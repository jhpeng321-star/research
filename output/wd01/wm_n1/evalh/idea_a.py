"""P8 evaluation-harness extension for Idea A (amendment 3 sections A3.3-A3.5).

Adds, without editing other ``evalh`` files:

* variant **A** (controller-in-the-loop) as an evaluable predictor, composed
  from the P1 ``ctrl`` mover, the Part A baseline + P2 HDV residual, the P4
  arrival head and the P5 correction head through the P3 rollout;
* the three held-out-strategy tests (A3.4) (i) HEADLINE J4 -> J4B,
  (ii) drop every PI arm, (iii) drop one wall-speed factor's arms;
* the OOD "is this motion familiar?" score on CAV planned trajectories,
  calibrated on CAL only (A3.5);
* the owner action-descriptor requirement (2026-10-01): the action descriptor
  of a paired comparison is the PLANNED SCHEDULE (19 x 30 s segments per
  CAV/group from the features plan tensor + wall hold/speed/cap), never the
  arm name; sign accuracy and deltaV CRPS are reported stratified by action
  duration/onset x tau with world-bootstrap CIs; an a-priori representation
  analysis CLI (TRAIN plans/schedules only, zero SUMO) and a disclosure line
  generator.

Integration points owned by P8 but living in other packages (documented in
INTERFACES_V2.md, not edited here): ``combine.select_variant.VARIANTS`` gains
``"A"`` and the A3.3 tie order, and ``adapter.predictor`` loads the composed
Idea A bundle.  Those edits are made in their own files.

Rollout arrays are on CPU; correction inference follows the loaded modules'
dtype/device. The predictor never opens outcome files: the held-out protocol
carries the access guard, the OOD calibration is CAL-only, and the predictor
only consumes rows handed to it.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from evalh import metrics as _metrics
from evalh import seals as _seals
from evalh.protocol import (GATING_TAUS, HORIZONS_S, N_QUANTILES,
                            QUANTILE_LEVELS, OutcomeRow, PredictorBase,
                            RolloutBatch, RolloutRow)

# Read-only reference to the frozen features plan-tensor definitions (the
# owner requirement forbids redefining them here).
from features import constants as _FC

VARIANT_A = "A"

#: A3.3 candidate set with the owner-acked tie order ``T < A < C < N``.
IDEA_A_VARIANTS = ("T", "A", "C", "N")

#: The three A3.4 held-out-strategy tests, in registration order.
HELD_OUT_TESTS = ("i", "ii", "iii")

#: J4 / J4B wall-law tags (A3.4(i)); J4B is law v2.2, J4 is v2.1.
J4_WALL_LAW = "wall-v2.1-closure"
J4B_WALL_LAW = "wall-v2.2"

#: A3.4(iii): fixed a priori as the highest J4 wall-speed factor.
#: J4 factors are W-mid 0.85 and W-lo 0.60 (walls_design/WALL_FIX_OPTIONS.md:75-78).
HIGHEST_J4_WALL_SPEED_FACTOR = "mid"

_DT_S = 0.5


# --------------------------------------------------------------- held-out ----
@dataclass(frozen=True)
class RunTag:
    """One campaign run descriptor for the held-out-strategy selectors.

    ``source`` is ``"TRAIN"``, ``"CAL"``, ``"J4"``, ``"J4B"`` or ``"C5"``;
    ``policy`` is ``native`` / ``damping`` / ``pi`` / ``wall``.
    """

    source: str = "CAL"
    split: str = "train"
    policy: str = "native"
    wall_speed_factor: Optional[str] = None
    wall_law: Optional[str] = None


class HeldOutViolation(_seals.SealError):
    """A held-out read was attempted outside its registered configuration."""


@dataclass(frozen=True)
class HeldOutStrategyTest:
    """One A3.4 generalisation test (never a selection input).

    ``kind`` selects the train-exclusion / eval-inclusion rule:

    * ``"headline"`` (i): train on J4 (law v2.1), evaluate on J4B (law v2.2);
    * ``"pi"`` (ii): drop every PI arm from training, evaluate on CAL PI arms;
    * ``"wall_factor"`` (iii): drop one wall-speed factor from training,
      evaluate on that factor's CAL wall arms.
    """

    name: str
    train_filter: str
    eval_filter: str
    note: str = ""
    kind: str = ""
    factor: Optional[str] = None
    reads_j4b: bool = False

    def train_excludes(self, run: RunTag) -> bool:
        """True iff ``run`` is removed from this test's training set."""
        if self.kind == "headline":
            return str(run.source) != "J4"
        if self.kind == "pi":
            return str(run.policy) == "pi"
        if self.kind == "wall_factor":
            return (str(run.policy) == "wall"
                    and run.wall_speed_factor == self.factor)
        raise HeldOutViolation(f"unknown held-out kind {self.kind!r}")

    def eval_includes(self, run: RunTag) -> bool:
        """True iff ``run`` belongs to this test's evaluation set."""
        if self.kind == "headline":
            return str(run.source) == "J4B"
        if self.kind == "pi":
            return str(run.split) == "cal" and str(run.policy) == "pi"
        if self.kind == "wall_factor":
            return (str(run.split) == "cal" and str(run.policy) == "wall"
                    and run.wall_speed_factor == self.factor)
        raise HeldOutViolation(f"unknown held-out kind {self.kind!r}")


def held_out_protocol() -> tuple:
    """The three registered held-out-strategy tests (A3.4)."""
    return (
        HeldOutStrategyTest(
            name="i",
            train_filter="J4 wall law v2.1 only",
            eval_filter="J4B wall law v2.2 only",
            note="J4B outcomes read ONLY by this registered evaluation",
            kind="headline", reads_j4b=True),
        HeldOutStrategyTest(
            name="ii",
            train_filter="remove every PI arm (native-vs-PI pairs included)",
            eval_filter="the PI arms of CAL",
            note="", kind="pi", reads_j4b=False),
        HeldOutStrategyTest(
            name="iii",
            train_filter="remove the wall arms of one speed factor",
            eval_filter="that factor's wall arms of CAL",
            note="factor fixed a priori as the highest J4 factor",
            kind="wall_factor", factor=HIGHEST_J4_WALL_SPEED_FACTOR,
            reads_j4b=False),
    )


def assert_j4b_read_allowed(test: HeldOutStrategyTest) -> None:
    """Refuse a J4B read outside config (i) (A3.4)."""
    if not bool(test.reads_j4b):
        raise HeldOutViolation(
            f"held-out test {test.name!r} must not read J4B outcomes; only "
            "config (i) is registered to read J4B (A3.4)")


def assert_read_allowed(test: HeldOutStrategyTest, run: RunTag) -> None:
    """Refuse any J4B run not covered by this test's registration."""
    if str(run.source) == "J4B":
        assert_j4b_read_allowed(test)


def held_out_train_runs(test: HeldOutStrategyTest,
                        runs: Sequence[RunTag]) -> list:
    """The training runs left after this test's exclusions (A3.4)."""
    return [r for r in runs if not test.train_excludes(r)]


def held_out_eval_runs(test: HeldOutStrategyTest,
                       runs: Sequence[RunTag]) -> list:
    """The evaluation runs, refusing a J4B read outside config (i).

    The guard runs over the whole candidate pool: filtering the pool is itself
    a read, so any J4B run in it needs config (i) regardless of membership in
    the final evaluation set.
    """
    for r in runs:
        assert_read_allowed(test, r)
    return [r for r in runs if test.eval_includes(r)]


# --------------------------------------------------------------- predictor ---
class IdeaANotLoadedError(RuntimeError):
    """The variant A predictor was constructed without a bundle/rollout."""


def _to_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def _rollout_arrays(rollouts, *, need_accel: bool) -> tuple:
    """Pull ``positions/speeds/masks/accel`` as numpy ``[S,T,B,K]`` arrays."""
    if isinstance(rollouts, Mapping):
        pos = rollouts.get("positions")
        spd = rollouts.get("speeds")
        msk = rollouts.get("masks")
        acc = rollouts.get("accel") if need_accel else None
    else:
        pos = getattr(rollouts, "positions", None)
        spd = getattr(rollouts, "speeds", None)
        msk = getattr(rollouts, "masks", None)
        acc = getattr(rollouts, "accel", None) if need_accel else None
    if pos is None or spd is None:
        raise ValueError("rollouts must expose positions and speeds")
    pos = _to_np(pos)
    spd = _to_np(spd)
    if pos.ndim != 4:
        raise ValueError(
            f"rollout positions must be [S,T,B,K]; got {tuple(pos.shape)}")
    msk = _to_np(msk) if msk is not None else None
    acc = _to_np(acc) if (need_accel and acc is not None) else None
    return pos, spd, msk, acc


def _t0_channels(rollouts) -> tuple:
    """Resolve the explicit ORIGINAL t0 channels of a rollout.

    Returns ``(t0_kin, t0_mask, t0_counter, counter_present)`` where
    ``t0_kin``/``t0_mask`` are the raw channel values (or ``None``) and
    ``counter_present`` is True iff the rollout carries a counter CHANNEL
    (``t0_counter``/``counter``), regardless of its value.  Works for
    :class:`rollout2.state.Rollouts`/attribute objects and for injected
    mappings (keys ``t0_kin``/``t0_mask``/``t0_counter``).
    """
    if isinstance(rollouts, Mapping):
        kin = rollouts.get("t0_kin")
        msk = rollouts.get("t0_mask")
        if "t0_counter" in rollouts:
            counter, present = rollouts["t0_counter"], True
        elif "counter" in rollouts:
            counter, present = rollouts["counter"], True
        else:
            counter, present = None, False
    else:
        kin = getattr(rollouts, "t0_kin", None)
        msk = getattr(rollouts, "t0_mask", None)
        if hasattr(rollouts, "t0_counter"):
            counter, present = getattr(rollouts, "t0_counter"), True
        elif hasattr(rollouts, "counter"):
            counter, present = getattr(rollouts, "counter"), True
        else:
            counter, present = None, False
    return kin, msk, counter, present


def _kin_from_rollout(rollouts, *, sample: int = 0, required: bool = False):
    """``(kin [B,K,7], mask [B,K])`` for the A6 state token.

    The token must be built from the ORIGINAL t0 state the trainer saw, so
    the explicit channels (``t0_kin``/``t0_mask``, carried by every real P3
    engine rollout before any controller step) take precedence.  Only when
    the channels are absent AND ``required`` is False (a legacy synthetic
    test double scored with the token OFF) does the trajectory fallback
    reconstruct an approximation from ``t0`` snapshots.  With ``required``
    True (an A6-ON checkpoint) missing channels fail closed: the fallback
    uses the first FUTURE applied acceleration, a 5 m default length and an
    ones mask, none of which the trainer fed.
    """
    kin0, msk0, _, _ = _t0_channels(rollouts)
    if kin0 is not None and msk0 is not None:
        kin0 = _to_np(kin0)
        msk0 = _to_np(msk0)
        if kin0.ndim != 3 or kin0.shape[-1] != 7:
            raise ValueError(
                f"t0_kin must be [B,K,7]; got {tuple(kin0.shape)}")
        if msk0.shape != kin0.shape[:2]:
            raise ValueError(
                f"t0_mask must be [B,K]={tuple(kin0.shape[:2])}; got "
                f"{tuple(msk0.shape)}")
        return (torch.as_tensor(np.ascontiguousarray(kin0),
                                dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(msk0),
                                dtype=torch.bool))
    if required:
        raise IdeaANotLoadedError(
            "state_global_token is ON but the rollout carries no explicit "
            "t0 kin/mask channels; the trajectory fallback (first FUTURE "
            "acceleration, 5 m default length, ones mask) is not the t0 "
            "state the trainer fed and is refused for A6-ON checkpoints")
    pos, spd, msk, acc = _rollout_arrays(rollouts, need_accel=True)
    b, k = pos.shape[2], pos.shape[3]
    pos0 = pos[sample, 0]
    spd0 = spd[sample, 0]
    mask = (msk[sample, 0] if msk is not None
            else np.ones((b, k), dtype=bool))
    lanes = (rollouts.get("lanes") if isinstance(rollouts, Mapping)
             else getattr(rollouts, "lanes", None))
    kin = np.zeros((b, k, 7), dtype=np.float32)
    kin[..., 0] = pos0
    if lanes is not None:
        l0 = _to_np(lanes)[sample, 0]
        for li in range(3):
            kin[..., 1 + li] = (l0 == li)
    kin[..., 4] = spd0
    if acc is not None and acc.ndim >= 3:
        kin[..., 5] = acc[sample, 0]
    kin[..., 6] = 5.0
    return torch.as_tensor(kin), torch.as_tensor(mask)


def _collapse_rollouts(rollouts, horizons_s: tuple,
                       n_rows: int) -> RolloutBatch:
    """``[S,T,B,K]`` trajectories -> ``RolloutBatch [N,H,V]`` (sample mean)."""
    pos, spd, msk, _ = _rollout_arrays(rollouts, need_accel=False)
    pos = pos.mean(axis=0)
    spd = spd.mean(axis=0)
    if msk is not None:
        msk = msk > 0.5
        while msk.ndim > 2:
            msk = msk.any(axis=0)
    else:
        msk = np.ones(pos.shape[1:], dtype=bool)
    horizons = tuple(float(h) for h in horizons_s)
    h = len(horizons)
    steps = pos.shape[0]
    present = [np.flatnonzero(msk[b]).tolist() for b in range(pos.shape[1])]
    v_cols = max((len(p) for p in present), default=0)
    pos_out = np.zeros((n_rows, h, v_cols), dtype=float)
    spd_out = np.zeros((n_rows, h, v_cols), dtype=float)
    mask_out = np.zeros((n_rows, h, v_cols), dtype=float)
    step_idx = [min(steps - 1, int(round(hi / _DT_S))) for hi in horizons]
    for b in range(n_rows):
        slots = present[b] if b < len(present) else []
        n_use = min(len(slots), v_cols)
        for hi, si in enumerate(step_idx):
            pos_out[b, hi, :n_use] = pos[si, b, slots[:n_use]]
            spd_out[b, hi, :n_use] = spd[si, b, slots[:n_use]]
            mask_out[b, hi, :n_use] = 1.0
    return RolloutBatch(
        horizons, pos_out, spd_out, mask=mask_out,
        collision=np.zeros(n_rows, dtype=float),
        ttc_lt3_s=np.zeros(n_rows, dtype=float),
        hard_brake_s=np.zeros(n_rows, dtype=float),
        intervention=np.zeros(n_rows, dtype=float))


def delay_quantiles_from_head(head_q, rows: Sequence[OutcomeRow], *,
                              taus=None, delay_index: Optional[int] = None,
                              ) -> np.ndarray:
    """Select each row's own-tau delay quantiles from a P5 head output.

    ``head_q`` is ``[B, n_tau, n_outcome_targets, n_quantiles]``.  ``taus``
    defaults to the frozen ``model.constants.TAUS`` (mapped offline only).
    """
    head_q = _to_np(head_q)
    if head_q.ndim != 4:
        raise ValueError(
            f"head_q must be [B, n_tau, n_targets, n_quantiles]; got "
            f"{tuple(head_q.shape)}")
    if taus is None or delay_index is None:
        from model import constants as C
        if taus is None:
            taus = C.TAUS
        if delay_index is None:
            delay_index = C.OUTCOME_TARGETS.index("delay_s")
    taus = np.asarray(taus, dtype=float)
    rows = list(rows)
    out = np.empty((len(rows), head_q.shape[-1]), dtype=float)
    for i, r in enumerate(rows):
        ti = int(np.argmin(np.abs(taus - float(r.tau_s))))
        out[i] = head_q[i, ti, int(delay_index), :]
    return out


class IdeaAVariantPredictor(PredictorBase):
    """Variant A predictor: composed P1-P5 bundle through the P3 rollout.

    Implements the ``evalh.protocol.Predictor`` contract.  The composition is
    injectable for CPU tests: ``rollout_fn(rows, *, n_steps, n_samples, seed)``
    returns the P3 ``Rollouts`` and ``outcome_fn(rows, rollouts)`` returns the
    per-row delay quantiles ``[N, 19]``.  When they are omitted the real P3
    rollout and the existing trained P5 projection/head composition are used.
    """

    name = VARIANT_A
    quantile_levels = QUANTILE_LEVELS

    def __init__(self, bundle=None, *, device: str = "cpu", rollout_fn=None,
                 outcome_fn=None, batch_builder=None, seed: int = 0,
                 horizon_steps: Optional[int] = None) -> None:
        self.bundle = bundle
        self.device = str(device)
        self.rollout_fn = rollout_fn
        self.outcome_fn = outcome_fn
        self.batch_builder = batch_builder
        self.seed = int(seed)
        if horizon_steps is None:
            horizon_steps = max(1, int(round(max(HORIZONS_S) / _DT_S)))
        self.horizon_steps = int(horizon_steps)

    def _require_bundle(self) -> object:
        if self.bundle is None and self.rollout_fn is None:
            raise IdeaANotLoadedError(
                "IdeaAVariantPredictor has no composed bundle; construct it "
                "with an IdeaABundle (and optionally a rollout_fn for tests)")
        return self.bundle

    def _batch(self, rows):
        if self.batch_builder is not None:
            return self.batch_builder(rows)
        raise IdeaANotLoadedError(
            "the production Idea A rollout needs a batch_builder to resolve "
            "the packed world batch for these rows")

    def _rollout(self, rows, *, n_steps: int, n_samples: int, seed: int):
        if self.rollout_fn is not None:
            return self.rollout_fn(rows, n_steps=n_steps, n_samples=n_samples,
                                   seed=seed)
        from rollout2.engine import rollout2
        batch = self._batch(rows)
        return rollout2(self.bundle, batch, n_steps=n_steps,
                        n_samples=n_samples, seed=seed)

    @staticmethod
    def _rollout_counter(rollouts, explicit=None):
        """Resolve ``(counter, present)`` for the A6 state token.

        Training feeds ``StateGlobalToken(kin, kin_mask, counter)`` with the
        ACTUAL t0 entrance counters; evaluation must feed the SAME counter or
        the token is not reproducible (contrastfix A6 parity).  This resolver
        only reports WHAT the rollout carries: ``present`` is True iff a
        counter channel exists (``t0_counter``/``counter`` attribute or
        mapping key), regardless of its value.  Whether an absent/None
        counter is refused is decided by the caller from the checkpoint's A6
        state: an ON checkpoint refuses (no counter-zero fallback), an OFF
        checkpoint keeps the legacy ``None`` path.
        """
        if explicit is not None:
            return explicit, True
        return _t0_channels(rollouts)[2:4]

    def _state_global_token(self, proj, rollouts, *, counter=None):
        """Infer the original-t0 token using the trained encoder placement."""
        from corr.evaluation import state_token

        kin, mask, _, _ = _t0_channels(rollouts)
        t0_counter, _ = self._rollout_counter(rollouts, counter)
        try:
            return state_token(self.bundle, proj, kin=kin, mask=mask,
                               counter=t0_counter)
        except ValueError as exc:
            raise IdeaANotLoadedError(str(exc)) from exc

    @torch.no_grad()
    def _outcome(self, rows, rollouts) -> np.ndarray:
        if self.outcome_fn is not None:
            return np.asarray(self.outcome_fn(rows, rollouts), dtype=float)
        from corr.evaluation import (channel, module_data, trained_correction,
                                     trajectory_summary)
        from model import constants as _MC

        try:
            head = trained_correction(self.bundle)
            pos, spd, msk, acc = _rollout_arrays(rollouts, need_accel=True)
            ids = channel(rollouts, "slot_ids")
            if ids is None:
                raise ValueError("correction requires explicit slot_ids")
            ids = _to_np(ids)
            if ids.shape != pos.shape or msk is None or msk.shape != pos.shape:
                raise ValueError("slot_ids/masks must match positions [S,T+1,B,K]")
            if acc is None or acc.shape != pos[:, 1:].shape:
                raise ValueError("correction requires accel [S,T,B,K]")
            # Score each sample separately: averaging trajectories would mix
            # identities/survival and change the trained nonlinear summary.
            # The evaluation rollout is longer than training for the rollout
            # metrics; slice back to the registered training correction window
            # (single source ``model.constants.CORRECTION_WINDOW_STEPS``) so
            # the summary is the SAME distribution the head was trained on.
            window = _MC.correction_window_steps(pos.shape[1])
            quantiles = []
            for sample in range(pos.shape[0]):
                raw = trajectory_summary(
                    pos[sample][:window + 1], spd[sample][:window + 1],
                    acc[sample][:window], msk[sample][:window + 1],
                    t0_kin=channel(rollouts, "t0_kin"),
                    t0_mask=channel(rollouts, "t0_mask"),
                    t0_is_cav=channel(rollouts, "t0_is_cav"),
                    t0_slot_id=channel(rollouts, "t0_slot_id"),
                    slot_ids=ids[sample][:window + 1])
                raw = module_data(head, raw)
                reference = raw.new_zeros((raw.shape[0], int(head.cfg.d_model)))
                token = self._state_global_token(reference, rollouts)
                head_out = head(raw, token)
                head_q = (head_out["outcome_quantiles"]
                          if isinstance(head_out, Mapping)
                          else head_out.outcome_quantiles)
                norm = getattr(self.bundle, "outcome_norm", None)
                if norm is not None:
                    from model.outcome_norm import denormalize_quantiles

                    head_q = denormalize_quantiles(head_q, norm)
                quantiles.append(delay_quantiles_from_head(head_q, rows))
            return np.mean(quantiles, axis=0)
        except ValueError as exc:
            raise IdeaANotLoadedError(str(exc)) from exc

    def predict_quantiles(self, rows: Sequence[OutcomeRow]) -> np.ndarray:
        """Roll out each row and return ``[len(rows), 19]`` delay quantiles."""
        rows = list(rows)
        if not rows:
            return np.zeros((0, N_QUANTILES), dtype=float)
        self._require_bundle()
        rollouts = self._rollout(rows, n_steps=self.horizon_steps,
                                 n_samples=1, seed=self.seed + 7)
        q = self._outcome(rows, rollouts)
        if q.shape != (len(rows), self.quantile_levels.size):
            raise ValueError(
                f"{self.name}: predictor returned {q.shape}, expected "
                f"({len(rows)}, {self.quantile_levels.size})")
        q = np.maximum.accumulate(q, axis=1)
        return self._validate(q, len(rows))

    def sample_rollouts(self, rows: Sequence[RolloutRow],
                        horizons_s: Optional[tuple] = None) -> RolloutBatch:
        """Short-horizon trajectories from the Idea A rollout."""
        rows = list(rows)
        if horizons_s is None:
            horizons_s = (tuple(getattr(rows[0], "horizons_s", HORIZONS_S))
                          if rows else HORIZONS_S)
        horizons = tuple(float(h) for h in horizons_s)
        if not rows:
            return RolloutBatch(horizons, np.zeros((0, len(horizons), 0)),
                                np.zeros((0, len(horizons), 0)))
        self._require_bundle()
        n_steps = max(1, int(round(max(horizons) / _DT_S)))
        rollouts = self._rollout(rows, n_steps=n_steps, n_samples=1,
                                 seed=self.seed + 101)
        return _collapse_rollouts(rollouts, horizons, len(rows))


# -------------------------------------------------------------------- OOD ----
@dataclass
class OodConfig:
    """OOD score calibration (A3.5): CAL-only, diagnostic on TEST."""

    cal_quantile: float = 0.95
    min_summary_std: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < float(self.cal_quantile) <= 1.0:
            raise ValueError("cal_quantile must be in (0, 1]")
        if float(self.min_summary_std) <= 0.0:
            raise ValueError("min_summary_std must be positive")


def _summary_matrix(summaries, what: str) -> np.ndarray:
    x = _to_np(summaries)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{what} must be [N,F]; got {tuple(x.shape)}")
    return x


def _resolve_ood_primitive(name: str):
    """The P5 ``corr.ood`` primitive, or ``None`` before P5 lands."""
    try:
        import corr.ood as _ood
    except Exception:                                    # pragma: no cover
        return None
    return getattr(_ood, name, None)


def _local_fit(summaries, *, min_summary_std: float = 1e-6) -> dict:
    x = np.asarray(summaries, dtype=float)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), float(min_summary_std))
    return dict(mean=mean, std=std, dim=int(x.shape[1]),
                min_summary_std=float(min_summary_std))


def _local_score(summaries, model) -> np.ndarray:
    x = np.asarray(summaries, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    floor = float(model.get("min_summary_std", 1e-6))
    z = (x - mean) / np.maximum(std, floor)
    return (z * z).mean(axis=-1)


def calibrate_ood(train_summaries, cal_summaries, *, cfg: Optional[OodConfig] = None,
                  split: str = "cal", worlds=None, fit_fn=None,
                  score_fn=None) -> dict:
    """Calibrate the CAV-motion familiarity score (A3.5).

    Fits the diagonal-Gaussian density on ``train_summaries`` (via the P5
    ``corr.ood`` primitives when present) and sets the hold-back ``threshold``
    at the ``cal_quantile`` of the CAL scores.  CAL-ONLY guard: any ``split``
    other than ``"cal"`` is refused, and ``worlds`` (when given) go through the
    canonical HOLDOUT guard.
    """
    cfg = cfg or OodConfig()
    if str(split) != "cal":
        raise _seals.SealError(
            "OOD calibration is CAL-only (A3.5); refusing split="
            f"{split!r}")
    if worlds is not None:
        _seals.assert_holdout(list(worlds))
    train = _summary_matrix(train_summaries, "train_summaries")
    cal = _summary_matrix(cal_summaries, "cal_summaries")
    if train.shape[1] != cal.shape[1]:
        raise ValueError(
            "train/cal summaries disagree on feature count: "
            f"{train.shape[1]} vs {cal.shape[1]}")
    fit = fit_fn or _resolve_ood_primitive("motion_density_fit") or _local_fit
    score = score_fn or _resolve_ood_primitive("motion_density_score") \
        or _local_score
    try:
        model = fit(train, min_summary_std=cfg.min_summary_std)
    except TypeError:
        model = fit(train)
    cal_scores = np.asarray(score(cal, model), dtype=float).reshape(-1)
    threshold = (float(np.quantile(cal_scores, cfg.cal_quantile))
                 if cal_scores.size else float("inf"))
    return dict(kind="cav_motion_density", split="cal",
                cal_quantile=float(cfg.cal_quantile),
                min_summary_std=float(cfg.min_summary_std),
                threshold=threshold, model=model,
                n_train=int(train.shape[0]), n_cal=int(cal.shape[0]))


def ood_score(summaries, calibration: dict, *, score_fn=None) -> np.ndarray:
    """Per-world "is this motion familiar?" score; larger = less familiar."""
    x = _summary_matrix(summaries, "summaries")
    model = calibration.get("model", calibration)
    score = score_fn or _resolve_ood_primitive("motion_density_score") \
        or _local_score
    return np.asarray(score(x, model), dtype=float).reshape(-1)


def hold_back(scores, calibration: dict) -> np.ndarray:
    """The planner hold-back rule: ``score >= threshold`` (A3.5)."""
    return np.asarray(scores, dtype=float).reshape(-1) >= float(
        calibration["threshold"])


# ============================================== action descriptor (2026-10) ===
# Owner action-descriptor requirement (via main, 2026-10-01): the action
# descriptor of a paired comparison is the PLANNED SCHEDULE, never the arm
# name.  The descriptor is 19 x 30 s per-CAV/per-group segment actions taken
# from the features plan tensor plus the wall hold/speed/cap.  Everything here
# is a read-only consumer of ``features.constants`` (plan tensor) and of the
# frozen ``evalh.metrics`` primitives.  Zero SUMO, no physics outcomes.

TRAIN = "TRAIN"

#: Reserved pseudo-vehicle id carrying the wall plan inside a descriptor.
WALL_VEHICLE_ID = "__wall__"

#: Plan-tensor contract, mirrored read-only from features/constants.py.
PLAN_N_SEGMENTS = int(_FC.N_SEGMENTS)          # 19
PLAN_SEGMENT_S = float(_FC.SEGMENT_S)          # 30.0
PLAN_HORIZON_S = float(PLAN_N_SEGMENTS) * PLAN_SEGMENT_S   # 570.0
PLAN_F = int(_FC.PLAN_F)                       # 8
PLAN_ACTIONS = tuple(str(a) for a in _FC.ACTIONS)          # native..unknown

#: Schedule-event action aliases -> canonical plan action token.
ACTION_ALIASES = {
    "native": "native", "damping": "damping", "pi": "pi",
    "pi-external-u": "pi", "fleet-pi": "pi", "fleet-pi-external-u": "pi",
    "wall": "wall", "wall-v2": "wall", "unknown": "unknown",
}
DEFAULT_ACTION = "native"

#: Duration strata of an action difference (owner requirement item 3).
DURATION_STRATA = ("<=60s", "60-240s", "whole-window")
#: Onset bins of the first differing segment.
ONSET_BINS = ("0-30s", "30-60s", "60-240s", "after-240s")


class ActionDescriptorError(ValueError):
    """A planned-schedule descriptor was malformed or replaced by a label."""


def canon_action(action) -> str:
    """Canonical plan action token for a schedule event action string."""
    key = str(action).strip().lower()
    return ACTION_ALIASES.get(key, key)


def _event_triplet(event) -> tuple:
    """``(vehicle_id, time_s, action)`` from a mapping / object / 3-tuple."""
    if isinstance(event, Mapping):
        return (str(event.get("vehicle_id")), float(event["time_s"]),
                canon_action(event.get("action")))
    if hasattr(event, "vehicle_id"):
        return (str(event.vehicle_id), float(event.time_s),
                canon_action(event.action))
    vid, ts, act = event
    return (str(vid), float(ts), canon_action(act))


def _wall_segments(onset_s: float, hold_s: float, *,
                   n_segments: int = PLAN_N_SEGMENTS,
                   segment_s: float = PLAN_SEGMENT_S) -> tuple:
    """Segments where the wall is planned active (``wall`` else ``native``)."""
    start = int(round(float(onset_s) / segment_s))
    n = int(round(float(hold_s) / segment_s))
    return tuple("wall" if start <= k < start + n else DEFAULT_ACTION
                 for k in range(n_segments))


def _wall_fields(wall, t0_s: float, *, n_segments: int,
                 segment_s: float) -> tuple:
    """``(segments, hold_s, speed, cap, wall_id)`` from a plan wall block."""
    hold = float(wall.get("hold_s", 0.0) or 0.0)
    if wall.get("target_time_s") is not None:
        onset = float(wall["target_time_s"]) - float(t0_s)
    else:
        onset = float(wall.get("onset_s", 0.0) or 0.0)
    speed = wall.get("speed_factor", wall.get("speed", 0.0)) or 0.0
    cap = wall.get("speed_floor_mps", wall.get("cap", 0.0)) or 0.0
    wid = wall.get("wall_id", wall.get("id", 0))
    segs = _wall_segments(onset, hold, n_segments=n_segments,
                          segment_s=segment_s)
    return segs, hold, float(speed), float(cap), int(wid)


@dataclass(frozen=True)
class ActionDescriptor:
    """The planned schedule of one controller on one decision window.

    ``segments`` is a sorted tuple of ``(cav_id, (action_0..action_n-1))``
    per-CAV/group 30 s segment actions; the reserved :data:`WALL_VEHICLE_ID`
    carries the wall-active segments.  ``wall_hold_s``/``wall_speed``/
    ``wall_cap``/``wall_id`` carry the wall hold, speed and cap.  The type
    deliberately has NO arm/contrast/policy field: the descriptor IS the
    action, never a name.
    """

    segments: tuple = ()
    wall_hold_s: float = 0.0
    wall_speed: float = 0.0
    wall_cap: float = 0.0
    wall_id: int = -1

    @property
    def n_segments(self) -> int:
        for _, segs in self.segments:
            return len(segs)
        return int(PLAN_N_SEGMENTS)

    def cav_ids(self) -> tuple:
        return tuple(cid for cid, _ in self.segments)

    def as_mapping(self) -> dict:
        return {cid: segs for cid, segs in self.segments}

    def has_wall(self) -> bool:
        return bool(self.wall_id >= 0 or self.wall_hold_s > 0.0
                    or WALL_VEHICLE_ID in dict(self.segments))

    def wall_segments(self) -> tuple:
        return dict(self.segments).get(WALL_VEHICLE_ID, ())

    def describe(self) -> str:
        """Compact human-readable schedule digest (no arm name)."""
        parts = [f"{cid}:" + "".join(s[0] for s in segs)
                 for cid, segs in self.segments]
        wall = (f"wall[{self.wall_id}] hold={self.wall_hold_s:g}"
                f" speed={self.wall_speed:g} cap={self.wall_cap:g}")
        return ";".join(parts + [wall]) if parts else wall


def descriptor_from_events(events, t0_s: float, *,
                           n_segments: int = PLAN_N_SEGMENTS,
                           segment_s: float = PLAN_SEGMENT_S,
                           wall=None,
                           default_action: str = DEFAULT_ACTION
                           ) -> ActionDescriptor:
    """Build a descriptor from planned-schedule events.

    ``events`` is an iterable of ``{vehicle_id, time_s, action}`` mappings (or
    ``(vehicle_id, time_s, action)`` tuples).  For segment ``k`` starting at
    ``t0_s + k * segment_s`` a CAV's action is the latest event at or before
    that time, defaulting to ``native``.  A CAV absent from the events stays
    ``native`` everywhere.  ``wall`` (a plan wall block) adds the reserved
    wall pseudo-vehicle plus the wall hold/speed/cap scalars.  Therefore
    "damp 60 s then native" and "damp 570 s" are different descriptors.
    """
    per: dict = {}
    for event in events or ():
        vid, ts, act = _event_triplet(event)
        per.setdefault(vid, []).append((ts, act))
    rows = []
    for vid in sorted(per):
        evs = sorted(per[vid], key=lambda x: x[0])
        segs = []
        for k in range(n_segments):
            t = float(t0_s) + float(segment_s) * k
            act = default_action
            for tt, aa in evs:
                if tt <= t + 1e-9:
                    act = aa
                else:
                    break
            segs.append(act)
        rows.append((vid, tuple(segs)))
    kw: dict = {}
    if wall is not None:
        segs, hold, speed, cap, wid = _wall_fields(
            wall, t0_s, n_segments=n_segments, segment_s=segment_s)
        rows = [r for r in rows if r[0] != WALL_VEHICLE_ID]
        rows.append((WALL_VEHICLE_ID, segs))
        kw = dict(wall_hold_s=hold, wall_speed=speed, wall_cap=cap,
                  wall_id=wid)
    return ActionDescriptor(segments=tuple(sorted(rows)), **kw)


def descriptor_from_schedule_dict(sched, t0_s: float, *,
                                  n_segments: int = PLAN_N_SEGMENTS,
                                  segment_s: float = PLAN_SEGMENT_S
                                  ) -> ActionDescriptor:
    """Descriptor from a ``switch_schedule.json`` payload (schedule fields)."""
    if isinstance(sched, Mapping) and "schedule" in sched:
        sched = sched["schedule"]
    events = (sched or {}).get("events", []) if isinstance(sched, Mapping) \
        else []
    return descriptor_from_events(events, t0_s, n_segments=n_segments,
                                  segment_s=segment_s)


def descriptor_from_plan_tensor(plan, *, cav_ids=None, wall=None,
                                n_segments: int = PLAN_N_SEGMENTS,
                                segment_s: float = PLAN_SEGMENT_S
                                ) -> ActionDescriptor:
    """Descriptor from a features plan tensor ``[S, C, PLAN_F]`` (or ``[S, F]``).

    The action token is the ``argmax`` over the first five (action one-hot)
    columns; wall speed/hold/id come from ``PLAN_WALL_SPEED``/
    ``PLAN_WALL_HOLD``/``PLAN_WALL_ID``.  An all-zero action one-hot is
    ``unknown`` and drops that CAV unless ``cav_ids`` names it.
    """
    arr = np.asarray(plan, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim != 3 or arr.shape[0] != n_segments:
        raise ActionDescriptorError(
            f"plan tensor must be [n_segments, C, F] or [n_segments, F]; got "
            f"{tuple(arr.shape)}")
    n_act = len(PLAN_ACTIONS)
    rows = []
    wall_speed = wall_hold = 0.0
    wall_id = -1
    n_cav = arr.shape[1]
    for j in range(n_cav):
        segs = []
        for s in range(n_segments):
            vec = arr[s, j, :n_act]
            idx = int(np.argmax(vec))
            segs.append(PLAN_ACTIONS[idx] if vec[idx] > 0.0 else "unknown")
        cid = (str(cav_ids[j]) if cav_ids is not None else f"slot{j}")
        include = cav_ids is not None or any(a != "unknown" for a in segs)
        if include:
            rows.append((cid, tuple(segs)))
        if _FC.PLAN_WALL_SPEED < arr.shape[2]:
            wall_speed = max(wall_speed,
                             float(np.nanmax(arr[:, j, _FC.PLAN_WALL_SPEED])))
        if _FC.PLAN_WALL_HOLD < arr.shape[2]:
            wall_hold = max(wall_hold,
                            float(np.nanmax(arr[:, j, _FC.PLAN_WALL_HOLD])))
        if _FC.PLAN_WALL_ID < arr.shape[2]:
            wid = float(np.nanmax(arr[:, j, _FC.PLAN_WALL_ID]))
            if wid >= 0:
                wall_id = int(wid)
    kw: dict = dict(wall_hold_s=wall_hold * 600.0,
                    wall_speed=wall_speed * 40.0, wall_id=wall_id)
    if wall is not None:
        segs, hold, speed, cap, wid = _wall_fields(
            wall, 0.0, n_segments=n_segments, segment_s=segment_s)
        rows = [r for r in rows if r[0] != WALL_VEHICLE_ID]
        rows.append((WALL_VEHICLE_ID, segs))
        kw = dict(wall_hold_s=hold, wall_speed=speed, wall_cap=cap,
                  wall_id=wid)
    return ActionDescriptor(segments=tuple(sorted(rows)), **kw)


@dataclass(frozen=True)
class ScheduleDifference:
    """Stratified difference of two planned schedules (no arm names)."""

    differs: bool
    onset_s: float
    duration_s: float
    n_differing_segments: int
    duration_stratum: str
    onset_bin: str
    first_60_only: bool
    wall_differs: bool


def _duration_stratum(span_s: float) -> str:
    if span_s <= 60.0 + 1e-9:
        return "<=60s"
    if span_s <= 240.0 + 1e-9:
        return "60-240s"
    return "whole-window"


def _onset_bin(onset_s: float) -> str:
    if onset_s < 30.0 - 1e-9:
        return "0-30s"
    if onset_s < 60.0 - 1e-9:
        return "30-60s"
    if onset_s < 240.0 - 1e-9:
        return "60-240s"
    return "after-240s"


_SAME_DIFF = ScheduleDifference(False, 0.0, 0.0, 0, "same", "none", False,
                                False)


def schedule_difference(a: ActionDescriptor, b: ActionDescriptor, *,
                        n_segments: Optional[int] = None,
                        segment_s: float = PLAN_SEGMENT_S
                        ) -> ScheduleDifference:
    """Stratify how two planned schedules differ.

    A segment differs when any CAV's per-segment action differs (missing CAVs
    default to ``native``); a changed wall hold/speed/cap marks that
    descriptor's wall-active segments.  ``duration_s`` is the onset-to-last
    differing span in seconds and ``first_60_only`` is true iff every
    differing segment starts before 60 s (the "60 s then revert" pair).
    """
    n = int(n_segments or max(a.n_segments, b.n_segments, PLAN_N_SEGMENTS))
    ma, mb = a.as_mapping(), b.as_mapping()
    mask = [False] * n
    for cid in sorted(set(ma) | set(mb)):
        sa = ma.get(cid, ())
        sb = mb.get(cid, ())
        for k in range(n):
            va = sa[k] if k < len(sa) else DEFAULT_ACTION
            vb = sb[k] if k < len(sb) else DEFAULT_ACTION
            if va != vb:
                mask[k] = True
    wall_union = set()
    for d in (a, b):
        for k, v in enumerate(d.wall_segments()):
            if v == "wall":
                wall_union.add(k)
    wall_differs = ((a.wall_hold_s, a.wall_speed, a.wall_cap, a.wall_id)
                    != (b.wall_hold_s, b.wall_speed, b.wall_cap, b.wall_id))
    if wall_differs:
        for k in wall_union:
            if k < n:
                mask[k] = True
    idx = [k for k, flag in enumerate(mask) if flag]
    if not idx:
        return _SAME_DIFF
    onset = idx[0] * segment_s
    span = (idx[-1] - idx[0] + 1) * segment_s
    last_start = idx[-1] * segment_s
    return ScheduleDifference(
        True, float(onset), float(span), len(idx),
        _duration_stratum(span), _onset_bin(onset),
        bool(last_start < 60.0 - 1e-9), bool(wall_differs))


# ------------------------------------------------- paired action comparisons --
@dataclass
class ActionPair:
    """One planned-schedule pair on one window (descriptors, never names)."""

    world: str
    t0_s: float
    decision_t_s: float
    tau_s: float
    treatment: ActionDescriptor
    reference: ActionDescriptor
    difference: ScheduleDifference
    dy: float
    variant_crps: float
    hybrid_crps: float
    variant_sign: float
    hybrid_sign: float
    variant_dv: float
    hybrid_dv: float


def _resolve_descriptor(descriptors, world: str, policy: str, row):
    """Look up the planned-schedule descriptor for one row."""
    if callable(descriptors):
        return descriptors(world, policy, row)
    if descriptors is None:
        raise ActionDescriptorError(
            "a paired comparison needs planned-schedule descriptors")
    if (world, policy) in descriptors:
        return descriptors[(world, policy)]
    if policy in descriptors:
        return descriptors[policy]
    feat = getattr(row, "features", None) or {}
    if "action_descriptor" in feat:
        return feat["action_descriptor"]
    raise ActionDescriptorError(
        f"no planned-schedule descriptor for (world={world!r}, "
        f"policy={policy!r}); the descriptor is never the arm name")


def build_action_pairs(rows: Sequence[OutcomeRow], variant_q: np.ndarray,
                       hybrid_q: np.ndarray, descriptors, *, levels=None,
                       taus=GATING_TAUS,
                       reference_policy: str = "native",
                       n_segments: int = PLAN_N_SEGMENTS,
                       segment_s: float = PLAN_SEGMENT_S) -> list:
    """Pair each non-reference row with its reference row on the same window.

    Every returned :class:`ActionPair` carries the two PLANNED-SCHEDULE
    descriptors and their :class:`ScheduleDifference`; there is intentionally
    no contrast/arm-name argument or field.
    """
    lv = np.asarray(QUANTILE_LEVELS if levels is None else levels, float)
    mi = _metrics.median_index(lv)
    taus = {round(float(t), 6) for t in taus}
    rows = list(rows)
    idx = {}
    for i, r in enumerate(rows):
        idx[(r.world, round(float(r.t0_s), 6), round(float(r.decision_t_s), 6),
             round(float(r.tau_s), 6), r.policy)] = i
    keys = sorted({(r.world, round(float(r.t0_s), 6),
                    round(float(r.decision_t_s), 6),
                    round(float(r.tau_s), 6)) for r in rows})
    pairs = []
    for (world, t0, dec, tau) in keys:
        if tau not in taus:
            continue
        ir = idx.get((world, t0, dec, tau, reference_policy))
        if ir is None:
            continue
        ref_rows = [(p, i) for (w, a, b, c, p), i in idx.items()
                    if (w, a, b, c) == (world, t0, dec, tau)
                    and p != reference_policy]
        for pol, it in sorted(ref_rows):
            ref_desc = _resolve_descriptor(descriptors, world,
                                           reference_policy, rows[ir])
            tr_desc = _resolve_descriptor(descriptors, world, pol, rows[it])
            dy = float(rows[it].y_delay_s - rows[ir].y_delay_s)
            dv_v = np.asarray(variant_q[it] - variant_q[ir], float)
            dv_h = np.asarray(hybrid_q[it] - hybrid_q[ir], float)
            pairs.append(ActionPair(
                world=world, t0_s=float(t0), decision_t_s=float(dec),
                tau_s=float(tau), treatment=tr_desc, reference=ref_desc,
                difference=schedule_difference(tr_desc, ref_desc,
                                               n_segments=n_segments,
                                               segment_s=segment_s),
                dy=dy,
                variant_crps=float(_metrics.crps_of_quantiles(dv_v, dy, lv)),
                hybrid_crps=float(_metrics.crps_of_quantiles(dv_h, dy, lv)),
                variant_sign=float(np.sign(dv_v[mi])),
                hybrid_sign=float(np.sign(dv_h[mi])),
                variant_dv=float(np.mean(dv_v)),
                hybrid_dv=float(np.mean(dv_h))))
    return pairs


def _cell_key(diff: ScheduleDifference) -> str:
    return f"{diff.duration_stratum}|{diff.onset_bin}"


def action_cell_means(pairs, *, levels=None) -> dict:
    """TRAIN cell mean of ``dy`` per ``(duration|onset, tau)`` (baseline)."""
    out: dict = {}
    for p in pairs:
        key = (_cell_key(p.difference), round(float(p.tau_s), 6))
        out.setdefault(key, []).append(float(p.dy))
    return {k: float(np.mean(v)) for k, v in out.items()}


def stratified_action_metrics(pairs, *, levels=None, n_boot: int = 2000,
                              alpha: float = 0.10, seed: int = 0,
                              threshold: float = 0.0,
                              cell_means=None) -> dict:
    """Sign accuracy and deltaV CRPS stratified by duration/onset x tau.

    Reports, per stratum and tau, the variant and the world-bootstrap 90 % CI
    of its deltaV CRPS difference against the ZERO and TRAIN-cell-mean
    baselines, the three sign accuracies on ``|dy| > threshold`` pairs, and
    the pair/world counts.  ``cell_means`` maps ``(cell, tau)`` (or a nested
    ``cell -> {tau: mean}``) to a TRAIN mean; absent, the in-sample cell mean
    is used.
    """
    lv = np.asarray(QUANTILE_LEVELS if levels is None else levels, float)
    q = int(lv.size)
    groups: dict = {}
    for p in pairs:
        groups.setdefault(_cell_key(p.difference), {}).setdefault(
            round(float(p.tau_s), 6), []).append(p)
    result: dict = {"schema": "wm_n1_stratified_action_metrics/1",
                    "levels": [float(x) for x in lv], "strata": {}}
    for key in sorted(groups):
        result["strata"][key] = {}
        for tau in sorted(groups[key]):
            ps = groups[key][tau]
            worlds = [p.world for p in ps]
            dy = np.asarray([p.dy for p in ps], float)
            vcrps = np.asarray([p.variant_crps for p in ps], float)
            zero = np.asarray([_metrics.crps_of_quantiles(np.zeros(q), d, lv)
                               for d in dy], float)
            cm = None
            if isinstance(cell_means, Mapping):
                if (key, tau) in cell_means:
                    cm = cell_means[(key, tau)]
                elif key in cell_means and isinstance(cell_means[key], Mapping):
                    cm = cell_means[key].get(tau)
                elif f"{key}|{tau:g}" in cell_means:
                    cm = cell_means[f"{key}|{tau:g}"]
            if cm is None:
                cm = float(np.mean(dy)) if dy.size else 0.0
            cm = float(cm)
            cmq = np.full(q, cm)
            cmcrps = np.asarray([_metrics.crps_of_quantiles(cmq, d, lv)
                                 for d in dy], float)
            z_lo, z_hi, z_pt = _metrics.world_bootstrap_ci(
                vcrps - zero, worlds, n_boot=n_boot, alpha=alpha, seed=seed)
            c_lo, c_hi, c_pt = _metrics.world_bootstrap_ci(
                vcrps - cmcrps, worlds, n_boot=n_boot, alpha=alpha, seed=seed)
            v_sign = np.asarray([p.variant_sign for p in ps], float)
            va, n_used, v_lo, v_hi = _metrics.sign_accuracy_ci(
                v_sign, dy, worlds, threshold, n_boot=n_boot, alpha=alpha,
                seed=seed)
            za, _, zs_lo, zs_hi = _metrics.sign_accuracy_ci(
                np.zeros_like(v_sign), dy, worlds, threshold, n_boot=n_boot,
                alpha=alpha, seed=seed)
            cma, _, cs_lo, cs_hi = _metrics.sign_accuracy_ci(
                np.sign(np.full_like(v_sign, cm)), dy, worlds, threshold,
                n_boot=n_boot, alpha=alpha, seed=seed)
            result["strata"][key][f"{tau:g}"] = dict(
                n_pairs=int(len(ps)), n_worlds=int(len(set(worlds))),
                variant_crps=float(np.mean(vcrps)),
                zero_crps=float(np.mean(zero)),
                cellmean_crps=float(np.mean(cmcrps)),
                cell_mean=cm,
                variant_vs_zero_diff=float(z_pt),
                variant_vs_zero_ci=[float(z_lo), float(z_hi)],
                variant_vs_cellmean_diff=float(c_pt),
                variant_vs_cellmean_ci=[float(c_lo), float(c_hi)],
                variant_sign_accuracy=va, zero_sign_accuracy=za,
                cellmean_sign_accuracy=cma,
                variant_sign_ci=[float(v_lo), float(v_hi)],
                zero_sign_ci=[float(zs_lo), float(zs_hi)],
                cellmean_sign_ci=[float(cs_lo), float(cs_hi)],
                sign_n_used=int(n_used), sign_threshold=float(threshold))
    return result


# ------------------------------------------------- a-priori representation ----
#: TRAIN campaign sources: (name, WORLD_PLAN.json rel, runs-root rel).  The
#: chunk-5/J4B and CAL/HOLDOUT splits are never present here.
REPRESENTATION_SPECS = (
    ("chunk1",
     "output/campaign_stage1_chunk1/results_plan_wm44/WORLD_PLAN.json",
     "runs/campaign_stage1_chunk1"),
    ("chunk1_fix27",
     "output/campaign_stage1_v3/results_plan_fix27/WORLD_PLAN.json",
     "runs/campaign_stage1_chunk1_fix27"),
    ("chunk2",
     "output/campaign_stage1_v2/results_plan_chunk2/WORLD_PLAN.json",
     "runs/campaign_stage1_chunk2"),
    ("chunk3",
     "output/campaign_stage1_v3/results_plan_chunk3/WORLD_PLAN.json",
     "runs/campaign_stage1_chunk3"),
    ("chunk4",
     "output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json",
     "runs/campaign_stage1_chunk4"),
    ("truearms_chunk2",
     "output/campaign_stage1_v3/results_plan_truearms_chunk2/WORLD_PLAN.json",
     "runs/campaign_stage1_chunk2_truearms"),
    ("truearms_chunk3",
     "output/campaign_stage1_v3/results_plan_truearms_chunk3/WORLD_PLAN.json",
     "runs/campaign_stage1_chunk3_truearms"),
    ("j4_walls",
     "output/campaign_walls_v1/results_plan/WORLD_PLAN.json",
     "runs/campaign_walls_pilot_chunk4"),
)

RESEARCH_DENSE_ROOT = Path(_seals.RD)

#: File names a representation read must never open (defence in depth).
FORBIDDEN_READ_NAMES = ("physics_steps", "dispatcher_rows", "insertions",
                        "evaluate_summary", "outcome_record")


def _read_json(path: Path) -> dict:
    name = Path(path).name.lower()
    if any(bad in name for bad in FORBIDDEN_READ_NAMES):
        raise ActionDescriptorError(
            f"refusing to open non-schedule file {path!r}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_worlds(plan: Mapping) -> list:
    """The TRAIN worlds of a ``WORLD_PLAN.json`` (before any run file open)."""
    return [w for w in plan.get("worlds", []) if w.get("split") == TRAIN]


def read_switch_schedule(run_dir) -> Optional[dict]:
    """The ``switch_schedule.json`` payload of a run, else ``None``."""
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    for cand in (run_dir / "run" / "switch_schedule.json",
                 run_dir / "switch_schedule.json"):
        if cand.is_file():
            return _read_json(cand)
    return None


def branch_descriptor(branch, t0_s: float, run_dir=None, *,
                      n_segments: int = PLAN_N_SEGMENTS,
                      segment_s: float = PLAN_SEGMENT_S) -> ActionDescriptor:
    """Planned-schedule descriptor of one plan branch (wall block or schedule)."""
    if isinstance(branch, Mapping) and branch.get("wall"):
        segs, hold, speed, cap, wid = _wall_fields(
            branch["wall"], t0_s, n_segments=n_segments, segment_s=segment_s)
        return ActionDescriptor(segments=((WALL_VEHICLE_ID, segs),),
                                wall_hold_s=hold, wall_speed=speed,
                                wall_cap=cap, wall_id=wid)
    sched = read_switch_schedule(run_dir)
    if sched is not None:
        return descriptor_from_schedule_dict(sched, t0_s,
                                             n_segments=n_segments,
                                             segment_s=segment_s)
    return ActionDescriptor()


def _empty_tally() -> dict:
    return {"by_duration": {}, "by_onset": {}, "by_duration_onset": {},
            "first_60_only": 0, "n_pairs": 0}


def _tally(tally: dict, diff: ScheduleDifference) -> None:
    tally["n_pairs"] += 1
    tally["by_duration"][diff.duration_stratum] = (
        tally["by_duration"].get(diff.duration_stratum, 0) + 1)
    tally["by_onset"][diff.onset_bin] = (
        tally["by_onset"].get(diff.onset_bin, 0) + 1)
    dk = _cell_key(diff)
    tally["by_duration_onset"][dk] = (
        tally["by_duration_onset"].get(dk, 0) + 1)
    if diff.first_60_only:
        tally["first_60_only"] += 1


def representation_counts(specs=None, *, root=None,
                          n_segments: int = PLAN_N_SEGMENTS,
                          segment_s: float = PLAN_SEGMENT_S) -> dict:
    """Count TRAIN sibling schedule pairs by difference duration stratum.

    For every TRAIN world the branch planned schedules are canonicalised and
    every unordered pair of distinct descriptors is stratified.  This reads
    ONLY the chunk ``WORLD_PLAN.json`` (to fix the split first) and
    ``switch_schedule.json`` schedule fields; no physics/outcome file is
    opened.  Walls (J4) are included via the plan wall blocks.
    """
    root = Path(root) if root is not None else RESEARCH_DENSE_ROOT
    specs = REPRESENTATION_SPECS if specs is None else specs
    sources: dict = {}
    summary = _empty_tally()
    summary.update({"by_source": {}, "n_train_worlds": 0, "n_runs": 0,
                    "n_descriptors": 0, "n_worlds_with_pairs": 0})
    for spec in specs:
        name, plan_rel, runs_rel = spec
        plan_path = Path(plan_rel)
        if not plan_path.is_absolute():
            plan_path = root / plan_rel
        runs_root = Path(runs_rel)
        if not runs_root.is_absolute():
            runs_root = root / runs_rel
        plan = _read_json(plan_path)
        src = _empty_tally()
        src.update({"plan": str(plan_path), "n_train_worlds": 0, "n_runs": 0,
                    "n_descriptors": 0, "n_worlds_with_pairs": 0})
        for world in train_worlds(plan):
            t0 = float(world["t0_s"])
            descs: dict = {}
            for branch in world.get("branches", []):
                run = branch.get("run")
                run_dir = (runs_root / run) if run else None
                desc = branch_descriptor(branch, t0, run_dir,
                                         n_segments=n_segments,
                                         segment_s=segment_s)
                descs.setdefault(desc, None)
                src["n_runs"] += 1
            uniq = [d for d in descs]
            src["n_train_worlds"] += 1
            src["n_descriptors"] += len(uniq)
            if len(uniq) < 2:
                continue
            src["n_worlds_with_pairs"] += 1
            for da, db in combinations(uniq, 2):
                diff = schedule_difference(da, db, n_segments=n_segments,
                                           segment_s=segment_s)
                if not diff.differs:
                    continue
                _tally(src, diff)
                _tally(summary, diff)
        sources[name] = src
        summary["by_source"][name] = src["by_duration"]
        summary["n_train_worlds"] += src["n_train_worlds"]
        summary["n_runs"] += src["n_runs"]
        summary["n_descriptors"] += src["n_descriptors"]
        summary["n_worlds_with_pairs"] += src["n_worlds_with_pairs"]
    return {"schema": "wm_n1_action_representation/1",
            "root": str(root),
            "method": ("canonical planned-schedule descriptors (19 x 30 s per "
                       "CAV/group + wall hold/speed/cap); all unordered "
                       "distinct-descriptor sibling pairs per TRAIN world; "
                       "duration = onset-to-last differing span"),
            "sources": sources, "summary": summary}


def disclosure_line(counts: Mapping) -> str:
    """One disclosure line stating whole-window dominance with the counts."""
    summary = counts.get("summary", counts) if isinstance(counts, Mapping) \
        else {}
    by = dict(summary.get("by_duration", {}))
    total = int(summary.get("n_pairs", sum(int(v) for v in by.values())))
    ww = int(by.get("whole-window", 0))
    short = int(by.get("<=60s", 0))
    mid = int(by.get("60-240s", 0))
    first60 = int(summary.get("first_60_only", 0))
    pct = (100.0 * ww / total) if total else float("nan")
    return (
        f"Disclosure: the TRAIN planned-schedule comparison set is dominated "
        f"by whole-window actions ({ww}/{total} = {pct:.1f}% whole-window); "
        f"short commitments are {short} pair(s) at <=60 s and {mid} at "
        f"60-240 s, and {first60} pair(s) differ solely in the first 60 s.  "
        f"The per-CAV switching schedules and the J4 wall arms are the only "
        f"source of start/stop variation; registration and report must state "
        f"this.")


def _md_table(title: str, headers, rows) -> list:
    out = [f"### {title}", "", "| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out.append("")
    return out


def write_representation_report(out_dir, counts: Mapping, *,
                                stem: str = "ACTION_REPRESENTATION") -> tuple:
    """Write ``ACTION_REPRESENTATION.json`` and ``.md``; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(counts)
    report["generated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    report["disclosure"] = disclosure_line(counts)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=1, default=float),
                         encoding="utf-8")

    summary = counts.get("summary", {})
    by = dict(summary.get("by_duration", {}))
    md = ["# ACTION_REPRESENTATION", "",
          report["disclosure"], "",
          f"Schema `{counts.get('schema')}`; generated "
          f"{report['generated_utc']}.", "",
          f"Method: {counts.get('method')}.", "",
          "## TRAIN schedule-pair strata (aggregate)", ""]
    md += _md_table("Duration", ["stratum", "pairs"],
                    [[s, int(by.get(s, 0))] for s in DURATION_STRATA]
                    + [["first-60-only", int(summary.get('first_60_only', 0))]])
    onset = dict(summary.get("by_onset", {}))
    md += _md_table("Onset", ["bin", "pairs"],
                    [[s, int(onset.get(s, 0))] for s in ONSET_BINS])
    rows = []
    for name in sorted(counts.get("sources", {})):
        src = counts["sources"][name]
        bd = src.get("by_duration", {})
        rows.append([name, src.get("n_train_worlds", 0), src.get("n_runs", 0),
                     src.get("n_descriptors", 0), src.get("n_pairs", 0),
                     bd.get("<=60s", 0), bd.get("60-240s", 0),
                     bd.get("whole-window", 0), src.get("first_60_only", 0)])
    md += _md_table(
        "Per source",
        ["source", "TRAIN worlds", "runs", "distinct descriptors", "pairs",
         "<=60s", "60-240s", "whole-window", "first-60-only"], rows)
    md += _md_table(
        "Duration x onset (aggregate)", ["stratum", "pairs"],
        [[k, v] for k, v in sorted(
            dict(summary.get("by_duration_onset", {})).items())])
    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path


def representation_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="evalh.idea_a representation")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                        / "reports"))
    p.add_argument("--root", default=str(RESEARCH_DENSE_ROOT))
    args = p.parse_args(argv)
    counts = representation_counts(root=args.root)
    json_path, md_path = write_representation_report(args.out, counts)
    print(disclosure_line(counts))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def heldout_config_main(argv=None) -> int:
    """Registered held-out-strategy config check (A3.4), no data read.

    Held-out test (i) evaluates J4B (the J4 wall panel re-run under wall law
    v2.2, ``REGISTRATION_J4B.md``); J4B outcome reads are gated by that
    registration and allowed ONLY for config (i)
    (:func:`assert_j4b_read_allowed`).  This command validates the requested
    config and writes a small receipt; it reads no CAL/HOLDOUT/chunk-5/J4B
    bytes and consumes no seal marker.  The chunk-5 sealed final TEST is a
    separate ``evalh.run_eval --split final_test`` run.
    """
    p = argparse.ArgumentParser(prog="evalh.idea_a heldout-config")
    p.add_argument("--test", required=True, choices=("i", "ii", "iii"))
    p.add_argument("--out", required=True)
    p.add_argument("--predictor",
                   default="adapter.predictor:IdeaABundlePredictor")
    p.add_argument("--reference",
                   default="evalh.climatology:ClimatologyPredictor")
    args = p.parse_args(argv)
    tests = {t.name: t for t in held_out_protocol()}
    test = tests[args.test]
    if test.kind == "headline":
        assert_j4b_read_allowed(test)          # the registered (i) guard
    receipt = {
        "schema": "wm_n1_heldout_config/1",
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test": test.name,
        "kind": test.kind,
        "train_filter": test.train_filter,
        "eval_filter": test.eval_filter,
        "reads_j4b": bool(test.reads_j4b),
        "predictor": args.predictor,
        "reference": args.reference,
        "note": ("config check only; J4B outcome reads are gated by "
                 "REGISTRATION_J4B (LF 2463f5be) and read ONLY by held-out "
                 "test (i); chunk-5 is the separate sealed final TEST"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "HELDOUT_CONFIG.json"
    path.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(f"heldout-config test={test.name} reads_j4b={test.reads_j4b}")
    print(f"  train_filter={test.train_filter}")
    print(f"  eval_filter={test.eval_filter}")
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="evalh.idea_a")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("representation",
                   help="a-priori TRAIN schedule representation analysis")
    sub.add_parser("heldout-config",
                   help="registered A3.4 held-out config check (no data read)")
    args, rest = p.parse_known_args(argv)
    if args.command == "representation":
        return representation_main(rest)
    if args.command == "heldout-config":
        return heldout_config_main(rest)
    p.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
