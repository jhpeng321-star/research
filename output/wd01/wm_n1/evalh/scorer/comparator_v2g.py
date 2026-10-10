"""Part A v2g baseline binding + delay predictor (WM-N1 scorer contract 4).

Registered baseline (REGISTRATION_WM_N1.md sec 2, lines 38-41): the hybrid T
is "Part A v2g, frozen nominal (params 079a5576...)" plus the Part B v0.4a
tree; the v2g rollout supplies only the STEP dynamics, never a delay head.

The two delay definitions, side by side
---------------------------------------
LABEL ``total_delay_ref_s`` (Part B v0.4a outcome producer):

    Total delay over ``[t, t + tau]`` measured Monte-Carlo on the LOGGED
    SUMO campaign trajectory under the arm's policy (part4 delay library
    conventions, prefix-sum equivalence tested;
    E:/research-dense/output/wd01_partB_v0data/DATA_CARD_V0.md:11-13).
    It is an ALL-vehicle delay: labels include ENTRY WAITING from the
    scheduled departures inside the window (DATA_CARD_V0.md:40-41;
    WM_N1_DESIGN.md:85-86 "total delay (all vehicles, including
    arrivals)"; PARTB_COUPLING_NOTE_V2GENERAL.md:7-16 puts the
    arrivals term explicitly on Part B's side).  The column is reused
    read-only from the v0.4a parquets
    (features/outcomes.py:48-53 TARGET_COLUMNS) and bound as
    ``y_delay_s`` at evalh/real_source.py:314.

PREDICTOR (Part A v2g rollout started at the t0 snapshot):

    ``evalh.scorer.actions.rollout_with_schedule(..., integrator="v2g",
    residual_enabled=False)`` = the registered v2g baseline mode
    (inc1eval/rollout_baseline.py:19-37): deterministic IDM/MOBIL/merge
    surrogate with ARRIVALS OFF and no insertions ("no vehicle is ever
    created", partA_v2g/engine.py:19-20), finite horizon, no outcome
    head.  Any delay it can form covers only the TRACKED (t0-present)
    roster (REGISTRATION_C2L1_V2G_DRAFT.md:26 "part_a_scope =
    'tracked'"; :185-186 flag (c): "The untracked-arrival remainder is
    Part B's job").

VERDICT: NOT EXACTLY COMPUTABLE.  The label counts (i) vehicles that
ENTER after t0 -- absent from the arrivals-off rollout -- and (ii) the
SUMO ground-truth realization, while the rollout is a deterministic
surrogate world.  No registration or doc defines a v2g-to-``total_
delay_ref_s`` mapping (the INC1 predictor's trajectory delay,
inc1eval/predictor.py:71-72, is explicitly NOT the SUMO truth,
INC1_EVAL.md:128-129; the draft C2L1 registration was never frozen nor
run).  By R8 the predictor therefore FAILS CLOSED: constructing it
without an explicit owner-selected ``option`` raises
:class:`ComparatorUnavailable` (no native default, no zero/oracle
substitution).  Owner options:

(a) ``option="trajectory_window_delay"`` -- implemented here: run the
    registered v2g rollout over the item's tau window and report the
    tracked-roster trajectory delay (sum of per-step lost time
    ``dt * clip(1 - v / segment_limit, 0, 1)``, the INC1 convention),
    i.e. a DIFFERENT estimand than the label (tracked vs all-vehicle +
    entry waiting; surrogate vs SUMO realization).  Selecting it is an
    owner-visible approximation decision, recorded by construction.
(b) ``option="hybrid_correction_head"`` -- NOT implemented: push the
    v2g trajectory through model A's correction/outcome head; that is
    the registered hybrid T's delay model in spirit but is NOT the pure
    baseline and needs the checkpoint bundle; raises
    :class:`ComparatorUnavailable` pending owner decision.
(c) anything else (e.g. a frozen Part B tree bundle as the T delay
    model, or a new registered producer) -- propose via the owner.

The v2g rollout is deterministic, so the implemented option returns
DEGENERATE quantiles: all 19 levels of a row equal its point delay.
For a point-mass predictive distribution CRPS degenerates to the
absolute error |prediction - label| (exact, not an approximation).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS

REGISTERED_PARAMS_LF_SHA256 = (
    "079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7")
DT_S = 0.5

OPTION_TRAJECTORY_WINDOW = "trajectory_window_delay"
OPTION_HYBRID_HEAD = "hybrid_correction_head"
_DELAY_OPTIONS = (OPTION_TRAJECTORY_WINDOW, OPTION_HYBRID_HEAD)

_UNAVAILABLE_MESSAGE = (
    "the pure Part A v2g rollout cannot reproduce the label "
    "total_delay_ref_s: the label is the all-vehicle SUMO-logged total "
    "delay over [t, t+tau] including entry waiting from scheduled "
    "departures (DATA_CARD_V0.md:11-13,40-41), while the registered v2g "
    "baseline (rollout2 integrator='v2g', residual off, arrivals off, "
    "inc1eval/rollout_baseline.py:19-37) never inserts vehicles "
    "(partA_v2g/engine.py:19-20) and covers only the tracked t0 roster "
    "(REGISTRATION_C2L1_V2G_DRAFT.md:26,185-186); no registered v2g-to-"
    "delay mapping exists (the registered hybrid T takes its delay from "
    "the Part B tree, REGISTRATION_WM_N1.md:38-41). Owner decision "
    "required: (a) option='trajectory_window_delay' for the tracked-"
    "roster trajectory delay approximation (different estimand, "
    "documented); (b) option='hybrid_correction_head' (not implemented; "
    "v2g trajectory through model A's outcome head = hybrid, not the "
    "pure baseline); (c) register another producer/mapping.")


class ComparatorUnavailable(RuntimeError):
    """The pure v2g delay baseline is unavailable without an owner decision."""


class V2GArtifactRefusal(ComparatorUnavailable):
    """The frozen v2g artifacts are absent or differ from their pins."""


def lf_sha256(path) -> str:
    """SHA-256 of the file's bytes after LF normalisation (CRLF/CR -> LF)."""
    raw = Path(path).read_bytes()
    return hashlib.sha256(
        raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def bind_v2g_artifacts(root, *,
                       expected_params_lf_sha256=REGISTERED_PARAMS_LF_SHA256,
                       expected_engine_sha256=None) -> dict:
    """Hash-bind the frozen v2g params and engine before any use.

    ``root`` is the ``partA_v2g`` directory containing ``params_nominal
    .json`` and ``engine.py``.  The params file is hashed LF-normalised
    and must equal ``expected_params_lf_sha256`` (the registered
    079a5576... pin by default); the engine is hashed raw and must equal
    ``expected_engine_sha256`` when that pin is supplied.  Any absence
    or mismatch raises :class:`V2GArtifactRefusal`.  Returns
    ``{root, params_path, params_lf_sha256, engine_path, engine_sha256,
    params}`` with the parsed params mapping.
    """
    root = Path(root)
    params_path = root / "params_nominal.json"
    engine_path = root / "engine.py"
    if not params_path.is_file():
        raise V2GArtifactRefusal(f"v2g params file is absent: {params_path}")
    if not engine_path.is_file():
        raise V2GArtifactRefusal(f"v2g engine file is absent: {engine_path}")
    params_lf = lf_sha256(params_path)
    if params_lf != str(expected_params_lf_sha256):
        raise V2GArtifactRefusal(
            f"v2g params LF-sha mismatch: registered/expected "
            f"{expected_params_lf_sha256}, observed {params_lf} at "
            f"{params_path}")
    engine_sha = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if expected_engine_sha256 is not None \
            and engine_sha != str(expected_engine_sha256):
        raise V2GArtifactRefusal(
            f"v2g engine sha mismatch: expected {expected_engine_sha256}, "
            f"observed {engine_sha} at {engine_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise V2GArtifactRefusal(f"v2g params must be a JSON object: "
                                 f"{params_path}")
    return {"root": root, "params_path": params_path,
            "params_lf_sha256": params_lf, "engine_path": engine_path,
            "engine_sha256": engine_sha, "params": params}


class V2GBaselinePredictor:
    """The Part A v2g baseline as a quantile predictor (fail-closed).

    ``artifacts`` is a ``bind_v2g_artifacts`` record (or the partA_v2g
    root, bound here).  ``option`` is REQUIRED with no default: the pure
    v2g rollout has no registered mapping to ``total_delay_ref_s``, so
    construction without an explicit owner-selected option raises
    :class:`ComparatorUnavailable`.  Only
    ``option="trajectory_window_delay"`` is implemented; it needs a
    ``resolve_window(item) -> (bundle, world, schedule)`` callable and,
    unless ``rollout_fn`` is injected (tests), calls
    ``evalh.scorer.actions.rollout_with_schedule`` lazily with
    ``integrator="v2g"``, ``residual_enabled=False``.  ``predict_
    quantiles(items)`` returns ``[N, 19]`` degenerate quantiles
    (``QUANTILE_LEVELS``): every level of a row equals that row's
    trajectory delay, so CRPS = absolute error.
    """

    name = "v2g"
    quantile_levels = QUANTILE_LEVELS

    def __init__(self, artifacts, *, option=None, resolve_window=None,
                 rollout_fn=None, dt_s: float = DT_S, seed: int = 0):
        # ``option=None`` is the fail-closed sentinel: omitting the owner
        # decision raises ComparatorUnavailable; there is no default path.
        if option is None:
            raise ComparatorUnavailable(_UNAVAILABLE_MESSAGE)
        if option not in _DELAY_OPTIONS:
            raise ComparatorUnavailable(
                f"unknown v2g delay option {option!r}; registered choices "
                f"are {list(_DELAY_OPTIONS)}")
        if option == OPTION_HYBRID_HEAD:
            raise ComparatorUnavailable(
                "option 'hybrid_correction_head' is not implemented and "
                "is not a pure baseline (v2g trajectory through model A's "
                "correction/outcome head = the hybrid T's delay model, "
                "REGISTRATION_WM_N1.md:38-41); an owner decision and the "
                "checkpoint bundle are required before it can exist")
        if isinstance(artifacts, dict):
            known = {"root", "params_path", "params_lf_sha256",
                     "engine_path", "engine_sha256", "params"}
            if set(artifacts) < known:
                raise V2GArtifactRefusal(
                    f"artifacts record lacks keys {sorted(known - set(artifacts))}")
            if lf_sha256(artifacts["params_path"]) \
                    != artifacts["params_lf_sha256"]:
                raise V2GArtifactRefusal(
                    "v2g params changed after bind_v2g_artifacts; refusing "
                    f"{artifacts['params_path']}")
            self.artifacts = artifacts
        else:
            self.artifacts = bind_v2g_artifacts(artifacts)
        if resolve_window is None:
            raise ComparatorUnavailable(
                "option 'trajectory_window_delay' needs a "
                "resolve_window(item) -> (bundle, world, schedule) "
                "callable wired to the real CAL t0 snapshot; refusing to "
                "guess one")
        self.option = option
        self._resolve_window = resolve_window
        self._rollout_fn = rollout_fn
        self.dt_s = float(dt_s)
        self.seed = int(seed)
        self._engine_mod = None

    # ----------------------------------------------------------- helpers
    def _engine_module(self):
        """The frozen partA_v2g engine module, lazily loaded by path."""
        if self._engine_mod is None:
            path = Path(self.artifacts["engine_path"])
            spec = importlib.util.spec_from_file_location(
                "wm_n1_scorer_frozen_v2g_engine", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "segment_limits"):
                raise V2GArtifactRefusal(
                    f"frozen v2g engine lacks segment_limits: {path}")
            self._engine_mod = module
        return self._engine_mod

    def _window_delay(self, rollouts, tau_s: float) -> float:
        """Tracked-roster trajectory delay (veh-s) over the tau window."""
        pos = np.asarray(rollouts.positions, dtype=np.float64)
        spd = np.asarray(rollouts.speeds, dtype=np.float64)
        msk = np.asarray(rollouts.masks).astype(bool)
        lan = np.asarray(rollouts.lanes)
        if pos.ndim == 4:
            if pos.shape[0] != 1:
                raise ComparatorUnavailable(
                    f"v2g rollout returned {pos.shape[0]} samples; the "
                    "deterministic baseline pins n_samples=1")
            pos, spd, msk, lan = pos[0], spd[0], msk[0], lan[0]
        if pos.ndim != 3 or pos.shape[1] != 1:
            raise ComparatorUnavailable(
                f"v2g rollout must cover exactly one world per item, got "
                f"shape {pos.shape}")
        steps = int(round(float(tau_s) / self.dt_s))
        if steps <= 0 or abs(float(tau_s) - steps * self.dt_s) > 1e-6:
            raise ComparatorUnavailable(
                f"tau {tau_s!r} s is not a positive multiple of the "
                f"{self.dt_s} s physics step")
        if pos.shape[0] < steps + 1:
            raise ComparatorUnavailable(
                f"v2g rollout has {pos.shape[0]} snapshots, shorter than "
                f"the tau={tau_s} s window needs ({steps + 1})")
        pos, spd, msk, lan = pos[:steps + 1], spd[:steps + 1], \
            msk[:steps + 1], lan[:steps + 1]
        roster = msk[0, 0]
        if not roster.any():
            raise ComparatorUnavailable(
                "v2g rollout has an empty t0 roster; refusing a zero-delay "
                "fallback")
        engine = self._engine_module()
        total = 0.0
        for t in range(1, steps + 1):
            live = roster & msk[t, 0]
            if not live.any():
                continue
            limits = np.asarray(
                engine.segment_limits(lan[t, 0, live], pos[t, 0, live],
                                      self.artifacts["params"]),
                dtype=np.float64)
            if not np.isfinite(limits).all() or (limits <= 0.0).any():
                raise ComparatorUnavailable(
                    "nonfinite/nonpositive segment limit in the v2g "
                    f"trajectory delay at step {t}")
            total += self.dt_s * float(
                np.clip(1.0 - spd[t, 0, live] / limits, 0.0, 1.0).sum())
        if not math.isfinite(total):
            raise ComparatorUnavailable("v2g trajectory delay is nonfinite")
        return total

    def _rollout(self, bundle, world, schedule, tau_s: float):
        steps = int(round(float(tau_s) / self.dt_s))
        if self._rollout_fn is not None:
            return self._rollout_fn(bundle, world, schedule, n_steps=steps,
                                    integrator="v2g", residual_enabled=False,
                                    n_samples=1, seed=self.seed,
                                    t0_s=0.0)
        from evalh.scorer.actions import rollout_with_schedule
        return rollout_with_schedule(bundle, world, schedule, n_steps=steps,
                                     integrator="v2g",
                                     residual_enabled=False, n_samples=1,
                                     seed=self.seed, t0_s=0.0)

    # ----------------------------------------------------------- predict
    def predict_quantiles(self, items) -> np.ndarray:
        items = list(items)
        delays = np.empty(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            bundle, world, schedule = self._resolve_window(item)
            rollouts = self._rollout(bundle, world, schedule,
                                     float(item.tau_s))
            delays[i] = self._window_delay(rollouts, float(item.tau_s))
        q = np.repeat(delays[:, None], N_QUANTILES, axis=1)
        if q.shape != (len(items), N_QUANTILES) \
                or not np.isfinite(q).all() \
                or not (np.diff(q, axis=1) >= 0.0).all():
            raise ComparatorUnavailable(
                f"v2g predictor produced an invalid quantile curve {q.shape}")
        return q
