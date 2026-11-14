"""P7 Idea A trainer extension (amendment 3 section A3.2).

Extends the existing ``train`` package (``TrainConfig``, ``MemmapCorpus``)
without editing its files.  The five Idea A losses:

* one-step HDV residual NLL (excess form, see :meth:`IdeaATrainer.residual_nll`);
* multi-step rollout loss (differentiable rollout through the HDV residual,
  CAVs teacher-forced to the recorded controller trajectory);
* arrival negative-binomial loss;
* correction-head CRPS;
* paired-branch CONTRAST loss on the HDV response and outcomes (sibling arms
  at the same world + t0, treatment minus reference).

Training is member-1-first; a partial ensemble is any ``k`` in ``1..5`` and is
recorded in the checkpoint (``partial: true`` when ``k < 5``); the final-test
path (:func:`require_full_ensemble`) refuses ``k < 5``.

Canonical batch contract (the trainer consumes P2/P4/P5 ONLY through their
INTERFACES_V2 signatures; tests and the pre-flight pass fakes)
---------------------------------------------------------------------------
``kin`` [B,K,KIN_F] float, ``kin_mask`` [B,K] bool, ``target_accel`` [B,K],
``target_mask`` [B,K]; optional ``is_cav`` [B,K] (used ONLY to select HDV
loss slots - never a net input), ``baseline_idm`` [B,K], ``lane_target``
[B,K] int (-1 = none); ``counter`` [B,2,30], ``arrivals`` [B,2];
``summary_raw`` [B,SUMMARY_RAW_F], ``global_token`` [B,D],
``outcome_quantiles`` [B,N_TAU,N_TGT], ``outcome_mask`` [B,N_TAU];
optional verified ``vehicle_ids`` [B,K] int64 (canonical sidecar IDs, -1 =
absent slot; loss-slot alignment ONLY - never a net input); ``meta``
list[B] with ``pair_key``/``arm``/``policy``; rollout keys
``teacher`` [T+1,B,K,4] ``(s, lane, v, a)`` (the recorded trajectory, CAV
slots teacher-forced), ``rollout_accel`` [T,B,K] recorded HDV accelerations,
optional ``rollout_baseline_idm`` [T,B,K] / ``rollout_mask`` [T,B,K], and the
t0 world ``s0/v0/a0/lane0/length0/present0`` [B,K].

Held-out dry-run (amendment A3.4) data filters: :func:`held_out_split`,
:func:`held_out_corpus_view` and :func:`held_out_dry_run` build the
(i) J4->J4B, (ii) leave-PI-out and (iii) leave-highest-J4-wall-speed-out
train/eval splits with the excluded arms ABSENT from the training view.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from model import constants as C
from model import losses as L
from evalh import cal_split
from evalh import seals as _seals
from train import contrast_support as CS
from train.dataset import PairAwareBatchSampler

MEMBER_ORDER = (1, 2, 3, 4, 5)
DEFAULT_SEEDS = (101, 202, 303, 404, 505)
CHECKPOINT_FORMAT = "wm_n1_idea_a_member/1"
DT_S = C.DT_S

# teacher [T+1, B, K, 4] column order (rollout2.engine docstring: s, lane, v, a)
TEACHER_S, TEACHER_LANE, TEACHER_V, TEACHER_A = 0, 1, 2, 3

# namespaced module prefixes inside the flat checkpoint state_dict
MODULE_PREFIXES = ("hdv.", "arrival.", "corr.")
#: prefix -> IdeaABundle attribute (``state.`` is the M1FIX2 state encoder)
_PREFIX_ATTR = {"hdv.": "hdv", "arrival.": "arrival", "corr.": "corr",
                "state.": "state_enc", "paired.": "paired_effect"}


def _bundle_module(bundle, prefix: str):
    return getattr(bundle, _PREFIX_ATTR.get(prefix, prefix[:-1]), None)

HELD_OUT_TESTS = ("i_j4_to_j4b", "ii_leave_pi_out",
                  "iii_leave_top_wall_speed_out")


class IdeaAConfigError(ValueError):
    """An Idea A training config or request violates the contract."""


@dataclass
class IdeaATrainConfig:
    """Idea A training + architecture config (INTERFACES_V2 config keys)."""

    # --- architecture ---
    hdv_d_model: int = 128
    hdv_n_heads: int = 4
    hdv_n_layers: int = 3
    hdv_ffn: int = 512
    hdv_use_cav_flag: bool = False          # ablation only
    corr_summary_dim: int = 64
    arrival_counter_channels: int = 128
    # --- schedule (member-1-first) ---
    member: int = 1
    n_members: int = 5
    seeds: tuple = DEFAULT_SEEDS
    ensemble_k: int = 5                     # partial ensemble size in 1..5
    epochs: int = 15
    steps_per_epoch: int = 2000
    batch_size: int = 8
    # Pair-dense batching (T2): when set to K >= 1 every batch drawn from the
    # eligible sibling-pair pool holds exactly K complete pairs (T2 default
    # None keeps the historical one-completed-pair-per-batch path unchanged).
    pairs_per_batch: Optional[int] = None
    # A8 batch fix: fold the coverage items into the pair batches so every
    # pair-dense batch keeps K pairs (default False = T2 trailing coverage).
    merge_coverage_into_pairs: bool = False
    lr: float = 1e-3
    grad_clip: float = 1.0
    # --- losses (A3.2) ---
    lambda_residual_nll: float = 1.0
    lambda_rollout: float = 1.0
    lambda_arrival: float = 1.0
    lambda_correction_crps: float = 1.0
    lambda_response_contrast: float = 1.0
    #: Pair-dense contrast sampling (T6 fix).  When True the eligible
    #: sibling-pair pool keeps ONLY pairs sharing at least one realised
    #: outcome tau, so the pair-aware sampler forces an outcome-carrying
    #: sibling pair into every batch instead of an HDV-only pair (which
    #: produced the ``out=0/0`` steps).  Default False preserves the
    #: historical outcome-OR-HDV eligibility.
    contrast_require_outcome_support: bool = False
    #: T6 follow-up (combined A8): give the OUTCOME sub-term of the paired
    #: response contrast its own explicit weight instead of letting the
    #: ``sum(terms)/len(terms)`` average halve it whenever the HDV sub-term is
    #: present (it is, on every identified-source step).  ``None`` (default,
    #: OFF) keeps the historical equal average byte-for-byte.  A float ``w``
    #: combines the present sub-terms as ``w * outcome + 1.0 * hdv`` (so
    #: ``w=1.0`` removes the halving and ``w=2.0`` doubles the outcome
    #: sub-term); a lone sub-term is returned unchanged.
    contrast_outcome_weight: Optional[float] = None
    # --- rollout / teacher forcing ---
    #: Rollout horizon AND the window the P5 correction summary is built over.
    #: The single source is ``model.constants.CORRECTION_WINDOW_STEPS``; the
    #: evaluator / scoring API slice their longer rollouts to the same window.
    rollout_horizon_steps: int = C.CORRECTION_WINDOW_STEPS
    teacher_forcing_cavs: bool = True
    # --- M1FIX2: state global token + normalization + batching ---
    state_global_token: bool = True         # False reproduces the zeros path
    outcome_norm_min_scale: float = 1e-3
    loss_balance_batches: int = 4           # first-N batches for the receipt
    # per-step telemetry: write <out>/train_log.jsonl every step and print to
    # stdout every log_every steps (1 = every step; 0 disables the print).
    log_every: int = 10
    # num_workers: 0 keeps the in-process path; None -> min(8, cpu-2).  The
    # dataclass default stays 0 so unit/preflight fits do not spawn workers;
    # the box CLI passes None to get the host-scaled default.
    num_workers: Optional[int] = 0
    prefetch_factor: int = 2
    # --- quick-tune (CAL-only improvement loop; same seed) ---
    quick_tune: bool = False
    subset_frac: float = 1.0
    epoch_frac: float = 1.0
    # --- io ---
    device: str = "cpu"
    out_dir: str = "runs/wm_n1_idea_a"
    resume: Optional[str] = None
    # --- format K_max (amendment 5): None = take the pack's K_max; an
    # explicit value (128 v2 / 192 v3) refuses a mismatching pack ---
    k_max: Optional[int] = None
    # TEST-ONLY: allow a corpus run to train with unavailable A3.2 terms.  The
    # flag and its active-term list are recorded in TRAIN_RECEIPT.json; a real
    # run must leave it False so a dropped term fails closed.
    allow_incomplete_losses: bool = False
    # --- optional climatology-anchored paired-effect head (default OFF) ---
    #: When True the bundle carries a :class:`model.paired_effect.
    #: PairedEffectHead` trained with the pinball loss on outcome-supported
    #: sibling contrast pairs.  Default OFF leaves every existing path
    #: byte-for-byte unchanged.
    paired_effect_head: bool = False
    #: Weight of the paired-effect pinball term (recorded in the receipt).
    paired_effect_weight: float = 1.0
    #: TRAIN-derived climatology JSON the head anchors on (required when the
    #: head is enabled).
    paired_effect_climatology: Optional[str] = None
    #: Optional SHA-256 pin of the climatology bytes (recorded/verified).
    paired_effect_climatology_sha256: Optional[str] = None
    #: A9 PLAN-AWARE head (default OFF): the residual trunk additionally reads
    #: the TREATMENT arm's CAV-motion summary (``summary_raw``) through a small
    #: learned plan embedding, so ``dV`` can depend on the plan (arm) taken.
    #: Requires ``paired_effect_head``; the zero-init gate keeps an untrained
    #: head exactly at the climatology.
    paired_effect_plan_aware: bool = False
    #: A11 FULL-HORIZON plan input (default None = the historical 10 s / 20-step
    #: ``summary_raw``).  When set, the plan-aware head consumes the treatment
    #: arm's planned CAV-motion summary over ``[t0, t0 + H]`` (H seconds, binned
    #: in 10 s bins) through its ``plan_proj`` (a new input width
    #: ``long_plan_raw_f(H)``).  Requires ``paired_effect_head`` AND
    #: ``paired_effect_plan_aware``.
    paired_effect_plan_horizon_s: Optional[float] = None
    #: A11 within-decision pairwise RANKING loss weight (default 0.0 = OFF).
    #: For each decision group with >= 2 outcome-supported treatment arms in the
    #: batch, a logistic loss on the head's predicted mean dV orders the arms by
    #: realised delay (lower delay must rank below higher delay), per tau and
    #: scaled by ``s_tau``; ties are ignored.  Finite and O(1).
    paired_effect_rank_weight: float = 0.0
    #: A11 group-aware pair-dense batching (default OFF): draw the eligible
    #: ``pairs_per_batch`` pairs grouped by ``pair_key`` so multiple treatment
    #: arms of one decision co-occur in a batch (needed for the ranking loss on
    #: real data).  Requires ``pairs_per_batch`` to be set.
    paired_effect_group_batching: bool = False

    def __post_init__(self) -> None:
        if not 1 <= int(self.member) <= int(self.n_members):
            raise IdeaAConfigError(f"member must be in 1..{self.n_members}")
        if not 1 <= int(self.ensemble_k) <= int(self.n_members):
            raise IdeaAConfigError("ensemble_k must be in 1..5")
        if self.hdv_use_cav_flag:
            raise IdeaAConfigError(
                "hdv_use_cav_flag=True is the ablation only; the headline "
                "model has no CAV flag (A3.1)")
        if not 0.0 < float(self.subset_frac) <= 1.0:
            raise IdeaAConfigError("subset_frac must be in (0, 1]")
        if not 0.0 < float(self.epoch_frac) <= 1.0:
            raise IdeaAConfigError("epoch_frac must be in (0, 1]")
        if self.num_workers is not None and int(self.num_workers) < 0:
            raise IdeaAConfigError("num_workers must be >= 0 (or None)")
        if int(self.loss_balance_batches) < 1:
            raise IdeaAConfigError("loss_balance_batches must be >= 1")
        if self.pairs_per_batch is not None:
            k = int(self.pairs_per_batch)
            if k < 1:
                raise IdeaAConfigError("pairs_per_batch must be >= 1 (or None)")
            if 2 * k > int(self.batch_size):
                raise IdeaAConfigError(
                    f"pairs_per_batch {k} needs 2*K = {2 * k} batch slots but "
                    f"batch_size is {int(self.batch_size)}")
        if self.contrast_outcome_weight is not None:
            w = float(self.contrast_outcome_weight)
            if not math.isfinite(w) or w < 0.0:
                raise IdeaAConfigError(
                    "contrast_outcome_weight must be a finite value >= 0 "
                    "(or None to keep the historical equal average)")
        if self.k_max is not None:
            from model import constants as C

            if int(self.k_max) not in C.FORMAT_K_MAX.values():
                raise IdeaAConfigError(
                    f"k_max {self.k_max} is not a registered format K "
                    f"({sorted(C.FORMAT_K_MAX.values())})")
        if not math.isfinite(float(self.paired_effect_weight)) \
                or float(self.paired_effect_weight) < 0.0:
            raise IdeaAConfigError(
                "paired_effect_weight must be a finite value >= 0")
        if self.paired_effect_plan_aware and not self.paired_effect_head:
            raise IdeaAConfigError(
                "paired_effect_plan_aware requires paired_effect_head; the "
                "plan-aware residual is a property of the paired-effect head")
        if self.paired_effect_plan_horizon_s is not None:
            h = float(self.paired_effect_plan_horizon_s)
            if not math.isfinite(h) or h <= 0.0:
                raise IdeaAConfigError(
                    "paired_effect_plan_horizon_s must be a finite value > 0 "
                    "(or None to keep the 10 s summary_raw plan)")
            if not self.paired_effect_head or not self.paired_effect_plan_aware:
                raise IdeaAConfigError(
                    "paired_effect_plan_horizon_s requires both "
                    "paired_effect_head and paired_effect_plan_aware; a "
                    "full-horizon plan is an input width of the plan-aware "
                    "paired-effect head")
        if not math.isfinite(float(self.paired_effect_rank_weight)) \
                or float(self.paired_effect_rank_weight) < 0.0:
            raise IdeaAConfigError(
                "paired_effect_rank_weight must be a finite value >= 0")
        if self.paired_effect_group_batching and self.pairs_per_batch is None:
            raise IdeaAConfigError(
                "paired_effect_group_batching requires pairs_per_batch; the "
                "group-aware reordering only applies to pair-dense batching")

    def effective_epochs(self) -> int:
        """Epochs actually run: ``epoch_frac``-scaled when quick-tune."""
        total = max(1, int(self.epochs))
        if not self.quick_tune or float(self.epoch_frac) >= 1.0:
            return total
        return max(1, int(round(float(self.epoch_frac) * total)))

    def resolved_num_workers(self) -> int:
        """DataLoader worker count: explicit value or ``min(8, cpu-2)``."""
        if self.num_workers is not None:
            return max(0, int(self.num_workers))
        import os

        cpus = os.cpu_count() or 2
        return max(0, min(8, int(cpus) - 2))

    def seed_for_member(self, member: int) -> int:
        if not 1 <= int(member) <= len(self.seeds):
            raise IdeaAConfigError(f"member {member} has no registered seed")
        return int(self.seeds[int(member) - 1])

    def loss_weights(self) -> dict:
        return dict(residual_nll=self.lambda_residual_nll,
                    rollout=self.lambda_rollout,
                    arrival=self.lambda_arrival,
                    correction_crps=self.lambda_correction_crps,
                    response_contrast=self.lambda_response_contrast,
                    paired_effect=self.paired_effect_weight,
                    paired_effect_rank=float(self.paired_effect_rank_weight))

    def model_configs(self) -> dict:
        """The three P2/P4/P5 config objects."""
        from arrival import ArrivalConfig
        from corr import CorrConfig
        from hdv import HdvConfig

        return dict(
            hdv=HdvConfig(d_model=self.hdv_d_model, n_heads=self.hdv_n_heads,
                          n_layers=self.hdv_n_layers, ffn=self.hdv_ffn,
                          use_cav_flag=self.hdv_use_cav_flag),
            arrival=ArrivalConfig(counter_channels=self.arrival_counter_channels),
            corr=CorrConfig(summary_dim=self.corr_summary_dim))

    @classmethod
    def from_dict(cls, data: dict) -> "IdeaATrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise IdeaAConfigError(f"unknown IdeaATrainConfig key(s): {unknown}")
        data = dict(data)
        if "seeds" in data:
            data["seeds"] = tuple(int(x) for x in data["seeds"])
        return cls(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = [int(s) for s in self.seeds]
        return d


@dataclass
class IdeaABundle:
    """The composed P1-P5 modules handed to the trainer/evaluator."""

    ctrl: object = None
    hdv: object = None
    arrival: object = None
    corr: object = None
    state_enc: object = None
    #: Optional :class:`model.paired_effect.PairedEffectHead` (default None;
    #: only built when ``IdeaATrainConfig.paired_effect_head`` is set).
    paired_effect: object = None
    config: IdeaATrainConfig = field(default_factory=IdeaATrainConfig)


def build_idea_a_bundle(config: Optional[IdeaATrainConfig] = None, *,
                        ctrl: object = None) -> "IdeaABundle":
    """Build the REAL P2/P4/P5 modules from an :class:`IdeaATrainConfig`.

    Integration wiring: the trainer/evaluator/queue consume P2/P4/P5 through
    their INTERFACES_V2 signatures; this factory constructs the real
    :class:`hdv.net.HdvResidualNet`, :class:`arrival.head.IdeaAArrivalHead` and
    :class:`corr.head.CorrectionHead`.  ``ctrl`` (P1) defaults to the REAL
    :class:`ctrl.mover.CavMover`; pass ``ctrl=...`` only to inject a test
    double.
    """
    from arrival import IdeaAArrivalHead
    from hdv import HdvResidualNet
    from train.g4_real import RealCorrBundle

    cfg = config or IdeaATrainConfig()
    mc = cfg.model_configs()
    if ctrl is None:
        from ctrl import CavMover, CtrlConfig

        ctrl = CavMover(CtrlConfig())
    # the canonical batch carries the 16-d raw CAV-motion summary; the real P5
    # projection (CavMotionSummary) + CorrectionHead is the composition the
    # trainer/evaluator consume (the same path as train.g4_real / evalh).
    state_enc = None
    if cfg.state_global_token:
        from corr import StateGlobalToken

        state_enc = StateGlobalToken(mc["corr"].d_model)
    paired_effect = None
    if bool(cfg.paired_effect_head):
        from model.paired_effect import (PLAN_RAW_F, PairedEffectHead,
                                         load_climatology, long_plan_raw_f)

        if not cfg.paired_effect_climatology:
            raise IdeaAConfigError(
                "paired_effect_head is enabled but no "
                "paired_effect_climatology path is configured; the head is "
                "anchored on the TRAIN-derived climatology (fail closed)")
        clim_path = cfg.paired_effect_climatology
        # A checkpoint records the TRAINING box's path; on another machine the
        # same bytes may live elsewhere.  WM_N1_PAIRED_EFFECT_CLIMATOLOGY may
        # point at a local copy, and the recorded sha256 pin is still enforced.
        override = os.environ.get("WM_N1_PAIRED_EFFECT_CLIMATOLOGY")
        if override and not Path(str(clim_path)).is_file():
            clim_path = override
        anchor, _sha, _contrast = load_climatology(
            clim_path,
            expected_sha256=cfg.paired_effect_climatology_sha256)
        plan_raw = (long_plan_raw_f(cfg.paired_effect_plan_horizon_s)
                    if cfg.paired_effect_plan_horizon_s is not None
                    else PLAN_RAW_F)
        paired_effect = PairedEffectHead(
            int(getattr(mc["corr"], "d_model", 128)),
            plan_aware=bool(cfg.paired_effect_plan_aware),
            plan_raw_dim=int(plan_raw))
        paired_effect.set_climatology(anchor)
    return IdeaABundle(
        ctrl=ctrl,
        hdv=HdvResidualNet(mc["hdv"]),
        arrival=IdeaAArrivalHead(mc["arrival"]),
        corr=RealCorrBundle(mc["corr"]),
        state_enc=state_enc,
        paired_effect=paired_effect,
        config=cfg,
    )


def ensemble_members(k: int, order: Sequence[int] = MEMBER_ORDER) -> tuple:
    """The first ``k`` members in member-1-first order (``k`` in 1..5)."""
    if not 1 <= int(k) <= len(order):
        raise IdeaAConfigError(f"ensemble k must be in 1..{len(order)}")
    return tuple(order)[:int(k)]


def ensemble_flags(k: int) -> dict:
    """Checkpoint flags for a partial ensemble of size ``k`` (1..5).

    ``k < 5`` is flagged early/internal (``partial=True``,
    ``final_test_eligible=False``); the chunk-5 TEST path must call
    :func:`require_full_ensemble`.
    """
    members = ensemble_members(k)
    return {"ensemble_k": int(k), "members": list(members),
            "partial": int(k) < len(MEMBER_ORDER),
            "final_test_eligible": int(k) == len(MEMBER_ORDER)}


def require_full_ensemble(ensemble_k) -> tuple:
    """Final-test gate: refuse any partial ensemble (``k < 5``)."""
    k = int(ensemble_k.get("ensemble_k") if isinstance(ensemble_k, dict)
            else ensemble_k)
    if k < len(MEMBER_ORDER):
        raise IdeaAConfigError(
            f"final test refuses a partial ensemble (k={k} < "
            f"{len(MEMBER_ORDER)}); train members "
            f"{list(MEMBER_ORDER)[k:]} first")
    return ensemble_members(k)


# ---------------------------------------------------------------- batching --
Batch2D_KEYS = ("kin", "kin_mask", "is_cav", "vehicle_ids", "baseline_idm",
                "target_accel", "target_mask", "lane_target", "counter",
                "arrivals", "summary_raw", "plan_long", "global_token",
                "outcome_quantiles", "outcome_mask", "collision", "s0", "v0",
                "a0", "lane0", "length0", "present0", "wall_group", "slot_id",
                "hdv_base", "event_tok", "event_mask")
BatchTime_KEYS = ("teacher", "rollout_accel", "rollout_baseline_idm",
                  "rollout_mask")

_WORLD0_REQUIRED = ("s0", "v0", "lane0", "length0", "present0")

#: the five packed P9 Idea A channels the full canonical batch is built from.
IDEA_A_PACK_ARRAYS = ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                      "event_mask")
#: ``hdv_base`` column of the one-step IDM baseline acceleration (P9 field 0).
HDV_BASE_IDM_IDX = 0
#: the five registered A3.2 loss terms a real corpus run MUST produce.
REQUIRED_LOSS_TERMS = ("residual_nll", "rollout", "arrival",
                       "correction_crps", "response_contrast")
# veh_kin column order (features.idea_a.KIN_SOURCE_INDICES).
_KIN_S, _KIN_L0, _KIN_L1, _KIN_L2, _KIN_V, _KIN_A, _KIN_LEN = range(7)


def collate_canonical(windows: Sequence[dict]) -> dict:
    """Concatenate canonical batch dicts (windows) along the batch axis.

    Time-leading rollout tensors (``teacher`` ``[T,B,...]``) concatenate on
    the batch axis (dim 1).  Optional keys must be present in every window or
    they are dropped from the merged batch (all-or-none).
    """
    windows = list(windows)
    if not windows:
        raise IdeaAConfigError("no windows to collate")
    out: dict = {}
    for key in Batch2D_KEYS:
        vals = [w[key] for w in windows if key in w]
        if len(vals) == len(windows) and vals:
            out[key] = torch.cat(vals, dim=0)
    for key in BatchTime_KEYS:
        vals = [w[key] for w in windows if key in w]
        if len(vals) == len(windows) and vals:
            out[key] = torch.cat(vals, dim=1)
    meta: list = []
    for w in windows:
        meta.extend(list(w.get("meta", [])))
    if meta:
        out["meta"] = meta
    return out


def batch_from_corpus_items(items: Sequence[dict]) -> dict:
    """Map ``MemmapCorpus`` items (v1 shard tensors) to the canonical batch.

    ``kin`` comes from the P9 ``kinematics_view`` (veh_kin columns) - NEVER
    the raw ``veh_step`` (action labels leak, INTERFACES mismatch 4).  The
    v1 corpus has no Idea A teacher trajectory, so the rollout term is
    unavailable for such batches (a fit receipt records that).
    """
    from train.dataset import collate
    from features.idea_a import kinematics_view

    batch = collate(list(items))
    veh = batch["veh_step"]
    mask = batch["veh_mask"]
    lane0 = veh[..., C.LANE_LEFT:C.LANE_ACCEL + 1].argmax(dim=-1).float()
    lane0 = torch.where(mask, lane0, torch.full_like(lane0, -1.0))
    out = {
        "kin": kinematics_view(veh),
        "kin_mask": mask,
        "is_cav": veh[..., C.IS_CAV_IDX] > 0.5,   # loss-slot selection only
        "baseline_idm": veh.new_zeros(veh.shape[:2]),
        "target_accel": batch["accel"],
        "target_mask": batch["target_mask"],
        "lane_target": batch["lane"],
        "counter": batch["counter"],
        "arrivals": batch["arrivals"],
        "outcome_quantiles": batch["outcome_quantiles"],
        "outcome_mask": batch["outcome_mask"],
        "collision": batch["collision"],
        "meta": batch["meta"],
        "s0": veh[..., C.S_DOWN],
        "v0": veh[..., C.V_IDX],
        "a0": veh[..., C.A_IDX],
        "lane0": lane0,
        "length0": veh[..., C.LENGTH_IDX],
        "present0": mask,
    }
    return out


def _collate_for_source(source):
    """The collate function matching a batching source's item type.

    A ``MemmapCorpus`` (v1) yields items carrying ``veh_step`` and needs the
    v1 -> canonical mapper; an ``IdeaACorpus``/window view already yields
    canonical windows and uses ``collate_canonical``.
    """
    try:
        first = source[0]
    except Exception:                                       # noqa: BLE001
        return collate_canonical
    return batch_from_corpus_items if "veh_step" in first else collate_canonical


def _norm_corpus_of(source):
    """The ``MemmapCorpus`` behind a batching source (or ``None``).

    ``IdeaACorpus`` wraps ``_v1``; ``HeldOutTrainView`` wraps ``corpus``; a
    plain ``MemmapCorpus`` is returned as-is.  Used only to read the packed
    TRAIN outcome rows for the normalization statistics.
    """
    if hasattr(source, "outcome_target"):
        return source
    if hasattr(source, "_v1"):
        return source._v1
    inner = getattr(source, "corpus", None)
    if inner is not None:
        return getattr(inner, "_v1", inner)
    return None


def fit_outcome_norm(source=None, *, windows=None,
                     min_scale: float = 1e-3):
    """Fit the outcome normalization from TRAIN items ONLY.

    ``windows`` (explicit canonical windows) are used directly; otherwise the
    source's packed TRAIN outcome rows are read through
    ``MemmapCorpus.outcome_target`` without building any full window.  CAL/TEST
    data is never reachable from here (the source is the fit's TRAIN source).
    """
    from model.outcome_norm import OutcomeNorm

    if windows is not None:
        if len(windows) == 0:
            return OutcomeNorm.identity()
        return OutcomeNorm.from_items(list(windows), min_scale=min_scale)
    if source is not None:
        try:
            if len(source) == 0:
                return OutcomeNorm.identity()
        except TypeError:
            pass
    corpus = _norm_corpus_of(source)
    if corpus is None or not hasattr(corpus, "outcome_target"):
        # a corpus-like stand-in (tests / quick-tune view) whose items are
        # canonical windows already
        if source is not None and hasattr(source, "__getitem__"):
            return OutcomeNorm.from_items(
                [source[i] for i in range(len(source))], min_scale=min_scale)
        raise IdeaAConfigError(
            "cannot fit outcome normalization: the source exposes no TRAIN "
            "packed outcome rows")
    targets, masks = [], []
    for i in range(len(corpus)):
        y, m, _ = corpus.outcome_target(i)
        targets.append(y)
        masks.append(m)
    return OutcomeNorm.fit(targets, masks, min_scale=min_scale)


def loss_balance_report(magnitudes, *, factor: float = 100.0,
                        at_step: int = 0) -> dict:
    """Receipt check: no term may exceed ``factor`` x the others' median.

    This WARNS only; the registered lambdas are never changed.  ``magnitudes``
    is the per-step list of ``{term: value}`` recorded on the first batches.
    """
    if not magnitudes:
        return {"at_step": at_step, "magnitudes": {}, "others_median": {},
                "ratios": {}, "balanced": None, "factor": float(factor),
                "warnings": ["no loss magnitudes recorded"]}
    row = magnitudes[max(0, min(int(at_step), len(magnitudes) - 1))]
    names = sorted(row)
    med, ratios, warnings = {}, {}, []
    for name in names:
        others = [abs(float(row[o])) for o in names if o != name]
        m = float(np.median(others)) if others else 0.0
        med[name] = m
        value = abs(float(row[name]))
        ratios[name] = (value / m) if m > 0 else (
            float("inf") if value > 0 else 1.0)
        if ratios[name] > float(factor):
            warnings.append(
                f"{name} magnitude {float(row[name]):.4g} exceeds "
                f"{float(factor):g}x the median of the other terms "
                f"({m:.4g}); lambda unchanged")
    return {"at_step": int(at_step),
            "magnitudes": {k: float(v) for k, v in row.items()},
            "others_median": med, "ratios": ratios,
            "balanced": not warnings, "factor": float(factor),
            "warnings": warnings}


def _cav_motion_summary(pos_seq: Tensor, cav_plan_seq: Tensor,
                        cav_mask: Tensor) -> Tensor:
    """``summary_raw`` ``[B, SUMMARY_RAW_F]`` from the CAV planned channels.

    The P5 correction head consumes the registered 16-feature CAV planned-motion
    summary (``corr.summary.cav_motion_summary``); the pack stores the CAV
    planned speed/accel per tick in ``cav_plan_ch`` (fields 0/1, scaled by the
    frozen :data:`features.idea_a.CAV_TGT_SPEED_SCALE` /
    :data:`features.idea_a.CAV_ACCEL_SCALE`) and the recorded positions in
    ``veh_kin``.  No new field is invented: the same deterministic summary the
    evaluator builds from a CAV rollout is rebuilt from the packed channels.
    """
    from corr.summary import cav_motion_summary
    from features.idea_a import CAV_ACCEL_SCALE, CAV_TGT_SPEED_SCALE

    # [T, B, K] (batch axis inserted), planner channels de-scaled to SI.
    speeds = cav_plan_seq[..., 0] * float(CAV_TGT_SPEED_SCALE)
    accel = cav_plan_seq[..., 1] * float(CAV_ACCEL_SCALE)
    return cav_motion_summary(pos_seq, speeds, accel, cav_mask)


def remap_canonical_ids(values, vehicle_map, lookup):
    """Pack-local canonical IDs -> one shared canonical index space.

    ``vehicle_map`` maps a pack-local canonical index to its
    ``(chunk, world, vehicle_id)`` key; ``lookup`` maps that key to a shared
    index.  ``-1`` (absent slot) stays ``-1``.  Pure and deterministic so the
    cross-pack HDV alignment contract is unit-testable without a corpus.
    """
    values = np.asarray(values, dtype=np.int64)
    out = np.empty(values.shape, dtype=np.int64)
    for j, raw in enumerate(values.tolist()):
        out[j] = -1 if raw < 0 else int(lookup[tuple(vehicle_map[int(raw)])])
    return out


class IdeaACorpus:
    """Read-only packed-corpus view yielding the FULL canonical Idea A batch.

    ``train.dataset.MemmapCorpus`` (the frozen v1 shard reader) exposes the v1
    tensors and the packed outcome rows; this view adds the five P9 Idea A
    channels (``veh_kin``/``hdv_base``/``cav_plan_ch``/``event_tok`` and
    ``event_mask``) and the consecutive-tick rollout window at the registered
    horizon, so :meth:`IdeaATrainer.fit` (and the timing/G4/G5 gates) compute
    every registered A3.2 loss term instead of silently dropping them.

    Construction FAILS CLOSED when the pack lacks any Idea A channel; the view
    returns one canonical window per item (leading batch axis 1) that
    :func:`collate_canonical` concatenates exactly like the synthetic windows.

    ``k_max`` (amendment 5) is forwarded to the frozen ``MemmapCorpus``: None
    takes the pack's own K_max, an explicit 128 (v2) / 192 (v3) refuses a
    mismatching pack, so a pack/model K mismatch refuses on the Idea A path too.

    Verified identity access (contrastfix identity recovery): for a TRAIN
    corpus, windows carry verified canonical ``vehicle_ids`` loaded lazily
    through the portable TRAIN identity sidecar
    (``train.identity_sidecar.load_identity_sidecar(pack_dir, sidecar_dir)``,
    sidecar directory from ``WM_N1_TRAIN_IDENTITIES``).  The sidecar loader
    applies strict path checks that reject any symlink/reparse ancestor - but
    a deployed GPU pack location is often an INTENTIONAL frozen alias (e.g.
    ``$WORK/pack`` -> the real upload tree).  The separately
    coordinator-validated REAL canonical target may be supplied through the
    OPTIONAL ``WM_N1_CANONICAL_PACK`` environment: ONLY when it is set, and
    ``pack_dir`` resolves (strict) to EXACTLY that directory, is the literal
    canonical target (never the alias) handed to the identity loader - whose
    own strict checks then prove the target itself is link-free.  A mismatch
    refuses BEFORE the loader reads anything; with the env unset there is NO
    automatic resolve fallback (the literal ``pack_dir`` is passed unchanged,
    so an alias still refuses).  ``pack_dir``, the corpus attributes and the
    identity receipt's ``pack_alias`` keep the ORIGINAL literal for the
    handoff gate; ``canonical_pack`` records the validated target.  Rows are
    indexed by
    the item's ORIGINAL GLOBAL manifest position - resolved here from the pack
    manifest, never from the TRAIN-filtered dataset position - and every row
    is validated against the item's HDV-valid slots: a missing, unknown or
    duplicated identity raises (fail closed, never guessed).  CAL corpora get
    no identities (the sidecar is TRAIN-only; CAL is never read here).  The
    ``identity_loader`` injection is TEST-ONLY (synthetic fixtures).
    """

    def __init__(self, pack_dir, split: str = "TRAIN", *,
                 pair_sampling: bool = False, horizon_steps: int = 20,
                 allow_single_run_fallback: bool = False,
                 k_max: Optional[int] = None,
                 identity_loader: Optional[Callable] = None,
                 extra_pack_dirs: Optional[Sequence] = None,
                 extra_overlay_dir: Optional[str] = None,
                 extra_identity_dirs: Optional[Sequence] = None,
                 extra_identity_loader: Optional[Callable] = None,
                 plan_horizon_s: Optional[float] = None) -> None:
        from train.dataset import MemmapCorpus

        pack_dir = Path(pack_dir)
        self._is_multi_pack = bool(extra_pack_dirs)
        self._extra_pack_dirs = [str(p) for p in (extra_pack_dirs or [])]
        # A11 FULL-HORIZON plan summary: when set, each window also carries a
        # fixed-size ``plan_long`` vector (the treatment arm's planned CAV
        # motion over [t0, t0 + H], binned in 10 s bins); None keeps every
        # existing path byte-for-byte unchanged (no plan_long key emitted).
        self._plan_horizon_s = (None if plan_horizon_s is None
                                else float(plan_horizon_s))
        # ``k_max`` is the FORMAT slot width (amendment 5): None takes the
        # pack's own K_max, and an explicit value (128 v2 / 192 v3) makes the
        # frozen MemmapCorpus refuse a mismatching pack before any Idea A
        # channel is read, so a K mismatch refuses on the Idea A path too.
        if extra_pack_dirs:
            # A9: item-level composite of pack_v3 + extra (dspeed/damping)
            # pack; the physical single-pack concatenation would need > the
            # available disk.  The composite keeps every MemmapCorpus
            # contract the rest of this class relies on.
            from dspeed.composite import ConcatCorpus
            self._v1 = ConcatCorpus(
                [pack_dir, *extra_pack_dirs], split=split,
                pair_sampling=pair_sampling, k_max=k_max,
                extra_overlay_dir=extra_overlay_dir)
        else:
            self._v1 = MemmapCorpus(pack_dir, split=split,
                                    pair_sampling=pair_sampling, k_max=k_max)
        arrays = set(self._v1.manifest.get("arrays", {}))
        missing = [name for name in IDEA_A_PACK_ARRAYS if name not in arrays]
        if missing:
            raise IdeaAConfigError(
                f"pack {pack_dir} lacks the Idea A channel(s) {missing}; the "
                "registered A3.2 losses (rollout / correction / contrast) "
                "cannot be built - refusing instead of silently dropping them "
                "(re-featurize with the P9 Idea A channels)")
        self.pack_dir = pack_dir
        self.split = self._v1.split
        # resolved format slot width + declared format (None when unclaimed)
        self.k_max = self._v1.k_max
        self.format = self._v1.format
        self.pair_sampling = bool(pair_sampling)
        self._items = self._v1._items
        self._order = self._v1._order
        self.horizon_steps = max(1, int(horizon_steps))
        # TEST-ONLY: allow a pack without INDEX.json to be treated as one run.
        # A real corpus run must leave this False so a missing/unparsable run
        # table fails closed instead of silently crossing run boundaries.
        self.allow_single_run_fallback = bool(allow_single_run_fallback)
        self._mm: dict = {}
        self._runs = self._load_run_ranges()
        # verified identity state (contrastfix identity recovery): the
        # sidecar (or False = unavailable), per-global-position validated ID
        # rows, and the ORIGINAL global manifest position of every dataset
        # position (resolved via the pack manifest, NOT the TRAIN position).
        self._identity_loader = identity_loader
        self._identity = None
        self._id_rows: dict = {}
        # the literal canonical pack target this corpus validated and loaded
        # against (WM_N1_CANONICAL_PACK), or None when the env is unset;
        # recorded in the identity receipt for audit alongside pack_alias
        self._canonical_pack_used: Optional[str] = None
        # A9 composite: one identity sidecar per extra pack (the pack_v3
        # sidecar never covers the extra items), plus the TEST-ONLY injected
        # extra loader.  Each pack's canonical ID index is PACK-LOCAL, so in
        # composite mode every row is re-projected into one shared index space
        # built from the union of the packs' canonical vehicle maps: two packs
        # then agree on the same physical vehicle's canonical ID.
        self._extra_identity_dirs = self._resolve_extra_identity_dirs(
            extra_identity_dirs)
        self._extra_identity_loader = extra_identity_loader
        self._extra_identities: list = [None] * len(self._extra_pack_dirs)
        self._reproject_ids = bool(self._is_multi_pack)
        self._global_lookup: Optional[dict] = None
        self._global_packs: list = []
        self._manifest_pos = self._build_manifest_position_index()
        self._global_positions = self._resolve_global_positions()

    # -- pickling (DataLoader workers) --------------------------------------
    def __getstate__(self) -> dict:
        """Drop open memmaps before pickling to a spawned worker.

        Mirrors :meth:`train.dataset.MemmapCorpus.__getstate__`: the cached
        ``np.memmap`` arrays serialize their whole content, so a spawned
        worker would re-serialize a real pack (multi-GB).  Workers re-open the
        arrays lazily; fork inherits them.  The validated per-item ID-row
        cache is dropped too (re-validated lazily from the immutable
        sidecar); the loaded sidecar itself pickles as compact bytes.
        """
        state = dict(self.__dict__)
        state["_mm"] = {}
        state["_id_rows"] = {}
        state["_global_lookup"] = None      # rebuilt lazily from the maps
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_mm", {})
        self.__dict__.setdefault("_id_rows", {})
        self.__dict__.setdefault("_global_lookup", None)

    # -- pack access --------------------------------------------------------
    def _array(self, name):
        if name not in self._mm:
            # delegate to the wrapped corpus so a multi-pack composite
            # (A9 --extra-corpus) serves the concatenated view; a plain
            # MemmapCorpus._array loads the identical bytes from pack_dir.
            self._mm[name] = self._v1._array(name)
        return self._mm[name]

    def _load_run_ranges(self) -> list:
        """Per-run ``(tick_start, tick_end)`` from the corpus ``INDEX.json``.

        The window never crosses a run boundary (the concatenated tick arrays
        continue with another run's vehicles).  This FAILS CLOSED: a real pack
        whose ``INDEX.json`` is missing, unparsable, or has any run without
        ``tick_start``/``tick_end`` raises rather than guessing a single range
        and silently windowing across a boundary.  The single-range fallback is
        reachable only through the explicit TEST-ONLY
        ``allow_single_run_fallback`` constructor flag.
        """
        if getattr(self._v1, "_multi_pack", False):
            # A9 composite: the merged per-pack run ranges (global tick
            # offsets) are authoritative, so a window never crosses a run
            # boundary in any part.
            return list(self._v1._runs)
        path = self.pack_dir / "INDEX.json"
        if not path.is_file():
            if self.allow_single_run_fallback:
                n_ticks = int(self._v1.manifest["arrays"]["veh_kin"]["shape"][0])
                return [(0, n_ticks)]
            raise IdeaAConfigError(
                f"pack {self.pack_dir} has no INDEX.json; the run "
                "(tick_start/tick_end) table is required to keep a window "
                "inside one run.  Pass allow_single_run_fallback=True only in "
                "tests (a real corpus run must fail closed)")
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise IdeaAConfigError(
                f"pack {self.pack_dir} INDEX.json is not parsable: {exc}") \
                from exc
        runs = index.get("runs") if isinstance(index, dict) else None
        records = list(runs.values()) if isinstance(runs, dict) else runs
        if not records:
            raise IdeaAConfigError(
                f"pack {self.pack_dir} INDEX.json carries no runs; the "
                "tick_start/tick_end table is required")
        ranges = []
        for rec in records:
            if not isinstance(rec, dict):
                raise IdeaAConfigError(
                    f"pack {self.pack_dir} INDEX.json run entry is not an "
                    f"object: {rec!r}")
            start, end = rec.get("tick_start"), rec.get("tick_end")
            if start is None or end is None:
                raise IdeaAConfigError(
                    f"pack {self.pack_dir} INDEX.json run {rec.get('world')!r}"
                    f"/{rec.get('arm')!r} lacks tick_start/tick_end; refusing "
                    "to guess the run boundary")
            ranges.append((int(start), int(end)))
        return sorted(ranges)

    def _run_end(self, tick: int) -> int:
        for start, end in self._runs:
            if start <= tick < end:
                return end
        return int(self._array("veh_kin").shape[0])

    # -- verified identity access (contrastfix identity recovery) -----------
    @staticmethod
    def _item_position_key(item: dict) -> tuple:
        """The manifest-unique item identity tuple used for position lookup."""
        return (str(item.get("chunk")), str(item.get("world")),
                str(item.get("arm")), float(item.get("t0_s")),
                int(item.get("tick")), int(item.get("plan_row")))

    def _build_manifest_position_index(self) -> dict:
        """``identity tuple -> ORIGINAL global manifest position`` (or None).

        ``MemmapCorpus._items`` is the SPLIT-FILTERED item list, so a dataset
        position is a TRAIN position, not the sidecar's row key.  The reliable
        mapping is rebuilt here from ``manifest["items"]`` itself; an identity
        tuple that is not unique across the manifest is marked unusable.  In
        composite mode the per-pack positions come from the ``_manifest_gpos``
        the composite stamps on each item (never the offset-shifted dataset
        position), so this base index is only consulted for a single pack.
        """
        index: dict = {}
        for gpos, raw in enumerate(self._v1.manifest.get("items") or []):
            key = self._item_position_key(raw)
            index[key] = None if key in index else gpos
        return index

    def _component_position(self, item: dict) -> tuple:
        """``(pack index, ORIGINAL per-pack manifest position)`` (strict).

        A single pack resolves the position from its own manifest.  A composite
        pack carries ``_pack`` + ``_manifest_gpos`` (set by
        :class:`dspeed.composite.ConcatCorpus`): the composite dataset ``tick``/
        ``plan_row`` are OFFSET-SHIFTED, so the manifest-position lookup MUST
        use the unshifted per-pack value, never the shifted dataset position.
        """
        pack = int(item.get("_pack", 0))
        if self._is_multi_pack or "_manifest_gpos" in item:
            gpos = item.get("_manifest_gpos")
            if gpos is None:
                raise IdeaAConfigError(
                    f"item {item.get('world')!r}/{item.get('arm')!r} does not "
                    "map to a unique manifest position in its own pack; the "
                    "sidecar identity lookup needs the ORIGINAL per-pack "
                    "manifest position, never the composite dataset position")
            return pack, int(gpos)
        gpos = self._manifest_pos.get(self._item_position_key(item))
        if gpos is None:
            raise IdeaAConfigError(
                f"pack {self.pack_dir} item {item.get('world')!r}/"
                f"{item.get('arm')!r} does not map to a unique global "
                "manifest position; the sidecar identity lookup needs "
                "the ORIGINAL manifest position, never the "
                "TRAIN-filtered dataset position")
        return 0, int(gpos)

    def _resolve_global_positions(self) -> list:
        """Dataset position -> ORIGINAL manifest position (strict, per pack).

        ``_global_packs`` is the parallel pack index (all 0 for one pack).
        """
        positions, packs = [], []
        for i in range(len(self._order)):
            pack, gpos = self._component_position(self._items[self._order[i]])
            positions.append(gpos)
            packs.append(pack)
        self._global_packs = packs
        return positions

    def _resolve_extra_identity_dirs(self, explicit) -> list:
        """One identity sidecar directory per extra pack (explicit or env)."""
        if not self._is_multi_pack:
            return []
        dirs = [str(x) for x in (explicit or []) if str(x).strip()]
        if not dirs:
            raw = os.environ.get("WM_N1_EXTRA_TRAIN_IDENTITIES")
            if raw and str(raw).strip():
                dirs = [p for p in str(raw).split(os.pathsep) if p.strip()]
        if dirs and len(dirs) != len(self._extra_pack_dirs):
            raise IdeaAConfigError(
                "the composite corpus has "
                f"{len(self._extra_pack_dirs)} extra pack(s) but "
                f"{len(dirs)} extra identity sidecar dir(s); one verified "
                "sidecar per extra pack is required (--extra-identities / "
                "WM_N1_EXTRA_TRAIN_IDENTITIES, os.pathsep-separated)")
        return dirs

    def _extra_sidecar(self, pack: int):
        """Verified sidecar for extra pack ``pack`` (1-based), or ``None``.

        Loaded lazily through :func:`dspeed.identity.load_extra_identity_sidecar`
        (which dispatches to the physics-sourced dspeed loader or the standard
        shard-sourced ``train.identity_sidecar`` loader by the receipt's
        ``source``) or the TEST-ONLY injected ``extra_identity_loader``.
        """
        if pack <= 0:
            raise IdeaAConfigError("base pack uses the pack_v3 identity sidecar")
        idx = pack - 1
        cached = self._extra_identities[idx]
        if cached is not None:
            return cached if cached is not False else None
        if self.split != "TRAIN":
            self._extra_identities[idx] = False
            return None
        if idx >= len(self._extra_identity_dirs) or \
                not self._extra_identity_dirs[idx]:
            self._extra_identities[idx] = False
            return None
        loader = self._extra_identity_loader
        if loader is None:
            try:
                from dspeed.identity import load_extra_identity_sidecar
            except ImportError as exc:
                raise IdeaAConfigError(
                    "an extra identity sidecar is configured but "
                    "dspeed.identity is not importable; deploy the verified "
                    "extra identity module beside this package") from exc
            loader = load_extra_identity_sidecar
        part = self._v1.parts[pack]
        side = loader(str(part.pack_dir), self._extra_identity_dirs[idx])
        self._extra_identities[idx] = side
        return side

    def _global_vehicle_lookup(self) -> dict:
        """One shared canonical index space over every pack's vehicle map.

        Each sidecar's canonical IDs are PACK-LOCAL; re-projecting through the
        union of the ``(chunk, world, vehicle_id)`` maps makes integer equality
        across a trajectory item and its pack_v3 native sibling identify the
        same physical vehicle (fail closed if a map is unavailable).
        """
        if self._global_lookup is not None:
            return self._global_lookup
        maps = []
        base = self._identity_sidecar()
        if base is None or base.vehicle_map is None:
            raise IdeaAConfigError(
                "composite identity re-projection needs the base pack "
                "vehicle_map; refusing to align by pack-local indices")
        maps.append(base.vehicle_map)
        for pack in range(1, len(self._extra_pack_dirs) + 1):
            side = self._extra_sidecar(pack)
            if side is None or side.vehicle_map is None:
                raise IdeaAConfigError(
                    f"composite identity re-projection needs the extra pack "
                    f"{pack} vehicle_map; refusing to align by pack-local "
                    "indices")
            maps.append(side.vehicle_map)
        all_keys = sorted(set().union(*maps))
        self._global_lookup = {key: i for i, key in enumerate(all_keys)}
        return self._global_lookup

    def _reproject_row(self, side, row: Tensor) -> Tensor:
        lookup = self._global_vehicle_lookup()
        return torch.as_tensor(
            remap_canonical_ids(np.asarray(row, dtype=np.int64),
                                side.vehicle_map, lookup))

    def _sidecar_dir(self) -> Optional[str]:
        raw = os.environ.get("WM_N1_TRAIN_IDENTITIES")
        return raw if raw and str(raw).strip() else None

    @staticmethod
    def _canonical_pack_env() -> Optional[str]:
        """The OPTIONAL ``WM_N1_CANONICAL_PACK`` value (None when unset/blank).

        Root sets it ONLY to the separately coordinator-validated REAL
        canonical pack target (after the raw SHA verify); it is never guessed
        here.
        """
        raw = os.environ.get("WM_N1_CANONICAL_PACK")
        return raw if raw and str(raw).strip() else None

    def _canonical_pack_target(self) -> Optional[str]:
        """The validated canonical target for the sidecar read, or ``None``.

        The identity loader's strict path checks reject any symlink/reparse
        ancestor, and a deployed ``$WORK/pack`` location is an INTENTIONAL
        frozen alias.  ONLY when ``WM_N1_CANONICAL_PACK`` is set may the
        loader read the canonical target instead - and only after a strict
        ``Path.resolve(strict=True)`` equality check against ``pack_dir``
        (refused BEFORE any loader read).  The canonical target itself must
        be link-free: the loader's own strict checks enforce that on the
        literal path this corpus hands it.  When the env is absent there is
        NO automatic resolve fallback: ``None`` is returned and the literal
        ``pack_dir`` is passed unchanged (an alias then refuses, as it
        should, until root publishes the validated target).
        """
        canon = self._canonical_pack_env()
        if canon is None:
            return None
        try:
            resolved_alias = Path(self.pack_dir).resolve(strict=True)
            resolved_canon = Path(canon).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise IdeaAConfigError(
                f"WM_N1_CANONICAL_PACK {canon!r} or the pack alias "
                f"{str(self.pack_dir)!r} cannot be resolved (strict): "
                f"{exc}; refusing the identity sidecar read") from exc
        if resolved_alias != resolved_canon:
            raise IdeaAConfigError(
                f"WM_N1_CANONICAL_PACK {canon!r} resolves to "
                f"{str(resolved_canon)!r}, which is NOT the pack alias "
                f"{str(self.pack_dir)!r} ({str(resolved_alias)!r}); refusing "
                "to read the identity sidecar against an unvalidated pack "
                "location (no alias/canonical mismatch may reach the loader)")
        return canon

    def _identity_sidecar(self):
        """The verified TRAIN identity sidecar, or ``None`` when unavailable.

        Loading follows the exact sidecar interface
        (``load_identity_sidecar(pack_dir, sidecar_dir)``); the returned
        object exposes ``ids_for_item(global_manifest_position) -> int64[K]``
        (``-1`` absent) and ``receipt_sha256``.  When
        ``WM_N1_CANONICAL_PACK`` is set and validated (strict-resolve
        equality, checked here BEFORE the loader reads anything) the LITERAL
        canonical target is passed to the loader instead of the frozen
        ``pack_dir`` alias; ``pack_dir`` itself, the receipt's ``pack_alias``
        and every handoff-gate literal keep the original alias.  CAL corpora
        never load one (TRAIN-only provenance; CAL is never read here).  A
        load failure propagates - a broken sidecar refuses the corpus
        instead of degrading to slot-guessing.
        """
        if self._identity is not None:
            return self._identity if self._identity is not False else None
        if self.split != "TRAIN":
            self._identity = False
            return None
        loader = self._identity_loader
        side_dir = self._sidecar_dir()
        if loader is None:
            if not side_dir:
                self._identity = False
                return None
            try:
                from train.identity_sidecar import load_identity_sidecar
            except ImportError as exc:
                raise IdeaAConfigError(
                    "WM_N1_TRAIN_IDENTITIES is set but "
                    "train.identity_sidecar is not importable; deploy the "
                    "verified identity module beside this package") from exc
            loader = load_identity_sidecar
        load_pack = str(self.pack_dir)
        canonical = self._canonical_pack_target()
        if canonical is not None:
            # frozen-alias allowance: the loader reads the validated REAL
            # canonical target; the corpus/receipt literals below retain the
            # original alias (the gate refuses any mismatch above)
            load_pack = canonical
        self._canonical_pack_used = canonical
        self._identity = loader(load_pack, side_dir)
        return self._identity

    def _validate_identity_row(self, item: dict, ids) -> Tensor:
        """Fail closed on a missing/unknown/duplicated verified identity.

        Every HDV-valid slot (``target_mask & ~is_cav``) must carry a
        non-negative canonical ID; ``-1`` is allowed only on absent/CAV
        slots; an ID may appear at most once per row.
        """
        ids = np.asarray(ids)
        if ids.dtype != np.int64:
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: verified "
                f"identity row dtype {ids.dtype} is not int64")
        if ids.ndim != 1:
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: verified "
                f"identity row is not 1-D (shape {ids.shape})")
        target = np.asarray(
            self._v1._array("target_mask")[int(item["tick"])], bool)
        veh = self._v1._array("veh_step")[int(item["tick"])]
        cav = np.asarray(veh[:, C.IS_CAV_IDX] > 0.5, bool)
        if ids.shape[0] != int(target.shape[0]):
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: verified "
                f"identity row width {ids.shape[0]} != pack K "
                f"{int(target.shape[0])}")
        if ids.size and int(ids.min()) < -1:
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: unknown "
                "verified identity (< -1) in row")
        present = ids >= 0
        if int(present.sum()) != len(np.unique(ids[present])):
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: "
                "duplicated verified identity in row")
        missing = (target & ~cav) & ~present
        if bool(missing.any()):
            slots = np.flatnonzero(missing).tolist()
            raise IdeaAConfigError(
                f"item {item.get('world')!r}/{item.get('arm')!r}: missing "
                f"verified identity for HDV-valid slot(s) {slots}; refusing "
                "to disable the HDV sub-term or guess the identity")
        # copy: sidecar rows may back onto read-only buffers
        return torch.as_tensor(np.array(ids, dtype=np.int64, copy=True))

    def _ids_for_item(self, item: dict) -> Optional[Tensor]:
        """The item's validated ``int64[K]`` canonical IDs (or ``None``).

        The sidecar is queried with the item's ORIGINAL per-pack manifest
        position (never the TRAIN-filtered / composite-offset dataset
        position).  In composite mode each pack uses its OWN sidecar and rows
        are re-projected into one shared canonical space.  Rows are validated
        once and cached by ``(pack, position)``.
        """
        pack = int(item.get("_pack", 0))
        if pack == 0:
            side = self._identity_sidecar()
        else:
            side = self._extra_sidecar(pack)
            if side is None:
                raise IdeaAConfigError(
                    f"extra corpus item {item.get('world')!r}/"
                    f"{item.get('arm')!r} has no verified identity sidecar; "
                    "set --extra-identities / WM_N1_EXTRA_TRAIN_IDENTITIES to "
                    "one sidecar per extra pack (never guess identities)")
        if side is None:
            return None
        _, gpos = self._component_position(item)
        key = (pack, int(gpos))
        if key not in self._id_rows:
            try:
                row = side.ids_for_item(int(gpos))
            except Exception as exc:
                raise IdeaAConfigError(
                    f"item {item.get('world')!r}/{item.get('arm')!r} "
                    f"(pack {pack} manifest position {int(gpos)}) lacks "
                    f"verified TRAIN identities: {exc}") from exc
            row = self._validate_identity_row(item, row)
            if self._reproject_ids:
                row = self._reproject_row(side, row)
            self._id_rows[key] = row
        return self._id_rows[key]

    def verified_identities(self) -> dict:
        """Verify EVERY TRAIN item's identity; return the gate receipt.

        Raises :class:`IdeaAConfigError` when the sidecar is unavailable
        (``WM_N1_TRAIN_IDENTITIES`` unset) or any item lacks a verified,
        known, non-duplicated ID on every HDV-valid slot.  The receipt is
        recorded in ``TRAIN_RECEIPT.json``; the HDV sub-term is never
        silently disabled for an actual fit.  ``pack_alias`` (the original
        literal pack location) and ``canonical_pack`` (the validated
        ``WM_N1_CANONICAL_PACK`` target, or None) sit alongside the sidecar
        ``receipt_sha256`` for audit.
        """
        side = self._identity_sidecar()
        if side is None:
            raise IdeaAConfigError(
                "an actual fit requires VERIFIED per-item TRAIN vehicle "
                "identities: set WM_N1_TRAIN_IDENTITIES to the identity "
                "sidecar directory built for this pack "
                "(load_identity_sidecar(pack_dir, sidecar_dir)); the corpus "
                "never guesses identities")
        extra_receipts = []
        for pack in range(1, len(self._extra_pack_dirs) + 1):
            extra = self._extra_sidecar(pack)
            if extra is None:
                raise IdeaAConfigError(
                    f"extra corpus pack {self._extra_pack_dirs[pack - 1]!r} "
                    "requires its OWN verified identity sidecar; set "
                    "--extra-identities / WM_N1_EXTRA_TRAIN_IDENTITIES (one "
                    "per extra pack); the corpus never guesses identities")
            extra_sha = getattr(extra, "receipt_sha256", None)
            if not extra_sha:
                raise IdeaAConfigError(
                    "an extra verified identity sidecar exposes no "
                    "receipt_sha256")
            extra_receipts.append(str(extra_sha))
        n_ids = 0
        for i in range(len(self._order)):
            row = self._ids_for_item(self._items[self._order[i]])
            if row is None:
                raise IdeaAConfigError(
                    f"item {self._items[self._order[i]].get('world')!r} "
                    "lacks verified TRAIN identities; the HDV sub-term is "
                    "never silently disabled")
            n_ids += int((row >= 0).sum())
        receipt_sha = getattr(side, "receipt_sha256", None)
        if not receipt_sha:
            raise IdeaAConfigError(
                "the verified identity sidecar exposes no receipt_sha256")
        out = {"status": "verified", "mode": "identity-sidecar",
               "env": "WM_N1_TRAIN_IDENTITIES",
               "pack_alias": str(self.pack_dir),
               "canonical_pack":
                   getattr(self, "_canonical_pack_used", None),
               "n_items": len(self._order),
               "n_verified_id_slots": int(n_ids),
               "receipt_sha256": str(receipt_sha)}
        if self._is_multi_pack:
            out["mode"] = "identity-sidecar+extra"
            out["extra_env"] = "WM_N1_EXTRA_TRAIN_IDENTITIES"
            out["extra_receipt_sha256"] = extra_receipts
            out["extra_identity_dirs"] = list(self._extra_identity_dirs)
        return out

    # -- protocol -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._order)

    def worlds(self) -> list:
        return self._v1.worlds()

    def pair_keys(self) -> list:
        return self._v1.pair_keys()

    def __getitem__(self, i: int) -> dict:
        return self.window(self._v1[i])

    # -- canonical window ---------------------------------------------------
    @staticmethod
    def _lane_of(kin_row: Tensor) -> Tensor:
        """``[..., 7]`` kinematics -> int lane (0/1/2, -1 empty)."""
        one_hot = kin_row[..., _KIN_L0:_KIN_L2 + 1]
        has = one_hot.sum(dim=-1) > 0
        return torch.where(has, one_hot.argmax(dim=-1),
                           torch.full_like(one_hot[..., 0], -1,
                                           dtype=torch.long))

    def window(self, item: dict) -> dict:
        """One packed item -> the full canonical window (batch axis 1)."""
        tick = int(item["tick"])
        veh_kin = np.asarray(self._array("veh_kin")[tick], np.float32)
        hdv_base = np.asarray(self._array("hdv_base")[tick], np.float32)
        event_tok = np.asarray(self._array("event_tok")[tick], np.float32)
        event_mask = np.asarray(self._array("event_mask")[tick], np.float32)
        n_ticks = int(self._array("veh_kin").shape[0])
        end = min(self._run_end(tick), n_ticks)
        horizon = max(0, min(self.horizon_steps, end - tick - 1))

        kin = torch.as_tensor(veh_kin).unsqueeze(0)
        kin_mask = item["veh_mask"].to(torch.bool).unsqueeze(0)
        lane0 = self._lane_of(torch.as_tensor(veh_kin)).unsqueeze(0)
        is_cav = item["veh_step"][:, C.IS_CAV_IDX] > 0.5
        meta = {key: item[key] for key in (
            "world", "arm", "chunk", "split", "t0_s", "pair_key", "policy")}
        # (a) telemetry: carry the v1 item's per-item t0 assertion counts and
        # the (b) exclusion flag into the canonical window's meta, so
        # label_t0_telemetry() sees them (it reads batch["meta"]).
        for key in ("label_t0_rows", "label_t0_violations",
                    "outcome_supervision_excluded"):
            if key in item:
                meta[key] = item[key]
        out = {
            "kin": kin,
            "kin_mask": kin_mask,
            "is_cav": is_cav.unsqueeze(0),
            "hdv_base": torch.as_tensor(hdv_base).unsqueeze(0),
            "baseline_idm": torch.as_tensor(
                hdv_base[:, HDV_BASE_IDM_IDX]).unsqueeze(0),
            "target_accel": item["accel"].unsqueeze(0),
            "target_mask": item["target_mask"].unsqueeze(0),
            "lane_target": item["lane"].unsqueeze(0),
            "counter": item["counter"].unsqueeze(0),
            "arrivals": item["arrivals"].unsqueeze(0),
            "outcome_quantiles": item["outcome_quantiles"].unsqueeze(0),
            "outcome_mask": item["outcome_mask"].unsqueeze(0),
            "collision": item["collision"].unsqueeze(0),
            "event_tok": torch.as_tensor(event_tok).unsqueeze(0),
            "event_mask": torch.as_tensor(event_mask).unsqueeze(0),
            "s0": kin[:, :, _KIN_S],
            "v0": kin[:, :, _KIN_V],
            "a0": kin[:, :, _KIN_A],
            "lane0": lane0,
            "length0": kin[:, :, _KIN_LEN],
            "present0": kin_mask.clone(),
            # top-level world/arm so callers that filter by CAL half can read
            # the item's world without reaching into the meta list
            "world": meta.get("world"),
            "arm": meta.get("arm"),
            "meta": [meta],
        }
        if self.split == "TRAIN":
            ids = self._ids_for_item(item)
            if ids is not None:
                out["vehicle_ids"] = ids.unsqueeze(0)
        if horizon >= 1:
            horizon_max = self.horizon_steps
            k_slots = int(kin.shape[1])
            stop = tick + horizon + 1
            kin_seq = torch.as_tensor(np.asarray(
                self._array("veh_kin")[tick:stop], np.float32))   # [T,K,7]
            plan_seq = torch.as_tensor(np.asarray(
                self._array("cav_plan_ch")[tick:stop], np.float32))  # [T,K,8]
            acc_seq = torch.as_tensor(
                np.asarray(self._array("accel_next")[tick:tick + horizon],
                           np.float32))
            base_seq = torch.as_tensor(np.asarray(
                self._array("hdv_base")[tick:tick + horizon,
                                        ..., HDV_BASE_IDM_IDX], np.float32))
            # ALIVE-SINCE-t0: target_mask[tick+u] is 1 iff the SAME vehicle
            # is still in the slot at tick+u+1 (features/FORMAT.md section 6).
            # A slot reused mid-window must NOT become alive again for the NEW
            # vehicle: the rollout keeps integrating the departed vehicle's
            # state, so a per-step mask would compare the wrong vehicle.  The
            # cumulative product makes alive monotone: once a slot's t0
            # vehicle departs it stays masked, and mid-window entrants are
            # never added.  ``alive_state[t]`` is the t0 vehicle's presence at
            # tick+t (present0 AND the transitions before t); rollout step t
            # predicts tick+t -> tick+t+1 and so uses ``alive_state[t+1]``.
            present0 = kin_mask[0]
            tm = torch.as_tensor(np.asarray(
                self._array("target_mask")[tick:tick + horizon],
                bool)).to(torch.bool)                              # [h,K]
            alive_state = torch.cumprod(
                torch.cat([present0.unsqueeze(0).to(torch.int64),
                           tm.to(torch.int64)], dim=0),
                dim=0).to(torch.bool)                              # [h+1,K]
            mask_seq = alive_state[1:horizon + 1]                  # [h,K]
            # the summary is over the ACTUAL window (never padded); the time
            # tensors are zero-padded to the registered horizon so a batch of
            # runs with different lifetimes collates, with the padded steps
            # masked out of the rollout loss.  The CAV summary uses only the
            # steps each CAV slot is alive: the per-step mask excludes dead
            # steps from the speed/accel statistics, and the trajectory is
            # forward-filled so the displacement feature uses the slot's LAST
            # ALIVE position, never a departed/ reused slot's later state.
            # vectorized forward-fill over ALIVE steps: the last alive row
            # index per (t, slot) is a cummax of the alive positions, and the
            # gathered rows reproduce the per-t loop exactly.
            t_steps = int(alive_state.shape[0])
            ar = torch.arange(t_steps).view(t_steps, 1)
            fill = torch.cummax(
                torch.where(alive_state, ar, torch.zeros_like(ar)),
                dim=0).values                                   # [T,K]
            pos_ff = kin_seq[..., _KIN_S].gather(0, fill)       # [T,K]
            plan_ff = plan_seq.gather(
                0, fill.unsqueeze(-1).expand(-1, -1, plan_seq.shape[-1]))
            cav_alive = (is_cav.unsqueeze(0) & alive_state).unsqueeze(1)
            summary_raw = _cav_motion_summary(
                pos_ff.unsqueeze(1), plan_ff.unsqueeze(1), cav_alive)
            # A11 full-horizon plan: the SAME planned CAV channels, read over
            # [t0, t0+H] (not the 10 s rollout window) and binned.  Only built
            # when the flag is set, so the default batch is unchanged.
            plan_long = None
            plan_horizon_s = getattr(self, "_plan_horizon_s", None)
            if plan_horizon_s is not None:
                from features.idea_a import (CAV_ACCEL_SCALE,
                                             CAV_TGT_SPEED_SCALE)
                from model.paired_effect import long_plan_summary

                n_long_req = max(
                    1, int(round(plan_horizon_s / float(C.DT_S))))
                n_long = max(0, min(n_long_req, end - tick))
                if n_long >= 1:
                    pl = np.asarray(self._array("cav_plan_ch")[
                        tick:tick + n_long], np.float32)
                    sp_l = torch.as_tensor(pl[..., 0]).to(torch.float32) * \
                        float(CAV_TGT_SPEED_SCALE)
                    ac_l = torch.as_tensor(pl[..., 1]).to(torch.float32) * \
                        float(CAV_ACCEL_SCALE)
                    tm_l = torch.as_tensor(np.asarray(
                        self._array("target_mask")[
                            tick:tick + n_long - 1], bool)).to(torch.bool)
                    alive_l = torch.cumprod(torch.cat([
                        kin_mask[0].unsqueeze(0).to(torch.int64),
                        tm_l.to(torch.int64)], dim=0), dim=0).to(torch.bool)
                else:
                    sp_l = torch.zeros(1, kin.shape[1])
                    ac_l = torch.zeros(1, kin.shape[1])
                    alive_l = kin_mask[0].unsqueeze(0).to(torch.bool)
                cav_l = (is_cav.unsqueeze(0) & alive_l).to(
                    torch.float32).unsqueeze(1)
                plan_long = long_plan_summary(
                    sp_l.unsqueeze(1), ac_l.unsqueeze(1), cav_l,
                    horizon_s=float(plan_horizon_s))
            teacher = torch.zeros(horizon_max + 1, 1, k_slots, 4,
                                  dtype=kin_seq.dtype)
            teacher[:horizon + 1, 0, :, TEACHER_S] = kin_seq[..., _KIN_S]
            teacher[:horizon + 1, 0, :, TEACHER_LANE] = \
                self._lane_of(kin_seq).float()
            teacher[:horizon + 1, 0, :, TEACHER_V] = kin_seq[..., _KIN_V]
            # the CAV action of step t is the recorded acceleration that
            # produced tick t+1 (accel_next[tick+t]); HDV slots use the learned
            # residual, so the column is only consumed on CAV slots.
            teacher[0, 0, :, TEACHER_A] = kin_seq[0, :, _KIN_A]
            teacher[1:horizon + 1, 0, :, TEACHER_A] = acc_seq

            def _pad_time(seq):
                pad = torch.zeros(horizon_max - horizon, *seq.shape[1:],
                                  dtype=seq.dtype)
                return torch.cat([seq, pad], dim=0)

            out.update({
                "teacher": teacher,
                "rollout_accel": _pad_time(acc_seq).unsqueeze(1),
                "rollout_baseline_idm": _pad_time(base_seq).unsqueeze(1),
                "rollout_mask": _pad_time(mask_seq.bool()).unsqueeze(1),
                "summary_raw": summary_raw,
            })
            if plan_long is not None:
                out["plan_long"] = plan_long
        return out


    def support_masks(self, i: int) -> dict:
        """Cheap per-item contrast support (``outcome`` taus, ``hdv`` slots).

        Reads only ``target_mask`` + ``veh_step`` for the HDV slots and the
        packed outcome rows for the per-tau availability, so the eligible
        CONTRAST pair pool is precomputed without building a full rollout
        window per item.  When verified TRAIN identities are available the
        result also carries ``ids`` (``int64[K]``, ``-1`` absent); HDV
        eligibility is then common-ID based, not slot intersection.
        """
        if i < 0:
            i += len(self._order)
        if not 0 <= i < len(self._order):
            raise IndexError(f"item position {i} out of range")
        item = self._items[self._order[i]]
        tick = int(item["tick"])
        target = np.asarray(self._v1._array("target_mask")[tick], bool)
        veh = self._v1._array("veh_step")[tick]
        is_cav = np.asarray(veh[:, C.IS_CAV_IDX] > 0.5, bool)
        _, omask, _ = self._v1.outcome_target(i)
        ids = self._ids_for_item(item) if self.split == "TRAIN" else None
        return {"outcome": torch.as_tensor(np.asarray(omask, bool)),
                "hdv": torch.as_tensor(target & ~is_cav),
                "ids": ids}


def _kin_features(s_m: Tensor, lane: Tensor, v_mps: Tensor, a_mps2: Tensor,
                  length_m: Tensor) -> Tensor:
    """World tensors -> ``[..., KIN_F]`` in ``hdv.config.KIN_FIELDS`` order."""
    one_hot = (lane.unsqueeze(-1) == torch.arange(
        C.N_LANES, device=lane.device, dtype=lane.dtype)).to(s_m.dtype)
    return torch.stack([s_m, one_hot[..., 0], one_hot[..., 1],
                        one_hot[..., 2], v_mps, a_mps2, length_m], dim=-1)


def _make_optimizer(trainer: "IdeaATrainer", lr: float,
                    total_steps: Optional[int] = None):
    """Adam (+ cosine decay to ~0 when ``total_steps`` is known).

    The pinball/CRPS family is piecewise linear, so a decaying tail is needed
    to actually reach the ~zero floor of the G4 overfit gate.
    """
    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    scheduler = None
    if total_steps:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(total_steps), eta_min=float(lr) * 1e-3)
    return optimizer, scheduler


def _move_to_device(obj, device):
    """Recursively move tensors in dicts/lists/tuples to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def _masked_mean(x: Tensor, mask: Optional[Tensor], *,
                 strict: bool = False, context: str = "term") -> Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    while m.dim() < x.dim():                 # broadcast over trailing dims
        m = m.unsqueeze(-1)
    m = m.expand_as(x)
    denom = m.sum()
    if strict and float(denom) == 0.0:
        raise IdeaAConfigError(
            f"{context} evaluated over an EMPTY mask (0 slots); refusing to "
            "return the fail-open 0.0 on a real (fail-closed) run")
    return (x * m).sum() / denom.clamp(min=1.0)


def _d1_step(s_m: Tensor, v_mps: Tensor, a_mps2: Tensor,
             dt_s: float = DT_S) -> tuple:
    """Differentiable D1 trapezoidal step (rollout2.validity.integrate_d1).

    The frozen validity helper runs under ``no_grad``; the training rollout
    needs gradients through the integration, so the identical rule is applied
    here: ``v_new = max(0, v + a*dt)``, ``s_new = s + 0.5*(v + v_new)*dt``.
    """
    v_new = (v_mps + a_mps2 * dt_s).clamp(min=0.0)
    s_new = s_m + 0.5 * (v_mps + v_new) * dt_s
    return s_new, v_new


# ----------------------------------------------------------- held-out tests --
def _norm_world(world) -> str:
    return str(world or "").strip().upper()


def is_j4b_world(world) -> bool:
    """True for the J4B (wall law v2.2) worlds of held-out test (i)."""
    w = _norm_world(world)
    return w == "J4B" or w.startswith("J4B_") or w.startswith("J4B-")


def is_pi_item(item: dict) -> bool:
    """True for a PI arm (held-out test (ii) removes every PI arm)."""
    for key in ("policy", "arm"):
        value = str(item.get(key, "")).strip().lower()
        if value == "pi" or value.startswith("pi_") or value.startswith("pi-"):
            return True
    return False


def is_wall_item(item: dict) -> bool:
    """True for a wall arm (wall-speed arms of held-out test (iii))."""
    for key in ("policy", "arm"):
        value = str(item.get(key, "")).strip().lower()
        if value == "wall" or value.startswith("wall_") \
                or value.startswith("wall-") or value.startswith("wallv"):
            return True
    return False


# The J4 pilot uses ordinary world IDs shared with chunk-4 base runs.  The
# batch, preserved by corpus.pack._train_item, identifies its wall-law source.
J4_WALL_BATCH = "walls_pilot_chunk4"
_J4_ARM_RE = re.compile(r"w-(mid|lo)(?:[-_]|$)", re.IGNORECASE)


def is_j4_item(item: dict) -> bool:
    """True only for a J4 TRAIN wall arm with registered batch metadata."""
    if str(item.get("batch", "")) != J4_WALL_BATCH:
        return False
    # The inventory records the source chunk as "4"; the packed item records
    # the validated canonical label "chunk4" (featurize_all.chunk_label).
    if not str(item.get("world", "")).strip() or \
            str(item.get("chunk", "")) not in ("4", "chunk4") or \
            str(item.get("split", "")).upper() != "TRAIN" or \
            str(item.get("policy", "")).lower() != "wall" or \
            _J4_ARM_RE.match(str(item.get("arm", ""))) is None:
        raise IdeaAConfigError(
            f"J4 batch item {item.get('world')}/{item.get('arm')} has "
            "contradictory chunk/split/policy/arm metadata")
    return True


_WALL_FACTOR_RE = re.compile(r"wall[^0-9]*v?([0-9]+(?:\.[0-9]+)?)")


def wall_speed_factor(item: dict) -> Optional[float]:
    """The arm's wall speed factor (explicit key wins, else parsed from
    ``arm``/``policy`` names like ``wall_v31`` / ``wall-27mph``)."""
    for key in ("wall_speed_factor", "wall_speed_mps", "speed_factor"):
        raw = item.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    if str(item.get("batch", "")) == J4_WALL_BATCH:
        j4_arm = _J4_ARM_RE.match(str(item.get("arm", "")))
        if j4_arm:
            return _J4_WALL_FACTORS[j4_arm.group(1).lower()]
    for key in ("arm", "policy"):
        m = _WALL_FACTOR_RE.search(str(item.get(key, "")).lower())
        if m:
            return float(m.group(1))
    return None


def held_out_split(items: Sequence[dict], test: str, *,
                   factor: Optional[float] = None) -> dict:
    """Partition items into ``(train, eval)`` for one A3.4 held-out test.

    ``train`` keeps every item that is not excluded; ``eval`` is exactly the
    excluded arms; the two are disjoint by construction so the excluded arms
    are ABSENT from training.  ``factor`` overrides the a-priori highest
    J4 wall-speed factor for test (iii).
    """
    items = list(items)                       # keep dict identity for ids
    if test == "i_j4_to_j4b":
        excluded = [it for it in items if is_j4b_world(it.get("world"))]
    elif test == "ii_leave_pi_out":
        excluded = [it for it in items if is_pi_item(it)]
    elif test == "iii_leave_top_wall_speed_out":
        if any(not str(it.get("batch", "")) and is_wall_item(it) and
               _J4_ARM_RE.match(str(it.get("arm", ""))) is not None
               for it in items):
            raise IdeaAConfigError(
                "held-out test (iii) received a J4-shaped wall arm without "
                "the registered batch identity")
        cands = [it for it in items if is_j4_item(it)]
        for it in cands:
            arm_factor = _J4_WALL_FACTORS[
                _J4_ARM_RE.match(str(it["arm"])).group(1).lower()]
            if wall_speed_factor(it) != arm_factor:
                raise IdeaAConfigError(
                    f"J4 arm {it.get('arm')!r} disagrees with its explicit "
                    "wall-speed factor")
        factors = {wall_speed_factor(it) for it in cands}
        factors.discard(None)
        if not factors:
            raise IdeaAConfigError(
                "held-out test (iii) needs J4 wall arms with a parseable "
                "speed factor (registered W-mid/W-lo arms)")
        top = float(factor) if factor is not None else _J4_WALL_FACTORS["mid"]
        if top not in factors:
            raise IdeaAConfigError(
                f"held-out test (iii) registered factor {top} is absent "
                f"from the J4 wall arms {sorted(factors)}")
        excluded = [it for it in cands if wall_speed_factor(it) == top]
    else:
        raise IdeaAConfigError(
            f"unknown held-out test {test!r}; expected one of {HELD_OUT_TESTS}")
    eval_keys = {id(it) for it in excluded}
    train = [it for it in items if id(it) not in eval_keys]
    return {"test": test, "train": train, "eval": excluded,
            "excluded": excluded}


def _dataset_items(corpus) -> list:
    """Item metadata in dataset-position order (read-only access)."""
    items = getattr(corpus, "_items", None)
    if items is None:
        raise IdeaAConfigError(
            f"corpus {type(corpus).__name__} exposes no item metadata; "
            "pass an item list to held_out_split instead")
    order = getattr(corpus, "_order", None)
    if order is None:
        order = range(len(items))
    return [items[j] for j in order]


def _source_item_meta(source) -> Optional[list]:
    """Item metadata in dataset-position order for a batching source.

    Handles ``MemmapCorpus``/``IdeaACorpus`` (``_items`` + ``_order``) and the
    read-only ``HeldOutTrainView``/quick-tune view (a ``corpus`` + ``kept``
    position list) so the pair-aware sampler can see every item's ``pair_key``
    and ``policy``.  Returns ``None`` when the source exposes no cheap metadata
    (the caller then falls back to plain random batching).
    """
    items = getattr(source, "_items", None)
    order = getattr(source, "_order", None)
    meta = None
    if items is not None and order is not None:
        meta = [items[j] for j in order]
    else:
        corpus = getattr(source, "corpus", None)
        kept = getattr(source, "kept", None)
        if corpus is not None and kept is not None:
            c_items = getattr(corpus, "_items", None)
            c_order = getattr(corpus, "_order", None)
            if c_items is not None and c_order is not None:
                meta = [c_items[c_order[k]] for k in kept]
    # pair sampling needs a ``pair_key`` on every item; a corpus-like stand-in
    # that carries only world/arm metadata (e.g. the quick-tune window corpus)
    # falls back to the plain random batching it used before.
    if meta is None or not all("pair_key" in it for it in meta):
        return None
    return meta


def held_out_corpus_view(corpus, test: str, *,
                         factor: Optional[float] = None) -> "HeldOutTrainView":
    """A train-side corpus view with the excluded arms ABSENT."""
    items = _dataset_items(corpus)
    split = held_out_split(items, test, factor=factor)
    kept = [pos for pos, it in enumerate(items)
            if id(it) in {id(x) for x in split["train"]}]
    return HeldOutTrainView(corpus, kept, test, split["eval"])


class HeldOutTrainView:
    """Read-only train view over a corpus with excluded arms removed."""

    def __init__(self, corpus, kept_positions: Sequence[int], test: str,
                 excluded_items: Sequence[dict]) -> None:
        self.corpus = corpus
        self.kept = list(kept_positions)
        self.test = test
        self.excluded_items = list(excluded_items)

    def __len__(self) -> int:
        return len(self.kept)

    def __getitem__(self, i: int) -> dict:
        return self.corpus[self.kept[i]]

    @property
    def pair_sampling(self) -> bool:
        return bool(getattr(self.corpus, "pair_sampling", False))

    @property
    def pack_dir(self):
        return getattr(self.corpus, "pack_dir", None)

    @property
    def split(self) -> str:
        return str(getattr(self.corpus, "split", "TRAIN"))

    def arms(self) -> list:
        """``(world, arm)`` of every kept item (absence checks in tests)."""
        return [(it.get("world"), it.get("arm")) for it in
                (_dataset_items(self.corpus)[p] for p in self.kept)]

    def eval_arms(self) -> list:
        return [(it.get("world"), it.get("arm")) for it in self.excluded_items]

    def verified_identities(self) -> dict:
        """Delegate to the wrapped corpus (kept windows share its ID rows).

        The view never re-maps positions: the underlying corpus keys its
        verified identity rows by the ORIGINAL global manifest position, so
        the quick-tune/held-out kept subset is verified by the same strict
        per-item check as the full TRAIN corpus.
        """
        fn = getattr(self.corpus, "verified_identities", None)
        if not callable(fn):
            raise IdeaAConfigError(
                "the wrapped corpus exposes no verified_identities() "
                "accessor; verified TRAIN identities are required for an "
                "actual fit")
        return fn()


def held_out_dry_run(items_or_corpus, *,
                     factor: Optional[float] = None) -> dict:
    """G6 dry-run: build all three A3.4 splits and assert disjointness.

    Returns one receipt per test with the train/eval counts and the excluded
    arms; raises :class:`IdeaAConfigError` if any excluded arm would remain
    in its train split (they cannot, by construction - the check is the
    gate's receipt).
    """
    if isinstance(items_or_corpus, dict) or not hasattr(items_or_corpus,
                                                        "__len__"):
        raise IdeaAConfigError("expected an item list or a corpus")
    if isinstance(items_or_corpus, (list, tuple)):
        items = list(items_or_corpus)
    else:
        items = _dataset_items(items_or_corpus)
    if not items or not isinstance(items[0], dict):
        raise IdeaAConfigError(
            "held_out_dry_run needs item metadata dicts (or a corpus)")
    report = {}
    for test in HELD_OUT_TESTS:
        split = held_out_split(items, test, factor=factor)
        train_keys = {id(it) for it in split["train"]}
        overlap = [it for it in split["eval"] if id(it) in train_keys]
        if overlap:
            raise IdeaAConfigError(
                f"held-out {test}: excluded arms leaked into training: "
                f"{[(o.get('world'), o.get('arm')) for o in overlap]}")
        report[test] = {
            "n_train": len(split["train"]),
            "n_eval": len(split["eval"]),
            "excluded_arms": [(it.get("world"), it.get("arm"))
                              for it in split["eval"]],
        }
    return report


# ------------------------------------------------------------- quick-tune ----
def world_stratified_subset(items: Sequence[dict], subset_frac: float = 1.0,
                            seed: int = 0) -> dict:
    """Deterministic, world-stratified subset of item positions.

    Positions are grouped by ``world``; every world keeps
    ``round(subset_frac * n_world)`` items (at least one while
    ``subset_frac > 0``) drawn by a seeded RNG.  The result is fully
    reproducible for a given ``(items, subset_frac, seed)``: the same seed
    selects the same positions, and the per-world minimum guarantees no world
    is dropped, so a quick-tune run stays world-stratified.  Returns the kept
    positions plus a per-world receipt.
    """
    frac = float(subset_frac)
    if not 0.0 < frac <= 1.0:
        raise IdeaAConfigError("subset_frac must be in (0, 1]")
    by_world: dict = {}
    for pos, item in enumerate(items):
        by_world.setdefault(str(item.get("world", "")), []).append(pos)
    if not by_world:
        raise IdeaAConfigError("world_stratified_subset needs item metadata")
    gen = random.Random(int(seed))
    kept: list = []
    per_world: dict = {}
    for world in sorted(by_world):
        positions = by_world[world]
        if frac >= 1.0:
            chosen = list(positions)
        else:
            k = max(1, int(round(frac * len(positions))))
            chosen = sorted(gen.sample(positions, min(k, len(positions))))
        kept.extend(chosen)
        per_world[world] = len(chosen)
    kept.sort()
    return {"kept": kept, "per_world": per_world, "worlds": sorted(by_world),
            "n_total": len(items), "n_kept": len(kept), "subset_frac": frac,
            "seed": int(seed)}


def quick_tune_view(corpus, subset_frac: float = 0.25, seed: int = 0):
    """Wrap ``corpus`` with the deterministic world-stratified TRAIN subset.

    Returns ``(view, receipt)``; the view is read-only and exposes the same
    ``corpus``/``pack_dir``/``split`` attributes the trainer expects, so
    :meth:`IdeaATrainer.fit` can consume it unchanged.  Used only for the
    CAL-only improvement loop (quick-tune), never for a headline result.
    """
    items = _dataset_items(corpus)
    sel = world_stratified_subset(items, subset_frac, seed=seed)
    return HeldOutTrainView(corpus, sel["kept"], "quick-tune", []), sel


# ------------------------------------------------- Amendment 4 CAL split ----
# Amendment A4.2 froze a MANDATORY CAL-tune / CAL-cal split.  The improvement
# loop's selection metric (and any quick-tune scoring) reads CAL-tune ONLY;
# early stopping, calibration, blend weights and variant choice read CAL-cal
# ONLY.  The single loader/filter is :mod:`evalh.cal_split`; these helpers are
# the train-path entry points, and they refuse a world from the other half.
CAL_TUNE = cal_split.CAL_TUNE
CAL_CAL = cal_split.CAL_CAL


def selection_worlds(supplied=None, *, cal_split_path=None) -> list:
    """WORLDS the CAL-tune selection metric / quick-tune scoring may read.

    With ``supplied`` given, every world must belong to CAL-tune or
    :class:`evalh.cal_split.CalSplitError` is raised (so the selection path
    can never be handed a CAL-cal world).
    """
    return cal_split.resolve_worlds(
        cal_split.CAL_TUNE, supplied, path=cal_split_path,
        context="train selection/quick-tune scoring")


def calibration_worlds(supplied=None, *, cal_split_path=None) -> list:
    """WORLDS early stopping / calibration may read (CAL-cal ONLY)."""
    return cal_split.resolve_worlds(
        cal_split.CAL_CAL, supplied, path=cal_split_path,
        context="train calibration/early-stopping")


def filter_cal_items(items, half: str, *, cal_split_path=None) -> list:
    """Keep CAL items belonging to ``half`` (other half dropped, foreign refused)."""
    kept = set(cal_split.half_worlds(half, path=cal_split_path))
    out = []
    for it in items:
        world = it.get("world") if isinstance(it, dict) else None
        if world is not None and str(world) in kept:
            out.append(it)
    return out


# ----------------------------------------------- G6 held-out split report --
#: map the internal A3.4 test id -> (pre-flight gate config name, arm token)
_HELDOUT_GATE = {
    "i_j4_to_j4b": ("heldout-i-j4b", "j4b"),
    "ii_leave_pi_out": ("heldout-ii-leave-pi", "pi"),
    "iii_leave_top_wall_speed_out": (
        "heldout-iii-leave-highest-wall-speed", "wall-speed-highest"),
}
#: registered J4 wall-speed factors (walls_design/WALL_FIX_OPTIONS.md:75-78).
_J4_WALL_FACTORS = {"mid": 0.85, "lo": 0.60}


def _heldout_inventory_items(inventory: Path) -> list:
    """TRAIN inventory rows -> held-out taxonomy items (metadata only).

    Only the TRAIN split is read.  J4 wall-pilot rows retain actual world,
    batch, raw inventory chunk and policy; contradictory provenance is refused.
    No planned J4B rows are added here.  No physics/outcome is read.
    """
    import pandas as pd

    df = pd.read_parquet(inventory)
    required = ("batch", "chunk", "world", "split", "arm", "policy", "run")
    missing = [key for key in required if key not in df.columns]
    if missing:
        raise IdeaAConfigError(
            f"held-out inventory is missing provenance columns {missing}")
    df = df[df["split"] == "TRAIN"]
    items: list = []
    for row in df.itertuples():
        empty = [key for key in required
                 if pd.isna(getattr(row, key)) or
                 not str(getattr(row, key)).strip()]
        if empty:
            raise IdeaAConfigError(
                f"held-out TRAIN inventory row has empty provenance {empty}")
        batch = str(getattr(row, "batch", ""))
        chunk = str(getattr(row, "chunk", ""))
        policy = str(getattr(row, "policy", ""))
        arm = str(getattr(row, "arm", ""))
        item = dict(world=str(getattr(row, "world", "")), batch=batch,
                    chunk=chunk, arm=arm, policy=policy,
                    split=str(getattr(row, "split", "")),
                    run=str(getattr(row, "run", "")))
        if is_j4_item(item):
            item["wall_speed_factor"] = wall_speed_factor(item)
            item["law"] = "wall-v2.1"
        items.append(item)
    return items


def _declared_j4b_items() -> list:
    """Metadata-only J4B placeholders for registered G6 report (i)."""
    items = []
    for name, factor in _J4_WALL_FACTORS.items():
        items.append(dict(world="J4B", arm=f"wall_v{factor}", policy="wall",
                          wall_speed_factor=factor, split="TRAIN",
                          planned=True, law="wall-v2.2"))
    return items


def _item_name(item: dict) -> str:
    if item.get("planned") and is_j4b_world(item.get("world")):
        return f"J4B::{item.get('arm')}"
    batch = str(item.get("batch", ""))
    if not batch:
        raise IdeaAConfigError(
            "held-out report item lacks a batch identity")
    return f"{item.get('world')}::{batch}::{item.get('arm')}"


def heldout_split_report(inventory: Optional[Path] = None) -> dict:
    """G6 pre-flight entry point (owner ``PREFLIGHT_GATES_IDEA_A.md``).

    Builds the three A3.4 held-out train/eval splits from the real TRAIN
    inventory metadata (TRAIN split first; HOLDOUT/CAL/chunk-5 never read) and
    returns the gate schema.  The excluded arms are ABSENT from every train
    split by construction (asserted before returning).
    """
    inv = Path(inventory) if inventory is not None else (
        Path(__file__).resolve().parents[1] / "inventory" / "inventory.parquet")
    if not inv.is_file():
        raise IdeaAConfigError(f"held-out inventory not found: {inv}")
    items = _heldout_inventory_items(inv)
    splits: dict = {}
    for test, (gate_name, token) in _HELDOUT_GATE.items():
        candidates = (items + _declared_j4b_items()
                      if test == "i_j4_to_j4b" else items)
        split = held_out_split(candidates, test)
        train = [_item_name(it) for it in split["train"]]
        eval_ = [_item_name(it) for it in split["eval"]]
        train_keys = {id(it) for it in split["train"]}
        leaked = [it for it in split["eval"] if id(it) in train_keys]
        if leaked:
            raise IdeaAConfigError(
                f"held-out {test}: excluded arms leaked into training")
        if not train or not eval_:
            raise IdeaAConfigError(
                f"held-out {test}: empty train ({len(train)}) or eval "
                f"({len(eval_)}) split")
        excluded = [token] + eval_
        splits[gate_name] = {"train": train, "eval": eval_,
                             "excluded": excluded}
    return {"splits": splits,
            "inventory": str(inv),
            "note": ("J4B is the J4 wall panel re-run under wall law v2.2 "
                     "(REGISTRATION_J4B.md, LF 2463f5be), read ONLY by "
                     "held-out test (i); metadata only, never opened.  It is "
                     "NOT the chunk-5 sealed final TEST (f8c7d760)")}


class _LoaderPassSampler(PairAwareBatchSampler):
    """A ONE-pass pair-aware batch sampler advanced via ``set_epoch``.

    The persistent DataLoader iterator needs the sampler to yield exactly one
    seeded corpus pass per ``iter(loader)`` so the pass can be advanced once
    per COMPLETE pass (``set_epoch``) and the same single DataLoader can be
    re-iterated.  The inherited unbounded ``epochs=None`` generator cannot do
    that: a zero-yield source spins ``__iter__`` forever inside the DataLoader
    worker feed (busy hang) instead of returning, and a terminated stream ends
    the whole batch generator silently.  Ordering is unchanged: pass ``e`` is
    exactly :func:`train.dataset.epoch_batches` for ``(items, batch_size,
    seed, e, pairs)`` - every primary item is still visited once per pass and
    only a pair partner may repeat (nothing is dropped, no mid-pass reset).
    """

    def set_epoch(self, epoch: int) -> None:
        """Select the seeded corpus pass the next ``__iter__`` yields."""
        self.start_epoch = int(epoch)
        self.epochs = 1


# ------------------------------------------------------------------ trainer --
class IdeaATrainer:
    """Member-first Idea A trainer (A3.2 losses, checkpoint/resume)."""

    def __init__(self, config: IdeaATrainConfig, corpus, *,
                 member: Optional[int] = None, seed: Optional[int] = None,
                 device: Optional[str] = None,
                 bundle: Optional[IdeaABundle] = None) -> None:
        self.config = config
        self.corpus = corpus
        self.bundle = bundle or IdeaABundle(config=config)
        self.member = int(member if member is not None else config.member)
        if not 1 <= self.member <= int(config.n_members):
            raise IdeaAConfigError(
                f"member {self.member} outside 1..{config.n_members}")
        self.seed = int(seed if seed is not None
                        else config.seed_for_member(self.member))
        self.device = device or config.device
        # record the ACTUAL device the trainer uses (the CLI may pass
        # --device cuda while the config default stays "cpu"; the checkpoint
        # sidecar and TRAIN_RECEIPT must not claim cpu in that case).  Store a
        # string so the config stays JSON-serializable for the sidecar.
        self.config.device = str(self.device)
        for name in ("hdv", "arrival", "corr", "state_enc", "paired_effect"):
            module = getattr(self.bundle, name, None)
            if isinstance(module, torch.nn.Module):
                module.to(self.device)
        from model.outcome_norm import OutcomeNorm

        self.outcome_norm = OutcomeNorm.identity()
        self._outcome_norm_sha = None
        self.loss_magnitudes: list = []
        self.loss_balance: dict = {}
        torch.manual_seed(self.seed)
        self.epoch = 0
        self.step = 0
        self.losses: list = []
        self.history: list = []
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(self.seed)
        # no optimizer exists until the first fit step creates one (the
        # canonical preparation contract asserts its absence BEFORE that)
        self._optimizer = None
        self._scheduler = None

    # ----------------------------------------------------------- modules ----
    def _require(self, name: str):
        module = getattr(self.bundle, name, None)
        if module is None:
            raise IdeaAConfigError(
                f"the {name} module is required for this loss; pass an "
                "IdeaABundle (P2/P4/P5 through their INTERFACES_V2 "
                "signatures, fakes in tests)")
        return module

    def _accel_scale(self) -> float:
        hdv = self._require("hdv")
        cfg = getattr(hdv, "cfg", None)
        scale = float(getattr(cfg, "accel_delta_scale_mps2", 1.5))
        return max(scale, 1e-3)

    def _strict_masks(self) -> bool:
        """True on a fail-closed (real corpus) run: an empty loss mask raises."""
        return bool(getattr(self, "_require_all_terms", False))

    def global_token(self, batch: dict) -> Tensor:
        """The correction head's global token from STATE ONLY.

        With ``state_global_token`` on and a state encoder in the bundle, the
        token is ``StateGlobalToken(kin, kin_mask, counter)`` (kinematics +
        entrance counters; no CAV flag, action, strategy or label channel).
        Otherwise the pre-M1FIX2 behaviour is reproduced exactly: the batch
        ``global_token`` when supplied, else a zero token of the head's width.
        """
        enc = getattr(self.bundle, "state_enc", None)
        if (bool(self.config.state_global_token)
                and isinstance(enc, torch.nn.Module) and "kin" in batch):
            counter = batch.get("counter")
            if isinstance(counter, Tensor):
                counter = counter.to(self.device)
            return enc(batch["kin"].to(self.device),
                       batch["kin_mask"].to(self.device), counter)
        token = batch.get("global_token")
        if isinstance(token, Tensor):
            return token
        corr = getattr(self.bundle, "corr", None)
        d_model = int(getattr(getattr(corr, "cfg", None), "d_model", 0))
        if d_model <= 0:
            raise IdeaAConfigError(
                "the correction head exposes no d_model; pass "
                "'global_token' explicitly or set state_global_token")
        summary = batch.get("summary_raw")
        ref = (summary if isinstance(summary, Tensor)
               else torch.zeros(1, dtype=torch.float32, device=self.device))
        return torch.zeros(int(ref.shape[0]), d_model, dtype=ref.dtype,
                           device=ref.device)

    def a6_binding(self) -> Optional[str]:
        """The frozen Amendment 6 LF-sha when the flag is ON, else ``None``.

        ``state_global_token`` defaults ON only while A6 is frozen, so calling
        this with the flag ON refuses (``evalh.seals.Amendment6Error``) unless
        the frozen ``REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md`` is present
        and its CRLF-safe re-hash matches.  With the flag OFF the legacy zeros
        path is reproduced and no A6 file is required.
        """
        if not bool(self.config.state_global_token):
            return None
        sha = _seals.require_frozen_amendment_6()
        self._a6_sha = sha
        return sha

    def norm_targets(self, y: Tensor) -> Tensor:
        """Apply the fitted outcome normalization to raw targets."""
        return self.outcome_norm.normalize(y)

    def _paired_effect_climatology_sha(self) -> Optional[str]:
        """SHA-256 of the configured climatology (pin, else computed)."""
        pin = getattr(self.config, "paired_effect_climatology_sha256", None)
        path = getattr(self.config, "paired_effect_climatology", None)
        if pin:
            return str(pin).strip().lower()
        if not path:
            return None
        from pathlib import Path as _Path

        if not _Path(path).is_file():
            return None
        from model.paired_effect import sha256_file

        return sha256_file(path)

    def _module_prefixes(self) -> tuple:
        """Namespaced prefixes for the bundle's present modules.

        ``state.`` is only included when the state global-token encoder is
        present, so a zeros-token/fake bundle keeps the frozen
        ``hdv./arrival./corr.`` namespacing.
        """
        prefixes = list(MODULE_PREFIXES)
        if isinstance(getattr(self.bundle, "state_enc", None),
                      torch.nn.Module):
            prefixes.append("state.")
        if isinstance(getattr(self.bundle, "paired_effect", None),
                      torch.nn.Module):
            prefixes.append("paired.")
        return tuple(prefixes)

    def parameters(self):
        """Unique trainable parameters across the bundle modules."""
        seen, out = set(), []
        for prefix in self._module_prefixes():
            module = _bundle_module(self.bundle, prefix)
            if isinstance(module, torch.nn.Module):
                for p in module.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
        if not out:
            raise IdeaAConfigError("the bundle holds no trainable parameters")
        return out

    def state_dict(self) -> dict:
        """Flat, namespaced state dict (``hdv.*``, ``arrival.*``, ``corr.*``)."""
        flat: dict = {}
        for prefix in self._module_prefixes():
            module = _bundle_module(self.bundle, prefix)
            if isinstance(module, torch.nn.Module):
                for key, value in module.state_dict().items():
                    flat[prefix + key] = value
        return flat

    def load_state_dict(self, flat: dict) -> None:
        prefixes = self._module_prefixes()
        missing = []
        for prefix in prefixes:
            module = _bundle_module(self.bundle, prefix)
            if not isinstance(module, torch.nn.Module):
                continue
            own = {k[len(prefix):]: v for k, v in flat.items()
                   if k.startswith(prefix)}
            if not own and any(k.startswith(prefix) for k in self.state_dict()):
                missing.append(prefix[:-1])
                continue
            module.load_state_dict(own)
        absent = [k for k in flat
                  if not any(k.startswith(p) for p in prefixes)]
        if missing or absent:
            raise IdeaAConfigError(
                f"checkpoint mismatch; unmatched={missing} unknown={absent[:5]}")

    # ------------------------------------------------------------- losses ---
    def residual_nll(self, batch: dict) -> Tensor:
        """One-step HDV residual NLL (excess form).

        The residual target is ``target_accel - baseline_idm`` on HDV slots;
        the excess over the irreducible entropy floor of the fixed-scale
        Gaussian (constant ``log sigma0 + 0.5 log 2*pi`` dropped; it carries
        no gradient) is ``0.5 * (e / sigma0)**2`` plus the lane-change
        cross-entropy of the residual lane logits.  Zero-reachable, so the
        G4 overfit gate can drive the total loss to ~0.
        """
        hdv = self._require("hdv")
        out = hdv(batch["kin"], batch["kin_mask"])
        delta = out["accel_delta_mps2"]
        base = batch.get("baseline_idm")
        if base is None:
            base = torch.zeros_like(delta)
        mask = batch["target_mask"]
        cav = batch.get("is_cav")
        if cav is not None:
            mask = mask & ~cav.to(torch.bool)
        sigma0 = self._accel_scale()
        e = (batch["target_accel"] - (base + delta)) / sigma0
        loss = _masked_mean(0.5 * e.pow(2), mask, strict=self._strict_masks(),
                            context="residual_nll")
        lane = batch.get("lane_target")
        if lane is not None:
            loss = loss + L.lane_change_nll(lane, out["lane_logit_delta"],
                                            mask)
        return loss

    def rollout_loss(self, batch: dict) -> Tensor:
        """Multi-step rollout loss with teacher-forced CAVs.

        HDV slots advance by the differentiable D1 rule under
        ``baseline_idm + residual``; CAV slots are replaced by the recorded
        trajectory (detached), so gradients flow only through the HDV
        residual.  The per-step term is the same excess NLL as
        :meth:`residual_nll` on the recorded HDV accelerations.
        """
        hdv = self._require("hdv")
        if not self.config.teacher_forcing_cavs:
            raise IdeaAConfigError(
                "the training rollout always teacher-forces the recorded CAV "
                "trajectory; an unforced CAV rollout is the P3 engine's job")
        teacher = batch.get("teacher")
        target = batch.get("rollout_accel")
        if teacher is None or target is None:
            raise IdeaAConfigError(
                "rollout_loss needs the recorded 'teacher' [T+1,B,K,4] and "
                "'rollout_accel' [T,B,K] tensors")
        for key in _WORLD0_REQUIRED:
            if key not in batch:
                raise IdeaAConfigError(f"rollout_loss needs world0 key {key!r}")
        horizon = min(int(self.config.rollout_horizon_steps),
                      int(target.shape[0]))
        if horizon < 1:
            raise IdeaAConfigError("rollout horizon is zero")
        s = batch["s0"].clone()
        v = batch["v0"].clone()
        a_prev = batch.get("a0", torch.zeros_like(v))
        lane = batch["lane0"].float()
        length = batch["length0"]
        present0 = batch["present0"].to(torch.bool)
        cav0 = batch.get("is_cav")
        if cav0 is None:
            cav0 = torch.zeros_like(present0)
        cav0 = cav0.to(torch.bool)
        mask_all = batch.get("rollout_mask")
        base_seq = batch.get("rollout_baseline_idm")
        sigma0 = self._accel_scale()
        total = torch.zeros((), dtype=s.dtype, device=s.device)
        valid = None
        # ``alive`` is the t0 vehicle's presence, monotonically shrinking as
        # the recorded transitions say it left.  ``present`` (the HDV
        # attention mask) is updated every step, so a departed vehicle leaves
        # the neighbour set and a mid-window entrant is never added.  A CAV
        # slot only teacher-forces the recorded trajectory while it is alive;
        # after its departure its teacher columns are not consumed.
        alive = present0
        for t in range(horizon):
            if mask_all is not None:
                alive = alive & mask_all[t].to(torch.bool)
            present = present0 & alive
            cav = cav0 & present
            out = hdv(_kin_features(s, lane, v, a_prev, length), present)
            a_hdv = out["accel_delta_mps2"]
            if base_seq is not None:
                a_hdv = a_hdv + base_seq[t]
            a = torch.where(cav, teacher[t + 1][..., TEACHER_A].detach(),
                            a_hdv)
            hdv_mask = present & ~cav
            e = (target[t] - a) / sigma0
            total = total + _masked_mean(0.5 * e.pow(2), hdv_mask)
            step_valid = hdv_mask.sum()
            valid = step_valid if valid is None else valid + step_valid
            s_new, v_new = _d1_step(s, v, a)
            s = torch.where(cav, teacher[t + 1][..., TEACHER_S].detach(), s_new)
            v = torch.where(cav, teacher[t + 1][..., TEACHER_V].detach(), v_new)
            lane = torch.where(cav,
                               teacher[t + 1][..., TEACHER_LANE].detach(),
                               lane)
            a_prev = a.detach()
        if self._strict_masks() and (valid is None or int(valid) == 0):
            raise IdeaAConfigError(
                "rollout evaluated over an EMPTY HDV mask (no present non-CAV "
                "slot in any step); refusing to return the fail-open 0.0 on a "
                "real (fail-closed) run")
        return total / horizon

    def arrival_loss(self, batch: dict) -> Tensor:
        """Arrival negative-binomial loss via the P4 head signature."""
        head = self._require("arrival")
        pred = head(batch["counter"])
        return L.arrival_loss(batch["arrivals"], pred)

    def correction_crps(self, batch: dict) -> Tensor:
        """Correction-head CRPS via the P5 head signature."""
        corr = self._require("corr")
        summary = batch.get("summary_raw")
        if summary is None:
            raise IdeaAConfigError(
                "correction_crps needs 'summary_raw' [B, SUMMARY_RAW_F]")
        mask = batch.get("outcome_mask")
        if mask is not None and not bool(mask.any()):
            return summary.sum() * 0.0
        token = self.global_token(batch)
        pred = corr(summary, token)
        targets = self.norm_targets(batch["outcome_quantiles"])
        return L.outcome_crps(pred["outcome_quantiles"], targets, mask)

    def _pairs(self, batch: dict) -> dict:
        from train.dataset import contrast_batch_indices

        meta = batch.get("meta") or []
        # the pair positions index the batch tensors, so they must live on the
        # batch device (a host index tensor on a CUDA batch is an implicit
        # host->device copy at best, an error at worst).
        device = next((v.device for v in batch.values()
                       if torch.is_tensor(v)), None)
        idx = contrast_batch_indices(meta, device=device)
        return idx if int(idx["a"].numel()) > 0 else {}

    @staticmethod
    def _contrast_residual_check(residual: Tensor) -> dict:
        """Detached summary of the supported target-minus-prediction residual."""
        values = residual.detach().reshape(-1)
        finite = torch.isfinite(values)
        finite_values = values[finite]
        return {"count": int(values.numel()),
                "nonzero_count": int((values != 0).sum()),
                "max_abs": (float(finite_values.abs().max())
                            if finite_values.numel() else 0.0),
                "finite": bool(finite.all())}

    def _aligned_hdv_pinball(self, r_pred: Tensor, r_targ: Tensor, ids: Tensor,
                             target_mask: Tensor, is_cav: Optional[Tensor],
                             a: Tensor, b: Tensor, diag: dict) -> Optional[Tensor]:
        """HDV response contrast aligned by VERIFIED canonical vehicle IDs.

        A shared slot NUMBER is not an identity: FORMAT slots are
        persistent/reused run-local positions, so sibling arms may hold the
        same vehicle at DIFFERENT indices and the same index at DIFFERENT
        vehicles.  For every common canonical ID that is HDV-valid in both
        arms, each arm's predicted and realised response is gathered at ITS
        OWN slot index - differing A/B indices - BEFORE the frozen pinball
        primitive (whose TARGET is the scalar per-matched-ID realised
        difference); levels, weights and the loss math are unchanged and the
        IDs never enter the network.  Returns ``None`` when no common
        verified vehicle exists (the sub-term is then excluded, never
        averaged in as a fake zero).
        """
        ids_a = ids[a].to(torch.long)
        ids_b = ids[b].to(torch.long)
        valid_a = target_mask[a].to(torch.bool)
        valid_b = target_mask[b].to(torch.bool)
        if is_cav is not None:
            cav = is_cav.to(torch.bool)
            valid_a = valid_a & ~cav[a]
            valid_b = valid_b & ~cav[b]
        match = ((ids_a[:, :, None] == ids_b[:, None, :])
                 & (ids_a >= 0)[:, :, None] & (ids_b >= 0)[:, None, :]
                 & valid_a[:, :, None] & valid_b[:, None, :])
        p_idx, ja, jb = torch.nonzero(match, as_tuple=True)
        diag["hdv_id_aligned"] = True
        diag["hdv_pairs"] = int(torch.unique(p_idx).numel())
        diag["hdv_slots"] = int(p_idx.numel())
        diag["hdv_subterm"] = bool(p_idx.numel() > 0)
        if p_idx.numel() == 0:
            return None
        ra_pred = r_pred[a[p_idx], ja]
        rb_pred = r_pred[b[p_idx], jb]
        ra_targ = r_targ[a[p_idx], ja]
        rb_targ = r_targ[b[p_idx], jb]
        nq = C.N_QUANTILES
        qa = (ra_pred - rb_pred).unsqueeze(-1).expand(ra_pred.shape[0], nq)
        # the frozen pinball primitive unsqueezes its TARGET (it must
        # broadcast to [..., 1]): the realised difference enters as the
        # SCALAR per matched ID.  A 19-wide target would broadcast against
        # the [nIDs, NQ] prediction - a RuntimeError for every nIDs outside
        # {1, NQ} and a silently wrong cross-quantile mean at nIDs == NQ.
        qb = ra_targ - rb_targ
        diag["residual_checks"]["hdv"] = self._contrast_residual_check(
            qb - (ra_pred - rb_pred))
        levels = torch.tensor(C.QUANTILE_LEVELS, dtype=qa.dtype,
                              device=qa.device)
        return L.pinball(qa, qb, levels).mean()

    def paired_response_contrast(self, batch: dict) -> Tensor:
        """Paired-branch CONTRAST (treatment minus reference, same world+t0).

        Two sibling sub-terms, averaged: the correction-head OUTCOME
        quantile contrast (``model.losses.contrast_loss`` on ``q_a - q_b``
        vs the realised ``y_a - y_b``) and the HDV RESPONSE contrast (the
        predicted residual acceleration difference against the realised
        one, a point prediction broadcast over the quantile levels).
        ``a`` is always the treatment arm, ``b`` the all-native reference
        (``train.dataset.contrast_batch_indices``).

        Contrastfix: a shared ``(world, t0)`` does NOT imply that the two arms
        share an outcome tau or a valid HDV slot.  Each sub-term contributes
        ONLY for the pairs with real shared support; a sub-term with no support
        is excluded entirely (never averaged in as a fake ``0.0``) and a batch
        where neither sub-term has support fails closed with its counts.  The
        support masks are re-derived from the canonical batch here (the
        sampling pool only decides which partners are forced into a batch).
        With verified ``vehicle_ids`` on the batch, the HDV sub-term matches
        sibling arms by COMMON CANONICAL ID (per-arm slot indices may
        differ); without IDs it keeps the documented legacy slot
        intersection (synthetic windows only - an actual packed fit is
        gated on verified identities first).  ``self._contrast_diag``
        records the per-step support counts.
        """
        idx = self._pairs(batch)
        if not idx:
            raise IdeaAConfigError(
                "paired_response_contrast needs a batch with complete "
                "sibling pairs (treatment + native reference)")
        a, b = idx["a"], idx["b"]
        outcome_term: Optional[Tensor] = None
        hdv_term: Optional[Tensor] = None
        diag = {"pairs": int(a.numel()), "outcome_pairs": 0, "outcome_taus": 0,
                "hdv_pairs": 0, "hdv_slots": 0, "outcome_subterm": False,
                "hdv_subterm": False, "hdv_id_aligned": False,
                "residual_checks": {}}
        if self.bundle.corr is not None and batch.get("summary_raw") is not None \
                and batch.get("outcome_quantiles") is not None:
            corr = self._require("corr")
            summary = batch["summary_raw"]
            token = self.global_token(batch)
            q = corr(summary, token)["outcome_quantiles"]
            y = self.norm_targets(batch["outcome_quantiles"])
            pair_mask = CS.shared_outcome_mask(batch.get("outcome_mask"), a, b)
            if pair_mask is None:
                pair_mask = torch.ones(a.shape[0], q.shape[1], dtype=torch.bool,
                                       device=q.device)
            support = pair_mask.any(dim=-1)                 # [P] >=1 shared tau
            diag["outcome_pairs"] = int(support.sum())
            diag["outcome_taus"] = int(pair_mask.sum())
            if bool(support.any()):
                sub = torch.nonzero(support, as_tuple=False).flatten()
                outcome_term = L.contrast_loss(
                    q[a[sub]], q[b[sub]], y[a[sub]], y[b[sub]],
                    mask=pair_mask[sub])
                residual = ((y[a[sub]] - y[b[sub]]).unsqueeze(-1)
                            - (q[a[sub]] - q[b[sub]]))
                diag["residual_checks"]["outcome"] = (
                    self._contrast_residual_check(residual[pair_mask[sub]]))
                diag["outcome_subterm"] = True
        if self.bundle.hdv is not None and batch.get("kin") is not None:
            hdv = self._require("hdv")
            out = hdv(batch["kin"], batch["kin_mask"])
            r_pred = out["accel_delta_mps2"]
            base = batch.get("baseline_idm")
            if base is None:
                base = torch.zeros_like(r_pred)
            r_targ = batch["target_accel"] - base
            ids = batch.get("vehicle_ids")
            if ids is not None:
                hdv_term = self._aligned_hdv_pinball(
                    r_pred, r_targ, ids, batch["target_mask"],
                    batch.get("is_cav"), a, b, diag)
            else:
                pmask = CS.shared_hdv_mask(
                    batch["target_mask"], a, b, batch.get("is_cav"))
                support = pmask.any(dim=-1)                 # [P] >=1 valid slot
                diag["hdv_pairs"] = int(support.sum())
                diag["hdv_slots"] = int(pmask.sum())
                if bool(support.any()):
                    sub = torch.nonzero(support, as_tuple=False).flatten()
                    nq = C.N_QUANTILES
                    qa = r_pred[a[sub]].unsqueeze(-1).expand(
                        *r_pred[a[sub]].shape, nq)
                    qb = r_pred[b[sub]].unsqueeze(-1).expand(
                        *r_pred[b[sub]].shape, nq)
                    levels = torch.tensor(C.QUANTILE_LEVELS, dtype=qa.dtype,
                                          device=qa.device)
                    # the frozen contrast_loss mask layout is pair/tau-shaped,
                    # so the per-slot response term uses the same frozen
                    # pinball primitive over the pairs with real slot support
                    pb = L.pinball(qa - qb,
                                   r_targ[a[sub]] - r_targ[b[sub]], levels)
                    hdv_term = _masked_mean(pb, pmask[sub])
                    residual = ((r_targ[a[sub]] - r_targ[b[sub]])
                                - (r_pred[a[sub]] - r_pred[b[sub]]))
                    diag["residual_checks"]["hdv"] = (
                        self._contrast_residual_check(residual[pmask[sub]]))
                    diag["hdv_subterm"] = True
        self._contrast_diag = diag
        if outcome_term is None and hdv_term is None:
            raise IdeaAConfigError(
                "paired_response_contrast has NO valid contrast support for "
                f"{diag['pairs']} sibling pair(s): 0 shared outcome tau and 0 "
                "shared valid HDV slot; refusing the fail-open zero contrast "
                f"({diag})")
        return self._combine_contrast_subterms(outcome_term, hdv_term)

    def _combine_contrast_subterms(self, outcome_term: Optional[Tensor],
                                   hdv_term: Optional[Tensor]) -> Tensor:
        """Combine the present paired-contrast sub-terms.

        Default (``contrast_outcome_weight is None``) is the historical
        ``sum(terms)/len(terms)`` equal average, so the outcome sub-term is
        halved whenever the HDV sub-term is also present.  With an explicit
        weight ``w`` the outcome sub-term keeps its own weight instead:
        ``w * outcome + 1.0 * hdv`` (a lone sub-term is returned unchanged, so
        a weight can never scale away the only supervised term).
        """
        if outcome_term is not None and hdv_term is not None:
            weight = getattr(self.config, "contrast_outcome_weight", None)
            if weight is None:
                return (outcome_term + hdv_term) / 2.0
            return float(weight) * outcome_term + hdv_term
        term = outcome_term if outcome_term is not None else hdv_term
        assert term is not None
        return term

    # ------------------------------------- paired-effect (climatology anchor) --
    def _paired_plan_vector(self, batch: dict) -> Tensor:
        """The plan input a plan-aware head consumes (A9 short / A11 long).

        The head's ``plan_raw_f`` selects the tensor: the historical 16-D
        ``summary_raw`` (10 s / 20-step CAV-motion summary), or -- when a
        ``--paired-effect-plan-horizon-s`` build widened it -- the A11
        ``plan_long`` vector (same planned CAV channels over a longer window).
        Fails closed if the expected key is absent or its width mismatches the
        head, so a plan-aware head never trains on the wrong plan.
        """
        from model.paired_effect import PLAN_RAW_F

        head = self._require("paired_effect")
        expected = int(getattr(head, "plan_raw_f", 0))
        key = "summary_raw" if expected == int(PLAN_RAW_F) else "plan_long"
        plan = batch.get(key)
        if plan is None:
            raise IdeaAConfigError(
                f"paired_effect_plan_aware needs batch[{key!r}] (the treatment "
                f"arm's {expected}-D CAV-motion plan); refusing to train a "
                "plan-aware head with no plan input")
        if int(plan.shape[-1]) != expected:
            raise IdeaAConfigError(
                f"plan input {key!r} has {int(plan.shape[-1])} features but "
                f"the head expects {expected} (plan_raw_f)")
        return plan

    def paired_effect_rank_loss(self, batch: dict) -> Tensor:
        """A11 within-decision pairwise ranking loss on the head's mean dV.

        Default-OFF (only reached when ``paired_effect_rank_weight > 0`` and the
        bundle carries a paired-effect head).  For every decision group
        (``pair_key`` = same world + t0) with >= 2 outcome-supported treatment
        arms in the batch, the arms are ordered by realised delay contrast
        ``dY = delay(arm) - delay(native)`` (seconds, lower is better) and a
        per-tau logistic penalty presses the head's predicted mean dV to match
        that order; the per-tau ``s_tau`` scale keeps it O(1) and exact ties are
        ignored.  A batch with no rankable group returns an exactly-zero
        differentiable term.  ``self._rank_diag`` records the counts.
        """
        from model.paired_effect import paired_effect_rank_loss as rank_fn

        head = self._require("paired_effect")
        y = batch.get("outcome_quantiles")
        om = batch.get("outcome_mask")
        if y is None or om is None:
            raise IdeaAConfigError(
                "paired_effect_rank_loss needs batch['outcome_quantiles'] and "
                "batch['outcome_mask']")
        meta = batch.get("meta") or []
        from train.dataset import is_reference

        params = list(head.parameters())
        dtype = params[0].dtype if params else y.dtype
        y = y.to(dtype)
        while y.dim() > 2:                 # delay target only (index 0)
            y = y[..., 0]
        om = om.to(torch.bool)
        while om.dim() > 2:
            om = om[..., 0]
        token = self.global_token(batch)
        if bool(getattr(head, "plan_aware", False)):
            plan = self._paired_plan_vector(batch)
            dv = head.dv_quantiles(token, plan=plan)
        else:
            dv = head.dv_quantiles(token)
        pred = dv.mean(dim=-1)             # [B, N_TAU] mean dV per tau
        scale = head.loss_scale()

        groups: dict = {}
        for pos, item in enumerate(meta):
            if isinstance(item, dict) and "pair_key" in item:
                groups.setdefault(item["pair_key"], []).append(pos)
        scores: list = []
        deltas: list = []
        valids: list = []
        rank_pairs = 0
        rank_cells = 0
        for positions in groups.values():
            refs = [p for p in positions
                    if isinstance(meta[p], dict) and is_reference(meta[p])]
            if not refs:
                continue
            ref = refs[0]
            treats = [p for p in positions if p != ref]
            for x in range(len(treats)):
                for z in range(x + 1, len(treats)):
                    i, j = treats[x], treats[z]
                    valid = om[i] & om[j] & om[ref]
                    delta = y[i] - y[j]
                    usable = valid & (delta != 0)
                    if not bool(usable.any()):
                        continue
                    rank_pairs += 1
                    rank_cells += int(usable.sum())
                    scores.append(pred[i] - pred[j])
                    deltas.append(delta)
                    valids.append(valid)
        self._rank_diag = {"rank_pairs": int(rank_pairs),
                           "rank_cells": int(rank_cells),
                           "n_groups": int(len(groups))}
        if not scores:
            return pred.sum() * 0.0
        return rank_fn(torch.stack(scores), torch.stack(deltas),
                       torch.stack(valids), scale=scale, weight=1.0)

    def paired_effect_loss(self, batch: dict) -> Tensor:
        """Pinball loss of the climatology-anchored paired-effect head.

        Default-OFF (only reached when ``paired_effect_head`` is set and the
        bundle carries a :class:`model.paired_effect.PairedEffectHead`).
        Consumes the SAME sibling contrast pairs the response contrast uses:
        only pairs sharing at least one realised outcome tau contribute (never
        a fake zero).  The target is the realised delay contrast
        ``dV = delay_treatment - delay_native`` in seconds and the prediction
        is the head's per-tau dV quantiles on the registered 19-level grid.
        """
        from model.paired_effect import paired_effect_pinball

        head = getattr(self.bundle, "paired_effect", None)
        if not isinstance(head, torch.nn.Module):
            raise IdeaAConfigError(
                "paired_effect_head is enabled but the bundle carries no "
                "paired-effect head")
        idx = self._pairs(batch)
        if not idx:
            raise IdeaAConfigError(
                "paired_effect_loss needs sibling contrast pairs")
        y = batch.get("outcome_quantiles")
        if y is None:
            raise IdeaAConfigError(
                "paired_effect_loss needs batch['outcome_quantiles']")
        a, b = idx["a"], idx["b"]
        params = list(head.parameters())
        dtype = params[0].dtype if params else y.dtype
        y = y.to(dtype)
        while y.dim() > 2:                 # keep the delay target (index 0)
            y = y[..., 0]
        token = self.global_token(batch)
        if bool(getattr(head, "plan_aware", False)):
            plan = self._paired_plan_vector(batch)
            # the plan vector is item a's plan (a = treatment arm)
            dv = head.dv_quantiles(token, plan=plan)[a]   # [P, N_TAU, NQ]
        else:
            dv = head.dv_quantiles(token)[a]              # [P, N_TAU, NQ]
        dy = y[a] - y[b]                   # [P, N_TAU]
        omask = batch.get("outcome_mask")
        if omask is not None:
            om = omask.to(torch.bool)
            while om.dim() > 2:            # [B, N_TAU, ...] -> [B, N_TAU]
                om = om[..., 0]
            mask = om[a] & om[b]
            support = mask.any(dim=-1)
            if not bool(support.any()):
                raise IdeaAConfigError(
                    "paired_effect_loss found no outcome-supported sibling "
                    "pair; refusing a zero-only term")
            sel = torch.nonzero(support, as_tuple=False).flatten()
            return paired_effect_pinball(
                dv[sel], dy[sel], mask=mask[sel],
                weight=float(self.config.paired_effect_weight),
                scale=head.loss_scale())
        return paired_effect_pinball(
            dv, dy, weight=float(self.config.paired_effect_weight),
            scale=head.loss_scale())

    # ------------------------------------------------------- loss assembly --
    def loss_terms(self, batch: dict, *, require_all: Optional[bool] = None
                   ) -> dict:
        """Every computable A3.2 term for this batch (name -> tensor).

        On a real corpus run the trainer sets ``_require_all_terms``: a term
        that the canonical batch builder cannot produce is then a hard error
        (fail closed) instead of a silent skip; only the receipt-recorded
        ``allow_incomplete_losses`` test flag relaxes it.
        """
        batch = self._to_device(batch)
        terms: dict = {}
        if self.bundle.hdv is not None and all(
                k in batch for k in ("kin", "kin_mask", "target_accel",
                                     "target_mask")):
            terms["residual_nll"] = self.residual_nll(batch)
        if self.bundle.hdv is not None and "teacher" in batch \
                and "rollout_accel" in batch:
            terms["rollout"] = self.rollout_loss(batch)
        if self.bundle.arrival is not None and "counter" in batch \
                and "arrivals" in batch:
            terms["arrival"] = self.arrival_loss(batch)
        if self.bundle.corr is not None and "summary_raw" in batch \
                and "outcome_quantiles" in batch:
            terms["correction_crps"] = self.correction_crps(batch)
        if self._pairs(batch):
            terms["response_contrast"] = self.paired_response_contrast(batch)
        if (bool(self.config.paired_effect_head)
                and isinstance(getattr(self.bundle, "paired_effect", None),
                               torch.nn.Module)
                and self._pairs(batch)):
            terms["paired_effect"] = self.paired_effect_loss(batch)
        rank_w = float(getattr(self.config, "paired_effect_rank_weight", 0.0))
        if (rank_w > 0.0 and bool(self.config.paired_effect_head)
                and isinstance(getattr(self.bundle, "paired_effect", None),
                               torch.nn.Module)
                and "outcome_quantiles" in batch):
            terms["paired_effect_rank"] = self.paired_effect_rank_loss(batch)
        if not terms:
            raise IdeaAConfigError(
                "no A3.2 loss term is computable from this batch; check the "
                "bundle modules and the batch keys")
        enforce = (getattr(self, "_require_all_terms", False)
                   if require_all is None else bool(require_all))
        if enforce:
            missing = [name for name in REQUIRED_LOSS_TERMS
                       if name not in terms]
            if missing:
                raise IdeaAConfigError(
                    "registered A3.2 loss term(s) unavailable for a real "
                    f"corpus run: {missing}; the canonical Idea A batch "
                    "builder must supply every term.  Set "
                    "allow_incomplete_losses=True only for a documented, "
                    "receipt-recorded test run")
        return terms

    def _weighted_total(self, terms: dict,
                        weights: Optional[dict] = None) -> Tensor:
        w = self.config.loss_weights()
        if weights:
            w.update(weights)
        total = None
        for name, value in terms.items():
            term = w[name] * value
            total = term if total is None else total + term
        assert total is not None
        return total

    def total_loss(self, batch: dict, weights: Optional[dict] = None) -> Tensor:
        """The weighted sum over the available terms (pre-flight entry)."""
        return self._weighted_total(self.loss_terms(batch), weights)

    def _to_device(self, batch: dict) -> dict:
        if str(self.device) == "cpu":
            return batch
        return _move_to_device(batch, self.device)

    def _assert_on_device(self, batch: dict, *, context: str = "batch") -> None:
        """Fail loudly if params or batch tensors are not on ``self.device``.

        Guards the GPU box path: a host-created tensor that reaches the loss
        would otherwise surface as a confusing cross-device RuntimeError later.
        """
        target = torch.device(self.device)
        for name, module in (("hdv", self.bundle.hdv),
                             ("arrival", self.bundle.arrival),
                             ("corr", self.bundle.corr),
                             ("state_enc", self.bundle.state_enc),
                             ("paired_effect", self.bundle.paired_effect)):
            if not isinstance(module, torch.nn.Module):
                continue
            for p in module.parameters():
                if p.device.type != target.type:
                    raise IdeaAConfigError(
                        f"{name} parameter on {p.device} != trainer device "
                        f"{target} ({context})")
        for key, value in batch.items():
            if torch.is_tensor(value) and value.device.type != target.type:
                raise IdeaAConfigError(
                    f"{context} tensor {key!r} on {value.device} != trainer "
                    f"device {target}; a host-created tensor reached the loss")

    # ------------------------------------------------------------- training --
    def _contrast_pool(self, source, meta) -> Optional[list]:
        """The seeded-canonical eligible sibling pair pool for ``source``.

        Computed once per ``fit`` from the source's real mask arrays via
        :func:`train.contrast_support.build_eligible_pair_pool`; the same list
        is handed to the in-process :func:`epoch_batches` and the
        ``PairAwareBatchSampler`` so the seeded pair-aware order is identical.
        Returns ``None`` when the source exposes no item metadata (explicit
        windows / quick-tune window corpora fall back to their prior path).
        """
        if meta is None:
            return None
        return CS.build_eligible_pair_pool(
            source, meta,
            require_outcome=bool(
                getattr(self.config, "contrast_require_outcome_support",
                        False)))

    def _pairs_for(self, source, meta, *, contrast_pairs) -> list:
        """Eligible pool when precomputed, else the raw policy-native pairs."""
        pool = getattr(self, "_contrast_pair_pool", None)
        if pool is not None:
            return pool
        return contrast_pairs(meta) if meta else []

    def _iter_batches(self, source):
        """Yield canonical batches forever (one epoch per outer loop).

        On the real packed-corpus path the epoch is a seeded random
        permutation of every item (unchanged: each item is seen once per
        epoch), but every yielded batch is completed to contain at least one
        sibling pair (a treatment arm + the policy-native reference of the
        same ``pair_key``) so the registered CONTRAST loss is always
        computable and ``_require_all_terms`` stays fail-closed.  The pair
        partner may repeat across batches; a group with no policy-native
        reference (chunk1/fix27 switching ``__native``) forms no pair and its
        items still train the other losses.
        """
        if isinstance(source, dict) or not hasattr(source, "__len__"):
            raise IdeaAConfigError(
                "fit expects a corpus (len + getitem) or a list of windows")
        from train.dataset import contrast_pairs, epoch_batches

        n = len(source)
        if n == 0:
            return
        bs = max(1, int(self.config.batch_size))
        corpus_style = hasattr(source, "corpus") or hasattr(source, "pack_dir")
        meta = _source_item_meta(source) if corpus_style else None
        pairs = self._pairs_for(source, meta, contrast_pairs=contrast_pairs)
        epoch = 0
        while True:
            if meta is not None:
                epoch_idx = epoch_batches(
                    meta, bs, self.seed, epoch, pairs,
                    pairs_per_batch=getattr(self.config, "pairs_per_batch",
                                            None),
                    merge_coverage=bool(getattr(
                        self.config, "merge_coverage_into_pairs", False)),
                    group_aware=bool(getattr(
                        self.config, "paired_effect_group_batching", False)))
            else:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int((self.seed * 1000003 + epoch) % (2 ** 63 - 1)))
                order = torch.randperm(n, generator=generator).tolist()
                epoch_idx = [order[s:s + bs] for s in range(0, n, bs)]
            for idx in epoch_idx:
                if not idx:
                    continue
                if corpus_style:
                    items = [source[j] for j in idx]
                    first = items[0]
                    if "veh_step" in first:
                        yield batch_from_corpus_items(items)
                    else:
                        yield collate_canonical(items)
                else:                       # windows are already batches
                    yield collate_canonical([source[j] for j in idx])
            epoch += 1

    def _iter_batches_loader(self, source):
        """Yield canonical batches forever through a pair-aware DataLoader.

        The batch positions come from :class:`_LoaderPassSampler` (a
        :class:`train.dataset.PairAwareBatchSampler` yielding the same
        :func:`train.dataset.epoch_batches` the in-process path uses), so the
        DataLoader batches equal the in-process batches for the same seed.
        Exactly ONE DataLoader is constructed (persistent workers with
        prefetch when workers are configured; ``pin_memory`` only for CUDA)
        and re-iterated once per corpus pass: ``set_epoch`` advances the
        seeded pass exactly once per COMPLETE pass, so the sampler is never
        reset mid-step or mid-pass and no primary item is ever dropped.  A
        pass that yields no batch (empty / zero-yield source) raises
        fail-closed instead of busy-spinning the worker feed or silently
        ending the schedule.
        """
        from torch.utils.data import DataLoader

        from train.dataset import contrast_pairs

        meta = _source_item_meta(source)
        if meta is None:
            raise IdeaAConfigError(
                "the DataLoader path needs a source exposing item metadata "
                "(pair_key/policy); use num_workers=0 for explicit windows")
        pairs = self._pairs_for(source, meta, contrast_pairs=contrast_pairs)
        nw = int(self.config.resolved_num_workers())
        collate_fn = _collate_for_source(source)
        sampler = _LoaderPassSampler(
            meta, self.config.batch_size, self.seed, epoch=0, epochs=1,
            pairs=pairs,
            pairs_per_batch=getattr(self.config, "pairs_per_batch", None),
            merge_coverage=bool(getattr(
                self.config, "merge_coverage_into_pairs", False)),
            group_aware=bool(getattr(
                self.config, "paired_effect_group_batching", False)))
        kwargs = dict(batch_sampler=sampler, collate_fn=collate_fn,
                      num_workers=nw)
        if nw > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = int(self.config.prefetch_factor)
        if str(self.device).startswith("cuda"):
            kwargs["pin_memory"] = True
        loader = DataLoader(source, **kwargs)
        epoch = 0
        while True:
            sampler.set_epoch(epoch)          # once per COMPLETE corpus pass
            yielded = 0
            for batch in loader:
                yielded += 1
                yield batch
            if not yielded:
                raise IdeaAConfigError(
                    "the DataLoader batch source yielded 0 batches for "
                    f"seeded corpus pass {epoch} over {len(meta)} item(s); "
                    "refusing a zero-yield schedule instead of busy-spinning "
                    "or ending the batch stream early")
            epoch += 1

    def zero_grad_modules(self, set_to_none: bool = True) -> None:
        """Zero the gradients of every bundle module."""
        for module in (self.bundle.hdv, self.bundle.arrival, self.bundle.corr,
                       self.bundle.state_enc, self.bundle.paired_effect):
            if isinstance(module, torch.nn.Module):
                module.zero_grad(set_to_none=set_to_none)

    def _ensure_optimizer(self, total_steps: int):
        """Create the Adam optimizer + cosine schedule on the first batch.

        A pending optimizer/scheduler state staged by :meth:`resume` (saved
        when no frame existed yet) is loaded here, so ``resume(); fit()``
        continues the recorded schedule instead of silently restarting it
        (resume determinism).
        """
        if getattr(self, "_optimizer", None) is None:
            self._optimizer, self._scheduler = _make_optimizer(
                self, self.config.lr, total_steps)
            pending = getattr(self, "_pending_optimizer_state", None)
            if pending is not None:
                self._optimizer.load_state_dict(pending)
                self._pending_optimizer_state = None
            pending = getattr(self, "_pending_scheduler_state", None)
            if pending is not None and self._scheduler is not None:
                self._scheduler.load_state_dict(pending)
                self._pending_scheduler_state = None
        return self._optimizer, self._scheduler

    @staticmethod
    def label_t0_telemetry(batch: dict) -> dict:
        """Per-step (a) telemetry: t0-aligned label rows asserted / violating.

        The dataset asserts the t0 pick PER ITEM (a misaligned or duplicated
        row raises there), so a step that completes always reports zero
        violations; the counters are plain host ints written identically on
        CPU and CUDA.  ``train_log.jsonl`` carries ``label_t0_asserted`` (the
        summed per-item t0 label rows) and ``label_t0_violations``, and the
        standalone ``tools/wmn1_log_check.py`` fails closed on any violation
        or on a zero total asserted count.
        """
        asserted = 0
        violations = 0
        for row in batch.get("meta") or []:
            if not isinstance(row, dict):
                continue
            try:
                asserted += int(row.get("label_t0_rows", 0) or 0)
                violations += int(row.get("label_t0_violations", 0) or 0)
            except (TypeError, ValueError):
                continue
        return {"label_t0_asserted": asserted,
                "label_t0_violations": violations}

    def _optim_step(self, batch: dict, optimizer) -> float:
        optimizer.zero_grad(set_to_none=True)
        self._contrast_diag = None
        self._rank_diag = None
        self._last_label_t0 = self.label_t0_telemetry(batch)
        terms = self.loss_terms(batch)
        self._last_terms = sorted(terms)
        self._last_term_values = {k: float(v.detach()) for k, v in terms.items()}
        if len(self.loss_magnitudes) < int(self.config.loss_balance_batches):
            self.loss_magnitudes.append(
                {k: float(v.detach()) for k, v in terms.items()})
        loss = self._weighted_total(terms)
        loss.backward()
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           float(self.config.grad_clip))
        optimizer.step()
        return float(loss.detach())

    def _log_step(self, log_file, log_every: int,
                  step_seconds: float) -> None:
        """Append one JSON line to ``train_log.jsonl`` (print every N)."""
        t0 = getattr(self, "_last_label_t0", None) or {}
        record = {"step": int(self.step),
                  "total": float(self.losses[-1]) if self.losses else None,
                  "terms": dict(getattr(self, "_last_term_values", {})),
                  "contrast_support": getattr(self, "_contrast_diag", None),
                  "rank_support": getattr(self, "_rank_diag", None),
                  "s_per_step": float(step_seconds),
                  "label_t0_asserted": int(t0.get("label_t0_asserted", 0)),
                  "label_t0_violations": int(t0.get("label_t0_violations", 0))}
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            record["cuda_mem_bytes"] = int(torch.cuda.memory_allocated())
            record["cuda_peak_bytes"] = int(torch.cuda.max_memory_allocated())
        else:
            record["cuda_mem_bytes"] = None
            record["cuda_peak_bytes"] = None
        log_file.write(json.dumps(record, sort_keys=True) + "\n")
        log_file.flush()
        if log_every > 0 and int(self.step) % log_every == 0:
            terms = " ".join(f"{k}={v:.4g}"
                             for k, v in sorted(record["terms"].items()))
            diag = record["contrast_support"]
            support = ("" if not diag else
                       f" contrast[pairs={diag['pairs']} "
                       f"out={diag['outcome_pairs']}/{diag['outcome_taus']} "
                       f"hdv={diag['hdv_pairs']}/{diag['hdv_slots']}]")
            rd = record.get("rank_support")
            support += ("" if not rd else
                        f" rank[pairs={rd['rank_pairs']} "
                        f"cells={rd['rank_cells']} groups={rd['n_groups']}]")
            print(f"[train] step={record['step']} "
                  f"total={record['total']:.6g} {terms}{support} "
                  f"s/step={record['s_per_step']:.4f} "
                  f"cuda_mem={record['cuda_mem_bytes']}", flush=True)

    def _require_member_1_first(self) -> None:
        if self.member > 1:
            first = Path(self.config.out_dir) / "member_1.pt"
            if not first.is_file():
                raise IdeaAConfigError(
                    f"member {self.member} refused: member 1 has not "
                    f"completed ({first} is absent; amendment A1)")

    def _require_train_source(self, source) -> None:
        """``fit`` trains on TRAIN corpora only (amendment A3.2).

        CAL is calibration and variant selection only (:meth:`evaluate_cal`
        reads it as a diagnostic), HOLDOUT/TEST are sealed; a corpus-like
        source carrying any other ``split`` is refused before step one.
        """
        split = getattr(source, "split", None)
        if split is None:
            return
        text = str(split).strip().upper()
        resolved = {"TRAIN": "TRAIN", "CAL": "CALIBRATION",
                    "CALIBRATION": "CALIBRATION"}.get(text, text)
        if resolved != "TRAIN":
            raise IdeaAConfigError(
                f"fit refuses a {resolved} corpus: CAL is calibration and "
                "variant selection only and HOLDOUT/TEST are sealed (A3.2); "
                "pass the TRAIN corpus or explicit TRAIN windows")

    def _require_verified_identities(self, source, windows) -> dict:
        """The verified-identity gate (step 3 of :meth:`prepare_training_source`).

        An actual (packed-corpus) fit REQUIRES the source to verify every
        item's identity (missing/unknown/duplicated raises inside the
        accessor); the HDV sub-term is never silently disabled.  Explicit
        synthetic windows carry their own explicit fixture IDs (or none, on
        the documented legacy path) and are never identity-guessed.
        """
        if windows is not None:
            with_ids = sum(1 for w in windows
                           if isinstance(w, dict)
                           and w.get("vehicle_ids") is not None)
            return {"mode": "explicit-windows", "n_windows": len(windows),
                    "n_with_vehicle_ids": int(with_ids)}
        fn = getattr(source, "verified_identities", None)
        if not callable(fn):
            raise IdeaAConfigError(
                "an actual fit requires VERIFIED per-item TRAIN vehicle "
                f"identities; source {type(source).__name__} exposes no "
                "verified_identities() accessor (missing/unknown/duplicated "
                "identity must fail, never silently disable the HDV "
                "sub-term)")
        receipt = fn()
        if not isinstance(receipt, dict) \
                or receipt.get("status") != "verified":
            raise IdeaAConfigError(
                f"verified identity gate failed: {receipt!r}")
        if not receipt.get("receipt_sha256"):
            raise IdeaAConfigError(
                "the verified identity receipt carries no receipt_sha256")
        return receipt

    def prepare_training_source(self, source, windows=None) -> dict:
        """The ONE strict fit-preparation entry (identity recovery contract).

        Runs, in order, BEFORE any batch is built and before the optimizer
        exists: (1) the TRAIN-split gate, (2) the Amendment 6 seal,
        (2b) the quick-tune TRAIN narrowing when configured, (3) the
        VERIFIED per-item identity gate, (4) the ID-aware eligible sibling
        pair pool with its empty-pool fail-closed refusal, (5) the
        TRAIN-only outcome normalization fit.  CAL is never read here.

        The helper also BINDS the exact prepared state on the trainer
        (``_contrast_pair_pool``, ``_identity_receipt``, ``outcome_norm``)
        BEFORE it returns, so a strict timing caller that only runs this
        helper - never ``fit`` - iterates and scales with the SAME sampler
        pool, normalization and identity receipt instead of the
        constructor's identity norm or an unfiltered legacy pair fallback.
        ``fit`` consumes the returned record without refitting anything.
        ``windows`` (explicit synthetic windows) replaces
        ``source`` for this fit and skips the packed-corpus identity gate
        (windows carry their own explicit fixture IDs, if any).
        """
        prepared = windows if windows is not None else source
        quick_receipt = None
        if self.config.quick_tune:
            if windows is not None or not hasattr(source, "_items"):
                raise IdeaAConfigError(
                    "quick-tune needs a packed corpus with item metadata "
                    "(train.dataset.MemmapCorpus), not explicit windows")
            prepared, quick_receipt = quick_tune_view(
                source, self.config.subset_frac, seed=self.seed)
        self._require_train_source(prepared)
        # A6 gate: state_global_token defaults ON only while the frozen A6
        # record is present; refuse before touching any data otherwise.
        self.a6_binding()
        require_all = bool(windows is None
                           and not self.config.allow_incomplete_losses)
        self._require_all_terms = require_all
        identity = self._require_verified_identities(prepared, windows)
        meta = _source_item_meta(prepared)
        pool = self._contrast_pool(prepared, meta)
        if require_all and meta is not None and not pool:
            requirement = ("shared outcome-tau"
                           if bool(getattr(self.config,
                                           "contrast_require_outcome_support",
                                           False))
                           else "shared outcome-tau or valid-HDV-slot")
            raise IdeaAConfigError(
                "the TRAIN source has no eligible sibling pair with "
                f"{requirement} support for the registered "
                f"CONTRAST loss ({len(meta)} item(s), 0 eligible pairs); "
                "refusing a zero-contrast loss (contrastfix)")
        # Owner decision (b): arms in OUTCOME_SUPERVISION_EXCLUDED_ARMS lose
        # outcome supervision.  This FAIL-CLOSED receipt re-reads each such
        # item through the same outcome_target accessor the normalization
        # consumes and refuses if any label still leaks; the count/list is
        # carried into TRAIN_RECEIPT.json.  Synthetic windows have no packed
        # item list and yield an empty receipt.
        from train.dataset import (OUTCOME_SUPERVISION_EXCLUDED_ARMS,
                                   outcome_supervision_exclusion)

        norm_corpus = _norm_corpus_of(prepared)
        if windows is None and norm_corpus is not None:
            exclusion = outcome_supervision_exclusion(norm_corpus)
        else:
            exclusion = {"arms": list(OUTCOME_SUPERVISION_EXCLUDED_ARMS),
                         "count": 0, "items": []}
        self._outcome_supervision_receipt = exclusion
        outcome_norm = fit_outcome_norm(
            prepared, windows=windows,
            min_scale=float(self.config.outcome_norm_min_scale))
        # CANONICAL BINDING (before ANY return, fit and timing alike): the
        # exact prepared pool / normalization / identity receipt are bound on
        # the trainer here, once, so both fit and the strict timing path see
        # the identical sampler and normalization state.
        self._contrast_pair_pool = pool
        self._identity_receipt = identity
        self.outcome_norm = outcome_norm
        return {"source": prepared, "windows": windows,
                "require_all_terms": require_all, "identity": identity,
                "pair_pool": pool,
                "n_meta_items": None if meta is None else len(meta),
                "quick_receipt": quick_receipt,
                "amendment_6_lf_sha256": getattr(self, "_a6_sha", None),
                "outcome_supervision_exclusion": exclusion,
                "outcome_norm": outcome_norm}

    def fit(self, windows: Optional[Sequence[dict]] = None) -> dict:
        """Train one member; member 1 first.

        ALL preparation (TRAIN gate, A6 seal, verified identities, eligible
        pool, fail-closed checks, outcome normalization) runs through
        :meth:`prepare_training_source` BEFORE any batch or optimizer
        exists.  Writes ``member_<k>.pt`` (state_dict + JSON sidecar,
        ensemble flags recorded) and ``TRAIN_RECEIPT.json`` (with the
        identity block) under ``config.out_dir``.

        Both batch iterators are infinite seeded corpus passes; a batch
        stream that ends before the FULL requested schedule is an unexpected
        exhaustion and refuses fail-closed (the partial ``train_log.jsonl``
        is preserved, nothing is cleaned up, and no checkpoint,
        ``OUTCOME_NORM.json`` or ``TRAIN_RECEIPT.json`` is written) - never a
        silent early stop with an under-scheduled success receipt.
        """
        self._require_member_1_first()
        prep = self.prepare_training_source(self.corpus, windows=windows)
        source = prep["source"]
        # The canonical helper above already bound the exact pair pool,
        # identity receipt, outcome normalization and _require_all_terms on
        # this trainer BEFORE returning; fit consumes that SAME prepared
        # state (no refit, no divergent sampler/norm).
        self._last_terms = []
        self._last_term_values = {}
        self._last_label_t0 = {"label_t0_asserted": 0,
                               "label_t0_violations": 0}
        quick_receipt = prep["quick_receipt"]
        if int(self.config.resolved_num_workers()) > 0:
            batches = self._iter_batches_loader(source)
        else:
            batches = self._iter_batches(source)
        epochs = self.config.effective_epochs()
        total_steps = int(epochs) * int(self.config.steps_per_epoch)
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_every = max(0, int(getattr(self.config, "log_every", 10)))
        optimizer, scheduler = None, None
        log_file = (out_dir / "train_log.jsonl").open("w", encoding="utf-8")
        try:
            for epoch in range(self.epoch, int(epochs)):
                epoch_losses = []
                for _ in range(int(self.config.steps_per_epoch)):
                    batch = next(batches, None)
                    if batch is None:
                        raise IdeaAConfigError(
                            "the batch iterator ended early: epoch "
                            f"{epoch + 1}/{int(epochs)} stopped after "
                            f"{len(epoch_losses)}/"
                            f"{int(self.config.steps_per_epoch)} step(s) "
                            f"(total step counter {self.step}); refusing an "
                            "under-scheduled receipt - the partial "
                            "train_log.jsonl is preserved and no checkpoint, "
                            "OUTCOME_NORM.json or TRAIN_RECEIPT.json is "
                            "written")
                    if optimizer is None:
                        optimizer, scheduler = self._ensure_optimizer(total_steps)
                    if self.step == 0:
                        self._assert_on_device(
                            self._to_device(batch),
                            context="first fit batch")
                    t0 = time.perf_counter()
                    loss = self._optim_step(batch, optimizer)
                    dt = time.perf_counter() - t0
                    if scheduler is not None:
                        scheduler.step()
                    epoch_losses.append(loss)
                    self.losses.append(loss)
                    self.step += 1
                    self._log_step(log_file, log_every, dt)
                self.epoch = epoch + 1
                mean = (sum(epoch_losses) / len(epoch_losses)
                        if epoch_losses else float("nan"))
                self.history.append({"epoch": self.epoch, "mean_loss": mean})
        finally:
            log_file.close()
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        norm_path = out_dir / "OUTCOME_NORM.json"
        self._outcome_norm_sha = self.outcome_norm.write(norm_path)
        self.loss_balance = loss_balance_report(self.loss_magnitudes)
        ckpt = self.save_checkpoint(out_dir / f"member_{self.member}.pt")
        receipt = {"member": self.member, "seed": self.seed,
                   "steps": self.step, "epochs": self.epoch,
                   "losses": list(self.losses), "history": list(self.history),
                   "checkpoint": str(ckpt),
                   "quick_tune": bool(self.config.quick_tune),
                   "subset_frac": float(self.config.subset_frac),
                   "epoch_frac": float(self.config.epoch_frac),
                   "subset": quick_receipt,
                   "active_loss_terms": list(getattr(self, "_last_terms", [])),
                   "required_loss_terms": list(REQUIRED_LOSS_TERMS),
                   "allow_incomplete_losses":
                       bool(self.config.allow_incomplete_losses),
                   "state_global_token": bool(self.config.state_global_token),
                   "amendment_6_lf_sha256":
                       getattr(self, "_a6_sha", None),
                   "num_workers": int(self.config.resolved_num_workers()),
                   "pack_source": (str(source.pack_dir)
                                   if getattr(source, "pack_dir", None)
                                   else None),
                   "outcome_norm_sha256": self._outcome_norm_sha,
                   "outcome_norm_file": str(norm_path),
                   "outcome_norm": self.outcome_norm.to_dict(),
                   "outcome_supervision_excluded_arms": list(getattr(
                       self, "_outcome_supervision_receipt",
                       {}).get("arms", [])),
                   "outcome_supervision_excluded_count": int(getattr(
                       self, "_outcome_supervision_receipt",
                       {}).get("count", 0)),
                   "outcome_supervision_excluded_items": list(getattr(
                       self, "_outcome_supervision_receipt",
                       {}).get("items", [])),
                   "loss_magnitudes": list(self.loss_magnitudes),
                   "loss_balance": self.loss_balance,
                   "contrast_support": getattr(self, "_contrast_diag", None),
                   "contrast_pair_pool": (None if getattr(
                       self, "_contrast_pair_pool", None) is None
                       else len(self._contrast_pair_pool)),
                   "pairs_per_batch": getattr(self.config, "pairs_per_batch",
                                              None),
                   "merge_coverage_into_pairs": bool(getattr(
                       self.config, "merge_coverage_into_pairs", False)),
                   "contrast_require_outcome_support": bool(getattr(
                       self.config, "contrast_require_outcome_support",
                       False)),
                   "contrast_outcome_weight": getattr(
                       self.config, "contrast_outcome_weight", None),
                   "paired_effect_head": bool(self.config.paired_effect_head),
                   "paired_effect_weight": float(
                       self.config.paired_effect_weight),
                   "paired_effect_climatology": getattr(
                       self.config, "paired_effect_climatology", None),
                   "paired_effect_climatology_sha256": (
                       self._paired_effect_climatology_sha()),
                   "paired_effect_climatology_pin": getattr(
                       self.config, "paired_effect_climatology_sha256", None),
                   "paired_effect_loss_scale_s": (
                       [float(x) for x in self.bundle.paired_effect
                        .loss_scale().detach().cpu().tolist()]
                       if isinstance(getattr(self.bundle, "paired_effect",
                                             None), torch.nn.Module)
                       else None),
                   "paired_effect_plan_aware": bool(getattr(
                       self.config, "paired_effect_plan_aware", False)),
                   "paired_effect_plan_dim": (
                       int(getattr(self.bundle.paired_effect, "plan_dim", 0))
                       if isinstance(getattr(self.bundle, "paired_effect",
                                             None), torch.nn.Module)
                       else None),
                   "paired_effect_plan_raw_f": (
                       int(getattr(self.bundle.paired_effect, "plan_raw_f", 0))
                       if isinstance(getattr(self.bundle, "paired_effect",
                                             None), torch.nn.Module)
                       else None),
                   "paired_effect_plan_horizon_s": getattr(
                       self.config, "paired_effect_plan_horizon_s", None),
                   "paired_effect_rank_weight": float(getattr(
                       self.config, "paired_effect_rank_weight", 0.0)),
                   "paired_effect_rank_diag": getattr(
                       self, "_rank_diag", None),
                   "paired_effect_group_batching": bool(getattr(
                       self.config, "paired_effect_group_batching", False)),
                   "identity": getattr(self, "_identity_receipt", None),
                   **ensemble_flags(self.config.ensemble_k)}
        (out_dir / "TRAIN_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return receipt

    # ---------------------------------------------------------- checkpoint --
    def save_checkpoint(self, path) -> Path:
        """Persist plain ``state_dict`` + optimizer + ensemble flags.

        The JSON sidecar carries the same metadata without tensors; every
        ``torch.load`` sets ``weights_only`` explicitly (torch-2.5 safe).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = ensemble_flags(self.config.ensemble_k)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "state_dict": self.state_dict(),
            "optimizer": self._optimizer_state(),
            "scheduler": self._scheduler_state(),
            "epoch": int(self.epoch),
            "step": int(self.step),
            "member": self.member,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "outcome_norm": self.outcome_norm.to_dict(),
            "outcome_norm_sha256": self._outcome_norm_sha,
            "state_global_token": bool(self.config.state_global_token),
            "amendment_6_lf_sha256": self.a6_binding(),
            "num_workers": int(self.config.resolved_num_workers()),
            "loss_balance": getattr(self, "loss_balance", None),
            "gen_state": self.gen.get_state(),
            "rng_state": torch.get_rng_state(),
            **flags,
        }
        torch.save(payload, path)
        sidecar = {k: v for k, v in payload.items()
                   if k not in ("state_dict", "optimizer", "scheduler",
                                "gen_state", "rng_state")}
        path.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return path

    def _optimizer_state(self):
        frame = getattr(self, "_optimizer", None)
        if frame is not None:
            return frame.state_dict()
        return getattr(self, "_pending_optimizer_state", None)

    def _scheduler_state(self):
        frame = getattr(self, "_scheduler", None)
        if frame is not None:
            return frame.state_dict()
        return getattr(self, "_pending_scheduler_state", None)

    def resume(self, path, *, optimizer=None) -> "IdeaATrainer":
        """Restore weights/optimizer/epoch from ``path`` and return self.

        Optimizer/scheduler states are restored into the given or existing
        frames; when no frame exists yet (the natural ``resume(); fit()``
        flow) they are staged and loaded by :meth:`_ensure_optimizer`, so a
        resumed member continues the recorded Adam moments and cosine
        schedule instead of silently restarting them.
        """
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise IdeaAConfigError(
                f"checkpoint {path} format {ckpt.get('format')!r} != "
                f"{CHECKPOINT_FORMAT!r}")
        self.load_state_dict(ckpt["state_dict"])
        opt_state = ckpt.get("optimizer")
        frame = optimizer if optimizer is not None else getattr(
            self, "_optimizer", None)
        if opt_state is not None and frame is not None:
            frame.load_state_dict(opt_state)
        elif opt_state is not None:
            self._pending_optimizer_state = opt_state
        sched_state = ckpt.get("scheduler")
        sched = getattr(self, "_scheduler", None)
        if sched_state is not None and sched is not None:
            sched.load_state_dict(sched_state)
        elif sched_state is not None:
            self._pending_scheduler_state = sched_state
        self.epoch = int(ckpt.get("epoch", 0))
        self.step = int(ckpt.get("step", 0))
        self.member = int(ckpt.get("member", self.member))
        self.seed = int(ckpt.get("seed", self.seed))
        if ckpt.get("ensemble_k") is not None:
            stored = int(ckpt["ensemble_k"])
            if stored != int(self.config.ensemble_k):
                raise IdeaAConfigError(
                    f"checkpoint ensemble_k={stored} != config "
                    f"ensemble_k={self.config.ensemble_k}")
        if "gen_state" in ckpt:
            self.gen.set_state(ckpt["gen_state"])
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
        else:                       # checkpoint written before rng_state
            torch.manual_seed(self.seed)
        if ckpt.get("outcome_norm") is not None:
            from model.outcome_norm import OutcomeNorm

            self.outcome_norm = OutcomeNorm.from_dict(ckpt["outcome_norm"])
            self._outcome_norm_sha = ckpt.get("outcome_norm_sha256")
        return self

    # ------------------------------------------------------------- cal eval --
    def evaluate_cal(self, corpus=None, *, half: str = None,
                     cal_split_path=None) -> dict:
        """Diagnostic CAL mean loss.

        ``half`` is REQUIRED for a non-empty CAL corpus (``"cal_cal"`` for
        calibration/early stopping or ``"cal_tune"`` for the improvement-loop
        selection metric): the corpus is restricted to that half of
        ``CAL_SPLIT_A4.json`` and the other half is never scored.  Callers on
        the selection path must pass ``half="cal_tune"`` and callers on the
        calibration path ``half="cal_cal"`` -- the single
        :mod:`evalh.cal_split` filter refuses a world from the wrong half.  A
        missing or unknown half raises (A4.2); an empty/unavailable CAL corpus
        returns ``empty``/``unavailable`` without scoring.
        """
        cal = corpus if corpus is not None else self.corpus
        if cal is None:
            return {"status": "unavailable", "loss": None}
        if getattr(cal, "split", "TRAIN") != "CALIBRATION":
            pack_dir = getattr(cal, "pack_dir", None)
            if pack_dir is None:
                return {"status": "unavailable", "loss": None}
            try:
                cal = IdeaACorpus(pack_dir, split="CALIBRATION",
                                  horizon_steps=self.config.rollout_horizon_steps,
                                  k_max=self.config.k_max)
            except Exception as exc:                       # noqa: BLE001
                return {"status": "unavailable", "loss": None,
                        "error": str(exc)}
        if len(cal) == 0:
            return {"status": "empty", "loss": None, "n": 0}
        # Amendment A4.2: every CAL score must name its half.  A CAL corpus
        # with items and no (or an unknown) half is refused, so neither the
        # selection nor the calibration path can read the whole CAL set.
        if half not in (cal_split.CAL_TUNE, cal_split.CAL_CAL):
            raise IdeaAConfigError(
                f"evaluate_cal requires half in {list(cal_split.HALVES)} "
                f"(the A4.2 CAL-half rule); got {half!r}")
        allowed = set(cal_split.half_worlds(half, path=cal_split_path))
        positions = []
        for i in range(len(cal)):
            item = cal[i]
            world = item.get("world") if isinstance(item, dict) else None
            if world is None:
                raise IdeaAConfigError(
                    f"evaluate_cal(half={half!r}) needs CAL item metadata "
                    "carrying 'world' to enforce the A4.2 split")
            if str(world) in allowed:
                positions.append(i)
        if not positions:
            return {"status": "empty", "loss": None, "n": 0, "half": half}
        total, n = 0.0, 0
        with torch.no_grad():
            for i in positions:
                item = cal[i]
                batch = batch_from_corpus_items([item]) \
                    if "veh_step" in item else collate_canonical([item])
                loss = self.total_loss(batch)
                total += float(loss.detach())
                n += 1
        out = {"status": "ok", "member": self.member, "seed": self.seed,
               "n": n, "loss": total / max(1, n)}
        if half is not None:
            out["half"] = half
        return out


# ------------------------------------------------------- pre-flight gates ----
def write_g4_tiny_pack(windows: Sequence[dict], root) -> Path:
    """Encode canonical windows into a tiny real-format packed corpus.

    The G4 gate overfits the REAL Idea A corpus path; this writes the crafted
    ``train.g4_standins.make_windows`` values as the frozen P9 channels
    (``veh_kin``/``hdv_base``/``cav_plan_ch``/events) plus the v1 tensors and
    the outcome rows, so :class:`IdeaACorpus` rebuilds the batch through the
    SAME builder a GPU-box run uses.  ``summary_raw`` is not stored - it is the
    deterministic :func:`corr.summary.cav_motion_summary` of the CAV channels.
    """
    import pandas as pd

    from model import constants as C

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    windows = list(windows)
    H = int(windows[0]["teacher"].shape[0]) - 1
    K = int(windows[0]["s0"].shape[1])
    per_run = H + 1
    runs = [(wi, b) for wi, w in enumerate(windows)
            for b in range(int(w["s0"].shape[0]))]
    n_runs, n_tick = len(runs), len(runs) * per_run

    veh = np.zeros((n_tick, K, C.VEH_F), np.float16)
    mask = np.zeros((n_tick, K), np.uint8)
    plan = np.zeros((n_runs, K, C.N_SEGMENTS, C.PLAN_F), np.float16)
    wall = np.zeros((n_tick, C.MAX_WALLS, C.WALL_F), np.float16)
    wall_mask = np.zeros((n_tick, C.MAX_WALLS), np.uint8)
    counter = np.zeros((n_tick, C.N_ENTRANCES, C.COUNTER_BINS), np.float16)
    accel = np.zeros((n_tick, K), np.float16)
    target_mask = np.zeros((n_tick, K), np.uint8)
    lane_next = np.full((n_tick, K), -1, np.int8)
    arrivals = np.zeros((n_tick, C.N_ENTRANCES), np.int16)
    veh_kin = np.zeros((n_tick, K, 7), np.float16)
    hdv_base = np.zeros((n_tick, K, 6), np.float16)
    cav_plan = np.zeros((n_tick, K, 8), np.float16)
    event_tok = np.zeros((n_tick, 8, 8), np.float16)
    event_mask = np.zeros((n_tick, 8), np.uint8)
    rows, items, ranges = [], [], []
    for r, (wi, b) in enumerate(runs):
        w = windows[wi]
        base = r * per_run
        teacher = w["teacher"][:, b]
        roll_acc = w["rollout_accel"][:, b]
        roll_mask = w["rollout_mask"][:, b]
        base_idm = w["baseline_idm"][b]
        kmask = w["kin_mask"][b].numpy().astype(bool)
        present = w["present0"][b].numpy().astype(bool)
        is_cav = w["is_cav"][b].numpy().astype(bool)
        length0 = w["length0"][b].numpy().astype(np.float64)
        lane_t = w["lane_target"][b].numpy().astype(np.int64)
        for t in range(per_run):
            row = teacher[t]
            s = row[:, TEACHER_S].numpy().astype(np.float64)
            lane = row[:, TEACHER_LANE].numpy().astype(np.float64)
            v = row[:, TEACHER_V].numpy().astype(np.float64)
            a = row[:, TEACHER_A].numpy().astype(np.float64)
            veh[base + t, :, C.S_DOWN] = s.astype(np.float16)
            for li in range(C.N_LANES):
                veh[base + t, :, C.LANE_LEFT + li] = (lane == li).astype(
                    np.float16)
            veh[base + t, :, C.V_IDX] = v.astype(np.float16)
            veh[base + t, :, C.A_IDX] = a.astype(np.float16)
            veh[base + t, :, C.LENGTH_IDX] = length0.astype(np.float16)
            veh[base + t, :, C.IS_CAV_IDX] = is_cav.astype(np.float16)
            veh[base + t, :, C.ACT_NATIVE] = is_cav.astype(np.float16)
            veh[base + t, :, C.ACT_UNKNOWN] = (~is_cav).astype(np.float16)
            veh[base + t, :, C.PRESENT_IDX] = present.astype(np.float16)
            mask[base + t] = (present & kmask).astype(np.uint8)
            veh_kin[base + t, :, 0] = s.astype(np.float16)
            for li in range(C.N_LANES):
                veh_kin[base + t, :, 1 + li] = (lane == li).astype(np.float16)
            veh_kin[base + t, :, 4] = v.astype(np.float16)
            veh_kin[base + t, :, 5] = a.astype(np.float16)
            veh_kin[base + t, :, 6] = length0.astype(np.float16)
            hdv_base[base + t, :, 0] = base_idm.numpy().astype(np.float16)
            cav_plan[base + t, is_cav, 0] = (v[is_cav] / 40.0).astype(
                np.float16)
            cav_plan[base + t, is_cav, 1] = (a[is_cav] / 3.0).astype(
                np.float16)
            for li in range(C.N_LANES):
                cav_plan[base + t, is_cav, 5 + li] = \
                    (lane[is_cav] == li).astype(np.float16)
            if t < H:
                accel[base + t] = roll_acc[t].numpy().astype(np.float16)
                target_mask[base + t] = roll_mask[t].numpy().astype(np.uint8)
            lane_next[base + t] = lane_t.astype(np.int8)
        counter[base] = w["counter"][b].numpy().astype(np.float16)
        arrivals[base] = w["arrivals"][b].numpy().astype(np.int16)
        plan[r, :, :, 0] = 1.0
        y = w["outcome_quantiles"][b].numpy()
        omask = w["outcome_mask"][b].numpy().astype(bool)
        first = len(rows)
        for j, tau in enumerate(C.TAUS):
            rows.append({
                "world": "J4", "source": ("base" if b == 0 else "native"),
                "chunk": "chunk4", "split": "TRAIN", "t0_s": 100.0,
                "t": 100.0, "tau": float(tau),
                "total_delay_ref_s": float(y[j, 0]),
                "ttc_lt3_veh_s": float(y[j, 1]),
                "hard_brake_s": float(y[j, 2]),
                "collision_any": 0.0, "tau_available": bool(omask[j]),
            })
        items.append({
            "world": "J4", "arm": ("base" if b == 0 else "native"),
            "chunk": "chunk4", "split": "TRAIN", "t0_s": 100.0,
            "pair_key": f"J4w{wi}@t0=100",
            "policy": ("damping" if b == 0 else "native"),
            "tick": int(base), "plan_row": int(r),
            "outcome_rows": list(range(first, first + len(C.TAUS))),
        })
        ranges.append(dict(tick_start=int(base),
                           tick_end=int(base + per_run)))

    import json as _json

    arrays = {
        "veh_step": veh, "veh_mask": mask, "plan": plan, "wall": wall,
        "wall_mask": wall_mask, "counter": counter, "accel_next": accel,
        "target_mask": target_mask, "lane_next": lane_next,
        "arrivals": arrivals, "veh_kin": veh_kin, "hdv_base": hdv_base,
        "cav_plan_ch": cav_plan, "event_tok": event_tok,
        "event_mask": event_mask,
    }
    for name, arr in arrays.items():
        np.save(root / f"{name}.npy", arr)
    pd.DataFrame(rows).to_parquet(root / "outcomes.parquet", index=False)
    (root / "INDEX.json").write_text(_json.dumps(
        {"schema": "wm_n1_corpus_pack/2", "k_max": int(K), "runs": ranges},
        indent=1), encoding="utf-8")
    manifest = {
        "schema": "wm_n1_pack/1", "k_max": int(K),
        "arrays": {name: {"file": f"{name}.npy", "dtype": str(arr.dtype),
                          "shape": list(arr.shape)}
                   for name, arr in arrays.items()},
        "outcomes": {"file": "outcomes.parquet"},
        "items": items,
    }
    (root / "manifest.json").write_text(
        _json.dumps(manifest, indent=1), encoding="utf-8")
    return root


def corpus_overfit_probe(pack_dir, config: IdeaATrainConfig, *,
                         bundle: Optional[IdeaABundle] = None,
                         max_steps: int = 400, lr: Optional[float] = None,
                         threshold: float = 0.05) -> dict:
    """G4 overfit on the REAL corpus path (``IdeaACorpus`` canonical batch)."""
    cfg = config
    if lr is not None:
        cfg = IdeaATrainConfig.from_dict({**cfg.to_dict(), "lr": lr})
    corpus = IdeaACorpus(pack_dir, split="TRAIN", pair_sampling=True,
                         horizon_steps=max(1, int(cfg.rollout_horizon_steps)),
                         k_max=cfg.k_max)
    windows = [corpus[i] for i in range(len(corpus))]
    trainer = IdeaATrainer(cfg, corpus=None, bundle=bundle)
    optimizer, scheduler = _make_optimizer(trainer, cfg.lr, max_steps)
    trainer._optimizer = optimizer
    trainer._scheduler = scheduler
    history: list = []
    terms_seen: set = set()
    final = None
    final_terms: dict = {}
    for _ in range(int(max_steps)):
        batch = collate_canonical(windows)
        optimizer.zero_grad(set_to_none=True)
        loss = trainer.total_loss(batch)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(trainer.parameters(),
                                           float(cfg.grad_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        value = float(loss.detach())
        history.append(value)
        final = value
        final_terms = {k: float(v.detach()) for k, v in
                       trainer.loss_terms(batch).items()}
        terms_seen.update(final_terms)
    result = {
        "passed": final is not None and final < float(threshold),
        "final_total": final,
        "threshold": float(threshold),
        "steps": int(max_steps),
        "terms": final_terms,
        "terms_seen": sorted(terms_seen),
        "history": history,
    }
    for required in ("response_contrast", "rollout"):
        if required not in terms_seen:
            result["passed"] = False
            result.setdefault("warnings", []).append(
                f"gate requires the {required} term; batch lacks it")
    return result


def overfit_probe(windows: Sequence[dict], config: Optional[IdeaATrainConfig]
                  = None, *, bundle: Optional[IdeaABundle] = None,
                  max_steps: int = 400, lr: Optional[float] = None,
                  threshold: float = 0.05) -> dict:
    """G4 overfit gate: drive the TOTAL A3.2 loss toward zero on tiny windows.

    Trains one member (all supplied windows every step) on CPU and reports
    whether the final total loss is below ``threshold``.  The receipt lists
    the terms actually included (the gate requires the contrast and rollout
    terms to be present).
    """
    cfg = config or IdeaATrainConfig()
    cfg = (IdeaATrainConfig.from_dict(
        {**cfg.to_dict(), "lr": lr if lr is not None else cfg.lr})
        if lr is not None else cfg)
    trainer = IdeaATrainer(cfg, corpus=None, bundle=bundle)
    windows = list(windows)
    optimizer, scheduler = _make_optimizer(trainer, cfg.lr, max_steps)
    trainer._optimizer = optimizer
    trainer._scheduler = scheduler
    history: list = []
    terms_seen: set = set()
    final = None
    final_terms: dict = {}
    for _ in range(int(max_steps)):
        batch = collate_canonical(windows)
        optimizer.zero_grad(set_to_none=True)
        loss = trainer.total_loss(batch)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(trainer.parameters(),
                                           float(cfg.grad_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        value = float(loss.detach())
        history.append(value)
        final = value
        final_terms = {k: float(v.detach()) for k, v in
                       trainer.loss_terms(batch).items()}
        terms_seen.update(final_terms)
    result = {
        "passed": final is not None and final < float(threshold),
        "final_total": final,
        "threshold": float(threshold),
        "steps": int(max_steps),
        "terms": final_terms,
        "terms_seen": sorted(terms_seen),
        "history": history,
    }
    for required in ("response_contrast", "rollout"):
        if required not in terms_seen:
            result["passed"] = False
            result.setdefault("warnings", []).append(
                f"gate requires the {required} term; batch lacks it")
    return result


def overfit_report(*, n_windows: int = 2, max_steps: int = 1500,
                   lr: float = 0.03, threshold: float = 0.05,
                   seed: int = 0, bundle: Optional[IdeaABundle] = None,
                   windows: Optional[Sequence[dict]] = None,
                   standins: bool = False,
                   corpus_pack: Optional[str] = None) -> dict:
    """G4 pre-flight entry point (owner ``PREFLIGHT_GATES_IDEA_A.md``).

    Drives one Idea A member's TOTAL A3.2 loss (residual NLL, rollout,
    arrival, correction CRPS and the paired response contrast) near zero on two
    tiny TRAIN windows on CPU, and returns the gate schema.

    The headline path builds the REAL P2/P4/P5 bundle
    (:func:`train.g4_real.real_bundle`) and - by default - runs the probe
    THROUGH the corpus batch builder (:class:`IdeaACorpus`): the crafted
    ``train.g4_standins.make_windows`` values are written as a tiny real-format
    pack (:func:`write_g4_tiny_pack`) and read back, so the gate exercises the
    same builder a GPU-box run uses.  Pass ``corpus_pack=PACK`` to probe an
    existing pack, or ``windows=...`` for the legacy direct-window probe.

    ``standins=True`` (or a bundle built by ``train.g4_standins``) selects the
    documented linear stand-in fallback, which only proves the P7 loss
    machinery; the report is then FORCED to ``ok=False`` with a warning.
    """
    if bundle is None:
        if standins:
            from train.g4_standins import seeded_bundle

            bundle = seeded_bundle(101)
        else:
            from train.g4_real import real_bundle

            bundle = real_bundle(seed=101)
    used_standins = bool(standins)
    if not used_standins:
        from train.g4_real import is_standin_bundle

        used_standins = is_standin_bundle(bundle)

    import tempfile

    from train.g4_standins import make_windows

    work = None
    use_corpus = not used_standins and (corpus_pack is not None
                                        or windows is None)
    if use_corpus:
        if corpus_pack is None:
            crafted = make_windows(n_windows, seed=seed)
            work = tempfile.mkdtemp(prefix="wmn1_g4_")
            corpus_pack = str(write_g4_tiny_pack(crafted, Path(work) / "pack"))
            n_windows = len(crafted)
        horizon = int(windows[0]["teacher"].shape[0]) - 1 if windows else 3
        cfg = IdeaATrainConfig(out_dir="unused", grad_clip=0.0,
                               rollout_horizon_steps=horizon)
        probe = corpus_overfit_probe(corpus_pack, cfg, bundle=bundle,
                                     max_steps=max_steps, lr=lr,
                                     threshold=threshold)
    else:
        if windows is None:
            windows = make_windows(n_windows, seed=seed)
        windows = list(windows)
        probe = overfit_probe(windows, IdeaATrainConfig(out_dir="unused",
                                                        grad_clip=0.0),
                              bundle=bundle, max_steps=max_steps, lr=lr,
                              threshold=threshold)
    if work is not None:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    ok = bool(probe["passed"]) and not used_standins
    result = {
        "ok": ok,
        "device": "cpu",
        "n_windows": len(windows) if windows is not None else int(n_windows),
        "threshold": float(threshold),
        "final_loss": float(probe["final_total"]),
        "loss_curve": [float(x) for x in probe["history"]],
        "terms": {k: float(v) for k, v in probe["terms"].items()},
        "terms_seen": list(probe["terms_seen"]),
        "standins": used_standins,
        "real_modules": not used_standins,
        "corpus_path": bool(use_corpus),
    }
    if used_standins:
        result["warnings"] = [
            "G4 ran with the documented linear STAND-INS "
            "(train.g4_standins); the owner gate requires the REAL "
            "HdvResidualNet / IdeaAArrivalHead / CorrectionHead with gradients "
            "through the real contrast and rollout losses, so this report "
            "FAILS"]
    return result


# --------------------------------------------------------------- box CLI -----
def main(argv=None) -> int:
    """Idea A member training CLI (the GPU-box ``train`` argv).

    ``python -m train.idea_a --member k --seed S --corpus PACK --out RUN``
    trains one Idea A member with :class:`IdeaATrainer` (NOT the v1
    ``train.trainer.Trainer``).  Member-1-first is enforced by
    :meth:`IdeaATrainer.fit`.  Writes ``<RUN>/member_<k>.pt`` + its ``.json``
    sidecar + ``<RUN>/TRAIN_RECEIPT.json``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "dry_run", False):
        print(json.dumps({"status": "dry-run", "member": int(args.member),
                          "seed": args.seed, "corpus": str(args.corpus),
                          "out": str(args.out), "device": args.device,
                          "ensemble_k": int(args.ensemble_k),
                          "k_max": args.k_max,
                          "steps_per_epoch": args.steps_per_epoch,
                          "quick_tune": bool(args.quick_tune),
                          "subset_frac": float(args.subset_frac),
                          "epoch_frac": float(args.epoch_frac),
                          "pairs_per_batch": getattr(args,
                                                     "pairs_per_batch", None),
                          "lambda_contrast": getattr(args,
                                                     "lambda_contrast", None),
                          "contrast_outcome_weight": getattr(
                              args, "contrast_outcome_weight", None),
                          "require_outcome_contrast": bool(getattr(
                              args, "require_outcome_contrast", False)),
                          "paired_effect_head": bool(getattr(
                              args, "paired_effect_head", False)),
                          "paired_effect_weight": float(getattr(
                              args, "paired_effect_weight", 1.0)),
                          "paired_effect_climatology": getattr(
                              args, "paired_effect_climatology", None),
                          "paired_effect_climatology_sha256": getattr(
                              args, "paired_effect_climatology_sha256",
                              None),
                          "paired_effect_plan_aware": bool(getattr(
                              args, "paired_effect_plan_aware", False)),
                          "paired_effect_plan_horizon_s": getattr(
                              args, "paired_effect_plan_horizon_s", None),
                          "paired_effect_rank_weight": float(getattr(
                              args, "paired_effect_rank_weight", 0.0)),
                          "paired_effect_group_batching": bool(getattr(
                              args, "paired_effect_group_batching", False))},
                         sort_keys=True))
        return 0
    kwargs = dict(member=int(args.member), out_dir=str(args.out),
                  ensemble_k=int(args.ensemble_k),
                  quick_tune=bool(args.quick_tune),
                  subset_frac=float(args.subset_frac),
                  epoch_frac=float(args.epoch_frac),
                  allow_incomplete_losses=bool(args.allow_incomplete_losses),
                   state_global_token=bool(args.state_global_token),
                   num_workers=args.num_workers,
                   log_every=int(getattr(args, "log_every", 10)),
                   k_max=args.k_max,
                   pairs_per_batch=getattr(args, "pairs_per_batch", None),
                   merge_coverage_into_pairs=bool(getattr(
                       args, "merge_coverage_into_pairs", False)),
                   paired_effect_head=bool(getattr(args,
                                                   "paired_effect_head",
                                                   False)),
                   paired_effect_weight=float(getattr(
                       args, "paired_effect_weight", 1.0)),
                   paired_effect_climatology=getattr(
                       args, "paired_effect_climatology", None),
                   paired_effect_climatology_sha256=getattr(
                       args, "paired_effect_climatology_sha256", None),
                   paired_effect_plan_aware=bool(getattr(
                       args, "paired_effect_plan_aware", False)),
                   paired_effect_plan_horizon_s=getattr(
                       args, "paired_effect_plan_horizon_s", None),
                   paired_effect_rank_weight=float(getattr(
                       args, "paired_effect_rank_weight", 0.0)),
                   paired_effect_group_batching=bool(getattr(
                       args, "paired_effect_group_batching", False)))
    if getattr(args, "lambda_contrast", None) is not None:
        kwargs["lambda_response_contrast"] = float(args.lambda_contrast)
    if getattr(args, "contrast_outcome_weight", None) is not None:
        kwargs["contrast_outcome_weight"] = float(args.contrast_outcome_weight)
    kwargs["contrast_require_outcome_support"] = bool(
        getattr(args, "require_outcome_contrast", False))
    if args.epochs is not None:
        kwargs["epochs"] = int(args.epochs)
    if args.steps_per_epoch is not None:
        kwargs["steps_per_epoch"] = int(args.steps_per_epoch)
    cfg = IdeaATrainConfig(**kwargs)
    seed = (int(args.seed) if args.seed is not None
            else cfg.seed_for_member(int(args.member)))
    extra_packs = list(getattr(args, "extra_corpus", None) or [])
    extra_identities = list(getattr(args, "extra_identities", None) or [])
    extra_overlays = list(getattr(args, "extra_overlay", None) or [])
    corpus_kwargs = dict(
        split="TRAIN", pair_sampling=True,
        horizon_steps=cfg.rollout_horizon_steps, k_max=cfg.k_max,
        extra_pack_dirs=extra_packs or None,
        extra_overlay_dir=extra_overlays or None,
        extra_identity_dirs=extra_identities or None)
    if cfg.paired_effect_plan_horizon_s is not None:
        # A11 full-horizon plan: only the long-plan build widens the window;
        # the default corpus call signature stays unchanged (A10 form).
        corpus_kwargs["plan_horizon_s"] = float(
            cfg.paired_effect_plan_horizon_s)
    corpus = IdeaACorpus(str(args.corpus), **corpus_kwargs)
    torch.manual_seed(int(seed))
    trainer = IdeaATrainer(cfg, corpus, member=int(args.member), seed=seed,
                           device=args.device,
                           bundle=build_idea_a_bundle(cfg))
    receipt = trainer.fit()
    print(json.dumps({"status": "ok", "member": receipt["member"],
                      "seed": receipt["seed"], "steps": receipt["steps"],
                      "checkpoint": receipt["checkpoint"]},
                     sort_keys=True))
    return 0


def build_parser():
    """The ``train.idea_a`` box argv parser (also used by ``--dry-run``)."""
    import argparse

    parser = argparse.ArgumentParser(prog="train.idea_a",
                                     description=__doc__)
    parser.add_argument("--member", type=int, required=True,
                        choices=(1, 2, 3, 4, 5))
    parser.add_argument("--seed", type=int, default=None,
                        help="default: the registered member seed (A1)")
    parser.add_argument("--corpus", required=True, help="packed memmap dir")
    parser.add_argument("--extra-corpus", dest="extra_corpus", action="append",
                        default=None, metavar="PATH",
                        help="A9: repeatable additional packed memmap dir "
                             "trained alongside --corpus; items are merged at "
                             "the item level (pair_key grouping spans packs), "
                             "so trajectory items pair with their pack_v3 "
                             "native/pi siblings. Default OFF (unchanged).")
    parser.add_argument("--extra-overlay", dest="extra_overlay",
                        action="append", default=None, metavar="PATH",
                        help="A9: repeatable schema/3-style outcome overlay "
                             "sidecar, one per --extra-corpus pack (positional, "
                             "i-th overlay belongs to the i-th extra corpus); "
                             "supplies the zero-row extra items' labels by item "
                             "identity. Default OFF (unchanged).")
    parser.add_argument("--extra-identities", dest="extra_identities",
                        action="append", default=None, metavar="PATH",
                        help="A9: repeatable verified identity sidecar dir, one "
                             "per --extra-corpus pack (positional). Required "
                             "for a real fit that has extra packs; env fallback "
                             "WM_N1_EXTRA_TRAIN_IDENTITIES (os.pathsep). "
                             "Default OFF (unchanged).")
    parser.add_argument("--out", required=True, help="member output dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", dest="num_workers", type=int,
                        default=None,
                        help="DataLoader workers; default min(8, cpu-2); "
                             "0 keeps the in-process path")
    parser.add_argument("--no-state-global-token", dest="state_global_token",
                        action="store_false", default=True,
                        help="reproduce the pre-M1FIX2 zero global token")
    parser.add_argument("--ensemble-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="override TrainConfig.steps_per_epoch (a bounded "
                             "smoke/test run; the registered value is 2000)")
    parser.add_argument("--k-max", dest="k_max", type=int, default=None,
                        choices=(C.K_MAX_V2, C.K_MAX_V3),
                        help="FORMAT slot width (amendment 5): v2=128, "
                             "v3=192; default takes the pack's K_max and "
                             "refuses a mismatching pack")
    parser.add_argument("--quick-tune", action="store_true",
                        help="CAL-only improvement loop: train member 1 on a "
                             "fixed, world-stratified TRAIN subset and a "
                             "fraction of the epochs (same seed)")
    parser.add_argument("--subset-frac", type=float, default=1.0,
                        help="fraction of each world kept by quick-tune "
                             "(e.g. 0.25)")
    parser.add_argument("--epoch-frac", type=float, default=1.0,
                        help="fraction of the epochs run by quick-tune "
                             "(e.g. 0.35)")
    parser.add_argument("--allow-incomplete-losses", action="store_true",
                        help="TEST-ONLY: allow a corpus run to train with "
                             "unavailable A3.2 terms; the flag and the active "
                             "terms are recorded in TRAIN_RECEIPT.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse the argv and exit without touching data")
    parser.add_argument("--log-every", dest="log_every", type=int, default=10,
                        help="print the <out>/train_log.jsonl record every N "
                             "steps (1 = every step; 0 disables the print)")
    parser.add_argument("--merge-coverage-into-pairs",
                        dest="merge_coverage_into_pairs", action="store_true",
                        help="A8 batch fix: fold coverage items into the "
                             "pair-dense batches so every batch keeps K pairs")
    parser.add_argument("--pairs-per-batch", dest="pairs_per_batch", type=int,
                        default=None,
                        help="pair-dense batching (T2): each batch drawn from "
                             "the eligible sibling-pair pool holds exactly K "
                             "complete pairs; default None keeps the "
                             "historical one-pair-per-batch path (requires "
                             "2K <= batch_size)")
    parser.add_argument("--lambda-contrast", dest="lambda_contrast",
                        type=float, default=None,
                        help="override lambda_response_contrast, the paired "
                             "response CONTRAST loss weight (default 1.0)")
    parser.add_argument("--contrast-outcome-weight",
                        dest="contrast_outcome_weight", type=float,
                        default=None,
                        help="combined A8: give the OUTCOME sub-term of the "
                             "paired response contrast its own weight w "
                             "(w*outcome + 1.0*hdv) instead of halving it by "
                             "averaging with the HDV sub-term; default None "
                             "keeps the historical equal average")
    parser.add_argument("--require-outcome-contrast",
                        dest="require_outcome_contrast", action="store_true",
                        default=False,
                        help="combined A8: restrict the eligible sibling-pair "
                             "pool to pairs sharing a realised outcome tau "
                             "(T6), so every pair-dense batch forces an "
                             "outcome-carrying pair; default OFF keeps the "
                             "historical outcome-OR-HDV eligibility")
    parser.add_argument("--paired-effect-head", dest="paired_effect_head",
                        action="store_true", default=False,
                        help="enable the OPTIONAL climatology-anchored "
                             "paired-effect head (default OFF): per-tau dV "
                             "quantiles = TRAIN climatology + zero-init-gated "
                             "residual, trained by pinball on "
                             "outcome-supported contrast pairs")
    parser.add_argument("--paired-effect-weight", dest="paired_effect_weight",
                        type=float, default=1.0,
                        help="weight of the paired-effect pinball term "
                             "(default 1.0; recorded in TRAIN_RECEIPT.json)")
    parser.add_argument("--paired-effect-climatology",
                        dest="paired_effect_climatology", default=None,
                        help="TRAIN-derived climatology JSON the head anchors "
                             "on (required with --paired-effect-head)")
    parser.add_argument("--paired-effect-climatology-sha256",
                        dest="paired_effect_climatology_sha256", default=None,
                        help="optional SHA-256 pin of the climatology JSON")
    parser.add_argument("--paired-effect-plan-aware",
                        dest="paired_effect_plan_aware", action="store_true",
                        default=False,
                        help="A9 (default OFF): make the paired-effect head "
                             "plan-aware -- its residual trunk also reads the "
                             "TREATMENT arm's CAV-motion summary (summary_raw) "
                             "through a small learned plan embedding. Requires "
                             "--paired-effect-head. The zero-init gate keeps an "
                             "untrained head exactly at the climatology.")
    parser.add_argument("--paired-effect-plan-horizon-s",
                        dest="paired_effect_plan_horizon_s", type=float,
                        default=None,
                        help="A11 (default None = the historical 10 s "
                             "summary_raw plan): full-horizon plan INPUT -- "
                             "summarize the treatment arm's planned CAV motion "
                             "over [t0, t0+H] seconds in 10 s bins "
                             "(5 features/bin; H=120 -> 60 features) and feed "
                             "it to the plan-aware head. Requires "
                             "--paired-effect-head and "
                             "--paired-effect-plan-aware.")
    parser.add_argument("--paired-effect-rank-weight",
                        dest="paired_effect_rank_weight", type=float,
                        default=0.0,
                        help="A11 (default 0 = OFF): within-decision pairwise "
                             "ranking weight -- per decision group with >=2 "
                             "outcome-supported treatment arms, a logistic loss "
                             "orders the head's mean dV by realised delay, per "
                             "tau and scaled by s_tau (ties ignored).")
    parser.add_argument("--paired-effect-group-batching",
                        dest="paired_effect_group_batching",
                        action="store_true", default=False,
                        help="A11 (default OFF): group the pair-dense draws by "
                             "pair_key so multiple treatment arms of one "
                             "decision co-occur in a batch (needed for the "
                             "ranking loss on real data); requires "
                             "--pairs-per-batch.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
