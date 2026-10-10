"""Frozen data types and constants for the predict() contract v1 / v1.1 / v2.

Specs:
- output/wd01/PREDICT_CONTRACT_v1.md (frozen 2026-09-25; contract_version
  "wd01_predict/1"; §8 clarifications + §9 v1.1, contract_version
  "wd01_predict/1.1" — v1.1 stays frozen);
- output/wd01/PREDICT_CONTRACT_v2.md (contract_version "wd01_predict/2", a
  DELTA on v1: everything not changed there stays as in v1.1).

This module is a typed rendering of those texts. It is pure stdlib + numpy and
imports without torch. All v2-only fields default to None on v1/v1.1 outputs;
validate.py enforces presence (never defaulted True) exactly on v2 outputs.

Nothing in here interprets the contract semantically; see validate.py for the
§3.3/§4 rules, native_ref.py for §5, safety.py for the consumer rules,
conformance.py for the kit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np

CONTRACT_VERSION = "wd01_predict/1"
CONTRACT_VERSION_V1_1 = "wd01_predict/1.1"
CONTRACT_VERSION_V2 = "wd01_predict/2"
# §4 (v2.1, frozen 2026-09-29): amendment on top of v2 for Part A v2
# general (M = 1, tracked scope). Optional fields only; predictors that
# do not emit them remain valid.
PART_A_SCOPE_TRACKED = "tracked"
PART_A_SCOPE_ALL_VEHICLES = "all_vehicles"
PART_A_SCOPES = (PART_A_SCOPE_TRACKED, PART_A_SCOPE_ALL_VEHICLES)
# §9 (v1.1): implementations accept the supported versions and report the one
# they implement in meta.contract_version. CONTRACT_VERSION stays the v1
# default; v2 joins the supported set (PREDICT_CONTRACT_v2.md preamble).
SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION, CONTRACT_VERSION_V1_1, CONTRACT_VERSION_V2)
PLAN_VERSION = 1

# --- actions (§3.2 controller state; §3.3 per_cav) -------------------------
ACTION_NATIVE = "native"
ACTION_DAMPING = "damping"
ACTION_PI_EXTERNAL_U = "pi-external-u"
ACTION_WALL = "wall"
ACTIONS = (ACTION_NATIVE, ACTION_DAMPING, ACTION_PI_EXTERNAL_U, ACTION_WALL)
# per_cav may only carry the non-wall actions; wall members never appear there.
PER_CAV_ACTIONS = (ACTION_NATIVE, ACTION_DAMPING, ACTION_PI_EXTERNAL_U)

# --- wall phases (§3.3 WallSpec.phase; §3.2 wall_phase) --------------------
PHASE_ASSEMBLE = "assemble"
PHASE_HOLD = "hold"
PHASE_RELEASE = "release"
PHASES = (PHASE_ASSEMBLE, PHASE_HOLD, PHASE_RELEASE)

# --- wall abort codes (§4.4) -----------------------------------------------
ABORT_NONE = 0
ABORT_SAFETY = 1
ABORT_ASSEMBLY_TIMEOUT = 2
ABORT_REASONS = (ABORT_NONE, ABORT_SAFETY, ABORT_ASSEMBLY_TIMEOUT)

# --- WM-30 physical lanes (§4.3) -------------------------------------------
LANE_MAIN_LEFT = 0
LANE_MAIN_RIGHT = 1
LANE_ACCEL = 2
LANE_IDS = (LANE_MAIN_LEFT, LANE_MAIN_RIGHT, LANE_ACCEL)

# --- boundary entrances (§3.1 arrivals) ------------------------------------
# The contract names "the mainline entrance and the ramp" but does not fix the
# label strings; these are the canonical constants used by this implementation.
ENTRANCE_MAINLINE = "mainline"
ENTRANCE_RAMP = "ramp"

DEFAULT_HORIZON_S = 600.0

# §2 (v2): the commit window Δ and the post-Δ continuation modes. Under
# continuation="planner" the plan branch runs the planner's policy π after Δ,
# identified by meta.continuation_policy_id; under "native" all CAVs run
# native after Δ and the policy id is None.
DEFAULT_COMMIT_S = 30.0
CONTINUATION_PLANNER = "planner"
CONTINUATION_NATIVE = "native"
CONTINUATIONS = (CONTINUATION_PLANNER, CONTINUATION_NATIVE)

TRAJ_COLS = 4  # (t_s, s_m, lane_id, speed_mps)


@dataclass(frozen=True)
class Arrival:
    """One explicit boundary-arrival row (§3.1).

    §8.14: `t_s` may precede t0 (a vehicle that was due but not yet inserted
    at t0; its entry waiting counts from t0). `arrival_id` is OPTIONAL; when
    present it is identical between plan and native in every implementation,
    including the perfect simulator."""

    t_s: float
    entrance: str
    speed_mps: float
    is_cav: bool
    vehicle_type: str
    arrival_id: Optional[str] = None


@dataclass(frozen=True)
class ArrivalDraw:
    """A `{demand_model_id, seed}` pair that deterministically expands to an
    explicit arrival list (§3.1). Expansion is implementation-specific."""

    demand_model_id: str
    seed: int


Arrivals = Union[Tuple[Arrival, ...], ArrivalDraw]


@dataclass(frozen=True)
class SampleMeta:
    """Per-sample meta (§3.1). `time_s` is t0; all samples of one call share it.
    `merge_observed` and `coverage` are planner-side ABSTAIN inputs; models
    ignore them."""

    sample_id: str
    time_s: float
    merge_observed: bool
    coverage: float


    @classmethod
    def from_raw(cls, d: Mapping[str, Any]) -> "SampleMeta":
        _keys(d, {"sample_id", "time_s", "merge_observed", "coverage"})
        d = dict(d)
        # §8.15: hybrid_state/2 may carry merge_observed=None (no RSU inputs);
        # the StateSample builder maps None to False — fail closed, counts
        # toward ABSTAIN. Hand-constructed SampleMeta entries keep strict
        # typing: the validator still rejects a non-bool merge_observed.
        if d.get("merge_observed") is None:
            d["merge_observed"] = False
        return cls(**d)


@dataclass(frozen=True)
class StateSample:
    """One sample k (§3.1). `hybrid` is a `wm30_hybrid_state/1` reconstructor
    state (opaque here: observed-vehicle layer + hidden-road field + ONE
    sampled hidden-vehicle set)."""

    hybrid: Any
    arrivals: Arrivals
    meta: SampleMeta

    @classmethod
    def from_raw(cls, d: Mapping[str, Any]) -> "StateSample":
        _keys(d, {"hybrid", "arrivals", "meta"})
        meta = d["meta"]
        if isinstance(meta, Mapping):
            meta = SampleMeta.from_raw(meta)  # §8.15: maps merge_observed None -> False
        arrivals = d["arrivals"]
        if isinstance(arrivals, Mapping):
            _keys(arrivals, {"demand_model_id", "seed"})
            arrivals = ArrivalDraw(**arrivals)
        else:
            arrivals = tuple(
                a if isinstance(a, Arrival) else Arrival(**a) for a in arrivals
            )
        return cls(hybrid=d["hybrid"], arrivals=arrivals, meta=meta)


@dataclass(frozen=True)
class ControllerStateEntry:
    """Per-CAV controller state (§3.2). Ideal sensing: values are exact.

    §9 (v1.1): `lane_id` is OPTIONAL (None = unknown): 0 main_left,
    1 main_right, 2 accel. When present for every member of a wall, wall
    members must be ordered by ascending lane index, and `native_reference`
    orders members by lane_id."""

    action: str
    wall_id: Optional[str] = None
    wall_phase: Optional[str] = None
    time_in_phase_s: float = 0.0
    wall_speed_mps: Optional[float] = None
    locked: bool = False
    lane_id: Optional[int] = None

    @classmethod
    def from_raw(cls, d: Mapping[str, Any]) -> "ControllerStateEntry":
        _keys(
            d,
            {
                "action",
                "wall_id",
                "wall_phase",
                "time_in_phase_s",
                "wall_speed_mps",
                "locked",
                "lane_id",
            },
        )
        return cls(**d)


ControllerState = Mapping[str, ControllerStateEntry]


@dataclass(frozen=True)
class WallSpec:
    """One wall in a plan (§3.3). Exactly one of `release_at_s` / `hold_s` is
    set. `hold_s = math.inf` is reserved for the native reference (§5): a wall
    that stays in assemble until a safety abort ends it. Members are ordered
    by lane index, one per lane of the wall cross-section."""

    wall_id: str
    members: Tuple[str, ...]
    speed_mps: float
    phase: str
    release_at_s: Optional[float] = None
    hold_s: Optional[float] = None

    @classmethod
    def from_raw(cls, d: Mapping[str, Any]) -> "WallSpec":
        _keys(
            d,
            {
                "wall_id",
                "members",
                "speed_mps",
                "phase",
                "release_at_s",
                "hold_s",
            },
        )
        d = dict(d)
        d["members"] = tuple(d["members"])
        return cls(**d)


@dataclass(frozen=True)
class PlanCandidate:
    """One candidate plan (§3.3). v1 plans are constant from t0; a `schedule`
    field is reserved for a later version."""

    plan_version: int
    per_cav: Mapping[str, str]
    walls: Tuple[WallSpec, ...]

    @classmethod
    def from_raw(cls, d: Mapping[str, Any]) -> "PlanCandidate":
        _keys(d, {"plan_version", "per_cav", "walls"})
        walls = tuple(
            w if isinstance(w, WallSpec) else WallSpec.from_raw(w) for w in d["walls"]
        )
        return cls(plan_version=d["plan_version"], per_cav=dict(d["per_cav"]), walls=walls)


@dataclass(frozen=True)
class WallOutcome:
    """Per-wall outcome arrays (§4.4), each [M, K].

    `assembled_at_s` is NaN exactly where `assembled` is False.
    `abort_reason`: 0 none, 1 safety, 2 assembly_timeout; non-zero exactly
    where `safety_abort` is True."""

    assembled: np.ndarray
    assembled_at_s: np.ndarray
    safety_abort: np.ndarray
    abort_reason: np.ndarray
    # §3 (v2) future-wall heads [M, K] — REQUIRED (all three, together) on v2
    # outputs, None on v1/v1.1 outputs. `expected_assembly_time_s` is NaN
    # exactly where `p_assembly_completes` == 0; the two probabilities are in
    # [0, 1]. There are no native_ versions (walls belong to the plan branch).
    p_assembly_completes: Optional[np.ndarray] = None
    expected_assembly_time_s: Optional[np.ndarray] = None
    p_assembly_abort: Optional[np.ndarray] = None


@dataclass(frozen=True)
class HiddenTrajectorySet:
    """Optional per-k hidden-vehicle trajectories (§4.3): one ragged
    [n_k, T, 4] array plus its n_k ids (ids are not stable across samples)."""

    traj: np.ndarray
    ids: Tuple[str, ...]


@dataclass(frozen=True)
class Meta:
    """Prediction meta (§4.6). `weights_sha256` has one entry per ensemble
    member M ("simulator:<sumo version>+<network sha>" for the perfect
    simulator). `safety_window_s == part_a_horizon_s` on v1/v1.1 outputs;
    on v2 outputs `safety_window_s == commit_s` (§3: the v1 window fields
    cover the commit window Δ).

    §4 (v2) additions — all None/False on v1/v1.1 outputs, REQUIRED (and
    checked) on v2 outputs:
    - `commit_s`, `continuation` ("planner"|"native");
    - `continuation_policy_id`: str, None exactly when continuation=="native";
    - `value_model_id`, `value_model_sha256` (one per member);
    - `future_min_ttc_validated`: bool; `future_min_ttc_validation_id`: str
      non-None exactly when the flag is True (it points at the ledger/report
      record that validated the `future_min_ttc_q05_s` head)."""

    contract_version: str
    model_id: str
    weights_sha256: Tuple[str, ...]
    sample_ids: Tuple[str, ...]
    member_ids: Tuple[str, ...]
    part_a_horizon_s: float
    safety_window_s: float
    horizon_s: float
    native_kept_walls: Tuple[str, ...]
    native_unlocked_available: bool
    device: str
    wall_clock_s: float
    commit_s: Optional[float] = None
    continuation: Optional[str] = None
    continuation_policy_id: Optional[str] = None
    value_model_id: Optional[str] = None
    value_model_sha256: Optional[Tuple[str, ...]] = None
    future_min_ttc_validated: bool = False
    future_min_ttc_validation_id: Optional[str] = None
    # §4 (v2.1) optional meta additions (Part A v2 general): None on
    # every predictor that does not emit them.
    part_a_model: Optional[str] = None
    param_set_id: Optional[str] = None
    recon_schema_version: Optional[str] = None
    part_a_scope: Optional[str] = None

    @classmethod
    def from_raw(cls, d: Mapping[str, Any]) -> "Meta":
        _keys(
            d,
            {
                "contract_version",
                "model_id",
                "weights_sha256",
                "sample_ids",
                "member_ids",
                "part_a_horizon_s",
                "safety_window_s",
                "horizon_s",
                "native_kept_walls",
                "native_unlocked_available",
                "device",
                "wall_clock_s",
                "commit_s",
                "continuation",
                "continuation_policy_id",
                "value_model_id",
                "value_model_sha256",
                "future_min_ttc_validated",
                "future_min_ttc_validation_id",
                "part_a_model",
                "param_set_id",
                "recon_schema_version",
                "part_a_scope",
            },
            optional={
                "commit_s",
                "continuation",
                "continuation_policy_id",
                "value_model_id",
                "value_model_sha256",
                "future_min_ttc_validated",
                "future_min_ttc_validation_id",
                "part_a_model",
                "param_set_id",
                "recon_schema_version",
                "part_a_scope",
            },
        )
        d = dict(d)
        for k in ("weights_sha256", "sample_ids", "member_ids", "native_kept_walls"):
            d[k] = tuple(d[k])
        if d.get("value_model_sha256") is not None:
            d["value_model_sha256"] = tuple(d["value_model_sha256"])
        return cls(**d)


@dataclass(frozen=True)
class Prediction:
    """Output of one plan for one call (§4). Every array field is a numpy
    array; every [M, K] array shares its axes with native and with every other
    Prediction of the call (paired comparison).

    `trajectories_tracked` / `native_trajectories_tracked` map vehicle_id to
    float [M, K, T, 4] columns (t_s, s_m, lane_id, speed_mps), T =
    trajectory_s + 1; a vehicle that left the section is NaN (whole row) from
    the exit step on.

    The `native_unlocked_*` diagnostics (§4.5) are optional; they are present
    (all five) exactly when `meta.native_unlocked_available` is True."""

    # delay, vehicle-seconds (§4.1)
    delay_vs_native_s: np.ndarray
    plan_delay_s: np.ndarray
    native_delay_s: np.ndarray
    delay_part_a_s: np.ndarray
    delay_part_b_s: np.ndarray
    # safety over [t0, t0 + safety_window_s] (§4.2)
    collision: np.ndarray
    min_ttc_s: np.ndarray
    ttc_lt3_exposure_s: np.ndarray
    hard_brake_s: np.ndarray
    native_collision: np.ndarray
    native_min_ttc_s: np.ndarray
    native_ttc_lt3_exposure_s: np.ndarray
    native_hard_brake_s: np.ndarray
    # trajectories over the Part A window (§4.3)
    trajectories_tracked: Mapping[str, np.ndarray]
    native_trajectories_tracked: Mapping[str, np.ndarray]
    # per-wall outcomes (§4.4)
    walls: Mapping[str, WallOutcome]
    meta: Meta
    # optional (§4.3 / §4.5)
    trajectories_hidden: Optional[Tuple[HiddenTrajectorySet, ...]] = None
    native_unlocked_delay_s: Optional[np.ndarray] = None
    native_unlocked_min_ttc_s: Optional[np.ndarray] = None
    native_unlocked_ttc_lt3_exposure_s: Optional[np.ndarray] = None
    native_unlocked_hard_brake_s: Optional[np.ndarray] = None
    native_unlocked_collision: Optional[np.ndarray] = None
    # §9 (v1.1) REQUIRED cell-validity outputs. None = absent, which a 1.0
    # output legitimately is; a 1.1 output missing them is a validation error
    # and is never defaulted to True (fail closed).
    cell_valid: Optional[np.ndarray] = None          # bool [M, K]
    native_cell_valid: Optional[np.ndarray] = None   # bool [M, K]
    has_closing_pair: Optional[np.ndarray] = None    # bool [M, K]
    native_has_closing_pair: Optional[np.ndarray] = None  # bool [M, K]
    # §3 (v2) future safety over (t0+commit_s, t0+horizon_s] under the
    # continuation, all [M, K] with native_ versions. None = absent on
    # v1/v1.1 outputs; REQUIRED on v2 outputs and never defaulted (fail
    # closed). `future_cell_valid` / `native_future_cell_valid` cover
    # delay_part_b_s and every future_* float; in a valid future cell every
    # future_* float is finite EXCEPT `future_min_ttc_q05_s`, which is NaN
    # exactly where `future_has_closing_pair` is False. `p_future_collision`
    # is a probability in [0, 1].
    future_cell_valid: Optional[np.ndarray] = None   # bool [M, K]
    native_future_cell_valid: Optional[np.ndarray] = None  # bool [M, K]
    future_has_closing_pair: Optional[np.ndarray] = None   # bool [M, K]
    native_future_has_closing_pair: Optional[np.ndarray] = None  # bool [M, K]
    future_ttc_lt3_exposure_s: Optional[np.ndarray] = None
    future_hard_brake_s: Optional[np.ndarray] = None
    p_future_collision: Optional[np.ndarray] = None
    future_min_ttc_q05_s: Optional[np.ndarray] = None
    native_future_ttc_lt3_exposure_s: Optional[np.ndarray] = None
    native_future_hard_brake_s: Optional[np.ndarray] = None
    native_p_future_collision: Optional[np.ndarray] = None
    native_future_min_ttc_q05_s: Optional[np.ndarray] = None
    # §4 (v2.1) optional per-cell tracked-scope fields (Part A v2
    # general). None = absent; every field is OPTIONAL and no veto / no
    # abstain may be derived from n_free_road_assumed.
    n_tracked: Optional[np.ndarray] = None
    n_exited_tracked: Optional[np.ndarray] = None
    n_free_road_assumed: Optional[np.ndarray] = None
    untracked_arrivals_expected: Optional[np.ndarray] = None


def _keys(d: Mapping[str, Any], allowed: set, optional: "set | None" = None) -> None:
    """Strict raw-dict conversion: unknown keys are typos, not defaults.
    Keys in `optional` may be absent (they carry dataclass defaults — the §4
    v2 meta additions on a v1/v1.1 raw dict); when present they are checked
    against `allowed` like every other key."""
    optional = optional or set()
    unknown = set(d) - allowed
    missing = allowed - set(d) - optional
    if unknown or missing:
        raise TypeError(
            f"raw dict keys mismatch: unknown={sorted(unknown)} missing={sorted(missing)}"
        )


__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_V1_1",
    "CONTRACT_VERSION_V2",
    "SUPPORTED_CONTRACT_VERSIONS",
    "PLAN_VERSION",
    "DEFAULT_COMMIT_S",
    "CONTINUATION_PLANNER",
    "CONTINUATION_NATIVE",
    "CONTINUATIONS",
    "ACTION_NATIVE",
    "ACTION_DAMPING",
    "ACTION_PI_EXTERNAL_U",
    "ACTION_WALL",
    "ACTIONS",
    "PER_CAV_ACTIONS",
    "PHASE_ASSEMBLE",
    "PHASE_HOLD",
    "PHASE_RELEASE",
    "PHASES",
    "ABORT_NONE",
    "ABORT_SAFETY",
    "ABORT_ASSEMBLY_TIMEOUT",
    "ABORT_REASONS",
    "LANE_MAIN_LEFT",
    "LANE_MAIN_RIGHT",
    "LANE_ACCEL",
    "LANE_IDS",
    "ENTRANCE_MAINLINE",
    "ENTRANCE_RAMP",
    "DEFAULT_HORIZON_S",
    "TRAJ_COLS",
    "PART_A_SCOPE_TRACKED",
    "PART_A_SCOPE_ALL_VEHICLES",
    "PART_A_SCOPES",
    "Arrival",
    "ArrivalDraw",
    "Arrivals",
    "SampleMeta",
    "StateSample",
    "ControllerStateEntry",
    "ControllerState",
    "WallSpec",
    "PlanCandidate",
    "WallOutcome",
    "HiddenTrajectorySet",
    "Meta",
    "Prediction",
]
