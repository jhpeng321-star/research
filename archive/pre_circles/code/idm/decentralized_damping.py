"""Pure contracts for decentralized structured IDM damping.

Every CAV runs the same local policy.  A non-zero action is a compact V2V bid
for one structured damping option; deterministic contention chooses a
conflict-free winner without a lane-level learned selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


ABSTAIN = 0
MILD = 1
NOMINAL = 2
STRONG = 3
CONSERVATIVE_REFERENCE = 4
LEGACY_ACTION_SET = "gain_reference_v1"
ACCELERATION_CAP_ACTION_SET = "acceleration_caps_v1"
ACTIVATION_ONLY_ACTION_SET = "activation_only_v1"
ACTIVATION_SPEED_CEILING_ACTION_SET = "activation_speed_ceiling_v1"
ACTIVATION_SPEED_CEILING_P50_ACTION_SET = "activation_speed_ceiling_p50_v1"
ACTIVATION_SPEED_CEILING_P70_ACTION_SET = "activation_speed_ceiling_p70_v1"
ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET = (
    "activation_speed_ceiling_p30_projected_release_v1"
)
ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET = (
    "activation_speed_ceiling_p30_projected_release_v2"
)
ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET = (
    "activation_speed_ceiling_p30_settle_release_v1"
)
ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET = (
    "activation_modified_followerstopper_p30_ceiling_v1"
)
ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET = "activation_idm_p30_projected_blend_v1"
ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET = "activation_rear_aware_feedback_pulse_v1"
ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET = "activation_rear_aware_feedback_pulse_v2"


@dataclass(frozen=True)
class DampingMode:
    index: int
    name: str
    family: str | None
    value: float | None


LEGACY_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(MILD, "mild", "gain", 0.75),
    DampingMode(NOMINAL, "nominal", "gain", 1.00),
    DampingMode(STRONG, "strong", "gain", 1.25),
    DampingMode(CONSERVATIVE_REFERENCE, "conservative_reference", "reference_offset", -1.0),
)
ACCELERATION_CAP_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(1, "nominal", "gain", 1.00),
    DampingMode(2, "gentle_acceleration_cap", "acceleration_cap", 0.25),
    DampingMode(3, "coast_cap", "acceleration_cap", 0.0),
)
ACTIVATION_ONLY_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(1, "nominal", "gain", 1.00),
)
ACTIVATION_SPEED_CEILING_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    # The disturbance p30 is a terminal speed ceiling, not a stronger
    # braking-rate request. Projection retains every hard safety bound.
    DampingMode(1, "speed_ceiling", "speed_ceiling", 0.0),
)
ACTIVATION_SPEED_CEILING_P50_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(1, "speed_ceiling_p50", "speed_ceiling_p50", 0.0),
)
ACTIVATION_SPEED_CEILING_P70_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(1, "speed_ceiling_p70", "speed_ceiling_p70", 0.0),
)
# The active command is deliberately identical to frozen P30.  This separately
# named action set changes only the post-option release lifecycle.
ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(1, "speed_ceiling", "speed_ceiling", 0.0),
)
ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(
        1,
        "modified_followerstopper_p30_ceiling",
        "modified_followerstopper_p30_ceiling",
        0.0,
    ),
)
ACTIVATION_IDM_P30_PROJECTED_BLEND_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(1, "blend_p25", "idm_p30_projected_blend", 0.25),
    DampingMode(2, "blend_p50", "idm_p30_projected_blend", 0.50),
    DampingMode(3, "blend_p75", "idm_p30_projected_blend", 0.75),
    DampingMode(4, "p30_full", "idm_p30_projected_blend", 1.00),
)
ACTIVATION_REAR_AWARE_FEEDBACK_MODES: tuple[DampingMode, ...] = (
    DampingMode(ABSTAIN, "abstain", None, None),
    DampingMode(1, "rear_aware_feedback_pulse", "rear_aware_feedback_pulse", 0.0),
)
ACTION_SETS = {
    LEGACY_ACTION_SET: LEGACY_MODES,
    ACCELERATION_CAP_ACTION_SET: ACCELERATION_CAP_MODES,
    ACTIVATION_ONLY_ACTION_SET: ACTIVATION_ONLY_MODES,
    ACTIVATION_SPEED_CEILING_ACTION_SET: ACTIVATION_SPEED_CEILING_MODES,
    ACTIVATION_SPEED_CEILING_P50_ACTION_SET: ACTIVATION_SPEED_CEILING_P50_MODES,
    ACTIVATION_SPEED_CEILING_P70_ACTION_SET: ACTIVATION_SPEED_CEILING_P70_MODES,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET: (
        ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_MODES
    ),
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET: (
        ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_MODES
    ),
    ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET: (
        ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_MODES
    ),
    ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET: (
        ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_MODES
    ),
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET: (ACTIVATION_IDM_P30_PROJECTED_BLEND_MODES),
    ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET: ACTIVATION_REAR_AWARE_FEEDBACK_MODES,
    ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET: ACTIVATION_REAR_AWARE_FEEDBACK_MODES,
}
TERMINAL_REFERENCE_PERCENTILE_BY_ACTION_SET = {
    ACTIVATION_SPEED_CEILING_ACTION_SET: "p30",
    ACTIVATION_SPEED_CEILING_P50_ACTION_SET: "p50",
    ACTIVATION_SPEED_CEILING_P70_ACTION_SET: "p70",
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET: "p30",
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET: "p30",
    ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET: "p30",
    ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET: "p30",
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET: "p30",
    ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET: "p50",
    ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET: "p50",
}
# Backward-compatible exports for completed v1 artifacts.
MODES = LEGACY_MODES
MODE_BY_INDEX = {mode.index: mode for mode in LEGACY_MODES}
MODE_COUNT = len(LEGACY_MODES)


def modes_for(action_set: str) -> tuple[DampingMode, ...]:
    try:
        return ACTION_SETS[action_set]
    except KeyError as exc:
        raise ValueError(f"unknown decentralized action set {action_set!r}") from exc


def mode_by_index(action_set: str) -> dict[int, DampingMode]:
    return {mode.index: mode for mode in modes_for(action_set)}


def mode_count(action_set: str) -> int:
    return len(modes_for(action_set))


def nominal_mode_index(action_set: str) -> int:
    active = tuple(mode for mode in modes_for(action_set) if mode.index != ABSTAIN)
    return next(
        (mode.index for mode in active if mode.name in {"nominal", "p30_full"}),
        active[0].index,
    )


def terminal_reference_percentile(action_set: str) -> str | None:
    """Return the explicit terminal-reference identity for ceiling actions."""
    return TERMINAL_REFERENCE_PERCENTILE_BY_ACTION_SET.get(action_set)


@dataclass(frozen=True)
class Candidate:
    vehicle_id: str
    lane_id: str
    distance_to_merge_m: float
    disturbance_distance_to_merge_m: float
    runway_m: float
    rank: int


@dataclass(frozen=True)
class Bid:
    candidate: Candidate
    mode: DampingMode


def decode_mode(action: object, action_set: str = LEGACY_ACTION_SET) -> DampingMode:
    """Decode a policy action safely; malformed actions abstain."""
    try:
        if hasattr(action, "reshape"):
            action = action.reshape(-1)[0]
        index = int(action)
    except (TypeError, ValueError, IndexError):
        index = ABSTAIN
    table = mode_by_index(action_set)
    return table.get(index, table[ABSTAIN])


def enumerate_candidates(
    samples: Iterable[Any],
    cav_ids: Iterable[str],
    disturbances: Mapping[str, Any],
    *,
    minimum_runway_m: float,
    maximum_runway_m: float,
) -> dict[str, tuple[Candidate, ...]]:
    """Return every geometrically feasible CAV, ranked within its lane.

    Safety admission is deliberately a separate step: callers must test every
    enumerated candidate before contention, avoiding the former
    select-nearest-then-give-up failure.
    """
    cav_set = set(cav_ids)
    by_lane: dict[str, list[Candidate]] = {}
    for sample in samples:
        disturbance = disturbances.get(sample.lane_id)
        if sample.vehicle_id not in cav_set or disturbance is None:
            continue
        runway = float(sample.distance_to_merge_m - disturbance.distance_to_merge_m)
        if runway < minimum_runway_m or runway > maximum_runway_m:
            continue
        by_lane.setdefault(sample.lane_id, []).append(
            Candidate(
                vehicle_id=str(sample.vehicle_id),
                lane_id=str(sample.lane_id),
                distance_to_merge_m=float(sample.distance_to_merge_m),
                disturbance_distance_to_merge_m=float(disturbance.distance_to_merge_m),
                runway_m=runway,
                rank=0,
            )
        )
    ranked: dict[str, tuple[Candidate, ...]] = {}
    for lane_id, candidates in sorted(by_lane.items()):
        ordered = sorted(candidates, key=lambda row: (row.runway_m, row.vehicle_id))
        ranked[lane_id] = tuple(
            Candidate(**{**row.__dict__, "rank": rank}) for rank, row in enumerate(ordered, start=1)
        )
    return ranked


def contend(
    bids: Iterable[Bid],
    *,
    reserved: Iterable[Candidate] = (),
    minimum_spacing_m: float,
) -> tuple[dict[str, Bid], dict[str, str]]:
    """Resolve local V2V bids using a deterministic distributed priority.

    The same result can be computed by every participant from the broadcast
    tuple ``(lane, runway, distance_to_merge, vehicle_id, mode)``.  Retained
    option members have priority and are supplied through ``reserved``.
    """
    occupied = list(reserved)
    winners: dict[str, Bid] = {}
    rejected: dict[str, str] = {}
    ordered = sorted(
        bids,
        key=lambda bid: (
            bid.candidate.runway_m,
            bid.candidate.vehicle_id,
            bid.mode.index,
        ),
    )
    for bid in ordered:
        candidate = bid.candidate
        if candidate.lane_id in winners:
            rejected[candidate.vehicle_id] = "lower_lane_priority"
            continue
        conflict = next(
            (
                other
                for other in occupied
                if abs(candidate.distance_to_merge_m - other.distance_to_merge_m)
                < minimum_spacing_m
            ),
            None,
        )
        if conflict is not None:
            rejected[candidate.vehicle_id] = f"spacing_conflict:{conflict.vehicle_id}"
            continue
        winners[candidate.lane_id] = bid
        occupied.append(candidate)
    return winners, rejected


def action_mask(
    *,
    eligible: bool,
    latched_mode: int | None = None,
    action_set: str = LEGACY_ACTION_SET,
) -> tuple[int, ...]:
    """Return the five-action local mask used by the decentralized actor."""
    table = mode_by_index(action_set)
    count = len(table)
    if latched_mode in table and latched_mode != ABSTAIN:
        return tuple(1 if index == latched_mode else 0 for index in range(count))
    if eligible:
        return tuple(1 for _ in range(count))
    return (1, *(0 for _ in range(count - 1)))
