"""Wall-v2 proposer, folding, and the planner wall registry (PL01-B).

Proposer: full cross-section member sets -- one free (unlocked) CAV per lane
of the observed lane set (lane set and longitudinal positions read from the
current plan's t0 tracked trajectories), lane-ordered, longitudinal spread
<= wall_max_member_gap_m, crossed with wall_speeds_mps (all > 2.0, the
dispatcher floor) and wall_release_options_s (every option is a hold_s:
numeric options release after that many seconds past assembly; the None
option is the OPEN HOLD, hold_s = wall_open_hold_s -- no automatic release,
the wall runs until the planner releases it by wall omission, mixed_assignment/3).

Folding: CAVs whose tracked trajectories deviate under the wall candidate vs
the current plan (same thresholds as loop step 2) join the wall's group; the
cap applies with members counted; the weakest-interaction non-members are
dropped first; members are never dropped (members alone above the cap make
the candidate infeasible).

Registry: the planner owns wall_id -> record (lane-ordered members, speed,
phase, hold_s/release_at_s, dispatch/assembled times), updated from its own
decisions, from controller_state (execution reality: phases, unlocks), and
from prediction wall events (assembled / safety_abort / abort_reason).

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import PlannerConfig
from .grouping import trajectory_deviation
from .interface import (
    PlanCandidate,
    Prediction,
    WallSpec,
)

ACTIVE_WALL_PHASES = ("assemble", "hold")  # phases with a wall on the road

#: Dispatcher telemetry ``wall_phase`` -> planner wall phase. The final WM-40
#: dispatcher reports ``assembly | hold | handback`` (mixed_assignment/3
#: telemetry); the planner vocabulary is ``assemble | hold | release``. Both
#: vocabularies are accepted on ingestion.
TELEMETRY_WALL_PHASES = {
    "assembly": "assemble",  # dispatcher vocabulary
    "assemble": "assemble",  # planner vocabulary
    "hold": "hold",
    "handback": "release",  # dispatcher vocabulary
    "release": "release",  # planner vocabulary
}


def wall_id_for(members: Sequence[str], speed_mps: float) -> str:
    """Deterministic wall id for a member set and speed."""
    return f"wall:{'+'.join(members)}@{speed_mps:g}"


def assembly_bar_ok(
    prediction: Prediction, wall_id: str, config: PlannerConfig
) -> "tuple[bool, str]":
    """Contract v2 future-wall bar (planner defaults pending owner
    confirmation): a proposed wall additionally requires the MEDIAN over
    M*K cells of ``p_assembly_completes`` >= ``config.wall_min_p_assembly``
    and the median of ``p_assembly_abort`` <= ``config.wall_max_p_abort``.

    A prediction without v2 wall heads (v1/v1.1, or v2 with the block
    absent) passes vacuously -- the bar gates only what the contract
    reports. NaN medians fail closed (``not (median >= floor)``)."""
    block = prediction.walls.get(wall_id)
    if block is None or getattr(block, "p_assembly_completes", None) is None:
        return True, "no v2 future-wall heads: assembly bar not applicable"
    completes = np.asarray(block.p_assembly_completes, dtype=float)
    median_completes = float(np.median(completes))
    if not (median_completes >= config.wall_min_p_assembly):
        return False, (
            f"median p_assembly_completes {median_completes:.3f} < "
            f"wall_min_p_assembly {config.wall_min_p_assembly:g}"
        )
    aborts = np.asarray(block.p_assembly_abort, dtype=float)
    median_abort = float(np.median(aborts))
    if not (median_abort <= config.wall_max_p_abort):
        return False, (
            f"median p_assembly_abort {median_abort:.3f} > "
            f"wall_max_p_abort {config.wall_max_p_abort:g}"
        )
    return True, (
        f"v2 assembly bar passed (median completes {median_completes:.3f}, "
        f"median abort {median_abort:.3f})"
    )


def cells_rule(mask, rule: str) -> bool:
    """Aggregate a bool[M, K] wall-event mask over cells ('any' | 'majority')."""
    mask = np.asarray(mask, dtype=bool)
    if rule == "any":
        return bool(mask.any())
    if rule == "majority":
        return bool(mask.mean() > 0.5)  # strict majority
    raise ValueError(f"unknown wall cell rule {rule!r}")


@dataclass(frozen=True)
class WallCandidate:
    """One proposable wall: spec + the cross-section spread it requires."""

    spec: WallSpec
    spread_m: float

    @property
    def order_key(self):
        # None-sorted entries map to +inf so numeric hold options sort first
        # and the key always compares
        hold = float("inf") if self.spec.hold_s is None else float(self.spec.hold_s)
        release = -1.0 if self.spec.release_at_s is None else float(self.spec.release_at_s)
        return (self.spec.members, self.spec.speed_mps, hold, release)


def t0_lane_position(
    prediction: Prediction, cav_ids: Iterable[str]
) -> "dict[str, tuple[float, float]]":
    """{cav: (lane, s)} at t0: median over ALL M*K cells of the tracked columns."""
    out = {}
    for cav in cav_ids:
        arr = np.asarray(prediction.trajectories_tracked[cav], dtype=float)
        out[cav] = (float(np.median(arr[:, :, 0, 2])), float(np.median(arr[:, :, 0, 1])))
    return out


def feasible_wall_members(
    free_ids, controller_state, samples, config: PlannerConfig, *, t0_s: float
) -> Tuple[Tuple[str, ...], List[str]]:
    """PL01-E4 proposer feasibility filter (fail-closed toward the
    dispatcher, which aborts a wall on ``member_frame_missing``).

    A free CAV is eligible as a proposed wall member only if
    - it is LIVE: present in the ``controller_state`` roster (designated and
      inserted), and
    - its frame is FRESH: the freshest observation across the state samples
      is at most ``PlannerConfig.wall_member_max_frame_age_s`` old at the
      decision time (0.0 = a frame at this tick). Frame sources, declared:
      the samples' optional ``frames`` mapping ({vehicle: last-frame time};
      an interface request to WD-01 -- the contract state sample has no
      frame-age field), else membership of the hybrid ``tracked``
      observed-vehicle layer (a frame at that sample's time).

    When NO sample carries any frame information the filter passes everyone
    through (legacy worlds without frame data -- declared interpretation).

    Returns (eligible ids, per-CAV exclusion reasons)."""
    max_age = config.wall_member_max_frame_age_s
    roster = set(controller_state or {})
    have_frames = any(_frame_map(sample) is not None for sample in samples)
    eligible: List[str] = []
    reasons: List[str] = []
    for cav in free_ids:
        if cav not in roster:
            reasons.append(f"{cav}: not in the controller_state roster (not designated)")
            continue
        if not have_frames:
            eligible.append(cav)  # legacy call: no frame info anywhere
            continue
        age = _frame_age(cav, samples, t0_s)
        if age is None:
            reasons.append(f"{cav}: no observed frame (not yet inserted or not reporting)")
        elif age > max_age + 1e-9:
            reasons.append(f"{cav}: stale frame (age {age:g} s > {max_age:g} s)")
        else:
            eligible.append(cav)
    return tuple(eligible), reasons


def _frame_map(sample):
    """The frame source of one sample, if it carries any (else None)."""
    frames = getattr(sample, "frames", None)
    if isinstance(frames, Mapping):
        return frames
    hybrid = getattr(sample, "hybrid", None)
    if isinstance(hybrid, Mapping) and isinstance(hybrid.get("tracked"), Mapping):
        return hybrid["tracked"]
    return None


def _frame_age(cav, samples, t0_s: float) -> Optional[float]:
    """Freshest observed-frame age (s) of ``cav`` across samples; None when
    no sample observes it."""
    best = None
    for sample in samples:
        frames = getattr(sample, "frames", None)
        if isinstance(frames, Mapping) and cav in frames:
            age = max(0.0, t0_s - float(frames[cav]))
            best = age if best is None else min(best, age)
            continue
        hybrid = getattr(sample, "hybrid", None)
        tracked = hybrid.get("tracked") if isinstance(hybrid, Mapping) else None
        if isinstance(tracked, Mapping) and cav in tracked:
            sample_time = float((sample.meta or {}).get("time_s", t0_s))
            age = max(0.0, t0_s - sample_time)
            best = age if best is None else min(best, age)
    return best


def propose_walls(
    t0_state: Mapping[str, "tuple[float, float]"],
    free_ids: Iterable[str],
    config: PlannerConfig,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[WallCandidate, ...]:
    """Deterministic wall candidates over full cross-sections of free CAVs.

    The lane universe is the sorted set of t0 lanes among the free CAVs (the
    planner observes only tracked CAVs); every combination of one CAV per
    lane whose longitudinal spread fits becomes member sets, crossed with the
    speed and release-option grids. Fewer than two observed lanes -> no
    proposal (a wall spans a cross-section, not a single lane).
    """
    free = sorted(free_ids)
    lanes = sorted({t0_state[cav][0] for cav in free})
    if len(lanes) < 2:
        if stats is not None:
            stats["lanes"] = len(lanes)
            stats["cross_sections"] = 0
            stats["after_gap"] = 0
            stats["candidates"] = 0
        return ()
    if stats is not None:
        stats["lanes"] = len(lanes)
    per_lane = []
    for lane in lanes:
        per_lane.append(
            sorted(
                (cav for cav in free if t0_state[cav][0] == lane), key=lambda c: (t0_state[c][1], c)
            )
        )
    out = []
    n_cross_sections = 0
    n_after_gap = 0
    for combo in itertools.product(*per_lane):
        members = tuple(sorted(combo, key=lambda c: (t0_state[c][0], c)))
        positions = [t0_state[c][1] for c in members]
        spread = max(positions) - min(positions)
        if spread > config.wall_max_member_gap_m:
            continue
        n_cross_sections += 1
        for speed in config.wall_speeds_mps:
            for option in config.wall_release_options_s:
                # Every release option encodes as hold_s (seconds after
                # assembly completes); the None option is the open hold --
                # hold until the planner releases by omission, never an
                # automatic (or release_at_s) release.
                hold_s = config.wall_open_hold_s if option is None else float(option)
                spec = WallSpec(
                    wall_id_for(members, speed),
                    members,
                    float(speed),
                    "assemble",
                    hold_s=float(hold_s),
                )
                out.append(WallCandidate(spec, spread))
                n_after_gap += 1
    if stats is not None:
        stats["cross_sections"] = n_cross_sections
        stats["after_gap"] = n_after_gap
        stats["candidates"] = len(out)
    return tuple(sorted(out, key=lambda c: c.order_key))


def fold_wall_group(
    spec: WallSpec,
    wall_prediction: Prediction,
    current_prediction: Prediction,
    other_ids: Iterable[str],
    config: PlannerConfig,
) -> "Optional[tuple[list[str], list[str]]]":
    """Fold measured-affected non-member CAVs into the wall's group.

    Returns (folded, dropped): folded = the strongest-interaction CAVs that
    fit under the cap (members counted), dropped = the weakest-interaction
    remainder. None when the members alone already exceed the cap (members
    are never dropped).
    """
    if len(spec.members) > config.group_cap:
        return None
    member_set = set(spec.members)
    weights = {}
    for cav in sorted(other_ids):
        if cav in member_set:
            continue
        d_pos, d_speed = trajectory_deviation(
            wall_prediction.trajectories_tracked[cav], current_prediction.trajectories_tracked[cav]
        )
        pos = float(np.median(d_pos))
        speed = float(np.median(d_speed))
        if pos > config.interaction_pos_m or speed > config.interaction_speed_mps:
            weights[cav] = max(pos / config.interaction_pos_m, speed / config.interaction_speed_mps)
    budget = config.group_cap - len(spec.members)
    weakest_first = sorted(weights, key=lambda c: (weights[c], c))
    dropped = weakest_first[: max(0, len(weakest_first) - budget)]
    folded = sorted(cav for cav in weights if cav not in set(dropped))
    return folded, dropped


@dataclass
class WallRecord:
    """Planner-side registry entry for one dispatched wall."""

    wall_id: str
    members: Tuple[str, ...]  # lane-ordered
    speed_mps: float
    phase: str  # assemble | hold | release
    hold_s: Optional[float] = None
    release_at_s: Optional[float] = None
    dispatched_at_s: Optional[float] = None
    assembled_at_s: Optional[float] = None

    def to_wall_spec(self) -> WallSpec:
        return WallSpec(
            wall_id=self.wall_id,
            members=self.members,
            speed_mps=self.speed_mps,
            phase=self.phase,
            release_at_s=self.release_at_s,
            hold_s=self.hold_s,
        )

    @classmethod
    def from_wall_spec(cls, spec: WallSpec, dispatched_at_s: Optional[float] = None):
        return cls(
            wall_id=spec.wall_id,
            members=tuple(spec.members),
            speed_mps=spec.speed_mps,
            phase=spec.phase,
            hold_s=spec.hold_s,
            release_at_s=spec.release_at_s,
            dispatched_at_s=dispatched_at_s,
        )


class WallRegistry:
    """The planner's own wall book (supersedes 'explicit current_plan' adoptions)."""

    def __init__(self):
        self._walls: "dict[str, WallRecord]" = {}

    def __len__(self):
        return len(self._walls)

    def __contains__(self, wall_id):
        return wall_id in self._walls

    def get(self, wall_id) -> Optional[WallRecord]:
        return self._walls.get(wall_id)

    def all(self) -> "list[WallRecord]":
        return [self._walls[wid] for wid in sorted(self._walls)]

    def ids(self) -> "list[str]":
        return sorted(self._walls)

    def register_spec(self, spec: WallSpec, dispatched_at_s: Optional[float] = None) -> WallRecord:
        record = self._walls.get(spec.wall_id)
        if record is None:
            record = WallRecord.from_wall_spec(spec, dispatched_at_s)
            self._walls[spec.wall_id] = record
        else:
            record.phase = spec.phase
            if record.dispatched_at_s is None:
                record.dispatched_at_s = dispatched_at_s
        return record

    def sync_from_plan(self, plan: PlanCandidate) -> None:
        """Adopt walls found in a (constructor-provided) current plan."""
        for wall in plan.walls:
            self.register_spec(wall)

    def sync_from_controller_state(self, controller_state) -> "list[str]":
        """Adopt execution reality: phases, and walls the dispatcher ended.

        A registry wall whose members are all present in controller_state and
        none of them locked to it anymore has been released/aborted by the
        dispatcher -> dissolved here. Phases are adopted from the entries.
        Returns the dissolved wall ids so the caller can drop them from the
        current plan (members -> native).
        """
        if controller_state is None:
            return []
        locked_to: "dict[str, list[tuple[str, Mapping]]]" = {}
        for cav in sorted(controller_state):
            entry = controller_state[cav]
            if entry.get("locked") and entry.get("wall_id"):
                locked_to.setdefault(entry["wall_id"], []).append((cav, entry))
        dissolved = []
        for wall_id in list(self._walls):
            record = self._walls[wall_id]
            members = locked_to.get(wall_id)
            if not members:
                self.dissolve(wall_id)  # no member locked to it anymore
                dissolved.append(wall_id)
                continue
            phase = members[0][1].get("wall_phase")
            if phase in TELEMETRY_WALL_PHASES:
                record.phase = TELEMETRY_WALL_PHASES[phase]
        return sorted(dissolved)

    def walls_in(self, plan: PlanCandidate, phase: Optional[str] = None) -> "list[WallSpec]":
        return [
            w
            for w in plan.walls
            if w.wall_id in self._walls and (phase is None or w.phase == phase)
        ]

    def assembled_ids(
        self, plan: PlanCandidate, prediction: Prediction, config: PlannerConfig
    ) -> "list[str]":
        """Assemble-phase walls the prediction reports as assembled."""
        out = []
        for wall in self.walls_in(plan, "assemble"):
            block = prediction.walls.get(wall.wall_id)
            if block is not None and cells_rule(block.assembled, config.wall_cell_rule):
                out.append(wall.wall_id)
        return sorted(out)

    def abort_ids(
        self, plan: PlanCandidate, prediction: Prediction, config: PlannerConfig
    ) -> "list[str]":
        """Assemble-phase walls with a per-wall safety abort signal."""
        out = []
        for wall in self.walls_in(plan, "assemble"):
            block = prediction.walls.get(wall.wall_id)
            if block is None:
                continue
            if cells_rule(block.safety_abort, config.wall_cell_rule) or bool(
                np.any(np.asarray(block.abort_reason) != 0)
            ):
                out.append(wall.wall_id)
        return sorted(out)

    def mark_assembled(self, wall_id: str, now_s: float) -> None:
        record = self._walls[wall_id]
        record.phase = "hold"
        if record.assembled_at_s is None:
            record.assembled_at_s = float(now_s)

    def mark_release(self, wall_id: str) -> None:
        self._walls[wall_id].phase = "release"

    def release_ready(self) -> "list[WallRecord]":
        """Walls whose release was decided and that await dissolution."""
        return [
            self._walls[wid] for wid in sorted(self._walls) if self._walls[wid].phase == "release"
        ]

    def dissolve(self, wall_id: str) -> None:
        self._walls.pop(wall_id, None)
