"""Native reference resolution (§5) and its unlocked diagnostic variant.

§5: in native, every CAV that is not a member of a kept wall runs native IDM.
- Walls in phase ASSEMBLE continue: they stay locked; only a safety abort ends
  them. Their wall_ids are reported in meta.native_kept_walls.
- Walls in phase HOLD are dissolved at t0; members hand back to native.
- Walls in phase RELEASE finish their handback and count as dissolved.

Encoding note (v1): dissolved walls stay in the candidate as WallSpec with
phase="release", release_at_s=0.0. Their members are locked (§3.2/§3.3), and a
locked CAV may appear only as a member of its existing wall — so "hand back to
native" cannot be expressed by moving members into per_cav. phase="release"
(release now) is exactly "dissolve at t0"; the dissolution itself is model
semantics driven by native_kept_walls: everything that is not a kept ASSEMBLE
wall runs native.

The native reference is returned as a PlanCandidate so it flows through the
same plan pipeline (and the same paired [M, K] axes) as any planner plan.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple

from .contract import (
    ACTION_NATIVE,
    PHASE_ASSEMBLE,
    PHASE_RELEASE,
    ControllerStateEntry,
    PlanCandidate,
    WallSpec,
)
from .validate import ContractViolation, _as_entry


class NativeReferenceError(ContractViolation):
    """controller_state is inconsistent with the requested CAV set."""


def _check_inputs(
    controller_state: Mapping[str, object],
    cav_ids: Sequence[str],
) -> Tuple[Mapping[str, ControllerStateEntry], Tuple[str, ...]]:
    if not isinstance(controller_state, Mapping):
        raise NativeReferenceError("controller_state must be a mapping cav_id -> entry")
    entries = {c: _as_entry(c, e) for c, e in controller_state.items()}
    if not isinstance(cav_ids, (tuple, list)) or not all(
        isinstance(c, str) and c for c in cav_ids
    ):
        raise NativeReferenceError("cav_ids must be a sequence of non-empty str")
    ids = tuple(cav_ids)
    if len(set(ids)) != len(ids):
        raise NativeReferenceError(f"cav_ids contains duplicates: {sorted(ids)}")
    missing = [c for c in ids if c not in entries]
    if missing:
        raise NativeReferenceError(
            f"cav_ids not present in controller_state: {sorted(missing)}"
        )
    extra = [c for c in entries if c not in set(ids)]
    if extra:
        raise NativeReferenceError(
            f"controller_state has entries outside cav_ids: {sorted(extra)}"
        )
    return entries, ids


def _order_members(
    members: Sequence[str],
    entries: Mapping[str, ControllerStateEntry],
) -> Tuple[str, ...]:
    """§9 (v1.1): order wall members by lane_id when every member has one;
    otherwise fall back to sorted(cav_id) — the §8.3 v1 behaviour
    (controller_state carries no lane bookkeeping there)."""
    lanes = [entries[m].lane_id for m in members]
    if all(ln is not None for ln in lanes):
        return tuple(sorted(members, key=lambda m: entries[m].lane_id))
    return tuple(sorted(members))


def _wall_groups(
    entries: Mapping[str, ControllerStateEntry],
) -> Mapping[str, Tuple[str, ...]]:
    groups: dict = {}
    for cav_id, e in entries.items():
        if e.wall_id is None:
            continue
        groups.setdefault(e.wall_id, []).append(cav_id)
    return {w: _order_members(ms, entries) for w, ms in groups.items()}


def _wall_speed(
    wall_id: str,
    members: Tuple[str, ...],
    entries: Mapping[str, ControllerStateEntry],
) -> float:
    speeds = {entries[m].wall_speed_mps for m in members}
    if len(speeds) != 1 or None in speeds:
        raise NativeReferenceError(
            f"wall {wall_id!r}: members disagree on wall_speed_mps {speeds!r}"
        )
    return float(speeds.pop())


def _consensus_phase(
    wall_id: str,
    members: Tuple[str, ...],
    entries: Mapping[str, ControllerStateEntry],
) -> str:
    phases = {entries[m].wall_phase for m in members}
    if len(phases) != 1 or None in phases:
        raise NativeReferenceError(
            f"wall {wall_id!r}: members disagree on wall_phase {phases!r}"
        )
    return phases.pop()


def native_reference(
    controller_state: Mapping[str, object],
    cav_ids: Sequence[str],
) -> Tuple[PlanCandidate, Tuple[str, ...]]:
    """Resolve the §5 native reference for the current controller state.

    Returns (native PlanCandidate, native_kept_walls):
    - ASSEMBLE walls continue: kept with members locked to the wall, encoded
      as phase="assemble" with hold_s=math.inf ("until a safety abort ends
      it"; v1 WallSpec has no explicit indefinite-hold flag — see the spec
      questions in REPORT_PREDICT.md). Ids listed in native_kept_walls.
    - HOLD and RELEASE walls are dissolved: encoded as phase="release",
      release_at_s=0.0 ("release now"); model-side they hand members back to
      native (§5).
    - Every other CAV is "native" in per_cav.
    - Member order inside a wall is sorted(cav_id): controller_state carries
      no lane ids, so the §3.3 "ordered by lane index" rule cannot be resolved
      here (spec question in REPORT_PREDICT.md).
    """
    entries, ids = _check_inputs(controller_state, cav_ids)
    groups = _wall_groups(entries)

    kept_ids = []
    wall_specs = []
    kept_members = set()
    for wall_id in sorted(groups):
        members = groups[wall_id]
        phase = _consensus_phase(wall_id, members, entries)
        speed = _wall_speed(wall_id, members, entries)
        if phase == PHASE_ASSEMBLE:
            # continues: locked, only a safety abort ends it (§5)
            wall_specs.append(
                WallSpec(
                    wall_id=wall_id,
                    members=members,
                    speed_mps=speed,
                    phase=PHASE_ASSEMBLE,
                    release_at_s=None,
                    hold_s=math.inf,
                )
            )
            kept_ids.append(wall_id)
            kept_members.update(members)
        else:
            # HOLD / RELEASE: dissolved now; members hand back to native (§5)
            wall_specs.append(
                WallSpec(
                    wall_id=wall_id,
                    members=members,
                    speed_mps=speed,
                    phase=PHASE_RELEASE,
                    release_at_s=0.0,
                    hold_s=None,
                )
            )

    # members of dissolved walls stay wall members in the release encoding, so
    # per_cav excludes every wall member (kept assemble + release-encoded)
    all_wall_members = set().union(*groups.values()) if groups else set()
    per_cav = {c: ACTION_NATIVE for c in ids if c not in all_wall_members}
    native_plan = PlanCandidate(
        plan_version=1,
        per_cav=per_cav,
        walls=tuple(wall_specs),
    )
    return native_plan, tuple(kept_ids)


def native_unlocked_reference(
    controller_state: Mapping[str, object],
    cav_ids: Sequence[str],
) -> PlanCandidate:
    """§4.5 diagnostic variant: every wall is dissolved at t0, including walls
    still in assembly. Same v1 encoding as native_reference: each existing wall
    becomes phase="release", release_at_s=0.0; per_cav covers every unlocked
    CAV (wall members stay in their walls — they are locked)."""
    entries, ids = _check_inputs(controller_state, cav_ids)
    groups = _wall_groups(entries)
    wall_specs = []
    locked_members = set()
    for wall_id in sorted(groups):
        members = groups[wall_id]
        locked_members.update(members)
        wall_specs.append(
            WallSpec(
                wall_id=wall_id,
                members=members,
                speed_mps=_wall_speed(wall_id, members, entries),
                phase=PHASE_RELEASE,
                release_at_s=0.0,
                hold_s=None,
            )
        )
    return PlanCandidate(
        plan_version=1,
        per_cav={c: ACTION_NATIVE for c in ids if c not in locked_members},
        walls=tuple(wall_specs),
    )


__all__ = [
    "NativeReferenceError",
    "native_reference",
    "native_unlocked_reference",
]
