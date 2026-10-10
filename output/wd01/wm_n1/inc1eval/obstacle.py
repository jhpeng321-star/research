"""Fixed stopped-obstacle support for the INC1 incident (registration section 3).

The obstacle is a non-CAV slot pinned at constant ``s`` with ``v = 0`` and
``a = 0`` for the whole window.  ``rollout2.step`` / ``rollout2.validity`` keep
its slot present and immobile while every neighbour still perceives it as a
normal stopped same-lane leader.  This module owns the placement rule and the
injection of the obstacle slot into a :class:`rollout2.state.WorldState`.

Zero SUMO.  No file in this module reads an outcome or a held-out split.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_LENGTH_M = 5.0
DEFAULT_MIN_GAP_M = 15.0
DEFAULT_SEARCH_M = 20.0

# registration section 3: mainline right lane is ``lane_phys`` 1
MAIN_RIGHT_LANE = 1


@dataclass(frozen=True)
class Obstacle:
    """One resolved fixed obstacle (registration section 3).

    ``s_m`` is the FINAL downstream position (after any logged shift); the
    merge-relative frame has the merge start at ``s_down = 0``.
    """

    lane: int
    s_m: float
    length_m: float = DEFAULT_LENGTH_M
    slot: int = -1
    shift_m: float = 0.0

    def to_record(self) -> dict:
        return dict(lane=int(self.lane), s_m=float(self.s_m),
                    length_m=float(self.length_m), slot=int(self.slot),
                    shift_m=float(self.shift_m))


def nearest_free_shift(desired_s_m: float, occupied_s_m, *,
                       min_gap_m: float = DEFAULT_MIN_GAP_M,
                       search_m: float = DEFAULT_SEARCH_M,
                       resolution_m: float = 1.0) -> float:
    """Mechanical placement rule (registration section 3).

    If ``desired_s_m`` keeps a bumper gap of at least ``min_gap_m`` from every
    occupied position it returns ``0.0``.  Otherwise it returns the smallest
    ``|shift| <= search_m`` (scanned at ``resolution_m``) that does, preferring
    the nearest free gap; it falls back to ``0.0`` when none is found.
    """
    occupied = [float(x) for x in occupied_s_m]
    if all(abs(float(desired_s_m) - o) >= float(min_gap_m) for o in occupied):
        return 0.0
    steps = int(round(float(search_m) / float(resolution_m)))
    mags = sorted(range(1, steps + 1), key=lambda k: k)
    for k in mags:
        for sign in (1.0, -1.0):
            cand = float(desired_s_m) + sign * k * float(resolution_m)
            if all(abs(cand - o) >= float(min_gap_m) for o in occupied):
                return float(sign * k * float(resolution_m))
    return 0.0


def _free_slot(present: torch.Tensor, slot) -> int:
    k_n = int(present.shape[-1])
    if slot is not None:
        slot = int(slot)
        if slot < 0 or slot >= k_n:
            raise ValueError(f"obstacle slot {slot} outside [0, {k_n})")
        if bool(present.reshape(-1)[slot]):
            raise ValueError(f"obstacle slot {slot} already occupied")
        return slot
    free = torch.nonzero(~present.reshape(-1), as_tuple=False).reshape(-1)
    if free.numel() == 0:
        raise ValueError("no free slot available for the obstacle (K full)")
    return int(free[0].item())


def obstacle_mask(world) -> torch.Tensor:
    """The obstacle mask of a world (all-false when the world has none)."""
    obst = getattr(world, "obstacle", None)
    if obst is None:
        return torch.zeros_like(world.present)
    return obst.to(torch.bool)


@torch.no_grad()
def inject_obstacle(world, *, lane: int, s_m: float,
                    length_m: float = DEFAULT_LENGTH_M,
                    min_gap_m: float = DEFAULT_MIN_GAP_M,
                    search_m: float = DEFAULT_SEARCH_M,
                    slot=None) -> tuple:
    """Return ``(world_with_obstacle, Obstacle)``.

    Batch size must be 1 (INC1 evaluates one saved state at a time).  The
    chosen slot is made present, assigned the target ``lane`` and the resolved
    position, and flagged in ``WorldState.obstacle``.  The slot's ``slot_id`` is
    set to a fresh non-negative id so it is never mistaken for the empty (-1)
    sentinel.
    """
    if int(world.batch) != 1:
        raise ValueError("inject_obstacle expects a single-world batch")
    lane = int(lane)
    present = world.present.detach().to(torch.bool)
    lane_of = world.lane.detach().to(torch.int64)
    occupied = [float(world.s_m[0, i])
                for i in range(world.slots)
                if bool(present[0, i]) and int(lane_of[0, i]) == lane]
    shift = nearest_free_shift(float(s_m), occupied, min_gap_m=min_gap_m,
                               search_m=search_m)
    slot = _free_slot(present, slot)
    s_final = float(s_m) + float(shift)

    s = world.s_m.detach().clone().to(torch.float64)
    v = world.v_mps.detach().clone().to(torch.float64)
    a = world.a_mps2.detach().clone().to(torch.float64)
    ln = world.lane.detach().clone().to(torch.int64)
    length = world.length_m.detach().clone().to(torch.float64)
    pres = world.present.detach().clone().to(torch.bool)
    is_cav = world.is_cav.detach().clone().to(torch.bool)
    wall_group = world.wall_group.detach().clone().to(torch.int64)
    slot_id = world.slot_id.detach().clone().to(torch.int64)
    obst = obstacle_mask(world).clone()

    s[0, slot] = s_final
    v[0, slot] = 0.0
    a[0, slot] = 0.0
    ln[0, slot] = lane
    length[0, slot] = float(length_m)
    pres[0, slot] = True
    is_cav[0, slot] = False
    wall_group[0, slot] = -1
    next_id = int(slot_id.max().item()) + 1
    if next_id < 0:
        next_id = 0
    slot_id[0, slot] = next_id
    obst[0, slot] = True

    from rollout2.state import WorldState

    out = WorldState(
        s_m=s.to(world.s_m.dtype), v_mps=v.to(world.v_mps.dtype),
        a_mps2=a.to(world.a_mps2.dtype), lane=ln,
        length_m=length.to(world.length_m.dtype), present=pres, is_cav=is_cav,
        wall_group=wall_group, slot_id=slot_id, obstacle=obst)
    return out, Obstacle(lane=lane, s_m=s_final, length_m=float(length_m),
                         slot=slot, shift_m=float(shift))
