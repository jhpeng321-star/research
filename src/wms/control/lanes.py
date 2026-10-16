"""Per-CAV lane decisions of the learned policy.

The policy outputs, per CAV and per 1 s decision, a categorical lane choice
``stay / left / right`` (``LANE_STAY / LANE_LEFT / LANE_RIGHT``).  WM lane 0 is the
LEFT mainline lane, so ``left`` = lane - 1 and ``right`` = lane + 1.  This module holds
the feasibility mask, the conflict-checked execution of the choices and the per-CAV
dwell clock.  The SAME functions run in imagination (``wms.control.bridge``) and in
SUMO deployment (``wms.control.policy.load_policy_fn`` / ``wms.control.deploy``), so a
choice that is feasible in one is feasible in the other.

Feasibility (``lane_feasibility``), all from the state the safety layer already uses:

* ``stay`` is always feasible.
* Mainline CAVs (lanes 0, 1) may only move between lanes 0 and 1: never outside 0..1 and
  never into the ramp/acceleration lane 2.
* Ramp-lane CAVs (lane 2) may only choose ``left`` (= "merge now" into lane 1) inside the
  merge zone ``[MERGE_S0_M, MERGE_S1_M]``.  The bridge's existing forced-merge rule is
  applied afterwards and is never blocked by the head.
* Target-lane bumper gaps: front ``>= gap_base_m + gap_time_s * v_ego + closing_time_s *
  max(0, v_ego - v_front)``, rear ``>= rear_gap_base_m + rear_gap_time_s * v_rear +
  rear_closing_time_s * max(0, v_rear - v_ego)`` (the ``rear_*`` parameters default to the front
  ones when ``None``); an overlapping (alongside) vehicle blocks the change.  The defaults were
  calibrated against SUMO's acceptance of ``changeLane`` requests (``LANECAL_NOTES.md``).
* Dwell: at least ``dwell_s`` since this CAV's previous lane change.
* Edge guard (mainline only): no discretionary change when the CAV would reach a SUMO
  edge boundary (``EDGE_BOUNDARIES_S_M``) within ``CONTROL_DT_S`` plus ``edge_guard_m``.
  SUMO lane indices are per edge and differ across these boundaries, so a held
  ``changeLane`` request must never straddle one; the guard keeps imagination and SUMO
  identical.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M

from .contract import CONTROL_DT_S, MERGE_S0_M, MERGE_S1_M, N_LANES

LANE_STAY, LANE_LEFT, LANE_RIGHT = 0, 1, 2
N_LANE_CHOICES = 3
LANE_DELTA = (0, -1, 1)          # choice -> WM lane offset (lane 0 = left mainline)
LANE_CHOICE_NAMES = ("stay", "left", "right")
#: WM-frame s of SUMO edge boundaries on the mainline (upstream|merge edge, merge|downstream edge).
EDGE_BOUNDARIES_S_M = (MERGE_S0_M, MERGE_S1_M)


@dataclass(frozen=True)
class LaneRules:
    gap_base_m: float = 1.0       # FRONT: minimum bumper gap to the target-lane leader
    gap_time_s: float = 1.2       # + this time gap x ego speed
    closing_time_s: float = 4.5   # + closing speed (ego faster than the leader) x this time
    dwell_s: float = 3.0          # minimum time between two lane changes of one CAV
    edge_guard_m: float = 10.0    # mainline: no change within v*CONTROL_DT_S + this of an edge end
    # REAR (target-lane follower); None = same as the front parameter (legacy symmetric rule).
    rear_gap_base_m: Optional[float] = 3.0
    rear_gap_time_s: Optional[float] = 1.25   # x speed of the follower
    rear_closing_time_s: Optional[float] = 6.0  # x closing speed (follower faster than ego)

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None and name.startswith("rear_"):
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"lane rule {name} must be finite and nonnegative")

    @property
    def rear(self) -> tuple[float, float, float]:
        """(base, time gap, closing time) of the rear check, falling back to the front values."""
        return (self.gap_base_m if self.rear_gap_base_m is None else self.rear_gap_base_m,
                self.gap_time_s if self.rear_gap_time_s is None else self.rear_gap_time_s,
                self.closing_time_s if self.rear_closing_time_s is None else self.rear_closing_time_s)

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping]) -> "LaneRules":
        raw = dict(raw or {})
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown lane rule keys: {sorted(unknown)}")
        if raw and not any(k.startswith("rear_") for k in raw):
            # legacy payload (before LANECAL, symmetric rule): keep the symmetric semantics
            raw.update(rear_gap_base_m=None, rear_gap_time_s=None, rear_closing_time_s=None)
        return cls(**{k: (None if v is None else float(v)) for k, v in raw.items()})


def _kinematics(state: Mapping[str, Tensor]):
    present = state["present"].bool()
    lane = state["lane"].long()
    valid = present & (lane >= 0) & (lane < N_LANES)
    s = torch.where(valid, state["s_m"], torch.zeros_like(state["s_m"]))
    v = torch.where(valid, state["v_mps"], torch.zeros_like(s)).clamp_min(0.0)
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
    length = torch.where(valid, length, torch.zeros_like(s))
    return present, lane, valid, s, v, length


def _target_gap_ok(s, v, length, valid, other_lane, target, rules: LaneRules) -> Tensor:
    """[B,K] bool: ego i fits into lane ``target[i]`` given the others' lanes ``other_lane``."""
    k = s.shape[-1]
    ds = s[:, None, :] - s[:, :, None]                       # [B, ego, other] = s_j - s_i
    other = ~torch.eye(k, dtype=torch.bool, device=s.device)[None]
    in_target = valid[:, None, :] & (other_lane[:, None, :] == target[:, :, None]) & other
    inf = torch.full_like(ds, math.inf)
    front = torch.where(in_target & (ds >= 0), ds - length[:, None, :], inf)
    rear = torch.where(in_target & (ds < 0), -ds - length[:, :, None], inf)
    front_gap, front_idx = front.min(-1)
    rear_gap, rear_idx = rear.min(-1)
    v_front = v.gather(-1, front_idx)
    v_rear = v.gather(-1, rear_idx)
    need_front = (rules.gap_base_m + rules.gap_time_s * v
                  + rules.closing_time_s * (v - v_front).clamp_min(0.0))
    rear_base, rear_time, rear_closing = rules.rear
    need_rear = rear_base + rear_time * v_rear + rear_closing * (v_rear - v).clamp_min(0.0)
    front_ok = torch.isinf(front_gap) | (front_gap >= need_front)
    rear_ok = torch.isinf(rear_gap) | (rear_gap >= need_rear)
    return front_ok & rear_ok


def lane_feasibility(state: Mapping[str, Tensor], dwell_age_s: Optional[Tensor] = None,
                     rules: LaneRules = LaneRules()) -> Tensor:
    """Feasible lane choices ``[B,K,3]`` (stay/left/right); non-CAV slots: stay only."""
    present, lane, valid, s, v, length = _kinematics(state)
    cav = valid & state["is_cav"].bool()
    mainline = (lane == 0) | (lane == 1)
    zone = (s >= MERGE_S0_M) & (s <= MERGE_S1_M)
    left_geom = cav & (((lane == 1)) | ((lane == 2) & zone))
    right_geom = cav & (lane == 0)
    guard = torch.zeros_like(cav)
    reach = v * CONTROL_DT_S + rules.edge_guard_m
    for boundary in EDGE_BOUNDARIES_S_M:
        distance = boundary - s
        guard = guard | ((distance >= 0) & (distance < reach))
    discretionary = ~(mainline & guard)
    dwell_ok = (torch.ones_like(cav) if dwell_age_s is None
                else dwell_age_s.to(s.dtype) >= rules.dwell_s - 1e-6)
    left = left_geom & discretionary & dwell_ok
    right = right_geom & discretionary & dwell_ok
    left = left & _target_gap_ok(s, v, length, valid, lane, lane - 1, rules)
    right = right & _target_gap_ok(s, v, length, valid, lane, lane + 1, rules)
    return torch.stack((torch.ones_like(cav), left, right), -1)


def apply_lane_choices(state: Mapping[str, Tensor], choice: Tensor, feasible: Tensor,
                       rules: LaneRules = LaneRules(), acting: Optional[Tensor] = None
                       ) -> tuple[Tensor, Tensor]:
    """Execute feasible choices; returns ``(new_lane [B,K] long, changed [B,K] bool)``.

    Simultaneous changes are re-checked against everyone's PROPOSED lanes (one
    conservative pass: a changer whose target gap is violated by another changer is
    cancelled), so two CAVs never merge into the same gap in one tick.
    """
    present, lane, valid, s, v, length = _kinematics(state)
    choice = choice.long().clamp(0, N_LANE_CHOICES - 1)
    allowed = feasible.bool().gather(-1, choice[..., None]).squeeze(-1)
    want = allowed & (choice != LANE_STAY) & valid & state["is_cav"].bool()
    if acting is not None:
        want = want & acting.bool()
    delta = torch.tensor(LANE_DELTA, device=lane.device)[choice]
    target = lane + delta
    proposed = torch.where(want, target, lane)
    ok = _target_gap_ok(s, v, length, valid, proposed, target, rules)
    changed = want & ok
    return torch.where(changed, target, lane), changed


class LaneDwell:
    """Per-slot time since the last observed lane change (starts at ``dwell_s`` = allowed).

    ``step(state)`` is called once per tick (``dt_s`` apart) with the CURRENT state and
    returns the ages for the decision made at that tick.  A slot that newly becomes
    present starts at ``dwell_s``; junction transits (lane -1) keep the last lane.  An
    optional ``state["lane_dwell_s"]`` [B,K] seeds the first call (deployment check)."""

    def __init__(self, dwell_s: float, dt_s: float):
        self.dwell_s, self.dt_s = float(dwell_s), float(dt_s)
        self.reset()

    def reset(self) -> None:
        self.age = self.lane = self.present = None

    def step(self, state: Mapping[str, Tensor]) -> Tensor:
        lane = state["lane"].long()
        present = state["present"].bool()
        if self.age is None or self.age.shape != lane.shape:
            seed = state.get("lane_dwell_s")
            age = (torch.as_tensor(seed, dtype=torch.float32, device=lane.device).expand(lane.shape)
                   if seed is not None else torch.full(lane.shape, self.dwell_s, device=lane.device))
            age = age.clone().float()
            last = torch.where(present, lane, torch.full_like(lane, -1))
        else:
            age = self.age + self.dt_s
            known = (self.lane >= 0) & (lane >= 0) & present & self.present
            age = torch.where(known & (lane != self.lane), torch.zeros_like(age), age)
            new = present & ~self.present
            age = torch.where(new, torch.full_like(age, self.dwell_s), age)
            last = torch.where(new, torch.full_like(lane, -1), self.lane)
            last = torch.where(lane >= 0, lane, last)
        age = torch.where(present, age, torch.full_like(age, self.dwell_s))
        self.age, self.lane, self.present = age.detach(), last, present
        return self.age

    def peek(self) -> Optional[Tensor]:
        return None if self.age is None else self.age.clone()
