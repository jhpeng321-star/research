"""Per-CAV lane decisions of the learned policy (owner 2026-10-12 late).

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
* Optional ``consequence_mask`` checks the new follower's unclipped IDM response
  against its own WM-inferred comfortable deceleration and forbids overlap.
* Optional ``LaneRules.secure_gap_hook = "module:function"``: a callable
  ``hook(state, other_lane, target) -> bool Tensor[B,K]`` may only veto the existing
  gap mask. It is re-run against proposed simultaneous lanes at execution; stay
  remains feasible. Empty string keeps the existing mask without importing a hook.
* Edge guard (mainline only): no discretionary change when the CAV would reach a SUMO
  edge boundary (``EDGE_BOUNDARIES_S_M``) within ``CONTROL_DT_S`` plus ``edge_guard_m``.
  SUMO lane indices are per edge and differ across these boundaries, so a held
  ``changeLane`` request must never straddle one; the guard keeps imagination and SUMO
  identical.
"""
from __future__ import annotations

import importlib
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
    secure_gap_hook: str = ""    # optional module:function, resolved only when checking gaps
    consequence_mask: bool = False  # CAVCUTIN: inferred new-follower IDM >= -b_F

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "consequence_mask":
                if not isinstance(value, bool):
                    raise ValueError("lane rule consequence_mask must be a boolean")
                continue
            if name == "secure_gap_hook":
                if not isinstance(value, str):
                    raise ValueError("lane rule secure_gap_hook must be a module:function string")
                if value:
                    module, sep, function = value.partition(":")
                    if (not sep or not all(part.isidentifier() for part in module.split("."))
                            or not function.isidentifier()):
                        raise ValueError("lane rule secure_gap_hook must be a module:function string")
                continue
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
        if set(raw) - {"secure_gap_hook", "consequence_mask"} and not any(k.startswith("rear_") for k in raw):
            # legacy payload (before LANECAL, symmetric rule): keep the symmetric semantics
            raw.update(rear_gap_base_m=None, rear_gap_time_s=None, rear_closing_time_s=None)
        return cls(**{k: (v if k in ("secure_gap_hook", "consequence_mask") or v is None else float(v))
                      for k, v in raw.items()})


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


def consequence_gap_ok(state: Mapping[str, Tensor], other_lane: Tensor, target: Tensor) -> Tensor:
    """CAVCUTIN consequence veto, using only the WM's inferred IDM parameters.

    Judge the nearest new follower's *unclipped* IDM response to insertion against
    its own learned comfortable deceleration. Clipping at -b before comparison
    would make this check vacuous. Ego must not overlap its new leader or follower.
    Emergency braking and the optional 13E secure-gap tier are separate rules.
    """
    from wms.model.driver_encoder import PARAMETER_NAMES
    from wms.physics.idm import IDMParams, idm_accel

    _, _, valid, s, v, length = _kinematics(state)
    values = state.get("inferred_driver_parameters")
    if (not isinstance(values, Tensor) or not values.is_floating_point()
            or values.shape != (*s.shape, len(PARAMETER_NAMES)) or values.device != s.device):
        raise ValueError("consequence mask requires inferred_driver_parameters floating [B,K,9]")
    params = {name: values[..., PARAMETER_NAMES.index(name)]
              for name in ("v0", "time_headway", "min_gap", "accel", "decel")}
    for name, value in params.items():
        if not bool((torch.isfinite(value[valid]) & (value[valid] > 0)).all()):
            raise ValueError(f"consequence mask requires finite positive inferred {name}")
        params[name] = torch.where(valid, value, torch.ones_like(value))
    delta = state.get("inferred_idm_delta")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("consequence mask requires inferred_idm_delta from WM physics")
    if not bool((torch.isfinite(s[valid]) & torch.isfinite(state["v_mps"][valid])
                 & (state["v_mps"][valid] >= 0) & torch.isfinite(length[valid])
                 & (length[valid] > 0)).all()):
        raise ValueError("consequence mask requires finite kinematics and positive lengths")
    ds = s[:, None, :] - s[:, :, None]
    other = ~torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)[None]
    in_target = valid[:, None, :] & (other_lane[:, None, :] == target[:, :, None]) & other
    inf = torch.full_like(ds, math.inf)
    front_gap = torch.where(in_target & (ds >= 0), ds - length[:, None, :], inf).amin(-1)
    rear_gap, rear_idx = torch.where(in_target & (ds < 0), -ds - length[:, :, None], inf).min(-1)
    follower = {name: value.gather(-1, rear_idx) for name, value in params.items()}
    response = idm_accel(v.gather(-1, rear_idx), v, rear_gap,
                         IDMParams.from_driver_params(follower, delta=delta),
                         accel_min=-math.inf, accel_max=math.inf)
    return ((front_gap >= 0) & (rear_gap >= 0)
            & (torch.isinf(rear_gap) | (response >= -follower["decel"])))


def _secure_gap_ok(state, other_lane, target, rules: LaneRules) -> Tensor:
    """Resolve an explicit, trusted hook; unavailable/malformed hooks fail closed.

    ``other_lane`` contains current lanes for feasibility and proposed lanes for
    execution. Hooks must use it (not state['lane']) to find target-lane neighbours.
    Results must be boolean [B,K] on the state device; broadcasting is not allowed.
    Hooks must be pure functions of their inputs, without hidden driver truth or
    mutation. No registry, inferred-parameter lookup or global hook state is used.
    """
    if not rules.secure_gap_hook:
        return torch.ones_like(target, dtype=torch.bool)
    module, function = rules.secure_gap_hook.split(":")
    try:
        hook = getattr(importlib.import_module(module), function)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"secure gap hook unavailable: {rules.secure_gap_hook}") from error
    if not callable(hook):
        raise ValueError(f"secure gap hook is not callable: {rules.secure_gap_hook}")
    result = hook(state, other_lane, target)
    if (not isinstance(result, Tensor) or result.dtype != torch.bool
            or result.shape != target.shape or result.device != target.device):
        raise ValueError("secure gap hook must return a bool [B,K] tensor on the state device")
    return result


def secure_gap_gs(state: Mapping[str, Tensor], other_lane: Tensor, target: Tensor) -> Tensor:
    """Two-sided GAPACC ``gs`` veto using explicitly supplied inferred parameters.

    Enable with ``secure_gap_hook="wms.control.lanes:secure_gap_gs"``. State must
    contain floating [B,K] ``secure_gap_T_s`` and ``secure_gap_b_mps2`` tensors on
    the kinematics device: finite positive inferred headway T and comfortable
    braking b for every present, valid-lane vehicle, including ego and neighbours.
    Missing/invalid parameters raise ValueError; no constants, driver types, truth
    sidecars or implicit model lookup substitute for them. Upstream inference is
    responsible for the permitted observation provenance of these explicit inputs.

    For each new follower F / leader R pair require bumper gap >=
    T_F*v_F + relu(v_F**2/(2*b_F) - v_R**2/(2*b_R)), for both ego's new leader
    and its new follower. Absent neighbours impose no constraint. This helper
    neither supplies emergency braking nor overrides the existing overlap veto.
    """
    _, _, valid, s, v, length = _kinematics(state)
    parameters = []
    for name in ("secure_gap_T_s", "secure_gap_b_mps2"):
        value = state.get(name)
        if (not isinstance(value, Tensor) or not value.is_floating_point()
                or value.shape != s.shape or value.device != s.device):
            raise ValueError(f"gs requires explicit inferred {name} floating [B,K] on the state device")
        if not (torch.isfinite(value[valid]) & (value[valid] > 0)).all():
            raise ValueError(f"gs requires finite positive inferred {name} for present vehicles")
        parameters.append(torch.where(valid, value, torch.ones_like(value)))
    headway, braking = parameters
    if not (torch.isfinite(s[valid]) & torch.isfinite(state["v_mps"][valid])
            & (state["v_mps"][valid] >= 0) & torch.isfinite(length[valid])
            & (length[valid] > 0)).all():
        raise ValueError("gs requires finite positions, nonnegative speeds and positive lengths")
    k = s.shape[-1]
    ds = s[:, None, :] - s[:, :, None]
    other = ~torch.eye(k, dtype=torch.bool, device=s.device)[None]
    in_target = valid[:, None, :] & (other_lane[:, None, :] == target[:, :, None]) & other
    inf = torch.full_like(ds, math.inf)
    front_gap, front_idx = torch.where(in_target & (ds >= 0), ds - length[:, None, :], inf).min(-1)
    rear_gap, rear_idx = torch.where(in_target & (ds < 0), -ds - length[:, :, None], inf).min(-1)
    stopping = v.square() / (2 * braking)
    need_front = headway * v + (stopping - stopping.gather(-1, front_idx)).clamp_min(0)
    need_rear = ((headway * v).gather(-1, rear_idx)
                 + (stopping.gather(-1, rear_idx) - stopping).clamp_min(0))
    # Actual 13E also retains the inferred follower's minimum bumper gap. The
    # standalone GAPACC formula omits it; use it when the WM explicitly supplies
    # that setting, without substituting a driver-type constant.
    if "secure_gap_s0_m" in state:
        s0 = state["secure_gap_s0_m"]
        if (not isinstance(s0, Tensor) or s0.shape != s.shape or s0.device != s.device
                or not s0.is_floating_point()
                or not bool((torch.isfinite(s0[valid]) & (s0[valid] >= 0)).all())):
            raise ValueError("gs requires finite nonnegative inferred secure_gap_s0_m [B,K]")
        s0 = torch.where(valid, s0, torch.zeros_like(s0))
        need_front = need_front + s0
        need_rear = need_rear + s0.gather(-1, rear_idx)
    return ((torch.isinf(front_gap) | (front_gap >= need_front))
            & (torch.isinf(rear_gap) | (rear_gap >= need_rear)))


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
    if rules.secure_gap_hook:
        left = left & _secure_gap_ok(state, lane, lane - 1, rules)
        right = right & _secure_gap_ok(state, lane, lane + 1, rules)
    if rules.consequence_mask:
        left = left & consequence_gap_ok(state, lane, lane - 1)
        right = right & consequence_gap_ok(state, lane, lane + 1)
    return torch.stack((torch.ones_like(cav), left, right), -1)


def apply_lane_choices(state: Mapping[str, Tensor], choice: Tensor, feasible: Tensor,
                       rules: LaneRules = LaneRules(), acting: Optional[Tensor] = None
                       ) -> tuple[Tensor, Tensor]:
    """Execute feasible choices; returns ``(new_lane [B,K] long, changed [B,K] bool)``.

    Simultaneous changes are re-checked against everyone's PROPOSED lanes. Legacy
    rules use one conservative pass; the consequence tier also propagates cancelled
    leavers back into occupied gaps until no further cancellation is needed.
    """
    present, lane, valid, s, v, length = _kinematics(state)
    choice = choice.long().clamp(0, N_LANE_CHOICES - 1)
    allowed = feasible.bool().gather(-1, choice[..., None]).squeeze(-1)
    want = allowed & (choice != LANE_STAY) & valid & state["is_cav"].bool()
    if acting is not None:
        want = want & acting.bool()
    delta = torch.tensor(LANE_DELTA, device=lane.device)[choice]
    target = lane + delta
    changed = want
    # With the consequence tier, cancellations are propagated to a fixed point:
    # a cancelled leaver can reappear in an accepted changer's destination gap.
    # Only remove proposals, so this terminates in at most K cancellation rounds.
    for _ in range(s.shape[-1] + 1 if rules.consequence_mask else 1):
        proposed = torch.where(changed, target, lane)
        ok = _target_gap_ok(s, v, length, valid, proposed, target, rules)
        if rules.secure_gap_hook:
            ok = ok & _secure_gap_ok(state, proposed, target, rules)
        if rules.consequence_mask:
            ok = ok & consequence_gap_ok(state, proposed, target)
        accepted = changed & ok
        if torch.equal(accepted, changed):
            break
        changed = accepted
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
