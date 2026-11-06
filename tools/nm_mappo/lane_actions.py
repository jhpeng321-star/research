"""Lane-action provider for the lane-capable MAPPO comparator on wz3to2 (S-MAPPOWZL, N7).

Owner 2026-11-06 18:02: the MAPPO lane action set must be IDENTICAL to the clean5 policy's:
adjacent lanes including entries into the closing lane on wz3to2, through the LCEXEC
interface (c4/lcexec).  The MAPPO actor keeps its three-way lane head (stay / left = WM lane-1 /
right = WM lane+1); this provider decides which of left/right are feasible for each CAV and the
env writer (``n1_adapter.make_env(lane_writer="lcexec")``) executes the chosen change.

Feasibility = an execution gate in the N9 ``safe_targets`` form, evaluated for every CAV on its
OWN sensing (vehicles within 100 m): an allowed WM transition (source -> target, the ego's s inside
the transition's zone, optionally outside the edge guard of the guard points) AND the unchanged
``wms.control.lanes`` LaneRules front/rear gap check against the sensed target-lane vehicles; a
sensed vehicle in a junction (lane -1) blocks, as in N9 (fails closed).  The per-ego gate is far
too slow inside a 512-slot rollout (~ms per CAV and target), so :class:`VectorLaneGate` evaluates
the same rule for all CAVs in one tensor pass; :func:`parity` compares it with any per-ego gate on
sampled states (run against LCEXEC's gate when it lands; the launcher refuses an unverified gate).

Rule sources:
* ``lcexec()`` (config entry point for SUMO): LCEXEC's transition table
  (``LCEXEC_RULES`` export); raises :class:`LaneBindingError` until c4/lcexec provides it.
* ``provisional_rules(desc)``: the owner's clean5 description on the N1 frame (all adjacent
  existing-lane changes: 0<->1 outside the edge guard, 1<->2 inside the closing-lane change zone
  before the drop); CPU tests / parity harness only.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import torch

from wms.control.contract import CONTROL_DT_S
from wms.control.lanes import LANE_STAY, LaneRules, apply_lane_choices

LCEXEC_RULES = "tools.coopmwm.lc_runtime:mappo_lane_rules"   # expected LCEXEC export (confirm on landing)
SENSOR_RANGE_M = 100.0


class LaneBindingError(RuntimeError):
    """The LCEXEC lane interface is not available in this tree."""


@dataclass(frozen=True)
class Transition:
    source: int
    target: int
    s_lo: float = -math.inf
    s_hi: float = math.inf
    guard: bool = False          # not issued within v*dt + edge_guard before a guard point


@dataclass(frozen=True)
class GateRules:
    transitions: tuple
    guard_points_s_m: tuple = ()
    n_lanes: int = 3
    sensor_range_m: float = SENSOR_RANGE_M
    source: str = ""

    def allows(self, source: int, target: int, s: float, v: float, rules: LaneRules) -> bool:
        for t in self.transitions:
            if t.source == source and t.target == target and t.s_lo <= s <= t.s_hi:
                if t.guard and any(0 <= p - s < v * CONTROL_DT_S + rules.edge_guard_m
                                   for p in self.guard_points_s_m):
                    continue
                return True
        return False


def provisional_rules(desc=None) -> GateRules:
    """Owner's clean5 lane action set on the N1 frame (CPU tests / parity only, never SUMO)."""
    drop = 75.44
    if desc is not None and getattr(desc, "bottleneck", None):
        drop = float(desc.bottleneck.get("drop_s_m", drop))
    zone = (-math.inf, drop)
    return GateRules(transitions=(Transition(2, 1, *zone), Transition(1, 2, *zone, guard=True),
                                  Transition(1, 0, guard=True), Transition(0, 1, guard=True)),
                     guard_points_s_m=(drop,), source="provisional clean5 (owner 17:55), not LCEXEC")


def _kin(state):
    present = state["present"].bool()
    lane = state["lane"].long()
    valid = present & (lane >= 0)
    s, v = state["s_m"].float(), state["v_mps"].float()
    length = state["length_m"].float() if "length_m" in state else torch.full_like(s, 5.0)
    return present, lane, valid, s, v, length


def gap_ok_local(s, v, length, valid, lane, target, rules: LaneRules, sensor_m: float):
    """``wms.control.lanes._target_gap_ok`` restricted to each ego's own sensing (|ds| <= sensor_m)."""
    k = s.shape[-1]
    ds = s[:, None, :] - s[:, :, None]
    other = ~torch.eye(k, dtype=torch.bool, device=s.device)[None]
    in_target = (valid[:, None, :] & (lane[:, None, :] == target[:, :, None]) & other
                 & (ds.abs() <= sensor_m))
    inf = torch.full_like(ds, math.inf)
    front = torch.where(in_target & (ds >= 0), ds - length[:, None, :], inf)
    rear = torch.where(in_target & (ds < 0), -ds - length[:, :, None], inf)
    front_gap, front_idx = front.min(-1)
    rear_gap, rear_idx = rear.min(-1)
    v_front, v_rear = v.gather(-1, front_idx), v.gather(-1, rear_idx)
    need_front = rules.gap_base_m + rules.gap_time_s * v + rules.closing_time_s * (v - v_front).clamp_min(0.0)
    rear_base, rear_time, rear_closing = rules.rear
    need_rear = rear_base + rear_time * v_rear + rear_closing * (v_rear - v).clamp_min(0.0)
    return (torch.isinf(front_gap) | (front_gap >= need_front)) & (torch.isinf(rear_gap) | (rear_gap >= need_rear))


class VectorLaneGate:
    """All-CAV feasibility of the per-ego gate rule in one tensor pass."""

    def __init__(self, rules: GateRules):
        self.rules = rules

    def feasibility(self, state, ages, lane_rules: LaneRules):
        """``[B,K,3]``; absent slots never matter to anyone, so each batch row is evaluated on its
        present slots only (exact; O(Kp^2) instead of O(K^2)) and scattered back."""
        b_n, k_n = state["present"].shape
        mask = torch.zeros((b_n, k_n, 3), dtype=torch.bool, device=state["s_m"].device)
        mask[..., LANE_STAY] = True
        for b in range(b_n):
            idx = torch.nonzero(state["present"][b].bool()).flatten()
            if idx.numel() == 0:
                continue
            sub = {k: state[k][b:b + 1, idx] for k in ("present", "lane", "s_m", "v_mps", "is_cav", "length_m")
                   if k in state}
            mask[b, idx] = self._dense(sub, lane_rules)[0]
        return mask

    def _dense(self, state, lane_rules: LaneRules):
        r = self.rules
        present, lane, valid, s, v, length = _kin(state)
        cav = valid & state["is_cav"].bool()
        ds = (s[:, None, :] - s[:, :, None]).abs()
        sensed = present[:, None, :] & (ds <= r.sensor_range_m)
        junction_blocked = (sensed & (lane[:, None, :] < 0)).any(-1)
        reach = v * CONTROL_DT_S + lane_rules.edge_guard_m
        near_guard = torch.zeros_like(cav)
        for p in r.guard_points_s_m:
            dist = float(p) - s
            near_guard = near_guard | ((dist >= 0) & (dist < reach))
        mask = torch.zeros(cav.shape + (3,), dtype=torch.bool, device=s.device)
        mask[..., LANE_STAY] = True
        for choice, delta in ((1, -1), (2, +1)):
            target = lane + delta
            geom = torch.zeros_like(cav)
            for t in r.transitions:
                if t.target - t.source != delta:
                    continue
                ok = (lane == t.source) & (s >= t.s_lo) & (s <= t.s_hi)
                if t.guard:
                    ok = ok & ~near_guard
                geom = geom | ok
            in_range = (target >= 0) & (target < r.n_lanes)
            gap = gap_ok_local(s, v, length, valid, lane, target, lane_rules, r.sensor_range_m)
            mask[..., choice] = cav & in_range & geom & gap & ~junction_blocked
        return mask

    def apply(self, state, choice, mask, rules, cav):
        return apply_lane_choices(state, choice, mask, rules, cav)


def per_ego_reference(rules: GateRules, lane_rules: LaneRules = LaneRules()):
    """The N9-``safe_targets``-form per-ego gate for ``rules`` (parity reference; slow)."""
    from wms.control.lanes import _target_gap_ok

    def gate(requests, local, eligible):
        safe, blocked = {}, {}
        for vid, target in requests.items():
            rows = local.get(vid, {}).get("rows") or []
            ego = next((x for x in rows if x["vid"] == vid), None)
            if vid not in eligible or ego is None or any(x["lane"] not in range(rules.n_lanes) for x in rows):
                blocked[vid] = "missing_observable_gap_or_cav"
                continue
            if not rules.allows(int(ego["lane"]), int(target), float(ego["s_m"]), float(ego["v_mps"]), lane_rules):
                blocked[vid] = "not_allowed"
                continue
            t = lambda k, kind=float: torch.tensor([[kind(x[k]) for x in rows]])  # noqa: E731
            lane = t("lane", int)
            goal = lane.clone()
            i = rows.index(ego)
            goal[0, i] = target
            ok = _target_gap_ok(t("s_m"), t("v_mps"), t("length_m"), torch.ones_like(lane, dtype=torch.bool),
                                lane, goal, lane_rules)[0, i]
            (safe if bool(ok) else blocked)[vid] = target if bool(ok) else "unsafe_gap_or_edge"
        return safe, blocked
    return gate


def rows_of(state, b: int = 0) -> list[dict]:
    present = state["present"][b].bool()
    out = []
    for k in torch.nonzero(present).flatten().tolist():
        out.append({"vid": int(k), "s_m": float(state["s_m"][b, k]), "v_mps": float(state["v_mps"][b, k]),
                    "lane": int(state["lane"][b, k]), "is_cav": bool(state["is_cav"][b, k]),
                    "length_m": float(state["length_m"][b, k]) if "length_m" in state else 5.0})
    return out


def gate_mask(gate, state, sensor_m: float = SENSOR_RANGE_M):
    """Mask from a per-ego gate ``gate(requests, local, eligible)`` (slow; parity only)."""
    b_n, k_n = state["present"].shape
    mask = torch.zeros((b_n, k_n, 3), dtype=torch.bool)
    mask[..., LANE_STAY] = True
    for b in range(b_n):
        rows = rows_of(state, b)
        cavs = [x for x in rows if x["is_cav"] and x["lane"] >= 0]
        local = {x["vid"]: {"self": x["vid"], "rows": [y for y in rows if abs(y["s_m"] - x["s_m"]) <= sensor_m]}
                 for x in cavs}
        eligible = {x["vid"] for x in cavs}
        for choice, delta in ((1, -1), (2, +1)):
            requests = {x["vid"]: x["lane"] + delta for x in cavs}
            safe, _ = gate(requests, local, eligible)
            for vid, target in safe.items():
                if target == requests.get(vid):
                    mask[b, vid, choice] = True
    return mask


def parity(vector: VectorLaneGate, gate, states, lane_rules: LaneRules = LaneRules()) -> dict:
    """Compare the vector mask with a per-ego gate on ``states``; returns mismatch counts."""
    checked = mismatched = 0
    for state in states:
        a = vector.feasibility(state, None, lane_rules)
        b = gate_mask(gate, state, vector.rules.sensor_range_m)
        cav = state["present"].bool() & state["is_cav"].bool()
        checked += int(cav.sum()) * 2
        mismatched += int((a[..., 1:] != b[..., 1:])[cav].sum())
    return {"checked": checked, "mismatched": mismatched}


def lcexec():
    """Config entry point: ``[policy] lane_actions = "tools.nm_mappo.lane_actions:lcexec"``."""
    module_name, _, attr = LCEXEC_RULES.partition(":")
    try:
        rules = getattr(importlib.import_module(module_name), attr)()
    except (ImportError, AttributeError) as exc:
        raise LaneBindingError(f"LCEXEC lane rules {LCEXEC_RULES!r} not available ({exc}); the lane "
                               "MAPPO needs the c4/lcexec interface merged") from exc
    if not isinstance(rules, GateRules):
        raise LaneBindingError("LCEXEC lane rules must be a GateRules (adapter to be written on landing)")
    return VectorLaneGate(rules)


def provisional():
    """CPU tests only: the provisional clean5 rule set."""
    return VectorLaneGate(provisional_rules())
