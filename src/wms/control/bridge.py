"""Differentiable WM-S callback with a one-second CAV decision clock."""
from __future__ import annotations

from importlib import import_module
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M
from .contract import (ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, MERGE_S0_M, MERGE_S1_M,
                       NATIVE_CAV_ACCEL_MAX_MPS2, PHYSICS_DT_S, STEPS_PER_CONTROL, CavController,
                       Policy, SafetyLayer)
from .obs import build_observations
from .rsu_field import build_rsu_field


def _merge_lane(state: Mapping[str, Tensor], cav_mask: Tensor) -> Tensor:
    """Merge downstream-first, counting earlier accepted CAVs in target gaps.

    The result is a discrete lane index (no gradient), so the geometry is read detached:
    identical values, and autograd keeps none of the K-iteration loop's temporaries."""
    lane, present = state["lane"], state["present"].bool()
    s = torch.where(present, state["s_m"], torch.zeros_like(state["s_m"])).detach()
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M)).detach()
    length = torch.where(present, length, torch.zeros_like(length))
    candidates = cav_mask & (lane == 2) & (s >= MERGE_S0_M) & (s <= MERGE_S1_M)
    rank = torch.where(candidates, s, torch.full_like(s, -float("inf")))
    order = rank.argsort(dim=-1, descending=True, stable=True)
    assigned = lane
    # K is small in the slot representation. Each iteration handles one ego
    # per batch, keeping all kinematics/device operations in Torch.
    for i in range(s.shape[1]):
        ego = order[:, i:i + 1]
        ds = s - s.gather(1, ego)
        target = present & (assigned == 1)
        # An alongside target vehicle counts as a blocked front gap.
        front = torch.where(target & (ds >= 0), ds - length,
                            torch.full_like(ds, float("inf"))).amin(-1, keepdim=True)
        rear = torch.where(target & (ds < 0), -ds - length.gather(1, ego),
                           torch.full_like(ds, float("inf"))).amin(-1, keepdim=True)
        merge = candidates.gather(1, ego) & (front >= 6.0) & (rear >= 6.0)
        next_lane = torch.where(merge, torch.ones_like(ego), assigned.gather(1, ego))
        assigned = assigned.scatter(1, ego, next_lane)
    return assigned


def make_cav_controller(policy: Policy, group_encoder, safety: SafetyLayer, record=None, *,
                        lane_rules=None, act_mask: Optional[Tensor] = None,
                        native: Optional[Callable[[Mapping[str, Tensor]], Tensor]] = None,
                        accel_max: Optional[float] = None,
                        native_accel_max: Optional[float] = None) -> CavController:
    """Create an episode-local stateful callback producing [B,K,4] motion.

    Rebuild the controller for independent episodes; step zero also resets its
    action cache. A scalar or uniform tensor physics_step_index is accepted.
    Missing indices use an internal clock. Policy decisions happen on even
    physics steps; the proposed action is held on odd steps, while safety is
    re-evaluated every tick (also after an allowed merge). ``record`` may be a
    list or callable receiving graph-connected decision dictionaries containing
    obs, actions (pre-safety), safe_actions, group_signal and log_probs. Logging
    callers can detach these if they do not need the imagination graph.

    ``lane_rules`` (:class:`wms.control.lanes.LaneRules`) enables the policy's lane head:
    at each decision the policy receives ``lane_mask`` (feasible stay/left/right, with the
    per-CAV dwell clock) and returns a ``PolicyOutput``; the chosen, conflict-checked lane
    is applied at that decision tick (discrete, held until the next decision).  The
    existing forced merge of ramp-lane CAVs runs afterwards on every tick and is never
    blocked by the head.  ``act_mask`` [B,K] (counterfactual rows): only these CAVs follow
    the policy; the other CAVs follow ``native(state) -> accel [B,K]`` (re-evaluated each
    physics step) and stay in lane (forced merge still applies).

    ``accel_max`` (ACCELCAP_NOTES.md): the upper bound of the POLICY's commands, used as the
    safety ceiling and the motion clamp of policy-driven CAVs; default ``policy.accel_max``
    (a wrapper function may expose it as an attribute), else the legacy 1.5.  Native CAVs of
    counterfactual rows use ``native_accel_max`` (default ``NATIVE_CAV_ACCEL_MAX_MPS2``).  A
    safety layer without an ``accel_max`` keyword is called unchanged.
    """
    # The WM dependency is needed only when constructing the rollout bridge;
    # importing standalone observation/policy/safety modules for SUMO is light.
    cav_module = import_module("wms.model.cav")
    direct_command_motion = cav_module.direct_command_motion
    apply_exec_cap = cav_module.apply_exec_cap
    if act_mask is not None and native is None:
        raise ValueError("act_mask requires a native CAV behaviour")
    from .safety import with_accel_max
    safety = with_accel_max(safety)
    policy_max = float(accel_max if accel_max is not None
                       else getattr(policy, "accel_max", ACCEL_MAX_MPS2))
    native_max = float(native_accel_max if native_accel_max is not None
                       else NATIVE_CAV_ACCEL_MAX_MPS2)
    # Motion clamp (float): the largest ceiling in use; per-slot ceilings go through target_v.
    motion_max = policy_max if act_mask is None else max(policy_max, native_max)
    dwell = None
    if lane_rules is not None:
        from .lanes import LaneDwell
        dwell = LaneDwell(lane_rules.dwell_s, PHYSICS_DT_S)
    held = None
    previous_mask = None
    internal_index = 0
    # Reported to the other CAVs: each CAV's last commanded (safe)
    # acceleration and lane intent; seeded from state["cmd_accel_mps2"/"lane_intent"] if given.
    last_command = last_intent = None

    def controller(state: Mapping[str, Tensor]) -> Tensor:
        nonlocal held, previous_mask, internal_index, last_command, last_intent
        index = state.get("physics_step_index", internal_index)
        if torch.is_tensor(index):
            if index.numel() == 0 or not (index == index.reshape(-1)[0]).all():
                raise ValueError("physics_step_index must be scalar or uniform across slots")
            index = int(index.reshape(-1)[0].item())
        index = int(index)
        internal_index = index + 1
        mask = state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)
        acting = mask if act_mask is None else mask & act_mask.to(mask.device).bool()
        if index == 0 or (held is not None and held.shape != state["s_m"].shape):
            held = None
            previous_mask = None
            if dwell is not None:
                dwell.reset()
            zeros = torch.zeros_like(state["s_m"])
            last_command = state.get("cmd_accel_mps2", zeros).to(zeros.dtype)
            last_intent = state.get("lane_intent", zeros).to(zeros.dtype)
        ages = dwell.step(state) if dwell is not None else None
        decision = index % STEPS_PER_CONTROL == 0
        log_entry = None
        lane_now = state["lane"]
        intent = torch.zeros_like(state["s_m"])
        if decision:
            known = mask if previous_mask is None else mask & previous_mask
            if last_command is None or last_command.shape != state["s_m"].shape:
                last_command = last_intent = torch.zeros_like(state["s_m"])
            obs = build_observations({
                **state, "cmd_accel_mps2": torch.where(known, last_command, torch.zeros_like(last_command)),
                "lane_intent": torch.where(known, last_intent, torch.zeros_like(last_intent))})
            signal = group_encoder(obs, state["s_m"], build_rsu_field(state))
            if lane_rules is not None:
                from .lanes import apply_lane_choices, lane_feasibility
                lane_mask = lane_feasibility(state, ages, lane_rules)
                out = policy(obs, signal, lane_mask=lane_mask)
                held, log_probs = out.accel, out.log_prob_accel
                lane_now, changed = apply_lane_choices(state, out.lane_choice, lane_mask,
                                                       lane_rules, acting)
                from .lanes import LANE_DELTA
                intent = torch.where(acting, torch.tensor(LANE_DELTA, device=lane_now.device)[
                    out.lane_choice].to(held.dtype), torch.zeros_like(held))
                log_entry = {"lane_choice": out.lane_choice, "lane_mask": lane_mask,
                             "lane_changed": changed, "acting": acting}
            else:
                held, log_probs = policy(obs, signal)
                log_entry = {}
            log_entry.update({"physics_step_index": index, "obs": obs, "actions": held,
                              "log_probs": log_probs, "group_signal": signal})
        elif held is None:
            held = torch.zeros_like(state["s_m"])
        # A newly present slot waits until the next decision, never inheriting an
        # absent slot's old action. Keep cache changes functional for autograd.
        if previous_mask is not None and not decision:
            held = torch.where(mask & previous_mask, held, torch.zeros_like(held))
        held = torch.where(mask, held, torch.zeros_like(held))
        previous_mask = mask
        lane = _merge_lane({**state, "lane": lane_now}, mask)
        command_accel = held
        ceiling = policy_max
        if act_mask is not None:
            command_accel = torch.where(acting, held, native({**state, "lane": lane}))
            command_accel = torch.where(mask, command_accel, torch.zeros_like(command_accel))
            ceiling = torch.where(acting, torch.full_like(held, policy_max),
                                  torch.full_like(held, native_max))
        executed = safety({**state, "lane": lane}, command_accel, accel_max=ceiling)
        if decision:
            last_command = torch.where(mask, executed, torch.zeros_like(executed))
            last_intent = intent
        if log_entry is not None and record is not None:
            log_entry["safe_actions"] = executed
            record(log_entry) if callable(record) else record.append(log_entry)
        # A nonbinding target avoids imposing a hidden speed cap on the actor.
        present = state["present"].bool()
        s = torch.where(present, state["s_m"], torch.zeros_like(state["s_m"]))
        v = torch.where(present, state["v_mps"], torch.zeros_like(state["v_mps"]))
        target_v = v + ceiling * PHYSICS_DT_S
        # EXECCAP_NOTES.md: with the WM's cav_exec_cap on, policy-driven CAVs realise at most the
        # SUMO-consistent ceiling (identity when off; native counterfactual CAVs are not capped).
        executed = apply_exec_cap(state, executed, acting)
        command = torch.stack((executed, target_v), -1)
        next_s, next_v, applied, _ = direct_command_motion(
            s, v, command, PHYSICS_DT_S,
            ACCEL_MIN_MPS2, motion_max)
        return torch.stack((next_s, lane.to(next_s.dtype), next_v, applied), -1)

    return controller
