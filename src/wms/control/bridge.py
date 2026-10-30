"""Differentiable WM-S callback with a one-second CAV decision clock."""
from __future__ import annotations

from importlib import import_module
from typing import Callable, Mapping, MutableMapping, Optional

import torch
from torch import Tensor

from wms.core import speed_flags
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
    egos = s.shape[1]
    if speed_flags.enabled("fast_merge_lane") and egos and s.shape[0]:
        # S-SPEED2: candidates sort first; a non-candidate ego never merges, so every
        # iteration past the largest per-row candidate count rewrites its own lane (no-op).
        egos = int(candidates.sum(1).max())
    for i in range(egos):
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
                        native_accel_max: Optional[float] = None,
                        zone: Optional[Callable[[Mapping[str, Tensor]], Tensor]] = None,
                        symmetric_exec_cap: bool = False,
                        mask_forced_merge: bool = False,
                        policy_ramp_merge: bool = False,
                        native_ramp_merge: str = "gap6",
                        native_lane_change: bool = False,
                        policy_native_lanes: bool = False) -> CavController:
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
    physics step) and stay in lane (forced merge still applies).  ``zone(state) -> [B,K] bool``
    (V10ENTZONE, optional) additionally releases the CAVs it marks to ``native`` on every tick
    (their policy command and lane choice are dropped), whether or not ``act_mask`` is given; it
    is re-evaluated each tick so a CAV entering/leaving the zone switches behaviour immediately.

    ``accel_max`` (ACCELCAP_NOTES.md): the upper bound of the POLICY's commands, used as the
    safety ceiling and the motion clamp of policy-driven CAVs; default ``policy.accel_max``
    (a wrapper function may expose it as an attribute), else the legacy 1.5.  Native CAVs of
    counterfactual rows use ``native_accel_max`` (default ``NATIVE_CAV_ACCEL_MAX_MPS2``).  A
    safety layer without an ``accel_max`` keyword is called unchanged.

    LOOP14C ramp merges (both default off = the legacy forced 6 m rule for every CAV):

    * ``policy_ramp_merge``: policy-acting CAVs are never auto-merged; a ramp CAV reaches
      lane 1 only through its own lane head (``left`` inside the merge zone), filtered by the
      same feasibility / consequence / optional secure-gap tiers as mainline changes.  Needs
      ``lane_rules``.
    * ``native_ramp_merge = "wm_law"``: every CAV whose ramp merge is still automatic (non-acting
      CAVs; also acting ones when ``policy_ramp_merge`` is off) stays on the ramp in this callback
      and is handed to the WM's HDV merge law (``state[WM_MERGE_KEY]``, requires a mutable
      callback state and a first-passage WM); ``"gap6"`` keeps the fixed-threshold
      :func:`_merge_lane`.
    * ``native_lane_change``: non-acting CAVs use the WM's complete HDV lane path,
      including its inferred-driver discretionary decisions and first-passage merge.
      The callback advertises a pure ``native_lane_mask(state)`` before WM forward;
      the WM supplies the resulting lane targets. Policy-acting lanes retain ownership.
    """
    if native_ramp_merge not in ("gap6", "wm_law"):
        raise ValueError("native_ramp_merge must be 'gap6' or 'wm_law'")
    if not isinstance(native_lane_change, bool):
        raise ValueError("native_lane_change must be a boolean")
    if not isinstance(policy_native_lanes, bool):
        raise ValueError("policy_native_lanes must be a boolean")
    if policy_native_lanes:
        if not native_lane_change:
            raise ValueError("policy_native_lanes requires native_lane_change")
        policy_ramp_merge = False
    if policy_ramp_merge and lane_rules is None:
        raise ValueError("policy_ramp_merge needs lane_rules (the lane head is the only merge path)")
    residual_mode = bool(getattr(policy, "residual_on_native", False))
    from .execution import enabled as execution_parity_enabled
    execution_parity = residual_mode and execution_parity_enabled()
    if residual_mode:
        if native is None or not native_lane_change or lane_rules is not None:
            raise ValueError("native residual needs native command, native lanes, no policy lane rules")
        policy_ramp_merge = False
    # The WM dependency is needed only when constructing the rollout bridge;
    # importing standalone observation/policy/safety modules for SUMO is light.
    cav_module = import_module("wms.model.cav")
    direct_command_motion = cav_module.direct_command_motion
    apply_exec_cap = cav_module.apply_exec_cap
    wm_merge_key = cav_module.WM_MERGE_KEY
    if (act_mask is not None or zone is not None) and native is None:
        raise ValueError("act_mask / zone requires a native CAV behaviour")
    from .safety import with_accel_max
    safety = with_accel_max(safety)
    policy_max = float(accel_max if accel_max is not None
                       else getattr(policy, "accel_max", ACCEL_MAX_MPS2))
    native_max = float(native_accel_max if native_accel_max is not None
                       else NATIVE_CAV_ACCEL_MAX_MPS2)
    native_in_use = act_mask is not None or zone is not None or residual_mode
    if residual_mode:
        policy_max = native_max  # identical reference envelope, including exactly-zero residual
    if execution_parity:
        from .execution import ACCEL_MAX
        policy_max = min(float(getattr(policy, "accel_max", ACCEL_MAX)), ACCEL_MAX)
    # Motion clamp (float): the largest ceiling in use; per-slot ceilings go through target_v.
    motion_max = policy_max if not native_in_use else max(policy_max, native_max)
    dwell = None
    if lane_rules is not None:
        from .lanes import LaneDwell
        dwell = LaneDwell(lane_rules.dwell_s, PHYSICS_DT_S)
    held = None
    previous_mask = None
    internal_index = 0
    # Reported to the other CAVs (owner 2026-10-12 late): each CAV's last commanded (safe)
    # acceleration and lane intent; seeded from state["cmd_accel_mps2"/"lane_intent"] if given.
    last_command = last_intent = None

    def native_lane_mask(state):
        mask = state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)
        if residual_mode or policy_native_lanes:
            return mask
        acting = mask if act_mask is None else mask & act_mask.to(mask.device).bool()
        if zone is not None:
            acting = acting & ~zone(state).to(mask.device).bool()
        return mask & ~acting

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
        if zone is not None:                       # V10ENTZONE: in-zone CAVs released to native
            acting = acting & ~zone(state).to(mask.device).bool()
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
            observed_command = last_command
            observed_intent = last_intent
            if "external_command_mask" in state:
                observed_command = torch.where(state["external_command_mask"].bool(),
                                                state["cmd_accel_mps2"], last_command)
                observed_intent = torch.where(state["external_command_mask"].bool(),
                                               state["lane_intent"], last_intent)
            obs = build_observations({
                **state, "cmd_accel_mps2": (torch.zeros_like(last_command) if residual_mode else
                    torch.where(known, observed_command, torch.zeros_like(last_command))),
                "lane_intent": (torch.zeros_like(last_intent) if residual_mode else
                    torch.where(known, observed_intent, torch.zeros_like(last_intent)))})
            signal = (group_encoder(obs, state) if getattr(group_encoder, 'requires_state', False)
                      else group_encoder(obs, state["s_m"], build_rsu_field(state)))
            if lane_rules is not None:
                from .lanes import apply_lane_choices, lane_feasibility
                lane_mask = lane_feasibility(state, ages, lane_rules)
                out = policy(obs, signal, lane_mask=lane_mask)
                held, log_probs = out.accel, out.log_prob_accel
                lane_now, changed = apply_lane_choices(state, out.lane_choice, lane_mask,
                                                        lane_rules, acting)
                if policy_native_lanes:
                    lane_now, changed = state['lane'], torch.zeros_like(acting)
                from .lanes import LANE_DELTA
                intent = torch.where(acting, torch.tensor(LANE_DELTA, device=lane_now.device)[
                    out.lane_choice].to(held.dtype), torch.zeros_like(held))
                if policy_native_lanes:
                    intent = torch.zeros_like(intent)
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
        native_lanes = mask if residual_mode or policy_native_lanes else mask & ~acting
        if native_lane_change:
            targets = state.get(cav_module.WM_LANE_TARGET_KEY)
            if targets is None:
                raise ValueError("native_lane_change requires WM-computed HDV lane targets")
            # Ramp first passage is resolved after longitudinal execution by the WM.
            lane_now = torch.where(native_lanes & (state["lane"] != 2), targets, lane_now)
        # LOOP14C: CAVs whose ramp merge is automatic (all, unless the head owns acting CAVs).
        auto = mask & ~acting if policy_ramp_merge else mask
        if native_ramp_merge == "wm_law" or native_lane_change:
            # Stay on the ramp here; the WM's HDV merge law decides (and may veto) the crossing.
            if not isinstance(state, MutableMapping):
                raise TypeError("native_ramp_merge='wm_law' needs a mutable callback state")
            delegated = (mask if policy_native_lanes else auto) if native_ramp_merge == "wm_law" else native_lanes
            state[wm_merge_key] = delegated & (lane_now == 2)
            lane = (lane_now if native_ramp_merge == "wm_law" else
                    _merge_lane({**state, "lane": lane_now}, auto & ~native_lanes))
        else:
            lane = _merge_lane({**state, "lane": lane_now}, auto)
        if mask_forced_merge and lane_rules is not None:
            from .lanes import apply_lane_choices, lane_feasibility
            # The old bridge's forced ramp merge bypassed the learned head's veto.
            forced = (lane_now == 2) & (lane == 1)
            forced_state = {**state, "lane": lane_now}
            feasible = lane_feasibility(forced_state, ages, lane_rules)
            lane, _ = apply_lane_choices(forced_state, forced.long(), feasible,
                                         lane_rules, acting=forced)
        command_accel = held
        ceiling = policy_max
        motion_lane = lane
        if native_lane_change:
            # The WM's HDV lane substep follows longitudinal execution. Native
            # CAVs keep the learned head/safety evaluation in their CURRENT lane;
            # their proposed HDV lane may still be cancelled by final conflicts.
            motion_lane = torch.where(native_lanes, state["lane"], lane)
        if native_in_use:
            baseline = native({**state, "lane": motion_lane})
            if residual_mode:
                from .nativeinit import compose_native
                command_accel = compose_native(baseline, torch.where(acting, held, 0.0),
                                              ACCEL_MIN_MPS2, native_max)
            else:
                command_accel = torch.where(acting, held, baseline)
            command_accel = torch.where(mask, command_accel, torch.zeros_like(command_accel))
            ceiling = torch.where(acting, torch.full_like(held, policy_max),
                                  torch.full_like(held, native_max))
            if execution_parity:
                ceiling = torch.where(acting & (held != 0.), torch.full_like(held, policy_max),
                                      torch.full_like(held, native_max))
        executed = safety({**state, "lane": motion_lane}, command_accel, accel_max=ceiling)
        if execution_parity:
            from .execution import executed_residual
            from .continuous import project_native
            native_executed = project_native(state, baseline)
            executed = torch.where(acting, executed_residual(
                state, held, native_executed, ACCEL_MIN_MPS2, policy_max), native_executed)
        # Deployment parity (PolicyExecutor): a change is executed by SUMO during the next
        # physics step (and may be refused), so the command must be safe in BOTH the lane the
        # CAV is leaving and the target lane.  Without this, imagination credited a change
        # with an acceleration only the target lane allowed (the old-lane leader was ignored
        # for one tick), over-crediting CAV lane changes.  Only changed slots are affected.
        changed_now = log_entry.get("lane_changed") if log_entry is not None else None
        if not execution_parity and changed_now is not None and bool(changed_now.any()):
            old_lane_safe = safety({**state, "lane": state["lane"]}, command_accel,
                                   accel_max=ceiling)
            executed = torch.where(changed_now, torch.minimum(executed, old_lane_safe), executed)
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
        if not execution_parity:
            executed = apply_exec_cap(state, executed, mask if symmetric_exec_cap else acting)
        command = torch.stack((executed, target_v), -1)
        next_s, next_v, applied, _ = direct_command_motion(
            s, v, command, PHYSICS_DT_S,
            ACCEL_MIN_MPS2, motion_max)
        if execution_parity:
            # speedMode27 disables bit-2 comfortable-decel clipping; a binding
            # safe-speed cap may legitimately brake harder than the policy bound.
            # E3's default SUMO update is semi-implicit Euler, not ballistic.
            commanded = mask
            parity_v = (v + executed * PHYSICS_DT_S).clamp_min(0.)
            next_s = torch.where(commanded, s + parity_v * PHYSICS_DT_S, next_s)
            next_v = torch.where(commanded, parity_v, next_v)
            applied = torch.where(commanded, (parity_v-v)/PHYSICS_DT_S, applied)
        return torch.stack((next_s, lane.to(next_s.dtype), next_v, applied), -1)

    if native_lane_change:
        controller.native_lane_mask = native_lane_mask
    return controller
