"""Differentiable WM-S callback with a one-second CAV decision clock."""
from __future__ import annotations

from importlib import import_module
from typing import Mapping

import torch
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M
from .contract import (ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, MERGE_S0_M, MERGE_S1_M,
                       PHYSICS_DT_S, STEPS_PER_CONTROL, CavController, Policy, SafetyLayer)
from .obs import build_observations
from .rsu_field import build_rsu_field


def _merge_lane(state: Mapping[str, Tensor], cav_mask: Tensor) -> Tensor:
    """Merge downstream-first, counting earlier accepted CAVs in target gaps."""
    lane, present = state["lane"], state["present"].bool()
    s = torch.where(present, state["s_m"], torch.zeros_like(state["s_m"]))
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
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


def make_cav_controller(policy: Policy, group_encoder, safety: SafetyLayer, record=None
                        ) -> CavController:
    """Create an episode-local stateful callback producing [B,K,4] motion.

    Rebuild the controller for independent episodes; step zero also resets its
    action cache. A scalar or uniform tensor physics_step_index is accepted.
    Missing indices use an internal clock. Policy decisions happen on even
    physics steps; the proposed action is held on odd steps, while safety is
    re-evaluated every tick (also after an allowed merge). ``record`` may be a
    list or callable receiving graph-connected decision dictionaries containing
    obs, actions (pre-safety), safe_actions, group_signal and log_probs. Logging
    callers can detach these if they do not need the imagination graph.
    """
    # The WM dependency is needed only when constructing the rollout bridge;
    # importing standalone observation/policy/safety modules for SUMO is light.
    direct_command_motion = import_module("wms.model.cav").direct_command_motion
    held = None
    previous_mask = None
    internal_index = 0

    def controller(state: Mapping[str, Tensor]) -> Tensor:
        nonlocal held, previous_mask, internal_index
        index = state.get("physics_step_index", internal_index)
        if torch.is_tensor(index):
            if index.numel() == 0 or not (index == index.reshape(-1)[0]).all():
                raise ValueError("physics_step_index must be scalar or uniform across slots")
            index = int(index.reshape(-1)[0].item())
        index = int(index)
        internal_index = index + 1
        mask = state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)
        if index == 0 or (held is not None and held.shape != state["s_m"].shape):
            held = None
            previous_mask = None
        decision = index % STEPS_PER_CONTROL == 0
        log_entry = None
        if decision:
            obs = build_observations(state)
            signal = group_encoder(obs, state["s_m"], build_rsu_field(state))
            held, log_probs = policy(obs, signal)
            log_entry = {"physics_step_index": index, "obs": obs, "actions": held,
                         "log_probs": log_probs, "group_signal": signal}
        elif held is None:
            held = torch.zeros_like(state["s_m"])
        # A newly present slot waits until the next decision, never inheriting an
        # absent slot's old action. Keep cache changes functional for autograd.
        if previous_mask is not None and not decision:
            held = torch.where(mask & previous_mask, held, torch.zeros_like(held))
        held = torch.where(mask, held, torch.zeros_like(held))
        previous_mask = mask
        lane = _merge_lane(state, mask)
        executed = safety({**state, "lane": lane}, held)
        if log_entry is not None and record is not None:
            log_entry["safe_actions"] = executed
            record(log_entry) if callable(record) else record.append(log_entry)
        # A nonbinding target avoids imposing a hidden speed cap on the actor.
        present = state["present"].bool()
        s = torch.where(present, state["s_m"], torch.zeros_like(state["s_m"]))
        v = torch.where(present, state["v_mps"], torch.zeros_like(state["v_mps"]))
        target_v = v + ACCEL_MAX_MPS2 * PHYSICS_DT_S
        command = torch.stack((executed, target_v), -1)
        next_s, next_v, applied, _ = direct_command_motion(
            s, v, command, PHYSICS_DT_S,
            ACCEL_MIN_MPS2, ACCEL_MAX_MPS2)
        return torch.stack((next_s, lane.to(next_s.dtype), next_v, applied), -1)

    return controller
