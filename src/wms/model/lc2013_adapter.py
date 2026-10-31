"""Small opt-in adapter; learned lane/merge decisions cannot overwrite LC2013."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wms.physics.lc2013 import STATE_KEYS, lane_change
from wms.physics.lane_conflict import resolve_lane_conflicts
from .driver_encoder import PARAMETER_NAMES
from .cav import WM_MERGE_KEY


def _identity(state):
    return state.get("vehicle_id", state.get("vid"))


def evaluate(model, state, kin, present, deciders, params):
    return lane_change(kin, present, model.physics, memory=state,
                       identity=_identity(state), driver_params=params,
                       decision_mask=deciders,
                       remaining_distance=state.get("lc2013_remaining_distance"),
                       target_remaining_distance=state.get("lc2013_target_remaining_distance"),
                       best_lane_offset=state.get("lc2013_best_lane_offset"),
                       best_occupation=state.get("lc2013_best_occupation"))


def forward_output(model, state, kin, present, deciders, params, result):
    """Forward diagnostics on current snapshot; step recomputes after motion."""
    lc = evaluate(model, state, kin, present, deciders, params)
    result.update({k: v for k, v in lc.items() if k != "next_lane"})
    result["lane_probabilities"] = F.one_hot(lc["next_lane"].clamp(0, 2), 3).float()
    # Previous-step LC messages affect this longitudinal update, as in SUMO.
    if "lc2013_advice_accel" in state:
        same = present & state["lc2013_present"].bool()
        same &= state["lc2013_lane"] == kin[..., 1:4].argmax(-1)
        ids = _identity(state)
        if ids is not None and "lc2013_identity" in state:
            same &= ids == state["lc2013_identity"]
        advice = state["lc2013_advice_accel"].to(result["accel_mps2"])
        b = (torch.full_like(advice, model.physics.idm()[1]) if params is None
             else params["decel"].to(advice))
        # SUMO patchSpeed ignores advice outside the feasible acceleration range.
        valid = same & result["hdv_mask"] & torch.isfinite(advice) & (advice >= -b)
        accel = torch.where(valid, torch.minimum(result["accel_mps2"], advice),
                             result["accel_mps2"])
        # Zero advice means no message, not a request to suppress acceleration.
        result["accel_mps2"] = torch.where(valid & (advice < 0), accel,
                                             result["accel_mps2"])
        result["accel_delta_mps2"] = result["accel_mps2"] - result["physics_accel_mps2"]


def step_output(model, state, kin, present, out, next_s, next_v, next_lane, params):
    moved = kin.clone()
    moved[..., 0], moved[..., 4], moved[..., 5] = next_s, next_v, out["accel_mps2"]
    end = model.physics.merge_zone_s1_m
    pinned = present & (state["lane"] == model.physics.merge_lane) & (next_s > end)
    moved[..., 0] = torch.where(pinned, end, moved[..., 0])
    moved[..., 4] = torch.where(pinned, 0.0, moved[..., 4])
    if params is None and "driver_parameters" in out:
        params = {name: out["driver_parameters"][..., i]
                  for i, name in enumerate(PARAMETER_NAMES)}
    deciders = out.get("lane_decider_mask", out["hdv_mask"])
    delegated = out.get(WM_MERGE_KEY)
    if delegated is not None:
        deciders = deciders | delegated.bool()
    lc = evaluate(model, state, moved, present, deciders, params)
    proposed = torch.where(deciders, lc["next_lane"], next_lane)
    resolved = resolve_lane_conflicts(next_s, kin[..., 6], state["lane"].long(),
                                      proposed, present, deciders, 0.0)
    out.update({k: v for k, v in lc.items() if k != "next_lane"})
    out["lane_probabilities"] = F.one_hot(resolved.clamp(0, 2), 3).float()
    return resolved


def carry_state(current, out, present):
    for key in STATE_KEYS:
        if key in out:
            current[key] = out[key].detach()
    current["lc2013_present"] = current["lc2013_present"] & present


def reset_newborn(current, newborn):
    if "lc2013_present" in current:
        current["lc2013_present"] = current["lc2013_present"] & ~newborn.bool()
