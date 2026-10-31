"""Small opt-in adapter; learned lane/merge decisions cannot overwrite LC2013."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wms.physics.lc2013 import STATE_KEYS, lane_change
from wms.physics.lane_conflict import resolve_lane_conflicts
from wms.physics.lc2013_geometry import lane_position_shift
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
                       target_best_lane_offset=state.get("lc2013_target_best_lane_offset"),
                       best_occupation=state.get("lc2013_best_occupation"),
                       current_lane_position=state.get("lc2013_current_lane_position"))


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
        if "lc2013_own_has_advice" in state:
            own = state["lc2013_own_advice_accel"].to(advice)
            a = (torch.full_like(advice, model.physics.idm()[0]) if params is None
                 else params["accel"].to(advice))
            valid = (same & result["hdv_mask"] & state["lc2013_own_has_advice"].bool()
                     & torch.isfinite(own) & (own >= -b) & (own <= a))
            result["accel_mps2"] = torch.where(valid,
                torch.minimum(result["accel_mps2"], own), result["accel_mps2"])
        result["accel_delta_mps2"] = result["accel_mps2"] - result["physics_accel_mps2"]


def step_output(model, state, kin, present, out, next_s, next_v, next_lane, params):
    if params is None and "driver_parameters" in out:
        params = {name: out["driver_parameters"][..., i]
                  for i, name in enumerate(PARAMETER_NAMES)}
    deciders = out.get("lane_decider_mask", out["hdv_mask"])
    delegated = out.get(WM_MERGE_KEY)
    if delegated is not None:
        deciders = deciders | delegated.bool()
    normal = kin[..., 1:4].sum(-1) > 0
    deciders = deciders & present & normal
    source = state["lane"].long()
    end = model.physics.merge_zone_s1_m
    pinned = deciders & (source == model.physics.merge_lane) & (next_s > end)
    motion_s = torch.where(pinned, end, next_s)
    motion_v = torch.where(pinned, 0.0, next_v)
    applied_a = ((motion_v - kin[..., 4]) / model.physics.dt_s).to(out["accel_mps2"])
    out["accel_mps2"] = torch.where(pinned, applied_a, out["accel_mps2"])
    out["accel_delta_mps2"] = out["accel_mps2"] - out["physics_accel_mps2"]
    moved = kin.clone()
    moved[..., 0], moved[..., 4], moved[..., 5] = motion_s, motion_v, out["accel_mps2"]
    # Fixed CAV/controller motion has already executed. Its destination lane,
    # as well as its position, constrains this HDV decision snapshot. Connector
    # rows retain their empty one-hot instead of manufacturing a lane0 body.
    fixed = present & normal & ~deciders
    fixed_lanes = F.one_hot(next_lane.clamp(0, 2), 3).to(kin)
    fixed_lanes *= ((next_lane >= 0) & (next_lane < 3)).unsqueeze(-1)
    moved[..., 1:4] = torch.where(fixed.unsqueeze(-1), fixed_lanes, moved[..., 1:4])
    lc = evaluate(model, state, moved, present, deciders, params)
    proposed = torch.where(deciders, lc["next_lane"], next_lane)
    resolved, shifted_s = resolve_shifted_lanes(motion_s, kin[..., 6], source,
                                               proposed, present, deciders, model.physics)
    out.update({k: v for k, v in lc.items() if k != "next_lane"})
    out['lc2013_position_shift'] = shifted_s - motion_s
    out['lc2013_motion_clamped'] = pinned
    out["lane_probabilities"] = F.one_hot(resolved.clamp(0, 2), 3).float()
    return resolved, shifted_s, motion_v


def resolve_shifted_lanes(s, length, source, proposed, present, deciders, physics):
    """Recheck after rejected moves restore their original coordinate origin.

    Candidates only move toward STAY, so at most K rejections reach a fixed
    point. Fixed/teacher positions already use their supplied destination frame.
    """
    candidate = proposed
    for _ in range(s.shape[-1] + 1):
        shifted = s + torch.where(deciders, lane_position_shift(s, source, candidate, physics), 0.)
        resolved = resolve_lane_conflicts(shifted, length, source, candidate, present, deciders, 0.0)
        if torch.equal(resolved, candidate):
            return resolved, shifted
        candidate = resolved
    raise RuntimeError('LC2013 lane-coordinate arbitration did not converge')


def carry_state(current, out, present):
    for key in STATE_KEYS:
        if key in out:
            current[key] = out[key].detach()
    current["lc2013_present"] = current["lc2013_present"] & present


def reset_newborn(current, newborn):
    if "lc2013_present" in current:
        current["lc2013_present"] = current["lc2013_present"] & ~newborn.bool()
