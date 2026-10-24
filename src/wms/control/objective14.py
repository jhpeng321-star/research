"""POLICY14's unshaped, source-inclusive finite-population cost, in vehicle-seconds.

This uses the existing SUMO speed-deficit convention (end-of-physics-step speed).
Every requested vehicle accrues delay on road or waiting at a source until the
registered endpoint. Exits stop accruing; future requests start at their due time.
There is no admitted/completed-population divisor or throughput bonus. A terminal
queue is not erased: its accrued waiting is included, and its size is reported;
continuation beyond a short imagination is an explicitly estimated critic tail.
"""
from __future__ import annotations

import math

import torch

from .contract import PHYSICS_DT_S, SPEED_LIMIT_MPS
from .reward import team_reward


def authoritative_reward(state, next_state, accel, weights):
    parts = team_reward(state, next_state, accel, weights)
    cost = parts["delay"] + parts["source_wait"]
    hdv = state["present"].bool() & ~state["is_cav"].bool()
    # HDV followers must be monitored even when TTC optimisation is CAV-follower-only.
    hard = torch.where(hdv, torch.relu(-accel - 3.0).square(), torch.zeros_like(accel))
    return {**parts, "reward": -cost, "J": cost,
            "hdv_hard_brake": hard.sum(-1),
            "hdv_hard_brake_seconds": PHYSICS_DT_S * (hdv & (accel < -3.0)).sum(-1)}


def score_components(components, requested_population):
    """Independent *accrued* J, never a lambda return. Components are [E,H,N]."""
    road = components["delay"].sum(1).mean(0)
    source = components["source_wait"].sum(1).mean(0)
    population = torch.as_tensor(requested_population, device=road.device, dtype=road.dtype)
    if population.shape != road.shape or not bool(torch.isfinite(population).all()) or bool((population <= 0).any()):
        raise ValueError("one fixed, positive exogenous requested population is required per row")
    return {"J": road + source, "road_delay": road, "source_wait": source,
            "seconds_per_requested_vehicle": (road + source) / population}


def deployment_accel(proposal, safe_hold_speed, *, calibrated_probability=None,
                     calibration_supported=False):
    """Only calibrated learned direct outputs may replace safety-projected a=0.

    Probability is supplied by a separately fitted development-data calibrator,
    never an ensemble vote. Unsupported/missing calibration always abstains.
    """
    p = calibrated_probability
    accept = (calibration_supported and p is not None and math.isfinite(float(p))
              and 0.8 <= float(p) <= 1.0)
    return proposal if accept else safe_hold_speed


def road_delay_steps(speeds, present):
    """Recorded end-step speeds [T,N,K] -> vehicle-second road cost [T,N]."""
    return (PHYSICS_DT_S * torch.where(present.bool(),
            torch.relu(1.0 - speeds / SPEED_LIMIT_MPS), torch.zeros_like(speeds))).sum(-1)
