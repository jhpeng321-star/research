"""Observable highway geometry/speed limits, not driver behavior thresholds.

scenarios/idm_v5_7_extended_zone/osm.v5_7.extended.net.xml:
11553048#1_0 is 378.66m at22.22m/s (line22); acceleration lane
1487606179_0 is75.44m at31.29m/s (line45). The model's s=0 is
the start of that acceleration lane. Mainline reference limit is31.29.
Encoder physical v0 is a ratio of that reference limit (driver_encoder.py:23-24).
"""
import torch

RAMP_APPROACH_LIMIT_MPS = 22.22


def lane_desired_speed(v0, lane_limit, reference_limit):
    """SUMO getVehicleMaxSpeed: lane limit times the existing inferred speed factor."""
    return v0 * lane_limit / reference_limit


def model_lane_limit(kin, physics):
    lane = kin[..., 1:4].argmax(-1)
    approach = (lane == physics.merge_lane) & (kin[..., 0] < physics.merge_zone_s0_m)
    return torch.where(approach, torch.full_like(kin[..., 0], RAMP_APPROACH_LIMIT_MPS),
                       torch.full_like(kin[..., 0], physics.speed_limit_mps))
