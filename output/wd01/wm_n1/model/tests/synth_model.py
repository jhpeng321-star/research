"""Synthetic WM-N1 batches in the frozen featurizer format (no SUMO/data).

Shapes and feature indices mirror ``features/FORMAT.md`` exactly; used by the
model unit tests and the tiny overfit smoke.
"""
from __future__ import annotations

import torch

from model import constants as C
from model.net import WmN1Config


def small_cfg() -> WmN1Config:
    return WmN1Config(d_model=48, n_heads=4, n_layers=2, ffn=96,
                      bias_hidden=16, counter_channels=16)


def make_batch(B=2, K=8, W=2, n_cav=3, seed=0):
    gen = torch.Generator().manual_seed(int(seed))
    veh = torch.zeros(B, K, C.VEH_F)
    mask = torch.zeros(B, K, dtype=torch.bool)
    for b in range(B):
        for i in range(K):
            lane = i % C.N_LANES
            veh[b, i, C.S_DOWN] = -150.0 + 30.0 * i + 5.0 * b
            veh[b, i, C.LANE_LEFT + lane] = 1.0
            veh[b, i, C.V_IDX] = 8.0 + 3.0 * i
            veh[b, i, C.A_IDX] = 0.0
            veh[b, i, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
            veh[b, i, C.IS_CAV_IDX] = 1.0 if i < n_cav else 0.0
            veh[b, i, C.ACT_NATIVE if i < n_cav else C.ACT_UNKNOWN] = 1.0
            veh[b, i, C.PRESENT_IDX] = 1.0
            mask[b, i] = True
    plan = torch.zeros(B, K, C.N_SEGMENTS, C.PLAN_F)
    for b in range(B):
        for i in range(min(n_cav, K)):
            plan[b, i, :, 0] = 1.0            # plan slot 0 = native
    wall = torch.zeros(B, W, C.WALL_F)
    wall_mask = torch.zeros(B, W, dtype=torch.bool)
    wall[:, 0, 0] = 1.0                       # requested phase
    wall[:, 0, 5] = 0.5                       # speed / 40
    wall_mask[:, 0] = True
    counter = torch.rand(B, C.N_ENTRANCES, C.COUNTER_BINS, generator=gen) * 2.0
    return dict(veh_step=veh, veh_mask=mask, plan=plan, wall=wall,
                wall_mask=wall_mask, counter=counter)


def make_step_targets(batch, seed=1):
    gen = torch.Generator().manual_seed(int(seed))
    mask = batch["veh_mask"] > 0
    B, K = mask.shape
    accel = torch.randn(B, K, generator=gen, dtype=torch.float32)
    return dict(accel=accel, target_mask=mask,
                lane=torch.zeros(B, K, dtype=torch.long),
                arrivals=torch.zeros(B, C.N_ENTRANCES))


def make_outcome_targets(batch, seed=2):
    B = batch["veh_step"].shape[0]
    return dict(
        outcome_quantiles=torch.rand(B, C.N_TAU, C.N_OUTCOME_TARGETS) * 10.0,
        collision=torch.zeros(B, C.N_TAU))
