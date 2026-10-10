"""Synthetic fixtures for the adapter CPU tests (no real data, no SUMO).

Builds tiny randomly-initialised 5-member WM-N1 ensembles, windows in the
frozen featurizer format, a dict-backed fake corpus, and the evalh outcome /
rollout rows the predictor consumes.
"""
from __future__ import annotations

import numpy as np
import torch

from evalh.protocol import OutcomeRow, RolloutRow
from model import constants as C
from model.ensemble import WmN1Ensemble
from model.net import WmN1Config

#: A tiny architecture (fast on CPU) that still exercises every net branch.
TINY_ARCH = dict(d_model=16, n_heads=2, n_layers=1, ffn=32, bias_hidden=8,
                 counter_channels=8, dropout=0.0)

TINY_SEEDS = (11, 22, 33, 44, 55)


def tiny_ensemble(seeds=TINY_SEEDS, seed: int = 0) -> WmN1Ensemble:
    torch.manual_seed(int(seed))
    return WmN1Ensemble(cfg=WmN1Config(**TINY_ARCH), seeds=seeds)


def make_window(K: int = 6, seed: int = 0, n_cav: int = 3) -> dict:
    """One frozen-format window (unbatched tensors, all K slots occupied)."""
    g = torch.Generator().manual_seed(int(seed))
    veh = torch.zeros(K, C.VEH_F)
    mask = torch.zeros(K)
    for i in range(K):
        lane = i % C.N_LANES
        veh[i, C.S_DOWN] = -100.0 + 20.0 * i
        veh[i, C.LANE_LEFT + lane] = 1.0
        veh[i, C.V_IDX] = 10.0 + 2.0 * i
        veh[i, C.A_IDX] = 0.0
        veh[i, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
        veh[i, C.IS_CAV_IDX] = 1.0 if i < n_cav else 0.0
        veh[i, C.ACT_NATIVE if i < n_cav else C.ACT_UNKNOWN] = 1.0
        veh[i, C.PRESENT_IDX] = 1.0
        mask[i] = 1.0
    plan = torch.zeros(K, C.N_SEGMENTS, C.PLAN_F)
    plan[:n_cav, :, 0] = 1.0
    wall = torch.zeros(C.MAX_WALLS, C.WALL_F)
    wall[0, 0] = 1.0
    wall[0, 5] = 0.5
    wall_mask = torch.zeros(C.MAX_WALLS)
    wall_mask[0] = 1.0
    counter = torch.rand(C.N_ENTRANCES, C.COUNTER_BINS, generator=g)
    return dict(veh_step=veh, veh_mask=mask, plan=plan, wall=wall,
                wall_mask=wall_mask, counter=counter)


def fake_corpus(worlds, *, K: int = 6, seed0: int = 0) -> dict:
    """A ``{world: window}`` mapping (the adapter's mapping corpus path)."""
    return {str(w): make_window(K=K, seed=seed0 + i)
            for i, w in enumerate(worlds)}


def outcome_rows(worlds, taus=(60.0,), policies=("native",), t0_s=0.0) -> list:
    return [OutcomeRow(world=str(w), t0_s=t0_s, decision_t_s=0.0,
                       tau_s=float(tau), policy=str(pol), y_delay_s=1.0)
            for w in worlds for tau in taus for pol in policies]


def rollout_rows(worlds, policies=("native",), K: int = 6,
                 horizons=(10.0, 30.0), t0_s=0.0) -> list:
    h = len(horizons)
    return [RolloutRow(world=str(w), t0_s=t0_s, policy=str(pol),
                       true_pos=np.zeros((h, K)), true_speed=np.zeros((h, K)),
                       horizons_s=tuple(horizons), true_dist=1.0,
                       true_speed_scale=1.0)
            for w in worlds for pol in policies]
