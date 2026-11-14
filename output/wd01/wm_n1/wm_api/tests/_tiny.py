"""Synthetic fixtures for the wm_api tests (no real data, no SUMO).

Builds a TINY randomly-initialised Idea A member-checkpoint directory via
the REAL trainer save path (``train.idea_a.IdeaATrainer.save_checkpoint``,
format ``wm_n1_idea_a_member/1``) and small t0 snapshots.
"""
from __future__ import annotations

import numpy as np
import torch

from train.idea_a import IdeaATrainConfig, IdeaATrainer
from wm_api import WmScorer
from wm_api.state import ObstacleRecord, StateSnapshot

TINY_CONFIG = dict(hdv_d_model=16, hdv_n_heads=2, hdv_n_layers=1,
                   hdv_ffn=32, corr_summary_dim=16,
                   arrival_counter_channels=8)

TINY_SEED = 7


def tiny_config(k: int = 2) -> IdeaATrainConfig:
    return IdeaATrainConfig(**TINY_CONFIG, ensemble_k=k)


def write_tiny_checkpoint(tmp_path, k: int = 2):
    """Write ``k`` tiny member checkpoints; return (dir, config).

    Each member's bundle is built with the REAL factory
    (``train.idea_a.build_idea_a_bundle``) and saved through the REAL
    trainer checkpoint path, so the payload format is production-faithful.
    """
    from train.idea_a import build_idea_a_bundle

    cfg = tiny_config(k)
    for m in range(1, k + 1):
        seed = 101 * m + TINY_SEED
        torch.manual_seed(seed)
        bundle = build_idea_a_bundle(cfg)
        trainer = IdeaATrainer(cfg, corpus=None, member=m, seed=seed,
                               bundle=bundle)
        trainer.save_checkpoint(tmp_path / f"member_{m}.pt")
    return tmp_path, cfg


def tiny_scorer(tmp_path, k: int = 2) -> WmScorer:
    write_tiny_checkpoint(tmp_path, k=k)
    return WmScorer(tmp_path, threads=1)


def tiny_state(n_cav: int = 3, n_hdv: int = 3, *,
               t0_s: float = 540.0,
               counters: bool = False,
               obstacles: int = 0) -> StateSnapshot:
    n = n_cav + n_hdv
    return StateSnapshot(
        ids=[f"cav{i}" for i in range(n_cav)]
        + [f"hdv{i}" for i in range(n_hdv)],
        lane_phys=np.array([1, 0, 1] + [1] * n_hdv)[:n],
        s_down_m=np.linspace(-140.0, 40.0, n),
        v_mps=np.full(n, 18.0),
        a_mps2=np.zeros(n),
        length_m=np.full(n, 5.0),
        is_cav=np.array([True] * n_cav + [False] * n_hdv),
        t0_s=t0_s,
        entrance_counters=(np.zeros((2, 30)) if counters else None),
        obstacles=tuple(
            ObstacleRecord(id=f"obs{i}", lane_phys=1,
                           s_down_m=40.0 + 6.0 * i)
            for i in range(obstacles)))


def default_plans() -> list:
    """native reference vs a 60 s damping commitment (different inputs)."""
    return [
        {"actions": {}},
        {"actions": {"cav0": [{"time_s": 0.0, "action": "damping"},
                              {"time_s": 60.0, "action": "native"}],
                     "cav1": "damping"}},
    ]


def wall_plan(speed_mps: float = 8.0) -> dict:
    return {"actions": {},
            "wall": {"wall_id": 3, "speed_mps": speed_mps, "hold_s": 120.0,
                     "members": ["cav0", "cav1"]}}
