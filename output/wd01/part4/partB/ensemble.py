"""Part B ensemble wrapper (DESIGN_PARTS_A_B.md §3.3, §2.6).

M_B members (default 5) with distinct init seeds, bootstrap over traffic
roots at training time (not implemented here -- no real data exists).
Checkpoints are per-member state_dicts saved with torch.save and loaded
with torch.load(..., weights_only=True); device resolution from
WD_DEVICE, else cuda-if-available, else cpu; no .cuda() calls anywhere.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import DelayEstimator, DelayEstimatorConfig


def resolve_device() -> torch.device:
    env = os.environ.get("WD_DEVICE", "").strip()
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DelayEnsemble(nn.Module):
    """Members are plain DelayEstimators; forward stacks their outputs.

    forward(batch) -> [M, B, 4]; predict() -> (means [M,B,2], sigmas).
    """

    def __init__(self, cfg: DelayEstimatorConfig | None = None,
                 n_members: int = 5, base_seed: int = 20260926):
        super().__init__()
        self.cfg = cfg or DelayEstimatorConfig()
        self.n_members = int(n_members)
        self.base_seed = int(base_seed)
        members = []
        for m in range(self.n_members):
            seed_everything(base_seed + m)
            members.append(DelayEstimator(self.cfg))
        self.members = nn.ModuleList(members)

    def forward(self, batch: dict) -> torch.Tensor:
        return torch.stack([m(batch) for m in self.members], dim=0)

    @torch.no_grad()
    def predict(self, batch: dict):
        dev = resolve_device()
        out = self.to(dev).eval()(batch)          # [M, B, 4]
        mean = out[..., 0::2]                     # [M, B, 2]
        sigma = torch.exp(0.5 * out[..., 1::2]).clamp(min=1e-3)
        return mean, sigma

    # ------------------------------------------------------------ persist --
    def save_pretrained(self, out_dir) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for m, model in enumerate(self.members):
            torch.save(model.state_dict(), out_dir / f"member_{m}.pt")
        meta = dict(format="wd01_partB_ensemble/1",
                    n_members=self.n_members, base_seed=self.base_seed,
                    cfg=vars(self.cfg))
        (out_dir / "config.json").write_text(json.dumps(meta, indent=1),
                                             encoding="utf-8")
        return meta

    @classmethod
    def from_pretrained(cls, out_dir, map_location=None) -> "DelayEnsemble":
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "config.json").read_text("utf-8"))
        cfg = DelayEstimatorConfig(**meta["cfg"])
        ens = cls(cfg, n_members=meta["n_members"],
                  base_seed=meta["base_seed"])
        dev = torch.device(map_location) if map_location else resolve_device()
        for m, model in enumerate(ens.members):
            sd = torch.load(out_dir / f"member_{m}.pt", map_location=dev,
                            weights_only=True)
            model.load_state_dict(sd)
        return ens.to(dev)
