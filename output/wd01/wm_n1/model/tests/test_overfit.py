"""Tiny CPU overfit smoke: the loss decreases on one fixed batch.

Uses the chunk-4 TRAIN micro-smoke shards when present, otherwise a synthetic
batch in the same format (the shards are absent in this checkout).  No real
training; a handful of steps on a few vehicles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model import losses as L
from model.net import WmN1Net
from synth_model import make_batch, make_step_targets, small_cfg

MICRO_ROOT = Path(__file__).resolve().parents[2] / "features" / "data"


def _load_micro_shards():
    """Return (batch, targets) from a TRAIN micro-shard, or None if absent."""
    index_p = MICRO_ROOT / "INDEX.json"
    if not index_p.is_file():
        return None
    index = json.loads(index_p.read_text(encoding="utf-8"))
    runs = index.get("runs", {})
    if not runs:
        return None
    rec = next(iter(runs.values()))

    def npy(name):
        return np.load(Path(rec["shard_dir"]) / f"{name}.npy")

    veh = torch.tensor(npy("veh_step")[0:1].astype("float32"))
    mask = torch.tensor(npy("veh_mask")[0:1].astype("bool"))
    plan = torch.tensor(npy("plan")[0:1].astype("float32"))
    wall = torch.tensor(npy("wall")[0:1].astype("float32"))
    wall_mask = torch.tensor(npy("wall_mask")[0:1].astype("bool"))
    counter = torch.tensor(npy("counter")[0:1].astype("float32"))
    batch = dict(veh_step=veh, veh_mask=mask, plan=plan, wall=wall,
                 wall_mask=wall_mask, counter=counter)
    targets = dict(
        accel=torch.tensor(npy("accel_next")[0:1].astype("float32")),
        target_mask=torch.tensor(npy("target_mask")[0:1].astype("bool")),
        lane=torch.tensor(npy("lane_next")[0:1].astype("int64")),
        arrivals=torch.tensor(npy("arrivals")[0:1].astype("float32")))
    return batch, targets


def test_tiny_overfit_loss_decreases():
    torch.manual_seed(0)
    loaded = _load_micro_shards()
    if loaded is not None:
        batch, targets = loaded
    else:
        batch = make_batch(B=2, K=8)
        targets = make_step_targets(batch)

    net = WmN1Net(small_cfg()).train()
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        out = net(**batch)
        loss = L.total_loss(out, targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    print(f"overfit smoke: first={losses[0]:.4f} last={losses[-1]:.4f}")
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.98 * losses[0]
