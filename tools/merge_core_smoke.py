"""SECUREGAP CPU 2-step wiring proof (no data, no SUMO, no training run).

Builds a small structured model with ``model.merge_decision='secure_gap'``, rolls three 0.5 s
steps on a synthetic ramp / leader / follower scene, evaluates the real
``MergeCoreTerm`` (survival + teacher-forced acceptance) on that rollout and runs two Adam
steps on the core parameters.  This proves the switch, the persistent alpha state, the loss
composition and the gradient path on the CPU without touching a dataset or a checkpoint.

    python tools/merge_core_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wms.model import ModelSpec, WmSModel  # noqa: E402
from wms.physics.config import PhysicsConfig  # noqa: E402
from wms.train.losses.base import LossContext, UnrollTargets  # noqa: E402
from wms.train.losses.merge_core import MergeCoreConfig, MergeCoreTerm  # noqa: E402


def scene():
    b, k, n = 1, 3, 4
    states = torch.zeros(b, n + 1, k, 7)
    for t in range(n + 1):
        states[0, t, 0, 0] = 20.0 + 5.0 * t
        states[0, t, 1, 0] = 40.0 + 5.0 * t
        states[0, t, 2, 0] = 5.0 + 5.0 * t
        for j in range(k):
            states[0, t, j, 4] = 8.0
            states[0, t, j, 6] = 5.0
            lane = 2 if (j == 0 and t <= 2) else 1
            states[0, t, j, 1 + lane] = 1.0
    initial = {
        "s_m": states[:, 0, :, 0], "v_mps": states[:, 0, :, 4],
        "a_mps2": torch.zeros(b, k), "lane": states[:, 0, :, 1:4].argmax(-1),
        "length_m": states[:, 0, :, 6], "present": torch.ones(b, k, dtype=torch.bool),
        "is_cav": torch.zeros(b, k, dtype=torch.bool),
        "lane_age": torch.zeros(b, k, dtype=torch.long),
    }
    targets = UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1),
        hdv_mask=torch.ones(b, n, k), aux_mask=torch.ones(b, n, k),
        aux_count=b * n * k, denom=torch.tensor(1.0), lane_target=torch.zeros(b, n + 1, k),
        loss_valid=torch.ones(b, n + 1, k), is_cav=torch.zeros(b, n + 1, k), dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)
    return initial, targets


def main() -> int:
    torch.manual_seed(0)
    model = WmSModel(ModelSpec(lane_model="hazard_v5", lane_decision="gated",
                               merge_decision="secure_gap",
                               rollout_mode="driver_params"),
                     physics=PhysicsConfig()).eval()
    initial, targets = scene()
    term = MergeCoreTerm(MergeCoreConfig(weight=1.0, tf_weight=1.0, yield_weight=0.0))
    term.prepare(targets)
    opt = torch.optim.Adam(model.merge_core.parameters(), lr=1.0e-3)
    losses = []
    for step in range(2):
        opt.zero_grad()
        roll = model.rollout(initial, n_steps=3, k0=0)
        value = term(LossContext(pred=roll, current=initial, targets=targets, c0=0, c1=3))
        if value is None or not torch.isfinite(value.value):
            raise SystemExit(f"step {step}: non-finite or missing merge-core loss")
        value.value.backward()
        opt.step()
        losses.append(float(value.value.detach()))
        print(f"step {step}: loss {losses[-1]:.6f} finite", flush=True)
    print(f"persistent alpha present: {'merge_core_alpha' in roll['state']}", flush=True)
    print(f"merge_core parameters: {sum(p.numel() for p in model.merge_core.parameters())}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
