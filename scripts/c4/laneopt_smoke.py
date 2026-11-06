#!/usr/bin/env python3
"""V10LANEOPT (2026-10-19) lane-mode smoke: staged mainline decisions in both modes.

Runs the v5m+hazard_v5 architecture with ``physics.lane_learnable=true`` over a small set of
seeded starts in ``physics.mainline_lane_mode='free'`` and ``'courtesy'`` and counts the lane
transitions the world model decides:

  merge     lane 2 -> 1 (ramp), any position inside the merge zone;
  courtesy  lane 1 -> 0 with the source inside the courtesy approach zone upstream of the merge
            end (``[merge_zone_s1_m - mainline_courtesy_zone_m, merge_zone_s1_m]``);
  overtake  lane 1 -> 0 outside that zone;
  return    lane 0 -> 1 (frozen in courtesy).

The counts are decided by the WM only (no SUMO, no data); the script is a mechanism smoke that
proves the courtesy restriction (merges kept, courtesy overtakes kept, returns and outside
overtakes frozen).  The SUMO reference for the courtesy moves (D_hdv_courtesy_per_hour ~= 100/h,
coop_metrics/out/coop.md) and the ds10/30/60/120 fidelity need the real v5m checkpoint on the
training host:

  python scripts/c4/wm_fidelity_v10.py --wm <WM>/wm_s_best.pt \\
      --states <fixed_train_val_states.json> --horizons 10,30,60,120 --device cpu \\
      --out <out>/wm_fidelity_<mode>.json

Usage:
  python scripts/c4/laneopt_smoke.py [--starts 4] [--horizon-s 120] [--out smoke.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from wms.cli.assemble import build_model, build_physics
from wms.config.load import load_train_config
from wms.physics.config import MAINLINE_COURTESY_ZONE_M

DT_S = 0.5
V5 = ('model.lane_model="hazard_v5"', 'model.lane_gap_rule="speed"',
      "loss.lane_event.weight=3.0", "loss.lane_hazard_decouple=true")
LANES = ("free", "courtesy")


def build_smoke_model(config: str, bias: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    cfg = load_train_config(config, ["runtime.device=cpu", *V5,
                                     "physics.lane_learnable=true"])
    physics = build_physics(cfg)
    model = build_model(cfg, physics).eval()
    with torch.no_grad():
        model.lane_v5.heads.bias.fill_(float(bias))
    return model


def smoke_batch(starts: int, slots: int, horizon_steps: int, seed: int = 11):
    """Seeded synthetic TRAIN-val-like scene: cars spread over the approach, all lanes.

    Returns the ``rollout`` batch (states/present/is_cav/driver history) and the initial lanes.
    """
    g = torch.Generator().manual_seed(seed)
    b, k = int(starts), int(slots)
    s = (torch.arange(k, dtype=torch.float32).unsqueeze(0).repeat(b, 1) - k // 2.0) * 45.0
    s = s + torch.rand(b, k, generator=g)
    lane = torch.arange(k).remainder(3).unsqueeze(0).repeat(b, 1)   # 0,1,2,0,1,2,...
    kin0 = torch.zeros(b, k, 7)
    kin0[..., 0] = s
    kin0[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    kin0[..., 4] = 12.0 + 4.0 * torch.rand(b, k, generator=g)
    kin0[..., 5] = 0.1
    kin0[..., 6] = 5.0
    present = torch.ones(b, k, dtype=torch.bool)
    is_cav = torch.zeros(b, k, dtype=torch.bool)
    teacher = torch.zeros(b, horizon_steps + 1, k, 4)
    teacher[..., 0] = s.unsqueeze(1).expand(-1, horizon_steps + 1, -1)
    teacher[..., 1] = lane.float().unsqueeze(1).expand(-1, horizon_steps + 1, -1)
    teacher[..., 2] = kin0[..., 4].unsqueeze(1).expand(-1, horizon_steps + 1, -1)
    h = 61
    dh = torch.rand(b, k, h, 9, generator=g)
    dh[..., 2:5] = torch.nn.functional.one_hot(lane, 3).float().unsqueeze(2).expand(-1, -1, h, -1)
    return {"states": kin0.unsqueeze(1), "present": present.unsqueeze(1), "is_cav": is_cav.unsqueeze(1),
            "lane0": lane, "cav_teacher": teacher, "driver_hist": dh,
            "driver_hist_mask": torch.ones(b, k, h, dtype=torch.bool)}


def count_transitions(lanes: torch.Tensor, positions: torch.Tensor, lane0: torch.Tensor,
                      s0: torch.Tensor, end_s: float, zone_m: float) -> dict:
    """Count lane transitions over ``[B, T, K]`` lanes/positions (source = step i-1)."""
    prev_lane = torch.cat([lane0.unsqueeze(1), lanes[:, :-1]], 1)
    prev_pos = torch.cat([s0.unsqueeze(1), positions[:, :-1]], 1)
    src, dst = prev_lane.long(), lanes.long()
    zone = (prev_pos >= end_s - zone_m) & (prev_pos <= end_s)
    counts = {
        "merge": int(((src == 2) & (dst == 1)).sum()),
        "courtesy": int(((src == 1) & (dst == 0) & zone).sum()),
        "overtake": int(((src == 1) & (dst == 0) & ~zone).sum()),
        "return": int(((src == 0) & (dst == 1)).sum()),
    }
    counts["overtake_total"] = counts["courtesy"] + counts["overtake"]
    return counts


def run_arm(model, batch, mode: str, horizon_steps: int, lane_seed: int) -> dict:
    from wms.train.inputs import state_from_kin

    physics = replace(model.physics, mainline_lane_mode=mode)
    model.physics = physics
    state = state_from_kin(batch["states"][:, 0], batch["present"][:, 0])
    state["is_cav"] = batch["is_cav"][:, 0]
    with torch.no_grad():
        out = model.rollout(
            state, n_steps=int(horizon_steps), cav_teacher=batch["cav_teacher"],
            driver_history=batch["driver_hist"], driver_history_mask=batch["driver_hist_mask"],
            lane_seed=int(lane_seed))
    counts = count_transitions(out["lanes"], out["positions"], batch["lane0"],
                               batch["states"][:, 0, :, 0], physics.merge_zone_s1_m,
                               float(physics.mainline_courtesy_zone_m))
    counts["mode"] = mode
    counts["horizon_s"] = float(int(horizon_steps) * DT_S)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/c4_wm_v10loop_iter0.toml")
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--slots", type=int, default=15)
    ap.add_argument("--horizon-s", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lane-seed", type=int, default=1234)
    ap.add_argument("--bias", type=float, default=1.0)
    ap.add_argument("--zone-m", type=float, default=MAINLINE_COURTESY_ZONE_M)
    ap.add_argument("--out", default="laneopt_smoke.json")
    args = ap.parse_args(argv)

    steps = int(round(float(args.horizon_s) / DT_S))
    model = build_smoke_model(args.config, bias=args.bias)
    model.physics = replace(model.physics, mainline_courtesy_zone_m=float(args.zone_m))
    batch = smoke_batch(args.starts, args.slots, steps, seed=args.seed)
    arms = {mode: run_arm(model, batch, mode, steps, args.lane_seed) for mode in LANES}
    report = {"schema": "c4_laneopt_smoke/1", "starts": int(args.starts),
              "horizon_s": float(args.horizon_s), "dt_s": DT_S, "slots": int(args.slots),
              "zone_m": float(args.zone_m), "lane_seed": int(args.lane_seed),
              "note": ("structural CPU smoke, random-init v5m + lane_learnable; counts are the "
                       "WM's own lane decisions, no SUMO and no ds (needs the v5m checkpoint)"),
              "arms": arms}
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
