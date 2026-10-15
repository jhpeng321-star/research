"""V6MERGE speed check: WM-S step time with the two switches OFF vs ON (GPU-free, synthetic).

python tools/v6merge_bench.py [--batch 16] [--slots 96] [--steps 20] [--repeats 5] [--threads 2]

Builds the v6 fine-tune architecture (configs/c4_wm_v6_merge_ft.toml, random init) and times
(a) a no-grad rollout of ``--steps`` model steps and (b) a training-like unroll with backward,
alternating OFF / ON ``--repeats`` times on the same synthetic batch (ramp queue at the
accel-lane end + mainline traffic). Reports the median ms per step and the ON/OFF ratio.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import replace

import torch
import torch.nn.functional as F


def batch(b: int, k: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(b, k, generator=g) * 3000.0 - 2400.0
    lane = torch.randint(0, 2, (b, k), generator=g)
    n_ramp = max(4, k // 12)
    s[:, :n_ramp] = 73.0 - 8.0 * torch.arange(n_ramp, dtype=torch.float32)
    lane[:, :n_ramp] = 2
    v = torch.rand(b, k, generator=g) * 25.0
    v[:, :n_ramp] = 0.0
    lane1 = torch.arange(n_ramp, n_ramp + 8)
    s[:, lane1] = 76.0 - 7.0 * torch.arange(1, 9, dtype=torch.float32)
    lane[:, lane1] = 1
    v[:, lane1] = 3.0
    return dict(s_m=s, v_mps=v, a_mps2=torch.zeros(b, k), lane=lane,
                length_m=torch.full_like(s, 5.0), present=torch.ones(b, k, dtype=torch.bool))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--slots", type=int, default=96)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.environ.setdefault("WMS_DATA_V6_LONGPAIR", ".")
    from wms.cli.assemble import build_physics
    from wms.config.load import load_train_config
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    cfg = load_train_config("configs/c4_wm_v6_merge_ft.toml")
    on = build_physics(cfg, resolver=lambda name: name)
    off = replace(on, merge_end_wait=False, mainline_merge_yield=False)
    torch.manual_seed(0)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=off)
    state0 = batch(a.batch, a.slots)
    b, k = a.batch, a.slots
    hist = torch.zeros(b, k, model.driver_history_len, 9)
    hist[:, :, -1, 0] = state0["v_mps"] / 30.0
    hist[:, :, -1, 2:5] = F.one_hot(state0["lane"], 3).float()
    mask = torch.zeros(hist.shape[:-1], dtype=torch.bool)
    mask[:, :, -1] = True

    def unroll(grad: bool) -> float:
        state = dict(state0)
        acc = None
        t = time.perf_counter()
        with torch.set_grad_enabled(grad):
            total = 0.0
            for i in range(a.steps):
                out = model.step(state, k_steps=i + 1, driver_history=hist,
                                 driver_history_mask=mask, lane_v5_acc=acc,
                                 lane_uniform=torch.full((b, k), 0.5))
                acc = out.get("lane_v5_acc")
                state = dict(state, s_m=out["next_s_m"], v_mps=out["next_v_mps"],
                             lane=out["next_lane"], a_mps2=out["accel_mps2"])
                total = total + out["next_s_m"].sum()
            if grad:
                model.zero_grad(set_to_none=True)
                total.backward()
        return (time.perf_counter() - t) / a.steps * 1000.0

    res = {}
    for grad in (False, True):
        times = {"OFF": [], "ON": []}
        unroll(grad)                                  # warm-up
        for _ in range(a.repeats):
            for name, phys in (("OFF", off), ("ON", on)):
                model.physics = phys
                times[name].append(unroll(grad))
        res[grad] = {n: statistics.median(x) for n, x in times.items()}
        label = "train (fwd+bwd)" if grad else "no-grad rollout"
        print(f"{label}: B={b} K={k} steps={a.steps}: OFF {res[grad]['OFF']:.1f} ms/step, "
              f"ON {res[grad]['ON']:.1f} ms/step, ratio {res[grad]['ON'] / res[grad]['OFF']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
