"""Run-3 controlled CPU step benchmark (min of N reps, fresh model each rep).

Loads the cached real TRAIN batch (``_out/profile_batch.pt``), then times one
optimiser step at the 10 s and 30 s stages for a C1-equivalent baseline, each
run-2 flag alone, and all flags together.  Batch is capped (memory rule), one
batch in RAM at a time, packs untouched.  TRAIN-only, CPU-only.

Usage:: python _out/bench_step.py --batch 2 --reps 3 --tag after
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

CWD = Path(__file__).resolve().parents[1] / "output" / "wd01" / "wm_n1"
if str(CWD) not in sys.path:
    sys.path.insert(0, str(CWD))

FLAG_GROUPS = {
    "entries": dict(entries_v2=True, entry_pad=8),
    "projection": dict(projection_enabled=True, projection_min_gap_m=2.0),
    "residual": dict(residual_cap_mps2=1.5, lane_logit_cap=3.0),
    "stability": dict(stability_weight=0.01, stability_margin=1.0, stability_probes=1),
    "conservation": dict(conservation_enabled=True, conservation_weight=0.01),
    "traffic": dict(traffic_loss=True),
}
ALL = {k: v for g in FLAG_GROUPS.values() for k, v in g.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="after")
    a = ap.parse_args()
    import torch
    from wms.model import PhysicsConfig, WmSModel
    from wms.train import WmSTrainConfig, WmSTrainer
    batch = torch.load(Path(__file__).resolve().parent / "profile_batch.pt")
    batch = {k: (v[:a.batch] if hasattr(v, "shape") and v.dim() >= 1 else v)
             for k, v in batch.items()}
    variants = {"baseline": {}}
    variants.update(FLAG_GROUPS)
    variants["all"] = ALL
    results = {}
    for label, steps in (("10s", 20), ("30s", 60)):
        for name, flags in variants.items():
            times = []
            for _ in range(a.reps):
                cfg = WmSTrainConfig(device="cpu", threads=4, num_workers=0,
                                     batch_size=a.batch, lr=1e-3, truncation_steps=20,
                                     **flags)
                model = WmSModel(physics=PhysicsConfig(sumo_fidelity=True, sumo_idm=True,
                                                       merge_cal=True))
                trainer = WmSTrainer(model, cfg)
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer._forward_backward(batch, unroll_steps=steps)
                trainer.optimizer.zero_grad(set_to_none=True)
                t0 = time.perf_counter()
                trainer.train_batch(batch, unroll_steps=steps)
                times.append(time.perf_counter() - t0)
            results[f"{label}/{name}"] = {"min_s": round(min(times), 4),
                                          "median_s": round(statistics.median(times), 4)}
            print(f"{label:4s} {name:13s} min={min(times):8.3f}s "
                  f"med={statistics.median(times):8.3f}s", flush=True)
    out = Path(__file__).resolve().parent / f"bench_step_{a.tag}.json"
    out.write_text(json.dumps({"batch": a.batch, "reps": a.reps, "results": results},
                              indent=2, sort_keys=True), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
