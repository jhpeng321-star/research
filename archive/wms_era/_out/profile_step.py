"""Run-3 CPU step profiler: real data path, flag toggles, per-component time.

Builds one real TRAIN batch (same packs / argv contract as ``_out/run3_chain.sh``
and ``_out/run3_dry.py``), caches it, then times one optimiser step at the 10 s
(unroll 20) and 30 s (unroll 60) stages for a baseline (C1-equivalent) config and
for each run-2 flag alone and all together.  ``--profile`` additionally runs
cProfile on the selected variants.  TRAIN-only, no SUMO, no GPU; the batch is
materialised one at a time and packs stay memory-mapped.

Usage::

    python _out/profile_step.py --build
    python _out/profile_step.py --run --profile all,baseline
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[1]
CWD = PY_ROOT / "output" / "wd01" / "wm_n1"
if str(CWD) not in sys.path:
    sys.path.insert(0, str(CWD))

BASE = "D:/rd-staging/wmn1_feat/pack_v3"
ID_BASE = "E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities"
D9 = "E:/rd-wt/dspeed-data/_out"
D10 = "E:/rd-wt/damping-snap/_out"
ID_DS = "E:/rd-wt/a9/_out/identities_dspeed"
ID_DM = "E:/rd-wt/damping-snap/_out/identities_damping"
LPO = "E:/rd-wt/ws-p5/_out/lpo"
DEFAULT_OUTCOMES = "E:/rd-wt/ws-p2/_ext/labels_cov"

BATCH_PATH = Path(__file__).resolve().parent / "profile_batch.pt"

# Flag groups added on top of the C1-equivalent baseline.
FLAG_GROUPS = {
    "entries": dict(entries_v2=True, entry_pad=8),
    "projection": dict(projection_enabled=True, projection_min_gap_m=2.0),
    "residual": dict(residual_cap_mps2=1.5, lane_logit_cap=3.0),
    "stability": dict(stability_weight=0.01, stability_margin=1.0,
                      stability_probes=1),
    "conservation": dict(conservation_enabled=True, conservation_weight=0.01),
    "traffic": dict(traffic_loss=True),
}


def base_config(**extra):
    from wms.train import WmSTrainConfig
    cfg = dict(device="cpu", threads=4, num_workers=0, batch_size=8, lr=1e-3,
               truncation_steps=20, entries_v2=False)
    cfg.update(extra)
    return WmSTrainConfig(**cfg)


def build_batch():
    from torch.utils.data import DataLoader
    from wms.train import TrajectoryDataset, collate

    os.environ.setdefault("WM_N1_TRAIN_IDENTITIES", ID_BASE)
    os.environ.setdefault("WM_N1_TRAIN_OUTCOMES", DEFAULT_OUTCOMES)
    ds = TrajectoryDataset.from_real_packs(
        [BASE, f"{D9}/pack_dspeed", f"{D10}/pack_damping"],
        identity_dirs=[ID_BASE, ID_DS, ID_DM],
        overlays=[f"{D9}/labels_overlay_dspeed", f"{D10}/labels_overlay_damping"],
        horizon_steps=600, start_shift_prob=0.75,
        window_list=f"{LPO}/windows_train_noPI.json",
        exclude_window_list=[f"{LPO}/pi_excluded_runs.json"],
        entries_v2=True)
    print("dataset receipt n_records", ds.receipt.get("n_records"), flush=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate,
                        num_workers=0)
    batch = next(iter(loader))
    shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
              for k, v in batch.items()}
    print("batch shapes", json.dumps(shapes), flush=True)
    import torch
    torch.save(batch, BATCH_PATH)
    print("saved", BATCH_PATH, BATCH_PATH.stat().st_size, flush=True)


def _make_trainer(cfg):
    from wms.model import PhysicsConfig, WmSModel
    from wms.train import WmSTrainer
    model = WmSModel(physics=PhysicsConfig(sumo_fidelity=True, sumo_idm=True,
                                           merge_cal=True))
    return WmSTrainer(model, cfg)


def run_variant(name, batch, steps, flags, profiler=None):
    cfg = base_config(**flags)
    trainer = _make_trainer(cfg)
    # warmup forward to trigger one-time autograd/init costs (not timed)
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._forward_backward(batch, unroll_steps=steps)
    trainer.optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    rec = trainer.train_batch(batch, unroll_steps=steps)
    dt = time.perf_counter() - t0
    if profiler is not None:
        profiler.enable()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._forward_backward(batch, unroll_steps=steps)
        profiler.disable()
    return dt, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--profile", default="",
                    help="comma variants to cProfile: baseline,all,entries,projection")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    if args.build:
        build_batch()
        return 0
    if not args.run:
        ap.error("one of --build/--run")
    import torch
    batch = torch.load(BATCH_PATH)
    if args.batch is not None:
        batch = {k: (v[:args.batch] if hasattr(v, "shape") and v.dim() >= 1
                     else v) for k, v in batch.items()}
    print("batch size", batch["states"].shape[0], "K", batch["states"].shape[2],
          "T", batch["states"].shape[1], flush=True)

    variants = {"baseline": {}, "all": {k: v for g in FLAG_GROUPS.values()
                                        for k, v in g.items()}}
    for g, fl in FLAG_GROUPS.items():
        variants[g] = fl
    prof_targets = set(x for x in args.profile.split(",") if x)

    results = {}
    prof_out = {}
    for label, steps in (("10s", 20), ("30s", 60)):
        for vname, flags in variants.items():
            prof = cProfile.Profile() if vname in prof_targets else None
            try:
                dt, rec = run_variant(vname, batch, steps, flags, prof)
            except Exception as exc:  # noqa: BLE001
                results[f"{label}/{vname}"] = {"error": repr(exc)}
                continue
            results[f"{label}/{vname}"] = {"seconds": round(dt, 4),
                                           "loss": rec.get("loss")}
            print(f"{label:4s} {vname:13s} {dt:8.3f}s  loss={rec.get('loss'):.4f}",
                  flush=True)
            if prof is not None:
                st = pstats.Stats(prof)
                st.sort_stats("cumulative")
                buf = io.StringIO()
                st.stream = buf
                st.print_stats(25)
                prof_out[f"{label}/{vname}"] = buf.getvalue()
    out = {"tag": args.tag, "results": results, "profiles": prof_out}
    (Path(__file__).resolve().parent / f"profile_step_{args.tag}.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    for key, text in prof_out.items():
        (Path(__file__).resolve().parent /
         f"profile_{args.tag}_{key.replace('/', '_')}.txt").write_text(
            text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
