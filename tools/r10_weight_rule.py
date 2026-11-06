"""R10 cut-in loss weight rule: match its gradient norm to the existing braking terms.

Rule (no hand-set number): with the round-10 trainable set, measure on real TRAIN windows the
accumulated gradient norm of every loss term (``loss_grad_norm`` from the real unroll runner,
unweighted cut-in = weight 1) and set

    weight = G_brake / G_cutin ,   G_brake = sum of the accumulated gradient norms of the explicit braking terms in the composed
    loss (round 9/10: brake_persist + brake_persist_quantile),   both as the geometric mean over the measured batches.

At that weight the cut-in term's gradient norm equals the combined gradient norm of the braking
terms it corrects at the init checkpoint (gradient share parity, "matches the existing braking
terms").  Weights are reported with the braking terms' *configured* weights already inside their
norms (the runner captures weighted terms).  The tool only measures; it trains nothing and
never updates parameters.

    python tools/r10_weight_rule.py --config configs/c4_wm_v10round10.toml --ckpt <init.pt> \
        --out tmp_r10/weight_rule.json --n-batches 6 --unroll-s 60 --batch-size 2 \
        --set 'data.c4_sources=[...]' ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from torch.utils.data import ConcatDataset, DataLoader, Subset  # noqa: E402

from wms.cli.assemble import TrainingInputs, assemble, build_model, build_physics  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.data.c4_episodes import build_c4_datasets  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402

# Explicit braking loss terms (brake_follow_weight only re-weights steps inside position/speed and
# is not a separable term).  Terms that are not in the composed loss simply do not appear.
BRAKING = ("brake_persist", "brake_persist_quantile", "asym_speed", "decel_weight",
           "propagation", "brake_wave", "brake_wave_residual", "brake_wave_spurious")


def event_windows(ds, pre_steps: int, warmup_ticks: int = 600) -> list[int]:
    """Indices of windows whose recorded ramp->lane-1 insertion falls ``pre_steps`` steps after the start.

    Positions come from the recorded arrays only (same rule as tools/r10_micro_smoke.py): lane 2 -> lane 1 for the
    same vehicle id between consecutive ticks.  Without an insertion the cut-in term (and its gradient) is zero.
    """
    picked, cache = [], {}
    for index, (ep, tick) in enumerate(ds.positions):
        epdir = Path(ds._loader.root) / ds._loader.entries[ep]["dir"]
        if not (epdir / "veh_kin.npy").is_file():
            continue
        if ep not in cache:
            kin = np.load(epdir / "veh_kin.npy", mmap_mode="r")
            vid = np.load(epdir / "vid.npy", mmap_mode="r")
            lane = np.where(kin[..., 1:4].sum(-1) > 0, kin[..., 1:4].argmax(-1), -1)
            ins = (lane[:-1] == 2) & (lane[1:] == 1) & (vid[:-1] == vid[1:]) & (vid[1:] >= 0)
            cache[ep] = [int(t) for t in np.nonzero(ins.any(-1))[0] + 1 if t > warmup_ticks]
        if any(0 <= e - tick - pre_steps <= 1 for e in cache[ep]):
            picked.append(index)
    return picked


def measure(args) -> dict:
    cfg = load_train_config(ROOT / args.config, [
        f"model.init_from={args.ckpt}", "runtime.device=cpu",
        f"runtime.threads={args.threads}", "loss.cutin_reaction.weight=1.0", *args.set])
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="r10_weight_", dir=output.parent))
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.runtime.seed)
    chunk = max(1, round(args.unroll_s / 0.5))
    if args.roots:
        datasets = [build_c4_datasets(
            root, horizon_steps=chunk, history_s=cfg.data.history_s,
            history_stride_s=cfg.data.history_stride_s, entries_v2=bool(cfg.data.entries_v2),
            stride_s=1.0 if args.pre_steps is not None else 10.0, junction_fix=cfg.data.junction_fix,
            command_known=True, merge_pivot_history=True)["train"] for root in args.roots]
        if args.pre_steps is not None:
            datasets = [Subset(ds, event_windows(ds, args.pre_steps)) for ds in datasets]
            print("insertion windows per root:", [len(d) for d in datasets], flush=True)
        physics = build_physics(cfg)
        inputs = TrainingInputs(ConcatDataset(datasets), None, build_model(cfg, physics), physics,
                                {f"local_train_{i}": str(Path(root).resolve())
                                 for i, root in enumerate(args.roots)})
    else:
        inputs = assemble(cfg, out)
    trainer = WmSTrainer(cfg, inputs.model, inputs.train, val_dataset=inputs.val, out_dir=out)
    trainer.model.train()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(trainer.dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=generator)
    rows = []
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.capture_loss_shares = True
        record = trainer.forward_backward(batch, unroll_steps=chunk,
                                          truncation_steps=args.truncation_steps)
        norms = record.get("loss_grad_norm") or {}
        rows.append({"norms": norms,
                     "cutin": {k: v for k, v in record.items() if k.startswith("cutin_reaction")}})
        print(f"batch {i}: " + json.dumps({k: round(v, 5) for k, v in norms.items()}), flush=True)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    names = sorted({k for r in rows for k in r["norms"]})
    gmean = {}
    for name in names:
        vals = [r["norms"][name] for r in rows if r["norms"].get(name, 0.0) > 0]
        gmean[name] = float(math.exp(np.mean(np.log(vals)))) if vals else 0.0
    braking = {k: v for k, v in gmean.items() if k in BRAKING}
    g_brake = sum(braking.values())
    # per-batch ratio as a robustness band
    ratios = []
    for r in rows:
        b = sum(v for k, v in r["norms"].items() if k in BRAKING)
        c = r["norms"].get("cutin_reaction", 0.0)
        if b > 0 and c > 0:
            ratios.append(b / c)
    g_cut = gmean.get("cutin_reaction", 0.0)
    if len(rows) != args.n_batches or not all(
            math.isfinite(v) and v > 0 for v in (g_brake, g_cut)):
        raise RuntimeError("gradient parity requires all requested batches and finite positive norms")
    return {"config": args.config, "ckpt": str(args.ckpt), "n_batches": len(rows),
            "unroll_s": args.unroll_s, "batch_size": args.batch_size,
            "method": "Sum of per-term geometric-mean accumulated weighted braking gradient norms / "
                      "geometric-mean accumulated cut-in gradient norm at weight 1; "
                      "real WmSTrainer.forward_backward, no optimizer steps",
            "overrides": args.set, "sampler_seed": args.seed,
            "model_seed": cfg.runtime.seed, "threads": cfg.runtime.threads,
            "truncation_steps": args.truncation_steps,
            "resolved_data": inputs.resolved, "init_report": trainer.init_report,
            "trainable_parameters": trainable,
            "geomean_grad_norm": gmean, "braking_terms": braking, "g_brake": g_brake,
            "g_cutin_at_weight_1": g_cut,
            "weight_geomean_ratio": (g_brake / g_cut) if g_cut > 0 else None,
            "per_batch_ratio": ratios,
            "per_batch_ratio_median": float(np.median(ratios)) if ratios else None,
            "per_batch_ratio_min_max": [float(min(ratios)), float(max(ratios))] if ratios else None,
            "rows": rows}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/c4_wm_v10round10.toml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--roots", nargs="+", help="Local TRAIN episode sources; uniform 10 s window grid")
    p.add_argument("--pre-steps", type=int, default=None,
                   help="with --roots: use only 1 s-stride windows whose recorded insertion is at this step")
    p.add_argument("--n-batches", type=int, default=6)
    p.add_argument("--unroll-s", type=float, default=60.0)
    p.add_argument("--truncation-steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    result = measure(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()
