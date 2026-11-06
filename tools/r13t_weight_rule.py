"""R13T tail-term weight rule: gradient-norm parity with the existing braking terms (as R10, docs/r10_weight_rule.json).

Rule (no hand-set number): with the round-13T config and its trainable set, measure on real TRAIN windows the
accumulated gradient norm of every loss term (``loss_grad_norm`` of the real unroll runner; the tail terms at
weight 1, the braking terms at their configured weights) and set

    accel_weight = (G_brake / 2) / G_accel_tail,   jerk_weight = (G_brake / 2) / G_jerk_tail,
    G_brake = G(brake_persist) + G(brake_persist_quantile),

each G the geometric mean over the measured batches: together the two tail terms get the gradient norm of
the onset-only braking terms they complement (all rows vs recorded onsets), split equally between
acceleration and |jerk|.  Numbers are rounded to 4 significant digits.  The tool trains nothing.

    python tools/r13t_weight_rule.py --ckpt <init.pt> --table docs/r13t_tail_table.json --roots <TRAIN roots> \
        --out docs/r13t_weight_rule.json [--allow-provisional-table] [--set key=value ...]

``--allow-provisional-table`` admits a table built from another split role (local offline measurement only;
the training loss itself always refuses it).
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

from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from wms.cli.assemble import TrainingInputs, build_model, build_physics  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.data.c4_episodes import build_c4_datasets  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.train.losses import accel_tail as tail_mod  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402

BRAKING = ("brake_persist", "brake_persist_quantile")
TAIL = ("accel_tail", "jerk_tail")


def _gmean(values):
    values = [v for v in values if v > 0]
    return float(math.exp(np.mean(np.log(values)))) if values else 0.0


def measure(args) -> dict:
    table = str(Path(args.table).resolve())
    if args.allow_provisional_table:
        original = tail_mod._load_table
        tail_mod._load_table = lambda path, require: original(path, False)
    cfg = load_train_config(ROOT / args.config, [
        f"model.init_from={args.ckpt}", "runtime.device=cpu", f"runtime.threads={args.threads}",
        f"loss.accel_tail.table={table}", "loss.accel_tail.weight=1.0", "loss.accel_tail.jerk_weight=1.0",
        *args.set])
    out = Path(tempfile.mkdtemp(prefix="r13t_weight_", dir=Path(args.out).resolve().parent))
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.runtime.seed)
    chunk = max(1, round(args.unroll_s / 0.5))
    datasets = []
    for root in args.roots:
        manifest = json.loads((Path(root) / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
        role_of = {e["world"]: "train" for e in manifest["episodes"]} if args.all_roles_as_train else None
        datasets.append(build_c4_datasets(
            root, horizon_steps=chunk, history_s=cfg.data.history_s,
            history_stride_s=cfg.data.history_stride_s, entries_v2=bool(cfg.data.entries_v2),
            stride_s=args.stride_s, junction_fix=cfg.data.junction_fix, command_known=True,
            merge_pivot_history=True, **({"role_of": role_of} if role_of else {}))["train"])
    physics = build_physics(cfg)
    inputs = TrainingInputs(ConcatDataset(datasets), None, build_model(cfg, physics), physics,
                            {f"local_{i}": str(Path(r).resolve()) for i, r in enumerate(args.roots)})
    trainer = WmSTrainer(cfg, inputs.model, inputs.train, val_dataset=None, out_dir=out)
    trainer.model.train()
    loader = DataLoader(trainer.dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    rows = []
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.capture_loss_shares = True
        record = trainer.forward_backward(batch, unroll_steps=chunk, truncation_steps=args.truncation_steps)
        norms = record.get("loss_grad_norm") or {}
        rows.append({"norms": norms, "loss": float(record.get("loss", float("nan")))})
        print(f"batch {i}: " + json.dumps({k: round(v, 5) for k, v in norms.items()}), flush=True)
    names = sorted({k for r in rows for k in r["norms"]})
    gmean = {name: _gmean([r["norms"].get(name, 0.0) for r in rows]) for name in names}
    g_brake = sum(gmean.get(k, 0.0) for k in BRAKING)
    g_accel, g_jerk = gmean.get("accel_tail", 0.0), gmean.get("jerk_tail", 0.0)
    if len(rows) != args.n_batches or not all(math.isfinite(v) and v > 0 for v in (g_brake, g_accel, g_jerk)):
        raise RuntimeError(f"parity needs every batch and finite positive norms: {g_brake}, {g_accel}, {g_jerk}")
    per_batch = []
    for r in rows:
        b = sum(r["norms"].get(k, 0.0) for k in BRAKING)
        a, j = r["norms"].get("accel_tail", 0.0), r["norms"].get("jerk_tail", 0.0)
        if b > 0 and a > 0 and j > 0:
            per_batch.append({"accel": b / 2 / a, "jerk": b / 2 / j})
    return {"rule": "accel_weight = (G_brake/2)/G_accel_tail, jerk_weight = (G_brake/2)/G_jerk_tail; G = geometric "
                    "mean accumulated gradient norm (real WmSTrainer.forward_backward, no optimizer step), braking "
                    "terms at their configured weights, tail terms at weight 1",
            "config": args.config, "ckpt": str(args.ckpt), "table": table, "roots": args.roots,
            "overrides": args.set, "n_batches": len(rows), "batch_size": args.batch_size,
            "unroll_s": args.unroll_s, "truncation_steps": args.truncation_steps, "sampler_seed": args.seed,
            "init_report": trainer.init_report, "geomean_grad_norm": gmean, "g_brake": g_brake,
            "g_accel_tail_w1": g_accel, "g_jerk_tail_w1": g_jerk,
            "accel_weight": float(f"{g_brake / 2 / g_accel:.4g}"), "jerk_weight": float(f"{g_brake / 2 / g_jerk:.4g}"),
            "per_batch_weights": per_batch, "rows": rows}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/c4_wm_v10round13t.toml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--all-roles-as-train", action="store_true",
                   help="window every local episode (local subsets that only hold one split role)")
    p.add_argument("--allow-provisional-table", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--n-batches", type=int, default=4)
    p.add_argument("--unroll-s", type=float, default=20.0)
    p.add_argument("--stride-s", type=float, default=10.0)
    p.add_argument("--truncation-steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    result = measure(args)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "init_report")}, indent=1, default=str))


if __name__ == "__main__":
    main()
