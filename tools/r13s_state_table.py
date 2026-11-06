"""R13S fitted init table of the state-conditional temporal emission mean (CPU, no training, no SUMO).

S-FACTORIAL's winning "T mean" arm (E:/agent-work/c4-factorial/FACTORIAL.md; scripts/fit_table.py) replaces the
learned temporal acceleration emission by its state-conditional mean: per-level cell means over (lane, v, gap,
closing, s, k) with backoff (>= 30 rows), fitted on the AS-IS rollouts of the checkpoint.  This tool captures, on the
learned HDV rows of free rollouts from real TRAIN windows (the init checkpoint with the run's init transforms, via the
trainer's init_from path; real unroll runner, no gradient), the features of ``wms.model.temporal_state_table`` and the
PRE-GATE emission, fits the table with the same bins/levels (``temporal_state_table.fit``) and writes it as .npz plus a
JSON summary (rows, per-lane emission mean/sd, in-sample R2 and held-out-by-window R2).

    python tools/r13s_state_table.py --config configs/c4_wm_v10round13f.toml --set model.temporal_accel_off=false
        --ckpt <init.pt> --roots <TRAIN roots> --out docs/r13s_state_table.npz
"""
from __future__ import annotations

import argparse
import json
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

from wms.cli.assemble import build_model, build_physics  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.data.c4_episodes import build_c4_datasets  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.physics.state import state_kin, state_present  # noqa: E402
from wms.model import temporal_state_table as tst  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402


class Capture:
    """Rows (lane, v, gap, closing, s, k, pre-gate emission, window id) of learned on-road HDV rows."""

    def __init__(self, model):
        self.model = model
        self.rows = []
        self.window = 0
        self._orig = model.forward

    def __call__(self, state, **kwargs):
        out = self._orig(state, **kwargs)
        with torch.no_grad():
            kin = state_kin(state).float()
            present = out["hdv_mask"].bool() | state.get("is_cav", torch.zeros_like(out["hdv_mask"])).bool()
            k = torch.as_tensor(kwargs.get("k_steps", 0), device=kin.device)
            feats = tst.state_features(kin, state_present(state, kin), k)
            on_road = kin[..., 1:4].sum(-1) > 0
            learned = out["hdv_mask"].bool() & on_road & present
            emission = out["temporal_accel_pregate_mps2"].float()
            window = torch.arange(kin.shape[0], device=kin.device).view(-1, 1).expand_as(emission) + self.window
            cols = [f.float() for f in feats] + [emission, window.float()]
            self.rows.append(torch.stack(cols, -1)[learned].double().numpy())
        return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/c4_wm_v10round13f.toml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--all-roles-as-train", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--n-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--unroll-s", type=float, default=60.0)
    p.add_argument("--stride-s", type=float, default=30.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    cfg = load_train_config(ROOT / args.config, [f"model.init_from={args.ckpt}", "runtime.device=cpu",
                                                 f"runtime.threads={args.threads}", *args.set])
    if cfg.model.temporal_accel_off or cfg.model.temporal_accel_lane_bias or cfg.model.temporal_accel_state_table:
        raise SystemExit("measure the DIRECT emission: run with temporal_accel_off / lane_bias false")
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.runtime.seed)
    chunk = max(1, round(args.unroll_s / 0.5))
    datasets = []
    for root in args.roots:
        manifest = json.loads((Path(root) / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
        role_of = {e["world"]: "train" for e in manifest["episodes"]} if args.all_roles_as_train else None
        datasets.append(build_c4_datasets(
            root, horizon_steps=chunk, history_s=cfg.data.history_s, history_stride_s=cfg.data.history_stride_s,
            entries_v2=bool(cfg.data.entries_v2), stride_s=args.stride_s, junction_fix=cfg.data.junction_fix,
            command_known=True, merge_pivot_history=True, **({"role_of": role_of} if role_of else {}))["train"])
    physics = build_physics(cfg)
    out_dir = Path(tempfile.mkdtemp(prefix="r13m_init_", dir=Path(args.out).resolve().parent))
    trainer = WmSTrainer(cfg, build_model(cfg, physics), ConcatDataset(datasets), val_dataset=None, out_dir=out_dir)
    model = trainer.model
    model.diag_emit_temporal = True
    acc = Capture(model)
    model.forward = acc
    model.eval()
    loader = DataLoader(trainer.dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            acc.window = i * args.batch_size
            trainer.forward_backward(batch, unroll_steps=chunk, backward=False, truncation_steps=chunk)
            print(f"batch {i}: rows {sum(len(r) for r in acc.rows)}", flush=True)
    data = np.concatenate(acc.rows)
    x, y, win = data[:, :6], data[:, 6], data[:, 7].astype(int)
    table = tst.fit(x, y)
    out_path = Path(args.out)
    np.savez(out_path, **table)

    def predict(tab, xx):
        b = tst.bin_index(*(torch.as_tensor(xx[:, i]) for i in range(6)))
        b = tuple(t.long() for t in b)
        means = [torch.as_tensor(tab[f"mean{lv}"]) for lv in range(len(tst.LEVELS))]
        counts = [torch.as_tensor(tab[f"n{lv}"]) for lv in range(len(tst.LEVELS))]
        return tst.lookup(means, counts, b).double().numpy()

    def r2(yy, pp):
        return float(1 - np.mean((yy - pp) ** 2) / np.var(yy))
    train = win % 2 == 0
    held = predict(tst.fit(x[train], y[train]), x[~train])
    lanes = {}
    for lane in (0, 1, 2):
        sel = x[:, 0] == lane
        if sel.any():
            lanes[str(lane)] = {"rows": int(sel.sum()), "mean": float(y[sel].mean()), "sd": float(y[sel].std())}
    summary = {"rule": "S-FACTORIAL T-mean arm: per-level cell means of the pre-gate temporal emission over (lane, v, "
                       "gap, closing, s, k), backoff >= 30 rows (wms.model.temporal_state_table), fitted on as-is "
                       "free rollouts of the init checkpoint (run init transforms applied)",
               "config": args.config, "overrides": args.set, "ckpt": str(args.ckpt), "roots": args.roots,
               "n_batches": args.n_batches, "batch_size": args.batch_size, "unroll_s": args.unroll_s,
               "sampler_seed": args.seed, "rows": int(len(y)), "lanes": lanes,
               "r2_in_sample": r2(y, predict(table, x)), "r2_held_out_by_window": r2(y[~train], held),
               "cells_used_level0": int((table["n0"] >= tst.MIN_ROWS).sum()),
               "npz": out_path.name, "npz_sha256": __import__("hashlib").sha256(out_path.read_bytes()).hexdigest()}
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rows", "lanes", "r2_in_sample", "r2_held_out_by_window",
                                              "cells_used_level0")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
