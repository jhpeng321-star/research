"""C4 Dreamer-loop CPU smoke: assemble configs/c4_wm_refresh.toml with a converted episode
source and run the two-update dry run (wms.train.preflight.dry_run) on a batch that holds
BOTH original pack windows and collected C4 windows.

Usage (workspace root as CWD, PYTHONPATH=src, WMS_DATA_* pointing the registry names at a
data cache, WMS_DATA_CKPT_C4_WM at the C4 WM checkpoint or a stand-in):

    python tools/c4_loop_smoke.py --c4-source <converted dir> --out <new dir>
        [--keep-switching] [--n-original 2] [--n-c4 2]

``--keep-switching`` keeps the switching direct-mix source of c4_wm.toml; it is dropped by
default because that source's guard reads the zero-shot held-out-world list (the collected
windows are mixed identically either way; the three-source mix is unit-tested).
Writes SMOKE_SUMMARY.json + the dry-run TRAIN_RECEIPT.json into ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4-source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/c4_wm_refresh.toml")
    ap.add_argument("--keep-switching", action="store_true")
    ap.add_argument("--n-original", type=int, default=2)
    ap.add_argument("--n-c4", type=int, default=2)
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output directory {out_dir} is not empty")

    from torch.utils.data import Subset

    from wms.cli.assemble import assemble
    from wms.config.load import load_train_config
    from wms.config.train import TrainConfig
    from wms.data.action_timeline import CONTROLLERS
    from wms.data.collate import collate
    from wms.train.preflight import dry_run

    cfg = load_train_config(args.config, [
        f"runtime.out_dir={out_dir.as_posix()}", "runtime.device=cpu", "runtime.threads=4",
        f"data.c4_episodes.path={Path(args.c4_source).resolve().as_posix()}"])
    if not args.keep_switching:
        tree = cfg.to_dict()
        tree["data"].pop("switching")
        cfg = TrainConfig.from_dict(tree)
    inputs = assemble(cfg, out_dir)
    train = inputs.train
    names = [s.name for s in train.sources]
    c4_k = names.index("c4_episodes")
    orig = [train.offsets[0] + i for i in range(args.n_original)]
    c4 = [train.offsets[c4_k] + i * max(1, len(train.sources[c4_k].dataset) // max(1, args.n_c4))
          for i in range(args.n_c4)]
    indices = orig + c4
    subset = Subset(train, indices)
    subset.receipt = train.receipt
    records = [train[i] for i in indices]
    batch = collate(records)
    direct = CONTROLLERS.index("direct")
    per_record = []
    for rec in records:
        ctrl = np.asarray(rec["action_timeline"]["controller"])
        cav0 = np.asarray(rec["is_cav"][0], bool)
        per_record.append({
            "mix_source": rec["mix_source"], "meta_world": (rec.get("meta") or {}).get("world"),
            "meta_arm": (rec.get("meta") or {}).get("arm"),
            "n_present_t0": int(np.asarray(rec["present"][0]).sum()),
            "n_cav_t0": int(cav0.sum()),
            "direct_cav_steps": int((ctrl == direct).sum()),
            "direct_on_cav_slots_only": bool(not (ctrl[:, ~cav0] == direct).any()),
            "cav_slots_with_direct": int((ctrl == direct).any(0).sum()),
            "max_abs_issued_accel": float(np.abs(rec["action_timeline"]["acceleration_mps2"]).max()),
            "synthetic_timeline": bool(rec.get("action_timeline_synthetic", False))})
    receipt = dry_run(cfg, inputs.model, subset, out_dir=out_dir)
    summary = {
        "config": args.config, "switching_source": bool(args.keep_switching),
        "init_from": cfg.model.init_from,
        "init_from_env": os.environ.get("WMS_DATA_CKPT_C4_WM"),
        "dry_run": {"status": receipt.get("status"), "optimizer_steps": receipt.get("optimizer_steps"),
                    "batch_size": receipt.get("batch_size"),
                    "loss": [u.get("loss") for u in receipt["updates"]],
                    "finite": [u.get("finite") for u in receipt["updates"]],
                    "profiles": receipt.get("profiles")},
        "batch": {"indices": indices, "keys": sorted(batch),
                  "states": list(batch["states"].shape),
                  "action_timeline_controller": list(batch["action_timeline"]["controller"].shape),
                  "records": per_record},
        "mix": {"sources": [{"name": s.name, "n": len(s.dataset), "share": round(sh, 4)}
                            for s, sh in zip(train.sources, train.shares)],
                "n_val": 0 if inputs.val is None else len(inputs.val),
                "val_sources": [] if inputs.val is None else
                [{"name": s.name, "n": len(s.dataset)} for s in inputs.val.sources],
                "stage_sampling": train.receipt.get("stage_sampling"),
                "c4_receipt": {k: v for k, v in train.receipt["c4_episodes"].items()
                               if k not in ("train", "val")}},
        "resolved": inputs.resolved,
    }
    (out_dir / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True,
                                                           default=str), encoding="utf-8")
    print(json.dumps({"dry_run": summary["dry_run"], "records": per_record,
                      "mix": summary["mix"]["sources"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
