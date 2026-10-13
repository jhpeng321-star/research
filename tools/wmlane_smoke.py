"""WMLANE CPU smoke: the tools/c4_smoke.py two-update dry run for any config + overrides.

Usage (workspace root as CWD, PYTHONPATH=src):
    python tools/wmlane_smoke.py OUT_DIR [--config configs/c4_wm_v2_lane.toml]
                                 [--set a.b=c ...] [--data-root E:/research-dense]

``--data-root`` points the logical datasets at another checkout's data (a git worktree has
no ``data_cache``): every registry entry whose relative path exists below the root gets a
``WMS_DATA_<NAME>`` variable unless one is already set.  Writes TRAIN_RECEIPT (dry run) and
SMOKE_SUMMARY.json into OUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def point_data(data_root: Path) -> list[str]:
    """Set ``WMS_DATA_*`` for registry entries present below ``data_root``."""
    from wms.core import toml
    from wms.data.registry import env_var_name

    entries: dict[str, str] = {}
    reg_json = ROOT / "tools" / "data_registry.json"
    if reg_json.is_file():
        for name, entry in json.loads(reg_json.read_text(encoding="utf-8"))["datasets"].items():
            entries[name] = entry["local_path"]
    reg_toml = ROOT / "configs" / "data_registry.toml"
    if reg_toml.is_file():
        for name, entry in toml.load(reg_toml).get("datasets", {}).items():
            if "path" in entry and not Path(entry["path"]).is_absolute():
                entries[name] = entry["path"]
    pointed = []
    for name, rel in entries.items():
        var = env_var_name(name)
        target = data_root / rel
        if var not in os.environ and target.exists() and not (ROOT / rel).exists():
            os.environ[var] = str(target)
            pointed.append(name)
    return pointed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--config", default="configs/c4_wm_v2_lane.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output directory {out_dir} is not empty")
    pointed = point_data(Path(args.data_root)) if args.data_root else []
    from wms.cli.assemble import assemble
    from wms.config.load import load_train_config
    from wms.train.preflight import dry_run

    cfg = load_train_config(args.config, [f"runtime.out_dir={out_dir.as_posix()}",
                                          "runtime.device=cpu", "runtime.threads=4",
                                          *args.overrides])
    inputs = assemble(cfg, out_dir)
    receipt = dry_run(cfg, inputs.model, inputs.train, out_dir=out_dir)
    model = inputs.model
    summary = {
        "config": args.config, "overrides": args.overrides,
        "smoke": {"status": receipt.get("status"),
                  "optimizer_steps": receipt.get("optimizer_steps"),
                  "unroll_s": receipt.get("unroll_s"), "batch_size": receipt.get("batch_size"),
                  "loss_first": receipt["updates"][0].get("loss"),
                  "loss_second": receipt["updates"][1].get("loss"),
                  "lane_event": [u.get("lane_event") for u in receipt["updates"]],
                  "finite": [u.get("finite") for u in receipt["updates"]],
                  "profiles": receipt.get("profiles")},
        "switches": {"lane_model": getattr(model, "lane_model", None),
                     "driver_update": getattr(model, "driver_update", None),
                     "lane_hazard": getattr(model, "lane_hazard", None) is not None,
                     "driver_updater": getattr(model, "driver_updater", None) is not None,
                     "history_s": getattr(model, "history_s", None),
                     "lane_event_weight": cfg.loss.lane_event.weight,
                     "driver_reveal_prob": cfg.model.driver_reveal_prob},
        "windows": {"n_train": len(inputs.train),
                    "n_val": 0 if inputs.val is None else len(inputs.val)},
        "physics": {"lane_rule": inputs.physics.lane_rule},
        "data_pointed": pointed,
    }
    (out_dir / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True,
                                                           default=str), encoding="utf-8")
    print(json.dumps(summary["smoke"], indent=1, default=str))
    print(json.dumps(summary["switches"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
