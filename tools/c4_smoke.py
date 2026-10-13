"""C4 2-step CPU smoke: assemble the configs/c4_wm.toml data mix and run the
existing two-update dry run (wms.train.preflight.dry_run) on it.

Usage (env as in the preflight; workspace root as CWD):
    python tools/c4_smoke.py [out_dir] [config]

``config`` defaults to configs/c4_wm.toml (e.g. configs/c4_wm_v2_loss.toml for the v2 loss).

Writes the dry-run TRAIN_RECEIPT plus a trimmed assemble receipt (kept windows
per pack / switching filter census) into the output directory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/active/c4_wm_smoke")
    config = sys.argv[2] if len(sys.argv) > 2 else "configs/c4_wm.toml"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output directory {out_dir} is not empty")
    from wms.cli.assemble import assemble
    from wms.config.load import load_train_config
    from wms.train.preflight import dry_run

    cfg = load_train_config(config,
                            [f"runtime.out_dir={out_dir.as_posix()}", "runtime.device=cpu",
                             "runtime.threads=4"])
    inputs = assemble(cfg, out_dir)
    receipt = dry_run(cfg, inputs.model, inputs.train, out_dir=out_dir)

    def trim(node, depth=0):
        if isinstance(node, dict):
            return {k: trim(v, depth + 1) for k, v in node.items() if depth < 6}
        if isinstance(node, (list, tuple)):
            return f"<{len(node)} entries>"
        return node

    summary = {
        "smoke": {"config": config, "status": receipt.get("status"),
                  "optimizer_steps": receipt.get("optimizer_steps"),
                  "unroll_s": receipt.get("unroll_s"),
                  "batch_size": receipt.get("batch_size"),
                  "loss_first": receipt["updates"][0].get("loss"),
                  "loss_second": receipt["updates"][1].get("loss"),
                  "finite": [u.get("finite") for u in receipt["updates"]],
                  "profiles": receipt.get("profiles")},
        "windows": {"n_train": len(inputs.train),
                    "n_val": 0 if inputs.val is None else len(inputs.val),
                    "train_receipt": trim(inputs.train.receipt)},
        "physics": {"lane_rule": inputs.physics.lane_rule,
                    "lane_gap_rule": inputs.physics.lane_gap_rule,
                    "gap_stopped_relax": getattr(inputs.physics, "gap_stopped_relax", False),
                    "sumo_fidelity": inputs.physics.sumo_fidelity,
                    "merge_cal": inputs.physics.merge_cal},
        "resolved": inputs.resolved,
    }
    (out_dir / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                                encoding="utf-8")
    print(json.dumps(summary["smoke"], indent=1))
    print(json.dumps(summary["physics"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
