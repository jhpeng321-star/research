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
    if getattr(inputs.train, "pairs", None) is not None:
        # PAIRLOSS: two updates on one PAIRED batch (pair groups of the paired C4 source)
        summary["paired"] = paired_smoke(cfg, inputs.model, inputs.train)
    receipt_sources = inputs.train.receipt.get("sources", [])
    summary["windows"]["by_source"] = {s["name"]: {"n": s["n"], "share": s["share"]}
                                       for s in receipt_sources}
    (out_dir / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                                encoding="utf-8")
    print(json.dumps(summary["smoke"], indent=1))
    print(json.dumps(summary["physics"], indent=1))
    return 0


def paired_smoke(cfg, model, dataset) -> dict:
    """Two CPU updates (10 s unroll) on one paired batch; loss terms and paired value share."""
    import numpy as np

    from wms.data.collate import collate
    from wms.train.trainer import WmSTrainer

    flat = dict(cfg.to_legacy_flat())
    flat.update(device="cpu", steps_per_epoch=2, stage_steps=None, unroll_curriculum_s=[10.0])
    for per_stage in ("truncation_steps_per_stage", "lr_per_stage"):
        if flat.get(per_stage) is not None:
            flat[per_stage] = list(flat[per_stage])[:1]
    trainer = WmSTrainer(type(cfg).from_legacy_flat(flat), model, dataset,
                         out_dir=Path(cfg.runtime.out_dir))
    batch = collate(dataset.paired_batch(np.random.default_rng(0)))
    keys = ("loss", "position", "speed", "lane", "agg_field", "agg_reward", "pair_pos",
            "pair_speed", "pair_agg", "paired_value_share", "aggregate_value_share",
            "paired_batch", "paired_arms", "finite")
    updates = [trainer.train_batch(batch, unroll_steps=20) for _ in range(2)]
    out = {"rows": int(batch["states"].shape[0]),
           "groups": sorted(set(batch["pair_group"].tolist())),
           "horizon_steps": int(batch["states"].shape[1]) - 1,
           "updates": [{k: u.get(k) for k in keys} for u in updates],
           "source": dataset.pairs.receipt}
    print(json.dumps({"paired": out["updates"]}, indent=1))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
