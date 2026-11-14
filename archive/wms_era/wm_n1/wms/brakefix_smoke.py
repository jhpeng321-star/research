"""Coordinator-only bounded CPU smoke via the actual wms.cli train argv path.

Reads the authorized TRAIN keep/exclude metadata, selects at most two windows
per supplied pack without outcome-based selection, then performs one update per
curriculum stage. External pack reads occur only inside the existing trainer.
Use tools/brakefix_verify.py to capture full logs and enforce the memory limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import lpo
from .brakefix_audit import workspace_path
from .train import validate_train_source


def tiny_manifest(source: dict, packs: list[str], excluded: set, per_pack: int = 2) -> dict:
    """Deterministic metadata-only sample preserving exact structured identities."""
    if source.get("split", "TRAIN") != "TRAIN" or per_pack < 1:
        raise ValueError("positive budget and TRAIN window list required")
    selected = []
    for pack in packs:
        candidates = [row for row in source["windows"] if row.get("pack") == pack
                      and not lpo._excluded(row, excluded)]
        if not candidates:
            raise ValueError(f"no eligible TRAIN windows for {pack}")
        for row in candidates[:per_pack]:
            validate_train_source("/".join(str(row.get(key, "")) for key in
                                           ("pack", "batch", "chunk", "world", "arm", "key")))
            validate_train_source("TRAIN", row)
            if row.get("split", "TRAIN") != "TRAIN":
                raise ValueError("non-TRAIN row in keep list")
            selected.append(dict(row))
    return {**source, "windows": selected, "n_windows": len(selected),
            "list": "brakefix_metadata_only_smoke", "selection": "first eligible per pack"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Coordinator CPU TRAIN smoke (five updates)")
    p.add_argument("--pack", action="append", required=True)
    p.add_argument("--identities", action="append", required=True)
    p.add_argument("--overlay", action="append", default=[])
    p.add_argument("--outcomes")
    p.add_argument("--window-list", required=True)
    p.add_argument("--exclude-window-list", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = workspace_path(args.out, new=True)
    out.mkdir(exist_ok=False)
    keep = lpo.load_window_manifest(validate_train_source(args.window_list))
    tiny = tiny_manifest(keep, lpo.pack_names(args.pack),
                         lpo.exclude_runs(args.exclude_window_list))
    tiny_path = out / "windows_tiny.json"
    tiny_path.write_text(json.dumps(tiny, indent=2) + "\n", encoding="utf-8")
    train_argv = ["train"]
    for pack, identities in zip(args.pack, args.identities):
        train_argv.extend(["--pack", pack, "--identities", identities])
    if len(args.pack) != len(args.identities):
        raise ValueError("one identities path per pack required")
    for overlay in args.overlay:
        train_argv.extend(["--overlay", overlay])
    if args.outcomes:
        train_argv.extend(["--outcomes", args.outcomes])
    train_argv.extend([
        "--window-list", str(tiny_path), "--exclude-window-list", args.exclude_window_list,
        "--out", str(out / "run"), "--device", "cpu", "--threads", "1", "--num-workers", "0",
        "--batch-size", "1", "--sumo-fidelity", "--start-shift-prob", "0.75",
        "--truncation-steps", "20", "--rollout-mode", "hybrid", "--checkpoint-residual",
        "--asym-speed-loss", "2", "--decel-weight", "5", "--decel-threshold", "3",
        "--propagation-loss", "1", "--prop-window", "3", "--residual-cap-mps2", "1.5",
        "--residual-cap-neg-mps2", "3", "--loss-share-logging", "--stage-steps", "1,1,1,1,1"])
    stages = (10, 30, 60, 120, 300)
    for stage in stages:
        train_argv.extend(["--unroll-s", str(stage)])
    (out / "argv.json").write_text(json.dumps(train_argv, indent=2) + "\n", encoding="utf-8")
    package = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, package)
    from wms import cli, gate_s2, gate_stages
    rc = cli.main(train_argv)
    if rc:
        return rc
    log = out / "run" / "train_log.jsonl"
    gate_s2.check(log, stages=stages, steps_per_stage=1, required_terms=())
    result = gate_stages.check(log, stages=stages, terms=("position", "speed", "lane"))
    (out / "gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
