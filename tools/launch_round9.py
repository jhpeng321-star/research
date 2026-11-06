"""Round-9 draft launcher: workspace-local round-8 stage300 initialization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    path = args.init_from.resolve()
    if not path.is_relative_to(root):
        parser.error("--init-from must be inside this worktree")
    if not path.is_file():
        parser.error(f"round-8 stage300 checkpoint is missing: {path}")
    import torch
    from wms.cli.train import execute
    from wms.config.load import load_train_config

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "wm_s/1" or payload.get("config", {}).get("merge_decision") != "secure_gap":
        parser.error("--init-from must be a round-8 secure_gap wm_s/1 checkpoint")
    if float(payload.get("stage_s", -1)) != 300.0:
        parser.error("--init-from must record stage_s=300")
    overrides = [*args.overrides, "model.init_from=" + json.dumps(str(path))]
    cfg = load_train_config(root / "configs/c4_wm_v10round9.toml", overrides)
    return execute(cfg, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
