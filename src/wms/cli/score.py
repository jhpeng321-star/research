"""``wms score``: build the result JSON for the models and sets of an eval config.

Every model listed under ``[models]`` of the config (plus ``previous_best``) is scored on every
set; ``result.json`` holds the per-state results and the comparison of each model with the
previous best.  ``--set models.NAME=CHECKPOINT`` adds or replaces a model from the command line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config.load import load_eval_config, write_resolved
from ..core.io import write_json_atomic


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Options of ``wms score``."""
    parser.add_argument("--config", required=True, help="preset name or TOML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="A.B=C",
                        help="override a config value (repeatable)")
    parser.add_argument("--out", default=None, help="result directory (overrides eval.out_dir)")


def run(args: argparse.Namespace) -> int:
    """Score every configured model on every configured set and write ``result.json``."""
    from ..eval.run import build_result

    overrides = list(args.overrides)
    if args.out:
        overrides.append(f"out_dir={json.dumps(args.out)}")
    cfg = load_eval_config(args.config, overrides)
    result = build_result(cfg.models, cfg.sets, config=cfg)
    out = Path(cfg.out_dir)
    write_resolved(cfg, out)
    write_json_atomic(out / "result.json", result)
    print(json.dumps({"status": "scored", "result": str(out / "result.json")}))
    return 0
