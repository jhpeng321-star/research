"""``wms eval``: score a checkpoint against the previous best and write the comparison.

The candidate is ``--checkpoint`` (logical name or path); the reference is ``--previous-best``
(default ``previous_best`` of the config).  Both are rolled out on the sets of the config and
compared with :func:`wms.eval.compare.compare`.
Outputs in ``--out`` (default ``out_dir`` of the config): ``result.json`` (all per-state results),
``config.resolved.json`` and ``<candidate>/COMPARISON.md|json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config.load import load_eval_config, write_resolved
from ..core.io import write_json_atomic

CANDIDATE = "candidate"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Options of ``wms eval``."""
    parser.add_argument("--config", required=True, help="preset name or TOML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="A.B=C",
                        help="override a config value (repeatable)")
    parser.add_argument("--checkpoint", required=True, help="logical checkpoint name or path")
    parser.add_argument("--previous-best", default=None,
                        help="logical name of the previous best checkpoint (default: config)")
    parser.add_argument("--out", default=None, help="result directory (overrides eval.out_dir)")


def run(args: argparse.Namespace) -> int:
    """Score the candidate and the previous best, compare, and write the report."""
    from ..eval.compare import markdown, write_comparison
    from ..eval.run import build_result

    overrides = list(args.overrides)
    if args.previous_best:
        overrides.append(f"previous_best={json.dumps(args.previous_best)}")
    if args.out:
        overrides.append(f"out_dir={json.dumps(args.out)}")
    cfg = load_eval_config(args.config, overrides)
    reference = cfg.models.get(cfg.previous_best, cfg.previous_best)   # name -> logical checkpoint
    result = build_result({CANDIDATE: args.checkpoint, cfg.previous_best: reference}, cfg.sets,
                          config=cfg)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved(cfg, out)
    write_json_atomic(out / "result.json", result)
    comparison = result["comparisons"][CANDIDATE]
    md_path, json_path = write_comparison(comparison, out / CANDIDATE)
    print(markdown(comparison))
    print(json.dumps({"status": "scored", "verdict": comparison["verdict"],
                      "comparison": str(json_path), "result": str(out / "result.json")}))
    return 0
