"""CLI for the C4 MAPPO baseline (universal: penetration and demand sampled per episode).

    python -m wms.control.mappo.train --config configs/c4_mappo.toml --workers N --out DIR
        [--max-sumo-seconds X] [--iterations N] [--episodes-per-iteration E]
        [--horizon-s H] [--warmup-s W] [--seed S] [--update-threads T]
        [--set KEY=VALUE ...] [--resume] [--smoke]

Re-running with the same ``--out`` resumes automatically from the newest valid checkpoint
(``DIR/checkpoints/LATEST`` pointer, sha256-checked; truncated files skipped); ``--resume``
additionally REQUIRES one.  A ``DIR/STOP_REQUESTED`` file (spot termination watcher) stops the
run cleanly at the next episode / minibatch boundary: exit code 3, status
``stopped_by_request``; the last per-iteration checkpoint is the resume point.
``--smoke`` merges the config's ``[smoke]`` table (CPU fake env, tiny budget).  Workers are
``spawn`` processes (Windows and Linux), one SUMO each.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wms.core.toml import loads

from .config_helpers import build_trainer, merge_override

_FLAGS = {"iterations": int, "episodes_per_iteration": int, "horizon_s": float,
          "warmup_s": float, "seed": int, "update_threads": int}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the C4 MAPPO baseline in SUMO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-sumo-seconds", type=float, default=None)
    for name, kind in _FLAGS.items():
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=kind, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="any [mappo]/[rollout]/[lanes] key, TOML value syntax (repeatable)")
    parser.add_argument("--resume", action="store_true",
                        help="require an existing valid checkpoint in --out (resume is automatic)")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path | str) -> dict:
    return loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke and "smoke" in cfg:
        cfg = merge_override(cfg, cfg["smoke"])
    overrides = {k: getattr(args, k) for k in _FLAGS if getattr(args, k) is not None}
    if args.max_sumo_seconds is not None:
        overrides["max_sumo_seconds"] = float(args.max_sumo_seconds)
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key.strip()] = loads(f"v = {value.strip()}")["v"]
    cfg = merge_override(cfg, overrides)
    trainer = build_trainer(cfg, workers=args.workers, out_dir=args.out,
                            config_path=str(args.config), smoke=bool(args.smoke))
    if args.resume and not trainer._checkpoints():
        print(f"--resume: no checkpoint under {trainer.ckpt_dir}", file=sys.stderr)
        return 2
    result = trainer.run()
    print(json.dumps(result, default=float))
    if result["status"] == "stopped_by_request":
        return 3
    return 0 if result["status"] in ("completed", "budget_exhausted") else 1


if __name__ == "__main__":
    sys.exit(main())
