"""CLI for the C4 MAPPO baseline.

    python -m wms.control.mappo.train --config configs/c4_mappo.toml \
        --workers N --out DIR [--max-sumo-seconds X] [--smoke]

Partner modules and the SUMO env are resolved lazily from the dotted names in
the config (fake shortcuts: ``fake``/``fake_obs``/... in
``wms.control.mappo.rollout.FAKE_NAMES``), so this module never imports them at
load time.  ``--smoke`` merges the config's ``[smoke]`` overrides (fake env,
tiny budget) for a CPU-only pipeline check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wms.core.toml import loads

from .config_helpers import build_trainer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the C4 MAPPO baseline in SUMO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-sumo-seconds", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path | str) -> dict:
    return loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke and "smoke" in cfg:
        cfg = apply_smoke(cfg, cfg["smoke"])
    if args.max_sumo_seconds is not None:
        cfg.setdefault("rollout", {})["max_sumo_seconds"] = float(args.max_sumo_seconds)
    trainer = build_trainer(
        cfg,
        workers=args.workers,
        out_dir=args.out,
        config_path=str(args.config),
        smoke=bool(args.smoke),
    )
    result = trainer.run()
    print(result)
    return 0


def apply_smoke(cfg: dict, smoke: dict) -> dict:
    """Merge the config's [smoke] table into the base sections."""
    from .config_helpers import merge_override

    return merge_override(cfg, smoke)


if __name__ == "__main__":
    sys.exit(main())
