"""``wms train`` / ``wms preflight``: load a config, validate it, hand over to the trainer.

The heavy modules (data, model, trainer) are imported only when a run starts, so
``--help`` stays instant.  Assembly of the datasets and the model lives in
:mod:`wms.cli.assemble`; this module only sequences preflight, assembly and ``fit``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from ..config.load import load_train_config, write_resolved
from ..config.train import TrainConfig
from ..core.errors import ConfigError

MODES = ("preflight", "stage", "run")


def add_arguments(parser: argparse.ArgumentParser, *, with_mode: bool = True) -> None:
    """Options shared by ``train`` and ``preflight``."""
    parser.add_argument("--config", required=True, help="preset name or TOML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="A.B=C",
                        help="override a config value (repeatable), e.g. --set optim.lr=5e-4")
    parser.add_argument("--out", default=None, help="run directory (overrides runtime.out_dir)")
    if with_mode:
        parser.add_argument("--mode", choices=MODES, default="preflight",
                            help="preflight (default) | run; 'stage' is replaced by `wms fetch get`")


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Load the preset named by ``--config`` with ``--set`` overrides and ``--out`` applied."""
    overrides = list(args.overrides)
    if args.out:
        overrides.append(f"runtime.out_dir={json.dumps(args.out)}")
    return load_train_config(args.config, overrides)


def run(args: argparse.Namespace) -> int:
    """Entry point of ``wms train``."""
    return execute(config_from_args(args), args.mode)


def run_preflight(args: argparse.Namespace) -> int:
    """Entry point of ``wms preflight`` (``wms train --mode preflight``)."""
    return execute(config_from_args(args), "preflight")


def execute(cfg: TrainConfig, mode: str) -> int:
    """Validate ``cfg`` for training and perform ``mode``."""
    if mode == "stage":
        raise ConfigError("data staging is replaced by `python -m wms fetch get <name>`")
    cfg.validate_for_training()
    if mode == "preflight":
        return _preflight(cfg)
    return _train(cfg)


def _preflight(cfg: TrainConfig) -> int:
    """Config checks + every declared dataset resolves, exists and passes the restricted guard."""
    from ..train.preflight import preflight
    from .assemble import resolve_all

    paths, errors = resolve_all(cfg)
    out = Path(cfg.runtime.out_dir)
    report = preflight(cfg, max_hours=cfg.curriculum.max_hours or 10.0, out_dir=out,
                       inputs=list(paths.values()))
    report.failures.extend(f"unresolvable data {label}: {message}"
                           for label, message in errors.items())
    print(json.dumps({"status": "passed" if report.ok else "failed", "config": cfg.name,
                      "fingerprint": cfg.fingerprint(), "failures": report.failures,
                      "plan": report.plan, "data": {k: str(v) for k, v in paths.items()}},
                     indent=2, sort_keys=True))
    return 0 if report.ok else 2


def _fresh_out_dir(cfg: TrainConfig) -> Path:
    out = Path(cfg.runtime.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ConfigError(f"output directory {out} is not empty; choose a fresh one "
                          "(failed runs are preserved, never overwritten)")
    return out


def _train(cfg: TrainConfig) -> int:
    from ..train.trainer import WmSTrainer
    from .assemble import assemble

    out = _fresh_out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved(cfg, out)
    inputs = assemble(cfg, out)
    result = WmSTrainer(cfg, inputs.model, inputs.train, val_dataset=inputs.val,
                        out_dir=out).fit()
    summary = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
    if isinstance(summary, dict) and summary.get("selection", 0) is None:
        summary.pop("selection")  # SCHED off: RUN_SUMMARY keys unchanged
    # The full summary goes to a file; printing it would block any pipe-fed caller
    # once the text exceeds the OS pipe buffer (round-5 exit hang).
    (out / "RUN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"status": "done", "out": str(out),
                      "summary": str(out / "RUN_SUMMARY.json")}))
    return 0
