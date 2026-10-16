"""``python -m wms <command>``: train | preflight | eval | score | fetch.

Old invocations (``python -m wms.cli train --pack ... --stage-steps ...``) are still
understood: a ``train``/``dry-run`` command without ``--config`` is parsed with the
legacy flag table and run through the same code path.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from ..config.legacy_flags import COMMANDS as LEGACY_COMMANDS
from ..config.legacy_flags import parse_legacy_argv
from ..core.errors import NotYetImplemented, WmsError
from . import eval as eval_cmd
from . import fetch, score, train

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_NOT_IMPLEMENTED = 3


def build_parser() -> argparse.ArgumentParser:
    """Parser of the new-style commands (``fetch`` is forwarded before parsing)."""
    parser = argparse.ArgumentParser(prog="python -m wms", description="WM-S training and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    train.add_arguments(sub.add_parser("train", help="train from a config (default mode: preflight)"))
    train.add_arguments(sub.add_parser("preflight", help="check a training config without training"),
                        with_mode=False)
    eval_cmd.add_arguments(sub.add_parser("eval", help="evaluate a checkpoint"))
    score.add_arguments(sub.add_parser("score", help="build the result JSON for an eval config"))
    sub.add_parser("fetch", help="list|status|verify|get datasets (forwards to tools/fetch_data.py)")
    return parser


def _is_legacy_train(argv: Sequence[str]) -> bool:
    return bool(argv) and argv[0] in LEGACY_COMMANDS and "--config" not in argv


def _dispatch(argv: Sequence[str]) -> int:
    if argv and argv[0] == "fetch":
        return fetch.run(argv[1:])
    if _is_legacy_train(argv):
        _command, cfg = parse_legacy_argv(argv)  # dry-run differs only by its 2-step default
        return train.execute(cfg, "run")
    args = build_parser().parse_args(list(argv))
    handlers = {"train": train.run, "preflight": train.run_preflight,
                "eval": eval_cmd.run, "score": score.run}
    return handlers[args.command](args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI; refusals print one JSON line and exit with status 2."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        return _dispatch(arguments)
    except NotYetImplemented as exc:
        print(json.dumps({"status": "not_implemented", "error": str(exc)}))
        return EXIT_NOT_IMPLEMENTED
    except (WmsError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
