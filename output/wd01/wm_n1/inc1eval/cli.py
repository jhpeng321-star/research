"""``python -m inc1eval`` CLI: model-lock writer and zero-shot assertion.

There is deliberately NO ``metrics`` subcommand: the section-5 metrics run is
reached only through :func:`inc1eval.metrics.run_inc1_metrics`, which enforces the
one-shot gate and the verified lock.  A CLI that could bypass those guards
would defeat registration section 6.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import zeroshot
from .model_lock import build_lock, write_lock


def _member(value: str) -> tuple:
    if "=" not in value:
        raise argparse.ArgumentTypeError("member must be name=path")
    name, path = value.split("=", 1)
    return name, path


def _model_lock(args) -> int:
    payload = build_lock(
        members=dict(args.member), pack_manifest=args.pack_manifest,
        code_commit=args.code_commit, v2g_params=args.v2g_params,
        extra=({"pack_dir": args.pack_dir} if args.pack_dir else None))
    path = write_lock(args.out_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[inc1eval.model-lock] wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inc1eval", description="WM-N1-INC1 tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    z = sub.add_parser("zeroshot",
                       help="event_mask + obstacle/setStop zero-shot assertion")
    z.add_argument("--pack", required=True)
    z.add_argument("--source-root", required=True)

    m = sub.add_parser("model-lock", help="write INC1_MODEL_LOCK.json")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--member", action="append", type=_member, default=[],
                   required=True, metavar="NAME=PATH")
    m.add_argument("--pack-manifest", required=True)
    m.add_argument("--v2g-params", required=True)
    m.add_argument("--code-commit", required=True)
    m.add_argument("--pack-dir", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "zeroshot":
        return zeroshot.main(["--pack", args.pack,
                              "--source-root", args.source_root])
    if args.cmd == "model-lock":
        return _model_lock(args)
    raise SystemExit(f"unknown command {args.cmd!r}")   # pragma: no cover


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
