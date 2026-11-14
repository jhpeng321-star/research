"""Bundle queue CLI.

    python -m bundle --manifest MANIFEST.json --work-dir RUN \
        --approval APPROVAL.json [--stop-after timing-gate] [--device cuda]

Run with the default ``--stop-after timing-gate`` for the hard STOP before any
training; continue with ``--stop-after member-1`` (the M1-FIRST mode: train
Idea A member 1 alone, then STOP) or ``--stop-after members-2-5`` once the
owner APPROVAL.json is in place.  Members 2-5 additionally require the separate
owner ``APPROVAL_A_M2_5.json`` (``--a-m2-5-approval``) binding the member-1
checkpoint sha and the winning-config sha (``--winning-config``).  ``--tiny``
runs the real-member timing wrapper on a tiny CPU config (tests / smoke).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import queue
from .queue import STAGE_ORDER


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bundle")
    p.add_argument("--manifest", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--approval", required=True)
    p.add_argument("--stop-after", default="timing-gate",
                   choices=(None,) + STAGE_ORDER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--a-m2-5-approval", default=None,
                   help="separate owner APPROVAL_A_M2_5.json (required before "
                        "members 2-5; binds the member-1 checkpoint sha and the "
                        "winning-config sha)")
    p.add_argument("--winning-config", default=None,
                   help="selected WINNING_CONFIG.json bound by the A_M2_5 file")
    p.add_argument("--tiny", action="store_true",
                   help="tiny CPU real-member timing config (smoke)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    queue.TINY = bool(args.tiny)
    result = queue.bundle_queue(
        args.manifest, args.work_dir, args.approval,
        device=args.device, stop_after=args.stop_after,
        a_m2_5_approval=args.a_m2_5_approval,
        winning_config=args.winning_config)
    print(json.dumps(result, indent=1))
    print(f"[bundle] stopped after {result['stopped_after']!r} "
          f"(work_dir={result['work_dir']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
