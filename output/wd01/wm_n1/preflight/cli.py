"""``python -m preflight`` CLI.

    python -m preflight <zero-residual|ctrl-parity|label-leak|overfit|e2e-smoke|heldout-dryrun|all>
    python -m preflight all --dry

Prints PASS/FAIL per gate and writes JSON reports under ``preflight/reports/``
(plus ``PREFLIGHT_SUMMARY.json`` for ``all``).  ``--dry`` only lists the gates
and the entry points they need; it imports no piece and runs nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import imports
from .registry import GATES, GATE_ORDER, describe
from .report import write_gate_report, write_summary
from .settings import DEFAULT_REPORTS_DIRNAME, WM_N1_DIR, Settings

GATE_CHOICES = GATE_ORDER + ("all",)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preflight",
        description="WM-N1 Idea A owner pre-flight gates G1-G6 (CPU, no SUMO)")
    p.add_argument("gate", choices=GATE_CHOICES,
                   help="which gate to run, or 'all'")
    p.add_argument("--dry", action="store_true",
                   help="list the gates and their entry points; run nothing")
    p.add_argument("--reports-dir", default=None,
                   help=f"report directory (default "
                        f"preflight/{DEFAULT_REPORTS_DIRNAME})")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default=None,
                   help="root containing the runs/ tree for the TRAIN-data "
                        "gates (default: the inventory's recorded data_root)")
    return p


def dry_listing() -> str:
    """The exact dry-run listing: 6 gates + the entry points they need."""
    lines = ["WM-N1 Idea A pre-flight gates (dry run; nothing executed)"]
    for row in describe():
        eps = ", ".join(row["entry_points"]) or "(none)"
        lines.append(f"  {row['id']} {row['gate']:<18} -> {eps}")
    lines.append(f"  ({len(GATE_ORDER)} gates)")
    return "\n".join(lines)


def run_gates(names, settings, resolve=imports.resolve):
    return [GATES[name].run(settings, resolve) for name in names]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry:
        print(dry_listing())
        return 0

    if args.data_root:
        os.environ["WM_N1_DATA_ROOT"] = str(args.data_root)

    reports_dir = Path(args.reports_dir) if args.reports_dir else \
        (WM_N1_DIR / "preflight" / DEFAULT_REPORTS_DIRNAME)
    settings = Settings(wm_n1_dir=WM_N1_DIR, reports_dir=reports_dir,
                        device=args.device)
    names = list(GATE_ORDER) if args.gate == "all" else [args.gate]
    results = run_gates(names, settings)
    for r in results:
        write_gate_report(settings.resolve_report_dir(), r)
        print(f"{r.to_json()['verdict']} {r.gate}: {r.detail}")
    if args.gate == "all":
        path = write_summary(settings.resolve_report_dir(), results, dry=False)
        print(f"[preflight] summary -> {path}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
