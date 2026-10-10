"""Analyze oracle treatment headroom in a paired counterfactual dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from counterfactual.oracle import analyze_oracle


from experiments import REPO_ROOT

PROJECT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure safety-first oracle headroom in paired SUMO cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[PROJECT / "runs" / "probe" / "counterfactual_pilot.jsonl"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_oracle_report.json",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on {path}:{line_number}: {error}") from error
    return rows


def main() -> int:
    args = parse_args()
    input_paths = [path.resolve() for path in args.input]
    rows = [row for input_path in input_paths for row in _read_jsonl(input_path)]
    report = analyze_oracle(rows)
    report["datasets"] = [str(input_path) for input_path in input_paths]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
