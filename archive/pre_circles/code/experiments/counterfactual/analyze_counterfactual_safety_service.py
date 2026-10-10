"""Audit paired counterfactual datasets with the frozen safety/service oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from counterfactual.safety_service_oracle import (
    SafetyServiceMargins,
    summarize_staged_safety_service,
)
from experiments.io import atomic_write_json as _atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("pooled inputs contain duplicate case IDs")
    return rows


def main() -> int:
    args = parse_args()
    rows = _read_jsonl([path.resolve() for path in args.input])
    report = summarize_staged_safety_service(rows, SafetyServiceMargins())
    report["input_paths"] = [str(path.resolve()) for path in args.input]
    _atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "independent_seed_count": report["independent_seed_count"],
                "engaged_case_count": report["engaged_case_count"],
                "safe_engagement_case_count": report["safe_engagement_case_count"],
                "unsafe_engagement_case_count": report["unsafe_engagement_case_count"],
                "selected_case_count": report["selected_case_count"],
                "selected_seed_count": report["selected_seed_count"],
                "model_fit_permitted": report["model_fit_permitted"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
