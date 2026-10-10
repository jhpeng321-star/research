"""Compare simple timing/penetration rules with the safety/service oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from counterfactual.safety_service_oracle import evaluate_safety_service_baselines
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
    paths = [path.resolve() for path in args.input]
    report = evaluate_safety_service_baselines(_read_jsonl(paths))
    report["input_paths"] = [str(path) for path in paths]
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report["baselines"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
