"""Audit whether calibrated paired data are ready for gate training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from counterfactual.readiness import evaluate_readiness


from experiments import REPO_ROOT

PROJECT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine label calibration and paired outcomes into a gate-readiness audit."
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--minimum-paired-cases", type=int, default=10)
    parser.add_argument("--minimum-off-labels", type=int, default=3)
    parser.add_argument("--minimum-active-labels", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_readiness.json",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    args = parse_args()
    calibration_path = args.calibration.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    input_paths = [path.resolve() for path in args.input]
    rows = [row for path in input_paths for row in _read_jsonl(path)]
    report = evaluate_readiness(
        calibration,
        rows,
        minimum_paired_cases=args.minimum_paired_cases,
        minimum_off_labels=args.minimum_off_labels,
        minimum_active_labels=args.minimum_active_labels,
    )
    report["calibration_report"] = str(calibration_path)
    report["paired_datasets"] = [str(path) for path in input_paths]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")
    return 0 if report["ready_to_train_deployable_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
