"""Generate a post-pilot actionability diagnosis without changing labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from counterfactual.actionability import summarize_actionability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.resolve().read_text(encoding="utf-8").splitlines()
        if line
    ]
    report = summarize_actionability(rows)
    report["input_path"] = str(args.input.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "dose_qualified_fraction": report["dose_qualified_fraction"],
                "safety_admissible_fraction": report["safety_admissible_fraction"],
                "selected_case_count": report["selected_case_count"],
                "selected_seed_count": report["selected_seed_count"],
                "by_decision_step": report["by_decision_step"],
                "refusal_reason_counts": report["refusal_reason_counts"],
            },
            indent=2,
        )
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
