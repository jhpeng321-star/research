"""Build the state-aligned RL timing/candidate option library.

Runs the frozen paired protocol validation (identity, effect reproduction,
constraint reproduction via idm.paired_option_dataset.records_from_roots) on
the timing/candidate sweep roots, then groups single-action records into
per-state rows with abstain/candidate-1/candidate-2 outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from idm.paired_option_dataset import records_from_roots
from idm.rl_option_library import rl_rows_from_raw_records, write_rl_library


def _seeds(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seeds", type=_seeds, required=True)
    parser.add_argument("--validation-seeds", type=_seeds, required=True)
    args = parser.parse_args()
    raw_records, raw_manifest = records_from_roots(args.inputs, args.refs)
    rows = rl_rows_from_raw_records(raw_records, raw_manifest)
    counts = Counter(
        (round(row.penetration, 2), row.demand, row.recurrence) for row in rows
    )
    outcome_counts = Counter()
    for row in rows:
        for action in ("candidate_1", "candidate_2"):
            outcome_counts[f"{action}:{row.actions[action].classification}"] += 1
    cores = {
        "raw_dataset_schema": raw_manifest.get("schema"),
        "reward_contract": raw_manifest.get("reward_contract"),
        "reward_refs_path": raw_manifest.get("reward_refs_path"),
        "reward_refs_sha256": raw_manifest.get("reward_refs_sha256"),
        "record_count_per_cell": {
            f"{p:.2f}/m{d}/r{r}": count for (p, d, r), count in sorted(counts.items())
        },
        "outcome_counts": dict(outcome_counts),
        "sources": raw_manifest.get("sources"),
    }
    manifest = write_rl_library(
        args.output.resolve(),
        rows,
        cores,
        train_seeds=args.train_seeds,
        validation_seeds=args.validation_seeds,
    )
    print(
        json.dumps(
            {
                "rows": manifest["record_count"],
                "train": len(args.train_seeds),
                "validation": len(args.validation_seeds),
                "outcome_counts": cores["outcome_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
