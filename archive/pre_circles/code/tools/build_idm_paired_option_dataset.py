"""Build an immutable paired contextual-bandit dataset from SUMO branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idm.paired_option_dataset import records_from_roots, write_dataset


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
    records, manifest = records_from_roots(args.inputs, args.refs)
    payload = write_dataset(
        args.output.resolve(),
        records,
        manifest,
        train_seeds=args.train_seeds,
        validation_seeds=args.validation_seeds,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
