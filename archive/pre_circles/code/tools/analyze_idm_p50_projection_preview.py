"""Audit whether a p50 terminal-speed ceiling is distinct before treatment use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.io import atomic_write_json


SCHEMA = "idm_p50_projection_preview/1"


def _json_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
    return sorted(set(files))


def analyze(
    inputs: Iterable[Path],
    distinct_threshold_mps2: float = 0.05,
    reference_distinct_threshold_mps: float = 0.10,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in _json_files(inputs):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        decision = payload.get("decision")
        if not isinstance(decision, dict):
            continue
        for candidate in decision.get("candidates", []):
            previews = candidate.get("terminal_speed_previews") or {}
            if not all(name in previews for name in ("p30", "p50", "p70")):
                continue
            accelerations = {
                name: float(previews[name]["projected_acceleration_mps2"])
                for name in ("p30", "p50", "p70")
            }
            desired = {
                name: float(previews[name]["desired_acceleration_mps2"])
                for name in ("p30", "p50", "p70")
            }
            references = {
                name: float(previews[name]["reference_speed_mps"])
                for name in ("p30", "p50", "p70")
            }
            rows.append(
                {
                    "source": str(path),
                    "demand": payload.get("demand"),
                    "seed": payload.get("seed"),
                    "decision_step": decision.get("decision_step"),
                    "slot": candidate.get("slot"),
                    "vehicle_id": candidate.get("vehicle_id"),
                    "references_mps": references,
                    "desired_accelerations_mps2": desired,
                    "projected_accelerations_mps2": accelerations,
                    "p50_minus_p30_reference_mps": references["p50"] - references["p30"],
                    "p70_minus_p50_reference_mps": references["p70"] - references["p50"],
                    "p50_minus_p30_desired_mps2": desired["p50"] - desired["p30"],
                    "p70_minus_p50_desired_mps2": desired["p70"] - desired["p50"],
                    "p50_minus_p30_mps2": accelerations["p50"] - accelerations["p30"],
                    "p70_minus_p50_mps2": accelerations["p70"] - accelerations["p50"],
                    "reference_ordered": references["p30"] <= references["p50"] <= references["p70"],
                }
            )

    threshold = float(distinct_threshold_mps2)
    reference_threshold = float(reference_distinct_threshold_mps)
    p50_p30_reference = np.asarray(
        [abs(row["p50_minus_p30_reference_mps"]) for row in rows], dtype=float
    )
    p70_p50_reference = np.asarray(
        [abs(row["p70_minus_p50_reference_mps"]) for row in rows], dtype=float
    )
    p50_p30_desired = np.asarray(
        [abs(row["p50_minus_p30_desired_mps2"]) for row in rows], dtype=float
    )
    p70_p50_desired = np.asarray(
        [abs(row["p70_minus_p50_desired_mps2"]) for row in rows], dtype=float
    )
    p50_p30 = np.asarray([abs(row["p50_minus_p30_mps2"]) for row in rows], dtype=float)
    p70_p50 = np.asarray([abs(row["p70_minus_p50_mps2"]) for row in rows], dtype=float)

    def summarize(
        values: np.ndarray, median_key: str, distinct_threshold: float
    ) -> dict[str, Any]:
        if not values.size:
            return {median_key: None, "distinct_fraction": None}
        return {
            median_key: float(np.median(values)),
            "distinct_fraction": float(np.mean(values >= distinct_threshold)),
        }

    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "scientific_claim": "No treatment-outcome or training claim is supported by this preview.",
        "distinct_threshold_mps2": threshold,
        "reference_distinct_threshold_mps": reference_threshold,
        "candidate_count": len(rows),
        "reference_order_fraction": (
            float(np.mean([row["reference_ordered"] for row in rows])) if rows else None
        ),
        "reference_distinctness": {
            "p50_vs_p30": summarize(
                p50_p30_reference, "median_absolute_delta_mps", reference_threshold
            ),
            "p50_vs_p70": summarize(
                p70_p50_reference, "median_absolute_delta_mps", reference_threshold
            ),
        },
        "desired_command_distinctness": {
            "p50_vs_p30": summarize(
                p50_p30_desired, "median_absolute_delta_mps2", threshold
            ),
            "p50_vs_p70": summarize(
                p70_p50_desired, "median_absolute_delta_mps2", threshold
            ),
        },
        "p50_vs_p30": summarize(p50_p30, "median_absolute_delta_mps2", threshold),
        "p50_vs_p70": summarize(p70_p50, "median_absolute_delta_mps2", threshold),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distinct-threshold-mps2", type=float, default=0.05)
    parser.add_argument("--reference-distinct-threshold-mps", type=float, default=0.10)
    args = parser.parse_args()
    result = analyze(
        args.inputs,
        args.distinct_threshold_mps2,
        args.reference_distinct_threshold_mps,
    )
    output = args.output
    if output.suffix.lower() != ".json":
        output = output / "summary.json"
    atomic_write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
