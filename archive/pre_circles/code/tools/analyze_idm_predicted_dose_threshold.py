"""Post-hoc sensitivity of paired IDM outcomes to predicted-dose admission."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.io import atomic_write_json


SCHEMA = "idm_predicted_dose_threshold_diagnostic/1"


def _directories(inputs: Iterable[Path]) -> list[Path]:
    directories: list[Path] = []
    for path in inputs:
        if path.is_dir() and (path / "summary.json").exists():
            directories.append(path)
        elif path.is_dir():
            directories.extend(
                parent
                for summary in path.rglob("summary.json")
                for parent in [summary.parent]
                if (parent / "discoveries").is_dir()
            )
    return sorted(set(directories))


def _demand(case_id: str) -> int:
    match = re.match(r"m(\d+)\.", case_id)
    if match is None:
        raise ValueError(f"cannot infer demand from case_id {case_id!r}")
    return int(match.group(1))


def _rows(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for directory in _directories(inputs):
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for case in summary.get("cases", []):
            case_id = str(case["case_id"])
            seed = int(case["seed"])
            demand = _demand(case_id)
            discovery_path = directory / "discoveries" / f"m{demand}.seed{seed}.json"
            discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
            decision = discovery.get("decision")
            if not isinstance(decision, dict):
                raise ValueError(f"missing decision in {discovery_path}")
            candidates = {
                int(candidate["slot"]): candidate
                for candidate in decision.get("candidates", [])
            }
            for effect in case.get("effects", []):
                slot = int(effect["candidate_slot"])
                key = (case_id, slot)
                if key in seen:
                    raise ValueError(f"duplicate paired effect {key}")
                seen.add(key)
                candidate = candidates.get(slot)
                if candidate is None or candidate.get("predicted_dose_s") is None:
                    raise ValueError(f"missing predicted dose for {key}")
                constraints = dict(effect.get("constraints") or {})
                constraints_without_dose = {
                    name: bool(value)
                    for name, value in constraints.items()
                    if name != "dose_at_least_minimum"
                }
                rows.append(
                    {
                        "source": str(directory),
                        "case_id": case_id,
                        "demand": demand,
                        "seed": seed,
                        "decision_index": int(decision.get("decision_index", 0)),
                        "decision_step": int(case["decision_step"]),
                        "candidate_slot": slot,
                        "predicted_dose_s": float(candidate["predicted_dose_s"]),
                        "realized_dose_steps": int(effect["target_command_step_count"]),
                        "classification": str(effect["classification"]),
                        "effect_reference_units": float(
                            effect["safety_composite_reference_units"]
                        ),
                        "constraints_pass": bool(effect["constraints_pass"]),
                        "constraints_excluding_dose_pass": all(
                            constraints_without_dose.values()
                        ),
                        "failed_constraints": sorted(
                            name for name, value in constraints.items() if not value
                        ),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["demand"],
            row["decision_index"],
            row["seed"],
            row["candidate_slot"],
        ),
    )


def _effect_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p10": None, "median": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
    }


def analyze(
    inputs: Iterable[Path],
    thresholds_s: Iterable[float] = (16.0, 20.0),
    adequate_dose_steps: int = 16,
    requested_dose_steps: int = 20,
    deadband: float = 0.02,
) -> dict[str, Any]:
    rows = _rows(inputs)
    threshold_rows: list[dict[str, Any]] = []
    for threshold in sorted(set(float(value) for value in thresholds_s)):
        admitted = [row for row in rows if row["predicted_dose_s"] >= threshold]
        excluded = [row for row in rows if row["predicted_dose_s"] < threshold]
        qualified = [row for row in admitted if row["constraints_pass"]]
        qualified_harmful = [
            row
            for row in qualified
            if row["effect_reference_units"] < -float(deadband)
        ]
        all_harmful = [row for row in admitted if row["classification"] == "harmful"]
        threshold_rows.append(
            {
                "threshold_s": threshold,
                "admitted_count": len(admitted),
                "opportunity_coverage_fraction": len(admitted) / len(rows) if rows else 0.0,
                "adequate_realized_dose_fraction": (
                    sum(row["realized_dose_steps"] >= adequate_dose_steps for row in admitted)
                    / len(admitted)
                    if admitted
                    else 0.0
                ),
                "full_realized_dose_fraction": (
                    sum(row["realized_dose_steps"] >= requested_dose_steps for row in admitted)
                    / len(admitted)
                    if admitted
                    else 0.0
                ),
                "safety_service_admissible_excluding_dose_fraction": (
                    sum(row["constraints_excluding_dose_pass"] for row in admitted)
                    / len(admitted)
                    if admitted
                    else 0.0
                ),
                "contract_qualified_count": len(qualified),
                "contract_qualified_fraction": (
                    len(qualified) / len(admitted) if admitted else 0.0
                ),
                "all_harmful_count": len(all_harmful),
                "all_harmful_fraction": (
                    len(all_harmful) / len(admitted) if admitted else 0.0
                ),
                "qualified_harmful_count": len(qualified_harmful),
                "qualified_harmful_fraction": (
                    len(qualified_harmful) / len(qualified) if qualified else 0.0
                ),
                "qualified_nonneutral_count": sum(
                    abs(row["effect_reference_units"]) > float(deadband)
                    for row in qualified
                ),
                "qualified_effect": _effect_summary(
                    [row["effect_reference_units"] for row in qualified]
                ),
                "classification_counts": dict(
                    Counter(row["classification"] for row in admitted)
                ),
                "failed_constraint_counts": dict(
                    Counter(
                        constraint
                        for row in admitted
                        for constraint in row["failed_constraints"]
                    )
                ),
                "excluded_classification_counts": dict(
                    Counter(row["classification"] for row in excluded)
                ),
                "excluded_case_ids": [row["case_id"] for row in excluded],
            }
        )
    return {
        "schema": SCHEMA,
        "post_hoc_diagnostic_only": True,
        "scientific_claim": (
            "Threshold sensitivity on previously opened outcomes does not validate "
            "an admission policy or authorize training."
        ),
        "opportunity_count": len(rows),
        "adequate_dose_steps": int(adequate_dose_steps),
        "requested_dose_steps": int(requested_dose_steps),
        "deadband": float(deadband),
        "thresholds": threshold_rows,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds-s", nargs="+", type=float, default=(16.0, 20.0))
    parser.add_argument("--adequate-dose-steps", type=int, default=16)
    parser.add_argument("--requested-dose-steps", type=int, default=20)
    parser.add_argument("--deadband", type=float, default=0.02)
    args = parser.parse_args()
    result = analyze(
        args.inputs,
        args.thresholds_s,
        args.adequate_dose_steps,
        args.requested_dose_steps,
        args.deadband,
    )
    output = args.output
    if output.suffix.lower() != ".json":
        output = output / "summary.json"
    atomic_write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
