"""Characterize fixed-p30 response and residual risk by realized decision state."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.io import atomic_write_json


SCHEMA = "idm_p30_operating_domain_diagnostic/1"
PATTERN_FIELDS = (
    "idm_severe_level",
    "idm_wave_energy_level",
    "idm_severe_short_ema",
    "idm_wave_short_ema",
    "idm_severe_trend",
    "idm_wave_energy_trend",
)


def _demand(case_id: str) -> int:
    match = re.match(r"m(\d+)\.", case_id)
    if match is None:
        raise ValueError(f"cannot infer demand from case_id {case_id!r}")
    return int(match.group(1))


def _finite(values: Iterable[object]) -> np.ndarray:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def _summary(values: Iterable[object]) -> dict[str, float | None]:
    array = _finite(values)
    if not array.size:
        return {"mean": None, "median": None, "p10": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
    }


def _load_rows(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root in sorted(set(path.resolve() for path in inputs)):
        payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            case_id = str(case["case_id"])
            seed = int(case["seed"])
            demand = _demand(case_id)
            discovery = json.loads(
                (root / "discoveries" / f"m{demand}.seed{seed}.json").read_text(
                    encoding="utf-8"
                )
            )
            decision = discovery.get("decision")
            if not isinstance(decision, dict):
                raise ValueError(f"missing decision for {case_id}")
            candidates = {
                int(candidate["slot"]): candidate
                for candidate in decision.get("candidates", [])
            }
            control = json.loads(
                (root / "branches" / f"{case_id}.abstain.json").read_text(
                    encoding="utf-8"
                )
            )
            for effect in case.get("effects", []):
                arm = str(effect["arm"])
                key = (case_id, arm)
                if key in seen:
                    raise ValueError(f"duplicate effect {key}")
                seen.add(key)
                branch = json.loads(
                    (root / "branches" / f"{case_id}.{arm}.json").read_text(
                        encoding="utf-8"
                    )
                )
                candidate = candidates[int(effect["candidate_slot"])]
                vehicle_id = str(candidate["vehicle_id"])
                observation = list(decision["candidate_observations"][vehicle_id])
                pattern = dict(decision.get("pattern_context") or {})
                constraints = dict(effect.get("constraints") or {})
                projection = [
                    float(value)
                    for value in branch.get("projection_survival_values", [])
                    if np.isfinite(float(value))
                ]
                rows.append(
                    {
                        "source": str(root),
                        "case_id": case_id,
                        "arm": arm,
                        "demand": demand,
                        "seed": seed,
                        "decision_index": int(decision.get("decision_index", 0)),
                        "decision_step": int(case["decision_step"]),
                        "classification": str(effect["classification"]),
                        "effect_reference_units": float(
                            effect["safety_composite_reference_units"]
                        ),
                        "constraints_pass": bool(effect["constraints_pass"]),
                        "constraints_excluding_dose_pass": all(
                            bool(value)
                            for name, value in constraints.items()
                            if name != "dose_at_least_minimum"
                        ),
                        "failed_constraints": sorted(
                            name for name, value in constraints.items() if not value
                        ),
                        "predicted_dose_s": float(candidate["predicted_dose_s"]),
                        "realized_dose_steps": int(effect["target_command_step_count"]),
                        "ego_speed_mps": float(observation[1]) * 35.0,
                        "terminal_reference_mps": float(
                            candidate["preview_reference_speed_mps"]
                        ),
                        "terminal_speed_drop_mps": max(
                            0.0,
                            float(observation[1]) * 35.0
                            - float(candidate["preview_reference_speed_mps"]),
                        ),
                        "projection_survival_values": projection,
                        "control_severe_sum": float(control["severe_braking_count_sum"]),
                        "treatment_severe_sum": float(
                            branch["severe_braking_count_sum"]
                        ),
                        "control_wave_sum": float(control["acceleration_energy_sum"]),
                        "treatment_wave_sum": float(branch["acceleration_energy_sum"]),
                        "control_collapse_max_steps": int(
                            control["collapse_max_consecutive_steps"]
                        ),
                        "treatment_collapse_max_steps": int(
                            branch["collapse_max_consecutive_steps"]
                        ),
                        "pattern_context": {
                            name: float(pattern.get(name, float("nan")))
                            for name in PATTERN_FIELDS
                        },
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["demand"],
            row["decision_index"],
            row["seed"],
            row["arm"],
        ),
    )


def _group(rows: list[dict[str, Any]], minimum_dose_steps: int, deadband: float) -> dict[str, Any]:
    adequate = [row for row in rows if row["realized_dose_steps"] >= minimum_dose_steps]
    qualified = [
        row
        for row in adequate
        if row["constraints_excluding_dose_pass"]
    ]
    qualified_harmful = [
        row for row in qualified if row["effect_reference_units"] < -deadband
    ]
    projection = [
        value for row in adequate for value in row["projection_survival_values"]
    ]
    pattern = {
        name: _summary(row["pattern_context"][name] for row in rows)
        for name in PATTERN_FIELDS
    }
    return {
        "opportunity_count": len(rows),
        "seed_count": len({row["seed"] for row in rows}),
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "adequate_dose_fraction": len(adequate) / len(rows) if rows else 0.0,
        "safety_service_admissible_excluding_dose_fraction": (
            sum(row["constraints_excluding_dose_pass"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "qualified_count": len(qualified),
        "qualified_harmful_fraction": (
            len(qualified_harmful) / len(qualified) if qualified else 0.0
        ),
        "qualified_response_fraction": (
            sum(abs(row["effect_reference_units"]) > deadband for row in qualified)
            / len(qualified)
            if qualified
            else 0.0
        ),
        "qualified_effect": _summary(
            row["effect_reference_units"] for row in qualified
        ),
        "predicted_dose_s": _summary(row["predicted_dose_s"] for row in rows),
        "terminal_speed_drop_mps": _summary(
            row["terminal_speed_drop_mps"] for row in rows
        ),
        "command_projection_survival": _summary(projection),
        "command_projection_survival_at_least_half_fraction": (
            sum(value >= 0.5 for value in projection) / len(projection)
            if projection
            else None
        ),
        "mean_severe_braking_delta": (
            float(
                np.mean(
                    [
                        row["treatment_severe_sum"] - row["control_severe_sum"]
                        for row in rows
                    ]
                )
            )
            if rows
            else None
        ),
        "mean_acceleration_energy_delta": (
            float(
                np.mean(
                    [
                        row["treatment_wave_sum"] - row["control_wave_sum"]
                        for row in rows
                    ]
                )
            )
            if rows
            else None
        ),
        "collapse_max_improved_count": sum(
            row["treatment_collapse_max_steps"]
            < row["control_collapse_max_steps"]
            for row in rows
        ),
        "collapse_max_worsened_count": sum(
            row["treatment_collapse_max_steps"]
            > row["control_collapse_max_steps"]
            for row in rows
        ),
        "failed_constraint_counts": dict(
            Counter(
                constraint
                for row in rows
                for constraint in row["failed_constraints"]
            )
        ),
        "realized_predecision_pattern": pattern,
    }


def analyze(
    inputs: Iterable[Path], minimum_dose_steps: int = 16, deadband: float = 0.02
) -> dict[str, Any]:
    rows = _load_rows(inputs)
    demands = sorted({row["demand"] for row in rows})
    return {
        "schema": SCHEMA,
        "development_only": True,
        "scientific_claim": (
            "Configured demand is a reporting stratum, not a universal deployment "
            "threshold. Residual risk must be interpreted with realized state fields."
        ),
        "minimum_dose_steps": int(minimum_dose_steps),
        "deadband": float(deadband),
        "overall": _group(rows, minimum_dose_steps, deadband),
        "by_demand": {
            str(demand): _group(
                [row for row in rows if row["demand"] == demand],
                minimum_dose_steps,
                deadband,
            )
            for demand in demands
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-dose-steps", type=int, default=16)
    parser.add_argument("--deadband", type=float, default=0.02)
    args = parser.parse_args()
    result = analyze(args.inputs, args.minimum_dose_steps, args.deadband)
    output = args.output
    if output.suffix.lower() != ".json":
        output = output / "summary.json"
    atomic_write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
