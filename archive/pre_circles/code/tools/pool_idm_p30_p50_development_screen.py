"""Pool predeclared matched P30/P50 recurrence analyses for the retention gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.io import atomic_write_json


SCHEMA = "idm_p30_p50_development_pool/1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eligible_cases(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in summary.get("cases", ())
        if row.get("common_qualified")
        and all(bool(value) for value in row["identity"].values())
        and row.get("same_target_vehicle")
    ]


def _metrics(cases: list[dict[str, Any]], arm: str, deadband: float) -> dict[str, Any]:
    effects = [row[f"{arm}_effect"] for row in cases]
    values = np.asarray(
        [float(row["safety_composite_reference_units"]) for row in effects],
        dtype=float,
    )
    return {
        "qualified_case_count": len(effects),
        "qualified_harmful_count": sum(
            row["classification"] == "harmful" for row in effects
        ),
        "qualified_harmful_fraction": (
            sum(row["classification"] == "harmful" for row in effects) / len(effects)
            if effects
            else None
        ),
        "qualified_effect_mean": float(np.mean(values)) if values.size else None,
        "qualified_effect_p10": (
            float(np.percentile(values, 10.0)) if values.size else None
        ),
        "helpful_seeds": sorted(
            {
                int(case["seed"])
                for case in cases
                if case[f"{arm}_effect"]["classification"] == "helpful"
            }
        ),
        "nonbeneficial_seeds": sorted(
            {
                int(case["seed"])
                for case in cases
                if case[f"{arm}_effect"]["classification"] != "helpful"
            }
        ),
        "effect_beyond_replay_count": (
            int(np.sum(np.abs(values) > deadband)) if values.size else 0
        ),
        "effect_beyond_replay_fraction": (
            float(np.mean(np.abs(values) > deadband)) if values.size else 0.0
        ),
    }


def pool(paths: list[Path]) -> dict[str, Any]:
    summaries = [_read(path) for path in paths]
    if not summaries:
        raise ValueError("at least one matched analysis is required")
    deadbands = [float(summary.get("frozen_deadband", 0.02)) for summary in summaries]
    if len(set(deadbands)) != 1:
        raise ValueError(f"matched analyses use different replay deadbands: {deadbands}")
    deadband = deadbands[0]
    cases = []
    for recurrence, (path, summary) in enumerate(zip(paths, summaries), start=1):
        for row in _eligible_cases(summary):
            cases.append({**row, "recurrence": recurrence, "analysis_path": str(path)})
    p30, p50 = _metrics(cases, "p30", deadband), _metrics(cases, "p50", deadband)
    gates = {
        "first_onset_integrity_pass": bool(summaries[0]["p50_integrity_pass"]),
        "has_common_qualified_cases": bool(cases),
        "p50_harmful_fraction_lower_than_p30": bool(cases)
        and p50["qualified_harmful_fraction"] < p30["qualified_harmful_fraction"],
        "p50_p10_higher_than_p30": bool(cases)
        and p50["qualified_effect_p10"] > p30["qualified_effect_p10"],
        "p50_mean_no_more_than_0p02_below_p30": bool(cases)
        and p50["qualified_effect_mean"] >= p30["qualified_effect_mean"] - 0.02,
        "p50_helpful_in_at_least_three_seeds": len(p50["helpful_seeds"]) >= 3,
        "p50_nonbeneficial_in_at_least_three_seeds": (
            len(p50["nonbeneficial_seeds"]) >= 3
        ),
        "p50_effect_beyond_replay_fraction_at_least_30pct": (
            p50["effect_beyond_replay_fraction"] >= 0.30
        ),
    }
    return {
        "schema": SCHEMA,
        "development_only": True,
        "analysis_paths": [str(path) for path in paths],
        "frozen_deadband": deadband,
        "common_qualified_case_count": len(cases),
        "p30_metrics": p30,
        "p50_metrics": p50,
        "retention_gates": gates,
        "p50_retained_for_broader_development": all(gates.values()),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.analyses]
    payload = pool(paths)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    print(
        json.dumps(
            {
                "common_qualified_case_count": payload["common_qualified_case_count"],
                "p30_metrics": payload["p30_metrics"],
                "p50_metrics": payload["p50_metrics"],
                "retention_gates": payload["retention_gates"],
                "p50_retained_for_broader_development": payload[
                    "p50_retained_for_broader_development"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
