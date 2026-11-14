"""Joined P30-versus-v2-pulse analysis for the frozen six-seed mechanism pilot.

The fixed-P30 arm is stored in the paired-option grid source and the v2 pulse
arm in the mechanism-pilot output.  This analyzer joins only rows that share
the exact seed, decision step and target vehicle identity; it never compares
unmatched rows.  It is read-only and does not rerun the simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.io import atomic_write_json

SCHEMA = "idm_rear_aware_feedback_v2_pilot_join/1"
P30_ARM = "candidate1_speed_ceiling"
V2_ARM = "candidate1_rear_aware_feedback_pulse"
DEADBAND = 0.02
PROBE_STEPS = 5


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cases(summary: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    """Index cases by (seed, decision_step)."""
    return {
        (int(row["seed"]), int(row["decision_step"])): row
        for row in summary.get("cases", [])
    }


def _arm_effect(
    case: Mapping[str, Any], arm: str
) -> Mapping[str, Any] | None:
    return next((row for row in case.get("effects", []) if row.get("arm") == arm), None)


def _branch_index(
    branch_paths: Iterable[Path], arms: tuple[str, ...]
) -> dict[tuple[int, int, str, str], Mapping[str, Any]]:
    """Index branches by (seed, decision_step, target, arm)."""
    index: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    for path in branch_paths:
        row = _load(path)
        if row.get("arm") not in arms:
            continue
        key = (
            int(row["seed"]),
            int(row["decision_step"]),
            str(row["target_vehicle_id"]),
            str(row["arm"]),
        )
        index[key] = row
    return index


def join_pilot_and_p30(
    *,
    pilot_summary: Mapping[str, Any],
    pilot_discoveries: Iterable[Mapping[str, Any]],
    pilot_branch_paths: Iterable[Path],
    source_summary: Mapping[str, Any],
    source_branch_paths: Iterable[Path],
) -> dict[str, Any]:
    pilot_cases = _cases(pilot_summary)
    source_cases = _cases(source_summary)
    pilot_branches = _branch_index(pilot_branch_paths, (V2_ARM,))
    source_branches = _branch_index(source_branch_paths, (P30_ARM,))

    matched: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for key, pilot_case in sorted(pilot_cases.items()):
        seed, decision_step = key
        source_case = source_cases.get(key)
        if source_case is None:
            missing.append({"seed": seed, "decision_step": decision_step, "reason": "no_source_case"})
            continue
        pilot_effect = _arm_effect(pilot_case, V2_ARM)
        p30_effect = _arm_effect(source_case, P30_ARM)
        if pilot_effect is None or p30_effect is None:
            excluded.append(
                {
                    "seed": seed,
                    "decision_step": decision_step,
                    "reason": "missing_effect_arm",
                    "pilot_arm_present": pilot_effect is not None,
                    "p30_arm_present": p30_effect is not None,
                }
            )
            continue
        pilot_branch = next(
            (
                row
                for (s, d, t, a), row in pilot_branches.items()
                if (s, d) == (seed, decision_step) and a == V2_ARM
            ),
            None,
        )
        source_branch = next(
            (
                row
                for (s, d, t, a), row in source_branches.items()
                if (s, d) == (seed, decision_step) and a == P30_ARM
            ),
            None,
        )
        identity = {
            "seed": seed,
            "decision_step": decision_step,
            "target_match": (
                pilot_branch is not None
                and source_branch is not None
                and str(pilot_branch["target_vehicle_id"])
                == str(source_branch["target_vehicle_id"])
            ),
        }
        if pilot_branch is None or source_branch is None or not identity["target_match"]:
            excluded.append(
                {
                    **identity,
                    "reason": "branch_identity",
                    "pilot_branch": pilot_branch is not None,
                    "source_branch": source_branch is not None,
                }
            )
            continue
        matched.append(
            {
                **identity,
                "target_vehicle_id": str(pilot_branch["target_vehicle_id"]),
                "p30": _effect_row(p30_effect, source_branch),
                "v2": _effect_row(pilot_effect, pilot_branch),
            }
        )

    discovery_identity = {
        str(row.get("seed")): dict(
            (row.get("decision") or {}).get("matched_decision_source") or {}
        )
        for row in pilot_discoveries
        if row.get("decision") is not None
    }
    report_reasons: dict[str, list[str]] = {}
    for row in matched:
        seed = str(row["seed"])
        reasons = [
            str(release.get("reason"))
            for release in row["v2"]["release_reasons"]
            if release.get("reason")
        ]
        reasons.extend(
            str(abort.get("reason"))
            for abort in row["v2"]["abort_reasons"]
            if abort.get("reason")
        )
        report_reasons.setdefault(seed, []).extend(reasons)

    joined_count = len(matched)
    p30_harmful = sum(1 for row in matched if row["p30"]["classification"] == "harmful")
    v2_harmful = sum(1 for row in matched if row["v2"]["classification"] == "harmful")
    helpful_seeds = sorted(
        {row["seed"] for row in matched if row["v2"]["classification"] == "helpful"}
    )
    nonbeneficial_seeds = sorted(
        {
            row["seed"]
            for row in matched
            if row["v2"]["classification"] in {"neutral", "harmful"}
        }
    )
    probe_complete = [
        row["seed"] for row in matched if row["v2"]["realized_active_steps"] >= PROBE_STEPS
    ]
    early_release_explained = [
        row["seed"]
        for row in matched
        if row["v2"]["realized_active_steps"] < PROBE_STEPS
        and bool(row["v2"]["release_reasons"])
    ]
    trivial_dose_only = [
        row["seed"]
        for row in matched
        if row["v2"]["classification"] == "helpful"
        and row["v2"]["realized_active_steps"] < PROBE_STEPS
    ]
    v2_release_breakdown: dict[str, int] = {}
    for reasons in report_reasons.values():
        for reason in reasons:
            v2_release_breakdown[reason] = v2_release_breakdown.get(reason, 0) + 1

    return {
        "schema": SCHEMA,
        "matched_case_count": joined_count,
        "excluded_count": len(excluded),
        "missing_source_case_count": len(missing),
        "joined_seeds": sorted(row["seed"] for row in matched),
        "matched": matched,
        "excluded": excluded,
        "missing_source_cases": missing,
        "no_decision_seeds": sorted(
            int(row.get("seed"))
            for row in pilot_discoveries
            if row.get("decision") is None
        ),
        "discovery_matched_identity": discovery_identity,
        "gates": {
            "p30_and_v2_share_exact_identity": joined_count == len(pilot_cases)
            and not excluded,
            "v2_fewer_harmful_than_p30": v2_harmful < p30_harmful,
            "v2_benefit_beyond_deadband_in_at_least_two_seeds": len(helpful_seeds) >= 2,
            "at_least_two_nonbeneficial_seeds": len(nonbeneficial_seeds) >= 2,
            "v2_harmful_count": v2_harmful,
            "p30_harmful_count": p30_harmful,
            "v2_helpful_seeds": helpful_seeds,
            "v2_nonbeneficial_seeds": nonbeneficial_seeds,
            "probe_survived_seeds": probe_complete,
            "early_releases_with_reason": early_release_explained,
            "helpful_with_trivial_dose": trivial_dose_only,
            "release_reason_counts": v2_release_breakdown,
        },
    }


def _effect_row(effect: Mapping[str, Any], branch: Mapping[str, Any]) -> dict[str, Any]:
    feedback = branch.get("treatment_dose") or {}
    return {
        "arm": effect.get("arm"),
        "mode": effect.get("mode"),
        "classification": effect.get("classification"),
        "safety_composite_reference_units": effect.get(
            "safety_composite_reference_units"
        ),
        "constraints": effect.get("constraints"),
        "constraints_pass": effect.get("constraints_pass"),
        "target_command_step_count": effect.get("target_command_step_count"),
        "realized_active_steps": effect.get("target_command_step_count"),
        "treatment_dose": feedback,
        "release_reasons": list(branch.get("feedback_release_rows") or []),
        "abort_reasons": list(branch.get("treatment_safety_abort_rows") or []),
        "completed_horizon_steps": branch.get("completed_horizon_steps"),
        "low_ttc_exposure_step_delta": effect.get("low_ttc_exposure_step_delta"),
        "delay_rate_delta": effect.get("delay_rate_delta"),
        "arrival_delta": effect.get("arrival_delta"),
        "merge_discharge_count_delta": effect.get("merge_discharge_count_delta"),
        "treatment_safety_abort_count": effect.get("treatment_safety_abort_count"),
        "projection_survival_values": effect.get("projection_survival_values"),
        "paired_projection_survival": effect.get("paired_projection_survival"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--p30-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot_summary = _load(args.pilot / "summary.json")
    source_summary = _load(args.p30_source / "summary.json")
    pilot_discoveries = [
        _load(path) for path in sorted((args.pilot / "discoveries").glob("*.json"))
    ]
    report = join_pilot_and_p30(
        pilot_summary=pilot_summary,
        pilot_discoveries=pilot_discoveries,
        pilot_branch_paths=sorted((args.pilot / "branches").glob("*.json")),
        source_summary=source_summary,
        source_branch_paths=sorted((args.p30_source / "branches").glob("*.json")),
    )
    atomic_write_json(args.output, report)
    print(json.dumps(
        {
            "matched_case_count": report["matched_case_count"],
            "excluded_count": report["excluded_count"],
            "no_decision_seeds": report["no_decision_seeds"],
            "gates": report["gates"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
