"""Descriptive, no-model screen for the owner-approved escalation estimand."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from counterfactual.actionability import dose_qualified
from counterfactual.idm_predictor_contract import (
    FORBIDDEN_FEATURE_TOKENS,
    OBSERVATION_V2_FEATURES,
    feature_contract_manifest,
)
from experiments import REPO_ROOT
from experiments.io import atomic_write_json


LEGACY_ROOT = REPO_ROOT / "runs" / "probe" / "counterfactual_escalation_v2_headroom_v2"
V5_2_ROOT = REPO_ROOT / "runs" / "eval" / "idm_wall_v5_2_prospective_p30_s4301_4312"
CAPTURE_ROOT = REPO_ROOT / "runs" / "probe" / "idm_observation_v2_capture_v5_2_completed"
STAGE1_ROOT = REPO_ROOT / "runs" / "probe" / "idm_wall_v5_3_stage1_replay"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_observation_v2_screen"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _legacy_rows() -> list[dict[str, Any]]:
    return _jsonl(LEGACY_ROOT / "cases.jsonl")


def _v5_2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((V5_2_ROOT / "seeds").glob("seed*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["cases"])
    return rows


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _standardized_difference(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return 0.0
    pooled = positive + negative
    center = fmean(pooled)
    variance = fmean((value - center) ** 2 for value in pooled)
    return (fmean(positive) - fmean(negative)) / math.sqrt(variance) if variance > 1e-12 else 0.0


def _feature_screen(rows: list[dict[str, Any]], label_name: str) -> dict[str, Any]:
    feature_names = sorted(
        set.intersection(*(set(row["features"]) for row in rows)) if rows else set()
    )
    numeric_names = [
        name
        for name in feature_names
        if all(_finite_number(row["features"].get(name)) is not None for row in rows)
        and not any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    global_rows = []
    for name in numeric_names:
        positive = [float(row["features"][name]) for row in rows if row[label_name]]
        negative = [float(row["features"][name]) for row in rows if not row[label_name]]
        effect = _standardized_difference(positive, negative)
        loo = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            subset = [row for row in rows if int(row["seed"]) != seed]
            pos = [float(row["features"][name]) for row in subset if row[label_name]]
            neg = [float(row["features"][name]) for row in subset if not row[label_name]]
            if pos and neg:
                loo.append(_standardized_difference(pos, neg))
        sign = 0 if effect == 0.0 else (1 if effect > 0.0 else -1)
        global_rows.append(
            {
                "feature": name,
                "standardized_difference": effect,
                "abs_standardized_difference": abs(effect),
                "positive_mean": fmean(positive) if positive else None,
                "negative_mean": fmean(negative) if negative else None,
                "leave_one_seed_out_count": len(loo),
                "leave_one_seed_out_min": min(loo) if loo else None,
                "leave_one_seed_out_max": max(loo) if loo else None,
                "leave_one_seed_out_sign_stability": (
                    sum((1 if value > 0 else -1 if value < 0 else 0) == sign for value in loo)
                    / len(loo)
                    if loo
                    else None
                ),
            }
        )
    global_rows.sort(key=lambda row: (-row["abs_standardized_difference"], row["feature"]))
    return {
        "case_count": len(rows),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "positive_case_count": sum(bool(row[label_name]) for row in rows),
        "positive_seed_count": len({int(row["seed"]) for row in rows if row[label_name]}),
        "negative_case_count": sum(not bool(row[label_name]) for row in rows),
        "negative_seed_count": len({int(row["seed"]) for row in rows if not row[label_name]}),
        "numeric_feature_count": len(numeric_names),
        "top": global_rows[:15],
    }


def _fold_audit(rows: Iterable[dict[str, Any]], folds: int = 3) -> dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in rows})
    assignment = {seed: index % folds for index, seed in enumerate(seeds)}
    groups: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        groups[int(row["seed"])].add(assignment[int(row["seed"])])
    return {
        "fold_count": folds,
        "seed_to_fold": {str(seed): fold for seed, fold in assignment.items()},
        "seed_spans_multiple_folds": any(len(values) != 1 for values in groups.values()),
        "passed": all(len(values) == 1 for values in groups.values()),
    }


def build_report() -> dict[str, Any]:
    legacy = _legacy_rows()
    v5_2_all = _v5_2_rows()
    completed = [row for row in v5_2_all if dose_qualified(row)]
    capture = _jsonl(CAPTURE_ROOT / "rows.jsonl")
    capture_by_key = {(int(row["seed"]), int(row["decision_step"])): row for row in capture}
    enriched = []
    for row in completed:
        copy = dict(row)
        copy["features"] = dict(row["features"])
        captured = capture_by_key.get((int(row["seed"]), int(row["decision_step"])))
        if captured and captured["identity"]["passed"]:
            copy["features"].update(captured["features"])
        oracle = row["safety_service_oracle"]
        copy["safe_label"] = bool(oracle["safety_admissible"])
        copy["benefit_label"] = bool(oracle["meaningful_benefit"])
        copy["joint_y_label"] = bool(
            oracle["safety_admissible"]
            and oracle["traffic_noninferior"]
            and oracle["meaningful_benefit"]
        )
        enriched.append(copy)

    legacy_engaged = []
    for row in legacy:
        oracle = row.get("safety_service_oracle") or row.get("oracle") or {}
        if not oracle.get("wall_engaged"):
            continue
        copy = dict(row)
        copy["safe_label"] = bool(oracle.get("safety_admissible"))
        copy["benefit_label"] = bool(oracle.get("meaningful_benefit"))
        legacy_engaged.append(copy)

    d1 = [row for row in v5_2_all if bool((row.get("timing_scan") or {}).get("admitted")) and dose_qualified(row)]
    d0 = [row for row in v5_2_all if row not in d1]
    y1 = [row for row in enriched if row["joint_y_label"]]
    y0 = [row for row in enriched if not row["joint_y_label"]]
    legacy_digests = [str(row.get("decision_state_digest")) for row in legacy]
    v5_2_keys = [(int(row["seed"]), int(row["decision_step"])) for row in v5_2_all]
    v5_2_summary = json.loads((V5_2_ROOT / "summary.json").read_text(encoding="utf-8"))
    stage1_summary = (
        json.loads((STAGE1_ROOT / "summary.json").read_text(encoding="utf-8"))
        if (STAGE1_ROOT / "summary.json").exists()
        else None
    )
    leakage_names = sorted(
        {
            name
            for row in enriched
            for name in row["features"]
            if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
            or name.lower() == "seed"
        }
    )
    capture_complete = len(capture) == len(completed) == 10 and all(
        row["identity"]["passed"] for row in capture
    )
    minimum_gates = {
        "d1_spans_at_least_3_seeds": len({int(row["seed"]) for row in d1}) >= 3,
        "d0_spans_at_least_3_seeds": len({int(row["seed"]) for row in d0}) >= 3,
        "y1_spans_at_least_3_seeds": len({int(row["seed"]) for row in y1}) >= 3,
        "y0_spans_at_least_3_seeds": len({int(row["seed"]) for row in y0}) >= 3,
        "no_leakage_feature_names": not leakage_names,
        "no_duplicate_legacy_start_digests": len(set(legacy_digests)) == len(legacy_digests),
        "no_duplicate_v5_2_seed_decisions": len(set(v5_2_keys)) == len(v5_2_keys),
        "grouped_folds_keep_seeds_intact": _fold_audit(enriched)["passed"],
        "effects_exceed_replay_noise": bool(
            v5_2_summary["effect_vs_repeat_noise_evidence"][
                "observed_effect_exceeds_zero_replay_noise"
            ]
        ),
        "observation_v2_exact_capture_complete": capture_complete,
        "v5_3_stage1_integrity_pass": bool(stage1_summary and stage1_summary["passed"]),
    }
    return {
        "schema": "idm_observation_v2_value_screen/1",
        "scientific_role": "diagnostic_only_no_model_no_thresholds_no_probabilities",
        "feature_contract": feature_contract_manifest(),
        "artifact_inventory": {
            "legacy_v2_headroom": {
                "path": str(LEGACY_ROOT),
                "case_count": len(legacy),
                "engaged_case_count": len(legacy_engaged),
                "hdv_model": "route-file Krauss mixture",
                "observation_v2_available": False,
                "pooling_role": "historical_separate_screen_only",
            },
            "v5_2_completed": {
                "path": str(V5_2_ROOT),
                "case_count": len(completed),
                "seed_count": len({int(row["seed"]) for row in completed}),
                "observation_v2_capture_count": len(capture),
            },
            "diagnosis_cases": {
                "count": 7,
                "role": "references_to_existing_v5_2_starts_not_additional_independent_rows",
                "duplicate_rows_added": 0,
            },
        },
        "class_support": {
            "d1_case_count": len(d1),
            "d1_seed_count": len({int(row["seed"]) for row in d1}),
            "d0_case_count": len(d0),
            "d0_seed_count": len({int(row["seed"]) for row in d0}),
            "y1_case_count": len(y1),
            "y1_seed_count": len({int(row["seed"]) for row in y1}),
            "y0_case_count": len(y0),
            "y0_seed_count": len({int(row["seed"]) for row in y0}),
        },
        "v5_2_completed_screen": {
            "safe_vs_unsafe": _feature_screen(enriched, "safe_label"),
            "beneficial_vs_nonbeneficial": _feature_screen(enriched, "benefit_label"),
            "joint_y_vs_not_y": _feature_screen(enriched, "joint_y_label"),
        },
        "legacy_krauss_engaged_screen": {
            "safe_vs_unsafe": _feature_screen(legacy_engaged, "safe_label"),
            "beneficial_vs_nonbeneficial": _feature_screen(
                legacy_engaged, "benefit_label"
            ),
            "not_poolable_with_idm_reason": (
                "different HDV model, damping checkpoint, wall treatment, dose contract, and feature schema"
            ),
        },
        "grouped_fold_audit": _fold_audit(enriched),
        "leakage_feature_names": leakage_names,
        "minimum_gates": minimum_gates,
        "all_descriptive_headroom_gates_pass": all(
            value
            for key, value in minimum_gates.items()
            if key != "v5_3_stage1_integrity_pass"
        ),
        "scale_up_recommendation": (
            "GO_TO_SEPARATELY_APPROVED_TREATMENT_BLIND_V5_3_FEASIBILITY_SCAN"
            if all(minimum_gates.values())
            else "NO_GO_DIAGNOSE_FAILED_PHASE_A_GATES"
        ),
        "predictor_authorized": False,
        "phase_b_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "screen.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
