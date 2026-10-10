"""Run the development-only IDM v5.4 observation-sufficiency diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from counterfactual.idm_observation_sufficiency import (
    audit_features,
    current_model_error_analysis,
    family_model_screen,
    feature_families,
    leave_one_family_out_ablation,
    nearest_neighbor_observability,
    nested_feature_model_selection,
    population_integrity,
    prepare_population,
    shadow_rollout_reconstructibility,
    shuffled_label_controls,
    univariate_information,
)
from experiments import REPO_ROOT
from experiments.idm.wall_v5_4_predictor_contract import DEVELOPMENT_SEEDS
from experiments.io import atomic_write_json


DEFAULT_DATA = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_predictor_development_p30_s4401_4520"
)
DEFAULT_MODEL_SCREEN = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_gate_b_observation_model_repair_v1"
    / "oof_predictions.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_observation_sufficiency_diagnosis"
)


def load_development_rows(data_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((data_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "idm_wall_v5_4_predictor_development_summary/1":
        raise ValueError("wrong development summary schema")
    manifest = summary.get("manifest") or {}
    support_contract = manifest.get("support_contract") or {}
    if support_contract.get("role") != "development":
        raise ValueError("source is not the frozen development partition")
    if tuple(int(value) for value in support_contract.get("seeds", ())) != DEVELOPMENT_SEEDS:
        raise ValueError("source uses the wrong development seeds")
    rows = []
    for seed in DEVELOPMENT_SEEDS:
        payload = json.loads(
            (data_root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        if payload.get("schema") != "idm_wall_v5_4_predictor_development_seed/1":
            raise ValueError(f"wrong development seed schema for {seed}")
        if int(payload.get("seed")) != seed:
            raise ValueError(f"wrong development seed identity for {seed}")
        rows.extend(payload["cases"])
    if len(rows) != 480:
        raise ValueError("development source does not contain 480 declared windows")
    if len({str(row["case_id"]) for row in rows}) != len(rows):
        raise ValueError("development source has duplicate case IDs")
    return summary, rows


def _best_family(
    family_reports: list[Mapping[str, Any]], target: str
) -> dict[str, Any]:
    candidates = []
    for report in family_reports:
        metrics = report["targets"][target]["nested_metrics"]
        candidates.append(
            {
                "family": report["family"],
                "feature_count": report["feature_count"],
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
            }
        )
    return max(
        candidates,
        key=lambda row: (
            float(row["roc_auc"] or -1.0),
            float(row["average_precision"] or -1.0),
            -int(row["feature_count"]),
        ),
    )


def proposed_v2_contract(
    family_screen: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> dict[str, Any]:
    reports = list(family_screen["families"])
    best = {
        target: _best_family(reports, target)
        for target in (
            "dose_80pct",
            "safety_admissible",
            "conditional_joint_benefit",
        )
    }
    by_name = {str(row["family"]): row for row in reports}
    supported = []
    for target, selection in best.items():
        report = by_name[str(selection["family"])]
        importance = report["targets"][target]["held_out_permutation_importance"]
        for row in importance[:10]:
            if float(row["held_out_permutation_auc_decrease_mean"]) > 0.0:
                supported.append(str(row["feature"]))
    supported = list(dict.fromkeys(supported))[:24]
    shadow_outputs = list(
        shadow["deterministic_shadow_rollout_protocol"]["required_outputs"]
    )
    return {
        "schema": "idm_v5_4_observation_sufficiency_v2_proposal/1",
        "status": "review_only_not_authorized_for_training_or_new_seeds",
        "selection_data": "development_seeds_4401_4520_only",
        "best_current_family_by_target": best,
        "current_features_with_positive_held_out_importance_top24": supported,
        "mandatory_new_shadow_rollout_outputs": shadow_outputs,
        "smallest_recommended_blocks": [
            {
                "block": "static_and_temporal_bottleneck_state",
                "purpose": "arrival-discharge imbalance storage and compression persistence",
            },
            {
                "block": "member_and_rear_trajectory_projection",
                "purpose": "safety and treatment-dose observability",
            },
            {
                "block": "short_horizon_damping_vs_wall_shadow_delta",
                "purpose": "projected dose authority and inflow reduction",
            },
        ],
        "prohibited_inputs": [
            "actual treatment dose or completion reason",
            "actual treatment or control outcome",
            "oracle label or counterfactual effect",
            "seed identity or absolute vehicle identity",
        ],
        "estimated_next_sample_and_runtime": {
            "first_shadow_feature_replay": (
                "recompute decision-time shadow features on the already-opened 120 "
                "development seeds and join to the existing frozen labels"
            ),
            "minimum_grouped_development_support": (
                "retain at least the current 179 qualified cases across 101 seeds; "
                "prefer 250-350 qualified cases if a newly allocated development cohort is approved"
            ),
            "untouched_v2_validation": (
                "60-120 newly allocated independent seeds, excluding 4601-4660 and 5001-5060"
            ),
            "runtime_estimate_four_workers": (
                "about 1.5-2.5 hours to recapture the 120 development prefixes plus "
                "under about 15 minutes for pure 10-second shadows on 199 admissions; "
                "then 3-6 hours for an untouched 60-seed paired test; benchmark before scale-up"
            ),
        },
        "gate_before_v2_training": (
            "seed-grouped safety or benefit AUC must improve by at least about 0.03 "
            "and fixed-threshold policy precision/coverage must improve without collapsing "
            "below the predeclared command/seed minima"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-screen-oof", type=Path, default=DEFAULT_MODEL_SCREEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--null-repeats", type=int, default=20)
    args = parser.parse_args()

    started = time.time()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    print("[1/8] loading frozen development rows", flush=True)
    summary, rows = load_development_rows(data_root)
    prediction_payload = json.loads(
        args.model_screen_oof.resolve().read_text(encoding="utf-8")
    )
    if prediction_payload.get("schema") != "idm_v5_4_gate_b_observation_model_repair_oof/1":
        raise ValueError("wrong current model-screen prediction schema")

    print("[2/8] auditing populations and every saved feature", flush=True)
    integrity = population_integrity(rows)
    if not all(integrity["frozen_count_checks"].values()):
        raise RuntimeError("development population integrity check failed")
    audit = audit_features(rows)
    families = feature_families(audit)
    population = prepare_population(rows)
    errors = current_model_error_analysis(rows, prediction_payload)

    print("[3/8] computing seed-clustered univariate information", flush=True)
    univariate = univariate_information(
        population, audit, bootstrap_repeats=max(20, args.bootstrap_repeats)
    )

    print("[4/8] running nested grouped feature-family/model screen", flush=True)
    family_screen = family_model_screen(population, families)
    model_predictions = family_screen.pop("prediction_payload")

    print("[5/8] running fully nested family+model selection and ablations", flush=True)
    nested_selection = nested_feature_model_selection(population, families)
    nested_predictions = nested_selection.pop("prediction_payload")
    ablations = leave_one_family_out_ablation(population, families)

    print("[6/8] running shuffled-label controls and observability tests", flush=True)
    null_controls = shuffled_label_controls(
        population,
        families["G_combined_physical"],
        repeats=max(5, args.null_repeats),
    )
    nearest = nearest_neighbor_observability(
        population, families["G_combined_physical"]
    )

    print("[7/8] auditing short-horizon shadow-rollout reconstructibility", flush=True)
    shadow = shadow_rollout_reconstructibility(rows)
    proposed = proposed_v2_contract(family_screen, shadow)

    report = {
        "schema": "idm_v5_4_observation_sufficiency_diagnosis/1",
        "scientific_role": "development_only_observation_discovery",
        "source": {
            "data_root": str(data_root),
            "development_seed_partition": list(DEVELOPMENT_SEEDS),
            "source_summary_schema": summary["schema"],
            "validation_or_final_outcomes_read": False,
            "new_sumo_branches_generated": False,
        },
        "population_integrity": integrity,
        "feature_audit": audit,
        "feature_families": {
            name: {"feature_count": len(values), "feature_names": list(values)}
            for name, values in families.items()
        },
        "current_model_error_analysis": errors,
        "univariate_information": univariate,
        "family_model_screen": family_screen,
        "nested_feature_model_selection": nested_selection,
        "leave_one_family_out_ablation": ablations,
        "shuffled_label_controls": null_controls,
        "nearest_neighbor_observability": nearest,
        "shadow_rollout_reconstructibility": shadow,
        "proposed_v2_observation_contract": proposed,
        "wall_clock_seconds": time.time() - started,
        "guardrails": [
            "Development-only discovery; does not validate or reinterpret the frozen v1 policy.",
            "No treatment outcome, realized dose, completion/release reason, oracle, effect, seed ID or vehicle ID is a model input.",
            "An AUC gain below about 0.03 is weak unless deployable policy metrics also improve.",
        ],
    }
    print("[8/8] writing machine-readable diagnosis", flush=True)
    atomic_write_json(output / "report.json", report)
    atomic_write_json(
        output / "model_predictions.json",
        {
            "schema": "idm_v5_4_observation_sufficiency_predictions/1",
            "development_seed_partition": list(DEVELOPMENT_SEEDS),
            "family_nested_predictions": model_predictions,
            "fully_nested_predictions": nested_predictions,
        },
    )
    atomic_write_json(output / "proposed_v2_observation_contract.json", proposed)
    print(json.dumps({
        "output": str(output),
        "wall_clock_seconds": report["wall_clock_seconds"],
        "best_current_family_by_target": proposed["best_current_family_by_target"],
        "shadow_rollout_reconstructible": shadow["can_compute_exact_shadow_rollout_from_saved_rows"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
