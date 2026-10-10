"""Audit v5.2 decision observations before any two-stage gate fitting.

The audit is descriptive and read-only.  It checks feature completeness,
variation, label headroom, seed diversity, and whether the current contract
observes the post-merge edge where delayed unsafe divergence was diagnosed.
It deliberately does not fit or select a predictor.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from counterfactual.actionability import dose_qualified, summarize_actionability
from counterfactual.idm_predictor_contract import (
    FORBIDDEN_FEATURE_TOKENS,
    IDM_DECISION_FEATURES,
    IDM_PRIMARY_FEATURES,
    staged_targets,
)


SCHEMA = "idm_wall_v5_2_decision_observation_audit/1"
DEFAULT_INPUT = Path("runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312")


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _seed_count(rows: Iterable[Mapping[str, Any]]) -> int:
    return len({int(row["seed"]) for row in rows})


def _feature_audit(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted({str(name) for row in rows for name in (row.get("features") or {})})
    missing_counts: dict[str, int] = {}
    nonfinite_counts: dict[str, int] = {}
    distinct_counts: dict[str, int] = {}
    for name in names:
        values: list[float] = []
        missing = 0
        nonfinite = 0
        for row in rows:
            features = row.get("features") or {}
            if name not in features:
                missing += 1
                continue
            value = _finite(features[name])
            if value is None:
                nonfinite += 1
            else:
                values.append(value)
        missing_counts[name] = missing
        nonfinite_counts[name] = nonfinite
        distinct_counts[name] = len({round(value, 12) for value in values})

    zero_variance = sorted(
        name
        for name in names
        if missing_counts[name] == 0
        and nonfinite_counts[name] == 0
        and distinct_counts[name] <= 1
    )
    missing_primary = sorted(
        name
        for name in IDM_PRIMARY_FEATURES
        if name not in {"context_decision_step", "context_cav_penetration"}
        and name not in names
    )
    primary_zero_variance = sorted(
        name for name in IDM_PRIMARY_FEATURES if name in zero_variance
    )
    forbidden_named = sorted(
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    actual_downstream_names = sorted(
        name
        for name in names
        if name.startswith("decision_post_merge_")
        or name.startswith("pattern60_post_merge_")
        or name.startswith("traffic_lag_") and "post_merge" in name
    )
    premerge_relative_names = sorted(
        name for name in names if name == "local_downstream_minus_upstream_speed_mps"
    )
    return {
        "observed_feature_count": len(names),
        "frozen_expanded_allowlist_count": len(IDM_DECISION_FEATURES),
        "frozen_primary_count": len(IDM_PRIMARY_FEATURES),
        "all_primary_features_available": not missing_primary,
        "missing_primary_features": missing_primary,
        "features_missing_in_any_row": sorted(
            name for name, count in missing_counts.items() if count
        ),
        "features_nonfinite_in_any_row": sorted(
            name for name, count in nonfinite_counts.items() if count
        ),
        "zero_variance_feature_count": len(zero_variance),
        "zero_variance_features": zero_variance,
        "primary_zero_variance_features": primary_zero_variance,
        "forbidden_token_feature_names": forbidden_named,
        "explicit_post_merge_edge_feature_names": actual_downstream_names,
        "premerge_relative_feature_names_that_are_not_post_merge_observations": (
            premerge_relative_names
        ),
        "has_explicit_post_merge_edge_state": bool(actual_downstream_names),
        "has_60s_predecision_history": any(name.startswith("pattern60_") for name in names),
        "has_physical_preflight_state": all(
            name in names
            for name in (
                "preflight_member_count",
                "preflight_required_lane_count",
                "preflight_predicted_max_span_m",
                "preflight_min_front_ttc_s_capped",
                "preflight_min_rear_ttc_s_capped",
            )
        ),
    }


def audit_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    admitted = [
        row
        for row in rows
        if bool((row.get("predecision_supervisor") or {}).get("admitted"))
    ]
    completed = [row for row in admitted if dose_qualified(row)]
    failed_dose = [row for row in admitted if not dose_qualified(row)]
    safe = [
        row
        for row in completed
        if bool((row.get("safety_service_oracle") or {}).get("safety_admissible"))
    ]
    unsafe = [row for row in completed if row not in safe]
    selected = [
        row
        for row in completed
        if bool((row.get("safety_service_oracle") or {}).get("wall_selected"))
    ]
    not_selected = [row for row in completed if row not in selected]
    prefix_failures = [
        str(row.get("case_id"))
        for row in rows
        if not bool((row.get("prefix_identity") or {}).get("passed"))
    ]
    feature_contracts = Counter(str(row.get("feature_contract_version")) for row in rows)
    actionability = summarize_actionability(rows)
    feature_audit = _feature_audit(admitted)
    target_contract_errors: dict[str, str] = {}
    target_label_mismatches: list[str] = []
    for row in admitted:
        case_id = str(row.get("case_id"))
        try:
            targets = staged_targets(row)
        except (KeyError, TypeError, ValueError) as error:
            target_contract_errors[case_id] = f"{type(error).__name__}: {error}"
            continue
        expected = bool(
            dose_qualified(row)
            and (row.get("safety_service_oracle") or {}).get("wall_selected", False)
        )
        if targets.joint_safe_beneficial != expected:
            target_label_mismatches.append(case_id)

    headroom_checks = {
        "gate_f_positive_dose_spans_at_least_3_seeds": _seed_count(completed) >= 3,
        "gate_f_failed_dose_spans_at_least_3_seeds": _seed_count(failed_dose) >= 3,
        "gate_b_joint_positive_spans_at_least_3_seeds": _seed_count(selected) >= 3,
        "gate_b_joint_negative_spans_at_least_3_seeds": _seed_count(not_selected) >= 3,
        "safety_positive_spans_at_least_3_seeds": _seed_count(safe) >= 3,
        "safety_negative_spans_at_least_3_seeds": _seed_count(unsafe) >= 3,
        "explicit_post_merge_edge_state_available": feature_audit[
            "has_explicit_post_merge_edge_state"
        ],
        "all_prefix_identity_checks_pass": not prefix_failures,
        "one_feature_contract_only": len(feature_contracts) == 1,
        "current_target_contract_valid": (
            not target_contract_errors and not target_label_mismatches
        ),
    }
    return {
        "schema": SCHEMA,
        "source_case_count": len(rows),
        "source_seed_count": _seed_count(rows),
        "supervisor_admitted_case_count": len(admitted),
        "supervisor_admitted_seed_count": _seed_count(admitted),
        "dose_qualified_case_count": len(completed),
        "dose_qualified_seed_count": _seed_count(completed),
        "failed_dose_case_count": len(failed_dose),
        "failed_dose_seed_count": _seed_count(failed_dose),
        "safe_completed_case_count": len(safe),
        "safe_completed_seed_count": _seed_count(safe),
        "unsafe_completed_case_count": len(unsafe),
        "unsafe_completed_seed_count": _seed_count(unsafe),
        "joint_positive_case_count": len(selected),
        "joint_positive_seed_count": _seed_count(selected),
        "joint_negative_case_count": len(not_selected),
        "joint_negative_seed_count": _seed_count(not_selected),
        "feature_contract_counts": dict(feature_contracts),
        "target_contract_error_by_case": target_contract_errors,
        "target_joint_label_mismatch_case_ids": target_label_mismatches,
        "prefix_identity_failure_case_ids": prefix_failures,
        "feature_audit": feature_audit,
        "minimum_headroom_checks": headroom_checks,
        "minimum_headroom_pass": all(headroom_checks.values()),
        "predecision_separation_diagnostic": actionability[
            "predecision_separation"
        ],
        "interpretation": (
            "The current artifacts support outcome diversity for a diagnostic "
            "Gate-B screen, but do not support Gate-F verification: only two "
            "seeds contain failed dose, and the frozen observation contract has "
            "no explicit post-merge edge state despite delayed harm concentrating "
            "there. Extend and replay-verify the treatment-blind observation "
            "contract before fitting."
        ),
        "guardrail": (
            "Descriptive development audit only; feature separation is not "
            "prospective performance and may not be used to claim prediction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.input / "seeds").glob("seed*.json"))
    ]
    if not payloads:
        raise SystemExit(f"no seed artifacts found under {args.input / 'seeds'}")
    result = audit_rows(
        [row for payload in payloads for row in payload.get("cases") or []]
    )
    output = args.output or args.input / "decision_observation_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {key: value for key, value in result.items() if key not in {
        "predecision_separation_diagnostic", "feature_audit"
    }}
    compact["feature_audit"] = {
        key: result["feature_audit"][key]
        for key in (
            "observed_feature_count",
            "zero_variance_feature_count",
            "primary_zero_variance_features",
            "has_explicit_post_merge_edge_state",
            "has_60s_predecision_history",
            "has_physical_preflight_state",
        )
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
