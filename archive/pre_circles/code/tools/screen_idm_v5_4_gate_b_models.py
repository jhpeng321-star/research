"""Run the frozen IDM v5.4 Gate-B observation/model repair screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Any, Sequence

import joblib
import sklearn

from counterfactual.idm_gate_b_model_screen import (
    CANDIDATE_SPECS,
    fit_full_candidate,
    prepare_gate_b_rows_with_features,
    run_model_screen,
)
from counterfactual.idm_predictor_contract import feature_contract_manifest
from experiments import REPO_ROOT
from experiments.idm.wall_v5_4_predictor_contract import (
    CAV_PENETRATION,
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    DEVELOPMENT_SEEDS,
    predictor_manifest,
)
from experiments.io import atomic_write_json


DEFAULT_DATA = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_predictor_development_p30_s4401_4520"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_gate_b_observation_model_repair_v1"
)

SUMMARY_SCHEMA = "idm_wall_v5_4_predictor_development_summary/1"
MANIFEST_SCHEMA = "idm_wall_v5_4_predictor_development_manifest/1"
SEED_SCHEMA = "idm_wall_v5_4_predictor_development_seed/1"
CASE_SCHEMA = "idm_wall_v5_4_predictor_development_case/1"
BUNDLE_SCHEMA = "idm_v5_4_gate_b_observation_model_repair_bundle/1"
SCIENTIFIC_ROLE = "grouped_predictor_development_not_validation"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _load_authorized_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the complete, frozen 4401--4520 development grid."""

    root = Path(root).resolve()
    summary = _load_json_object(root / "summary.json")
    manifest = _load_json_object(root / "manifest.json")

    _require_equal(
        summary.get("schema"), SUMMARY_SCHEMA, "wrong predictor-development summary schema"
    )
    if summary.get("predictor_fitting_authorized") is not True:
        raise RuntimeError("predictor fitting is not authorized by development gates")
    _require_equal(
        summary.get("scientific_role"),
        SCIENTIFIC_ROLE,
        "wrong predictor-development scientific role",
    )
    _require_equal(
        manifest.get("schema"), MANIFEST_SCHEMA, "wrong predictor-development manifest schema"
    )
    _require_equal(
        manifest.get("scientific_role"),
        SCIENTIFIC_ROLE,
        "wrong predictor-development manifest role",
    )
    _require_equal(
        summary.get("manifest"), manifest, "summary manifest differs from manifest.json"
    )

    expected_contract = predictor_manifest("development", DEVELOPMENT_SEEDS)
    expected_features = feature_contract_manifest()
    _require_equal(
        manifest.get("support_contract"),
        expected_contract,
        "predictor-development contract differs from the frozen development contract",
    )
    _require_equal(
        manifest.get("feature_contract"),
        expected_features,
        "manifest feature schema differs from the frozen feature contract",
    )
    _require_equal(
        manifest.get("gate_b_label_contract"),
        expected_contract["gate_b"]["label_contract"],
        "manifest Gate-B label schema differs from the frozen contract",
    )
    _require_equal(
        summary.get("feature_contract"),
        expected_features,
        "summary feature schema differs from the frozen feature contract",
    )
    _require_equal(
        summary.get("gate_b_label_contract"),
        expected_contract["gate_b"]["label_contract"],
        "summary Gate-B label schema differs from the frozen contract",
    )

    seed_root = root / "seeds"
    expected_files = {f"seed{seed}.json" for seed in DEVELOPMENT_SEEDS}
    actual_files = {path.name for path in seed_root.glob("seed*.json") if path.is_file()}
    _require_equal(
        actual_files,
        expected_files,
        "predictor-development seed files do not exactly match seeds 4401--4520",
    )

    rows: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    observed_grid: list[tuple[int, int]] = []
    for seed in DEVELOPMENT_SEEDS:
        payload = _load_json_object(seed_root / f"seed{seed}.json")
        _require_equal(payload.get("schema"), SEED_SCHEMA, f"seed {seed} has the wrong schema")
        _require_equal(payload.get("seed"), seed, f"seed {seed} payload has the wrong seed")
        _require_equal(
            tuple(payload.get("decision_steps") or ()),
            DECISION_STEPS,
            f"seed {seed} has the wrong decision-step grid",
        )
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"seed {seed} cases must be a list")
        _require_equal(
            tuple(row.get("decision_step") for row in cases),
            DECISION_STEPS,
            f"seed {seed} cases do not exactly cover the decision-step grid",
        )
        for row in cases:
            if not isinstance(row, dict):
                raise ValueError(f"seed {seed} contains a non-object case")
            step = int(row.get("decision_step"))
            _require_equal(row.get("schema"), CASE_SCHEMA, f"seed {seed} step {step} has the wrong schema")
            _require_equal(row.get("seed"), seed, f"seed {seed} step {step} has the wrong seed")
            _require_equal(
                row.get("demand_mainline_vph"),
                DEMAND_MAINLINE_VPH,
                f"seed {seed} step {step} has the wrong mainline demand",
            )
            _require_equal(
                row.get("demand_ramp_vph"),
                DEMAND_RAMP_VPH,
                f"seed {seed} step {step} has the wrong ramp demand",
            )
            _require_equal(
                float(row.get("cav_penetration")),
                CAV_PENETRATION,
                f"seed {seed} step {step} has the wrong CAV penetration",
            )
            _require_equal(row.get("hdv_model"), "IDM", f"seed {seed} step {step} is not IDM")
            case_id = str(row.get("case_id"))
            if not case_id or case_id == "None" or case_id in case_ids:
                raise ValueError(f"seed {seed} step {step} has a missing or duplicate case_id")
            case_ids.add(case_id)
            observed_grid.append((seed, step))
            rows.append(row)

    expected_grid = [(seed, step) for seed in DEVELOPMENT_SEEDS for step in DECISION_STEPS]
    _require_equal(observed_grid, expected_grid, "development case grid is incomplete or reordered")
    return rows, summary


def _atomic_dump_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _selected_bundle(
    rows: Sequence[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any] | None:
    selected = report.get("selected_candidate")
    if selected is None:
        return None
    selection_role = report.get("selected_candidate_role")
    thresholds = report.get("selected_policy_thresholds")
    if not isinstance(selected, dict) or not isinstance(selection_role, str):
        raise ValueError("selected candidate metadata is incomplete")
    if not isinstance(thresholds, dict):
        raise ValueError("selected candidate lacks frozen top-level policy thresholds")
    try:
        spec = next(spec for spec in CANDIDATE_SPECS if spec.name == selected.get("name"))
    except StopIteration:
        raise ValueError("selected candidate is outside the frozen candidate library") from None
    _require_equal(
        selected.get("feature_scope"),
        spec.feature_scope,
        "selected candidate feature scope differs from its frozen specification",
    )
    _require_equal(
        selected.get("model_family"),
        spec.model_family,
        "selected candidate model family differs from its frozen specification",
    )
    prepared = prepare_gate_b_rows_with_features(rows, spec.feature_names)
    safety_model, benefit_model = fit_full_candidate(prepared, spec)
    return {
        "schema": BUNDLE_SCHEMA,
        "screen_contract": report["screen_contract"],
        "predictor_contract": predictor_manifest("development", DEVELOPMENT_SEEDS),
        "feature_contract": feature_contract_manifest(),
        "selection_role": selection_role,
        "selected_candidate": {
            "name": spec.name,
            "candidate_order": selected.get("candidate_order"),
            "feature_scope": spec.feature_scope,
            "model_family": spec.model_family,
        },
        "feature_names": list(spec.feature_names),
        "thresholds": thresholds,
        "safety_model": safety_model,
        "conditional_benefit_model": benefit_model,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    data = args.data.resolve()
    output = args.output.resolve()
    rows, source_summary = _load_authorized_rows(data)
    report, predictions = run_model_screen(rows)
    report["source"] = {
        "data_root": str(data),
        "summary_schema": source_summary["schema"],
        "seed_count": len(DEVELOPMENT_SEEDS),
        "case_count": len(rows),
    }
    report["runtime"] = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }

    selected_available = report.get("selected_candidate") is not None
    primary_or_exploratory_passed = bool(
        report.get("aggregate_policy_gate_passed")
        or report.get("exploratory_policy_gate_passed")
    )
    if selected_available != primary_or_exploratory_passed:
        raise ValueError("selected candidate availability disagrees with the policy-gate report")

    bundle = _selected_bundle(rows, report)

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "report.json", report)
    atomic_write_json(output / "oof_predictions.json", predictions)
    if bundle is not None:
        _atomic_dump_joblib(output / "model_bundle.joblib", bundle)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if selected_available else 2


if __name__ == "__main__":
    raise SystemExit(main())
