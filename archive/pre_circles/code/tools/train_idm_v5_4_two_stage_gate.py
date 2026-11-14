"""Fit the frozen interpretable Gate-B models on authorized development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import joblib
import sklearn

from counterfactual.idm_predictor_contract import IDM_PRIMARY_FEATURES, feature_contract_manifest
from counterfactual.idm_two_stage_gate import (
    development_report,
    fit_full_gate_b,
    grouped_oof_gate_b,
    prediction_record,
    prepare_gate_b_rows,
)
from experiments import REPO_ROOT
from experiments.idm.wall_v5_4_predictor_contract import (
    DEVELOPMENT_SEEDS,
    predictor_manifest,
)
from experiments.io import atomic_write_json


DEFAULT_DATA = REPO_ROOT / "runs" / "eval" / "idm_wall_v5_4_predictor_development_p30_s4401_4520"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "eval" / "idm_wall_v5_4_two_stage_gate_development"


def _load_authorized_rows(root: Path) -> tuple[list[dict], dict]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "idm_wall_v5_4_predictor_development_summary/1":
        raise ValueError("wrong predictor-development summary schema")
    if summary.get("predictor_fitting_authorized") is not True:
        raise RuntimeError("predictor fitting is not authorized by development gates")
    rows = []
    for seed in DEVELOPMENT_SEEDS:
        payload = json.loads((root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
        rows.extend(payload["cases"])
    if len(rows) != len(DEVELOPMENT_SEEDS) * 4:
        raise ValueError("development case grid is incomplete")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.output.resolve()
    rows, source_summary = _load_authorized_rows(data)
    prepared = prepare_gate_b_rows(rows)
    safety_prob, benefit_prob, folds = grouped_oof_gate_b(prepared)
    report = development_report(prepared, safety_prob, benefit_prob, folds)
    report.update(
        {
            "schema": "idm_v5_4_two_stage_gate_development_report/1",
            "predictor_contract": predictor_manifest("development", DEVELOPMENT_SEEDS),
            "feature_contract": feature_contract_manifest(),
            "primary_feature_names": list(IDM_PRIMARY_FEATURES),
            "source_summary_schema": source_summary["schema"],
            "runtime": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
            },
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "development_report.json", report)
    atomic_write_json(
        output / "oof_predictions.json",
        {
            "schema": "idm_v5_4_gate_b_oof_predictions/1",
            "rows": [
                prediction_record(row, safe, benefit)
                for row, safe, benefit in zip(prepared.rows, safety_prob, benefit_prob)
            ],
        },
    )
    if report["predictor_ready_for_untouched_validation"]:
        safety_model, benefit_model = fit_full_gate_b(prepared)
        joblib.dump(
            {
                "schema": "idm_v5_4_gate_b_model_bundle/1",
                "feature_names": list(IDM_PRIMARY_FEATURES),
                "thresholds": report["selected_thresholds"],
                "safety_model": safety_model,
                "conditional_benefit_model": benefit_model,
            },
            output / "model_bundle.joblib",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["predictor_ready_for_untouched_validation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
