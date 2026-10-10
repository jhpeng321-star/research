"""Apply the frozen Gate-B bundle once to untouched validation outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from counterfactual.actionability import dose_qualified
from counterfactual.idm_gate_b_prospective_validation import (
    evaluate_fixed_policy,
    probability_diagnostics,
    score_admitted_rows,
)
from experiments import REPO_ROOT
from experiments.idm.wall_v5_4_predictor_contract import VALIDATION_SEEDS
from experiments.idm.wall_v5_4_predictor_validation_contract import (
    MODEL_BUNDLE,
    load_frozen_model_bundle,
    require_frozen_model_bundle,
)
from experiments.io import atomic_write_json


DEFAULT_DATA = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_predictor_validation_p30_s4601_4660"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_gate_b_prospective_validation_v1"
    / "report.json"
)


def _load_rows(data_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((data_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "idm_wall_v5_4_predictor_validation_summary/1":
        raise ValueError("wrong validation summary schema")
    if summary.get("validation_analysis_authorized") is not True:
        raise RuntimeError("paired validation integrity did not authorize analysis")
    rows = []
    for seed in VALIDATION_SEEDS:
        payload = json.loads(
            (data_root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        if payload.get("schema") != "idm_wall_v5_4_predictor_validation_seed/1":
            raise ValueError(f"wrong validation seed schema for {seed}")
        if int(payload.get("seed")) != seed:
            raise ValueError(f"wrong validation seed identity for {seed}")
        rows.extend(payload["cases"])
    if len(rows) != len(VALIDATION_SEEDS) * 4:
        raise ValueError("validation does not contain the frozen 60-by-4 grid")
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bundle", type=Path, default=MODEL_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    bundle_contract = require_frozen_model_bundle(args.bundle.resolve())
    summary, rows = _load_rows(data_root)
    manifest_bundle = summary["manifest"]["support_contract"].get("frozen_gate_b_bundle")
    if manifest_bundle != bundle_contract:
        raise ValueError("paired outcomes were not generated under the frozen bundle manifest")
    bundle = load_frozen_model_bundle(args.bundle.resolve())
    predictions = score_admitted_rows(rows, bundle)
    report = evaluate_fixed_policy(
        rows, predictions, integrity_passed=bool(summary["passed"])
    )
    report["probability_diagnostics"] = probability_diagnostics(rows, predictions)
    report.update(
        {
            "data_root": str(data_root),
            "model_bundle": bundle_contract,
            "validation_seed_partition": list(VALIDATION_SEEDS),
            "development_refit_performed": False,
            "threshold_retuning_performed": False,
        }
    )
    output = args.output.resolve()
    atomic_write_json(output, report)
    rows_by_id = {str(row["case_id"]): row for row in rows}
    enriched_predictions = []
    for prediction in predictions:
        row = rows_by_id[str(prediction["case_id"])]
        oracle = row["safety_service_oracle"]
        dose = row["treatment_dose"]
        enriched_predictions.append(
            {
                **prediction,
                "dose_qualified": bool(dose_qualified(row)),
                "realized_duration_fraction": float(
                    dose.get("realized_duration_fraction", 0.0)
                ),
                "completion_reason": str(dose.get("completion_reason", "unknown")),
                "observed_safety_admissible": bool(oracle["safety_admissible"]),
                "observed_traffic_noninferior": bool(oracle["traffic_noninferior"]),
                "observed_meaningful_benefit": bool(oracle["meaningful_benefit"]),
                "observed_joint_beneficial": bool(oracle["wall_selected"]),
                "operational_joint_beneficial": bool(
                    dose_qualified(row) and oracle["wall_selected"]
                ),
                "refusal_reasons": list(oracle.get("refusal_reasons", ())),
                "effects": dict(oracle["effects"]),
            }
        )
    atomic_write_json(
        output.with_name("predictions.json"),
        {
            "schema": "idm_v5_4_gate_b_exploratory_prospective_predictions/1",
            "model_bundle": bundle_contract,
            "predictions": enriched_predictions,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
