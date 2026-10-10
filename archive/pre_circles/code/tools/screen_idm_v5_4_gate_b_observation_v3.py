"""Run the frozen one-shot IDM v5.4 Gate-B observation-v3 repair screen."""

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
    OBSERVATION_V3_CANDIDATE_SPECS,
    fit_full_candidate,
    prepare_gate_b_rows_with_features,
    run_observation_v3_screen,
)
from counterfactual.idm_observation_v3_contract import feature_contract_manifest
from experiments import REPO_ROOT
from experiments.idm.wall_v5_4_predictor_contract import DEVELOPMENT_SEEDS, predictor_manifest
from experiments.io import atomic_write_json
from tools.screen_idm_v5_4_gate_b_models import _load_authorized_rows


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
    / "idm_wall_v5_4_gate_b_observation_v3_repair_v1"
)


def _atomic_dump_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bundle(rows: Sequence[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any] | None:
    selected = report.get("selected_candidate")
    thresholds = report.get("selected_policy_thresholds")
    if selected is None:
        return None
    if not isinstance(thresholds, dict):
        raise ValueError("v3 selected candidate lacks frozen thresholds")
    try:
        spec = next(
            value
            for value in OBSERVATION_V3_CANDIDATE_SPECS
            if value.name == selected.get("name")
        )
    except StopIteration:
        raise ValueError("v3 selected candidate is outside the frozen library") from None
    prepared = prepare_gate_b_rows_with_features(rows, spec.feature_names)
    safety_model, benefit_model = fit_full_candidate(prepared, spec)
    return {
        "schema": "idm_v5_4_gate_b_observation_v3_bundle/1",
        "selection_role": report["selected_candidate_role"],
        "screen_contract": report["screen_contract"],
        "predictor_contract": predictor_manifest("development", DEVELOPMENT_SEEDS),
        "feature_contract": feature_contract_manifest(),
        "candidate": {
            "name": spec.name,
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

    rows, source_summary = _load_authorized_rows(args.data.resolve())
    report, predictions = run_observation_v3_screen(rows)
    report["source"] = {
        "data_root": str(args.data.resolve()),
        "summary_schema": source_summary["schema"],
        "seed_count": len(DEVELOPMENT_SEEDS),
        "case_count": len(rows),
    }
    report["runtime"] = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "report.json", report)
    atomic_write_json(output / "oof_predictions.json", predictions)
    bundle = _bundle(rows, report)
    if bundle is not None:
        _atomic_dump_joblib(output / "model_bundle.joblib", bundle)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["selected_candidate"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

