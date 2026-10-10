"""Frozen exploratory Gate-B prospective-validation contract."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib

from experiments import REPO_ROOT


CONTRACT_VERSION = "idm_v5_4_gate_b_exploratory_prospective_validation/1"
MODEL_BUNDLE = (
    REPO_ROOT
    / "runs"
    / "eval"
    / "idm_wall_v5_4_gate_b_observation_model_repair_v1"
    / "model_bundle.joblib"
)
MODEL_BUNDLE_SHA256 = "9cea403b96815dae0c856de88771cc49d03b1f8057d7fb0d9454afd593418662"
PRIMARY_FEATURE_SHA256 = "3FB9061E85073EAF65A788132ECAFDED4839CCE4D1F9A4DF1F0CC28E74D1A729"
SELECTED_CANDIDATE = "primary_shallow_boosting"
SELECTION_ROLE = "exploratory_selective_enrichment"
SAFETY_THRESHOLD = 0.7000000000000002
BENEFIT_THRESHOLD = 0.7499999999999999


def prospective_gate() -> dict[str, Any]:
    return {
        "minimum_command_cases": 5,
        "minimum_command_seeds": 4,
        "minimum_selected_safety_rate": 0.80,
        "minimum_joint_benefit_precision": 0.45,
        "minimum_safety_enrichment": 0.15,
        "minimum_joint_benefit_enrichment": 0.15,
        "minimum_selected_dose_success_rate": 0.80,
        "maximum_selected_excess_collision_or_teleport_cases": 0,
        "scientific_role": "metastability_tolerant_exploratory_validation",
    }


def require_frozen_model_bundle(path: Path = MODEL_BUNDLE) -> dict[str, Any]:
    payload_bytes = Path(path).read_bytes()
    if sha256(payload_bytes).hexdigest() != MODEL_BUNDLE_SHA256:
        raise ValueError("frozen exploratory model bundle hash changed")
    bundle = joblib.load(path)
    if bundle.get("schema") != "idm_v5_4_gate_b_observation_model_repair_bundle/1":
        raise ValueError("wrong exploratory model bundle schema")
    if bundle.get("selection_role") != SELECTION_ROLE:
        raise ValueError("model bundle is not the frozen exploratory policy")
    selected = bundle.get("selected_candidate") or {}
    if selected.get("name") != SELECTED_CANDIDATE:
        raise ValueError("wrong exploratory candidate")
    feature_contract = bundle.get("feature_contract") or {}
    if feature_contract.get("primary_feature_names_sha256") != PRIMARY_FEATURE_SHA256:
        raise ValueError("wrong primary feature contract")
    if tuple(bundle.get("feature_names") or ()) != tuple(
        feature_contract.get("primary_feature_names") or ()
    ):
        raise ValueError("bundle feature order differs from the frozen primary contract")
    thresholds = bundle.get("thresholds") or {}
    if float(thresholds.get("safety_threshold")) != SAFETY_THRESHOLD:
        raise ValueError("frozen safety threshold changed")
    if float(thresholds.get("benefit_threshold")) != BENEFIT_THRESHOLD:
        raise ValueError("frozen benefit threshold changed")
    return {
        "path": str(Path(path).resolve()),
        "sha256": MODEL_BUNDLE_SHA256,
        "schema": bundle["schema"],
        "selection_role": bundle["selection_role"],
        "selected_candidate": selected,
        "thresholds": {
            "safety_threshold": SAFETY_THRESHOLD,
            "benefit_threshold": BENEFIT_THRESHOLD,
        },
        "primary_feature_names_sha256": PRIMARY_FEATURE_SHA256,
        "feature_count": len(bundle["feature_names"]),
        "prospective_gate": prospective_gate(),
    }


def load_frozen_model_bundle(path: Path = MODEL_BUNDLE) -> dict[str, Any]:
    require_frozen_model_bundle(path)
    return joblib.load(path)


__all__ = [
    "BENEFIT_THRESHOLD",
    "CONTRACT_VERSION",
    "MODEL_BUNDLE",
    "MODEL_BUNDLE_SHA256",
    "PRIMARY_FEATURE_SHA256",
    "SAFETY_THRESHOLD",
    "SELECTED_CANDIDATE",
    "SELECTION_ROLE",
    "load_frozen_model_bundle",
    "prospective_gate",
    "require_frozen_model_bundle",
]
