"""Analyze the frozen IDM v5.4 shadow-observation v2 development capture.

This runner is deliberately development-only.  It validates all 480 capture
records before opening the already-existing labels, joins by exact ``case_id``,
and never reads either sealed predictor partition.  Every learned comparison
uses complete-seed grouped outer and inner folds.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from counterfactual.actionability import dose_qualified
from counterfactual.escalation_v2_model import probability_metrics
from counterfactual.idm_gate_b_model_screen import make_candidate_estimator
from counterfactual.idm_predictor_contract import IDM_PRIMARY_FEATURES
from counterfactual.idm_observation_sufficiency import feature_value
from counterfactual.idm_shadow_observation_v2_contract import (
    ANALYSIS_ROOT,
    ANALYSIS_SCHEMA,
    CAPTURE_ROOT,
    CAPTURE_SCHEMA,
    CONFIRMATORY_GATES,
    CONTRACT_VERSION,
    INNER_FOLDS,
    OUTER_FOLDS,
    PROHIBITED_INPUT_TOKENS,
    RANDOM_STATE,
    SEED_SCHEMA,
    SOURCE_FEASIBILITY_ROOT,
    SOURCE_LABEL_ROOT,
    SOURCE_SHA256,
    TABULAR_MODEL_FAMILIES,
    contract_manifest,
)
from experiments import REPO_ROOT
from experiments.idm.wall_v5_4_predictor_contract import (
    DECISION_STEPS,
    DEVELOPMENT_SEEDS,
)
from experiments.io import atomic_write_json


CASE_SCHEMA = "idm_v5_4_shadow_observation_v2_case/1"
CAPTURE_PHASE = "after_identical_damping_before_wall_override"
EXACT_BASELINE = "current_engineered_v1_exact"
DEPLOYABLE_BASELINE = "current_engineered_deployable"
PHYSICAL_BLOCK = "physical_trajectory_20s"
SHADOW_BLOCK = "exact_wall_shadow_10s"
COMBINED_BLOCK = "trajectory_plus_shadow"
CORE_BLOCKS = (EXACT_BASELINE, DEPLOYABLE_BASELINE, PHYSICAL_BLOCK, SHADOW_BLOCK, COMBINED_BLOCK)
PHYSICAL_SUBBLOCKS = (
    "sectional_transitional_instability",
    "merge_pressure_breakdown_state",
    "trajectory_physical_20s",
)
SHADOW_SUBBLOCK = "proposed_wall_shadow_10s"
TARGET_NAMES = ("dose", "safety", "benefit")
SECONDARY_EFFECTS = (
    "hard_brake_onset_excess",
    "hard_brake_excess_vehicle_steps",
    "low_ttc_excess_steps",
    "merge_discharge_benefit",
    "delay_benefit_s",
)
AUXILIARY_EFFECTS = (
    "collision_excess_vehicle_count",
    "teleport_excess_vehicle_count",
)


@dataclass(frozen=True)
class FeatureMatrix:
    names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class DevelopmentPopulation:
    """Gate-F-admitted development cases in canonical case-id order."""

    rows: tuple[Mapping[str, Any], ...]
    groups: np.ndarray
    dose: np.ndarray
    safety: np.ndarray
    benefit: np.ndarray
    effects: Mapping[str, np.ndarray]
    blocks: Mapping[str, FeatureMatrix]
    physical_subblocks: Mapping[str, FeatureMatrix]

    def target_mask(self, target: str) -> np.ndarray:
        if target == "dose":
            return np.ones(len(self.rows), dtype=bool)
        if target == "safety":
            return self.dose == 1
        if target == "benefit":
            return (self.dose == 1) & (self.safety == 1)
        raise ValueError(f"unknown target {target!r}")

    def target(self, target: str) -> np.ndarray:
        if target == "dose":
            return self.dose
        if target == "safety":
            return self.safety
        if target == "benefit":
            return self.benefit
        raise ValueError(f"unknown target {target!r}")


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_digest(payload: Mapping[str, Any]) -> str | None:
    direct = payload.get("contract_sha256")
    if direct is not None:
        return str(direct)
    for key in ("contract", "shadow_observation_v2_contract", "protocol_contract"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value.get("contract_sha256") is not None:
            return str(value["contract_sha256"])
    return None


def validate_capture_payloads(
    manifest: Mapping[str, Any],
    seed_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_decisions: Sequence[int],
) -> list[dict[str, Any]]:
    """Fail closed on a complete capture before any label is opened."""

    expected_contract = contract_manifest()
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("wrong shadow-observation capture manifest schema")
    if _contract_digest(manifest) != expected_contract["contract_sha256"]:
        raise ValueError("capture manifest differs from the frozen v2 contract")
    declared_seeds = manifest.get("seeds", manifest.get("source_seeds"))
    if declared_seeds is not None and tuple(int(value) for value in declared_seeds) != tuple(
        int(value) for value in expected_seeds
    ):
        raise ValueError("capture manifest uses the wrong development seeds")
    declared_decisions = manifest.get("decision_steps")
    if declared_decisions is not None and tuple(int(value) for value in declared_decisions) != tuple(
        int(value) for value in expected_decisions
    ):
        raise ValueError("capture manifest uses the wrong decision grid")

    by_seed = {int(payload.get("seed")): payload for payload in seed_payloads}
    if set(by_seed) != set(int(value) for value in expected_seeds):
        raise ValueError("capture seed artifacts do not exactly match the development partition")
    if len(by_seed) != len(seed_payloads):
        raise ValueError("duplicate capture seed artifact")

    rows: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for seed in expected_seeds:
        payload = by_seed[int(seed)]
        if payload.get("schema") != SEED_SCHEMA:
            raise ValueError(f"seed {seed} has the wrong shadow-observation schema")
        if _contract_digest(payload) != expected_contract["contract_sha256"]:
            raise ValueError(f"seed {seed} differs from the frozen v2 contract")
        if tuple(int(value) for value in payload.get("decision_steps", ())) != tuple(
            int(value) for value in expected_decisions
        ):
            raise ValueError(f"seed {seed} has an incomplete decision grid")
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"seed {seed} cases must be a list")
        if tuple(int(row.get("decision_step")) for row in cases) != tuple(
            int(value) for value in expected_decisions
        ):
            raise ValueError(f"seed {seed} cases are incomplete or reordered")
        for row in cases:
            if not isinstance(row, dict):
                raise ValueError(f"seed {seed} contains a non-object case")
            step = int(row.get("decision_step"))
            if row.get("schema") != CASE_SCHEMA:
                raise ValueError(f"seed {seed} step {step} has the wrong case schema")
            if int(row.get("seed")) != int(seed):
                raise ValueError(f"seed {seed} step {step} has the wrong seed metadata")
            case_id = str(row.get("case_id"))
            if not case_id or case_id == "None" or case_id in case_ids:
                raise ValueError(f"seed {seed} step {step} has a missing or duplicate case_id")
            case_ids.add(case_id)
            if row.get("capture_phase") != CAPTURE_PHASE:
                raise ValueError(f"{case_id} was captured at the wrong causal phase")
            baseline = row.get("baseline_observation")
            if not isinstance(baseline, Mapping):
                raise ValueError(f"{case_id} lacks the predecision observation")
            _validate_feature_container(baseline, case_id=case_id, role="baseline")
            admitted = bool(row.get("gate_f_admitted"))
            shadow = row.get("shadow_observation")
            integrity = row.get("integrity") or {}
            if admitted:
                if not isinstance(shadow, Mapping):
                    raise ValueError(f"admitted case {case_id} lacks its exact shadow")
                _validate_feature_container(shadow, case_id=case_id, role="shadow")
                if integrity.get("prefix_identity_passed") is not True:
                    raise ValueError(f"admitted case {case_id} failed prefix identity")
                if integrity.get("shadow_generated") is not True:
                    raise ValueError(f"admitted case {case_id} was not shadowed")
            elif shadow is not None:
                raise ValueError(f"infeasible case {case_id} unexpectedly contains a shadow")
            if integrity.get("feasibility_source_treatment_blind") is not True:
                raise ValueError(f"{case_id} does not bind treatment-blind Gate F")
            rows.append(dict(row))
    return rows


def _validate_feature_container(
    payload: Mapping[str, Any], *, case_id: str, role: str
) -> None:
    model_features = payload.get("model_features")
    blocks = payload.get("feature_blocks")
    if not isinstance(model_features, Mapping) or not isinstance(blocks, Mapping):
        raise ValueError(f"{case_id} {role} observation lacks frozen feature containers")
    _validate_feature_names(model_features, context=f"{case_id}:{role}:model_features")
    for block_name, block in blocks.items():
        _validate_feature_names(_flatten_numeric(block), context=f"{case_id}:{role}:{block_name}")


def _validate_feature_names(values: Mapping[str, Any], *, context: str) -> None:
    for name, value in values.items():
        lowered = str(name).lower()
        hit = next((token for token in PROHIBITED_INPUT_TOKENS if token in lowered), None)
        if hit is not None:
            raise ValueError(f"prohibited input token {hit!r} in {context}:{name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"non-scalar model feature {context}:{name}")
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite model feature {context}:{name}")


def load_capture_rows(capture_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(capture_root).resolve()
    manifest = _json_object(root / "manifest.json")
    seed_root = root / "seeds"
    expected_names = {f"seed{seed}.json" for seed in DEVELOPMENT_SEEDS}
    actual_names = {path.name for path in seed_root.glob("seed*.json") if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("capture directory does not contain exactly development seeds 4401--4520")
    payloads = [_json_object(seed_root / f"seed{seed}.json") for seed in DEVELOPMENT_SEEDS]
    rows = validate_capture_payloads(
        manifest,
        payloads,
        expected_seeds=DEVELOPMENT_SEEDS,
        expected_decisions=DECISION_STEPS,
    )
    if len(rows) != 480:
        raise ValueError("capture does not contain all 480 declared decisions")
    return manifest, rows


def _verify_frozen_source_hashes(label_root: Path) -> dict[str, str]:
    paths = {
        "label_manifest": Path(label_root) / "manifest.json",
        "label_summary": Path(label_root) / "summary.json",
        "feasibility_analysis": REPO_ROOT / SOURCE_FEASIBILITY_ROOT / "analysis.json",
        "v1_diagnosis": (
            REPO_ROOT
            / "runs"
            / "eval"
            / "idm_wall_v5_4_observation_sufficiency_diagnosis"
            / "report.json"
        ),
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": SOURCE_SHA256[name], "observed": observed[name]}
        for name in SOURCE_SHA256
        if observed.get(name) != SOURCE_SHA256[name]
    }
    if mismatches:
        raise ValueError(f"frozen development source hash mismatch: {mismatches}")
    return observed


def load_label_rows(label_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Open labels only after the caller has validated the complete capture."""

    root = Path(label_root).resolve()
    summary = _json_object(root / "summary.json")
    manifest = _json_object(root / "manifest.json")
    if summary.get("schema") != "idm_wall_v5_4_predictor_development_summary/1":
        raise ValueError("wrong development label summary schema")
    if manifest.get("schema") != "idm_wall_v5_4_predictor_development_manifest/1":
        raise ValueError("wrong development label manifest schema")
    contract = manifest.get("support_contract") or {}
    if contract.get("role") != "development":
        raise ValueError("label source is not the development partition")
    if tuple(int(value) for value in contract.get("seeds", ())) != DEVELOPMENT_SEEDS:
        raise ValueError("label source does not use development seeds 4401--4520")
    if summary.get("manifest") != manifest:
        raise ValueError("development label manifest differs from summary")
    expected_files = {f"seed{seed}.json" for seed in DEVELOPMENT_SEEDS}
    seed_root = root / "seeds"
    actual_files = {path.name for path in seed_root.glob("seed*.json") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("label source does not contain exactly the development seeds")
    rows: list[dict[str, Any]] = []
    for seed in DEVELOPMENT_SEEDS:
        payload = _json_object(seed_root / f"seed{seed}.json")
        if payload.get("schema") != "idm_wall_v5_4_predictor_development_seed/1":
            raise ValueError(f"seed {seed} has the wrong label schema")
        if int(payload.get("seed")) != seed:
            raise ValueError(f"seed {seed} has the wrong label identity")
        cases = payload.get("cases") or []
        if tuple(int(row.get("decision_step")) for row in cases) != DECISION_STEPS:
            raise ValueError(f"seed {seed} has the wrong label decision grid")
        rows.extend(dict(row) for row in cases)
    if len(rows) != 480 or len({str(row.get("case_id")) for row in rows}) != 480:
        raise ValueError("label source lacks 480 unique case IDs")
    return summary, rows


def join_capture_labels(
    capture_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Exact case-id join; positional or seed-only joins are prohibited."""

    capture_by_id = {str(row.get("case_id")): row for row in capture_rows}
    label_by_id = {str(row.get("case_id")): row for row in label_rows}
    if len(capture_by_id) != len(capture_rows) or len(label_by_id) != len(label_rows):
        raise ValueError("duplicate case ID before label join")
    if set(capture_by_id) != set(label_by_id):
        raise ValueError("capture and label case IDs do not match exactly")
    joined = []
    for case_id in sorted(capture_by_id):
        capture = capture_by_id[case_id]
        label = label_by_id[case_id]
        if int(capture["seed"]) != int(label["seed"]) or int(
            capture["decision_step"]
        ) != int(label["decision_step"]):
            raise ValueError(f"metadata mismatch for exact case join {case_id}")
        if bool(capture["gate_f_admitted"]) != bool(
            (label.get("predecision_supervisor") or {}).get("admitted")
        ):
            raise ValueError(f"Gate-F admission mismatch for {case_id}")
        joined.append({"case_id": case_id, "capture": capture, "label": label})
    return joined


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric(value[key], name))
    elif isinstance(value, (list, tuple)):
        # Scalar physical blocks may contain short fixed vectors/quantiles.
        for index, item in enumerate(value):
            name = f"{prefix}[{index}]"
            out.update(_flatten_numeric(item, name))
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        out[prefix] = float(value)
    elif value is not None:
        raise ValueError(f"non-numeric feature payload at {prefix}")
    return out


def _matrix_from_rows(rows: Sequence[Mapping[str, float]], *, prefix: str) -> FeatureMatrix:
    if not rows:
        raise ValueError(f"empty feature block {prefix}")
    names = tuple(sorted(rows[0]))
    if not names:
        raise ValueError(f"feature block {prefix} is empty")
    for index, row in enumerate(rows):
        if tuple(sorted(row)) != names:
            raise ValueError(f"feature schema drift in {prefix} row {index}")
    qualified_names = tuple(f"{prefix}::{name}" for name in names)
    _validate_feature_names(
        {name: rows[0][raw] for name, raw in zip(qualified_names, names)},
        context=prefix,
    )
    matrix = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"non-finite feature matrix {prefix}")
    return FeatureMatrix(qualified_names, matrix)


def _concatenate(*matrices: FeatureMatrix) -> FeatureMatrix:
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for matrix in matrices:
        names.extend(matrix.names)
        arrays.append(matrix.values)
    if len(names) != len(set(names)):
        raise ValueError("combined feature blocks have duplicate names")
    return FeatureMatrix(tuple(names), np.column_stack(arrays))


def prepare_population(joined: Sequence[Mapping[str, Any]]) -> DevelopmentPopulation:
    admitted = [row for row in joined if bool(row["capture"].get("gate_f_admitted"))]
    if not admitted:
        raise ValueError("no Gate-F-admitted v2 development cases")

    exact_rows: list[dict[str, float]] = []
    deployable_rows: list[dict[str, float]] = []
    physical_rows: dict[str, list[dict[str, float]]] = {
        name: [] for name in PHYSICAL_SUBBLOCKS
    }
    shadow_rows: list[dict[str, float]] = []
    dose: list[int] = []
    safety: list[int] = []
    benefit: list[int] = []
    effects: dict[str, list[float]] = {
        name: [] for name in (*SECONDARY_EFFECTS, *AUXILIARY_EFFECTS)
    }
    groups: list[str] = []
    for joined_row in admitted:
        capture = joined_row["capture"]
        label = joined_row["label"]
        # Historical v1 rows store physical features under ``features`` but
        # derive the two context fields from top-level case metadata.  Reuse
        # the frozen accessor so the matched baseline is byte-semantically
        # identical to the completed v1 diagnosis.
        exact = {name: float(feature_value(label, name)) for name in IDM_PRIMARY_FEATURES}
        exact_rows.append(exact)
        deployable_rows.append(
            {name: value for name, value in exact.items() if name not in {"health", "health_change_30"}}
        )
        base_blocks = (capture.get("baseline_observation") or {}).get("feature_blocks") or {}
        for name in PHYSICAL_SUBBLOCKS:
            if name not in base_blocks:
                raise ValueError(f"{joined_row['case_id']} lacks physical block {name}")
            physical_rows[name].append(_flatten_numeric(base_blocks[name]))
        shadow = capture.get("shadow_observation") or {}
        shadow_blocks = shadow.get("feature_blocks") or {}
        if SHADOW_SUBBLOCK not in shadow_blocks:
            raise ValueError(f"{joined_row['case_id']} lacks exact shadow block")
        shadow_rows.append(_flatten_numeric(shadow_blocks[SHADOW_SUBBLOCK]))

        oracle = label.get("safety_service_oracle") or {}
        dose.append(int(dose_qualified(label)))
        safety.append(int(bool(oracle.get("safety_admissible"))))
        benefit.append(int(bool(oracle.get("wall_selected"))))
        effect_values = oracle.get("effects") or {}
        for name in (*SECONDARY_EFFECTS, *AUXILIARY_EFFECTS):
            effects[name].append(float(effect_values[name]))
        groups.append(f"seed{int(label['seed'])}")

    exact_matrix = _matrix_from_rows(exact_rows, prefix=EXACT_BASELINE)
    deployable_matrix = _matrix_from_rows(deployable_rows, prefix=DEPLOYABLE_BASELINE)
    subblocks = {
        name: _matrix_from_rows(values, prefix=name) for name, values in physical_rows.items()
    }
    physical_matrix = _concatenate(*(subblocks[name] for name in PHYSICAL_SUBBLOCKS))
    shadow_matrix = _matrix_from_rows(shadow_rows, prefix=SHADOW_SUBBLOCK)
    combined_matrix = _concatenate(physical_matrix, shadow_matrix)
    for forbidden in ("health", "health_change_30"):
        if any(forbidden in name.lower() for name in (*physical_matrix.names, *shadow_matrix.names, *combined_matrix.names)):
            raise ValueError(f"v2 physical/shadow candidate contains prohibited primary {forbidden}")
    return DevelopmentPopulation(
        rows=tuple(admitted),
        groups=np.asarray(groups),
        dose=np.asarray(dose, dtype=int),
        safety=np.asarray(safety, dtype=int),
        benefit=np.asarray(benefit, dtype=int),
        effects={name: np.asarray(values, dtype=float) for name, values in effects.items()},
        blocks={
            EXACT_BASELINE: exact_matrix,
            DEPLOYABLE_BASELINE: deployable_matrix,
            PHYSICAL_BLOCK: FeatureMatrix(physical_matrix.names, physical_matrix.values),
            SHADOW_BLOCK: FeatureMatrix(shadow_matrix.names, shadow_matrix.values),
            COMBINED_BLOCK: FeatureMatrix(combined_matrix.names, combined_matrix.values),
        },
        physical_subblocks=subblocks,
    )


def development_outer_folds(
    population: DevelopmentPopulation,
    *,
    n_splits: int = OUTER_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = np.zeros(len(population.rows), dtype=int)
    qualified = population.dose == 1
    safe = qualified & (population.safety == 1)
    strata[qualified & ~safe] = 1
    strata[safe & (population.benefit == 0)] = 2
    strata[safe & (population.benefit == 1)] = 3
    return _stratified_group_folds(
        np.arange(len(population.rows)),
        strata,
        population.groups,
        n_splits=n_splits,
        random_state=RANDOM_STATE,
    )


def _stratified_group_folds(
    indices: np.ndarray,
    strata: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    unique_groups = np.unique(groups[indices])
    folds_count = min(int(n_splits), len(unique_groups))
    if folds_count < 2:
        raise ValueError("at least two independent seed groups are required")
    splitter = StratifiedGroupKFold(
        n_splits=folds_count, shuffle=True, random_state=random_state
    )
    folds = []
    for local_train, local_test in splitter.split(
        np.zeros((len(indices), 1)), strata[indices], groups[indices]
    ):
        train = indices[local_train]
        test = indices[local_test]
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("seed leakage across grouped folds")
        folds.append((train, test))
    return folds


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import recall_score

    report = probability_metrics(labels, probabilities)
    report["recall_at_0_5"] = float(
        recall_score(labels, probabilities >= 0.5, zero_division=0)
    )
    return report


def _fit_predict_inner(
    matrix: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    model_family: str,
    *,
    random_state: int,
) -> np.ndarray:
    from sklearn.base import clone

    probability = np.full(len(labels), np.nan, dtype=float)
    estimator = make_candidate_estimator(model_family, random_state=random_state)
    for fold_index, (train_all, test_all) in enumerate(inner_folds):
        train = train_all[eligible[train_all]]
        if len(np.unique(labels[train])) != 2:
            raise ValueError(
                f"inner fold {fold_index} lacks one class for {model_family}"
            )
        model = clone(estimator).fit(matrix[train], labels[train])
        probability[test_all] = model.predict_proba(matrix[test_all])[:, 1]
    covered = np.concatenate([test for _, test in inner_folds])
    if not np.all(np.isfinite(probability[covered])):
        raise AssertionError("inner grouped probabilities are incomplete")
    return probability


def _select_inner_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    random_state: int,
) -> tuple[str, np.ndarray, list[dict[str, Any]]]:
    reports = []
    predictions: dict[str, np.ndarray] = {}
    covered = np.concatenate([test for _, test in inner_folds])
    evaluation = covered[eligible[covered]]
    for order, model_family in enumerate(TABULAR_MODEL_FAMILIES):
        try:
            probability = _fit_predict_inner(
                matrix,
                labels,
                eligible,
                inner_folds,
                model_family,
                random_state=random_state + order,
            )
            metric = _metrics(labels[evaluation], probability[evaluation])
            reports.append(
                {
                    "model_family": model_family,
                    "model_order": order,
                    "metrics": metric,
                    "error": None,
                }
            )
            predictions[model_family] = probability
        except (ValueError, FloatingPointError) as error:
            reports.append(
                {
                    "model_family": model_family,
                    "model_order": order,
                    "metrics": None,
                    "error": str(error),
                }
            )
    valid = [row for row in reports if row["metrics"] is not None]
    if not valid:
        raise ValueError("every bounded model failed inside the grouped fold")
    selected = max(
        valid,
        key=lambda row: (
            float(row["metrics"].get("roc_auc", -1.0)),
            float(row["metrics"].get("average_precision", -1.0)),
            -int(row["model_order"]),
        ),
    )
    family = str(selected["model_family"])
    return family, predictions[family], reports


def _policy_metrics(
    population: DevelopmentPopulation,
    command: np.ndarray,
    probabilities: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    command = np.asarray(command, dtype=bool)
    count = int(command.sum())
    selected_groups = population.groups[command]
    effects = population.effects
    catastrophic = (
        (effects.get("collision_excess_vehicle_count", np.zeros(len(command))) > 0)
        | (effects.get("teleport_excess_vehicle_count", np.zeros(len(command))) > 0)
    )
    physically_harmful = (
        (effects["hard_brake_onset_excess"] > 0)
        | (effects["hard_brake_excess_vehicle_steps"] > 0)
        | (effects["low_ttc_excess_steps"] > 0)
        | catastrophic
    )

    def inventory(mask: np.ndarray) -> list[dict[str, Any]]:
        rows = []
        for index in np.flatnonzero(command & mask):
            joined = population.rows[index]
            label = joined["label"]
            item = {
                "case_id": str(joined["case_id"]),
                "seed": int(label["seed"]),
                "decision_step": int(label["decision_step"]),
                "dose_qualified": bool(population.dose[index]),
                "safety_admissible": bool(population.safety[index]),
                "joint_beneficial": bool(population.benefit[index]),
                "effects": {
                    name: float(values[index]) for name, values in effects.items()
                },
            }
            if probabilities is not None:
                item["oof_probabilities"] = {
                    name: float(values[index]) for name, values in probabilities.items()
                }
            rows.append(item)
        return rows

    beneficial_seed_counts = Counter(
        str(group)
        for group, take in zip(population.groups, command & (population.benefit == 1))
        if take
    )
    return {
        "command_case_count": count,
        "command_seed_count": len(set(str(value) for value in selected_groups)),
        "coverage_fraction_of_gate_f_admissions": count / len(command) if len(command) else 0.0,
        "selected_dose_success_rate": float(np.mean(population.dose[command])) if count else 0.0,
        "selected_safety_rate": float(np.mean(population.safety[command])) if count else 0.0,
        "joint_benefit_precision": float(np.mean(population.benefit[command])) if count else 0.0,
        "feasible_always_wall_dose_success_rate": float(np.mean(population.dose)),
        "feasible_always_wall_safety_rate": float(np.mean(population.safety)),
        "feasible_always_wall_joint_benefit_rate": float(np.mean(population.benefit)),
        "dose_success_enrichment_over_feasible_always_wall": (
            (float(np.mean(population.dose[command])) if count else 0.0)
            - float(np.mean(population.dose))
        ),
        "safety_enrichment_over_feasible_always_wall": (
            (float(np.mean(population.safety[command])) if count else 0.0)
            - float(np.mean(population.safety))
        ),
        "joint_benefit_enrichment_over_feasible_always_wall": (
            (float(np.mean(population.benefit[command])) if count else 0.0)
            - float(np.mean(population.benefit))
        ),
        "joint_benefit_recall": (
            int(np.sum(command & (population.benefit == 1))) / int(np.sum(population.benefit == 1))
            if int(np.sum(population.benefit == 1))
            else 0.0
        ),
        "selected_failed_dose_case_count": int(np.sum(command & (population.dose == 0))),
        "selected_unsafe_case_count": int(np.sum(command & (population.safety == 0))),
        "selected_nonbeneficial_case_count": int(np.sum(command & (population.benefit == 0))),
        "selected_physically_harmful_case_count": int(np.sum(command & physically_harmful)),
        "selected_excess_collision_or_teleport_case_count": int(np.sum(command & catastrophic)),
        "beneficial_command_seed_count": len(beneficial_seed_counts),
        "largest_single_beneficial_seed_fraction": (
            max(beneficial_seed_counts.values()) / sum(beneficial_seed_counts.values())
            if beneficial_seed_counts
            else 0.0
        ),
        "inventories": {
            "failed_dose": inventory(population.dose == 0),
            "unsafe": inventory(population.safety == 0),
            "nonbeneficial": inventory(population.benefit == 0),
            "physically_harmful": inventory(physically_harmful),
            "collision_or_teleport": inventory(catastrophic),
        },
    }


def _select_policy_thresholds(
    population: DevelopmentPopulation,
    include: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    values = np.round(np.arange(0.30, 0.901, 0.05), 2)
    reports = []
    for dose_threshold in values:
        dose_take = probabilities["dose"] >= dose_threshold
        for safety_threshold in values:
            safe_take = dose_take & (probabilities["safety"] >= safety_threshold)
            for benefit_threshold in values:
                command = include & safe_take & (probabilities["benefit"] >= benefit_threshold)
                metric = _policy_metrics(population, command)
                coverage_ok = (
                    metric["command_case_count"] >= CONFIRMATORY_GATES["minimum_selected_cases"]
                    and metric["command_seed_count"] >= CONFIRMATORY_GATES["minimum_selected_seeds"]
                )
                passes = bool(
                    coverage_ok
                    and metric["selected_dose_success_rate"]
                    >= CONFIRMATORY_GATES["minimum_selected_dose_success_rate"]
                    and metric["selected_safety_rate"]
                    >= CONFIRMATORY_GATES["minimum_selected_safety_rate"]
                    and metric["joint_benefit_precision"]
                    >= CONFIRMATORY_GATES["minimum_joint_benefit_precision"]
                    and metric["selected_excess_collision_or_teleport_case_count"] == 0
                )
                reports.append(
                    {
                        "dose_threshold": float(dose_threshold),
                        "safety_threshold": float(safety_threshold),
                        "benefit_threshold": float(benefit_threshold),
                        **{key: value for key, value in metric.items() if key != "inventories"},
                        "coverage_eligible": coverage_ok,
                        "passes_inner_confirmatory_rates": passes,
                    }
                )
    eligible = [row for row in reports if row["coverage_eligible"]]
    if not eligible:
        return None, reports
    passing = [row for row in eligible if row["passes_inner_confirmatory_rates"]]
    pool = passing or eligible
    selected = max(
        pool,
        key=lambda row: (
            int(row["passes_inner_confirmatory_rates"]),
            row["joint_benefit_precision"],
            row["selected_safety_rate"],
            row["selected_dose_success_rate"],
            row["joint_benefit_recall"],
            row["command_seed_count"],
            -row["command_case_count"],
            row["dose_threshold"],
            row["safety_threshold"],
            row["benefit_threshold"],
        ),
    )
    return dict(selected), reports


def nested_block_screen(
    population: DevelopmentPopulation,
    block: FeatureMatrix,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    permutation_repeats: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    """Nested model and threshold selection for one immutable feature block."""

    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    n = len(population.rows)
    probabilities = {target: np.full(n, np.nan, dtype=float) for target in TARGET_NAMES}
    command = np.zeros(n, dtype=bool)
    target_folds: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGET_NAMES}
    policy_folds = []
    permutation: dict[str, dict[str, list[float]]] = {
        target: defaultdict(list) for target in TARGET_NAMES
    }
    strata = np.zeros(n, dtype=int)
    qualified = population.dose == 1
    safe = qualified & (population.safety == 1)
    strata[qualified & ~safe] = 1
    strata[safe & (population.benefit == 0)] = 2
    strata[safe & (population.benefit == 1)] = 3

    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        inner_folds = _stratified_group_folds(
            outer_train,
            strata,
            population.groups,
            n_splits=INNER_FOLDS,
            random_state=RANDOM_STATE + fold_index,
        )
        inner_probabilities: dict[str, np.ndarray] = {}
        for target_index, target in enumerate(TARGET_NAMES):
            labels = population.target(target)
            eligible = population.target_mask(target)
            family, inner_probability, candidates = _select_inner_model(
                block.values,
                labels,
                eligible,
                inner_folds,
                random_state=RANDOM_STATE + 100 * fold_index + 10 * target_index,
            )
            inner_probabilities[target] = inner_probability
            train = outer_train[eligible[outer_train]]
            model = clone(
                make_candidate_estimator(
                    family,
                    random_state=RANDOM_STATE + 1000 + 10 * fold_index + target_index,
                )
            ).fit(block.values[train], labels[train])
            probabilities[target][outer_test] = model.predict_proba(
                block.values[outer_test]
            )[:, 1]
            test = outer_test[eligible[outer_test]]
            fold_metric = _metrics(labels[test], probabilities[target][test])
            target_folds[target].append(
                {
                    "fold": fold_index,
                    "selected_model_family": family,
                    "inner_candidates": candidates,
                    "outer_metrics": fold_metric,
                    "train_seed_count": len(set(population.groups[outer_train])),
                    "test_seed_count": len(set(population.groups[outer_test])),
                    "test_seed_groups": sorted(set(str(value) for value in population.groups[outer_test])),
                }
            )
            if permutation_repeats > 0 and len(np.unique(labels[test])) == 2:
                base_auc = float(roc_auc_score(labels[test], probabilities[target][test]))
                rng = np.random.default_rng(
                    RANDOM_STATE + 10000 + fold_index * 101 + target_index
                )
                for feature_index, feature_name in enumerate(block.names):
                    decreases = []
                    for _ in range(permutation_repeats):
                        permuted = block.values[outer_test].copy()
                        rng.shuffle(permuted[:, feature_index])
                        perm_prob = model.predict_proba(permuted)[:, 1]
                        local = np.flatnonzero(eligible[outer_test])
                        decreases.append(
                            base_auc
                            - float(roc_auc_score(labels[test], perm_prob[local]))
                        )
                    permutation[target][feature_name].append(float(np.mean(decreases)))

        thresholds, _sweep = _select_policy_thresholds(
            population,
            np.isin(np.arange(n), outer_train),
            inner_probabilities,
        )
        if thresholds is not None:
            test_command = (
                (probabilities["dose"][outer_test] >= float(thresholds["dose_threshold"]))
                & (
                    probabilities["safety"][outer_test]
                    >= float(thresholds["safety_threshold"])
                )
                & (
                    probabilities["benefit"][outer_test]
                    >= float(thresholds["benefit_threshold"])
                )
            )
            command[outer_test] = test_command
        policy_folds.append(
            {
                "fold": fold_index,
                "selected_thresholds": thresholds,
                "threshold_selection_available": thresholds is not None,
                "outer_policy": _policy_metrics(
                    population,
                    np.isin(np.arange(n), outer_test) & command,
                    probabilities,
                ),
            }
        )

    if any(not np.all(np.isfinite(values)) for values in probabilities.values()):
        raise AssertionError("nested grouped OOF predictions are incomplete")
    targets = {}
    for target in TARGET_NAMES:
        eligible = population.target_mask(target)
        labels = population.target(target)
        targets[target] = {
            "eligible_case_count": int(eligible.sum()),
            "independent_seed_count": len(set(population.groups[eligible])),
            "nested_metrics": _metrics(labels[eligible], probabilities[target][eligible]),
            "selected_model_family_counts": dict(
                Counter(row["selected_model_family"] for row in target_folds[target])
            ),
            "folds": target_folds[target],
            "held_out_permutation_importance": sorted(
                (
                    {
                        "feature": feature,
                        "mean_auc_decrease": float(np.mean(values)),
                        "fold_count": len(values),
                    }
                    for feature, values in permutation[target].items()
                ),
                key=lambda row: row["mean_auc_decrease"],
                reverse=True,
            ),
        }
    return (
        {
            "feature_count": len(block.names),
            "feature_names": list(block.names),
            "targets": targets,
            "cross_fitted_policy": _policy_metrics(population, command, probabilities),
            "policy_folds": policy_folds,
        },
        probabilities,
        command,
    )


def _cluster_bootstrap_auc_gain(
    labels: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    repeats: int,
    random_state: int,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    unique = np.unique(groups[eligible])
    gains = []
    for _ in range(max(0, repeats)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(eligible & (groups == group)) for group in sampled]
        )
        if len(np.unique(labels[indices])) < 2:
            continue
        gains.append(
            float(roc_auc_score(labels[indices], candidate[indices]))
            - float(roc_auc_score(labels[indices], baseline[indices]))
        )
    return {
        "requested_repeats": int(repeats),
        "valid_repeats": len(gains),
        "mean_auc_gain": float(np.mean(gains)) if gains else None,
        "auc_gain_95": (
            [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))]
            if gains
            else None
        ),
    }


def matched_comparisons(
    population: DevelopmentPopulation,
    screen_reports: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    *,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    baseline_report = screen_reports[EXACT_BASELINE]
    baseline_predictions = predictions[EXACT_BASELINE]
    report: dict[str, Any] = {}
    for block_name in CORE_BLOCKS:
        if block_name == EXACT_BASELINE:
            continue
        targets = {}
        for target in TARGET_NAMES:
            eligible = population.target_mask(target)
            labels = population.target(target)
            candidate_auc = float(
                screen_reports[block_name]["targets"][target]["nested_metrics"]["roc_auc"]
            )
            baseline_auc = float(
                baseline_report["targets"][target]["nested_metrics"]["roc_auc"]
            )
            leave_one_seed = []
            for seed in np.unique(population.groups):
                include = eligible & (population.groups != seed)
                from sklearn.metrics import roc_auc_score

                if len(np.unique(labels[include])) < 2:
                    continue
                gain = float(
                    roc_auc_score(labels[include], predictions[block_name][target][include])
                    - roc_auc_score(labels[include], baseline_predictions[target][include])
                )
                leave_one_seed.append({"excluded_seed_group": str(seed), "auc_gain": gain})
            targets[target] = {
                "exact_v1_baseline_auc": baseline_auc,
                "candidate_auc": candidate_auc,
                "matched_auc_gain": candidate_auc - baseline_auc,
                "clustered_bootstrap": _cluster_bootstrap_auc_gain(
                    labels,
                    eligible,
                    population.groups,
                    predictions[block_name][target],
                    baseline_predictions[target],
                    repeats=bootstrap_repeats,
                    random_state=RANDOM_STATE + len(target) + len(block_name),
                ),
                "leave_one_seed_out_auc_gain": leave_one_seed,
                "minimum_leave_one_seed_out_auc_gain": min(
                    (row["auc_gain"] for row in leave_one_seed), default=None
                ),
            }
        report[block_name] = {"targets": targets}
    return report


def shuffled_label_controls(
    population: DevelopmentPopulation,
    block: FeatureMatrix,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    repeats: int,
) -> dict[str, Any]:
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(RANDOM_STATE + 991)
    report = {}
    for target_index, target in enumerate(TARGET_NAMES):
        labels = population.target(target)
        eligible = population.target_mask(target)
        values = []
        for repeat in range(max(0, repeats)):
            shuffled = labels.copy()
            shuffled_values = shuffled[eligible].copy()
            rng.shuffle(shuffled_values)
            shuffled[eligible] = shuffled_values
            probability = np.full(len(labels), np.nan, dtype=float)
            valid = True
            for fold_index, (train_all, test_all) in enumerate(outer_folds):
                train = train_all[eligible[train_all]]
                if len(np.unique(shuffled[train])) < 2:
                    valid = False
                    break
                model = clone(
                    make_candidate_estimator(
                        "regularized_logistic",
                        random_state=RANDOM_STATE + repeat * 17 + fold_index + target_index,
                    )
                ).fit(block.values[train], shuffled[train])
                probability[test_all] = model.predict_proba(block.values[test_all])[:, 1]
            if valid and np.all(np.isfinite(probability[eligible])):
                values.append(float(roc_auc_score(shuffled[eligible], probability[eligible])))
        report[target] = {
            "requested_repeats": int(repeats),
            "valid_repeats": len(values),
            "null_auc_mean": float(np.mean(values)) if values else None,
            "null_auc_95": (
                [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                if values
                else None
            ),
        }
    return report


def feature_block_diagnostics(block: FeatureMatrix) -> dict[str, Any]:
    """Constants, exact duplicates, and near rank-redundancy for one block."""

    from scipy.stats import rankdata

    rows = []
    by_bytes: dict[bytes, list[str]] = defaultdict(list)
    variable = []
    for index, name in enumerate(block.names):
        values = block.values[:, index]
        unique = int(len(np.unique(values)))
        rows.append(
            {
                "name": name,
                "unique_count": unique,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
                "constant": unique == 1,
            }
        )
        by_bytes[np.ascontiguousarray(values).tobytes()].append(name)
        if unique > 1:
            variable.append(index)
    redundancy = []
    if len(variable) > 1:
        ranked = np.column_stack(
            [rankdata(block.values[:, index]) for index in variable]
        )
        correlation = np.corrcoef(ranked, rowvar=False)
        for left in range(len(variable)):
            for right in range(left + 1, len(variable)):
                value = float(correlation[left, right])
                if math.isfinite(value) and abs(value) >= 0.98:
                    redundancy.append(
                        {
                            "left": block.names[variable[left]],
                            "right": block.names[variable[right]],
                            "spearman": value,
                        }
                    )
    return {
        "feature_count": len(block.names),
        "constant_feature_count": sum(row["constant"] for row in rows),
        "exact_duplicate_groups": [
            names for names in by_bytes.values() if len(names) > 1
        ],
        "absolute_spearman_at_least_0_98": redundancy,
        "features": rows,
    }


def cross_seed_nearest_neighbor_observability(
    population: DevelopmentPopulation,
    block: FeatureMatrix,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Quantify close cross-seed states with opposing frozen labels."""

    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    report = {}
    for target in TARGET_NAMES:
        labels = population.target(target)
        eligible = population.target_mask(target)
        nearest_label = np.full(len(labels), np.nan)
        five_probability = np.full(len(labels), np.nan)
        nearest_distance = np.full(len(labels), np.nan)
        pairs = []
        for train_all, test_all in outer_folds:
            train = train_all[eligible[train_all]]
            test = test_all[eligible[test_all]]
            scaler = StandardScaler().fit(block.values[train])
            train_matrix = scaler.transform(block.values[train])
            test_matrix = scaler.transform(block.values[test])
            for local, row_index in enumerate(test):
                distance = np.sqrt(
                    np.mean((train_matrix - test_matrix[local]) ** 2, axis=1)
                )
                order = np.argsort(distance)
                nearest = train[order[0]]
                nearest_label[row_index] = labels[nearest]
                nearest_distance[row_index] = distance[order[0]]
                neighbors = train[order[: min(5, len(order))]]
                five_probability[row_index] = float(np.mean(labels[neighbors]))
                opposite = [position for position in order if labels[train[position]] != labels[row_index]]
                if opposite:
                    other = train[opposite[0]]
                    pairs.append(
                        {
                            "case_id": str(population.rows[row_index]["case_id"]),
                            "opposite_case_id": str(population.rows[other]["case_id"]),
                            "seed_group": str(population.groups[row_index]),
                            "opposite_seed_group": str(population.groups[other]),
                            "standardized_rms_distance": float(distance[opposite[0]]),
                            "label": int(labels[row_index]),
                            "opposite_label": int(labels[other]),
                        }
                    )
        indices = np.flatnonzero(eligible)
        threshold = float(np.quantile(nearest_distance[indices], 0.25))
        close = indices[nearest_distance[indices] <= threshold]
        pairs.sort(key=lambda row: row["standardized_rms_distance"])
        report[target] = {
            "eligible_case_count": len(indices),
            "one_nearest_neighbor_accuracy": float(
                accuracy_score(labels[indices], nearest_label[indices])
            ),
            "five_neighbor_probability_auc": float(
                roc_auc_score(labels[indices], five_probability[indices])
            ),
            "nearest_distance_first_quartile": threshold,
            "close_neighbor_opposite_label_fraction": float(
                np.mean(nearest_label[close] != labels[close])
            ),
            "closest_opposite_label_pairs": pairs[:20],
        }
    return report


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mean_absolute_error": float(mean_absolute_error(observed, predicted)),
        "root_mean_squared_error": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson_correlation": (
            float(np.corrcoef(observed, predicted)[0, 1])
            if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0
            else 0.0
        ),
    }


def secondary_effect_regressions(
    population: DevelopmentPopulation,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    eligible = population.dose == 1
    report = {}
    for block_name in (EXACT_BASELINE, PHYSICAL_BLOCK, SHADOW_BLOCK, COMBINED_BLOCK):
        matrix = population.blocks[block_name].values
        effects = {}
        for effect_index, effect_name in enumerate(SECONDARY_EFFECTS):
            observed = population.effects[effect_name]
            predicted = np.full(len(observed), np.nan, dtype=float)
            folds = []
            for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
                inner_indices = outer_train[eligible[outer_train]]
                inner_folds = _stratified_group_folds(
                    inner_indices,
                    np.digitize(
                        observed,
                        np.quantile(observed[inner_indices], [0.25, 0.5, 0.75]),
                    ),
                    population.groups,
                    n_splits=INNER_FOLDS,
                    random_state=RANDOM_STATE + 500 + fold_index + effect_index,
                )
                candidates = []
                for alpha in (1.0, 10.0, 100.0):
                    inner_pred = np.full(len(observed), np.nan, dtype=float)
                    for train, test in inner_folds:
                        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                        model.fit(matrix[train], observed[train])
                        inner_pred[test] = model.predict(matrix[test])
                    candidates.append(
                        {
                            "alpha": alpha,
                            "mae": float(
                                np.mean(np.abs(observed[inner_indices] - inner_pred[inner_indices]))
                            ),
                        }
                    )
                selected = min(candidates, key=lambda row: (row["mae"], row["alpha"]))
                train = outer_train[eligible[outer_train]]
                test = outer_test[eligible[outer_test]]
                model = make_pipeline(StandardScaler(), Ridge(alpha=float(selected["alpha"])))
                model.fit(matrix[train], observed[train])
                predicted[test] = model.predict(matrix[test])
                folds.append(
                    {
                        "fold": fold_index,
                        "selected_alpha": selected["alpha"],
                        "inner_candidates": candidates,
                    }
                )
            if not np.all(np.isfinite(predicted[eligible])):
                raise AssertionError("secondary paired-effect predictions are incomplete")
            effects[effect_name] = {
                "eligible_case_count": int(eligible.sum()),
                "metrics": _regression_metrics(observed[eligible], predicted[eligible]),
                "folds": folds,
            }
        report[block_name] = effects
    return report


def _raw_tensor_payload(
    row: Mapping[str, Any], history_key: str = "history_20s"
) -> tuple[np.ndarray, np.ndarray] | None:
    history = ((row.get("capture") or {}).get("baseline_observation") or {}).get(
        history_key
    )
    if not isinstance(history, Mapping):
        return None
    values = history.get("tensor_values", history.get("values"))
    mask = history.get(
        "tensor_mask", history.get("presence_mask", history.get("mask"))
    )
    if values is None or mask is None:
        return None
    array = np.asarray(values, dtype=float)
    masks = np.asarray(mask, dtype=float)
    if array.ndim != 3 or masks.shape not in {array.shape, array.shape[:2], array.shape[:2] + (1,)}:
        raise ValueError("history_20s raw tensor/mask has an unsupported shape")
    if masks.ndim == 2:
        masks = masks[..., None]
    if masks.shape[-1] == 1 and array.shape[-1] != 1:
        masks = np.broadcast_to(masks, array.shape)
    # Presence masks intentionally vanish when a trajectory exits.  The
    # extractor therefore carries a separate event mask so the exit-event
    # channel is not silently erased by masked pooling.  This remains fully
    # predecision and anonymous; it only changes which recorded numeric event
    # cells are valid.
    channel_names = history.get("channel_names", history.get("slot_feature_names"))
    event_mask = history.get("event_mask")
    if channel_names is not None:
        if not isinstance(channel_names, list) or len(channel_names) != array.shape[-1]:
            raise ValueError("history raw tensor has inconsistent channel names")
        if event_mask is not None:
            event = np.asarray(event_mask, dtype=float)
            if event.shape != array.shape[:2]:
                raise ValueError("history raw tensor has an inconsistent event mask")
            masks = np.array(masks, copy=True)
            for index, name in enumerate(channel_names):
                if str(name).endswith("_event"):
                    masks[..., index] = np.maximum(masks[..., index], event)
    if not np.all(np.isfinite(array)) or not np.all(np.isfinite(masks)):
        raise ValueError("history_20s raw tensor is non-finite")
    return array, masks


def _sequence_matrix(
    payloads: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Masked slot pooling while preserving the complete temporal axis."""

    time_channels = {(value.shape[0], value.shape[2]) for value, _mask in payloads}
    if len(time_channels) != 1:
        raise ValueError("raw tensors do not share frozen time/channel dimensions")
    time_count, channel_count = next(iter(time_channels))
    rows = []
    slot_counts = []
    for values, mask in payloads:
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[-1] == 1 and values.shape[-1] != 1:
            mask = np.broadcast_to(mask, values.shape)
        if mask.shape != values.shape:
            raise ValueError("raw tensor and mask shapes are incompatible")
        slot_counts.append(int(values.shape[1]))
        weighted = values * mask
        denominator = np.maximum(np.sum(mask, axis=1), 1.0)
        mean = np.sum(weighted, axis=1) / denominator
        centered = (values - mean[:, None, :]) * mask
        std = np.sqrt(np.sum(centered * centered, axis=1) / denominator)
        masked_for_max = np.where(mask > 0.0, values, -np.inf)
        maximum = np.max(masked_for_max, axis=1)
        maximum[~np.isfinite(maximum)] = 0.0
        # Availability is retained as a real temporal channel; it is not an
        # imputation or a summary of future treatment behavior.
        presence = np.mean(mask[..., 0], axis=1, keepdims=True)
        rows.append(np.column_stack((mean, maximum, std, presence)))
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (len(payloads), time_count, channel_count * 3 + 1):
        raise AssertionError("unexpected pooled temporal tensor shape")
    return matrix, {
        "raw_time_steps": time_count,
        "raw_channel_count": channel_count,
        "pooled_channel_count": channel_count * 3 + 1,
        "minimum_anonymous_slot_count": min(slot_counts),
        "maximum_anonymous_slot_count": max(slot_counts),
        "slot_pooling": "permutation_invariant_masked_mean_max_std_plus_presence_fraction",
    }


def _torch_tcn_fit_predict(
    sequence: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    predict: np.ndarray,
    *,
    channels: int,
    layers: int,
    weight_decay: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch import nn

    torch.manual_seed(int(random_state))
    torch.set_num_threads(1)
    mean = np.mean(sequence[train], axis=(0, 1), keepdims=True)
    scale = np.std(sequence[train], axis=(0, 1), keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = (sequence - mean) / scale
    x = torch.as_tensor(normalized.transpose(0, 2, 1), dtype=torch.float32)
    y = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32)

    class CausalConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2)

        def forward(self, values):
            return self.conv(values)[..., :-2]

    class TinyTCN(nn.Module):
        def __init__(self):
            super().__init__()
            modules: list[nn.Module] = []
            current = int(sequence.shape[2])
            for _ in range(int(layers)):
                modules.extend((CausalConv(current, channels), nn.ReLU()))
                current = channels
            self.encoder = nn.Sequential(*modules)
            self.head = nn.Linear(current, 1)

        def forward(self, values):
            encoded = self.encoder(values)
            return self.head(torch.mean(encoded, dim=-1)).squeeze(-1)

    model = TinyTCN()
    positive = max(float(np.sum(labels[train] == 1)), 1.0)
    negative = max(float(np.sum(labels[train] == 0)), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(negative / positive, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=float(weight_decay)
    )
    train_tensor = torch.as_tensor(train, dtype=torch.long)
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x[train_tensor]), y[train_tensor])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_probability = torch.sigmoid(model(x[train_tensor])).cpu().numpy()
        predict_probability = torch.sigmoid(
            model(x[torch.as_tensor(predict, dtype=torch.long)])
        ).cpu().numpy()
    return train_probability.astype(float), predict_probability.astype(float)


def _temporal_convolution_window(
    population: DevelopmentPopulation,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    *,
    history_key: str,
    reference_probabilities: Mapping[str, np.ndarray] | None,
    null_repeats: int,
) -> dict[str, Any]:
    """Conditionally run the frozen small causal-convolution screen."""

    raw = [_raw_tensor_payload(row, history_key) for row in population.rows]
    available = sum(value is not None for value in raw)
    if available == 0:
        return {
            "status": f"unavailable_raw_tensor_not_captured:{history_key}",
            "available_case_count": 0,
            "required_case_count": len(raw),
            "fabricated_sequence_features": False,
        }
    if available != len(raw):
        raise ValueError("raw 20-second tensor is missing for some Gate-F admissions")
    payloads = [value for value in raw if value is not None]
    sequence, tensor_audit = _sequence_matrix(payloads)
    if importlib.util.find_spec("torch") is None:
        return {
            "status": f"unavailable_torch_backend:{history_key}",
            "available_case_count": available,
            "required_case_count": len(raw),
            "tensor_audit": tensor_audit,
            "fabricated_sequence_features": False,
        }

    folds = list(outer_folds or development_outer_folds(population))
    n = len(population.rows)
    strata = np.zeros(n, dtype=int)
    qualified = population.dose == 1
    safe = qualified & (population.safety == 1)
    strata[qualified & ~safe] = 1
    strata[safe & (population.benefit == 0)] = 2
    strata[safe & (population.benefit == 1)] = 3
    configs = (
        {"channels": 8, "layers": 1, "weight_decay": 1e-2},
        {"channels": 8, "layers": 2, "weight_decay": 1e-2},
        {"channels": 16, "layers": 2, "weight_decay": 1e-3},
    )
    target_reports = {}
    held_probabilities: dict[str, np.ndarray] = {}
    for target_index, target in enumerate(TARGET_NAMES):
        labels = population.target(target)
        eligible = population.target_mask(target)
        held_probability = np.full(n, np.nan, dtype=float)
        fold_reports = []
        for fold_index, (outer_train, outer_test) in enumerate(folds):
            inner_folds = _stratified_group_folds(
                outer_train,
                strata,
                population.groups,
                n_splits=INNER_FOLDS,
                random_state=RANDOM_STATE + 700 + fold_index,
            )
            candidates = []
            for config_index, config in enumerate(configs):
                inner_probability = np.full(n, np.nan, dtype=float)
                for inner_index, (inner_train_all, inner_test_all) in enumerate(inner_folds):
                    inner_train = inner_train_all[eligible[inner_train_all]]
                    if len(np.unique(labels[inner_train])) != 2:
                        raise ValueError("inner TCN training fold lacks a target class")
                    _train_probability, test_probability = _torch_tcn_fit_predict(
                        sequence,
                        labels,
                        inner_train,
                        inner_test_all,
                        random_state=(
                            RANDOM_STATE
                            + target_index * 10000
                            + fold_index * 1000
                            + config_index * 100
                            + inner_index
                        ),
                        **config,
                    )
                    inner_probability[inner_test_all] = test_probability
                inner_eval = outer_train[eligible[outer_train]]
                metric = _metrics(labels[inner_eval], inner_probability[inner_eval])
                candidates.append(
                    {"config": dict(config), "config_order": config_index, "metrics": metric}
                )
            selected = max(
                candidates,
                key=lambda row: (
                    float(row["metrics"].get("roc_auc", -1.0)),
                    float(row["metrics"].get("average_precision", -1.0)),
                    -int(row["config_order"]),
                ),
            )
            train = outer_train[eligible[outer_train]]
            train_probability, test_probability = _torch_tcn_fit_predict(
                sequence,
                labels,
                train,
                outer_test,
                random_state=RANDOM_STATE + 50000 + target_index * 100 + fold_index,
                **selected["config"],
            )
            held_probability[outer_test] = test_probability
            test = outer_test[eligible[outer_test]]
            local = np.flatnonzero(eligible[outer_test])
            fold_reports.append(
                {
                    "fold": fold_index,
                    "selected_config": selected["config"],
                    "inner_candidates": candidates,
                    "training_metrics": _metrics(labels[train], train_probability),
                    "held_out_metrics": _metrics(labels[test], test_probability[local]),
                }
            )
        if not np.all(np.isfinite(held_probability[eligible])):
            raise AssertionError("TCN held-out predictions are incomplete")
        held_probabilities[target] = held_probability
        leave_one_seed = []
        for seed_group in np.unique(population.groups):
            include = eligible & (population.groups != seed_group)
            if len(np.unique(labels[include])) < 2:
                continue
            from sklearn.metrics import roc_auc_score

            candidate_auc = float(
                roc_auc_score(labels[include], held_probability[include])
            )
            item = {
                "excluded_seed_group": str(seed_group),
                "candidate_auc": candidate_auc,
            }
            if reference_probabilities is not None:
                reference_auc = float(
                    roc_auc_score(
                        labels[include], reference_probabilities[target][include]
                    )
                )
                item.update(
                    reference_physical_auc=reference_auc,
                    auc_gain_over_physical=candidate_auc - reference_auc,
                )
            leave_one_seed.append(item)
        reference_auc = None
        matched_gain = None
        if reference_probabilities is not None:
            from sklearn.metrics import roc_auc_score

            reference_auc = float(
                roc_auc_score(labels[eligible], reference_probabilities[target][eligible])
            )
            matched_gain = float(
                roc_auc_score(labels[eligible], held_probability[eligible]) - reference_auc
            )
        target_reports[target] = {
            "eligible_case_count": int(eligible.sum()),
            "held_out_metrics": _metrics(labels[eligible], held_probability[eligible]),
            "selected_config_counts": dict(
                Counter(json.dumps(row["selected_config"], sort_keys=True) for row in fold_reports)
            ),
            "matched_physical_reference_auc": reference_auc,
            "matched_auc_gain_over_physical": matched_gain,
            "leave_one_seed_out_sensitivity": leave_one_seed,
            "minimum_leave_one_seed_auc_gain_over_physical": (
                min(
                    (
                        row["auc_gain_over_physical"]
                        for row in leave_one_seed
                        if "auc_gain_over_physical" in row
                    ),
                    default=None,
                )
            ),
            "folds": fold_reports,
        }

    # A real sequence null uses the raw tensor and the same outer seed folds.
    # It deliberately fixes the smallest architecture instead of searching a
    # favorable null architecture.
    null_report = {}
    rng = np.random.default_rng(RANDOM_STATE + 88000)
    null_config = configs[0]
    from sklearn.metrics import roc_auc_score

    for target_index, target in enumerate(TARGET_NAMES):
        labels = population.target(target)
        eligible = population.target_mask(target)
        null_auc = []
        for repeat in range(max(0, null_repeats)):
            shuffled = labels.copy()
            values = shuffled[eligible].copy()
            rng.shuffle(values)
            shuffled[eligible] = values
            probability = np.full(n, np.nan, dtype=float)
            valid = True
            for fold_index, (train_all, test_all) in enumerate(folds):
                train = train_all[eligible[train_all]]
                if len(np.unique(shuffled[train])) != 2:
                    valid = False
                    break
                _train_probability, test_probability = _torch_tcn_fit_predict(
                    sequence,
                    shuffled,
                    train,
                    test_all,
                    random_state=(
                        RANDOM_STATE
                        + 90000
                        + target_index * 1000
                        + repeat * 10
                        + fold_index
                    ),
                    **null_config,
                )
                probability[test_all] = test_probability
            if valid and np.all(np.isfinite(probability[eligible])):
                null_auc.append(
                    float(roc_auc_score(shuffled[eligible], probability[eligible]))
                )
        null_report[target] = {
            "fixed_null_config": dict(null_config),
            "requested_repeats": int(null_repeats),
            "valid_repeats": len(null_auc),
            "null_auc_mean": float(np.mean(null_auc)) if null_auc else None,
            "null_auc_95": (
                [
                    float(np.quantile(null_auc, 0.025)),
                    float(np.quantile(null_auc, 0.975)),
                ]
                if null_auc
                else None
            ),
        }
    qualifying_targets = []
    if reference_probabilities is not None:
        for target in ("safety", "benefit"):
            target_report = target_reports[target]
            null_mean = null_report[target]["null_auc_mean"]
            if (
                target_report["matched_auc_gain_over_physical"] is not None
                and target_report["matched_auc_gain_over_physical"] >= 0.03
                and target_report["minimum_leave_one_seed_auc_gain_over_physical"] is not None
                and target_report["minimum_leave_one_seed_auc_gain_over_physical"] >= 0.0
                and null_mean is not None
                and null_mean <= 0.55
            ):
                qualifying_targets.append(target)
    return {
        "status": f"executed_nested_seed_grouped:{history_key}",
        "available_case_count": available,
        "required_case_count": len(raw),
        "tensor_audit": tensor_audit,
        "candidate_library": list(configs),
        "targets": target_reports,
        "shuffled_label_controls": null_report,
        "qualifying_targets_over_physical": qualifying_targets,
        "observation_gate_passed": bool(qualifying_targets),
        "fabricated_sequence_features": False,
    }


def temporal_convolution_screen(
    population: DevelopmentPopulation,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    *,
    reference_probabilities: Mapping[str, np.ndarray] | None = None,
    null_repeats: int = 5,
) -> dict[str, Any]:
    """Primary 20-second TCN plus the conditional frozen 60-second sensitivity."""

    folds = list(outer_folds or development_outer_folds(population))
    primary = _temporal_convolution_window(
        population,
        folds,
        history_key="history_20s",
        reference_probabilities=reference_probabilities,
        null_repeats=null_repeats,
    )
    if not str(primary.get("status", "")).startswith("executed_"):
        return primary
    if primary.get("observation_gate_passed"):
        primary["sixty_second_sensitivity"] = {
            "status": "not_triggered_20s_observation_gate_passed"
        }
        return primary
    primary["sixty_second_sensitivity"] = _temporal_convolution_window(
        population,
        folds,
        history_key="history_60s",
        reference_probabilities=reference_probabilities,
        null_repeats=null_repeats,
    )
    return primary


def _ablation_blocks(population: DevelopmentPopulation) -> dict[str, FeatureMatrix]:
    shadow = population.blocks[SHADOW_BLOCK]
    result = {"without_shadow": population.blocks[PHYSICAL_BLOCK]}
    for omitted in PHYSICAL_SUBBLOCKS:
        remaining = [
            population.physical_subblocks[name]
            for name in PHYSICAL_SUBBLOCKS
            if name != omitted
        ]
        result[f"without_{omitted}"] = _concatenate(*remaining, shadow)
    return result


def _gate_report(
    block_name: str,
    block_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    nulls: Mapping[str, Any],
) -> dict[str, Any]:
    target_checks = {}
    for target in ("safety", "benefit"):
        comparison = comparisons[block_name]["targets"][target]
        candidate_metrics = block_report["targets"][target]["nested_metrics"]
        baseline_metrics = baseline_report["targets"][target]["nested_metrics"]
        baseline_auc = comparison["exact_v1_baseline_auc"]
        # Matched exact-v1 calibration values are attached by the caller below.
        loo_min = comparison["minimum_leave_one_seed_out_auc_gain"]
        target_checks[target] = {
            "matched_auc_gain_at_least_0_03": comparison["matched_auc_gain"]
            >= CONFIRMATORY_GATES["minimum_matched_auc_gain"],
            "leave_one_seed_gain_directionally_positive": loo_min is not None and loo_min >= 0.0,
            "shuffle_auc_near_chance": (
                nulls[target]["null_auc_mean"] is not None
                and float(nulls[target]["null_auc_mean"]) <= 0.55
            ),
            "brier_not_materially_worse": float(candidate_metrics["brier_score"])
            <= float(baseline_metrics["brier_score"]) + 0.02,
            "log_loss_not_materially_worse": float(candidate_metrics["log_loss"])
            <= float(baseline_metrics["log_loss"]) + 0.05,
            "candidate_auc": candidate_metrics["roc_auc"],
            "exact_v1_baseline_auc": baseline_auc,
            "matched_auc_gain": comparison["matched_auc_gain"],
            "minimum_leave_one_seed_auc_gain": loo_min,
        }
    improved = [
        target
        for target, checks in target_checks.items()
        if checks["matched_auc_gain_at_least_0_03"]
        and checks["leave_one_seed_gain_directionally_positive"]
        and checks["shuffle_auc_near_chance"]
        and checks["brier_not_materially_worse"]
        and checks["log_loss_not_materially_worse"]
    ]
    policy = block_report["cross_fitted_policy"]
    policy_checks = {
        "minimum_selected_cases": policy["command_case_count"]
        >= CONFIRMATORY_GATES["minimum_selected_cases"],
        "minimum_selected_seeds": policy["command_seed_count"]
        >= CONFIRMATORY_GATES["minimum_selected_seeds"],
        "minimum_selected_dose_success_rate": policy["selected_dose_success_rate"]
        >= CONFIRMATORY_GATES["minimum_selected_dose_success_rate"],
        "minimum_selected_safety_rate": policy["selected_safety_rate"]
        >= CONFIRMATORY_GATES["minimum_selected_safety_rate"],
        "minimum_joint_benefit_precision": policy["joint_benefit_precision"]
        >= CONFIRMATORY_GATES["minimum_joint_benefit_precision"],
        "no_selected_collision_or_teleport": policy[
            "selected_excess_collision_or_teleport_case_count"
        ]
        <= CONFIRMATORY_GATES["maximum_selected_excess_collisions"],
        "no_single_beneficial_outlier": (
            policy["beneficial_command_seed_count"] >= 2
            and policy["largest_single_beneficial_seed_fraction"] <= 0.50
        ),
    }
    return {
        "block": block_name,
        "target_checks": target_checks,
        "qualifying_improved_targets": improved,
        "policy_checks": policy_checks,
        "passed": bool(improved and all(policy_checks.values())),
    }


def run_development_analysis(
    population: DevelopmentPopulation,
    *,
    bootstrap_repeats: int = 200,
    null_repeats: int = 20,
    permutation_repeats: int = 3,
    run_ablations: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    outer_folds = development_outer_folds(population)
    screens: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    commands: dict[str, np.ndarray] = {}
    for block_name in CORE_BLOCKS:
        report, probability, command = nested_block_screen(
            population,
            population.blocks[block_name],
            outer_folds,
            permutation_repeats=(permutation_repeats if block_name == COMBINED_BLOCK else 0),
        )
        screens[block_name] = report
        predictions[block_name] = probability
        commands[block_name] = command

    comparisons = matched_comparisons(
        population,
        screens,
        predictions,
        bootstrap_repeats=bootstrap_repeats,
    )
    nulls = {
        block_name: shuffled_label_controls(
            population,
            population.blocks[block_name],
            outer_folds,
            repeats=null_repeats,
        )
        for block_name in (PHYSICAL_BLOCK, SHADOW_BLOCK, COMBINED_BLOCK)
    }
    ablations = {}
    if run_ablations:
        for name, block in _ablation_blocks(population).items():
            report, _probability, _command = nested_block_screen(
                population, block, outer_folds, permutation_repeats=0
            )
            ablations[name] = report
    effects = secondary_effect_regressions(population, outer_folds)
    sequence = temporal_convolution_screen(
        population,
        outer_folds,
        reference_probabilities=predictions[PHYSICAL_BLOCK],
        null_repeats=min(max(1, null_repeats), 10),
    )
    feature_diagnostics = {
        block_name: feature_block_diagnostics(population.blocks[block_name])
        for block_name in CORE_BLOCKS
    }
    nearest_neighbor_observability = cross_seed_nearest_neighbor_observability(
        population,
        population.blocks[COMBINED_BLOCK],
        outer_folds,
    )
    gates = {
        block_name: _gate_report(
            block_name,
            screens[block_name],
            screens[EXACT_BASELINE],
            comparisons,
            nulls[block_name],
        )
        for block_name in (PHYSICAL_BLOCK, SHADOW_BLOCK, COMBINED_BLOCK)
    }
    passed = [name for name, value in gates.items() if value["passed"]]
    report = {
        "schema": ANALYSIS_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract_manifest()["contract_sha256"],
        "scientific_role": "development_only_shadow_observation_v2_screen",
        "population": {
            "gate_f_admitted_case_count": len(population.rows),
            "independent_seed_count": len(set(population.groups)),
            "dose_success_case_count": int(population.dose.sum()),
            "dose_failure_case_count": int(np.sum(population.dose == 0)),
            "safe_case_count": int(np.sum((population.dose == 1) & (population.safety == 1))),
            "unsafe_case_count": int(np.sum((population.dose == 1) & (population.safety == 0))),
            "safe_joint_beneficial_case_count": int(
                np.sum((population.dose == 1) & (population.safety == 1) & (population.benefit == 1))
            ),
            "safe_joint_nonbeneficial_case_count": int(
                np.sum((population.dose == 1) & (population.safety == 1) & (population.benefit == 0))
            ),
        },
        "folds": [
            {
                "fold": index,
                "train_seed_groups": sorted(set(str(value) for value in population.groups[train])),
                "test_seed_groups": sorted(set(str(value) for value in population.groups[test])),
            }
            for index, (train, test) in enumerate(outer_folds)
        ],
        "feature_blocks": {
            name: {"feature_count": len(block.names), "feature_names": list(block.names)}
            for name, block in population.blocks.items()
        },
        "feature_block_diagnostics": feature_diagnostics,
        "combined_cross_seed_nearest_neighbor_observability": nearest_neighbor_observability,
        "matched_nested_model_screen": screens,
        "matched_exact_v1_comparisons": comparisons,
        "shuffled_label_controls": nulls,
        "leave_one_block_out_ablation": ablations,
        "secondary_paired_effect_regressors": effects,
        "temporal_convolution_screen": sequence,
        "confirmatory_gates": gates,
        "passing_v2_blocks": passed,
        "observation_repair_passed": bool(passed),
        "guardrails": {
            "labels_joined_only_after_capture_validation": True,
            "label_join_key": "exact_case_id",
            "validation_or_final_paths_read": False,
            "health_in_exact_v1_benchmark_only": True,
            "health_in_v2_candidates": False,
            "threshold_selection": "inside_outer_train_via_grouped_inner_oof",
            "sequence_surrogate_fabricated": False,
            "no_training_or_validation_authorized_by_this_report": True,
        },
    }
    oof = {
        "schema": "idm_v5_4_shadow_observation_v2_oof/1",
        "case_ids": [str(row["case_id"]) for row in population.rows],
        "seed_groups": [str(value) for value in population.groups],
        "observed": {
            "dose": population.dose.tolist(),
            "safety": population.safety.tolist(),
            "benefit": population.benefit.tolist(),
        },
        "blocks": {
            block_name: {
                "probabilities": {
                    target: values.tolist() for target, values in target_values.items()
                },
                "cross_fitted_command": commands[block_name].astype(int).tolist(),
            }
            for block_name, target_values in predictions.items()
        },
    }
    return report, oof


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=REPO_ROOT / CAPTURE_ROOT)
    parser.add_argument("--label-root", type=Path, default=REPO_ROOT / SOURCE_LABEL_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ANALYSIS_ROOT)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--null-repeats", type=int, default=20)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    args = parser.parse_args(argv)

    started = time.time()
    print("[1/8] validating all capture artifacts before opening labels", flush=True)
    capture_manifest, capture_rows = load_capture_rows(args.capture_root)
    print("[2/8] verifying frozen source hashes", flush=True)
    source_hashes = _verify_frozen_source_hashes(args.label_root.resolve())
    print("[3/8] opening development labels and joining exact case IDs", flush=True)
    label_summary, label_rows = load_label_rows(args.label_root)
    joined = join_capture_labels(capture_rows, label_rows)
    print("[4/8] constructing matched leakage-safe feature blocks", flush=True)
    population = prepare_population(joined)
    print("[5/8] running nested seed-grouped model and policy screens", flush=True)
    report, oof = run_development_analysis(
        population,
        bootstrap_repeats=max(20, args.bootstrap_repeats),
        null_repeats=max(5, args.null_repeats),
        permutation_repeats=max(1, args.permutation_repeats),
    )
    report["source"] = {
        "capture_root": str(args.capture_root.resolve()),
        "label_root": str(args.label_root.resolve()),
        "capture_manifest_schema": capture_manifest["schema"],
        "label_summary_schema": label_summary["schema"],
        "verified_source_sha256": source_hashes,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "validation_or_final_paths_read": False,
    }
    report["wall_clock_seconds"] = time.time() - started
    print("[6/8] preserving selected unsafe/harmful/dose-failure inventories", flush=True)
    inventories = {
        name: screen["cross_fitted_policy"]["inventories"]
        for name, screen in report["matched_nested_model_screen"].items()
    }
    print("[7/8] writing machine-readable development artifacts", flush=True)
    output = args.output.resolve()
    atomic_write_json(output / "report.json", report)
    atomic_write_json(output / "oof_predictions.json", oof)
    atomic_write_json(
        output / "selected_case_inventories.json",
        {
            "schema": "idm_v5_4_shadow_observation_v2_selected_case_inventory/1",
            "development_only": True,
            "inventories": inventories,
        },
    )
    print("[8/8] complete", flush=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "observation_repair_passed": report["observation_repair_passed"],
                "passing_v2_blocks": report["passing_v2_blocks"],
                "wall_clock_seconds": report["wall_clock_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
