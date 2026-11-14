"""Policy-value screen for P30 timing/candidate selection.

Per-action value regression (ridge) on causal, penetration-free features;
policy selects argmax(abstain=0, candidate-1, candidate-2) where actions are
feasible.  Features are frozen families A (candidate-local), B (A + classic
RSU merge-zone block) and the F ablation (RSU + patterns only).  Held
development is opened once after threshold selection on grouped training
predictions.  Value, not AUROC, is the authorization metric.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.io import atomic_write_json
from idm.rl_option_library import RLOptionRow, load_rl_library
from tools.screen_idm_expected_option_value import fit_ridge, predict_ridge

COVERAGE_GATE = 0.20
VALUE_MARGIN_GATE = 0.01
HARMFUL_FRACTION_GATE = 0.25
ORACLE_RECOVERY_GATE = 0.25


def _feature_vector(row: RLOptionRow, action: str, family: str) -> np.ndarray | None:
    if family == "rsu_pattern":
        context = np.asarray(
            [row.pattern_context.get(key, 0.0) for key in sorted(row.pattern_context)],
            dtype=float,
        )
        rsu = np.asarray(row.observation[9:17], dtype=float)
        return np.concatenate([rsu, context])
    if action not in row.candidate_features:
        return None
    features = row.candidate_features[action]
    base = [
        features.get("runway_m"),
        features.get("predicted_dose_s"),
        features.get("front_gap_m"),
        features.get("rear_gap_m"),
        features.get("front_ttc_s"),
        features.get("rear_ttc_s"),
        features.get("front_secure_gap_m"),
        features.get("rear_secure_gap_m"),
        features.get("prospective_rear_headroom_m"),
        features.get("command_separation_mps2"),
        features.get("p30_projected_acceleration_mps2"),
        features.get("disturbance_distance_to_merge_m"),
        features.get("distance_to_merge_m"),
    ]
    values = [0.0 if value is None else float(value) for value in base]
    context = np.asarray(
        [row.pattern_context.get(key, 0.0) for key in sorted(row.pattern_context)],
        dtype=float,
    )
    local = np.asarray(values, dtype=float)
    if family == "candidate_local":
        return np.concatenate([local, context])
    if family == "candidate_local_rsu":
        rsu = np.asarray(row.observation[9:17], dtype=float)
        return np.concatenate([local, context, rsu])
    raise ValueError(f"unknown family {family!r}")


def _samples(rows: list[RLOptionRow], family: str) -> tuple[np.ndarray, np.ndarray, list[tuple[RLOptionRow, str]]]:
    x, y, meta = [], [], []
    for row in rows:
        for action in ("candidate_1", "candidate_2"):
            if not row.actions[action].feasible:
                continue
            vector = _feature_vector(row, action, family)
            if vector is None:
                continue
            x.append(vector)
            y.append(float(row.actions[action].reward))
            meta.append((row, action))
    if not x:
        raise ValueError(f"no feasible candidate samples for family {family}")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), meta


def _train_value_models(train_rows: list[RLOptionRow], family: str) -> dict[str, dict[str, np.ndarray]]:
    x, y, meta = _samples(train_rows, family)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    models, best = {}, None
    for alpha in alphas:
        model = fit_ridge(x, y, np.ones(len(x)), alpha)
        # seed-grouped OOF MSE for alpha selection
        oof = np.zeros(len(x), dtype=float)
        for seed in sorted({row.seed for row, _ in meta}):
            test = np.asarray([row.seed == seed for row, _ in meta], dtype=bool)
            train = ~test
            if not np.any(train):
                continue
            oof_model = fit_ridge(x[train], y[train], np.ones(int(np.sum(train))), alpha)
            oof[test] = predict_ridge(oof_model, x[test])
        mse = float(np.mean((oof - y) ** 2))
        score = (-mse, alpha)
        if best is None or score > best[0]:
            best, models = (score,), {"alpha": alpha, "model": model, "mse": mse}
    return models


def _predict_action_values(rows: list[RLOptionRow], models: dict[str, dict[str, np.ndarray]], family: str) -> list[dict[str, float]]:
    predictions = []
    for row in rows:
        values = {"abstain": 0.0}
        for action in ("candidate_1", "candidate_2"):
            if not row.actions[action].feasible:
                continue
            vector = _feature_vector(row, action, family)
            values[action] = float(predict_ridge(models["model"], vector[None, :])[0])
        predictions.append(values)
    return predictions


def _policy(rows: list[RLOptionRow], predictions: list[dict[str, float]]) -> list[str]:
    selected = []
    for row, values in zip(rows, predictions):
        candidates = [action for action in ("candidate_1", "candidate_2") if row.actions[action].feasible]
        best, best_value = "abstain", 0.0
        for action in candidates:
            if values[action] > best_value:
                best, best_value = action, values[action]
        selected.append(best)
    return selected


def _metrics(rows: list[RLOptionRow], selected: list[str]) -> dict[str, Any]:
    rewards = np.asarray([row.actions[a].reward for row, a in zip(rows, selected)], dtype=float)
    counts = Counter(row.actions[a].classification for row, a in zip(rows, selected))
    nonabstain = [a for a in selected if a != "abstain"]
    failures = [row.sample_id for row, a in zip(rows, selected) if a != "abstain" and not row.actions[a].constraints_pass]
    by_demand: dict[str, dict[str, float]] = {}
    for demand in sorted({row.demand for row in rows}):
        mask = [row.demand == demand for row in rows]
        active = [a for a, keep in zip(selected, mask) if keep and a != "abstain"]
        by_demand[str(demand)] = {
            "mean_policy_reward": round(
                float(
                    np.mean(
                        [row.actions[a].reward for row, a, keep in zip(rows, selected, mask) if keep]
                    )
                ),
                5,
            ),
            "activations": int(len(active)),
        }
    by_pen = {}
    for pen in sorted({round(row.penetration, 2) for row in rows}):
        mask = [round(row.penetration, 2) == pen for row in rows]
        by_pen[f"{pen:.2f}"] = {
            "mean_policy_reward": round(
                float(
                    np.mean(
                        [row.actions[a].reward for row, a, keep in zip(rows, selected, mask) if keep]
                    )
                ),
                5,
            ),
            "activations": int(sum(1 for a, keep in zip(selected, mask) if keep and a != "abstain")),
        }
    return {
        "mean_policy_reward": float(np.mean(rewards)),
        "activation_coverage": float(np.mean([a != "abstain" for a in selected])),
        "activations": int(len(nonabstain)),
        "helpful": int(counts["helpful"]),
        "harmful": int(counts["harmful"]),
        "neutral": int(counts["neutral"]),
        "harmful_fraction_of_activations": float(
            counts["harmful"] / max(len(nonabstain), 1)
        ),
        "helpful_minus_harmful_fraction": float(
            (counts["helpful"] - counts["harmful"]) / max(len(nonabstain), 1)
        ),
        "constraint_failures": failures,
        "by_demand": by_demand,
        "by_penetration": by_pen,
    }


def analyze(library: Path) -> dict[str, Any]:
    rows, manifest = load_rl_library(library)
    train = [row for row in rows if row.seed in set(manifest["train_seeds"])]
    validation = [row for row in rows if row.seed in set(manifest["validation_seeds"])]
    fixed_rule_selection = [
        "candidate_1" if row.actions["candidate_1"].feasible else "abstain" for row in validation
    ]
    fixed_rule = _metrics(validation, fixed_rule_selection)
    oracle = _metrics(
        validation,
        [
            max(
                ("abstain", "candidate_1", "candidate_2"),
                key=lambda a: row.actions[a].reward if row.actions[a].feasible else -1e9,
            )
            for row in validation
        ],
    )
    results = {}
    for family in ("candidate_local", "candidate_local_rsu", "rsu_pattern"):
        models = _train_value_models(train, family)
        predictions = _predict_action_values(validation, models, family)
        selected = _policy(validation, predictions)
        metrics = _metrics(validation, selected)
        gain_fixed = float(metrics["mean_policy_reward"] - fixed_rule["mean_policy_reward"])
        oracle_gain = float(oracle["mean_policy_reward"] - fixed_rule["mean_policy_reward"])
        gates = {
            "coverage_at_least_20pct": metrics["activation_coverage"] >= COVERAGE_GATE,
            "improves_fixed_rule_by_0p01": gain_fixed >= VALUE_MARGIN_GATE,
            "helpful_more_frequent_than_harmful": metrics["helpful"] > metrics["harmful"],
            "harmful_fraction_at_most_25pct": metrics["harmful_fraction_of_activations"]
            <= HARMFUL_FRACTION_GATE,
            "positive_in_two_demands": sum(
                1 for value in metrics["by_demand"].values() if value["mean_policy_reward"] > 0.0
            ) >= 2,
            "positive_in_both_penetrations": all(
                value["mean_policy_reward"] > 0.0 for value in metrics["by_penetration"].values()
            ),
            "preserves_safety_service": not metrics["constraint_failures"],
            "recovers_oracle_fraction": (
                oracle_gain > 0.0 and gain_fixed / oracle_gain >= ORACLE_RECOVERY_GATE
            ),
        }
        results[family] = {
            "selected_alpha": models["alpha"],
            "training_mse": round(models["mse"], 6),
            "validation": metrics,
            "fixed_rule_gain": gain_fixed,
            "oracle_gain": oracle_gain,
            "gates": gates,
            "passes": all(gates.values()),
        }
    return {
        "schema": "idm_rl_timing_value_screen/1",
        "library": str(library.resolve()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "baselines": {
            "fixed_domain_rule": fixed_rule,
            "constrained_oracle": oracle,
        },
        "families": results,
        "authorized_families": [name for name, row in results.items() if row["passes"]],
        "rl_authorized": any(row["passes"] for row in results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.library.resolve())
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True)[:4500])


if __name__ == "__main__":
    main()
