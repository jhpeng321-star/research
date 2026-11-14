"""Diagnose predecision feature sufficiency for fixed p30 activation.

This is development-only feature-family diagnosis.  It never promotes a model
and deliberately separates currently actor-visible fields from causal fields
that are available to the deterministic admission/controller but not yet in
the actor observation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.io import atomic_write_json
from tools.audit_idm_phase5_observation_sufficiency import evaluate


EGO = slice(0, 9)
ZONE = slice(9, 21)
NEIGHBORS = slice(21, 121)
CONTEXT = slice(121, 134)


def _number(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _margin(gap: object, secure: object) -> float:
    if gap is None:
        return 250.0
    return _number(gap) - _number(secure, 0.0)


def _candidate_features(decision: dict, candidate: dict, observation: list[float]) -> list[float]:
    pattern = decision.get("pattern_context") or {}
    speed = _number(observation[1]) * 35.0
    reference = _number(candidate.get("preview_reference_speed_mps"))
    lane_id = str(candidate.get("lane_id", ""))
    try:
        lane_index = float(lane_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        lane_index = float("nan")
    return [
        _number(candidate.get("runway_m")),
        _number(candidate.get("distance_to_merge_m")),
        _number(candidate.get("disturbance_distance_to_merge_m")),
        _number(candidate.get("predicted_dose_s")),
        speed,
        reference,
        speed - reference,
        _number(candidate.get("prospective_rear_headroom_m")),
        _number(candidate.get("preview_nominal_acceleration_mps2")),
        _margin(candidate.get("front_gap_m"), candidate.get("front_secure_gap_m")),
        _margin(candidate.get("rear_gap_m"), candidate.get("rear_secure_gap_m")),
        _number(candidate.get("front_ttc_s"), 60.0),
        _number(candidate.get("rear_ttc_s"), 60.0),
        lane_index,
        _number(decision.get("decision_index")),
        _number(decision.get("decision_step")),
        _number(pattern.get("idm_severe_level")),
        _number(pattern.get("idm_wave_energy_level")),
        _number(pattern.get("idm_severe_short_ema")),
        _number(pattern.get("idm_wave_short_ema")),
        _number(pattern.get("idm_severe_trend")),
        _number(pattern.get("idm_wave_energy_trend")),
    ]


def load_rows(inputs: list[Path]) -> list[dict]:
    rows = []
    for root in (path.resolve() for path in inputs):
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        demand = int(summary["protocol"]["demand"])
        minimum_dose = int(summary["protocol"]["minimum_dose_steps"])
        for case in summary["cases"]:
            seed = int(case["seed"])
            discovery = json.loads(
                (root / "discoveries" / f"m{demand}.seed{seed}.json").read_text(
                    encoding="utf-8"
                )
            )
            decision = discovery["decision"]
            for effect in case["effects"]:
                if effect["classification"] not in {"helpful", "harmful"}:
                    continue
                constraints = dict(effect.get("constraints") or {})
                admissible_except_dose = all(
                    value for key, value in constraints.items() if key != "dose_at_least_minimum"
                )
                if (
                    int(effect["target_command_step_count"]) < minimum_dose
                    or not admissible_except_dose
                ):
                    continue
                slot = int(effect["candidate_slot"])
                candidate = decision["candidates"][slot - 1]
                vehicle_id = str(candidate["vehicle_id"])
                observation = list(decision["candidate_observations"][vehicle_id])[:134]
                rows.append(
                    {
                        "demand": demand,
                        "seed": seed,
                        "label": 1 if effect["classification"] == "helpful" else 0,
                        "classification": effect["classification"],
                        "effect": float(effect["safety_composite_reference_units"]),
                        "observation": observation,
                        "candidate": _candidate_features(decision, candidate, observation),
                        "decision_index": int(decision["decision_index"]),
                    }
                )
    return rows


def _score(features: np.ndarray, labels: np.ndarray, seeds: np.ndarray, demands: np.ndarray) -> dict:
    if len(features) < 2 or len(set(labels.tolist())) < 2 or len(set(seeds.tolist())) < 2:
        return {"auroc": float("nan"), "balanced_accuracy": float("nan"), "folds": []}
    return evaluate(features, labels, seeds, demands, slice(None))


def diagnose(inputs: list[Path]) -> dict:
    rows = load_rows(inputs)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    seeds = np.asarray([row["seed"] for row in rows], dtype=int)
    demands = np.asarray([row["demand"] for row in rows], dtype=int)
    actor = np.asarray([row["observation"] for row in rows], dtype=float)
    candidate = np.asarray([row["candidate"] for row in rows], dtype=float)
    families = {
        "actor_full_134": actor,
        "actor_ego_9": actor[:, EGO],
        "actor_zone_12": actor[:, ZONE],
        "actor_neighbors_100": actor[:, NEIGHBORS],
        "actor_nearest_neighbor_5": actor[:, 21:26],
        "actor_nearest_2_neighbors_10": actor[:, 21:31],
        "actor_nearest_3_neighbors_15": actor[:, 21:36],
        "actor_nearest_5_neighbors_25": actor[:, 21:46],
        "actor_nearest_10_neighbors_50": actor[:, 21:71],
        "actor_context_13": actor[:, CONTEXT],
        "actor_ego_zone_context_34": np.column_stack(
            [actor[:, EGO], actor[:, ZONE], actor[:, CONTEXT]]
        ),
        "actor_zone_plus_recurrence_13": np.column_stack(
            [actor[:, ZONE], candidate[:, 14:15]]
        ),
        "actor_ego_zone_context_plus_recurrence_35": np.column_stack(
            [actor[:, EGO], actor[:, ZONE], actor[:, CONTEXT], candidate[:, 14:15]]
        ),
        "admission_controller_22": candidate,
        "actor_context_plus_admission": np.column_stack([actor[:, CONTEXT], candidate]),
        "risk_recurrence_only": candidate[:, 14:15],
        "simulation_time_only": candidate[:, 15:16],
    }
    metrics = {
        name: _score(values, labels, seeds, demands)
        for name, values in families.items()
    }
    metrics_by_demand = {}
    for demand in sorted(set(demands.tolist())):
        selected = demands == demand
        metrics_by_demand[str(demand)] = {
            name: _score(values[selected], labels[selected], seeds[selected], demands[selected])
            for name, values in families.items()
        }
    recurrence = {}
    for index in sorted({row["decision_index"] for row in rows}):
        group = [row for row in rows if row["decision_index"] == index]
        recurrence[str(index)] = {
            "count": len(group),
            "helpful": sum(row["label"] == 1 for row in group),
            "harmful": sum(row["label"] == 0 for row in group),
            "mean_effect": float(np.mean([row["effect"] for row in group])),
        }
    return {
        "schema": "idm_decentralized_decision_observability_diagnosis/1",
        "development_only": True,
        "inputs": [str(path.resolve()) for path in inputs],
        "case_count": len(rows),
        "seed_count": len(set(seeds.tolist())),
        "helpful_count": int(np.sum(labels == 1)),
        "harmful_count": int(np.sum(labels == 0)),
        "forbidden_features": ["future outcome", "realized dose", "seed", "filename"],
        "metrics": metrics,
        "metrics_by_demand": metrics_by_demand,
        "by_risk_recurrence": recurrence,
        "interpretation": (
            "Feature-family results are development diagnostics only. Any newly selected "
            "observation family must be frozen and checked on new demand-seed cells before training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = diagnose(args.inputs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
