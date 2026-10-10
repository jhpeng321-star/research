"""Action-specific observation-sufficiency gate for decentralized IDM activation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.io import atomic_write_json
from tools.audit_idm_phase5_observation_sufficiency import evaluate


PHYSICAL_OBSERVATION_DIM = 134
MASKED_OBSERVATION_DIM = 136
MINIMUM_NONNEUTRAL_CASES = 24
MINIMUM_SEED_COVERAGE = 8


def _observation_hash(observation: list[float]) -> str:
    payload = np.asarray(observation, dtype=np.float32).tobytes()
    return sha256(payload).hexdigest()


def _records(inputs: list[Path]) -> tuple[list[dict], list[dict]]:
    summaries = []
    records = []
    for directory in inputs:
        root = directory.resolve()
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        demand = int(summary["protocol"]["demand"])
        for case in summary["cases"]:
            seed = int(case["seed"])
            discovery_path = root / "discoveries" / f"m{demand}.seed{seed}.json"
            discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
            decision = discovery.get("decision")
            if decision is None or int(decision["decision_step"]) != int(case["decision_step"]):
                raise RuntimeError(f"missing paired decision for m{demand} seed{seed}")
            branches = {
                row["arm"]: row
                for path in root.joinpath("branches").glob(f"{case['case_id']}.*.json")
                for row in [json.loads(path.read_text(encoding="utf-8"))]
            }
            for effect in case["effects"]:
                slot = int(effect["candidate_slot"])
                mode_index = int(effect["mode_index"])
                if mode_index == 0:
                    continue
                candidate = decision["candidates"][slot - 1]
                vehicle_id = str(candidate["vehicle_id"])
                observation = list(decision["candidate_observations"][vehicle_id])
                arm = next(
                    name
                    for name, branch in branches.items()
                    if not branch["is_control"]
                    and int(branch["candidate_slot"]) == slot
                    and int(branch["mode_index"]) == mode_index
                )
                branch = branches[arm]
                records.append(
                    {
                        "source": str(root),
                        "demand": demand,
                        "seed": seed,
                        "decision_index": int(decision.get("decision_index", 1)),
                        "decision_step": int(decision["decision_step"]),
                        "candidate_slot": slot,
                        "vehicle_id": vehicle_id,
                        "classification": str(effect["classification"]),
                        "effect": float(effect["safety_composite_reference_units"]),
                        "constraints_pass": bool(effect["constraints_pass"]),
                        "constraints": dict(effect.get("constraints") or {}),
                        "dose_steps": int(effect["target_command_step_count"]),
                        "observation": observation,
                        "observation_hash": _observation_hash(observation),
                    }
                )
    return summaries, records


def _protocol_identity(summaries: list[dict]) -> bool:
    keys = (
        "schema",
        "action_set",
        "option_steps",
        "minimum_dose_steps",
        "safety_horizon_steps",
        "service_horizon_steps",
        "landmark",
        "landmark_contract",
        "cav_penetration",
        "candidate_ranking",
        "candidate_count",
        "reward_refs_sha256",
    )
    reference = summaries[0]["protocol"]
    return all(
        all(summary["protocol"].get(key) == reference.get(key) for key in keys)
        for summary in summaries[1:]
    )


def analyze(inputs: list[Path], required_demands: set[int] | None = None) -> dict:
    summaries, records = _records(inputs)
    if not summaries:
        raise ValueError("at least one authority-verification input is required")
    reference = summaries[0]["protocol"]
    minimum_dose = int(reference["minimum_dose_steps"])
    dose_qualified = [row for row in records if row["dose_steps"] >= minimum_dose]
    def admissible_except_dose(row: dict) -> bool:
        return all(
            value
            for key, value in row["constraints"].items()
            if key != "dose_at_least_minimum"
        )

    safety_admissible = [row for row in records if admissible_except_dose(row)]
    qualified = [
        row
        for row in dose_qualified
        if row in safety_admissible
    ]
    nonneutral = [
        row for row in qualified if row["classification"] in {"helpful", "harmful"}
    ]
    labels = np.asarray(
        [1 if row["classification"] == "helpful" else 0 for row in nonneutral],
        dtype=int,
    )
    features = np.asarray(
        [row["observation"][:PHYSICAL_OBSERVATION_DIM] for row in nonneutral],
        dtype=np.float64,
    )
    seeds = np.asarray([row["seed"] for row in nonneutral], dtype=int)
    demands = np.asarray([row["demand"] for row in nonneutral], dtype=int)
    required_demands = set(required_demands or {2200})
    classes_present = len(set(labels.tolist())) == 2
    metrics = (
        evaluate(features, labels, seeds, demands, slice(None))
        if len(nonneutral) and classes_present and len(set(seeds.tolist())) >= 2
        else {
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "true_positive_rate": float("nan"),
            "true_negative_rate": float("nan"),
            "folds": [],
        }
    )
    demand_only_metrics = (
        evaluate(
            demands.astype(np.float64).reshape(-1, 1),
            labels,
            seeds,
            demands,
            slice(None),
        )
        if len(nonneutral) and classes_present and len(set(seeds.tolist())) >= 2
        else {
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "true_positive_rate": float("nan"),
            "true_negative_rate": float("nan"),
            "folds": [],
        }
    )
    zone_recurrence_features = (
        np.column_stack(
            [
                features[:, 9:21],
                np.asarray([row["decision_index"] for row in nonneutral], dtype=float),
            ]
        )
        if len(nonneutral)
        else np.empty((0, 13), dtype=float)
    )
    ranked_neighbor_features = (
        features[:, 21:36] if len(nonneutral) else np.empty((0, 15), dtype=float)
    )
    zone_recurrence_metrics = (
        evaluate(
            zone_recurrence_features,
            labels,
            seeds,
            demands,
            slice(None),
        )
        if len(nonneutral) and classes_present and len(set(seeds.tolist())) >= 2
        else {
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "true_positive_rate": float("nan"),
            "true_negative_rate": float("nan"),
            "folds": [],
        }
    )
    zone_recurrence_by_demand = {}
    ranked_neighbor_by_demand = {}
    for demand in sorted(set(demands.tolist())):
        selected = demands == demand
        demand_labels = labels[selected]
        if (
            int(selected.sum()) >= 2
            and len(set(demand_labels.tolist())) == 2
            and len(set(seeds[selected].tolist())) >= 2
        ):
            zone_recurrence_by_demand[str(demand)] = evaluate(
                zone_recurrence_features[selected],
                demand_labels,
                seeds[selected],
                demands[selected],
                slice(None),
            )
        else:
            zone_recurrence_by_demand[str(demand)] = {
                "auroc": float("nan"),
                "balanced_accuracy": float("nan"),
                "true_positive_rate": float("nan"),
                "true_negative_rate": float("nan"),
                "folds": [],
            }
        if (
            int(selected.sum()) >= 2
            and len(set(demand_labels.tolist())) == 2
            and len(set(seeds[selected].tolist())) >= 2
        ):
            ranked_neighbor_by_demand[str(demand)] = evaluate(
                ranked_neighbor_features[selected],
                demand_labels,
                seeds[selected],
                demands[selected],
                slice(None),
            )
        else:
            ranked_neighbor_by_demand[str(demand)] = {
                "auroc": float("nan"),
                "balanced_accuracy": float("nan"),
                "true_positive_rate": float("nan"),
                "true_negative_rate": float("nan"),
                "folds": [],
            }
    ranked_neighbor_metrics = (
        evaluate(ranked_neighbor_features, labels, seeds, demands, slice(None))
        if len(nonneutral) and classes_present and len(set(seeds.tolist())) >= 2
        else {
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "true_positive_rate": float("nan"),
            "true_negative_rate": float("nan"),
            "folds": [],
        }
    )
    class_seeds = defaultdict(set)
    for row in nonneutral:
        class_seeds[row["classification"]].add(row["seed"])
    counts = Counter(row["classification"] for row in records)
    dose_qualified_fraction = len(dose_qualified) / max(len(records), 1)
    safety_admissible_fraction = len(safety_admissible) / max(len(records), 1)
    gates = {
        "protocol_identity": _protocol_identity(summaries),
        "all_observations_match_136_field_contract": bool(records)
        and all(len(row["observation"]) == MASKED_OBSERVATION_DIM for row in records),
        "safety_service_admissible_fraction_at_least_90pct": bool(records)
        and safety_admissible_fraction >= 0.90,
        "dose_qualified_fraction_at_least_80pct": bool(records)
        and dose_qualified_fraction >= 0.80,
        "at_least_24_nonneutral_cases": len(nonneutral) >= MINIMUM_NONNEUTRAL_CASES,
        "at_least_8_independent_development_seeds": len(set(seeds.tolist()))
        >= MINIMUM_SEED_COVERAGE,
        "covers_every_predeclared_demand_cell": required_demands.issubset(
            set(row["demand"] for row in records)
        ),
        "both_classes_in_at_least_3_seeds": min(
            len(class_seeds["helpful"]), len(class_seeds["harmful"])
        )
        >= 3,
        "leave_one_seed_out_auroc_at_least_0p70": metrics["auroc"] >= 0.70,
        "leave_one_seed_out_balanced_accuracy_at_least_0p65": metrics[
            "balanced_accuracy"
        ]
        >= 0.65,
    }
    finite_demand_metrics = [
        row
        for row in zone_recurrence_by_demand.values()
        if np.isfinite(float(row["auroc"]))
        and np.isfinite(float(row["balanced_accuracy"]))
    ]
    proposed_upgrade_gates = {
        key: value
        for key, value in gates.items()
        if not key.startswith("leave_one_seed_out_")
    }
    proposed_upgrade_gates.update(
        {
            "zone_recurrence_loso_auroc_at_least_0p70": zone_recurrence_metrics[
                "auroc"
            ]
            >= 0.70,
            "zone_recurrence_loso_balanced_accuracy_at_least_0p65": (
                zone_recurrence_metrics["balanced_accuracy"] >= 0.65
            ),
            "zone_recurrence_passes_two_demand_cells": sum(
                row["auroc"] >= 0.60 and row["balanced_accuracy"] >= 0.55
                for row in finite_demand_metrics
            )
            >= 2,
            "zone_recurrence_no_cell_auroc_below_0p45": bool(finite_demand_metrics)
            and all(row["auroc"] >= 0.45 for row in finite_demand_metrics),
        }
    )
    ranked_neighbor_upgrade_gates = {
        key: value
        for key, value in gates.items()
        if not key.startswith("leave_one_seed_out_")
    }
    m2000_ranked = ranked_neighbor_by_demand.get("2000", {})
    m2000_records = [row for row in records if row["demand"] == 2000]
    m2000_qualified = [row for row in qualified if row["demand"] == 2000]
    m2000_effects = [row["effect"] for row in m2000_qualified]
    m2000_harmful_fraction = (
        sum(row["classification"] == "harmful" for row in m2000_qualified)
        / len(m2000_qualified)
        if m2000_qualified
        else float("nan")
    )
    ranked_neighbor_upgrade_gates.update(
        {
            "ranked_neighbor_loso_auroc_at_least_0p70": ranked_neighbor_metrics[
                "auroc"
            ]
            >= 0.70,
            "ranked_neighbor_loso_balanced_accuracy_at_least_0p65": (
                ranked_neighbor_metrics["balanced_accuracy"] >= 0.65
            ),
            "m2000_ranked_neighbor_auroc_at_least_0p65": float(
                m2000_ranked.get("auroc", float("nan"))
            )
            >= 0.65,
            "m2000_ranked_neighbor_balanced_accuracy_at_least_0p60": float(
                m2000_ranked.get("balanced_accuracy", float("nan"))
            )
            >= 0.60,
            "m2000_all_treatments_fully_qualified": bool(m2000_records)
            and len(m2000_qualified) == len(m2000_records),
            "m2000_mean_effect_at_least_0p02": bool(m2000_effects)
            and float(np.mean(m2000_effects)) >= 0.02,
            "m2000_harmful_fraction_at_most_0p35": bool(m2000_effects)
            and m2000_harmful_fraction <= 0.35,
            "m2000_tenth_percentile_effect_at_least_minus_0p25": bool(m2000_effects)
            and float(np.percentile(m2000_effects, 10.0)) >= -0.25,
        }
    )
    by_demand_recurrence = {}
    for demand in sorted({row["demand"] for row in records}):
        by_demand_recurrence[str(demand)] = {}
        for decision_index in sorted(
            {row["decision_index"] for row in records if row["demand"] == demand}
        ):
            group = [
                row
                for row in records
                if row["demand"] == demand and row["decision_index"] == decision_index
            ]
            group_qualified = [
                row
                for row in group
                if row["dose_steps"] >= minimum_dose and admissible_except_dose(row)
            ]
            effects = [row["effect"] for row in group_qualified]
            by_demand_recurrence[str(demand)][str(decision_index)] = {
                "case_count": len(group),
                "dose_qualified_count": sum(
                    row["dose_steps"] >= minimum_dose for row in group
                ),
                "safety_service_admissible_count": sum(
                    admissible_except_dose(row) for row in group
                ),
                "fully_qualified_count": len(group_qualified),
                "classification_counts": dict(
                    Counter(row["classification"] for row in group_qualified)
                ),
                "mean_effect": float(np.mean(effects)) if effects else float("nan"),
                "worst_effect": float(np.min(effects)) if effects else float("nan"),
            }
    constraint_failure_counts = {}
    for demand in sorted({row["demand"] for row in records}):
        counts = Counter(
            key
            for row in records
            if row["demand"] == demand
            for key, value in row["constraints"].items()
            if not value
        )
        constraint_failure_counts[str(demand)] = dict(counts)
    return {
        "schema": "idm_decentralized_activation_sufficiency/3",
        "development_only": True,
        "forbidden_inputs": ["seed", "arm", "filename", "future outcome"],
        "inputs": [str(path.resolve()) for path in inputs],
        "physical_observation_dimension": PHYSICAL_OBSERVATION_DIM,
        "masked_observation_dimension": MASKED_OBSERVATION_DIM,
        "case_count": len(records),
        "nonneutral_case_count": len(nonneutral),
        "dose_qualified_case_count": len(dose_qualified),
        "dose_qualified_fraction": dose_qualified_fraction,
        "safety_service_admissible_case_count": len(safety_admissible),
        "safety_service_admissible_fraction": safety_admissible_fraction,
        "seed_count": len(set(row["seed"] for row in records)),
        "classification_counts": dict(counts),
        "class_seed_coverage": {
            key: len(value) for key, value in sorted(class_seeds.items())
        },
        "metrics": metrics,
        "demand_only_metrics": demand_only_metrics,
        "zone_recurrence_metrics": zone_recurrence_metrics,
        "zone_recurrence_by_demand": zone_recurrence_by_demand,
        "ranked_neighbor_metrics": ranked_neighbor_metrics,
        "ranked_neighbor_by_demand": ranked_neighbor_by_demand,
        "m2000_operating_domain": {
            "case_count": len(m2000_records),
            "fully_qualified_count": len(m2000_qualified),
            "mean_effect": float(np.mean(m2000_effects))
            if m2000_effects
            else float("nan"),
            "harmful_fraction": m2000_harmful_fraction,
            "tenth_percentile_effect": float(np.percentile(m2000_effects, 10.0))
            if m2000_effects
            else float("nan"),
        },
        "by_demand_recurrence": by_demand_recurrence,
        "constraint_failure_counts": constraint_failure_counts,
        "required_demands": sorted(required_demands),
        "gates": gates,
        "observation_sufficiency_pass": all(gates.values()),
        "proposed_upgrade_gates": proposed_upgrade_gates,
        "proposed_upgrade_validation_pass": all(proposed_upgrade_gates.values()),
        "ranked_neighbor_upgrade_gates": ranked_neighbor_upgrade_gates,
        "ranked_neighbor_upgrade_validation_pass": all(
            ranked_neighbor_upgrade_gates.values()
        ),
        "records": [
            {key: value for key, value in row.items() if key != "observation"}
            for row in sorted(
                records,
                key=lambda row: (
                    row["seed"],
                    row["decision_index"],
                    row["candidate_slot"],
                ),
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-demands", type=int, nargs="+", default=[2200])
    args = parser.parse_args()
    payload = analyze(args.inputs, set(args.required_demands))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    report = [
        "# IDM Decentralized Activation Observation Sufficiency",
        "",
        "Status: **PASS**" if payload["observation_sufficiency_pass"] else "Status: **FAIL**",
        "",
        "Development-only action-specific evidence; this does not establish policy efficacy.",
        "",
        f"- Cases: {payload['case_count']} ({payload['nonneutral_case_count']} nonneutral)",
        f"- Seeds: {payload['seed_count']}",
        f"- Outcomes: {payload['classification_counts']}",
        f"- LOSO AUROC: {payload['metrics']['auroc']:.3f}",
        f"- LOSO balanced accuracy: {payload['metrics']['balanced_accuracy']:.3f}",
        f"- Demand-only AUROC diagnostic: {payload['demand_only_metrics']['auroc']:.3f}",
        f"- Zone + recurrence LOSO AUROC: {payload['zone_recurrence_metrics']['auroc']:.3f}",
        f"- Zone + recurrence LOSO balanced accuracy: {payload['zone_recurrence_metrics']['balanced_accuracy']:.3f}",
        "",
        "## Gates",
        "",
        *[
            f"- {'PASS' if value else 'FAIL'} — `{key}`"
            for key, value in payload["gates"].items()
        ],
        "",
        "## Proposed zone + recurrence upgrade gates",
        "",
        *[
            f"- {'PASS' if value else 'FAIL'} — `{key}`"
            for key, value in payload["proposed_upgrade_gates"].items()
        ],
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
