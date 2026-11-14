"""Build a development-only continuous authority map for frozen P30 damping.

The map reuses exact paired SUMO branches.  It deliberately keeps the frozen
helpful/neutral/harmful response class separate from absolute residual traffic
metrics.  It does not infer a critical density or manufacture a binary
``safe_but_insufficient`` label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.io import atomic_write_json


SCHEMA = "idm_damping_authority_map/1"
EXPECTED_SOURCE_SCHEMA = "idm_decentralized_authority/1"
EXPECTED_GRID_SCHEMA = "idm_paired_option_penetration_grid/1"
EXPECTED_ACTION_SET = "activation_speed_ceiling_v1"
EXPECTED_DECISION_INDEX = 3
EXPECTED_OPTION_STEPS = 20
EXPECTED_HORIZON_STEPS = 60
MINIMUM_DOSE_STEPS = 16
DEADBAND = 0.02
PROSPECTIVE_SEED_FLOOR = 4201
FROZEN_DOMAIN_PENETRATIONS = (0.05, 0.10)
FROZEN_DOMAIN_DEMANDS = (1900, 2000, 2100)

REFERENCE_FIELDS = {
    "severe": "corridor_severe_braking_count_ref_per_s",
    "energy": "corridor_acceleration_energy_sum_ref_m2ps4",
}
BRANCH_SCALAR_FIELDS = (
    "severe_braking_count_sum",
    "acceleration_energy_sum",
    "ttc_count_sum",
    "low_ttc_exposure_steps",
    "mean_corridor_delay_rate",
    "arrived_vehicles",
    "merge_discharge_count",
    "collision_observations",
    "teleport_observations",
    "collapse_max_consecutive_steps",
)
PATTERN_FIELDS = (
    "idm_severe_level",
    "idm_wave_energy_level",
    "idm_severe_short_ema",
    "idm_wave_short_ema",
    "idm_severe_trend",
    "idm_wave_energy_trend",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_number(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _references(path: Path) -> tuple[dict[int, dict[str, float]], str]:
    payload = _read(path)
    rows = payload.get("refs") or payload
    result: dict[int, dict[str, float]] = {}
    for row in rows.values():
        demand = int(row["demand_mainline_vph"])
        result[demand] = {
            name: _finite_number(row[field], field=field)
            for name, field in REFERENCE_FIELDS.items()
        }
    return result, _sha256(path)


def roots_from_grid_summary(path: Path) -> list[Path]:
    payload = _read(path)
    if payload.get("schema") != EXPECTED_GRID_SCHEMA:
        raise ValueError(f"unexpected grid schema {payload.get('schema')!r}")
    if not bool(payload.get("development_only")):
        raise ValueError("authority-map sources must be development-only")
    if payload.get("status") != "complete":
        raise ValueError("grid summary is not complete")
    roots = []
    for cell in payload.get("cells", []):
        if cell.get("status") != "complete":
            raise ValueError(f"incomplete grid cell {cell}")
        roots.append(Path(cell["output"]).resolve())
    if not roots:
        raise ValueError("grid summary contains no cells")
    return roots


def _source_contract(
    root: Path, protocol: Mapping[str, Any], reference_sha256: str
) -> dict[str, Any]:
    checks = {
        "source_schema": protocol.get("schema") == EXPECTED_SOURCE_SCHEMA,
        "development_only": bool(protocol.get("development_only")),
        "action_set": protocol.get("action_set") == EXPECTED_ACTION_SET,
        "decision_index": int(protocol.get("decision_index", -1))
        == EXPECTED_DECISION_INDEX,
        "option_steps": int(protocol.get("option_steps", -1))
        == EXPECTED_OPTION_STEPS,
        "safety_horizon_steps": int(protocol.get("safety_horizon_steps", -1))
        == EXPECTED_HORIZON_STEPS,
        "service_horizon_steps": int(protocol.get("service_horizon_steps", -1))
        == EXPECTED_HORIZON_STEPS,
        "candidate_ranking": protocol.get("candidate_ranking") == "runway",
        "candidate_count": int(protocol.get("candidate_count", -1)) == 1,
        "landmark": protocol.get("landmark") == "risk_onset",
        "reward_reference": protocol.get("reward_refs_sha256")
        == reference_sha256,
    }
    if not all(checks.values()):
        raise ValueError(f"{root}: incompatible source contract {checks}")
    seeds = sorted(int(seed) for seed in protocol.get("seeds", []))
    if not seeds:
        raise ValueError(f"{root}: source has no seeds")
    if max(seeds) >= PROSPECTIVE_SEED_FLOOR:
        raise ValueError(
            f"{root}: prospective seed {max(seeds)} is prohibited in development map"
        )
    return {
        "checks": checks,
        "demand": int(protocol["demand"]),
        "penetration": float(protocol["cav_penetration"]),
        "seeds": seeds,
    }


def _candidate(decision: Mapping[str, Any], slot: int) -> Mapping[str, Any]:
    by_slot = {
        int(row["slot"]): row for row in decision.get("candidates", [])
    }
    if slot not in by_slot:
        raise ValueError(f"candidate slot {slot} is missing")
    return by_slot[slot]


def _spatial_metric(branch: Mapping[str, Any], level: str, metric: str) -> float:
    return _finite_number(
        ((branch.get("spatial_totals") or {}).get(level) or {}).get(metric, 0.0),
        field=f"spatial_totals.{level}.{metric}",
    )


def _command_authority(branch: Mapping[str, Any]) -> dict[str, float | int | None]:
    requested: list[float] = []
    realized: list[float] = []
    for command in branch.get("command_rows", []):
        nominal = command.get("nominal_acceleration_mps2")
        desired = command.get("desired_acceleration_mps2")
        projected = command.get("projected_acceleration_mps2")
        if None in (nominal, desired, projected):
            continue
        requested.append(abs(float(desired) - float(nominal)))
        realized.append(abs(float(projected) - float(nominal)))
    projection = [
        float(value)
        for value in branch.get("projection_survival_values", [])
        if math.isfinite(float(value))
    ]
    return {
        "command_row_count": len(branch.get("command_rows", [])),
        "mean_requested_acceleration_delta_mps2": (
            float(np.mean(requested)) if requested else None
        ),
        "mean_realized_acceleration_delta_mps2": (
            float(np.mean(realized)) if realized else None
        ),
        "mean_projection_survival": (
            float(np.mean(projection)) if projection else None
        ),
        "median_projection_survival": (
            float(np.median(projection)) if projection else None
        ),
        "projection_survival_at_least_half_fraction": (
            float(np.mean(np.asarray(projection) >= 0.5)) if projection else None
        ),
    }


def _absolute_metrics(
    branch: Mapping[str, Any], refs: Mapping[str, float]
) -> dict[str, float | int]:
    severe = _finite_number(
        branch["severe_braking_count_sum"], field="severe_braking_count_sum"
    )
    energy = _finite_number(
        branch["acceleration_energy_sum"], field="acceleration_energy_sum"
    )
    horizon = int(branch["completed_horizon_steps"])
    normalizer_steps = max(horizon, 1)
    severe_ref_units = severe / (refs["severe"] * normalizer_steps)
    energy_ref_units = energy / (refs["energy"] * normalizer_steps)
    return {
        "completed_horizon_steps": horizon,
        "severe_braking_count_sum": severe,
        "severe_braking_count_per_step": severe / normalizer_steps,
        "severe_braking_absolute_reference_units": severe_ref_units,
        "acceleration_energy_sum": energy,
        "acceleration_energy_per_step": energy / normalizer_steps,
        "acceleration_energy_absolute_reference_units": energy_ref_units,
        "absolute_safety_burden_reference_units": 0.5
        * (severe_ref_units + energy_ref_units),
        "ttc_count_sum": _finite_number(
            branch.get("ttc_count_sum", 0.0), field="ttc_count_sum"
        ),
        "low_ttc_exposure_steps": int(branch["low_ttc_exposure_steps"]),
        "mean_corridor_delay_rate": _finite_number(
            branch["mean_corridor_delay_rate"], field="mean_corridor_delay_rate"
        ),
        "arrived_vehicles": int(branch["arrived_vehicles"]),
        "merge_discharge_count": int(branch["merge_discharge_count"]),
        "collision_observations": int(branch["collision_observations"]),
        "teleport_observations": int(branch["teleport_observations"]),
        "collapse_max_consecutive_steps": int(
            branch["collapse_max_consecutive_steps"]
        ),
        "corridor_spatial_severe_braking_count": _spatial_metric(
            branch, "corridor", "severe_braking_count"
        ),
        "corridor_spatial_low_ttc_count": _spatial_metric(
            branch, "corridor", "low_ttc_count"
        ),
        "corridor_integrated_negative_acceleration_mps": _spatial_metric(
            branch, "corridor", "integrated_negative_acceleration_mps"
        ),
        "immediate_follower_severe_braking_count": _spatial_metric(
            branch, "immediate_follower", "severe_braking_count"
        ),
        "immediate_follower_low_ttc_count": _spatial_metric(
            branch, "immediate_follower", "low_ttc_count"
        ),
    }


def _metric_deltas(
    control: Mapping[str, float | int], treatment: Mapping[str, float | int]
) -> dict[str, float]:
    return {
        name: float(treatment[name]) - float(control[name])
        for name in treatment
        if name != "completed_horizon_steps" and name in control
    }


def _replay_delta(
    control: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, float]:
    result = {}
    for field in BRANCH_SCALAR_FIELDS:
        left = control.get(field)
        right = replay.get(field)
        if left is None or right is None:
            continue
        result[field] = abs(float(right) - float(left))
    return result


def _response_class(effect: Mapping[str, Any]) -> str:
    classification = str(effect["classification"])
    if classification not in {"helpful", "neutral", "harmful"}:
        raise ValueError(f"unknown response class {classification!r}")
    utility = _finite_number(
        effect["safety_composite_reference_units"],
        field="safety_composite_reference_units",
    )
    constraints_pass = bool(effect["constraints_pass"])
    expected = (
        "harmful"
        if not constraints_pass or utility < -DEADBAND
        else "helpful"
        if utility > DEADBAND
        else "neutral"
    )
    if classification != expected:
        raise ValueError(
            f"stored response class {classification!r} disagrees with {expected!r}"
        )
    return classification


def _row(
    *,
    root: Path,
    source_contract: Mapping[str, Any],
    case: Mapping[str, Any],
    effect: Mapping[str, Any],
    refs: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    demand = int(source_contract["demand"])
    penetration = float(source_contract["penetration"])
    case_id = str(case["case_id"])
    seed = int(case["seed"])
    if not bool(case.get("all_branch_decision_identity")) or not bool(
        case.get("all_discovery_identity")
    ):
        raise ValueError(f"{root}: paired identity failed for {case_id}")
    discovery = _read(root / "discoveries" / f"m{demand}.seed{seed}.json")
    decision = discovery.get("decision")
    if not isinstance(decision, dict):
        raise ValueError(f"{root}: missing decision for {case_id}")
    if int(decision["decision_step"]) != int(case["decision_step"]):
        raise ValueError(f"{root}: decision step mismatch for {case_id}")
    arm = str(effect["arm"])
    control = _read(root / "branches" / f"{case_id}.abstain.json")
    replay = _read(root / "branches" / f"{case_id}.abstain_replay.json")
    treatment = _read(root / "branches" / f"{case_id}.{arm}.json")
    if int(control.get("control_command_count", 0)) != 0 or int(
        replay.get("control_command_count", 0)
    ) != 0:
        raise ValueError(f"{root}: abstain branch received a control command")
    for name, branch in (
        ("control", control),
        ("replay", replay),
        ("treatment", treatment),
    ):
        if not bool(branch.get("decision_identity_with_discovery")):
            raise ValueError(f"{root}: {name} identity failed for {case_id}")
    slot = int(effect["candidate_slot"])
    candidate = _candidate(decision, slot)
    control_absolute = _absolute_metrics(control, refs)
    treatment_absolute = _absolute_metrics(treatment, refs)
    dose_steps = int(effect["target_command_step_count"])
    pattern = dict(decision.get("pattern_context") or {})
    in_domain = (
        any(abs(penetration - value) <= 1e-12 for value in FROZEN_DOMAIN_PENETRATIONS)
        and demand in FROZEN_DOMAIN_DEMANDS
    )
    row = {
        "sample_id": (
            f"m{demand}.p{penetration:.3f}.seed{seed}."
            f"d{EXPECTED_DECISION_INDEX}.slot{slot}"
        ),
        "source": str(root),
        "case_id": case_id,
        "seed": seed,
        "demand_vph": demand,
        "configured_cav_penetration": penetration,
        "decision_index": EXPECTED_DECISION_INDEX,
        "decision_step": int(decision["decision_step"]),
        "landmark": str(decision["landmark"]),
        "candidate_vehicle_id": str(candidate["vehicle_id"]),
        "candidate_lane_id": str(candidate.get("lane_id", decision.get("lane_id", ""))),
        "predicted_dose_s": _finite_number(
            candidate["predicted_dose_s"], field="predicted_dose_s"
        ),
        "runway_m": _finite_number(candidate["runway_m"], field="runway_m"),
        "distance_to_merge_m": _finite_number(
            candidate["distance_to_merge_m"], field="distance_to_merge_m"
        ),
        "in_frozen_rule_evidence_domain": in_domain,
        "evidence_role": (
            "frozen_rule_in_domain" if in_domain else "out_of_domain_diagnostic"
        ),
        "response_class": _response_class(effect),
        "paired_physical_effect_reference_units": _finite_number(
            effect["safety_composite_reference_units"],
            field="safety_composite_reference_units",
        ),
        "constraints_pass": bool(effect["constraints_pass"]),
        "failed_constraints": sorted(
            name
            for name, passed in (effect.get("constraints") or {}).items()
            if not bool(passed)
        ),
        "realized_dose_steps": dose_steps,
        "realized_dose_fraction": dose_steps / EXPECTED_OPTION_STEPS,
        "adequate_dose": dose_steps >= MINIMUM_DOSE_STEPS,
        "predecision_pattern": {
            name: (
                _finite_number(pattern[name], field=name)
                if pattern.get(name) is not None
                else None
            )
            for name in PATTERN_FIELDS
        },
        "command_authority": _command_authority(treatment),
        "control_absolute": control_absolute,
        "treatment_absolute": treatment_absolute,
        "treatment_minus_control": _metric_deltas(
            control_absolute, treatment_absolute
        ),
    }
    return row, _replay_delta(control, replay)


CONTINUOUS_PATHS = (
    ("paired_physical_effect_reference_units",),
    ("realized_dose_fraction",),
    ("predicted_dose_s",),
    ("runway_m",),
    ("predecision_pattern", "idm_severe_short_ema"),
    ("predecision_pattern", "idm_wave_short_ema"),
    ("predecision_pattern", "idm_severe_trend"),
    ("predecision_pattern", "idm_wave_energy_trend"),
    ("command_authority", "mean_requested_acceleration_delta_mps2"),
    ("command_authority", "mean_realized_acceleration_delta_mps2"),
    ("command_authority", "mean_projection_survival"),
    ("treatment_absolute", "severe_braking_count_per_step"),
    ("treatment_absolute", "severe_braking_absolute_reference_units"),
    ("treatment_absolute", "acceleration_energy_per_step"),
    ("treatment_absolute", "acceleration_energy_absolute_reference_units"),
    ("treatment_absolute", "absolute_safety_burden_reference_units"),
    ("treatment_absolute", "ttc_count_sum"),
    ("treatment_absolute", "low_ttc_exposure_steps"),
    ("treatment_absolute", "mean_corridor_delay_rate"),
    ("treatment_absolute", "arrived_vehicles"),
    ("treatment_absolute", "merge_discharge_count"),
    ("treatment_absolute", "collapse_max_consecutive_steps"),
    ("treatment_absolute", "immediate_follower_severe_braking_count"),
    ("treatment_absolute", "immediate_follower_low_ttc_count"),
    ("control_absolute", "absolute_safety_burden_reference_units"),
    ("control_absolute", "severe_braking_count_per_step"),
    ("control_absolute", "acceleration_energy_per_step"),
    ("control_absolute", "low_ttc_exposure_steps"),
    ("control_absolute", "mean_corridor_delay_rate"),
    ("control_absolute", "merge_discharge_count"),
    ("treatment_minus_control", "severe_braking_count_per_step"),
    ("treatment_minus_control", "acceleration_energy_per_step"),
    ("treatment_minus_control", "low_ttc_exposure_steps"),
    ("treatment_minus_control", "mean_corridor_delay_rate"),
    ("treatment_minus_control", "arrived_vehicles"),
    ("treatment_minus_control", "merge_discharge_count"),
    ("treatment_minus_control", "collapse_max_consecutive_steps"),
    ("treatment_minus_control", "immediate_follower_severe_braking_count"),
    ("treatment_minus_control", "immediate_follower_low_ttc_count"),
)


def _path_value(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = row
    for name in path:
        value = value[name]
    return value


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    constraints = [row for row in rows if bool(row["constraints_pass"])]
    adequate = [row for row in rows if bool(row["adequate_dose"])]
    return {
        "paired_opportunity_count": len(rows),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "response_class_counts": dict(
            Counter(str(row["response_class"]) for row in rows)
        ),
        "adequate_dose_fraction": len(adequate) / len(rows) if rows else 0.0,
        "constraints_pass_fraction": len(constraints) / len(rows) if rows else 0.0,
        "hard_failure_counts": {
            "incomplete_horizon": sum(
                int(row["treatment_absolute"]["completed_horizon_steps"])
                != EXPECTED_HORIZON_STEPS
                for row in rows
            ),
            "collision_observation_rows": sum(
                int(row["treatment_absolute"]["collision_observations"]) > 0
                for row in rows
            ),
            "teleport_observation_rows": sum(
                int(row["treatment_absolute"]["teleport_observations"]) > 0
                for row in rows
            ),
        },
        "continuous_metrics": {
            ".".join(path): _summary(_path_value(row, path) for row in rows)
            for path in CONTINUOUS_PATHS
        },
        "failed_constraint_counts": dict(
            Counter(
                name for row in rows for name in row.get("failed_constraints", [])
            )
        ),
    }


def analyze(
    roots: Iterable[Path], reference_path: Path
) -> dict[str, Any]:
    references, reference_sha256 = _references(reference_path)
    unique_roots = sorted({Path(root).resolve() for root in roots})
    if not unique_roots:
        raise ValueError("at least one paired source is required")
    rows: list[dict[str, Any]] = []
    replay_deltas: list[dict[str, float]] = []
    sources: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    expected_seed_cells = 0
    no_decision_seed_cells = 0
    all_protocol_hashes: set[str] = set()
    for root in unique_roots:
        summary_path = root / "summary.json"
        summary = _read(summary_path)
        protocol = dict(summary.get("protocol") or {})
        contract = _source_contract(root, protocol, reference_sha256)
        demand = int(contract["demand"])
        if demand not in references:
            raise ValueError(f"{root}: no absolute metric references for demand {demand}")
        source_case_seeds = {int(case["seed"]) for case in summary.get("cases", [])}
        unknown_seeds = source_case_seeds - set(contract["seeds"])
        if unknown_seeds:
            raise ValueError(f"{root}: cases contain undeclared seeds {unknown_seeds}")
        expected_seed_cells += len(contract["seeds"])
        no_decision = sorted(set(contract["seeds"]) - source_case_seeds)
        no_decision_seed_cells += len(no_decision)
        protocol_hash = str(summary.get("protocol_hash", ""))
        if not protocol_hash:
            raise ValueError(f"{root}: missing protocol hash")
        all_protocol_hashes.add(protocol_hash)
        sources.append(
            {
                "root": str(root),
                "summary_sha256": _sha256(summary_path),
                "protocol_hash": protocol_hash,
                "demand_vph": demand,
                "configured_cav_penetration": float(contract["penetration"]),
                "declared_seeds": contract["seeds"],
                "paired_case_count": len(summary.get("cases", [])),
                "no_decision_seeds": no_decision,
                "contract_checks": contract["checks"],
            }
        )
        for case in summary.get("cases", []):
            effects = list(case.get("effects", []))
            if len(effects) != 1:
                raise ValueError(
                    f"{root}: expected one treatment effect for {case.get('case_id')}"
                )
            row, replay = _row(
                root=root,
                source_contract=contract,
                case=case,
                effect=effects[0],
                refs=references[demand],
            )
            if row["sample_id"] in seen_samples:
                raise ValueError(f"duplicate paired sample {row['sample_id']}")
            seen_samples.add(row["sample_id"])
            rows.append(row)
            replay_deltas.append(replay)
    rows.sort(
        key=lambda row: (
            int(row["demand_vph"]),
            float(row["configured_cav_penetration"]),
            int(row["seed"]),
        )
    )
    observed_seeds = sorted({int(row["seed"]) for row in rows})
    demands = sorted({int(row["demand_vph"]) for row in rows})
    penetrations = sorted(
        {float(row["configured_cav_penetration"]) for row in rows}
    )
    in_domain_rows = [row for row in rows if row["in_frozen_rule_evidence_domain"]]
    out_domain_rows = [row for row in rows if not row["in_frozen_rule_evidence_domain"]]
    replay_max = {
        field: max((row.get(field, 0.0) for row in replay_deltas), default=0.0)
        for field in BRANCH_SCALAR_FIELDS
    }
    return {
        "schema": SCHEMA,
        "development_only": True,
        "critical_density_defined": False,
        "binary_safe_but_insufficient_label_defined": False,
        "scientific_contract": (
            "Frozen response_class is reported separately from continuous absolute "
            "post-treatment residual metrics. Demand and configured penetration are "
            "strata, not a deployable critical-density threshold."
        ),
        "prohibited_interpretations": [
            "No row establishes a universal critical density.",
            "No continuous residual metric is an approved safety threshold.",
            "Out-of-domain P30 rows are counterfactual diagnostics, not actions of the frozen rule.",
            "Development rows are not prospective evidence for generalization or wall benefit.",
        ],
        "contamination": {
            "observed_development_seeds": observed_seeds,
            "prospective_seed_floor": PROSPECTIVE_SEED_FLOOR,
            "prospective_seed_used": any(
                seed >= PROSPECTIVE_SEED_FLOOR for seed in observed_seeds
            ),
            "all_observed_seeds_excluded_from_future_prospective_claims": True,
        },
        "contract": {
            "action_set": EXPECTED_ACTION_SET,
            "decision_index": EXPECTED_DECISION_INDEX,
            "option_steps": EXPECTED_OPTION_STEPS,
            "horizon_steps": EXPECTED_HORIZON_STEPS,
            "minimum_dose_steps": MINIMUM_DOSE_STEPS,
            "deadband": DEADBAND,
            "reference_path": str(Path(reference_path).resolve()),
            "reference_sha256": reference_sha256,
            "frozen_rule_evidence_domain": {
                "penetrations": list(FROZEN_DOMAIN_PENETRATIONS),
                "demands_vph": list(FROZEN_DOMAIN_DEMANDS),
            },
        },
        "source_protocol_hash_count": len(all_protocol_hashes),
        "sources": sources,
        "coverage": {
            "declared_seed_cell_count": expected_seed_cells,
            "paired_opportunity_count": len(rows),
            "no_decision_seed_cell_count": no_decision_seed_cells,
            "demands_vph": demands,
            "configured_penetrations": penetrations,
            "balanced_grid": all(
                any(
                    int(source["demand_vph"]) == demand
                    and abs(
                        float(source["configured_cav_penetration"]) - penetration
                    )
                    <= 1e-12
                    for source in sources
                )
                for demand in demands
                for penetration in penetrations
            ),
        },
        "integrity": {
            "all_source_contracts_pass": True,
            "all_branch_identity_pass": True,
            "maximum_absolute_abstain_replay_delta": replay_max,
            "maximum_absolute_abstain_replay_delta_overall": max(
                replay_max.values(), default=0.0
            ),
        },
        "overall": _group(rows),
        "frozen_rule_in_domain": _group(in_domain_rows),
        "out_of_domain_diagnostic": _group(out_domain_rows),
        "by_demand": {
            str(demand): _group(
                [row for row in rows if int(row["demand_vph"]) == demand]
            )
            for demand in demands
        },
        "by_penetration": {
            f"{penetration:.2f}": _group(
                [
                    row
                    for row in rows
                    if abs(
                        float(row["configured_cav_penetration"]) - penetration
                    )
                    <= 1e-12
                ]
            )
            for penetration in penetrations
        },
        "by_seed": {
            str(seed): _group([row for row in rows if int(row["seed"]) == seed])
            for seed in observed_seeds
        },
        "by_demand_and_penetration": {
            f"m{demand}_p{penetration:.2f}": _group(
                [
                    row
                    for row in rows
                    if int(row["demand_vph"]) == demand
                    and abs(
                        float(row["configured_cav_penetration"]) - penetration
                    )
                    <= 1e-12
                ]
            )
            for demand in demands
            for penetration in penetrations
            if any(
                int(row["demand_vph"]) == demand
                and abs(float(row["configured_cav_penetration"]) - penetration)
                <= 1e-12
                for row in rows
            )
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-summary", type=Path)
    parser.add_argument("--inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = list(args.inputs)
    if args.grid_summary is not None:
        roots.extend(roots_from_grid_summary(args.grid_summary.resolve()))
    result = analyze(roots, args.refs.resolve())
    output = args.output.resolve()
    if output.suffix.lower() != ".json":
        output = output / "summary.json"
    atomic_write_json(output, result)
    compact = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
