"""Frozen contract for the minimal v5.5 damping density mapper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from experiments import REPO_ROOT
from idm.contract import (
    NETWORK_PATH,
    NETWORK_SHA256,
    PREDICTOR_DEVELOPMENT_SEEDS,
    SCENARIO_ROOT,
    file_sha256,
    scenario_config,
    transition_scenario_config,
)
from policies.idm_analytical import frozen_idm_analytical_manifest


SCHEMA = "idm_v5_5_damping_density_mapper_contract/1"
SCIENTIFIC_ROLE = "development_regime_and_collapse_headroom_identification"
MAINLINE_DEMANDS_VPH = (2200, 2400, 2700)
RAMP_DEMAND_VPH = 400
CAV_PENETRATION = 0.30
DEVELOPMENT_SEEDS = tuple(range(4401, 4407))
REPEAT_SEED = 4401
ANALYSIS_CLOCK = "sumo_time"
ANALYSIS_START_S = 620
ANALYSIS_END_S = 1080
PREVALENCE_LOOKBACK_S = 60
PRIMARY_QUEUE_COUNT = 5
PRIMARY_SPEED_THRESHOLD_MPS = 15.0
PRIMARY_SUSTAIN_S = 60
SECONDARY_HEALTH_THRESHOLD = 0.15
SECONDARY_HEALTH_SUSTAIN_S = 60
TELEMETRY_PERIOD_S = 1
SUMO_STEP_LENGTH_S = 0.5
CONTROLLER_INTERVAL_S = 1.0
EXPECTED_PRIMARY_BRANCHES = 18
EXPECTED_REPEAT_BRANCHES = 3
EXPECTED_BRANCHES = EXPECTED_PRIMARY_BRANCHES + EXPECTED_REPEAT_BRANCHES
DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "probe" / "idm_v5_5_damping_density_mapper_p30_s4401_4406"
)


def scenario_path(demand_vph: int) -> Path:
    demand = int(demand_vph)
    if demand == 2200:
        return scenario_config(demand)
    return transition_scenario_config(demand)


def route_path(demand_vph: int) -> Path:
    return SCENARIO_ROOT / f"idm_m{int(demand_vph)}r{RAMP_DEMAND_VPH}.rou.xml"


def _sha256_json(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_inputs(
    *,
    demands: Iterable[int],
    ramp_demand: int,
    cav_penetration: float,
    seeds: Iterable[int],
    analysis_clock: str,
    analysis_start: int,
    analysis_end: int,
    prevalence_lookback_s: int,
    primary_queue_count: int,
    primary_speed_threshold_mps: float,
    primary_sustain_s: int,
    secondary_health_threshold: float,
    secondary_health_sustain_s: int,
    repeat_seed: int,
) -> None:
    values = {
        "demands": tuple(int(value) for value in demands),
        "ramp_demand": int(ramp_demand),
        "cav_penetration": float(cav_penetration),
        "seeds": tuple(int(value) for value in seeds),
        "analysis_clock": str(analysis_clock),
        "analysis_start": int(analysis_start),
        "analysis_end": int(analysis_end),
        "prevalence_lookback_s": int(prevalence_lookback_s),
        "primary_queue_count": int(primary_queue_count),
        "primary_speed_threshold_mps": float(primary_speed_threshold_mps),
        "primary_sustain_s": int(primary_sustain_s),
        "secondary_health_threshold": float(secondary_health_threshold),
        "secondary_health_sustain_s": int(secondary_health_sustain_s),
        "repeat_seed": int(repeat_seed),
    }
    expected = {
        "demands": MAINLINE_DEMANDS_VPH,
        "ramp_demand": RAMP_DEMAND_VPH,
        "cav_penetration": CAV_PENETRATION,
        "seeds": DEVELOPMENT_SEEDS,
        "analysis_clock": ANALYSIS_CLOCK,
        "analysis_start": ANALYSIS_START_S,
        "analysis_end": ANALYSIS_END_S,
        "prevalence_lookback_s": PREVALENCE_LOOKBACK_S,
        "primary_queue_count": PRIMARY_QUEUE_COUNT,
        "primary_speed_threshold_mps": PRIMARY_SPEED_THRESHOLD_MPS,
        "primary_sustain_s": PRIMARY_SUSTAIN_S,
        "secondary_health_threshold": SECONDARY_HEALTH_THRESHOLD,
        "secondary_health_sustain_s": SECONDARY_HEALTH_SUSTAIN_S,
        "repeat_seed": REPEAT_SEED,
    }
    if values != expected:
        raise ValueError(f"density mapper arguments differ from frozen contract: {values}")
    if not set(DEVELOPMENT_SEEDS).issubset(set(PREDICTOR_DEVELOPMENT_SEEDS)):
        raise RuntimeError("density mapper seeds escaped predictor-development partition")


def contract_manifest() -> dict:
    scenarios = []
    for demand in MAINLINE_DEMANDS_VPH:
        config = scenario_path(demand)
        route = route_path(demand)
        if not config.is_file() or not route.is_file():
            raise FileNotFoundError(f"missing frozen scenario assets for demand {demand}")
        scenarios.append(
            {
                "mainline_demand_vph": demand,
                "ramp_demand_vph": RAMP_DEMAND_VPH,
                "sumo_config": str(config),
                "sumo_config_sha256": file_sha256(config),
                "route_file": str(route),
                "route_file_sha256": file_sha256(route),
            }
        )
    payload = {
        "schema": SCHEMA,
        "scientific_role": SCIENTIFIC_ROLE,
        "interpretation_limit": (
            "development regime/headroom screen only; no no-op arm, critical-density "
            "estimate, damping-benefit claim, predictor fitting, or validation claim"
        ),
        "network": str(NETWORK_PATH),
        "network_sha256": NETWORK_SHA256,
        "scenarios": scenarios,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "intervention_schedule": "DAMPING_ONLY at every controller step",
        "cav_penetration": CAV_PENETRATION,
        "cav_placement": "seeded Bernoulli scatter at departure",
        "hdv_model": "IDM",
        "lane_change_model": "LC2013",
        "sumo_step_length_s": SUMO_STEP_LENGTH_S,
        "controller_interval_s": CONTROLLER_INTERVAL_S,
        "analysis_clock": ANALYSIS_CLOCK,
        "analysis_start_s": ANALYSIS_START_S,
        "analysis_end_s": ANALYSIS_END_S,
        "prevalence_lookback_s": PREVALENCE_LOOKBACK_S,
        "primary_label": {
            "endpoint": "premerge sustained congestion",
            "queue_count_at_least": PRIMARY_QUEUE_COUNT,
            "premerge_mean_speed_at_most_mps": PRIMARY_SPEED_THRESHOLD_MPS,
            "sustain_s": PRIMARY_SUSTAIN_S,
            "prevalence_incidence_separated": True,
        },
        "secondary_diagnostic": {
            "endpoint": "legacy merge-zone health",
            "health_strictly_below": SECONDARY_HEALTH_THRESHOLD,
            "sustain_s": SECONDARY_HEALTH_SUSTAIN_S,
            "not_a_primary_label": True,
        },
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "repeat_seed": REPEAT_SEED,
        "primary_branch_count": EXPECTED_PRIMARY_BRANCHES,
        "repeat_branch_count": EXPECTED_REPEAT_BRANCHES,
        "total_branch_count": EXPECTED_BRANCHES,
        "validation_and_final_partitions_opened": False,
    }
    return {**payload, "contract_fingerprint_sha256": _sha256_json(payload)}


__all__ = [
    "ANALYSIS_CLOCK",
    "ANALYSIS_END_S",
    "ANALYSIS_START_S",
    "CAV_PENETRATION",
    "DEFAULT_OUTPUT",
    "DEVELOPMENT_SEEDS",
    "EXPECTED_BRANCHES",
    "EXPECTED_PRIMARY_BRANCHES",
    "EXPECTED_REPEAT_BRANCHES",
    "MAINLINE_DEMANDS_VPH",
    "PREVALENCE_LOOKBACK_S",
    "PRIMARY_QUEUE_COUNT",
    "PRIMARY_SPEED_THRESHOLD_MPS",
    "PRIMARY_SUSTAIN_S",
    "RAMP_DEMAND_VPH",
    "REPEAT_SEED",
    "SCIENTIFIC_ROLE",
    "SECONDARY_HEALTH_SUSTAIN_S",
    "SECONDARY_HEALTH_THRESHOLD",
    "contract_manifest",
    "route_path",
    "scenario_path",
    "validate_inputs",
]
