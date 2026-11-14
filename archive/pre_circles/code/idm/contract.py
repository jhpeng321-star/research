"""Frozen provenance and seed partitions for the IDM-native paper contract."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

from idm.damping_campaign import (
    TRAINING_SEEDS as NATIVE_DAMPING_TRAINING_SEEDS,
    DEVELOPMENT_SEEDS as NATIVE_DAMPING_DEVELOPMENT_SEEDS,
    ACCEPTANCE_SEEDS as NATIVE_DAMPING_ACCEPTANCE_SEEDS,
)

CONTRACT_VERSION = "idm_decentralized_escalation/2"
ROOT = Path(__file__).resolve().parent.parent
SCENARIO_ROOT = ROOT / "scenarios" / "idm_transition"
NETWORK_PATH = Path(r"D:\research\highway_merge_shepherd_long\osm.net.xml.gz")
LEGACY_PRUNED_CHECKPOINT = (
    ROOT / "runs" / "rllib" / "shepherd_pruned_v1_dense_mpen" / "checkpoints" / "best"
)

DEVELOPMENT_SEEDS = tuple(range(4101, 4121))
ANALYTICAL_PILOT_SEEDS = tuple(range(4101, 4107))
# Added only after the 4107--4120 feature-family diagnosis was complete.
# These remain model-development evidence and never enter checkpoint acceptance.
CONTROLLER_FEATURE_VALIDATION_SEEDS = tuple(range(4121, 4133))
VALIDATION_SEEDS = tuple(range(4201, 4241))
DAMPING_ACCEPTANCE_SEEDS = tuple(range(4201, 4221))
DAMPING_CONFIRMATION_SEEDS = tuple(range(4221, 4241))
WALL_DEVELOPMENT_SEEDS = tuple(range(4301, 4321))
WALL_V5_2_PILOT_SEEDS = tuple(range(4301, 4313))
PREDICTOR_DEVELOPMENT_SEEDS = tuple(range(4401, 4521))
PREDICTOR_VALIDATION_SEEDS = tuple(range(4601, 4661))
FINAL_TEST_SEEDS = tuple(range(5001, 5061))
# 2026-09-02: Stage 1a sustained-wall effect-existence development block.
# Never used before this register (grep-audited); excluded from prospective
# evaluation with the other development and validation partitions.
SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS = tuple(range(6001, 6031))
# 2026-09-02: Stage 1b-prep band locator block (native-IDM arm C only): 30
# seeds per lower-demand cell. Never used before this register (grep-audited).
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_1900_SEEDS = tuple(range(6031, 6061))
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS = tuple(range(6061, 6091))
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS = tuple(range(6091, 6121))
SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_SEEDS = (
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_1900_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS
)
# 2026-09-03: Stage 1b-prep-5 locator blocks (2400/2600/2700 veh/h, p 0.20,
# ramp 400, long IDM map; 2700 block announced by the amendment).
# Grep-audited before registration (no prior assignment).
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2400_SEEDS = tuple(range(6121, 6151))
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2600_SEEDS = tuple(range(6151, 6181))
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2700_SEEDS = tuple(range(6181, 6211))
SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_SEEDS = (
    SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2400_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2600_SEEDS
    + SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2700_SEEDS
)
# 2026-09-03: Stage 1b generation population (demand drawn per seed in
# [2200, 2800] veh/h, ramp 400; p = 0.20 seeds 7001-7240, p = 0.30 seeds
# 7241-7360). The drawn demand is a hidden variable (recorded, never a
# feature). Grep-audited before registration.
STAGE1B_GENERATION_P20_SEEDS = tuple(range(7001, 7241))
STAGE1B_GENERATION_P30_SEEDS = tuple(range(7241, 7361))
STAGE1B_GENERATION_SEEDS = STAGE1B_GENERATION_P20_SEEDS + STAGE1B_GENERATION_P30_SEEDS
# 2026-09-03: Stage 1c release population (plateau-decline demand profile,
# p = 0.20). Grep-audited before registration.
STAGE1C_RELEASE_2200_SEEDS = tuple(range(7401, 7431))
STAGE1C_RELEASE_2400_SEEDS = tuple(range(7431, 7461))
STAGE1C_RELEASE_2600_SEEDS = tuple(range(7461, 7491))
STAGE1C_RELEASE_2700_SEEDS = tuple(range(7491, 7521))
STAGE1C_RELEASE_SEEDS = (STAGE1C_RELEASE_2200_SEEDS + STAGE1C_RELEASE_2400_SEEDS
                         + STAGE1C_RELEASE_2600_SEEDS + STAGE1C_RELEASE_2700_SEEDS)
# 2026-09-05: Stage 2 final-generation population (plateau-decline demand,
# M drawn per seed in [2200, 2800] veh/h from the seed number, ramp 400;
# p = 0.20 seeds 8001-8240, p = 0.30 seeds 8241-8360). The drawn demand is a
# hidden variable (recorded, never a feature). Grep-audited before
# registration.
STAGE2_GENERATION_P20_SEEDS = tuple(range(8001, 8241))
STAGE2_GENERATION_P30_SEEDS = tuple(range(8241, 8361))
STAGE2_GENERATION_SEEDS = STAGE2_GENERATION_P20_SEEDS + STAGE2_GENERATION_P30_SEEDS
# Every partition that may not enter a future prospective (final-test)
# evaluation. FINAL_TEST_SEEDS is deliberately absent: it is the only
# partition reserved for prospective evaluation.
PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS = tuple(
    sorted(
        set(DEVELOPMENT_SEEDS)
        | set(CONTROLLER_FEATURE_VALIDATION_SEEDS)
        | set(VALIDATION_SEEDS)
        | set(WALL_DEVELOPMENT_SEEDS)
        | set(PREDICTOR_DEVELOPMENT_SEEDS)
        | set(PREDICTOR_VALIDATION_SEEDS)
        | set(SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS)
        | set(SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_SEEDS)
        | set(SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_SEEDS)
        | set(STAGE1B_GENERATION_SEEDS)
        | set(STAGE1C_RELEASE_SEEDS)
        | set(STAGE2_GENERATION_SEEDS)
        | set(NATIVE_DAMPING_TRAINING_SEEDS)
        | set(NATIVE_DAMPING_DEVELOPMENT_SEEDS)
        | set(NATIVE_DAMPING_ACCEPTANCE_SEEDS)
    )
)
DEMANDS_VPH = (1900, 2000, 2100, 2200)
TRANSITION_SCAN_DEMANDS_VPH = (2300, 2400, 2500, 2600, 2700, 2800)
CAV_PENETRATION = 0.20

NETWORK_SHA256 = "F58C4EF5F0497CFB887A298CD0FA4876BFE7BCEF101CC0ED2FA6801D503E93F5"
LEGACY_PRUNED_ALGORITHM_STATE_SHA256 = (
    "1191F787C6AA9CF3EB969FA6729E16D0A1B31A58F21FBCA8ABAA9F9463FA6F71"
)


def scenario_config(demand_vph: int) -> Path:
    demand = int(demand_vph)
    if demand not in DEMANDS_VPH:
        raise ValueError(f"demand {demand} is outside frozen cells {DEMANDS_VPH}")
    return SCENARIO_ROOT / f"osm.dense.idm_m{demand}r400.sumocfg"


def transition_scenario_config(demand_vph: int) -> Path:
    demand = int(demand_vph)
    if demand not in TRANSITION_SCAN_DEMANDS_VPH:
        raise ValueError(
            f"demand {demand} is outside transition scan {TRANSITION_SCAN_DEMANDS_VPH}"
        )
    return SCENARIO_ROOT / f"osm.dense.idm_m{demand}r400.sumocfg"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_seed_role(role: str, seeds: Iterable[int]) -> tuple[int, ...]:
    requested = tuple(int(seed) for seed in seeds)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("seeds must be a non-empty unique sequence")
    allowed = {
        "development": set(DEVELOPMENT_SEEDS),
        "analytical_pilot": set(ANALYTICAL_PILOT_SEEDS),
        "controller_feature_validation": set(CONTROLLER_FEATURE_VALIDATION_SEEDS),
        "validation": set(VALIDATION_SEEDS),
        "damping_acceptance": set(DAMPING_ACCEPTANCE_SEEDS),
        "damping_confirmation": set(DAMPING_CONFIRMATION_SEEDS),
        "wall_development": set(WALL_DEVELOPMENT_SEEDS),
        "wall_v5_2_pilot": set(WALL_V5_2_PILOT_SEEDS),
        "predictor_development": set(PREDICTOR_DEVELOPMENT_SEEDS),
        "predictor_validation": set(PREDICTOR_VALIDATION_SEEDS),
        "final_test": set(FINAL_TEST_SEEDS),
        "sustained_wall_v1_stage1a_development": set(
            SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS
        ),
        "sustained_wall_v1_stage1b_band_locator": set(
            SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_SEEDS
        ),
        "sustained_wall_v1_stage1b_prep5_locator": set(
            SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_SEEDS
        ),
        "stage1b_generation": set(STAGE1B_GENERATION_SEEDS),
        "stage1c_release": set(STAGE1C_RELEASE_SEEDS),
        "stage2_generation": set(STAGE2_GENERATION_SEEDS),
    }.get(role)
    if allowed is None:
        raise ValueError(f"unknown seed role {role!r}")
    unexpected = sorted(set(requested) - allowed)
    if unexpected:
        raise ValueError(f"seeds {unexpected} are outside frozen {role} partition")
    return requested


def contract_manifest() -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "sumo_version": "1.27.1",
        "network": str(NETWORK_PATH),
        "network_sha256": NETWORK_SHA256,
        "simulation_step_s": 0.5,
        "controller_interval_s": 1.0,
        "demand_duration_s": 900,
        "mainline_demands_vph": list(DEMANDS_VPH),
        "development_transition_scan_demands_vph": list(TRANSITION_SCAN_DEMANDS_VPH),
        "ramp_demand_vph": 400,
        "cav_penetration": CAV_PENETRATION,
        "cav_placement": "seeded Bernoulli scatter at departure",
        "cav_type": "normal_car (IDM/LC2013)",
        "analytical_pilot_seeds": list(ANALYTICAL_PILOT_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "controller_feature_validation_seeds": list(
            CONTROLLER_FEATURE_VALIDATION_SEEDS
        ),
        "validation_seeds": list(VALIDATION_SEEDS),
        "damping_acceptance_seeds": list(DAMPING_ACCEPTANCE_SEEDS),
        "damping_confirmation_seeds": list(DAMPING_CONFIRMATION_SEEDS),
        "wall_development_seeds": list(WALL_DEVELOPMENT_SEEDS),
        "wall_v5_2_pilot_seeds": list(WALL_V5_2_PILOT_SEEDS),
        "predictor_development_seeds": list(PREDICTOR_DEVELOPMENT_SEEDS),
        "predictor_validation_seeds": list(PREDICTOR_VALIDATION_SEEDS),
        "final_test_seeds": list(FINAL_TEST_SEEDS),
        "sustained_wall_v1_stage1a_development_seeds": list(
            SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS
        ),
        "sustained_wall_v1_stage1b_band_locator_seeds": {
            "m1900": list(SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_1900_SEEDS),
            "m2000": list(SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS),
            "m2100": list(SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS),
        },
        "sustained_wall_v1_stage1b_prep5_locator_seeds": {
            "m2400": list(SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2400_SEEDS),
            "m2600": list(SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2600_SEEDS),
            "m2700": list(SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2700_SEEDS),
        },
        "stage1b_generation_seeds": {
            "p20": list(STAGE1B_GENERATION_P20_SEEDS),
            "p30": list(STAGE1B_GENERATION_P30_SEEDS),
        },
        "stage1c_release_seeds": {
            "m2200": list(STAGE1C_RELEASE_2200_SEEDS),
            "m2400": list(STAGE1C_RELEASE_2400_SEEDS),
            "m2600": list(STAGE1C_RELEASE_2600_SEEDS),
            "m2700": list(STAGE1C_RELEASE_2700_SEEDS),
        },
        "stage2_generation_seeds": {
            "p20": list(STAGE2_GENERATION_P20_SEEDS),
            "p30": list(STAGE2_GENERATION_P30_SEEDS),
        },
        "prospective_evaluation_excluded_seeds": list(
            PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS
        ),
        "collapse_definition": {
            "pre_merge_mean_speed_at_most_mps": 15.0,
            "pre_merge_slow_vehicle_count_at_least": 5,
            "slow_vehicle_speed_below_mps": 15.0,
            "sustained_controller_steps": 60,
        },
        "replay_tolerance": {"state_absolute": 1e-6, "metric_absolute": 1e-9},
        "legacy_pruned_checkpoint_role": "comparison only; not IDM control arm",
        "legacy_pruned_checkpoint": str(LEGACY_PRUNED_CHECKPOINT),
        "legacy_pruned_algorithm_state_sha256": LEGACY_PRUNED_ALGORITHM_STATE_SHA256,
    }
