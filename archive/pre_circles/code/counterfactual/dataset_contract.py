"""Frozen seed and sample-size contract for counterfactual escalation v2.

The locked-test assignments in this module are declarations, not permission to
open those outcomes during model development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CONTRACT_VERSION = "counterfactual_escalation_v2_dataset/5"
PRIMARY_OUTCOME_ORACLE_VERSION = "counterfactual_escalation_safety_service_oracle/1"


@dataclass(frozen=True)
class DatasetCell:
    demand: str
    cav_penetration: float
    model_development_seeds: tuple[int, ...]
    locked_test_seeds: tuple[int, ...]


def _block(first: int, count: int) -> tuple[int, ...]:
    return tuple(range(first, first + count))


PRIMARY_CELLS = (
    DatasetCell("dense", 0.10, _block(1001, 20), _block(2001, 10)),
    DatasetCell("dense", 0.20, _block(1021, 20), _block(2011, 10)),
    DatasetCell("dense", 0.30, _block(1041, 20), _block(2021, 10)),
    DatasetCell("hyper", 0.10, _block(1061, 20), _block(2031, 10)),
    DatasetCell("hyper", 0.20, _block(1081, 20), _block(2041, 10)),
    DatasetCell("hyper", 0.30, _block(1101, 20), _block(2051, 10)),
)

TRANSFER_IDM_SEEDS = _block(3001, 20)
EXCLUDED_PROTOCOL_DEVELOPMENT_SEEDS = tuple(
    sorted(
        {
            *range(101, 135),
            *range(201, 211),
            *range(401, 409),
            *range(501, 513),
            # Frozen next treatment-dose/headroom cohort. These seeds may be
            # used only for the gate and can never enter model evaluation.
            *range(601, 621),
            # Fresh v5.3 safety/service headroom cohort. Frozen before any
            # outcomes are generated and permanently excluded from modeling.
            *range(701, 721),
            # Fresh v5.4 prospective cohort for the versioned predecision
            # safety envelope; never available to model fitting or testing.
            *range(801, 821),
        }
    )
)

# These are pre-fit adequacy gates, not paper success criteria.  A model is not
# fit or described as verified unless the pooled development data satisfy them.
MIN_MODEL_DEVELOPMENT_GROUPS = 60
MIN_WALL_SELECTED_GROUPS = 15
MIN_WALL_SELECTED_CASES = 20


def cell_for(demand: str, cav_penetration: float) -> DatasetCell:
    for cell in PRIMARY_CELLS:
        if cell.demand == demand and abs(cell.cav_penetration - cav_penetration) < 1e-9:
            return cell
    raise ValueError(
        f"no frozen primary cell for demand={demand!r}, cav_penetration={cav_penetration!r}"
    )


def validate_assigned_seeds(
    *,
    role: str,
    demand: str,
    cav_penetration: float,
    seeds: Iterable[int],
) -> tuple[int, ...]:
    requested = tuple(int(seed) for seed in seeds)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("seeds must be a non-empty unique sequence")
    if role in ("model_development", "locked_test", "transfer_test") and set(requested) & set(
        EXCLUDED_PROTOCOL_DEVELOPMENT_SEEDS
    ):
        raise ValueError("protocol-development/headroom seeds cannot enter v2 modeling")
    cell = cell_for(demand, cav_penetration)
    allowed = {
        "model_development": set(cell.model_development_seeds),
        "locked_test": set(cell.locked_test_seeds),
    }.get(role)
    if allowed is None:
        return requested
    unexpected = sorted(set(requested) - allowed)
    if unexpected:
        raise ValueError(
            f"seeds {unexpected} are outside the frozen {role} block for "
            f"{demand}/p{int(round(cav_penetration * 100))}"
        )
    return requested


def contract_manifest() -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "primary_hdv_model": "route-file Krauss mixture",
        "primary_outcome_oracle_version": PRIMARY_OUTCOME_ORACLE_VERSION,
        "outcome_oracle_amendment": {
            "status": "development-stage amendment",
            "adopted_after_inspecting": ["hyper/p20", "hyper/p30"],
            "adopted_before": ["hyper/p10", "all locked tests"],
            "original_delay_oracle_retained_as_sensitivity": True,
            "delay_role": "noninferiority guardrail and secondary outcome",
            "primary_benefits": [
                "hard-brake episode-onset reduction",
                "low-TTC exposure reduction",
                "merge-discharge gain",
            ],
        },
        "prospective_headroom_amendment": {
            "primary_estimand": "safety_service",
            "collapse_prevention_role": "secondary_reported_outcome",
            "global_safety_gate_unchanged": True,
            "fresh_excluded_seed_block": list(range(701, 721)),
            "supervised_headroom_seed_block": list(range(801, 821)),
            "predecision_supervisor": "local_support_density_coherence_v1",
        },
        "decision_steps": [120, 180, 240],
        "model_fit_adequacy_gate": {
            "minimum_independent_development_groups": MIN_MODEL_DEVELOPMENT_GROUPS,
            "minimum_wall_selected_independent_groups": MIN_WALL_SELECTED_GROUPS,
            "minimum_wall_selected_cases": MIN_WALL_SELECTED_CASES,
            "rule": "do not fit or claim a verified predictor before all minima pass",
        },
        "excluded_protocol_development_seeds": list(EXCLUDED_PROTOCOL_DEVELOPMENT_SEEDS),
        "primary_cells": [
            {
                "demand": cell.demand,
                "cav_penetration": cell.cav_penetration,
                "model_development_seeds": list(cell.model_development_seeds),
                "locked_test_seeds": list(cell.locked_test_seeds),
            }
            for cell in PRIMARY_CELLS
        ],
        "idm_transfer_seeds": list(TRANSFER_IDM_SEEDS),
    }
