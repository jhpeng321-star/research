"""Run the frozen treatment-blind v5.3 prospective feasibility grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

import traci

from env.core import TrafficShepherdMultiAgentEnv
from experiments import REPO_ROOT
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.scan_idm_wall_timing import _env_config as _base_env_config
from experiments.idm.wall_v5_3_support_contract import (
    CAV_PENETRATIONS,
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    SUPPORT_SEEDS,
    assert_no_prior_seed_artifacts,
    availability_rule,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
    support_manifest,
    validate_support_seeds,
)
from experiments.io import atomic_write_json
from idm.contract import transition_scenario_config
from policies.counterfactual_escalation import CounterfactualEscalationPolicy
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


DEFAULT_OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_wall_v5_3_feasibility_p10_p30_s4313_4332"


def _env_config(seed: int, max_step: int, sumo_binary: str, penetration: float) -> dict[str, Any]:
    cfg = _base_env_config(seed, max_step, sumo_binary)
    cfg["sumo_config"] = transition_scenario_config(DEMAND_MAINLINE_VPH).name
    cfg["cav_penetration"] = float(penetration)
    return cfg


def _probe(env: Any, obs: dict[str, Any], seed: int) -> dict[str, Any]:
    factory = require_accepted_damping_factory(build_frozen_idm_analytical_policy)
    overlay = CounterfactualEscalationPolicy(base_policy=factory(), **overlay_overrides())
    try:
        overlay.reset(env, seed=seed)
        live_ids = overlay._live_ids()
        formation = overlay._formation_status(obs.keys(), live_ids)
        formation_dict = formation.to_dict()
        selected_lane_ids = tuple(
            sorted(
                {
                    lane_id
                    for cav_id in formation.controlled_cav_ids
                    if (lane_id := overlay._lane_id(cav_id)) is not None
                }
            )
        )
        occupied = set(formation.occupied_lane_ids)
        full_coverage = bool(occupied) and set(formation.required_lane_ids) == occupied and set(selected_lane_ids) == occupied
        requested = float(overlay.planned_treatment_duration_s or 0.0)
        projected = requested if formation.feasible else 0.0
        fraction = projected / requested if requested > 0.0 else 0.0
        safety = None
        if formation.feasible and full_coverage:
            safety = overlay._activation_safety_status(
                formation.controlled_cav_ids, live_ids
            ).to_dict()
        admitted = bool(
            formation.feasible
            and full_coverage
            and fraction >= float(availability_rule()["minimum_projected_duration_fraction"])
            and safety is not None
            and safety["safe"]
        )
        return {
            "formation": formation_dict,
            "lane_coverage": {
                "selected_lane_ids": selected_lane_ids,
                "full_occupied_lane_coverage": full_coverage,
                "coverage_fraction": len(set(selected_lane_ids) & occupied) / len(occupied) if occupied else 0.0,
            },
            "projected_dose": {
                "requested_duration_s": requested,
                "projected_duration_s": projected,
                "projected_duration_fraction": fraction,
                "is_predecision_projection": True,
                "is_realized_treatment_dose": False,
            },
            "activation_safety": safety,
            "admitted": admitted,
        }
    finally:
        overlay.close()


def run_cell(case: dict[str, Any]) -> dict[str, Any]:
    factory = require_accepted_damping_factory(build_frozen_idm_analytical_policy)
    seed = int(case["seed"])
    penetration = float(case["cav_penetration"])
    output = Path(case["output"])
    path = output / f"p{round(penetration * 100):02d}" / "seeds" / f"seed{seed}.json"
    if path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "penetration": penetration, "path": str(path)}
    env = TrafficShepherdMultiAgentEnv(
        _env_config(seed, max(DECISION_STEPS) + 1, str(case["sumo_binary"]), penetration)
    )
    base = factory()
    results: list[dict[str, Any]] = []
    started = time.time()
    try:
        obs, _ = env.reset(seed=seed)
        base.reset(env, seed=seed)
        decisions = set(DECISION_STEPS)
        for relative_step in range(1, max(DECISION_STEPS) + 2):
            base.act(obs)
            decision_step = relative_step - 1
            if decision_step in decisions:
                row = _probe(env, obs, seed)
                row.update(
                    decision_step=decision_step,
                    request_step=relative_step,
                    sim_time_s=float(traci.simulation.getTime()),
                    active_cav_count=len(obs),
                    damping_command_count=len(base.last_diagnostics.get("commands", ())),
                )
                results.append(row)
            obs, _, terminated, truncated, _ = env.step({})
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
    finally:
        base.close()
        env.close()
    payload = {
        "schema": "idm_wall_v5_3_feasibility_seed/1",
        "scientific_role": "treatment_blind_v5_3_feasibility",
        "wall_profile": profile_manifest(),
        "base_control_arm": frozen_idm_analytical_manifest(),
        "seed": seed,
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "cav_penetration": penetration,
        "decision_steps": list(DECISION_STEPS),
        "results": results,
        "wall_clock_seconds": time.time() - started,
        "wall_was_applied": False,
        "treatment_outcomes_read": False,
    }
    atomic_write_json(path, payload)
    return {"status": "completed", "seed": seed, "penetration": penetration, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = validate_support_seeds(SUPPORT_SEEDS)
    output = args.output.resolve()
    audit = assert_no_prior_seed_artifacts(REPO_ROOT / "runs", seeds, allowed_roots=(output,))
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output / "manifest.json",
        {
            "schema": "idm_wall_v5_3_feasibility_manifest/1",
            "scientific_role": "treatment_blind_v5_3_feasibility",
            "created_unix_s": time.time(),
            "support_contract": support_manifest(),
            "demand_mainline_vph": DEMAND_MAINLINE_VPH,
            "demand_ramp_vph": DEMAND_RAMP_VPH,
            "cav_penetrations": list(CAV_PENETRATIONS),
            "seeds": list(seeds),
            "decision_steps": list(DECISION_STEPS),
            "wall_was_applied": False,
            "treatment_outcomes_read": False,
            "wall_profile": profile_manifest(),
            "base_control_arm": frozen_idm_analytical_manifest(),
            "overlay_overrides": overlay_overrides(),
            "environment": supervisor_overrides(),
            "availability_rule": availability_rule(),
            "prospective_seed_artifact_audit": audit,
        },
    )
    cases = [
        {
            "seed": seed,
            "cav_penetration": penetration,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
        }
        for penetration in CAV_PENETRATIONS
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_cell, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

