"""Frozen treatment-blind compact-wall penetration feasibility scan.

This runner follows the analytical damping controller only.  At the four
predeclared decision steps it evaluates the implemented v5.2 compact full
occupied-lane wall contract without issuing a wall command.  It records no
treatment branch and no outcome or benefit label.
"""

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
from experiments.idm.run_idm_wall_pilot import supervisor_overrides
from experiments.idm.scan_idm_wall_timing import _env_config as _base_env_config
from experiments.io import atomic_write_json
from idm.contract import (
    WALL_V5_2_PILOT_SEEDS,
    transition_scenario_config,
)
from policies.counterfactual_escalation import CounterfactualEscalationPolicy
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)
from experiments.idm.wall_v5_2_contract import (
    DECISION_STEPS,
    PROFILE_ID,
    assert_no_prior_seed_artifacts,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
    validate_pilot_seeds,
)


ROOT = REPO_ROOT
DEFAULT_OUTPUT = ROOT / "runs" / "probe" / "idm_compact_wall_feasibility_p10_p30_s4301_4312"
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
DEMAND_MAINLINE_VPH = 2700
DEMAND_RAMP_VPH = 400
CAV_PENETRATIONS = (0.10, 0.15, 0.20, 0.25, 0.30)
SEEDS = WALL_V5_2_PILOT_SEEDS
MINIMUM_PROJECTED_DURATION_FRACTION = 0.80


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
        full_coverage = (
            bool(occupied)
            and set(formation.required_lane_ids) == occupied
            and set(selected_lane_ids) == occupied
        )
        projected_duration_s = (
            float(overlay.planned_treatment_duration_s or 0.0)
            if formation.feasible
            else 0.0
        )
        requested_duration_s = float(overlay.planned_treatment_duration_s or 0.0)
        projected_fraction = (
            projected_duration_s / requested_duration_s if requested_duration_s > 0.0 else 0.0
        )
        safety = None
        if formation.feasible and full_coverage:
            safety = overlay._activation_safety_status(
                formation.controlled_cav_ids, live_ids
            ).to_dict()
        admitted = bool(
            formation.feasible
            and full_coverage
            and projected_fraction >= MINIMUM_PROJECTED_DURATION_FRACTION
            and safety is not None
            and safety["safe"]
        )
        return {
            "formation": formation_dict,
            "lane_coverage": {
                "selected_lane_ids": selected_lane_ids,
                "full_occupied_lane_coverage": full_coverage,
                "coverage_fraction": (
                    len(set(selected_lane_ids) & occupied) / len(occupied) if occupied else 0.0
                ),
            },
            # Compact formation feasibility projects the complete frozen pulse.
            # This is a predecision contract projection, not realized treatment dose.
            "projected_dose": {
                "requested_duration_s": requested_duration_s,
                "projected_duration_s": projected_duration_s,
                "projected_duration_fraction": projected_fraction,
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
    token = f"p{round(penetration * 100):02d}"
    output = Path(case["output"])
    path = output / token / "seeds" / f"seed{seed}.json"
    if path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "penetration": penetration, "path": str(path)}

    decisions = list(DECISION_STEPS)
    env = TrafficShepherdMultiAgentEnv(
        _env_config(seed, max(decisions) + 1, str(case["sumo_binary"]), penetration)
    )
    base = factory()
    results = []
    started = time.time()
    try:
        obs, _reset_info = env.reset(seed=seed)
        base.reset(env, seed=seed)
        decision_set = set(decisions)
        for relative_step in range(1, max(decisions) + 2):
            base.act(obs)
            decision_step = relative_step - 1
            if decision_step in decision_set:
                result = _probe(env, obs, seed)
                result.update(
                    {
                        "decision_step": decision_step,
                        "request_step": relative_step,
                        "sim_time_s": float(traci.simulation.getTime()),
                        "active_cav_count": len(obs),
                        "damping_command_count": len(base.last_diagnostics.get("commands", ())),
                    }
                )
                results.append(result)
            obs, _rewards, terminated, truncated, _info = env.step({})
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
    finally:
        base.close()
        env.close()

    payload = {
        "schema": "idm_compact_wall_feasibility_seed/1",
        "scientific_role": "treatment_blind_compact_wall_feasibility",
        "wall_profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "seed": seed,
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "cav_penetration": penetration,
        "decision_steps": decisions,
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
    parser.add_argument(
        "--sumo-binary", default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = validate_pilot_seeds(SEEDS)
    output = args.output.resolve()
    contamination_audit = assert_no_prior_seed_artifacts(
        ROOT / "runs",
        seeds,
        allowed_roots=(output,),
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output / "manifest.json",
        {
            "schema": "idm_compact_wall_feasibility_manifest/1",
            "scientific_role": "treatment_blind_compact_wall_feasibility",
            "created_unix_s": time.time(),
            "demand_mainline_vph": DEMAND_MAINLINE_VPH,
            "demand_ramp_vph": DEMAND_RAMP_VPH,
            "cav_penetrations": list(CAV_PENETRATIONS),
            "seeds": list(seeds),
            "seed_role": "wall_v5_2_pilot_treatment_blind_feasibility",
            "prospective_seed_artifact_audit": contamination_audit,
            "decision_steps": list(DECISION_STEPS),
            "wall_was_applied": False,
            "treatment_outcomes_read": False,
            "wall_profile_id": PROFILE_ID,
            "wall_profile": profile_manifest(),
            "base_control_arm": frozen_idm_analytical_manifest(),
            "overlay_overrides": overlay_overrides(),
            "environment": supervisor_overrides(),
            "availability_rule": {
                "minimum_admitted_seeds": 8,
                "minimum_admitted_seed_windows": 12,
                "minimum_distinct_decision_steps": 2,
                "minimum_projected_duration_fraction": (
                    MINIMUM_PROJECTED_DURATION_FRACTION
                ),
                "select_lowest_passing_penetration": True,
            },
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
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(run_cell, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
