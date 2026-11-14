"""Lightweight pre-treatment feasibility scan for the frozen IDM wall.

No wall command is applied. Each seed follows only the frozen analytical IDM
damping controller; at predeclared decision windows a fresh overlay instance
evaluates the unchanged formation and activation-safety contract.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

import traci

from config.defaults import DEFAULT_CONFIG
from env.core import TrafficShepherdMultiAgentEnv
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    CAV_PENETRATION,
    SCENARIO_ROOT,
    WALL_V5_2_PILOT_SEEDS,
    transition_scenario_config,
)
from policies import DirectControlPolicy
from policies.counterfactual_escalation import CounterfactualEscalationPolicy
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)
from experiments.idm.run_idm_wall_pilot import supervisor_overrides
from experiments.idm.wall_v5_2_contract import (
    DECISION_STEPS,
    PROFILE_ID,
    assert_no_prior_seed_artifacts,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
    validate_pilot_seeds,
)


from experiments import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_OUTPUT = ROOT / "runs" / "eval" / "idm_wall_v5_2_timing_s4301_4312"
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
DEFAULT_DECISIONS = DECISION_STEPS


class _ProbeBase(DirectControlPolicy):
    def act(self, obs_dict):
        return {}


def _env_config(seed: int, max_step: int, sumo_binary: str) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "project_root": SCENARIO_ROOT,
            "sumo_binary": sumo_binary,
            "sumo_config": transition_scenario_config(2700).name,
            "sumo_seed": seed,
            "cav_penetration": CAV_PENETRATION,
            "sumo_step_length": 0.5,
            "max_episode_steps": max_step + 5,
            "eval_mode": False,
            "control_mode": "shepherd_hybrid",
            "reward_mode": "shepherd",
            "centralized_critic": True,
            "use_v2_obs": True,
            "shepherd_modes": ["coast", "recover", "guard"],
            "shepherd_disabled_modes": [],
            "max_neighbors": 20,
            "cav_warmup_steps": 400,
            "max_cavs_critic": 32,
            "local_obs_dim": 124,
            "action_masking": True,
            "use_meter_load": False,
            "use_cav_count": False,
            "use_health_obs": False,
            "use_dv_obs": False,
            "use_margin_obs": False,
            "feasibility_mask": False,
        }
    )
    cfg.update(supervisor_overrides())
    return cfg


def _probe(
    env: Any,
    obs: dict[str, Any],
    seed: int,
    *,
    base_policy_factory=None,
) -> dict[str, Any]:
    factory = require_accepted_damping_factory(base_policy_factory)
    overlay = CounterfactualEscalationPolicy(base_policy=factory(), **overlay_overrides())
    try:
        overlay.reset(env, seed=seed)
        live_ids = overlay._live_ids()
        formation = overlay._formation_status(obs.keys(), live_ids)
        safety = None
        if formation.feasible:
            safety = overlay._activation_safety_status(
                formation.controlled_cav_ids, live_ids
            ).to_dict()
        return {
            "formation": formation.to_dict(),
            "activation_safety": safety,
            "admitted": bool(formation.feasible and safety is not None and safety["safe"]),
        }
    finally:
        overlay.close()


def run_seed(case: dict[str, Any]) -> dict[str, Any]:
    factory = require_accepted_damping_factory(build_frozen_idm_analytical_policy)
    seed = int(case["seed"])
    output = Path(case["output"])
    path = output / "seeds" / f"seed{seed}.json"
    if path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(path)}
    decisions = sorted(int(step) for step in case["decision_steps"])
    env = TrafficShepherdMultiAgentEnv(
        _env_config(seed, max(decisions) + 1, str(case["sumo_binary"]))
    )
    base = factory()
    results = []
    started = time.time()
    try:
        obs, reset_info = env.reset(seed=seed)
        base.reset(env, seed=seed)
        decision_set = set(decisions)
        for relative_step in range(1, max(decisions) + 2):
            base.act(obs)
            decision_step = relative_step - 1
            if decision_step in decision_set:
                result = _probe(
                    env,
                    obs,
                    seed,
                    base_policy_factory=build_frozen_idm_analytical_policy,
                )
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
        "schema": "idm_wall_timing_feasibility_seed/2",
        "wall_profile": PROFILE_ID,
        "seed": seed,
        "reset_info": reset_info,
        "decision_steps": decisions,
        "results": results,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "wall_clock_seconds": time.time() - started,
        "wall_was_applied": False,
    }
    _atomic_json(path, payload)
    return {"status": "completed", "seed": seed, "path": str(path)}


def build_summary(output: Path, seeds: list[int], decisions: list[int]) -> dict[str, Any]:
    rows = []
    missing = []
    for seed in seeds:
        path = output / "seeds" / f"seed{seed}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(str(path))
    by_decision = {}
    for decision in decisions:
        samples = [
            result
            for row in rows
            for result in row["results"]
            if int(result["decision_step"]) == decision
        ]
        reason_counts: dict[str, int] = {}
        for sample in samples:
            reason = str(sample["formation"]["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        feasible = [sample for sample in samples if sample["formation"]["feasible"]]
        admitted = [sample for sample in samples if sample["admitted"]]
        by_decision[str(decision)] = {
            "case_count": len(samples),
            "formation_feasible_count": len(feasible),
            "formation_feasible_fraction": len(feasible) / len(samples) if samples else 0.0,
            "safety_admitted_count": len(admitted),
            "safety_admitted_fraction": len(admitted) / len(samples) if samples else 0.0,
            "formation_reason_counts": reason_counts,
            "feasible_seeds": [
                row["seed"]
                for row in rows
                if any(
                    int(result["decision_step"]) == decision and result["formation"]["feasible"]
                    for result in row["results"]
                )
            ],
        }
    summary = {
        "schema": "idm_wall_timing_feasibility_summary/2",
        "wall_profile": PROFILE_ID,
        "scientific_role": "treatment_timing_diagnosis_no_wall_applied",
        "base_control_arm": frozen_idm_analytical_manifest(),
        "seed_count": len(rows),
        "missing": missing,
        "decision_steps": decisions,
        "by_decision": by_decision,
        "best_decision_by_safety_admission": max(
            decisions,
            key=lambda step: (by_decision[str(step)]["safety_admitted_count"], -step),
        )
        if decisions
        else None,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(WALL_V5_2_PILOT_SEEDS))
    parser.add_argument("--decision-steps", type=int, nargs="+", default=list(DEFAULT_DECISIONS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary", default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = validate_pilot_seeds(args.seeds)
    decisions = sorted(set(int(step) for step in args.decision_steps))
    if not decisions or decisions[0] < 0:
        raise ValueError("decision steps must be nonnegative")
    if tuple(decisions) != DECISION_STEPS:
        raise ValueError(f"v5.2 decision windows are frozen at {DECISION_STEPS}")
    output = args.output.resolve()
    artifact_audit = assert_no_prior_seed_artifacts(
        ROOT / "runs",
        seeds,
        allowed_roots=(output,),
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "idm_wall_timing_feasibility_manifest/2",
            "created_unix_s": time.time(),
            "seeds": seeds,
            "decision_steps": decisions,
            "wall_was_applied": False,
            "base_control_arm": frozen_idm_analytical_manifest(),
            "overlay_overrides": overlay_overrides(),
            "wall_profile": profile_manifest(),
            "prior_seed_artifact_audit": artifact_audit,
            "environment": supervisor_overrides(),
        },
    )
    cases = [
        {
            "seed": seed,
            "decision_steps": decisions,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
        }
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps(build_summary(output, seeds, decisions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
