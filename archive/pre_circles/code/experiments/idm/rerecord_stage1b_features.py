"""Stage 1c-2 Part A: re-record the p20 RSU feature vectors on the D branches.

For every p20 generation seed (7001-7240) the D branch (frozen analytical
damping, no wall) is replayed with the original generation replay loop and
the corrected capture (``rsu_feature_capture.step_features_fixed``), the
recomputed trace digest is checked against the stored generation digest
(exact replay identity - the capture is metadata and cannot move it), and
the corrected feature vectors at t = 150/250/350 are written to
``stage1b_features_v2_p20_s7001_7240/``. Part C's L0'/L1' fits and the
onset AUCs consume these vectors; the original constant-zero fields are
replaced by real values (cav_count_band, cav_count_upstream, and
ramp_queue_length under the < 5 m/s definition).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import traci

from config.defaults import DEFAULT_CONFIG
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm import deterministic_sumo  # noqa: F401  (race-free SUMO launch, must precede env.core)
from experiments.idm.rsu_feature_capture import step_features_fixed
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    TOTAL_STEPS,
    _clean_reset_info,
    _json_safe,
    env_overrides,
)
from experiments.idm.run_stage1b_generation import (
    DECISION_TIMES,  # noqa: F401  (documented feature times of record)
    HISTORY_STEPS,
    _extract_features,
)
from experiments.idm.stage1b_generation_contract import (
    FEATURE_TIMES,
    SCENARIO_ROOT as GEN_SCENARIO_ROOT,
    demand_for_seed,
    penetration_for_seed,
    write_seed_scenario,
)
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import contract_manifest as idm_contract_manifest
from policies.idm_analytical import build_frozen_idm_analytical_policy

GENERATION_DIR = REPO_ROOT / "runs" / "probe" / "stage1b_generation_p20_s7001_7240"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "probe" / "stage1b_features_v2_p20_s7001_7240"


def replay_d_v2(
    *, seed: int, scenario_name: str, sumo_binary: str, total_steps: int,
    project_root: Optional[Path] = None,
) -> ReplayTrace:
    """The generation D replay with the corrected capture (digest-identical).

    Same loop as ``run_stage1b_generation._replay_d_with_features``; only the
    per-step capture function differs, and captures are controller metadata,
    which the trace digest does not cover. ``project_root`` defaults to the
    generation scenario dir; the smoke test passes the transition dir.
    """
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": str(project_root or GEN_SCENARIO_ROOT),
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_name,
            "sumo_seed": int(seed),
            "cav_penetration": float(penetration_for_seed(seed)),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy = build_frozen_idm_analytical_policy()
    snapshots: list[StepSnapshot] = []
    audits: list[Dict[str, Any]] = []
    health_history: list[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
    live_captures: Dict[int, Dict[str, Any]] = {}
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(env, None, 0, 0, 0, inflow_tracker, health_history)
        snapshots.append(capture_step(traci, relative_step=0, action=0,
                                      supervisor_observation=supervisor_observation,
                                      active_cav_ids=obs.keys(), digits=6))
        audits.append({"relative_step": 0, "treatment_state": "DAMPING", "wall_active": False})
        for relative_step in range(1, total_steps + 1):
            policy.act(obs)
            audit = _json_safe(policy.last_diagnostics)
            audit["relative_step"] = relative_step
            audit["treatment_state"] = "DAMPING"
            audit["wall_active"] = False
            audits.append(audit)
            obs, _rewards, terminated, truncated, _infos = env.step({})
            supervisor_observation = corridor_state(
                env, previous_health, 0, 0, relative_step, inflow_tracker, health_history
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(capture_step(traci, relative_step=relative_step, action=0,
                                          supervisor_observation=supervisor_observation,
                                          active_cav_ids=obs.keys(), digits=6))
            if relative_step in HISTORY_STEPS:
                live_captures[relative_step] = step_features_fixed(env)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_name)
        reset_info.setdefault("paired_seed", int(seed))
        return ReplayTrace(
            seed=int(seed), decision_step=0, horizon_steps=total_steps,
            prefix_action=0, post_action=0, treatment_steps=None,
            reset_info=reset_info, snapshots=tuple(snapshots), terminated_early=False,
            controller_metadata={"captures": live_captures},
            controller_audits=tuple(audits),
        )
    finally:
        try:
            policy.close()
        finally:
            env.close()


def run_seed(case: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    stored_path = GENERATION_DIR / "seeds" / f"seed{seed}.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored_digest = str(stored["arms"]["D"]["trace_digest"])
    demand = demand_for_seed(seed)
    scenario_name = write_seed_scenario(seed, demand)
    trace = replay_d_v2(
        seed=seed, scenario_name=scenario_name, sumo_binary=str(case["sumo_binary"]),
        total_steps=TOTAL_STEPS,
    )
    features = _extract_features(trace)
    payload: Dict[str, Any] = {
        "schema": "stage1b_features_v2_seed/1",
        "seed": seed,
        "hidden_demand_vph": demand,
        "cav_penetration": penetration_for_seed(seed),
        "scenario": scenario_name,
        "digest_stored": stored_digest,
        "digest_recomputed": trace.trace_digest,
        "digest_match": bool(trace.trace_digest == stored_digest),
        "features": features,
        "wall_clock_seconds": time.time() - started,
    }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "digest_match": payload["digest_match"],
            "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from experiments.idm.stage1b_generation_contract import SEEDS_P20

    seeds = tuple(args.seeds) if args.seeds else SEEDS_P20
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "stage1b_features_v2_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1c2_part_a_rerecord",
            "seeds": seeds,
            "seed_role": "stage1b_generation (dev; prospective-excluded)",
            "capture_fix": {
                "cav_counts": "read env._active_cavs (the public attribute does not exist)",
                "ramp_queue_length": "ramp-edge vehicles below 5 m/s (was < 15 m/s, never fired)",
                "ramp_30s_entries": "unchanged (per-step ramp_entries summed over 30 s)",
            },
            "digest_identity_required": True,
            "generation_reference": str(GENERATION_DIR),
            "idm_contract": idm_contract_manifest(),
            "command": {
                "seeds": seeds,
                "workers": args.workers,
                "sumo_binary": args.sumo_binary,
                "output": str(output),
            },
        },
    )
    cases = [
        {"seed": seed, "sumo_binary": args.sumo_binary, "output": str(output),
         "force": args.force}
        for seed in seeds
    ]
    mismatches: list[int] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            if result.get("status") == "completed" and not result["digest_match"]:
                mismatches.append(int(result["seed"]))
            print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"status": "all_workers_done", "seeds": len(seeds),
                      "digest_mismatches": sorted(mismatches)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
