"""Stage 1c release runner: 4 cells, 5 arms per seed, plateau-decline demand.

Per seed: write the plateau-decline scenario; replay D (damping only) with
per-step approach-density capture over steps 1-100 (for the treatment-blind
W_R2 threshold); replay W_ORACLE / W_R1 / W_R2 / W_HOLD with the release
rule; record rule inputs at 1 Hz; verify exact prefix identity with D
through t_d = 250 and first divergence at 251; repeat D every 10th seed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import traci

from config.constants import PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from config.defaults import DEFAULT_CONFIG
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step, compare_traces
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm import deterministic_sumo  # noqa: F401  race-free launch
from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    _branch_summary,
    _clean_reset_info,
    _json_safe,
    env_overrides,
)
from experiments.idm.stage1c_release_contract import (
    CAV_PENETRATION,
    CELLS,
    DECISION_STEP,
    HORIZON_S,
    SCENARIO_ROOT,
    contract_manifest,
    scene_name_for_seed,
    write_seed_scenario,
)
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import contract_manifest as idm_contract_manifest
from policies.damping_plus_wall import DampingPlusWallPolicy
from policies.idm_analytical import build_frozen_idm_analytical_policy
from policies.stage1c_release import (
    APPROACH_HI,
    APPROACH_LO,
    APPROACH_LANES,
    ReleaseRuleDampingWallPolicy,
)

TOTAL_STEPS = int(HORIZON_S)
APPROACH_LO_ = APPROACH_LO
APPROACH_HI_ = APPROACH_HI


def _approach_density(env: Any) -> float:
    live = {str(v) for v in traci.vehicle.getIDList()}
    count = 0
    for vid in sorted(live):
        try:
            if str(traci.vehicle.getRoadID(vid)) != PRE_MERGE_EDGE:
                continue
            lane_pos = float(traci.vehicle.getLanePosition(vid))
        except Exception:
            continue
        distance = max(0.0, PRE_MERGE_LANE_LENGTH - lane_pos)
        if APPROACH_LO_ <= distance <= APPROACH_HI_:
            count += 1
    return float(count) / ((APPROACH_HI_ - APPROACH_LO_) / 1000.0 * APPROACH_LANES)


class _DensityCaptureDamping:
    def __init__(self, densities: list) -> None:
        self._inner = build_frozen_idm_analytical_policy()
        self._densities = densities
        self.last_audit: Dict[str, Any] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        self._inner.reset(env, seed)
        self.last_audit = {}

    def act(self, obs: Dict[str, Any]) -> Dict[str, object]:
        result = self._inner.act(obs)
        step = getattr(self._inner, "step_index", 0)
        if 1 <= int(getattr(self._inner.engine, "step_index", 0)) <= 100:
            self._densities.append(_approach_density(getattr(self._inner, "env", None)))
        self.last_audit = {"treatment_state": "DAMPING", "wall_active": False}
        return result

    def close(self) -> None:
        self._inner.close()


def _replay_d(seed: int, scenario_name: str, sumo_binary: str) -> tuple[ReplayTrace, float]:
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": SCENARIO_ROOT,
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_name,
            "sumo_seed": int(seed),
            "cav_penetration": CAV_PENETRATION,
            "max_episode_steps": TOTAL_STEPS + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    densities: list[float] = []
    policy = _DensityCaptureDamping(densities)
    snapshots: list[StepSnapshot] = []
    audits: list[Dict[str, Any]] = []
    health_history: list[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(env, None, 0, 0, 0, inflow_tracker, health_history)
        snapshots.append(capture_step(traci, relative_step=0, action=0,
                                      supervisor_observation=supervisor_observation,
                                      active_cav_ids=obs.keys(), digits=6))
        audits.append({"relative_step": 0, "treatment_state": "DAMPING", "wall_active": False})
        for relative_step in range(1, TOTAL_STEPS + 1):
            policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit["relative_step"] = relative_step
            audits.append(audit)
            obs, _r, terminated, truncated, _i = env.step({})
            supervisor_observation = corridor_state(
                env, previous_health, 0, 0, relative_step, inflow_tracker, health_history
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(capture_step(traci, relative_step=relative_step, action=0,
                                          supervisor_observation=supervisor_observation,
                                          active_cav_ids=obs.keys(), digits=6))
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_name)
        reset_info.setdefault("paired_seed", int(seed))
        trace = ReplayTrace(
            seed=int(seed), decision_step=0, horizon_steps=TOTAL_STEPS,
            prefix_action=0, post_action=0, treatment_steps=None,
            reset_info=reset_info, snapshots=tuple(snapshots), terminated_early=False,
            controller_metadata={}, controller_audits=tuple(audits),
        )
        threshold = float(np.percentile(densities, 90.0)) if densities else 0.0
        return trace, threshold
    finally:
        try:
            policy.close()
        finally:
            env.close()


def _replay_w(
    seed: int, scenario_name: str, sumo_binary: str, factory: Callable[[], Any],
) -> ReplayTrace:
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": SCENARIO_ROOT,
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_name,
            "sumo_seed": int(seed),
            "cav_penetration": CAV_PENETRATION,
            "max_episode_steps": TOTAL_STEPS + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy = factory()
    snapshots: list[StepSnapshot] = []
    audits: list[Dict[str, Any]] = []
    health_history: list[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(env, None, 0, 0, 0, inflow_tracker, health_history)
        snapshots.append(capture_step(traci, relative_step=0, action=0,
                                      supervisor_observation=supervisor_observation,
                                      active_cav_ids=obs.keys(), digits=6))
        audits.append({"relative_step": 0, "treatment_state": "DAMPING", "wall_active": False})
        for relative_step in range(1, TOTAL_STEPS + 1):
            policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit["relative_step"] = relative_step
            audits.append(audit)
            obs, _r, terminated, truncated, _i = env.step({})
            supervisor_observation = corridor_state(
                env, previous_health, 0, 0, relative_step, inflow_tracker, health_history
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(capture_step(traci, relative_step=relative_step, action=0,
                                          supervisor_observation=supervisor_observation,
                                          active_cav_ids=obs.keys(), digits=6))
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_name)
        reset_info.setdefault("paired_seed", int(seed))
        audit_key = "_w_rule_inputs"
        return ReplayTrace(
            seed=int(seed), decision_step=0, horizon_steps=TOTAL_STEPS,
            prefix_action=0, post_action=0, treatment_steps=None,
            reset_info=reset_info, snapshots=tuple(snapshots), terminated_early=False,
            controller_metadata={
                audit_key: dict(getattr(policy, "rule_inputs", {})),
            },
            controller_audits=tuple(audits),
        )
    finally:
        try:
            policy.close()
        finally:
            env.close()


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    mainline_vph = int(case["mainline_vph"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    scenario_name = write_seed_scenario(seed, mainline_vph)
    sumo_binary = str(case["sumo_binary"])
    d_trace, r2_threshold = _replay_d(seed, scenario_name, sumo_binary)

    def arm_factory(arm: str) -> Callable[[], Any]:
        if arm == "W_ORACLE":
            return lambda: DampingPlusWallPolicy(decision_step=DECISION_STEP, wall_cap_s=1200.0)
        if arm == "W_HOLD":
            return lambda: DampingPlusWallPolicy(decision_step=DECISION_STEP, wall_cap_s=900.0)
        if arm == "W_R1":
            return lambda: ReleaseRuleDampingWallPolicy(
                decision_step=DECISION_STEP, rule="R1",
                min_hold_s=120.0, hard_max_hold_s=900.0)
        return lambda: ReleaseRuleDampingWallPolicy(
            decision_step=DECISION_STEP, rule="R2",
            r2_threshold_veh_km_lane=r2_threshold, min_hold_s=120.0,
            hard_max_hold_s=900.0)

    traces: Dict[str, ReplayTrace] = {"D": d_trace}
    for arm in ("W_ORACLE", "W_R1", "W_R2", "W_HOLD"):
        traces[arm] = _replay_w(seed, scenario_name, sumo_binary, arm_factory(arm))
    integrity: Dict[str, Any] = {}
    for arm in ("W_ORACLE", "W_R1", "W_R2", "W_HOLD"):
        comparison = compare_traces(d_trace, traces[arm])
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        integrity[arm] = {
            "prefix_identical_through_t_d": bool(
                first_divergence is None or first_divergence > DECISION_STEP
            ),
            "first_divergence_step": first_divergence,
            "expected_first_divergence_step": DECISION_STEP + 1,
        }
    payload: Dict[str, Any] = {
        "schema": "stage1c_release_seed/1",
        "seed": seed,
        "mainline_demand_vph": mainline_vph,
        "scenario": scenario_name,
        "r2_threshold_veh_km_lane": r2_threshold,
        "arms": {
            arm: _branch_summary(trace, arm=arm, decision_step=0)
            for arm, trace in traces.items()
        },
        "rule_inputs": {
            arm: traces[arm].controller_metadata.get("_w_rule_inputs", {})
            for arm in ("W_ORACLE", "W_R1", "W_R2", "W_HOLD")
        },
        "integrity": integrity,
        "repeat_check": None,
        "wall_dose": {arm: None for arm in ("W_ORACLE", "W_R1", "W_R2", "W_HOLD")},
        "wall_clock_seconds": time.time() - started,
    }
    if (seed - case["repeat_offset"]) % 10 == 0:
        repeated, _t = _replay_d(seed, scenario_name, sumo_binary)
        payload["repeat_check"] = {
            "second_digest": repeated.trace_digest,
            "exact_identity": repeated.trace_digest == d_trace.trace_digest,
        }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "mainline_vph": mainline_vph,
            "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-vph", type=int, default=2200)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    vph = int(args.mainline_vph)
    if vph not in CELLS:
        raise ValueError(f"cells are {sorted(CELLS)}")
    seeds = tuple(args.seeds) if args.seeds else CELLS[vph]
    if not set(seeds) <= set(CELLS[vph]):
        raise ValueError(f"seeds {seeds} are outside the m{vph} 1c block")
    output = (args.output or REPO_ROOT / "runs" / "probe"
              / f"stage1c_release_m{vph}_p20_s{seeds[0]}_{seeds[-1]}").resolve()
    family = tuple(
        (REPO_ROOT / "runs" / "probe" / p.name)
        for p in (REPO_ROOT / "runs" / "probe").glob("*stage1*")
        if p.is_dir()
    ) + tuple(
        (REPO_ROOT / "runs" / "probe" / p.name)
        for p in (REPO_ROOT / "runs" / "probe").glob("det_a*")
        if p.is_dir()
    )
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=(output, *family)
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "stage1c_release_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1c_release_state_based_release",
            "cell": {"mainline_demand_vph": vph, "ramp_vph": 400,
                     "cav_penetration": CAV_PENETRATION},
            "seeds": seeds,
            "seed_role": "stage1c_release",
            "prospective_evaluation_exclusion": True,
            "contract": contract_manifest(),
            "idm_contract": idm_contract_manifest(),
            "horizon_steps": TOTAL_STEPS,
            "prior_seed_artifact_audit": artifact_audit,
            "command": {"seeds": seeds, "workers": args.workers,
                        "sumo_binary": args.sumo_binary, "output": str(output)},
        },
    )
    cases = [
        {"seed": seed, "mainline_vph": vph, "sumo_binary": args.sumo_binary,
         "output": str(output), "force": args.force, "repeat_offset": seeds[0]}
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps({"status": "all_workers_done", "seeds": seeds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
