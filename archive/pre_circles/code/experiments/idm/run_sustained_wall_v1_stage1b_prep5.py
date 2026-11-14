"""Stage 1b-prep-5 (amended) runners: locator C, damping arm D, damping+wall W.

Cells: 2000 / 2100 / 2200 (existing seeds 6061-6090 / 6091-6120 / 6001-6030)
and 2400 / 2600 / 2700 (new seeds 6121-6150 / 6151-6180 / 6181-6210, ramp
400, p = 0.20, long IDM map).

Arms:
- C: native IDM no-command replay (only for the new cells; older cells have
  stored C). Used for D-C and W-C pairing baselines and the locator rows.
- D: frozen analytical damping (zero residual) applied from t = 0, no wall;
  records command activity (CAV-seconds, distinct CAVs, fraction of steps
  with a command) and the prep-4-style RSU observation vector at
  t = 150/200/250/300 (C2 payload).
- W: damping + wall full hold (composite: damping each step, wall override
  on in-band mainline CAVs 200-400 m at 10 m/s / tau 3.0 / 0.3 clip,
  engaged at 251, held to 900, then 5 s taper and tau restore); the damping
  branch is replayed in the same process for exact prefix identity (digest
  additionally compared to the stored D).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import traci

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH, RAMP_EDGE
from config.defaults import DEFAULT_CONFIG
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step, compare_traces
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    TOTAL_STEPS,
    _branch_summary,
    _clean_reset_info,
    _json_safe,
    _wall_dose,
    env_overrides,
)
from experiments.idm.run_sustained_wall_v1_stage1b_prep4_observation_map import (
    _step_features,
)
from experiments.idm.stage1b_runner_core import replay_branch
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2400_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2600_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2700_SEEDS,
    contract_manifest,
    transition_scenario_config,
    validate_seed_role,
)
from policies.damping_plus_wall import DampingPlusWallPolicy
from policies.idm_analytical import build_frozen_idm_analytical_policy
from policies.noop_base import NoOpNativeIDMPolicy

POST_MERGE_EDGE = "1487606178"
CELLS = {
    2000: ("sustained_wall_v1_stage1b_band_locator", SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS),
    2100: ("sustained_wall_v1_stage1b_band_locator", SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS),
    2200: ("sustained_wall_v1_stage1a_development", SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS),
    2400: ("sustained_wall_v1_stage1b_prep5_locator", SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2400_SEEDS),
    2600: ("sustained_wall_v1_stage1b_prep5_locator", SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2600_SEEDS),
    2700: ("sustained_wall_v1_stage1b_prep5_locator", SUSTAINED_WALL_V1_STAGE1B_PREP5_LOCATOR_2700_SEEDS),
}
OBSERVE_TIMES = (150, 200, 250, 300)
DECISION_STEP = 250


def scenario_for(vph: int) -> str:
    if vph in (1900, 2000, 2100, 2200):
        from idm.contract import scenario_config

        return scenario_config(vph).name
    return transition_scenario_config(vph).name


def default_output(arm: str, vph: int) -> Path:
    seeds = CELLS[vph][1]
    return (
        REPO_ROOT
        / "runs"
        / "probe"
        / f"sustained_wall_v1_stage1b_prep5_{arm}_m{vph}_p20_s{seeds[0]}_{seeds[-1]}"
    )


def _replay_damped(
    *, seed: int, scenario_file: str, scenario_root: Path, sumo_binary: str,
    total_steps: int,
) -> Dict[str, Any]:
    """D replay with command-activity + observation-vector capture."""
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": scenario_root,
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_file,
            "sumo_seed": int(seed),
            "cav_penetration": float(C.CAV_PENETRATION),
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
    step_features: Dict[int, Dict[str, Any]] = {}
    command_activity = {"commanded_steps": 0, "cav_seconds": 0.0, "distinct": set()}
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
            diagnostics = getattr(policy, "last_diagnostics", {}) or {}
            rows = diagnostics.get("commands", ())
            if isinstance(rows, list):
                commanded = [str(r["vehicle_id"]) for r in rows if isinstance(r, Mapping) and r.get("vehicle_id")]
            else:
                commanded = []
            audit = _json_safe(policy.last_diagnostics)
            audit["relative_step"] = relative_step
            audit["treatment_state"] = "DAMPING"
            audit["wall_active"] = False
            audits.append(audit)
            if commanded:
                command_activity["commanded_steps"] += 1
                command_activity["cav_seconds"] += len(commanded)
                command_activity["distinct"].update(commanded)
            obs, _rewards, terminated, truncated, _infos = env.step({})
            supervisor_observation = corridor_state(
                env, previous_health, 0, 0, relative_step, inflow_tracker, health_history
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(capture_step(traci, relative_step=relative_step, action=0,
                                          supervisor_observation=supervisor_observation,
                                          active_cav_ids=obs.keys(), digits=6))
            if relative_step in OBSERVE_TIMES:
                step_features[relative_step] = _step_features(env)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_file)
        reset_info.setdefault("paired_seed", int(seed))
        trace = ReplayTrace(
            seed=int(seed), decision_step=0, horizon_steps=total_steps,
            prefix_action=0, post_action=0, treatment_steps=None,
            reset_info=reset_info, snapshots=tuple(snapshots), terminated_early=False,
            controller_metadata={}, controller_audits=tuple(audits),
        )
        return {
            "trace": trace,
            "step_features": step_features,
            "command_activity": {
                "commanded_steps": command_activity["commanded_steps"],
                "cav_seconds": command_activity["cav_seconds"],
                "distinct_cavs": len(command_activity["distinct"]),
                "fraction_steps_with_command": (
                    command_activity["commanded_steps"] / len(snapshots)
                ),
            },
            "h_series": [float(s.supervisor_observation[0]) for s in snapshots],
        }
    finally:
        try:
            policy.close()
        finally:
            env.close()



class _AuditAdapter:
    """Give the damping policy the last_audit surface the core loop reads."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self.last_audit: Dict[str, Any] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        self._policy.reset(env, seed)
        self.last_audit = {}

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        result = self._policy.act(obs_dict)
        self.last_audit = {
            "treatment_state": "DAMPING",
            "wall_active": False,
        }
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def close(self) -> None:
        try:
            self._policy.close()
        except Exception:
            pass


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    mode = str(case["mode"])
    cell = int(case["mainline_vph"])
    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    scenario_file = scenario_for(cell)
    sumo_binary = str(case["sumo_binary"])
    if mode == "C":
        trace = replay_branch(
            seed=seed, arm="C", decision_step=None, scenario_file=scenario_file,
            scenario_root=C.SCENARIO_ROOT_PATH, sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS, policy_factory=NoOpNativeIDMPolicy,
            cav_penetration=C.CAV_PENETRATION,
        )
        summary = _branch_summary(trace, arm="C", decision_step=0)
        payload = {
            "schema": "sustained_wall_v1_stage1b_prep5_locator_seed/1",
            "seed": seed, "mainline_vph": cell, "arm": "C",
            "branch": summary,
            "health_collapse": summary["health_collapse"],
            "physical_collapse": summary["physical_collapse"],
            "delay_s_per_veh": summary["delay"]["mean_per_vehicle_delay_s"],
            "terminal": summary["termination"],
            "wall_clock_seconds": time.time() - started,
        }
    elif mode == "D":
        result = _replay_damped(
            seed=seed, scenario_file=scenario_file,
            scenario_root=C.SCENARIO_ROOT_PATH, sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
        )
        trace = result["trace"]
        payload = {
            "schema": "sustained_wall_v1_stage1b_prep5_damping_seed/1",
            "seed": seed, "mainline_vph": cell, "arm": "D",
            "trace_digest": trace.trace_digest,
            "branch": _branch_summary(trace, arm="D", decision_step=0),
            "command_activity": result["command_activity"],
            "step_features": {str(k): v for k, v in result["step_features"].items()},
            "h_series": [round(float(v), 6) for v in result["h_series"]],
            "wall_clock_seconds": time.time() - started,
        }
        if seed in case.get("repeat_seeds", ()):
            repeated = _replay_damped(
                seed=seed, scenario_file=scenario_file,
                scenario_root=C.SCENARIO_ROOT_PATH, sumo_binary=sumo_binary,
                total_steps=TOTAL_STEPS,
            )
            payload["repeat_check"] = {
                "second_digest": repeated["trace"].trace_digest,
                "exact_identity": repeated["trace"].trace_digest == trace.trace_digest,
            }
    else:  # W
        d_trace = replay_branch(
            seed=seed, arm="D", decision_step=None, scenario_file=scenario_file,
            scenario_root=C.SCENARIO_ROOT_PATH, sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
            policy_factory=lambda: _AuditAdapter(build_frozen_idm_analytical_policy()),
            cav_penetration=C.CAV_PENETRATION,
        )
        w_trace = replay_branch(
            seed=seed, arm="W", decision_step=None, scenario_file=scenario_file,
            scenario_root=C.SCENARIO_ROOT_PATH, sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
            policy_factory=lambda: DampingPlusWallPolicy(decision_step=DECISION_STEP),
            cav_penetration=C.CAV_PENETRATION,
        )
        comparison = compare_traces(d_trace, w_trace)
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        payload = {
            "schema": "sustained_wall_v1_stage1b_prep5_fullhold_seed/1",
            "seed": seed, "mainline_vph": cell, "arm": "W",
            "d_branch": _branch_summary(d_trace, arm="D", decision_step=0),
            "w_branch": _branch_summary(w_trace, arm="W", decision_step=0),
            "d_digest": d_trace.trace_digest,
            "w_digest": w_trace.trace_digest,
            "integrity": {
                "prefix_identical_through_decision_step": bool(
                    first_divergence is None or first_divergence > DECISION_STEP
                ),
                "first_divergence_step": first_divergence,
                "expected_first_divergence_step": DECISION_STEP + 1,
            },
            "wall_dose": _wall_dose(w_trace),
            "wall_clock_seconds": time.time() - started,
        }
    _atomic_json(result_path, payload)
    return {"status": "completed", "mode": mode, "seed": seed, "mainline_vph": cell,
            "path": str(result_path)}




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("C", "D", "W"), default="D")
    parser.add_argument("--mainline-vph", type=int, default=2400)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=2)
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
    role, cell_seeds = CELLS[vph]
    seeds = validate_seed_role(role, args.seeds or cell_seeds)
    if not set(seeds) <= set(cell_seeds):
        raise ValueError(f"seeds {seeds} are outside the m{vph} block")
    output = (args.output or default_output(args.mode, vph)).resolve()
    family = tuple(
        REPO_ROOT / "runs" / "probe" / name
        for name in (
            "sustained_wall_v1_stage1a_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_prep2_wallspeed_scan_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_prep3_hrule_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_prep3_hrule_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_prep3_hrule_m2000_p20_s6061_6090",
            "sustained_wall_v1_stage1b_prep3_se_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_prep4_minhold_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_prep4_legacy_probe",
            "sustained_wall_v1_stage1b_band_locator_m1900_s6031_6060",
            "sustained_wall_v1_stage1b_band_locator_m2000_s6061_6090",
            "sustained_wall_v1_stage1b_band_locator_m2100_s6091_6120",
            "sustained_wall_v1_stage1b_prep4_obsmap_m2000_p20_s6061_6090",
            "sustained_wall_v1_stage1b_prep4_obsmap_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_prep4_obsmap_m2200_p20_s6001_6030",
        )
    )
    prep5_siblings = tuple(
        (REPO_ROOT / "runs" / "probe" / p.name)
        for p in list((REPO_ROOT / "runs" / "probe").glob("sustained_wall_v1_stage1b_prep5_*"))
        + list((REPO_ROOT / "runs" / "probe").glob("det_a1_*"))
        if p.is_dir()
    )
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=(output, *family, *prep5_siblings)
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": f"sustained_wall_v1_stage1b_prep5_{args.mode}_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": f"stage1b_prep5_{args.mode}",
            "cell": {
                "mainline_demand_vph": vph,
                "ramp_demand_vph": C.RAMP_DEMAND_VPH,
                "cav_penetration": C.CAV_PENETRATION,
                "scenario": scenario_for(vph),
            },
            "seeds": seeds,
            "seed_role": role,
            "prospective_evaluation_exclusion": True,
            "arm_definition": {
                "C": "native IDM, no commands",
                "D": "frozen analytical damping (zero residual) from t=0, no wall",
                "W": "damping each step + wall override (200-400 m, 10 m/s, tau 3.0, 0.3 clip) engage 251, hold to 900, 5 s taper",
            }[args.mode],
            "decision_step": DECISION_STEP,
            "contract": C.contract_manifest(),
            "idm_contract": contract_manifest(),
            "prior_seed_artifact_audit": artifact_audit,
            "command": {
                "seeds": seeds,
                "workers": args.workers,
                "sumo_binary": args.sumo_binary,
                "output": str(output),
            },
        },
    )
    cases = [
        {
            "mode": args.mode, "seed": seed, "mainline_vph": vph,
            "repeat_seeds": (cell_seeds[0], cell_seeds[1]) if args.mode == "D" else (),
            "sumo_binary": args.sumo_binary, "output": str(output), "force": args.force,
        }
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps({"status": "all_workers_done", "seeds": seeds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
