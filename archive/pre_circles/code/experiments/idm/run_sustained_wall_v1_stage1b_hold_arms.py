"""Stage 1b-prep Part C: sustained-hold arms at 2200/p20 (seeds 6001-6030).

Arms (both native IDM + a band-passage wall branched from the Arm C prefix,
exact identity required through t_d):
  * Arm C (control, replayed for prefix verification)
  * Arm WHOLD_soft: Stage 1a Arm W semantics (band 200-400 m, 10 m/s, tau
    3.0 s, 0.3 m/s^2 clip, mainline only), held until t = 900 s, then 5 s
    taper and tau restore. No health-based release.
  * Arm WHOLD_strong: same, RUNGS[2] from env/wall_supervisor.py (band
    200-800 m, 12 m/s, tau 3.0 s), held until 900 s.

Fixed, rule-blind t_d = 200 s on every seed. The replay machinery is the
generic Stage 1b core (``stage1b_runner_core``), which reuses the Stage 1a
env/obs/digest conventions; Arm C is replayed once per seed so prefix
identity that exact (the stored Stage 1a C digest is checked against the
replayed one and recorded).
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

from config.constants import CAV_VTYPE
from counterfactual.twin_replay import ReplayTrace, compare_traces
from experiments import REPO_ROOT
from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    TOTAL_STEPS,
    _branch_summary,
    _wall_dose,
)
from experiments.idm.stage1b_runner_core import replay_branch
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS,
    contract_manifest,
    scenario_config,
    validate_seed_role,
)
from policies.noop_base import NoOpNativeIDMPolicy
from policies.sustained_wall import SustainedWallPolicy

FIXED_DECISION_STEP = 200
WALL_CAP_S = 900.0
HOLD_WALL_PARAMS = {
    "WHOLD_SOFT": dict(C.WALL_PARAMS),
    "WHOLD_STRONG": {
        "d_lo": 200.0,
        "d_hi": 800.0,
        "wall_speed": 12.0,
        "wall_tau": 3.0,
        "a_max_dec": 0.3,
    },
}
REPEAT_VERIFY_DIGESTS = True


def default_output() -> Path:
    return REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030"


def _hold_policy_factory(arm: str) -> Callable[[], Any]:
    def factory() -> SustainedWallPolicy:
        return SustainedWallPolicy(
            decision_step=FIXED_DECISION_STEP,
            wall_params=dict(HOLD_WALL_PARAMS[arm]),
            recovery_h=0.45,
            recovery_persist_s=1.0e6,  # no health-based release for the hold arms
            max_wall_duration_s=1.0e6,
            wall_cap_s=WALL_CAP_S,
            taper_duration_s=5.0,
        )

    return factory


def _tau_integrity(trace: ReplayTrace) -> Dict[str, Any]:
    release_audit_step: Optional[int] = None
    for audit_row in trace.controller_audits:
        if audit_row.get("release_step") is not None:
            release_audit_step = int(audit_row["release_step"])
            break
    if release_audit_step is None:
        return {"restored": False, "reason": "no_release_step_recorded"}
    probe_steps = [release_audit_step + offset for offset in (3, 6, 10)]
    by_step = {int(snapshot.relative_step): snapshot for snapshot in trace.snapshots}
    maxima: Dict[int, Optional[float]] = {}
    for probe_step in probe_steps:
        snapshot = by_step.get(probe_step)
        values = [
            float(row[9])
            for row in (snapshot.vehicles if snapshot is not None else [])
            if str(row[8]) == CAV_VTYPE
        ]
        maxima[probe_step] = max(values) if values else None
    return {
        "window": probe_steps,
        "max_cav_tau_s_by_step": maxima,
        "restored": all(
            value is not None and value <= C.WALL_PARAMS["wall_tau"] + 1.6
            for value in maxima.values()
        ),
    }


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    sumo_binary = str(case["sumo_binary"])
    scenario_file = scenario_config(C.MAINLINE_DEMAND_VPH).name
    arm_times: Dict[str, float] = {}
    branch_started = time.time()
    control = replay_branch(
        seed=seed,
        arm="C",
        decision_step=FIXED_DECISION_STEP,
        scenario_file=scenario_file,
        scenario_root=C.SCENARIO_ROOT_PATH,
        sumo_binary=sumo_binary,
        total_steps=TOTAL_STEPS,
        policy_factory=NoOpNativeIDMPolicy,
        cav_penetration=C.CAV_PENETRATION,
    )
    arm_times["C"] = time.time() - branch_started
    traces: Dict[str, ReplayTrace] = {"C": control}
    for arm in ("WHOLD_SOFT", "WHOLD_STRONG"):
        branch_started = time.time()
        traces[arm] = replay_branch(
            seed=seed,
            arm=arm,
            decision_step=FIXED_DECISION_STEP,
            scenario_file=scenario_file,
            scenario_root=C.SCENARIO_ROOT_PATH,
            sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
            policy_factory=_hold_policy_factory(arm),
            cav_penetration=C.CAV_PENETRATION,
        )
        arm_times[arm] = time.time() - branch_started

    stored_payload = json.loads(
        (
            REPO_ROOT
            / "runs"
            / "probe"
            / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
            / "seeds"
            / f"seed{seed}.json"
        ).read_text(encoding="utf-8")
    )
    stored_c_digest = str(stored_payload["arms"]["C"]["trace_digest"])

    integrity: Dict[str, Any] = {}
    for arm in ("WHOLD_SOFT", "WHOLD_STRONG"):
        comparison = compare_traces(control, traces[arm])
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        integrity[arm] = {
            "prefix_identical_through_decision_step": bool(
                first_divergence is None or first_divergence > FIXED_DECISION_STEP
            ),
            "first_divergence_step": first_divergence,
            "first_divergence_sim_time_s": None if mismatch is None else mismatch.get("left_sim_time"),
            "comparison_passed": bool(comparison["passed"]),
        }
    payload = {
        "schema": "sustained_wall_v1_stage1b_hold_seed/1",
        "seed": seed,
        "fixed_decision_step": FIXED_DECISION_STEP,
        "rule_blind_t_d": True,
        "cell": {
            "mainline_demand_vph": C.MAINLINE_DEMAND_VPH,
            "ramp_demand_vph": C.RAMP_DEMAND_VPH,
            "cav_penetration": C.CAV_PENETRATION,
        },
        "arms": {
            arm: _branch_summary(trace, arm=arm, decision_step=FIXED_DECISION_STEP)
            for arm, trace in traces.items()
        },
        "integrity": integrity,
        "tau_integrity": {arm: _tau_integrity(trace) for arm, trace in traces.items() if arm != "C"},
        "replayed_c_digest": control.trace_digest,
        "stored_stage1a_c_digest": stored_c_digest,
        "c_replay_digest_matches_stage1a": control.trace_digest == stored_c_digest,
        "wall_dose": {
            arm: _wall_dose(trace) for arm, trace in traces.items() if arm != "C"
        },
        "branch_wall_clock_s": arm_times,
        "wall_clock_seconds": time.time() - started,
    }
    # coverage_full_fraction: fraction of hold steps (201..900) in which every
    # occupied mainline lane has at least one wall member (coverage 1.0).
    hold_audits = {
        arm: [
            audit
            for audit in traces[arm].controller_audits
            if 201 <= int(audit.get("relative_step") or 0) <= 900
        ]
        for arm in ("WHOLD_SOFT", "WHOLD_STRONG")
    }
    for arm in ("WHOLD_SOFT", "WHOLD_STRONG"):
        rows = hold_audits[arm]
        full = sum(
            1
            for audit in rows
            if float(audit.get("lane_coverage_fraction") or 0.0) >= 1.0 - 1e-9
        )
        payload["wall_dose"][arm]["coverage_full_fraction"] = float(full / len(rows)) if rows else 0.0
        payload["wall_dose"][arm]["coverage_full_steps"] = int(full)
        payload["wall_dose"][arm]["hold_window_steps"] = int(len(rows))
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = validate_seed_role("sustained_wall_v1_stage1a_development", args.seeds)
    output = args.output.resolve()
    stage1a_root = (
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
    )
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=(output, stage1a_root)
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "sustained_wall_v1_stage1b_hold_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1b_hold_arms_240s_to_900_s",
            "cell": {
                "mainline_demand_vph": C.MAINLINE_DEMAND_VPH,
                "ramp_demand_vph": C.RAMP_DEMAND_VPH,
                "cav_penetration": C.CAV_PENETRATION,
                "scenario": scenario_config(C.MAINLINE_DEMAND_VPH).name,
            },
            "seeds": seeds,
            "seed_role": "sustained_wall_v1_stage1a_development",
            "prospective_evaluation_exclusion": True,
            "arms": {
                "C": {"definition": "native IDM, no commands; replayed for identity"},
                "WHOLD_SOFT": {
                    "definition": "band 200-400 m, wall_speed 10, tau 3.0, 0.3 clip, "
                    "held to t=900 s, 5 s taper",
                    "wall_params": HOLD_WALL_PARAMS["WHOLD_SOFT"],
                },
                "WHOLD_STRONG": {
                    "definition": "RUNGS[2] band 200-800 m, wall_speed 12, tau 3.0, "
                    "held to t=900 s, 5 s taper",
                    "wall_params": HOLD_WALL_PARAMS["WHOLD_STRONG"],
                },
            },
            "fixed_rule_blind_decision_step": FIXED_DECISION_STEP,
            "hold_until_s": WALL_CAP_S,
            "taper_duration_s": 5.0,
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
            "seed": seed,
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
    print(json.dumps({"status": "all_workers_done", "seeds": seeds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
