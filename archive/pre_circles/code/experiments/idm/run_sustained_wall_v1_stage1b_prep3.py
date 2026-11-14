"""Stage 1b-prep-3 runners: HRULE closed-loop h-rule and SE single-engagement.

HRULE (per cell 2200/2100/2000, seeds 6001-6030 / 6091-6120 / 6061-6090):
legacy hysteresis (arm h < 0.25, release h > 0.45, instant, re-engagement
allowed, gate at step 100) on the sustained wall actuator (200-400 m,
10 m/s, tau 3.0, 0.3 m/s^2 clip, mainline only).

SE (2100 only, seeds 6091-6120): engage at fixed rule-blind t_d = 150 or
250 regardless of h; release by the legacy rule (h > 0.45, instant) or at
the demand end; no re-engagement.

Arm C is replayed per seed (digest compared with the stored cell C); prefix
identity through t = 100 s (HRULE) or t = t_d (SE) is checked against the
replayed C.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from counterfactual.twin_replay import ReplayTrace, compare_traces
from experiments import REPO_ROOT
from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    TOTAL_STEPS,
    _branch_summary,
)
from experiments.idm.stage1b_runner_core import replay_branch
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
    contract_manifest,
    scenario_config,
    validate_seed_role,
)
from policies.noop_base import NoOpNativeIDMPolicy
from policies.sustained_wall_hrule import (
    HysteresisWallPolicy,
    SingleEngagementWallPolicy,
)

CELLS = {
    2200: {
        "seeds": SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS,
        "stored_c_root": REPO_ROOT
        / "runs"
        / "probe"
        / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030",
        "stored_c_key": "arms",
    },
    2100: {
        "seeds": SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
        "stored_c_root": REPO_ROOT
        / "runs"
        / "probe"
        / "sustained_wall_v1_stage1b_band_locator_m2100_s6091_6120",
        "stored_c_key": "branch",
    },
    2000: {
        "seeds": SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS,
        "stored_c_root": REPO_ROOT
        / "runs"
        / "probe"
        / "sustained_wall_v1_stage1b_band_locator_m2000_s6061_6090",
        "stored_c_key": "branch",
    },
}


def default_output(mode: str, mainline_vph: int, seeds: tuple) -> Path:
    if mode == "hrule":
        return (
            REPO_ROOT
            / "runs"
            / "probe"
            / f"sustained_wall_v1_stage1b_prep3_hrule_m{mainline_vph}_p20_s{seeds[0]}_{seeds[-1]}"
        )
    return (
        REPO_ROOT
        / "runs"
        / "probe"
        / f"sustained_wall_v1_stage1b_prep3_se_m{mainline_vph}_p20_s{seeds[0]}_{seeds[-1]}"
    )


def _tau_integrity(trace: ReplayTrace) -> Dict[str, Any]:
    """Max CAV tau 3/6/10 steps after the last wall-active step."""
    last_on = None
    for audit in trace.controller_audits:
        if bool(audit.get("wall_active")):
            last_on = int(audit.get("relative_step") or 0)
    if last_on is None:
        return {"restored": True, "reason": "never_engaged"}
    from config.constants import CAV_VTYPE

    by_step = {int(s.relative_step): s for s in trace.snapshots}
    maxima: Dict[int, Optional[float]] = {}
    for offset in (3, 6, 10):
        snapshot = by_step.get(last_on + offset)
        values = [
            float(row[9])
            for row in (snapshot.vehicles if snapshot is not None else [])
            if str(row[8]) == CAV_VTYPE
        ]
        maxima[last_on + offset] = max(values) if values else None
    return {
        "probe_steps": [last_on + 3, last_on + 6, last_on + 10],
        "max_cav_tau_s_by_step": maxima,
        "restored": all(
            value is not None and value <= C.WALL_PARAMS["wall_tau"] + 1.6
            for value in maxima.values()
        ),
    }


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    mode = str(case["mode"])
    mainline_vph = int(case["mainline_vph"])
    arms_spec: Dict[str, Callable[[], Any]] = {}
    if mode == "hrule":
        arms_spec["HRULE"] = lambda: HysteresisWallPolicy(
            arm_start_step=100,
            trigger_h=0.25,
            release_h=0.45,
        )
        decision = 0
    else:
        for step in case["decision_times"]:
            arms_spec[f"SE{int(step)}"] = (lambda s=int(step): SingleEngagementWallPolicy(
                decision_step=s,
                release_h=0.45,
                demand_end_s=900.0,
            ))
        decision = 0
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    sumo_binary = str(case["sumo_binary"])
    scenario_file = scenario_config(mainline_vph).name
    branch_started = time.time()
    control = replay_branch(
        seed=seed,
        arm="C",
        decision_step=None,
        scenario_file=scenario_file,
        scenario_root=C.SCENARIO_ROOT_PATH,
        sumo_binary=sumo_binary,
        total_steps=TOTAL_STEPS,
        policy_factory=NoOpNativeIDMPolicy,
        cav_penetration=C.CAV_PENETRATION,
    )
    arm_times = {"C": time.time() - branch_started}
    traces: Dict[str, ReplayTrace] = {"C": control}
    for arm, factory in arms_spec.items():
        branch_started = time.time()
        traces[arm] = replay_branch(
            seed=seed,
            arm=arm,
            decision_step=None,
            scenario_file=scenario_file,
            scenario_root=C.SCENARIO_ROOT_PATH,
            sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
            policy_factory=factory,
            cav_penetration=C.CAV_PENETRATION,
        )
        arm_times[arm] = time.time() - branch_started

    stored_root = Path(case["stored_c_root"])
    stored = json.loads(
        (stored_root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
    )
    if case["stored_c_key"] == "arms":
        stored_c_digest = str(stored["arms"]["C"]["trace_digest"])
    else:
        stored_c_digest = str(stored["branch"]["trace_digest"])

    prefix_before = 100 if mode == "hrule" else None
    integrity: Dict[str, Any] = {}
    for arm in arms_spec:
        comparison = compare_traces(control, traces[arm])
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        required_first = 101 if mode == "hrule" else int(arm[2:]) + 1
        integrity[arm] = {
            "prefix_identical_through_step": (
                int(prefix_before) if prefix_before is not None else int(arm[2:])
            ),
            "prefix_identical": bool(
                first_divergence is None or first_divergence > int(prefix_before or arm[2:])
            ),
            "first_divergence_step": first_divergence,
            "first_divergence_sim_time_s": None if mismatch is None else mismatch.get("left_sim_time"),
            "expected_first_divergence_step": required_first,
            "comparison_passed": bool(comparison["passed"]),
        }
    payload = {
        "schema": f"sustained_wall_v1_stage1b_prep3_{mode}_seed/1",
        "seed": seed,
        "mode": mode,
        "cell": {
            "mainline_demand_vph": mainline_vph,
            "ramp_demand_vph": C.RAMP_DEMAND_VPH,
            "cav_penetration": C.CAV_PENETRATION,
            "scenario": scenario_file,
        },
        "arms": {
            arm: _branch_summary(trace, arm=arm, decision_step=0)
            for arm, trace in traces.items()
        },
        "integrity": integrity,
        "wall_telemetry": {
            arm: _wall_telemetry(traces[arm]) for arm in arms_spec
        },
        "tau_integrity": {arm: _tau_integrity(traces[arm]) for arm in arms_spec},
        "replayed_c_digest": control.trace_digest,
        "stored_c_digest": stored_c_digest,
        "c_replay_matches_stored_cell": control.trace_digest == stored_c_digest,
        "branch_wall_clock_s": arm_times,
        "wall_clock_seconds": time.time() - started,
    }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "path": str(result_path)}


def _wall_telemetry(trace: ReplayTrace) -> Dict[str, Any]:
    wall_on_steps = []
    member_counts = []
    engagements = 0
    previous_on = False
    first_engage = None
    release_reasons = []
    for audit in trace.controller_audits:
        on = bool(audit.get("wall_active"))
        step = int(audit.get("relative_step") or 0)
        if on:
            wall_on_steps.append(step)
            member_counts.append(int(audit.get("member_count") or 0))
            if previous_on is False and first_engage is None:
                first_engage = step
        if on and not previous_on:
            engagements += 1
        if not on and previous_on:
            release_reasons.append(str(audit.get("release_reason") or "release"))
        previous_on = on
    return {
        "wall_on_steps": wall_on_steps,
        "wall_on_seconds": len(wall_on_steps),
        "engagements": engagements,
        "first_engage_step": first_engage,
        "member_count_mean": float(sum(member_counts) / len(member_counts)) if member_counts else 0.0,
        "max_member_count": max(member_counts) if member_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hrule", "se"), default="hrule")
    parser.add_argument("--mainline-vph", type=int, default=2200)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--decision-times", type=int, nargs="+", default=[150, 250])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mainline_vph = int(args.mainline_vph)
    if mainline_vph not in CELLS:
        raise ValueError(f"cells are {sorted(CELLS)}")
    cell = CELLS[mainline_vph]
    seeds = validate_seed_role(
        "sustained_wall_v1_stage1b_band_locator"
        if mainline_vph != 2200
        else "sustained_wall_v1_stage1a_development",
        args.seeds or cell["seeds"],
    )
    if not set(seeds) <= set(cell["seeds"]):
        raise ValueError(f"seeds {seeds} are outside the m{mainline_vph} cell block")
    output = (args.output or default_output(args.mode, mainline_vph, tuple(seeds))).resolve()
    allowed_roots = (
        output,
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030",
        REPO_ROOT
        / "runs"
        / "probe"
        / "sustained_wall_v1_stage1b_prep2_wallspeed_scan_m2200_p20_s6001_6030",
        cell["stored_c_root"],
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_hrule_m2100_p20_s6091_6120",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_se_m2100_p20_s6091_6120",
    )
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=allowed_roots
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": f"sustained_wall_v1_stage1b_prep3_{args.mode}_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": f"stage1b_prep3_{args.mode}",
            "cell": {
                "mainline_demand_vph": mainline_vph,
                "ramp_demand_vph": C.RAMP_DEMAND_VPH,
                "cav_penetration": C.CAV_PENETRATION,
                "scenario": scenario_config(mainline_vph).name,
            },
            "seeds": seeds,
            "seed_role": (
                "sustained_wall_v1_stage1b_band_locator"
                if mainline_vph != 2200
                else "sustained_wall_v1_stage1a_development"
            ),
            "prospective_evaluation_exclusion": True,
            "rule": (
                {
                    "arm": "h < 0.25 inst; release h > 0.45 inst; re-engage allowed; "
                    "gate at step 100; no taper on release",
                    "arm_start_s": 100,
                    "trigger_h": 0.25,
                    "release_h": 0.45,
                    "actuator": "wall 200-400 m, 10 m/s, tau 3.0, 0.3 m/s^2 clip, mainline",
                }
                if args.mode == "hrule"
                else {
                    "single_engagement": True,
                    "decision_times_s": args.decision_times,
                    "release": "h > 0.45 instant or demand end (t = 900 s); no re-engagement",
                    "release_h": 0.45,
                }
            ),
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
            "mode": args.mode,
            "mainline_vph": mainline_vph,
            "decision_times": args.decision_times,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "stored_c_root": str(cell["stored_c_root"]),
            "stored_c_key": cell["stored_c_key"],
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
