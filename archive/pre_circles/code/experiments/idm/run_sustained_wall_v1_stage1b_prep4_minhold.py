"""Stage 1b-prep-4 Part B runner: SE250 minimum-hold arms at 2100/p20.

SE250_H120 / SE250_H300: single engagement at rule-blind t_d = 250 (first
command at step 251), hold >= H_min, then release when h > 0.45 has
persisted for 30 consecutive seconds, or at the demand end (t = 900 s);
5 s taper; no re-engagement. Arm C is replayed per seed (digest compared
with the stored band-locator C); prefix identity through t_d = 250 is
checked against the replayed C.
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
    _wall_dose,
)
from experiments.idm.stage1b_runner_core import replay_branch
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
    contract_manifest,
    scenario_config,
    validate_seed_role,
)
from policies.noop_base import NoOpNativeIDMPolicy
from policies.sustained_wall_minhold import MinimumHoldWallPolicy

DECISION_STEP = 250
MIN_HOLDS = {"SE250_H120": 120.0, "SE250_H300": 300.0}


def default_output() -> Path:
    return (
        REPO_ROOT
        / "runs"
        / "probe"
        / "sustained_wall_v1_stage1b_prep4_minhold_m2100_p20_s6091_6120"
    )


def _factory(arm: str) -> Callable[[], Any]:
    def factory() -> MinimumHoldWallPolicy:
        return MinimumHoldWallPolicy(
            decision_step=DECISION_STEP,
            min_hold_s=MIN_HOLDS[arm],
            recovery_h=0.45,
            recovery_persist_s=30.0,
            wall_cap_s=900.0,
            taper_duration_s=5.0,
        )

    return factory


def _tau_integrity(trace: ReplayTrace) -> Dict[str, Any]:
    from config.constants import CAV_VTYPE

    last_on = None
    for audit in trace.controller_audits:
        if bool(audit.get("wall_active")):
            last_on = int(audit.get("relative_step") or 0)
    if last_on is None:
        return {"restored": True, "reason": "never_engaged"}
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
    mainline_vph = int(case["mainline_vph"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    sumo_binary = str(case["sumo_binary"])
    scenario_file = scenario_config(mainline_vph).name
    arm_times: Dict[str, float] = {}
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
    arm_times["C"] = time.time() - branch_started
    traces: Dict[str, ReplayTrace] = {"C": control}
    for arm in MIN_HOLDS:
        branch_started = time.time()
        traces[arm] = replay_branch(
            seed=seed,
            arm=arm,
            decision_step=None,
            scenario_file=scenario_file,
            scenario_root=C.SCENARIO_ROOT_PATH,
            sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
            policy_factory=_factory(arm),
            cav_penetration=C.CAV_PENETRATION,
        )
        arm_times[arm] = time.time() - branch_started

    stored = json.loads(
        (
            REPO_ROOT
            / "runs"
            / "probe"
            / "sustained_wall_v1_stage1b_band_locator_m2100_s6091_6120"
            / "seeds"
            / f"seed{seed}.json"
        ).read_text(encoding="utf-8")
    )
    stored_c_digest = str(stored["branch"]["trace_digest"])
    integrity: Dict[str, Any] = {}
    for arm in MIN_HOLDS:
        comparison = compare_traces(control, traces[arm])
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        integrity[arm] = {
            "prefix_identical_through_step": DECISION_STEP,
            "prefix_identical": bool(
                first_divergence is None or first_divergence > DECISION_STEP
            ),
            "first_divergence_step": first_divergence,
            "expected_first_divergence_step": DECISION_STEP + 1,
            "comparison_passed": bool(comparison["passed"]),
        }
    payload = {
        "schema": "sustained_wall_v1_stage1b_prep4_minhold_seed/1",
        "seed": seed,
        "decision_step": DECISION_STEP,
        "rule_blind_t_d": True,
        "cell": {
            "mainline_demand_vph": 2100,
            "ramp_demand_vph": C.RAMP_DEMAND_VPH,
            "cav_penetration": C.CAV_PENETRATION,
        },
        "arms": {
            arm: _branch_summary(trace, arm=arm, decision_step=0)
            for arm, trace in traces.items()
        },
        "integrity": integrity,
        "tau_integrity": {arm: _tau_integrity(traces[arm]) for arm in MIN_HOLDS},
        "replayed_c_digest": control.trace_digest,
        "stored_c_digest": stored_c_digest,
        "c_replay_matches_stored_cell": control.trace_digest == stored_c_digest,
        "wall_dose": {arm: _wall_dose(traces[arm]) for arm in MIN_HOLDS},
        "branch_wall_clock_s": arm_times,
        "wall_clock_seconds": time.time() - started,
    }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = validate_seed_role("sustained_wall_v1_stage1b_band_locator", args.seeds)
    if not set(seeds) <= set(SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS):
        raise ValueError(f"seeds {seeds} are outside the 2100 band-locator block")
    output = args.output.resolve()
    allowed_roots = (
        output,
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_band_locator_m2100_s6091_6120",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_hrule_m2100_p20_s6091_6120",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_se_m2100_p20_s6091_6120",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_legacy_probe",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_obsmap_m2000_p20_s6061_6090",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_obsmap_m2100_p20_s6091_6120",
        REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_obsmap_m2200_p20_s6001_6030",
    )
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=allowed_roots
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "sustained_wall_v1_stage1b_prep4_minhold_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1b_prep4_minimum_hold_pilot",
            "cell": {
                "mainline_demand_vph": 2100,
                "ramp_demand_vph": C.RAMP_DEMAND_VPH,
                "cav_penetration": C.CAV_PENETRATION,
                "scenario": scenario_config(2100).name,
            },
            "seeds": seeds,
            "seed_role": "sustained_wall_v1_stage1b_band_locator",
            "prospective_evaluation_exclusion": True,
            "rule": {
                "single_engagement": True,
                "decision_step": DECISION_STEP,
                "min_hold_s": MIN_HOLDS,
                "release": "h > 0.45 persisted 30 s after min hold, or demand end; 5 s taper; "
                "no re-engagement",
            },
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
            "mainline_vph": 2100,
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
