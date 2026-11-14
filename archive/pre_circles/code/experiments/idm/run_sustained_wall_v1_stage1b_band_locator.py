"""Stage 1b-prep band locator: native-IDM Arm C only, lower-demand cells.

Runs Arm C (native IDM via ``policies/noop_base.py``) on the 1900, 2000 and
2100 veh/h mainline cells (ramp 400, CAV penetration 0.20, long IDM scenario,
seeds 6031-6060 / 6061-6090 / 6091-6120) with the identical replay machinery
as Stage 1a (imported from ``run_sustained_wall_v1_stage1a``, which is
unchanged). Horizon 1080 s or verified natural clearance. The 2200/p20 cell
is reused from Stage 1a; it is NOT rerun here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping

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
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_1900_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
    contract_manifest,
    scenario_config,
    validate_seed_role,
)
from policies.noop_base import NoOpNativeIDMPolicy

CELL_SEEDS = {
    1900: SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_1900_SEEDS,
    2000: SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS,
    2100: SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
}


def default_output(mainline_vph: int) -> Path:
    seeds = CELL_SEEDS[int(mainline_vph)]
    return (
        REPO_ROOT
        / "runs"
        / "probe"
        / f"sustained_wall_v1_stage1b_band_locator_m{int(mainline_vph)}"
        f"_s{seeds[0]}_{seeds[-1]}"
    )


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    mainline_vph = int(case["mainline_vph"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    trace = replay_branch(
        seed=seed,
        arm="C",
        decision_step=None,
        scenario_file=scenario_config(mainline_vph).name,
        scenario_root=C.SCENARIO_ROOT_PATH,
        sumo_binary=str(case["sumo_binary"]),
        total_steps=TOTAL_STEPS,
        policy_factory=NoOpNativeIDMPolicy,
        cav_penetration=C.CAV_PENETRATION,
    )
    summary = _branch_summary(trace, arm="C", decision_step=0)
    h_series = summary["series"]["h"]
    h_10s = h_series[::10]
    payload = {
        "schema": "sustained_wall_v1_stage1b_band_locator_seed/1",
        "seed": seed,
        "cell": {
            "mainline_demand_vph": mainline_vph,
            "ramp_demand_vph": C.RAMP_DEMAND_VPH,
            "cav_penetration": C.CAV_PENETRATION,
            "scenario": scenario_config(mainline_vph).name,
        },
        "arm": "C",
        "branch": summary,
        "h_10s": {
            "step": list(summary["series"]["step"][::10]),
            "h": [round(float(v), 6) for v in h_10s],
        },
        "health_collapse": summary["health_collapse"],
        "physical_collapse": summary["physical_collapse"],
        "delay_s_per_veh": summary["delay"]["mean_per_vehicle_delay_s"],
        "incremental_delay_s": summary["delay"]["incremental_delay_s"],
        "hard_brake_steps": summary["outcomes"]["hard_brake_vehicle_steps"],
        "low_ttc_exposure_steps": summary["outcomes"]["low_ttc_exposure_steps"],
        "terminal": summary["termination"],
        "events": summary["events"],
        "wall_clock_seconds": time.time() - started,
    }
    _atomic_json(result_path, payload)
    return {
        "status": "completed",
        "seed": seed,
        "mainline_vph": mainline_vph,
        "path": str(result_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-vph", type=int, default=1900)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mainline_vph = int(args.mainline_vph)
    if mainline_vph not in CELL_SEEDS:
        raise ValueError(f"band locator cells are {sorted(CELL_SEEDS)}")
    cell_seeds = CELL_SEEDS[mainline_vph]
    seeds = validate_seed_role("sustained_wall_v1_stage1b_band_locator", args.seeds or cell_seeds)
    if not set(seeds) <= set(cell_seeds):
        raise ValueError(f"seeds {seeds} are not inside the m{mainline_vph} cell block")
    output = (args.output or default_output(mainline_vph)).resolve()
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=(output,)
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "sustained_wall_v1_stage1b_band_locator_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1b_band_locator_native_idm_arm_c",
            "cell": {
                "mainline_demand_vph": mainline_vph,
                "ramp_demand_vph": C.RAMP_DEMAND_VPH,
                "cav_penetration": C.CAV_PENETRATION,
                "scenario": scenario_config(mainline_vph).name,
            },
            "seeds": seeds,
            "seed_role": "sustained_wall_v1_stage1b_band_locator",
            "prospective_evaluation_exclusion": True,
            "contract": C.contract_manifest(),
            "idm_contract": contract_manifest(),
            "horizon_steps": TOTAL_STEPS,
            "arm": "C_native_idm_no_commands",
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
            "mainline_vph": mainline_vph,
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
