"""Stage 1c-2 treatment runner: fixed-hold, onset-triggered, and R1b arms.

Cells 2400/2600/2700 (seeds 7431-7520) on the Stage 1c plateau-decline
profile; the 2200 cell runs WON_H300 only (false-engagement cost below the
boundary), seeds 7401-7430. Per seed: replay D fresh with the 1c machinery
(digest cross-checked against the stored Stage 1c D digest), compute the
treatment-blind onset (first step in the D prefix with h < 0.15 for 10
consecutive seconds, search capped at t = 900), then run the arms:

- W250_H300 / W250_H450: fixed hold at t_d = 250 (300/450 s, 5 s taper);
- WON_H300 / WON_H450: onset-triggered (skipped, and counted, when the
  seed has no onset by t = 900);
- W250_R1b: corrected state release (speed gate over 400-800 m only,
  hard max 900 s of hold relative to engagement).

Every arm must have exact prefix identity with D through its engagement
step (t_d = 250 fixed / the onset step for WON). D itself is not repeated
as an arm: the stored Stage 1c D summary is the pairing reference; the
fresh replay exists for integrity and onset detection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from counterfactual.twin_replay import compare_traces
from experiments import REPO_ROOT
from experiments.idm import deterministic_sumo  # noqa: F401  (race-free launch)
from experiments.idm.run_stage1c_release import (
    DEFAULT_SUMO_EXE,
    _replay_d,
    _replay_w,
)
from experiments.idm.rsu_feature_capture import (
    first_onset_step,
    safety_events_by_location,
)
from experiments.idm.stage1c_release_contract import (
    CELLS,
    DECISION_STEP,
    contract_manifest as stage1c_contract_manifest,
    write_seed_scenario,
)
from experiments.idm.run_sustained_wall_v1_stage1a import _branch_summary
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import contract_manifest as idm_contract_manifest
from policies.stage1c2_arms import (
    FixedHoldDampingWallPolicy,
    OnsetTriggeredDampingWallPolicy,
    R1bReleaseDampingWallPolicy,
)

STAGE1C_DIRS = {
    2200: REPO_ROOT / "runs" / "probe" / "stage1c_release_m2200_p20_s7401_7430",
    2400: REPO_ROOT / "runs" / "probe" / "stage1c_release_m2400_p20_s7431_7460",
    2600: REPO_ROOT / "runs" / "probe" / "stage1c_release_m2600_p20_s7461_7490",
    2700: REPO_ROOT / "runs" / "probe" / "stage1c_release_m2700_p20_s7491_7520",
}
ONSET_SEARCH_MAX_STEP = 900
WON_ONSET_SUSTAIN_S = 10.0


def _release_step_from_audits(trace) -> Any:
    for audit in trace.controller_audits:
        if str(audit.get("treatment_state")) == "TAPERING":
            return int(audit.get("relative_step"))
    return None


def _wall_dose_from_audits(trace) -> Dict[str, Any]:
    active_steps = 0
    member_total = 0
    for audit in trace.controller_audits:
        if bool(audit.get("wall_active")):
            active_steps += 1
            member_total += int(audit.get("member_count") or 0)
    return {
        "wall_active_steps": active_steps,
        "mean_member_count": (member_total / active_steps) if active_steps else 0.0,
    }


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    mainline_vph = int(case["mainline_vph"])
    arms_spec = list(case["arms"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()

    stored_path = STAGE1C_DIRS[mainline_vph] / "seeds" / f"seed{seed}.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored_d_digest = str(stored["arms"]["D"]["trace_digest"])

    scenario_name = write_seed_scenario(seed, mainline_vph)
    sumo_binary = str(case["sumo_binary"])
    d_trace, _r2_threshold = _replay_d(seed, scenario_name, sumo_binary)
    digest_match = bool(d_trace.trace_digest == stored_d_digest)
    h_series = [float(s.supervisor_observation[0]) for s in d_trace.snapshots]
    onset = first_onset_step(
        h_series, sustain_s=WON_ONSET_SUSTAIN_S, max_step=ONSET_SEARCH_MAX_STEP
    )

    arm_tds: Dict[str, int] = {"W250_H300": DECISION_STEP, "W250_H450": DECISION_STEP,
                               "W250_R1b": DECISION_STEP}
    factories: Dict[str, Callable[[], Any]] = {
        "W250_H300": lambda: FixedHoldDampingWallPolicy(decision_step=DECISION_STEP, hold_s=300.0),
        "W250_H450": lambda: FixedHoldDampingWallPolicy(decision_step=DECISION_STEP, hold_s=450.0),
        "W250_R1b": lambda: R1bReleaseDampingWallPolicy(decision_step=DECISION_STEP),
    }
    if onset is not None:
        arm_tds["WON_H300"] = int(onset)
        arm_tds["WON_H450"] = int(onset)
        factories["WON_H300"] = (
            lambda o=int(onset): OnsetTriggeredDampingWallPolicy(onset_step=o, hold_s=300.0)
        )
        factories["WON_H450"] = (
            lambda o=int(onset): OnsetTriggeredDampingWallPolicy(onset_step=o, hold_s=450.0)
        )
    missing = [arm for arm in arms_spec if arm not in factories]
    if missing:
        skipped_no_onset = missing
    else:
        skipped_no_onset = []

    traces: Dict[str, Any] = {}
    integrity: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    for arm in arms_spec:
        if arm in skipped_no_onset:
            continue
        td = arm_tds[arm]
        trace = _replay_w(seed, scenario_name, sumo_binary, factories[arm])
        traces[arm] = trace
        comparison = compare_traces(d_trace, trace)
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        integrity[arm] = {
            "prefix_identical_through_t_d": bool(
                first_divergence is None or first_divergence > td
            ),
            "first_divergence_step": first_divergence,
            "expected_first_divergence_step": td + 1,
            "t_d": td,
        }
        release_step = _release_step_from_audits(trace)
        engage_step = td + 1
        summary = _branch_summary(trace, arm=arm, decision_step=0)
        summary["engage_step"] = engage_step
        summary["release_step"] = release_step
        summary["hold_duration_s"] = (
            (release_step - engage_step) if release_step is not None else None
        )
        summary["wall_dose"] = _wall_dose_from_audits(trace)
        summary["safety_by_location"] = safety_events_by_location(
            trace.snapshots, engage_step=engage_step, release_step=release_step
        )
        summaries[arm] = summary

    payload: Dict[str, Any] = {
        "schema": "stage1c2_arms_seed/1",
        "seed": seed,
        "mainline_demand_vph": mainline_vph,
        "scenario": scenario_name,
        "stored_d_digest_match": digest_match,
        "onset_step": int(onset) if onset is not None else None,
        "no_onset_by_900": onset is None,
        "skipped_no_onset_arms": skipped_no_onset,
        "arms": summaries,
        "integrity": integrity,
        "rule_inputs": {
            "W250_R1b": traces["W250_R1b"].controller_metadata.get("_w_rule_inputs", {})
            if "W250_R1b" in traces else {}
        },
        "repeat_check": None,
        "wall_clock_seconds": time.time() - started,
    }
    if (seed - int(case["repeat_offset"])) % 10 == 0:
        repeated, _t = _replay_d(seed, scenario_name, sumo_binary)
        payload["repeat_check"] = {
            "second_digest": repeated.trace_digest,
            "exact_identity": repeated.trace_digest == d_trace.trace_digest,
        }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "mainline_vph": mainline_vph,
            "onset": payload["onset_step"], "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, required=True, choices=(2200, 2400, 2600, 2700))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sumo-binary", default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cell = int(args.cell)
    seeds = tuple(args.seeds) if args.seeds else CELLS[cell]
    if cell == 2200:
        arms = ["WON_H300"]
        dir_name = f"stage1c2_won_m{cell}_p20_s{seeds[0]}_{seeds[-1]}"
    else:
        arms = ["W250_H300", "W250_H450", "WON_H300", "WON_H450", "W250_R1b"]
        dir_name = f"stage1c2_arms_m{cell}_p20_s{seeds[0]}_{seeds[-1]}"
    output = (args.output or REPO_ROOT / "runs" / "probe" / dir_name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "stage1c2_arms_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1c2_treatment_arms",
            "cell_mainline_vph": cell,
            "seeds": seeds,
            "seed_role": "stage1c_release (dev; prospective-excluded)",
            "arms": arms,
            "onset_rule": f"h < 0.15 for {WON_ONSET_SUSTAIN_S:.0f} consecutive s in the D "
            "prefix; no treatment branch when no onset by t=900",
            "stage1c_contract": stage1c_contract_manifest(),
            "idm_contract": idm_contract_manifest(),
            "reference_arms_not_rerun": {"W_HOLD": "1c", "W_ORACLE": "1c", "D": "1c"},
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
            "mainline_vph": cell,
            "arms": arms,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
            "repeat_offset": seeds[0],
        }
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps({"status": "all_workers_done", "cell": cell, "seeds": len(seeds)},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
