"""Treatment-blind early-formation feasibility scan for IDM wall v5.6.1."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from config.constants import PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from counterfactual.twin_replay import run_replay
from experiments import REPO_ROOT
from experiments.idm.idm_v5_5_t880_contract import (
    ANALYSIS_END_S, CAV_PENETRATION, DEMAND_MAINLINE_VPH, DEMAND_RAMP_VPH,
)
from experiments.idm.run_idm_v5_5_t880_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from idm.contract import transition_scenario_config
from policies.counterfactual_escalation import DAMPING_ONLY
from policies.idm_analytical import build_frozen_idm_analytical_policy, frozen_idm_analytical_manifest
from experiments.io import atomic_write_json


SEEDS = (4401, 4402, 4404, 4405)
ASSEMBLY_STARTS_S = (860, 865, 870, 875)
HOLD_START_S = 880
FINAL_BAND_M = (200.0, 1000.0)
FINAL_SPAN_M = 50.0
MAX_SPEED_SPREAD_MPS = 4.0
DECEL_CAP_MPS2 = 0.3
POSITION_KP_S_INV = 0.05
POSITION_KD = 0.5
POSITION_OFFSET_CAP_MPS = 1.0
# A five-second, 4 m/s relative-speed envelope beyond the 50 m final compact
# wall is a geometric bound, independent of observed treatment outcomes.
ASSEMBLY_SPAN_M = FINAL_SPAN_M + MAX_SPEED_SPREAD_MPS * 5.0
MIN_FRONT_REAR_GAP_M = 20.0
MIN_FRONT_TTC_S = 3.0
MIN_REAR_TTC_S = 4.0
SCHEMA = "idm_v5_6_1_early_formation_scan/1"
OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_v5_6_1_early_formation_scan_p30_s4401_4402_4404_4405"


def _zero_schedule(_step: int) -> int:
    return int(DAMPING_ONLY)


def _distance(row: tuple[Any, ...]) -> float | None:
    if str(row[1]) != PRE_MERGE_EDGE:
        return None
    return max(0.0, PRE_MERGE_LANE_LENGTH - float(row[3]))


def _rows(snapshot: Any) -> dict[str, dict[str, Any]]:
    active = {str(value) for value in snapshot.active_cavs}
    values: dict[str, dict[str, Any]] = {}
    for row in snapshot.vehicles:
        dist = _distance(row)
        if dist is None:
            continue
        values[str(row[0])] = {
            "id": str(row[0]), "lane": str(row[2]), "distance_m": dist,
            "speed_mps": float(row[4]), "accel_mps2": float(row[5]),
            "active_cav": str(row[0]) in active,
        }
    return values


def _ttc(gap: float, closing: float) -> float | None:
    return None if closing <= 1.0e-9 else gap / closing


def _member_safety(member: Mapping[str, Any], lane_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    distance = float(member["distance_m"]); speed = float(member["speed_mps"])
    ahead = [row for row in lane_rows if float(row["distance_m"]) < distance and row["id"] != member["id"]]
    behind = [row for row in lane_rows if float(row["distance_m"]) > distance and row["id"] != member["id"]]
    front = max(ahead, key=lambda row: float(row["distance_m"])) if ahead else None
    rear = min(behind, key=lambda row: float(row["distance_m"])) if behind else None
    front_gap = None if front is None else distance - float(front["distance_m"])
    rear_gap = None if rear is None else float(rear["distance_m"]) - distance
    front_ttc = None if front is None else _ttc(front_gap, speed - float(front["speed_mps"]))
    rear_ttc = None if rear is None else _ttc(rear_gap, float(rear["speed_mps"]) - speed)
    safe = (
        (front_gap is None or front_gap >= MIN_FRONT_REAR_GAP_M)
        and (rear_gap is None or rear_gap >= MIN_FRONT_REAR_GAP_M)
        and (front_ttc is None or front_ttc >= MIN_FRONT_TTC_S)
        and (rear_ttc is None or rear_ttc >= MIN_REAR_TTC_S)
    )
    return {"safe": safe, "front_gap_m": front_gap, "rear_gap_m": rear_gap, "front_ttc_s": front_ttc, "rear_ttc_s": rear_ttc}


def _project(members: tuple[Mapping[str, Any], ...], seconds: int) -> dict[str, Any]:
    distances = {str(row["id"]): float(row["distance_m"]) for row in members}
    speeds = {str(row["id"]): float(row["speed_mps"]) for row in members}
    common_target = min(speeds.values())
    spans = [max(distances.values()) - min(distances.values())]
    speed_spreads = [max(speeds.values()) - min(speeds.values())]
    ready_second = 0 if spans[-1] <= FINAL_SPAN_M and speed_spreads[-1] <= MAX_SPEED_SPREAD_MPS else None
    for second in range(1, seconds + 1):
        center = sum(distances.values()) / len(distances); mean_speed = sum(speeds.values()) / len(speeds)
        targets: dict[str, float] = {}
        for key in distances:
            offset = max(-POSITION_OFFSET_CAP_MPS, min(POSITION_OFFSET_CAP_MPS, POSITION_KP_S_INV * (distances[key] - center) - POSITION_KD * (speeds[key] - mean_speed)))
            targets[key] = max(0.0, common_target + offset)
        for key in distances:
            old = speeds[key]; target = targets[key]
            new = max(target, old - DECEL_CAP_MPS2) if old > target else min(target, old + 1.0)
            distances[key] -= 0.5 * (old + new); speeds[key] = new
        span = max(distances.values()) - min(distances.values()); spread = max(speeds.values()) - min(speeds.values())
        spans.append(span); speed_spreads.append(spread)
        if ready_second is None and span <= FINAL_SPAN_M and spread <= MAX_SPEED_SPREAD_MPS:
            ready_second = second
    return {"initial_span_m": spans[0], "maximum_assembly_span_m": max(spans), "final_span_at_hold_m": spans[-1], "maximum_speed_spread_mps": max(speed_spreads), "final_speed_spread_mps": speed_spreads[-1], "time_to_compact_ready_s": ready_second, "projected_member_distances_at_hold_m": distances, "spans_monotone_nonincreasing": all(left >= right for left, right in zip(spans, spans[1:]))}


def assess_snapshot(start_snapshot: Any, hold_snapshot: Any, *, assembly_start_s: int) -> dict[str, Any]:
    start_rows = _rows(start_snapshot); hold_rows = _rows(hold_snapshot)
    occupied = tuple(sorted({row["lane"] for row in start_rows.values()}))
    cavs = [row for row in start_rows.values() if row["active_cav"] and FINAL_BAND_M[0] <= row["distance_m"] <= PRE_MERGE_LANE_LENGTH]
    by_lane = {lane: [row for row in cavs if row["lane"] == lane] for lane in occupied}
    if not occupied or any(not by_lane.get(lane) for lane in occupied):
        return {"assembly_start_s": assembly_start_s, "feasible": False, "reason": "uncovered_occupied_lane", "occupied_lanes": occupied}
    choices = []; rejected: dict[str, int] = {}
    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1
    for members in itertools.product(*(by_lane[lane] for lane in occupied)):
        safety = [_member_safety(row, start_rows.values()) for row in members]
        if not all(row["safe"] for row in safety):
            reject("assembly_activation_safety")
            continue
        if max(row["speed_mps"] for row in members) - min(row["speed_mps"] for row in members) > MAX_SPEED_SPREAD_MPS:
            reject("assembly_speed_spread")
            continue
        projection = _project(tuple(members), HOLD_START_S - assembly_start_s)
        if projection["maximum_assembly_span_m"] > ASSEMBLY_SPAN_M or not projection["spans_monotone_nonincreasing"]:
            reject("assembly_envelope_or_convergence")
            continue
        if projection["final_span_at_hold_m"] > FINAL_SPAN_M or projection["time_to_compact_ready_s"] is None:
            reject("final_compactness")
            continue
        ids = tuple(sorted(row["id"] for row in members))
        hold_members = [hold_rows.get(key) for key in ids]
        if any(row is None or not row["active_cav"] or not (FINAL_BAND_M[0] <= row["distance_m"] <= FINAL_BAND_M[1]) for row in hold_members):
            reject("identity_continuity_or_final_band")
            continue
        hold_safety = [_member_safety(row, hold_rows.values()) for row in hold_members if row is not None]
        if not all(row["safe"] for row in hold_safety):
            reject("hold_activation_safety")
            continue
        choices.append((projection["final_span_at_hold_m"], projection["maximum_assembly_span_m"], ids, members, safety, projection, hold_safety))
    if not choices:
        return {"assembly_start_s": assembly_start_s, "feasible": False, "reason": "no_safe_continuous_compact_cohort", "occupied_lanes": occupied, "candidate_rejection_counts": rejected, "candidate_combination_count": len(list(itertools.product(*(by_lane[lane] for lane in occupied))))}
    _, _, ids, members, safety, projection, hold_safety = min(choices, key=lambda row: row[:3])
    return {"assembly_start_s": assembly_start_s, "hold_start_s": HOLD_START_S, "feasible": True, "reason": "feasible", "occupied_lanes": occupied, "member_ids": ids, "members_at_assembly": list(members), "member_safety_at_assembly": safety, "member_safety_at_hold": hold_safety, "candidate_identity_continuity": True, "expected_handoffs": 0, "assembly_command_duration_s": HOLD_START_S - assembly_start_s, "projection": projection, "temporary_assembly_span_cap_m": ASSEMBLY_SPAN_M, "final_holding_span_cap_m": FINAL_SPAN_M}


def _relative(absolute_s: int, reset_time: float) -> int:
    value = int(round(absolute_s - reset_time))
    if value < 0 or abs(reset_time + value - absolute_s) > 1.0e-6: raise ValueError("nonintegral SUMO clock mapping")
    return value


def run_seed(case: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"]); output = Path(case["output"]); path = output / "seeds" / f"seed{seed}.json"
    if path.exists() and not case.get("force"): return {"status": "skipped", "seed": seed, "path": str(path)}
    # Existing Gate-F records provide only reset-clock provenance, never outcomes.
    gate = json.loads((REPO_ROOT / "runs" / "probe" / "idm_v5_5_1_four_case_repair_p30_s4401_4402_4404_4405" / "cases" / f"seed{seed}.json").read_text(encoding="utf-8"))["gate_f"]
    reset_time = float(gate["reset_info"]["episode_time"])
    earliest = _relative(min(ASSEMBLY_STARTS_S), reset_time); hold = _relative(HOLD_START_S, reset_time)
    trace = run_replay(project_root=REPO_ROOT / "scenarios" / "idm_transition", sumo_binary=str(case["sumo_binary"]), sumo_config=transition_scenario_config(DEMAND_MAINLINE_VPH).name, seed=seed, decision_step=earliest, horizon_steps=hold-earliest, treatment_steps=1, cav_penetration=CAV_PENETRATION, sumo_step_length=0.5, supervisor_overrides={**supervisor_overrides(), "use_health_obs": False, "use_dv_obs": False, "feasibility_mask": False, "action_masking": False}, digits=6, base_policy_factory=build_frozen_idm_analytical_policy, intervention_schedule=_zero_schedule, overlay_overrides=None, controller_metadata={"scientific_role": "v5_6_1_treatment_blind_early_formation_scan", "treatment_outcomes_read": False})
    snapshots = {int(item.relative_step): item for item in trace.snapshots}
    rows = []
    for start in ASSEMBLY_STARTS_S:
        result = assess_snapshot(snapshots[_relative(start, reset_time)], snapshots[hold], assembly_start_s=start)
        result["relative_decision_step"] = _relative(start, reset_time); result["relative_hold_step"] = hold; rows.append(result)
    payload = {"schema": SCHEMA, "seed": seed, "reset_time_s": reset_time, "grid_absolute_s": list(ASSEMBLY_STARTS_S), "hold_start_s": HOLD_START_S, "treatment_outcomes_read": False, "base_control_arm": frozen_idm_analytical_manifest(), "rows": rows, "trace_digest": trace.trace_digest}
    atomic_write_json(path, payload); return {"status": "completed", "seed": seed, "path": str(path)}


def build_summary(output: Path) -> dict[str, Any]:
    records = [json.loads((output / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS]
    grid = {start: all(next(row for row in record["rows"] if row["assembly_start_s"] == start)["feasible"] for record in records) for start in ASSEMBLY_STARTS_S}
    selected = max((start for start, passed in grid.items() if passed), default=None)
    return {"schema": "idm_v5_6_1_early_formation_scan_summary/1", "grid_absolute_s": list(ASSEMBLY_STARTS_S), "selection_rule": "latest_common_feasible_start", "common_feasible_by_start": grid, "selected_assembly_start_s": selected, "selection_passed": selected is not None, "treatment_outcomes_read": False, "seeds": list(SEEDS), "assembly_span_cap_m": ASSEMBLY_SPAN_M, "final_holding_span_cap_m": FINAL_SPAN_M, "base_control_arm": frozen_idm_analytical_manifest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE)); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    output = args.output.resolve(); (output / "seeds").mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "grid_absolute_s": list(ASSEMBLY_STARTS_S), "hold_start_s": HOLD_START_S, "seeds": list(SEEDS), "temporary_assembly_span_cap_m": ASSEMBLY_SPAN_M, "final_holding_span_cap_m": FINAL_SPAN_M, "treatment_outcomes_read": False, "base_control_arm": frozen_idm_analytical_manifest()}; manifest["fingerprint_sha256"] = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); atomic_write_json(output / "manifest.json", manifest)
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), len(SEEDS))) as pool:
        for future in as_completed([pool.submit(run_seed, {"seed": seed, "output": str(output), "sumo_binary": args.sumo_binary, "force": args.force}) for seed in SEEDS]): print(json.dumps(future.result(), sort_keys=True), flush=True)
    atomic_write_json(output / "summary.json", build_summary(output)); return 0


if __name__ == "__main__": raise SystemExit(main())
