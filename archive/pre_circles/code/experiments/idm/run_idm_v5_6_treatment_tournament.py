"""Run each independently eligible IDM v5.6 moving-wall tournament candidate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import os
from typing import Any, Mapping

from counterfactual.idm_gate_labels import IDMGateBMargins, classify_idm_gate_b_case
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import _audit_contract, _prefix_identity
from experiments.idm.idm_v5_5_t880_contract import (
    ACTIVATION_TIME_S, ANALYSIS_END_S, CAV_PENETRATION, DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH, MINIMUM_SHORTENING_S, PRIMARY_QUEUE_COUNT,
    PRIMARY_SPEED_THRESHOLD_MPS, PRIMARY_SUSTAIN_S,
)
from experiments.idm.run_idm_v5_5_1_four_case_repair import (
    CAUSAL_CHECK_NAMES, DEFAULT_GATE_F, DEFAULT_SUMO_EXE, SEEDS, TREATMENT_DURATION_S,
    _adaptive_dose, _compression_summary, _integrity, _physical_effect, _relative_decision,
    _replay,
)
from experiments.idm.run_idm_v5_5_t880_wall_pilot import _arm_record, _gate_f_row
if os.environ.get("IDM_V5_6_PROFILE") == "phase_split_authority":
    from experiments.idm.wall_v5_6_2_phase_split_contract import overlay_overrides, profile_manifest
    CANDIDATE_ID = "v5_6_2_phase_split_authority"
    SCIENTIFIC_ROLE = "v5_6_2_phase_split_authority_four_case_gate"
else:
    from experiments.idm.wall_v5_6_a_low_authority_contract import overlay_overrides, profile_manifest
    CANDIDATE_ID = "A_low_authority"
    SCIENTIFIC_ROLE = "v5_6_treatment_tournament_candidate_a"
from experiments.io import atomic_write_json


OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_v5_6_tournament_a_low_authority_p30_s4401_4402_4404_4405"
V54_REFERENCE = REPO_ROOT / "runs" / "probe" / "idm_v5_5_t880_wall_pilot_p30_s4401_4406"
V551_REFERENCE = REPO_ROOT / "runs" / "probe" / "idm_v5_5_1_four_case_repair_p30_s4401_4402_4404_4405"


def _reference(root: Path, seed: int) -> dict[str, Any]:
    return json.loads((root / "cases" / f"seed{seed}.json").read_text(encoding="utf-8")) if root == V551_REFERENCE else json.loads((root / "cases" / f"seed{seed}.primary.json").read_text(encoding="utf-8"))


def _phase_dose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize explicit v5.6.2 phases without reclassifying treatment dose."""
    phases: dict[str, dict[str, Any]] = {}
    for row in rows:
        phase = str(row.get("treatment_state", "DAMPING"))
        deltas = row.get("control_deltas") or {}
        if not deltas:
            continue
        record = phases.setdefault(
            phase,
            {"command_steps": 0, "controlled_cav_seconds": 0.0, "max_authority_mps2": 0.0,
             "min_lane_coverage": 1.0, "max_span_m": 0.0, "max_settling_jerk_mps3": 0.0,
             "all_settling_tau_frozen": True, "member_ids": set()},
        )
        record["command_steps"] += 1
        record["controlled_cav_seconds"] += float(len(deltas))
        record["max_authority_mps2"] = max(
            float(record["max_authority_mps2"]),
            max((float(value.get("wall_a_max_dec_mps2") or 0.0) for value in deltas.values()), default=0.0),
        )
        coverage = (row.get("lane_coverage") or {}).get("coverage_fraction")
        if coverage is not None:
            record["min_lane_coverage"] = min(float(record["min_lane_coverage"]), float(coverage))
        span = row.get("current_member_span_m")
        if span is not None:
            record["max_span_m"] = max(float(record["max_span_m"]), float(span))
        record["max_settling_jerk_mps3"] = max(
            float(record["max_settling_jerk_mps3"]), float(row.get("settling_max_jerk_mps3") or 0.0)
        )
        if phase == "SETTLING_TO_LOW_AUTHORITY":
            record["all_settling_tau_frozen"] = bool(record["all_settling_tau_frozen"]) and bool(row.get("settling_tau_frozen"))
        record["member_ids"].update(str(member_id) for member_id in deltas)
    return {
        phase: {**record, "member_ids": tuple(sorted(record["member_ids"]))}
        for phase, record in phases.items()
    }


def _phase_split_integrity(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Machine-check the frozen authority and in-window settling contract."""
    dose = _phase_dose(rows)
    assembly = dose.get("ASSEMBLING", {})
    settling = dose.get("SETTLING_TO_LOW_AUTHORITY", {})
    low = [dose.get(phase, {}) for phase in ("HOLDING", "TAPERING")]
    active_rows = [row for row in rows if row.get("control_deltas")]
    return {
        "one_high_authority_assembly_step": int(assembly.get("command_steps", 0)) <= 1,
        "assembly_cap_within_1_0": float(assembly.get("max_authority_mps2", 0.0)) <= 1.0 + 1.0e-9,
        "settling_duration_within_two_steps": int(settling.get("command_steps", 0)) <= 2,
        "settling_jerk_within_0_5": float(settling.get("max_settling_jerk_mps3", 0.0)) <= 0.5 + 1.0e-9,
        "settling_tau_not_tightened": bool(settling.get("all_settling_tau_frozen", True)),
        "hold_and_taper_within_0_3": all(float(item.get("max_authority_mps2", 0.0)) <= 0.3 + 1.0e-9 for item in low),
        "full_coverage_during_active_phases": all(float(item.get("min_lane_coverage", 1.0)) >= 1.0 - 1.0e-9 for item in dose.values()),
        "span_within_50_m_during_active_phases": all(float(item.get("max_span_m", 0.0)) <= 50.0 + 1.0e-9 for item in dose.values()),
        "treatment_window_not_extended": all(int(row.get("active_treatment_steps", 0)) <= TREATMENT_DURATION_S for row in active_rows),
    }


def run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    path = output / "cases" / f"seed{seed}.json"
    if path.is_file() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(path)}
    gate_f = _gate_f_row(Path(case["gate_f"]), seed)
    if not bool((gate_f.get("gate_f") or {}).get("admitted")):
        raise RuntimeError(f"candidate {CANDIDATE_ID} seed {seed} is not admitted by frozen Gate F")
    decision = _relative_decision(gate_f)
    started = time.time()
    control_trace = _replay(seed=seed, decision_step=decision, treatment=False, sumo_binary=str(case["sumo_binary"]))
    treatment_trace = _replay(
        seed=seed, decision_step=decision, treatment=True, sumo_binary=str(case["sumo_binary"]),
        wall_overlay_overrides=overlay_overrides, wall_profile_manifest=profile_manifest,
        treatment_arm_name=f"damping_plus_{CANDIDATE_ID}_wall",
    )
    control = _arm_record(control_trace, decision_step=decision)
    treatment = _arm_record(treatment_trace, decision_step=decision)
    audit = _audit_contract(control_trace, treatment_trace, decision_step=decision, treatment_steps=TREATMENT_DURATION_S, step_length_s=1.0)
    audit["treatment_audit_rows"] = [dict(row) for row in treatment_trace.controller_audits]
    prefix = _prefix_identity(control_trace, treatment_trace, decision)
    dose = _adaptive_dose(audit, decision)
    rows = [dict(row) for row in treatment_trace.controller_audits]
    safety_service = classify_idm_gate_b_case(
        {"arms": {"0": {"outcomes": control["outcomes_880_to_1080"], "wall_engaged": False}, "1": {"outcomes": treatment["outcomes_880_to_1080"], "wall_engaged": bool(audit["wall_override_steps"])}}},
        IDMGateBMargins(),
    )
    integrity = _integrity(audit, prefix, control, treatment, dose)
    phase_dose = _phase_dose(rows)
    phase_integrity = (
        _phase_split_integrity(rows)
        if CANDIDATE_ID == "v5_6_2_phase_split_authority"
        else {}
    )
    integrity.update(phase_integrity)
    record = {
        "schema": "idm_v5_6_tournament_candidate_case/1", "candidate_id": CANDIDATE_ID,
        "seed": seed, "scientific_role": SCIENTIFIC_ROLE, "contract": profile_manifest(),
        "contract_fingerprint_sha256": profile_manifest()["profile_sha256"], "gate_f": gate_f,
        "decision_relative_step": decision, "frozen_conditions": {"mainline_vph": DEMAND_MAINLINE_VPH, "ramp_vph": DEMAND_RAMP_VPH, "cav_penetration": CAV_PENETRATION, "activation_absolute_s": ACTIVATION_TIME_S, "window_s": TREATMENT_DURATION_S, "endpoint_absolute_s": ANALYSIS_END_S, "primary_endpoint": {"queue_count": PRIMARY_QUEUE_COUNT, "speed_mps": PRIMARY_SPEED_THRESHOLD_MPS, "sustain_s": PRIMARY_SUSTAIN_S, "minimum_shortening_s": MINIMUM_SHORTENING_S}},
        "prefix_identity": prefix, "integrity": integrity, "treatment_contract": audit,
        "treatment_dose": dose, "phase_dose": phase_dose, "phase_integrity": phase_integrity,
        "follower_compression": _compression_summary(rows),
        "arms": {"0": control, "1": treatment}, "safety_service_guardrail": safety_service,
        "primary_physical_effect": _physical_effect(control, treatment),
        "v5_4_reference": _reference(V54_REFERENCE, seed),
        "v5_5_1_reference": _reference(V551_REFERENCE, seed),
        "wall_clock_seconds": time.time() - started,
    }
    atomic_write_json(path, record)
    return {"status": "completed", "candidate_id": CANDIDATE_ID, "seed": seed, "path": str(path), "integrity": all(integrity.values()), "phase_integrity": phase_integrity, "safety_admissible": safety_service["safety_admissible"], "service_noninferior": safety_service["traffic_noninferior"], "physical_benefit": record["primary_physical_effect"]["primary_physical_benefit"], "seconds": record["wall_clock_seconds"]}


def build_summary(output: Path) -> dict[str, Any]:
    rows = [json.loads((output / "cases" / f"seed{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS]
    by_seed = {row["seed"]: row for row in rows}
    gates = {
        "all_integrity": all(all(row["integrity"].values()) for row in rows),
        "all_safety_admissible": all(row["safety_service_guardrail"]["safety_admissible"] for row in rows),
        "all_service_admissible": all(row["safety_service_guardrail"]["traffic_noninferior"] for row in rows),
        "4402_retains_material_benefit": bool(by_seed[4402]["primary_physical_effect"]["primary_physical_benefit"]),
        "4405_retains_material_benefit": bool(by_seed[4405]["primary_physical_effect"]["primary_physical_benefit"]),
        "4401_no_physical_harm": not bool(by_seed[4401]["primary_physical_effect"]["induces_sustained_congestion"]),
        "4404_no_physical_harm": not bool(by_seed[4404]["primary_physical_effect"]["induces_sustained_congestion"]),
        "no_negligible_authority": all(float(row["treatment_dose"]["controlled_cav_seconds"]) >= 2.0 for row in rows),
    }
    return {"schema": "idm_v5_6_tournament_candidate_summary/1", "candidate_id": CANDIDATE_ID, "contract": profile_manifest(), "seeds": list(SEEDS), "case_count": len(rows), "gates": gates, "four_case_passed": all(gates.values()), "predictor_training_authorized": False, "fresh_seeds_opened": False, "per_seed": [{"seed": row["seed"], "integrity": row["integrity"], "dose": row["treatment_dose"], "safety_service": row["safety_service_guardrail"], "physical_effect": row["primary_physical_effect"], "compression": row["follower_compression"]} for row in rows]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--gate-f", type=Path, default=DEFAULT_GATE_F)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE)); parser.add_argument("--workers", type=int, default=6); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = args.output.resolve(); (output / "cases").mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "idm_v5_6_tournament_manifest/1", "candidate_id": CANDIDATE_ID, "contract": profile_manifest(), "seeds": list(SEEDS), "workers": min(max(1, args.workers), 6), "gate_f": str(args.gate_f.resolve()), "sumo_binary": args.sumo_binary}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.force and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("existing candidate manifest differs from frozen protocol")
    atomic_write_json(manifest_path, manifest)
    cases = [{"seed": seed, "output": str(output), "gate_f": str(args.gate_f.resolve()), "sumo_binary": args.sumo_binary, "force": args.force} for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 6, len(cases))) as pool:
        for future in as_completed([pool.submit(run_case, case) for case in cases]): print(json.dumps(future.result(), sort_keys=True), flush=True)
    atomic_write_json(output / "summary.json", build_summary(output)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
