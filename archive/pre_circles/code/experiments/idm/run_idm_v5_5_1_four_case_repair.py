"""Paired four-case v5.5.1 adaptive hold/release mechanism repair gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from counterfactual.idm_gate_labels import IDMGateBMargins, classify_idm_gate_b_case
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import _audit_contract, _prefix_identity
from experiments.idm.idm_v5_5_t880_contract import (
    ACTIVATION_TIME_S,
    ANALYSIS_END_S,
    CAV_PENETRATION,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    MINIMUM_SHORTENING_S,
    PRIMARY_QUEUE_COUNT,
    PRIMARY_SPEED_THRESHOLD_MPS,
    PRIMARY_SUSTAIN_S,
    RELEASE_TIME_S,
)
from experiments.idm.run_idm_v5_5_t880_wall_pilot import (
    CAUSAL_CHECK_NAMES,
    DEFAULT_GATE_F,
    DEFAULT_SUMO_EXE,
    _arm_record,
    _gate_f_row,
    _physical_effect,
    _relative_decision,
    _replay,
)
from experiments.idm.wall_v5_5_1_contract import overlay_overrides, profile_manifest
from experiments.io import atomic_write_json


SEEDS = (4401, 4402, 4404, 4405)
TREATMENT_DURATION_S = 20
OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_v5_5_1_four_case_repair_p30_s4401_4402_4404_4405"
V54_REFERENCE = REPO_ROOT / "runs" / "probe" / "idm_v5_5_t880_wall_pilot_p30_s4401_4406"
MANIFEST_SCHEMA = "idm_v5_5_1_four_case_repair_manifest/1"
CASE_SCHEMA = "idm_v5_5_1_four_case_repair_case/1"
SUMMARY_SCHEMA = "idm_v5_5_1_four_case_repair_summary/1"


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _adaptive_dose(audit: Mapping[str, Any], decision: int) -> dict[str, Any]:
    """Translate generic lifecycle audit into the v5.5.1 phase-aware dose law."""

    rows = [
        dict(row)
        for row in audit.get("treatment_audit_rows", ())
        if decision < int(row.get("relative_step", -1)) <= decision + TREATMENT_DURATION_S
    ]
    if not rows:
        rows = []
    dose = dict(audit.get("treatment_dose") or {})
    active = [row for row in rows if row.get("wall_override_ids")]
    taper = [row for row in active if row.get("treatment_state") == "TAPERING"]
    holds = [row for row in active if row.get("treatment_state") == "HOLDING"]
    assembly = [row for row in active if row.get("treatment_state") == "ASSEMBLING"]
    coverage = [
        float((row.get("lane_coverage") or {}).get("coverage_fraction", 0.0))
        for row in active
    ]
    adaptive_trigger_rows = [
        row for row in rows if str(row.get("taper_reason") or "").startswith("adaptive_soft_compression:")
    ]
    hard_abort_rows = [row for row in rows if row.get("abort_reason")]
    terminal_rows = [
        row
        for row in audit.get("treatment_audit_rows", ())
        if int(row.get("relative_step", -1)) == decision + TREATMENT_DURATION_S + 1
    ]
    release_identity = bool(
        audit.get("checks", {}).get("release_returns_to_damping")
        and terminal_rows
        and not terminal_rows[0].get("wall_override_ids")
    )
    blends = [
        float(row["taper_blend_fraction"])
        for row in taper
        if row.get("taper_blend_fraction") is not None
    ]
    taper_monotone = bool(
        blends and all(left >= right for left, right in zip(blends, blends[1:])) and blends[-1] == 0.0
    )
    full_coverage = bool(active) and all(value >= 1.0 for value in coverage)
    policy_completed = bool(
        not hard_abort_rows
        and bool(active)
        and full_coverage
        and taper_monotone
        and release_identity
    )
    completion_reason = (
        "hard_abort"
        if hard_abort_rows
        else (
            "adaptive_release_complete"
            if policy_completed and adaptive_trigger_rows
            else "planned_window_complete" if policy_completed else "incomplete_lifecycle"
        )
    )
    return {
        **dose,
        "requested_intervention_window_s": TREATMENT_DURATION_S,
        "policy_window_command_steps": len(rows),
        "realized_assembly_duration_s": len(assembly),
        "realized_holding_duration_s": len(holds),
        "realized_adaptive_release_duration_s": len(taper),
        "effective_holding_fraction_of_window": len(holds) / TREATMENT_DURATION_S,
        "adaptive_soft_release_triggered": bool(adaptive_trigger_rows),
        "adaptive_soft_release_trigger_steps": [
            int(row["relative_step"]) for row in adaptive_trigger_rows
        ],
        "hard_abort_steps": [int(row["relative_step"]) for row in hard_abort_rows],
        "full_lane_coverage_while_active": full_coverage,
        "policy_completed": policy_completed,
        "completion_reason": completion_reason,
        "release_returns_to_exact_damping": release_identity,
        "taper_monotone_to_zero": taper_monotone,
        "requested_and_realized_are_distinct": True,
    }


def _integrity(audit: Mapping[str, Any], prefix: Mapping[str, Any], control: Mapping[str, Any], treatment: Mapping[str, Any], dose: Mapping[str, Any]) -> dict[str, bool]:
    checks = audit.get("checks") or {}
    return {
        "predecision_identity": bool(prefix.get("passed")),
        "causal_action_path": all(bool(checks.get(name)) for name in CAUSAL_CHECK_NAMES),
        "control_common_horizon": bool(control["timing_integrity"]["passed"]),
        "treatment_common_horizon": bool(treatment["timing_integrity"]["passed"]),
        "control_telemetry": bool(control["telemetry_integrity"]["passed"]),
        "treatment_telemetry": bool(treatment["telemetry_integrity"]["passed"]),
        "adaptive_policy_completed": bool(dose["policy_completed"]),
        "lane_coverage_active": bool(dose["full_lane_coverage_while_active"]),
        "release_to_identical_damping": bool(dose["release_returns_to_exact_damping"]),
        "no_hard_abort": not bool(dose["hard_abort_steps"]),
    }


def _v54_reference(seed: int) -> dict[str, Any]:
    path = V54_REFERENCE / "cases" / f"seed{seed}.primary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _compression_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    samples = [row.get("adaptive_release") or {} for row in rows if row.get("adaptive_release")]
    numeric = lambda key, fn: [float(item[key]) for item in samples if item.get(key) is not None]
    gaps = numeric("min_rear_gap_m", min)
    ttcs = numeric("min_rear_ttc_s", min)
    closing = numeric("max_closing_speed_mps", max)
    return {
        "sample_count": len(samples),
        "minimum_rear_gap_m": min(gaps) if gaps else None,
        "minimum_rear_ttc_s": min(ttcs) if ttcs else None,
        "maximum_closing_speed_mps": max(closing) if closing else None,
        "maximum_braking_follower_count": max(
            (int(item.get("braking_follower_count", 0)) for item in samples), default=0
        ),
        "soft_trigger_rows": [
            {"relative_step": int(row["relative_step"]), "state": row.get("adaptive_release")}
            for row in rows
            if (row.get("adaptive_release") or {}).get("soft_reasons")
        ],
    }


def run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    path = output / "cases" / f"seed{seed}.json"
    if path.is_file() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(path)}
    gate_f = _gate_f_row(Path(case["gate_f"]), seed)
    if not bool((gate_f.get("gate_f") or {}).get("admitted")):
        raise RuntimeError(f"four-case seed {seed} is not admitted by frozen treatment-blind Gate F")
    decision = _relative_decision(gate_f)
    started = time.time()
    control_trace = _replay(seed=seed, decision_step=decision, treatment=False, sumo_binary=str(case["sumo_binary"]))
    treatment_trace = _replay(
        seed=seed,
        decision_step=decision,
        treatment=True,
        sumo_binary=str(case["sumo_binary"]),
        wall_overlay_overrides=overlay_overrides,
        wall_profile_manifest=profile_manifest,
        treatment_arm_name="damping_plus_v5_5_1_adaptive_wall",
    )
    control = _arm_record(control_trace, decision_step=decision)
    treatment = _arm_record(treatment_trace, decision_step=decision)
    audit = _audit_contract(control_trace, treatment_trace, decision_step=decision, treatment_steps=TREATMENT_DURATION_S, step_length_s=1.0)
    audit["treatment_audit_rows"] = [dict(row) for row in treatment_trace.controller_audits]
    prefix = _prefix_identity(control_trace, treatment_trace, decision)
    dose = _adaptive_dose(audit, decision)
    audit["treatment_dose_v5_5_1"] = dose
    safety_service = classify_idm_gate_b_case(
        {"arms": {"0": {"outcomes": control["outcomes_880_to_1080"], "wall_engaged": False}, "1": {"outcomes": treatment["outcomes_880_to_1080"], "wall_engaged": bool(audit["wall_override_steps"])}}},
        IDMGateBMargins(),
    )
    effect = _physical_effect(control, treatment)
    integrity = _integrity(audit, prefix, control, treatment, dose)
    rows = [dict(row) for row in treatment_trace.controller_audits]
    reference = _v54_reference(seed)
    record = {
        "schema": CASE_SCHEMA,
        "seed": seed,
        "scientific_role": "v5_5_1_development_only_hold_release_repair",
        "contract": profile_manifest(),
        "contract_fingerprint_sha256": profile_manifest()["profile_sha256"],
        "frozen_conditions": {"mainline_vph": DEMAND_MAINLINE_VPH, "ramp_vph": DEMAND_RAMP_VPH, "cav_penetration": CAV_PENETRATION, "activation_absolute_s": ACTIVATION_TIME_S, "window_s": TREATMENT_DURATION_S, "endpoint_absolute_s": ANALYSIS_END_S, "primary_endpoint": {"queue_count": PRIMARY_QUEUE_COUNT, "speed_mps": PRIMARY_SPEED_THRESHOLD_MPS, "sustain_s": PRIMARY_SUSTAIN_S}},
        "gate_f": gate_f,
        "decision_relative_step": decision,
        "prefix_identity": prefix,
        "integrity": integrity,
        "treatment_contract": audit,
        "treatment_dose": dose,
        "follower_compression": _compression_summary(rows),
        "arms": {"0": control, "1": treatment},
        "safety_service_guardrail": safety_service,
        "primary_physical_effect": effect,
        "v5_4_reference": {"classification": reference["classification"], "safety_service_effects": reference["safety_service_guardrail"]["effects"], "physical_effect": reference["primary_physical_effect"]},
        "wall_clock_seconds": time.time() - started,
    }
    atomic_write_json(path, record)
    return {"status": "completed", "seed": seed, "path": str(path), "integrity": all(integrity.values()), "policy_completed": dose["policy_completed"], "safety_admissible": safety_service["safety_admissible"], "service_noninferior": safety_service["traffic_noninferior"], "physical_benefit": effect["primary_physical_benefit"], "seconds": record["wall_clock_seconds"]}


def build_summary(output: Path) -> dict[str, Any]:
    rows = [json.loads((output / "cases" / f"seed{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS]
    gates = {
        "all_integrity": all(all(row["integrity"].values()) for row in rows),
        "all_safety_admissible": all(row["safety_service_guardrail"]["safety_admissible"] for row in rows),
        "all_service_admissible": all(row["safety_service_guardrail"]["traffic_noninferior"] for row in rows),
        "4402_retains_material_benefit": bool(rows[1]["primary_physical_effect"]["primary_physical_benefit"]),
        "4405_retains_material_benefit": bool(rows[3]["primary_physical_effect"]["primary_physical_benefit"]),
        "4401_no_physical_harm": not bool(rows[0]["primary_physical_effect"]["induces_sustained_congestion"]),
        "4404_no_physical_harm": not bool(rows[2]["primary_physical_effect"]["induces_sustained_congestion"]),
        "no_negligible_authority": all(float(row["treatment_dose"]["controlled_cav_seconds"]) >= 2.0 for row in rows),
    }
    passed = all(gates.values())
    return {"schema": SUMMARY_SCHEMA, "contract": profile_manifest(), "seeds": list(SEEDS), "case_count": len(rows), "gates": gates, "four_case_repair_passed": passed, "predictor_training_authorized": False, "validation_or_final_seeds_opened": False, "per_seed": [{"seed": row["seed"], "integrity": row["integrity"], "dose": row["treatment_dose"], "safety_service": row["safety_service_guardrail"], "physical_effect": row["primary_physical_effect"], "compression": row["follower_compression"]} for row in rows]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--gate-f", type=Path, default=DEFAULT_GATE_F)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); (output / "cases").mkdir(exist_ok=True)
    manifest = {"schema": MANIFEST_SCHEMA, "contract": profile_manifest(), "seeds": list(SEEDS), "workers": min(max(1, args.workers), 6), "gate_f": str(args.gate_f.resolve()), "sumo_binary": args.sumo_binary}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.force and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("existing v5.5.1 manifest differs from the frozen protocol")
    atomic_write_json(manifest_path, manifest)
    cases = [{"seed": seed, "output": str(output), "gate_f": str(args.gate_f.resolve()), "sumo_binary": args.sumo_binary, "force": args.force} for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 6, len(cases))) as pool:
        for future in as_completed([pool.submit(run_case, case) for case in cases]):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    atomic_write_json(output / "summary.json", build_summary(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
