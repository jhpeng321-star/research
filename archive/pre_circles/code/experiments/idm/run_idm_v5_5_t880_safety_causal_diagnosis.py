"""Diagnosis-only ablations for the frozen t880 compact moving wall.

This runner is deliberately not a treatment repair or predictor experiment.
It asks one causal question on the already-admitted development seeds: does a
wall safety failure come from software/replay, the first assembly command, or
the subsequent hold/release lifecycle?  The frozen IDM damping controller,
wall geometry, members, and safety rules are reused unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from counterfactual.outcomes import OutcomeDefinition, summarize_branch
from counterfactual.twin_replay import FrozenPulseSchedule, ReplayTrace, run_replay
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import (
    _audit_contract,
    _prefix_identity,
)
from experiments.idm.idm_v5_5_t880_contract import (
    ACTIVATION_TIME_S,
    ANALYSIS_END_S,
    CAV_PENETRATION,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    DEVELOPMENT_SEEDS,
    PRIMARY_QUEUE_COUNT,
    PRIMARY_SPEED_THRESHOLD_MPS,
    PRIMARY_SUSTAIN_S,
    RELEASE_TIME_S,
    TREATMENT_DURATION_S,
    contract_manifest,
)
from experiments.idm.run_idm_v5_5_t880_wall_pilot import (
    DEFAULT_SUMO_EXE,
    _gate_f_row,
    _relative_decision,
)
from experiments.idm.run_idm_wall_pilot import supervisor_overrides
from experiments.idm.wall_v5_4_contract import overlay_overrides, profile_manifest
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from idm.density_mapper import summarize_density_trace
from idm.wall_t880 import congestion_run_summary, upstream_storage_summary
from policies.inflow_control import WallHoldPolicy
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


SCHEMA = "idm_v5_5_t880_safety_causal_diagnosis/1"
MANIFEST_SCHEMA = "idm_v5_5_t880_safety_causal_diagnosis_manifest/1"
VARIANTS = ("sham", "assembly_only")
DIAGNOSIS_SEEDS = (4401, 4402, 4404)
OUTCOME_STEPS = int(ANALYSIS_END_S - ACTIVATION_TIME_S)
DEFAULT_GATE_F = REPO_ROOT / "runs" / "probe" / "idm_v5_5_t880_gate_f_p30_s4401_4406"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_v5_5_t880_safety_causal_diagnosis"


class _ZeroEffectWall(WallHoldPolicy):
    """Keep wall state/configuration but issue no speed or tau commands."""

    def act(self, obs_dict: Mapping[str, Any]) -> dict[str, object]:
        self.last_speed_commands_mps = {}
        return {}

    def act_with_targets(
        self, obs_dict: Mapping[str, Any], targets: Mapping[str, float]
    ) -> dict[str, object]:
        self.last_speed_commands_mps = {}
        return {}

    def restore_tau(self, cav_ids: Any) -> list[dict[str, object]]:
        return []

    def set_tau_blend(
        self, cav_ids: Any, fraction: float
    ) -> list[dict[str, object]]:
        return []


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _schedule(decision_step: int, variant: str):
    if variant == "sham":
        return FrozenPulseSchedule(decision_step, TREATMENT_DURATION_S)
    if variant == "assembly_only":
        return FrozenPulseSchedule(decision_step, 1)
    raise ValueError(f"unknown diagnosis variant {variant!r}")


def _replay(
    *, seed: int, decision_step: int, variant: str | None, sumo_binary: str
) -> ReplayTrace:
    treatment = variant is not None
    metadata = {
        "diagnosis_schema": SCHEMA,
        "diagnosis_variant": variant or "damping_control",
        "scientific_role": "t880_safety_causality_diagnosis_only",
        "activation_absolute_s": ACTIVATION_TIME_S,
        "release_absolute_s": RELEASE_TIME_S,
        "common_endpoint_absolute_s": ANALYSIS_END_S,
        "wall_profile": profile_manifest(),
    }
    overrides = overlay_overrides()
    if variant == "sham":
        overrides["wall_factory"] = _ZeroEffectWall
    treatment_steps = TREATMENT_DURATION_S if variant in (None, "sham") else 1
    return run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=sumo_binary,
        sumo_config=transition_scenario_config(DEMAND_MAINLINE_VPH).name,
        seed=seed,
        decision_step=decision_step,
        horizon_steps=OUTCOME_STEPS,
        treatment_steps=treatment_steps,
        cav_penetration=CAV_PENETRATION,
        sumo_step_length=0.5,
        supervisor_overrides={
            **supervisor_overrides(),
            "use_health_obs": False,
            "use_dv_obs": False,
            "feasibility_mask": False,
            "action_masking": False,
        },
        digits=6,
        base_policy_factory=build_frozen_idm_analytical_policy,
        intervention_schedule=(
            (lambda _step: 0) if not treatment else _schedule(decision_step, variant)
        ),
        overlay_overrides=overrides,
        controller_metadata=metadata,
    )


def _physical_state(snapshot: Any) -> tuple[Any, ...]:
    """State used for causal identity; exclude command and supervisor metadata."""
    return (
        snapshot.relative_step,
        snapshot.sim_time,
        snapshot.vehicles,
        snapshot.active_cavs,
        snapshot.collision_vehicle_ids,
        snapshot.starting_teleport_vehicle_ids,
    )


def _first_physical_divergence(left: ReplayTrace, right: ReplayTrace) -> int | None:
    for left_step, right_step in zip(left.snapshots, right.snapshots):
        if _physical_state(left_step) != _physical_state(right_step):
            return int(left_step.relative_step)
    if len(left.snapshots) != len(right.snapshots):
        return min(len(left.snapshots), len(right.snapshots))
    return None


def _outcome(trace: ReplayTrace) -> dict[str, Any]:
    definition = OutcomeDefinition(
        queue_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        congestion_queue_count=PRIMARY_QUEUE_COUNT,
        congestion_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        sustain_steps=PRIMARY_SUSTAIN_S,
    )
    outcome = summarize_branch(
        trace,
        decision_step=trace.decision_step,
        outcome_steps=OUTCOME_STEPS,
        definition=definition,
    )
    density = summarize_density_trace(
        trace,
        analysis_start_s=ACTIVATION_TIME_S,
        analysis_end_s=ANALYSIS_END_S,
        prevalence_lookback_s=PRIMARY_SUSTAIN_S,
        primary_queue_count=PRIMARY_QUEUE_COUNT,
        primary_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        primary_sustain_s=PRIMARY_SUSTAIN_S,
        secondary_health_threshold=0.15,
        secondary_health_sustain_s=60,
    )
    telemetry = density["telemetry"]
    return {
        "trace_summary": trace.summary(),
        "trace_digest": trace.trace_digest,
        "controller_audit_digest": _sha256_json(trace.controller_audits),
        "outcomes": outcome,
        "density_telemetry": density,
        "physical_congestion_880_to_1080": congestion_run_summary(
            telemetry,
            start_s=ACTIVATION_TIME_S,
            end_s=ANALYSIS_END_S,
            sustain_s=PRIMARY_SUSTAIN_S,
        ),
        "primary_physical_congestion_900_to_1080": congestion_run_summary(
            telemetry,
            start_s=RELEASE_TIME_S,
            end_s=ANALYSIS_END_S,
            sustain_s=PRIMARY_SUSTAIN_S,
        ),
        "upstream_storage_880_to_1080": upstream_storage_summary(
            telemetry, start_s=ACTIVATION_TIME_S, end_s=ANALYSIS_END_S
        ),
    }


def _event_keys(arm: Mapping[str, Any]) -> set[tuple[str, int]]:
    attribution = ((arm.get("outcomes") or {}).get("hard_brake_attribution") or {})
    return {
        (str(row.get("vehicle_id")), int(row.get("relative_step")))
        for row in attribution.get("event_starts", ())
    }


def _audit_summary(trace: ReplayTrace) -> dict[str, Any]:
    audits = [
        row
        for row in trace.controller_audits
        if int(row.get("relative_step", -1)) > trace.decision_step
    ]
    wall_rows = [row for row in audits if row.get("wall_override_ids")]
    return {
        "wall_override_steps": [int(row["relative_step"]) for row in wall_rows],
        "wall_override_ids": sorted(
            {str(v) for row in wall_rows for v in row.get("wall_override_ids", ())}
        ),
        "states": sorted({str(row.get("treatment_state")) for row in audits}),
        "transition_reason_counts": {
            reason: sum(1 for row in audits if str(row.get("transition_reason")) == reason)
            for reason in sorted({str(row.get("transition_reason")) for row in audits})
        },
        "first_request": next((dict(row) for row in audits if row.get("formation")), None),
        "release_rows": [
            {
                "relative_step": int(row["relative_step"]),
                "command": int(row.get("command", 0)),
                "state": row.get("treatment_state"),
                "transition_reason": row.get("transition_reason"),
                "wall_override_ids": list(row.get("wall_override_ids", ())),
            }
            for row in audits
            if int(row.get("relative_step", -1)) > trace.decision_step + 1
        ][:8],
        "controller_audits": [dict(row) for row in audits],
    }


def _variant_record(seed: int, control: ReplayTrace, variant: str, treatment: ReplayTrace) -> dict[str, Any]:
    decision = control.decision_step
    audit = _audit_contract(
        control,
        treatment,
        decision_step=decision,
        treatment_steps=(TREATMENT_DURATION_S if variant == "sham" else 1),
        step_length_s=1.0,
    )
    control_arm = _outcome(control)
    treatment_arm = _outcome(treatment)
    control_events = _event_keys(control_arm)
    treatment_events = _event_keys(treatment_arm)
    divergence = _first_physical_divergence(control, treatment)
    prefix = _prefix_identity(control, treatment, decision)
    return {
        "schema": SCHEMA,
        "seed": seed,
        "variant": variant,
        "decision_relative_step": decision,
        "control": control_arm,
        "treatment": treatment_arm,
        "prefix_identity": prefix,
        "physical_identity": {
            "passed": divergence is None,
            "first_physical_divergence_step": divergence,
            "control_trace_digest": control.trace_digest,
            "treatment_trace_digest": treatment.trace_digest,
        },
        "hard_brake_event_comparison": {
            "control_event_count": len(control_events),
            "treatment_event_count": len(treatment_events),
            "shared_event_count": len(control_events & treatment_events),
            "treatment_only_event_count": len(treatment_events - control_events),
            "control_only_event_count": len(control_events - treatment_events),
            "treatment_only_events": [
                {"vehicle_id": vid, "relative_step": step}
                for vid, step in sorted(treatment_events - control_events)
            ],
        },
        "treatment_audit": audit,
        "lifecycle": _audit_summary(treatment),
    }


def _run_seed(case: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    gate = _gate_f_row(Path(case["gate_f"]), seed)
    if not bool(gate["gate_f"]["admitted"]):
        raise RuntimeError(f"diagnosis seed {seed} is not Gate-F admitted")
    decision = _relative_decision(gate)
    control = _replay(seed=seed, decision_step=decision, variant=None, sumo_binary=str(case["sumo_binary"]))
    variant = str(case["variant"])
    treatment = _replay(seed=seed, decision_step=decision, variant=variant, sumo_binary=str(case["sumo_binary"]))
    record = _variant_record(seed, control, variant, treatment)
    path = Path(case["output"]) / "cases" / f"seed{seed}.{variant}.json"
    atomic_write_json(path, record)
    return {
        "status": "completed",
        "seed": seed,
        "variant": variant,
        "path": str(path),
        "physical_identity_passed": record["physical_identity"]["passed"],
        "first_physical_divergence_step": record["physical_identity"]["first_physical_divergence_step"],
        "wall_override_steps": record["treatment_audit"]["wall_override_steps"],
    }


def _build_summary(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for seed in DIAGNOSIS_SEEDS:
        for variant in VARIANTS:
            path = output / "cases" / f"seed{seed}.{variant}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    sham = [row for row in rows if row["variant"] == "sham"]
    assembly = [row for row in rows if row["variant"] == "assembly_only"]
    sham_identity = all(row["physical_identity"]["passed"] for row in sham)
    assembly_divergence = [row["physical_identity"]["first_physical_divergence_step"] for row in assembly]
    assembly_only_first_command = all(
        row["treatment_audit"]["wall_override_steps"][:1]
        == [row["decision_relative_step"] + 1]
        for row in assembly
    )
    assembly_no_hold = all(
        not any(state == "HOLDING" for state in row["lifecycle"]["states"])
        for row in assembly
    )
    summary = {
        "schema": SCHEMA,
        "manifest_schema": manifest.get("schema"),
        "contract": contract_manifest(),
        "wall_profile": profile_manifest(),
        "seeds": list(DIAGNOSIS_SEEDS),
        "variants": list(VARIANTS),
        "branch_count": len(DIAGNOSIS_SEEDS) * len(VARIANTS) * 2,
        "per_case": [
            {
                "seed": row["seed"],
                "variant": row["variant"],
                "physical_identity": row["physical_identity"],
                "hard_brake_event_comparison": row["hard_brake_event_comparison"],
                "wall_override_steps": row["treatment_audit"]["wall_override_steps"],
                "states": row["lifecycle"]["states"],
            }
            for row in rows
        ],
        "gates": {
            "sham_physical_identity_exact": sham_identity,
            "assembly_only_first_command_only": assembly_only_first_command,
            "assembly_only_has_no_hold": assembly_no_hold,
            "common_horizon": all(
                bool(row["control"]["trace_summary"]["completed_steps"] >= OUTCOME_STEPS)
                and bool(row["treatment"]["trace_summary"]["completed_steps"] >= OUTCOME_STEPS)
                for row in rows
            ),
        },
        "interpretation": {
            "sham_identity_pass": sham_identity,
            "assembly_first_physical_divergence_steps": assembly_divergence,
            "classification": "pending_mechanism_review",
            "predictor_training_authorized": False,
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--gate-f", type=Path, default=DEFAULT_GATE_F)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "cases").mkdir(parents=True, exist_ok=True)
    gate_root = args.gate_f.resolve()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract": contract_manifest(),
        "wall_profile": profile_manifest(),
        "gate_f_root": str(gate_root),
        "seeds": list(DIAGNOSIS_SEEDS),
        "variants": list(VARIANTS),
        "branch_count": len(DIAGNOSIS_SEEDS) * len(VARIANTS) * 2,
        "workers": min(max(1, int(args.workers)), 8),
        "sumo_binary": str(args.sumo_binary),
        "diagnosis_only": True,
        "new_seeds_opened": False,
        "predictor_training_authorized": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        expected = dict(manifest)
        comparable.pop("created_unix_s", None)
        expected.pop("created_unix_s", None)
        if comparable != expected:
            raise RuntimeError("existing diagnosis manifest does not match frozen protocol")
    else:
        atomic_write_json(manifest_path, manifest)
    cases = [
        {
            "seed": seed,
            "variant": variant,
            "gate_f": str(gate_root),
            "output": str(output),
            "sumo_binary": str(args.sumo_binary),
        }
        for seed in DIAGNOSIS_SEEDS
        for variant in VARIANTS
    ]
    with ProcessPoolExecutor(max_workers=min(manifest["workers"], len(cases))) as pool:
        futures = [pool.submit(_run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    summary = _build_summary(output, manifest)
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

