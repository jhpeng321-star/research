"""Run the frozen 21-branch IDM v5.5 damping-only density mapper.

The mapper is a development identification screen, not an efficacy test.  It
uses the exact v5.4 analytical damping control arm at 30% CAV penetration and
keeps the wall command off for every step.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

from counterfactual.twin_replay import run_replay
from experiments.idm.idm_v5_5_density_mapper_contract import (
    ANALYSIS_CLOCK,
    ANALYSIS_END_S,
    ANALYSIS_START_S,
    CAV_PENETRATION,
    DEFAULT_OUTPUT,
    DEVELOPMENT_SEEDS,
    EXPECTED_BRANCHES,
    EXPECTED_PRIMARY_BRANCHES,
    EXPECTED_REPEAT_BRANCHES,
    MAINLINE_DEMANDS_VPH,
    PREVALENCE_LOOKBACK_S,
    PRIMARY_QUEUE_COUNT,
    PRIMARY_SPEED_THRESHOLD_MPS,
    PRIMARY_SUSTAIN_S,
    RAMP_DEMAND_VPH,
    REPEAT_SEED,
    SCIENTIFIC_ROLE,
    SECONDARY_HEALTH_SUSTAIN_S,
    SECONDARY_HEALTH_THRESHOLD,
    contract_manifest,
    scenario_path,
    validate_inputs,
)
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT
from idm.density_mapper import branch_integrity, summarize_density_trace
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


MANIFEST_SCHEMA = "idm_v5_5_damping_density_mapper_manifest/1"
SUMMARY_SCHEMA = "idm_v5_5_damping_density_mapper_summary/1"


def _always_damping(_step: int) -> int:
    return 0


def _manifest_arguments(namespace: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-safe copy of frozen command arguments."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(namespace).items()
    }


def _branch_name(demand: int, seed: int, repeat: bool) -> str:
    suffix = ".repeat" if repeat else ".primary"
    return f"m{int(demand)}.seed{int(seed)}{suffix}.json"


def _run_branch(case: dict[str, Any]) -> dict[str, Any]:
    demand = int(case["demand"])
    seed = int(case["seed"])
    repeat = bool(case["repeat"])
    output = Path(case["output"])
    path = output / "cases" / _branch_name(demand, seed, repeat)
    if path.is_file() and not bool(case.get("force")):
        return {
            "status": "skipped",
            "demand": demand,
            "seed": seed,
            "repeat": repeat,
            "path": str(path),
        }
    started = time.time()
    control_manifest = frozen_idm_analytical_manifest()
    config = scenario_path(demand)
    trace = run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=str(case["sumo_binary"]),
        sumo_config=config.name,
        seed=seed,
        decision_step=0,
        horizon_steps=ANALYSIS_END_S,
        treatment_steps=None,
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
        intervention_schedule=_always_damping,
        overlay_overrides=None,
        controller_metadata={
            "scientific_role": SCIENTIFIC_ROLE,
            "demand_mainline_vph": demand,
            "demand_ramp_vph": RAMP_DEMAND_VPH,
            "cav_penetration": CAV_PENETRATION,
            "base_control_arm_id": control_manifest["controller_id"],
            "base_control_arm_fingerprint_sha256": control_manifest[
                "contract_fingerprint_sha256"
            ],
            "wall_treatment_present": False,
        },
    )
    record = summarize_density_trace(
        trace,
        analysis_start_s=ANALYSIS_START_S,
        analysis_end_s=ANALYSIS_END_S,
        prevalence_lookback_s=PREVALENCE_LOOKBACK_S,
        primary_queue_count=PRIMARY_QUEUE_COUNT,
        primary_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        primary_sustain_s=PRIMARY_SUSTAIN_S,
        secondary_health_threshold=SECONDARY_HEALTH_THRESHOLD,
        secondary_health_sustain_s=SECONDARY_HEALTH_SUSTAIN_S,
    )
    record.update(
        {
            "demand_mainline_vph": demand,
            "demand_ramp_vph": RAMP_DEMAND_VPH,
            "cav_penetration": CAV_PENETRATION,
            "repeat": repeat,
            "scenario_config": str(config),
            "base_control_arm": control_manifest,
            "integrity": branch_integrity(
                record,
                analysis_start_s=ANALYSIS_START_S,
                analysis_end_s=ANALYSIS_END_S,
                prevalence_lookback_s=PREVALENCE_LOOKBACK_S,
            ),
            "wall_clock_seconds": time.time() - started,
        }
    )
    atomic_write_json(path, record)
    return {
        "status": "completed",
        "demand": demand,
        "seed": seed,
        "repeat": repeat,
        "path": str(path),
        "wall_clock_seconds": record["wall_clock_seconds"],
        "primary_classification": record["primary_premerge_endpoint"]["classification"],
    }


def _run_phase(cases: list[dict[str, Any]], workers: int) -> None:
    progress: dict[int, dict[str, int]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_run_branch, case) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result, sort_keys=True), flush=True)
            record = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
            if not bool(record.get("repeat")):
                demand = int(record["demand_mainline_vph"])
                classification = str(
                    record["primary_premerge_endpoint"]["classification"]
                )
                bucket = progress.setdefault(demand, {})
                bucket[classification] = bucket.get(classification, 0) + 1
                if bool(
                    record["primary_premerge_endpoint"].get(
                        "diagnostic_right_censored"
                    )
                ):
                    bucket["diagnostic_right_censored"] = (
                        bucket.get("diagnostic_right_censored", 0) + 1
                    )
                print(
                    json.dumps(
                        {
                            "incremental_manifest_analysis": {
                                str(key): value for key, value in sorted(progress.items())
                            }
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


def _load(output: Path, demand: int, seed: int, repeat: bool) -> dict[str, Any]:
    path = output / "cases" / _branch_name(demand, seed, repeat)
    return json.loads(path.read_text(encoding="utf-8"))


def _repeat_check(primary: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "trace_digest_exact": primary["trace_digest"] == repeat["trace_digest"],
        "controller_audit_digest_exact": (
            primary["controller_audit_digest"] == repeat["controller_audit_digest"]
        ),
        "telemetry_digest_exact": (
            primary["telemetry_digest"] == repeat["telemetry_digest"]
        ),
        "primary_endpoint_exact": (
            primary["primary_premerge_endpoint"]
            == repeat["primary_premerge_endpoint"]
        ),
        "secondary_endpoint_exact": (
            primary["secondary_legacy_health_diagnostic"]
            == repeat["secondary_legacy_health_diagnostic"]
        ),
        "terminal_exact": primary["terminal"] == repeat["terminal"],
        "flow_population_exact": (
            primary["flow_and_population"] == repeat["flow_and_population"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _regime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = []
    for row in rows:
        endpoint = row["primary_premerge_endpoint"]
        value = str(endpoint["classification"])
        if bool(endpoint.get("diagnostic_right_censored")) and value == "no_incident_or_recovered":
            value = "censored_or_invalid"
        classes.append(value)
    event_count = sum(
        value in {"prevalent_at_anchor", "incident_after_anchor"} for value in classes
    )
    no_event_count = classes.count("no_incident_or_recovered")
    censored_count = classes.count("censored_or_invalid")
    prevalent_count = classes.count("prevalent_at_anchor")
    incident_count = classes.count("incident_after_anchor")
    if censored_count:
        category = "censored_or_invalid"
    elif prevalent_count >= 4:
        category = "anchor_too_late"
    elif prevalent_count > 0:
        category = "prevalence_contaminated"
    elif incident_count >= 5:
        category = "overload_candidate"
    elif incident_count <= 1:
        category = "subcritical_candidate"
    elif incident_count == 2:
        category = "lower_transition_candidate"
    elif incident_count in (3, 4):
        category = "upper_transition_candidate"
    else:
        category = "inconclusive"
    secondary_events = sum(
        bool(row["secondary_legacy_health_diagnostic"].get("sustained_event"))
        for row in rows
    )
    disagreements = sum(not bool(row["label_agreement"]) for row in rows)
    return {
        "seed_count": len(rows),
        "event_count": event_count,
        "actionable_incident_count": incident_count,
        "prevalent_at_anchor_count": prevalent_count,
        "incident_after_anchor_count": incident_count,
        "no_incident_or_recovered_count": no_event_count,
        "censored_or_invalid_count": censored_count,
        "secondary_health_event_count": secondary_events,
        "primary_health_disagreement_count": disagreements,
        "primary_health_disagreement_fraction": disagreements / len(rows) if rows else 0.0,
        "category": category,
    }


def build_summary(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    primary = [
        _load(output, demand, seed, False)
        for demand in MAINLINE_DEMANDS_VPH
        for seed in DEVELOPMENT_SEEDS
    ]
    repeats = {
        str(demand): _repeat_check(
            _load(output, demand, REPEAT_SEED, False),
            _load(output, demand, REPEAT_SEED, True),
        )
        for demand in MAINLINE_DEMANDS_VPH
    }
    expected_fingerprint = frozen_idm_analytical_manifest()[
        "contract_fingerprint_sha256"
    ]
    branch_gates = [row["integrity"] for row in primary]
    gates = {
        "declared_21_branch_artifacts_present": (
            len(list((output / "cases").glob("*.json"))) == EXPECTED_BRANCHES
        ),
        "eighteen_independent_primary_branches": len(primary) == EXPECTED_PRIMARY_BRANCHES,
        "three_repeats_not_counted_as_seeds": len(repeats) == EXPECTED_REPEAT_BRANCHES,
        "exact_repeat_identity_every_demand": all(
            value["passed"] for value in repeats.values()
        ),
        "frozen_analytical_controller_exact": all(
            row["base_control_arm"]["contract_fingerprint_sha256"]
            == expected_fingerprint
            for row in primary
        ),
        "p30_exact": all(abs(float(row["cav_penetration"]) - 0.30) <= 1e-12 for row in primary),
        "one_hz_absolute_clock": all(
            value["one_hz_observations"]
            and value["prevalence_lookback_observed"]
            and value["exact_anchor_observed"]
            and value["absolute_window_coverage"]
            and value["exact_endpoint_or_proven_natural_clearance"]
            for value in branch_gates
        ),
        "all_terminal_reasons_valid": all(
            value["valid_terminal_classification"] for value in branch_gates
        ),
        "no_wall_command_or_override": all(
            value["no_wall_command_or_override"] for value in branch_gates
        ),
        "zero_collision": all(value["no_collision"] for value in branch_gates),
        "zero_teleport": all(value["no_teleport"] for value in branch_gates),
    }
    by_demand = {
        str(demand): _regime(
            [row for row in primary if int(row["demand_mainline_vph"]) == demand]
        )
        for demand in MAINLINE_DEMANDS_VPH
    }
    mixed_demands = [
        int(demand)
        for demand, values in by_demand.items()
        if values["category"] in {
            "lower_transition_candidate",
            "upper_transition_candidate",
        }
    ]
    anchor_too_late = [
        int(demand)
        for demand, values in by_demand.items()
        if int(values["prevalent_at_anchor_count"]) >= 4
    ]
    high_disagreement = [
        int(demand)
        for demand, values in by_demand.items()
        if float(values["primary_health_disagreement_fraction"]) > 1.0 / 3.0
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "manifest": manifest,
        "support": {
            "primary_branch_count": len(primary),
            "independent_seed_count_per_demand": len(DEVELOPMENT_SEEDS),
            "repeat_branch_count": len(repeats),
        },
        "integrity_gates": gates,
        "integrity_passed": all(gates.values()),
        "repeat_checks": repeats,
        "by_demand": by_demand,
        "identification": {
            "transition_candidate_demands_vph": mixed_demands,
            "anchor_too_late_demands_vph": anchor_too_late,
            "health_unsuitable_as_primary_label_demands_vph": high_disagreement,
            "headroom_diversity_identified": bool(mixed_demands),
            "next_step_if_integrity_and_headroom_pass": (
                "choose exactly one matched damping-versus-damping-plus-wall treatment question"
            ),
            "next_step_if_no_mixed_cell": (
                "stop duration/predictor expansion; revise measured-state stratification or "
                "freeze one bounded additional density cell"
            ),
        },
        "claims": {
            "damping_effectiveness_verified": False,
            "critical_density_estimated": False,
            "wall_headroom_verified": False,
            "predictor_training_authorized": False,
            "validation_or_final_partition_opened": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-demands", nargs="+", type=int, default=list(MAINLINE_DEMANDS_VPH))
    parser.add_argument("--ramp-demand", type=int, default=RAMP_DEMAND_VPH)
    parser.add_argument("--cav-penetration", type=float, default=CAV_PENETRATION)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEVELOPMENT_SEEDS))
    parser.add_argument("--analysis-clock", default=ANALYSIS_CLOCK)
    parser.add_argument("--analysis-start", type=int, default=ANALYSIS_START_S)
    parser.add_argument("--analysis-end", type=int, default=ANALYSIS_END_S)
    parser.add_argument("--prevalence-lookback-s", type=int, default=PREVALENCE_LOOKBACK_S)
    parser.add_argument("--primary-queue-count", type=int, default=PRIMARY_QUEUE_COUNT)
    parser.add_argument(
        "--primary-speed-threshold-mps", type=float, default=PRIMARY_SPEED_THRESHOLD_MPS
    )
    parser.add_argument("--primary-sustain-s", type=int, default=PRIMARY_SUSTAIN_S)
    parser.add_argument(
        "--secondary-health-threshold", type=float, default=SECONDARY_HEALTH_THRESHOLD
    )
    parser.add_argument(
        "--secondary-health-sustain-s", type=int, default=SECONDARY_HEALTH_SUSTAIN_S
    )
    parser.add_argument("--repeat-seed", type=int, default=REPEAT_SEED)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    validate_inputs(
        demands=args.mainline_demands,
        ramp_demand=args.ramp_demand,
        cav_penetration=args.cav_penetration,
        seeds=args.seeds,
        analysis_clock=args.analysis_clock,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        prevalence_lookback_s=args.prevalence_lookback_s,
        primary_queue_count=args.primary_queue_count,
        primary_speed_threshold_mps=args.primary_speed_threshold_mps,
        primary_sustain_s=args.primary_sustain_s,
        secondary_health_threshold=args.secondary_health_threshold,
        secondary_health_sustain_s=args.secondary_health_sustain_s,
        repeat_seed=args.repeat_seed,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(parents=True, exist_ok=True)
    frozen_contract = contract_manifest()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_fingerprint = existing["frozen_contract"]["contract_fingerprint_sha256"]
        if existing_fingerprint != frozen_contract["contract_fingerprint_sha256"]:
            raise RuntimeError("existing output uses a different frozen density-mapper contract")
    else:
        atomic_write_json(
            manifest_path,
            {
                "schema": MANIFEST_SCHEMA,
                "created_unix_s": time.time(),
                "frozen_contract": frozen_contract,
                "command_arguments": _manifest_arguments(args),
                "execution_phases": [
                    "18 independent damping-only branches",
                    "3 exact repeat branches after primary completion",
                ],
            },
        )
    common = {
        "output": str(output),
        "sumo_binary": args.sumo_binary,
        "force": args.force,
    }
    primaries = [
        {**common, "demand": demand, "seed": seed, "repeat": False}
        for demand in MAINLINE_DEMANDS_VPH
        for seed in DEVELOPMENT_SEEDS
    ]
    repeats = [
        {**common, "demand": demand, "seed": REPEAT_SEED, "repeat": True}
        for demand in MAINLINE_DEMANDS_VPH
    ]
    _run_phase(primaries, args.workers)
    _run_phase(repeats, min(args.workers, len(repeats)))
    summary = build_summary(output)
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
