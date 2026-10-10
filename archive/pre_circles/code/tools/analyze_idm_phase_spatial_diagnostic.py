"""Aggregate frozen P30 phase/spatial diagnostics across development cases.

This is a mechanism audit, not a predictor and not a controller-selection
tool.  It consumes outputs from ``run_idm_decentralized_authority_verification``
that opted into the versioned phase-response and spatial-history contracts.
All paired signs are explicit:

* raw deltas are treatment minus control;
* normalized safety scores are control minus treatment, so positive is better.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.io import atomic_write_json
from idm.phase_diagnostics import PHASE_DIAGNOSTIC_SCHEMA
from idm.spatial_history import SPATIAL_HISTORY_SCHEMA


SCHEMA = "idm_phase_spatial_mechanism_audit/1"
EXPECTED_ACTION_SET = "activation_speed_ceiling_v1"
EXPECTED_MODE = "speed_ceiling"
PHASE_ORDER = ("probe", "hold", "release", "post_release")
BURDEN_METRICS = (
    "severe_braking_count",
    "hard_braking_count",
    "low_ttc_count",
    "acceleration_energy_m2ps4",
    "integrated_negative_acceleration_mps",
)
SERVICE_METRICS = ("arrived_this_step", "merge_discharge_this_step")
BEHAVIOR_FIELDS = (
    "completed_horizon_steps",
    "acceleration_energy_sum",
    "severe_braking_count_sum",
    "low_ttc_exposure_steps",
    "mean_corridor_delay_rate",
    "arrived_vehicles",
    "merge_discharge_count",
    "collision_observations",
    "teleport_observations",
    "target_command_step_count",
    "unexpected_command_count",
    "member_change_rows",
    "target_admission_rows",
    "spatial_totals",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_reward_refs(protocol: Mapping[str, Any]) -> dict[str, float]:
    path = Path(str(protocol["reward_refs_path"]))
    payload = path.read_bytes()
    expected = protocol.get("reward_refs_sha256")
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and str(expected) != actual:
        raise ValueError(f"reward-reference hash mismatch for {path}")
    refs = json.loads(payload.decode("utf-8"))["refs"]
    key = f"osm.dense.idm_m{int(protocol['demand'])}r400.sumocfg"
    selected = refs[key]
    return {
        "severe": float(selected["corridor_severe_braking_count_ref_per_s"]),
        "wave": float(selected["corridor_acceleration_energy_sum_ref_m2ps4"]),
    }


def _normalized_safety_score(
    cumulative_delta: Mapping[str, Any],
    *,
    elapsed_steps: int,
    refs: Mapping[str, float],
) -> dict[str, float]:
    """Return partial-horizon control-minus-treatment reference units."""
    severe_delta = float(cumulative_delta.get("severe_braking_count") or 0.0)
    wave_delta = float(cumulative_delta.get("acceleration_energy_m2ps4") or 0.0)
    severe = -severe_delta / max(float(refs["severe"]) * elapsed_steps, 1e-12)
    wave = -wave_delta / max(float(refs["wave"]) * elapsed_steps, 1e-12)
    severe_cumulative = -severe_delta / max(float(refs["severe"]), 1e-12)
    wave_cumulative = -wave_delta / max(float(refs["wave"]), 1e-12)
    return {
        "severe_braking_effect_reference_units": severe,
        "wave_effect_reference_units": wave,
        "safety_composite_reference_units": 0.5 * (severe + wave),
        "cumulative_safety_benefit_reference_steps": 0.5
        * (severe_cumulative + wave_cumulative),
    }


def _score_class(score: float, deadband: float) -> str:
    if score > deadband:
        return "helpful"
    if score < -deadband:
        return "harmful"
    return "neutral"


def _phase_endpoints(
    effect: Mapping[str, Any],
    *,
    refs: Mapping[str, float],
    deadband: float,
) -> dict[str, Any]:
    paired = effect.get("paired_phase_response") or {}
    if paired.get("schema") != PHASE_DIAGNOSTIC_SCHEMA:
        raise ValueError("effect lacks the required paired phase-response contract")
    steps = list(paired.get("steps") or [])
    if not steps:
        raise ValueError("paired phase response is empty")
    if [int(row["relative_step"]) for row in steps] != list(
        range(1, len(steps) + 1)
    ):
        raise ValueError("paired phase response must be contiguous from step one")

    scored: list[dict[str, Any]] = []
    for row in steps:
        step = int(row["relative_step"])
        score = _normalized_safety_score(
            row["cumulative_treatment_minus_control"],
            elapsed_steps=step,
            refs=refs,
        )
        scored.append(
            {
                "relative_step": step,
                "phase": str(row["phase"]),
                **score,
                "classification": _score_class(
                    score["safety_composite_reference_units"], deadband
                ),
                "cumulative_treatment_minus_control": dict(
                    row["cumulative_treatment_minus_control"]
                ),
            }
        )

    endpoints: dict[str, Any] = {}
    for phase in PHASE_ORDER:
        phase_rows = [row for row in scored if row["phase"] == phase]
        if phase_rows:
            endpoints[phase] = phase_rows[-1]
    final = scored[-1]
    active_phases = [phase for phase in ("probe", "hold") if phase in endpoints]
    active_endpoint = endpoints[active_phases[-1]] if active_phases else None
    active_end_step = int(active_endpoint["relative_step"]) if active_endpoint else 0

    first_attenuated = next(
        (
            row
            for row in scored
            if int(row["relative_step"]) > active_end_step
            and row["classification"] != "helpful"
        ),
        None,
    )
    first_raw_nonpositive = next(
        (
            row
            for row in scored
            if int(row["relative_step"]) > active_end_step
            and row["cumulative_safety_benefit_reference_steps"] <= 0.0
        ),
        None,
    )
    first_harmful = next(
        (
            row
            for row in scored
            if int(row["relative_step"]) > active_end_step
            and row["classification"] == "harmful"
        ),
        None,
    )
    active_helpful = bool(
        active_endpoint and active_endpoint["classification"] == "helpful"
    )
    phase_increments: dict[str, Any] = {}
    previous_endpoint: Mapping[str, Any] | None = None
    previous_step = 0
    for phase in PHASE_ORDER:
        endpoint = endpoints.get(phase)
        if endpoint is None:
            continue
        cumulative = endpoint["cumulative_treatment_minus_control"]
        previous_cumulative = (
            previous_endpoint["cumulative_treatment_minus_control"]
            if previous_endpoint is not None
            else {}
        )
        increment = {
            metric: float(cumulative.get(metric) or 0.0)
            - float(previous_cumulative.get(metric) or 0.0)
            for metric in (*BURDEN_METRICS, *SERVICE_METRICS)
        }
        phase_steps = int(endpoint["relative_step"]) - previous_step
        phase_increments[phase] = {
            "step_count": phase_steps,
            "treatment_minus_control": increment,
            **_normalized_safety_score(
                increment,
                elapsed_steps=max(phase_steps, 1),
                refs=refs,
            ),
        }
        previous_endpoint = endpoint
        previous_step = int(endpoint["relative_step"])
    return {
        "sign_convention": {
            "raw_cumulative_metrics": "treatment_minus_control",
            "normalized_safety_scores": "control_minus_treatment_positive_is_better",
        },
        "endpoints": endpoints,
        "phase_increments": phase_increments,
        "active_endpoint": active_endpoint,
        "final_endpoint": final,
        "active_helpful": active_helpful,
        "first_post_active_deadband_attenuation_step": (
            int(first_attenuated["relative_step"]) if first_attenuated else None
        ),
        "first_post_active_raw_nonpositive_step": (
            int(first_raw_nonpositive["relative_step"])
            if first_raw_nonpositive
            else None
        ),
        "first_post_active_harmful_step": (
            int(first_harmful["relative_step"]) if first_harmful else None
        ),
        "helpful_active_to_final_harmful_reversal": bool(
            active_helpful and effect.get("classification") == "harmful"
        ),
    }


def _phase_event_counts(branch: Mapping[str, Any]) -> dict[str, Any]:
    response = branch.get("phase_response") or {}
    if response.get("schema") != PHASE_DIAGNOSTIC_SCHEMA:
        raise ValueError(f"branch {branch.get('branch_id')} lacks phase diagnostics")
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in response.get("steps") or []:
        phase = str(row.get("phase", "inactive"))
        for event in row.get("safety_events") or []:
            relation = str(event.get("relation_to_target", "corridor_other"))
            for metric, field in (
                ("severe_braking", "severe_braking"),
                ("hard_braking", "hard_braking"),
                ("low_ttc", "low_ttc"),
            ):
                if bool(event.get(field, False)):
                    counts[(phase, relation, metric)] += 1
    return {
        phase: {
            relation: {
                metric: counts[(phase, relation, metric)]
                for metric in ("severe_braking", "hard_braking", "low_ttc")
            }
            for relation in sorted(
                {
                    key[1]
                    for key in counts
                    if key[0] == phase
                }
            )
        }
        for phase in sorted({key[0] for key in counts})
    }


def _flatten_event_counts(value: Mapping[str, Any]) -> Counter[tuple[str, str, str]]:
    output: Counter[tuple[str, str, str]] = Counter()
    for phase, relations in value.items():
        for relation, metrics in relations.items():
            for metric, count in metrics.items():
                output[(str(phase), str(relation), str(metric))] = int(count)
    return output


def _event_delta(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    control_response = control.get("phase_response") or {}
    treatment_response = treatment.get("phase_response") or {}
    if (
        control_response.get("schema") != PHASE_DIAGNOSTIC_SCHEMA
        or treatment_response.get("schema") != PHASE_DIAGNOSTIC_SCHEMA
    ):
        raise ValueError("paired branches lack phase diagnostics")
    control_by_step = {
        int(row["relative_step"]): row
        for row in control_response.get("steps") or []
    }
    treatment_by_step = {
        int(row["relative_step"]): row
        for row in treatment_response.get("steps") or []
    }
    if set(control_by_step) != set(treatment_by_step):
        raise ValueError("paired event traces are not step-aligned")
    control_counts: Counter[tuple[str, str, str]] = Counter()
    treatment_counts: Counter[tuple[str, str, str]] = Counter()
    for step in sorted(treatment_by_step):
        treatment_row = treatment_by_step[step]
        phase = str(treatment_row.get("phase", "inactive"))
        for destination, source in (
            (control_counts, control_by_step[step]),
            (treatment_counts, treatment_row),
        ):
            for event in source.get("safety_events") or []:
                relation = str(event.get("relation_to_target", "corridor_other"))
                for metric in ("severe_braking", "hard_braking", "low_ttc"):
                    if bool(event.get(metric, False)):
                        destination[(phase, relation, metric)] += 1
    keys = sorted(set(control_counts) | set(treatment_counts))
    nested: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for phase, relation, metric in keys:
        nested[phase][relation][metric] = (
            treatment_counts[(phase, relation, metric)]
            - control_counts[(phase, relation, metric)]
        )
    return {
        phase: {relation: dict(metrics) for relation, metrics in relations.items()}
        for phase, relations in nested.items()
    }


def _decision_time_cohort(treatment: Mapping[str, Any]) -> dict[str, str]:
    response = treatment.get("phase_response") or {}
    for row in response.get("steps") or []:
        targets = list(row.get("target_states") or [])
        if not targets:
            continue
        output = {str(targets[0]["vehicle_id"]): "target"}
        for follower in row.get("upstream_followers") or []:
            output[str(follower["vehicle_id"])] = str(
                follower.get("relation_to_target", "decision_follower")
            )
        return output
    return {}


def _stable_event_stats(
    branch: Mapping[str, Any],
    *,
    phase_by_step: Mapping[int, str],
    cohort: Mapping[str, str],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    response = branch.get("phase_response") or {}
    rows = {
        int(row["relative_step"]): row for row in response.get("steps") or []
    }
    previous: dict[str, set[str]] = {
        metric: set()
        for metric in ("severe_braking", "hard_braking", "low_ttc")
    }
    stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"exposure_vehicle_steps": 0, "onset_count": 0, "vehicle_ids": set()}
    )
    for step in sorted(phase_by_step):
        row = rows.get(step)
        if row is None:
            raise ValueError("paired stable-cohort traces are not step-aligned")
        phase = str(phase_by_step[step])
        for metric in ("severe_braking", "hard_braking", "low_ttc"):
            current = {
                str(event["vehicle_id"])
                for event in row.get("safety_events") or []
                if bool(event.get(metric, False))
            }
            for vehicle_id in current:
                relation = cohort.get(vehicle_id, "corridor_other")
                key = (phase, relation, metric)
                stats[key]["exposure_vehicle_steps"] += 1
                stats[key]["vehicle_ids"].add(vehicle_id)
                if vehicle_id not in previous[metric]:
                    stats[key]["onset_count"] += 1
            previous[metric] = current
    return stats


def _stable_cohort_event_summary(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    treatment_rows = (treatment.get("phase_response") or {}).get("steps") or []
    phase_by_step = {
        int(row["relative_step"]): str(row.get("phase", "inactive"))
        for row in treatment_rows
    }
    cohort = _decision_time_cohort(treatment)
    control_stats = _stable_event_stats(
        control, phase_by_step=phase_by_step, cohort=cohort
    )
    treatment_stats = _stable_event_stats(
        treatment, phase_by_step=phase_by_step, cohort=cohort
    )
    nested: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for phase, relation, metric in sorted(set(control_stats) | set(treatment_stats)):
        control_row = control_stats.get(
            (phase, relation, metric),
            {"exposure_vehicle_steps": 0, "onset_count": 0, "vehicle_ids": set()},
        )
        treatment_row = treatment_stats.get(
            (phase, relation, metric),
            {"exposure_vehicle_steps": 0, "onset_count": 0, "vehicle_ids": set()},
        )
        nested[phase][relation][metric] = {
            "control_exposure_vehicle_steps": int(
                control_row["exposure_vehicle_steps"]
            ),
            "treatment_exposure_vehicle_steps": int(
                treatment_row["exposure_vehicle_steps"]
            ),
            "exposure_vehicle_step_delta": int(
                treatment_row["exposure_vehicle_steps"]
                - control_row["exposure_vehicle_steps"]
            ),
            "control_onset_count": int(control_row["onset_count"]),
            "treatment_onset_count": int(treatment_row["onset_count"]),
            "onset_count_delta": int(
                treatment_row["onset_count"] - control_row["onset_count"]
            ),
            "control_unique_vehicle_ids": sorted(control_row["vehicle_ids"]),
            "treatment_unique_vehicle_ids": sorted(treatment_row["vehicle_ids"]),
        }
    return {
        "cohort_vehicle_relation": dict(sorted(cohort.items())),
        "event_unit": "vehicle_step_exposure_with_separate_onsets",
        "by_phase_relation": {
            phase: {relation: dict(metrics) for relation, metrics in relations.items()}
            for phase, relations in nested.items()
        },
    }


def _spatial_availability(branch: Mapping[str, Any]) -> dict[str, Any]:
    history = branch.get("diagnostic_spatial_history") or {}
    if history.get("schema") != SPATIAL_HISTORY_SCHEMA:
        raise ValueError(f"branch {branch.get('branch_id')} lacks spatial history")
    pre = list(history.get("predecision_window") or [])
    post = list(history.get("postdecision_frames") or [])
    frames = pre + post
    valid_frames = [row for row in frames if row.get("snapshot_available", False)]
    fit_slots = valid_fits = detected_frames = 0
    queue_detected_frames = detached_detected_frames = 0
    valid_fit_quality_count = 0
    nonempty_bands = band_slots = 0
    for frame in valid_frames:
        detected = False
        queue_detected = False
        detached_detected = False
        for front in (frame.get("fronts_by_lane") or {}).values():
            queue_detected = queue_detected or (
                front.get("queue_tail_distance_m") is not None
            )
            detached_detected = detached_detected or (
                front.get("detached_wave_front_distance_m") is not None
            )
        detected = queue_detected or detached_detected
        detected_frames += int(detected)
        queue_detected_frames += int(queue_detected)
        detached_detected_frames += int(detached_detected)
        for windows in (frame.get("front_velocity_by_lane") or {}).values():
            for fit in windows.values():
                fit_slots += 1
                valid_fits += int(fit.get("velocity_mps") is not None)
                valid_fit_quality_count += int(
                    fit.get("velocity_mps") is not None
                    and fit.get("r_squared") is not None
                )
        for band in ((frame.get("mainline") or {}).get("all_lanes") or []):
            band_slots += 1
            nonempty_bands += int(int(band.get("vehicle_count") or 0) > 0)
    return {
        "predecision_frame_count": len(pre),
        "postdecision_frame_count": len(post),
        "frame_count": len(frames),
        "snapshot_available_fraction": (
            len(valid_frames) / len(frames) if frames else 0.0
        ),
        "front_detected_frame_fraction": (
            detected_frames / len(valid_frames) if valid_frames else 0.0
        ),
        "queue_front_detected_frame_fraction": (
            queue_detected_frames / len(valid_frames) if valid_frames else 0.0
        ),
        "detached_front_detected_frame_fraction": (
            detached_detected_frames / len(valid_frames) if valid_frames else 0.0
        ),
        "front_velocity_valid_fit_fraction": (
            valid_fits / fit_slots if fit_slots else 0.0
        ),
        "front_velocity_valid_fit_count": valid_fits,
        "front_velocity_fit_with_quality_count": valid_fit_quality_count,
        "front_velocity_fit_slot_count": fit_slots,
        "front_velocity_contract_warning": (
            "existing fits may join queue tails and detached fronts across missing frames"
        ),
        "mainline_nonempty_band_fraction": (
            nonempty_bands / band_slots if band_slots else 0.0
        ),
    }


def _branch_index(root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted((root / "branches").glob("*.json")):
        branch = _read_json(path)
        branch_id = str(branch["branch_id"])
        if branch_id in output:
            raise ValueError(f"duplicate branch id {branch_id} in {root}")
        output[branch_id] = branch
    return output


def _behavior_fingerprint(branch: Mapping[str, Any]) -> str:
    """Hash controller/trajectory outputs while excluding added diagnostics."""
    commands = []
    for row in branch.get("command_rows") or []:
        commands.append(
            {
                key: row.get(key)
                for key in (
                    "relative_step",
                    "vehicle_id",
                    "lane_id",
                    "mode_index",
                    "action_family",
                    "current_speed_mps",
                    "nominal_acceleration_mps2",
                    "desired_acceleration_mps2",
                    "projected_acceleration_mps2",
                    "projection_reasons",
                    "projection_input",
                    "reference",
                )
            }
        )
    payload = {
        key: branch.get(key) for key in BEHAVIOR_FIELDS
    }
    payload["command_rows"] = commands
    return _canonical_hash(payload)


def _sum_event_deltas(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        flattened = _flatten_event_counts(row.get("event_delta_treatment_minus_control") or {})
        totals.update(flattened)
    nested: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (phase, relation, metric), value in sorted(totals.items()):
        nested[phase][relation][metric] = int(value)
    return {
        phase: {relation: dict(metrics) for relation, metrics in relations.items()}
        for phase, relations in nested.items()
    }


def analyze(run_dirs: Sequence[Path]) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for raw_root in run_dirs:
        root = Path(raw_root).resolve()
        summary = _read_json(root / "summary.json")
        protocol = summary["protocol"]
        if protocol.get("action_set") != EXPECTED_ACTION_SET:
            raise ValueError(f"{root} is not the frozen P30 action set")
        if list(protocol.get("active_modes") or []) != [1]:
            raise ValueError(f"{root} does not contain only P30 mode 1")
        if list(protocol.get("candidate_slots") or []) != [1]:
            raise ValueError(f"{root} does not contain only candidate slot 1")
        if protocol.get("response_trace_contract") != PHASE_DIAGNOSTIC_SCHEMA:
            raise ValueError(f"{root} has the wrong phase-response contract")
        spatial_contract = protocol.get("diagnostic_spatial_history_contract") or {}
        if spatial_contract.get("schema") != SPATIAL_HISTORY_SCHEMA:
            raise ValueError(f"{root} has the wrong spatial-history contract")
        if not bool(protocol.get("development_only", False)):
            raise ValueError(f"{root} is not marked development-only")
        refs = _load_reward_refs(protocol)
        branches = _branch_index(root)
        source_root = (
            Path(str(protocol["decision_source_path"])).resolve()
            if protocol.get("decision_source_path") is not None
            else None
        )
        source_branches = (
            _branch_index(source_root)
            if source_root is not None and (source_root / "branches").is_dir()
            else {}
        )
        discoveries = {
            int(row["seed"]): row
            for path in sorted((root / "discoveries").glob("*.json"))
            for row in [_read_json(path)]
        }
        matched_source_contract = (
            protocol.get("decision_source_purpose")
            == "p30_phase_spatial_replay"
            and protocol.get("decision_source_path") is not None
            and protocol.get("decision_source_manifest_sha256") is not None
            and protocol.get("decision_source_protocol_hash") is not None
        )
        source_rows.append(
            {
                "path": str(root),
                "protocol_hash": str(summary["protocol_hash"]),
                "decision_source_path": protocol.get("decision_source_path"),
                "decision_source_manifest_sha256": protocol.get(
                    "decision_source_manifest_sha256"
                ),
                "decision_source_protocol_hash": protocol.get(
                    "decision_source_protocol_hash"
                ),
                "decision_source_purpose": protocol.get("decision_source_purpose"),
                "matched_frozen_source_contract": matched_source_contract,
                "case_count": int(summary.get("case_count", 0)),
            }
        )
        for case in summary.get("cases") or []:
            case_id = str(case["case_id"])
            if case_id in seen_cases:
                raise ValueError(f"duplicate case across inputs: {case_id}")
            seen_cases.add(case_id)
            effects = list(case.get("effects") or [])
            if len(effects) != 1 or effects[0].get("mode") != EXPECTED_MODE:
                raise ValueError(f"{case_id} does not contain exactly one P30 effect")
            effect = effects[0]
            discovery = discoveries.get(int(case["seed"])) or {}
            matched = ((discovery.get("decision") or {}).get(
                "matched_decision_source"
            ) or {})
            matched_identity = bool(
                matched_source_contract
                and matched.get("purpose") == "p30_phase_spatial_replay"
                and matched.get("physical_state_identity") is True
                and matched.get("treatment_identity") is True
                and matched.get("nonclassic_observation_identity") is True
                and matched.get("decision_fingerprint_identity") is True
            )
            control_id = f"{case_id}.abstain"
            replay_id = f"{case_id}.abstain_replay"
            treatment_id = f"{case_id}.{effect['arm']}"
            for branch_id in (control_id, replay_id, treatment_id):
                if branch_id not in branches:
                    raise ValueError(f"missing branch {branch_id}")
            control, replay, treatment = (
                branches[control_id],
                branches[replay_id],
                branches[treatment_id],
            )
            source_behavior_identity = bool(
                control_id in source_branches
                and treatment_id in source_branches
                and _behavior_fingerprint(control)
                == _behavior_fingerprint(source_branches[control_id])
                and _behavior_fingerprint(treatment)
                == _behavior_fingerprint(source_branches[treatment_id])
            )
            control_pre = (control.get("diagnostic_spatial_history") or {}).get(
                "predecision_window"
            ) or []
            replay_pre = (replay.get("diagnostic_spatial_history") or {}).get(
                "predecision_window"
            ) or []
            treatment_pre = (treatment.get("diagnostic_spatial_history") or {}).get(
                "predecision_window"
            ) or []
            pre_hashes = {
                _canonical_hash(control_pre),
                _canonical_hash(replay_pre),
                _canonical_hash(treatment_pre),
            }
            endpoints = _phase_endpoints(
                effect,
                refs=refs,
                deadband=float(summary["frozen_deadband"]),
            )
            phase_snapshot_unavailable_steps = sum(
                not bool(row.get("snapshot_available", False))
                for branch in (control, treatment)
                for row in ((branch.get("phase_response") or {}).get("steps") or [])
            )
            rows.append(
                {
                    "case_id": case_id,
                    "seed": int(case["seed"]),
                    "decision_step": int(case["decision_step"]),
                    "decision_index": int(protocol["decision_index"]),
                    "classification": str(effect["classification"]),
                    "constraints_pass": bool(effect["constraints_pass"]),
                    "target_command_step_count": int(
                        effect["target_command_step_count"]
                    ),
                    "safety_composite_reference_units": float(
                        effect["safety_composite_reference_units"]
                    ),
                    "delay_rate_delta": float(effect["delay_rate_delta"]),
                    "merge_discharge_count_delta": int(
                        effect["merge_discharge_count_delta"]
                    ),
                    "low_ttc_exposure_step_delta": int(
                        effect["low_ttc_exposure_step_delta"]
                    ),
                    "all_branch_decision_identity": bool(
                        case["all_branch_decision_identity"]
                    ),
                    "all_discovery_identity": bool(case["all_discovery_identity"]),
                    "matched_frozen_source_identity": matched_identity,
                    "diagnostic_enablement_behavior_identity": source_behavior_identity,
                    "phase_snapshot_unavailable_steps": int(
                        phase_snapshot_unavailable_steps
                    ),
                    "predecision_spatial_history_identity": len(pre_hashes) == 1,
                    "phase": endpoints,
                    "event_delta_treatment_minus_control": _event_delta(
                        control, treatment
                    ),
                    "stable_decision_cohort_events": _stable_cohort_event_summary(
                        control, treatment
                    ),
                    "spatial_availability": _spatial_availability(treatment),
                }
            )

    rows.sort(key=lambda row: (row["seed"], row["decision_index"], row["case_id"]))
    classifications = Counter(row["classification"] for row in rows)
    reversal_rows = [
        row
        for row in rows
        if row["phase"]["helpful_active_to_final_harmful_reversal"]
    ]
    final_harmful = [row for row in rows if row["classification"] == "harmful"]
    front_fit_slots = sum(
        row["spatial_availability"]["front_velocity_fit_slot_count"] for row in rows
    )
    front_valid_fits = sum(
        row["spatial_availability"]["front_velocity_valid_fit_count"] for row in rows
    )
    integrity = {
        "case_ids_unique": len(rows) == len(seen_cases),
        "all_branch_decision_identity": bool(rows)
        and all(row["all_branch_decision_identity"] for row in rows),
        "all_discovery_identity": bool(rows)
        and all(row["all_discovery_identity"] for row in rows),
        "all_predecision_spatial_histories_identical": bool(rows)
        and all(row["predecision_spatial_history_identity"] for row in rows),
        "all_matched_frozen_source_identities": bool(rows)
        and all(row["matched_frozen_source_identity"] for row in rows),
        "diagnostic_enablement_preserves_behavior": bool(rows)
        and all(row["diagnostic_enablement_behavior_identity"] for row in rows),
        "all_phase_snapshots_available": bool(rows)
        and all(row["phase_snapshot_unavailable_steps"] == 0 for row in rows),
        "all_constraints_pass": bool(rows)
        and all(row["constraints_pass"] for row in rows),
        "all_full_p30_dose": bool(rows)
        and all(row["target_command_step_count"] == 20 for row in rows),
    }
    by_final_class = {
        classification: {
            "case_count": len(group),
            "event_delta_treatment_minus_control": _sum_event_deltas(group),
            "active_helpful_count": sum(
                row["phase"]["active_helpful"] for row in group
            ),
            "active_helpful_to_final_harmful_reversal_count": sum(
                row["phase"]["helpful_active_to_final_harmful_reversal"]
                for row in group
            ),
        }
        for classification in ("helpful", "neutral", "harmful")
        for group in [[row for row in rows if row["classification"] == classification]]
    }
    return {
        "schema": SCHEMA,
        "scope": {
            "development_only": True,
            "new_seeds_added": False,
            "treatment": "frozen_p30_speed_ceiling",
            "purpose": "mechanism_diagnosis_not_predictor_training",
        },
        "sources": source_rows,
        "case_count": len(rows),
        "seed_count": len({row["seed"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "classification_counts": dict(classifications),
        "integrity_gates": integrity,
        "integrity_pass": all(integrity.values()),
        "helpful_active_to_final_harmful_reversal_count": len(reversal_rows),
        "helpful_active_to_final_harmful_reversal_fraction_of_harmful": (
            len(reversal_rows) / len(final_harmful) if final_harmful else None
        ),
        "front_velocity_valid_fit_count": front_valid_fits,
        "front_velocity_fit_slot_count": front_fit_slots,
        "front_velocity_valid_fit_fraction": (
            front_valid_fits / front_fit_slots if front_fit_slots else 0.0
        ),
        "event_delta_treatment_minus_control": _sum_event_deltas(rows),
        "event_delta_scope": (
            "vehicle_step_exposures_using_dynamic_same_lane_relation; "
            "use each row's stable_decision_cohort_events for causal cohort attribution"
        ),
        "phase_low_ttc_scope": (
            "TraCI snapshot leader gaps; distinct from frozen top-level "
            "low_ttc_exposure_step constraint"
        ),
        "by_final_class": by_final_class,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.runs)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
