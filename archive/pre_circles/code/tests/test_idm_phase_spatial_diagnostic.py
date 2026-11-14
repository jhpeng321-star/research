from __future__ import annotations

import hashlib
import json

import pytest

from tools.analyze_idm_phase_spatial_diagnostic import (
    _event_delta,
    _normalized_safety_score,
    _phase_endpoints,
    _spatial_availability,
    analyze,
)


def _event(vehicle: str, relation: str) -> dict:
    return {
        "vehicle_id": vehicle,
        "relation_to_target": relation,
        "severe_braking": True,
        "hard_braking": False,
        "low_ttc": False,
    }


def _response(phases, events_by_step=()):
    event_rows = list(events_by_step) + [[] for _ in range(len(phases))]
    return {
        "schema": "idm_phase_response/1",
        "steps": [
            {
                "relative_step": index,
                "phase": phase,
                "snapshot_available": True,
                "safety_events": event_rows[index - 1],
            }
            for index, phase in enumerate(phases, start=1)
        ],
    }


def _spatial_history() -> dict:
    frame = {
        "schema": "idm_spatial_rsu_history/1",
        "snapshot_available": True,
        "simulation_time_s": 1.0,
        "mainline": {
            "all_lanes": [
                {"vehicle_count": 1},
                {"vehicle_count": 0},
            ]
        },
        "fronts_by_lane": {
            "lane_0": {
                "queue_tail_distance_m": 100.0,
                "detached_wave_front_distance_m": None,
            }
        },
        "front_velocity_by_lane": {
            "lane_0": {
                "10s": {"velocity_mps": 2.0},
                "20s": {"velocity_mps": None},
            }
        },
    }
    return {
        "schema": "idm_spatial_rsu_history/1",
        "contract": {"schema": "idm_spatial_rsu_history/1"},
        "predecision_window": [frame],
        "postdecision_frames": [{**frame, "simulation_time_s": 2.0}],
    }


def test_partial_safety_score_uses_positive_for_treatment_improvement():
    result = _normalized_safety_score(
        {
            "severe_braking_count": -2.0,
            "acceleration_energy_m2ps4": -20.0,
        },
        elapsed_steps=10,
        refs={"severe": 0.2, "wave": 2.0},
    )
    assert result["severe_braking_effect_reference_units"] == pytest.approx(1.0)
    assert result["wave_effect_reference_units"] == pytest.approx(1.0)
    assert result["safety_composite_reference_units"] == pytest.approx(1.0)


def test_phase_endpoints_detect_helpful_hold_to_harmful_final_reversal():
    effect = {
        "classification": "harmful",
        "paired_phase_response": {
            "schema": "idm_phase_response/1",
            "steps": [
                {
                    "relative_step": 1,
                    "phase": "hold",
                    "cumulative_treatment_minus_control": {
                        "severe_braking_count": -1.0,
                        "acceleration_energy_m2ps4": -2.0,
                    },
                },
                {
                    "relative_step": 2,
                    "phase": "release",
                    "cumulative_treatment_minus_control": {
                        "severe_braking_count": 0.0,
                        "acceleration_energy_m2ps4": 0.0,
                    },
                },
                {
                    "relative_step": 3,
                    "phase": "post_release",
                    "cumulative_treatment_minus_control": {
                        "severe_braking_count": 1.0,
                        "acceleration_energy_m2ps4": 2.0,
                    },
                },
            ],
        },
    }
    result = _phase_endpoints(
        effect,
        refs={"severe": 1.0, "wave": 1.0},
        deadband=0.02,
    )
    assert result["active_endpoint"]["classification"] == "helpful"
    assert result["first_post_active_deadband_attenuation_step"] == 2
    assert result["first_post_active_raw_nonpositive_step"] == 2
    assert result["first_post_active_harmful_step"] == 3
    assert result["helpful_active_to_final_harmful_reversal"] is True


def test_event_delta_uses_treatment_phase_for_step_aligned_control_events():
    control = {
        "phase_response": _response(
            ["inactive", "inactive"],
            [[_event("follower", "upstream_follower_1")], []],
        )
    }
    treatment = {
        "phase_response": _response(
            ["hold", "release"],
            [[], [_event("follower", "upstream_follower_1")]],
        )
    }
    result = _event_delta(control, treatment)
    assert result["hold"]["upstream_follower_1"]["severe_braking"] == -1
    assert result["release"]["upstream_follower_1"]["severe_braking"] == 1
    assert "inactive" not in result


def test_spatial_availability_reports_detection_fit_and_band_coverage():
    result = _spatial_availability(
        {"branch_id": "case.arm", "diagnostic_spatial_history": _spatial_history()}
    )
    assert result["frame_count"] == 2
    assert result["snapshot_available_fraction"] == 1.0
    assert result["front_detected_frame_fraction"] == 1.0
    assert result["front_velocity_valid_fit_count"] == 2
    assert result["front_velocity_fit_slot_count"] == 4
    assert result["mainline_nonempty_band_fraction"] == 0.5


def test_full_analyzer_requires_and_preserves_frozen_contract(tmp_path):
    refs_path = tmp_path / "refs.json"
    refs_bytes = json.dumps(
        {
            "refs": {
                "osm.dense.idm_m2000r400.sumocfg": {
                    "corridor_severe_braking_count_ref_per_s": 1.0,
                    "corridor_acceleration_energy_sum_ref_m2ps4": 1.0,
                }
            }
        }
    ).encode("utf-8")
    refs_path.write_bytes(refs_bytes)
    root = tmp_path / "run"
    (root / "branches").mkdir(parents=True)
    (root / "discoveries").mkdir(parents=True)
    source_root = tmp_path / "source"
    (source_root / "branches").mkdir(parents=True)
    case_id = "m2000.seed4129.t354"
    phases = ["hold", "release", "post_release"]
    history = _spatial_history()
    common = {
        "case_id": case_id,
        "seed": 4129,
        "decision_step": 354,
        "diagnostic_spatial_history": history,
    }
    branches = [
        {
            **common,
            "branch_id": f"{case_id}.abstain",
            "arm": "abstain",
            "phase_response": _response(["inactive"] * 3),
        },
        {
            **common,
            "branch_id": f"{case_id}.abstain_replay",
            "arm": "abstain_replay",
            "phase_response": _response(["inactive"] * 3),
        },
        {
            **common,
            "branch_id": f"{case_id}.candidate1_speed_ceiling",
            "arm": "candidate1_speed_ceiling",
            "phase_response": _response(phases),
        },
    ]
    for branch in branches:
        for branch_root in (root, source_root):
            (branch_root / "branches" / f"{branch['branch_id']}.json").write_text(
                json.dumps(branch), encoding="utf-8"
            )
    cumulative = [
        (-1.0, -2.0),
        (0.0, 0.0),
        (1.0, 2.0),
    ]
    effect = {
        "arm": "candidate1_speed_ceiling",
        "mode": "speed_ceiling",
        "classification": "harmful",
        "constraints_pass": True,
        "target_command_step_count": 20,
        "safety_composite_reference_units": -0.5,
        "delay_rate_delta": 0.0,
        "merge_discharge_count_delta": 0,
        "low_ttc_exposure_step_delta": 0,
        "paired_phase_response": {
            "schema": "idm_phase_response/1",
            "steps": [
                {
                    "relative_step": step,
                    "phase": phase,
                    "cumulative_treatment_minus_control": {
                        "severe_braking_count": severe,
                        "acceleration_energy_m2ps4": energy,
                    },
                }
                for step, (phase, (severe, energy)) in enumerate(
                    zip(phases, cumulative), start=1
                )
            ],
        },
    }
    protocol = {
        "action_set": "activation_speed_ceiling_v1",
        "active_modes": [1],
        "candidate_slots": [1],
        "response_trace_contract": "idm_phase_response/1",
        "diagnostic_spatial_history_contract": {
            "schema": "idm_spatial_rsu_history/1"
        },
        "development_only": True,
        "reward_refs_path": str(refs_path),
        "reward_refs_sha256": hashlib.sha256(refs_bytes).hexdigest(),
        "demand": 2000,
        "decision_index": 1,
        "decision_source_path": str(source_root),
        "decision_source_manifest_sha256": "manifest",
        "decision_source_protocol_hash": "protocol",
        "decision_source_purpose": "p30_phase_spatial_replay",
    }
    (root / "discoveries" / "m2000.seed4129.json").write_text(
        json.dumps(
            {
                "seed": 4129,
                "decision": {
                    "matched_decision_source": {
                        "purpose": "p30_phase_spatial_replay",
                        "physical_state_identity": True,
                        "treatment_identity": True,
                        "nonclassic_observation_identity": True,
                        "decision_fingerprint_identity": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "protocol_hash": "run-protocol",
                "protocol": protocol,
                "case_count": 1,
                "frozen_deadband": 0.02,
                "cases": [
                    {
                        "case_id": case_id,
                        "seed": 4129,
                        "decision_step": 354,
                        "all_branch_decision_identity": True,
                        "all_discovery_identity": True,
                        "effects": [effect],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = analyze([root])

    assert report["integrity_pass"] is True
    assert report["classification_counts"] == {"harmful": 1}
    assert report["helpful_active_to_final_harmful_reversal_count"] == 1
    assert report["front_velocity_valid_fit_fraction"] == 0.5
    assert report["rows"][0]["predecision_spatial_history_identity"] is True
