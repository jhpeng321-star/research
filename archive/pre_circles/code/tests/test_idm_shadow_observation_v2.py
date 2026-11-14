from __future__ import annotations

from dataclasses import replace
import json

import pytest

from config.constants import DOWNSTREAM_EDGE, MERGE_EDGE, PRE_MERGE_EDGE, RAMP_EDGE
from counterfactual.idm_shadow_observation_v2 import (
    MAX_ANONYMOUS_SLOTS,
    ShadowObservationError,
    capture_predecision_record,
    model_feature_vector,
    shadow_observation_features,
    validate_model_feature_names,
)
from counterfactual.idm_shadow_observation_v2_contract import PROHIBITED_INPUT_TOKENS
from counterfactual.twin_replay import ReplayTrace, StepSnapshot


DECISION = 60


def _row(
    vehicle: str,
    edge: str,
    lane: str,
    position: float,
    speed: float,
    acceleration: float = 0.0,
    *,
    vehicle_type: str = "normal_car",
) -> tuple[object, ...]:
    return (
        vehicle,
        edge,
        lane,
        position,
        speed,
        acceleration,
        0.0,
        0,
        vehicle_type,
        1.0,
    )


def _vehicles(step: int, *, wall: bool = False, reverse: bool = False):
    elapsed = max(0, step - DECISION)
    member_speed = 20.0 if not wall else max(15.0, 20.0 - 0.5 * elapsed)
    member_position = 500.0 + (20.0 if not wall else 17.5) * elapsed
    follower_acceleration = -0.2 if not wall else -1.0
    rows = [
        _row("f", PRE_MERGE_EDGE, f"{PRE_MERGE_EDGE}_0", member_position - 20.0, member_speed + 1.0, follower_acceleration),
        _row("m", PRE_MERGE_EDGE, f"{PRE_MERGE_EDGE}_0", member_position, member_speed),
        _row("l", PRE_MERGE_EDGE, f"{PRE_MERGE_EDGE}_0", member_position + 25.0, member_speed),
        _row("c2", PRE_MERGE_EDGE, f"{PRE_MERGE_EDGE}_1", 510.0 + 19.0 * elapsed, 19.0),
        _row("r", RAMP_EDGE, f"{RAMP_EDGE}_0", 250.0 + 12.0 * elapsed, 12.0),
        _row("d", DOWNSTREAM_EDGE, f"{DOWNSTREAM_EDGE}_0", 50.0 + 17.0 * elapsed, 17.0),
    ]
    if step <= DECISION:
        rows.append(_row("x", MERGE_EDGE, f"{MERGE_EDGE}_1", 60.0, 16.0))
    else:
        rows.append(_row("x", DOWNSTREAM_EDGE, f"{DOWNSTREAM_EDGE}_1", 10.0 + 16.0 * elapsed, 16.0))
    if reverse:
        rows.reverse()
    return tuple(rows)


def _snapshot(step: int, *, wall: bool = False, reverse: bool = False) -> StepSnapshot:
    vehicles = _vehicles(step, wall=wall, reverse=reverse)
    speeds = [float(row[4]) for row in vehicles]
    return StepSnapshot(
        relative_step=step,
        sim_time=float(step),
        action=int(wall and step > DECISION),
        supervisor_observation=(1.0,) * 16,
        vehicles=vehicles,
        active_cavs=("m", "c2"),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=f"unused-{step}-{wall}",
        vehicle_count=len(vehicles),
        mean_speed=sum(speeds) / len(speeds),
        active_time_loss_sum=0.0,
    )


def _audit(step: int, *, wall: bool) -> dict[str, object]:
    active = wall and step > DECISION
    return {
        "relative_step": step,
        "wall_active": active,
        "wall_override_ids": ("m",) if active else (),
        "wall_member_ids": ("m",) if active else (),
        "out_of_band_damping_ids": ("c2",),
        "base_actions": {"m": (0.0, 0.1), "c2": (0.0, 0.2)},
        "applied_env_actions": {"c2": (0.0, 0.2)},
        "treatment_state": "ASSEMBLING" if active and step <= DECISION + 2 else "HOLDING",
        "assembly_elapsed_s": float(max(0, step - DECISION)),
        "current_member_span_m": 0.0,
        "member_speed_spread_mps": 0.0,
        "lane_coverage": {"coverage_fraction": 1.0 if active else 0.0},
        "reserve_ids_by_lane": {f"{PRE_MERGE_EDGE}_0": ("reserve",)},
        "handoff_counts_by_lane": {f"{PRE_MERGE_EDGE}_0": 0},
        "control_deltas": {
            "m": {
                "pre_override_speed_mps": 20.0,
                "member_raw_speed_target_mps": 18.0,
                "member_projected_speed_target_mps": 18.0,
            }
        }
        if active
        else {},
    }


def _trace(*, wall: bool, reverse: bool = False, stop: int = DECISION + 10) -> ReplayTrace:
    snapshots = tuple(
        _snapshot(
            step,
            wall=wall and step > DECISION,
            reverse=reverse,
        )
        for step in range(stop + 1)
    )
    audits = tuple(_audit(step, wall=wall) for step in range(stop + 1))
    return ReplayTrace(
        seed=4401,
        decision_step=DECISION,
        horizon_steps=10,
        prefix_action=0,
        post_action=int(wall),
        treatment_steps=30 if wall else None,
        reset_info={},
        snapshots=snapshots,
        terminated_early=False,
        controller_metadata={},
        controller_audits=audits,
    )


def _feasibility() -> dict[str, object]:
    return {
        "formation": {
            "feasible": True,
            "controlled_cav_ids": ("m",),
            "occupied_lane_ids": (f"{PRE_MERGE_EDGE}_0",),
            "required_lane_ids": (f"{PRE_MERGE_EDGE}_0",),
            "longitudinal_span_m": 0.0,
            "predicted_max_span_m": 5.0,
            "predicted_terminal_span_m": 2.0,
        },
        "lane_coverage": {
            "coverage_fraction": 1.0,
            "full_occupied_lane_coverage": True,
        },
        "projected_dose": {"projected_duration_fraction": 1.0},
        "activation_safety": {
            "safe": True,
            "min_acceleration_mps2": -0.2,
            "min_front_gap_m": 20.0,
            "min_rear_gap_m": 15.0,
            "min_front_ttc_s": 5.0,
            "min_rear_ttc_s": 4.0,
        },
        "admitted": True,
    }


def _available(value):
    return {"value": value, "available": True}


def _live_record() -> dict[str, object]:
    def vehicle(identity: str, position: float, tau: float):
        return {
            "internal_vehicle_id": identity,
            "road": _available(PRE_MERGE_EDGE),
            "lane": _available(f"{PRE_MERGE_EDGE}_0"),
            "lane_index": _available(0),
            "lane_position_m": _available(position),
            "distance_to_merge_m": _available(1144.95 - position),
            "speed_mps": _available(20.0),
            "acceleration_mps2": _available(-0.2),
            "length_m": _available(5.0),
            "type": _available("normal_car"),
            "route_index": _available(0),
            "leader": {
                "internal_vehicle_id": "linked",
                "net_gap_m": 18.0,
                "relative_speed_mps": 1.0,
                "available": True,
            },
            "follower": {
                "internal_vehicle_id": "linked",
                "net_gap_m": 15.0,
                "relative_speed_mps": -1.0,
                "available": True,
            },
            "parameters": {
                "tau_s": _available(tau),
                "min_gap_m": _available(2.5),
                "accel_mps2": _available(2.0),
                "decel_mps2": _available(4.5),
                "emergency_decel_mps2": _available(9.0),
                "apparent_decel_mps2": _available(4.5),
                "speed_factor": _available(1.0),
                "allowed_speed_mps": _available(30.0),
                "max_speed_mps": _available(40.0),
                "action_step_length_s": _available(1.0),
                "speed_mode": _available(31),
                "lane_change_mode": _available(1621),
            },
        }

    return {
        "schema": "idm_v5_4_shadow_observation_v2_predecision_live/1",
        "capture_phase": "after_identical_damping_before_wall_override",
        "decision_step": DECISION,
        "vehicle_selection": "all_live_corridor_vehicles_internal_linkage_only",
        "vehicles": [vehicle("m", 500.0, 1.0), vehicle("f", 480.0, 1.4)],
        "active_cav_internal_ids": ["m"],
        "controlled_cav_internal_ids": ["m"],
        "damping": {
            "action_dict_by_internal_id": {},
            "base_actions_by_internal_id": {},
            "projection_diagnostics": {
                "commands": [
                    {
                        "vehicle_id": "m",
                        "nominal_acceleration_mps2": 0.5,
                        "desired_acceleration_mps2": 0.0,
                        "projected": {
                            "acceleration_mps2": -0.1,
                            "constraint_conflict": False,
                        },
                    }
                ]
            },
        },
        "proposed_wall": {
            "member_internal_ids": ["m"],
            "required_lane_ids": [f"{PRE_MERGE_EDGE}_0"],
            "ranked_same_lane_reserves": {
                f"{PRE_MERGE_EDGE}_0": [
                    {
                        "internal_vehicle_id": "f",
                        "reserve_rank": 1,
                        "distance_to_merge_m": 664.95,
                        "distance_from_primary_reference_m": 20.0,
                    }
                ]
            },
            "release_boundary_m": 50.0,
        },
        "causal_detector_history": {},
        "internal_identifiers_are_linkage_metadata_only": True,
    }


def test_predecision_capture_has_exact_causal_histories_and_numeric_tensors():
    record = capture_predecision_record(
        _trace(wall=False), decision_step=DECISION, feasibility_row=_feasibility()
    )

    assert record["integrity"]["causal_cutoff_passed"] is True
    assert record["integrity"]["primary_sample_count"] == 21
    assert record["integrity"]["captured_sample_count"] == 61
    assert record["history_20s"]["relative_times_s"] == [float(x) for x in range(-20, 1)]
    assert len(record["history_20s"]["tensor_values"]) == 21
    assert len(record["history_20s"]["tensor_values"][0]) == record["history_20s"][
        "anonymous_slot_count"
    ]
    assert record["history_20s"]["anonymous_slot_count"] <= MAX_ANONYMOUS_SLOTS
    assert len(record["history_20s"]["channel_names"]) == 19
    assert record["history_20s"]["tensor_values"] == record["history_20s"]["values"]
    assert record["history_20s"]["tensor_mask"] == record["history_20s"]["presence_mask"]
    names, values = model_feature_vector(record)
    assert names == tuple(sorted(names))
    assert all(isinstance(value, float) for value in values)
    assert not any(name.startswith("history60_") for name in names)
    assert record["sensitivity_model_features_60s"]


def test_exact_shadow_reports_physical_differences_and_audit_identity():
    record = shadow_observation_features(
        _trace(wall=False),
        _trace(wall=True),
        decision_step=DECISION,
        feasibility_row=_feasibility(),
    )

    assert record["report_times_s"] == [1, 2, 5, 10]
    assert record["integrity"]["prefix_identity"] is True
    assert record["integrity"]["target_only_wall_override"] is True
    assert record["integrity"]["out_of_band_damping_identity"] is True
    assert record["integrity"]["checked_out_of_band_commands"] == 10
    features = record["model_features"]
    assert features["shadow_t10_wall_minus_damping_premerge_mean_speed_mps"] < 0.0
    assert features["shadow_t10_minimum_follower_acceleration_mps2"] == -1.0
    assert features["shadow_t1_assembly_state"] == 1.0
    assert features["shadow_t10_hold_state"] == 1.0
    assert features["shadow_t10_lane_coverage_fraction"] == 1.0
    assert features["shadow_10s_projected_override_fraction"] == 1.0
    assert features["shadow_10s_any_assembly_success"] == 1.0
    assert "shadow_t10_wall_lane0_ttc_q10_s" in features
    assert features["shadow_t10_required_follower_deceleration_mps2"] == 1.0


def test_prefix_mismatch_fails_closed():
    wall = _trace(wall=True)
    changed = list(wall.snapshots)
    changed[DECISION] = replace(changed[DECISION], supervisor_observation=(0.0,) * 16)
    wall = replace(wall, snapshots=tuple(changed))

    with pytest.raises(ShadowObservationError, match="prefix mismatch"):
        shadow_observation_features(
            _trace(wall=False), wall, decision_step=DECISION, feasibility_row=_feasibility()
        )


def test_missing_exact_report_fails_closed():
    with pytest.raises(ShadowObservationError, match=r"missing exact shadow report at \+10s"):
        shadow_observation_features(
            _trace(wall=False, stop=DECISION + 9),
            _trace(wall=True, stop=DECISION + 9),
            decision_step=DECISION,
            feasibility_row=_feasibility(),
        )


def test_prohibited_scan_mutations_cannot_change_feature_bytes():
    clean = _feasibility()
    mutated = _feasibility()
    mutated.update(
        {
            "seed": 999999,
            "actual_duration_s": 0.0,
            "realized_duration_fraction": 0.0,
            "completion_reason": "changed",
            "release_reason": "changed",
            "abort_reason": "changed",
            "outcome": {"joint_benefit": True},
            "oracle": {"label": 1},
            "counterfactual_effect": -999.0,
            "future_trajectory": [999.0],
            "wall_selected": True,
        }
    )
    mutated["projected_dose"] = {
        **mutated["projected_dose"],
        "realized_duration_fraction": 0.0,
        "actual_duration_s": 0.0,
    }
    control = _trace(wall=False)
    wall = _trace(wall=True)
    clean_pre = capture_predecision_record(
        control, decision_step=DECISION, feasibility_row=clean
    )
    mutated_pre = capture_predecision_record(
        control, decision_step=DECISION, feasibility_row=mutated
    )
    clean_shadow = shadow_observation_features(
        control, wall, decision_step=DECISION, feasibility_row=clean
    )
    mutated_shadow = shadow_observation_features(
        control, wall, decision_step=DECISION, feasibility_row=mutated
    )
    assert json.dumps(clean_pre["model_features"], sort_keys=True) == json.dumps(
        mutated_pre["model_features"], sort_keys=True
    )
    assert json.dumps(clean_shadow["model_features"], sort_keys=True) == json.dumps(
        mutated_shadow["model_features"], sort_keys=True
    )


def test_vehicle_row_order_does_not_change_features_or_tensor():
    ordinary = capture_predecision_record(
        _trace(wall=False), decision_step=DECISION, feasibility_row=_feasibility()
    )
    reversed_rows = capture_predecision_record(
        _trace(wall=False, reverse=True),
        decision_step=DECISION,
        feasibility_row=_feasibility(),
    )
    assert model_feature_vector(ordinary) == model_feature_vector(reversed_rows)
    assert ordinary["history_20s"]["tensor_values"] == reversed_rows["history_20s"][
        "tensor_values"
    ]

    shadow = shadow_observation_features(
        _trace(wall=False, reverse=True),
        _trace(wall=True),
        decision_step=DECISION,
        feasibility_row=_feasibility(),
    )
    assert shadow["integrity"]["prefix_identity"] is True


def test_exported_names_and_tensor_channels_exclude_identity_and_labels():
    record = capture_predecision_record(
        _trace(wall=False), decision_step=DECISION, feasibility_row=_feasibility()
    )
    shadow = shadow_observation_features(
        _trace(wall=False),
        _trace(wall=True),
        decision_step=DECISION,
        feasibility_row=_feasibility(),
    )
    for exported in (record["model_features"], shadow["model_features"]):
        validate_model_feature_names(exported)
        joined = " ".join(exported).lower()
        assert not any(token in joined for token in PROHIBITED_INPUT_TOKENS)
    channel_names = " ".join(record["history_20s"]["channel_names"]).lower()
    assert not any(token in channel_names for token in PROHIBITED_INPUT_TOKENS)
    assert "4401" not in json.dumps(record["history_20s"])


def test_validator_rejects_prohibited_or_nonfinite_model_inputs():
    with pytest.raises(ShadowObservationError, match="prohibited"):
        validate_model_feature_names({"seed_identity": 1.0})
    with pytest.raises(ShadowObservationError, match="finite scalar"):
        validate_model_feature_names({"safe_feature": float("nan")})


def test_enriched_live_record_is_anonymized_used_and_mutation_safe():
    control = _trace(wall=False)
    live = _live_record()
    record = capture_predecision_record(
        control,
        decision_step=DECISION,
        feasibility_row=_feasibility(),
        live_predecision_record=live,
    )
    features = record["feature_blocks"]["live_predecision_physical"]
    assert features["live_record_available"] == 1.0
    assert features["live_idm_tau_s_mean"] == pytest.approx(1.2)
    assert features["live_damping_command_count"] == 1.0
    assert features["live_damping_authority_mean_mps2"] == pytest.approx(0.6)
    assert features["live_wall_reserve_count"] == 1.0
    assert not any("vehicle_id" in name for name in features)

    renamed = json.loads(json.dumps(live))
    rename = {"m": "absolute-name-changed", "f": "another-name", "linked": "third"}
    for row in renamed["vehicles"]:
        row["internal_vehicle_id"] = rename[row["internal_vehicle_id"]]
        row["leader"]["internal_vehicle_id"] = rename["linked"]
        row["follower"]["internal_vehicle_id"] = rename["linked"]
    renamed["active_cav_internal_ids"] = [rename["m"]]
    renamed["controlled_cav_internal_ids"] = [rename["m"]]
    renamed["proposed_wall"]["member_internal_ids"] = [rename["m"]]
    renamed["proposed_wall"]["ranked_same_lane_reserves"][f"{PRE_MERGE_EDGE}_0"][0][
        "internal_vehicle_id"
    ] = rename["f"]
    renamed["damping"]["projection_diagnostics"]["commands"][0]["vehicle_id"] = rename["m"]
    renamed.update({"outcome": 999, "oracle": 1, "actual_duration_s": 999})
    renamed_record = capture_predecision_record(
        control,
        decision_step=DECISION,
        feasibility_row=_feasibility(),
        live_predecision_record=renamed,
    )
    assert record["model_features"] == renamed_record["model_features"]


def test_tensor_ignores_non_influence_edges_and_marks_exit_events_separately():
    base = _trace(wall=False)
    changed = []
    for snapshot in base.snapshots:
        rows = list(snapshot.vehicles)
        rows.append(_row(f"irrelevant-{snapshot.relative_step}", "unrelated", "unrelated_0", 1.0, 1.0))
        if snapshot.relative_step == 50:
            rows.append(_row("exiting", PRE_MERGE_EDGE, f"{PRE_MERGE_EDGE}_2", 100.0, 10.0))
        changed.append(replace(snapshot, vehicles=tuple(rows), vehicle_count=len(rows)))
    record = capture_predecision_record(
        replace(base, snapshots=tuple(changed)),
        decision_step=DECISION,
        feasibility_row=_feasibility(),
    )
    history = record["history_20s"]
    # Seven base influence-region vehicles plus the one temporary linked vehicle.
    assert history["anonymous_slot_count"] == 8
    entry_index = history["channel_names"].index("entry_event")
    start_index = history["channel_names"].index("window_start_present")
    exit_index = history["channel_names"].index("exit_event")
    assert all(frame[entry_index] == 0.0 for frame in history["tensor_values"][0])
    assert all(
        frame[start_index] == frame[history["channel_names"].index("present")]
        for frame in history["tensor_values"][0]
    )
    exit_locations = [
        (time_index, slot_index)
        for time_index, frame in enumerate(history["tensor_values"])
        for slot_index, slot in enumerate(frame)
        if slot[exit_index] == 1.0
    ]
    assert len(exit_locations) == 1
    time_index, slot_index = exit_locations[0]
    assert history["presence_mask"][time_index][slot_index] == 0.0
    assert history["event_mask"][time_index][slot_index] == 1.0


def test_tensor_capacity_overflow_fails_closed_instead_of_truncating():
    base = _trace(wall=False)
    changed = []
    # Spread 257 unique influence-region vehicles across the 61-second window.
    for snapshot in base.snapshots:
        rows = list(snapshot.vehicles)
        if snapshot.relative_step <= DECISION:
            for ordinal in range(5):
                identity = snapshot.relative_step * 5 + ordinal
                if identity < MAX_ANONYMOUS_SLOTS + 1:
                    rows.append(
                        _row(
                            f"overflow-{identity}",
                            PRE_MERGE_EDGE,
                            f"{PRE_MERGE_EDGE}_2",
                            float(identity % 200),
                            15.0,
                        )
                    )
        changed.append(replace(snapshot, vehicles=tuple(rows), vehicle_count=len(rows)))
    with pytest.raises(ShadowObservationError, match="slot capacity"):
        capture_predecision_record(
            replace(base, snapshots=tuple(changed)),
            decision_step=DECISION,
            feasibility_row=_feasibility(),
        )
