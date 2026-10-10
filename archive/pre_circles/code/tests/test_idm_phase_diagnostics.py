import pytest

from idm.phase_diagnostics import (
    TreatmentPhaseTracker,
    cumulative_paired_deltas,
    phase_step_diagnostic,
    upstream_follower_states,
    vehicle_safety_events,
)
from tools.run_idm_decentralized_authority_verification import _effect


class _Snapshot:
    def __init__(self):
        self._rows = {
            "target": {"pos": 100.0, "speed": 10.0, "accel": -0.5},
            "f1": {"pos": 85.0, "speed": 14.0, "accel": -4.8},
            "f2": {"pos": 65.0, "speed": 15.0, "accel": -1.0},
            "other": {"pos": 120.0, "speed": 8.0, "accel": 0.0},
        }

    def has(self, vehicle_id):
        return vehicle_id in self._rows

    def ordered_ids(self):
        return tuple(self._rows)

    def lane_id(self, _vehicle_id):
        return "mainline_0"

    def lane_pos(self, vehicle_id):
        return self._rows[vehicle_id]["pos"]

    def speed(self, vehicle_id):
        return self._rows[vehicle_id]["speed"]

    def accel(self, vehicle_id):
        return self._rows[vehicle_id]["accel"]

    def road(self, _vehicle_id):
        return "mainline"

    def leader(self, vehicle_id):
        return {
            "target": ("other", 15.0),
            "f1": ("target", 10.0),
            "f2": ("f1", 15.0),
            "other": None,
        }[vehicle_id]


def test_phase_tracker_labels_realized_probe_hold_release_and_native_steps():
    tracker = TreatmentPhaseTracker(probe_steps=2, release_steps=2)
    assert tracker.observe(relative_step=1, active_vehicle_ids=["a"])["phase"] == "probe"
    assert tracker.observe(relative_step=2, active_vehicle_ids=["a"])["phase"] == "probe"
    assert tracker.observe(relative_step=3, active_vehicle_ids=["a"])["phase"] == "hold"
    release = tracker.observe(
        relative_step=4,
        active_vehicle_ids=[],
        member_changes=[{"reason": "option_completed"}],
    )
    assert release["phase"] == "release"
    assert release["release_reason"] == "option_completed"
    assert tracker.observe(relative_step=5, active_vehicle_ids=[])["phase"] == "release"
    assert tracker.observe(relative_step=6, active_vehicle_ids=[])["phase"] == "post_release"


def test_control_branch_remains_inactive_without_realized_commands():
    tracker = TreatmentPhaseTracker()
    rows = [
        tracker.observe(relative_step=step, active_vehicle_ids=[])
        for step in range(1, 8)
    ]
    assert {row["phase"] for row in rows} == {"inactive"}


def test_current_fixed_option_has_no_fabricated_probe_phase():
    tracker = TreatmentPhaseTracker(probe_steps=0)
    rows = [
        tracker.observe(relative_step=step, active_vehicle_ids=["a"])
        for step in range(1, 4)
    ]
    assert {row["phase"] for row in rows} == {"hold"}


def test_upstream_followers_are_ranked_and_include_chain_ttc():
    rows = upstream_follower_states(_Snapshot(), "target", count=2)
    assert [row["vehicle_id"] for row in rows] == ["f1", "f2"]
    assert rows[0]["relation_to_target"] == "upstream_follower_1"
    assert rows[0]["gap_to_leader_m"] == pytest.approx(10.0)
    assert rows[0]["ttc_s"] == pytest.approx(2.5)
    assert rows[1]["leader_vehicle_id"] == "f1"
    assert rows[1]["gap_to_leader_m"] == pytest.approx(15.0)
    assert rows[1]["ttc_s"] == pytest.approx(15.0)


def test_vehicle_safety_events_attribute_vehicle_and_relation():
    rows = vehicle_safety_events(
        _Snapshot(),
        ["target", "f1", "f2", "other"],
        relative_step=3,
        simulation_time_s=12.5,
        relation_by_vehicle={"target": "target", "f1": "upstream_follower_1"},
        low_ttc_threshold_s=3.0,
    )
    by_id = {row["vehicle_id"]: row for row in rows}
    assert by_id["f1"]["hard_braking"] is True
    assert by_id["f1"]["low_ttc"] is True
    assert by_id["f1"]["ttc_s"] == pytest.approx(2.5)
    assert by_id["f1"]["relation_to_target"] == "upstream_follower_1"
    assert by_id["f1"]["lane_position_m"] == pytest.approx(85.0)
    assert by_id["f1"]["simulation_time_s"] == pytest.approx(12.5)


def test_cumulative_paired_deltas_are_step_aligned_and_explicitly_signed():
    control = [
        {
            "relative_step": 1,
            "severe_braking_count": 1,
            "acceleration_energy_m2ps4": 2.0,
            "merge_discharge_this_step": 1,
        },
        {
            "relative_step": 2,
            "severe_braking_count": 0,
            "acceleration_energy_m2ps4": 3.0,
            "merge_discharge_this_step": 0,
        },
    ]
    treatment = [
        {
            "relative_step": 1,
            "phase": "probe",
            "severe_braking_count": 0,
            "acceleration_energy_m2ps4": 4.0,
            "merge_discharge_this_step": 0,
        },
        {
            "relative_step": 2,
            "phase": "hold",
            "severe_braking_count": 2,
            "acceleration_energy_m2ps4": 1.0,
            "merge_discharge_this_step": 1,
        },
    ]
    rows = cumulative_paired_deltas(control, treatment)
    assert rows[0]["step_treatment_minus_control"]["severe_braking_count"] == -1
    assert rows[1]["cumulative_treatment_minus_control"]["severe_braking_count"] == 1
    assert rows[1]["cumulative_treatment_minus_control"]["acceleration_energy_m2ps4"] == 0
    assert rows[1]["cumulative_treatment_minus_control"]["merge_discharge_this_step"] == 0


def test_phase_step_diagnostic_preserves_events_and_realized_phase():
    row = phase_step_diagnostic(
        {"relative_step": 2, "phase": "probe", "active_vehicle_ids": ["target"]},
        {
            "simulation_time_s": 9.0,
            "snapshot_available": True,
            "absorbers": [{"vehicle_id": "target"}],
            "upstream_followers": [{"vehicle_id": "f1"}],
            "safety_events": [{"vehicle_id": "f1", "hard_braking": True}],
            "corridor": {
                "severe_braking_count": 1,
                "hard_braking_count": 1,
                "low_ttc_count": 0,
                "acceleration_energy_m2ps4": 8.0,
                "integrated_negative_acceleration_mps": 4.0,
            },
            "merge_service": {"arrived_this_step": 2, "collapse_condition": False},
        },
        merge_discharge_this_step=1,
    )
    assert row["phase"] == "probe"
    assert row["safety_events"][0]["vehicle_id"] == "f1"
    assert row["upstream_followers"][0]["vehicle_id"] == "f1"
    assert row["merge_discharge_this_step"] == 1


def test_cumulative_paired_deltas_reject_misaligned_replays():
    with pytest.raises(ValueError, match="must match exactly"):
        cumulative_paired_deltas(
            [{"relative_step": 1}],
            [{"relative_step": 2}],
        )


def test_authority_effect_attaches_opt_in_paired_phase_trace():
    base = {
        "arm": "abstain",
        "candidate_slot": None,
        "mode_index": 0,
        "mode": "abstain",
        "completed_horizon_steps": 60,
        "target_command_step_count": 0,
        "unexpected_command_count": 0,
        "collision_observations": 0,
        "teleport_observations": 0,
        "low_ttc_exposure_steps": 0,
        "mean_corridor_delay_rate": 0.0,
        "arrived_vehicles": 10,
        "merge_discharge_count": 20,
        "severe_braking_count_sum": 1.0,
        "acceleration_energy_sum": 10.0,
        "spatial_totals": {
            "influence_region": {"acceleration_energy_m2ps4": 5.0}
        },
        "projection_survival_values": [],
        "phase_response": {
            "schema": "idm_phase_response/1",
            "steps": [{"relative_step": 1, "phase": "inactive", "severe_braking_count": 1}],
        },
    }
    action = {
        **base,
        "arm": "candidate1_nominal",
        "candidate_slot": 1,
        "mode_index": 2,
        "mode": "nominal",
        "target_command_step_count": 20,
        "phase_response": {
            "schema": "idm_phase_response/1",
            "steps": [{"relative_step": 1, "phase": "hold", "severe_braking_count": 2}],
        },
    }
    effect = _effect(base, action, {"severe": 1.0, "wave": 1.0}, deadband=0.02)
    paired = effect["paired_phase_response"]
    assert paired["sign_convention"] == "treatment_minus_control"
    assert paired["steps"][0]["cumulative_treatment_minus_control"][
        "severe_braking_count"
    ] == 1
