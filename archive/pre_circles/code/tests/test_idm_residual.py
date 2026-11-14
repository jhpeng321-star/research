"""Contract tests for the IDM-native analytical/residual actuator."""

import numpy as np
import pytest

from env.idm_residual import (
    Disturbance,
    IDMResidualActionEngine,
    IDMResidualConfig,
    ProjectionInput,
    VehicleSample,
    _MemberState,
    analytical_acceleration,
    detect_disturbances,
    project_command,
    select_absorbers,
)
from idm.decentralized_damping import (
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
    ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
    ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P50_ACTION_SET,
    Candidate,
)
from idm.rear_aware_feedback import FeedbackResponse, REAR_AWARE_FEEDBACK_V2_SCHEMA


def sample(vid, lane, distance, speed, accel=0.0):
    return VehicleSample(
        vehicle_id=vid,
        road_id="1191554636",
        lane_id=lane,
        lane_index=int(lane.rsplit("_", 1)[-1]),
        lane_pos_m=1000.0 - distance,
        distance_to_merge_m=distance,
        speed_mps=speed,
        acceleration_mps2=accel,
    )


def test_frozen_analytical_equation():
    cfg = IDMResidualConfig()
    value = analytical_acceleration(
        speed_mps=20.0,
        reference_speed_mps=16.0,
        gap_m=40.0,
        min_gap_m=2.5,
        leader_speed_mps=18.0,
        cfg=cfg,
    )
    # .15*(-4) + .02*(40-(2.5+1.4*20)) + .20*(-2)
    assert value == pytest.approx(-0.6 + 0.19 - 0.4)


def test_projection_obeys_jerk_and_comfort_bounds():
    cfg = IDMResidualConfig(dt_s=1.0)
    out = project_command(
        ProjectionInput(
            current_speed_mps=20.0,
            desired_acceleration_mps2=-3.0,
            prior_acceleration_mps2=0.0,
            allowed_speed_mps=30.0,
        ),
        cfg,
    )
    assert out.acceleration_mps2 == pytest.approx(-0.5)
    assert "comfortable_acceleration" in out.reasons
    assert "jerk" in out.reasons


def test_front_constraint_wins_in_front_rear_conflict():
    cfg = IDMResidualConfig(dt_s=1.0)
    out = project_command(
        ProjectionInput(
            current_speed_mps=20.0,
            desired_acceleration_mps2=0.0,
            prior_acceleration_mps2=0.0,
            allowed_speed_mps=30.0,
            leader_gap_m=3.0,
            leader_speed_mps=10.0,
            follower_gap_m=1.0,
            follower_speed_mps=25.0,
        ),
        cfg,
    )
    assert out.constraint_conflict
    assert out.lower_speed_mps == out.upper_speed_mps
    assert "front_follower_conflict" in out.reasons


def test_disturbance_and_absorber_selection_are_lane_local_and_deterministic():
    cfg = IDMResidualConfig(disturbance_min_vehicles=4)
    rows = [
        sample("h0", "1191554636_0", 100.0, 10.0),
        sample("h1", "1191554636_0", 120.0, 12.0),
        sample("h2", "1191554636_0", 140.0, 22.0),
        sample("h3", "1191554636_0", 160.0, 24.0),
        sample("cav_b", "1191554636_0", 360.0, 24.0),
        sample("cav_a", "1191554636_0", 360.0, 24.0),
        sample("other", "1191554636_1", 500.0, 20.0),
    ]
    disturbances = detect_disturbances(rows, cfg)
    assert set(disturbances) == {"1191554636_0"}
    selected = select_absorbers(rows, {"cav_a", "cav_b", "other"}, disturbances, cfg)
    assert selected == {"1191554636_0": "cav_a"}


def test_zero_residual_executes_analytical_path_and_out_of_band_is_untouched(
    monkeypatch,
):
    cfg = IDMResidualConfig(disturbance_min_vehicles=4)
    engine = IDMResidualActionEngine(cfg)
    rows = [
        sample("h0", "1191554636_0", 100.0, 10.0),
        sample("h1", "1191554636_0", 120.0, 12.0),
        sample("h2", "1191554636_0", 140.0, 22.0),
        sample("h3", "1191554636_0", 160.0, 24.0),
        sample("selected", "1191554636_0", 360.0, 24.0),
        sample("out_of_band", "1191554636_1", 500.0, 20.0),
    ]
    monkeypatch.setattr(engine, "_samples", lambda: rows)
    monkeypatch.setattr(engine, "_candidate_safe", lambda _vid: True)
    monkeypatch.setattr(engine, "_release", lambda _active: [])
    called = []

    def record(member, residual, _samples, disturbance=None):
        called.append((member.vehicle_id, residual))
        return {"vehicle_id": member.vehicle_id, "residual_mps2": residual}

    monkeypatch.setattr(engine, "_command_member", record)
    diag = engine.apply_all({}, {"selected", "out_of_band"})
    assert called == [("selected", 0.0)]
    assert diag["controlled_cav_seconds"] == pytest.approx(cfg.dt_s)
    assert "out_of_band" not in {row["vehicle_id"] for row in diag["commands"]}


def test_explicit_zero_residual_matches_implicit_zero(monkeypatch):
    cfg = IDMResidualConfig(disturbance_min_vehicles=4)
    rows = [
        sample("h0", "1191554636_0", 100.0, 10.0),
        sample("h1", "1191554636_0", 120.0, 12.0),
        sample("h2", "1191554636_0", 140.0, 22.0),
        sample("h3", "1191554636_0", 160.0, 24.0),
        sample("selected", "1191554636_0", 360.0, 24.0),
    ]

    def run(actions):
        engine = IDMResidualActionEngine(cfg)
        monkeypatch.setattr(engine, "_samples", lambda: rows)
        monkeypatch.setattr(engine, "_candidate_safe", lambda _vid: True)
        monkeypatch.setattr(engine, "_release", lambda _active: [])
        seen = []
        monkeypatch.setattr(
            engine,
            "_command_member",
            lambda member, residual, _rows, disturbance=None: (
                seen.append((member.vehicle_id, residual))
                or {"vehicle_id": member.vehicle_id, "residual_mps2": residual}
            ),
        )
        return engine.apply_all(actions, {"selected"}), seen

    implicit, implicit_seen = run({})
    explicit, explicit_seen = run({"selected": np.array([0.0], dtype=np.float32)})
    assert implicit_seen == explicit_seen == [("selected", 0.0)]
    assert implicit["commands"] == explicit["commands"]


def test_member_latch_then_same_lane_handoff(monkeypatch):
    from env.idm_residual import _MemberState

    cfg = IDMResidualConfig(minimum_hold_steps=10)
    engine = IDMResidualActionEngine(cfg)
    lane = "1191554636_0"
    rows = {
        "old": sample("old", lane, 400.0, 20.0),
        "new": sample("new", lane, 300.0, 20.0),
    }
    monkeypatch.setattr(engine, "_candidate_safe", lambda _vid: True)
    engine.members[lane] = _MemberState("old", selected_step=1)

    engine.step_index = 5
    engine._update_members({lane: "new"}, rows)
    assert engine.members[lane].vehicle_id == "old"

    engine.step_index = 11
    changes, active = engine._update_members({lane: "new"}, rows)
    assert engine.members[lane].vehicle_id == "new"
    assert active == {"new"}
    assert "old" in engine.releasing
    assert any(row["reason"] == "handoff" and row["new_member"] == "new" for row in changes)


def test_release_returns_vehicle_to_native_idm(monkeypatch):
    import env.idm_residual as module
    from env.idm_residual import _ReleaseState

    engine = IDMResidualActionEngine(IDMResidualConfig(dt_s=1.0))
    engine.releasing["cav"] = _ReleaseState(prior_acceleration_mps2=0.0)
    calls = []
    monkeypatch.setattr(module.traci.vehicle, "getIDList", lambda: ("cav",))
    monkeypatch.setattr(module.traci.vehicle, "getSpeed", lambda _vid: 20.0)
    monkeypatch.setattr(module.traci.vehicle, "getSpeedWithoutTraCI", lambda _vid: 20.1)
    monkeypatch.setattr(
        module.traci.vehicle, "setSpeed", lambda vid, speed: calls.append((vid, speed))
    )

    diagnostics = engine._release(set())
    assert calls == [("cav", -1)]
    assert "cav" not in engine.releasing
    assert diagnostics[0]["released_to_idm"] is True


def test_release_command_respects_jerk_before_returning_to_idm(monkeypatch):
    import env.idm_residual as module
    from env.idm_residual import _ReleaseState

    engine = IDMResidualActionEngine(IDMResidualConfig(dt_s=1.0))
    engine.releasing["cav"] = _ReleaseState(prior_acceleration_mps2=-0.5)
    calls = []
    monkeypatch.setattr(module.traci.vehicle, "getIDList", lambda: ("cav",))
    monkeypatch.setattr(module.traci.vehicle, "getSpeed", lambda _vid: 20.0)
    monkeypatch.setattr(module.traci.vehicle, "getSpeedWithoutTraCI", lambda _vid: 25.0)
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(
        module.traci.vehicle, "setSpeed", lambda vid, speed: calls.append((vid, speed))
    )

    diagnostics = engine._release(set())
    assert calls == [("cav", pytest.approx(20.0))]
    assert diagnostics[0]["released_to_idm"] is False
    assert diagnostics[0]["acceleration_mps2"] == pytest.approx(0.0)


def test_disturbance_reference_uses_detected_lane_p30(monkeypatch):
    import env.idm_residual as module

    cfg = IDMResidualConfig(reference_mode="disturbance_p30")
    engine = IDMResidualActionEngine(cfg)
    member = sample("cav", "1191554636_0", 400.0, 25.0)
    downstream = [sample("fast", "1191554636_0", 150.0, 28.0)]
    disturbance = Disturbance(
        lane_id="1191554636_0",
        distance_to_merge_m=120.0,
        speed_p30_mps=12.0,
        speed_p50_mps=18.0,
        speed_p70_mps=25.0,
        speed_std_mps=5.0,
        vehicle_count=8,
    )
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    reference, diagnostics = engine._reference(member, downstream, disturbance=disturbance)
    assert reference == pytest.approx(12.0)
    assert diagnostics["source_mode"] == "disturbance_p30"


def test_p50_reference_selects_detected_lane_median_without_changing_p30(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(IDMResidualConfig(reference_mode="disturbance_p30"))
    member = sample("cav", "1191554636_0", 400.0, 25.0)
    disturbance = Disturbance(
        lane_id="1191554636_0",
        distance_to_merge_m=120.0,
        speed_p30_mps=12.0,
        speed_p50_mps=18.0,
        speed_p70_mps=25.0,
        speed_std_mps=5.0,
        vehicle_count=8,
    )
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)

    p30, p30_diag = engine._reference(
        member, [member], disturbance=disturbance, disturbance_percentile="p30"
    )
    engine.reference_by_lane.clear()
    p50, p50_diag = engine._reference(
        member, [member], disturbance=disturbance, disturbance_percentile="p50"
    )

    assert p30 == pytest.approx(12.0)
    assert p30_diag["source_mode"] == "disturbance_p30"
    assert p50 == pytest.approx(18.0)
    assert p50_diag["source_mode"] == "disturbance_p50"


@pytest.mark.parametrize(
    ("action_set", "expected_terminal_speed"),
    [
        (ACTIVATION_SPEED_CEILING_ACTION_SET, 10.0),
        (ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET, 10.0),
        (ACTIVATION_SPEED_CEILING_P50_ACTION_SET, 17.0),
    ],
)
def test_terminal_rear_admission_uses_action_specific_target(
    monkeypatch, action_set, expected_terminal_speed
):
    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=action_set,
        )
    )
    lane = "1191554636_0"
    candidate = sample("cav", lane, 400.0, 20.0)
    disturbance = Disturbance(lane, 100.0, 10.0, 17.0, 24.0, 5.0, 8)
    monkeypatch.setattr(
        engine,
        "_candidate_admission",
        lambda *_args, **_kwargs: {"admissible": True, "reasons": ()},
    )
    targets = []

    def terminal_admission(_vehicle_id, target_speed_mps):
        targets.append(target_speed_mps)
        return {
            "terminal_rear_admissible": True,
            "terminal_rear_secure_gap_m": 10.0,
            "terminal_rear_headroom_m": 10.0,
            "terminal_rear_reason": None,
        }

    monkeypatch.setattr(engine, "_terminal_rear_admission", terminal_admission)
    _, admissions = engine._decentralized_candidate_state(
        [candidate], {"cav"}, {lane: disturbance}
    )
    assert admissions["cav"]["admissible"] is True
    assert targets == [pytest.approx(expected_terminal_speed)]


def test_preview_matches_next_applied_members(monkeypatch):
    cfg = IDMResidualConfig(disturbance_min_vehicles=4)
    engine = IDMResidualActionEngine(cfg)
    rows = [
        sample("h0", "1191554636_0", 100.0, 10.0),
        sample("h1", "1191554636_0", 120.0, 12.0),
        sample("h2", "1191554636_0", 140.0, 22.0),
        sample("h3", "1191554636_0", 160.0, 24.0),
        sample("selected", "1191554636_0", 360.0, 24.0),
        sample("other", "1191554636_1", 500.0, 20.0),
    ]
    monkeypatch.setattr(engine, "_samples", lambda: rows)
    monkeypatch.setattr(engine, "_candidate_safe", lambda _vid: True)
    monkeypatch.setattr(engine, "_release", lambda _active: [])
    monkeypatch.setattr(
        engine,
        "_command_member",
        lambda member, residual, _rows, disturbance=None: {
            "vehicle_id": member.vehicle_id,
            "residual_mps2": residual,
        },
    )
    preview = engine.preview_active_members({"selected", "other"})
    diag = engine.apply_all({"selected": 0.1, "other": -0.4}, {"selected", "other"})
    assert preview == set(diag["active_members"].values()) == {"selected"}
    assert diag["commands"][0]["residual_mps2"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("family", "action_value", "expected_filtered", "expected_desired"),
    [
        ("gain", 1.5, 1.0 + 0.5 / 6.0, -0.65),
        ("reference_offset", 2.0, 2.0 / 6.0, -0.55),
    ],
)
def test_structured_action_transforms_before_unchanged_projection(
    monkeypatch, family, action_value, expected_filtered, expected_desired
):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(
        IDMResidualConfig(action_family=family, structured_action_filter_tau_s=5.0)
    )
    member = sample("cav", "1191554636_0", 400.0, 24.0)
    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: {
        "leader_id": None,
        "leader_gap_m": None,
        "leader_speed_mps": None,
        "follower_id": None,
        "follower_gap_m": None,
        "follower_speed_mps": None,
    })
    monkeypatch.setattr(
        engine,
        "_reference",
        lambda *_args, **_kwargs: (
            20.0,
            {"raw_reference_mps": 20.0, "filtered_reference_mps": 20.0},
        ),
    )
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)

    result = engine._command_member(member, action_value, [member])

    assert result["action_family"] == family
    assert result["filtered_action"] == pytest.approx(expected_filtered)
    assert result["nominal_acceleration_mps2"] == pytest.approx(-0.6)
    assert result["desired_acceleration_mps2"] == pytest.approx(expected_desired)
    assert result["projected"]["reasons"] == ("jerk",)


def test_acceleration_cap_reduces_positive_nominal_without_extra_braking(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(IDMResidualConfig())
    member = sample("cav", "1191554636_0", 400.0, 20.0)
    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: {
        "leader_id": None,
        "leader_gap_m": None,
        "leader_speed_mps": None,
        "follower_id": None,
        "follower_gap_m": None,
        "follower_speed_mps": None,
    })
    monkeypatch.setattr(
        engine,
        "_reference",
        lambda *_args, **_kwargs: (
            25.0,
            {"raw_reference_mps": 25.0, "filtered_reference_mps": 25.0},
        ),
    )
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)
    result = engine._command_member(
        member, 0.25, [member], action_family="acceleration_cap", mode_index=2
    )
    assert result["nominal_acceleration_mps2"] == pytest.approx(0.75)
    assert result["desired_acceleration_mps2"] == pytest.approx(0.25)
    assert result["projected"]["acceleration_mps2"] == pytest.approx(0.25)


def test_speed_ceiling_changes_terminal_speed_not_safety_limits(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(IDMResidualConfig())
    member = sample("cav", "1191554636_0", 400.0, 24.0)
    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: {
        "leader_id": None,
        "leader_gap_m": None,
        "leader_speed_mps": None,
        "follower_id": None,
        "follower_gap_m": None,
        "follower_speed_mps": None,
    })
    monkeypatch.setattr(
        engine,
        "_reference",
        lambda *_args, **_kwargs: (
            20.0,
            {"raw_reference_mps": 20.0, "filtered_reference_mps": 20.0},
        ),
    )
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)
    result = engine._command_member(
        member, 0.0, [member], action_family="speed_ceiling", mode_index=1
    )
    assert result["reference"]["effective_reference_mps"] == pytest.approx(20.0)
    assert result["desired_acceleration_mps2"] == pytest.approx(-4.0)
    assert result["projected"]["acceleration_mps2"] == pytest.approx(-0.5)
    assert set(result["projected"]["reasons"]) == {"comfortable_acceleration", "jerk"}


def test_p50_speed_ceiling_uses_same_hard_projection(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(IDMResidualConfig())
    member = sample("cav", "1191554636_0", 400.0, 24.0)
    monkeypatch.setattr(
        engine,
        "_leader_follower",
        lambda _vid: {
            "leader_id": None,
            "leader_gap_m": None,
            "leader_speed_mps": None,
            "follower_id": None,
            "follower_gap_m": None,
            "follower_speed_mps": None,
        },
    )
    percentiles = []

    def reference(*_args, **kwargs):
        percentiles.append(kwargs["disturbance_percentile"])
        return 20.0, {"raw_reference_mps": 20.0, "filtered_reference_mps": 20.0}

    monkeypatch.setattr(engine, "_reference", reference)
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)

    result = engine._command_member(
        member, 0.0, [member], action_family="speed_ceiling_p50", mode_index=1
    )

    assert percentiles == ["p50"]
    assert result["projected"]["acceleration_mps2"] == pytest.approx(-0.5)
    assert set(result["projected"]["reasons"]) == {
        "comfortable_acceleration",
        "jerk",
    }


def test_modified_followerstopper_is_a_p30_front_feedback_ceiling(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(IDMResidualConfig())
    member = sample("cav", "1191554636_0", 400.0, 20.0)
    context = {
        "leader_id": "leader",
        "leader_gap_m": 30.0,
        "leader_speed_mps": 15.0,
        "follower_id": None,
        "follower_gap_m": None,
        "follower_speed_mps": None,
    }
    percentiles = []

    def reference(*_args, **kwargs):
        percentiles.append(kwargs["disturbance_percentile"])
        return 18.0, {"raw_reference_mps": 18.0, "filtered_reference_mps": 18.0}

    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: context)
    monkeypatch.setattr(engine, "_reference", reference)
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "getDecel", lambda _vid: 4.5)
    monkeypatch.setattr(module.traci.vehicle, "getFollowSpeed", lambda *_args: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "getSecureGap", lambda *_args: 5.0)
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)

    result = engine._command_member(
        member,
        0.0,
        [member],
        action_family="modified_followerstopper_p30_ceiling",
        mode_index=1,
    )

    feedback = result["modified_follower_stopper"]
    assert percentiles == ["p30"]
    assert feedback["zone"] == "closing_transition"
    assert 0.0 < feedback["desired_speed_mps"] < 15.0
    assert result["desired_acceleration_mps2"] == pytest.approx(
        feedback["ceiling_acceleration_mps2"]
    )
    assert result["desired_acceleration_mps2"] < result["nominal_acceleration_mps2"]
    assert result["treatment_infeasible"] is False


def test_terminal_speed_admission_rejects_future_rear_secure_gap(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(IDMResidualConfig())
    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: {
        "leader_id": None,
        "leader_gap_m": None,
        "leader_speed_mps": None,
        "follower_id": "follower",
        "follower_gap_m": 20.0,
        "follower_speed_mps": 25.0,
    })
    monkeypatch.setattr(module.traci.vehicle, "getDecel", lambda _vid: 4.5)
    monkeypatch.setattr(module.traci.vehicle, "getSecureGap", lambda *_args: 25.0)
    result = engine._terminal_rear_admission("cav", 15.0)
    assert result["terminal_rear_admissible"] is False
    assert result["terminal_rear_headroom_m"] == pytest.approx(-5.0)
    assert result["terminal_rear_reason"] == "terminal_rear_secure_gap"


def test_modified_followerstopper_projection_conflict_aborts_and_blocks_lane(
    monkeypatch,
):
    lane = "1191554636_0"
    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=(
                ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET
            ),
            release_max_steps=10,
        )
    )
    member = sample("cav", lane, 400.0, 20.0)
    engine.members[lane] = _MemberState(
        vehicle_id="cav",
        selected_step=1,
        action_family="modified_followerstopper_p30_ceiling",
        action_value=0.0,
        mode_index=1,
    )
    monkeypatch.setattr(engine, "_samples", lambda: [member])
    monkeypatch.setattr(
        engine,
        "_decentralized_candidate_state",
        lambda *_args: ({}, {}),
    )
    monkeypatch.setattr(
        engine,
        "_update_decentralized_members",
        lambda *_args: ([], {"cav"}, {}, []),
    )
    monkeypatch.setattr(
        engine,
        "_command_member",
        lambda *_args, **_kwargs: {
            "vehicle_id": "cav",
            "treatment_infeasible": True,
        },
    )
    release_calls = []

    def release(active, *, skip_vehicle_ids=()):
        release_calls.append((set(active), set(skip_vehicle_ids)))
        return []

    monkeypatch.setattr(engine, "_release", release)
    diag = engine.apply_all({}, {"cav"})

    assert engine.members == {}
    assert "cav" in engine.releasing
    assert engine.projected_release_cooldown_until[lane] == 11
    assert release_calls == [(set(), {"cav"})]
    assert diag["treatment_safety_aborts"] == [
        {
            "vehicle_id": "cav",
            "lane_id": lane,
            "reason": "projection_interval_conflict",
            "selected_step": 1,
            "hold_steps": 0,
            "release_deferred_until_next_step": True,
        }
    ]


def test_modified_followerstopper_unsafe_member_blocks_same_lane_handoff(
    monkeypatch,
):
    lane = "1191554636_0"
    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=(
                ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET
            ),
            decentralized_option_steps=20,
            release_max_steps=10,
        )
    )
    old = sample("old", lane, 400.0, 20.0)
    new = sample("new", lane, 300.0, 20.0)
    engine.members[lane] = _MemberState(
        vehicle_id="old",
        selected_step=1,
        action_family="modified_followerstopper_p30_ceiling",
        action_value=0.0,
        mode_index=1,
    )
    engine.step_index = 5
    monkeypatch.setattr(
        engine,
        "_candidate_admission",
        lambda vehicle_id, *_args: {
            "admissible": vehicle_id == "new",
            "reasons": () if vehicle_id == "new" else ("rear_ttc",),
        },
    )
    candidate = Candidate(
        vehicle_id="new",
        lane_id=lane,
        distance_to_merge_m=300.0,
        disturbance_distance_to_merge_m=200.0,
        runway_m=100.0,
        rank=1,
    )

    changes, active, _rejected, bids = engine._update_decentralized_members(
        {lane: (candidate,)},
        {"new": {"admissible": True, "reasons": ()}},
        {"new": 1},
        {"old": old, "new": new},
    )

    assert engine.members == {}
    assert active == set()
    assert "old" in engine.releasing
    assert engine.projected_release_cooldown_until[lane] == 15
    assert bids[0]["outcome"] == "projected_release_or_cooldown"
    assert any(row["reason"] == "unsafe_or_exited" for row in changes)


def test_projected_blend_full_strength_matches_frozen_p30_command(monkeypatch):
    import env.idm_residual as module

    member = sample("cav", "1191554636_0", 400.0, 20.0)
    context = {
        "leader_id": "leader",
        "leader_gap_m": 30.0,
        "leader_speed_mps": 17.0,
        "follower_id": "follower",
        "follower_gap_m": 40.0,
        "follower_speed_mps": 20.0,
    }
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "getAccel", lambda _vid: 2.0)
    monkeypatch.setattr(module.traci.vehicle, "getDecel", lambda _vid: 4.5)
    monkeypatch.setattr(module.traci.vehicle, "getFollowSpeed", lambda *_args: 21.0)
    monkeypatch.setattr(module.traci.vehicle, "getSecureGap", lambda *_args: 5.0)
    monkeypatch.setattr(
        module.traci.vehicle, "getSpeedWithoutTraCI", lambda _vid: 20.0
    )
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)

    def run(family, value):
        engine = IDMResidualActionEngine(
            IDMResidualConfig(
                decentralized_damping_enabled=True,
                decentralized_action_set=(
                    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET
                ),
            )
        )
        monkeypatch.setattr(engine, "_leader_follower", lambda _vid: context)
        monkeypatch.setattr(
            engine,
            "_reference",
            lambda *_args, **_kwargs: (
                18.0,
                {"raw_reference_mps": 18.0, "filtered_reference_mps": 18.0},
            ),
        )
        return engine._command_member(
            member,
            value,
            [member],
            action_family=family,
            mode_index=4 if family == "idm_p30_projected_blend" else 1,
        )

    p30 = run("speed_ceiling", 0.0)
    blend = run("idm_p30_projected_blend", 1.0)
    assert blend["projected"]["acceleration_mps2"] == pytest.approx(
        p30["projected"]["acceleration_mps2"], abs=1e-12
    )
    assert blend["projected"]["target_speed_mps"] == pytest.approx(
        p30["projected"]["target_speed_mps"], abs=1e-12
    )
    assert blend["projected_blend"]["strength"] == 1.0
    assert (
        blend["projected_blend"]["final_projection_idempotence_error_mps2"]
        == pytest.approx(0.0, abs=1e-12)
    )


def test_rear_aware_feedback_probe_blends_from_native_with_p50(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        )
    )
    member = sample("cav", "1191554636_0", 400.0, 20.0)
    context = {
        "leader_id": "leader",
        "leader_gap_m": 30.0,
        "leader_speed_mps": 15.0,
        "follower_id": None,
        "follower_gap_m": None,
        "follower_speed_mps": None,
    }
    percentiles = []

    def reference(*_args, **kwargs):
        percentiles.append(kwargs["disturbance_percentile"])
        return 18.0, {"raw_reference_mps": 18.0, "filtered_reference_mps": 18.0}

    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: context)
    monkeypatch.setattr(engine, "_reference", reference)
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "getAccel", lambda _vid: 2.0)
    monkeypatch.setattr(module.traci.vehicle, "getDecel", lambda _vid: 4.5)
    monkeypatch.setattr(module.traci.vehicle, "getFollowSpeed", lambda *_args: 21.0)
    monkeypatch.setattr(
        module.traci.vehicle, "getSpeedWithoutTraCI", lambda _vid: 20.5
    )
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)

    result = engine._command_member(
        member,
        0.0,
        [member],
        action_family="rear_aware_feedback_pulse",
        mode_index=1,
        active_age_steps=1,
    )

    feedback = result["rear_aware_feedback"]
    native = feedback["projected_native_endpoint"]["acceleration_mps2"]
    full = feedback["projected_feedback_endpoint"]["acceleration_mps2"]
    assert percentiles == ["p50"]
    assert feedback["phase"] == "probe"
    assert feedback["strength"] == pytest.approx(0.2)
    assert result["desired_acceleration_mps2"] == pytest.approx(
        native + 0.2 * (full - native)
    )
    assert result["treatment_infeasible"] is False


def test_rear_aware_probe_without_response_releases_smoothly(monkeypatch):
    lane = "1191554636_0"
    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
            decentralized_option_steps=20,
        )
    )
    member = sample("cav", lane, 400.0, 20.0)
    baseline = FeedbackResponse(
        disturbance_present=True,
        disturbance_std_mps=4.0,
        rear_gap_m=30.0,
        rear_closing_speed_mps=2.0,
        follower_acceleration_mps2=-0.5,
        local_acceleration_energy_m2ps4=2.0,
        rear_floor_feasible=True,
        projection_survived=True,
        command_delta_from_native_mps2=0.2,
    )
    engine.members[lane] = _MemberState(
        vehicle_id="cav",
        selected_step=1,
        action_family="rear_aware_feedback_pulse",
        action_value=0.0,
        mode_index=1,
        feedback_baseline=baseline,
        feedback_projection_survival_count=4,
        feedback_meaningful_command_count=4,
    )
    engine.step_index = 4
    monkeypatch.setattr(engine, "_samples", lambda: [member])
    monkeypatch.setattr(
        engine, "_decentralized_candidate_state", lambda *_args: ({}, {})
    )
    monkeypatch.setattr(
        engine,
        "_update_decentralized_members",
        lambda *_args: ([], {"cav"}, {}, []),
    )
    monkeypatch.setattr(
        engine,
        "_command_member",
        lambda *_args, **_kwargs: {
            "vehicle_id": "cav",
            "lane_id": lane,
            "rear_aware_feedback": {
                "active_age_steps": 5,
                "current_response": baseline.to_dict(),
            },
        },
    )
    monkeypatch.setattr(engine, "_release", lambda _active: [])
    projected_calls = []

    def projected_release(active, *, skip_vehicle_ids=()):
        projected_calls.append((set(active), set(skip_vehicle_ids)))
        return [{"phase": "projected_release_pending"}]

    monkeypatch.setattr(engine, "_release_projected_rear_aware", projected_release)

    diag = engine.apply_all({}, {"cav"})

    assert engine.members == {}
    assert "cav" in engine.projected_releasing
    assert projected_calls == [(set(), {"cav"})]
    assert diag["feedback_releases"][0]["reason"] == "probe_no_improving_response"
    assert diag["feedback_releases"][0]["probe_completed"] is True
    assert diag["treatment_safety_aborts"] == []
    assert diag["release"] == [{"phase": "projected_release_pending"}]


def test_rear_aware_follower_identity_change_releases_and_logs(monkeypatch):
    lane = "1191554636_0"
    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
            decentralized_option_steps=20,
        )
    )
    member = sample("cav", lane, 400.0, 20.0)
    baseline = FeedbackResponse(
        disturbance_present=True,
        disturbance_std_mps=4.0,
        rear_gap_m=30.0,
        rear_closing_speed_mps=2.0,
        follower_acceleration_mps2=-0.5,
        local_acceleration_energy_m2ps4=2.0,
        rear_floor_feasible=True,
        projection_survived=True,
        command_delta_from_native_mps2=0.2,
        follower_id="follower.old",
    )
    engine.members[lane] = _MemberState(
        vehicle_id="cav",
        selected_step=1,
        action_family="rear_aware_feedback_pulse",
        action_value=0.0,
        mode_index=1,
        feedback_baseline=baseline,
        feedback_projection_survival_count=3,
        feedback_meaningful_command_count=3,
    )
    engine.step_index = 3
    replaced = FeedbackResponse(
        disturbance_present=True,
        disturbance_std_mps=4.0,
        rear_gap_m=28.0,
        rear_closing_speed_mps=2.2,
        follower_acceleration_mps2=-0.6,
        local_acceleration_energy_m2ps4=2.0,
        rear_floor_feasible=True,
        projection_survived=True,
        command_delta_from_native_mps2=0.2,
        follower_id="follower.new",
    )
    monkeypatch.setattr(engine, "_samples", lambda: [member])
    monkeypatch.setattr(
        engine, "_decentralized_candidate_state", lambda *_args: ({}, {})
    )
    monkeypatch.setattr(
        engine,
        "_update_decentralized_members",
        lambda *_args: ([], {"cav"}, {}, []),
    )
    monkeypatch.setattr(
        engine,
        "_command_member",
        lambda *_args, **_kwargs: {
            "vehicle_id": "cav",
            "lane_id": lane,
            "rear_aware_feedback": {
                "active_age_steps": 4,
                "current_response": replaced.to_dict(),
            },
        },
    )
    monkeypatch.setattr(engine, "_release", lambda _active: [])
    monkeypatch.setattr(
        engine,
        "_release_projected_rear_aware",
        lambda _active, *, skip_vehicle_ids=(): [{"phase": "projected_release_pending"}],
    )

    diag = engine.apply_all({}, {"cav"})

    assert engine.members == {}
    assert "cav" in engine.projected_releasing
    assert diag["feedback_releases"][0]["reason"] == "follower_identity_changed"
    assert diag["feedback_releases"][0]["probe_completed"] is False
    assert diag["feedback_releases"][0]["active_steps"] == 4
    assert diag["treatment_safety_aborts"] == []
    assert diag["member_changes"][0]["reason"] == "feedback_early_release"
    assert diag["member_changes"][0]["reason_details"] == (
        "follower_identity_changed",
    )


def test_rear_aware_feedback_v2_contract_is_selected_explicitly(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        )
    )
    member = sample("cav", "1191554636_0", 400.0, 20.0)
    context = {
        "leader_id": "leader",
        "leader_gap_m": 30.0,
        "leader_speed_mps": 15.0,
        "follower_id": None,
        "follower_gap_m": None,
        "follower_speed_mps": None,
    }

    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: context)
    monkeypatch.setattr(
        engine, "_reference", lambda *_args, **_kwargs: (18.0, {})
    )
    monkeypatch.setattr(module.traci.vehicle, "getMinGap", lambda _vid: 2.5)
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "getAccel", lambda _vid: 2.0)
    monkeypatch.setattr(module.traci.vehicle, "getDecel", lambda _vid: 4.5)
    monkeypatch.setattr(module.traci.vehicle, "getFollowSpeed", lambda *_args: 21.0)
    monkeypatch.setattr(
        module.traci.vehicle, "getSpeedWithoutTraCI", lambda _vid: 20.5
    )
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    monkeypatch.setattr(module.traci.vehicle, "setSpeed", lambda *_args: None)

    result = engine._command_member(
        member,
        0.0,
        [member],
        action_family="rear_aware_feedback_pulse",
        mode_index=1,
        active_age_steps=1,
    )
    feedback = result["rear_aware_feedback"]
    assert feedback["schema"] == REAR_AWARE_FEEDBACK_V2_SCHEMA
    assert feedback["contract"]["schema"] == REAR_AWARE_FEEDBACK_V2_SCHEMA
    assert feedback["contract"]["soft_response_checks_after_probe"] is True
    assert feedback["contract"]["risk_conditioned_rear_response"] is True
    assert feedback["follower_id"] is None
