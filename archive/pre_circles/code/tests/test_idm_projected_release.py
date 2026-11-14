"""Contracts for the isolated P30 projected-release treatment."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from env.idm_residual import (
    IDMResidualActionEngine,
    IDMResidualConfig,
    ProjectionInput,
    VehicleSample,
    _MemberState,
    _ProjectedReleaseState,
    project_command,
)
from idm.decentralized_damping import (
    ACTIVATION_SPEED_CEILING_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
    mode_count,
    modes_for,
    terminal_reference_percentile,
)
from idm.projected_release import (
    PROJECTED_RELEASE_V1,
    PROJECTED_RELEASE_V2,
    PROJECTED_RELEASE_V3,
    RearFloorInput,
    handoff_converged,
    release_blend_fraction,
    solve_rear_speed_floor,
)


LANE = "1191554636_0"


def _sample(vehicle_id: str = "cav", speed: float = 20.0) -> VehicleSample:
    return VehicleSample(
        vehicle_id=vehicle_id,
        road_id="1191554636",
        lane_id=LANE,
        lane_index=0,
        lane_pos_m=500.0,
        distance_to_merge_m=500.0,
        speed_mps=speed,
        acceleration_mps2=0.0,
    )


def _no_neighbors() -> dict[str, object]:
    return {
        "leader_id": None,
        "leader_gap_m": None,
        "leader_speed_mps": None,
        "follower_id": None,
        "follower_gap_m": None,
        "follower_speed_mps": None,
    }


def test_projected_release_action_is_p30_during_active_control():
    assert mode_count(ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET) == 2
    active = modes_for(ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET)[1]
    frozen = modes_for(ACTIVATION_SPEED_CEILING_ACTION_SET)[1]
    assert active == frozen
    assert (
        terminal_reference_percentile(
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET
        )
        == "p30"
    )
    assert modes_for(ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET)[
        1
    ] == frozen
    assert (
        terminal_reference_percentile(
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET
        )
        == "p30"
    )


def test_v2_smoothstep_release_is_slower_early_and_exact_at_endpoint():
    assert release_blend_fraction(1, PROJECTED_RELEASE_V2) < (
        release_blend_fraction(1, PROJECTED_RELEASE_V1)
    )
    assert release_blend_fraction(5, PROJECTED_RELEASE_V2) == pytest.approx(0.5)
    assert release_blend_fraction(9, PROJECTED_RELEASE_V2) > 0.9
    assert release_blend_fraction(10, PROJECTED_RELEASE_V2) == pytest.approx(1.0)
    assert modes_for(ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET)[1] == (
        modes_for(ACTIVATION_SPEED_CEILING_ACTION_SET)[1]
    )


def test_v3_settles_five_steps_then_matches_native_at_step_ten():
    assert [release_blend_fraction(step, PROJECTED_RELEASE_V3) for step in range(1, 6)] == [
        0.0
    ] * 5
    assert 0.0 < release_blend_fraction(6, PROJECTED_RELEASE_V3) < 0.5
    assert release_blend_fraction(8, PROJECTED_RELEASE_V3) > 0.5
    assert release_blend_fraction(10, PROJECTED_RELEASE_V3) == pytest.approx(1.0)


def test_omitted_projected_release_floor_is_exact_legacy_projection():
    cfg = IDMResidualConfig()
    inp = ProjectionInput(
        current_speed_mps=20.0,
        desired_acceleration_mps2=-0.4,
        prior_acceleration_mps2=0.0,
        allowed_speed_mps=30.0,
        follower_gap_m=20.0,
        follower_speed_mps=21.0,
    )
    assert project_command(inp, cfg) == project_command(
        inp, cfg, minimum_target_speed_mps=None
    )


def test_release_floor_survives_jerk_and_front_conflict_still_wins():
    cfg = IDMResidualConfig()
    inp = ProjectionInput(
        current_speed_mps=20.0,
        desired_acceleration_mps2=-1.0,
        prior_acceleration_mps2=-0.5,
        allowed_speed_mps=30.0,
    )
    protected = project_command(
        inp,
        cfg,
        minimum_target_speed_mps=20.2,
        minimum_speed_reason="rear_secure_gap",
    )
    assert protected.target_speed_mps == pytest.approx(20.2)
    assert "rear_secure_gap" in protected.reasons

    conflict = project_command(
        ProjectionInput(
            **{
                **asdict(inp),
                "leader_gap_m": 0.0,
                "leader_speed_mps": 19.0,
            }
        ),
        cfg,
        minimum_target_speed_mps=20.2,
        minimum_speed_reason="rear_secure_gap",
    )
    assert conflict.constraint_conflict is True
    assert conflict.target_speed_mps == pytest.approx(19.0)
    assert "front_follower_conflict" in conflict.reasons


def _rear_floor(*, gap: float, follower_speed: float):
    inp = RearFloorInput(
        current_ego_speed_mps=20.0,
        follower_speed_mps=follower_speed,
        follower_gap_m=gap,
        minimum_next_speed_mps=19.0,
        maximum_next_speed_mps=20.5,
        dt_s=1.0,
        rear_ttc_floor_s=4.0,
        follower_decel_mps2=4.5,
    )

    def response(next_speed: float) -> tuple[float, float]:
        predicted = follower_speed - 0.5
        secure = max(0.0, follower_speed - next_speed) * 2.0 + 2.0
        return predicted, secure

    return solve_rear_speed_floor(inp, follower_response_at_speed=response)


def test_rear_floor_tightens_for_faster_follower_and_smaller_gap():
    base = _rear_floor(gap=12.0, follower_speed=21.0)
    faster = _rear_floor(gap=12.0, follower_speed=22.0)
    smaller = _rear_floor(gap=10.0, follower_speed=22.0)
    assert base.feasible_within_upper_bound
    assert faster.minimum_next_speed_mps > base.minimum_next_speed_mps
    assert smaller.minimum_next_speed_mps >= faster.minimum_next_speed_mps
    assert not smaller.feasible_within_upper_bound


def test_missing_follower_response_requests_maximum_and_fails_feasibility():
    result = solve_rear_speed_floor(
        RearFloorInput(
            current_ego_speed_mps=20.0,
            follower_speed_mps=22.0,
            follower_gap_m=10.0,
            minimum_next_speed_mps=19.0,
            maximum_next_speed_mps=20.5,
            dt_s=1.0,
            rear_ttc_floor_s=4.0,
            follower_decel_mps2=4.5,
        )
    )
    assert result.minimum_next_speed_mps == pytest.approx(20.5)
    assert not result.feasible_within_upper_bound
    assert "sumo_follower_response_unavailable" in result.reasons


def test_raw_ttc_floor_above_upper_bound_cannot_be_reported_feasible():
    result = solve_rear_speed_floor(
        RearFloorInput(
            current_ego_speed_mps=10.0,
            follower_speed_mps=30.0,
            follower_gap_m=4.0,
            minimum_next_speed_mps=9.0,
            maximum_next_speed_mps=10.5,
            dt_s=1.0,
            rear_ttc_floor_s=4.0,
            follower_decel_mps2=4.5,
        ),
        follower_response_at_speed=lambda speed: (29.0, 1.0),
    )
    assert result.ttc_floor_mps == pytest.approx(29.0)
    assert result.minimum_next_speed_mps == pytest.approx(10.5)
    assert not result.feasible_within_upper_bound
    assert "rear_raw_floor_above_upper_bound" in result.reasons


def test_handoff_contract_compares_projected_and_native_commands():
    assert handoff_converged(
        projected_target_speed_mps=20.4,
        native_target_speed_mps=20.5,
        projected_acceleration_mps2=0.4,
        native_acceleration_mps2=0.5,
        constraint_conflict=False,
        rear_feasible=True,
    )
    assert not handoff_converged(
        projected_target_speed_mps=20.5,
        native_target_speed_mps=21.5,
        projected_acceleration_mps2=0.5,
        native_acceleration_mps2=1.5,
        constraint_conflict=False,
        rear_feasible=True,
    )


def test_only_normal_option_completion_enters_projected_release(monkeypatch):
    cfg = IDMResidualConfig(
        decentralized_damping_enabled=True,
        decentralized_action_set=(
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET
        ),
        decentralized_option_steps=20,
    )
    engine = IDMResidualActionEngine(cfg)
    member = _MemberState(
        "cav",
        selected_step=1,
        action_family="speed_ceiling",
        action_value=0.0,
        mode_index=1,
    )
    engine.members[LANE] = member
    engine.step_index = 21
    monkeypatch.setattr(
        engine,
        "_candidate_admission",
        lambda *_args, **_kwargs: {"admissible": True, "reasons": ()},
    )
    engine._update_decentralized_members({}, {}, {}, {"cav": _sample()})
    assert "cav" in engine.projected_releasing
    assert "cav" not in engine.releasing

    unsafe = IDMResidualActionEngine(cfg)
    unsafe.members[LANE] = member
    unsafe.step_index = 5
    unsafe._update_decentralized_members({}, {}, {}, {})
    assert "cav" not in unsafe.projected_releasing
    assert "cav" in unsafe.releasing


def test_projected_release_runs_fixed_ten_commands_then_hands_off(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=(
                ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET
            ),
        )
    )
    state = _ProjectedReleaseState(
        lane_id=LANE,
        selected_step=1,
        initial_acceleration_mps2=0.5,
        prior_acceleration_mps2=0.5,
        age_steps=8,
    )
    engine.projected_releasing["cav"] = state
    monkeypatch.setattr(engine, "_leader_follower", lambda _vid: _no_neighbors())
    monkeypatch.setattr(module.traci.vehicle, "getIDList", lambda: ("cav",))
    speeds = iter((20.0, 20.5))
    natives = iter((20.5, 21.0))
    monkeypatch.setattr(module.traci.vehicle, "getSpeed", lambda _vid: next(speeds))
    monkeypatch.setattr(
        module.traci.vehicle, "getSpeedWithoutTraCI", lambda _vid: next(natives)
    )
    monkeypatch.setattr(module.traci.vehicle, "getAllowedSpeed", lambda _vid: 30.0)
    monkeypatch.setattr(module.traci.vehicle, "getAccel", lambda _vid: 0.5)
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    calls: list[float] = []
    monkeypatch.setattr(
        module.traci.vehicle, "setSpeed", lambda _vid, speed: calls.append(speed)
    )

    first = engine._release_projected_rear_aware(set())
    second = engine._release_projected_rear_aware(set())
    assert calls == [pytest.approx(20.5), pytest.approx(21.0)]
    assert first[0]["age_steps"] == 9
    assert second[0]["age_steps"] == 10
    assert second[0]["handoff_ready_next_step"] is True
    assert "cav" in engine.projected_releasing

    handoff = engine._release_projected_rear_aware(set())
    assert calls[-1] == -1
    assert handoff[0]["handoff_contract_pass"] is True
    assert handoff[0]["unmatched_handoff"] is False
    assert "cav" not in engine.projected_releasing


def test_unmatched_budget_handoff_is_explicit_hard_failure(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=(
                ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET
            ),
        )
    )
    engine.projected_releasing["cav"] = _ProjectedReleaseState(
        lane_id=LANE,
        selected_step=1,
        initial_acceleration_mps2=0.5,
        prior_acceleration_mps2=0.5,
        age_steps=PROJECTED_RELEASE_V1.nominal_budget_steps,
        converged_steps=1,
    )
    monkeypatch.setattr(module.traci.vehicle, "getIDList", lambda: ("cav",))
    monkeypatch.setattr(module.traci.vehicle, "setSpeedMode", lambda *_args: None)
    calls: list[float] = []
    monkeypatch.setattr(
        module.traci.vehicle, "setSpeed", lambda _vid, speed: calls.append(speed)
    )
    row = engine._release_projected_rear_aware(set())[0]
    assert calls == [-1]
    assert row["unmatched_handoff"] is True
    assert row["handoff_contract_pass"] is False


def test_apply_dispatch_preserves_p30_release_and_isolates_new_path(monkeypatch):
    def configure(engine: IDMResidualActionEngine):
        monkeypatch.setattr(engine, "_samples", lambda: [])
        monkeypatch.setattr(
            engine,
            "_decentralized_candidate_state",
            lambda *_args: ({}, {}),
        )
        monkeypatch.setattr(
            engine,
            "_update_decentralized_members",
            lambda *_args: ([], set(), {}, []),
        )

    frozen = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=ACTIVATION_SPEED_CEILING_ACTION_SET,
        )
    )
    configure(frozen)
    monkeypatch.setattr(frozen, "_release", lambda _active: [{"path": "legacy"}])
    monkeypatch.setattr(
        frozen,
        "_release_projected_rear_aware",
        lambda _active: pytest.fail("frozen P30 called projected release"),
    )
    assert frozen.apply_all({}, set())["release"] == [{"path": "legacy"}]

    new = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_action_set=(
                ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET
            ),
        )
    )
    configure(new)
    monkeypatch.setattr(new, "_release", lambda _active: [])
    monkeypatch.setattr(
        new,
        "_release_projected_rear_aware",
        lambda _active: [{"path": "projected"}],
    )
    assert new.apply_all({}, set())["release"] == [{"path": "projected"}]


def test_reset_clears_projected_release_state():
    engine = IDMResidualActionEngine()
    engine.projected_releasing["cav"] = _ProjectedReleaseState(
        lane_id=LANE,
        selected_step=1,
        initial_acceleration_mps2=0.0,
        prior_acceleration_mps2=0.0,
    )
    engine.projected_release_cooldown_until[LANE] = 99
    engine.reset()
    assert engine.projected_releasing == {}
    assert engine.projected_release_cooldown_until == {}
