"""Contracts for the decentralized structured IDM controller."""

import numpy as np
import pytest

from env.idm_residual import (
    Disturbance,
    IDMResidualActionEngine,
    IDMResidualConfig,
    VehicleSample,
)
from idm.decentralized_damping import (
    ACCELERATION_CAP_ACTION_SET,
    ACTIVATION_SPEED_CEILING_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P50_ACTION_SET,
    Bid,
    Candidate,
    MODE_BY_INDEX,
    action_mask,
    contend,
    decode_mode,
    enumerate_candidates,
    terminal_reference_percentile,
)


LANE0 = "1191554636_0"
LANE1 = "1191554636_1"


def _sample(vehicle_id, lane, distance, speed=20.0):
    return VehicleSample(
        vehicle_id=vehicle_id,
        road_id="1191554636",
        lane_id=lane,
        lane_index=int(lane[-1]),
        lane_pos_m=1000.0 - distance,
        distance_to_merge_m=distance,
        speed_mps=speed,
        acceleration_mps2=0.0,
    )


def _disturbance(lane, distance=100.0):
    return Disturbance(lane, distance, 10.0, 17.0, 24.0, 5.0, 8)


def test_candidate_enumeration_exposes_ranked_fallbacks():
    rows = [_sample("second", LANE0, 500.0), _sample("first", LANE0, 400.0)]
    candidates = enumerate_candidates(
        rows,
        {"first", "second"},
        {LANE0: _disturbance(LANE0)},
        minimum_runway_m=150.0,
        maximum_runway_m=800.0,
    )
    assert [(row.vehicle_id, row.rank) for row in candidates[LANE0]] == [
        ("first", 1),
        ("second", 2),
    ]


def test_contention_uses_next_local_bid_and_enforces_spacing():
    first = Candidate("first", LANE0, 400.0, 100.0, 300.0, 1)
    second = Candidate("second", LANE0, 500.0, 100.0, 400.0, 2)
    nearby = Candidate("nearby", LANE1, 520.0, 100.0, 420.0, 1)
    winners, rejected = contend(
        [
            Bid(second, MODE_BY_INDEX[2]),
            Bid(nearby, MODE_BY_INDEX[3]),
        ],
        minimum_spacing_m=150.0,
    )
    # Candidate one abstained and never entered contention; candidate two is
    # therefore the lane-local fallback. Its broadcast wins over the nearby bid.
    assert winners[LANE0].candidate.vehicle_id == "second"
    assert LANE1 not in winners
    assert rejected["nearby"] == "spacing_conflict:second"
    assert first.vehicle_id not in rejected


def test_modes_and_masks_fail_safe_to_abstain():
    assert decode_mode(np.array([3])).name == "strong"
    assert decode_mode(99).name == "abstain"
    assert action_mask(eligible=False) == (1, 0, 0, 0, 0)
    assert action_mask(eligible=True) == (1, 1, 1, 1, 1)
    assert action_mask(eligible=True, latched_mode=4) == (0, 0, 0, 0, 1)
    assert decode_mode(3, ACCELERATION_CAP_ACTION_SET).name == "coast_cap"
    assert action_mask(
        eligible=False, action_set=ACCELERATION_CAP_ACTION_SET
    ) == (1, 0, 0, 0)
    assert decode_mode(1, ACTIVATION_SPEED_CEILING_ACTION_SET).family == "speed_ceiling"
    assert action_mask(
        eligible=True, action_set=ACTIVATION_SPEED_CEILING_ACTION_SET
    ) == (1, 1)
    assert (
        decode_mode(1, ACTIVATION_SPEED_CEILING_P50_ACTION_SET).family
        == "speed_ceiling_p50"
    )
    assert action_mask(
        eligible=False, action_set=ACTIVATION_SPEED_CEILING_P50_ACTION_SET
    ) == (1, 0)
    assert action_mask(
        eligible=True, action_set=ACTIVATION_SPEED_CEILING_P50_ACTION_SET
    ) == (1, 1)
    assert terminal_reference_percentile(ACTIVATION_SPEED_CEILING_ACTION_SET) == "p30"
    assert terminal_reference_percentile(ACTIVATION_SPEED_CEILING_P50_ACTION_SET) == "p50"


def test_engine_selects_safe_fallback_and_latches_its_mode(monkeypatch):
    cfg = IDMResidualConfig(
        decentralized_damping_enabled=True,
        decentralized_option_steps=20,
        disturbance_min_vehicles=4,
    )
    engine = IDMResidualActionEngine(cfg)
    rows = [
        _sample("h0", LANE0, 80.0, 10.0),
        _sample("h1", LANE0, 100.0, 12.0),
        _sample("h2", LANE0, 120.0, 22.0),
        _sample("h3", LANE0, 140.0, 24.0),
        _sample("unsafe_nearest", LANE0, 400.0, 20.0),
        _sample("safe_fallback", LANE0, 500.0, 20.0),
        _sample("out_of_band", LANE1, 500.0, 20.0),
    ]
    monkeypatch.setattr(engine, "_samples", lambda: rows)

    def admission(vehicle_id, candidate=None):
        safe = vehicle_id != "unsafe_nearest"
        return {
            "admissible": safe,
            "reasons": () if safe else ("front_ttc",),
            "predicted_dose_s": None if candidate is None else 20.0,
        }

    monkeypatch.setattr(engine, "_candidate_admission", admission)
    monkeypatch.setattr(engine, "_release", lambda _active: [])
    commands = []

    def command(member, value, _rows, **kwargs):
        commands.append((member.vehicle_id, value, kwargs))
        return {
            "vehicle_id": member.vehicle_id,
            "residual_mps2": 0.0,
            "mode_index": kwargs["mode_index"],
        }

    monkeypatch.setattr(engine, "_command_member", command)
    diag = engine.apply_all(
        {"unsafe_nearest": 3, "safe_fallback": 2, "out_of_band": 4},
        {"unsafe_nearest", "safe_fallback", "out_of_band"},
    )
    assert diag["active_members"] == {LANE0: "safe_fallback"}
    assert len(commands) == 1
    vehicle_id, value, kwargs = commands[0]
    assert vehicle_id == "safe_fallback"
    assert value == pytest.approx(1.0)
    assert kwargs["action_family"] == "gain"
    assert kwargs["mode_index"] == 2
    assert kwargs["disturbance"].lane_id == LANE0
    assert "out_of_band" not in {row["vehicle_id"] for row in diag["commands"]}
    assert next(
        row for row in diag["candidate_admissions"] if row["vehicle_id"] == "unsafe_nearest"
    )["reasons"] == ("front_ttc",)


def test_secure_gap_admission_rejects_front_and_rear_violations(monkeypatch):
    import env.idm_residual as module

    engine = IDMResidualActionEngine(
        IDMResidualConfig(decentralized_damping_enabled=True)
    )
    monkeypatch.setattr(
        engine,
        "_leader_follower",
        lambda _vehicle_id: {
            "leader_id": "leader",
            "leader_gap_m": 12.0,
            "leader_speed_mps": 20.0,
            "follower_id": "follower",
            "follower_gap_m": 10.0,
            "follower_speed_mps": 20.0,
        },
    )
    monkeypatch.setattr(module.traci.vehicle, "getSpeed", lambda _vehicle_id: 20.0)
    monkeypatch.setattr(module.traci.vehicle, "getDecel", lambda _vehicle_id: 4.5)
    monkeypatch.setattr(module.traci.vehicle, "getSecureGap", lambda *_args: 15.0)
    result = engine._candidate_admission("candidate")
    assert not result["admissible"]
    assert result["reasons"] == ("front_secure_gap", "rear_secure_gap")


def test_option_boundary_reopens_all_local_actions(monkeypatch):
    from env.idm_residual import _MemberState

    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_option_steps=20,
            disturbance_min_vehicles=4,
        )
    )
    rows = [
        _sample("h0", LANE0, 80.0, 10.0),
        _sample("h1", LANE0, 100.0, 12.0),
        _sample("h2", LANE0, 120.0, 22.0),
        _sample("h3", LANE0, 140.0, 24.0),
        _sample("candidate", LANE0, 500.0, 20.0),
    ]
    monkeypatch.setattr(engine, "_samples", lambda: rows)
    monkeypatch.setattr(
        engine,
        "_candidate_admission",
        lambda _vehicle_id, candidate=None: {
            "admissible": True,
            "reasons": (),
            "predicted_dose_s": 20.0 if candidate is not None else None,
        },
    )
    engine.members[LANE0] = _MemberState(
        "candidate", selected_step=1, action_family="gain", action_value=1.25, mode_index=3
    )
    engine.step_index = 19
    _, held_masks, _ = engine.preview_decentralized_state({"candidate"})
    assert held_masks["candidate"] == (0, 0, 0, 1, 0)
    engine.step_index = 20
    _, boundary_masks, _ = engine.preview_decentralized_state({"candidate"})
    assert boundary_masks["candidate"] == (1, 1, 1, 1, 1)


def test_latched_lane_masks_backup_candidates_until_option_boundary(monkeypatch):
    from env.idm_residual import _MemberState

    engine = IDMResidualActionEngine(
        IDMResidualConfig(
            decentralized_damping_enabled=True,
            decentralized_option_steps=20,
            disturbance_min_vehicles=4,
        )
    )
    rows = [
        _sample("h0", LANE0, 80.0, 10.0),
        _sample("h1", LANE0, 100.0, 12.0),
        _sample("h2", LANE0, 120.0, 22.0),
        _sample("h3", LANE0, 140.0, 24.0),
        _sample("member", LANE0, 500.0, 20.0),
        _sample("backup", LANE0, 550.0, 21.0),
    ]
    monkeypatch.setattr(engine, "_samples", lambda: rows)
    monkeypatch.setattr(
        engine,
        "_candidate_admission",
        lambda _vehicle_id, candidate=None: {
            "admissible": True,
            "reasons": (),
            "predicted_dose_s": 20.0 if candidate is not None else None,
        },
    )
    engine.members[LANE0] = _MemberState(
        "member", selected_step=1, action_family="gain", action_value=1.25, mode_index=3
    )
    engine.step_index = 10
    _, masks, _ = engine.preview_decentralized_state({"member", "backup"})
    assert masks["member"] == (0, 0, 0, 1, 0)
    assert masks["backup"] == (1, 0, 0, 0, 0)
