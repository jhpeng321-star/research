"""§1/§3 constants and frozen dataclass behaviour."""
import dataclasses

import pytest

from predict import contract as C
from predict import (
    ACTIONS,
    ACTION_DAMPING,
    ACTION_NATIVE,
    ACTION_PI_EXTERNAL_U,
    ACTION_WALL,
    ABORT_ASSEMBLY_TIMEOUT,
    ABORT_NONE,
    ABORT_REASONS,
    ABORT_SAFETY,
    CONTRACT_VERSION,
    DEFAULT_HORIZON_S,
    ENTRANCE_MAINLINE,
    ENTRANCE_RAMP,
    LANE_ACCEL,
    LANE_IDS,
    LANE_MAIN_LEFT,
    LANE_MAIN_RIGHT,
    PER_CAV_ACTIONS,
    PHASES,
    PHASE_ASSEMBLE,
    PHASE_HOLD,
    PHASE_RELEASE,
    PLAN_VERSION,
    Arrival,
    ControllerStateEntry,
    Meta,
    PlanCandidate,
    SampleMeta,
    StateSample,
    WallSpec,
)


def test_constants_values():
    assert CONTRACT_VERSION == "wd01_predict/1"
    assert PLAN_VERSION == 1
    assert DEFAULT_HORIZON_S == 600.0
    assert (ABORT_NONE, ABORT_SAFETY, ABORT_ASSEMBLY_TIMEOUT) == (0, 1, 2)
    assert ABORT_REASONS == (0, 1, 2)
    assert (LANE_MAIN_LEFT, LANE_MAIN_RIGHT, LANE_ACCEL) == (0, 1, 2)
    assert LANE_IDS == (0, 1, 2)
    assert ACTIONS == ("native", "damping", "pi-external-u", "wall")
    assert PER_CAV_ACTIONS == ("native", "damping", "pi-external-u")
    assert ACTION_WALL not in PER_CAV_ACTIONS
    assert PHASES == ("assemble", "hold", "release")
    assert {PHASE_ASSEMBLE, PHASE_HOLD, PHASE_RELEASE} == set(PHASES)
    assert {ENTRANCE_MAINLINE, ENTRANCE_RAMP} == {"mainline", "ramp"}


def test_dataclasses_are_frozen():
    w = WallSpec("W1", ("a",), 5.0, PHASE_ASSEMBLE, None, 10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.speed_mps = 6.0
    p = PlanCandidate(1, {"a": "native"}, (w,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.per_cav = {}
    e = ControllerStateEntry("native")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.locked = True
    s = StateSample(object(), (), SampleMeta("s0", 0.0, True, 1.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.hybrid = None


def test_wall_spec_from_raw_roundtrip():
    raw = {
        "wall_id": "W1",
        "members": ["a", "b"],
        "speed_mps": 5.0,
        "phase": "assemble",
        "release_at_s": None,
        "hold_s": 30.0,
    }
    w = WallSpec.from_raw(raw)
    assert w == WallSpec("W1", ("a", "b"), 5.0, "assemble", None, 30.0)


def test_plan_candidate_from_raw():
    raw = {
        "plan_version": 1,
        "per_cav": {"a": "native", "b": "damping"},
        "walls": [
            {
                "wall_id": "W1",
                "members": ["x"],
                "speed_mps": 5.0,
                "phase": "hold",
                "release_at_s": None,
                "hold_s": 0.0,
            }
        ],
    }
    p = PlanCandidate.from_raw(raw)
    assert p.plan_version == 1
    assert p.per_cav == {"a": "native", "b": "damping"}
    assert p.walls[0].members == ("x",)


def test_meta_and_state_sample_from_raw():
    meta_raw = {
        "contract_version": CONTRACT_VERSION,
        "model_id": "m",
        "weights_sha256": ["w0"],
        "sample_ids": ["s0"],
        "member_ids": ["m0"],
        "part_a_horizon_s": 60.0,
        "safety_window_s": 60.0,
        "horizon_s": 600.0,
        "native_kept_walls": ["W1"],
        "native_unlocked_available": False,
        "device": "cpu",
        "wall_clock_s": 0.0,
    }
    meta = Meta.from_raw(meta_raw)
    assert meta.contract_version == CONTRACT_VERSION
    s = StateSample.from_raw(
        {
            "hybrid": {"x": 1},
            "arrivals": [{"t_s": 1.0, "entrance": "ramp", "speed_mps": 20.0,
                          "is_cav": True, "vehicle_type": "cav"}],
            "meta": {"sample_id": "s0", "time_s": 5.0, "merge_observed": True,
                     "coverage": 0.9},
        }
    )
    assert isinstance(s.arrivals[0], Arrival)
    assert s.meta == SampleMeta("s0", 5.0, True, 0.9)


def test_from_raw_rejects_unknown_keys():
    with pytest.raises(TypeError):
        WallSpec.from_raw(
            {"wall_id": "W", "members": ["a"], "speed_mps": 5.0, "phase": "hold",
             "release_at_s": None, "hold_s": 0.0, "schedule": []}  # reserved, not v1
        )
    with pytest.raises(TypeError):
        SampleMeta.from_raw(
            {"sample_id": "s", "time_s": 0.0, "merge_observed": True, "coverage": 1.0,
             "extra": 1}
        )
