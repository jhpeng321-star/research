"""State + plan schema contracts (api_version 1.0)."""
from __future__ import annotations

import numpy as np
import pytest

from wm_api.errors import (PlanSchemaError, StateSchemaError,
                           UnknownActionError)
from wm_api.plans import (ACTION_ALIASES, PLANNABLE_ACTIONS, canon_action,
                          resolve_plan)
from wm_api.state import (K_MAX, StateSnapshot, counters_from_ticks,
                          state_from_physics_tick)

from _tiny import default_plans, tiny_state, wall_plan


def test_state_schema_round_trip():
    st = tiny_state()
    st2 = StateSnapshot.from_json(st.to_json())
    assert st2.ids == st.ids
    assert np.array_equal(st2.lane_phys, st.lane_phys)
    assert np.array_equal(st2.is_cav, st.is_cav)
    assert st2.t0_s == pytest.approx(st.t0_s)
    assert st2.n_model_vehicles == st.n_model_vehicles == 6


def test_state_schema_validation():
    with pytest.raises(StateSchemaError):
        StateSnapshot(
            ids=["a", "b"], lane_phys=np.array([1, 0, 1]),
            s_down_m=np.zeros(2), v_mps=np.zeros(2), a_mps2=np.zeros(2),
            length_m=np.zeros(2), is_cav=np.array([True, False]))
    bad = tiny_state()
    bad.lane_phys[0] = 9
    with pytest.raises(StateSchemaError):
        bad.__post_init__()
    with pytest.raises(StateSchemaError):
        StateSnapshot(
            ids=["a"], lane_phys=np.array([1]), s_down_m=np.zeros(1),
            v_mps=np.zeros(1), a_mps2=np.zeros(1), length_m=np.zeros(1),
            is_cav=np.array([True]), entrance_counters=np.zeros((2, 3)))


def test_physics_tick_converter_frozen_lane_table():
    tick = {
        "time_s": 540.0,
        "vehicles": {
            "cav0": {"lane_id": "1191554636_0", "lane_position_m": 100.0,
                     "speed_mps": 20.0, "acceleration_mps2": 0.5,
                     "road_id": "1191554636"},
            "hdv0": {"lane_id": "11553048#1_0", "lane_position_m": 10.0,
                     "speed_mps": 15.0, "acceleration_mps2": 0.0,
                     "road_id": "11553048#1"},
            "junc0": {"lane_id": ":junction_0_0", "lane_position_m": 3.0,
                      "speed_mps": 5.0, "road_id": ":junction_0_0"},
        },
        "departed_ids": [],
        "cav_ids": ["cav0"],
    }
    st = state_from_physics_tick(tick)
    assert st.ids == ["cav0", "hdv0"]
    assert list(st.lane_phys) == [1, 2]
    # main_right entry lane: s = 0 + 100, offset 2644.22
    assert st.s_down_m[0] == pytest.approx(100.0 - 2644.22)
    # accel lane: s = 0 + 10, offset 378.66
    assert st.s_down_m[1] == pytest.approx(10.0 - 378.66)
    assert list(st.is_cav) == [True, False]
    assert st.excluded == {"junc0": "junction_or_missing_lane"}
    assert st.t0_s == pytest.approx(540.0)


def test_counters_from_ticks_bins():
    ticks = [
        {"time_s": 240.0, "vehicles": {"a": {"road_id": "1191554636"}},
         "departed_ids": ["a"]},
        {"time_s": 245.0, "vehicles": {"b": {"road_id": "11553048#1"}},
         "departed_ids": ["b"]},
        {"time_s": 100.0, "vehicles": {"old": {"road_id": "1191554636"}},
         "departed_ids": ["old"]},
        {"time_s": 541.0, "vehicles": {}, "departed_ids": []},
    ]
    c = counters_from_ticks(ticks, t_now=540.0)
    assert c.shape == (2, 30)
    assert c[1, 29] == 1.0            # age 295 s -> bin 29
    assert c.sum() == 1.0             # 300 s old is outside (t-300, t];
                                      # the t=541 departure is in the future
    assert counters_from_ticks(ticks, t_now=245.0)[0, 0] == 1.0


def test_canon_action_aliases_and_plannable():
    assert canon_action("PI-External-U") == "pi"
    assert canon_action("wall-v2") == "wall"
    assert canon_action("damping") == "damping"
    assert set(PLANNABLE_ACTIONS) == {"native", "damping", "pi", "wall"}


def test_plan_aliases_mirror_evalh_descriptor_convention():
    from evalh.idea_a import ACTION_ALIASES as EVALH_ALIASES

    assert ACTION_ALIASES == EVALH_ALIASES


def test_plan_schedule_segments_distinguish_durations():
    st = tiny_state()
    short = resolve_plan(
        {"actions": {"cav0": [{"time_s": 0.0, "action": "damping"},
                              {"time_s": 60.0, "action": "native"}]}},
        st, n_segments=10)
    whole = resolve_plan({"actions": {"cav0": "damping"}}, st, n_segments=10)
    assert short.segment_tokens[0]["cav0"] == "damping"
    assert short.segment_tokens[2]["cav0"] == "native"
    assert all(t == "damping" for t in whole.segment_tokens[0].values())
    assert short.describe() != whole.describe()
    ref = resolve_plan(default_plans()[0], st, n_segments=10)
    assert all(t == "native" for seg in ref.segment_tokens
               for t in seg.values())


def test_plan_wall_membership_and_gates():
    st = tiny_state()
    wp = resolve_plan(wall_plan(), st, n_segments=10)
    assert wp.wall["members"] == ["cav0", "cav1"]
    for k in range(4):                 # hold 120 s = 4 segments
        assert wp.segment_tokens[k]["cav0"] == "wall"
        assert wp.segment_tokens[k]["cav1"] == "wall"
    assert wp.segment_tokens[4]["cav0"] == "native"
    assert wp.wall["speed_mps"] == pytest.approx(8.0)
    assert "pair_gate" in wp.gates and "dmin_refusal" in wp.gates
    assert "hold_cap_s" in wp.gates
    member_s = [float(st.s_down_m[st.ids.index(cid)])
                for cid in ("cav0", "cav1")]
    lead = max(member_s) + 2644.2
    assert wp.gates["lead_s_road_m"] == pytest.approx(round(lead, 6))
    # the tiny state's lead is 104 m from the merge -> D_min = 560 m refuses
    assert wp.gates["pair_gate"]["feasible"] is False
    assert wp.gates["pair_gate"]["refusal"]["reason"] == "dmin_unreachable"
    assert wp.gates["dmin_refusal"] is not None
    assert wp.gates["hold_cap_s"] < wp.wall["hold_s"]


def test_wall_pair_gate_feasible_when_far_enough():
    st = tiny_state()
    st.s_down_m = np.array([-640.0, -600.0, -300.0, -200.0, -100.0, -50.0])
    st.lane_phys = np.array([1, 0, 1, 1, 1, 1])
    wp = resolve_plan(wall_plan(), st, n_segments=10)
    assert wp.gates["pair_gate"]["feasible"] is True
    assert wp.gates["pair_gate"]["refusal"] is None
    assert wp.gates["dmin_refusal"] is None
    # lead 600 m from the merge at 8 m/s: hold cap = min(120, 600/8) = 75 s
    assert wp.gates["hold_cap_s"] == pytest.approx(75.0)


def test_plan_refusals():
    st = tiny_state()
    with pytest.raises(UnknownActionError):
        resolve_plan({"actions": {"cav0": "teleport"}}, st, 10)
    with pytest.raises(UnknownActionError):
        resolve_plan({"actions": {"cav0": "unknown"}}, st, 10)
    with pytest.raises(PlanSchemaError):
        resolve_plan({"actions": {"hdv0": "damping"}}, st, 10)
    with pytest.raises(PlanSchemaError):
        resolve_plan({"actions": {"ghost": "damping"}}, st, 10)
    with pytest.raises(PlanSchemaError):
        resolve_plan(
            {"actions": {"cav0": "native"},
             "wall": {"wall_id": 1, "speed_mps": 8.0, "hold_s": 60.0,
                      "members": ["cav0", "cav0"]}}, st, 10)
    with pytest.raises(PlanSchemaError):
        resolve_plan({"actions": {}, "wall": {"speed_mps": 0.0,
                                              "hold_s": 60.0,
                                              "members": ["cav0", "cav1"]}},
                     st, 10)
    sched = [{"vehicle_id": "cav0", "time_s": 0.0, "action": "damping"}]
    with pytest.raises(PlanSchemaError):
        resolve_plan({"actions": {"cav0": sched},
                      "wall": {"wall_id": 1, "speed_mps": 8.0,
                               "hold_s": 60.0, "members": ["cav0", "cav1"]}},
                     st, 10)


def test_k_max_frozen():
    assert K_MAX == 128
