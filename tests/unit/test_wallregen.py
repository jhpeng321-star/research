"""WALLREGEN unit tests (no SUMO): seed block, WM-59 v2.1 request rule, arm runner, converter."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from traffic_shepherd import seeds as S
from traffic_shepherd.c4 import wallregen as W
from traffic_shepherd.c4 import wallregen_states as WS

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Frame:
    vehicle_id: str
    lane_id: str
    lane_index: int
    position_m: float
    distance_to_merge_m: float
    speed_mps: float


def frame(vid, lane, pos, speed=20.0, edge="1191554636"):
    return Frame(vid, f"{edge}_{lane}", lane, pos, W.UPSTREAM_EDGE_LENGTH_M - pos, speed)


# ------------------------------------------------------------------ seed block
def test_seed_block_disjoint_from_every_registered_table():
    block = set(S.C4_WALLREGEN_EVAL_SEEDS)
    assert block == set(range(962000, 962200))
    for name, value in vars(S).items():
        if name.startswith("C4_WALLREGEN") or name.startswith("_"):
            continue
        if isinstance(value, dict):
            ints = {k for k in value if isinstance(k, int)}
        elif isinstance(value, (set, frozenset, tuple, list, range)):
            ints = {x for x in value if isinstance(x, int)}
        else:
            continue
        assert not (ints & block), name


def test_seed_guard_admits_block_only_with_registered_loadings():
    from traffic_shepherd.c4.seeds import SeedRefusal, check_case
    check_case(962000, 2700)
    check_case(962199, 2600)
    check_case(962100, 1800)
    with pytest.raises(ValueError):
        check_case(962000, 1025)
    with pytest.raises(SeedRefusal):
        check_case(962200, 2700)


# ------------------------------------------------------------------ law + rule pins
def test_wall_law_is_the_pinned_v21_closure_file():
    data = (ROOT / "traffic_shepherd" / "wall_group.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == \
        "09ba6cbcf1c0b21f8449c96b44d52541fe6a7b16fe704c38081412059f0a11bf"  # export copy: governance comments removed
    from traffic_shepherd.wall_group import WALL_LAW_VERSION
    assert WALL_LAW_VERSION == W.WALL_LAW == "wall-v2.1-closure"


def test_pilot_constants():
    assert W.ARM_SPEED_FACTORS == {"w-lo": 0.60, "w-mid": 0.85}
    assert W.TIMING_OFFSET_S == {"early": 0.0, "late": 60.0}
    assert (W.REQUEST_WINDOW_S, W.SPEED_FLOOR_MPS, W.REQUEST_SPAN_CAP_M) == (120.0, 2.5, 75.0)
    assert W.HOLD_CHOICES_S == (60.0, 120.0, 240.0)
    assert (W.ROAD_LEFT_REQUEST_S, W.ROAD_END_RELEASE_S) == (80.0, 10.0)
    assert W.MEMBER_EDGES == frozenset({"1191554636"})


def test_choose_pair_minimal_span_then_road_left():
    frames = {
        "a": frame("a", 0, 100.0), "b": frame("b", 1, 130.0),      # span 30
        "c": frame("c", 0, 500.0), "d": frame("d", 1, 505.0),      # span 5 -> chosen
        "e": frame("e", 1, 600.0, edge="1487606178"),              # wrong edge
    }
    pair = W.choose_pair(frames, 0.85)
    assert pair["members"] == ("c", "d") and pair["span_m"] == pytest.approx(5.0)
    assert W.wall_speed_for(0.6, {"x": 10.0, "y": 20.0}) == pytest.approx(9.0)
    assert W.wall_speed_for(0.6, {"x": 1.0, "y": 1.0}) == W.SPEED_FLOOR_MPS


def test_pair_refused_same_lane_wide_span_or_short_road():
    assert W.choose_pair({"a": frame("a", 0, 10.0), "b": frame("b", 0, 12.0)}, 0.6) is None
    assert W.choose_pair({"a": frame("a", 0, 10.0), "b": frame("b", 1, 90.0)}, 0.6) is None
    # road left = 2644.2 - 2400 + 553.3 = 797.5 m < 0.85 * 20 * 80 = 1360 m
    assert W.choose_pair({"a": frame("a", 0, 2400.0), "b": frame("b", 1, 2401.0)}, 0.85) is None


def test_arm_runner_request_assembly_hold_release_and_not_feasible():
    run = W.WallArmRunner(arm="w-lo-early", run_name="w1__w-lo-early", t_branch_s=300.0,
                          hold_s=60.0)
    fr = {"a": frame("a", 0, 100.0, 20.0), "b": frame("b", 1, 101.0, 20.0)}
    cmds = run.tick(fr, 300.0)
    assert run.state == "requested" and set(cmds) == {"a", "b"}
    assert run.wall_speed_mps == pytest.approx(12.0)
    # members at wall speed and aligned -> assembled
    fr = {"a": frame("a", 0, 120.0, 12.0), "b": frame("b", 1, 121.0, 12.0)}
    run.tick(fr, 301.0)
    assert run.assembled_at == 301.0
    for t in range(302, 361):
        fr = {"a": frame("a", 0, 120.0 + 12 * (t - 301), 12.0),
              "b": frame("b", 1, 121.0 + 12 * (t - 301), 12.0)}
        assert run.tick(fr, float(t))
    fr = {"a": frame("a", 0, 900.0, 12.0), "b": frame("b", 1, 901.0, 12.0)}
    assert run.tick(fr, 361.0) == {}                  # hold 60 s elapsed -> released
    assert run.outcome() == "assembled_released"
    tl = run.timeline(900.0)
    assert tl["achieved_hold_s"] == 60.0 and tl["request_rel_s"] == 0.0

    late = W.WallArmRunner(arm="w-mid-late", run_name="w1__w-mid-late", t_branch_s=300.0,
                           hold_s=60.0)
    for t in range(300, 482):
        late.tick({}, float(t))
    assert late.outcome() == "not_feasible"
    assert late.target_s == 360.0 and late.deadline_s == 480.0


def test_plan_is_deterministic_and_guarded():
    kw = dict(seeds=list(range(962000, 962006)), penetrations=(0.1, 0.2, 0.4),
              demands=(2400, 2600, 2700), t_choices=(240, 300), plan_seed=1, branch_s=600)
    a, b = W.make_plan(**kw), W.make_plan(**kw)
    assert a["worlds"] == b["worlds"]
    assert [w["penetration"] for w in a["worlds"][:3]] == [0.1, 0.2, 0.4]
    assert a["worlds"][0]["world"].startswith("w962000m2400p10t")
    assert "native_repeat" in a["worlds"][0]["arms"]
    with pytest.raises(ValueError):
        W.make_plan(**{**kw, "seeds": [962200]})      # outside every allowed block


# ------------------------------------------------------------------ converter
def _arm(t0, n, vids, s0, cav=()):
    k = 6
    T = n
    time_s = t0 + 0.5 * np.arange(T)
    s = np.zeros((T, k), np.float32)
    lane = np.full((T, k), -1, np.int8)
    v = np.zeros((T, k), np.float32)
    a = np.zeros((T, k), np.float32)
    mask = np.zeros((T, k), bool)
    vid = np.full((T, k), -1, np.int32)
    for slot, x in enumerate(vids):
        for j in range(T):
            mask[j, slot], vid[j, slot] = True, x
            s[j, slot], lane[j, slot], v[j, slot] = s0[slot] + 10 * 0.5 * j, slot % 2, 10.0
    return {"time_s": time_s, "s": s, "lane": lane, "v": v, "a": a, "mask": mask, "vid": vid,
            "first_seen_s": np.array([0.0, 5.0, 100.0, 200.0, 290.0, np.nan]),
            "first_entrance": np.array([0, 1, 0, 0, 1, -1], np.int8),
            "is_cav": np.array([x in cav for x in range(6)], bool)}


def test_timeline_records_history_counter_and_matching():
    base = _arm(180.0, 241, [0, 1, 2, 3, 4], [0, 50, 100, 150, 200])
    arm = _arm(300.0, 400, [0, 1, 2, 3, 4], [600, 650, 700, 750, 800], cav=(1,))
    reg = [f"v{i}" for i in range(6)]
    tl = WS.Timeline(base, arm, reg, arm["is_cav"])
    j0 = tl.index_of(304.5)
    assert j0 == 240 + 9
    roster = sorted(tl.present(j0))
    fut = tl.future(j0, roster)
    assert len(fut) == 241 and len(fut[0]["vehicles"]) == 5
    h = tl.history(j0, 11, 60.0, True)
    assert np.array(h["kin"]).shape == (11, 6, 7) and h["available_span_s"] == 5.0
    assert h["run_start_limited"] is True and h["ids"][5] == "__history_context_slot5"
    c = np.array(tl.counter(300.0))
    assert c.sum() == 4 and c[1, 1] == 1 and c[0, 10] == 1 and c[0, 20] == 1 and c[1, 29] == 1
    f = tl.frame(j0)
    assert WS.frames_match(f, tl.frame(j0))
    g = json.loads(json.dumps(f))
    g["vehicles"][0]["s"] += 1.0
    assert not WS.frames_match(f, g)
    assert WS.phase_at({"request_s": 300.0, "assembled_at_s": 302.0, "owner_end_s": 400.0,
                        "handback_s": 400.0}, 301.0) == "assembly"
    assert WS.phase_at({"request_s": 300.0, "assembled_at_s": 302.0, "owner_end_s": 400.0,
                        "handback_s": 400.0}, 401.0) == "handed_back"
    assert WS.phase_at({"request_s": None}, 301.0) == "none"
