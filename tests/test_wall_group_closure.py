"""Wall assembly closure law (owner decision 2026-09-30, Option B).

Front member holds v_wall; each rear member targets
v_wall + min(2.0, 0.1 * gap_to_front) during assembly; hold paces all to
v_wall; assembly_timeout aborts at 60 s.  Fakes only, zero simulation.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from traffic_shepherd import wall_group as wg
from traffic_shepherd.wall_group import WallGroupSession, _pace


def fr(vid, now, speed, s, lane):
    """Report frame; s is the downstream-increasing coordinate."""
    return SimpleNamespace(vehicle_id=vid, timestamp_s=now, speed_mps=speed,
                           distance_to_merge_m=-float(s), lane_index=lane)


class World:
    """Point-mass members; commands applied as v += a*dt, x += v*dt."""

    def __init__(self, members, wall_speed, *, pinned_front=None):
        self.ids = list(members)            # id -> (s, v)
        self.state = {k: list(v) for k, v in members.items()}
        self.session = WallGroupSession("w", members=frozenset(self.ids),
                                        wall_speed_mps=wall_speed)
        self.v_wall = wall_speed
        self.pinned = pinned_front          # (id, speed) scripted actuator
        self.t = 0.0
        self.cmd_log = []

    def frames(self):
        return {k: fr(k, self.t, v, s, i)
                for i, (k, (s, v)) in enumerate(self.state.items())}

    def step(self):
        cmds = self.session.tick(self.frames(), self.t)
        self.cmd_log.append((self.t, dict(cmds), {k: tuple(x) for k, x in self.state.items()}))
        for k, st in self.state.items():
            if self.pinned and k == self.pinned[0]:
                st[1] = self.pinned[1]
            elif k in cmds:
                st[1] = max(0.0, st[1] + cmds[k] * 1.0)
            st[0] += st[1] * 1.0
        self.t += 1.0
        return cmds

    def span(self):
        pos = [s for s, _ in self.state.values()]
        return max(pos) - min(pos)


def test_a_wide_pair_converges_and_front_is_paced():
    for gap in (50.0, 75.0):
        w = World({"f": (gap, 20.0), "r": (0.0, 20.0)}, 20.0)
        for _ in range(60):
            w.step()
            if w.session.phase != wg.PHASE_ASSEMBLY:
                break
        assert w.session.phase == wg.PHASE_HOLD, gap
        assert w.session.assembled_at_s is not None and w.session.assembled_at_s <= 60.0
        # front always paced to v_wall exactly; rear inside the bounded clip
        for _t, c, st in w.cmd_log:
            if not c:
                continue
            assert c["f"] == _pace(20.0, st["f"][1], 1.0)
            assert -3.0 <= c["r"] <= 1.5
        # span <= 6 m at the assembled tick (positions before the last advance)
        _t, _c, st = w.cmd_log[-1]
        assert abs(st["f"][0] - st["r"][0]) <= wg.ASSEMBLY_SPAN_TOL_M
        assert abs(st["r"][1] - 20.0) <= wg.ASSEMBLY_SPEED_TOL_MPS


def test_a_rear_command_equals_formula():
    w = World({"f": (50.0, 20.0), "r": (0.0, 18.0)}, 20.0)
    cmds = w.step()
    assert cmds["f"] == _pace(20.0, 20.0, 1.0)
    assert cmds["r"] == _pace(20.0 + min(2.0, 0.1 * 50.0), 18.0, 1.0)
    # small gap: Delta = 0.1 * gap
    w = World({"f": (5.0, 20.0), "r": (0.0, 19.0)}, 20.0)
    assert w.step()["r"] == _pace(20.5, 19.0, 1.0)


def test_b_close_pair_assembles_on_first_tick():
    w = World({"f": (4.0, 20.0), "r": (0.0, 20.0)}, 20.0)
    cmds = w.step()
    assert w.session.phase == wg.PHASE_HOLD and w.session.assembled_at_s == 0.0
    assert cmds["f"] == 0.0 and cmds["r"] == pytest.approx(0.4)
    # regression schedule: same tick as the old identical-pacing law would
    old = WallGroupSession("o", members=frozenset({"f", "r"}), wall_speed_mps=20.0)
    old.tick({"f": fr("f", 0.0, 20.0, 4.0, 0), "r": fr("r", 0.0, 20.0, 0.0, 1)}, 0.0)
    assert old.assembled_at_s == 0.0


def test_speed_test_passes_with_delta_at_span_tolerance():
    # rear tracking Delta = 0.6 m/s at exactly 6 m span still passes 1 m/s tol
    w = World({"f": (6.0, 20.0), "r": (0.0, 20.6)}, 20.0)
    w.step()
    assert w.session.phase == wg.PHASE_HOLD


def test_c_assembly_timeout_and_handback():
    # front pinned fast (like the WM-43 A3 injector): closure impossible
    w = World({"f": (50.0, 22.5), "r": (0.0, 20.0)}, 20.0, pinned_front=("f", 22.5))
    last = None
    for _ in range(70):
        cmds = w.step()
        if w.session.phase == wg.PHASE_HANDBACK:
            last = cmds
            break
    s = w.session
    assert s.abort_reason == "assembly_timeout"
    assert w.t - 1.0 == 60.0 and s.assembly_since == 0.0
    assert last == {}
    ev = s.events[-1]
    assert ev["event"] == "safety_abort" and ev["reason"] == "assembly_timeout"
    assert ev["handback"] == "native" and ev["timestamp_s"] == 60.0
    # 59 s: still assembling, no abort yet
    assert all(c for t, c, _ in w.cmd_log if t < 60.0)
    assert not s.active
    assert w.session.tick(w.frames(), 61.0) == {}


def test_timeout_after_other_aborts_order():
    # member_span_exceeded is still evaluated first
    s = WallGroupSession("w", members=frozenset({"a", "b"}), wall_speed_mps=20.0)
    s.tick({"a": fr("a", 0.0, 20.0, 0.0, 0), "b": fr("b", 0.0, 20.0, 10.0, 1)}, 0.0)
    out = s.tick({"a": fr("a", 60.0, 20.0, 0.0, 0), "b": fr("b", 60.0, 20.0, 200.0, 1)}, 60.0)
    assert out == {} and s.abort_reason == "member_span_exceeded"


def test_d_hold_paces_all_to_v_wall():
    s = WallGroupSession("w", members=frozenset({"a", "b"}), wall_speed_mps=20.0)
    s.tick({"a": fr("a", 0.0, 20.0, 0.0, 0), "b": fr("b", 0.0, 20.0, 3.0, 1)}, 0.0)
    assert s.phase == wg.PHASE_HOLD
    # even with a large longitudinal offset, HOLD paces both to v_wall
    fr2 = {"a": fr("a", 1.0, 17.0, 0.0, 0), "b": fr("b", 1.0, 23.0, 30.0, 1)}
    cmds = s.tick(fr2, 1.0)
    assert cmds == {"a": _pace(20.0, 17.0, 1.0), "b": _pace(20.0, 23.0, 1.0)}
    # a hold longer than the timeout never times out
    assert s.tick({"a": fr("a", 500.0, 20.0, 0.0, 0), "b": fr("b", 500.0, 20.0, 3.0, 1)},
                  500.0)
    assert s.abort_reason is None


def test_e_three_members_close_to_front():
    w = World({"a": (60.0, 20.0), "b": (30.0, 20.0), "c": (0.0, 20.0)}, 20.0)
    cmds = w.step()
    assert cmds["a"] == 0.0
    assert cmds["b"] == _pace(22.0, 20.0, 1.0)     # gap 30 m -> Delta capped 2
    assert cmds["c"] == _pace(22.0, 20.0, 1.0)
    for _ in range(80):
        if w.session.phase != wg.PHASE_ASSEMBLY:
            break
        w.step()
    assert w.session.phase == wg.PHASE_HOLD
    _t, _c, st = w.cmd_log[-1]
    ss = [x[0] for x in st.values()]
    assert max(ss) - min(ss) <= wg.ASSEMBLY_SPAN_TOL_M


def test_front_is_most_downstream_regardless_of_id_or_lane_order():
    s = WallGroupSession("w", members=frozenset({"a", "b"}), wall_speed_mps=20.0)
    cmds = s.tick({"a": fr("a", 0.0, 20.0, 40.0, 1), "b": fr("b", 0.0, 20.0, 0.0, 0)}, 0.0)
    assert cmds["a"] == 0.0 and cmds["b"] == _pace(22.0, 20.0, 1.0)


def test_f_law_version_in_events():
    w = World({"f": (4.0, 20.0), "r": (0.0, 20.0)}, 20.0)
    w.step()
    evs = {e["event"]: e for e in w.session.events}
    assert evs["assembly_start"]["law_version"] == wg.WALL_LAW_VERSION == "wall-v2.1-closure"
    assert evs["assembled"]["law_version"] == "wall-v2.1-closure"
