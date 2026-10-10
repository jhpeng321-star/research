"""P1 interface tests: signatures, batched dispatch, wall laws, parity helpers.

The wall-v2.2 merge-release rule (release when the lead member's cumulative
road coordinate reaches the merge start ``s_road = 2644.2 m``, HOLD only,
hold capped there) is pinned here.  No run data is read: every fixture is
hand-built; the CIRCLES ONNX payload is the code-voted model asset.
"""
from __future__ import annotations

import math

import pytest
import torch

import ctrl
from ctrl import (ACTION_CODES, CavMover, CavStepInput, CtrlConfig,
                  MERGE_START_S_ROAD_M, MergeReleaseWallLaw, WALL_LAW_V21,
                  WALL_LAW_V22, WALL_PHASE_ASSEMBLY, WALL_PHASE_HOLD,
                  WallLawUnavailable, execution_speed, register_wall_law,
                  resolve_wall_law, s_road_m, wall_law_status)


def _row(v, s=100.0, lane=1, cav=True, group=-1, length=5.0, allowed=31.29,
         phase=None):
    one = lambda x: torch.tensor([[x]], dtype=torch.float64)  # noqa: E731
    return CavStepInput(
        s_m=one(s), v_mps=one(v),
        lane=torch.tensor([[lane]], dtype=torch.int64),
        length_m=one(length),
        present=torch.ones(1, 1, dtype=torch.bool),
        is_cav=torch.tensor([[cav]], dtype=torch.bool),
        wall_group=torch.tensor([[group]], dtype=torch.int64),
        segment_lo_m=torch.zeros(1, 1), segment_hi_m=one(3000.0),
        segment_limit_mps=one(allowed), t_s=100.0,
        wall_phase=(None if phase is None else torch.tensor(
            [[phase]], dtype=torch.int64)))


def _grid(**overrides):
    """A [2, 4] batch of live CAV rows (the INTERFACES_V2 slot grid)."""
    b, k = 2, 4
    n = b * k
    fill = lambda val, dt=torch.float64: torch.full(  # noqa: E731
        (b, k), val, dtype=dt)
    fields = dict(
        s_m=torch.arange(n, dtype=torch.float64).reshape(b, k) * 30.0 + 100.0,
        v_mps=fill(27.0),
        lane=torch.full((b, k), 1, dtype=torch.int64),
        length_m=fill(5.0),
        present=torch.ones(b, k, dtype=torch.bool),
        is_cav=torch.ones(b, k, dtype=torch.bool),
        wall_group=torch.full((b, k), -1, dtype=torch.int64),
        segment_lo_m=torch.zeros(b, k), segment_hi_m=fill(3000.0),
        segment_limit_mps=fill(31.29), t_s=100.0)
    fields.update(overrides)
    return CavStepInput(**fields)


# --------------------------------------------------- INTERFACES_V2 surface

def test_public_symbols_present():
    assert ACTION_CODES["native"] == 0 and ACTION_CODES["damping"] == 1
    assert ACTION_CODES["pi"] == 2 and ACTION_CODES["wall"] == 3
    assert ACTION_CODES["none"] == -1
    assert ctrl.CavMover is CavMover
    assert WALL_LAW_V21 == "wall-v2.1-closure"
    assert WALL_LAW_V22 == "wall-v2.2"


def test_config_contract_defaults():
    cfg = CtrlConfig()
    assert cfg.dt_s == 0.5 and cfg.control_dt_s == 1.0
    assert cfg.pi_u_mps == 31.29
    assert cfg.wall_law == WALL_LAW_V22
    assert cfg.model_sha256 == ctrl.CIRCLES_MODEL_SHA256
    with pytest.raises(ValueError):
        CtrlConfig(wall_law="wall-v9")


def test_config_law_version_selects_both_wall_laws():
    """BOTH wall laws are selectable via law_version (== wall_law)."""
    v22 = CtrlConfig()
    assert v22.law_version == "wall-v2.2"
    v21 = CtrlConfig(wall_law="wall-v2.1-closure")
    assert v21.law_version == "wall-v2.1-closure" == WALL_LAW_V21
    assert v21.wall_law == "wall-v2.1-closure"
    # the law version selects the mover path, not just a tag: v2.1 never
    # merge-releases, v2.2 does (see the mover wall tests below)


def test_execution_speed_rule():
    v = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    tgt = torch.tensor([20.0, 0.0, 10.5], dtype=torch.float64)
    idm = torch.full((3,), 3.0, dtype=torch.float64)
    out = execution_speed(tgt, v, idm)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(10.0 + 1.5 * 0.5)   # accel-limited
    assert out[1] == pytest.approx(10.0 - 3.0 * 0.5)   # decel-limited
    assert out[2] == pytest.approx(10.5)               # target reachable


# ------------------------------------------------------------- wall hook

def test_wall_hook_unregistered_then_registered():
    ctrl.clear_registry()
    assert wall_law_status()[WALL_LAW_V22] is False
    with pytest.raises(WallLawUnavailable):
        resolve_wall_law(WALL_LAW_V22)

    class _Law:
        name = WALL_LAW_V22

        def gap_to_front(self, positions_m, group, closure=None):
            return torch.zeros_like(positions_m)

        def pace(self, wall_speed_mps, speed_mps, *, dt_s=1.0,
                 gap_to_front_m=None):
            return torch.zeros_like(speed_mps)

    register_wall_law(_Law())
    assert wall_law_status()[WALL_LAW_V22] is True
    assert resolve_wall_law(WALL_LAW_V22).name == WALL_LAW_V22
    ctrl.clear_registry()
    ctrl.ensure_builtin_laws()
    assert wall_law_status()[WALL_LAW_V22] is True


def test_s_road_mapping_constants():
    # entry 1191554636 ends at the merge start 2644.2; merge 1487606179 is
    # 75.4 m; exit 1487606178 is 477.9 m
    assert s_road_m("1191554636", 0.0) == 0.0
    assert s_road_m("1191554636", 2644.2) == pytest.approx(MERGE_START_S_ROAD_M)
    assert MERGE_START_S_ROAD_M == 2644.2
    assert s_road_m("1487606179", 75.4) == pytest.approx(2644.2 + 75.4)
    assert s_road_m("1487606178", 477.9) == pytest.approx(2644.2 + 75.4 + 477.9)
    with pytest.raises(KeyError):
        s_road_m(":13638027693_0", 1.0)


def _release_cases():
    lead = torch.tensor([[2600.0], [2644.2], [2700.0]],
                        dtype=torch.float64)
    group = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    return lead, group


def test_wall_v22_release_in_hold_only():
    law = MergeReleaseWallLaw()
    lead, group = _release_cases()
    hold = torch.ones(3, 1, dtype=torch.bool)
    released = law.release_mask(lead, group, hold)
    assert released.tolist() == [[False], [True], [True]]
    assembly = torch.zeros(3, 1, dtype=torch.bool)   # not HOLD -> never
    assert law.release_mask(lead, group, assembly).sum() == 0
    # deterministic: identical inputs -> identical outputs
    assert torch.equal(law.release_mask(lead, group, hold), released)
    # non-members are never released
    lone = torch.tensor([[-1.0, 9999.0]], dtype=torch.float64)
    grp = torch.tensor([[0, -1]], dtype=torch.int64)
    hld = torch.ones(1, 2, dtype=torch.bool)
    got = law.release_mask(lone, grp, hld)
    assert got.tolist() == [[False, False]]


def test_wall_v22_pacing_matches_v21_mirrors():
    """The v2.2 law paces exactly like the J4 v2.1 mirrors."""
    from ctrl._bridge import emulators

    emu = emulators()
    law = MergeReleaseWallLaw()
    pos = torch.tensor([100.0, 130.0, 160.0], dtype=torch.float64)
    group = torch.zeros(3, dtype=torch.int64)
    closure = torch.ones(3, dtype=torch.bool)
    assert torch.allclose(law.gap_to_front(pos, group, closure),
                          emu.wall_gap_to_front(pos, group, closure=closure))
    w = torch.full((3,), 2.5, dtype=torch.float64)
    v = torch.tensor([2.5, 3.0, 4.0], dtype=torch.float64)
    gap = law.gap_to_front(pos, group, closure)
    assert torch.allclose(law.pace(w, v, gap_to_front_m=gap),
                          emu.wall_pace(w, v, gap_to_front_m=gap))


# --------------------------------------------- frozen v2.2 law (wall_v22)

def test_wall_v22_frozen_constants():
    """Law constants equal the frozen registration (never tuned)."""
    assert ctrl.MERGE_START_S == 2644.2
    assert ctrl.MERGE_END_S == 2719.6
    assert ctrl.NETWORK_END_S == 3197.5
    assert ctrl.MERGE_RELEASE_X_M == 0.0            # owner binding X = 0
    assert ctrl.MERGE_RELEASE_POINT_S == 2644.2     # = START - X
    assert ctrl.MIN_MEANINGFUL_HOLD_S == 60.0
    assert ctrl.ASSEMBLY_TIME_ASSUMED_S == 10.0
    assert ctrl.FIXED_ARM_DRAWS_S == (60.0, 120.0, 240.0)
    assert ctrl.REQUEST_SPAN_CAP_M == 75.0
    assert ctrl.MERGE_RELEASE_POINT_S == ctrl.MERGE_START_S_ROAD_M


def test_wall_v22_dmin_is_70_times_wall_speed():
    """D_min = X + v*(t_assemble + H_min) = 70 * v_wall (a priori)."""
    assert ctrl.min_proposal_distance_m(15.3) == pytest.approx(1071.0)
    assert ctrl.min_proposal_distance_m(17.0) == pytest.approx(1190.0)
    assert ctrl.min_proposal_distance_m(2.5) == pytest.approx(175.0)
    with pytest.raises(ValueError):
        ctrl.min_proposal_distance_m(0.0)
    with pytest.raises(ValueError):
        ctrl.min_proposal_distance_m(-1.0)


def test_wall_v22_dmin_refusal_record():
    """Reachable leads pass; closer leads are refused ``dmin_unreachable``."""
    # distance to merge 2644.2 - 1500 = 1144.2 >= D_min(15.3) = 1071 -> ok
    assert ctrl.dmin_refusal(1500.0, 15.3) is None
    # distance 1044.2 < 1071 -> refused at the proposal, full record
    rec = ctrl.dmin_refusal(1600.0, 15.3)
    assert rec is not None
    assert rec["reason"] == "dmin_unreachable"
    assert rec["lead_s_road_m"] == pytest.approx(1600.0)
    assert rec["lead_distance_to_merge_m"] == pytest.approx(1044.2)
    assert rec["dmin_m"] == pytest.approx(1071.0)
    assert rec["wall_speed_mps"] == pytest.approx(15.3)
    # the gate boundary honours the frozen 1e-9 tolerance
    edge = 2644.2 - 1071.0
    assert ctrl.dmin_refusal(edge, 15.3) is None
    assert ctrl.dmin_refusal(edge + 1e-6, 15.3)["reason"] == "dmin_unreachable"


def test_wall_v22_pair_gate_2_abreast():
    """The proposal gate: 2 abreast, span cap, then D_min."""
    ok = ctrl.check_pair([0, 1], [1500.0, 1530.0], 15.3)
    assert ok["feasible"] and ok["refusal"] is None
    assert ok["span_m"] == pytest.approx(30.0)
    assert ok["lead_distance_to_merge_m"] == pytest.approx(2644.2 - 1530.0)
    same_lane = ctrl.check_pair([1, 1], [1500.0, 1530.0], 15.3)
    assert not same_lane["feasible"]
    assert same_lane["refusal"]["reason"] == "not_2_abreast"
    wide = ctrl.check_pair([0, 1], [1400.0, 1490.0], 15.3)
    assert not wide["feasible"]
    assert wide["refusal"]["reason"] == "span_exceeds_cap"
    close = ctrl.check_pair([0, 1], [1600.0, 1590.0], 15.3)
    assert not close["feasible"]
    assert close["refusal"]["reason"] == "dmin_unreachable"
    assert close["refusal"]["dmin_m"] == pytest.approx(1071.0)
    with pytest.raises(ValueError):
        ctrl.check_pair([0], [1500.0], 15.3)


def test_wall_v22_hold_cap_formula():
    """hold_cap = min(H, max(0, 2644.2 - s_lead)/v_wall), rounded 6."""
    # merge-capped: 644.2 m / 15.3 m/s = 42.1 s < the 60 s draw
    assert ctrl.hold_cap_s(60.0, 2000.0, 15.3) == pytest.approx(
        round(644.2 / 15.3, 6))
    # uncapped: the 120 s draw expires before the lead reaches the point
    assert ctrl.hold_cap_s(120.0, 0.0, 15.3) == pytest.approx(120.0)
    # a 240 s draw from the entry start is merge-capped at 2644.2/15.3
    assert ctrl.hold_cap_s(240.0, 0.0, 15.3) == pytest.approx(
        round(2644.2 / 15.3, 6))
    assert ctrl.hold_cap_s(240.0, 2600.0, 15.3) == pytest.approx(
        round(44.2 / 15.3, 6))
    # at/past the release point the cap is zero (release fires this tick)
    assert ctrl.hold_cap_s(240.0, 2644.2, 15.3) == pytest.approx(0.0)
    assert ctrl.hold_cap_s(240.0, 2700.0, 15.3) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ctrl.hold_cap_s(60.0, 2000.0, 0.0)


def test_wall_v22_hold_cap_record_fields():
    """The hold_cap log row carries the frozen wall_driver field set."""
    row = ctrl.hold_cap_record(wall_id="w1", timestamp_s=510.0,
                               requested_hold_s=240.0, wall_speed_mps=15.3,
                               lead_s_road_m=2000.0, effective_hold_s=42.1,
                               capped_at_merge=True)
    assert set(row) == {
        "kind", "wall_id", "timestamp_s", "requested_hold_s",
        "effective_hold_s", "capped_at_merge", "cap_point_s",
        "a_priori_effective_hold_s", "lead_s_road_at_release_m",
        "wall_speed_mps", "source"}
    assert row["kind"] == "hold_cap"
    assert row["cap_point_s"] == pytest.approx(2644.2)
    assert row["a_priori_effective_hold_s"] == pytest.approx(
        ctrl.hold_cap_s(240.0, 2000.0, 15.3))
    assert row["source"] == "cav_report_frames"
    # derived flags when the engine does not supply them
    row2 = ctrl.hold_cap_record(wall_id="w2", timestamp_s=1.0,
                                requested_hold_s=60.0, wall_speed_mps=15.3,
                                lead_s_road_m=0.0)
    assert row2["capped_at_merge"] is False
    assert row2["effective_hold_s"] == pytest.approx(60.0)


def test_wall_v22_release_event_fields():
    """merge_release: a RELEASE kind, handback native, frozen fields."""
    ev = ctrl.merge_release_event(wall_id=3, timestamp_s=512.5)
    assert ev["kind"] == "wall_event"
    assert ev["event"] == "merge_release"
    assert ev["release_kind"] == "merge_release"    # never an abort
    assert ev["release_point_s"] == pytest.approx(2644.2)
    assert ev["merge_start_s"] == pytest.approx(2644.2)
    assert ev["handback"] == "native"
    assert ev["law_version"] == "wall-v2.2"
    assert ev["wall_id"] == 3 and ev["timestamp_s"] == pytest.approx(512.5)


def test_merge_release_session_fires_once_per_wall():
    """The session emits the event exactly once per (row, wall)."""
    from ctrl import MergeReleaseSession

    lead = torch.tensor([[2600.0], [2644.2]], dtype=torch.float64)
    group = torch.tensor([[0], [1]], dtype=torch.int64)
    hold = torch.ones(2, 1, dtype=torch.bool)
    sess = MergeReleaseSession()
    r1 = sess.update(500.0, lead, group, hold)
    assert r1.tolist() == [[False], [True]]
    assert len(sess.events) == 1
    assert sess.events[0]["wall_id"] == 1
    assert sess.fired == {(1, 1)}
    # the same wall at/after the point: mask stays True, NO second event;
    # wall 0 now reaches the point for the FIRST time and fires once
    lead2 = torch.tensor([[2644.2], [2700.0]], dtype=torch.float64)
    r2 = sess.update(501.0, lead2, group, hold)
    assert r2.tolist() == [[True], [True]]
    assert len(sess.events) == 2
    assert [e["wall_id"] for e in sess.events] == [1, 0]
    assert sess.events[0]["timestamp_s"] == pytest.approx(500.0)
    assert sess.events[1]["timestamp_s"] == pytest.approx(501.0)
    assert sess.fired == {(1, 1), (0, 0)}
    # a third tick with both walls past the point: nothing new fires
    sess.update(502.0, lead2, group, hold)
    assert len(sess.events) == 2
    # reset clears the latch for a new episode
    sess.reset()
    assert sess.events == [] and sess.fired == set()
    sess.update(0.0, lead2, group, hold)
    assert len(sess.events) == 2
    assert sess.fired == {(0, 0), (1, 1)}


# ---------------------------------------------------------------- mover

def test_mover_native_and_non_cav_slots_get_no_command():
    mover = CavMover()
    state = mover.init_state(2, 4)
    inp = _grid()
    actions = torch.full((2, 4), ACTION_CODES["native"], dtype=torch.int64)
    out = mover.step(state, inp, actions)
    assert out.target_speed_mps.shape == (2, 4)
    assert out.target_speed_mps.dtype == torch.float64
    assert bool(torch.isnan(out.target_speed_mps).all())
    assert (out.reason_code == 1).all()          # selected_native
    # non-CAV slots are never commanded, whatever their action row says
    inp2 = _grid(is_cav=torch.tensor(
        [[True, False, True, False], [True, True, False, False]]))
    act2 = torch.full((2, 4), ACTION_CODES["damping"], dtype=torch.int64)
    out2 = mover.step(state, inp2, act2)
    assert bool(torch.isnan(out2.target_speed_mps[0, 1]))
    assert bool(torch.isnan(out2.target_speed_mps[1, 2]))


def test_mover_step_is_deterministic():
    mover = CavMover()
    state = mover.init_state(2, 4)
    inp = _grid()
    act = torch.full((2, 4), ACTION_CODES["native"], dtype=torch.int64)
    a = mover.step(state, inp, act)
    b = mover.step(state, inp, act)

    def same(x, y):
        return bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all())

    assert same(a.target_speed_mps, b.target_speed_mps)
    assert torch.equal(a.reason_code, b.reason_code)
    assert torch.equal(a.state.pi_raw_prev, b.state.pi_raw_prev)


def test_mover_wall_requires_wall_speed():
    mover = CavMover(ctrl.CtrlConfig(wall_law=WALL_LAW_V21))
    state = mover.init_state(1, 1)
    act = torch.tensor([[ACTION_CODES["wall"]]], dtype=torch.int64)
    with pytest.raises(ValueError):
        mover.step(state, _row(2.5, group=0, phase=WALL_PHASE_HOLD), act)


def test_mover_wall_v21_matches_j4_mirrors():
    """HOLD and assembly targets equal wall_pace -> direct_target."""
    from ctrl._bridge import emulators

    emu = emulators()
    mover = CavMover(ctrl.CtrlConfig(wall_law=WALL_LAW_V21))
    state = mover.init_state(1, 2)
    act = torch.full((1, 2), ACTION_CODES["wall"], dtype=torch.int64)
    wall_speed = torch.full((1, 2), 3.0, dtype=torch.float64)
    # HOLD: both members paced to v_wall (lead at 2650 < no v2.2 release)
    inp = CavStepInput(
        s_m=torch.tensor([[2630.0, 2650.0]],
                         dtype=torch.float64) - MERGE_START_S_ROAD_M,
        v_mps=torch.tensor([[3.5, 3.5]], dtype=torch.float64),
        lane=torch.zeros(1, 2, dtype=torch.int64),
        length_m=torch.full((1, 2), 5.0),
        present=torch.ones(1, 2, dtype=torch.bool),
        is_cav=torch.ones(1, 2, dtype=torch.bool),
        wall_group=torch.zeros(1, 2, dtype=torch.int64),
        segment_lo_m=torch.zeros(1, 2),
        segment_hi_m=torch.full((1, 2), 3000.0),
        segment_limit_mps=torch.full((1, 2), 31.29), t_s=500.0,
        wall_phase=torch.full((1, 2), WALL_PHASE_HOLD, dtype=torch.int64))
    out = mover.step(state, inp, act, wall_speed_mps=wall_speed)
    speed = inp.v_mps[0]
    allowed = inp.segment_limit_mps[0]
    _cmd, ref = emu.direct_target(speed, allowed, emu.wall_pace(
        wall_speed[0], speed, dt_s=1.0, gap_to_front_m=None))
    assert torch.allclose(out.target_speed_mps[0], ref)
    assert (out.reason_code[0] == 0).all()
    # ASSEMBLY: rear member (20 m behind front) closes at v + min(2, 0.1g)
    inp.wall_phase = torch.full((1, 2), WALL_PHASE_ASSEMBLY,
                                dtype=torch.int64)
    out2 = mover.step(state, inp, act, wall_speed_mps=wall_speed)
    gap = emu.wall_gap_to_front(
        (inp.s_m[0] + MERGE_START_S_ROAD_M), inp.wall_group[0],
        closure=torch.ones(2, dtype=torch.bool))
    accel = emu.wall_pace(wall_speed[0], speed, dt_s=1.0, gap_to_front_m=gap)
    _cmd, ref2 = emu.direct_target(speed, allowed, accel)
    assert torch.allclose(out2.target_speed_mps[0], ref2)
    # the closure targets exceed the wall speed for the rear member
    assert out2.target_speed_mps[0, 0] > 3.0
    assert out2.target_speed_mps[0, 1] == pytest.approx(3.0, abs=1e-9)


def test_mover_wall_v22_releases_at_merge_start_in_hold():
    mover = CavMover()                      # default wall law: wall-v2.2
    state = mover.init_state(1, 1)
    act = torch.tensor([[ACTION_CODES["wall"]]], dtype=torch.int64)
    wall_speed = torch.full((1, 1), 2.5, dtype=torch.float64)
    # s in _row is the merge-relative s_m: s_road = s + 2644.2
    below = mover.step(state, _row(2.5, s=-44.2,
                                   group=0, phase=WALL_PHASE_HOLD),
                       act, wall_speed_mps=wall_speed)
    assert below.reason_code[0, 0] == 0     # issued
    assert below.target_speed_mps[0, 0] == pytest.approx(2.5, abs=1e-9)
    at = mover.step(state, _row(2.5, s=0.0, group=0,
                                phase=WALL_PHASE_HOLD),
                    act, wall_speed_mps=wall_speed)
    assert math.isnan(at.target_speed_mps[0, 0])
    assert at.reason_code[0, 0] == 1        # handed back: selected_native
    past = mover.step(state, _row(2.5, s=55.8, group=0,
                                  phase=WALL_PHASE_HOLD),
                      act, wall_speed_mps=wall_speed)
    assert math.isnan(past.target_speed_mps[0, 0])
    # ASSEMBLY at/past the merge start is NOT released (HOLD only)
    asm = mover.step(state, _row(2.5, s=55.8, group=0,
                                 phase=WALL_PHASE_ASSEMBLY),
                     act, wall_speed_mps=wall_speed)
    assert asm.reason_code[0, 0] == 0
    # v2.1 keeps pacing past the merge start (no release rule)
    mover21 = CavMover(ctrl.CtrlConfig(wall_law=WALL_LAW_V21))
    state21 = mover21.init_state(1, 1)
    old = mover21.step(state21, _row(2.5, s=55.8, group=0,
                                     phase=WALL_PHASE_HOLD),
                       act, wall_speed_mps=wall_speed)
    assert old.reason_code[0, 0] == 0
    assert old.target_speed_mps[0, 0] == pytest.approx(2.5, abs=1e-9)


def test_mover_wall_v22_release_fires_once_and_logs_merge_release():
    """v2.2: release at s_road 2644.2, HOLD only, ONCE, event logged."""
    mover = CavMover()                      # default wall law: wall-v2.2
    state = mover.init_state(1, 1)
    act = torch.tensor([[ACTION_CODES["wall"]]], dtype=torch.int64)
    wall_speed = torch.full((1, 1), 2.5, dtype=torch.float64)
    # tick 1: below the point (s_road = 2600.0) -> commanded, no event
    t1 = mover.step(state, _row(2.5, s=-44.2, group=0,
                                phase=WALL_PHASE_HOLD),
                    act, wall_speed_mps=wall_speed)
    assert t1.reason_code[0, 0] == 0
    assert t1.merge_release_events is None
    assert not bool(t1.state.merge_released[0, 0])
    # tick 2: AT the point (s_road = 2644.2, X = 0) -> released, ONE event
    t2 = mover.step(t1.state, _row(2.5, s=0.0, group=0,
                                   phase=WALL_PHASE_HOLD),
                    act, wall_speed_mps=wall_speed)
    assert math.isnan(t2.target_speed_mps[0, 0])
    assert t2.reason_code[0, 0] == 1        # handed back to native
    assert bool(t2.state.merge_released[0, 0])
    assert t2.merge_release_events is not None
    assert len(t2.merge_release_events) == 1
    ev = t2.merge_release_events[0]
    assert ev["event"] == "merge_release" and ev["release_kind"] == "merge_release"
    assert ev["release_point_s"] == pytest.approx(2644.2)
    assert ev["handback"] == "native" and ev["wall_id"] == 0
    assert ev["timestamp_s"] == pytest.approx(100.0)
    # tick 3: the latch keeps the wall released and fires NO second event
    t3 = mover.step(t2.state, _row(2.5, s=30.0, group=0,
                                   phase=WALL_PHASE_HOLD),
                    act, wall_speed_mps=wall_speed)
    assert math.isnan(t3.target_speed_mps[0, 0])
    assert t3.reason_code[0, 0] == 1
    assert t3.merge_release_events is None
    # even a phase report flip back to ASSEMBLY never re-enslaves (once)
    t4 = mover.step(t3.state, _row(2.5, s=30.0, group=0,
                                   phase=WALL_PHASE_ASSEMBLY),
                    act, wall_speed_mps=wall_speed)
    assert math.isnan(t4.target_speed_mps[0, 0])
    assert t4.merge_release_events is None


def test_mover_wall_v22_release_is_batched_per_row_and_wall():
    """The release mask and events are batched [B, K] per (row, wall)."""
    b, k = 2, 4
    mover = CavMover()
    state = mover.init_state(b, k)
    act = torch.tensor([[ACTION_CODES["wall"]] * 4,
                        [ACTION_CODES["wall"], ACTION_CODES["wall"],
                         ACTION_CODES["native"], ACTION_CODES["native"]]],
                       dtype=torch.int64)
    wall_speed = torch.full((b, k), 2.5, dtype=torch.float64)
    fields = dict(
        v_mps=torch.full((b, k), 2.5),
        lane=torch.zeros(b, k, dtype=torch.int64),
        length_m=torch.full((b, k), 5.0),
        present=torch.ones(b, k, dtype=torch.bool),
        is_cav=torch.ones(b, k, dtype=torch.bool),
        # row 0: wall 0 (past the point) + wall 1 (before it);
        # row 1: wall 0 only, still before the point
        wall_group=torch.tensor([[0, 0, 1, 1], [0, 0, -1, -1]],
                                dtype=torch.int64),
        wall_phase=torch.full((b, k), WALL_PHASE_HOLD, dtype=torch.int64))
    fields["s_m"] = torch.tensor(
        [[10.0, 20.0, -100.0, -120.0], [5.0, 9.0, 0.0, 0.0]],
        dtype=torch.float64)
    inp = CavStepInput(
        segment_lo_m=torch.zeros(b, k), segment_hi_m=torch.full((b, k), 3000.0),
        segment_limit_mps=torch.full((b, k), 31.29), t_s=700.0, **fields)
    out = mover.step(state, inp, act, wall_speed_mps=wall_speed)
    assert out.target_speed_mps.shape == (b, k)
    # row 0 wall 0 (s_road 2654.2/2664.2 >= 2644.2): released to native
    assert math.isnan(out.target_speed_mps[0, 0])
    assert math.isnan(out.target_speed_mps[0, 1])
    # row 0 wall 1 (s_road 2544.2/2524.2 < 2644.2): still commanded
    assert out.reason_code[0, 2] == 0 and out.reason_code[0, 3] == 0
    # row 1 wall 0 (s_road 2649.2/2653.2 >= 2644.2): released too
    assert math.isnan(out.target_speed_mps[1, 0])
    # exactly one event per released wall, none for the held wall
    fired = {(e["wall_id"]) for e in out.merge_release_events}
    assert fired == {0}
    assert len(out.merge_release_events) == 2   # row 0 + row 1 of wall 0
    assert all(e["timestamp_s"] == pytest.approx(700.0)
               for e in out.merge_release_events)


def test_mover_wall_v22_release_honours_frozen_tolerance():
    """The rule fires at MERGE_RELEASE_POINT_S - 1e-9 (frozen wall_group)."""
    mover = CavMover()
    state = mover.init_state(1, 1)
    act = torch.tensor([[ACTION_CODES["wall"]]], dtype=torch.int64)
    wall_speed = torch.full((1, 1), 2.5, dtype=torch.float64)
    just_below = mover.step(
        state, _row(2.5, s=-5e-10, group=0, phase=WALL_PHASE_HOLD),
        act, wall_speed_mps=wall_speed)
    assert math.isnan(just_below.target_speed_mps[0, 0])
    assert just_below.reason_code[0, 0] == 1
    margin = mover.step(
        state, _row(2.5, s=-2e-9, group=0, phase=WALL_PHASE_HOLD),
        act, wall_speed_mps=wall_speed)
    assert margin.reason_code[0, 0] == 0
    assert margin.target_speed_mps[0, 0] == pytest.approx(2.5, abs=1e-9)


def test_mover_damping_matches_parta_emulator():
    """The damping path equals the E1-validated emulator on the same frame."""
    from ctrl._bridge import emulators

    emu_mod = emulators()
    mover = CavMover()
    state = mover.init_state(1, 2)
    inp = CavStepInput(
        s_m=torch.tensor([[50.0, 80.0]], dtype=torch.float64),
        v_mps=torch.tensor([[27.17, 27.14]], dtype=torch.float64),
        lane=torch.zeros(1, 2, dtype=torch.int64),
        length_m=torch.full((1, 2), 5.0),
        present=torch.ones(1, 2, dtype=torch.bool),
        is_cav=torch.ones(1, 2, dtype=torch.bool),
        wall_group=torch.full((1, 2), -1, dtype=torch.int64),
        segment_lo_m=torch.zeros(1, 2),
        segment_hi_m=torch.full((1, 2), 2644.2),
        segment_limit_mps=torch.full((1, 2), 31.29), t_s=420.0)
    act = torch.tensor([[ACTION_CODES["damping"], ACTION_CODES["native"]]],
                       dtype=torch.int64)
    out = mover.step(state, inp, act)
    frame = emu_mod.FrameBatch(
        now_s=torch.tensor([420.0]), ts_s=torch.tensor([420.0]),
        speed_mps=torch.tensor([27.17], dtype=torch.float64),
        allowed_speed_mps=torch.tensor([31.29], dtype=torch.float64),
        front_coverage_m=torch.tensor([min(100.0, 2644.2 - 50.0)]),
        rear_coverage_m=torch.tensor([min(100.0, 50.0 - 5.0)]),
        front_present=torch.tensor([True]),
        front_gap_m=torch.tensor([80.0 - 5.0 - 50.0]),
        front_rel_mps=torch.tensor([27.14 - 27.17]),
        rear_present=torch.tensor([False]),
        rear_gap_m=torch.tensor([0.0]), rear_rel_mps=torch.tensor([0.0]),
        frame_present=torch.tensor([True]))
    ref_target, ref_code, _ = emu_mod.DampingEmulator(
        str(ctrl._bridge.model_onnx())).decide(frame)
    assert int(ref_code[0]) == 0
    assert out.target_speed_mps[0, 0].item() == pytest.approx(
        float(ref_target[0]), abs=1e-9)
    assert out.reason_code[0, 0].item() == 0
    # the issued commanded acceleration is the bridge slope
    assert out.accel_cmd_mps2[0, 0].item() == pytest.approx(
        float(ref_target[0]) - 27.17, abs=1e-9)


def test_mover_pi_lifecycle_and_state():
    from ctrl._bridge import emulators

    emu_mod = emulators()
    mover = CavMover()
    state = mover.init_state(1, 3)
    inp = CavStepInput(
        s_m=torch.tensor([[50.0, 80.0, 20.0]], dtype=torch.float64),
        v_mps=torch.tensor([[27.17, 27.14, 27.2]], dtype=torch.float64),
        lane=torch.zeros(1, 3, dtype=torch.int64),
        length_m=torch.full((1, 3), 5.0),
        present=torch.ones(1, 3, dtype=torch.bool),
        is_cav=torch.ones(1, 3, dtype=torch.bool),
        wall_group=torch.full((1, 3), -1, dtype=torch.int64),
        segment_lo_m=torch.zeros(1, 3),
        segment_hi_m=torch.full((1, 3), 2644.2),
        segment_limit_mps=torch.full((1, 3), 31.29), t_s=420.0)
    act = torch.tensor([[ACTION_CODES["damping"], ACTION_CODES["native"],
                         ACTION_CODES["pi"]]], dtype=torch.int64)
    out = mover.step(state, inp, act)
    assert out.reason_code[0, 2].item() == 0
    # the raw command equals the reference recurrence on the same frame
    # (auto sensing: the ego at 20 m sees the nearest front at 50 m)
    pi = emu_mod.PIEmulator()
    ref_raw, ref_state = pi.raw_command(
        emu_mod.PIState.zeros(1),
        torch.tensor([True]),
        torch.tensor([420.0], dtype=torch.float64),
        torch.tensor([27.2], dtype=torch.float64),
        torch.tensor([27.17], dtype=torch.float64),
        torch.tensor([50.0 - 5.0 - 20.0], dtype=torch.float64))
    assert out.state.pi_raw_prev[0, 2].item() == pytest.approx(
        float(ref_raw[0]), abs=1e-9)
    assert bool(out.state.pi_active[0, 2])
    # a same-tick repeat violates the one-sample clock -> defensive re-init
    out2 = mover.step(out.state, inp, act)
    assert not bool(out2.state.pi_active[0, 2])


# -------------------------------------------------------- parity helpers

def test_parity_sumo_idm_accel_value():
    from ctrl.parity import sumo_idm_accel

    # free road (sentinel gap 250), normal_car: the (s*/gap)^2 term survives
    a = sumo_idm_accel(10.0, 250.0, 0.0, 2.6, 4.5, 1.1, 2.5, 31.29)
    s_star = 2.5 + 10.0 * 1.1
    assert a == pytest.approx(
        2.6 * (1.0 - (10.0 / 31.29) ** 4 - (s_star / 250.0) ** 2),
        abs=1e-12)
    # a stopped leader 10 m ahead brakes hard
    a2 = sumo_idm_accel(15.0, 10.0, 15.0, 2.6, 4.5, 1.1, 2.5, 31.29)
    assert a2 < -3.0


def test_parity_row_geometry():
    from ctrl.parity import row_geometry

    states = {"a": ("1191554636_0", 100.0, 20.0, "normal_car", False),
              "b": ("1487606179_0", 10.0, 5.0, "normal_car", False),
              "c": (":13638027693_0_1", 1.0, 20.0, "normal_car", False)}
    geom = row_geometry(states)
    assert geom["a"] == (1, 100.0)                 # entry right lane
    assert geom["b"] == (2, 378.66 + 10.0)         # accel chain
    assert geom["c"] == (None, None)               # junction internal


def test_parity_reconstruct_frames_front_rear():
    from ctrl.parity import reconstruct_frames

    states = {
        "ego": ("1191554636_0", 100.0, 25.0, "normal_car", False),
        "front": ("1191554636_0", 140.0, 24.0, "normal_car", False),
        "rear": ("1191554636_0", 60.0, 26.0, "normal_car", False),
        "other_lane": ("1191554636_1", 110.0, 25.0, "normal_car", False),
    }
    lanes = {"1191554636_0": (2644.22, 31.29),
             "1191554636_1": (2645.67, 31.29)}
    frames = reconstruct_frames(states, ["ego"], lanes,
                                {"normal_car": (1.0, 1.1, 2.6, 4.5, 2.5)})
    fr = frames["ego"]
    assert fr["front_present"] and fr["rear_present"]
    assert fr["front_gap_m"] == pytest.approx(140.0 - 5.0 - 100.0)
    assert fr["front_rel_mps"] == pytest.approx(-1.0)
    assert fr["rear_gap_m"] == pytest.approx(100.0 - 5.0 - 60.0)
    assert fr["rear_rel_mps"] == pytest.approx(1.0)
    assert fr["allowed_speed_mps"] == pytest.approx(31.29)
    assert fr["front_coverage_m"] == pytest.approx(100.0)
    # a far front beyond the coverage is not seen
    states["front"] = ("1191554636_0", 400.0, 24.0, "normal_car", False)
    fr2 = reconstruct_frames(states, ["ego"], lanes,
                             {"normal_car": (1.0, 1.1, 2.6, 4.5, 2.5)})["ego"]
    assert not fr2["front_present"]
