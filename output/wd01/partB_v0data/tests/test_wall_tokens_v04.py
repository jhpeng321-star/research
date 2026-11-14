"""v0.4 wall token tests (brief item 2) on synthetic wall_events.jsonl /
branch_record.json fixtures in the exact campaign_walls_v1/wall_driver.py
schema (header lines 1-30 + the emitted row shapes), covering the three
lifecycles: assemble -> hold -> release; abort in assembly; not_feasible.
Includes the leakage shuffle test and the strict/deployability audit
extensions (no outcome fields, no HDV-derived wall features).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import featurize_v0 as F  # noqa: E402
import wall_tokens_v04 as W  # noqa: E402

T0 = 480.0
WID = "wall_wsyntp02t480_early"


def request_row(wall_id=WID, ts=T0, members=("cav.a", "cav.c"),
                speed=17.0, hold=60.0):
    return dict(kind="wall_request", wall_id=wall_id,
                timestamp_s=round(float(ts), 6), timing="early",
                target_time_s=ts, deadline_s=ts + 120.0,
                edge="1191554636_0", members=list(members), lanes=[0, 1],
                span_m=40.0, speed_factor=0.85,
                mean_member_speed_mps=20.0, wall_speed_mps=float(speed),
                hold_s=float(hold), source="cav_report_frames")


def wall_event(event, ts, wall_id=WID, **extra):
    return dict(kind="wall_event", wall_id=wall_id,
                timestamp_s=round(float(ts), 6), event=event, **extra)


def not_feasible_row(wall_id=WID, ts=T0 + 121.0):
    return dict(kind="not_feasible", wall_id=wall_id,
                timestamp_s=round(float(ts), 6), timing="early",
                target_time_s=ts - 1.0, deadline_s=ts - 1.0,
                reason="no feasible pair within the 120 s request window")


def handback_row(how, phase, ts, wall_id=WID,
                 members=("cav.a", "cav.c")):
    return dict(kind="wall_handback", wall_id=wall_id,
                timestamp_s=round(float(ts), 6), how=how, phase=phase,
                reason=None, members=list(members), handback="native")


def branch_record(outcome, **overrides):
    rec = dict(schema="wm59_wall_branch_record/1", root_id="wsynt",
               branch_id="wsyntp02t480_early", branch_type="early",
               t0_s=T0, wall_id=WID, members=["cav.a", "cav.c"],
               speed_factor=0.85, wall_speed_mps=17.0, timing="early",
               hold_s=60.0, request_s=T0, outcome=outcome,
               assembled_at_s=None, released_at_s=None, aborted_at_s=None,
               abort_reason=None)
    rec.update(overrides)
    return rec


def write_branch(run_dir: Path, events: list[dict], record: dict | None):
    (run_dir / "run").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run" / "wall_events.jsonl", "w",
              encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    if record is not None:
        (run_dir / "run" / "branch_record.json").write_text(
            json.dumps(record), encoding="utf-8")


def released_branch(tmp_path):
    run = tmp_path / "released"
    events = [request_row(), wall_event("assembly_start", T0 + 1.0),
              wall_event("assembled", T0 + 3.0,
                         members=["cav.a", "cav.c"]),
              wall_event("released", T0 + 63.0),
              handback_row("released", "hold", T0 + 63.0)]
    write_branch(run, events, branch_record("assembled_released",
                                            assembled_at_s=T0 + 3.0,
                                            released_at_s=T0 + 63.0))
    return run


def aborted_assembly_branch(tmp_path):
    run = tmp_path / "aborted"
    events = [request_row(), wall_event("assembly_start", T0 + 1.0),
              wall_event("safety_abort", T0 + 5.0, reason="collision"),
              handback_row("safety_abort", "assembly", T0 + 5.0)]
    write_branch(run, events,
                 branch_record("aborted_in_assembly",
                               assembled_at_s=None,
                               aborted_at_s=T0 + 5.0,
                               abort_reason="collision"))
    return run


def not_feasible_branch(tmp_path):
    run = tmp_path / "notfeasible"
    write_branch(run, [not_feasible_row()],
                 branch_record("not_feasible", members=None,
                               wall_speed_mps=None, request_s=None))
    return run


def positions_at(t):
    """Two member CAVs + one HDV; only the members may ever be consulted."""
    return {"cav.a": (1200.0 + 15.0 * (t - T0), 0),
            "cav.c": (1240.0 + 15.0 * (t - T0), 1),
            "hdv.9": (0.0, 2)}


# --------------------------------------------------------- lifecycles ----

def test_lifecycle_assemble_hold_release(tmp_path):
    s = W.WallEpisodeSet.from_branch_dir(released_branch(tmp_path))
    assert len(s) == 1
    # requested at the request tick
    st = s.walls_at(T0, positions=positions_at(T0), cav_roster={"cav.a",
                                                                "cav.c"})
    assert st[0]["phase"] == "requested"
    assert st[0]["planned_hold_remaining_s"] == pytest.approx(60.0)
    # assembling
    st = s.walls_at(T0 + 2.0, positions=positions_at(T0 + 2.0),
                    cav_roster={"cav.a", "cav.c"})[0]
    assert st["phase"] == "assembling"
    assert st["time_in_phase_s"] == pytest.approx(1.0)
    assert st["planned_hold_remaining_s"] == pytest.approx(60.0)
    # holding: remaining counts from the assembly completion
    st = s.walls_at(T0 + 13.0, positions=positions_at(T0 + 13.0),
                    cav_roster={"cav.a", "cav.c"})[0]
    assert st["phase"] == "holding"
    assert st["time_in_phase_s"] == pytest.approx(10.0)
    assert st["planned_hold_remaining_s"] == pytest.approx(50.0)
    assert st["speed_mps"] == pytest.approx(17.0)
    assert st["n_members"] == 2
    assert st["mean_s_down_m"] == pytest.approx(1415.0)   # (1395 + 1435)/2
    assert st["lane_span"] == pytest.approx(1.0)          # lanes 0,1
    # terminal phase shows at its own tick, then the wall is dropped
    st = s.walls_at(T0 + 63.0, positions=positions_at(T0 + 63.0))[0]
    assert st["phase"] == "released"
    assert s.walls_at(T0 + 63.5) == []
    # members are locked until the handback, released afterwards
    assert set(s.member_flags(["cav.a", "cav.c", "cav.z"], T0 + 10.0)) == \
        {"cav.a", "cav.c"}
    assert s.member_flags(["cav.a", "cav.c"], T0 + 63.5) == {}


def test_lifecycle_abort_in_assembly(tmp_path):
    s = W.WallEpisodeSet.from_branch_dir(aborted_assembly_branch(tmp_path))
    st = s.walls_at(T0 + 3.0)[0]
    assert st["phase"] == "assembling"
    assert st["planned_hold_remaining_s"] == pytest.approx(60.0)
    st = s.walls_at(T0 + 5.0)[0]
    assert st["phase"] == "aborted"
    assert s.walls_at(T0 + 5.5) == []
    assert s.member_flags(["cav.a", "cav.c"], T0 + 5.5) == {}


def test_lifecycle_not_feasible_never_yields_a_wall(tmp_path):
    s = W.WallEpisodeSet.from_branch_dir(not_feasible_branch(tmp_path))
    assert len(s) == 1                      # episode recorded
    for t in (T0, T0 + 100.0, T0 + 121.0, T0 + 200.0):
        assert s.walls_at(t) == []
        assert s.member_flags(["cav.a", "cav.c"], t) == {}


def test_wall_before_request_is_invisible(tmp_path):
    s = W.WallEpisodeSet.from_branch_dir(released_branch(tmp_path))
    assert s.walls_at(T0 - 0.5) == []


# ------------------------------------------------------ leakage / audit --

def test_no_future_or_outcome_leakage_shuffle_after_t(tmp_path):
    """Brief item 2: shuffle the events after t -- the features at t are
    unchanged.  Also randomising branch_record's OUTCOME fields changes
    nothing (outcomes are never features)."""
    run = released_branch(tmp_path)
    s = W.WallEpisodeSet.from_branch_dir(run)
    t = T0 + 13.0
    base = [dict(w) for w in
            s.walls_at(t, positions=positions_at(t),
                       cav_roster={"cav.a", "cav.c"})]
    tok_base = F._wall_token_row(base[0])

    # shuffle every event strictly after t (and the outcome record)
    events = [json.loads(line) for line in
              (run / "run" / "wall_events.jsonl")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    tail = [e for e in events if float(e["timestamp_s"]) > t]
    head = [e for e in events if float(e["timestamp_s"]) <= t]
    rng = np.random.default_rng(0)
    for perm in range(3):
        tail_shuf = [tail[i] for i in
                     rng.permutation(len(tail)).astype(int)] if tail else []
        rec = branch_record("assembled_released",
                            assembled_at_s=T0 + 3.0,
                            released_at_s=T0 + 63.0)
        rec["outcome"] = "aborted_in_hold"      # corrupted on purpose
        rec["released_at_s"] = 9999.0
        write_branch(run, head + tail_shuf, rec)
        s2 = W.WallEpisodeSet.from_branch_dir(run)
        got = s2.walls_at(t, positions=positions_at(t),
                          cav_roster={"cav.a", "cav.c"})
        assert [dict(w) for w in got] == base
        assert np.array_equal(F._wall_token_row(got[0]), tok_base)


def test_member_geometry_uses_member_cav_rows_only(tmp_path):
    """Corrupting every non-member (HDV) position leaves the wall token
    unchanged; a non-CAV MEMBER is a loud contract error."""
    s = W.WallEpisodeSet.from_branch_dir(released_branch(tmp_path))
    t = T0 + 13.0
    pos = positions_at(t)
    base = s.walls_at(t, positions=pos, cav_roster={"cav.a", "cav.c"})[0]
    pos_bad = dict(pos, hdv9=(-999.0, -1), hdvX=(3.0, 3.0))
    got = s.walls_at(t, positions=pos_bad,
                     cav_roster={"cav.a", "cav.c"})[0]
    assert got == base
    with pytest.raises(W.WallTokenError, match="non-CAV member"):
        s.walls_at(t, positions=dict(pos, **{"hdv.9": (1.0, 2)}),
                   cav_roster={"cav.c"})     # cav.a missing from roster
    # absent members drop out of the mean instead of inventing positions
    got = s.walls_at(t, positions={"cav.c": pos["cav.c"]},
                     cav_roster={"cav.a", "cav.c"})[0]
    assert got["mean_s_down_m"] == pytest.approx(pos["cav.c"][0])
    assert got["n_members"] == 2            # planned count is planner-known


def test_absent_events_file_yields_empty_set(tmp_path):
    run = tmp_path / "empty"
    (run / "run").mkdir(parents=True)
    s = W.WallEpisodeSet.from_branch_dir(run)
    assert len(s) == 0 and s.walls_at(T0) == []


# ------------------------------------------------- featurizer wiring ----

def _synth_rt(synth_world):
    import v0lib as V
    rt = V.build_run_trace("wsyntm000p02t240", "base",
                           synth_world["runs"] /
                           "wsyntm000p02t240__base",
                           synth_world["t0"], t_ins_pre={})
    rt.demand_loading = 2700
    return rt


def test_featurize_at_wall_member_tokens(synth_world, geom):
    """Member CAV tokens carry member=1 + the wall's phase/speed/time in
    phase; non-members keep their own action; the wall token row matches
    the field table (inline episode on the synthetic trace window)."""
    from actions_v04 import ACTIONS_V04
    rt = _synth_rt(synth_world)
    ep = W._WallEp(WID)
    ep.observe(request_row(ts=100.0, members=("a", "c"), speed=17.0,
                           hold=60.0))
    ep.observe(wall_event("assembly_start", 101.0))
    ep.observe(wall_event("assembled", 103.0, members=["a", "c"]))
    ep.observe(wall_event("released", 170.0))
    walls = W.WallEpisodeSet({WID: ep}, {})
    t = 150.0                                  # holding; both CAVs present
    out = F.featurize_at(rt, t, geom, walls=walls)
    tok, msk = out["cav_tokens"], out["cav_mask"]
    k = int(msk.sum())
    assert k == 2                              # the synthetic CAVs a and c
    ids = tok[:k, 5:5 + len(ACTIONS_V04)].argmax(axis=1)
    assert (ids == ACTIONS_V04.index("wall")).all()
    rows = tok[:k]
    assert np.isclose(rows[:, 16], 17.0 / 40.0, atol=1e-6).all()
    assert (rows[:, 18] == 1.0).all()
    assert (rows[:, 10:16].argmax(axis=1) ==
            1 + W.WALL_PHASES.index("holding")).all()
    assert np.isclose(rows[:, 19], 13.0 / 600.0, atol=1e-6).all()  # 60-47
    assert out["wall_mask"].sum() == 1.0
    wrow = out["wall_tokens"][0]
    assert wrow[W.WALL_PHASES.index("holding")] == 1.0
    assert np.isclose(wrow[5], 17.0 / 40.0, atol=1e-6)
    assert np.isclose(wrow[6], 47.0 / 600.0, atol=1e-6)  # in hold: 150-103
    assert np.isclose(wrow[7], 13.0 / 600.0, atol=1e-6)
    assert np.isclose(wrow[8], 2.0 / 4.0, atol=1e-6)
    assert np.isclose(wrow[9], 2521.25 / 1000.0, atol=1e-6)
    assert np.isclose(wrow[10], 0.5, atol=1e-6)     # lanes 0 and 1
    # after the handback the members are back to plain tokens
    out = F.featurize_at(rt, 175.0, geom, walls=walls)
    tok = out["cav_tokens"][:int(out["cav_mask"].sum())]
    assert out["wall_mask"].sum() == 0.0
    assert (tok[:, 18] == 0.0).all()
    assert (tok[:, 10] == 1.0).all()                # phase = none again


def test_featurize_at_wall_tokens_absent_without_pilot(synth_world, geom):
    rt = _synth_rt(synth_world)
    out = F.featurize_at(rt, 250.0, geom)
    assert (out["wall_tokens"] == 0.0).all()
    assert (out["wall_mask"] == 0.0).all()
