"""Label-machinery tests: prefix-sum windows vs the part4 delay library,
entry-waiting semantics, TD additivity, and cav/speed-mode bookkeeping
(synthetic mini-world; no simulation, no real data required)."""
from __future__ import annotations

import numpy as np
import pytest

import v0lib as V
from delay import core

TOL = 1e-9


def _library_window(rt, t, tau, geom):
    k0 = int(round((t - rt.t_origin) / V.DT))
    kH = k0 + int(round(tau / V.DT))
    veh = []
    for vi in range(len(rt.vid_table)):
        lo, hi = int(rt.start[vi]), int(rt.end[vi])
        a = np.searchsorted(rt.tick[lo:hi], k0, "left")
        b = np.searchsorted(rt.tick[lo:hi], kH, "right")
        if b <= a:
            continue
        sel = slice(lo + a, lo + b)
        tr = core.VehicleTrace(vid=str(rt.vid_table[vi]),
                               t=rt.t_origin + rt.tick[sel] * V.DT,
                               lane=rt.lane[sel], s=rt.s[sel],
                               vehicle_type=-1)
        veh.append(core.vehicle_delay(tr, float(t), tau, geom, V.DT,
                                      strict=False))
    sched_w = {vid: (td, ent) for vid, td, ent in
               zip(rt.sched_vid, rt.sched_t, rt.sched_ent)
               if td <= t + tau + core.EPS}
    ins = dict(rt.t_ins_pre)
    for vid, ti in rt.t_ins.items():
        if ti <= t + tau + core.EPS:
            ins[vid] = ti
    waits, _ = core.waiting_component(sched_w, ins, float(t), tau)
    return core.window_totals(veh, waits, float(t), tau)


def test_label_matches_delay_library_synthetic(synth_base_trace, geom):
    rt = synth_base_trace
    checked = 0
    for t in (130.0, 150.0, 200.0, 250.0, 295.0):
        ta = rt.data_end - t
        for tau in V.TAUS:
            if tau > ta + 1e-9:
                continue
            mine = V.window_label(rt, t, tau)
            lib = _library_window(rt, t, tau, geom)
            assert abs(mine["time_on_section_s"]
                       - lib["time_on_section_s"]) < TOL
            assert abs(mine["free_flow_ref_s"]
                       - lib["free_flow_ref_s"]) < TOL
            assert abs(mine["traversed_m"] - lib["traversed_m"]) < TOL
            assert abs(mine["wait_mainline_s"] - lib["wait_mainline_s"]) < TOL
            assert abs(mine["wait_ramp_s"] - lib["wait_ramp_s"]) < TOL
            assert abs(mine["total_delay_ref_s"]
                       - lib["total_delay_ref_s"]) < TOL
            checked += 1
    assert checked >= 10


def test_entry_waiting_semantics(synth_base_trace):
    rt = synth_base_trace
    # vehicle f: scheduled 295 s, never inserted.
    # window [280, 300): f contributes wait = 300 - 295 = 5 s (queued).
    m = V.window_label(rt, 280.0, 20.0)
    assert abs(m["wait_mainline_s"] - 5.0) < TOL
    # window [130, 160): f's depart is beyond the horizon -> no contribution
    m = V.window_label(rt, 130.0, 30.0)
    assert m["wait_mainline_s"] == 0.0 and m["wait_ramp_s"] == 0.0
    # window [285, 300): wait 5 s again, all from f
    m = V.window_label(rt, 285.0, 15.0)
    assert abs(m["wait_mainline_s"] - 5.0) < TOL
    # b: scheduled 99, inserted 101 -> pre-window for t >= 120: wait 0
    m = V.window_label(rt, 130.0, 30.0)
    assert m["wait_mainline_s"] == 0.0


def test_td_additivity_synthetic(synth_world, geom):
    t0 = synth_world["t0"]
    rows = synth_world["runs"] / "wsyntm000p02t240__native"
    rt = V.build_run_trace("wsyntm000p02t240", "native", rows, t0,
                           t_ins_pre={})
    L = {}
    for t in V.decision_grid(t0, rt.data_end, "native"):
        ta = rt.data_end - t
        for tau in V.TAUS:
            if tau <= ta + 1e-9:
                L[(float(t), tau)] = \
                    V.window_label(rt, float(t), tau)["total_delay_ref_s"]
    n = 0
    for (t, tau), y in L.items():
        if tau < 60.0:
            continue
        d30 = L.get((t, 30.0))
        nxt = L.get((t + 30.0, tau - 30.0))
        if d30 is None or nxt is None:
            continue
        assert abs(y - (d30 + nxt)) < 1e-6, (t, tau)
        n += 1
    assert n > 0


def test_cav_and_speedmode_bookkeeping(synth_base_trace):
    rt = synth_base_trace
    assert rt.stats["smode_values"] == [31.0]
    assert set(np.unique(rt.is_cav)) <= {0, 1}
    # designated CAVs are a and c; every record of theirs is flagged
    for vid in ("a", "c"):
        vi = int(np.flatnonzero(rt.vid_table == vid)[0])
        lo, hi = int(rt.start[vi]), int(rt.end[vi])
        assert rt.is_cav[lo:hi].all()
    for vid in ("b", "d", "e"):
        vi = int(np.flatnonzero(rt.vid_table == vid)[0])
        lo, hi = int(rt.start[vi]), int(rt.end[vi])
        assert not rt.is_cav[lo:hi].any()


def test_decision_grid_spec():
    # base over [120, t0+570], native over (t0, t0+570], capped at data end
    ts = V.decision_grid(360.0, 960.0, "base")
    assert ts[0] == 120.0 and ts[-1] == 930.0 and len(ts) == 82
    ts = V.decision_grid(360.0, 960.0, "native")
    assert ts[0] == 370.0 and ts[-1] == 930.0 and len(ts) == 57
    ts = V.decision_grid(240.0, 300.0, "base")
    assert ts[-1] <= 300.0 + 1e-9


def test_world_sources_bases_root_and_null_base_arm():
    """chunk1_fix27 resolution: bases in a separate root, plan base_arm
    null -> <world>__base; default (no bases_root) keeps v0.2 behaviour."""
    from pathlib import Path
    w = dict(world="w8465m2700p40t240", t0_s=240.0, base_arm=None)
    t0, runs = V.world_sources(w, Path("E:/rd/runs/fix27"),
                               Path("E:/rd/runs/verify27"))
    assert t0 == 240.0
    assert runs["base"] == Path("E:/rd/runs/verify27/w8465m2700p40t240__base")
    assert runs["native"] == Path("E:/rd/runs/fix27/w8465m2700p40t240__native")
    # no bases_root -> both arms under runs_root
    _, runs2 = V.world_sources(w, Path("E:/rd/runs/fix27"))
    assert runs2["base"] == Path("E:/rd/runs/fix27/w8465m2700p40t240__base")
    # explicit base_arm still honoured under runs_root (v0.2 plans)
    w3 = dict(world="wsynt", t0_s=240.0,
              base_arm=dict(arm="base", run="wsynt__damping"))
    _, runs3 = V.world_sources(w3, Path("E:/rd/runs/c1"))
    assert runs3["base"] == Path("E:/rd/runs/c1/wsynt__damping")
