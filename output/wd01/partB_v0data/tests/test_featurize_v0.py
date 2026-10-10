"""Featurizer tests: shapes, dtypes, geometry mask, demand vector, context
slots, and the no-vType guarantee."""
from __future__ import annotations

import numpy as np
import pytest

import featurize_v0 as F
import v0lib as V


def test_feature_shapes_and_dtypes(synth_world, geom):
    t0 = synth_world["t0"]
    rows = synth_world["runs"] / "wsyntm000p02t240__native"
    rt = V.build_run_trace("wsyntm000p02t240", "native", rows, t0,
                           t_ins_pre={})
    rt.demand_loading = 2700
    ts = V.decision_grid(t0, rt.data_end, "native")
    tau_avail = rt.data_end - ts
    feats = F.build_features(rt, ts, tau_avail, geom)
    n = len(ts)
    assert feats["lane_bins"].shape == (n, 3, F.LMAX_BINS, 3)
    assert feats["lane_bins"].dtype == np.float32
    assert feats["lane_mask"].shape == (n, 3, F.LMAX_BINS)
    assert feats["cav_tokens"].shape == (n, F.MAX_CAVS, F.CAV_RAW)
    assert feats["cav_mask"].shape == (n, F.MAX_CAVS)
    assert feats["wall_tokens"].shape == (n, F.MAX_WALLS, F.WALL_RAW)
    assert feats["demand"].shape == (n, F.DEMAND_RAW)
    assert feats["demand"].dtype == np.float32
    assert feats["context"].shape == (n, F.CTX_RAW)
    assert feats["t"].shape == (n,) and feats["tau_available"].shape == (n,)
    # skeleton-compatible block sizes (v0.4 token widths: BRIEF Part B
    # wall/action inputs items 1-2 -- bumped ONLY in part4/partB/featurize)
    assert (F.LMAX_BINS, F.CAV_RAW, F.WALL_RAW, F.DEMAND_RAW, F.CTX_RAW) == \
        (320, 20, 11, 21, 8)
    # not all-empty: snapshots populated
    assert (feats["raw_off"][1:] - feats["raw_off"][:-1]).min() > 0
    assert feats["cav_mask"].sum() > 0


def test_geometry_mask_matches_lane_lengths(geom):
    m = F.geometry_mask(geom)
    assert m.shape == (3, F.LMAX_BINS)
    for lane in (0, 1, 2):
        nbin = int(np.ceil(geom.length(lane) / 10.0))
        assert m[lane, :nbin].all() and not m[lane, nbin:].any()
    assert m.sum() == 320 + 320 + 46      # 3199.0, 3197.55, 454.1 m


def test_demand_vector_counts_scheduled_arrivals(synth_world, synth_base_trace,
                                                 geom):
    rt = synth_base_trace
    d = F._demand_vector(rt, 200.0)
    grid = d[:20].reshape(10, 2)
    # only f (t=295, mainline) is scheduled after t=200 within 600 s
    assert grid[1, 0] == 1.0 / 50.0 and grid.sum() == 1.0 / 50.0
    assert d[20] == 0.0                   # f's designation unknown -> share 0


def test_context_slots(synth_world, synth_base_trace):
    rt = synth_base_trace
    rt.demand_loading = 2700
    ctx = F._context_vector(rt, 250.0, (3.0, 1.0))
    assert ctx[0] == pytest.approx(250.0 / 1000.0)
    assert ctx[1] == pytest.approx(10.0 / 600.0)         # t - t0 = 10
    assert ctx[2] == 0.0                                  # native: no pre-t0
    assert ctx[3] == pytest.approx(2700.0 / 2700.0)
    assert ctx[4] == 0.0 and ctx[5] == 0.0
    assert ctx[6] == pytest.approx(3.0 / 50.0)
    assert ctx[7] == pytest.approx(1.0 / 50.0)
    # base source: time-to-branch slot is positive before t0
    ctx_b = F._context_vector(rt, 150.0, (0.0, 0.0))
    assert ctx_b[2] == pytest.approx(90.0 / 600.0)


def test_cav_token_layout_default_unknown(synth_world, synth_base_trace,
                                          geom):
    """Without an action timeline every CAV carries the explicit
    'unknown' action slot -- never a silent 'native' (brief item 1)."""
    from actions_v04 import ACTIONS_V04
    rt = synth_base_trace
    out = F.featurize_at(rt, 150.0, geom)
    tok, msk = out["cav_tokens"], out["cav_mask"]
    k = int(msk.sum())
    assert k == out["n_cav"] > 0
    rows = tok[:k]
    assert (rows[:, 5 + ACTIONS_V04.index("unknown")] == 1.0).all()
    for a in ("native", "damping", "pi", "wall"):
        assert (rows[:, 5 + ACTIONS_V04.index(a)] == 0.0).all()
    assert (rows[:, 10] == 1.0).all()     # phase = none (slot after actions)
    assert (rows[:, 16:] == 0.0).all()    # no wall quantities without walls


def test_cav_token_layout_with_actions_timeline(synth_world,
                                                synth_base_trace, geom):
    """Per-CAV actions from an ActionTimeline land in the one-hot slots
    (and stay 'unknown' before the CAV's first evidence)."""
    from actions_v04 import ACTIONS_V04, ActionTimeline
    rt = synth_base_trace
    rt.demand_loading = 2700
    tl = ActionTimeline(
        {"a": [(150.0, ACTIONS_V04.index("damping")),
               (175.0, ACTIONS_V04.index("native"))],
         "c": [(150.0, ACTIONS_V04.index("pi"))]},
        fleet_policy=None, command_owner_counts={},
        n_schedule_events=3, n_command_rows=0, sources={"test": "inline"})
    out = F.featurize_at(rt, 140.0, geom, actions=tl)   # before any event
    tok = out["cav_tokens"][:int(out["cav_mask"].sum())]
    assert (tok[:, 5 + ACTIONS_V04.index("unknown")] == 1.0).all()
    out = F.featurize_at(rt, 190.0, geom, actions=tl)   # after a's switch
    tok = out["cav_tokens"][:int(out["cav_mask"].sum())]
    ids = tok[:, 5:5 + len(ACTIONS_V04)].argmax(axis=1)
    # CAV token order is sorted by vid index; 'a' inserted at t=1 (idx 0)
    # and 'c' at t=40 (idx 2) in the synthetic fixture
    assert ACTIONS_V04[int(ids[0])] == "native"
    assert ACTIONS_V04[int(ids[1])] == "pi"


def test_no_vtype_anywhere(synth_world, synth_base_trace, geom):
    rt = synth_base_trace
    rt.demand_loading = 2000
    ts = V.decision_grid(rt.world_t0, rt.data_end, rt.source)
    feats = F.build_features(rt, ts, rt.data_end - ts, geom)
    for key in feats:
        assert "type" not in key.lower()
    # physics type_ids are read for the trace but never featurised: the
    # raw block carries no type column
    assert not any(k.startswith("raw_type") for k in feats)
