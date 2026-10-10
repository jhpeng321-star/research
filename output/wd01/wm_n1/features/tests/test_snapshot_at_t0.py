"""``build_shard(snapshot_at_t0=...)`` re-aligns a full-run arm to t0.

Default OFF reproduces the frozen origin-0 full-run snapshot; ON places the
snapshot origin at the item's t0 (first physics row t0+0.5 s, like a branch
arm).  Synthesizes a small full run (no SUMO).
"""
from __future__ import annotations

import numpy as np

from features import featurizer as F
from synth_features import MAIN_LEFT, plan_row, row, veh, write_run


def _full_rows(t_end=10.0):
    rows = []
    t = 0.5
    while t <= t_end + 1e-9:
        rows.append(row(t, {"mainline.0": veh(MAIN_LEFT, 5.0 + t, 10.0)},
                        departed=("mainline.0",)))
        t += F.C.DT_S
    return rows


def _build(tmp_path, *, snapshot, t0):
    root = write_run(tmp_path / ("s" if snapshot else "d") / "base",
                     _full_rows())
    return F.build_shard("chunk1", "wsynth", "base", root, split="TRAIN",
                         plan_row=plan_row(t0), outcomes_root=None,
                         format="v3", policy="damping",
                         snapshot_at_t0=snapshot)


def test_default_base_still_origin_zero(tmp_path):
    sh = _build(tmp_path, snapshot=False, t0=4.0)
    m = sh["meta"]
    assert m["t_origin"] == 0.0
    assert "snapshot_at_t0" not in m
    nz = np.flatnonzero(np.asarray(sh["veh_mask"]).any(axis=1))
    assert int(nz[0]) == 1 and 0.0 + int(nz[0]) * F.C.DT_S == 0.5
    # the frozen full-run grid runs to t_end from origin 0
    assert m["n_ticks"] == int(round((10.0 - 0.0) / F.C.DT_S)) + 1
    assert m["decision_t"][0] == 4.0


def test_snapshot_at_t0_realigns(tmp_path):
    sh = _build(tmp_path, snapshot=True, t0=4.0)
    m = sh["meta"]
    assert m["t_origin"] == 4.0
    assert m["snapshot_at_t0"] is True
    nz = np.flatnonzero(np.asarray(sh["veh_mask"]).any(axis=1))
    assert int(nz[0]) == 1
    assert 4.0 + int(nz[0]) * F.C.DT_S == 4.5  # first physics row t0+0.5
    assert m["n_ticks"] == int(round((10.0 - 4.0) / F.C.DT_S)) + 1
    assert m["decision_t"][0] == 4.0


def test_default_bytes_are_stable_and_differ_from_realigned(tmp_path):
    a = _build(tmp_path, snapshot=False, t0=4.0)
    b = _build(tmp_path, snapshot=False, t0=4.0)
    assert np.array_equal(np.asarray(a["veh_step"]), np.asarray(b["veh_step"]))
    c = _build(tmp_path, snapshot=True, t0=4.0)
    assert a["veh_step"].shape[0] != c["veh_step"].shape[0]
