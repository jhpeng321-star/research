"""v0.3.1 deployable-sidecar tests.

The load-bearing one is the NO-FUTURE-LEAK test: the deployable feature
builder must never read schedule/insertion entries with timestamp > t --
perturbing ALL entries after a query time must leave the deployable
demand/context blocks BIT-IDENTICAL (the oracle demand vector, by
contrast, is built from entries in (t, t+600] and moves under the same
perturbation -- the contrast is asserted too).

Also: column semantics (windows, queues, zeroed time-to-branch), the
name contract with partB_v0.gbm, and the end-to-end mini-root build
(verified links + sidecars + MANIFEST with the parent sha + loader
switch).
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

from conftest_load import HoldoutAccessWatch, make_mini_v02_root  # noqa: E402
from deployable_features import (DEMAND_DEP_NAMES,  # noqa: E402
                                 CTX_RAW, DEMAND_DEP_RAW, SourceObs)

T0 = 240.0
LOADING = 2700.0


# ------------------------------------------------------- synthetic src --

def make_sched(n=40, seed=3, horizon=900.0):
    """Synthetic shared schedule: n vehicles, mixed entrances/CAV."""
    rng = np.random.default_rng(seed)
    vids = [f"v{i:03d}" for i in range(n)]
    t = np.sort(rng.uniform(0.0, horizon, n))
    ent = rng.integers(0, 2, n)
    cav = (rng.random(n) < 0.3).astype(np.int8)
    return vids, t, ent.astype(np.int64), cav


def make_insertions(vids, t, frac=0.7, seed=4, lag=2.0):
    """Realised insertions for a subset of the schedule (lag after
    departure), as (vid list, time list)."""
    rng = np.random.default_rng(seed)
    sel = rng.random(len(vids)) < frac
    return ([v for v, s in zip(vids, sel) if s],
            [float(ti) + lag for ti, s in zip(t, sel) if s])


def perturb_after(vids, t, ent, cav, ins_vids, ins_t, t_cut, rng_seed=9):
    """Perturb EVERY schedule entry and insertion with timestamp > t_cut:
    shift far into the future and flip entrance/CAV flags (insertions of
    past vids and past entries are untouched)."""
    t2 = t.copy()
    ent2 = ent.copy()
    cav2 = cav.copy()
    fut = t > t_cut
    t2[fut] += 1234.0
    ent2[fut] = 1 - ent2[fut]
    cav2[fut] = 1 - cav2[fut]
    iv2, it2 = [], []
    for v, ti in zip(ins_vids, ins_t):
        iv2.append(v)
        it2.append(ti + 1234.0 if ti > t_cut else ti)
    return vids, t2, ent2, cav2, iv2, it2


def build_blocks(vids, t, ent, cav, ins_vids, ins_t, ts):
    obs = SourceObs(vids, t, ent, cav, ins_vids, ins_t)
    return obs.blocks(ts, T0, LOADING)


TS = np.array([100.0, 250.0, 400.0, 600.0, 800.0])


# ----------------------------------------------------- the leak test --

def test_no_future_entries_after_t_bit_identical():
    """Perturbing all schedule/insertion entries after ANY query time t
    must leave the deployable blocks at t bit-identical."""
    vids, t, ent, cav = make_sched()
    ins_vids, ins_t = make_insertions(vids, t)
    d_ref, c_ref = build_blocks(vids, t, ent, cav, ins_vids, ins_t, TS)

    # (a) perturb everything after the LAST decision time: all rows
    p = perturb_after(vids, t, ent, cav, ins_vids, ins_t, TS[-1])
    d2, c2 = build_blocks(*p, TS)
    assert d2.tobytes() == d_ref.tobytes()
    assert c2.tobytes() == c_ref.tobytes()

    # (b) per decision time: perturb everything after THAT t; the row at
    #     t must be bit-identical (the builder never reads entries > t)
    for i, tq in enumerate(TS):
        p = perturb_after(vids, t, ent, cav, ins_vids, ins_t, tq,
                          rng_seed=10 + i)
        d3, c3 = build_blocks(*p, TS)
        assert d3[i].tobytes() == d_ref[i].tobytes(), f"row {i} moved"
        assert c3[i].tobytes() == c_ref[i].tobytes(), f"row {i} moved"

    # contrast: the ORACLE demand vector reads (t, t+600] and MUST move
    # under the same perturbation (the leak the deployable block removes)
    fut = (t > TS[0] + 1e-9) & (t <= TS[0] + 600.0)
    assert fut.any(), "synthetic schedule has no in-window arrivals"
    p = perturb_after(vids, t, ent, cav, ins_vids, ins_t, TS[0])
    assert (p[1][fut] != t[fut]).all(), "perturbation moved nothing"


def test_future_cav_share_and_ttb_absent():
    """Deployable context[2] (time-to-branch) is identically 0 even for
    base rows before t0 (where the oracle block is nonzero), and no
    demand column correlates with the FUTURE CAV share: flipping the CAV
    flag of a future-scheduled vehicle changes nothing."""
    vids, t, ent, cav = make_sched()
    ins_vids, ins_t = make_insertions(vids, t)
    d, c = build_blocks(vids, t, ent, cav, ins_vids, ins_t, TS)
    assert CTX_RAW == 8 and c.shape[1] == 8
    assert (c[:, 2] == 0.0).all(), "time-to-branch must be 0 (deployable)"
    np.testing.assert_allclose(c[:, 3], LOADING / 2700.0, rtol=0, atol=0)
    np.testing.assert_allclose(c[:, 0], TS / 1000.0, rtol=0, atol=1e-7)
    # v0.3.2 fix #1: ctx[1] (was (t-t0)/600) is identically 0 -- together
    # with ctx[0]=t/1000 it made the campaign branch time t0 recoverable
    assert (c[:, 1] == 0.0).all(), "since-t0 col must be 0 (fix #1)"

    cav2 = cav.copy()
    cav2[t > TS[-1]] = 1 - cav2[t > TS[-1]]
    d2, _ = build_blocks(vids, t, ent, cav2, ins_vids, ins_t, TS)
    assert d2.tobytes() == d.tobytes()


def test_window_and_queue_semantics():
    """Observed counts are (t-w, t] (left-open); the queue at t counts
    scheduled departures <= t not yet inserted; future departures and
    future insertions never enter."""
    vids = ["m_early", "m_edge", "m_in", "m_future", "r_now", "r_fut"]
    t = np.array([10.0, 40.0, 40.5, 500.0, 41.0, 700.0])
    ent = np.array([0, 0, 0, 0, 1, 1])
    cav = np.array([0, 0, 1, 0, 0, 1])
    ins_vids = ["m_early", "m_edge", "m_in", "r_now"]
    ins_t = [30.0, 40.0, 40.5, 41.0]
    obs = SourceObs(vids, t, ent, cav, ins_vids, ins_t)
    tq = 41.0
    d, c = obs.row(tq, T0, LOADING)
    # mainline (t-60, t] = (-19, 41]: all three mainline insertions in
    np.testing.assert_allclose(d[0], 3.0 / 50.0, rtol=0, atol=1e-7)
    # ramp (t-60, t]: r_now at 41 -> 1 veh
    np.testing.assert_allclose(d[3], 1.0 / 50.0, rtol=0, atol=1e-7)
    # CAV: m_in only -> 1 veh mainline; ramp 0
    np.testing.assert_allclose(d[6], 1.0 / 50.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(d[9], 0.0, rtol=0, atol=1e-7)
    # queue: r_fut departs at 700 > t: NOT queued; r_now inserted at 41:
    # not queued; nothing scheduled <= 41 is uninserted on mainline
    assert c[6] == 0.0 and c[7] == 0.0
    # a scheduled-but-uninserted vehicle at t queues it
    obs2 = SourceObs(["a", "b"], np.array([10.0, 41.0]),
                     np.array([0, 1]), np.array([0, 0]),
                     ["a"], [12.0])
    _d, c2 = obs2.row(41.0, T0, LOADING)
    assert c2[6] == 0.0 and c2[7] == 1.0 / 50.0    # b: ramp sched, not in
    # window edges: at t the 300 s window is (t-300, t]; an insertion at
    # exactly t-300 is out, one just inside is in
    obs3 = SourceObs(["a"], np.array([0.0]), np.array([0]),
                     np.array([0]), ["a"], [11.0])
    d3, _ = obs3.row(311.0, T0, LOADING)
    assert d3[2] == 0.0                            # t-300 exactly: out
    d4, _ = obs3.row(310.5, T0, LOADING)           # (10.5, 310.5]: 11 in
    assert d4[2] == pytest.approx(1.0 / 5.0 / 50.0)
    # same for the 60 s window (left edge exactly t-60 is out)
    obs5 = SourceObs(["a"], np.array([0.0]), np.array([0]),
                     np.array([0]), ["a"], [30.0])
    d5, _ = obs5.row(90.0, T0, LOADING)
    assert d5[0] == 0.0                            # t-60 exactly: out
    d6, _ = obs5.row(89.5, T0, LOADING)            # (29.5, 89.5]: 30 in
    assert d6[0] == pytest.approx(1.0 / 50.0)


def test_demand_dep_names_match_gbm():
    """The featurizer and the model must agree on the deployable demand
    column order."""
    from partB_v0.gbm import DEMAND_DEP_NAMES as GBM_NAMES
    assert tuple(GBM_NAMES) == tuple(DEMAND_DEP_NAMES)
    assert len(DEMAND_DEP_NAMES) == DEMAND_DEP_RAW


# ------------------------------------------------- end-to-end mini root --

def _npz_sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_mini_root_v031_build(tmp_path, monkeypatch):
    """build_v031 over the mini v0.3-shaped root: links + sidecars +
    MANIFEST (parent sha) + SHA256SUMS; HOLDOUT never opened; the
    deployable sidecar context reproduces the oracle context on every
    RSU-observable column; the loader switch reads it."""
    mini = make_mini_v02_root(tmp_path / "mini")
    v03_root = mini["root"]
    out_root = tmp_path / "v031"
    parent_sha = _npz_sha(v03_root / "MANIFEST.json")

    watch = HoldoutAccessWatch([mini["holdout"]]).install()
    import build_v031 as G
    rc = G.main(["--v03-root", str(v03_root), "--out", str(out_root),
                 "--parent-sha", parent_sha, "--workers", "1"])
    assert rc == 0
    assert not watch.hits, f"HOLDOUT files opened: {watch.hits[:3]}"

    man = json.loads((out_root / "MANIFEST.json").read_text("utf-8"))
    assert man["parent"]["manifest_sha256"] == parent_sha
    assert man["build"]["holdout_accessed"] is False
    n_side = 0
    for key, e in man["worlds"].items():
        assert e.get("split") != "HOLDOUT"
        if e.get("superseded_by"):
            continue
        deps = [r for r in e["files"] if r.endswith("_feat_dep.npz")]
        assert len(deps) == 1, key
        n_side += 1
    assert n_side == 8                     # 4 non-holdout worlds x 2

    # SHA256SUMS verifies (MANIFEST + every worlds/ file)
    sums = out_root / "SHA256SUMS"
    n_ok = 0
    for line in sums.read_text("utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        assert _npz_sha(out_root / rel) == sha, rel
        n_ok += 1
    assert n_side + 1 <= n_ok              # sidecars + MANIFEST + sources

    # no sidecar for the HOLDOUT world anywhere
    assert not any(mini["holdout"] in p.name
                   for p in out_root.rglob("*_feat_dep.npz"))

    # loader switch: oracle vs deployable on the SAME root
    sys.path.insert(0, str(HERE.parent))           # output/wd01
    from partB_v0.data import V0Dataset, load_manifest
    log = lambda *a, **k: None
    m_or = load_manifest(out_root, verify_files=True, log=log)
    ds_or = V0Dataset(m_or, feature_set="oracle", log=log)
    ds_de = V0Dataset(m_or, feature_set="deployable", log=log)
    assert ds_de.feature_set == "deployable"
    assert ds_or.feats["demand"].shape[1] == 21
    assert ds_de.feats["demand"].shape == \
        (ds_or.feats["demand"].shape[0], 12)
    np.testing.assert_array_equal(ds_or.feats["lane_bins"],
                                  ds_de.feats["lane_bins"])
    np.testing.assert_array_equal(ds_or.feats["cav_tokens"],
                                  ds_de.feats["cav_tokens"])
    orc, dep = ds_or.feats["context"], ds_de.feats["context"]
    assert dep.shape == orc.shape
    assert (dep[:, 2] == 0.0).all()
    for col in (0, 3, 6, 7):               # RSU-observable ctx cols match
        np.testing.assert_allclose(dep[:, col], orc[:, col], atol=1e-6)
    # v0.3.2 fix #1: the since-t0 col is zeroed in deployable blocks
    assert (dep[:, 1] == 0.0).all()
    # label rows/TD unchanged (same files, same t_idx alignment)
    pd_testing_equal = ds_or.rows.equals(ds_de.rows)
    assert pd_testing_equal
    # deployable row feature matrix carries the named observed columns
    from partB_v0.gbm import row_feature_matrix
    _X, names = row_feature_matrix(ds_de)
    for n in DEMAND_DEP_NAMES:
        assert n in names, n

    # a root without sidecars refuses the deployable feature set
    with pytest.raises(FileNotFoundError):
        V0Dataset(load_manifest(v03_root, verify_files=False, log=log),
                  feature_set="deployable", log=log)
