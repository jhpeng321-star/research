"""v0.3.2 data-side fix tests (#1, #2, #3, #5) + the build_v032 builder.

#1  t0 non-recoverability: no deployable block may depend on the
    campaign branch time (ctx[1] identically 0; two sources differing
    only in t0 produce byte-identical deployable blocks).
#2  insertions.jsonl is never read by label/feature code (functional
    open-guard + the w8473-style derivation from the base physics pass).
#3  policy relabels: '__native'/'__pi' -> switching_mix; per-CAV actions
    only from switch_schedule.json / suffix_commands.jsonl -- no
    dispatcher_rows read anywhere.
#5  strict (HEADLINE) vs detector (SENSITIVITY) sensing variants.
"""
from __future__ import annotations

import builtins
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
                                 DEMAND_STRICT_RAW, STRICT_DEMAND_NAMES,
                                 VARIANTS, SourceObs)

T0 = 240.0
LOADING = 2700.0
TS = np.array([100.0, 250.0, 400.0, 600.0, 800.0])


def make_sched(n=40, seed=3, horizon=900.0):
    rng = np.random.default_rng(seed)
    vids = [f"v{i:03d}" for i in range(n)]
    t = np.sort(rng.uniform(0.0, horizon, n))
    ent = rng.integers(0, 2, n)
    cav = (rng.random(n) < 0.3).astype(np.int8)
    return vids, t, ent.astype(np.int64), cav


def make_insertions(vids, t, frac=0.7, seed=4, lag=2.0):
    rng = np.random.default_rng(seed)
    sel = rng.random(len(vids)) < frac
    return ([v for v, s in zip(vids, sel) if s],
            [float(ti) + lag for ti, s in zip(t, sel) if s])


# ------------------------------------------------------------------ #1 --

def test_fix1_t0_not_recoverable():
    """ctx[1] is identically 0 in BOTH variants and the whole deployable
    block is invariant to the campaign branch time: two sources that
    differ ONLY in t0 produce byte-identical deployable blocks, so t0
    cannot be recovered from any deployable feature."""
    vids, t, ent, cav = make_sched()
    ins_vids, ins_t = make_insertions(vids, t)
    for variant in VARIANTS:
        obs = SourceObs(vids, t, ent, cav, ins_vids, ins_t)
        d1, c1 = obs.blocks(TS, T0, LOADING, variant=variant)
        d2, c2 = obs.blocks(TS, T0 + 777.0, LOADING, variant=variant)
        assert d1.tobytes() == d2.tobytes(), variant
        assert c1.tobytes() == c2.tobytes(), variant
        assert (c1[:, 1] == 0.0).all(), variant
        # and no OTHER context column sneaks t0 back in either
        assert (c2[:, 1] == 0.0).all(), variant


def test_fix1_row_level_t0_invariance():
    """Row-level: SourceObs.row ignores t0 for every deployable column."""
    vids, t, ent, cav = make_sched(n=12, seed=5)
    ins_vids, ins_t = make_insertions(vids, t)
    obs = SourceObs(vids, t, ent, cav, ins_vids, ins_t)
    for tq in TS:
        for variant in VARIANTS:
            d1, c1 = obs.row(float(tq), T0, LOADING, variant=variant)
            d2, c2 = obs.row(float(tq), T0 + 1234.0, LOADING,
                             variant=variant)
            assert d1.tobytes() == d2.tobytes()
            assert c1.tobytes() == c2.tobytes()
    with pytest.raises(ValueError):
        obs.row(float(TS[0]), T0, LOADING, variant="oracle")


# ------------------------------------------------------------------ #2 --

def test_fix2_no_insertions_jsonl_in_label_feature_code():
    """Static: the label/feature modules contain no insertions.jsonl
    path or the removed fallback; the only derivation helper is the
    base-physics pass."""
    for mod in ("v0lib.py", "featurize_v0.py", "deployable_features.py",
                "actions_v04.py", "wall_tokens_v04.py"):
        src = (HERE.parent / mod).read_text(encoding="utf-8")
        assert "insertions.jsonl" not in src, mod
    b = (HERE.parent / "build_v0data.py").read_text(encoding="utf-8")
    assert "t_ins_pre_from_insertions_file" not in b
    assert "t_ins_pre_from_base_physics" in b


def test_fix2_labels_build_without_insertions_file(synth_world,
                                                   tmp_path,
                                                   monkeypatch):
    """Functional: a full label/feature build succeeds with
    builtins.open RAISING on any insertions.jsonl path (the w8473 root
    cause: the old fallback silently returned an empty history from
    that file)."""
    import featurize_v0 as F
    import v0lib as V
    import build_v0data as B

    real_open = builtins.open

    def guarded_open(file, *a, **k):
        if "insertions.jsonl" in str(file):
            raise AssertionError(
                f"label/feature code opened {file} (v0.3.2 fix #2)")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    runs = synth_world["runs"]
    t0 = synth_world["t0"]
    wname = "wsyntm000p02t240"
    base_rt = V.build_run_trace(wname, "base",
                                runs / f"{wname}__base", t0,
                                t_ins_pre={})
    t_pre = V.build_t_ins_pre(base_rt, t0)
    # the fallback-free derivation agrees with the cached-trace path
    t_pre2 = B.t_ins_pre_from_base_physics(runs / f"{wname}__base", t0)
    assert t_pre2 == t_pre
    rt = V.build_run_trace(wname, "native", runs / f"{wname}__native",
                           t0, t_ins_pre=t_pre)
    rt.demand_loading = 2700
    ts = V.decision_grid(t0, rt.data_end, "native")
    geom = V.cached_geometry()
    _feats = F.attach_meta(rt, F.build_features(rt, ts, ts + 570, geom))
    lab = V.window_label(rt, float(ts[0]), 300.0)
    assert np.isfinite(lab["total_delay_ref_s"])
    q = V.queue_lengths(rt, float(ts[0]))
    assert all(np.isfinite(x) for x in q)


def test_fix2_missing_base_insertions_yields_physics_history(
        synth_world, monkeypatch):
    """A base arm with NO insertions.jsonl at all (the w8473/verify27
    shape) still yields the shared pre-t0 history from departed_ids."""
    import build_v0data as B
    import v0lib as V
    runs = synth_world["runs"]
    t0 = synth_world["t0"]
    wname = "wsyntm000p02t240"
    arm = runs / f"{wname}__base"
    ins_p = arm / "run" / "insertions.jsonl"
    backup = None
    if ins_p.exists():
        backup = ins_p.read_bytes()
        ins_p.unlink()
    try:
        t_pre = B.t_ins_pre_from_base_physics(arm, t0)
        base_rt = V.build_run_trace(wname, "base", arm, t0, t_ins_pre={})
        assert t_pre == V.build_t_ins_pre(base_rt, t0)
        assert len(t_pre) > 0
    finally:
        if backup is not None:
            ins_p.write_bytes(backup)


# ------------------------------------------------------------------ #3 --

def test_fix3_policy_map_no_native_claim():
    """POLICY_OF_SOURCE maps __native/__pi to switching_mix in BOTH the
    trainer and the builder, the shared map is equal, and the provenance
    prose makes no fleet-native claim for the campaign arms."""
    from partB_v0.data import POLICY_OF_SOURCE as TRAINER_MAP
    import build_v032 as G
    assert TRAINER_MAP == G.POLICY_OF_SOURCE
    assert TRAINER_MAP["native"] == "switching_mix"
    assert TRAINER_MAP["pi"] == "switching_mix"
    assert TRAINER_MAP["base"] == "damping_fleet"
    prose_native = V_SOURCE_POLICY()["native"]
    assert "switching" in prose_native.lower()
    assert "NOT a fleet-wide native/PI policy" in prose_native
    # the old false claim is gone
    assert "native-IDM branch" not in prose_native


def V_SOURCE_POLICY():
    import v0lib
    return v0lib.SOURCE_POLICY


def test_fix3_no_dispatcher_rows_read():
    """Per-CAV actions are attributed only from switch_schedule.json /
    suffix_commands.jsonl; no label/feature or builder module reads the
    dispatcher action log (the builder only carries the explicit
    dispatcher_rows_read=false attestation field)."""
    mods = ["v0lib.py", "featurize_v0.py", "deployable_features.py",
            "build_v0data.py", "build_v031.py", "actions_v04.py",
            "wall_tokens_v04.py"]
    for mod in mods:
        src = (HERE.parent / mod).read_text(encoding="utf-8")
        assert "dispatcher_rows" not in src, mod
    v032 = (HERE.parent / "build_v032.py").read_text(encoding="utf-8")
    assert "switch_schedule.json" in v032
    assert "suffix_commands.jsonl" in v032


# ------------------------------------------------------------------ #5 --

def test_fix5_strict_has_no_all_vehicle_counts_and_no_ratio():
    """The strict block is exactly the detector block's CAV columns (no
    all-vehicle entrance counts), and it is invariant to EVERY non-CAV
    insertion -- so it carries no signal about the non-CAV (all-vehicle)
    count and no penetration-revealing ratio can be formed from it."""
    vids, t, ent, cav = make_sched()
    ins_vids, ins_t = make_insertions(vids, t)
    obs = SourceObs(vids, t, ent, cav, ins_vids, ins_t)
    d_det, c_det = obs.blocks(TS, T0, LOADING, variant="detector")
    d_str, c_str = obs.blocks(TS, T0, LOADING, variant="strict")
    assert d_det.shape[1] == len(DEMAND_DEP_NAMES) == 12
    assert d_str.shape[1] == DEMAND_STRICT_RAW == len(STRICT_DEMAND_NAMES)
    np.testing.assert_array_equal(d_str, d_det[:, 6:])
    # detector queue cols present, strict queue cols zeroed
    assert (c_det[:, 6:8] >= 0).all()
    assert (c_str[:, 6:8] == 0.0).all()

    # perturb ONLY non-CAV insertions (a different realised non-CAV
    # count, e.g. a different true penetration): the detector all-vehicle
    # counts move, the strict block does not
    cav_set = set(np.asarray(vids)[np.asarray(cav) == 1].tolist())
    pv, pt = list(ins_vids), list(ins_t)
    for i, (v, ti) in enumerate(zip(ins_vids, ins_t)):
        if v not in cav_set:
            pt[i] = float(ti) + 1000.0      # out of every window
    obs2 = SourceObs(vids, t, ent, cav, pv, pt)
    d_det2, c_det2 = obs2.blocks(TS, T0, LOADING, variant="detector")
    d_str2, c_str2 = obs2.blocks(TS, T0, LOADING, variant="strict")
    assert d_det2.tobytes() != d_det.tobytes(), "detector must see all"
    assert d_str2.tobytes() == d_str.tobytes(), \
        "strict must not see non-CAV insertions"
    assert c_str2.tobytes() == c_str.tobytes()


# ------------------------------------------------- build_v032 mini root --

def _npz_sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_mini_root_v032_build(tmp_path):
    """build_v032 over the mini v0.3-shaped root (rebuild key absent ->
    skipped): links + relabels + BOTH sidecar variants + MANIFEST (parent
    sha, fixes, policies, feature_sets) + SHA256SUMS; HOLDOUT never
    opened; the loader reads both new feature sets."""
    mini = make_mini_v02_root(tmp_path / "mini")
    v03_root = mini["root"]
    out_root = tmp_path / "v032"
    parent_sha = _npz_sha(v03_root / "MANIFEST.json")

    watch = HoldoutAccessWatch([mini["holdout"]]).install()
    import build_v032 as G
    rc = G.main(["--v03-root", str(v03_root), "--out", str(out_root),
                 "--parent-sha", parent_sha, "--no-rebuild",
                 "--workers", "1"])
    assert rc == 0
    assert not watch.hits, f"HOLDOUT files opened: {watch.hits[:3]}"

    man = json.loads((out_root / "MANIFEST.json").read_text("utf-8"))
    assert man["parent"]["manifest_sha256"] == parent_sha
    assert man["build"]["holdout_accessed"] is False
    assert len(man["build"]["fixes"]) == len(G.FIXES)
    assert man["policies"]["canonical_by_source"]["native"] == \
        "switching_mix"
    assert man["policies"]["dispatcher_rows_read"] is False
    assert set(man["feature_sets"]) >= {"oracle", "deployable_strict",
                                        "deployable_detector"}
    n_side = 0
    for key, e in man["worlds"].items():
        assert e.get("split") != "HOLDOUT"
        if e.get("superseded_by"):
            continue
        dets = [r for r in e["files"] if r.endswith("_feat_dep_det.npz")]
        strs = [r for r in e["files"] if r.endswith("_feat_dep_strict.npz")]
        assert len(dets) == 1 and len(strs) == 1, key
        assert e["policy_v032"] == G.POLICY_OF_SOURCE[e["source"]]
        n_side += 2
    assert n_side == 16                    # 4 non-holdout worlds x 2 src x 2

    # relabels landed in the label parquets (fixture parquets may already
    # carry the new prose -> the rewrite is then an idempotent no-op)
    import pandas as pd
    for key, e in man["worlds"].items():
        if e.get("superseded_by") or e["source"] != "native":
            continue
        rel = next(r for r in e["files"] if r.endswith("_labels.parquet"))
        df = pd.read_parquet(out_root / rel)
        assert (df["policy"] == G.V.SOURCE_POLICY["native"]).all()
        assert e.get("policy_relabelled_v032", False) in (True, False)

    # SHA256SUMS verifies
    for line in (out_root / "SHA256SUMS").read_text("utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        assert _npz_sha(out_root / rel) == sha, rel

    # strict sidecars: ctx cols 1/6/7 zero; demand == detector CAV cols;
    # detector ctx matches oracle on (0, 3, 6, 7)
    for key, e in man["worlds"].items():
        if e.get("superseded_by"):
            continue
        z_or = np.load(out_root / f"worlds/{e['chunk_id']}/"
                       f"{e['world']}__{e['source']}_feat.npz")
        z_d = np.load(out_root / f"worlds/{e['chunk_id']}/"
                      f"{e['world']}__{e['source']}_feat_dep_det.npz")
        z_s = np.load(out_root / f"worlds/{e['chunk_id']}/"
                      f"{e['world']}__{e['source']}_feat_dep_strict.npz")
        assert z_d["feature_set"] == "deployable_detector"
        assert z_s["feature_set"] == "deployable_strict"
        assert (z_d["context"][:, 1] == 0.0).all()
        assert (z_s["context"][:, 1] == 0.0).all()
        assert (z_s["context"][:, 6] == 0.0).all()
        assert (z_s["context"][:, 7] == 0.0).all()
        np.testing.assert_array_equal(z_s["demand"], z_d["demand"][:, 6:])
        for col in (0, 3, 6, 7):
            np.testing.assert_allclose(z_d["context"][:, col],
                                       z_or["context"][:, col], atol=1e-6)
        for col in (0, 3):
            np.testing.assert_allclose(z_s["context"][:, col],
                                       z_or["context"][:, col], atol=1e-6)
        z_or.close(), z_d.close(), z_s.close()

    # loader: both feature sets on the SAME root
    sys.path.insert(0, str(HERE.parent))           # output/wd01
    from partB_v0.data import V0Dataset, load_manifest
    log = lambda *a, **k: None
    m = load_manifest(out_root, verify_files=False, log=log)
    ds_s = V0Dataset(m, feature_set="deployable_strict", log=log)
    ds_d = V0Dataset(m, feature_set="deployable_detector", log=log)
    assert ds_s.feats["demand"].shape[1] == 6
    assert ds_d.feats["demand"].shape[1] == 12
    np.testing.assert_array_equal(ds_s.feats["demand"],
                                  ds_d.feats["demand"][:, 6:])
    assert (ds_s.feats["context"][:, 1] == 0.0).all()
