"""v0.3.2 trainer-side fix tests (#9, #10, #11, #12, #13).

#9  bootstrap on WORLD ids (base + switching arm of a world together);
#10 HGB early stopping on a world-grouped holdout (never random rows);
#11 deployable models fix pre_fix to the post-fix value at predict;
#12 the CAL-based selection path is hard-guarded; selection uses TRAIN
    world-grouped OOB only;
#13 no 'damping - native' claim remains anywhere (damping_fleet -
    switching_mix everywhere).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
PART = HERE.parent
WD01 = PART.parent
for p in (str(WD01),):
    if p not in sys.path:
        sys.path.insert(0, p)

from partB_v0.data import V0Dataset, bootstrap_world_ids, load_manifest  # noqa: E402
from partB_v0.gbm import (PRE_FIX_PREDICT_VALUE, STRICT_DEMAND_DEP_NAMES,  # noqa: E402
                          GbmValueModel, guard_no_calibration_rows,
                          row_feature_matrix)
from partB_v0.model import POLICIES, policy_id  # noqa: E402


def _ds(root):
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    return V0Dataset(man, log=lambda *_: None)


# ------------------------------------------------------------------ #9 --

def test_fix9_bootstrap_world_grouped():
    """Bootstrap unit = "<chunk>/<world>": a sampled world contributes
    BOTH its sources; an OOB world excludes BOTH."""
    names = [f"chunk1/w{i}__{s}" for i in "ABC" for s in ("base",
                                                          "native")]
    splits = ["TRAIN"] * len(names)
    for m in range(12):
        sampled, oob = bootstrap_world_ids(names, splits, seed=5, member=m)
        assert sampled and all("__" not in w for w in sampled)
        assert all("__" not in w for w in oob)
        assert set(sampled).isdisjoint(oob)
        s_worlds = {w.split("/")[-1] for w in sampled}
        o_worlds = {w.split("/")[-1] for w in oob}
        for w in "ABC":
            srcs = {n.rsplit("__", 1)[1]
                    for n in names if f"/w{w}__" in n}
            if w in s_worlds:
                assert srcs == {"base", "native"}, (m, w)
                assert w not in o_worlds
            if w in o_worlds:
                assert w not in s_worlds


def test_fix9_member_trainer_pools_world_grouped(tiny_dataset):
    """MemberTrainer pools/OOB rows follow the world-grouped bootstrap:
    every TRAIN source of a sampled world is in the pool; every source of
    an OOB world is in oob_rows (and CAL rows never are)."""
    from partB_v0.train import MemberTrainer, TrainConfig, resolve_device
    root, _ = tiny_dataset
    ds = _ds(root)
    cfg = TrainConfig(data_root=str(root), out_dir=str(HERE / "unused_out"),
                      members=1, probe_steps=0, verify=False)
    for m in range(4):
        tr = MemberTrainer(cfg, ds, m, resolve_device("cpu"),
                           log=lambda *_: None)
        pool_src = set(ds.rows.loc[tr.pool, "src_idx"])
        oob_src = set(ds.rows.loc[tr.oob_rows, "src_idx"])
        pool_names = {ds.src_names[s] for s in pool_src}
        oob_names = {ds.src_names[s] for s in oob_src}
        sampled_worlds = {w.split("/")[-1] for w in tr.worlds}
        oob_worlds = {w.split("/")[-1] for w in tr.oob_worlds}
        for name in ds.src_names:
            if ds.src_split[ds.src_names.index(name)] != "TRAIN":
                assert name not in pool_names
                assert name not in oob_names
                continue
            world = name.rsplit("__", 1)[0].split("/")[-1]
            if world in sampled_worlds:
                assert name in pool_names, (m, name)
            if world in oob_worlds:
                assert name in oob_names, (m, name)
        # OOB rows are TRAIN rows by construction
        assert (ds.rows.loc[tr.oob_rows, "split"] == "TRAIN").all()


# ----------------------------------------------------------------- #10 --

def test_fix10_es_split_world_disjoint():
    """The early-stopping holdout splits by WORLD: no world has rows on
    both sides; a single-world slice gets no split (fixed max_iter)."""
    m = GbmValueModel(max_iter=5)
    rng = np.random.default_rng(0)
    grp = np.array(["wA"] * 50 + ["wB"] * 50 + ["wC"] * 50)[rng.permutation(
        150)]
    X = np.zeros((150, 3))
    y = rng.normal(size=150)
    out = m._es_val_split(X, y, grp)
    assert out is not None
    fit_i, val_i = out
    assert set(grp[fit_i]).isdisjoint(grp[val_i])
    assert 0 < len(val_i) < len(fit_i)
    # tiny / single-world slices: no grouped holdout possible
    assert m._es_val_split(np.zeros((10, 3)), np.zeros(10),
                           np.array(["wA"] * 10)) is None
    assert m._es_val_split(np.zeros((100, 3)), np.zeros(100),
                           np.array(["wA"] * 100)) is None


def test_fix10_fit_passes_grouped_validation(tiny_dataset, monkeypatch):
    """HGB fits receive the grouped validation set via X_val (sklearn
    >= 1.7) or run at fixed max_iter with early stopping OFF -- never a
    random-row internal split."""
    import sklearn.ensemble as ske
    root, _ = tiny_dataset
    ds = _ds(root)
    calls = []

    orig_fit = ske.HistGradientBoostingRegressor.fit

    def rec_fit(self, X, y, *a, **k):
        calls.append(dict(n=int(len(X)), X_val=k.get("X_val"),
                          early_stopping=self.early_stopping))
        return orig_fit(self, X, y, *a, **k)

    monkeypatch.setattr(ske.HistGradientBoostingRegressor, "fit", rec_fit)
    model = GbmValueModel(max_iter=4, cal_folds=2, threads=1)
    model.fit(ds, log=lambda *_: None)
    assert calls, "no HGB fit recorded"
    assert model.es_grouping in ("world-grouped", "fixed-iter")
    if model.es_val_support:
        assert any(c["X_val"] is not None for c in calls)
        for c in calls:
            if c["X_val"] is not None:
                assert c["early_stopping"] is True
    else:
        assert all(c["early_stopping"] is False for c in calls)


def test_fix10_baselines_have_no_random_row_es():
    """No random-row validation split survives anywhere in gbm.py: the
    ONLY validation_fraction=0.1 sits in the early_stopping=True branch
    of _hgb (which receives the world-grouped X_val), and the
    fixed-iter/no-ES constructors are used everywhere else."""
    import inspect
    from partB_v0 import gbm
    src = inspect.getsource(gbm)
    assert src.count("validation_fraction=0.1") == 1
    seg = src[:src.index("validation_fraction=0.1")]
    assert "early_stopping=True" in seg[-200:]
    # fixed max_iter fallbacks: the no-split _hgb branch + the baselines
    assert src.count("early_stopping=False") >= 2


# ----------------------------------------------------------------- #11 --

def _fit_two_models(ds):
    orc = GbmValueModel(max_iter=4, cal_folds=0, threads=1,
                        feature_set="oracle")
    orc.fit(ds, log=lambda *_: None)
    dep = GbmValueModel(max_iter=4, cal_folds=0, threads=1,
                        feature_set="deployable_strict")
    dep.fit(ds, log=lambda *_: None)
    return orc, dep


def test_fix11_pre_fix_fixed_at_predict_for_deployable(tiny_dataset):
    """A deployable model's predictions are INVARIANT to the rows'
    pre_fix values (always predicted at the post-fix value); the oracle
    model, which keeps the true flag, does move."""
    root, _ = tiny_dataset
    ds = _ds(root)
    # fit with a MIXED pre_fix pattern so the feature genuinely matters
    pre = (np.arange(len(ds.rows)) % 2).astype(float)
    ds.rows["pre_fix"] = pre
    orc, dep = _fit_two_models(ds)
    rows = np.arange(len(ds.rows))
    assert dep.pre_fix_at_predict == PRE_FIX_PREDICT_VALUE == 0.0
    assert orc.pre_fix_at_predict is None
    base_orc = orc.predict_quantiles_rows(ds, rows)
    base_dep = dep.predict_quantiles_rows(ds, rows)
    # flip EVERY row's pre_fix to the other value and re-predict
    ds.rows["pre_fix"] = 1.0 - pre
    flip_orc = orc.predict_quantiles_rows(ds, rows)
    flip_dep = dep.predict_quantiles_rows(ds, rows)
    dep_same = all(np.array_equal(base_dep[q], flip_dep[q])
                   for q in base_dep)
    assert dep_same, "deployable model must fix pre_fix at predict (#11)"
    assert not all(np.array_equal(base_orc[q], flip_orc[q])
                   for q in base_orc), \
        "oracle model should still read the rows' true pre_fix"
    ds.rows["pre_fix"] = pre


def test_fix11_override_in_row_feature_matrix(tiny_dataset):
    root, _ = tiny_dataset
    ds = _ds(root)
    X_plain, names = row_feature_matrix(ds)
    X_fixed, _ = row_feature_matrix(ds, pre_fix_override=0.0)
    assert names[-1] == "pre_fix"
    assert not np.allclose(X_plain[:, -1], X_fixed[:, -1]) or \
        (X_plain[:, -1] == 0.0).all()
    assert (X_fixed[:, -1] == 0.0).all()


def test_fix11_sidecar_records_the_fix(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    ds = _ds(root)
    dep = GbmValueModel(max_iter=3, cal_folds=0, threads=1,
                        feature_set="deployable_strict")
    dep.fit(ds, log=lambda *_: None)
    side = dep.save(tmp_path / "m")
    assert side["pre_fix_at_predict"] == 0.0
    assert "post-fix" in side["pre_fix_predict_note"]
    m2 = GbmValueModel.load(tmp_path / "m", log=lambda *_: None)
    assert m2.pre_fix_at_predict == 0.0


def test_fix11_feature_set_validation_and_names():
    for fs in ("oracle", "deployable", "deployable_detector",
               "deployable_strict"):
        GbmValueModel(feature_set=fs)
    with pytest.raises(ValueError):
        GbmValueModel(feature_set="deployable_strict_x")
    assert len(STRICT_DEMAND_DEP_NAMES) == 6
    assert all(n.startswith("obs_cav_")
               for n in STRICT_DEMAND_DEP_NAMES)


def test_fix11_summary_names_strict(tiny_dataset):
    from partB_v0.gbm import summary_features
    root, _ = tiny_dataset
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    ds_o = V0Dataset(man, feature_set="oracle", log=lambda *_: None)
    _X, names_o = summary_features(ds_o)
    assert "demand_0" in names_o
    # a strict-shaped fake ds: demand width 6 -> the CAV-only names
    ds_s = V0Dataset(man, feature_set="oracle", log=lambda *_: None)
    ds_s.feature_set = "deployable_strict"
    ds_s.feats = dict(ds_s.feats)
    ds_s.feats["demand"] = ds_s.feats["demand"][:, :6]
    _X, names_s = summary_features(ds_s)
    assert [n for n in names_s if n.startswith("obs_")] == \
        list(STRICT_DEMAND_DEP_NAMES)


# ----------------------------------------------------------------- #12 --

def test_fix12_cal_selection_raises(tiny_dataset):
    from partB_v0.train import CalSelectionRemoved, cal_native_crps300
    root, _ = tiny_dataset
    ds = _ds(root)
    net = torch.nn.Module()
    with pytest.raises(CalSelectionRemoved):
        cal_native_crps300(net, ds)


def test_fix12_oob_selector_guard(tiny_dataset):
    from partB_v0.gbm import CalibrationLeakError
    from partB_v0.train import oob_crps300
    root, _ = tiny_dataset
    ds = _ds(root)
    r = ds.rows
    cal_idx = r[(r["split"] == "CALIBRATION")
                & np.isclose(r["tau"], 300.0)].index.to_numpy()
    assert len(cal_idx)
    net = torch.nn.Sequential(torch.nn.Flatten())
    with pytest.raises(CalibrationLeakError):
        oob_crps300(net, ds, cal_idx)


def test_fix12_guard_rejects_non_train():
    from partB_v0.gbm import CalibrationLeakError
    with pytest.raises(CalibrationLeakError):
        guard_no_calibration_rows(np.array(["TRAIN", "CALIBRATION"]),
                                  "test")


def test_fix12_run_training_selects_on_oob(tiny_dataset, tmp_path):
    """run_training's selection records are OOB-keyed; the CAL key is
    gone; the member sidecar states the OOB selection contract."""
    from partB_v0.train import TrainConfig, run_training
    root, _ = tiny_dataset
    out = tmp_path / "out_oob"
    cfg = TrainConfig(data_root=str(root), out_dir=str(out), members=1,
                      batch_size=32, td_batch=16, max_steps=2,
                      probe_steps=0, seed=5, verify=False)
    metrics, _ = run_training(cfg, log=lambda *_: None)
    rep = metrics["members"][0]
    assert "best_oob_crps300" in rep and "best_cal_native_crps300" not in rep
    assert "final_oob_crps300" in rep
    side = json.loads((out / "member_0.json").read_text("utf-8"))
    assert "best_oob_crps300" in side
    assert "OOB" in side["selection"] and "CAL never read" in \
        side["selection"]
    meta = json.loads((out / "ckpt" / "m0_meta.json").read_text("utf-8"))
    assert "best_oob_crps300" in meta


# ----------------------------------------------------------------- #13 --

def test_fix13_no_damping_minus_native_claim():
    """No 'damping - native' claim remains in the live code or the wd01
    docs (the campaign arms are switching mixes; v0.3.2 fix #13)."""
    claim = "damping - native"
    roots = [PART / "partB_v0", PART / "partB_v0data", WD01]
    for root in roots:
        for pat in ("*.py", "*.md"):
            for p in root.rglob(pat):
                if "archive" in p.parts or p.name.startswith(
                        "test_v032_fixes"):
                    continue
                if p.name.startswith("test_v032_fixes"):
                    continue
                txt = p.read_text(encoding="utf-8", errors="replace")
                assert claim not in txt, p


def test_fix13_policies_canonical():
    assert POLICIES[0] == "switching_mix"
    assert policy_id("switching_mix") == 0
    with pytest.raises(ValueError):
        policy_id("native")
