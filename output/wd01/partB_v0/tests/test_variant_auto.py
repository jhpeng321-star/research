"""--gbm-variant auto tests (v0.3.1 per-tau selection).

Covers: the CALIBRATION-rows-never-reach-selection guard (the brief's
required unit test), the selection rule (hybrid better -> hybrid; ridge
better -> ridge; exact tie -> ridge), tie-aware acceptance verdicts,
the auto fit end-to-end on tiny synthetic data (hybrid AND ridge-only
taus; save/load roundtrip), the in-fit (no-fold) fallback and the CLI
wiring incl. eval_report section 0 + the headline.

CPU only; tiny fits with small --gbm-max-iter (owner rule respected).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import wdpaths

from conftest_load import make_tiny_dataset

from partB_v0.data import V0Dataset, load_manifest
from partB_v0.gbm import (PRE_FIX_PREDICT_VALUE, CalibrationLeakError,
                          GbmValueModel, acceptance_verdict,
                          guard_no_calibration_rows, row_feature_matrix,
                          select_variant_by_oob)


def _load_ds(root, feature_set="oracle"):
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    return V0Dataset(man, feature_set=feature_set, log=lambda *_: None)


def _fit(ds, **kw):
    kw.setdefault("quantiles", (0.1, 0.5, 0.9))
    kw.setdefault("max_iter", 15)
    kw.setdefault("cal_folds", 2)
    return GbmValueModel(**kw).fit(ds, log=lambda *_: None)


@pytest.fixture()
def v031_dataset(tmp_path):
    """Tiny dataset WITH deployable sidecars (test_features_deployable
    pattern: same state blocks, replaced demand/context)."""
    root = tmp_path / "data"
    make_tiny_dataset(root)
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    rng = np.random.default_rng(99)
    for key, entry in man.entries.items():
        rel = f"worlds/{entry['chunk_id']}/{entry['world']}__" \
              f"{entry['source']}_feat.npz"
        z = np.load(root / rel, allow_pickle=False)
        payload = {k: z[k] for k in z.files}
        n = int(z["t"].shape[0])
        payload["demand"] = rng.random((n, 12)).astype(np.float32)
        payload["feature_set"] = np.str_("deployable")
        z.close()
        dep_rel = rel.replace("_feat.npz", "_feat_dep.npz")
        np.savez_compressed(root / dep_rel, **payload)
        doc = json.loads((root / "MANIFEST.json").read_text("utf-8"))
        doc["worlds"][key]["files"][dep_rel] = \
            hashlib.sha256((root / dep_rel).read_bytes()).hexdigest()
        (root / "MANIFEST.json").write_text(
            json.dumps(doc, indent=1), encoding="utf-8")
    return root


# ------------------------------------------------------- selection guard --

def _preds(y, lo, hi):
    return {0.1: y - lo, 0.5: y, 0.9: y + hi}


def test_select_variant_guard_rejects_calibration_rows():
    """The required guard: ANY CALIBRATION row reaching the selection
    function raises CalibrationLeakError (selection is TRAIN-only)."""
    y = np.arange(5, dtype=float)
    splits = np.array(["TRAIN", "TRAIN", "CALIBRATION", "TRAIN", "TRAIN"])
    with pytest.raises(CalibrationLeakError) as e:
        select_variant_by_oob(300.0, "delay", y, splits,
                              _preds(y, 1.0, 1.0), _preds(y, 2.0, 2.0))
    assert "300" in str(e.value) and "CALIBRATION" in str(e.value)


def test_guard_no_calibration_rows_direct():
    guard_no_calibration_rows(["TRAIN", "TRAIN"], "ctx")
    guard_no_calibration_rows(np.array(["TRAIN"]), "ctx")
    for bad in ("CALIBRATION", "HOLDOUT", ""):
        with pytest.raises(CalibrationLeakError):
            guard_no_calibration_rows(
                np.array(["TRAIN", bad]), "ctx")


def test_select_variant_rule_and_alignment():
    y = np.arange(6, dtype=float)
    splits = np.array(["TRAIN"] * 6)
    # hybrid strictly sharper (width 1 vs 3) -> ships hybrid
    e = select_variant_by_oob(30.0, "delay", y, splits,
                              _preds(y, 0.5, 0.5), _preds(y, 1.5, 1.5))
    assert e["chosen"] == "hybrid" and e["select_target"] == "delay"
    assert e["n"] == 6
    assert e["oob_crps"]["hybrid"] < e["oob_crps"]["ridge"]
    # ridge sharper -> ships ridge
    e2 = select_variant_by_oob(300.0, "delay", y, splits,
                               _preds(y, 2.0, 2.0), _preds(y, 0.5, 0.5))
    assert e2["chosen"] == "ridge"
    assert e2["oob_crps"]["ridge"] < e2["oob_crps"]["hybrid"]
    # exact CRPS tie -> ships ridge (the simpler model)
    p = _preds(y, 1.0, 1.0)
    e3 = select_variant_by_oob(570.0, "delay", y, splits, p, p)
    assert e3["chosen"] == "ridge"
    assert e3["oob_crps"]["hybrid"] == e3["oob_crps"]["ridge"]
    # misaligned predictive arrays are refused
    with pytest.raises(ValueError):
        select_variant_by_oob(60.0, "delay", y, splits,
                              {0.1: y[:3], 0.5: y, 0.9: y},
                              _preds(y, 1.0, 1.0))


# ------------------------------------------------------ acceptance ties --

def test_acceptance_verdict_tie_semantics():
    """A tie is a TIE, not a pass: exact-equality metrics never yield
    PASS; TIE = no strict loss and no strict win on both."""
    v = acceptance_verdict(10.0, 10.0, 5.0, 5.0)
    assert v["result"] == "TIE" and not v["hard_pass"]
    assert v["tie_mae"] and v["tie_crps"]
    p = acceptance_verdict(9.0, 10.0, 4.0, 5.0)
    assert p["result"] == "PASS" and p["hard_pass"]
    f = acceptance_verdict(11.0, 10.0, 4.0, 5.0)
    assert f["result"] == "FAIL" and not f["hard_pass"]
    # MAE tie + CRPS win: not a pass, no loss -> TIE
    t = acceptance_verdict(10.0, 10.0, 4.0, 5.0)
    assert t["result"] == "TIE" and t["beats_ridge_crps"]
    # MAE tie + CRPS loss -> FAIL
    fl = acceptance_verdict(10.0, 10.0, 6.0, 5.0)
    assert fl["result"] == "FAIL"
    nan = acceptance_verdict(float("nan"), 10.0, 4.0, 5.0)
    assert nan["result"] == "FAIL" and not nan["hard_pass"]


# ------------------------------------------------- auto fit end-to-end --

def test_gbm_auto_fit_selection_records_and_roundtrip(v031_dataset,
                                                      tmp_path):
    from partB_v0.gbm import SCHEMA
    root = v031_dataset
    ds = _load_ds(root, feature_set="deployable")
    model = _fit(ds, variant="auto", feature_set="deployable")
    assert model.variant == "auto"
    assert set(model.taus) == set(model.tau_variants)
    assert set(model.tau_variants.values()) <= {"hybrid", "ridge"}
    assert len(model.selection) == 5
    for e in model.selection:
        assert e["select_target"] == "delay" and e["n"] > 0
        assert e["chosen"] == model.tau_variants[e["tau"]]
        assert np.isfinite(e["oob_crps"]["hybrid"])
        assert np.isfinite(e["oob_crps"]["ridge"])
    # a ridge tau skips the GBM quantile fits ENTIRELY (ridge point +
    # conformal offsets only); a hybrid tau has both parts
    for tau, v in model.tau_variants.items():
        has_gbm = any(k[0] == tau for k in model.models)
        assert has_gbm == (v == "hybrid"), (tau, v)
        assert (tau, "delay") in model.point_models
        assert set(model.oof_delay[tau]) == \
            set(model.target_quantiles["delay"])

    # predictions: monotone, finite, on-grid point == predictive median
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    idx = cal_idx[:40]
    q = model.predict_quantiles_rows(ds, idx, "delay")
    qs_sorted = sorted(q)
    for a, b in zip(qs_sorted[:-1], qs_sorted[1:]):
        assert np.all(q[b] >= q[a] - 1e-9)
    assert np.isfinite(q[0.5]).all()
    X, _names = row_feature_matrix(ds, idx)
    taus = ds.rows["tau"].to_numpy()[idx]
    pt = model.point_matrix(X, taus, "delay")
    assert np.allclose(pt, q[0.5])
    p90 = model.point_matrix(X, np.full(len(idx), 90.0), "delay")
    assert np.isfinite(p90).all()

    # a ridge tau's calibrated quantiles == ridge point + cumulative-max
    # of its conformal offsets (the shipped construction, verified)
    for tau, v in model.tau_variants.items():
        if v != "ridge":
            continue
        m_tau = np.isclose(taus, tau)
        if not m_tau.any():
            continue
        base = model.point_models[(tau, "delay")].predict(X[m_tau])
        for qq in qs_sorted:
            off = max(model.shifts[(tau, "delay", a)]
                      for a in qs_sorted if a <= qq + 1e-12)
            assert np.allclose(q[qq][m_tau], base + off, atol=1e-8)

    # a HYBRID tau's predictive median must sit AT the ridge point plus a
    # residual-scale correction (regression: _base_part once dropped the
    # point for auto hybrid taus, leaving residual-only predictions)
    from sklearn.linear_model import Ridge
    X_all, _names = row_feature_matrix(ds)
    tau_all = ds.rows["tau"].to_numpy()
    tr_all = (ds.rows["split"] == "TRAIN").to_numpy()
    y_all_rows = ds.rows["total_delay_ref_s"].to_numpy()
    y_idx = y_all_rows[idx]
    for tau, v in model.tau_variants.items():
        if v != "hybrid":
            continue
        m_tau = np.isclose(taus, tau)
        if not m_tau.any():
            continue
        tr = np.isclose(tau_all, tau) & tr_all
        rid = Ridge(alpha=1.0).fit(X_all[tr], y_all_rows[tr])
        pt = rid.predict(X[m_tau])
        scale = float(np.mean(np.abs(y_idx[m_tau] - pt)))
        drift = float(np.mean(np.abs(q[0.5][m_tau] - pt)))
        assert drift < 5.0 * scale + 10.0, \
            f"tau={tau}: median drifted off the ridge point " \
            f"({drift:.1f} vs residual scale {scale:.1f})"

    # save/load roundtrip: schema /3, choice + selection preserved,
    # identical predictions
    out = tmp_path / "out"
    model.save(out)
    side = json.loads((out / "gbm_value.json").read_text("utf-8"))
    assert side["schema"] == SCHEMA
    assert side["variant"] == "auto"
    assert side["tau_variants"] and side["selection"]
    m2 = GbmValueModel.load(out, log=lambda *_: None)
    assert m2.variant == "auto"
    assert m2.tau_variants == model.tau_variants
    assert m2.selection == model.selection
    q2 = m2.predict_quantiles_rows(ds, idx, "delay")
    for k in q:
        assert np.allclose(q[k], q2[k])


def test_gbm_auto_all_ridge_taus_ship_point_plus_offsets(tiny_dataset,
                                                         monkeypatch):
    """Forcing every tau to 'ridge' must leave ZERO GBM quantile models
    and still answer every predict path (quantiles, median, off-grid
    point, collision head) -- the ridge-only shipping construction."""
    import partB_v0.gbm as G

    def _always_ridge(tau, target, y, splits, ph, pr):
        # the guard must still run in production code; this fake keeps
        # the record shape only
        return dict(tau=float(tau), select_target=target, chosen="ridge",
                    n=int(len(y)),
                    oob_crps={"hybrid": 1.0, "ridge": 0.5})

    monkeypatch.setattr(G, "select_variant_by_oob", _always_ridge)
    root, _ = tiny_dataset
    ds = _load_ds(root)
    model = _fit(ds, variant="auto")
    assert model.tau_variants and \
        set(model.tau_variants.values()) == {"ridge"}
    assert not model.models                      # no GBM quantile fits
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    idx = cal_idx[:30]
    q = model.predict_quantiles_rows(ds, idx, "delay")
    qs_sorted = sorted(q)
    for a, b in zip(qs_sorted[:-1], qs_sorted[1:]):
        assert np.all(q[b] >= q[a] - 1e-9)
    assert np.isfinite(q[0.5]).all()
    X, _names = row_feature_matrix(ds, idx)
    taus = ds.rows["tau"].to_numpy()[idx]
    assert np.allclose(model.point_matrix(X, taus, "delay"), q[0.5])
    assert np.isfinite(model.point_matrix(
        X, np.full(len(idx), 90.0), "delay")).all()
    pc = model.coll_prob_rows(ds, idx)
    assert np.all((pc >= 0) & (pc <= 1))


def _force_choice(monkeypatch, chosen: str) -> None:
    """Pin the per-tau auto selection to one variant (the forcing the
    regression tests need; the CAL-leak guard itself keeps its own
    dedicated tests)."""
    import partB_v0.gbm as G

    def _fixed(tau, target, y, splits, ph, pr):
        return dict(tau=float(tau), select_target=str(target),
                    chosen=chosen, n=int(len(y)),
                    oob_crps={"hybrid": 1.0, "ridge": 0.5})

    monkeypatch.setattr(G, "select_variant_by_oob", _fixed)


def test_gbm_auto_forced_all_hybrid_matches_plain_hybrid(tiny_dataset,
                                                         monkeypatch):
    """Regression (reviewer bug): an auto model whose per-tau choice is
    forced to all-'hybrid' must predict EXACTLY like the plain hybrid
    model (median + all quantiles, on-grid and off-grid points) -- the
    buggy _base_part dropped the ridge point for auto hybrid taus,
    leaving residual-only predictions."""
    _force_choice(monkeypatch, "hybrid")
    root, _ = tiny_dataset
    ds = _load_ds(root)
    auto = _fit(ds, variant="auto")
    plain = _fit(ds, variant="hybrid")
    assert auto.tau_variants and \
        set(auto.tau_variants.values()) == {"hybrid"}
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    qa = auto.predict_quantiles_rows(ds, cal_idx, "delay")
    qh = plain.predict_quantiles_rows(ds, cal_idx, "delay")
    for k in qh:
        assert np.allclose(qa[k], qh[k], rtol=0.0, atol=1e-9), k
    X, _names = row_feature_matrix(ds, cal_idx)
    taus = ds.rows["tau"].to_numpy()[cal_idx]
    assert np.allclose(auto.point_matrix(X, taus, "delay"),
                       plain.point_matrix(X, taus, "delay"),
                       rtol=0.0, atol=1e-9)
    off_t = np.full(len(cal_idx), 90.0)      # off-grid interpolation too
    assert np.allclose(auto.point_matrix(X, off_t, "delay"),
                       plain.point_matrix(X, off_t, "delay"),
                       rtol=0.0, atol=1e-9)


def test_gbm_auto_forced_all_ridge_equals_ridge_only(tiny_dataset, tmp_path,
                                                     monkeypatch):
    """Regression: an auto model forced to all-'ridge' must be the
    ridge-only shipping construction at every tau -- ridge point +
    cumulative-max of the recorded conformal offsets, ZERO GBM
    correction (and the loaded pkl keeps it)."""
    _force_choice(monkeypatch, "ridge")
    root, _ = tiny_dataset
    ds = _load_ds(root)
    model = _fit(ds, variant="auto")
    assert set(model.tau_variants.values()) == {"ridge"}
    assert not model.models
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    q = model.predict_quantiles_rows(ds, cal_idx, "delay")
    qs_sorted = sorted(q)
    X, _names = row_feature_matrix(ds, cal_idx)
    taus = ds.rows["tau"].to_numpy()[cal_idx]
    for tau in model.taus:
        m = np.isclose(taus, tau)
        base = model.point_models[(float(tau), "delay")].predict(X[m])
        for qq in qs_sorted:
            off = max(model.shifts[(float(tau), "delay", a)]
                      for a in qs_sorted if a <= qq + 1e-12)
            assert np.allclose(q[qq][m], base + off, rtol=0.0,
                               atol=1e-9), (tau, qq)
    assert np.allclose(model.point_matrix(X, taus, "delay"), q[0.5],
                       rtol=0.0, atol=1e-9)
    # save/load roundtrip preserves the forced construction
    out = tmp_path / "out_ridge"
    model.save(out)
    m2 = GbmValueModel.load(out, log=lambda *_: None)
    q2 = m2.predict_quantiles_rows(ds, cal_idx, "delay")
    for k in q:
        assert np.allclose(q2[k], q[k], rtol=0.0, atol=1e-9), k


def test_legacy_schema2_hybrid_pkl_loads_as_hybrid(tiny_dataset, tmp_path):
    """Regression: /2 pkls (the pre-auto hybrid schema, e.g. the accepted
    v0.3.1 hybrid run) record the variant in the body and must load as
    'hybrid' -- load() once mapped every legacy schema to 'pure', which
    turned loaded hybrid models into residual-only predictors."""
    root, _ = tiny_dataset
    ds = _load_ds(root)
    model = _fit(ds, variant="hybrid")
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    q = model.predict_quantiles_rows(ds, cal_idx, "delay")
    out = tmp_path / "out_s2"
    model.save(out)
    import pickle
    pkl = out / "gbm_value.pkl"
    body = pickle.loads(pkl.read_bytes())
    body["schema"] = "wd01_partB_v0_gbm/2"
    for k in ("tau_variants", "selection"):
        body.pop(k, None)                    # /2 predates the auto record
    pkl.write_bytes(pickle.dumps(body, protocol=5))
    side = json.loads((out / "gbm_value.json").read_text("utf-8"))
    side.pop("pkl_sha256")                   # body was rewritten above
    (out / "gbm_value.json").write_text(json.dumps(side), encoding="utf-8")
    m2 = GbmValueModel.load(out, log=lambda *_: None)
    assert m2.variant == "hybrid"
    q2 = m2.predict_quantiles_rows(ds, cal_idx, "delay")
    for k in q:
        assert np.allclose(q2[k], q[k], rtol=0.0, atol=1e-9), k


def test_gbm_auto_infit_fallback_smoke(tiny_dataset):
    """cal_folds=0 (no folds): auto still selects via in-fit candidates
    (smoke-only semantics) and fits the chosen variant."""
    root, _ = tiny_dataset
    ds = _load_ds(root)
    model = _fit(ds, variant="auto", cal_folds=0)
    assert model.tau_variants
    assert set(model.tau_variants.values()) <= {"hybrid", "ridge"}
    cal_idx = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    q = model.predict_quantiles_rows(ds, cal_idx[:20], "delay")
    assert np.isfinite(q[0.5]).all()


def test_fit_guard_raises_if_train_mask_leaks_cal(tiny_dataset,
                                                  monkeypatch):
    """Regression: if the tau TRAIN slice ever leaked a CALIBRATION row
    into the conformal/selection stage, fit() must raise instead of
    silently calibrating on CAL (simulated mask bug)."""
    import partB_v0.gbm as G
    root, _ = tiny_dataset
    ds = _load_ds(root)
    cal_pos = np.flatnonzero(
        (ds.rows["split"] == "CALIBRATION").to_numpy())
    orig = G._tau_mask

    def leaky(rows_, tau, split=None):
        m = orig(rows_, tau, split)
        if split == "TRAIN" and m.any():
            m[cal_pos[0]] = True      # the simulated bug: a CAL row in
        return m                      # the TRAIN slice

    monkeypatch.setattr(G, "_tau_mask", leaky)
    with pytest.raises(CalibrationLeakError):
        _fit(ds, variant="auto")


def test_crps_empirical_chunked_exact():
    """The chunked NRG CRPS (added after the auto run OOMed on the
    20715-row OOB table) is exact: any block size equals the naive
    full-matrix computation, incl. the S < 2 degenerate case."""
    from partB_v0.distributions import crps_empirical
    rng = np.random.default_rng(3)
    s = rng.normal(100, 30, 37)
    y = rng.normal(100, 30, 53)
    naive = np.abs(s[None, :] - y[:, None]).mean(axis=1) \
        - 0.5 * float(np.abs(s[None, :] - s[:, None]).mean())
    for blk in (1, 2, 7, 1000):
        assert np.allclose(crps_empirical(s, y, block=blk), naive)
    deg = crps_empirical(np.array([5.0]), y, block=2)
    assert np.allclose(deg, np.abs(5.0 - y))


# ------------------------------------------------------------- CLI ------

def test_cli_gbm_variant_auto(tiny_dataset, tmp_path, capsys):
    from partB_v0.__main__ import main
    root, _ = tiny_dataset
    out = tmp_path / "out_auto"
    rc = main(["train", "--value-model", "gbm", "--gbm-variant", "auto",
               "--data", str(root), "--out", str(out), "--gbm-max-iter",
               "15", "--gbm-cal-folds", "2", "--device", "cpu",
               "--no-verify"])
    assert rc == 0
    capsys.readouterr()
    side = json.loads((out / "gbm_value.json").read_text("utf-8"))
    assert side["variant"] == "auto"
    assert side["tau_variants"] and len(side["selection"]) == 5
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["gbm"]["variant"] == "auto"
    assert metrics["gbm"]["tau_variants"] == side["tau_variants"]
    assert metrics["gbm"]["selection"] == side["selection"]
    ev = metrics["eval"]
    assert ev["gbm_variant"] == "auto"
    assert ev["gbm_selection"] == side["selection"]
    report = (out / "eval_report.md").read_text("utf-8")
    assert "gbm — AUTO" in report
    assert ("## 0. Per-tau value-model variant selected on TRAIN OOB"
            in report)
    assert "verdict |" in report
    assert "tie is a TIE" in report
    if any(v == "ridge" for v in side["tau_variants"].values()):
        assert "deployable long-horizon skill ~ ridge" in report
    rc = main(["verify", "--out", str(out)])
    assert rc == 0
    assert "verify: OK" in capsys.readouterr().out


# ------------------------- v0.3.1 real-artifact regression (read-only) --

_V031_DATA = (Path(wdpaths.RESEARCH_DENSE_ROOT) / "output"
              / "wd01_partB_v0data_v031")
_V031_HYBRID_DIR = (Path(wdpaths.RESEARCH_DENSE_ROOT) / "output"
                    / "partB_results" / "v0.3.1_hybrid_deployable")
_V031_AUTO_R2_DIR = (Path(wdpaths.RESEARCH_DENSE_ROOT) / "output"
                     / "partB_results" / "v0.3.1_selected_deployable_r2")

_V031_READY = ((_V031_DATA / "MANIFEST.json").exists()
               and (_V031_HYBRID_DIR / "gbm_value.pkl").exists()
               and (_V031_AUTO_R2_DIR / "gbm_value.pkl").exists())


def _ridge_only_reference(ds, tau: float, X_cal: np.ndarray,
                          taus_cal: np.ndarray, cal_folds: int) -> dict:
    """Independent reimplementation of the ridge-only shipping
    construction for one grid tau (fit contract: Ridge(alpha=1) on the
    TRAIN rows; world-grouped OOF (or in-fit) residuals; conformal
    shift = q-quantile of the OOF residuals; cumulative-max across the
    sorted levels).  Used to cross-check the trained artifact."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    r = ds.rows
    X_all, _names = row_feature_matrix(ds)
    tr = (np.isclose(r["tau"].to_numpy(), tau)
          & (r["split"] == "TRAIN").to_numpy())
    y = r["total_delay_ref_s"].to_numpy()[tr]
    Xtr = X_all[tr]
    grp = (r["chunk_id"].astype(str) + "|"
           + r["world"].astype(str)).to_numpy()[tr]
    n = int(tr.sum())
    assert n >= 20
    folds = max(2, min(cal_folds, n // 10))
    oof = np.full(n, np.nan)
    if len(np.unique(grp)) >= folds:
        splits = GroupKFold(n_splits=folds).split(Xtr, groups=grp)
    else:
        from sklearn.model_selection import KFold
        splits = KFold(n_splits=folds, shuffle=True,
                       random_state=0).split(Xtr)
    for fit_i, cal_i in splits:
        rf = Ridge(alpha=1.0).fit(Xtr[fit_i], y[fit_i])
        oof[cal_i] = rf.predict(Xtr[cal_i])
    resid = y - oof
    fin = resid[np.isfinite(resid)]
    m = np.isclose(taus_cal, tau)
    base = Ridge(alpha=1.0).fit(Xtr, y).predict(X_cal[m])
    out, run = {}, None
    for qq in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        v = base + (float(np.quantile(fin, qq)) if len(fin) else 0.0)
        run = v if run is None else np.maximum(run, v)
        out[qq] = run
    return out


@pytest.mark.skipif(not _V031_READY,
                    reason="v0.3.1 data root or trained hybrid/auto_r2 "
                           "model dirs not available (read-only "
                           "artifacts)")
def test_v031_real_cal_auto_forced_choices_match_references():
    """The brief's regression on REAL CAL rows of the v0.3.1 root
    (read-only; TRAIN+CALIBRATION only, HOLDOUT never requested):

    1. the trained auto model at the HYBRID-chosen taus predicts exactly
       like the accepted plain-hybrid run (median + all quantiles);
    2. an all-'hybrid' forced auto path (the plain hybrid model read as
       auto) reproduces its own hybrid predictions bit-for-bit/1e-9;
    3. an all-'ridge' forced auto path is the ridge-only construction
       everywhere and equals the natural auto predictions at the
       ridge-chosen taus;
    4. the ridge-chosen taus match an independently recomputed
       ridge-only reference on the same TRAIN rows.

    Regression: _base_part once dropped the ridge point for auto hybrid
    taus (CAL MAE tau30 1891 vs 86) and load() once mapped /2 hybrid
    pkls to the pure variant."""
    silent = lambda *_: None                     # noqa: E731
    man = load_manifest(_V031_DATA, verify_files=False, log=silent)
    ds = V0Dataset(man, feature_set="deployable", log=silent)
    hyb = GbmValueModel.load(_V031_HYBRID_DIR, log=silent)
    auto = GbmValueModel.load(_V031_AUTO_R2_DIR, log=silent)
    assert hyb.variant == "hybrid"       # /2 body records it (regression)
    assert auto.variant == "auto"
    assert auto.tau_variants == {30.0: "hybrid", 60.0: "hybrid",
                                 120.0: "hybrid", 300.0: "ridge",
                                 570.0: "ridge"}
    assert auto.data_manifest_sha256 == man.sha256
    assert hyb.data_manifest_sha256 == man.sha256
    side = json.loads((_V031_AUTO_R2_DIR / "gbm_value.json")
                      .read_text("utf-8"))
    cal_folds = int(side["cal_folds"])

    cal = ds.rows.index[ds.rows["split"] == "CALIBRATION"].to_numpy()
    taus = ds.rows["tau"].to_numpy()[cal]
    # v0.3.2 fix #11: a loaded deployable model predicts with pre_fix
    # FIXED to the post-fix value, so every raw-matrix reference below
    # must use the same override or it disagrees by construction.
    X, _names = row_feature_matrix(
        ds, cal, pre_fix_override=PRE_FIX_PREDICT_VALUE)
    qh = hyb.predict_quantiles_rows(ds, cal, "delay")
    qa = auto.predict_quantiles_rows(ds, cal, "delay")

    # (1) natural auto == plain hybrid at the HYBRID-chosen taus
    for tau in (30.0, 60.0, 120.0):
        m = np.isclose(taus, tau)
        assert m.any()
        for k in qh:
            d = float(np.max(np.abs(qa[k][m] - qh[k][m])))
            assert d <= 1e-9, (tau, k, d)

    # (2) forced all-'hybrid' auto path == plain hybrid
    hyb_auto = GbmValueModel.load(_V031_HYBRID_DIR, log=silent)
    hyb_auto.variant = "auto"            # tau_variants empty -> all hybrid
    qf = hyb_auto.predict_quantiles_rows(ds, cal, "delay")
    for k in qh:
        assert np.allclose(qf[k], qh[k], rtol=0.0, atol=1e-9), k
    assert np.allclose(hyb_auto.point_matrix(X, taus, "delay"),
                       hyb.point_matrix(X, taus, "delay"),
                       rtol=0.0, atol=1e-9)

    # (3) forced all-'ridge' == ridge-only construction; == natural at
    #     the ridge-chosen taus
    auto_r = GbmValueModel.load(_V031_AUTO_R2_DIR, log=silent)
    auto_r.tau_variants = {t: "ridge" for t in auto_r.taus}
    qr = auto_r.predict_quantiles_rows(ds, cal, "delay")
    qs_sorted = sorted(qr)
    for tau in auto_r.taus:
        m = np.isclose(taus, tau)
        base = auto_r.point_models[(float(tau), "delay")].predict(X[m])
        for qq in qs_sorted:
            off = max(auto_r.shifts[(float(tau), "delay", a)]
                      for a in qs_sorted if a <= qq + 1e-12)
            assert np.max(np.abs(qr[qq][m] - (base + off))) <= 1e-9, \
                (tau, qq)
    for tau in (300.0, 570.0):
        m = np.isclose(taus, tau)
        for k in qr:
            assert float(np.max(np.abs(qr[k][m] - qa[k][m]))) <= 1e-9, \
                (tau, k)
        # (4) independent ridge-only reference on the same TRAIN rows
        ref = _ridge_only_reference(ds, tau, X, taus, cal_folds)
        for qq in qs_sorted:
            d = float(np.max(np.abs(qa[qq][m] - ref[qq])))
            assert d <= 1e-9, (tau, qq, d)
