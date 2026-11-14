"""Adapter export output is accepted by combine stacking / oof_tree."""
from __future__ import annotations

import numpy as np

from adapter import checkpoint as CK
from adapter.export import export_cal, export_oof
from adapter.predictor import WmN1Predictor
from combine.crps import QuantilePred
from combine.oof_tree import FULL_FOLD, load_oof
from combine.stacking import fit_stack
from combine.synthetic import gaussian_pred, synthetic_rows

import synth_adapter as synth


def _predictor(worlds, **extra):
    ensemble = synth.tiny_ensemble()
    return WmN1Predictor(ensemble, synth.fake_corpus(list(worlds)),
                         device="cpu")


def test_export_cal_accepted_by_stacking(tmp_path):
    tab = synthetic_rows(n_train=4, n_cal=4, n_test=0, seed=1)
    tab = tab[tab["split"] == "CALIBRATION"].reset_index(drop=True)
    worlds = sorted(tab["world"].unique())
    p = _predictor(worlds)
    res = export_cal(p, tab, tmp_path / "cal")

    assert (tmp_path / "cal.parquet").is_file()
    assert (tmp_path / "cal.npz").is_file()
    lv = np.asarray(res["levels"], float)
    q = np.asarray(res["qvals"], float)
    assert q.shape == (len(tab), lv.size)
    assert np.all(np.diff(q, axis=1) >= -1e-12)      # monotone across 7 levels
    assert set(res["table"]["fold_id"]) == {FULL_FOLD}

    fn = QuantilePred(lv, q)
    ft = gaussian_pred(lv, np.zeros(len(tab)), 1.0)
    y = tab["y"].to_numpy(float)
    model, xf = fit_stack(fn, ft, y, tab["world"].to_numpy(),
                          tab["split"].to_numpy(), folds=2)
    assert np.isfinite(model.weight)
    assert model.predict(fn, ft).shape == (len(tab), lv.size)


def test_export_cal_roundtrips_through_oof_tree_loader(tmp_path):
    tab = synthetic_rows(n_train=4, n_cal=3, n_test=0, seed=3)
    tab = tab[tab["split"] == "CALIBRATION"].reset_index(drop=True)
    p = _predictor(sorted(tab["world"].unique()))
    res = export_cal(p, tab, tmp_path / "cal")
    loaded = load_oof(tmp_path / "cal")
    assert tuple(loaded.quantiles) == tuple(res["levels"])
    assert len(loaded.table) == len(tab)


def test_export_oof_train_only_documents_member_limit(tmp_path):
    tab = synthetic_rows(n_train=6, n_cal=0, n_test=0, seed=2)
    p = _predictor(sorted(tab["world"].unique()))
    res = export_oof(p, tab, tmp_path / "oof", folds=3)
    assert (tmp_path / "oof.parquet").is_file()
    assert res["kind"] == "TRAIN"
    assert res["member_oof_available"] is False
    assert "member_train_worlds" in res["limit_note"]
    assert set(res["table"]["model_kind"]) == {"full-in-sample"}
    assert set(res["table"]["fold_id"]) == {FULL_FOLD}


def test_export_oof_uses_eligible_members(tmp_path):
    tab = synthetic_rows(n_train=6, n_cal=0, n_test=0, seed=4)
    worlds = sorted(tab["world"].unique())
    ckpt = tmp_path / "ens"
    CK.save_ensemble(ckpt, synth.tiny_ensemble(),
                     member_train_worlds={str(i + 1): [w]
                                          for i, w in enumerate(worlds[:5])})
    p = WmN1Predictor.from_checkpoint(
        ckpt, synth.fake_corpus(worlds), device="cpu")
    res = export_oof(p, tab, tmp_path / "oof2", folds=3)
    assert res["member_oof_available"] is True
    assert set(res["table"]["model_kind"]) == {"oof"}
    assert res["folds"] == 3
    assert sorted(set(res["table"]["fold_id"])) == [0, 1, 2]
    # every fold's OOF rows are scored with a member subset
    assert len(res["fold_of"]) == len(worlds)
