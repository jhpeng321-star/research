"""End-to-end pipeline integration on SYNTHETIC rows: pairs ->
planned-schedule descriptors -> direct-dY fit -> TRAIN-OOB acceptance
+ stratified tables + artifacts (no real data, no CAL, no SUMO).

The fit-time grid is SHRUNK (SMALL_GRID) to keep the suite fast; the
registered 6-candidate grid itself is covered by test_reg and by the
real dry-run.
"""
from __future__ import annotations

import json

import numpy as np

from partB_v04b import fit_v04b as FI
from partB_v04b import pairs_v04b as PB
from partB_v04b import reg

SMALL_GRID = ((80, 0.05, 4, 20), (80, 0.1, 4, 10))


def _fit(variant, rows3, pairs3, ablation=True):
    return FI.fit_variant(pairs3, rows3, variant, log=lambda m: None,
                          ablation=ablation, grid=SMALL_GRID, cv_folds=3)


def test_pipeline_train_oob_end_to_end(rows3, pairs3, tmp_path):
    art = _fit("deployable_strict", rows3, pairs3, ablation=True)
    t = art["train_oob"]
    assert t["label"].startswith("TRAIN-OOB")
    a1 = t["a1"]
    assert len(a1["per_tau"]) == len(reg.TAUS)
    # the synthetic wall arms lower delay deterministically and the OOB
    # residuals are tiny relative to |dY| -> A3-i must pass everywhere
    a3 = t["a3"]
    assert a3["contrast"] == reg.HEADLINE_CONTRAST
    assert a3["passes"]
    # aux numbers are reported, never gating
    contrasts = {r["contrast"] for r in t["aux_records"]}
    assert {"damping-native", "pi-native", "fixed-native",
            "switching-native"} <= contrasts
    assert t["strata"], "s.12.1 stratified tables must be reported"
    FI.save_artifacts(art, rows3, tmp_path, log=lambda m: None)
    side = json.loads(
        (tmp_path / "fit_sidecar_deployable_strict.json").read_text(
            encoding="utf-8"))
    assert side["schema"] == reg.SCHEMA
    assert len(side["selection"]) == len(reg.TAUS)
    assert (tmp_path / "model_deployable_strict.pkl").exists()


def test_pipeline_detector_variant_fits(rows3, pairs3, tmp_path):
    art = _fit("deployable_detector", rows3, pairs3, ablation=False)
    assert art["models"]
    assert art["train_oob"]["a1"]["per_tau"]


def test_fit_refuses_non_train_rows(rows3, pairs3):
    from partB_v04b.cal_guard import CalReadNotAllowed
    p = PB.build_pairs(rows3)
    p.loc[0, "split"] = "CALIBRATION"
    try:
        _fit("deployable_strict", rows3, p, ablation=False)
    except CalReadNotAllowed as e:
        assert "CALIBRATION" in str(e)
    else:
        raise AssertionError("a CAL row reached the TRAIN fit")
