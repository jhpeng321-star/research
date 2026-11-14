"""combine CAL-half guards: blend/calibration/variant choice read CAL-cal only.

Synthetic world lists + a synthetic split file; no real CAL data.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from combine import guards as G
from combine import select_variant as SV
from evalh import cal_split as CS


def _split(tmp_path, *, tune=("t1", "t2"), cal=("c1", "c2")):
    p = tmp_path / "CAL_SPLIT_A4.json"
    p.write_text(json.dumps({
        "schema": CS.CAL_SPLIT_SCHEMA, "seed": 20261001,
        "cal_tune": list(tune), "cal_cal": list(cal)}), encoding="utf-8")
    return p


def test_selection_path_refuses_cal_cal(tmp_path):
    p = _split(tmp_path)
    assert G.selection_worlds(None, cal_split_path=p) == ["t1", "t2"]
    with pytest.raises(CS.CalSplitError):
        G.selection_worlds(["c1"], cal_split_path=p)


def test_calibration_path_refuses_cal_tune(tmp_path):
    p = _split(tmp_path)
    assert G.calibration_worlds(None, cal_split_path=p) == ["c1", "c2"]
    with pytest.raises(CS.CalSplitError):
        G.calibration_worlds(["t1"], cal_split_path=p)


def test_guard_cal_only_half_refuses_other_half(tmp_path):
    p = _split(tmp_path)
    splits = np.array(["CALIBRATION", "CALIBRATION"])
    with pytest.raises(CS.CalSplitError):
        G.guard_cal_only(splits, "blend", half=CS.CAL_CAL,
                         worlds=["t1", "t2"], cal_split_path=p)
    assert G.guard_cal_only(splits, "blend", half=CS.CAL_CAL,
                            worlds=["c1", "c2"], cal_split_path=p)


def test_guard_cal_only_half_needs_worlds(tmp_path):
    p = _split(tmp_path)
    splits = np.array(["CALIBRATION"])
    with pytest.raises(G.SplitLeakError):
        G.guard_cal_only(splits, "blend", half=CS.CAL_CAL,
                         cal_split_path=p)


def test_variant_choice_refuses_other_half(tmp_path):
    p = _split(tmp_path)
    taus = (60.0, 120.0, 300.0)
    counts = len(taus) * len(SV.DEFAULT_CONTRASTS)
    flat = lambda v: {float(t): {c: v for c in SV.DEFAULT_CONTRASTS}  # noqa: E731
                      for t in taus}
    crps = {v: flat(1.0) for v in SV.VARIANTS}
    coverage = {v: 0.80 for v in SV.VARIANTS}
    with pytest.raises(CS.CalSplitError):
        SV.select_planner_variant(crps=crps, coverage=coverage,
                                  worlds=["t1"], cal_half=CS.CAL_CAL,
                                  cal_split_path=p)
    rec = SV.select_planner_variant(crps=crps, coverage=coverage,
                                    worlds=["c1"] * counts,
                                    cal_half=CS.CAL_CAL, cal_split_path=p)
    assert rec["chosen"] in SV.VARIANTS
