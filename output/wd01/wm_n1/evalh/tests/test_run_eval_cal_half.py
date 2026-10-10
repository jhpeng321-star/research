"""evalh CAL-half path: selection vs calibration/variant refusal (A4.2)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evalh import cal_split as CS
from evalh import run_eval
from evalh import seals as S


def _split(tmp_path, *, tune=("t1", "t2"), cal=("c1", "c2")):
    p = tmp_path / "CAL_SPLIT_A4.json"
    p.write_text(json.dumps({
        "schema": CS.CAL_SPLIT_SCHEMA, "seed": 20261001,
        "cal_tune": list(tune), "cal_cal": list(cal)}), encoding="utf-8")
    return p


def test_selection_path_refuses_cal_cal(tmp_path):
    p = _split(tmp_path)
    assert run_eval.selection_worlds(None, cal_split_path=p) == ["t1", "t2"]
    with pytest.raises(CS.CalSplitError):
        run_eval.selection_worlds(["c1"], cal_split_path=p)


def test_calibration_path_refuses_cal_tune(tmp_path):
    p = _split(tmp_path)
    assert run_eval.calibration_worlds(None, cal_split_path=p) == ["c1", "c2"]
    with pytest.raises(CS.CalSplitError):
        run_eval.calibration_worlds(["t1"], cal_split_path=p)


def test_cal_half_rows_filters_and_refuses_foreign(tmp_path):
    p = _split(tmp_path)
    rows = [SimpleNamespace(world=w) for w in ("t1", "c1", "t2")]
    kept = run_eval.cal_half_rows(rows, CS.CAL_TUNE, cal_split_path=p)
    assert [r.world for r in kept] == ["t1", "t2"]
    with pytest.raises(CS.CalSplitError):
        run_eval.cal_half_rows(
            [SimpleNamespace(world="sealed")], CS.CAL_CAL, cal_split_path=p)


def test_parser_has_cal_half():
    p = run_eval.build_parser()
    ns = p.parse_args(["--synthetic-only", "--split", "cal", "--out", "o",
                       "--predictor", "esynth:variant", "--reference",
                       "esynth:hybrid", "--cal-half", "cal_cal"])
    assert ns.cal_half == "cal_cal"
    assert ns.synthetic_only is True


def test_eval_without_half_refused_before_marker(tmp_path):
    """A --split cal run with no --cal-half refuses BEFORE any seal read."""
    args = run_eval.build_parser().parse_args([
        "--synthetic-only", "--split", "cal", "--out", str(tmp_path / "out"),
        "--predictor", "esynth:variant", "--reference", "esynth:hybrid"])
    with pytest.raises(S.SealError, match="requires --cal-half"):
        run_eval.run(args)
    assert not (tmp_path / "out" / "seals" / S.CAL_MARKER).exists()
