"""End-to-end run_eval CLI tests (synthetic data and predictors only)."""
from __future__ import annotations

import csv
import json

import pytest

from evalh import predictors as P
from evalh import run_eval as R
from evalh import seals as S


def _args(tmp_path, split="train_oof", **extra):
    base = dict(out=str(tmp_path / "run"), predictor="esynth:variant",
                reference="esynth:hybrid", split=split, synthetic_n=8,
                seed=1, n_boot=200, sign_threshold=1.0,
                synthetic_only=True)
    base.update(extra)
    out = []
    for k, v in base.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        out += [flag, str(v)] if not isinstance(v, bool) else [flag]
    return out


def test_run_eval_writes_rows_before_report(tmp_path):
    args = R.build_parser().parse_args(_args(tmp_path))
    report = R.run(args)
    run = tmp_path / "run"
    assert (run / "rows.csv").is_file()
    assert (run / "REPORT.json").is_file()
    assert (run / "REPORT.md").is_file()
    rows = list(csv.reader((run / "rows.csv").open(encoding="utf-8")))
    assert len(rows) - 1 == report["n_rows"]
    assert report["decision"]["verdict"] in ("WM-N1_BEATS_HYBRID",
                                            "HYBRID_STAYS")
    assert report["delta_crps"]["damping-native"][60.0]["improves"] is True


def test_run_eval_main_returns_zero(tmp_path):
    assert R.main(_args(tmp_path)) == 0


def test_legacy_fixture_preserves_existing_output_before_scoring(tmp_path,
                                                                monkeypatch):
    args = R.build_parser().parse_args(_args(tmp_path))
    R.run(args)
    before = (tmp_path / "run" / "rows.csv").read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("rerun reached scoring")

    monkeypatch.setattr(R, "SyntheticSource", forbidden)
    with pytest.raises(S.SealError, match="fresh --out"):
        R.run(args)
    assert (tmp_path / "run" / "rows.csv").read_bytes() == before


def test_run_eval_final_test_without_flag_refused(tmp_path):
    with pytest.raises(S.SealError):
        R.run(R.build_parser().parse_args(_args(tmp_path, split="final_test")))


def test_run_eval_final_test_dry_run_checks_seals(tmp_path):
    args = _args(tmp_path, split="final_test", final_test=True, dry_run=True,
                 chunk5_sha=S.LF_SHA_CHUNK5, registration_sha=S.LF_SHA_WM_N1)
    with pytest.raises(S.SealError, match="final-test access"):
        R.run(R.build_parser().parse_args(args))
    assert not (tmp_path / "run" / "seals" / S.TEST_MARKER).exists()


def test_run_eval_production_options_refused(tmp_path):
    args = _args(tmp_path, marker_dir=str(tmp_path / "marker"))
    with pytest.raises(S.SealError, match="seal, marker"):
        R.run(R.build_parser().parse_args(args))
    assert not (tmp_path / "marker").exists()


def test_run_eval_cal_uses_generated_split_without_marker(tmp_path):
    split = tmp_path / "SYNTHETIC_CAL_SPLIT_A4.json"
    split.write_text(json.dumps({
        "schema": "wm_n1_cal_split_a4/1",
        "synthetic_fixture": True,
        "cal_tune": [f"w{i:03d}" for i in range(4)],
        "cal_cal": [f"w{i:03d}" for i in range(4, 8)]}), encoding="utf-8")
    args = _args(tmp_path, split="cal", cal_half="cal_tune",
                 cal_split_path=str(split))
    R.run(R.build_parser().parse_args(args))
    assert not (tmp_path / "run" / "seals" / S.CAL_MARKER).exists()


def test_legacy_refusal_precedes_sources_predictors_and_markers(
        tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy refused path reached data/predictor/seal")

    monkeypatch.setattr(S, "guard_split", forbidden)
    monkeypatch.setattr(P, "load_object", forbidden)
    monkeypatch.setattr(P, "load_predictor", forbidden)
    monkeypatch.setattr(R, "SyntheticSource", forbidden)
    for extra, reason in (
            ({"synthetic_only": False}, "requires --synthetic-only"),
            ({"source": "foreign:source"}, "custom --source"),
            ({"predictor": "real:checkpoint"}, "only esynth:variant"),
            ({"reference": "real:hybrid"}, "only esynth:variant")):
        args = R.build_parser().parse_args(_args(tmp_path, **extra))
        with pytest.raises(S.SealError, match=reason):
            R.run(args)
    assert not (tmp_path / "run").exists()


def test_synthetic_cal_refuses_unpinned_or_foreign_split_before_seal(
        tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("production seal was reached")

    monkeypatch.setattr(S, "guard_split", forbidden)
    missing = R.build_parser().parse_args(
        _args(tmp_path, split="cal", cal_half="cal_tune"))
    with pytest.raises(S.SealError, match="explicit generated"):
        R.run(missing)
    foreign = tmp_path / "CAL_SPLIT_A4.json"
    foreign.write_text("{}", encoding="utf-8")
    supplied = R.build_parser().parse_args(_args(
        tmp_path, split="cal", cal_half="cal_tune",
        cal_split_path=str(foreign)))
    with pytest.raises(S.SealError, match="generated sibling"):
        R.run(supplied)


def test_predictor_loader_errors():
    with pytest.raises(P.PredictorLoadError):
        P.load_predictor("no_colon_here")
    with pytest.raises(P.PredictorLoadError):
        P.load_predictor("esynth:NoSuchAttr")
    with pytest.raises(P.PredictorLoadError):
        P.load_predictor("evalh.sources:DEFAULT_TAUS")


def test_predictor_loader_factory():
    obj = P.load_predictor("esynth:variant")
    assert hasattr(obj, "predict_quantiles")
