"""E2 sealed extra CAL-tune label pack tests (synthetic only)."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from evalh.scorer import arm_contrasts as AC
from evalh.scorer import extra_labels as XL

_COLS = ["world", "source", "arm", "split", "t0_s", "t", "tau",
         "tau_available", "total_delay_ref_s"]


def _row(world="w1", source="fixed1", arm=None, split="CALIBRATION", t0=0.0,
         t=0.0, tau=60.0, avail=1.0, delay=140.0):
    return dict(world=world, source=source, arm=arm or source, split=split,
                t0_s=t0, t=t, tau=tau, tau_available=avail,
                total_delay_ref_s=delay)


def _seal(tmp_path, rows, name="pack", *, manifest=True, corrupt=False):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    table = d / "outcomes.parquet"
    pd.DataFrame(rows, columns=_COLS).to_parquet(table, index=False)
    if manifest:
        sha = hashlib.sha256(table.read_bytes()).hexdigest()
        if corrupt:
            sha = "0" * 64
        (d / "SHA256SUMS").write_text(
            f"{sha}  outcomes.parquet\n", encoding="utf-8")
    return d


def _load(d, *, selected, allowed=None):
    return XL.load_extra_cal_labels(
        [str(d)], selected_keys=selected, taus=(30.0, 60.0, 120.0),
        allowed_sources=allowed if allowed is not None else AC.E2_ALLOWED_ARMS)


def test_loads_listed_rows(tmp_path):
    d = _seal(tmp_path, [_row(), _row(tau=120.0)])
    rows, receipt = _load(d, selected={("w1", "fixed1")})
    assert len(rows) == 2
    assert receipt[0]["tables"] == ["outcomes.parquet"]
    assert len(receipt[0]["sha256"]["outcomes.parquet"]) == 64


def test_refuses_unselected_world(tmp_path):
    d = _seal(tmp_path, [_row(world="w9")])
    with pytest.raises(XL.ExtraLabelRefusal):
        _load(d, selected={("w1", "fixed1")})


def test_refuses_unregistered_arm(tmp_path):
    d = _seal(tmp_path, [_row(source="fixed9", arm="fixed9")])
    with pytest.raises(XL.ExtraLabelRefusal):
        _load(d, selected={("w1", "fixed9")})


def test_refuses_missing_manifest(tmp_path):
    d = _seal(tmp_path, [_row()], manifest=False)
    with pytest.raises(XL.ExtraLabelRefusal):
        _load(d, selected={("w1", "fixed1")})


def test_refuses_sha_mismatch(tmp_path):
    d = _seal(tmp_path, [_row()], corrupt=True)
    with pytest.raises(XL.ExtraLabelRefusal):
        _load(d, selected={("w1", "fixed1")})


def test_refuses_foreign_split(tmp_path):
    d = _seal(tmp_path, [_row(split="HOLDOUT")])
    with pytest.raises(XL.ExtraLabelRefusal):
        _load(d, selected={("w1", "fixed1")})


def test_accepts_sealed_train_split_for_selected_item(tmp_path):
    # the E4 trajectory labels were written with the TRAIN token inside an
    # explicitly sealed CAL-tune pack; the (world, arm) gate is the authority.
    d = _seal(tmp_path, [_row(source="arm0", split="TRAIN")])
    rows, _ = _load(d, selected={("w1", "arm0")})
    assert len(rows) == 1


def test_refuses_empty_after_filters(tmp_path):
    d = _seal(tmp_path, [_row(tau=999.0)])
    with pytest.raises(XL.ExtraLabelRefusal):
        _load(d, selected={("w1", "fixed1")})


def test_refuses_non_directory(tmp_path):
    with pytest.raises(XL.ExtraLabelRefusal):
        _load(tmp_path / "nope", selected=set())


def test_refuses_no_paths():
    with pytest.raises(XL.ExtraLabelRefusal):
        XL.load_extra_cal_labels([], selected_keys=set(), taus=(60.0,),
                                 allowed_sources=AC.E2_ALLOWED_ARMS)


def test_selected_key_string_form(tmp_path):
    d = _seal(tmp_path, [_row()])
    rows, _ = _load(d, selected={"batch/w1__fixed1"})
    assert len(rows) == 1
