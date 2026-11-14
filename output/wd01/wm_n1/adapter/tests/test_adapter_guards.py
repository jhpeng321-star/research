"""Split/seal refusals for the adapter export guards."""
from __future__ import annotations

import pandas as pd
import pytest

from adapter.export import ExportGuardError, export_cal, export_oof
from adapter.predictor import WmN1Predictor
from combine.guards import SplitLeakError

import synth_adapter as synth


def _pred(worlds=("w0",)):
    return WmN1Predictor(synth.tiny_ensemble(), synth.fake_corpus(list(worlds)),
                         device="cpu")


def _tab(split, world="w0"):
    return pd.DataFrame([dict(row_id="r0", world=world, split=split, tau=60.0,
                              target="delay", policy="native", y=1.0,
                              t0_s=0.0, decision_t_s=0.0)])


def test_export_cal_refuses_non_cal(tmp_path):
    p = _pred()
    for split in ("TRAIN", "TEST", "HOLDOUT"):
        with pytest.raises(SplitLeakError):
            export_cal(p, _tab(split), tmp_path / f"x_{split}")


def test_export_cal_requires_explicit_split(tmp_path):
    p = _pred()
    tab = _tab("CALIBRATION").drop(columns=["split"])
    with pytest.raises(SplitLeakError):
        export_cal(p, tab, tmp_path / "nosplit")


def test_export_cal_refuses_chunk5(tmp_path):
    p = _pred(("chunk5/x",))
    with pytest.raises(ExportGuardError):
        export_cal(p, _tab("CALIBRATION", world="chunk5/x"), tmp_path / "c5")


def test_export_oof_refuses_sealed_splits(tmp_path):
    p = _pred()
    for split in ("TEST", "HOLDOUT", "CALIBRATION"):
        with pytest.raises(SplitLeakError):
            export_oof(p, _tab(split), tmp_path / f"o_{split}")


def test_export_oof_refuses_chunk5(tmp_path):
    p = _pred(("chunk5/x",))
    with pytest.raises(ExportGuardError):
        export_oof(p, _tab("TRAIN", world="chunk5/x"), tmp_path / "o5")
