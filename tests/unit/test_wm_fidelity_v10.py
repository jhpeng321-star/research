"""scripts/c4/wm_fidelity_v10.py: the v10 WM-fidelity scorer (pure helpers + CLI wiring).

No torch, no GPU, no checkpoint: the module is loaded via ``importlib.util`` and only its
torch-free helpers (selection, summary, path guard, CLI parser) are exercised.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "wm_fidelity_v10", REPO / "scripts" / "c4" / "wm_fidelity_v10.py")
WF = importlib.util.module_from_spec(_spec)
sys.modules["wm_fidelity_v10"] = WF
_spec.loader.exec_module(WF)


def _record(i: int) -> dict:
    return {"world": f"w{i:03d}", "key": f"k{i:03d}", "source_arm": "wall", "t0_s": float(i)}


def test_select_states_is_deterministic_and_evenly_spaced():
    records = [_record(i) for i in range(100)]
    chosen = WF.select_states(records, 48)
    assert len(chosen) == 48
    ordered = sorted(records, key=WF.selection_key)
    assert WF.selection_key(chosen[0]) == WF.selection_key(ordered[0])
    assert WF.selection_key(chosen[-1]) == WF.selection_key(ordered[-1])
    again = WF.select_states(records, 48)
    assert [WF.selection_key(r) for r in chosen] == [WF.selection_key(r) for r in again]
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    assert ([WF.selection_key(r) for r in WF.select_states(shuffled, 48)]
            == [WF.selection_key(r) for r in chosen])


def test_select_states_fewer_than_n():
    records = [_record(i) for i in range(3)]
    assert WF.select_states(records, 48) == sorted(records, key=WF.selection_key)


def _row(ds60, ds120, error=None) -> dict:
    car = {"60.0": {"hdv": {"ds_med": ds60}}, "120.0": {"hdv": {"ds_med": ds120}}}
    arm = {"error": error, "car": car} if error is None else {"error": error}
    return {"arms": {"native": arm}}


def test_summarize_reads_hdv_ds_med_and_blowups():
    rows = [_row(10.0, 20.0), _row(20.0, 40.0), _row(60.0, 150.0), _row(None, None, error="boom")]
    out = WF.summarize(rows, "native", (10, 30, 60, 120))
    assert out["schema"] == "c4_wm_fidelity_v10/1"
    assert out["hdv"]["abs_ds"]["60"] == 20.0
    assert out["horizons"]["60"]["hdv_abs_ds_med"] == 20.0
    assert out["hdv"]["abs_ds"]["10"] is None
    assert out["blowups"]["n"] == 1
    assert out["blowups"]["denominator"] == 3
    assert out["errors"] == 1
    assert out["n_states"] == 4


def test_restricted_paths_refused():
    with pytest.raises(SystemExit):
        WF.refuse_restricted(str(Path("data") / "holdout" / "states.json"))
    with pytest.raises(SystemExit):
        WF.refuse_restricted("runs/sealed/x.json")
    assert WF.refuse_restricted("data_cache/fixed_train_val_states.json").endswith(".json")


def test_cli_parses_and_requires_wm_and_out():
    ap = WF.parser()
    args = ap.parse_args(["--wm", "x.pt", "--states", "s.json", "--out", "o.json"])
    assert args.wm == "x.pt" and args.out == "o.json"
    assert args.device == "cpu" and args.select == 48 and args.entries == "off"
    with pytest.raises(SystemExit):
        ap.parse_args(["--states", "s.json", "--out", "o.json"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--wm", "x.pt", "--states", "s.json"])


def test_parse_horizons():
    assert WF.parse_horizons("10,30,60,120") == (10.0, 30.0, 60.0, 120.0)
