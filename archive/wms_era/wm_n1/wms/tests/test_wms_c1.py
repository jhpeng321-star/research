from __future__ import annotations

import json

import numpy as np
import pytest

from output.wd01.wm_n1.wms import c1, lpo


def test_window_and_item_key_are_batch_first_and_t0_rounded():
    window = {"batch": "truearms_chunk2", "chunk": "chunk2", "world": "w1",
              "arm": "native", "t0_s": 300.04}
    item = {"batch": "truearms_chunk2", "chunk": "chunk2", "world": "w1",
            "arm": "native", "t0_s": 300.04}
    assert lpo.window_key(window) == lpo.item_key(item) == (
        "truearms_chunk2", "w1", "native", 300.04)
    # chunk is only a fallback when batch is absent
    assert lpo.window_key({"chunk": "c", "world": "w", "arm": "a", "t0_s": 0})[0] == "c"


def test_exclude_runs_splits_prefix_from_run_id():
    keys = ["truearms_chunk2/w9__pi", "chunk1_fix27/w8__native", "bare"]
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pi_excluded_runs.json"
        p.write_text(json.dumps({"keys": keys}), encoding="utf-8")
        out = lpo.exclude_runs(p)
    assert ("truearms_chunk2", "w9__pi") in out
    assert ("chunk1_fix27", "w8__native") in out
    assert ("bare", "") in out


def test_filter_positions_matches_keep_and_fails_closed():
    items = [{"batch": "chunk1", "world": "w1", "arm": "native", "t0_s": 240.0},
             {"batch": "chunk2", "world": "w2", "arm": "pi", "t0_s": 240.0}]
    packs, names = [0, 0], ["pack_v3"]
    keep = {"pack_v3": {lpo.item_key(items[0])}}
    selected, receipt = lpo.filter_positions(items, packs, names, keep=keep)
    assert selected == [0]
    assert receipt["kept"] == 1 and receipt["matched"] == 1
    bad = {"pack_v3": {("chunk9", "wX", "native", 1.0)}}
    with pytest.raises(ValueError):
        lpo.filter_positions(items, packs, names, keep=bad)


def test_filter_positions_excludes_on_batch_not_colliding_chunk():
    # pack_v3 reuses chunk='chunk2' across the old and truearms batches; only the
    # batch named in the exclusion list may be dropped.
    items = [{"batch": "chunk2", "chunk": "chunk2", "world": "w1", "arm": "native",
              "t0_s": 300.0},
             {"batch": "truearms_chunk2", "chunk": "chunk2", "world": "w1",
              "arm": "native", "t0_s": 300.0}]
    packs, names = [0, 0], ["pack_v3"]
    selected, receipt = lpo.filter_positions(
        items, packs, names, exclude={("chunk2", "w1__native")})
    assert selected == [1]
    assert receipt["dropped"] == 1


def test_cav_affected_and_car_level_abs_ds():
    true_s = np.zeros((3, 2))
    cf = np.zeros((3, 2))
    true_s[1, 0] = 5.0  # HDV 0 is affected; HDV 1 is not
    affected = c1.cav_affected_hdvs(true_s, cf)
    assert affected.tolist() == [True, False]
    pred_s = np.zeros((3, 2))
    pred_s[1, 0] = 3.0
    out = c1.car_level_abs_ds(pred_s, true_s, affected, horizons_s=(1.0,), dt_s=1.0)
    assert out[1.0]["abs_ds_median_m"] == 2.0
    assert out[1.0]["n"] == 1


def test_traffic_level_and_paired_and_ratio():
    true_v = np.array([[20.0], [0.0]])
    pred_v = np.array([[20.0], [5.0]])
    sec = c1.traffic_level_per_section(pred_v, true_v, np.array([0]))
    assert sec[0]["speed_mae_mps"] == 2.5
    assert sec[0]["n_veh"] == 1
    pred_pi = np.array([[1.0], [2.0]])
    pred_nat = np.array([[0.0], [1.0]])
    true_pi = np.array([[2.0], [3.0]])
    true_nat = np.array([[0.0], [2.0]])
    paired = c1.paired_pi_native(pred_pi, pred_nat, true_pi, true_nat)
    assert paired["sign_agreement"] == 1.0
    assert paired["n"] == 1
    ratio = c1.seen_unseen_error_ratio([1.0, 2.0], [2.0, 4.0])
    assert ratio["ratio"] == 2.0


def test_protocol_descriptor_is_metadata_only():
    d = c1.protocol_descriptor()
    assert d["held_out_controller"] == "pi"
    assert d["horizons_s"] == [10.0, 30.0, 120.0, 300.0]
    assert "black_box" in d["controller_enters"]
