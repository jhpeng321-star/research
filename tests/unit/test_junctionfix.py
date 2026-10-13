"""JUNCTIONFIX: junction-transit rows of TRAIN pack windows (``data.junction_fix``).

Synthetic tests of :func:`wms.data.junction_fix.fill_junction_rows` (always run) plus real
TRAIN-window tests (skipped when the E:/research-dense data cache is absent): ``off`` is
bit-identical to the pre-JUNCTIONFIX loader (golden record hashes captured with c4/int a8f552e),
``ffill`` removes the frozen-s lane-0 rows and the fake 1->0->1 changes and keeps the recorded
(loss-valid) changes.  Only TRAIN windows of configs/c4_wm_v3.toml / v4 are read.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.config.load import load_train_config
from wms.data.junction_fix import fill_junction_rows, pre_t0_rows, run_prefix

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("WMS_JUNCTIONFIX_DATA_ROOT", "E:/research-dense"))
HAS_DATA = (DATA_ROOT / "data_cache" / "pack_v3" / "manifest.json").is_file()


# ----------------------------------------------------------------------------- synthetic
def _teacher(rows):
    """rows: list over t of list over k of (s, lane) -> teacher [T,K,4] (v = 10, a = 0)."""
    t_n, k = len(rows), len(rows[0])
    out = np.zeros((t_n, k, 4), np.float32)
    for t, row in enumerate(rows):
        for j, (s, lane) in enumerate(row):
            out[t, j] = (s, lane, 10.0, 0.0)
    return out


def _fill(teacher, occupied=None, same_next=None, pre=None):
    t_n, k = teacher.shape[:2]
    occupied = np.ones((t_n, k), bool) if occupied is None else occupied
    same_next = np.ones((t_n - 1, k), bool) if same_next is None else same_next
    if pre is None:
        pre = (np.full(k, -1), np.zeros(k, np.float32), np.zeros(k, np.int64))
    return fill_junction_rows(teacher, occupied, same_next, *pre)


def test_transit_keeps_lane_and_interpolates_s():
    teach = _teacher([[(70.0, 1)], [(70.0, -1)], [(70.0, -1)], [(85.0, 1)], [(90.0, 1)]])
    out, filled, unfilled = _fill(teach)
    assert filled[:, 0].tolist() == [False, True, True, False, False] and not unfilled.any()
    assert out[:, 0, 1].tolist() == [1, 1, 1, 1, 1]                 # no 1 -> 0 -> 1
    np.testing.assert_allclose(out[:, 0, 0], [70, 75, 80, 85, 90])  # no frozen s
    np.testing.assert_array_equal(out[:, 0, 2:], teach[:, 0, 2:])   # v, a untouched


def test_real_change_and_on_lane_rows_untouched():
    teach = _teacher([[(10.0, 1)], [(20.0, 0)], [(30.0, 0)], [(40.0, 1)]])
    out, filled, _ = _fill(teach)
    np.testing.assert_array_equal(out, teach)
    assert not filled.any()
    # a change across a junction is kept at the first on-lane row after the transit
    teach = _teacher([[(70.0, 2)], [(70.0, -1)], [(80.0, 1)]])
    out, _, _ = _fill(teach)
    assert out[:, 0, 1].tolist() == [2, 2, 1]


def test_transit_at_window_end_holds_s():
    teach = _teacher([[(60.0, 1)], [(70.0, 1)], [(70.0, -1)], [(70.0, -1)]])
    out, filled, _ = _fill(teach)
    assert filled[2:, 0].all()
    np.testing.assert_allclose(out[:, 0, 0], [60, 70, 70, 70])
    assert out[:, 0, 1].tolist() == [1, 1, 1, 1]


def test_t0_transit_uses_pre_t0_row_or_stays_unfilled():
    teach = _teacher([[(74.0, -1), (74.0, -1)], [(74.0, -1), (74.0, -1)], [(84.0, 1), (84.0, 1)]])
    pre = (np.array([1, -1]), np.array([64.0, 0.0], np.float32), np.array([1, 0]))
    out, filled, unfilled = _fill(teach, pre=pre)
    np.testing.assert_allclose(out[:, 0, 0], [70.666667, 77.333333, 84], rtol=1e-6)
    assert out[:, 0, 1].tolist() == [1, 1, 1] and filled[:2, 0].all()
    assert unfilled[:2, 1].all() and out[0, 1, 1] == -1             # no history: unfilled


def test_new_vehicle_in_a_reused_slot_is_not_filled_from_the_old_one():
    teach = _teacher([[(70.0, 0)], [(-2.0, -1)], [(5.0, 1)]])
    same_next = np.array([[False], [True]])                          # slot reused at t=1
    out, filled, unfilled = _fill(teach, same_next=same_next)
    assert unfilled[1, 0] and not filled.any() and out[1, 0, 1] == -1


def test_absent_rows_are_ignored():
    teach = _teacher([[(70.0, 1)], [(0.0, 0)], [(0.0, 0)]])
    occupied = np.array([[True], [False], [False]])
    out, filled, _ = _fill(teach, occupied=occupied)
    np.testing.assert_array_equal(out, teach)
    assert not filled.any()


def test_run_prefix_stops_at_slot_reuse():
    present = np.array([[1, 1], [1, 1], [1, 1]], bool)
    cont = np.array([[1, 1], [0, 1]], bool)
    got = run_prefix(present, cont, np.array([True, False]))
    assert got[:, 0].tolist() == [True, True, False] and not got[:, 1].any()


class _FakeCorpus:
    def __init__(self, kin, occ, same, start=0):
        self.arrays = {"veh_kin": kin, "veh_mask": occ, "target_mask": same}
        self.start = start

    def run_start(self, tick):
        return self.start

    def array(self, name):
        return self.arrays[name]


def test_pre_t0_rows_follow_the_same_vehicle_only():
    kin = np.zeros((4, 3, 7), np.float32)
    kin[:, :, 0] = np.arange(4)[:, None] * 10.0
    kin[0:2, :, 2] = 1.0                    # lane 1 at ticks 0-1, junction at ticks 2-3
    occ = np.ones((4, 3), bool)
    same = np.ones((4, 3), bool)
    same[0, 1] = False                      # slot 1: a different vehicle before tick 1
    same[1, 2] = False                      # slot 2: different vehicle before tick 2
    lane, s, age = pre_t0_rows(_FakeCorpus(kin, occ, same), 3, np.array([True, True, True]))
    assert lane.tolist() == [1, 1, -1] and s.tolist()[:2] == [10.0, 10.0]
    assert age.tolist()[:2] == [2, 2]


# ------------------------------------------------------------------------------- config
def test_v4_config_differs_from_v3_only_by_the_switch_and_out_dir():
    from wms.config.train import TrainConfig
    from wms.train.settings import TrainSettings

    v4 = load_train_config("configs/c4_wm_v4.toml", ["runtime.device=cpu"])
    v3 = load_train_config("configs/c4_wm_v3.toml", ["runtime.device=cpu"])
    f4, f3 = v4.to_legacy_flat(), v3.to_legacy_flat()
    diff = {k for k in set(f4) | set(f3) if f4.get(k) != f3.get(k)}
    assert diff == {"junction_fix", "out_dir"}
    assert f4["junction_fix"] == "ffill" and "junction_fix" not in f3
    assert f4["out_dir"] == "runs/active/c4_wm_v4"
    n4, n3 = v4.to_dict(), v3.to_dict()
    n4["data"].pop("junction_fix"), n3["data"].pop("junction_fix")
    n4["runtime"].pop("out_dir"), n3["runtime"].pop("out_dir")
    n4.pop("name"), n3.pop("name")
    assert n4 == n3
    assert TrainConfig.from_legacy_flat(f4).to_legacy_flat() == f4
    assert TrainSettings.from_flat(f4).junction_fix == "ffill"
    with pytest.raises(ValueError):
        TrainSettings.from_flat({"junction_fix": "bogus"})


# ------------------------------------------------------------------------------ real data
def record_hash(rec) -> str:
    h = hashlib.sha256()

    def feed(key, value):
        h.update(key.encode())
        if isinstance(value, dict):
            for k in sorted(value):
                feed(key + "." + k, value[k])
        elif isinstance(value, np.ndarray):
            a = np.ascontiguousarray(value)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
        elif torch.is_tensor(value):
            feed(key, value.numpy())
        else:
            h.update(repr(value).encode())

    for k in sorted(rec):
        feed(k, rec[k])
    return h.hexdigest()[:16]


#: configs/c4_wm_v3.toml TRAIN mix, torch.manual_seed(1000 + w), c4/int a8f552e (pre-JUNCTIONFIX).
GOLDEN_OFF = {0: "395850e032c4b3e9", 7: "5f04ddaca460f5f9", 300: "8a17889202e95a11",
              582: "30597c651c2d9138", 583: "8c2a9c59bc9429f9", 1200: "22915406f3fabef2",
              2040: "dfbee64c4ecfcb80"}


@pytest.fixture(scope="module")
def train_sets(tmp_path_factory):
    if not HAS_DATA:
        pytest.skip("C4 TRAIN data cache not present")
    sys.path.insert(0, str(ROOT / "tools"))
    from wmlane_smoke import point_data

    point_data(DATA_ROOT)
    from wms.cli.assemble import build_datasets

    out = {}
    for name in ("c4_wm_v3", "c4_wm_v4"):
        cfg = load_train_config(f"configs/{name}.toml", ["runtime.device=cpu"])
        out[name], _, _ = build_datasets(cfg, tmp_path_factory.mktemp(name))
    return out


def _item(ds, w, seed=1000):
    torch.manual_seed(seed + w)
    return ds[w]


@pytest.mark.parametrize("w", sorted(GOLDEN_OFF))
def test_off_is_bit_identical_to_the_pre_fix_loader(train_sets, w):
    assert record_hash(_item(train_sets["c4_wm_v3"], w)) == GOLDEN_OFF[w]


def test_ffill_batch_collates_with_the_same_keys_and_shapes(train_sets):
    from wms.data.collate import collate

    off = collate([_item(train_sets["c4_wm_v3"], w) for w in (0, 583)])
    on = collate([_item(train_sets["c4_wm_v4"], w) for w in (0, 583)])
    assert set(off) == set(on)
    for key, value in off.items():
        if torch.is_tensor(value):
            assert value.shape == on[key].shape and value.dtype == on[key].dtype, key


def _census(rec, junction):
    st, pr, lv, cav = rec["states"], rec["present"], rec["loss_valid"], rec["is_cav"]
    lanes = st[..., 1:4].argmax(-1)
    hdv = pr & ~cav
    both = hdv[:-1] & hdv[1:]
    change = both & (lanes[:-1] != lanes[1:])
    jn_end = junction[:-1] | junction[1:]
    fake_101 = 0
    for t in range(1, st.shape[0] - 1):
        fake_101 += int((hdv[t - 1] & hdv[t] & hdv[t + 1] & (lanes[t - 1] == 1) & (lanes[t] == 0)
                         & (lanes[t + 1] == 1) & junction[t]).sum())
    frozen = np.zeros_like(pr)
    frozen[1:] = np.abs(st[1:, :, 0] - st[:-1, :, 0]) < 1e-4
    lvb = lv & ~cav
    real = lvb[:-1] & lvb[1:] & (lanes[:-1] != lanes[1:])
    return {"junction_end_changes": int((change & jn_end).sum()), "fake_101": fake_101,
            "frozen_lane0_junction": int((junction & frozen & (lanes == 0)).sum()),
            "teacher_minus1": int((junction & (np.rint(rec["cav_teacher"][:st.shape[0], :, 1])
                                               < 0)).sum()),
            "real_changes": real, "loss_valid": lv}


def test_ffill_removes_the_artifact_and_keeps_recorded_changes(train_sets):
    off_ds, on_ds = train_sets["c4_wm_v3"], train_sets["c4_wm_v4"]
    totals = {"off": {}, "on": {}}
    for w in (0, 7, 50, 300, 582, 583, 900, 1200, 2040):
        off, on = _item(off_ds, w), _item(on_ds, w)
        junction = off["present"] & ~off["loss_valid"]
        c_off, c_on = _census(off, junction), _census(on, junction & on["present"])
        for key in ("junction_end_changes", "fake_101", "frozen_lane0_junction",
                    "teacher_minus1"):
            totals["off"][key] = totals["off"].get(key, 0) + c_off[key]
            totals["on"][key] = totals["on"].get(key, 0) + c_on[key]
        # recorded (loss-valid) changes and supervised rows: unchanged except dropped slots
        dropped = off["present"][0] & ~on["present"][0]
        keep = ~dropped
        np.testing.assert_array_equal(c_off["real_changes"][:, keep], c_on["real_changes"][:, keep])
        np.testing.assert_array_equal(off["loss_valid"][:, keep], on["loss_valid"][:, keep])
        assert not (on["loss_valid"] & ~off["loss_valid"]).any()
        # on-lane rows and speeds are untouched
        lv = off["loss_valid"]
        np.testing.assert_array_equal(off["states"][lv], on["states"][lv])
        np.testing.assert_array_equal(off["states"][..., 4], on["states"][..., 4])
        if "road_present" in on:
            assert not (on["road_present"] & ~off["road_present"]).any()
    assert totals["off"]["fake_101"] > 0 and totals["off"]["frozen_lane0_junction"] > 0
    assert totals["off"]["junction_end_changes"] > 10 * max(1, totals["on"]["junction_end_changes"])
    assert totals["on"]["fake_101"] == 0
    assert totals["on"]["teacher_minus1"] == 0                 # every kept transit filled
    # the remaining frozen lane-0 junction rows are real lane-0 cars held at a window end
    assert totals["on"]["frozen_lane0_junction"] < totals["off"]["frozen_lane0_junction"] / 10
