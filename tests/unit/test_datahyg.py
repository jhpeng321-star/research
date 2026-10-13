"""DATAHYG: no designed controllers in the WM-S training windows (``data.controller_hygiene``).

Synthetic tests always run; real TRAIN-window tests are skipped without the E:/research-dense
data cache.  ``off`` is bit-identical to c4/int a2d3716 (golden record hashes of TRAIN and
validation windows of configs/c4_wm_v3.toml and v4, captured before this change).  Only TRAIN
windows are read; no guard / CAL / holdout file is opened.
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
from wms.data import controller_hygiene as H

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("WMS_DATAHYG_DATA_ROOT", "E:/research-dense"))
HAS_DATA = (DATA_ROOT / "data_cache" / "pack_v3" / "manifest.json").is_file()
NATIVE, DAMPING, DIRECT = 0, 1, 3


# ------------------------------------------------------------------ generator goldens
def test_switch_schedule_matches_the_generator_golden():
    # traffic_shepherd.switching_data.sample_switch_schedule (branch wms/swpack), direct-mix
    # choices (native, damping, direct), duration 1200 s, seed = world seed.
    assert H.switch_schedule(8857, "mainline.7")[:5] == [
        (0, 5, "native"), (5, 12, "damping"), (12, 14, "direct"), (14, 21, "damping"),
        (21, 23, "native")]
    assert H.switch_schedule(8857, "mainline.2")[:2] == [(0, 30, "native"), (30, 45, "damping")]
    s = H.scheduled_damping_seconds(8857, "mainline.7")
    assert s.shape == (1200,) and np.flatnonzero(s)[:9].tolist() == [5, 6, 7, 8, 9, 10, 11, 14, 15]
    assert H.world_seed("w8857m2600p05") == 8857


def test_dspeed_plan_matches_the_generator_golden():
    # output/campaign_stage1_v3/trajectory_arm.py (branch impl/dspeed-arm)
    seed = H.dspeed_seed("w8481m2700p02t480", 1)
    roster = [f"mainline.{i}" for i in (162, 165, 167, 173, 175)]
    assert H.dspeed_selected(roster, seed, 1) == ["mainline.162", "mainline.173"]
    segs = H.dspeed_segments(seed, "mainline.165")
    assert [k for _a, _b, k in segs] == [
        "random_target", "speedup_ramp", "random_target", "merge_aimed", "merge_aimed",
        "gap_opening", "gap_opening", "slowdown_ramp"]
    assert segs[0] == (0.0, 15.096643, "random_target") and segs[-1][1] == 120.0
    assert H.arm_index_of("arm3") == 3


# ------------------------------------------------------------------ dspeed synthetic
def _dspeed_run(rows=600, k=4):
    vm = np.ones((rows, k), bool)
    tm = np.ones((rows, k), bool)
    vm[:, 3] = False                                     # empty slot
    cav0 = np.array([True, True, False, False])
    vids = {0: "mainline.162", 1: "mainline.165", 2: "mainline.167"}
    return H.DspeedRun("w8481m2700p02t480", "arm1", vids, vm, tm, cav0), vm


def test_dspeed_run_directed_and_designed_ticks():
    run, _vm = _dspeed_run()
    seed = H.dspeed_seed("w8481m2700p02t480", 1)
    chosen = set(H.dspeed_selected(["mainline.162", "mainline.165"], seed, 1))
    assert [run_vid for run_vid in ("mainline.162", "mainline.165") if run_vid in chosen] == \
        [["mainline.162", "mainline.165"][k] for k in run.selected_slots]
    assert not run.directed[:, 2].any() and not run.directed[240:].any()   # HDV / after 120 s
    for k in run.selected_slots:
        assert run.directed[:240, k].all()
    expect = np.zeros(600, bool)
    for k in run.selected_slots:
        vid = {0: "mainline.162", 1: "mainline.165"}[k]
        for a, b, kind in H.dspeed_segments(seed, vid):
            if kind in H.DESIGNED_SEGMENTS:
                expect[max(0, int(np.floor(a * 2)) - 2):int(np.ceil(b * 2)) + 2] = True
    np.testing.assert_array_equal(run.designed, expect)
    assert not run.designed[250:].any()
    allowed = run.allowed_starts(40)
    for t in range(0, 560, 7):
        assert allowed[t] == (not run.designed[t:t + 41].any())


def test_dspeed_slot_reuse_ends_the_directed_span():
    rows, k = 300, 2
    vm = np.ones((rows, k), bool)
    tm = np.ones((rows, k), bool)
    tm[99] = False                                       # slots change vehicle at row 100
    run = H.DspeedRun("w1m1", "arm0", {0: "a", 1: "b"}, vm, tm, np.array([True, True]))
    for k_sel in run.selected_slots:
        assert run.directed[:100, k_sel].all() and not run.directed[100:, k_sel].any()


def _loader(start_shift_prob=0.5):
    from wms.data.dataset import _PackWindowLoader

    loader = object.__new__(_PackWindowLoader)
    loader.start_shift_prob = start_shift_prob
    return loader


def test_relabelled_dspeed_timeline():
    run, _vm = _dspeed_run()
    loader = _loader()
    tl = loader._dspeed_timeline(run, shift=200, horizon=121, k=4)
    assert tl["controller"].shape == (60, 4)
    for c in range(60):
        row = 200 + 2 * c
        expect = np.where(run.directed[row], DIRECT, NATIVE)
        np.testing.assert_array_equal(tl["controller"][c], expect)
    # rows >= 240 (after the 120 s directed window): every CAV native
    assert (tl["controller"][20:] == NATIVE).all() and (tl["controller"][:20, 2:] == NATIVE).all()
    assert not tl["command_known"].any()
    assert not tl["acceleration_mps2"].any() and not tl["target_speed_mps"].any()
    from wms.data.action_timeline import ActionTimeline

    ActionTimeline(tl["controller"], tl["acceleration_mps2"], tl["target_speed_mps"])


def test_hygiene_shift_moves_refused_starts_only():
    run, _vm = _dspeed_run()
    run.allowed_rows = np.flatnonzero(run.allowed_starts(40)[:500])
    allowed = set(run.allowed_rows.tolist())
    good = min(allowed)
    loader = _loader(0.5)
    assert loader._hygiene_shift(run, good) == good
    torch.manual_seed(0)
    for _ in range(20):
        assert loader._hygiene_shift(run, 0) in allowed
    assert _loader(0.0)._hygiene_shift(run, 0) == good       # validation: earliest allowed


# ---------------------------------------------------------------- direct-mix synthetic
def test_directmix_damping_cohort_entrant_and_unknown():
    rows, k = 121, 4
    vm = np.zeros((rows, k), bool)
    tm = np.ones((rows, k), bool)
    cav = np.zeros((rows, k), bool)
    vm[:, 0] = cav[:, 0] = True                           # t0 CAV mainline.2 (damping 30-45 s)
    vm[10:, 1] = cav[10:, 1] = True                       # entrant, named at the tick-50 snapshot
    vm[20:31, 2] = cav[20:31, 2] = True                   # entrant that leaves unnamed
    tm[30, 2] = False
    vm[:, 3] = True                                       # HDV
    recorded = np.zeros((120, k), np.int64)
    snaps = [(0, {0: "mainline.2"}), (50, {0: "mainline.2", 1: "mainline.1"})]

    def run(h, cav_rows=cav, rec=recorded):
        return H.directmix_damping_ticks("w8857m2600p05", 0, h, veh_mask=vm, target_mask=tm,
                                         is_cav=cav_rows, cohort_vid={0: "mainline.2"},
                                         recorded_ctrl=rec, snapshots=snaps)

    res = run(80)
    # mainline.1 (entrant) damping [18, 22) s = rows 36..43; mainline.2 from 30 s = row 60
    assert np.flatnonzero(res["damping"]).tolist() == list(range(36, 44)) + list(range(60, 81))
    assert np.flatnonzero(res["unknown"]).tolist() == list(range(20, 31))
    assert res["stats"]["recorded_damping_not_scheduled"] == 0
    only_cohort = cav.copy()
    only_cohort[:, 1:] = False
    assert not run(40, only_cohort)["damping"].any()       # 0-20 s of mainline.2: native
    rec = recorded.copy()
    rec[5, 0] = DAMPING                                    # a recorded damping label counts too
    out = run(40, only_cohort, rec)
    assert out["damping"][10:12].all() and out["stats"]["recorded_damping_not_scheduled"] == 1


# --------------------------------------------------------------------- plumbing
def test_v4_hygiene_config_differs_from_v3_only_by_the_switches_and_out_dir():
    from wms.config.train import TrainConfig
    from wms.train.settings import TrainSettings

    hy = load_train_config("configs/c4_wm_v4_hygiene.toml", ["runtime.device=cpu"])
    v3 = load_train_config("configs/c4_wm_v3.toml", ["runtime.device=cpu"])
    fh, f3 = hy.to_legacy_flat(), v3.to_legacy_flat()
    diff = {k for k in set(fh) | set(f3) if fh.get(k) != f3.get(k)}
    assert diff == {"junction_fix", "controller_hygiene", "out_dir"}
    assert fh["controller_hygiene"] == "strict" and fh["junction_fix"] == "ffill"
    assert "controller_hygiene" not in f3 and "junction_fix" not in f3
    assert fh["out_dir"] == "runs/active/c4_wm_v4_hygiene"
    nh, n3 = hy.to_dict(), v3.to_dict()
    for n in (nh, n3):
        n["data"].pop("junction_fix"), n["data"].pop("controller_hygiene")
        n["runtime"].pop("out_dir"), n.pop("name")
    assert nh == n3
    assert TrainConfig.from_legacy_flat(fh).to_legacy_flat() == fh
    assert TrainSettings.from_flat(fh).controller_hygiene == "strict"
    assert TrainSettings.from_flat(f3).controller_hygiene == "off"
    with pytest.raises(ValueError):
        TrainSettings.from_flat({"controller_hygiene": "bogus"})


def test_assemble_kwargs_omit_defaults():
    from types import SimpleNamespace

    from wms.cli.assemble import _junction_kwargs

    assert _junction_kwargs(SimpleNamespace(junction_fix="off", controller_hygiene="off")) == {}
    assert _junction_kwargs(SimpleNamespace(junction_fix="off", controller_hygiene="strict")) \
        == {"controller_hygiene": "strict"}


def _record(command_known=None):
    k, n = 3, 2
    tl = {"controller": np.full((n, k), DIRECT, np.int64),
          "acceleration_mps2": np.zeros((n, k), np.float32),
          "target_speed_mps": np.zeros((n, k), np.float32), "control_dt_s": 1.0}
    if command_known is not None:
        tl["command_known"] = np.full((n, k), command_known)
    return {"states": np.zeros((5, k, 7), np.float32), "present": np.ones((5, k), bool),
            "slot_index": np.zeros((5, k), np.int64), "action_timeline": tl}


def test_collate_carries_command_known_only_when_every_record_has_it():
    from wms.data.collate import collate

    assert "command_known" not in collate([_record(), _record()])["action_timeline"]
    assert "command_known" not in collate([_record(False), _record()])["action_timeline"]
    out = collate([_record(False), _record(True)])["action_timeline"]
    assert out["command_known"].dtype == torch.bool and out["command_known"].shape == (2, 2, 3)


def test_unknown_commands_refused_for_closed_loop_and_black_box_only():
    from types import SimpleNamespace

    from wms.data.collate import collate
    from wms.train.inputs import _refuse_unknown_commands

    tl = collate([_record(False)])["action_timeline"]
    _refuse_unknown_commands(tl, SimpleNamespace(cav_drive="teacher", black_box=False))
    for settings in (SimpleNamespace(cav_drive="closed_loop", black_box=False),
                     SimpleNamespace(cav_drive="teacher", black_box=True)):
        with pytest.raises(ValueError):
            _refuse_unknown_commands(tl, settings)
    ok = collate([_record(True)])["action_timeline"]
    _refuse_unknown_commands(ok, SimpleNamespace(cav_drive="closed_loop", black_box=False))


# ------------------------------------------------------------------------------ real data
#: torch.manual_seed(1000 + w), c4/int a2d3716 (pre-DATAHYG).  v3 TRAIN values equal the
#: JUNCTIONFIX goldens (a8f552e).
GOLDEN_OFF = {
    "c4_wm_v3": {"train0": "395850e032c4b3e9", "train7": "5f04ddaca460f5f9",
                 "train140": "dd3225921c9fb0af", "train300": "8a17889202e95a11",
                 "train582": "30597c651c2d9138", "train583": "8c2a9c59bc9429f9",
                 "train1200": "22915406f3fabef2", "train2040": "dfbee64c4ecfcb80",
                 "val0": "76f6d25a007d697c", "val50": "f35e9040deb73d88",
                 "val120": "74990b3e7ef228af", "val194": "87ed84b9190f065f"},
    "c4_wm_v4": {"train0": "e92e8aee4f28966b", "train7": "56348eccd31770bc",
                 "train140": "d47e7dc105983ae0", "train300": "921a7ce03e08ba94",
                 "train582": "484401c94f9793c6", "train583": "9955336305f404a4",
                 "train1200": "0a1da8cc98b68d0f", "train2040": "fd200031eeb6d848",
                 "val0": "db4899dae2b96b68", "val50": "2175f49332640249",
                 "val120": "bd387ce8498ede2b", "val194": "90cc911ab7578763"}}


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    if not HAS_DATA:
        pytest.skip("C4 TRAIN data cache not present")
    sys.path.insert(0, str(ROOT / "tools"))
    from wmlane_smoke import point_data

    point_data(DATA_ROOT)
    from wms.cli.assemble import build_datasets

    out = {}
    for name in ("c4_wm_v3", "c4_wm_v4", "c4_wm_v4_hygiene"):
        cfg = load_train_config(f"configs/{name}.toml", ["runtime.device=cpu"])
        out[name] = build_datasets(cfg, tmp_path_factory.mktemp(name))[:2]
    return out


def _item(ds, w, seed=1000):
    torch.manual_seed(seed + w)
    return ds[w]


def record_hash(rec) -> str:
    """Same digest as tests/unit/test_junctionfix.py::record_hash."""
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


@pytest.mark.parametrize("name", sorted(GOLDEN_OFF))
def test_off_is_bit_identical_to_a2d3716(datasets, name):
    train, val = datasets[name]
    for key, golden in GOLDEN_OFF[name].items():
        ds, w = (train, int(key[5:])) if key.startswith("train") else (val, int(key[3:]))
        assert record_hash(_item(ds, w)) == golden, (name, key)


def test_strict_disables_the_damping_saturated_direct_mix_source(datasets):
    train, val = datasets["c4_wm_v4_hygiene"]
    assert len(train) == 583 and len(val) == 99            # pack_v3 + pack_dspeed only
    rec = train.receipt["switching_disabled_by_controller_hygiene"]["controller_hygiene"]
    assert rec["source_disabled"] and rec["train"]["n_considered"] == 1458
    assert rec["train"]["drop_fraction"] > 0.6 and rec["val"]["n_considered"] == 162
    check = rec["train"]["identity_and_schedule_check"]
    # every recorded damping label is a scheduled damping second (schedule reconstruction)
    assert check["recorded_damping_steps"] > 100000
    assert check["recorded_damping_not_scheduled"] == 0
    hyg = train.receipt["controller_hygiene"]
    assert hyg["dspeed_items"] == 444 and hyg["dspeed_items_dropped_no_allowed_start"] == 0


def test_strict_dspeed_windows_avoid_designed_segments_and_relabel(datasets):
    from wms.data.collate import collate

    train, val = datasets["c4_wm_v4_hygiene"]
    off_train, _ = datasets["c4_wm_v4"]
    loader = train._loader
    horizon = loader.horizon_steps
    n_checked = 0
    for ds, idx in ((train, (139, 140, 200, 300, 450, 582)), (val, (23, 40, 98))):
        runs = ds._loader.hygiene_runs
        for w in idx:
            for seed in (1000, 2000, 3000):
                rec = _item(ds, w, seed)
                run = runs[w]
                s = rec["start_shift_ticks"]
                assert not run.designed[s:s + horizon + 1].any()
                tl = rec["action_timeline"]
                rows = s + 2 * np.arange(tl["controller"].shape[0])
                ok = rows < run.directed.shape[0]
                expect = np.where(run.directed[rows[ok]], DIRECT, NATIVE)
                np.testing.assert_array_equal(tl["controller"][ok], expect)
                assert not tl["command_known"].any() and not tl["acceleration_mps2"].any()
                n_checked += 1
    assert n_checked == 27
    # native pack_v3 windows: constant native, commands unknown, collate shapes unchanged
    rec = _item(train, 0)
    assert (rec["action_timeline"]["controller"] == NATIVE).all()
    on = collate([_item(train, w) for w in (0, 140)])
    off = collate([_item(off_train, w) for w in (0, 140)])
    assert set(on) == set(off)
    for key, value in off.items():
        if torch.is_tensor(value):
            assert value.shape == on[key].shape and value.dtype == on[key].dtype, key
    assert set(on["action_timeline"]) == set(off["action_timeline"]) | {"command_known"}
