"""Tests for the switching/direct + original data mix (wms.mixdata, trainer hooks)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from wms.data import mixdata as mx
from wms.data.action_timeline import CONTROLLERS
import mini_switch_pack as mini
from wms.core.errors import TrainDataRefusal
from wms.data.dataset import TrajectoryDataset
from wms.data.sources import validate_train_source

DIRECT = CONTROLLERS.index("direct")
WORLD = "w8857m2600p05"          # a TRAIN-style (non held-out) world, seed 8857


@pytest.fixture(autouse=True)
def _isolated_fixture_workspace(tmp_path, monkeypatch):
    # CI may use its own temporary root; synthetic tests own that root. The
    # actual CPU helper retains the real workspace boundary and new-dir refusal.
    monkeypatch.setattr(mini, "WORKSPACE", tmp_path)


def _family_vehicle(held: bool, held_family: int = 3, world: str = WORLD):
    seed = mx.world_seed(world)
    for i in range(400):
        vid = f"mainline.{i}"
        if (mx.direct_family(seed, vid) == held_family) == held:
            return vid
    raise AssertionError("no vehicle found")


def _heldout_doc(worlds=("w8859m2700p10",), family=3):
    return {"schema": mx.HELDOUT_SCHEMA, "held_family": family,
            "n_held_worlds": len(worlds), "held_worlds": list(worlds),
            "family_selection": {"clean": False, "train_val_overlap": 214},
            "overlap_guard": {"disjoint": True, "held_subset_of_test": True,
                              "overlap_guard_ok": True, "problems": []},
            "worlds": {w: {"n_families": 4, "held_family": family} for w in worlds}}


# ------------------------------------------------------------- held-out ----
def test_direct_family_golden_values_match_the_builder():
    # Seed 8859 held-family vehicles of ZERO_SHOT_HELDOUT.json are family 3.
    for vid in ("mainline.1001", "mainline.1037", "mainline.176", "ramp.108"):
        assert mx.direct_family(8859, vid) == 3
    assert [mx.direct_family(920002, v) for v in
            ("mainline.1001", "mainline.1037", "mainline.176", "mainline.229",
             "mainline.494")] == [1, 3, 1, 2, 0]
    assert mx.world_seed("w8859m2700p10") == 8859
    with pytest.raises(mx.MixDataRefusal):
        mx.world_seed("not-a-world")


def test_heldout_guard_refuses_inconsistent_docs_and_held_worlds(tmp_path):
    guard = mx.HeldOutGuard(_heldout_doc())
    assert guard.held_family == 3 and not guard.family_clean
    guard.refuse_worlds([WORLD], where="x")
    with pytest.raises(mx.MixDataRefusal, match="held-out worlds present"):
        guard.refuse_worlds([WORLD, "w8859m2700p10"], where="x")
    bad = _heldout_doc()
    bad["schema"] = "other/1"
    with pytest.raises(mx.MixDataRefusal):
        mx.HeldOutGuard(bad)
    bad = _heldout_doc()
    bad["overlap_guard"]["overlap_guard_ok"] = False
    with pytest.raises(mx.MixDataRefusal, match="overlap guard"):
        mx.HeldOutGuard(bad)
    bad = _heldout_doc()
    bad["n_held_worlds"] = 5
    with pytest.raises(mx.MixDataRefusal, match="bookkeeping"):
        mx.HeldOutGuard(bad)
    with pytest.raises(mx.MixDataRefusal, match="required and missing"):
        mx.HeldOutGuard.load(tmp_path / "absent.json")


# --------------------------------------------------------------- events ----
def test_event_steps_ignore_departures_via_continuity():
    ctrl = np.zeros((6, 2), np.int64)
    ctrl[2:, 0] = 1                      # col 0: native -> damping at step 2
    ctrl[3:, 1] = 2                      # col 1: ...
    assert mx.event_steps(ctrl) == (2, 6)
    cont = np.ones((5, 2), bool)
    cont[1, 0] = False                   # col 0's 1->2 change is a departure artefact
    assert mx.event_steps(ctrl, cont) == (3, 6)
    ctrl[4:, 1] = DIRECT
    assert mx.event_steps(ctrl, cont) == (3, 4)
    assert mx.event_steps(np.zeros((5, 3), np.int64)) == (5, 5)


# ------------------------------------------------------------ timelines ----
def _write_timeline_pack(root, ctrl, accel, target, items, *, split="TRAIN"):
    pack = root / "pack_switching_train"
    pack.mkdir(parents=True, exist_ok=True)
    np.save(pack / "controller_timeline.npy", ctrl.astype(np.int8))
    np.save(pack / "issued_acceleration_mps2.npy", accel.astype(np.float32))
    np.save(pack / "issued_target_speed_mps.npy", target.astype(np.float32))
    index = {"schema": mx.INDEX_SCHEMA, "split": split, "control_dt_s": 1.0,
             "control_steps": ctrl.shape[1], "k_max": ctrl.shape[2],
             "controllers": list(CONTROLLERS), "n_items": len(items), "items": items}
    path = root / "ACTION_TIMELINE_INDEX_train.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return pack, path


def _items(n, world=WORLD, valid=8):
    return [{"arm": "switching", "local_tick": 240 + 120 * i, "position": i,
             "snapshot_t_s": 120.0 + 60 * i, "tick": 240 + 120 * i,
             "valid_steps": valid, "world": world} for i in range(n)]


@pytest.fixture()
def tl_pack(tmp_path):
    n, steps, k = 4, 8, 3
    ctrl = np.zeros((n, steps, k), np.int8)
    acc = np.zeros((n, steps, k), np.float32)
    tgt = np.zeros((n, steps, k), np.float32)
    ctrl[1, 3:, 0] = 1                              # item 1: native -> damping
    ctrl[2, 2:5, 1] = DIRECT                         # item 2: direct segment
    acc[2, 2:5, 1], tgt[2, 2:5, 1] = 0.5, 18.0
    ctrl[3, :, 2] = 1                                # item 3: constant damping
    pack, index = _write_timeline_pack(tmp_path, ctrl, acc, tgt, _items(n, valid=6))
    return pack, index, ctrl


def test_switch_timelines_bind_slice_pad_and_check_join(tl_pack):
    pack, index, ctrl = tl_pack
    tl = mx.SwitchTimelines(pack, index)
    assert len(tl) == 4 and tl.worlds == [WORLD]
    with pytest.raises(mx.MixDataRefusal, match="exceeds stored"):
        tl.timeline(2, 10)
    t = tl.timeline(2, 8)
    assert t["controller"].shape == (8, 3) and t["controller"].dtype == np.int64
    assert t["controller"][2:5, 1].tolist() == [DIRECT] * 3
    assert t["acceleration_mps2"][2:5, 1].tolist() == [0.5] * 3
    assert t["target_speed_mps"][3, 1] == 18.0
    assert (t["controller"][6:] == 0).all()          # valid_steps=6 -> native afterwards
    assert tl.timeline(1, 4)["controller"].shape == (4, 3)
    tl.check_item(1, WORLD, "switching", 360)
    with pytest.raises(mx.MixDataRefusal, match="join mismatch"):
        tl.check_item(1, WORLD, "switching", 361)
    with pytest.raises(mx.MixDataRefusal, match="join mismatch"):
        tl.check_item(1, "w9999m1m1", "switching", 360)
    assert tl.full_scan() == {"items_with_direct": 1, "items_with_any_change": 2}


def test_switch_timelines_fail_closed_on_bad_inputs(tmp_path, tl_pack):
    pack, index, _ = tl_pack
    doc = json.loads(index.read_text(encoding="utf-8"))
    wrong_split = tmp_path / "wrong_split.json"
    wrong_split.write_text(json.dumps({**doc, "split": "VAL"}), encoding="utf-8")
    with pytest.raises(mx.MixDataRefusal, match="split"):
        mx.SwitchTimelines(pack, wrong_split)
    bad_vocab = tmp_path / "bad_vocab.json"
    bad_vocab.write_text(json.dumps({**doc, "controllers": ["a", "b"]}), encoding="utf-8")
    with pytest.raises(mx.MixDataRefusal, match="vocabulary"):
        mx.SwitchTimelines(pack, bad_vocab)
    bad_pos = tmp_path / "bad_pos.json"
    items = list(doc["items"])
    items[0] = {**items[0], "position": 7}
    bad_pos.write_text(json.dumps({**doc, "items": items}), encoding="utf-8")
    with pytest.raises(mx.MixDataRefusal, match="position"):
        mx.SwitchTimelines(pack, bad_pos)
    (pack / "issued_target_speed_mps.npy").unlink()
    with pytest.raises(mx.MixDataRefusal, match="missing"):
        mx.SwitchTimelines(pack, index)


def test_timeline_value_errors_surface_at_load(tmp_path):
    ctrl = np.zeros((1, 4, 2), np.int8)
    ctrl[0, 1, 0] = 9                                # outside the vocabulary
    pack, index = _write_timeline_pack(tmp_path, ctrl, np.zeros((1, 4, 2)),
                                       np.zeros((1, 4, 2)), _items(1, valid=4))
    tl = mx.SwitchTimelines(pack, index)
    with pytest.raises(ValueError):
        tl.timeline(0, 4)
    with pytest.raises(mx.MixDataRefusal, match="outside vocabulary"):
        tl.full_scan()


# --------------------------------------------------------------- filter ----
class _Book:
    def __init__(self, table):
        self.table = table

    def vehicle(self, position, slot):
        return self.table.get((position, slot))


def _fake_corpus(items, n_ticks=2000, k=3):
    tm = np.ones((n_ticks, k), np.uint8)
    return SimpleNamespace(
        _items=items, tick_offsets=lambda: None, meta=lambda i: items[i], pack_index=lambda i: 0,
        manifest_position=lambda i: i, array=lambda name: tm)


def _filter_setup(tl_pack, policy, **kw):
    pack, index, _ = tl_pack
    tl = mx.SwitchTimelines(pack, index)
    held_v, other_v = _family_vehicle(True), _family_vehicle(False)
    book = _Book({(2, 1): (WORLD, held_v), (1, 0): (WORLD, other_v),
                  (3, 2): (WORLD, other_v)})
    guard = mx.HeldOutGuard(_heldout_doc())
    flt = mx.SwitchItemFilter(tl, guard, book, policy=policy, **kw)
    corpus = _fake_corpus([{"world": WORLD, "arm": "switching", "tick": i["tick"]}
                           for i in tl.items])
    return flt, corpus


def test_filter_drops_held_family_direct_items_and_records_events(tl_pack):
    flt, corpus = _filter_setup(tl_pack, "drop")
    kept, receipt = flt(corpus, [0, 1, 2, 3])
    assert kept == [0, 1, 3]                         # item 2 has a held-family direct CAV
    assert receipt["n_dropped"] == 1 and receipt["n_items_with_held_family_direct"] == 1
    assert flt.events[0] == (8, 8) and flt.events[1] == (3, 8)
    assert flt.events[3] == (8, 8)                   # constant damping: no event
    assert 2 not in flt.events


def test_filter_policies_and_fail_closed_paths(tl_pack):
    flt, corpus = _filter_setup(tl_pack, "refuse")
    with pytest.raises(mx.MixDataRefusal, match="policy refuse"):
        flt(corpus, [0, 1, 2, 3])
    with pytest.raises(mx.MixDataRefusal, match="policy"):
        _filter_setup(tl_pack, "allow")
    flt, corpus = _filter_setup(tl_pack, "drop", max_drop_frac=0.1)
    with pytest.raises(mx.MixDataRefusal, match="drop"):
        flt(corpus, [0, 1, 2, 3])
    # a pack tick that does not match the index refuses
    flt, corpus = _filter_setup(tl_pack, "drop")
    corpus._items[1]["tick"] += 1
    with pytest.raises(mx.MixDataRefusal, match="join mismatch"):
        flt(corpus, [0, 1])
    # a held-out WORLD refuses (never silently dropped)
    pack, index, _ = tl_pack
    tl = mx.SwitchTimelines(pack, index)
    guard = mx.HeldOutGuard(_heldout_doc(worlds=(WORLD,)))
    flt = mx.SwitchItemFilter(tl, guard, _Book({}), policy="drop")
    corpus = _fake_corpus([{"world": WORLD, "arm": "switching", "tick": i["tick"]}
                           for i in tl.items])
    with pytest.raises(mx.MixDataRefusal, match="held-out worlds present"):
        flt(corpus, [0])


def test_direct_slot_without_identity_is_treated_as_held(tl_pack):
    pack, index, _ = tl_pack
    tl = mx.SwitchTimelines(pack, index)
    guard = mx.HeldOutGuard(_heldout_doc())
    flt = mx.SwitchItemFilter(tl, guard, _Book({}), policy="drop")
    corpus = _fake_corpus([{"world": WORLD, "arm": "switching", "tick": i["tick"]}
                           for i in tl.items])
    kept, receipt = flt(corpus, [0, 1, 2, 3])
    assert kept == [0, 1, 3] and receipt["n_direct_slots_without_identity"] == 1


def test_continuity_masks_departure_in_event_steps(tl_pack):
    flt, corpus = _filter_setup(tl_pack, "drop")
    tm = np.ones((2000, 3), np.uint8)
    tick1 = corpus._items[1]["tick"]
    tm[tick1 + 2 * 2: tick1 + 2 * 2 + 2, 0] = 0      # vehicle absent around step 2->3
    corpus.array = lambda name: tm
    flt(corpus, [1])
    assert flt.events[1] == (8, 8)                   # the only change is not a switch


# -------------------------------------------------------------- mixing -----
class _Src:
    def __init__(self, n, with_timeline=False):
        self.n, self.with_timeline = n, with_timeline
        self.receipt = {"mode": "fake", "n_records": n}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rec = {"states": np.zeros((5, 4, 7), np.float32), "i": i}
        if self.with_timeline:
            rec["action_timeline"] = {
                "controller": np.ones((2, 4), np.int64),
                "acceleration_mps2": np.zeros((2, 4), np.float32),
                "target_speed_mps": np.zeros((2, 4), np.float32), "control_dt_s": 1.0}
        return rec


def _mixed(ratio=0.5, over=3.0):
    events = np.asarray([[2, 99], [99, 99], [99, 20], [99, 99], [40, 99]], np.int64)
    return mx.MixedTrajectoryDataset(
        [mx.MixSource("original", _Src(8), share=1.0 - ratio),
         mx.MixSource("switching", _Src(5, True), share=ratio, events=events)],
        event_oversample=over)


def test_mixed_dataset_ratio_and_stage_dependent_oversampling():
    ds = _mixed(0.25, 3.0)
    assert len(ds) == 13
    w10 = np.asarray(ds.stage_weights(10.0))
    assert abs(w10[:8].sum() - 0.75) < 1e-9 and abs(w10[8:].sum() - 0.25) < 1e-9
    assert np.allclose(w10[:8], 0.75 / 8)            # original: uniform
    # 10 s stage: only item 0 (switch@2) has an event inside the horizon
    assert w10[8] / w10[9] == pytest.approx(3.0)
    assert w10[9] == w10[11]
    w30 = np.asarray(ds.stage_weights(30.0))          # item 2 (direct@20) joins
    assert w30[10] / w30[9] == pytest.approx(3.0)
    assert abs(w30[8:].sum() - 0.25) < 1e-9
    w120 = np.asarray(ds.stage_weights(120.0))        # every item has an event
    assert np.allclose(w120[8:], 0.25 / 5)
    rows = ds.describe_stages((10.0, 30.0, 120.0))
    assert rows[0]["sources"]["switching"]["n_event_items"] == 1
    assert rows[1]["sources"]["switching"]["n_event_items"] == 2
    assert rows[2]["sources"]["switching"]["event_fraction_sampled"] == 1.0
    assert rows[0]["sources"]["original"]["n_event_items"] == 0


def test_mixed_dataset_getitem_always_has_a_timeline():
    ds = _mixed()
    first, last = ds[0], ds[12]
    assert first["mix_source"] == "original" and first["action_timeline_synthetic"]
    assert first["action_timeline"]["controller"].shape == (2, 4)    # ceil((5-1)*0.5)
    assert (first["action_timeline"]["controller"] == 0).all()
    assert last["mix_source"] == "switching" and "action_timeline_synthetic" not in last
    assert ds.locate(-1) == (1, 4)
    with pytest.raises(IndexError):
        ds[13]
    with pytest.raises(mx.MixDataRefusal):
        mx.MixedTrajectoryDataset([], event_oversample=1.0)
    with pytest.raises(mx.MixDataRefusal):
        _mixed(over=0.5)
    with pytest.raises(mx.MixDataRefusal):
        mx.mix_datasets(_Src(2), {"train": _Src(2), "train_events": np.zeros((2, 2)),
                                  "val": None, "receipt": {}},
                        switch_ratio=1.0, event_oversample=1.0)


# -------------------------------------------------------- trainer hooks ----
def _tiny_arrays(n, with_timeline):
    states = np.zeros((n, 5, 2, 7), np.float32)       # 4 physics steps -> 2 control steps
    states[..., 1] = 1.0
    states[..., 4] = 10.0
    states[..., 0] = np.arange(5, dtype=np.float32)[None, :, None] * 5.0
    cav = np.zeros((n, 5, 2), bool)
    cav[:, :, 0] = True
    arrays = {"states": states, "is_cav": cav}
    if with_timeline:
        ids = np.zeros((n, 2, 2), np.int64)
        ids[:, 1, 0] = 1
        arrays.update(controller_timeline=ids,
                      controller_timeline_schema="wms_action_timeline/1",
                      issued_acceleration_mps2=np.zeros(ids.shape, np.float32),
                      issued_target_speed_mps=np.zeros(ids.shape, np.float32))
    return arrays


def test_start_shift_is_refused_with_timelines(tmp_path):
    with pytest.raises(TrainDataRefusal, match="start-shift"):
        TrajectoryDataset.from_real_packs(
            [tmp_path / "p"], identity_dirs=[tmp_path / "i"], start_shift_prob=0.5,
            timelines={0: object()})


def test_world_split_names_avoid_the_forbidden_token(tmp_path):
    windows = [{"pack": "pack_switching_train", "batch": "hetdata", "chunk": "hetdata",
                "world": f"w{8900 + i}m2700p05", "arm": "switching", "t0_s": 0.0,
                "key": f"hetdata/w{8900 + i}m2700p05__switching"} for i in range(10)]
    src = tmp_path / "windows.json"
    src.write_text(json.dumps({"schema": "wm_s_lpo_windows/1", "n_windows": 10,
                               "windows": windows}), encoding="utf-8")
    train, val = mx.write_world_split(src, tmp_path / "switch_valsplit", frac=0.2, seed=0)
    validate_train_source(train)
    validate_train_source(val)
    n_train = json.loads(open(train).read())["n_windows"]
    assert n_train == 8 and json.loads(open(val).read())["n_windows"] == 2


def _config():
    return {"schema": mx.DATA_CONFIG_SCHEMA,
            "switching": {"train_pack": "pack_switching_train",
                          "train_identities": "identities_switching_train",
                          "train_window_list": "windows_train_hetdata.json",
                          "train_timeline_index": "ACTION_TIMELINE_INDEX_train.json",
                          "heldout": "ZERO_SHOT_HELDOUT.json",
                          "required_train_files": ["pack_switching_train",
                                                   "windows_train_hetdata.json"],
                          "items": {"train": 4}, "runs": {"train": 1},
                          "k_max": 3, "control_steps": 8}}


def test_preflight_passes_and_refuses(tmp_path, tl_pack):
    pack, index, _ = tl_pack
    root = pack.parent
    ident = root / "identities_switching_train"
    ident.mkdir()
    (ident / "TRAIN_IDENTITY_RECEIPT.json").write_text(json.dumps(
        {"schema": "wm_n1_train_identity_sidecar/1", "split": "TRAIN"}), encoding="utf-8")
    np.save(ident / "identities.npy", np.full((4, 3), -1, np.int64))
    np.save(ident / "global_positions.npy", np.arange(4, dtype=np.int64))
    (ident / "vehicle_map.json").write_text("[]", encoding="utf-8")
    import hashlib
    (ident / "TRAIN_IDENTITY_RECEIPT.json").write_text(json.dumps(
        {"schema": "wm_n1_train_identity_sidecar/1", "split": "TRAIN", "files": {
            n: hashlib.sha256((ident / n).read_bytes()).hexdigest() for n in
            ("global_positions.npy", "identities.npy", "vehicle_map.json")}}), encoding="utf-8")
    (root / "windows_train_hetdata.json").write_text(json.dumps(
        {"n_windows": 1, "windows": [{"world": WORLD}]}), encoding="utf-8")
    (root / "ZERO_SHOT_HELDOUT.json").write_text(json.dumps(_heldout_doc()), encoding="utf-8")
    # every direct slot is unidentified -> the census counts it
    out = mx.preflight(_config(), root, full_scan=True)
    assert out["status"] == "passed"
    assert out["checks"]["held_family_census"]["n_direct_slots_without_identity"] == 1
    assert out["checks"]["full_scan"]["items_with_direct"] == 1
    with pytest.raises(mx.MixDataRefusal, match="drop cap"):
        mx.preflight(_config(), root, max_drop_frac=0.1)
    bad = _config()
    bad["switching"]["items"]["train"] = 5
    with pytest.raises(mx.MixDataRefusal, match="expected"):
        mx.preflight(bad, root)
    (root / "windows_train_hetdata.json").write_text(json.dumps(
        {"n_windows": 1, "windows": [{"world": "w8859m2700p10"}]}), encoding="utf-8")
    with pytest.raises(mx.MixDataRefusal, match="held-out worlds present"):
        mx.preflight(_config(), root)


# ----------------------------------------------- real mini-pack integration -
@pytest.mark.skipif(not mini.available(), reason="local hetpack_check mini pack absent")
def test_real_pack_binding_heldout_drop_and_batching(tmp_path):
    built = mini.build(tmp_path / "mini")
    kw = dict(pack=built["pack"], identities=built["identities"],
              window_list=built["windows"], index_path=built["index"],
              heldout_path=built["heldout"], horizon_steps=240, entries_v2=False,
              val_holdout_frac=0.0, work_dir=tmp_path / "out")
    out = mx.build_switch_datasets(**kw)
    ds, events = out["train"], out["train_events"]
    assert len(ds) == 6 and events.shape == (6, 2)
    assert events[0].tolist() == [120, 120] and events[2].tolist() == [10, 10]
    rec = ds[2]
    t = rec["action_timeline"]
    assert t["controller"].shape == (120, 192)
    cav = built["facts"][2]["cav_slots"]
    assert set(np.unique(t["controller"][12, cav])) == {DIRECT}
    assert (t["acceleration_mps2"][12, cav] > 0.3).all()
    assert (t["controller"][12, [s for s in range(192) if s not in cav]] == 0).all()
    assert rec["states"].shape[0] == 241 and rec["start_shift_ticks"] == 0
    # held-out family of a direct CAV -> items with direct segments are dropped
    fam = mx.direct_family(8993, f"mainline.{built['facts'][2]['cav_slots'][0]}")
    # Change exclusion metadata on the already-built fixture to introduce overlap.
    held = mini.build(tmp_path / "held")
    held_doc = json.loads(open(held["heldout"]).read())
    held_doc["held_family"] = fam
    from pathlib import Path
    Path(held["heldout"]).write_text(json.dumps(held_doc))
    kw.update(pack=held["pack"], identities=held["identities"],
              window_list=held["windows"], index_path=held["index"],
              heldout_path=held["heldout"], work_dir=tmp_path / "out2")
    dropped = mx.build_switch_datasets(**kw, max_drop_frac=1.0)
    assert len(dropped["train"]) < 6
    assert dropped["receipt"]["heldout"]["n_dropped"] >= 1
    with pytest.raises(mx.MixDataRefusal, match="held-out family"):
        mx.build_switch_datasets(**kw, family_policy="refuse")
    # a held-out world in the training window list refuses
    wl = json.loads(open(held["windows"]).read())
    wl["windows"][0]["world"] = "w8859m2700p10"
    bad = tmp_path / "held" / "bad_windows.json"
    bad.write_text(json.dumps(wl), encoding="utf-8")
    kw["window_list"] = str(bad)
    with pytest.raises(mx.MixDataRefusal, match="held-out worlds present"):
        mx.build_switch_datasets(**kw)


def test_original_trajectory_policy_keeps_constant_direct_ids_and_recorded_commands(tmp_path):
    from pathlib import Path
    built = mini.build(tmp_path / "orig", original=True)
    pack = Path(built["pack"])
    manifest = json.loads((pack / "manifest.json").read_text())
    for item in manifest["items"]:
        item["policy"] = "trajectory"
    (pack / "manifest.json").write_text(json.dumps(manifest))
    commands = np.lib.format.open_memmap(pack / "cav_plan_ch.npy", mode="r+")
    commands[:, :, 0] = .5             # normalized target -> 20 m/s
    commands[:, :, 1] = -.2            # normalized acceleration -> -.6 m/s2
    commands.flush()
    ds = TrajectoryDataset.from_real_packs([pack], identity_dirs=[built["identities"]],
        horizon_steps=240, identity_env_dir="")
    timeline = ds[0]["action_timeline"]
    assert (timeline["controller"] == DIRECT).all()
    assert np.allclose(timeline["target_speed_mps"], 20)
    assert np.allclose(timeline["acceleration_mps2"], -.6)


def test_explicit_switch_monitoring_world_overlap_refuses(tmp_path):
    built = mini.build(tmp_path / "overlap")
    with pytest.raises(mx.MixDataRefusal, match="worlds overlap"):
        mx.build_switch_datasets(pack=built["pack"], identities=built["identities"],
            window_list=built["windows"], val_window_list=built["windows"],
            index_path=built["index"], heldout_path=built["heldout"],
            horizon_steps=240, entries_v2=False)


def test_leave_pi_wall_out_removes_timeline_pi_and_packed_wall_windows(tmp_path):
    from pathlib import Path
    built = mini.build(tmp_path / "excluded")
    pack = Path(built["pack"])
    ctrl = np.lib.format.open_memmap(pack / "controller_timeline.npy", mode="r+")
    ctrl[0, 119, built["facts"][0]["cav_slots"][0]] = 2
    ctrl.flush()
    veh = np.lib.format.open_memmap(pack / "veh_step.npy", mode="r+")
    # This wall-owned entrant tick is inside every remaining window.
    veh[300, built["facts"][0]["cav_slots"][0], 13] = 1
    veh.flush()
    with pytest.raises(TrainDataRefusal, match="no TRAIN windows"):
        mx.build_switch_datasets(pack=built["pack"], identities=built["identities"],
            window_list=built["windows"], index_path=built["index"],
            heldout_path=built["heldout"], horizon_steps=240, entries_v2=False,
            val_holdout_frac=0, leave_pi_wall_out=True)


def test_entries_v2_windows_need_the_entries_model(tmp_path):
    pytest.importorskip("wms.model.entries")
    built = mini.build(tmp_path / "entries")
    out = mx.build_switch_datasets(
        pack=built["pack"], identities=built["identities"], window_list=built["windows"],
        index_path=built["index"], heldout_path=built["heldout"], horizon_steps=240,
        entries_v2=True, val_holdout_frac=0.0, work_dir=tmp_path / "out")
    assert out["train"][0]["entry_feat"].ndim == 1
