"""Generated metadata/shards only; no registered CAL or TRAIN data."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from evalh import real_source as RS
from evalh.real_source import (RealSourceRefusal, ReleasedCalSource,
                               select_cal_items)


def _item(world, split="CALIBRATION", first=0):
    return dict(batch="chunk2", chunk="chunk2", world=world, arm="native",
                policy="native", split=split, t0_s=300.0,
                tick=0, plan_row=0, outcome_rows=[first, first + 1])


def _run(world, split="CALIBRATION", first=0):
    return dict(key=f"chunk2/{world}__native", batch="chunk2",
                chunk="chunk2", world=world, arm="native", split=split,
                t0_s=300.0, tick_start=0, tick_end=241, dec_start=0,
                dec_end=20, n_ticks=241, outcome_first=first,
                n_outcome_rows=2)


def _metadata():
    items = [_item("c1"), _item("c2", first=2)]
    runs = [_run("c1"), _run("c2", first=2)]
    return ({"schema": "wm_n1_pack/1", "items": items},
            {"schema": "wm_n1_corpus_pack/2", "runs": runs})


def test_select_cal_items_only_named_half_from_pack_metadata():
    manifest, index = _metadata()
    picked = select_cal_items(manifest, index, ["c1"])
    assert len(picked) == 1
    assert picked[0][0] == 0
    assert picked[0][1]["world"] == "c1"


@pytest.mark.parametrize("change", [
    lambda m, i: m["items"][0].update(split="HOLDOUT"),
    lambda m, i: m["items"][0].update(batch="walls_pilot_j4b"),
    lambda m, i: m["items"][0].update(tick=241),
    lambda m, i: m["items"][0].update(outcome_rows=[0, 99]),
    lambda m, i: i["runs"].append(copy.deepcopy(i["runs"][0])),
])
def test_select_cal_items_refuses_protected_or_inconsistent_metadata(change):
    manifest, index = _metadata()
    change(manifest, index)
    with pytest.raises(RealSourceRefusal):
        select_cal_items(manifest, index, ["c1"])


def _label(t, tau):
    return dict(world="c1", source="native", split="CALIBRATION",
                t0_s=300.0, t=t, tau=tau, total_delay_ref_s=12.0,
                collision_any=0.0, ttc_lt3_veh_s=1.0, hard_brake_s=0.1,
                n_vehicles_present=10.0, tau_available=100.0)


def test_selected_labels_keep_only_item_t0_decision(tmp_path):
    # owner-approved fix (a): rows at later decision times are not the
    # item's targets and are skipped; only the t0 row is emitted.
    path = tmp_path / "selected-only.parquet"
    pd.DataFrame([_label(300.0, 60.0), _label(330.0, 60.0)])         .to_parquet(path, index=False)
    source = ReleasedCalSource.__new__(ReleasedCalSource)
    source.selected = ((0, _item("c1"), _run("c1")),)
    source._checked_file = lambda item, name: path
    rows = source.outcome_rows("cal")
    assert [(r.decision_t_s, r.tau_s) for r in rows] == [(300.0, 60.0)]


def test_selected_labels_refuse_duplicate_tau_even_at_same_decision(tmp_path):
    path = tmp_path / "selected-only.parquet"
    pd.DataFrame([_label(300.0, 60.0), _label(300.0, 60.0)]) \
        .to_parquet(path, index=False)
    source = ReleasedCalSource.__new__(ReleasedCalSource)
    source.selected = ((0, _item("c1"), _run("c1")),)
    source._checked_file = lambda item, name: path
    with pytest.raises(RealSourceRefusal, match="multiple decision labels"):
        source.outcome_rows("cal")


def test_selected_labels_emit_exact_t0_and_tau(tmp_path):
    path = tmp_path / "selected-only.parquet"
    pd.DataFrame([_label(300.0, 60.0), _label(300.0, 120.0)]) \
        .to_parquet(path, index=False)
    source = ReleasedCalSource.__new__(ReleasedCalSource)
    source.selected = ((0, _item("c1"), _run("c1")),)
    source._checked_file = lambda item, name: path
    rows = source.outcome_rows("cal")
    assert [(r.world, r.decision_t_s, r.tau_s) for r in rows] == [
        ("c1", 300.0, 60.0), ("c1", 300.0, 120.0)]


def test_released_source_never_accepts_other_split_without_reading_data():
    source = ReleasedCalSource.__new__(ReleasedCalSource)
    with pytest.raises(RealSourceRefusal, match="only CAL"):
        source.outcome_rows("final_test")
    with pytest.raises(RealSourceRefusal, match="only CAL"):
        source.rollout_rows("train_oof")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _released_fixture(tmp_path, monkeypatch):
    """A generated two-half pack with payload bytes only in the chosen half."""
    split_path = tmp_path / "CAL_SPLIT_A4.json"
    split_sha = _write_json(split_path, {
        "schema": "wm_n1_cal_split_a4/1", "cal_tune": ["c1"],
        "cal_cal": ["c2"]})
    monkeypatch.setattr(RS, "LF_SHA_CAL_SPLIT_A4", split_sha)
    manifest, index = _metadata()
    pack = tmp_path / "pack"
    manifest_sha = _write_json(pack / "manifest.json", manifest)
    index_sha = _write_json(pack / "INDEX.json", index)
    # Invalid combined payload and other-half payload are tripwires: neither
    # may be opened by the selected-half constructor or label reader.
    (pack / "outcomes.parquet").write_bytes(b"forbidden mixed table")
    shards = tmp_path / "shards"
    chosen = shards / "chunk2" / "c1" / "native"
    other = shards / "chunk2" / "c2" / "native"
    chosen.mkdir(parents=True)
    other.mkdir(parents=True)
    (other / "outcomes.parquet").write_bytes(b"forbidden other half")
    pd.DataFrame([_label(300.0, 60.0), _label(300.0, 120.0)]) \
        .to_parquet(chosen / "outcomes.parquet", index=False)
    n, k = 241, 2
    kin = np.zeros((n, k, 7), dtype=np.float16)
    kin[..., 0] = np.arange(n, dtype=np.float16)[:, None]
    kin[..., 4] = 5.0
    mask = np.ones((n, k), dtype=np.uint8)
    slot = np.broadcast_to(np.array([0, 1], dtype=np.int32), (n, k)).copy()
    slot[20:, 1] = 2                # slot reuse must exclude the original car
    target = np.ones((n, k), dtype=np.uint8)
    target[19, 1] = 0
    for name, arr in (("veh_kin", kin), ("veh_mask", mask),
                      ("slot_index", slot), ("target_mask", target)):
        np.save(chosen / f"{name}.npy", arr)
    files = {name: _file_sha(chosen / name) for name in (
        "outcomes.parquet", "veh_kin.npy", "veh_mask.npy",
        "slot_index.npy", "target_mask.npy")}
    done = dict(schema="wm_n1_features_index/1", batch="chunk2",
                chunk="chunk2", world="c1", arm="native",
                split="CALIBRATION", t0_s=300.0,
                outputs={"outcomes.parquet": files["outcomes.parquet"],
                         "veh_mask": files["veh_mask.npy"],
                         "slot_index": files["slot_index.npy"],
                         "target_mask": files["target_mask.npy"]})
    done_sha = _write_json(chosen / "DONE.json", done)
    idea_sha = _write_json(chosen / "IDEA_A_DONE.json", dict(
        schema="wm_n1_features_idea_a/1", batch="chunk2",
        chunk="chunk2", world="c1", arm="native",
        split="CALIBRATION", t0_s=300.0,
        arrays={"veh_kin": {"sha256": files["veh_kin.npy"]}}))
    spec = dict(schema="wm_n1_released_cal_source/1",
                allocation_id="generated-allocation",
                data_release_id="generated-release", source_head="synthetic",
                checkpoint_member_sha256="a" * 64, half="cal_tune",
                cal_split_path=str(split_path), pack_dir=str(pack),
                manifest_sha256=manifest_sha, index_sha256=index_sha,
                shards_root=str(shards), selected_shards={
                    "chunk2/c1__native": dict(done_sha256=done_sha,
                                               idea_a_done_sha256=idea_sha,
                                               files=files)})
    spec_path = tmp_path / "RELEASE.json"
    spec_sha = _write_json(spec_path, spec)
    return spec_path, spec_sha, chosen


def test_released_constructor_selects_before_any_payload_read(tmp_path, monkeypatch):
    path, sha, _ = _released_fixture(tmp_path, monkeypatch)
    source = ReleasedCalSource(path, expected_release_sha256=sha)
    assert source.half == "cal_tune"
    assert len(source.selected) == 1
    assert {r.world for r in source.outcome_rows("cal")} == {"c1"}


def test_released_source_rejects_changed_done_before_label_read(tmp_path,
                                                                 monkeypatch):
    path, sha, chosen = _released_fixture(tmp_path, monkeypatch)
    source = ReleasedCalSource(path, expected_release_sha256=sha)
    (chosen / "DONE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RealSourceRefusal, match="DONE"):
        source.outcome_rows("cal")


def test_released_source_rejects_changed_selected_label_bytes(tmp_path,
                                                               monkeypatch):
    path, sha, chosen = _released_fixture(tmp_path, monkeypatch)
    source = ReleasedCalSource(path, expected_release_sha256=sha)
    (chosen / "outcomes.parquet").write_bytes(b"changed selected payload")
    with pytest.raises(RealSourceRefusal, match="differs from pin"):
        source.outcome_rows("cal")


def test_released_truth_stays_in_run_and_excludes_reused_slot(tmp_path,
                                                               monkeypatch):
    path, sha, _ = _released_fixture(tmp_path, monkeypatch)
    source = ReleasedCalSource(path, expected_release_sha256=sha)
    rows = source.rollout_rows("cal")
    assert len(rows) == 1
    assert rows[0].features["t0_slots"] == [0]
    assert rows[0].true_pos.shape == (4, 1)
    with pytest.raises(RealSourceRefusal, match="0.5 s grid"):
        source.rollout_rows("cal", horizons_s=(10.1,))
    pos, item, run = source.selected[0]
    source.selected = ((pos, item, dict(run, n_ticks=240)),)
    with pytest.raises(RealSourceRefusal, match="run boundary"):
        source.rollout_rows("cal")
