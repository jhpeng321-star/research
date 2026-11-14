"""Generated-fixture tests for the CAL-tune scorer labels (no real CAL)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evalh import real_source as RS
from evalh.real_source import RealSourceRefusal, ReleasedCalSource
from evalh.scorer.labels import (OUTCOME_COLUMNS, ScorerLabelRefusal,
                                 masked_rollout_truth, planned_reads,
                                 tune_outcome_rows)
from evalh.tests.test_real_source import _file_sha, _write_json

_TUNE = ("c1", "c3")
_CAL = ("c2",)
_N_TICKS = 40


def _lab(world, tau, *, t=300.0, t0_s=300.0, arm="native"):
    return dict(world=world, source=arm, split="CALIBRATION", t0_s=t0_s, t=t,
                tau=tau, total_delay_ref_s=12.0, collision_any=0.0,
                ttc_lt3_veh_s=1.0, hard_brake_s=0.1, n_vehicles_present=10.0,
                tau_available=100.0)


def _item(world, *, first=0, n_rows=2):
    return dict(batch="chunk2", chunk="chunk2", world=world, arm="native",
                policy="native", split="CALIBRATION", t0_s=300.0, tick=0,
                plan_row=0, outcome_rows=list(range(first, first + n_rows)))


def _run(world, *, first=0, n_rows=2):
    return dict(key=f"chunk2/{world}__native", batch="chunk2", chunk="chunk2",
                world=world, arm="native", split="CALIBRATION", t0_s=300.0,
                tick_start=0, tick_end=_N_TICKS, dec_start=0, dec_end=10,
                n_ticks=_N_TICKS, outcome_first=first, n_outcome_rows=n_rows)


def _arrays(n=_N_TICKS, k=3):
    kin = np.zeros((n, k, 7), dtype=np.float16)
    kin[..., 0] = (np.arange(n, dtype=np.float16)[:, None]
                   + np.arange(k, dtype=np.float16)[None, :])
    kin[..., 4] = (5.0 + np.arange(k, dtype=np.float16)[None, :])
    mask = np.ones((n, k), dtype=np.uint8)
    slot = np.broadcast_to(np.arange(k, dtype=np.int32), (n, k)).copy()
    target = np.ones((n, k), dtype=np.uint8)
    target[14, 2] = 0                # car 2 leaves at step 15
    target[24, 1] = 0                # car 1 leaves at step 25
    slot[26:, 1] = 3                 # its slot is reused afterwards
    return kin, mask, slot, target


def _make_pack(tmp_path):
    split_path = tmp_path / "CAL_SPLIT_A4.json"
    split_sha = _write_json(split_path, {"schema": "wm_n1_cal_split_a4/1",
                                         "cal_tune": list(_TUNE),
                                         "cal_cal": list(_CAL)})
    order = (("c1", 0, 2), ("c3", 2, 0), ("c2", 2, 2))
    manifest = {"schema": "wm_n1_pack/1",
                "items": [_item(w, first=f, n_rows=n) for w, f, n in order]}
    index = {"schema": "wm_n1_corpus_pack/2",
             "runs": [_run(w, first=f, n_rows=n) for w, f, n in order]}
    pack = tmp_path / "pack"
    manifest_sha = _write_json(pack / "manifest.json", manifest)
    index_sha = _write_json(pack / "INDEX.json", index)
    (pack / "outcomes.parquet").write_bytes(b"forbidden mixed table")
    shards = tmp_path / "shards"
    file_shas, done_shas, idea_shas = {}, {}, {}
    for world, _first, n_rows in order:
        d = shards / "chunk2" / world / "native"
        d.mkdir(parents=True)
        kin, mask, slot, target = _arrays()
        for name, arr in (("veh_kin", kin), ("veh_mask", mask),
                          ("slot_index", slot), ("target_mask", target),
                          ("cav_plan_ch", np.zeros((_N_TICKS, 3, 8),
                                                   dtype=np.float16))):
            np.save(d / f"{name}.npy", arr)
        files = {f"{n}.npy": _file_sha(d / f"{n}.npy") for n in
                 ("veh_kin", "veh_mask", "slot_index", "target_mask",
                  "cav_plan_ch")}
        outputs = {"veh_mask": files["veh_mask.npy"],
                   "slot_index": files["slot_index.npy"],
                   "target_mask": files["target_mask.npy"]}
        if n_rows:
            pd.DataFrame([_lab(world, 60.0), _lab(world, 120.0)]) \
                .to_parquet(d / "outcomes.parquet", index=False)
            files["outcomes.parquet"] = _file_sha(d / "outcomes.parquet")
            outputs["outcomes.parquet"] = files["outcomes.parquet"]
        done = dict(schema="wm_n1_features_index/1", batch="chunk2",
                    chunk="chunk2", world=world, arm="native",
                    split="CALIBRATION", t0_s=300.0, n_outcome_rows=n_rows,
                    outputs=outputs)
        done_shas[world] = _write_json(d / "DONE.json", done)
        idea = dict(schema="wm_n1_features_idea_a/1", batch="chunk2",
                    chunk="chunk2", world=world, arm="native",
                    split="CALIBRATION", t0_s=300.0,
                    arrays={"veh_kin": {"sha256": files["veh_kin.npy"]},
                            "cav_plan_ch": {"sha256":
                                            files["cav_plan_ch.npy"]}})
        idea_shas[world] = _write_json(d / "IDEA_A_DONE.json", idea)
        file_shas[world] = files
    return dict(split_path=split_path, split_sha=split_sha, pack=pack,
                manifest_sha=manifest_sha, index_sha=index_sha, shards=shards,
                file_shas=file_shas, done_shas=done_shas, idea_shas=idea_shas)


def _release(tmp_path, ctx, *, half, worlds, omit_outcomes=()):
    selected = {}
    for world in worlds:
        files = dict(ctx["file_shas"][world])
        if world in omit_outcomes:
            files.pop("outcomes.parquet", None)
        selected[f"chunk2/{world}__native"] = dict(
            done_sha256=ctx["done_shas"][world],
            idea_a_done_sha256=ctx["idea_shas"][world], files=files)
    spec = dict(schema="wm_n1_released_cal_source/1",
                allocation_id="generated-allocation",
                data_release_id="generated-release", source_head="synthetic",
                checkpoint_member_sha256="a" * 64, half=half,
                cal_split_path=str(ctx["split_path"]), pack_dir=str(ctx["pack"]),
                manifest_sha256=ctx["manifest_sha"],
                index_sha256=ctx["index_sha"], shards_root=str(ctx["shards"]),
                selected_shards=selected)
    path = tmp_path / f"RELEASE_{half}.json"
    return path, _write_json(path, spec)


def _source(tmp_path, monkeypatch, *, half="cal_tune", worlds=_TUNE,
            omit_outcomes=()):
    ctx = _make_pack(tmp_path)
    monkeypatch.setattr(RS, "LF_SHA_CAL_SPLIT_A4", ctx["split_sha"])
    path, sha = _release(tmp_path, ctx, half=half, worlds=worlds,
                         omit_outcomes=omit_outcomes)
    source = ReleasedCalSource(path, expected_release_sha256=sha)
    return ctx, source


def _payload_guard(monkeypatch, forbidden=()):
    opened = []
    orig_hash, orig_load = RS._hash_file, np.load
    orig_pq = pd.read_parquet

    def check(path):
        parts = Path(str(path)).parts
        for name in forbidden:
            if name in parts:
                raise AssertionError(f"forbidden payload opened: {path}")

    def guard_hash(path, *a, **k):
        check(path)
        opened.append(str(path))
        return orig_hash(path, *a, **k)

    def guard_load(path, *a, **k):
        check(path)
        opened.append(str(path))
        return orig_load(path, *a, **k)

    def guard_pq(path, *a, **k):
        check(path)
        opened.append(str(path))
        return orig_pq(path, *a, **k)

    monkeypatch.setattr(RS, "_hash_file", guard_hash)
    monkeypatch.setattr(np, "load", guard_load)
    monkeypatch.setattr(pd, "read_parquet", guard_pq)
    return opened


def test_tune_outcome_rows_skips_zero_label_and_counts(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    _payload_guard(monkeypatch)
    rows, census = tune_outcome_rows(source)
    assert [(r.world, r.tau_s) for r in rows] == [("c1", 60.0), ("c1", 120.0)]
    assert census.n_items == 2
    assert census.n_zero_label_skipped == 1
    assert census.skipped_keys == ["chunk2/c3__native"]
    assert census.n_rows_by_tau == {60.0: 1, 120.0: 1}


def test_zero_label_shard_payload_is_never_opened(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    opened = _payload_guard(monkeypatch, forbidden=("c3",))
    tune_outcome_rows(source)
    assert not any("c3" in Path(p).parts for p in opened)


def test_index_rows_but_file_missing_fails_closed(tmp_path, monkeypatch):
    ctx, source = _source(tmp_path, monkeypatch)
    (ctx["shards"] / "chunk2" / "c1" / "native" / "outcomes.parquet").unlink()
    with pytest.raises(RealSourceRefusal):
        tune_outcome_rows(source)


def test_index_rows_but_release_pins_no_outcomes_fails_closed(tmp_path,
                                                              monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch, omit_outcomes=("c1",))
    with pytest.raises(ScorerLabelRefusal, match="pins no outcomes"):
        tune_outcome_rows(source)


def test_masked_rollout_truth_per_horizon_survivors(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    _pos, item, _run_ = source.selected[0]
    truth = masked_rollout_truth(source, item, (5.0, 10.0, 15.0))
    assert truth.t0_slot_ids.tolist() == [0, 1, 2]
    assert truth.survivor_counts.tolist() == [3, 2, 1]
    assert truth.mask[0].tolist() == [True, True, True]
    assert truth.mask[1].tolist() == [True, True, False]
    assert truth.mask[2].tolist() == [True, False, False]
    assert truth.horizon_masked.tolist() == [False, False, False]
    assert truth.positions[0].tolist() == [10.0, 11.0, 12.0]
    assert truth.positions[2][0] == 30.0
    assert np.isnan(truth.positions[2][1]) and np.isnan(truth.positions[2][2])
    assert truth.speeds[0].tolist() == [5.0, 6.0, 7.0]


def test_masked_rollout_truth_masks_horizon_past_run_end(tmp_path,
                                                         monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    _pos, item, _run_ = source.selected[0]
    truth = masked_rollout_truth(source, item, (5.0, 20.0))
    assert truth.horizon_masked.tolist() == [False, True]
    assert truth.survivor_counts.tolist() == [3, 0]
    assert truth.mask[1].tolist() == [False, False, False]
    assert np.isnan(truth.positions[1]).all()


def test_tune_refuses_cal_cal_release_before_payload(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch, half="cal_cal", worlds=_CAL)
    opened = _payload_guard(monkeypatch)
    with pytest.raises(ScorerLabelRefusal, match="only the 'cal_tune' half"):
        tune_outcome_rows(source)
    with pytest.raises(ScorerLabelRefusal):
        masked_rollout_truth(source, source.selected[0][1], (5.0,))
    assert opened == []


def test_scoring_opens_only_selected_half_payload(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    opened = _payload_guard(monkeypatch, forbidden=("c2",))
    rows, _census = tune_outcome_rows(source)
    for _pos, item, _run_ in source.selected:
        masked_rollout_truth(source, item, (5.0, 10.0))
    assert {r.world for r in rows} == {"c1"}
    assert any("c1" in Path(p).parts for p in opened)
    assert not any("c2" in Path(p).parts for p in opened)


def test_planned_reads_lists_outcome_columns_and_rollout_arrays(tmp_path,
                                                                monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    reads = planned_reads(source)
    assert ("chunk2/c1/native/outcomes.parquet",
            tuple(OUTCOME_COLUMNS)) in reads
    assert not any(path.endswith("outcomes.parquet") and "c3" in path
                   for path, _col in reads)
    assert ("chunk2/c3/native/veh_kin.npy", "veh_kin") in reads
    assert ("chunk2/c1/native/cav_plan_ch.npy", "cav_plan_ch") in reads
    assert len(reads) == 6 + 5
    assert not any("c2" in path for path, _col in reads)


def test_planned_reads_refuses_cal_cal_release(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch, half="cal_cal", worlds=_CAL)
    with pytest.raises(ScorerLabelRefusal):
        planned_reads(source)
