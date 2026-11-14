"""E10: owner rule-(b) outcome-supervision exclusions in the CAL-tune scorer.

Generated fixtures only (no CAL data, no real checkpoint).  The tests prove
that ``base`` / ``native_base`` items are skipped on metadata before any
payload open, counted in the census, excluded from ``planned_reads``, and
recorded by the CLI in RESULTS/DONE, while the remaining native/pi items still
score.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evalh import real_source as RS
from evalh.protocol import N_QUANTILES
from evalh.real_source import RealSourceRefusal, ReleasedCalSource
from evalh.scorer.cli import run_score
from evalh.scorer.labels import (OUTCOME_COLUMNS, planned_reads,
                                 selected_items, tune_outcome_rows)
from evalh.tests.test_real_source import _file_sha, _write_json

_TUNE = ("c1",)
_CAL = ("c2",)
_N_TICKS = 40
_ORDER = (("native", "native"), ("pi", "pi"), ("base", "damping"),
          ("native_base", "native"))
_EXCLUDED = ("base", "native_base")
_EXPECTED_KEYS = ["chunk2/c1__base", "chunk2/c1__native_base"]
_SPEC_SHA = "a" * 64


def _label(world, arm, tau, t, t0_s=300.0):
    return dict(world=world, source=arm, split="CALIBRATION", t0_s=t0_s, t=t,
                tau=tau, total_delay_ref_s=12.0, collision_any=0.0,
                ttc_lt3_veh_s=1.0, hard_brake_s=0.1, n_vehicles_present=10.0,
                tau_available=100.0)


def _arrays(n=_N_TICKS, k=3):
    kin = np.zeros((n, k, 7), dtype=np.float16)
    kin[..., 0] = (np.arange(n, dtype=np.float16)[:, None]
                   + np.arange(k, dtype=np.float16)[None, :])
    kin[..., 4] = (5.0 + np.arange(k, dtype=np.float16)[None, :])
    mask = np.ones((n, k), dtype=np.uint8)
    slot = np.broadcast_to(np.arange(k, dtype=np.int32), (n, k)).copy()
    target = np.ones((n, k), dtype=np.uint8)
    target[14, 2] = 0
    target[24, 1] = 0
    slot[26:, 1] = 3
    return kin, mask, slot, target


def _make_ctx(tmp_path):
    split_path = tmp_path / "CAL_SPLIT_A4.json"
    split_sha = _write_json(split_path, {"schema": "wm_n1_cal_split_a4/1",
                                         "cal_tune": list(_TUNE),
                                         "cal_cal": list(_CAL)})
    shards = tmp_path / "shards"
    items, runs, files_by_key, done_shas, idea_shas = [], [], {}, {}, {}
    cursor = 0
    for world in (_TUNE + _CAL):
        arms = _ORDER if world in _TUNE else (("native", "native"),)
        for arm, policy in arms:
            key = f"chunk2/{world}__{arm}"
            first = cursor
            items.append(dict(batch="chunk2", chunk="chunk2", world=world,
                              arm=arm, policy=policy, split="CALIBRATION",
                              t0_s=300.0, tick=0, plan_row=0,
                              outcome_rows=[first, first + 1]))
            runs.append(dict(key=key, batch="chunk2", chunk="chunk2",
                             world=world, arm=arm, split="CALIBRATION",
                             t0_s=300.0, tick_start=0, tick_end=_N_TICKS,
                             dec_start=0, dec_end=10, n_ticks=_N_TICKS,
                             outcome_first=first, n_outcome_rows=2))
            d = shards / "chunk2" / world / arm
            d.mkdir(parents=True)
            kin, mask, slot, target = _arrays()
            for name, arr in (("veh_kin", kin), ("veh_mask", mask),
                              ("slot_index", slot), ("target_mask", target),
                              ("cav_plan_ch", np.zeros((_N_TICKS, 3, 8),
                                                       dtype=np.float16))):
                np.save(d / f"{name}.npy", arr)
            # Excluded arms carry t != t0 label rows: if ever opened, the
            # real source aborts the pass ("decision time differs").
            label_t = 330.0 if arm in _EXCLUDED else 300.0
            pd.DataFrame([_label(world, arm, 60.0, label_t),
                          _label(world, arm, 120.0, label_t)]) \
                .to_parquet(d / "outcomes.parquet", index=False)
            files = {f"{n}.npy": _file_sha(d / f"{n}.npy") for n in
                     ("veh_kin", "veh_mask", "slot_index", "target_mask",
                      "cav_plan_ch")}
            files["outcomes.parquet"] = _file_sha(d / "outcomes.parquet")
            done = dict(schema="wm_n1_features_index/1", batch="chunk2",
                        chunk="chunk2", world=world, arm=arm,
                        split="CALIBRATION", t0_s=300.0, n_outcome_rows=2,
                        outputs={"outcomes.parquet":
                                 files["outcomes.parquet"],
                                 "veh_mask": files["veh_mask.npy"],
                                 "slot_index": files["slot_index.npy"],
                                 "target_mask": files["target_mask.npy"]})
            done_shas[key] = _write_json(d / "DONE.json", done)
            idea = dict(schema="wm_n1_features_idea_a/1", batch="chunk2",
                        chunk="chunk2", world=world, arm=arm,
                        split="CALIBRATION", t0_s=300.0,
                        arrays={"veh_kin": {"sha256": files["veh_kin.npy"]},
                                "cav_plan_ch": {"sha256":
                                                files["cav_plan_ch.npy"]}})
            idea_shas[key] = _write_json(d / "IDEA_A_DONE.json", idea)
            files_by_key[key] = files
            cursor += 2
    pack = tmp_path / "pack"
    manifest_sha = _write_json(pack / "manifest.json",
                               {"schema": "wm_n1_pack/1", "items": items})
    index_sha = _write_json(pack / "INDEX.json",
                            {"schema": "wm_n1_corpus_pack/2", "runs": runs})
    (pack / "outcomes.parquet").write_bytes(b"forbidden mixed table")
    selected_keys = [f"chunk2/{w}__{arm}" for w in _TUNE
                     for arm, _policy in _ORDER]
    selected = {key: dict(done_sha256=done_shas[key],
                          idea_a_done_sha256=idea_shas[key],
                          files=files_by_key[key]) for key in selected_keys}
    spec = dict(schema="wm_n1_released_cal_source/1",
                allocation_id="generated-allocation",
                data_release_id="generated-release", source_head="synthetic",
                checkpoint_member_sha256=_SPEC_SHA, half="cal_tune",
                cal_split_path=str(split_path), pack_dir=str(pack),
                manifest_sha256=manifest_sha, index_sha256=index_sha,
                shards_root=str(shards), selected_shards=selected)
    spec_path = tmp_path / "RELEASE.json"
    spec_sha = _write_json(spec_path, spec)
    return dict(split_sha=split_sha, spec_path=spec_path, spec_sha=spec_sha,
                shards=shards)


def _source(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    monkeypatch.setattr(RS, "LF_SHA_CAL_SPLIT_A4", ctx["split_sha"])
    source = ReleasedCalSource(ctx["spec_path"],
                               expected_release_sha256=ctx["spec_sha"])
    return ctx, source


def _payload_guard(monkeypatch, forbidden=()):
    opened = []
    orig_hash, orig_load = RS._hash_file, np.load
    orig_pq = pd.read_parquet

    def check(path):
        parts = Path(str(path)).parts
        if any(name in parts for name in forbidden):
            raise AssertionError(f"forbidden payload opened: {path}")

    def guard(fn):
        def inner(path, *a, **k):
            check(path)
            opened.append(str(path))
            return fn(path, *a, **k)
        return inner

    monkeypatch.setattr(RS, "_hash_file", guard(orig_hash))
    monkeypatch.setattr(np, "load", guard(orig_load))
    monkeypatch.setattr(pd, "read_parquet", guard(orig_pq))
    return opened


class _FakeBound:
    def __init__(self, sha256):
        self.bundle = object()
        self.sha256 = sha256


class _FakeA:
    name = "A:teacher"
    summary_source = "training_packed_plan"

    def predict_quantiles(self, rows):
        return np.stack([np.full(N_QUANTILES, 12.0) for _row in rows])


def test_base_arms_skipped_counted_without_payload(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    opened = _payload_guard(monkeypatch, forbidden=_EXCLUDED)
    rows, census = tune_outcome_rows(source)
    assert census.n_items == 4
    assert census.n_excluded_b == 2
    assert census.excluded_b_keys == _EXPECTED_KEYS
    assert census.n_zero_label_skipped == 0
    assert {r.policy for r in rows} == {"native", "pi"}
    assert {r.world for r in rows} == {"c1"}
    assert not any(name in Path(p).parts for p in opened
                   for name in _EXCLUDED)


def test_planned_reads_excludes_base_arms(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    reads = planned_reads(source)
    assert ("chunk2/c1/native/outcomes.parquet",
            tuple(OUTCOME_COLUMNS)) in reads
    assert ("chunk2/c1/pi/veh_kin.npy", "veh_kin") in reads
    assert not any(name in path for path, _col in reads
                   for name in _EXCLUDED)


def test_selected_items_reports_excluded_keys(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    kept, excluded_keys = selected_items(source)
    assert [entry[1]["arm"] for entry in kept] == ["native", "pi"]
    assert list(excluded_keys) == _EXPECTED_KEYS


def test_excluded_label_rows_would_abort_if_read(tmp_path, monkeypatch):
    _ctx, source = _source(tmp_path, monkeypatch)
    pos, item, run = next(entry for entry in source.selected
                          if entry[1]["arm"] == "base")
    with pytest.raises(RealSourceRefusal, match="decision time differs"):
        source.outcome_rows_for(item, run, pos, columns=OUTCOME_COLUMNS)


def test_cli_records_n_excluded_b(tmp_path, monkeypatch):
    ctx, _source_ = _source(tmp_path, monkeypatch)
    member = tmp_path / "member_1.pt"
    member.write_bytes(b"exclusion-b-test-member")
    out = tmp_path / "out"
    args = SimpleNamespace(
        command="score", release_spec=str(ctx["spec_path"]),
        release_sha=ctx["spec_sha"], checkpoint=str(member), out=str(out),
        comparators="zero", climatology=None, climatology_sha256=None,
        cav_motion="teacher", v2g_delay_option=None,
        dynamics_extrapolation=False, params_sha256=None,
        v2g_engine_sha256=None)
    done = run_score(
        args,
        source_factory=lambda spec_path,
        expected_release_sha256: ReleasedCalSource(
            spec_path, expected_release_sha256=expected_release_sha256),
        checkpoint_loader=lambda path, **kw: _FakeBound(_SPEC_SHA),
        a_factory=lambda bundle: _FakeA(),
        git_head="b" * 40)
    results = json.loads((out / "RESULTS.json").read_text("utf-8"))
    assert results["census"]["n_excluded_b"] == 2
    assert results["census"]["excluded_b_keys"] == _EXPECTED_KEYS
    assert done["n_excluded_b"] == 2
    assert done["census"]["n_excluded_b"] == 2
    assert done["n_rows"] == 4
    cells = results["metrics"]["cells"]
    assert "pi-native" in cells
    assert "damping-native" not in cells
    assert (out / "DONE.json").is_file() and (out / "rows.csv").is_file()
