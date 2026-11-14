"""FEAT1 --idea-fast flag plumbing (byte-identical fast Idea A baseline).

The flag is opt-in, recorded in ALLOCATION.json, injected into every job
dict (so the spawned workers pass it to ``write_idea_a_channels``), and
forwarded by ``write_idea_a_channels`` to
``features.idea_a.idea_a_channels_from_shard``.  The receipt schema and the
written arrays are unchanged.
"""
from __future__ import annotations

import json

import numpy as np

import synth
from corpus import _util as U
from corpus import featurize_all as FA
from corpus.idea_a import write_idea_a_channels

HEALTHY = {"free_virtual_gb": 30.0, "free_ram_gb": 16.0}


def _inventory_and_records(tmp_path, out):
    data = tmp_path / "data"
    synth.write_world_plan(
        data / "output/campaign_stage1_v3/results_plan_chunk4"
        "/WORLD_PLAN.json",
        [{"world": "w1", "split": "TRAIN", "t0_s": 540.0}])
    runs = [
        dict(batch="chunk4", chunk="chunk4", world="w1", arm="native",
             policy="native", split="TRAIN", t0_s=540.0,
             root_rel="runs/campaign_stage1_chunk4", run="w1__native"),
    ]
    inv = synth.make_inventory(tmp_path / "inv.parquet", runs)
    return inv, data


def _shard_record(tmp_path, store):
    shard = _shard_for_idea(tmp_path)
    from features import writer as W

    return W.write_shard(shard, store)["record"]


def _featurize(tmp_path, monkeypatch, *, idea_fast):
    out = tmp_path / ("out_fast" if idea_fast else "out_ref")
    inv, data = _inventory_and_records(tmp_path, out)
    store = out / "store"
    records = {U.run_key("chunk4", "w1", "native"):
               _shard_record(tmp_path, store)}
    jobs = []

    def fake_run_batch(batch, workers, failures):
        jobs.extend(batch)
        return [dict(key=U.run_key(j["batch"], j["world"], j["arm"]),
                     status="skipped",
                     record=records[U.run_key(j["batch"], j["world"],
                                              j["arm"])])
                for j in batch]

    monkeypatch.setattr(FA, "_run_batch", fake_run_batch)
    probe = lambda: dict(HEALTHY)  # noqa: E731
    FA.featurize_all(inv, data, out, workers=1, probe=probe,
                     idea_fast=idea_fast)
    receipt = json.loads(
        (out / FA.ALLOCATION_NAME).read_text(encoding="utf-8"))
    return jobs, receipt


def test_idea_fast_default_false(tmp_path, monkeypatch):
    jobs, receipt = _featurize(tmp_path, monkeypatch, idea_fast=False)
    assert jobs and all(job["idea_fast"] is False for job in jobs)
    assert receipt["idea_fast"] is False


def test_idea_fast_true_reaches_jobs_and_receipt(tmp_path, monkeypatch):
    jobs, receipt = _featurize(tmp_path, monkeypatch, idea_fast=True)
    assert jobs and all(job["idea_fast"] is True for job in jobs)
    assert receipt["idea_fast"] is True


def test_cli_parser_idea_fast():
    from corpus.cli import build_parser

    a = build_parser().parse_args(
        ["featurize-all", "--inventory", "i", "--data-root", "d",
         "--out", "o"])
    assert not a.idea_fast
    a2 = build_parser().parse_args(
        ["featurize-all", "--inventory", "i", "--data-root", "d",
         "--out", "o", "--idea-fast"])
    assert a2.idea_fast


def _shard_for_idea(tmp_path):
    k = 3
    inputs = tmp_path / "run_in" / "run" / "physics_steps.jsonl"
    inputs.parent.mkdir(parents=True, exist_ok=True)
    inputs.write_text('{"time_s": 0.5}\n', encoding="utf-8")
    z = np.zeros
    return dict(
        meta=dict(chunk="chunk4", world="w1", arm="native", split="TRAIN",
                  t0_s=540.0, n_ticks=1, n_slots=k, n_dropped_slots=0,
                  n_dropped_slot_steps=0, input_physics=str(inputs)),
        veh_step=z((1, k, 16), np.float16), veh_mask=z((1, k), np.uint8),
        slot_index=z((1, k), np.int32), plan=z((1, k, 19, 8), np.float16),
        decision_t=z((1,), np.float64), wall=z((1, 4, 11), np.float16),
        wall_mask=z((1, 4), np.uint8), counter=z((1, 2, 30), np.float16),
        accel_next=z((1, k), np.float16), target_mask=z((1, k), np.uint8),
        lane_next=z((1, k), np.int8), arrivals=z((1, 2), np.int16),
        slot_vid=[f"v{i}" for i in range(k)], outcomes=None)


def test_write_idea_a_channels_forwards_fast(tmp_path, monkeypatch):
    from features import idea_a as IA

    seen = []
    real = IA.idea_a_channels_from_shard

    def spy(shard, *, params=None, fast=False):
        seen.append(fast)
        return real(shard, params=params, fast=fast)

    monkeypatch.setattr(IA, "idea_a_channels_from_shard", spy)
    shard = _shard_for_idea(tmp_path)
    rec = write_idea_a_channels(shard, tmp_path / "shard", force=True,
                                fast=True)
    assert rec["status"] == "written" and seen == [True]
    done = json.loads(
        (tmp_path / "shard" / "IDEA_A_DONE.json").read_text(
            encoding="utf-8"))
    assert done["schema"] == "wm_n1_features_idea_a/1"
