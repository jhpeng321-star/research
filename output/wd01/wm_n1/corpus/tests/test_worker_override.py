"""FEAT1 v2 speed-up: owner-approved --max-workers-override + dynamic gate.

The classic hard maximum stays 6 workers; a higher ``--workers N`` is refused
unless the explicit owner-approved ``--max-workers-override N`` is passed
(recorded in ``ALLOCATION.json``).  A run above
``MEM.DYNAMIC_FALLBACK_WORKERS`` drops the current batch to 8 workers while
free virtual memory is < 8 GB or free RAM is < 6 GB (logged + receipted).
"""
import json

import numpy as np
import pytest

import synth
from corpus import _util as U
from corpus import featurize_all as FA
from corpus import memory as MEM

HEALTHY = {"free_virtual_gb": 30.0, "free_ram_gb": 16.0}
LOW_VIRTUAL = {"free_virtual_gb": 7.9, "free_ram_gb": 16.0}
LOW_RAM = {"free_virtual_gb": 30.0, "free_ram_gb": 5.9}
BOUNDARY = {"free_virtual_gb": 8.0, "free_ram_gb": 6.0}


# ------------------------------------------------- pure gate decision ----
def test_dynamic_worker_gate_pure():
    assert MEM.dynamic_worker_gate(HEALTHY, workers=12) == (12, False)
    assert MEM.dynamic_worker_gate(LOW_VIRTUAL, workers=12) == (8, True)
    assert MEM.dynamic_worker_gate(LOW_RAM, workers=12) == (8, True)
    assert MEM.dynamic_worker_gate(BOUNDARY, workers=12) == (12, False)
    assert MEM.dynamic_worker_gate(LOW_VIRTUAL, workers=8) == (8, False)
    assert MEM.dynamic_worker_gate(LOW_RAM, workers=6) == (6, False)


# ------------------------------------------------------------ fixtures ---
def _shard_record(tmp_path, store, *, world):
    k = 4
    inputs = tmp_path / f"run_{world}" / "run" / "physics_steps.jsonl"
    inputs.parent.mkdir(parents=True, exist_ok=True)
    inputs.write_text('{"time_s": 0.5}\n', encoding="utf-8")
    z = np.zeros
    shard = dict(
        meta=dict(chunk="chunk4", world=world, arm="native", split="TRAIN",
                  t0_s=540.0, n_ticks=1, n_slots=k, n_dropped_slots=0,
                  n_dropped_slot_steps=0, input_physics=str(inputs)),
        veh_step=z((1, k, 16), np.float16), veh_mask=z((1, k), np.uint8),
        slot_index=z((1, k), np.int32), plan=z((1, k, 19, 8), np.float16),
        decision_t=z((1,), np.float64), wall=z((1, 4, 11), np.float16),
        wall_mask=z((1, 4), np.uint8), counter=z((1, 2, 30), np.float16),
        accel_next=z((1, k), np.float16), target_mask=z((1, k), np.uint8),
        lane_next=z((1, k), np.int8), arrivals=z((1, 2), np.int16),
        slot_vid=[f"v{i}" for i in range(k)], outcomes=None)
    from features import writer as W

    return W.write_shard(shard, store)["record"]


def _inventory_and_records(tmp_path, out):
    data = tmp_path / "data"
    synth.write_world_plan(
        data / "output/campaign_stage1_v3/results_plan_chunk4"
             "/WORLD_PLAN.json",
        [{"world": "w1", "split": "TRAIN", "t0_s": 540.0},
         {"world": "w2", "split": "TRAIN", "t0_s": 600.0}])
    runs = [
        dict(batch="chunk4", chunk="chunk4", world="w1", arm="native",
             policy="native", split="TRAIN", t0_s=540.0,
             root_rel="runs/campaign_stage1_chunk4", run="w1__native"),
        dict(batch="chunk4", chunk="chunk4", world="w2", arm="native",
             policy="native", split="TRAIN", t0_s=600.0,
             root_rel="runs/campaign_stage1_chunk4", run="w2__native"),
    ]
    inv = synth.make_inventory(tmp_path / "inv.parquet", runs)
    store = out / "store"
    records = {U.run_key("chunk4", r["world"], "native"):
               _shard_record(tmp_path, store, world=r["world"]) for r in runs}
    return inv, data, records


def _patch_run_batch(monkeypatch, records, calls):
    def fake_run_batch(batch, workers, failures):
        calls.append((len(batch), workers))
        return [dict(key=U.run_key(j["batch"], j["world"], j["arm"]),
                     status="skipped", record=records[
                         U.run_key(j["batch"], j["world"], j["arm"])])
                for j in batch]

    monkeypatch.setattr(FA, "_run_batch", fake_run_batch)


# ------------------------------------------------- override validation ---
@pytest.mark.parametrize("workers", (7, 12))
def test_override_required_above_hard_max(tmp_path, workers):
    inv, data, _ = _inventory_and_records(tmp_path, tmp_path / "out")
    with pytest.raises(ValueError, match="max-workers-override"):
        FA.featurize_all(inv, data, tmp_path / "out", workers=workers)


def test_override_mismatch_refused(tmp_path):
    inv, data, _ = _inventory_and_records(tmp_path, tmp_path / "out")
    with pytest.raises(ValueError, match="max-workers-override"):
        FA.featurize_all(inv, data, tmp_path / "out", workers=12,
                         max_workers_override=10)


def test_override_accepted_and_recorded(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    inv, data, records = _inventory_and_records(tmp_path, out)
    calls = []
    _patch_run_batch(monkeypatch, records, calls)
    probe = lambda: dict(HEALTHY)  # noqa: E731
    rep = FA.featurize_all(inv, data, out, workers=12,
                           max_workers_override=12, probe=probe)
    assert calls == [(2, 12)]
    assert rep["written"] == 0 and rep["skipped"] == 2
    receipt = json.loads(
        (out / FA.ALLOCATION_NAME).read_text(encoding="utf-8"))
    assert receipt["max_workers_override"] == 12
    assert receipt["workers"] == 12
    assert receipt["worker_gate_events"] == []
    assert receipt["n_gate_drops"] == 0
    assert "worker-gate" not in capsys.readouterr().out


# ------------------------------------------------------- dynamic gate ----
def test_gate_drops_batch_to_8_and_logs(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    inv, data, records = _inventory_and_records(tmp_path, out)
    calls = []
    _patch_run_batch(monkeypatch, records, calls)
    probe = lambda: dict(LOW_VIRTUAL)  # noqa: E731
    # disable the 10 GB pause gate: this test targets the dynamic worker
    # gate, whose low-memory sample must not block the start gate
    FA.featurize_all(inv, data, out, workers=12, max_workers_override=12,
                     min_free_virtual_gb=0.0, min_free_ram_gb=0.0,
                     probe=probe)
    assert calls == [(2, 8)]                    # batch ran with 8 workers
    out_text = capsys.readouterr().out
    assert "worker-gate:" in out_text and "8 workers" in out_text
    receipt = json.loads(
        (out / FA.ALLOCATION_NAME).read_text(encoding="utf-8"))
    events = receipt["worker_gate_events"]
    assert len(events) == 1 and receipt["n_gate_drops"] == 1
    assert events[0]["effective_workers"] == 8
    assert events[0]["requested_workers"] == 12
    assert events[0]["free_virtual_gb"] < MEM.DYNAMIC_VIRTUAL_GB


def test_gate_drops_on_low_ram_too(tmp_path, monkeypatch):
    out = tmp_path / "out"
    inv, data, records = _inventory_and_records(tmp_path, out)
    calls = []
    _patch_run_batch(monkeypatch, records, calls)
    probe = lambda: dict(LOW_RAM)  # noqa: E731
    FA.featurize_all(inv, data, out, workers=12,
                     max_workers_override=12, min_free_virtual_gb=0.0,
                     min_free_ram_gb=0.0, probe=probe)
    assert calls == [(2, 8)]


def test_no_gate_probe_below_fallback(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    inv, data, records = _inventory_and_records(tmp_path, out)
    calls = []
    _patch_run_batch(monkeypatch, records, calls)
    probe = lambda: dict(LOW_RAM)  # noqa: E731

    FA.featurize_all(inv, data, out, workers=6, probe=probe)
    assert calls == [(2, 6)]           # classic run: batch at requested 6
    receipt = json.loads(
        (out / FA.ALLOCATION_NAME).read_text(encoding="utf-8"))
    assert receipt["worker_gate_events"] == []   # no dynamic gate at 6 workers
    assert "worker-gate" not in capsys.readouterr().out
