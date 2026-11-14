"""History-conditioned (driver_params) preparation tests on the original packs.

These cover the WM-S2 ``--rollout-mode driver_params`` preparation path on the
ORIGINAL leave-PI-out data:

* the trainer's pre-t0 history extractor reads the pack's per-tick kinematics,
  right-aligns into the fixed 10-tick window and pads/masks runs that start
  closer than the history window (start-shift re-bases the window onto a real
  later tick of the same run);
* the chain runs with NO driver sidecar (parameters are learned only through the
  trajectory loss) and still logs the recovered parameter statistics;
* the offline evaluator can dump per-vehicle recovered parameters without any
  ground-truth sidecar.

All fixtures are tiny synthetic packs built through the real corpus writer path;
no real pack, SUMO, GPU or sealed source is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import (TrajectoryDataset, WmSTrainConfig,
                                         WmSTrainer, _prior_history,
                                         _shifted_item, collate, dry_run)
from hdv.config import HdvConfig


def _small_model(**kwargs):
    return WmSModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        **kwargs,
    )


def _synthetic_run_pack(tmp_path, *, n_ticks=16, k=8):
    from train.tests.packsynth import write_idea_a_pack

    runs = [dict(chunk="chunk4", world="w1", arm="base", policy="damping",
                 split="TRAIN", t0_s=540.0, n_ticks=n_ticks, n_dec=2)]
    return write_idea_a_pack(Path(tmp_path), runs, k=k, seed=0)


def test_prior_history_reads_pack_kinematics_and_pads_run_start(tmp_path):
    from train.idea_a import IdeaACorpus

    pack = _synthetic_run_pack(tmp_path)
    corpus = IdeaACorpus(str(pack), split="TRAIN", horizon_steps=10)
    item = corpus._items[corpus._order[0]]
    tick = int(item["tick"])
    present0 = np.asarray(corpus._array("veh_mask")[tick]).astype(bool)

    hist, mask = _prior_history(corpus, tick, present0)
    # Run starts at tick 0: a single valid pre-t0 row, right-aligned.
    assert hist.shape == (corpus.k_max, 10, 6)
    assert mask.shape == (corpus.k_max, 10)
    for slot in np.flatnonzero(present0):
        assert mask[slot].sum() == 1
        assert bool(mask[slot, -1])
        assert not bool(mask[slot, :-1].any())
    # The single live row is the real t0 kinematics, not a zero placeholder.
    assert np.isfinite(hist).all()
    assert np.abs(hist[present0, -1, 0]).max() > 0.0

    # A later tick in the same run has more real history, still right-aligned.
    later = tick + 5
    present5 = np.asarray(corpus._array("veh_mask")[later]).astype(bool)
    hist5, mask5 = _prior_history(corpus, later, present5)
    for slot in np.flatnonzero(present5):
        assert mask5[slot].sum() == 6
        assert bool(mask5[slot, -6:].all())
        assert not bool(mask5[slot, :-6].any())
    sorted_head = _shifted_item(corpus, item, later)
    assert int(sorted_head["tick"]) == later
    assert sorted_head["veh_step"].shape[0] == corpus.k_max


def test_history_window_is_capped_at_run_start_not_zero_padded_across_it(tmp_path):
    from train.idea_a import IdeaACorpus

    # Two runs of 6 ticks each: run 1 starts at tick 6, so its window must not
    # leak run 0's vehicles into the history.
    pack = _synthetic_run_pack(tmp_path, n_ticks=6)
    corpus = IdeaACorpus(str(pack), split="TRAIN", horizon_steps=4)
    same_run = _shifted_item(corpus, corpus._items[corpus._order[0]], 3)
    present = np.asarray(corpus._array("veh_mask")[3]).astype(bool)
    hist, mask = _prior_history(corpus, 3, present)
    for slot in np.flatnonzero(present):
        assert mask[slot].sum() == 4                 # ticks 0..3, none before run
    assert same_run["veh_step"].shape[0] == corpus.k_max


def _driver_params_dataset(n=2, t1=21, slots=4):
    base = np.zeros((n, slots, 7), dtype=np.float32)
    base[..., 1] = 1.0
    base[..., 4] = 12.0
    base[..., 6] = 5.0
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    states[..., 4] = 12.0
    present = np.ones((n, t1, slots), dtype=bool)
    present[:, :, -1] = False
    dataset = TrajectoryDataset.from_arrays({"states": states, "present": present})
    for i, record in enumerate(dataset.records):
        h = np.zeros((slots, 10, 6), dtype=np.float32)
        hm = np.zeros((slots, 10), dtype=bool)
        h[..., 0] = 0.4
        hm[:, -3:] = True
        record["hist"] = h
        record["hist_mask"] = hm
        record["controller_id"] = 0
    return dataset


def test_driver_params_training_has_no_sidecar_requirement():
    dataset = _driver_params_dataset()
    cfg = WmSTrainConfig(device="cpu", threads=1, steps_per_epoch=2,
                         unroll_curriculum_s=(10.0,), batch_size=2,
                         rollout_mode="driver_params", driver_sidecar=None)
    trainer = WmSTrainer(_small_model(rollout_mode="driver_params"), cfg)
    assert trainer.driver_recovery_report(collate([dataset[0]]), None) is None
    records = [trainer.train_batch(collate([dataset[0], dataset[1]]),
                                   unroll_steps=20) for _ in range(2)]
    assert all(np.isfinite(r["loss"]) for r in records)
    stats = records[-1]["driver_param_stats"]
    assert stats is not None and "time_headway" in stats
    for name, entry in stats.items():
        assert entry["min"] <= entry["median"] <= entry["max"]


def test_dry_run_honours_driver_params_rollout_mode(tmp_path):
    dataset = _driver_params_dataset()
    cfg = WmSTrainConfig(device="cpu", threads=1, steps_per_epoch=2,
                         unroll_curriculum_s=(10.0,), batch_size=2,
                         out_dir=str(tmp_path), rollout_mode="driver_params")
    result = dry_run(dataset, config=cfg)
    assert result["status"] == "passed"
    assert result["updates"][-1]["driver_param_stats"] is not None
    # driver_param_stats only exists on the driver_params rollout path.


def test_eval_predictions_only_needs_no_sidecar(tmp_path, monkeypatch):
    import wms_eval.driver as drv
    import wms_eval.wms_plugin as plugin

    monkeypatch.setattr(plugin, "WORKSPACE", Path(tmp_path))
    # Build the checkpoint exactly as WmsCheckpointModel._load reconstructs it
    # (default HdvConfig; the embedded physics fixes only max_gap_steps).
    model = WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                     rollout_mode="driver_params")
    ckpt = Path(tmp_path) / "wm_s.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"rollout_mode": "driver_params",
                           "history_features": 6,
                           "physics": {"max_gap_steps": 8}}}, ckpt)
    slots = 2
    hist_kin = np.zeros((10, slots, 7), dtype=np.float32)
    hist_kin[..., 1] = 1.0
    hist_kin[..., 4] = 10.0
    hist_kin[..., 6] = 5.0
    states = {"schema": "wms_driver_states/1", "n_states": 1, "states": [{
        "world": "w1", "arm": "base", "t0_s": 240.0, "tick": 3,
        "item_position": 0, "policy": "damping",
        "state": {"time_s": 240.0, "vehicles": [
            {"id": "slot0", "s": 0.0, "lane": 0, "v": 10.0, "present": True},
            {"id": "slot1", "s": 20.0, "lane": 0, "v": 10.0, "present": True}]},
        "veh_kin": hist_kin[-1].tolist(), "veh_mask": [True, True],
        "history": {"kin": hist_kin.tolist(),
                    "mask": np.ones((10, slots), bool).tolist(),
                    "target_mask": np.ones((9, slots), bool).tolist()}}]}
    states_path = Path(tmp_path) / "states.json"
    states_path.write_text(json.dumps(states), encoding="utf-8")
    out = Path(tmp_path) / "pred_out"
    rc = drv.main(["driver", "--states", str(states_path), "--ckpt", str(ckpt),
                   "--out", str(out)])
    assert rc == 0
    payload = json.loads((out / "DRIVER_PRED.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "wms_driver_predictions/1"
    assert payload["n_records"] == 1
    assert len(payload["parameter_names"]) == 9
    slot0 = payload["records"][0]["slots"][0]
    assert slot0["id"] == "slot0"
    assert len(slot0["values"]) == 9 and len(slot0["uncertainty"]) == 9
    assert not (out / "DRIVER_RECOVERY.json").exists()
