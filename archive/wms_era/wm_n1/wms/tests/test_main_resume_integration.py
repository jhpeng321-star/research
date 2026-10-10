"""Synthetic CPU regressions for hybrid identity, entry resume and action decoding."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from output.wd01.wm_n1.hdv.config import HdvConfig
from output.wd01.wm_n1.wms.action_timeline import CONTROLLERS, SCHEMA, load_timeline
from output.wd01.wm_n1.wms.driver_encoder import PARAMETER_NAMES
from output.wd01.wm_n1.wms.entries import EntranceHistory
from output.wd01.wm_n1.wms.model import CONTROLLER_VOCAB, PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import (
    TrajectoryDataset,
    WmSTrainConfig,
    WmSTrainer,
    _state_from_kin,
    collate,
)

MODES = [("residual", "off"), ("driver_params", "off"), ("hybrid", "off"),
         ("driver_params", "history"), ("hybrid", "history")]


def _model(mode="residual", lane_head="off"):
    torch.manual_seed(9)
    model = WmSModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        rollout_mode=mode, lane_head=lane_head)
    if mode == "hybrid":
        # A zero-initialized adapter would hide changes to its conditioning.
        with torch.no_grad():
            model.residual_net.cond_proj[-1].weight.uniform_(-0.2, 0.2)
    return model.eval()


def _arrays(steps=6, cav=False):
    states = np.zeros((1, steps + 1, 2, 7), dtype=np.float32)
    states[..., 0] = np.array([-2000.0, -1970.0])[None, None, :] \
        + np.arange(steps + 1)[None, :, None] * 6.0
    states[:, :, 0, 1] = 1.0
    states[:, :, 1, 2] = 1.0
    states[..., 4] = 12.0
    states[..., 6] = 5.0
    arrays = {"states": states}
    if cav:
        is_cav = np.zeros(states.shape[:3], dtype=bool)
        is_cav[..., 0] = True
        arrays["is_cav"] = is_cav
    return arrays


def _batch(cav=False):
    record = TrajectoryDataset.from_arrays(_arrays(cav=cav))[0]
    rng = np.random.default_rng(8)
    record["hist"] = rng.normal(0, 0.2, (2, 10, 6)).astype(np.float32)
    record["hist_mask"] = np.ones((2, 10), dtype=bool)
    return collate([record])


def test_lazy_dataset_switch_sampling_keeps_only_membership(tmp_path, monkeypatch):
    from output.wd01.wm_n1.wms import train

    arrays = _arrays(steps=2)
    records = []
    for labels in ([0, 2], [0, 0]):
        row = TrajectoryDataset.from_arrays(arrays)[0]
        row["action_timeline"] = {
            "controller": np.broadcast_to(np.asarray(labels)[:, None], (2, 2)).copy(),
            "acceleration_mps2": np.zeros((2, 2), np.float32),
            "target_speed_mps": np.zeros((2, 2), np.float32), "control_dt_s": 1.0}
        records.append(row)
    loaded = []

    def load(index):
        loaded.append(index)
        return records[index]

    lazy = TrajectoryDataset([{}, {}], loader=load)
    weights = []
    original = train.WeightedRandomSampler

    def sampler(values, **kwargs):
        weights.extend(values)
        return original(values, **kwargs)

    monkeypatch.setattr(train, "WeightedRandomSampler", sampler)
    trainer = WmSTrainer(_model(), WmSTrainConfig(
        out_dir=str(tmp_path / "lazy_switch"), batch_size=1, steps_per_epoch=1,
        epochs_per_stage=1, unroll_curriculum_s=(1.0,), switch_window_fraction=0.75))
    trainer.fit(lazy)
    assert loaded[:2] == [0, 1]
    assert weights == [0.75, 0.25]


@pytest.mark.parametrize("lane_head", ["off", "history"])
def test_hybrid_embedding_fixed_across_steps_and_explicit_resume(lane_head):
    model = _model("hybrid", lane_head)
    batch = _batch()
    state = _state_from_kin(batch["states"][:, 0], batch["present"][:, 0])
    encodings, conditioning = [], []
    encoder_hook = model.driver_encoder.register_forward_hook(
        lambda _module, _args, out: encodings.append(out["embedding"].detach().clone()))
    residual_hook = model.residual_net.register_forward_pre_hook(
        lambda _module, _args, kwargs: conditioning.append(
            kwargs["conditioning"].detach().clone()), with_kwargs=True)
    kwargs = {"driver_history": batch["hist"], "driver_history_mask": batch["hist_mask"]}
    try:
        with torch.no_grad():
            whole = model.rollout(state, n_steps=6, **kwargs)
            assert len(encodings) == 1
            assert all(torch.equal(value, encodings[0]) for value in conditioning)
            encodings.clear()
            conditioning.clear()
            first = model.rollout(state, n_steps=2, **kwargs)
            params = {name: first["driver_parameters"][..., i]
                      for i, name in enumerate(PARAMETER_NAMES)}
            second = model.rollout(
                first["state"], n_steps=4, k0=2,
                history=first["history"], history_mask=first["history_mask"],
                driver_params=params, driver_embedding=first["driver_embedding"])
        assert len(encodings) == 1
        assert len(conditioning) == 6
        assert all(torch.equal(value, encodings[0]) for value in conditioning)
        for key in ("positions", "speeds", "lanes", "accel", "lane_logits"):
            torch.testing.assert_close(
                torch.cat((first[key], second[key]), 1), whole[key], rtol=0, atol=0)
    finally:
        encoder_hook.remove()
        residual_hook.remove()


@pytest.mark.parametrize("lane_head", ["off", "history"])
def test_trainer_hybrid_identity_survives_backward_chunks(lane_head):
    model = _model("hybrid", lane_head)
    trainer = WmSTrainer(model, WmSTrainConfig(
        threads=1, rollout_mode="hybrid", lane_head=lane_head, truncation_steps=2))
    encodings, conditioning, live_graph = [], [], []
    encoder_hook = model.driver_encoder.register_forward_hook(
        lambda _module, _args, out: encodings.append(out["embedding"].detach().clone()))

    def capture(_module, _args, kwargs):
        value = kwargs["conditioning"]
        conditioning.append(value.detach().clone())
        live_graph.append(value.requires_grad)

    residual_hook = model.residual_net.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        result = trainer.train_batch(_batch(), unroll_steps=6)
        assert result["finite"] and np.isfinite(result["loss"])
        assert len(encodings) == 1 and len(conditioning) == 6
        assert all(torch.equal(value, encodings[0]) for value in conditioning)
        assert live_graph == [True, True, False, False, False, False]
        grad = model.residual_net.cond_proj[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.driver_encoder.parameters())
    finally:
        encoder_hook.remove()
        residual_hook.remove()


@pytest.mark.parametrize("mode,lane_head", MODES)
def test_entries_switch_timeline_and_chunk_resume_fastpath_parity(mode, lane_head, monkeypatch):
    batch = _batch(cav=True)
    batch["action_timeline"] = {
        "controller": torch.tensor([[[0, 0], [3, 0], [0, 0]]]),
        "acceleration_mps2": torch.tensor([[[0.0, 0.0], [-0.5, 0.0], [0.0, 0.0]]]),
        "target_speed_mps": torch.full((1, 3, 2), 30.0),
        "control_dt_s": 1.0}
    entrance = EntranceHistory.defaults()
    entrance.rate[:] = [8.0, 0.0]
    entrance.headway_cv[:] = 0.2
    entrance.cav_share[:] = 1.0
    entrance.gap_mean[:] = 0.0
    entrance.gap_std[:] = 0.0
    entrance.speed_std[:] = 0.0
    entrance.vtype_share[:] = 0.0
    entrance.vtype_share[:, 3] = 1.0  # 7.5 m truck
    batch["entry_feat"] = torch.tensor(entrance.to_features()[None], dtype=torch.float32)
    original_plan = {key: value.clone() if torch.is_tensor(value) else value
                     for key, value in batch["action_timeline"].items()}

    def run(force_materialise):
        model = _model(mode, lane_head)
        trainer = WmSTrainer(model, WmSTrainConfig(
            threads=1, rollout_mode=mode, lane_head=lane_head,
            truncation_steps=2, entries_v2=True, entry_pad=4, seed=17))
        make_entries, rollout = trainer._entry_process, model.rollout
        starts, rolls, processes = [], [], []

        def entry_process(*args):
            process = make_entries(*args)
            process._FORCE_MATERIALISE = force_materialise
            processes.append(process)
            return process

        def capture(state, **kwargs):
            starts.append({key: value.detach().clone()
                           for key, value in state.items() if torch.is_tensor(value)})
            timeline = kwargs["action_timeline"]
            for key in ("controller", "acceleration_mps2", "target_speed_mps"):
                assert timeline[key].shape == (1, 3, 6)
                assert torch.equal(timeline[key][..., :2], original_plan[key])
                assert torch.count_nonzero(timeline[key][..., 2:]) == 0
            result = rollout(state, **kwargs)
            rolls.append({key: result[key].detach().clone() for key in
                          ("positions", "speeds", "lanes", "masks", "accel", "entrant_seq")})
            return result

        monkeypatch.setattr(trainer, "_entry_process", entry_process)
        monkeypatch.setattr(model, "rollout", capture)
        record = trainer.train_batch(batch, unroll_steps=6)
        assert record["finite"] and np.isfinite(record["loss"])
        assert record["valid_targets"] == 6  # original HDV only
        assert record["entrants"] > 0 and processes[0].summary()["inserted_cav"] > 0
        assert len(starts) == 3
        newcomers = starts[1]["entrant"]
        assert newcomers.any() and not newcomers[..., :2].any()
        for current in starts[1:]:
            assert current["is_cav"][newcomers].all()
            assert torch.all(current["length_m"][newcomers] == 7.5)
            assert torch.all(current["length_m"][..., :2] == 5.0)
            assert current["is_cav"][0, :2].tolist() == [True, False]
        # Native-default CAV newcomers retain hold-speed motion after resume;
        # they must not quietly become learned HDVs at the chunk boundary.
        for roll in rolls[1:]:
            assert torch.all(roll["accel"][:, :, newcomers[0]] == 0.0)
        return record, rolls, processes[0].summary(), model.state_dict()

    fast, fast_rolls, fast_summary, fast_weights = run(False)
    reference, reference_rolls, reference_summary, reference_weights = run(True)
    assert fast_summary == reference_summary
    for key in ("loss", "position", "speed", "lane", "entrants", "driver_param_stats"):
        assert fast[key] == reference[key]
    for fast_roll, reference_roll in zip(fast_rolls, reference_rolls):
        for key in fast_roll:
            assert torch.equal(fast_roll[key], reference_roll[key])
    for key in fast_weights:
        assert torch.equal(fast_weights[key], reference_weights[key])
    for key, value in original_plan.items():
        if torch.is_tensor(value):
            assert torch.equal(batch["action_timeline"][key], value)


def _timeline_arrays(ids, **metadata):
    arrays = _arrays(steps=2)
    arrays["controller_timeline"] = np.broadcast_to(
        np.asarray(ids).reshape(1, -1, 1), (1, len(ids), 2)).copy()
    arrays.update(metadata)
    return arrays


@pytest.mark.parametrize("legacy_ids,expected", [([1, 1], [0, 0]), ([3, 3], [1, 1])])
@pytest.mark.parametrize("vocab", [CONTROLLER_VOCAB, CONTROLLER_VOCAB[:-1]])
def test_declared_legacy_fixed_timeline_decodes_without_direct(legacy_ids, expected, vocab):
    record = TrajectoryDataset.from_arrays(_timeline_arrays(
        legacy_ids, controller_vocab=np.asarray(vocab)))[0]
    assert record["action_timeline"]["controller"][:, 0].tolist() == expected


@pytest.mark.parametrize("metadata", [
    {"schema": np.asarray(SCHEMA)},
    {"controller_timeline_schema": np.asarray(SCHEMA)},
    {"controller_vocab": np.asarray(CONTROLLERS)},
])
def test_declared_compact_timeline_retains_damping_and_direct(metadata):
    arrays = _timeline_arrays([1, 3], **metadata)
    arrays["issued_acceleration_mps2"] = np.full((1, 2, 2), -0.5, np.float32)
    arrays["issued_target_speed_mps"] = np.full((1, 2, 2), 20.0, np.float32)
    action = TrajectoryDataset.from_arrays(arrays)[0]["action_timeline"]
    assert action["controller"][:, 0].tolist() == [1, 3]
    np.testing.assert_array_equal(action["acceleration_mps2"], arrays[
        "issued_acceleration_mps2"][0])


@pytest.mark.parametrize("ids", [[1, 1], [3, 3], [0, 1, 2, 3]])
def test_ambiguous_unmarked_numeric_timeline_is_refused(ids):
    with pytest.raises(ValueError, match="ambiguous controller_timeline"):
        TrajectoryDataset.from_arrays(_timeline_arrays(ids))


def test_unambiguous_legacy_and_common_ids_remain_backcompatible():
    legacy = _timeline_arrays([1, 6])
    legacy["issued_acceleration_mps2"] = np.zeros((1, 2, 2), np.float32)
    legacy["issued_target_speed_mps"] = np.full((1, 2, 2), 20.0, np.float32)
    got = TrajectoryDataset.from_arrays(legacy)[0]["action_timeline"]
    assert got["controller"][:, 0].tolist() == [0, 3]
    common = TrajectoryDataset.from_arrays(_timeline_arrays([0, 2]))[0]
    assert common["action_timeline"]["controller"][:, 0].tolist() == [0, 2]
    named = TrajectoryDataset.from_arrays(_timeline_arrays(["native", "damping"]))[0]
    assert named["action_timeline"]["controller"][:, 0].tolist() == [0, 1]


@pytest.mark.parametrize("ids,metadata", [([3], {"schema": SCHEMA}), ([6], {})])
@pytest.mark.parametrize("missing", [
    "issued_acceleration_mps2", "issued_target_speed_mps", "both"])
def test_direct_requires_both_command_fields(ids, metadata, missing):
    arrays = _timeline_arrays(ids, **metadata)
    if missing not in ("issued_acceleration_mps2", "both"):
        arrays["issued_acceleration_mps2"] = np.zeros((1, 1, 2), np.float32)
    if missing not in ("issued_target_speed_mps", "both"):
        arrays["issued_target_speed_mps"] = np.full((1, 1, 2), 20.0, np.float32)
    with pytest.raises(ValueError, match="direct actions require both issued command fields"):
        TrajectoryDataset.from_arrays(arrays)


@pytest.mark.parametrize("metadata,match", [
    ({"schema": SCHEMA, "controller_vocab": CONTROLLER_VOCAB}, "conflicts"),
    ({"controller_timeline_schema": "unknown/1"}, "unsupported"),
    ({"schema": "wms_action_timeline/2"}, "unsupported"),
    ({"controller_vocab": ["native", "unknown"]}, "unknown controller_vocab"),
])
def test_invalid_or_conflicting_vocabulary_is_refused(metadata, match):
    with pytest.raises(ValueError, match=match):
        TrajectoryDataset.from_arrays(_timeline_arrays([1], **metadata))


@pytest.mark.parametrize("directory", [False, True])
def test_synthetic_pack_preserves_explicit_vocabulary_metadata(tmp_path, directory):
    arrays = _timeline_arrays([3], controller_vocab=np.asarray(CONTROLLER_VOCAB))
    if directory:
        path = tmp_path / "synthetic_pack"
        path.mkdir()
        for name, array in arrays.items():
            np.save(path / f"{name}.npy", array, allow_pickle=False)
    else:
        path = tmp_path / "synthetic_pack.npz"
        np.savez(path, **arrays)
    got = TrajectoryDataset.from_pack(path)[0]["action_timeline"]
    assert got["controller"][:, 0].tolist() == [1]


def test_loaded_schedule_roundtrips_through_declared_arrays_and_collate(tmp_path):
    import json

    schedule = tmp_path / "synthetic_schedule.json"
    schedule.write_text(json.dumps({
        "schema": "wms_switch_schedule/1", "control_steps": 2,
        "vehicle_ids": ["c1", "c2"],
        "switches": [{"step": 1, "vehicle_id": "c1", "controller": "direct"}],
    }), encoding="utf-8")
    commands = tmp_path / "synthetic_commands.jsonl"
    commands.write_text(json.dumps({
        "step": 1, "vehicle_id": "c1", "issued_acceleration_mps2": -0.5,
        "issued_target_speed_mps": 20.0,
    }) + "\n", encoding="utf-8")
    timeline = load_timeline(schedule, commands)
    arrays = _arrays(steps=4, cav=True)
    arrays.update({
        "schema": np.asarray(SCHEMA), "controller_timeline": timeline.controller[None],
        "issued_acceleration_mps2": timeline.acceleration_mps2[None],
        "issued_target_speed_mps": timeline.target_speed_mps[None],
    })
    record = TrajectoryDataset.from_arrays(arrays)[0]
    batch = collate([record])
    roll = _model().rollout(
        {**_state_from_kin(batch["states"][:, 0], batch["present"][:, 0]),
         "is_cav": batch["is_cav"][:, 0]}, n_steps=4,
        cav_teacher=batch["cav_teacher"], action_timeline=batch["action_timeline"])
    assert roll["accel"][0, :, 0].tolist() == [0.0, 0.0, -0.5, -0.5]
    assert roll["speeds"][0, 2:, 0].tolist() == [11.75, 11.5]
