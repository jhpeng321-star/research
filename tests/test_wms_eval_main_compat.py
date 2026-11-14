"""Synthetic evalshift acceptance: checkpoint layout, causal history, receipt replay.

All checkpoints are generated on CPU under pytest's workspace-confined basetemp.
No saved research checkpoint, pack, dataset, controller or simulator is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms.blackbox_traj import BlackBoxTrajModel  # noqa: E402
from wms.model import HISTORY_H, PhysicsConfig, WmSModel  # noqa: E402
from wms.train import _prior_history  # noqa: E402
from wms_eval.models import RolloutFlags  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointError, WmsCheckpointModel  # noqa: E402


MODERN = [("hybrid", "off"), ("driver_params", "history"),
          ("hybrid", "history")]


@pytest.fixture(autouse=True)
def _cpu_workspace_only(tmp_path, monkeypatch):
    # Refuse accidental default OS-temp use, rather than writing outside _out.
    tmp_path.resolve().relative_to((ROOT / "_out").resolve())
    monkeypatch.setenv("WMS_EVAL_THREADS", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _model(kind="structured", **kwargs):
    torch.manual_seed(117)
    if kind == "blackbox_traj":
        return BlackBoxTrajModel(action_conditioned=False, **kwargs).cpu().eval()
    return WmSModel(**kwargs).cpu().eval()


def _save(tmp_path, model, config=None, *, omit=()):
    weights = {k: v for k, v in model.state_dict().items()
               if not k.startswith(tuple(omit))}
    path = tmp_path / "synthetic_wm_s.pt"
    torch.save({"format": "wm_s/1", "model": weights,
                "config": {"history_features": 6, **(config or {})}}, path)
    return path, weights


def _record():
    # Deliberately unsorted: the CAV is a physical leader for the trailing HDV.
    rows = [{"id": "z_hdv", "s": -65.0, "lane": 1, "v": 12.0,
             "present": True, "is_cav": False},
            {"id": "a_cav", "s": -20.0, "lane": 1, "v": 11.0,
             "present": True, "is_cav": True}]
    frames = []
    for i in range(7):
        frame_rows = [dict(row, s=row["s"] + 5.5 * i) for row in rows]
        frame_rows[1]["v"] = 11.0 - 0.25 * i
        frames.append({"time_s": 120.0 + 0.5 * i, "vehicles": frame_rows})
    return {"world": "synthetic_compat", "t0_s": 120.0,
            "initial_state": {"time_s": 120.0, "vehicles": rows},
            "recorded": {"native": frames}}


def _initial(plugin, record):
    _, ids, kin, is_cav = plugin._inputs(record, "native")
    return ids, {
        "veh_kin": torch.as_tensor(kin[None]),
        "s_m": torch.as_tensor(kin[None, :, 0]),
        "v_mps": torch.as_tensor(kin[None, :, 4]),
        "a_mps2": torch.zeros(1, len(ids)),
        "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
        "length_m": torch.as_tensor(kin[None, :, 6]),
        "present": torch.ones(1, len(ids), dtype=torch.bool),
        "is_cav": torch.as_tensor(is_cav[None]),
    }


def _assert_frames_match_roll(frames, roll, ids):
    for i, frame in enumerate(frames[1:]):
        expected_ids = [vid for j, vid in enumerate(ids) if roll["masks"][0, i, j]]
        assert [row["id"] for row in frame["vehicles"]] == expected_ids
        for row in frame["vehicles"]:
            j = ids.index(row["id"])
            assert row["s"] == float(roll["positions"][0, i, j])
            assert row["v"] == float(roll["speeds"][0, i, j])
            assert row["lane"] == int(roll["lanes"][0, i, j])


@pytest.mark.parametrize("kind", ["structured", "blackbox_traj"])
def test_pre_switch_checkpoint_keeps_original_architecture_and_outputs(tmp_path, kind):
    original = _model(kind)
    path, weights = _save(tmp_path, original, omit=("driver_encoder.", "command_proj."))
    assert not any(k.startswith(("driver_encoder.", "command_proj.",
                                 "controller_embed.")) for k in weights)
    # Pin the historical 64 history + 32 gap embedding, with no 16-wide action input.
    assert tuple(weights["temporal.0.weight"].shape) == (64, 96)
    plugin = WmsCheckpointModel(path, model_kind=kind)
    loaded = plugin._load()
    assert tuple(loaded.temporal[0].weight.shape) == (64, 96)
    assert loaded.command_proj is None and loaded.controller_embed is None
    if kind == "blackbox_traj":
        assert loaded.action_conditioned is False
        with pytest.raises(NotImplementedError, match="no physics base"):
            loaded.physics_accel({})
    for key, value in weights.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)

    record = _record()
    ids, initial = _initial(plugin, record)
    # An independent pre-switch temporal equation checks a nonzero gate/history.
    history = torch.linspace(-0.5, 0.5, 120).reshape(1, 2, 10, 6)
    mask = torch.ones(1, 2, 10, dtype=torch.bool)
    mask[:, 0, :3] = False
    k = torch.full((1, 2), 3, dtype=torch.long)
    with torch.no_grad():
        flat = torch.cat(((history * mask[..., None]).reshape(1, 2, 60), mask.float()), -1)
        temporal = original.temporal(torch.cat((original.history_proj(flat),
                                                original.k_embedding(k)), -1))
        base = original.residual_net(initial["veh_kin"], initial["present"])
        gate = original.gate(k)
        if kind == "blackbox_traj":
            accel = (base["accel_mean_mps2"] + gate * temporal[..., 0]).clamp(
                original.physics.accel_min_mps2, original.physics.accel_max_mps2)
            lanes = base["lane_logits"] + gate[..., None] * temporal[..., 1:]
        else:
            physics = original.physics_accel(initial)
            accel = torch.where(~initial["is_cav"], physics + gate * (
                base["accel_delta_mps2"] + temporal[..., 0]), physics).clamp(
                    original.physics.accel_min_mps2, original.physics.accel_max_mps2)
            lanes = original.mobil_merge_logits(initial) + gate[..., None] * (
                base["lane_logit_delta"] + temporal[..., 1:])
        actual = loaded(initial, k_steps=k, history=history, history_mask=mask)
        torch.testing.assert_close(actual["accel_mps2"], accel, rtol=0, atol=0)
        torch.testing.assert_close(actual["lane_logits"], lanes, rtol=0, atol=0)
        _, _, kin, is_cav = plugin._inputs(record, "native")
        teacher, present, junction = plugin._teacher(record, 120.0, ids, kin, is_cav, 6,
                                                     "native")
        expected = original.rollout(initial, n_steps=6, cav_teacher=teacher,
                                    present_seq=present, junction_seq=junction)
    frames = plugin.predict_states(record, {"arm": "native"}, 3.0, 19)
    assert [f["time_s"] for f in frames] == [120.0 + i * 0.5 for i in range(7)]
    _assert_frames_match_roll(frames, expected, ids)
    for got, recorded in zip(frames, record["recorded"]["native"]):
        cav = next(row for row in got["vehicles"] if row["id"] == "a_cav")
        assert (cav["s"], cav["v"]) == (recorded["vehicles"][1]["s"],
                                       recorded["vehicles"][1]["v"])


@pytest.mark.parametrize("kind", ["structured", "blackbox_traj"])
def test_legacy_compatibility_does_not_allow_missing_core_weights(tmp_path, kind):
    path, _ = _save(tmp_path, _model(kind), omit=("driver_encoder.", "temporal.0.bias"))
    with pytest.raises(WmsCheckpointError, match=r"missing network weights.*temporal\.0\.bias"):
        WmsCheckpointModel(path, model_kind=kind)._load()


@pytest.mark.parametrize("mode,lane_head", MODERN)
@pytest.mark.parametrize("missing", ["all", "one"])
def test_modern_history_checkpoint_requires_every_encoder_weight(
        tmp_path, mode, lane_head, missing):
    model = _model(rollout_mode=mode, lane_head=lane_head)
    # Select an actual individual key; the encoder's internal head naming may evolve.
    omit = (("driver_encoder.",) if missing == "all" else
            (next(k for k in model.state_dict() if k.startswith("driver_encoder.")
                  and k.endswith("weight")),))
    path, _ = _save(tmp_path, model, {"rollout_mode": mode, "lane_head": lane_head},
                    omit=omit)
    with pytest.raises(WmsCheckpointError, match=r"missing network weights.*driver_encoder"):
        WmsCheckpointModel(path)._load()


def _shifted_history(tick):
    # Two packed runs; history must never import the first run or future suffix.
    kin = np.zeros((23, 3, 7), np.float32)
    for t in range(len(kin)):
        kin[t, :, 0] = np.array([-100.0, -40.0, -75.0]) + t * np.array([2.0, 3.0, 1.0])
        kin[t, :, 2] = 1.0
        kin[t, :, 4] = np.array([4.0, 6.0, 2.0]) + 0.1 * t
        kin[t, :, 5] = np.array([0.2, -0.4, 0.7])
        kin[t, :, 6] = np.array([5.0, 7.0, 4.0])
    present = np.ones((23, 3), dtype=bool)
    target = np.ones((23, 3), dtype=bool)
    # Slot reuse while still occupied, and an independently missing observation.
    target[5, 0] = False
    target[12, 2] = False
    present[6, 1] = False
    kin[:4] = 999.0
    kin[tick + 1:] = -999.0
    arrays = {"veh_kin": kin, "veh_mask": present, "target_mask": target}
    corpus = SimpleNamespace(_runs=[(0, 4), (4, 23)], _array=arrays.__getitem__)
    lo = max(4, tick - HISTORY_H + 1)
    record = {"t0_s": 120.0 + (tick - 4) * 0.5,
              "initial_state": {"time_s": 120.0 + (tick - 4) * 0.5, "vehicles": [
                  {"id": vid, "s": float(kin[tick, j, 0]), "lane": 1,
                   "v": float(kin[tick, j, 4]), "present": True, "is_cav": False}
                  for j, vid in enumerate(["z", "a", "m"])]},
              "history": {"ids": ["z", "a", "m"], "kin": kin[lo:tick + 1].tolist(),
                          "mask": present[lo:tick + 1].tolist(),
                          "target_mask": target[lo:tick].tolist()}}
    return corpus, record


@pytest.mark.parametrize("tick", [4, 7, 17])
def test_id_keyed_shifted_history_matches_trainer_in_sorted_id_order(tmp_path, tick):
    path, _ = _save(tmp_path, _model())
    plugin = WmsCheckpointModel(path)
    corpus, record = _shifted_history(tick)
    _, ids, _, _ = plugin._inputs(record, "native")
    assert ids == ["a", "m", "z"]
    got, mask = plugin._driver_history(record, ids)
    expected, expected_mask = _prior_history(corpus, tick, np.ones(3, dtype=bool))
    order = [1, 2, 0]
    torch.testing.assert_close(got[0], torch.as_tensor(expected[order]), rtol=0, atol=0)
    torch.testing.assert_close(mask[0], torch.as_tensor(expected_mask[order]), rtol=0, atol=0)
    assert tuple(got.shape) == (1, 3, HISTORY_H, 6)
    if tick == 4:
        assert mask[0].sum().item() == 3
        assert torch.count_nonzero(got[0, :, :-1]) == 0
    if tick == 7:
        assert mask[0, 2].sum().item() == 2  # occupied slot reused after tick 5
        assert not mask[0, 0, -2]           # tick 6 report missing
    if tick == 17:
        assert mask[0, 1].sum().item() == 5  # second slot reuse after tick 12
    # Serialised history columns can be permuted without changing the encoder input.
    hist = record["history"]
    perm = [2, 0, 1]
    permuted = {"ids": [hist["ids"][j] for j in perm],
                "kin": np.asarray(hist["kin"])[:, perm].tolist(),
                **{key: np.asarray(hist[key]).reshape(-1, 3)[:, perm].tolist()
                   for key in ("mask", "target_mask")}}
    again, again_mask = plugin._driver_history({**record, "history": permuted}, ids)
    torch.testing.assert_close(again, got, rtol=0, atol=0)
    torch.testing.assert_close(again_mask, mask, rtol=0, atol=0)


@pytest.mark.parametrize("mode,lane_head", MODERN)
def test_modern_load_consumes_shifted_history_and_matches_trainer_rollout(
        tmp_path, mode, lane_head):
    original = _model(rollout_mode=mode, lane_head=lane_head)
    path, _ = _save(tmp_path, original, {"rollout_mode": mode, "lane_head": lane_head})
    plugin = WmsCheckpointModel(path)
    loaded = plugin._load()
    assert loaded.rollout_mode == mode and loaded.lane_head_mode == lane_head
    with pytest.raises(WmsCheckpointError, match="requires pre-t0 history"):
        plugin.predict_states(_record(), {"arm": "native"}, 1.0, 0)
    corpus, record = _shifted_history(17)
    expected, mask = _prior_history(corpus, 17, np.ones(3, dtype=bool))
    expected = torch.as_tensor(expected[[1, 2, 0]][None])
    mask = torch.as_tensor(mask[[1, 2, 0]][None])
    ids, initial = _initial(plugin, record)
    calls = []
    hook = loaded.driver_encoder.register_forward_pre_hook(
        lambda _module, args: calls.append((args[0].clone(), args[1].clone())))
    try:
        frames = plugin.predict_states(record, {"arm": "native"}, 1.5, 0)
    finally:
        hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], expected, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], mask, rtol=0, atol=0)
    with torch.no_grad():
        roll = original.rollout(initial, n_steps=3, driver_history=expected,
                                driver_history_mask=mask)
    _assert_frames_match_roll(frames, roll, ids)


@pytest.mark.parametrize("physics_source", ["override", "embedded", "sibling"])
def test_receipt_physics_caps_and_projection_are_executed(tmp_path, physics_source):
    from wms.stability import FeasibilityProjection, ProjectionConfig

    # Block adjacent-lane escape so the gap constraint is genuinely exercised.
    physics = {"sumo_idm": False, "speed_limit_mps": 27.0, "lane_safe_gap_m": 1000.0}
    original = _model(physics=PhysicsConfig(**physics), residual_cap_mps2=0.125,
                      lane_logit_cap=0.25)
    with torch.no_grad():
        original.gate_increments.fill_(1.0)
        original.temporal[-1].weight.zero_()
        original.temporal[-1].bias.fill_(1000.0)
    config = {"projection_enabled": True, "projection_min_gap_m": 9.0,
              "residual_cap_mps2": 0.125, "lane_logit_cap": 0.25}
    if physics_source == "embedded":
        config["physics"] = physics
    if physics_source == "override":
        config["physics"] = {"sumo_idm": True, "speed_limit_mps": 13.0}
    (tmp_path / "physics.json").write_text(json.dumps(
        physics if physics_source == "sibling" else {"speed_limit_mps": 14.0}),
        encoding="utf-8")
    path, _ = _save(tmp_path, original, config, omit=("driver_encoder.",))
    plugin = WmsCheckpointModel(path, physics={**physics, "not_a_physics_key": 123}
                                if physics_source == "override" else None)
    loaded = plugin._load()
    assert loaded.physics.sumo_idm is False and loaded.physics.speed_limit_mps == 27.0
    assert not hasattr(loaded.physics, "not_a_physics_key")
    assert loaded.residual_cap_mps2 == 0.125 and loaded.lane_logit_cap == 0.25
    assert plugin.projection.config.min_gap_m == 9.0
    record = _record()
    # Begin inside the configured gap: projection must act, not merely exist.
    record["initial_state"]["vehicles"][0]["s"] = -27.0
    record.pop("recorded")
    ids, initial = _initial(plugin, record)
    with torch.no_grad():
        step = loaded(initial, k_steps=3)
        hdv = ~initial["is_cav"]
        torch.testing.assert_close(step["accel_delta_mps2"][hdv], torch.tensor([0.125]),
                                   rtol=0, atol=1e-6)
        lane_delta = step["lane_logits"] - loaded.mobil_merge_logits(initial)
        torch.testing.assert_close(lane_delta, torch.full_like(lane_delta, 0.25),
                                   rtol=0, atol=1e-6)
        expected = original.rollout(initial, n_steps=4, cav_controller=original.hold_teacher,
                                    projection=FeasibilityProjection(
                                        ProjectionConfig(min_gap_m=9.0)))
    frames = plugin.predict_states(record, {"arm": "native"}, 2.0, 0)
    _assert_frames_match_roll(frames, expected, ids)
    assert plugin.projection.stats["steps"] == 4
    assert plugin.projection.stats["s_clamped"] > 0
    for frame in frames[1:]:
        rows = {row["id"]: row for row in frame["vehicles"]}
        assert rows["a_cav"]["s"] == -20.0 + 11.0 * (frame["time_s"] - 120.0)
        assert rows["a_cav"]["v"] == 11.0
        assert rows["a_cav"]["s"] - rows["z_hdv"]["s"] - 5.0 >= 9.0 - 1e-6


def test_entries_v2_with_shifted_history_projection_and_teacher_replays(tmp_path):
    model = _model(rollout_mode="hybrid", lane_head="history")
    config = {"rollout_mode": "hybrid", "lane_head": "history",
              "projection_enabled": True, "projection_min_gap_m": 3.0,
              "residual_cap_mps2": 0.5, "lane_logit_cap": 0.5}
    path, _ = _save(tmp_path, model, config)
    _, record = _shifted_history(17)
    record["initial_state"]["vehicles"][1]["is_cav"] = True
    t0 = record["t0_s"]
    record["recorded"] = {"native": [
        {"time_s": t0 + 0.5 * i, "vehicles": [
            dict(row, s=row["s"] + row["v"] * 0.5 * i)
            for row in record["initial_state"]["vehicles"]]}
        for i in range(25)]}
    counter = np.zeros((2, 30), np.float32)
    counter[:, :12] = np.array([[6.0], [2.0]])
    record["counter"] = counter.tolist()
    flags = RolloutFlags(open_entries=True, entries_v2=True)
    first = WmsCheckpointModel(path, flags)
    first._load()
    histories = []
    hook = first.model.driver_encoder.register_forward_pre_hook(
        lambda _module, args: histories.append((args[0].clone(), args[1].clone())))
    try:
        frames = first.predict_states(record, {"arm": "native"}, 12.0, 71)
    finally:
        hook.remove()
    second = WmsCheckpointModel(path, flags)
    replay = second.predict_states(record, {"arm": "native"}, 12.0, 71)
    assert frames == replay
    assert first.entries_summary == second.entries_summary
    assert first.entries_summary["version"] == 2 and first.entries_summary["inserted"] > 0
    assert len(frames) == 25
    assert first.projection.stats["steps"] == 24
    assert len(histories) == 1
    hist, mask = histories[0]
    assert hist.shape[1] > 3  # entry pool was padded before the encoder ran
    assert torch.count_nonzero(hist[:, 3:]) == 0
    assert not mask[:, 3:].any()
    entrants = {row["id"] for frame in frames for row in frame["vehicles"]
                if row["id"].startswith("__entry_")}
    assert entrants
    for got, teacher in zip(frames, record["recorded"]["native"]):
        cav = next(row for row in got["vehicles"] if row["id"] == "a")
        recorded = next(row for row in teacher["vehicles"] if row["id"] == "a")
        assert (cav["s"], cav["v"]) == pytest.approx((recorded["s"], recorded["v"]),
                                                  abs=1e-5)
        assert all(np.isfinite(row["s"]) and np.isfinite(row["v"])
                   and row["v"] >= 0.0 and row["lane"] in (0, 1, 2)
                   for row in got["vehicles"])
