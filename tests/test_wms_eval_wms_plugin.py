import sys
import json
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))
from wms_eval.models import ModelUnavailable, load_model  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointError, WmsCheckpointModel  # noqa: E402


def _record():
    return {
        "world": "unit",
        "t0_s": 0.0,
        "initial_state": {
            "time_s": 0.0,
            "vehicles": [
                {"id": "h1", "s": 20.0, "lane": 1, "v": 20.0, "present": True, "is_cav": False},
                {"id": "c1", "s": 40.0, "lane": 1, "v": 21.0, "present": True, "is_cav": True},
            ],
        },
    }


def _checkpoint(out: Path, name: str = "wm_s.pt") -> Path:
    from wms.model import PhysicsConfig, WmSModel

    torch.manual_seed(0)
    model = WmSModel(physics=PhysicsConfig(), history_features=6)
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"history_features": 6}}, out / name)
    return out / name


def test_random_checkpoint_loads_and_rolls_out():
    out = ROOT / "_out" / "wms_plugin_test"
    out.mkdir(parents=True, exist_ok=True)
    model = WmsCheckpointModel(_checkpoint(out))
    frames = model.predict_states(_record(), {"arm": "native"}, 10.0, 0)
    assert len(frames) == 21
    assert {row["id"] for row in frames[0]["vehicles"]} == {"h1", "c1"}
    cav_end = [row["s"] for row in frames[-1]["vehicles"] if row["id"] == "c1"]
    assert cav_end and cav_end[0] != 40.0


def test_history_lane_head_requires_and_consumes_pre_t0_history():
    from wms.model import PhysicsConfig, WmSModel

    out = ROOT / "_out" / "wms_plugin_test"
    out.mkdir(parents=True, exist_ok=True)
    model = WmSModel(physics=PhysicsConfig(), rollout_mode="driver_params",
                     lane_head="history")
    ckpt = out / "lane_history.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"history_features": 6, "rollout_mode": "driver_params",
                           "lane_head": "history"}}, ckpt)
    plugin = WmsCheckpointModel(ckpt)
    with pytest.raises(WmsCheckpointError, match="requires pre-t0 history"):
        plugin.predict_states(_record(), {"arm": "native"}, 1.0, 0)

    rec = {"initial_state": {"time_s": 0.0, "vehicles": [
        {"id": "slot0", "s": 20.0, "lane": 1, "v": 20.0, "present": True},
        {"id": "slot1", "s": 40.0, "lane": 1, "v": 21.0, "present": True},
    ]}, "history": {"kin": [], "mask": []}}
    import numpy as np
    history = np.zeros((4, 2, 7), dtype=np.float32)
    history[..., 2] = 1.0
    history[..., 4] = 20.0
    history[..., 6] = 5.0
    rec["history"] = {"kin": history.tolist(),
                      "mask": np.ones((4, 2), dtype=bool).tolist()}
    frames = plugin.predict_states(rec, {"arm": "native"}, 1.0, 0)
    assert len(frames) == 3


def test_hybrid_checkpoint_builds_pre_t0_history_and_rolls_out():
    import numpy as np
    from wms.model import PhysicsConfig, WmSModel

    out = ROOT / "_out" / "wms_plugin_test"
    out.mkdir(parents=True, exist_ok=True)
    model = WmSModel(physics=PhysicsConfig(), rollout_mode="hybrid")
    ckpt = out / "hybrid.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"history_features": 6, "rollout_mode": "hybrid"}}, ckpt)
    plugin = WmsCheckpointModel(ckpt)
    # Without pre-t0 history the history-conditioned encoder cannot be built.
    with pytest.raises(WmsCheckpointError, match="requires pre-t0 history"):
        plugin.predict_states(_record(), {"arm": "native"}, 1.0, 0)

    history = np.zeros((4, 2, 7), dtype=np.float32)
    history[..., 2] = 1.0
    history[..., 4] = 20.0
    history[..., 6] = 5.0
    rec = {"initial_state": {"time_s": 0.0, "vehicles": [
        {"id": "slot0", "s": 20.0, "lane": 1, "v": 20.0, "present": True},
        {"id": "slot1", "s": 40.0, "lane": 1, "v": 21.0, "present": True},
    ]}, "history": {"kin": history.tolist(),
                    "mask": np.ones((4, 2), dtype=bool).tolist()}}
    frames = plugin.predict_states(rec, {"arm": "native"}, 1.0, 0)
    assert len(frames) == 3


def test_unknown_checkpoint_format_is_refused():
    out = ROOT / "_out" / "wms_plugin_test"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "other/0"}, out / "bad.pt")
    with pytest.raises(WmsCheckpointError):
        WmsCheckpointModel(out / "bad.pt").predict_states(
            _record(), {"arm": "native"}, 5.0, 0)


def test_wms_spec_requires_a_checkpoint():
    with pytest.raises(ModelUnavailable):
        load_model("wms")


def _entry_record():
    import numpy as np

    counter = np.zeros((2, 30), dtype=np.float32)
    counter[0, :12] = 6.0                       # mainline: 0.6 veh/s
    counter[1, :12] = 2.0                       # ramp: 0.2 veh/s
    rec = _record()
    rec["counter"] = counter.tolist()
    return rec


def test_entries_flag_defaults_off_and_overrides_wms_and_pure_physics():
    _, off = load_model("wms", checkpoint=str(
        _checkpoint(ROOT / "_out" / "wms_plugin_test")))
    assert off.open_entries is False
    _, on = load_model("wms", checkpoint=str(
        _checkpoint(ROOT / "_out" / "wms_plugin_test")), entries=True)
    assert on.open_entries is True
    _, phys = load_model("pure_physics", entries=True)
    assert phys.open_entries is True and phys.residual_enabled is False


def test_entries_rollout_inserts_counter_driven_vehicles():
    from wms_eval.models import RolloutFlags

    out = ROOT / "_out" / "wms_plugin_test"
    ckpt = _checkpoint(out)
    off = WmsCheckpointModel(ckpt, RolloutFlags())
    frames_off = off.predict_states(_entry_record(), {"arm": "native"}, 30.0, 0)
    ids_off = {row["id"] for row in frames_off[-1]["vehicles"]}
    assert not any(vid.startswith("__entry_") for vid in ids_off)

    on = WmsCheckpointModel(ckpt, RolloutFlags(open_entries=True))
    frames_on = on.predict_states(_entry_record(), {"arm": "native"}, 30.0, 0)
    ids_on = {row["id"] for row in frames_on[-1]["vehicles"]}
    assert any(vid.startswith("__entry_") for vid in ids_on)
    assert {"h1", "c1"} <= ids_on
    assert on.entries_summary["inserted"] > 0


def test_receipt_physics_override_is_replayed():
    out = ROOT / "_out" / "wms_plugin_test"
    out.mkdir(parents=True, exist_ok=True)
    model = WmsCheckpointModel(
        _checkpoint(out),
        physics={"sumo_fidelity": True, "speed_limit_mps": 30.0, "unknown_key": 1},
    )
    model._load()
    assert model.model.physics.sumo_fidelity is True
    assert model.model.physics.speed_limit_mps == 30.0
    assert not hasattr(model.model.physics, "unknown_key")


def test_embedded_config_physics_is_used_when_no_override():
    from wms.model import PhysicsConfig, WmSModel

    out = ROOT / "_out" / "wms_plugin_test"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    built = WmSModel(physics=PhysicsConfig(), history_features=6)
    torch.save({"format": "wm_s/1", "model": built.state_dict(),
                "config": {"history_features": 6, "physics": {"sumo_idm": False}}},
               out / "cfg_physics.pt")
    model = WmsCheckpointModel(out / "cfg_physics.pt")._load()
    assert model.physics.sumo_idm is False


def test_entries_v2_flag_defaults_off_and_implies_open_entries():
    _, off = load_model("wms", checkpoint=str(
        _checkpoint(ROOT / "_out" / "wms_plugin_test")))
    assert off.open_entries is False and off.entries_v2 is False
    _, on = load_model("wms", checkpoint=str(
        _checkpoint(ROOT / "_out" / "wms_plugin_test")), entries_v2=True)
    assert on.open_entries is True and on.entries_v2 is True
    _, phys = load_model("pure_physics", entries_v2=True)
    assert phys.entries_v2 is True and phys.residual_enabled is False


def test_entries_v2_rollout_inserts_history_driven_vehicles():
    from wms_eval.models import RolloutFlags

    out = ROOT / "_out" / "wms_plugin_test"
    ckpt = _checkpoint(out)
    model = WmsCheckpointModel(ckpt, RolloutFlags(open_entries=True, entries_v2=True))
    frames = model.predict_states(_entry_record(), {"arm": "native"}, 30.0, 0)
    ids = {row["id"] for row in frames[-1]["vehicles"]}
    assert any(vid.startswith("__entry_") for vid in ids)
    assert model.entries_summary["version"] == 2
    assert model.entries_summary["inserted"] > 0


def test_section_ids_matches_wms_eval_section_names():
    import numpy as np
    from wms.entries import section_ids
    from wms_eval.metrics import _section

    s = np.array([-100.0, -1e-9, 0.0, 37.7, 75.44, 75.441, 200.0])
    names = ("upstream", "merge", "downstream")
    ids = section_ids(s)
    assert ids.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert [names[int(i)] for i in ids] == [_section(float(x)) for x in s]


def test_closed_loop_fixed_native_cav_matches_physics_only_path():
    out = ROOT / "_out" / "wms_plugin_test"
    ckpt = _checkpoint(out, "closed_loop_native.pt")
    closed = WmsCheckpointModel(ckpt)
    got = closed.predict_states(_record(), {"arm": "native", "cav_mode": "closed_loop"},
                                2.0, 0)
    model = closed.model
    _, _, kin, is_cav = closed._inputs(_record(), "native")
    initial = {
        "s_m": torch.as_tensor(kin[None, :, 0]),
        "v_mps": torch.as_tensor(kin[None, :, 4]),
        "a_mps2": torch.zeros((1, len(kin))),
        "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
        "length_m": torch.as_tensor(kin[None, :, 6]),
        "present": torch.ones((1, len(kin)), dtype=torch.bool),
        "is_cav": torch.as_tensor(is_cav[None]),
    }

    def physics_only_cav(state):
        accel = model.physics_accel(state)[0]
        speed = (state["v_mps"][0] + accel * model.physics.dt_s).clamp_min(0.0)
        position = state["s_m"][0] + 0.5 * (state["v_mps"][0] + speed) * model.physics.dt_s
        return torch.stack((position, state["lane"][0].float(), speed, accel),
                           dim=-1).unsqueeze(0)

    with torch.no_grad():
        expected_roll = model.rollout(initial, n_steps=4,
                                      cav_controller=physics_only_cav)
    ids = closed._inputs(_record(), "native")[1]
    expected = [{"time_s": 0.0, "vehicles": [
        {"id": vid, "s": float(kin[i, 0]),
         "lane": int(kin[i, 1:4].argmax(-1)), "v": float(kin[i, 4]),
         "present": True}
        for i, vid in enumerate(ids)]}]
    for step in range(4):
        expected.append({"time_s": (step + 1) * 0.5, "vehicles": [
            {"id": vid, "s": float(expected_roll["positions"][0, step, i]),
             "lane": int(expected_roll["lanes"][0, step, i]),
             "v": float(expected_roll["speeds"][0, step, i]), "present": True}
            for i, vid in enumerate(ids)]})
    for actual_frame, expected_frame in zip(got, expected):
        actual = {row["id"]: row for row in actual_frame["vehicles"]}["c1"]
        baseline = {row["id"]: row for row in expected_frame["vehicles"]}["c1"]
        assert actual["s"] == pytest.approx(baseline["s"], abs=1e-6)
        assert actual["v"] == pytest.approx(baseline["v"], abs=1e-6)


def test_closed_loop_damping_executes_controller_from_predicted_state():
    out = ROOT / "_out" / "wms_plugin_test"
    plugin = WmsCheckpointModel(_checkpoint(out, "closed_loop_damping.pt"))
    state = _record()
    state["initial_state"]["vehicles"][0]["s"] = 60.0
    state["initial_state"]["vehicles"][1]["s"] = 40.0
    frames = plugin.predict_states(state,
        {"arm": "damping", "cav_mode": "closed_loop"}, 2.0, 0)
    assert len(frames) == 5
    closed_v = {row["id"]: row for row in frames[-1]["vehicles"]}["c1"]["v"]
    physics = WmsCheckpointModel(_checkpoint(out, "closed_loop_damping.pt"))
    state["initial_state"]["vehicles"][0]["is_cav"] = False
    physics_frames = physics.predict_states(state, {"arm": "native"}, 2.0, 0)
    physics_v = {row["id"]: row for row in physics_frames[-1]["vehicles"]}["c1"]["v"]
    assert closed_v >= 0.0
    assert closed_v != pytest.approx(physics_v, abs=1e-3)


def test_wms_eval_cli_closed_loop_smoke():
    from wms_eval.cli import main

    out_root = ROOT / "_out" / "wms_plugin_test"
    ckpt = _checkpoint(out_root, "closed_loop_cli.pt")
    record = _record()
    frame = {"time_s": 0.0, "vehicles": record["initial_state"]["vehicles"]}
    record["recorded"] = {"native": [frame], "pi": [frame]}
    nonce = uuid.uuid4().hex
    states = out_root / f"cli_state_{nonce}.json"
    output = out_root / f"cli_output_{nonce}"
    states.write_text(json.dumps([record]), encoding="utf-8")
    try:
        code = main(["run", "--model", "wms", "--ckpt", str(ckpt),
                     "--states", str(states), "--out", str(output),
                     "--horizon", "1", "--cav-mode", "closed_loop"])
        assert code == 0
        summary = json.loads((output / "EVAL.json").read_text(encoding="utf-8"))
        assert summary["cav_mode"] == "closed_loop"
    finally:
        states.unlink(missing_ok=True)
