"""Synthetic CPU contracts; no real packs, checkpoints, or recorded states."""

import io
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

from wms.history import history_spec, long_history_features  # noqa: E402
from wms.model import WmSModel  # noqa: E402
from wms_eval import materialize as M  # noqa: E402
from wms_eval import wms_plugin as P  # noqa: E402
from wms_eval import driver as D  # noqa: E402


def _motion(t=40, k=2):
    kin = np.zeros((t, k, 7), np.float32)
    kin[..., 0] = np.arange(t)[:, None] * 10 + np.arange(k)[None, :] * 30
    kin[..., 2] = 1.0
    kin[..., 4] = 20.0
    kin[..., 6] = 5.0
    return kin, np.ones((t, k), bool), np.ones((max(0, t - 1), k), bool)


def _history(t=21, history_s=10.0, stride_s=1.0, limited=False, by_id=True):
    kin, mask, target = _motion(t)
    history = dict(kin=kin.tolist(), mask=mask.tolist(), target_mask=target.tolist(),
                   history_s=history_s, history_stride_s=stride_s, dt_s=0.5,
                   available_span_s=(t - 1) * 0.5, run_start_limited=limited)
    if by_id:
        history["ids"] = ["a", "b"]
    return history


def _plugin(history_s=10.0, stride_s=1.0):
    plugin = P.WmsCheckpointModel.__new__(P.WmsCheckpointModel)
    plugin._torch = torch
    plugin.config = {"history_s": history_s, "history_stride_s": stride_s}
    plugin.model = SimpleNamespace(history_s=history_s, history_stride_s=stride_s,
                                   history_len=10)
    return plugin


class _MemoryPack:
    """Only in-memory synthetic arrays, with a nonzero run boundary."""

    def __init__(self, tick=24):
        kin, mask, target = _motion()
        step = np.zeros((len(kin), 2, 16), np.float32)
        step[:, 0, 7] = 1.0
        self.arrays = dict(veh_kin=kin, veh_mask=mask, veh_step=step,
                           target_mask=np.concatenate((target, np.zeros((1, 2), bool))))
        self.run = dict(key="chunk/unit__native", chunk="chunk", world="unit",
                        arm="native", t0_s=12.0, tick_start=4, tick_end=len(kin))
        self.runs = {self.run["key"]: self.run}
        self.item = dict(tick=tick)

    def array(self, name):
        return self.arrays[name]

    def item_for(self, run):
        assert run is self.run
        return 0, self.item


class _MemoryIdentity:
    def row_for_position(self, position):
        assert position == 0
        return np.asarray([0, 1], np.int64)

    def label(self, canonical):
        return ("a", "b")[canonical]


def _build(pack=None, **kwargs):
    pack = pack or _MemoryPack()
    window = dict(key="synthetic", pack="unit", chunk="chunk", world="unit",
                  arm="native", t0_s=12.0)
    return M.build_state(window, {"unit": pack}, {"unit": _MemoryIdentity()},
                         horizon_s=1.0, min_vehicles=1, **kwargs)


def _load_payload(monkeypatch, payload, kind="structured"):
    # No file is created or read: the loader receives an in-memory state dict.
    monkeypatch.setattr(P, "_checkpoint_path", lambda value: ROOT / "synthetic.pt")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    plugin = P.WmsCheckpointModel("synthetic.pt", physics={"dt_s": 0.5},
                                   model_kind=kind)
    plugin._load()
    return plugin


def test_checkpoint_restores_long_history_without_changing_rolling_geometry(monkeypatch):
    built = WmSModel(rollout_mode="hybrid", lane_head="history", history_s=10.0,
                     history_stride_s=1.0)
    payload = dict(format="wm_s/1", model=built.state_dict(),
                   config=dict(rollout_mode="hybrid", lane_head="history",
                               history_s=10.0, history_stride_s=1.0))
    plugin = _load_payload(monkeypatch, payload)
    assert plugin.model.history_s == 10.0
    assert plugin.model.history_stride_s == 1.0
    assert plugin.model.history_len == 10
    assert plugin.model.history_features == 6
    assert plugin.model.driver_embed_dim == 102
    for key, weight in built.state_dict().items():
        assert torch.equal(plugin.model.state_dict()[key], weight)
    features, mask = plugin._driver_history({"history": _history()}, ["b", "a"])
    assert features.shape == (1, 2, history_spec(10.0)[2], 9)
    assert mask.all()


def test_legacy_checkpoint_restores_with_long_history_disabled(monkeypatch):
    built = WmSModel()
    plugin = _load_payload(monkeypatch, dict(format="wm_s/1", model=built.state_dict(),
                                            config={"history_features": 6}))
    assert plugin.model.history_s is None
    assert plugin._driver_history({}, ["slot0"]) == (None, None)


@pytest.mark.parametrize("kind", ["blackbox_id", "blackbox_traj"])
def test_long_checkpoint_rejects_blackbox_kinds(monkeypatch, kind):
    with pytest.raises(P.WmsCheckpointError, match="only for structured"):
        _load_payload(monkeypatch, dict(format="wm_s/1", model={},
                                        config={"history_s": 10.0}), kind)


def test_long_residual_checkpoint_requires_encoder_weights(monkeypatch):
    built = WmSModel(history_s=10.0)
    weights = {k: v for k, v in built.state_dict().items()
               if not k.startswith("driver_encoder.")}
    with pytest.raises(P.WmsCheckpointError, match="missing network weights"):
        _load_payload(monkeypatch, dict(format="wm_s/1", model=weights,
                                        config={"history_s": 10.0}))


def test_materializer_default_record_matches_frozen_legacy_window(monkeypatch):
    pack = _MemoryPack()

    def legacy_window(pack, run, tick, slots, **kwargs):
        lo = max(int(run.get("tick_start", 0)), tick - 10 + 1)
        return dict(kin=np.asarray(pack.array("veh_kin")[lo:tick + 1][:, slots], np.float32),
                    mask=np.asarray(pack.array("veh_mask")[lo:tick + 1][:, slots], bool),
                    target_mask=np.asarray(pack.array("target_mask")[lo:tick][:, slots], bool))

    with monkeypatch.context() as patch:
        patch.setattr(M, "_history_window", legacy_window)
        expected = _build(pack)
    actual = _build(pack)
    explicit_off = _build(pack, history_s=None, history_stride_s=2.0)
    assert json.dumps(actual, separators=(",", ":")) == json.dumps(expected, separators=(",", ":"))
    assert actual == explicit_off
    assert actual["history"]["history_steps"] == 10
    assert set(actual["history"]) == {"ids", "history_steps", "kin", "mask", "target_mask"}


def test_materializer_long_extension_retains_native_ticks_and_future():
    legacy = _build()
    long = _build(history_s=10.0, history_stride_s=2.0)
    history = long["history"]
    assert history["history_steps"] == history_spec(10.0, 2.0)[0] == 21
    assert len(history["target_mask"]) == 20
    assert history["available_span_s"] == 10.0
    assert history["run_start_limited"] is False
    assert history["history_s"] == 10.0
    assert history["history_stride_s"] == 2.0
    assert history["dt_s"] == 0.5
    assert np.array_equal(np.asarray(history["kin"])[:, 0, 0], np.arange(4, 25) * 10)
    assert {k: v for k, v in long.items() if k != "history"} == {
        k: v for k, v in legacy.items() if k != "history"}


def test_materializer_run_start_clip_and_shift_are_t0_causal():
    pack = _MemoryPack(tick=6)
    early = _build(pack, history_s=10.0)
    shifted = _build(pack, history_s=10.0, shift_steps=4)
    for record, tick in ((early, 6), (shifted, 10)):
        hist = record["history"]
        assert hist["history_steps"] == tick - pack.run["tick_start"] + 1
        assert hist["available_span_s"] == (tick - pack.run["tick_start"]) * 0.5
        assert hist["run_start_limited"] is True
        assert hist["kin"][-1][0][0] == tick * 10
        assert hist["kin"][0][0][0] == pack.run["tick_start"] * 10
        assert hist["kin"][-1][0][0] == record["initial_state"]["vehicles"][0]["s"]


@pytest.mark.parametrize("by_id", [True, False])
def test_early_state_masks_padding_and_unsampled_slot_reuse(by_id):
    history = _history(t=5, history_s=6.0, limited=True, by_id=by_id)
    # Break a transition that a 1 s sampler does not directly observe.
    history["target_mask"][1][1] = False
    ids = ["b", "a"] if by_id else ["slot1", "slot0"]
    features, mask = _plugin(history_s=6.0)._driver_history({"history": history}, ids)
    assert features.shape == (1, 2, 7, 9)
    assert mask[0, 0].tolist() == [False, False, False, False, False, True, True]
    assert mask[0, 1].tolist() == [False, False, False, False, True, True, True]
    assert torch.equal(features[..., :4, :], torch.zeros_like(features[..., :4, :]))


@pytest.mark.parametrize("by_id", [True, False])
def test_eval_matches_shared_native_history_features_after_reordering(by_id):
    history = _history(by_id=by_id)
    history["target_mask"][7][0] = False
    ids = ["b", "a"] if by_id else ["slot1", "slot0"]
    features, mask = _plugin()._driver_history({"history": history}, ids)
    kin = torch.tensor(history["kin"], dtype=torch.float32)[:, [1, 0]]
    present = torch.tensor(history["mask"], dtype=torch.bool)[:, [1, 0]]
    target = torch.tensor(history["target_mask"], dtype=torch.bool)[:, [1, 0]]
    expected, expected_mask = long_history_features(kin, present, target,
                                                   torch.ones(2, dtype=torch.bool),
                                                   history_s=10.0, dt_s=0.5)
    assert torch.equal(features[0], expected)
    assert torch.equal(mask[0], expected_mask)


def test_long_checkpoint_requires_history_object():
    with pytest.raises(P.WmsCheckpointError, match="requires pre-t0 history"):
        _plugin()._driver_history({}, ["a"])


@pytest.mark.parametrize("by_id", [True, False])
def test_legacy_ten_tick_state_is_refused(by_id):
    history = _history(t=10, by_id=by_id)
    for key in ("history_s", "history_stride_s", "dt_s", "available_span_s",
                "run_start_limited"):
        history.pop(key)
    with pytest.raises(P.WmsCheckpointError, match="duration metadata"):
        _plugin()._driver_history({"history": history}, ["a"] if by_id else ["slot0"])


@pytest.mark.parametrize("changes, message", [
    ({"dt_s": 1.0}, "dt_s=0.5"),
    ({"history_stride_s": 2.0}, "matching history_stride_s"),
    ({"available_span_s": 9.0}, "does not match native ticks"),
    ({"available_span_s": float("nan")}, "dt_s=0.5"),
    ({"history_s": 4.5}, "shorter than checkpoint"),
    ({"target_mask": []}, "all native transitions"),
])
@pytest.mark.parametrize("by_id", [True, False])
def test_long_metadata_and_continuity_fail_closed(changes, message, by_id):
    history = _history(by_id=by_id)
    history.update(changes)
    with pytest.raises(P.WmsCheckpointError, match=message):
        _plugin()._driver_history({"history": history}, ["a"] if by_id else ["slot0"])


@pytest.mark.parametrize("limited", [False, None, "true", 1])
def test_short_history_requires_explicit_boolean_run_start_receipt(limited):
    history = _history(t=10, limited=limited)
    with pytest.raises(P.WmsCheckpointError, match="explicit run_start_limited"):
        _plugin()._driver_history({"history": history}, ["a", "b"])


def test_one_tick_run_start_is_padded_and_present_at_t0():
    history = _history(t=1, limited=True)
    features, mask = _plugin()._driver_history({"history": history}, ["a", "b"])
    assert features.shape == (1, 2, history_spec(10.0)[2], 9)
    assert not mask[..., :-1].any()
    assert mask[..., -1].all()


def test_stream_threads_flags_without_filesystem_output(monkeypatch):
    captured = []

    def build(window, packs, identities, **kwargs):
        captured.append(kwargs)
        return {"key": window["key"]}

    class Buffer(io.StringIO):
        def close(self):
            pass

    buffer = Buffer()
    monkeypatch.setattr(M, "build_state", build)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: buffer)
    summary = M._write_stream(ROOT / "synthetic.json", "unit", [{"key": "unit"}], {}, {},
                               horizon_s=1.0, shift_steps=4,
                               history_s=10.0, history_stride_s=2.0)
    assert summary["n_states"] == 1
    assert json.loads(buffer.getvalue())["windows"] == [{"key": "unit"}]
    assert captured[0]["history_s"] == 10.0
    assert captured[0]["history_stride_s"] == 2.0
    assert captured[0]["shift_steps"] == 4


def test_materializer_parser_history_defaults_and_opt_in():
    args = M.parser().parse_args(["--pack", "unit=synthetic", "--identities", "unit=synthetic"])
    assert args.history_s is None
    assert args.history_stride_s == 1.0
    long = M.parser().parse_args(["--pack", "unit=synthetic", "--identities", "unit=synthetic",
                                  "--history-s", "10", "--history-stride-s", "2"])
    assert long.history_s == 10.0
    assert long.history_stride_s == 2.0


@pytest.mark.parametrize("by_id", [True, False])
def test_historical_leader_context_survives_t0_exit(by_id):
    hist = _history(by_id=by_id)
    # The leader is observable in the past, but absent from the scored t0 roster.
    hist["mask"][-1][1] = False
    features, mask = _plugin()._driver_history({"history": hist},
                                              ["a"] if by_id else ["slot0"])
    assert features.shape == (1, 1, 11, 9)
    assert features[0, 0, -2, 8] == 1
    assert features[0, 0, -2, 5] == pytest.approx(.25)
    assert features[0, 0, -1, 8] == 0
    assert mask.all()


def test_driver_tools_materialize_and_predict_matching_long_history(monkeypatch, tmp_path):
    kin, mask, target = _motion(t=30)
    manifest = {"items": [{"tick": 24, "world": "unit", "arm": "native"}]}
    index = {"runs": [{"world": "unit", "arm": "native", "tick_start": 4}]}
    monkeypatch.setattr(D, "_read_pack_arrays", lambda path: (manifest, index, kin, mask, target))
    dest = tmp_path / "driver_states.json"
    D.materialise_pack_states(tmp_path / "synthetic", dest,
                               history_s=10, history_stride_s=1)
    rec = json.loads(dest.read_text())["states"][0]
    assert len(rec["history"]["kin"]) == 21
    assert rec["history"]["run_start_limited"] is False
    model = WmSModel(history_s=10, rollout_mode="hybrid").eval()
    values, uncertainties = D.predict_records(model, [rec])[0]
    assert values.shape == uncertainties.shape == (2, 9)
    assert np.isfinite(values).all()
    rec["history"].pop("history_s")
    with pytest.raises(P.WmsCheckpointError, match="duration metadata"):
        D.predict_records(model, [rec])


def test_clean_and_shifted_builders_thread_history_flags(monkeypatch):
    from wms_eval import c1sets

    captured = []

    def stream(*args, **kwargs):
        captured.append(kwargs)
        return {"n_states": 1}

    monkeypatch.setattr(M, "_write_stream", stream)
    monkeypatch.setattr(M, "_load_windows", lambda path: [])
    monkeypatch.setattr(c1sets, "unseen_split", lambda *args: dict(
        u_same=[{}], u_new=[{}], dropped=[]))
    monkeypatch.setattr(c1sets, "seen_selection", lambda *args, **kwargs: dict(
        seen=[{}], strict=[], paired=[]))
    monkeypatch.setattr(c1sets, "scan_states", lambda path: {"n_states": 1})
    monkeypatch.setattr(M, "_windows_from_set", lambda *args, **kwargs: iter([{}]))
    args = SimpleNamespace(pi_states=ROOT / "pi.json", windows_val=ROOT / "val.json",
                           train_windows=None, horizon_s=1.0, min_vehicles=1,
                           shift_steps=4, history_s=10.0, history_stride_s=2.0,
                           unseen_u_same=ROOT / "same.json", unseen_u_new=ROOT / "new.json",
                           seen_v2=ROOT / "seen.json", seen_size=1, seed=0,
                           out_dir=ROOT, from_set=[f"unit={ROOT / 'unit.json'}"], limit=None)
    M._write_clean_sets(args, {}, {}, {})
    M._write_shifted_sets(args, {}, {}, {})
    assert len(captured) == 4
    assert all(call["history_s"] == 10.0 and call["history_stride_s"] == 2.0
               and call["shift_steps"] == 4 for call in captured)


def test_standard_materializer_sets_thread_history_flags(monkeypatch, capsys):
    captured = []

    def stream(*args, **kwargs):
        captured.append(kwargs)
        return {"n_states": 1}

    monkeypatch.setattr(M, "Pack", lambda *args: _MemoryPack())
    monkeypatch.setattr(M, "Identity", lambda *args: _MemoryIdentity())
    monkeypatch.setattr(M, "_load_windows", lambda path: [{"policy": "pi"}])
    monkeypatch.setattr(M, "_write_stream", stream)
    args = M.parser().parse_args([
        "--pack", "unit=synthetic", "--identities", "unit=synthetic",
        "--pi-states", str(ROOT / "pi.json"), "--windows-val", str(ROOT / "val.json"),
        "--unseen-pi", str(ROOT / "fleet.json"),
        "--unseen-mixed", str(ROOT / "mixed.json"), "--seen-val", str(ROOT / "seen.json"),
        "--history-s", "10", "--history-stride-s", "2",
    ])
    assert M.run(args) == 0
    assert len(captured) == 3
    assert all(call["history_s"] == 10.0 and call["history_stride_s"] == 2.0
               for call in captured)
    assert len(json.loads(capsys.readouterr().out)) == 3
