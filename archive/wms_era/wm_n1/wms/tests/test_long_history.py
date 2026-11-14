"""Long-horizon fix #5: bounded synthetic CPU-only history contracts."""
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from output.wd01.wm_n1.wms.cli import _parser, _build_model
from output.wd01.wm_n1.wms.history import history_spec
from output.wd01.wm_n1.wms.driver_encoder import PARAMETER_NAMES
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import (
    TrajectoryDataset, WmSTrainConfig, WmSTrainer, _prior_history, _state_from_kin, collate,
)
from hdv.config import HdvConfig


def motion(t=141, k=3):
    kin = np.zeros((t, k, 7), np.float32)
    kin[..., 0] = np.arange(t)[:, None] * 5 + np.arange(k)[None] * 30
    kin[..., 2] = 1
    kin[..., 4] = 10 + np.arange(k)[None]
    kin[..., 6] = 5
    return kin, np.ones((t, k), bool), np.ones((t - 1, k), bool)


def corpus(t=141):
    kin, mask, target = motion(t)
    arrays = {"veh_kin": kin, "veh_mask": mask,
              "target_mask": np.concatenate((target, np.zeros((1, 3), bool)))}
    return SimpleNamespace(_runs=[(0, t)], _array=lambda key: arrays[key]), arrays


def small_model(**kwargs):
    return WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16), **kwargs)


@pytest.mark.parametrize("duration,stride,want", [(30, 1, 31), (60, 1, 61), (60, 2, 31)])
def test_shapes_sampling_and_observed_relative_features(duration, stride, want):
    c, a = corpus()
    hist, mask = _prior_history(c, 130, a["veh_mask"][130],
                                history_s=duration, history_stride_s=stride)
    assert hist.shape == (3, want, 9)
    assert mask.shape == (3, want) and mask.all()
    assert hist[0, -1, 5] == pytest.approx(.25)  # leader bumper gap /100
    assert hist[0, -1, 7] == pytest.approx(1 / 30)
    assert hist[0, -1, 8] == 1
    assert hist[-1, -1, 8] == 0
    assert hist[0, 0, 6] == pytest.approx(-duration * 10 / 250)


def test_early_t0_entries_slot_reuse_and_missing_observation_masks():
    c, a = corpus()
    a["target_mask"][1, 1] = False  # reuse BETWEEN downsampled ticks
    a["veh_mask"][2, 2] = False
    a["veh_kin"][2, 2] = np.nan
    hist, mask = _prior_history(c, 4, np.ones(3, bool), history_s=30)
    assert mask[0].sum() == 3
    assert mask[1].sum() == 2
    assert mask[2].sum() == 2
    assert not mask[:, :-3].any()
    assert np.isfinite(hist).all()
    assert np.all(hist[~mask] == 0)
    # A second run must never borrow the first run's history.
    c._runs = [(0, 3), (3, 141)]
    _, second_mask = _prior_history(c, 4, np.ones(3, bool), history_s=30)
    assert second_mask.sum() == 3


def test_encoder_is_invariant_to_any_post_t0_values():
    torch.manual_seed(7)
    model = small_model(history_s=30, rollout_mode="hybrid").eval()
    c, a = corpus()
    tick = 70
    h, m = _prior_history(c, tick, a["veh_mask"][tick], history_s=30)
    state = _state_from_kin(torch.tensor(a["veh_kin"][tick:tick + 1]),
                             torch.ones(1, 3, dtype=torch.bool))
    first = model(state, driver_history=torch.tensor(h[None]),
                   driver_history_mask=torch.tensor(m[None]))
    for values in a.values():
        values[tick + 1:] = 0 if values.dtype == bool else np.nan
    hh, mm = _prior_history(c, tick, a["veh_mask"][tick], history_s=30)
    second = model(state, driver_history=torch.tensor(hh[None]),
                    driver_history_mask=torch.tensor(mm[None]))
    assert np.array_equal(h, hh) and np.array_equal(m, mm)
    assert torch.equal(first["driver_embedding"], second["driver_embedding"])
    assert torch.equal(first["driver_parameters"], second["driver_parameters"])


def test_masked_nan_values_do_not_change_embedding_and_gru_sees_order():
    torch.manual_seed(3)
    model = small_model(history_s=30).eval()
    c, a = corpus()
    h, m = _prior_history(c, 10, a["veh_mask"][10], history_s=30)
    h, m = torch.tensor(h[None]), torch.tensor(m[None])
    state = _state_from_kin(torch.tensor(a["veh_kin"][10:11]),
                             torch.ones(1, 3, dtype=torch.bool))
    def embedding(hist):
        return model(state, driver_history=hist, driver_history_mask=m)["driver_embedding"]
    clean = embedding(h)
    dirty = h.clone()
    dirty[~m] = float("nan")
    assert torch.equal(clean, embedding(dirty))
    reversed_live = h.clone()
    reversed_live[:, :, -6:] = h[:, :, -6:].flip(-2)
    assert not torch.equal(clean, embedding(reversed_live))


def test_flags_off_preserves_weights_rng_forward_backward_and_records():
    kin, present, _ = motion(5)
    state = _state_from_kin(torch.tensor(kin[:1]), torch.tensor(present[:1]))
    torch.manual_seed(22)
    legacy = small_model(rollout_mode="hybrid", lane_head="history")
    rng = torch.random.get_rng_state()
    torch.manual_seed(22)
    off = small_model(rollout_mode="hybrid", lane_head="history",
                       history_s=None, history_stride_s=2)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not any("history_gru" in key for key in off.state_dict())
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, off.state_dict()[key])
    results = []
    for model in (legacy, off):
        out = model.rollout(state, n_steps=2)
        loss = out["positions"].sum() + out["speeds"].sum() + out["lane_logits"].sum()
        loss.backward()
        results.append(out)
    for key in ("positions", "speeds", "lane_logits", "history", "driver_embedding"):
        assert torch.equal(results[0][key], results[1][key])
    for left, right in zip(legacy.parameters(), off.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad)
    c, a = corpus()
    old = _prior_history(c, 90, a["veh_mask"][90])
    disabled = _prior_history(c, 90, a["veh_mask"][90], history_s=None,
                               history_stride_s=2)
    for x, y in zip(old, disabled):
        assert np.array_equal(x, y)
    arrays = {"states": kin}
    old_ds = TrajectoryDataset.from_arrays(arrays)
    off_ds = TrajectoryDataset.from_arrays(arrays, history_s=None, history_stride_s=2)
    assert old_ds[0].keys() == off_ds[0].keys()
    for key in old_ds[0]:
        assert np.array_equal(old_ds[0][key], off_ds[0][key])
    cfg = WmSTrainConfig(history_s=None, history_stride_s=2).to_dict()
    assert "history_s" not in cfg and "history_stride_s" not in cfg


@pytest.mark.parametrize("mode,lane", [("residual", "off"), ("driver_params", "off"),
                                       ("hybrid", "history")])
def test_tiny_synthetic_forward_backward_and_chunked_driver_identity(mode, lane):
    kin, mask, target = motion(70)
    tick = 64
    dataset = TrajectoryDataset.from_arrays(
        {"states": kin[tick:tick + 5], "history_kin": kin[:tick + 1],
         "history_mask": mask[:tick + 1], "history_target_mask": target[:tick]},
        history_s=30)
    batch = collate([dataset[0]])
    model = small_model(history_s=30, rollout_mode=mode, lane_head=lane)
    # The existing conditioning adapter is zero-initialised. A tiny synthetic
    # nonzero adapter exercises gradient flow into history without an update.
    torch.nn.init.normal_(model.residual_net.cond_proj[-1].weight, std=.01)
    trainer = WmSTrainer(model, WmSTrainConfig(history_s=30, rollout_mode=mode,
                                               lane_head=lane, threads=1,
                                               truncation_steps=2))
    result = trainer._forward_backward(batch, unroll_steps=4)
    assert np.isfinite(result["loss"])
    grad = model.driver_encoder.history_gru.weight_ih.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    state = _state_from_kin(batch["states"][:, 0], batch["present"][:, 0])
    dh, dm = batch["driver_hist"], batch["driver_hist_mask"]
    first = model.rollout(state, n_steps=2, driver_history=dh, driver_history_mask=dm)
    # Forward with an already fixed embedding needs no new history or future data.
    out = model(first["state"], driver_embedding=first["driver_embedding"],
                driver_params=({name: first["driver_parameters"][..., i]
                                for i, name in enumerate(PARAMETER_NAMES)}
                               if mode != "residual" else None))
    assert torch.equal(out["driver_embedding"], first["driver_embedding"])


def test_array_window_without_prior_is_padded_not_future_filled():
    kin, _, _ = motion(8)
    first = TrajectoryDataset.from_arrays({"states": kin}, history_s=60)[0]
    changed = kin.copy()
    changed[1:] = 999
    second = TrajectoryDataset.from_arrays({"states": changed}, history_s=60)[0]
    assert first["driver_hist"].shape == (3, 61, 9)
    assert first["driver_hist_mask"].sum() == 3
    assert np.array_equal(first["driver_hist"], second["driver_hist"])


@pytest.mark.parametrize("duration,stride", [(0, 1), (-1, 1), (float("nan"), 1),
                                            (30, 0), (30, .75), (30, float("inf"))])
def test_invalid_flags_refused(duration, stride):
    with pytest.raises(ValueError):
        history_spec(duration, stride)


def test_cli_model_config_and_default_flags():
    args = _parser().parse_args(["train", "--pack", "synthetic.npz"])
    assert args.history_s is None and args.history_stride_s == 1
    cfg = WmSTrainConfig(history_s=60, history_stride_s=2)
    model = _build_model(args, PhysicsConfig(), cfg)
    assert model.history_s == 60 and model.driver_history_len == 31
    assert WmSTrainConfig.from_dict(cfg.to_dict()).history_stride_s == 2
    args.blackbox_traj = True
    with pytest.raises(ValueError, match="structured"):
        _build_model(args, PhysicsConfig(), cfg)


@pytest.mark.parametrize("mode", ["residual", "driver_params", "hybrid"])
def test_flags_off_matches_actual_pre_change_source(mode, monkeypatch):
    """Compare to frozen bf298cf source, not two calls to the new implementation."""
    root = Path(__file__).resolve().parents[5]
    package = "output.wd01.wm_n1.wms"
    modules = {}
    for name in ("model", "train"):
        source = subprocess.check_output(
            ["git", "show", f"bf298cf:output/wd01/wm_n1/wms/{name}.py"],
            cwd=root, text=True, encoding="utf-8")
        module = types.ModuleType(f"{package}._r3_base_{name}")
        module.__file__ = str(root / "output" / "wd01" / "wm_n1" / "wms" / f"{name}.py")
        module.__package__ = package
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        modules[name] = module
    kwargs = dict(physics=PhysicsConfig(max_gap_steps=8),
                  hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
                  rollout_mode=mode)
    torch.manual_seed(97)
    before = modules["model"].WmSModel(**kwargs)
    old_rng = torch.random.get_rng_state()
    torch.manual_seed(97)
    after = WmSModel(**kwargs)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    for key, weight in before.state_dict().items():
        assert torch.equal(weight, after.state_dict()[key])
    cfg = dict(rollout_mode=mode, threads=1, truncation_steps=2)
    old_trainer = modules["train"].WmSTrainer(before, modules["train"].WmSTrainConfig(**cfg))
    new_trainer = WmSTrainer(after, WmSTrainConfig(**cfg))
    assert old_trainer.config.to_dict() == new_trainer.config.to_dict()
    kin, _, _ = motion(5)
    batch = collate([TrajectoryDataset.from_arrays({"states": kin})[0]])
    old_result = old_trainer._forward_backward(batch, unroll_steps=4)
    new_result = new_trainer._forward_backward(batch, unroll_steps=4)
    assert old_result == new_result
    for a, b in zip(before.parameters(), after.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
    c, arrays = corpus()
    expected = modules["train"]._prior_history(c, 90, arrays["veh_mask"][90])
    actual = _prior_history(c, 90, arrays["veh_mask"][90])
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual))


def test_lazy_original_and_switch_windows_share_long_history_contract(tmp_path):
    from output.wd01.wm_n1.wms.tests import mini_switch_pack as mini
    from output.wd01.wm_n1.wms.mixdata import build_switch_datasets

    built = mini.build(tmp_path / "pack", original=True, k=8, ticks=(4, 240))
    plain = TrajectoryDataset.from_real_packs(
        [built["pack"]], identity_dirs=[built["identities"]], horizon_steps=4,
        identity_env_dir="")
    long = TrajectoryDataset.from_real_packs(
        [built["pack"]], identity_dirs=[built["identities"]], horizon_steps=4,
        identity_env_dir="", history_s=60, history_stride_s=2)
    assert "driver_history" not in plain.receipt
    assert long.receipt["driver_history"]["history_s"] == 60
    assert long[0]["driver_hist"].shape == (8, 31, 9)
    assert long[0]["driver_hist_mask"].sum(-1).max() == 2
    assert long[1]["driver_hist_mask"].sum(-1).max() == 31
    for key in plain[0]:
        if key not in ("meta", "action_timeline"):
            assert np.array_equal(plain[0][key], long[0][key])
    switch = mini.build(tmp_path / "switch", k=8, ticks=(4, 240))
    mixed = build_switch_datasets(
        pack=switch["pack"], identities=switch["identities"],
        window_list=switch["windows"], index_path=switch["index"],
        heldout_path=switch["heldout"], horizon_steps=4, entries_v2=False,
        val_holdout_frac=0, history_s=60, history_stride_s=2)
    batch = collate([mixed["train"][0], mixed["train"][1]])
    assert batch["driver_hist"].shape == (2, 8, 31, 9)
    assert batch["hist"].shape == (2, 8, 10, 6)
