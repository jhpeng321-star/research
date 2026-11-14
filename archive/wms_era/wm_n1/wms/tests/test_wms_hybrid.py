"""Hybrid rollout tests: per-driver physics + retained gated residual.

``--rollout-mode hybrid`` keeps the history-inferred IDM/MOBIL parameters AND
the gated learned residual (unlike ``driver_params``, which zeroes the
residual), and additionally conditions the residual net on the driver
embedding.  These tests pin the parity contract, the gradient paths, the mode
plumbing and the optional residual-magnitude penalty/share log.
"""
from __future__ import annotations

import numpy as np
import torch

from output.wd01.wm_n1.hdv.config import HdvConfig
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import (TrajectoryDataset, WmSTrainConfig,
                                         WmSTrainer, collate, dry_run)


def _small_model(**kwargs):
    return WmSModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        **kwargs,
    )


def _state(batch=1, slots=2):
    kin = torch.zeros(batch, slots, 7)
    kin[..., 1] = 1.0
    kin[..., 4] = 12.0
    kin[..., 6] = 5.0
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6],
            "lane": kin[..., 1:4].argmax(-1),
            "present": torch.ones(batch, slots, dtype=torch.bool),
            "is_cav": torch.zeros(batch, slots, dtype=torch.bool)}


def _empty_history(batch=1, slots=2):
    return (torch.zeros(batch, slots, 10, 6),
            torch.zeros(batch, slots, 10, dtype=torch.bool))


def test_hybrid_parity_with_default_encoder_equals_residual_mode():
    # An empty pre-t0 history makes the encoder return exactly its vType
    # defaults, and the zero-initialised conditioning adapter is a no-op, so
    # hybrid reproduces residual mode bit-exactly on a follower-free scenario.
    torch.manual_seed(0)
    residual = _small_model(rollout_mode="residual")
    hybrid = _small_model(rollout_mode="hybrid")
    hybrid.load_state_dict(residual.state_dict(), strict=False)
    state = _state()
    state["lane"] = torch.tensor([[0, 2]])       # non-adjacent: no followers
    state["s_m"] = torch.tensor([[0.0, 20.0]])
    state["v_mps"] = torch.tensor([[12.0, 11.0]])
    state["veh_kin"][..., 0] = state["s_m"]
    state["veh_kin"][..., 4] = state["v_mps"]
    hist, hmask = _empty_history()
    base = residual.rollout(state, n_steps=5)
    rolled = hybrid.rollout(state, n_steps=5, driver_history=hist,
                            driver_history_mask=hmask)
    for key in ("positions", "speeds", "lanes", "lane_logits"):
        torch.testing.assert_close(base[key], rolled[key], rtol=0, atol=0)
    # The learned residual is retained (driver_params mode would zero it).
    assert float(rolled["residual_magnitude"].detach()) > 0.0
    # Hybrid surfaces the extra diagnostics; residual mode does not.
    assert "residual_magnitude" in rolled and "residual_magnitude" not in base


def test_hybrid_rollout_shapes_and_finite_gradients_to_both_paths():
    torch.manual_seed(3)
    model = _small_model(rollout_mode="hybrid")
    state = _state()
    hist = torch.randn(1, 2, 10, 6)
    hmask = torch.ones(1, 2, 10, dtype=torch.bool)
    out = model(state, driver_history=hist, driver_history_mask=hmask)
    assert out["accel_mps2"].shape == (1, 2)
    assert out["lane_logits"].shape == (1, 2, 3)
    assert out["driver_embedding"].shape == (1, 2, model.driver_embed_dim)
    assert "driver_parameters" in out
    # A rollout reaches k>0, where the gate g(k) opens and the residual net
    # actually contributes (g(0) == 0 is intentional).
    rolled = model.rollout(state, n_steps=2, driver_history=hist,
                           driver_history_mask=hmask)
    assert rolled["positions"].shape == (1, 2, 2)
    loss = (rolled["positions"].sum() + rolled["speeds"].sum()
            + rolled["lane_logits"].sum())
    loss.backward()
    enc_grads = [p.grad for p in model.driver_encoder.parameters()
                 if p.grad is not None]
    assert enc_grads and all(bool(torch.isfinite(g).all()) for g in enc_grads)
    assert any(float(g.abs().sum()) > 0 for g in enc_grads)
    res_grads = [p.grad for p in model.residual_net.parameters()
                 if p.grad is not None]
    assert res_grads and all(bool(torch.isfinite(g).all()) for g in res_grads)
    assert any(float(g.abs().sum()) > 0 for g in res_grads)
    # Conditioning adapter receives a gradient (its last layer is zero-init).
    cond = model.residual_net.cond_proj[-1].weight.grad
    assert cond is not None and bool(torch.isfinite(cond).all())


def test_driver_params_mode_still_zeroes_the_residual():
    model = _small_model(rollout_mode="driver_params")
    state = _state()
    hist = torch.randn(1, 2, 10, 6)
    hmask = torch.ones(1, 2, 10, dtype=torch.bool)
    out = model(state, driver_history=hist, driver_history_mask=hmask)
    assert torch.equal(out["accel_delta_mps2"],
                       torch.zeros_like(out["accel_delta_mps2"]))
    assert "residual_magnitude" not in model.rollout(
        state, n_steps=2, driver_history=hist, driver_history_mask=hmask)


def test_hybrid_lane_head_history_rollout_shapes():
    model = _small_model(rollout_mode="hybrid", lane_head="history")
    state = _state()
    hist = torch.randn(1, 2, 10, 6)
    hmask = torch.ones(1, 2, 10, dtype=torch.bool)
    out = model.rollout(state, n_steps=2, driver_history=hist,
                        driver_history_mask=hmask)
    assert out["lane_logits"].shape == (1, 2, 2, 3)
    assert out["lane_probabilities"].shape == (1, 2, 2, 3)
    torch.testing.assert_close(out["lane_probabilities"].sum(-1),
                               torch.ones(1, 2, 2))


def test_config_accepts_hybrid_and_lane_head_and_validates_weight():
    cfg = WmSTrainConfig(rollout_mode="hybrid")
    assert cfg.rollout_mode == "hybrid" and cfg.residual_magnitude_weight is None
    assert WmSTrainConfig(rollout_mode="hybrid", lane_head="history").lane_head == "history"
    WmSTrainConfig(rollout_mode="hybrid", residual_magnitude_weight=0.0)
    try:
        WmSTrainConfig(rollout_mode="hybrid", residual_magnitude_weight=-1.0)
    except ValueError as exc:
        assert "residual_magnitude_weight" in str(exc)
    else:
        raise AssertionError("negative residual magnitude weight must be refused")


def test_cli_plumbs_hybrid_mode_and_weight():
    from output.wd01.wm_n1.wms.cli import _parser

    args = _parser().parse_args([
        "train", "--pack", "p", "--identities", "i",
        "--rollout-mode", "hybrid", "--lane-head", "history",
        "--residual-magnitude-weight", "0.002"])
    assert args.rollout_mode == "hybrid" and args.lane_head == "history"
    assert args.residual_magnitude_weight == 0.002


def _hybrid_dataset(n=2, t1=21, slots=4):
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
    for record in dataset.records:
        h = np.zeros((slots, 10, 6), dtype=np.float32)
        hm = np.zeros((slots, 10), dtype=bool)
        h[..., 0] = 0.4
        hm[:, -3:] = True
        record["hist"] = h
        record["hist_mask"] = hm
        record["controller_id"] = 0
    return dataset


def test_hybrid_train_and_dry_run_log_residual_share_and_penalty():
    dataset = _hybrid_dataset()
    for lane_head in ("off", "history"):
        cfg = WmSTrainConfig(device="cpu", threads=1, steps_per_epoch=2,
                             unroll_curriculum_s=(10.0,), batch_size=2,
                             rollout_mode="hybrid", lane_head=lane_head)
        trainer = WmSTrainer(_small_model(rollout_mode="hybrid",
                                          lane_head=lane_head), cfg)
        record = trainer._forward_backward(
            collate([dataset[0], dataset[1]]), unroll_steps=20)
        assert np.isfinite(record["loss"])
        assert record["residual_magnitude_weight"] == 1.0e-3
        assert 0.0 <= record["residual_share"] <= 1.0
        assert abs(record["physics_share"] + record["residual_share"] - 1.0) < 1e-6
        assert "residual_penalty" in record
        assert all(np.isfinite(v) for v in record["driver_param_stats"][
            "time_headway"].values())


def test_dry_run_hybrid_and_history_lane_head():
    dataset = _hybrid_dataset()
    for lane_head in ("off", "history"):
        cfg = WmSTrainConfig(device="cpu", threads=1, steps_per_epoch=2,
                             unroll_curriculum_s=(10.0,), batch_size=2,
                             rollout_mode="hybrid", lane_head=lane_head)
        result = dry_run(dataset, config=cfg)
        assert result["status"] == "passed"
        assert all(row["finite"] for row in result["updates"])
