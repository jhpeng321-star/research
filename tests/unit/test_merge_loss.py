"""V7FT merge-zone calibration loss (wms.train.losses.merge).

* exactly zero when the rollout equals the record; positive, finite and differentiable
  otherwise (chunked like truncated BPTT, discharge carry across chunks);
* the soft discharge count follows a known deficit and the validation metrics have the
  documented sign;
* switches default OFF and bit-identical (flat forms, composer, unroll record + gradients);
* the v7 / v7ft configs reach the settings and the composer.
"""
from __future__ import annotations

import pytest
import torch

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.losses import aggregate as agg
from wms.train.losses import merge as mg
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

NEW_FLAT_KEYS = tuple(f"merge_loss_{k}" for k in (
    "speed_weight", "discharge_weight", "cell_edges_m", "line_m", "bin_s", "cell_temp_m",
    "cross_temp_m", "speed_scale_mps", "discharge_scale_veh"))


def _recorded(b=2, t=40, k=10, seed=0):
    """HDVs driving through the merge zone (s from -150 to 200 m), one CAV in slot 0."""
    g = torch.Generator().manual_seed(seed)
    s0 = torch.linspace(-150.0, 60.0, k).repeat(b, 1) + torch.rand(b, k, generator=g) * 5
    v = (6.0 + 10.0 * torch.rand(b, 1, k, generator=g)).expand(b, t + 1, k).clone()
    s = s0[:, None, :] + torch.cumsum(torch.cat((torch.zeros(b, 1, k), v[:, :-1] * 0.5), 1), 1)
    lane = torch.zeros(b, t + 1, k, dtype=torch.long)
    lane[..., 1::2] = 1
    states = torch.zeros(b, t + 1, k, 7)
    states[..., 0] = s
    states[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    states[..., 4] = v
    states[..., 6] = 5.0
    present = torch.ones(b, t + 1, k, dtype=torch.bool)
    is_cav = torch.zeros(b, t + 1, k, dtype=torch.bool)
    is_cav[..., 0] = True
    return states, present, is_cav, lane


def _targets(states, present, is_cav):
    b, t1, k = present.shape
    n = t1 - 1
    rt = agg.road_targets({"states": states}, states, present, k, entries_on=False,
                          device="cpu")
    return UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1) / n,
        hdv_mask=present[:, 1:] & ~is_cav[:, 1:], aux_mask=present[:, 1:], aux_count=0,
        denom=torch.tensor(1.0), lane_target=torch.zeros(b, n, k, dtype=torch.long),
        loss_valid=present, is_cav=is_cav, dt_s=0.5, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False, road=rt)


def _ctx(targets, s, v, lane, mask, c0=0, c1=None):
    c1 = targets.n_steps if c1 is None else c1
    pred = {"positions": s[:, c0 + 1:c1 + 1], "speeds": v[:, c0 + 1:c1 + 1],
            "lanes": lane[:, c0 + 1:c1 + 1], "masks": mask[:, c0 + 1:c1 + 1]}
    current = {"s_m": s[:, c0], "present": mask[:, c0]}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def _loss(**kw):
    return mg.MergeLoss(mg.MergeLossConfig(speed_weight=1.0, discharge_weight=1.0, **kw))


def _run_chunks(loss, targets, s, v, lane, mask, cuts):
    total, outs = 0.0, []
    for c0, c1 in cuts:
        out = loss.components(_ctx(targets, s, v, lane, mask, c0, c1))
        outs.append(out)
        total = total + sum(out.values())
    return total, outs


def test_zero_when_prediction_equals_record():
    states, present, is_cav, lane = _recorded()
    targets = _targets(states, present, is_cav)
    loss = _loss()
    loss.prepare(targets)
    _, outs = _run_chunks(loss, targets, states[..., 0], states[..., 4], lane, present,
                          ((0, 15), (15, 40)))
    for out in outs:
        assert set(out) == set(mg.TERM_NAMES)
        for name, value in out.items():
            assert float(value) == 0.0, name
    metrics = mg.merge_metrics(targets.prepared["merge"], targets.n_steps)
    assert metrics["merge_speed_err_mps"] == 0.0
    assert metrics["merge_discharge_err_frac"] == 0.0
    assert metrics["merge_discharge_rec_veh"] > 1.0          # vehicles do cross the line


def test_positive_finite_and_gradient_flows():
    states, present, is_cav, lane = _recorded()
    targets = _targets(states, present, is_cav)
    loss = _loss()
    loss.prepare(targets)
    # a too-slow merge: HDVs at 70 % of the recorded speed (positions integrate it)
    v = (states[..., 4] * 0.7).requires_grad_()
    s0 = states[:, :1, :, 0]
    s = s0 + torch.cumsum(torch.cat((torch.zeros_like(v[:, :1]), v[:, :-1] * 0.5), 1), 1)
    total, outs = _run_chunks(loss, targets, s, v, lane, present, ((0, 15), (15, 40)))
    assert all(float(x.detach()) >= 0.0 for out in outs for x in out.values())
    assert all(float(outs[1][n].detach()) > 0.0 for n in mg.TERM_NAMES)
    assert bool(torch.isfinite(total)) and float(total.detach()) > 0.0
    total.backward()
    assert v.grad is not None and bool(torch.isfinite(v.grad).all())
    assert float(v.grad[..., 1:].abs().sum()) > 0.0          # HDVs receive gradient
    assert float(v.grad[..., 0].abs().sum()) == 0.0          # the (teacher-forced) CAV does not
    metrics = mg.merge_metrics(targets.prepared["merge"], targets.n_steps)
    assert metrics["merge_speed_bias_mps"] < 0.0             # WM slower than the record
    assert metrics["merge_discharge_err_frac"] < 0.0         # WM discharges fewer
    assert metrics["merge_discharge_abs_err_frac"] >= -metrics["merge_discharge_err_frac"]


def test_chunking_and_recomputed_chunk_do_not_double_count():
    states, present, is_cav, lane = _recorded()
    targets = _targets(states, present, is_cav)
    s = states[..., 0] - 4.0
    v = states[..., 4] * 0.8
    loss = _loss(bin_s=2.5)                                    # 5-step bins; cut at 15
    loss.prepare(targets)
    whole = loss.components(_ctx(targets, s, v, lane, present))
    loss.prepare(targets)
    first = loss.components(_ctx(targets, s, v, lane, present, 0, 15))
    loss.components(_ctx(targets, s, v, lane, present, 0, 15))   # recomputed chunk
    second = loss.components(_ctx(targets, s, v, lane, present, 15, 40))
    for name in mg.TERM_NAMES:
        torch.testing.assert_close(first[name] + second[name], whole[name])


def test_window_padding_is_masked():
    states, present, is_cav, lane = _recorded()
    targets = _targets(states, present, is_cav)
    targets.road.valid[:, 25:] = False
    loss = _loss()
    loss.prepare(targets)
    s = states[..., 0].clone()
    s[:, 25:] -= 30.0                                          # errors only on padded steps
    v = states[..., 4].clone()
    v[:, 25:] *= 0.5
    out = loss.components(_ctx(targets, s, v, lane, present))
    assert all(float(x) == 0.0 for x in out.values())


def test_config_validation():
    with pytest.raises(ValueError):
        mg.MergeLossConfig(cell_edges_m=(0.0, 0.0))
    with pytest.raises(ValueError):
        mg.MergeLossConfig(speed_weight=-1.0)
    assert not mg.MergeLossConfig().enabled and mg.merge_terms(mg.MergeLossConfig()) == []


# ----------------------------------------------------------------------- switches default OFF
def _default_batch():
    b, t, k = 2, 7, 4
    torch.manual_seed(0)
    states = torch.zeros(b, t, k, 7)
    states[..., 0] = torch.rand(b, t, k) * 80 + torch.arange(k) * 22
    states[..., 2] = 1.0
    states[..., 4] = torch.rand(b, t, k) * 15 + 5
    states[..., 5] = 0.1
    states[..., 6] = 5.0
    present = torch.ones(b, t, k, dtype=torch.bool)
    present[:, :, 3] = False
    is_cav = torch.zeros(b, t, k, dtype=torch.bool)
    is_cav[:, :, 0] = True
    hist = torch.zeros(b, k, 10, 6)
    hist[..., 0] = 40.0
    hist[..., 4] = 18.0
    return {"states": states, "present": present, "is_cav": is_cav,
            "cav_teacher": torch.zeros(b, t, k, 4), "hist": hist,
            "hist_mask": torch.ones(b, k, 10, dtype=torch.bool)}


def _run(settings, batch, steps=6, truncation=None):
    torch.manual_seed(0)
    model = WmSModel(ModelSpec())
    model.train()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    record = runner.run(batch, unroll_steps=steps, backward=True, truncation_steps=truncation)
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return record, grads


def test_switches_default_off_and_flat_forms_unchanged():
    flat = TrainConfig().to_legacy_flat()
    assert not set(NEW_FLAT_KEYS) & set(flat)
    for name in ("c4_wm", "configs/c4_wm_v5.toml", "configs/c4_wm_v5m_refresh.toml"):
        assert not set(NEW_FLAT_KEYS) & set(load_train_config(name).to_legacy_flat()), name
    assert build_loss(TrainSettings()).names == ("position", "speed", "lane")
    assert not TrainSettings().merge_loss_on


def test_explicit_off_is_bit_identical():
    base_rec, base_grads = _run(TrainSettings(), _default_batch())
    off_rec, off_grads = _run(TrainSettings(merge_loss_speed_weight=0.0,
                                            merge_loss_discharge_weight=0.0,
                                            merge_loss_cell_edges_m=(-50.0, 0.0, 50.0),
                                            merge_loss_cross_temp_m=9.0), _default_batch())
    assert base_rec == off_rec
    assert len(base_grads) == len(off_grads)
    assert all(torch.equal(a, b) for a, b in zip(base_grads, off_grads))


def test_merge_on_in_the_unroll_logs_terms_and_metrics():
    base_rec, _ = _run(TrainSettings(), _default_batch(), truncation=2)
    settings = TrainSettings(merge_loss_speed_weight=2.0, merge_loss_discharge_weight=3.0,
                             merge_loss_cell_edges_m=(0.0, 60.0, 200.0), merge_loss_line_m=60.0)
    assert tuple(t.name for t in build_loss(settings).chunk_terms)[-2:] == ("merge_speed", "merge_discharge")
    rec, grads = _run(settings, _default_batch(), truncation=2)
    for key in ("position", "speed", "lane", "valid_targets"):
        assert rec[key] == base_rec[key], key
    for key in ("merge_speed", "merge_discharge", *mg.METRIC_NAMES, "merge_value_share",
                "merge_speed_bias_mps_c0", "merge_speed_bias_mps_c1"):
        assert key in rec and torch.isfinite(torch.tensor(float(rec[key]))), key
    assert rec["merge_speed"] >= 0.0 and rec["merge_discharge"] >= 0.0
    assert all(bool(torch.isfinite(g).all()) for g in grads)


# ----------------------------------------------------------------------- the new configs
@pytest.mark.parametrize("path,init", [("configs/c4_wm_v7.toml", False),
                                       ("configs/c4_wm_v7ft.toml", True)])
def test_v7_configs_reach_settings_and_composer(path, init):
    cfg = load_train_config(path)
    s = TrainSettings.from_config(cfg)
    assert s.merge_loss_on and s.merge_end_wait and s.mainline_merge_yield
    assert s.merge_loss_cell_edges_m == (-100.0, 0.0, 75.44, 300.0)
    assert tuple(t.name for t in build_loss(s).chunk_terms)[-2:] == ("merge_speed", "merge_discharge")
    assert cfg.required_log_terms()[-2:] == ("merge_speed", "merge_discharge")
    assert (cfg.model.init_from is not None) == init
    ratios = {src.name: src.mix_ratio for src in cfg.data.c4_sources}
    assert abs(sum(ratios.values()) - 1.0) < 1e-9 and "v6_policy" in ratios
    back = TrainConfig.from_legacy_flat(cfg.to_legacy_flat())
    assert back.loss.merge == cfg.loss.merge


# ----------------------------------------------------------------------- 2-step CPU fit smoke
def test_fit_two_steps_logs_merge_terms_in_train_and_val(tmp_path):
    import json

    import numpy as np

    from wms.train.trainer import WmSTrainer

    cfg = TrainConfig.from_dict({
        "runtime": {"device": "cpu", "threads": 2, "omp": 2},
        "curriculum": {"unroll_s": [3.0], "stage_steps": [2], "truncation_steps": 2},
        "optim": {"batch_size": 2},
        "selection": {"select_by_long_horizon": True, "val_every": 1},
        "loss": {"merge": {"speed_weight": 10.0, "discharge_weight": 20.0,
                           "cell_edges_m": [0.0, 60.0, 200.0], "line_m": 60.0}}})
    batch = _default_batch()

    class Windows(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            rec = {k: v[index].numpy() for k, v in batch.items() if torch.is_tensor(v)}
            rec["slot_index"] = np.arange(4)
            rec["meta"] = {"world": "world-A"}
            return rec

    torch.manual_seed(0)
    model = WmSModel(ModelSpec())
    trainer = WmSTrainer(cfg, model, Windows(), val_dataset=Windows(), out_dir=tmp_path)
    trainer.fit()
    rows = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["merge_speed"] >= 0.0 and row["merge_discharge"] >= 0.0
        assert "merge_discharge_err_frac" in row and "merge_value_share" in row
    val = [json.loads(x) for x in (tmp_path / "val_log.jsonl").read_text().splitlines()]
    assert val and all("mean_merge_speed_err_mps" in r and "mean_merge_discharge_err_frac" in r
                       and "mean_merge_speed" in r for r in val)
