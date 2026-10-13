"""C4 WM v2 loss (WMLOSS_NOTES.md): aggregate-consistency loss (B) and entries in training (E).

* switches default OFF and bit-identical (flat forms, composer, unroll record);
* the aggregate loss is exactly zero on identical trajectories, positive and differentiable
  otherwise, and its hard limit agrees with the RSU field the C4 policy reads;
* entries-in-training shapes (adaptive pad, padded inputs, full-road recorded targets);
* the new config reaches the settings and composer.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.entries import EntranceHistory
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.inputs import build_targets, entry_pad_for, prepare_inputs
from wms.train.losses import aggregate as agg
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

NEW_FLAT_KEYS = ("aggregate_weight", "aggregate_field_weight", "aggregate_reward_weight",
                 "aggregate_flow_weight", "aggregate_bin_s", "aggregate_every_steps",
                 "aggregate_horizon_start_s", "aggregate_horizon_full_s",
                 "aggregate_cell_temp_m", "entry_pad_auto", "entry_pad_max")


# ----------------------------------------------------------------------------- fixtures
def _recorded(b=2, t=30, k=12, seed=0):
    """Recorded window inside the RSU zones: three lanes, a slow (queued) group, an exit
    at t=10 and a NEW vehicle re-using that slot from t=15 (full-road presence)."""
    g = torch.Generator().manual_seed(seed)
    s0 = torch.linspace(-700.0, 180.0, k).repeat(b, 1) + torch.rand(b, k, generator=g) * 10
    v = 5.0 + 20.0 * torch.rand(b, 1, k, generator=g).expand(b, t + 1, k).clone()
    v[:, :, :3] = 1.0                                      # a slow group -> queue cells
    s = s0[:, None, :] + torch.cumsum(torch.cat((torch.zeros(b, 1, k), v[:, :-1] * 0.5), 1), 1)
    lane = torch.zeros(b, t + 1, k, dtype=torch.long)
    lane[..., 1::3] = 1
    lane[..., 2::3] = 2
    s = torch.where(lane == 2, s.clamp(max=70.0), s)       # ramp vehicles stay on the ramp
    s = s.clone()
    s[:, 15:, 4] = -790.0 + torch.arange(t + 1 - 15) * 0.5 * 20.0   # slot 4: new vehicle
    states = torch.zeros(b, t + 1, k, 7)
    states[..., 0] = s
    states[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    states[..., 4] = v
    states[..., 6] = 5.0
    road = torch.ones(b, t + 1, k, dtype=torch.bool)
    road[:, 10:15, 4] = False                              # exit, then a new arrival in slot 4
    cont = road[:, :-1] & road[:, 1:]
    cont[:, 14, 4] = False
    cohort = road.clone()
    cohort[:, 10:, 4] = False                              # alive-since-t0
    return states, road, cont, cohort, lane


def _targets(states, cohort, road=None, cont=None, *, full: bool):
    b, t1, k = cohort.shape
    n = t1 - 1
    batch = {"states": states}
    if road is not None:
        batch.update(road_present=road, road_continue=cont,
                     road_valid=torch.ones(b, t1, dtype=torch.bool))
    rt = agg.road_targets(batch, states, cohort, k, entries_on=full, device="cpu")
    return UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1) / n,
        hdv_mask=cohort[:, 1:], aux_mask=cohort[:, 1:], aux_count=0,
        denom=torch.tensor(1.0), lane_target=torch.zeros(b, n, k, dtype=torch.long),
        loss_valid=cohort, is_cav=torch.zeros_like(cohort), dt_s=0.5, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False, road=rt)


def _ctx(targets, s, v, lane, mask, c0=0, c1=None):
    c1 = targets.n_steps if c1 is None else c1
    pred = {"positions": s[:, c0 + 1:c1 + 1], "speeds": v[:, c0 + 1:c1 + 1],
            "lanes": lane[:, c0 + 1:c1 + 1], "masks": mask[:, c0 + 1:c1 + 1]}
    current = {"s_m": s[:, c0], "present": mask[:, c0]}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def _loss(**kw):
    cfg = agg.AggregateLossConfig(flow_weight=1.0, **kw)
    return agg.AggregateLoss(cfg)


# ----------------------------------------------------------------------- contract / geometry
def test_geometry_and_constants_match_the_c4_contract():
    from wms.control import contract, reward, rsu_field

    assert agg.CELLS == rsu_field.CELLS
    assert agg.SPEED_LIMIT_MPS == contract.SPEED_LIMIT_MPS
    assert agg.MERGE_S1_M == contract.MERGE_S1_M
    assert agg.RSU_MAINLINE_S_M == contract.RSU_MAINLINE_S_M
    assert agg.RSU_RAMP_S_M == contract.RSU_RAMP_S_M
    assert agg.RSU_CELL_M == contract.RSU_CELL_M
    assert agg.QUEUE_SPEED_MPS == rsu_field.QUEUE_SPEED_MPS
    assert agg.CROSSING_WIDTH_M == reward._CROSSING_WIDTH_M


def test_soft_fields_hard_limit_equals_rsu_field():
    from wms.control.rsu_field import build_rsu_field

    g = torch.Generator().manual_seed(4)
    k = 40
    cell = torch.randint(0, 20, (1, k), generator=g)
    s = -800.0 + 50.0 * cell + 10.0 + 30.0 * torch.rand(1, k, generator=g)  # away from edges
    lane = torch.randint(0, 2, (1, k), generator=g)
    v = torch.rand(1, k, generator=g) * 25.0
    v[0, :8] = 1.0
    present = torch.ones(1, k, dtype=torch.bool)
    present[0, -3:] = False
    cfg = agg.AggregateLossConfig(cell_temp_m=0.01, speed_prior_count=1e-6,
                                  queue_temp_mps=1e-3, occupied_temp=1e-3)
    soft = agg.soft_cell_fields(s[:, None], v[:, None], lane[:, None], present[:, None], cfg=cfg)
    hard = build_rsu_field({"s_m": s, "v_mps": v, "lane": lane, "present": present})[0]
    occupied = hard[:, 4] > 0
    torch.testing.assert_close(soft["density"][0, 0], hard[:, 0], atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(soft["mean_speed"][0, 0][occupied], hard[occupied, 1],
                               atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(soft["mean_speed"][0, 0][~occupied],
                               hard[~occupied, 1], atol=1e-2, rtol=0)
    queue_ok = (hard[:, 1] - agg.QUEUE_SPEED_MPS).abs() > 0.05
    torch.testing.assert_close(soft["queue"][0, 0][queue_ok], hard[queue_ok, 3], atol=1e-3,
                               rtol=0)
    # reward delay convention (dt * relu(1 - v / limit)) over present vehicles
    delay = (0.5 * torch.relu(1 - v / agg.SPEED_LIMIT_MPS) * present).sum()
    torch.testing.assert_close(soft["delay_road"][0, 0], delay)


# ----------------------------------------------------------------------- zero / positive / grad
@pytest.mark.parametrize("full", [True, False])
def test_aggregate_zero_on_identical_trajectories(full):
    states, road, cont, cohort, lane = _recorded()
    targets = _targets(states, cohort, road, cont, full=full)
    loss = _loss(bin_s=2.0)
    loss.prepare(targets)
    mask = road if full else cohort
    out = loss.components(_ctx(targets, states[..., 0], states[..., 4], lane, mask))
    assert set(out) == set(agg.TERM_NAMES)
    for name, value in out.items():
        assert float(value) == 0.0, name


@pytest.mark.parametrize("full", [True, False])
def test_aggregate_positive_and_differentiable(full):
    states, road, cont, cohort, lane = _recorded()
    targets = _targets(states, cohort, road, cont, full=full)
    loss = _loss(bin_s=2.0)
    loss.prepare(targets)
    g = torch.Generator().manual_seed(1)
    s = (states[..., 0] + 8.0 * torch.randn(states[..., 0].shape, generator=g)).requires_grad_()
    v = (states[..., 4] * 0.6).requires_grad_()
    mask = road if full else cohort
    total = 0.0
    for c0, c1 in ((0, 12), (12, 30)):                      # chunked like truncated BPTT
        out = loss.components(_ctx(targets, s, v, lane, mask, c0, c1))
        assert all(float(x.detach()) >= 0.0 for x in out.values())
        total = total + sum(out.values())
    assert float(total.detach()) > 0.0
    total.backward()
    for leaf in (s, v):
        assert leaf.grad is not None and bool(torch.isfinite(leaf.grad).all())
        assert float(leaf.grad.abs().sum()) > 0.0


def test_chunk_sum_equals_single_chunk_when_bins_align():
    states, road, cont, cohort, lane = _recorded()
    targets = _targets(states, cohort, road, cont, full=True)
    loss = _loss(bin_s=2.0)                                 # 4-step bins; chunks at 12
    loss.prepare(targets)
    s = states[..., 0] + 3.0
    v = states[..., 4] * 0.8
    whole = loss.components(_ctx(targets, s, v, lane, road))
    parts = [loss.components(_ctx(targets, s, v, lane, road, a, b)) for a, b in ((0, 12), (12, 30))]
    for name in agg.TERM_NAMES:
        torch.testing.assert_close(parts[0][name] + parts[1][name], whole[name])


def test_masked_where_recorded_future_unavailable():
    states, road, cont, cohort, lane = _recorded()
    targets = _targets(states, cohort, road, cont, full=True)
    targets.road.valid[:, 20:] = False                       # window padding from t=20
    loss = _loss(bin_s=2.0)
    loss.prepare(targets)
    s = states[..., 0].clone()
    s[:, 21:] += 50.0                                        # errors only on unavailable steps
    out = loss.components(_ctx(targets, s, states[..., 4], lane, road))
    assert all(float(x) == 0.0 for x in out.values())


def test_horizon_ramp():
    cfg = agg.AggregateLossConfig(horizon_start_s=5.0, horizon_full_s=15.0)
    r = cfg.ramp(torch.tensor([0, 10, 20, 30, 40]))
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    step = agg.AggregateLossConfig(horizon_start_s=5.0, horizon_full_s=5.0).ramp(
        torch.tensor([9, 10]))
    torch.testing.assert_close(step, torch.tensor([0.0, 1.0]))


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


def _run(settings, batch, steps=6):
    torch.manual_seed(0)
    model = WmSModel(ModelSpec())
    model.train()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    record = runner.run(batch, unroll_steps=steps, backward=True)
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return record, grads


def test_switches_default_off_and_flat_forms_unchanged():
    flat = TrainConfig().to_legacy_flat()
    assert not set(NEW_FLAT_KEYS) & set(flat)
    for name in ("c4_wm", "r4", "r5b"):
        assert not set(NEW_FLAT_KEYS) & set(load_train_config(name).to_legacy_flat()), name
    assert build_loss(TrainSettings()).names == ("position", "speed", "lane")
    assert TrainSettings().aggregate_weight == 0.0 and not TrainSettings().entry_pad_auto


def test_explicit_off_switches_are_bit_identical():
    base_rec, base_grads = _run(TrainSettings(), _default_batch())
    off_rec, off_grads = _run(TrainSettings(aggregate_weight=0.0, aggregate_flow_weight=0.7,
                                            entry_pad_auto=False, entry_pad_max=99),
                              _default_batch())
    assert base_rec == off_rec
    assert all(torch.equal(a, b) for a, b in zip(base_grads, off_grads))


def test_aggregate_on_leaves_per_car_terms_unchanged_and_logs_share():
    base_rec, _ = _run(TrainSettings(), _default_batch())
    rec, grads = _run(TrainSettings(aggregate_weight=1.0, aggregate_flow_weight=0.5),
                      _default_batch())
    for key in ("position", "speed", "lane", "valid_targets"):
        assert rec[key] == base_rec[key], key
    for key in ("agg_field", "agg_reward", "agg_flow"):
        assert key in rec and np.isfinite(rec[key]) and rec[key] >= 0.0
    assert 0.0 <= rec["aggregate_value_share"] < 1.0
    assert all(bool(torch.isfinite(g).all()) for g in grads)


# ----------------------------------------------------------------------- entries in training
def _entry_feat(rate_main=0.6, rate_ramp=0.2):
    h = dataclasses.replace(EntranceHistory.defaults(), rate=np.array([rate_main, rate_ramp]))
    return torch.as_tensor(h.to_features(), dtype=torch.float32)


def test_adaptive_entry_pad_formula():
    s = TrainSettings(entries_v2=True, entry_pad=8, entry_pad_auto=True, entry_pad_max=100)
    h = EntranceHistory.from_features(_entry_feat().numpy())
    rate = float(np.sum(h.rate))
    assert abs(rate - 0.8) < 1e-6
    assert entry_pad_for(s, [h], 60.0) == int(np.ceil(rate * 60 * 1.5)) + 16
    assert entry_pad_for(s, [h], 600.0) == 100                       # capped
    assert entry_pad_for(dataclasses.replace(s, entry_pad_auto=False), [h], 600.0) == 8


def test_entries_in_training_shapes_and_full_road_targets():
    batch = _default_batch()
    b, t1, k = batch["present"].shape
    batch["entry_feat"] = _entry_feat().expand(b, -1).clone()
    batch["road_present"] = batch["present"].clone()
    batch["road_continue"] = batch["present"][:, :-1] & batch["present"][:, 1:]
    batch["road_valid"] = torch.ones(b, t1, dtype=torch.bool)
    settings = TrainSettings(entries_v2=True, entry_pad=4, entry_pad_auto=True,
                             entry_pad_max=64, aggregate_weight=1.0, aggregate_flow_weight=0.5)
    torch.manual_seed(0)
    model = WmSModel(ModelSpec())
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 6)
    rate = float(np.sum(EntranceHistory.from_features(_entry_feat().numpy()).rate))
    pad = int(np.ceil(rate * 6 * 0.5 * 1.5)) + 16
    assert inp.entries is not None and inp.entries.train_pad == pad
    assert inp.states.shape == (b, t1, k + pad, 7)
    assert inp.present.shape == inp.present_seq.shape == inp.loss_valid.shape == (b, t1, k + pad)
    assert not bool(inp.present[..., k:].any()) and bool(inp.present_seq[..., k:].all())
    assert not bool(inp.loss_valid[..., k:].any())
    targets = build_targets(inp, batch, settings, 0.5, torch.device("cpu"))
    assert targets.road.full and targets.road.k == k
    assert targets.road.present.shape == (b, t1, k)
    # without the recorded road arrays the aggregate falls back to the t0 cohort
    bare = {key: value for key, value in batch.items() if not key.startswith("road_")}
    inp2 = prepare_inputs(bare, model, settings, torch.device("cpu"), 6)
    cohort = build_targets(inp2, bare, settings, 0.5, torch.device("cpu")).road
    assert not cohort.full and cohort.k == k
    # one full forward/backward with entries + aggregate
    record, grads = _run(settings, batch)
    assert record["unroll_steps"] == 6 and np.isfinite(record["loss"])
    assert {"agg_field", "agg_reward", "agg_flow"} <= set(record)
    assert all(bool(torch.isfinite(g).all()) for g in grads)


def test_road_presence_and_collate():
    from wms.data.collate import collate
    from wms.data.dataset import road_presence

    class _Source:
        def __init__(self):
            self.mask = np.zeros((10, 3), bool)
            self.mask[2:, 0] = True
            self.mask[5:, 1] = True                       # arrives after t0
            self.target = np.ones((10, 3), bool)

        def array(self, name):
            return {"veh_kin": np.zeros((10, 3, 7)), "veh_mask": self.mask,
                    "target_mask": self.target}[name]

        def run_end(self, tick):
            return 8

    out = road_presence(_Source(), 3, 6)                   # ticks 3..8 inside the run, 4 avail
    assert out["road_present"].shape == (7, 3) and out["road_continue"].shape == (6, 3)
    assert out["road_valid"].tolist() == [True] * 5 + [False] * 2
    assert out["road_present"][:, 1].tolist() == [False, False, True, True, True, False, False]
    records = [{"states": np.zeros((7, 3, 7), np.float32), "present": np.ones((7, 3), bool),
                "slot_index": np.zeros((7, 3), np.int64), **out} for _ in range(2)]
    batch = collate(records)
    assert batch["road_present"].dtype == torch.bool and batch["road_present"].shape == (2, 7, 3)
    assert "road_present" not in collate([{k: v for k, v in r.items()
                                           if not k.startswith("road_")} for r in records])


# ----------------------------------------------------------------------- the new config
def test_v2_loss_config_reaches_settings_and_composer():
    cfg = load_train_config("configs/c4_wm_v2_loss.toml")
    s = TrainSettings.from_config(cfg)
    assert s.unroll_curriculum_s == (10.0, 30.0, 60.0, 120.0)
    assert s.truncation_steps_per_stage == (20, 60, 120, 120)
    assert len(s.lr_per_stage) == 4 and s.grad_decay_per_step == 0.90 and s.checkpoint_step
    assert s.entries_v2 and s.entry_pad_auto and s.aggregate_weight > 0.0
    names = build_loss(s).names
    assert names[-3:] == ("agg_field", "agg_reward", "agg_flow")
    assert cfg.required_log_terms()[-3:] == ("agg_field", "agg_reward", "agg_flow")
    back = TrainConfig.from_legacy_flat(cfg.to_legacy_flat())
    assert back.loss.aggregate == cfg.loss.aggregate
