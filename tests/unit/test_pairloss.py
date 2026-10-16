"""PAIRLOSS: multi-source C4 data (GENRAND) and the paired-difference loss (PAIRLOSS_NOTES.md)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.core.errors import ConfigError
from wms.data.collate import collate
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.losses import aggregate as agg
from wms.train.losses import paired_diff as pdl
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

NEW_FLAT_KEYS = ("paired_diff_weight", "paired_diff_fraction", "paired_diff_groups",
                 "paired_diff_max_arms", "paired_diff_position_weight",
                 "paired_diff_speed_weight", "paired_diff_aggregate_weight")
SMOKE = Path("runs/active/C4-GENRAND-SMOKE-20261014")
PLAIN, BRANCH, DRIVER = SMOKE / "plain_conv", SMOKE / "branch_conv", SMOKE / "driver/c4_episodes"
needs_smoke = pytest.mark.skipif(not (BRANCH / "C4_EPISODES_MANIFEST.json").is_file(),
                                 reason="GENRAND smoke outputs not present")


# ----------------------------------------------------------------------------- config
def test_defaults_off_and_flat_forms_unchanged():
    flat = TrainConfig().to_legacy_flat()
    assert not set(NEW_FLAT_KEYS) & set(flat)
    for name in ("c4_wm", "c4_wm_v4_lane", "c4_wm_v4_hygiene"):
        cfg = load_train_config(name)
        assert not set(NEW_FLAT_KEYS) & set(cfg.to_legacy_flat()), name
        assert cfg.data.c4_sources == ()
    assert build_loss(TrainSettings()).names == ("position", "speed", "lane")
    assert TrainSettings().paired_diff_weight == 0.0


def test_v4_data_config_reaches_settings_and_composer():
    cfg = load_train_config("configs/c4_wm_v4_data.toml")
    assert cfg.data.controller_hygiene == "strict" and cfg.data.junction_fix == "ffill"
    names = {s.name: s for s in cfg.data.c4_sources}
    assert set(names) == {"genrand_plain", "genrand_branch"} and names["genrand_branch"].paired
    assert cfg.model.lane_model == "hazard" and cfg.loss.lane_hazard_decouple  # v4_lane kept
    settings = TrainSettings.from_config(cfg)
    assert settings.paired_diff_weight == cfg.loss.paired_diff.weight > 0
    composer = build_loss(settings)
    assert tuple(t.name for t in composer.chunk_terms)[-3:] == ("pair_pos", "pair_speed", "pair_agg")
    assert set(cfg.required_log_terms()) >= {"pair_pos", "pair_speed", "pair_agg"}
    round_trip = TrainConfig.from_legacy_flat(cfg.to_legacy_flat())
    assert round_trip.loss.paired_diff == cfg.loss.paired_diff


def test_config_validation():
    base = load_train_config("configs/c4_wm_v4_data.toml")
    with pytest.raises(ConfigError):
        dataclasses.replace(base.data.c4_sources[0], mix_ratio=0.0)   # non-paired needs a share
    with pytest.raises(ConfigError):
        dataclasses.replace(base.data, c4_sources=(base.data.c4_sources[0],) * 2)  # dup names
    with pytest.raises(ConfigError):
        dataclasses.replace(base.data, c4_sources=(
            dataclasses.replace(base.data.c4_sources[0], mix_ratio=0.99),
            base.data.c4_sources[1]))                                  # shares sum >= 1
    no_pair = dataclasses.replace(base, data=dataclasses.replace(
        base.data, c4_sources=base.data.c4_sources[:1]))
    with pytest.raises(ConfigError):
        no_pair.validate_for_training()
    with pytest.raises(ValueError):
        TrainSettings(paired_diff_fraction=1.5)


# ----------------------------------------------------------------------------- loss (synthetic)
def _recorded(b=2, t=30, k=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    s0 = torch.linspace(-700.0, 150.0, k).repeat(b, 1)
    v = (5.0 + 20.0 * torch.rand(1, 1, k, generator=g)).expand(b, t + 1, k).clone()
    s = s0[:, None, :] + torch.cumsum(torch.cat((torch.zeros(b, 1, k), v[:, :-1] * 0.5), 1), 1)
    lane = torch.zeros(b, t + 1, k, dtype=torch.long)
    lane[..., 1::3] = 1
    lane[..., 2::3] = 2
    s = torch.where(lane == 2, s.clamp(max=70.0), s)
    states = torch.zeros(b, t + 1, k, 7)
    states[..., 0] = s
    states[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    states[..., 4] = v
    states[..., 6] = 5.0
    cohort = torch.ones(b, t + 1, k, dtype=torch.bool)
    return states, cohort, lane


def _pair_targets(states, cohort, ref):
    b, t1, k = cohort.shape
    n = t1 - 1
    batch = {"states": states, "pair_ref": torch.as_tensor(ref)}
    targets = UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1) / n,
        hdv_mask=cohort[:, 1:], aux_mask=cohort[:, 1:], aux_count=0,
        denom=torch.tensor(1.0), lane_target=torch.zeros(b, n, k, dtype=torch.long),
        loss_valid=cohort, is_cav=torch.zeros_like(cohort), dt_s=0.5, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False)
    targets.pair = pdl.pair_targets(batch, states, cohort, k, device="cpu")
    return targets


def _ctx(targets, s, v, lane, mask):
    n = targets.n_steps
    pred = {"positions": s[:, 1:n + 1], "speeds": v[:, 1:n + 1], "lanes": lane[:, 1:n + 1],
            "masks": mask[:, 1:n + 1]}
    return LossContext(pred=pred, current={"s_m": s[:, 0], "present": mask[:, 0]},
                       targets=targets, c0=0, c1=n)


def _paired(cfg=None):
    return pdl.PairedDiffLoss(cfg or pdl.PairedDiffConfig())


def _arm_states(states):
    """Row 1 = row 0 with a different CAV action effect: vehicles 3..6 slowed down."""
    out = states.clone()
    slow = torch.zeros_like(out[1, :, :, 4])
    slow[:, 3:7] = torch.linspace(0.0, 4.0, out.shape[1]).unsqueeze(-1)
    out[1, :, :, 4] = (out[1, :, :, 4] - slow).clamp_min(0.0)
    out[1, :, :, 0] = out[1, :1, :, 0] + torch.cumsum(torch.cat(
        (torch.zeros(1, out.shape[2]), out[1, :-1, :, 4] * 0.5), 0), 0)
    return out


def test_zero_when_predicted_differences_equal_recorded():
    states, cohort, lane = _recorded()
    states = _arm_states(states)
    targets = _pair_targets(states, cohort, [0, 0])
    loss = _paired()
    loss.prepare(targets)
    # identical arms (prediction and record of both rows equal) -> exactly zero
    same = states[:1].expand(2, -1, -1, -1).clone()
    t_same = _pair_targets(same, cohort, [0, 0])
    loss.prepare(t_same)
    out = loss.components(_ctx(t_same, same[..., 0] + 3.0, same[..., 4] + 0.5, lane, cohort))
    assert all(float(v) == 0.0 for v in out.values())
    # a common offset of both arms' predictions leaves every difference exact
    loss.prepare(targets)
    out = loss.components(_ctx(targets, states[..., 0] + 2.0, states[..., 4], lane, cohort))
    assert float(out["pair_pos"]) == pytest.approx(0.0, abs=1e-6)
    assert float(out["pair_speed"]) == pytest.approx(0.0, abs=1e-6)


def test_positive_and_differentiable_when_differences_are_wrong():
    states, cohort, lane = _recorded()
    states = _arm_states(states)
    targets = _pair_targets(states, cohort, [0, 0])
    loss = _paired()
    loss.prepare(targets)
    s = states[..., 0].clone()
    v = states[..., 4].clone()
    s[1] = s[0]          # the model predicts NO action effect in arm 1
    v[1] = v[0]
    s.requires_grad_(True)
    v.requires_grad_(True)
    out = loss.components(_ctx(targets, s, v, lane, cohort))
    for name in pdl.TERM_NAMES:
        assert float(out[name]) > 0.0, name
    sum(out.values()).backward()
    assert float(s.grad.abs().sum()) > 0.0 and float(v.grad.abs().sum()) > 0.0
    # reference rows carry no term of their own, but receive gradient through the differences
    assert float(s.grad[0].abs().sum()) > 0.0


def test_aggregate_difference_matches_the_aggregate_code():
    """With the reference predicted exactly, pair_agg == agg_reward of the arm (cohort mode)."""
    states, cohort, lane = _recorded()
    states = _arm_states(states)
    targets = _pair_targets(states, cohort, [0, 0])
    loss = _paired(pdl.PairedDiffConfig(position_weight=0.0, speed_weight=0.0))
    loss.prepare(targets)
    s = states[..., 0].clone()
    v = states[..., 4].clone()
    s[1] = s[1] + torch.linspace(0.0, 30.0, s.shape[1]).unsqueeze(-1)   # arm 1 predicted wrong
    v[1] = v[1] * 1.3
    pair = loss.components(_ctx(targets, s, v, lane, cohort))["pair_agg"]
    # aggregate loss on the arm row alone, cohort targets, reward term only, no ramp
    arm_states, arm_cohort = states[1:], cohort[1:]
    n = targets.n_steps
    road = agg.road_targets({"states": arm_states}, arm_states, arm_cohort,
                            arm_cohort.shape[-1], entries_on=False, device="cpu")
    t_arm = UnrollTargets(
        states=arm_states, n_steps=n, weights=torch.ones(1, n, 1) / n,
        hdv_mask=arm_cohort[:, 1:], aux_mask=arm_cohort[:, 1:], aux_count=0,
        denom=torch.tensor(1.0), lane_target=torch.zeros(1, n, arm_cohort.shape[-1],
                                                         dtype=torch.long),
        loss_valid=arm_cohort, is_cav=torch.zeros_like(arm_cohort), dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False, road=road)
    aloss = agg.AggregateLoss(agg.AggregateLossConfig(field_weight=0.0, reward_weight=1.0))
    aloss.prepare(t_arm)
    ref = aloss.components(_ctx(t_arm, s[1:], v[1:], lane[1:], arm_cohort))["agg_reward"]
    assert float(pair) > 0.0
    assert float(pair) == pytest.approx(float(ref), rel=1e-5)


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


def test_off_is_bit_identical_one_training_step():
    base_rec, base_grads = _run(TrainSettings(), _default_batch())
    paired_batch = {**_default_batch(), "pair_group": torch.tensor([0, 0]),
                    "pair_ref": torch.tensor([0, 0])}
    off_rec, off_grads = _run(TrainSettings(paired_diff_weight=0.0, paired_diff_groups=7,
                                            paired_diff_fraction=1.0,
                                            paired_diff_aggregate_weight=3.0), paired_batch)
    assert base_rec == off_rec
    assert all(torch.equal(a, b) for a, b in zip(base_grads, off_grads))


def test_on_leaves_per_car_terms_unchanged_and_logs_share():
    base_rec, _ = _run(TrainSettings(), _default_batch())
    paired_batch = {**_default_batch(), "pair_group": torch.tensor([0, 0]),
                    "pair_ref": torch.tensor([0, 0])}
    rec, grads = _run(TrainSettings(paired_diff_weight=1.0), paired_batch)
    for key in ("position", "speed", "lane", "valid_targets"):
        assert rec[key] == base_rec[key], key
    for key in pdl.TERM_NAMES:
        assert key in rec and np.isfinite(rec[key]) and rec[key] >= 0.0
    assert rec["pair_pos"] > 0.0 and rec["paired_batch"] and rec["paired_arms"] == 1
    assert 0.0 < rec["paired_value_share"] < 1.0
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    # an ordinary batch: zero paired terms, keys still logged
    plain, _ = _run(TrainSettings(paired_diff_weight=1.0), _default_batch())
    assert plain["pair_pos"] == 0.0 and plain["paired_batch"] is False
    assert plain["position"] == base_rec["position"]


# ----------------------------------------------------------------------------- data (smoke)
def test_world_roles_val_wins():
    from wms.data.c4_pairs import world_roles

    docs = [{"episodes": [{"world": "a", "split_role": "train"},
                          {"world": "b", "split_role": "val"}]},
            {"episodes": [{"world": "a", "split_role": "val"},
                          {"world": "c", "split_role": "train"}]}]
    roles, overridden = world_roles(docs)
    assert roles == {"a": "val", "b": "val", "c": "train"} and overridden == ["a"]


@needs_smoke
def test_multiple_sources_load_split_by_world_and_weights_respected():
    from wms.data.c4_episodes import build_c4_datasets, read_source_manifest
    from wms.data.c4_pairs import add_c4_sources, world_roles

    docs = [read_source_manifest(p) for p in (PLAIN, DRIVER)]
    roles, _ = world_roles(docs)
    common = dict(horizon_steps=40, history_s=60.0, entries_v2=True, role_of=roles,
                  junction_fix="ffill", command_known=True)
    plain = build_c4_datasets(PLAIN, stride_s=10.0, **common)
    driver = build_c4_datasets(DRIVER, stride_s=10.0, **common)
    # the driver dir holds base + branch episodes of world w4110 on the VAL side and w4109 TRAIN
    train_worlds = set(driver["train"].receipt["worlds"])
    val_worlds = set(driver["val"].receipt["worlds"])
    assert not train_worlds & val_worlds
    val_eps = driver["val"].receipt["windows_by_episode"]
    assert any("_b0360_" in e for e in val_eps) and any(e.endswith("__generic_random")
                                                       for e in val_eps)
    original = build_c4_datasets(PLAIN, stride_s=20.0, horizon_steps=40, history_s=60.0,
                                 entries_v2=True)["train"]
    mixed, val = add_c4_sources(original, None, [("genrand_plain", plain, 0.30),
                                                 ("genrand_driver", driver, 0.05)])
    shares = dict(zip([s.name for s in mixed.sources], mixed.shares))
    assert shares == pytest.approx({"original": 0.65, "genrand_plain": 0.30,
                                    "genrand_driver": 0.05})
    w = np.asarray(mixed.stage_weights(10.0))
    for k, src in enumerate(mixed.sources):
        lo = mixed.offsets[k]
        assert w[lo:lo + len(src.dataset)].sum() == pytest.approx(shares[src.name])
    assert [s.name for s in val.sources] == ["genrand_driver_val"]
    rec = mixed[len(mixed) - 1]
    assert rec["action_timeline"]["command_known"].all()     # DATAHYG strict on C4 windows
    batch = collate([mixed[0], mixed[len(original)], mixed[len(mixed) - 1]])
    assert batch["states"].shape[1] == 41 and "pair_ref" not in batch


@needs_smoke
def test_loader_defaults_unchanged_by_new_options():
    from wms.data.c4_episodes import build_c4_datasets

    a = build_c4_datasets(PLAIN, horizon_steps=40, history_s=60.0, entries_v2=True)["train"]
    b = build_c4_datasets(PLAIN, horizon_steps=40, history_s=60.0, entries_v2=True,
                          junction_fix="off", command_known=False)["train"]
    for i in (0, len(a) - 1):
        ra, rb = a[i], b[i]
        assert set(ra) == set(rb) and "command_known" not in ra["action_timeline"]
        for key in ("states", "present", "loss_valid", "cav_teacher", "road_present"):
            assert np.array_equal(ra[key], rb[key]), key


@needs_smoke
def test_paired_batch_identical_start_and_differing_cav_teachers():
    from wms.data.c4_pairs import PairedBatchDataset, PairedBranchSource

    src = PairedBranchSource(BRANCH, horizon_steps=240, history_s=60.0, entries_v2=True,
                             junction_fix="ffill", command_known=True, name="genrand_branch")
    assert src.horizon == 120                      # capped at the 60 s branch
    assert src.receipt["n_groups"] == 1 and src.receipt["groups_with_native_reference"] == 1
    inner = PairedBranchSource(BRANCH, horizon_steps=240, history_s=60.0, entries_v2=True,
                               name="x").dataset
    wrapped = PairedBatchDataset(inner, src, fraction=1.0, groups=2, max_arms=5, batch_size=8)
    assert len(wrapped) == len(inner) and wrapped.receipt["paired_batches"]["fraction"] == 1.0
    assert getattr(wrapped, "stage_weights", None) is None
    records = wrapped.__getitems__([0, 1, 2, 0, 1, 2, 0, 1])
    assert len(records) == 3                        # native reference + 2 seed arms
    assert "_native" in records[0]["meta"]["arm"]
    assert len({r["meta"]["pair_key"] for r in records}) == 1
    batch = collate(records)
    assert batch["pair_ref"].tolist() == [0, 0, 0] and batch["pair_group"].tolist() == [0, 0, 0]
    for key in ("hist", "hist_mask", "driver_hist", "driver_hist_mask", "entry_feat"):
        assert all(torch.equal(batch[key][0], batch[key][i]) for i in (1, 2)), key
    assert all(torch.equal(batch["states"][0, 0], batch["states"][i, 0]) for i in (1, 2))
    assert all(torch.equal(batch["present"][0, 0], batch["present"][i, 0]) for i in (1, 2))
    cav = batch["is_cav"][0, 0] & batch["present"][0, 0]
    assert int(cav.sum()) > 0
    teacher = batch["cav_teacher"]
    assert torch.equal(teacher[0, 0], teacher[1, 0])
    for i in (1, 2):     # realized CAV motion differs after t0 (the arms' only difference)
        assert float((teacher[0, 1:, cav, 0] - teacher[i, 1:, cav, 0]).abs().max()) > 1.0
    ctrl = batch["action_timeline"]["controller"]
    assert int((ctrl[0] == 3).sum()) == 0 and int((ctrl[1] == 3).sum()) > 0
    # the trainer's DataLoader path (sampler + batch_size) reaches __getitems__
    from torch.utils.data import DataLoader, WeightedRandomSampler

    loader = DataLoader(wrapped, batch_size=8, collate_fn=collate, num_workers=0,
                        sampler=WeightedRandomSampler([1.0] * len(wrapped), 8, replacement=True))
    assert "pair_ref" in next(iter(loader))
    # fraction 0 -> ordinary batches, untouched
    plain = PairedBatchDataset(inner, src, fraction=0.0, groups=2, max_arms=5, batch_size=8)
    assert "pair_ref" not in collate(plain.__getitems__([0, 1]))


@needs_smoke
def test_paired_batches_are_reproducible_and_respect_fraction():
    from wms.data.c4_pairs import PairedBatchDataset, PairedBranchSource

    src = PairedBranchSource(BRANCH, horizon_steps=120, name="b")
    plain = PairedBranchSource(BRANCH, horizon_steps=120, name="p").dataset
    wrapped = PairedBatchDataset(plain, src, fraction=0.25, groups=1, max_arms=3, batch_size=4)
    picks = [len(wrapped.__getitems__([i % 3, (i * 7) % 3, i % 2])) for i in range(200)]
    again = [len(wrapped.__getitems__([i % 3, (i * 7) % 3, i % 2])) for i in range(200)]
    assert picks == again
    rng = np.random.default_rng(1)
    draws = [rng.integers(0, 3, 4).tolist() for _ in range(400)]
    paired = sum("pair_ref" in wrapped.__getitems__(d)[0] for d in draws)
    assert 0.15 < paired / len(draws) < 0.35
