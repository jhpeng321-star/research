"""CPU contracts for v6 paired returns, replay, and identity-keyed randomness."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.train.losses.aggregate import SPEED_LIMIT_MPS
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.paired_diff import (
    PairedDiffConfig,
    PairedDiffLoss,
    PairedDiffTerm,
    pair_targets,
    paired_diff_terms,
)
from wms.train.paired_random import PairedRandom
from wms.train.settings import TrainSettings


ROOT = Path(__file__).resolve().parents[2]
V6_DEFAULTS = {
    "couple_randomness": False,
    "return_weight": 0.0,
    "rank_weight": 0.0,
    "rank_margin_veh_s": 5.0,
}


def _runner_batch(duplicate=True):
    b, t, k = 4, 9, 4
    states = torch.zeros(b, t, k, 7)
    states[..., 0] = (torch.tensor([-600., -500., -400., -300.])[None, None]
                      + torch.arange(t)[None, :, None] * 7.5)
    states[..., 2] = 1.0
    states[..., 4] = 15.0
    states[..., 6] = 5.0
    if not duplicate:
        states[1, 1:, 1:, 4] = 5.0
        states[3, 1:, 1:, 4] = 25.0
    present = torch.ones(b, t, k, dtype=torch.bool)
    cav = torch.zeros_like(present)
    cav[..., 0] = True
    hist = torch.zeros(b, k, 10, 6)
    hist[..., 0] = 40.0
    hist[..., 4] = 15.0
    teacher = torch.stack((states[..., 0], states[..., 1:4].argmax(-1),
                           states[..., 4], states[..., 5]), -1)
    return dict(states=states, present=present, is_cav=cav, hist=hist,
                hist_mask=torch.ones(b, k, 10, dtype=torch.bool), cav_teacher=teacher,
                vehicle_ids=torch.arange(k).expand(b, k), pair_ref=torch.tensor([0, 0, 2, 2]),
                pair_group=torch.tensor([7, 7, 9, 9]),
                pair_world=["world-A"] * 4)


def _runner(settings, mode="hybrid"):
    from wms.model.residual_net import HdvConfig
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.train.losses.compose import build_loss
    from wms.train.unroll import UnrollRunner

    torch.manual_seed(17)
    model = WmSModel(ModelSpec(rollout_mode=mode, history_hidden=16, k_hidden=8,
                              lane_model="hazard_v5", lane_decision="sample",
                              driver_update="ema"),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32))
    return UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                        projection=None, device=torch.device("cpu"))


@pytest.mark.parametrize("mode", ["hybrid", "driver_params"])
@pytest.mark.parametrize("entries", [False, True])
def test_duplicate_arms_have_identical_predictions_and_exact_zero_loss(mode, entries):
    from wms.model.entries import EntranceHistory

    settings = TrainSettings(rollout_mode=mode, paired_diff_weight=1.0,
                             paired_diff_couple_randomness=True,
                             paired_diff_return_weight=1.0, paired_diff_rank_weight=0.1,
                             driver_grad_across_chunks=True, driver_reveal_prob=0.7,
                             driver_reveal_max_s=3.0, state_noise_std_m=0.2,
                             state_noise_std_mps=0.1, truncation_steps=3,
                             entries_v2=entries, entry_pad=3)
    runner = _runner(settings, mode)
    batch = _runner_batch()
    if entries:
        batch["entry_feat"] = torch.as_tensor(np.stack(
            [EntranceHistory.defaults().to_features()] * 4)).float()
    original = runner.model.rollout
    calls = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        for key in ("positions", "speeds", "lanes", "masks"):
            assert torch.equal(result[key][0], result[key][1]), key
            assert torch.equal(result[key][2], result[key][3]), key
        assert torch.equal(kwargs["observe_mask"][0], kwargs["observe_mask"][1])
        assert torch.equal(kwargs["lane_uniform"][0], kwargs["lane_uniform"][1])
        assert not torch.equal(kwargs["lane_uniform"][0], kwargs["lane_uniform"][2])
        calls.append({k: result[k].detach().clone() for k in ("positions", "speeds", "lanes")})
        return result

    runner.model.rollout = capture
    record = runner.run(batch, unroll_steps=8)
    assert len(calls) == 6  # three no-grad probe chunks, three exact training replays
    for first, replay in zip(calls[:3], calls[3:]):
        assert all(torch.equal(first[k], replay[k]) for k in first)
    for name in ("pair_pos", "pair_speed", "pair_agg", "pair_return", "pair_rank"):
        assert record[name] == 0.0, name
    assert record["paired_grad_norm"] == 0.0
    assert record["paired_unique_worlds"] == 1 and record["paired_groups"] == 2
    assert record["paired_effect_bins"] == {"negative": 0, "within_margin": 2, "positive": 0}


def test_paired_gradient_norm_includes_encoder_and_logs_whole_return_once():
    settings = TrainSettings(rollout_mode="hybrid", paired_diff_weight=1.0,
                             paired_diff_couple_randomness=True,
                             paired_diff_return_weight=1.0, paired_diff_rank_weight=0.1,
                             paired_diff_rank_margin_veh_s=0.1,
                             driver_grad_across_chunks=True, truncation_steps=3,
                             position_weight=0, speed_weight=0, lane_weight=0,
                             residual_magnitude_weight=0)
    runner = _runner(settings)
    record = runner.run(_runner_batch(False), unroll_steps=8)
    norm = sum(float(p.grad.double().square().sum()) for p in runner.model.parameters()
               if p.grad is not None) ** 0.5
    assert norm > 0
    assert record["paired_grad_norm"] == pytest.approx(norm, rel=1e-5)
    assert record["loss"] == pytest.approx(sum(record[n] for n in (
        "pair_pos", "pair_speed", "pair_agg", "pair_return", "pair_rank")), abs=1e-6)
    assert record["paired_effect_bins"] == {"negative": 1, "within_margin": 0, "positive": 1}


def test_new_defaults_keep_training_records_gradients_and_rng_identical():
    base = TrainSettings(rollout_mode="hybrid", paired_diff_weight=1.0,
                         driver_grad_across_chunks=True, state_noise_std_m=0.2,
                         state_noise_std_mps=0.1, driver_reveal_prob=0.4, truncation_steps=3)

    def run(settings):
        runner = _runner(settings)
        record = runner.run(_runner_batch(False), unroll_steps=8)
        grads = {n: p.grad.clone() for n, p in runner.model.named_parameters()
                 if p.grad is not None}
        return record, grads, torch.get_rng_state()

    a, ga, ra = run(base)
    b, gb, rb = run(replace(base, **{"paired_diff_" + k: v for k, v in V6_DEFAULTS.items()}))
    assert a == b and ga.keys() == gb.keys() and torch.equal(ra, rb)
    assert all(torch.equal(ga[k], gb[k]) for k in ga)


def test_return_evaluation_chunking_and_replay_do_not_change_rng():
    settings = TrainSettings(rollout_mode="hybrid", paired_diff_weight=1.0,
                             paired_diff_return_weight=1.0, paired_diff_rank_weight=0.1,
                             driver_grad_across_chunks=True, truncation_steps=3)
    runner = _runner(settings)
    batch = _runner_batch(False)
    rng = torch.get_rng_state()
    with torch.no_grad():
        evaluation = runner.run(batch, unroll_steps=8, backward=False)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    # Gradient-enabled forward runs the replay; no update is requested.
    replay = runner.run(batch, unroll_steps=8, backward=False)
    assert torch.equal(torch.get_rng_state(), after)
    for key in ("pair_return", "pair_rank", "loss"):
        assert replay[key] == pytest.approx(evaluation[key], abs=1e-6)
    assert replay["paired_grad_norm"] is None


def test_paired_collation_counts_worlds_not_groups():
    from wms.data.collate import collate

    records = [dict(states=np.zeros((2, 1, 7)), present=np.ones((2, 1), bool),
                    slot_index=np.zeros(1), vehicle_ids=np.array([5]), pair_group=g,
                    pair_ref=r, meta={"world": w, "pair_key": key})
               for g, r, w, key in [(0, 0, "a", "a@0"), (0, 0, "a", "a@0"),
                                    (1, 2, "a", "a@5"), (1, 2, "a", "a@5")]]
    batch = collate(records)
    assert len(set(batch["pair_world"])) == 1
    assert len(set(batch["pair_key"])) == 2


def test_entry_draws_stay_coupled_after_one_arm_defers(monkeypatch):
    from wms.model.entries import EntranceHistory, TorchEntryProcessV2

    process = TorchEntryProcessV2(history=EntranceHistory.defaults(), batch=2,
                                  row_seeds=[123, 123], couple_draws=True)
    calls = [[], []]
    blocked = True

    def try_lane(work, b, e, lane, cav, speed, gap, vtype):
        calls[b].append((cav, speed, gap, vtype))
        return not (b == 1 and blocked)

    monkeypatch.setattr(process, "_try_lane", try_lane)
    # Ramp has only one lane attempt. Arm 0 inserts three candidates; arm 1 defers.
    process._insert_entrance(None, 0, 1, 3)
    process._insert_entrance(None, 1, 1, 3)
    assert process.pending[1, 1] == 3
    assert calls[1] == calls[0][:1]
    # Insertion retries cannot perturb subsequent renewal times.
    assert [process._headway(0, 1) for _ in range(5)] == [
        process._headway(1, 1) for _ in range(5)]
    blocked = False
    process._insert_entrance(None, 1, 1, 0)
    assert calls[1][1:] == calls[0]  # same identities/attributes when eventually admitted


def test_fit_persists_paired_diagnostics_in_train_log(tmp_path):
    import json

    from wms.train.trainer import WmSTrainer

    cfg = TrainConfig.from_dict({
        "runtime": {"device": "cpu", "threads": 2, "omp": 2},
        "model": {"rollout_mode": "hybrid"},
        "curriculum": {"unroll_s": [4.0], "stage_steps": [1],
                       "truncation_steps": 3, "driver_grad_across_chunks": True},
        "optim": {"batch_size": 4},
        "loss": {"paired_diff": {"weight": 1.0, "couple_randomness": True,
                                  "return_weight": 1.0, "rank_weight": 0.1,
                                  "rank_margin_veh_s": 0.1}}})
    settings = TrainSettings.from_config(cfg)
    batch = _runner_batch(False)

    class Pairs(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            rec = {k: v[index].numpy() for k, v in batch.items() if torch.is_tensor(v)}
            rec["slot_index"] = np.arange(4)
            rec["meta"] = {"world": "world-A", "pair_key": f"world-A@{index // 2}"}
            return rec

        def __getitems__(self, indices):
            return [self[i] for i in range(4)]

    trainer = WmSTrainer(cfg, _runner(settings).model, Pairs(), out_dir=tmp_path)
    trainer.fit()
    row = json.loads((tmp_path / "train_log.jsonl").read_text().splitlines()[0])
    assert row["paired_batch"] and row["paired_unique_worlds"] == 1
    assert row["paired_groups"] == 2 and row["paired_grad_norm"] > 0
    assert sum(row["paired_effect_bins"].values()) == 2
    assert row["pair_return"] > 0 and row["pair_rank"] > 0


def test_v6_defaults_are_omitted_from_legacy_flat_and_restored():
    cfg = TrainConfig()
    flat = cfg.to_legacy_flat()
    restored = TrainConfig.from_legacy_flat(flat)
    settings = TrainSettings.from_config(restored)
    for name, default in V6_DEFAULTS.items():
        assert getattr(cfg.loss.paired_diff, name) == default
        assert "paired_diff_" + name not in flat
        assert getattr(restored.loss.paired_diff, name) == default
        assert getattr(settings, "paired_diff_" + name) == default
    assert PairedDiffLoss(PairedDiffConfig()).return_names == ()
    assert [term.name for term in paired_diff_terms(PairedDiffConfig())] == [
        "pair_pos", "pair_speed", "pair_agg"
    ]


@pytest.mark.parametrize("name,value", [
    ("couple_randomness", True), ("return_weight", 2.5),
    ("rank_weight", 0.3), ("rank_margin_veh_s", 7.0),
])
def test_v6_nondefault_flat_fields_round_trip_independently(name, value):
    cfg = TrainConfig.from_dict({"loss": {"paired_diff": {name: value}}})
    flat = cfg.to_legacy_flat()
    assert flat["paired_diff_" + name] == value
    assert all("paired_diff_" + other not in flat for other in V6_DEFAULTS if other != name)
    restored = TrainConfig.from_legacy_flat(flat)
    assert restored.loss.paired_diff == cfg.loss.paired_diff
    assert getattr(TrainSettings.from_config(restored), "paired_diff_" + name) == value


def test_v6_preset_inherits_v5_and_enables_return_terms():
    base = load_train_config(ROOT / "configs/c4_wm_v5.toml")
    cfg = load_train_config(ROOT / "configs/c4_wm_v6_pair.toml")
    pair = cfg.loss.paired_diff
    assert pair.couple_randomness is True
    assert (pair.return_weight, pair.rank_weight, pair.rank_margin_veh_s) == (1.0, 0.1, 5.0)
    assert (pair.weight, pair.fraction) == (1.0, 0.25)
    assert replace(pair, **V6_DEFAULTS) == base.loss.paired_diff
    assert replace(cfg.loss, paired_diff=base.loss.paired_diff) == base.loss
    for section in ("data", "model", "curriculum", "optim", "selection"):
        assert getattr(cfg, section) == getattr(base, section)
    assert cfg.model.lane_model == "hazard_v5"
    assert cfg.model.lane_decision == "gated"
    assert cfg.runtime.device == "cpu"
    assert (cfg.runtime.threads, cfg.runtime.omp) == (2, 2)
    assert cfg.runtime.out_dir == "runs/active/c4_wm_v6_pair"
    assert {"pair_return", "pair_rank"} <= set(cfg.required_log_terms())
    assert not {"pair_return", "pair_rank"} & set(base.required_log_terms())
    settings = TrainSettings.from_config(cfg)
    for name in V6_DEFAULTS:
        assert getattr(settings, "paired_diff_" + name) == getattr(pair, name)
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).loss.paired_diff == pair


def _v6_loss(**overrides):
    options = dict(dt_s=1.0, position_weight=0.0, speed_weight=0.0,
                   aggregate_weight=0.0, return_weight=1.0, rank_weight=1.0,
                   rank_margin_veh_s=1.0)
    options.update(overrides)
    return PairedDiffLoss(PairedDiffConfig(**options))


def _v6_targets(ratios, refs, *, present=None, is_cav=None, observed=None, n_k=None):
    """Synthetic recorded future speeds as fractions of the speed limit; dt = 1 s."""
    ratios = torch.as_tensor(ratios, dtype=torch.float64)
    b, n, k = ratios.shape
    states = torch.zeros(b, n + 1, k, 7, dtype=torch.float64)
    states[..., 2] = 1.0
    states[:, 0, :, 4] = 0.5 * SPEED_LIMIT_MPS
    states[:, 1:, :, 4] = ratios * SPEED_LIMIT_MPS
    if present is None:
        present = torch.ones(b, n + 1, k, dtype=torch.bool)
    if is_cav is None:
        is_cav = torch.zeros_like(present)
    if observed is None:
        observed = torch.ones(b, n + 1, dtype=torch.bool)
    hdv = present[:, 1:] & present[:, :1] & ~is_cav[:, :1]
    targets = UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1, dtype=torch.float64) / n,
        hdv_mask=hdv, aux_mask=hdv, aux_count=int(hdv.sum()),
        denom=torch.tensor(1.0, dtype=torch.float64),
        lane_target=torch.ones(b, n, k, dtype=torch.long), loss_valid=present,
        is_cav=is_cav, dt_s=1.0, position_delta_m=5.0, speed_delta_mps=1.0,
        history_lane_head=False,
    )
    targets.pair = pair_targets(
        {"pair_ref": torch.as_tensor(refs), "road_valid": observed},
        states, present, k if n_k is None else n_k, device="cpu",
    )
    return targets


def _v6_context(targets, speeds, *, c0=0, masks=None):
    """The supplied speeds are independent predictions for this chunk only."""
    c1 = c0 + speeds.shape[1]
    if masks is None:
        masks = targets.loss_valid[:, c0 + 1:c1 + 1]
    return LossContext(
        pred={"speeds": speeds, "positions": torch.zeros_like(speeds),
              "lanes": torch.ones_like(speeds, dtype=torch.long), "masks": masks},
        current={}, targets=targets, c0=c0, c1=c1,
    )


@pytest.mark.parametrize("term", ["pair_return", "pair_rank"])
def test_return_and_rank_each_have_analytic_gradients_in_both_arms(term):
    recorded = torch.tensor([0.25, 0.75], dtype=torch.float64)[:, None, None].expand(2, 4, 2)
    targets = _v6_targets(recorded, [0, 0])
    loss = _v6_loss(weight=2.0, return_weight=3.0, rank_weight=0.5)
    loss.prepare(targets)
    speeds = (torch.tensor([0.6, 0.4], dtype=torch.float64)[:, None, None]
              .expand(2, 4, 2) * SPEED_LIMIT_MPS).clone().requires_grad_()
    ctx = _v6_context(targets, speeds)
    torch.testing.assert_close(loss.recorded_return_difference(targets),
                               torch.tensor([0.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(loss.predicted_return_difference(ctx),
                               torch.tensor([0.0, -1.6], dtype=torch.float64))
    value = loss.components(ctx)[term]
    # Common scale = 2 HDVs * 4 s = 8 veh-s; normalized return error = -0.7.
    expected_value = 6.0 * 0.5 * 0.7 ** 2 if term == "pair_return" else 2.6 / 8.0
    expected_arm_grad = -6.0 * 0.7 / (8.0 * SPEED_LIMIT_MPS) if term == "pair_return" \
        else -1.0 / (8.0 * SPEED_LIMIT_MPS)
    assert value.item() == pytest.approx(expected_value)
    grad, = torch.autograd.grad(value, speeds)
    torch.testing.assert_close(grad[1], torch.full_like(grad[1], expected_arm_grad))
    torch.testing.assert_close(grad[0], -grad[1])
    assert torch.all(grad[0] > 0) and torch.all(grad[1] < 0)


@pytest.mark.parametrize("effect", [-1.001, -1.0, -0.5, 0.0, 0.5, 1.0, 1.001])
def test_rank_strict_margin_ties_and_submargin_effects_have_exact_zero_gradient(effect):
    recorded = torch.full((2, 4, 1), 0.5, dtype=torch.float64)
    # Endpoints 0 and speed_limit make the exact-margin cases exactly representable.
    # Multiplying 0.75 by the decimal speed limit can otherwise move a tie by one ULP.
    slow, fast = (0, 1) if effect >= 0 else (1, 0)
    recorded[slow, :2] = 0.0
    recorded[fast, 0] = min(abs(effect), 1.0)
    recorded[fast, 1] = max(abs(effect) - 1.0, 0.0)
    targets = _v6_targets(recorded, [0, 0])
    loss = _v6_loss()
    loss.prepare(targets)
    speeds = torch.full((2, 4, 1), 0.5 * SPEED_LIMIT_MPS,
                        dtype=torch.float64, requires_grad=True)
    value = loss.components(_v6_context(targets, speeds))["pair_rank"]
    grad, = torch.autograd.grad(value, speeds)
    bins = loss.recorded_effect_bins(targets)
    active = abs(effect) > 1.0
    assert bins["rank_active"].tolist() == [False, active]
    assert bins["effect_bin"].tolist() == [0, (1 if effect > 0 else -1) if active else 0]
    if active:
        assert value.item() == pytest.approx(0.25)
        assert torch.all(grad[1] * effect < 0)
        torch.testing.assert_close(grad[0], -grad[1])
    else:
        assert value.item() == 0.0
        assert torch.count_nonzero(grad).item() == 0


def test_loss_row_permutation_remaps_shared_and_nonzero_references():
    refs = torch.tensor([0, 0, 0, 3, 3])
    ratios = torch.tensor([0.2, 0.7, 0.45, 0.8, 0.3], dtype=torch.float64)
    recorded = ratios[:, None, None].expand(5, 4, 2).clone()
    prediction = torch.tensor([0.55, 0.4, 0.2, 0.3, 0.65], dtype=torch.float64)
    speeds = (prediction[:, None, None].expand_as(recorded) * SPEED_LIMIT_MPS).clone()
    order = torch.tensor([4, 2, 0, 1, 3])
    inverse = torch.argsort(order)
    remapped = inverse[refs[order]]
    assert remapped.tolist() == [4, 2, 2, 2, 4]
    loss = _v6_loss(rank_margin_veh_s=0.5)

    def evaluate(records, mapping, predictions):
        targets = _v6_targets(records, mapping)
        loss.prepare(targets)
        leaf = predictions.clone().requires_grad_()
        ctx = _v6_context(targets, leaf)
        values = loss.components(ctx)
        gradients = {name: torch.autograd.grad(values[name], leaf, retain_graph=True)[0]
                     for name in loss.return_names}
        return values, gradients, loss.recorded_effect_bins(targets), \
            loss.predicted_return_difference(ctx)

    original, grad, bins, difference = evaluate(recorded, refs, speeds)
    permuted, perm_grad, perm_bins, perm_difference = evaluate(
        recorded[order], remapped, speeds[order])
    for name in loss.return_names:
        torch.testing.assert_close(permuted[name], original[name])
        torch.testing.assert_close(perm_grad[name], grad[name][order])
    for name in bins:
        torch.testing.assert_close(perm_bins[name], bins[name][order])
    torch.testing.assert_close(perm_difference, difference[order])


def test_common_cohort_normalization_and_swap_symmetry_with_unequal_occupancy():
    recorded = torch.full((2, 4, 2), 0.5, dtype=torch.float64)
    present = torch.ones(2, 5, 2, dtype=torch.bool)
    present[1, 2:, 1] = False
    present[1, 4] = False  # observed empty road still counts toward the common horizon
    observed = torch.ones(2, 5, dtype=torch.bool)
    observed[0, 3] = False  # this tick must contribute to neither arm
    prediction = torch.tensor([0.75, 0.25], dtype=torch.float64)
    speeds = (prediction[:, None, None].expand_as(recorded) * SPEED_LIMIT_MPS).clone()
    loss = _v6_loss(rank_margin_veh_s=0.5)
    results = []
    for refs, arm, sign in (([0, 0], 1, 1), ([1, 1], 0, -1)):
        targets = _v6_targets(recorded, refs, present=present, observed=observed)
        loss.prepare(targets)
        bins = loss.recorded_effect_bins(targets)
        assert bins["hdv_count"][arm].item() == 2
        assert bins["horizon_s"][arm].item() == 3
        assert bins["scale_veh_s"][arm].item() == 6
        assert bins["difference_veh_s"][arm].item() == pytest.approx(sign * 1.5)
        leaf = speeds.clone().requires_grad_()
        ctx = _v6_context(targets, leaf)
        assert loss.predicted_return_difference(ctx)[arm].item() == pytest.approx(sign * -0.75)
        values = loss.components(ctx)
        assert values["pair_return"].item() == pytest.approx(0.5 * (2.25 / 6.0) ** 2)
        assert values["pair_rank"].item() == pytest.approx(1.25 / 6.0)
        gradients = {name: torch.autograd.grad(values[name], leaf, retain_graph=True)[0]
                     for name in loss.return_names}
        for grad in gradients.values():
            assert torch.count_nonzero(grad[:, 2]).item() == 0
            assert torch.count_nonzero(grad[~present[:, 1:]]).item() == 0
            assert grad[0, 0].abs().sum() > 0 and grad[1, 0].abs().sum() > 0
        results.append((values, gradients))
    for name in loss.return_names:
        torch.testing.assert_close(results[0][0][name], results[1][0][name])
        torch.testing.assert_close(results[0][1][name], results[1][1][name])


def test_return_cohort_excludes_cavs_late_vehicles_and_appended_entry_slots():
    recorded = torch.full((2, 4, 4), 0.5, dtype=torch.float64)
    recorded[0, :, 0], recorded[1, :, 0] = 0.25, 0.75
    present = torch.ones(2, 5, 4, dtype=torch.bool)
    present[:, 0, 2:] = False
    cav = torch.zeros_like(present)
    cav[:, :, 1] = True
    loss = _v6_loss()
    results = []
    for perturb in (False, True):
        rec = recorded.clone()
        pred = torch.full_like(rec, 0.5)
        if perturb:
            rec[0, :, 1:], rec[1, :, 1:] = -4.0, 3.0
            pred[0, :, 1:], pred[1, :, 1:] = 2.0, -5.0
        targets = _v6_targets(rec, [0, 0], present=present, is_cav=cav, n_k=3)
        loss.prepare(targets)
        bins = loss.recorded_effect_bins(targets)
        assert bins["hdv_count"].tolist() == [1, 1]
        assert bins["scale_veh_s"][1].item() == 4
        assert loss.recorded_return_difference(targets)[1].item() == pytest.approx(2.0)
        speeds = (pred * SPEED_LIMIT_MPS).requires_grad_()
        ctx = _v6_context(targets, speeds)
        assert loss.predicted_return_difference(ctx).tolist() == [0.0, 0.0]
        values = loss.components(ctx)
        assert values["pair_return"].item() == pytest.approx(0.125)
        assert values["pair_rank"].item() == pytest.approx(0.25)
        gradients = {name: torch.autograd.grad(values[name], speeds, retain_graph=True)[0]
                     for name in loss.return_names}
        for grad in gradients.values():
            assert torch.count_nonzero(grad[..., 1:]).item() == 0
            assert torch.all(grad[0, :, 0] > 0) and torch.all(grad[1, :, 0] < 0)
        results.append((values, gradients))
    for name in loss.return_names:
        torch.testing.assert_close(results[0][0][name], results[1][0][name])
        torch.testing.assert_close(results[0][1][name], results[1][1][name])


def test_full_cumulative_errors_cancel_across_chunks_before_huber():
    targets = _v6_targets(torch.full((2, 4, 1), 0.5), [0, 0])
    loss = _v6_loss()
    loss.prepare(targets)
    speeds = torch.full((2, 4, 1), 0.5, dtype=torch.float64)
    speeds[1, :, 0] = torch.tensor([1.0, 0.5, 0.0, 0.5])
    speeds *= SPEED_LIMIT_MPS
    pieces = []
    with torch.no_grad():
        for lo, hi in ((0, 2), (2, 4)):
            ctx = _v6_context(targets, speeds[:, lo:hi], c0=lo)
            difference = loss.predicted_return_difference(ctx)
            assert loss.predicted_return_difference(ctx) is difference
            assert all(loss.components(ctx)[name].item() == 0 for name in loss.return_names)
            pieces.append(difference)
        assert pieces[0][1].item() == pytest.approx(0.5)
        assert pieces[1][1].item() == pytest.approx(-0.5)
        whole = sum(pieces)
        assert all(value.item() == 0 for value in loss.return_losses(targets, whole).values())
        torch.testing.assert_close(
            sum(loss.recorded_return_difference(targets, lo, hi) for lo, hi in ((0, 2), (2, 4))),
            loss.recorded_return_difference(targets),
        )
    targets.prepared["paired_return_reference"] = whole.detach()
    for lo, hi in ((0, 2), (2, 4)):
        leaf = speeds[:, lo:hi].clone().requires_grad_()
        values = loss.components(_v6_context(targets, leaf, c0=lo))
        grad, = torch.autograd.grad(sum(values[name] for name in loss.return_names), leaf)
        assert all(values[name].item() == 0 for name in loss.return_names)
        assert torch.count_nonzero(grad).item() == 0


@pytest.mark.parametrize("term", ["pair_return", "pair_rank"])
@pytest.mark.parametrize("bounds", [(0, 1, 4, 6), (0, 2, 6)])
def test_replay_gradients_equal_full_objective_on_independent_prediction_tensors(term, bounds):
    recorded = torch.tensor([0.2, 0.8, 0.7, 0.3], dtype=torch.float64)
    recorded = recorded[:, None, None].expand(4, 6, 2).clone()
    targets = _v6_targets(recorded, [0, 0, 2, 2])
    loss = _v6_loss(weight=1.7, return_weight=0.8, rank_weight=0.3)
    loss.prepare(targets)
    ratios = torch.tensor([0.6, 0.4, 0.3, 0.55], dtype=torch.float64)[:, None, None]
    ratios = ratios.expand_as(recorded) + torch.arange(12, dtype=torch.float64).reshape(1, 6, 2) / 200
    full = (ratios * SPEED_LIMIT_MPS).clone().requires_grad_()
    full_value = loss.components(_v6_context(targets, full))[term]
    full_grad, = torch.autograd.grad(full_value, full)
    assert full_value.item() > 0
    assert all(row.abs().sum() > 0 for row in full_grad)
    chunks = list(zip(bounds[:-1], bounds[1:]))
    with torch.no_grad():
        probe = sum(loss.predicted_return_difference(
            _v6_context(targets, full[:, lo:hi].detach().clone(), c0=lo)) for lo, hi in chunks)
        torch.testing.assert_close(loss.return_losses(targets, probe)[term], full_value.detach())
        torch.testing.assert_close(
            sum(loss.recorded_return_difference(targets, lo, hi) for lo, hi in chunks),
            loss.recorded_return_difference(targets),
        )
    targets.prepared["paired_return_reference"] = probe.detach()
    wrapper = PairedDiffTerm(term, loss, owner=False)
    replay_grads, logs = [], []
    for lo, hi in chunks:
        # Each chunk is a new leaf: no shared rollout graph can hide incorrect replay weights.
        leaf = full[:, lo:hi].detach().clone().requires_grad_()
        assert leaf.is_leaf and leaf.data_ptr() != full.data_ptr()
        result = wrapper(_v6_context(targets, leaf, c0=lo))
        torch.testing.assert_close(result.value, full_value.detach())
        assert not result.logged.requires_grad
        if lo:
            assert result.logged.item() == 0
        grad, = torch.autograd.grad(result.value, leaf)
        replay_grads.append(grad)
        logs.append(result.logged)
    torch.testing.assert_close(torch.cat(replay_grads, dim=1), full_grad, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(sum(logs), full_value.detach())


def test_gradient_partial_chunk_requires_detached_whole_horizon_reference():
    targets = _v6_targets(torch.full((2, 4, 1), 0.5), [0, 0])
    loss = _v6_loss()
    loss.prepare(targets)
    speeds = torch.ones(2, 2, 1, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="whole-horizon probe"):
        loss.components(_v6_context(targets, speeds))
    for invalid in (torch.zeros(2, requires_grad=True), torch.zeros(2, 1)):
        targets.prepared["paired_return_reference"] = invalid
        with pytest.raises(ValueError, match="detached whole-horizon"):
            loss.components(_v6_context(targets, speeds))


def _v6_random_batch(*, episode_encoded=False):
    ids = torch.tensor([[0, 17, 53, -1]] * 4)
    if episode_encoded:
        prefixes = torch.tensor([11, 12, 21, 22])[:, None] * 1_000_000
        ids = torch.where(ids >= 0, ids + prefixes, ids)
    present = torch.ones(4, 9, 4, dtype=torch.bool)
    present[..., -1] = False
    return {"vehicle_ids": ids, "present": present, "pair_ref": torch.tensor([0, 0, 2, 2]),
            "pair_group": ["world-a", "world-a", "world-b", "world-b"]}


def _v6_random(batch, *, seed=123, dtype=torch.float64, pad=2):
    return PairedRandom.from_batch(batch, seed=seed, device=torch.device("cpu"), dtype=dtype, pad=pad)


@pytest.mark.parametrize("method", ["uniform", "normal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_paired_random_identity_streams_match_arms_and_separate_groups(method, dtype):
    batch = _v6_random_batch()
    random = _v6_random(batch, dtype=dtype)
    repeat = _v6_random(batch, dtype=dtype)
    other_seed = _v6_random(batch, seed=124, dtype=dtype)
    previous = None
    for stream in ("lane", "state_s", "state_v", "reveal_pick", "reveal_length", "t0_s", "t0_v"):
        draw = getattr(random, method)(stream, range(8))
        assert draw.shape == (4, 8, 6) and draw.dtype == dtype and draw.device.type == "cpu"
        assert torch.isfinite(draw).all()
        if method == "uniform":
            assert torch.all((draw >= 0) & (draw < 1))
        assert torch.equal(draw[0], draw[1])
        assert torch.equal(draw[2], draw[3])
        assert not torch.equal(draw[0], draw[2])
        assert torch.equal(draw, getattr(repeat, method)(stream, range(8)))
        assert not torch.equal(draw, getattr(other_seed, method)(stream, range(8)))
        if previous is not None:
            assert not torch.equal(draw, previous)
        previous = draw
        # Physical ID zero and synthetic entry ordinal zero are separate identities.
        assert not torch.equal(draw[..., 0], draw[..., 4])
        assert not torch.equal(draw[..., 4], draw[..., 5])
        for lo in (0, 2):
            isolated = {"vehicle_ids": batch["vehicle_ids"][lo:lo + 2],
                        "present": batch["present"][lo:lo + 2], "pair_ref": torch.tensor([0, 0]),
                        "pair_group": batch["pair_group"][lo:lo + 2]}
            assert torch.equal(getattr(_v6_random(isolated, dtype=dtype), method)(stream, range(8)),
                               draw[lo:lo + 2])
    assert random.row_seeds[0] == random.row_seeds[1]
    assert random.row_seeds[2] == random.row_seeds[3]
    assert random.row_seeds[0] != random.row_seeds[2]


@pytest.mark.parametrize("method", ["uniform", "normal"])
def test_paired_random_slot_row_time_and_chunk_permutations_are_exact(method):
    batch = _v6_random_batch(episode_encoded=True)
    random = _v6_random(batch)
    draw = getattr(random, method)("state_s", range(8))
    slots = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2], [3, 2, 1, 0], [1, 0, 2, 3]])
    rows = torch.tensor([3, 0, 2, 1])
    remapped = torch.argsort(rows)[batch["pair_ref"][rows]]
    permuted = {
        "vehicle_ids": batch["vehicle_ids"].gather(1, slots)[rows],
        "present": batch["present"].gather(2, slots[:, None].expand(-1, 9, -1))[rows],
        "pair_ref": remapped, "pair_group": [batch["pair_group"][i] for i in rows.tolist()],
    }
    shuffled = _v6_random(permuted)
    expected = torch.cat((draw[..., :4].gather(2, slots[:, None].expand(-1, 8, -1)),
                          draw[..., 4:]), dim=-1)[rows]
    assert torch.equal(getattr(shuffled, method)("state_s", range(8)), expected)
    assert shuffled.row_seeds == [random.row_seeds[i] for i in rows.tolist()]
    times = torch.tensor([7, 0, 4, 4, 2])
    assert torch.equal(getattr(shuffled, method)("state_s", times), expected[:, times])
    assert torch.equal(getattr(random, method)("state_s", 4), draw[:, 4:5])
    # Requests deliberately arrive out of time order and use another stream in between.
    tail = getattr(random, method)("state_s", range(5, 8))
    getattr(random, method)("unrelated", range(20))
    head = getattr(random, method)("state_s", range(2))
    middle = getattr(random, method)("state_s", range(2, 5))
    assert torch.equal(torch.cat((head, middle, tail), dim=1), draw)


@pytest.mark.parametrize("method", ["uniform", "normal"])
def test_episode_prefixed_ids_decode_to_same_physical_identity_with_explicit_opt_out(method):
    plain = _v6_random(_v6_random_batch())
    encoded_batch = _v6_random_batch(episode_encoded=True)
    encoded = _v6_random(encoded_batch)
    expected = getattr(plain, method)("lane", range(8))
    assert torch.equal(getattr(encoded, method)("lane", range(8)), expected)
    encoded_batch["vehicle_ids_episode_encoded"] = True
    assert torch.equal(getattr(_v6_random(encoded_batch), method)("lane", range(8)), expected)
    encoded_batch["vehicle_ids_episode_encoded"] = False
    literal = getattr(_v6_random(encoded_batch), method)("lane", range(8))
    assert not torch.equal(literal[0, :, :3], literal[1, :, :3])
    assert not torch.equal(literal[..., :3], expected[..., :3])
    assert torch.equal(literal[..., 3:], expected[..., 3:])


def test_paired_random_does_not_consume_global_rng_state():
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    random = _v6_random(_v6_random_batch())
    random.uniform("lane", range(10))
    random.normal("state_s", range(10))
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


@pytest.mark.parametrize("fault", ["missing", "unknown_present", "unknown_later", "duplicate",
                                   "wrong_shape", "floating"])
def test_paired_random_rejects_missing_or_invalid_stable_identity(fault):
    batch = _v6_random_batch()
    if fault == "missing":
        del batch["vehicle_ids"]
    elif fault == "unknown_present":
        batch["vehicle_ids"][0, 0] = -1
    elif fault == "unknown_later":
        batch["present"][0, -1, -1] = True
    elif fault == "duplicate":
        batch["vehicle_ids"][0, 1] = batch["vehicle_ids"][0, 0]
    elif fault == "wrong_shape":
        batch["vehicle_ids"] = batch["vehicle_ids"][:, :-1]
    else:
        batch["vehicle_ids"] = batch["vehicle_ids"].double()
    with pytest.raises(ValueError, match="vehicle_ids"):
        _v6_random(batch)
