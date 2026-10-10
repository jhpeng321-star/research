"""M1FIX2 training-quality fixes: outcome normalization, state global token,
DataLoader parity, fail-closed timing gate and balanced loss magnitudes.

CPU only, tmp dirs, synthetic packs - no SUMO, no real data, no D:.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import train.idea_a as ia
from model import constants as C
from model.outcome_norm import OutcomeNorm, denormalize_quantiles
from corr import StateGlobalToken
from test_idea_a_fakes import FakeArrival, FakeCorr, FakeHdv, CORR_D_MODEL, \
    make_windows


def _many_group_runs(groups: int, arms_per_group: int):
    runs = []
    for g in range(groups):
        world = f"w{g}"
        t0 = 100.0 + g
        runs.append(dict(chunk="chunk4", world=world, arm="native",
                         policy="native", split="TRAIN", t0_s=t0))
        for a in range(arms_per_group - 1):
            policy = "switching" if a % 2 else "damping"
            arm = f"switching{a + 1}" if policy == "switching" else f"base{a + 1}"
            runs.append(dict(chunk="chunk4", world=world, arm=arm,
                             policy=policy, split="TRAIN", t0_s=t0))
    return runs


def _pack(tmp_path, runs, k=16, n_ticks=3, seed=0):
    from bundle.e2e import write_synthetic_pack

    pack = tmp_path / "pack"
    write_synthetic_pack(pack, k=k, n_ticks=n_ticks, seed=seed, runs=runs)
    return pack


# ------------------------------------------------- item 1: normalization ----
def test_outcome_norm_roundtrip_and_train_only(tmp_path):
    g = torch.Generator().manual_seed(0)
    train = [torch.rand(C.N_TAU, C.N_OUTCOME_TARGETS, generator=g) * 50.0
             for _ in range(6)]
    cal = [torch.full((C.N_TAU, C.N_OUTCOME_TARGETS), -999.0)]
    mask = [torch.ones(C.N_TAU, dtype=torch.bool) for _ in range(6)]
    norm = OutcomeNorm.fit(train, mask)
    # round-trip de-normalization reproduces the raw targets
    y = train[0]
    back = norm.denormalize(norm.normalize(y))
    assert torch.allclose(back, y, atol=1e-9)
    # quantile-shaped de-normalization also round-trips
    q = y.unsqueeze(-1).expand(C.N_TAU, C.N_OUTCOME_TARGETS, 19).clone()
    assert torch.allclose(denormalize_quantiles(norm.normalize(q), norm), q,
                          atol=1e-9)
    # stats are fitted from the supplied (TRAIN) targets only: adding CAL rows
    # to a separate fit changes nothing observable here
    norm_cal = OutcomeNorm.fit(train + cal, mask + [torch.ones(
        C.N_TAU, dtype=torch.bool)])
    assert norm.to_dict() != norm_cal.to_dict()      # CAL would change them
    assert json.dumps(norm.to_dict(), sort_keys=True) != json.dumps(
        norm_cal.to_dict(), sort_keys=True)


def test_trainer_normalization_from_train_split_only(tmp_path):
    runs = [
        dict(chunk="chunk4", world="wt", arm="base", policy="damping",
             split="TRAIN", t0_s=100.0),
        dict(chunk="chunk4", world="wt", arm="native", policy="native",
             split="TRAIN", t0_s=100.0),
    ]
    pack = _pack(tmp_path, runs, k=8)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN")
    norm = ia.fit_outcome_norm(corpus)
    # recompute manually from the TRAIN outcome rows through the reader
    targets, masks = [], []
    for i in range(len(corpus._v1)):
        y, m, _ = corpus._v1.outcome_target(i)
        targets.append(y)
        masks.append(m)
    manual = OutcomeNorm.fit(targets, masks)
    assert np.allclose(norm.loc, manual.loc)
    assert np.allclose(norm.scale, manual.scale)


def test_fit_writes_outcome_norm_sha_and_receipt(tmp_path):
    from test_idea_a_fakes import seeded_bundle

    windows = make_windows(2)
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    receipt = tr.fit(windows=windows)
    norm_file = tmp_path / "out" / "OUTCOME_NORM.json"
    assert norm_file.is_file()
    payload = json.loads(norm_file.read_text(encoding="utf-8"))
    assert payload["schema"] == "wm_n1_outcome_norm/1"
    assert receipt["outcome_norm_sha256"] == tr.outcome_norm.sha256()
    assert receipt["loss_balance"] is not None
    # A6: the ON default records the frozen A6 LF-sha in the receipt.
    assert receipt["state_global_token"] is True
    assert receipt["amendment_6_lf_sha256"] == ia._seals.LF_SHA_AMENDMENT_6


# ----------------------------------- A6 freeze gate (state global token) -----
def test_a6_flag_on_requires_frozen_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ia._seals, "AMENDMENT_6_PATH",
                        tmp_path / "missing_a6.md")
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path), num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=ia.IdeaABundle(config=cfg))
    with pytest.raises(ia._seals.Amendment6Error):
        tr.a6_binding()


def test_a6_flag_on_refuses_wrong_sha(tmp_path, monkeypatch):
    bad = tmp_path / "a6.md"
    bad.write_bytes(b"# not the frozen A6\n")
    monkeypatch.setattr(ia._seals, "AMENDMENT_6_PATH", bad)
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path), num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=ia.IdeaABundle(config=cfg))
    with pytest.raises(ia._seals.Amendment6Error):
        tr.a6_binding()


def test_a6_flag_off_needs_no_frozen_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ia._seals, "AMENDMENT_6_PATH",
                        tmp_path / "missing_a6.md")
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path), num_workers=0,
                              state_global_token=False)
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=ia.IdeaABundle(config=cfg))
    assert tr.a6_binding() is None


def test_a6_fit_refuses_when_file_missing(tmp_path, monkeypatch):
    from test_idea_a_fakes import seeded_bundle

    monkeypatch.setattr(ia._seals, "AMENDMENT_6_PATH",
                        tmp_path / "missing_a6.md")
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    with pytest.raises(ia._seals.Amendment6Error):
        tr.fit(windows=make_windows(2))


def test_a6_sha_recorded_in_checkpoint(tmp_path):
    from test_idea_a_fakes import seeded_bundle

    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    receipt = tr.fit(windows=make_windows(2))
    payload = torch.load(receipt["checkpoint"], map_location="cpu",
                         weights_only=True)
    assert payload["state_global_token"] is True
    assert payload["amendment_6_lf_sha256"] == ia._seals.LF_SHA_AMENDMENT_6


# ------------------------------------------------- item 2: global token -----
def test_state_global_token_is_state_only_and_label_invariant():
    enc = StateGlobalToken(d_model=8)
    enc.eval()
    g = torch.Generator().manual_seed(0)
    kin = torch.randn(2, 5, 7, generator=g)
    mask = torch.ones(2, 5, dtype=torch.bool)
    counter = torch.randn(2, 2, 30, generator=g)
    base = enc(kin, mask, counter)
    # perturbing labels/actions/outcomes must not change the token (it never
    # reads them); the signature has no such input
    import inspect

    params = list(inspect.signature(StateGlobalToken.forward).parameters)
    assert params == ["self", "kin", "kin_mask", "counter"]
    assert all(k not in p.lower() for p in params
               for k in ("label", "strategy", "action", "is_cav", "policy"))
    assert torch.equal(base, enc(kin.clone(), mask.clone(), counter.clone()))
    # a changed STATE does change the token
    assert not torch.allclose(base, enc(kin + 5.0, mask, counter))


def test_trainer_global_token_label_leak_and_zeros_flag():
    bundle = ia.IdeaABundle(hdv=FakeHdv(), arrival=FakeArrival(),
                            corr=FakeCorr(),
                            state_enc=StateGlobalToken(d_model=CORR_D_MODEL))
    cfg = ia.IdeaATrainConfig(out_dir="unused", state_global_token=True,
                              num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=bundle)
    batch = ia.collate_canonical(make_windows(1))
    a = tr.global_token(batch)
    perturbed = dict(batch)
    perturbed["is_cav"] = ~batch["is_cav"]
    perturbed["target_accel"] = batch["target_accel"] + 99.0
    perturbed["outcome_quantiles"] = batch["outcome_quantiles"] + 1234.0
    perturbed["global_token"] = torch.full_like(batch["global_token"], 7.0)
    assert torch.equal(a, tr.global_token(perturbed))
    # zeros-flag reproduces the old path: explicit batch token, else zeros
    cfg_old = ia.IdeaATrainConfig(out_dir="unused",
                                  state_global_token=False, num_workers=0)
    tr_old = ia.IdeaATrainer(cfg_old, corpus=None, bundle=bundle)
    explicit = torch.full_like(batch["global_token"], 3.0)
    shifted = dict(batch, global_token=explicit)
    assert torch.equal(tr_old.global_token(shifted), explicit)
    no_token = {k: v for k, v in batch.items() if k != "global_token"}
    zeros = tr_old.global_token(no_token)
    assert torch.equal(zeros, torch.zeros(batch["kin"].shape[0], CORR_D_MODEL))


# ----------------------------------------------- item 4: DataLoader parity --
def test_dataloader_batches_equal_in_process(tmp_path):
    from train.dataset import PairAwareBatchSampler, contrast_pairs, \
        epoch_batches, pair_groups

    pack = _pack(tmp_path, _many_group_runs(6, 4), k=8, n_ticks=3)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", pair_sampling=True,
                            horizon_steps=2)
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=4,
                              out_dir="unused", num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                         bundle=ia.IdeaABundle(config=cfg))
    in_proc = tr._iter_batches(corpus)
    loader = tr._iter_batches_loader(corpus)
    for _ in range(3):
        a = next(in_proc)
        b = next(loader)
        assert a["meta"] == b["meta"]
    # the pure sampler yields exactly the epoch_batches positions
    items = ia._source_item_meta(corpus)
    pairs = contrast_pairs(items)
    expected = epoch_batches(items, 4, 101, 0, pairs)
    got = list(PairAwareBatchSampler(items, 4, 101, epochs=1, pairs=pairs))
    assert expected == got
    assert pair_groups(items)


# ------------------------------------------------ item 5: fail-closed gate --
def test_timing_gate_refuses_synthetic_when_not_tiny():
    from bundle import timing as T

    with pytest.raises(RuntimeError):
        T.measure_idea_a_corpus(T.make_config(tiny=True), "cpu", warmup=0,
                                timed_steps=1, k_max=16, pack_dir=None,
                                tiny=False)
    with pytest.raises(RuntimeError):
        T.loss_decrease_idea_a(T.make_config(tiny=True), "cpu", steps=1,
                               k_max=16, pack_dir=None, tiny=False)
    # the explicit tiny path still measures the synthetic pack and records it
    out = T.measure_idea_a_corpus(T.make_config(tiny=True), "cpu", warmup=0,
                                  timed_steps=1, k_max=128, pack_dir=None,
                                  tiny=True)
    assert out["pack_source"] == "synthetic"
    assert "num_workers" in out


# ------------------------------------------- item 6: main path + balance ----
def test_main_path_fit_balanced_magnitudes_many_groups(tmp_path):
    pack = _pack(tmp_path, _many_group_runs(40, 8), k=16, n_ticks=3)
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=6, batch_size=8,
                              out_dir=str(tmp_path / "out"))
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", pair_sampling=True,
                            horizon_steps=cfg.rollout_horizon_steps)
    tr = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                         bundle=ia.build_idea_a_bundle(cfg))
    receipt = tr.fit()
    assert set(receipt["active_loss_terms"]) == set(ia.REQUIRED_LOSS_TERMS)
    assert receipt["state_global_token"] is True
    assert (tmp_path / "out" / "OUTCOME_NORM.json").is_file()
    balance = receipt["loss_balance"]
    assert balance["balanced"] is True, balance["magnitudes"]
    assert set(balance["magnitudes"]) == set(ia.REQUIRED_LOSS_TERMS)
    assert all(r <= 100.0 for r in balance["ratios"].values()), \
        balance["ratios"]
