"""Trainer tests: tiny CPU loop, checkpoint/resume, seeds, weights_only."""
from __future__ import annotations

import inspect
import json

import pytest
import torch

from packsynth import make_runs, write_pack
from train import trainer as trainer_mod
from train.config import MEMBER_ORDER, TrainConfig
from train.dataset import MemmapCorpus
from train.trainer import Trainer


def tiny_config(tmp_path, **over):
    values = dict(d_model=32, n_heads=4, n_layers=1, ffn=48, bias_hidden=8,
                  counter_channels=8, epochs=1, steps_per_epoch=30,
                  batch_size=4, lr=1e-2, device="cpu",
                  out_dir=str(tmp_path / "run"))
    values.update(over)
    return TrainConfig(**values)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "pack"
    write_pack(root, make_runs())
    return MemmapCorpus(root, pair_sampling=True)


def test_tiny_train_loop_loss_decreases(tmp_path, corpus):
    trainer = Trainer(tiny_config(tmp_path), corpus, member=1, seed=101,
                      device="cpu")
    result = trainer.fit()
    assert result["losses"][-1] < result["losses"][0]
    assert result["member"] == 1 and result["seed"] == 101
    run = tmp_path / "run"
    assert (run / "member_1.pt").is_file()
    assert (run / "member_1.json").is_file()
    receipt = json.loads((run / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["seed"] == 101 and receipt["member"] == 1


def test_save_resume_gives_identical_next_step_loss(tmp_path, corpus):
    cfg = tiny_config(tmp_path, steps_per_epoch=5)
    trainer = Trainer(cfg, corpus, member=1, seed=101, device="cpu")
    trainer.fit()
    items = [corpus[0], corpus[1], corpus[2]]
    before = float(trainer._forward_loss(items, train=False).detach())

    ckpt = trainer.save_checkpoint(tmp_path / "run" / "member_1.pt")
    fresh = Trainer(cfg, corpus, member=2, seed=999, device="cpu")
    fresh.resume(ckpt)
    after = float(fresh._forward_loss(items, train=False).detach())

    assert after == before
    assert fresh.member == 1 and fresh.seed == 101


def test_resume_loads_with_weights_only_true(tmp_path, corpus, monkeypatch):
    cfg = tiny_config(tmp_path, steps_per_epoch=2)
    trainer = Trainer(cfg, corpus, member=1, seed=101, device="cpu")
    trainer.fit()
    ckpt = trainer.save_checkpoint(tmp_path / "run" / "member_1.pt")

    captured = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    Trainer(cfg, corpus, member=1, seed=101, device="cpu").resume(ckpt)
    assert captured.get("weights_only") is True
    assert "weights_only=True" in inspect.getsource(trainer_mod)


def test_member_bounds(tmp_path, corpus):
    with pytest.raises(ValueError):
        Trainer(tiny_config(tmp_path), corpus, member=0, seed=1, device="cpu")
    with pytest.raises(ValueError):
        Trainer(tiny_config(tmp_path), corpus, member=6, seed=1, device="cpu")


def test_member_seeds_independent_init(tmp_path, corpus):
    cfg = tiny_config(tmp_path, steps_per_epoch=1)
    seeds = [cfg.seed_for_member(m) for m in MEMBER_ORDER]
    t1 = Trainer(cfg, corpus, member=1, seed=seeds[0], device="cpu")
    t2 = Trainer(cfg, corpus, member=2, seed=seeds[1], device="cpu")
    v1 = torch.cat([p.detach().flatten() for p in t1.model.parameters()])
    v2 = torch.cat([p.detach().flatten() for p in t2.model.parameters()])
    assert not torch.allclose(v1, v2)


def test_evaluate_cal_diagnostics(tmp_path, corpus):
    trainer = Trainer(tiny_config(tmp_path), corpus, member=1, seed=101,
                      device="cpu")
    result = trainer.evaluate_cal()
    assert result["status"] == "ok"
    assert result["n"] == 1
    assert result["loss"] is not None
