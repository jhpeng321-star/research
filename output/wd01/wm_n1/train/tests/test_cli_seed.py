"""The Idea A CLI seeds model initialization before it builds the bundle."""
from __future__ import annotations

import torch

import train.idea_a as ia


def _run_cli(monkeypatch, *, member: int, seed: int | None,
             ambient_seed: int):
    seen = {}

    class FakeCorpus:
        def __init__(self, path, **kwargs):
            seen["corpus_args"] = (path, kwargs)

    def fake_bundle(cfg):
        seen["factory_cfg"] = cfg
        seen["initial_weights"] = torch.rand(4).clone()
        return object()

    class FakeTrainer:
        def __init__(self, cfg, corpus, *, member, seed, device, bundle):
            seen["trainer_args"] = (cfg, corpus, member, seed, device, bundle)
            self.member = member
            self.seed = seed

        def fit(self):
            return {"member": self.member, "seed": self.seed, "steps": 0,
                    "checkpoint": "synthetic-checkpoint"}

    monkeypatch.setattr(ia, "IdeaACorpus", FakeCorpus)
    monkeypatch.setattr(ia, "build_idea_a_bundle", fake_bundle)
    monkeypatch.setattr(ia, "IdeaATrainer", FakeTrainer)
    argv = ["--member", str(member), "--corpus", "synthetic-pack",
            "--out", "synthetic-output", "--device", "cpu",
            "--epochs", "1", "--steps-per-epoch", "2"]
    if seed is not None:
        argv += ["--seed", str(seed)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ambient_seed)
        assert ia.main(argv) == 0
    cfg, corpus, actual_member, actual_seed, device, bundle = seen["trainer_args"]
    assert seen["corpus_args"] == (
        "synthetic-pack",
        {"split": "TRAIN", "pair_sampling": True,
         "horizon_steps": cfg.rollout_horizon_steps, "k_max": cfg.k_max,
         "extra_pack_dirs": None, "extra_overlay_dir": None,
         "extra_identity_dirs": None},
    )
    assert isinstance(corpus, FakeCorpus)
    assert seen["factory_cfg"] is cfg
    assert bundle is not None
    assert (actual_member, actual_seed, device) == (
        member, seed if seed is not None else cfg.seed_for_member(member), "cpu")
    assert (cfg.epochs, cfg.steps_per_epoch, cfg.out_dir) == (
        1, 2, "synthetic-output")
    return seen["initial_weights"]


def test_cli_seed_controls_factory_before_trainer(monkeypatch):
    first = _run_cli(monkeypatch, member=1, seed=101, ambient_seed=7)
    repeated = _run_cli(monkeypatch, member=1, seed=101, ambient_seed=999)
    expected = torch.rand(4, generator=torch.Generator().manual_seed(101))
    assert torch.equal(first, expected)
    assert torch.equal(repeated, expected)


def test_cli_seed_override_and_member_default_change_initialization(monkeypatch):
    member_two = _run_cli(monkeypatch, member=2, seed=None, ambient_seed=7)
    override = _run_cli(monkeypatch, member=2, seed=303, ambient_seed=7)
    assert torch.equal(
        member_two, torch.rand(4, generator=torch.Generator().manual_seed(202)))
    assert torch.equal(
        override, torch.rand(4, generator=torch.Generator().manual_seed(303)))
    assert not torch.equal(member_two, override)
