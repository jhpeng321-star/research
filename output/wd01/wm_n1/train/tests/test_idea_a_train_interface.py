"""P7 interface contract for the Idea A trainer (symbols, config, fit)."""
from __future__ import annotations

import pytest

import train.idea_a as ia


def test_public_symbols():
    assert ia.MEMBER_ORDER == (1, 2, 3, 4, 5)
    assert ia.DEFAULT_SEEDS == (101, 202, 303, 404, 505)
    assert ia.CHECKPOINT_FORMAT == "wm_n1_idea_a_member/1"


def test_config_member_order_and_partial_ensemble():
    cfg = ia.IdeaATrainConfig()
    assert cfg.device == "cpu"
    assert cfg.seed_for_member(1) == 101
    assert cfg.seed_for_member(5) == 505
    assert ia.ensemble_members(1) == (1,)
    assert ia.ensemble_members(3) == (1, 2, 3)
    assert ia.ensemble_members(5) == (1, 2, 3, 4, 5)
    with pytest.raises(ia.IdeaAConfigError):
        ia.ensemble_members(0)
    with pytest.raises(ia.IdeaAConfigError):
        ia.ensemble_members(6)
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(ensemble_k=6)
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(member=0)
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(hdv_use_cav_flag=True)


def test_config_from_dict_and_helpers():
    cfg = ia.IdeaATrainConfig.from_dict({"member": 2, "ensemble_k": 3,
                                         "seeds": [1, 2, 3, 4, 5],
                                         "device": "cpu"})
    assert cfg.member == 2 and cfg.ensemble_k == 3 and cfg.seeds[-1] == 5
    assert cfg.loss_weights()["response_contrast"] == 1.0
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig.from_dict({"nope": 1})
    assert set(cfg.model_configs()) == {"hdv", "arrival", "corr"}


def test_trainer_defaults_and_member_first():
    tr = ia.IdeaATrainer(ia.IdeaATrainConfig(), corpus=None)
    assert tr.member == 1 and tr.seed == 101 and tr.device == "cpu"
    tr2 = ia.IdeaATrainer(ia.IdeaATrainConfig(), corpus=None, member=3,
                          seed=9)
    assert tr2.member == 3 and tr2.seed == 9
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainer(ia.IdeaATrainConfig(), corpus=None, member=6)


def test_fit_refuses_member_without_member_1(tmp_path):
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=None, member=2, seed=202)
    with pytest.raises(ia.IdeaAConfigError):
        tr.fit(windows=[])


def test_fit_empty_corpus_and_cal_paths(tmp_path):
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=[], member=1, seed=101)
    receipt = tr.fit()                       # no steps; still a checkpoint
    assert receipt["steps"] == 0
    assert receipt["members"] == list(ia.ensemble_members(cfg.ensemble_k))
    assert (tmp_path / "out" / "member_1.pt").is_file()
    assert tr.evaluate_cal()["status"] == "unavailable"


class _SplitCorpus:
    """A corpus-like stand-in exposing only a ``split`` label."""

    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self) -> int:
        return 0

    def __getitem__(self, i: int) -> dict:   # pragma: no cover - never reached
        raise IndexError(i)


@pytest.mark.parametrize("split", ["CALIBRATION", "CAL", "HOLDOUT", "TEST"])
def test_fit_refuses_non_train_split(tmp_path, split):
    # amendment A3.2: CAL is calibration/variant selection only; no CAL (or
    # sealed-split) data may reach the training step.
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=_SplitCorpus(split), member=1, seed=101)
    with pytest.raises(ia.IdeaAConfigError):
        tr.fit()
    assert not (tmp_path / "out" / "member_1.pt").exists()


def test_fit_accepts_train_split_corpus(tmp_path):
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=_SplitCorpus("TRAIN"), member=1,
                         seed=101)
    receipt = tr.fit()
    assert receipt["steps"] == 0
