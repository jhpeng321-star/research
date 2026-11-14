"""Train-path Amendment 4 CAL split: selection (tune) vs calibration (cal).

Synthetic world lists + a synthetic split file only; no real CAL data.
"""
from __future__ import annotations

import json

import pytest

import train.idea_a as ia
from test_idea_a_fakes import seeded_bundle


def _split(tmp_path, *, tune=("t1", "t2"), cal=("c1", "c2")):
    p = tmp_path / "CAL_SPLIT_A4.json"
    p.write_text(json.dumps({
        "schema": "wm_n1_cal_split_a4/1", "seed": 20261001,
        "cal_tune": list(tune), "cal_cal": list(cal)}), encoding="utf-8")
    return p


def test_selection_path_reads_tune_and_refuses_cal(tmp_path):
    p = _split(tmp_path)
    assert ia.selection_worlds(None, cal_split_path=p) == ["t1", "t2"]
    with pytest.raises(Exception) as ei:
        ia.selection_worlds(["c1"], cal_split_path=p)
    assert "cal_tune" in str(ei.value)


def test_calibration_path_reads_cal_and_refuses_tune(tmp_path):
    p = _split(tmp_path)
    assert ia.calibration_worlds(None, cal_split_path=p) == ["c1", "c2"]
    with pytest.raises(Exception):
        ia.calibration_worlds(["t1"], cal_split_path=p)


def test_filter_cal_items_keeps_only_the_half(tmp_path):
    p = _split(tmp_path)
    items = [{"world": "t1"}, {"world": "c1"}, {"world": "t2"}]
    kept = ia.filter_cal_items(items, ia.CAL_CAL, cal_split_path=p)
    assert [it["world"] for it in kept] == ["c1"]


class _CalCorpus:
    split = "CALIBRATION"
    pack_dir = None

    def __init__(self, worlds):
        self._worlds = list(worlds)

    def __len__(self):
        return len(self._worlds)

    def __getitem__(self, i):
        return {"world": self._worlds[i]}


def test_evaluate_cal_half_refuses_missing_item_world(tmp_path):
    p = _split(tmp_path)
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=1,
                              ensemble_k=1, out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=None, member=1, seed=101,
                         bundle=seeded_bundle(101))
    with pytest.raises(ia.IdeaAConfigError):
        tr.evaluate_cal(_CalCorpus([None]), half=ia.CAL_CAL,
                        cal_split_path=p)


def _cal_runs():
    return [
        dict(chunk="chunk2", world="calw_t", arm="native", policy="native",
             split="CALIBRATION", t0_s=300.0, n_ticks=4, n_dec=2),
        dict(chunk="chunk2", world="calw_c", arm="native", policy="native",
             split="CALIBRATION", t0_s=300.0, n_ticks=4, n_dec=2),
    ]


def test_evaluate_cal_scores_only_the_named_half(tmp_path):
    from train.tests import packsynth

    pack = packsynth.write_idea_a_pack(tmp_path, _cal_runs(), k=8)
    p = _split(tmp_path, tune=("calw_t",), cal=("calw_c",))
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=1,
                              ensemble_k=1, out_dir=str(tmp_path / "out"),
                              rollout_horizon_steps=3)
    corpus = ia.IdeaACorpus(str(pack), split="CALIBRATION", horizon_steps=3)
    tr = ia.IdeaATrainer(cfg, corpus=corpus, member=1, seed=101,
                         bundle=seeded_bundle(101))
    tune = tr.evaluate_cal(half=ia.CAL_TUNE, cal_split_path=p)
    cal = tr.evaluate_cal(half=ia.CAL_CAL, cal_split_path=p)
    assert tune["status"] == "ok" and tune["n"] == 1
    assert tune["half"] == ia.CAL_TUNE
    assert cal["status"] == "ok" and cal["n"] == 1
    assert cal["half"] == ia.CAL_CAL
    # A4.2: a non-empty CAL score without a named half is refused
    with pytest.raises(ia.IdeaAConfigError):
        tr.evaluate_cal(cal_split_path=p)
    with pytest.raises(ia.IdeaAConfigError):
        tr.evaluate_cal(half="not_a_half", cal_split_path=p)
