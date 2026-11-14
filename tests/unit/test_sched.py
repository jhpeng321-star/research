"""SCHED (2026-10-14): validation horizons, multi-horizon selection, mixed-horizon batches.

All switches default OFF; the off path must make exactly the same training-loop decisions
(train unrolls, validation calls, selection) as before.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "train"))

from _compat import FlatConfig, legacy  # noqa: E402
from synth import make_arrays, small_model  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.config.train import TrainConfig  # noqa: E402
from wms.train import selection  # noqa: E402
from wms.train.settings import TrainSettings  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402

train_mod = legacy("train")
STAGES = (10.0, 30.0, 60.0, 120.0)
V4_SCHED_DIFF = {"stage_steps": [1200, 1500, 400, 160], "val_all_horizons": True,
                 "selection_mode": "multi_horizon", "mixed_horizon_frac": 0.3,
                 "out_dir": "runs/active/c4_wm_v4_sched"}


def _flat(tmp_path, sched=None, **kw):
    base = dict(device="cpu", threads=1, truncation_steps=8, batch_size=2, seed=3,
                unroll_curriculum_s=STAGES, stage_steps=(3, 4, 2, 2), out_dir=str(tmp_path),
                select_by_long_horizon=True, val_every=2)
    base.update(kw)
    flat = train_mod.WmSTrainConfig(**base).to_dict()
    flat.update(sched or {})
    return flat


def _dataset():
    return train_mod.TrajectoryDataset.from_arrays(make_arrays())


def _fake_score(step: int, unroll: int) -> float:
    """Scripted validation mean_position: improves with steps, worse at long horizons."""
    return unroll * (1.0 + 0.5 * ((step * 7919 + unroll) % 5) / 5.0)


def _fake_fit(tmp_path, sched=None, **kw):
    """Run ``fit`` with the model calls faked; returns (calls, result, trainer)."""
    flat = _flat(tmp_path, sched, **kw)
    ds = _dataset()
    trainer = WmSTrainer(FlatConfig(flat), small_model(), ds,
                         val_dataset=ds if flat.get("select_by_long_horizon") else None,
                         out_dir=tmp_path)
    calls = []
    state = {"step": 0}

    def train_batch(batch, *, unroll_steps, truncation_steps=None):
        state["step"] += 1
        calls.append(("train", unroll_steps, truncation_steps))
        return {"loss": 1.0, "unroll_steps": int(unroll_steps), "seconds": 0.0, "finite": True}

    def evaluate(val_dataset, *, unroll_steps, require_full_horizon=False):
        calls.append(("eval", state["step"], unroll_steps))
        return {"n_batches": 1, "valid_targets": 5, "reached_steps": unroll_steps,
                "available_steps": unroll_steps, "unroll_steps": unroll_steps,
                "mean_position": _fake_score(state["step"], unroll_steps),
                "mean_agg_reward": 1.0}

    trainer.train_batch = train_batch
    trainer.evaluate = evaluate
    result = trainer.fit()
    return calls, result, trainer


# ------------------------------------------------------------------- off == unchanged
#: Decisions of the pre-SCHED loop for stage_steps (3, 4, 2, 2), val_every 2 (60 s + stage
#: at >= 60 s stages, the stage horizon otherwise).
EXPECTED_OFF = (
    [("train", 20, 8)] * 3 + [("train", 60, 8)] * 4 + [("train", 120, 8)] * 2
    + [("train", 240, 8)] * 2)
EXPECTED_EVAL_OFF = [(2, 20), (3, 20), (4, 60), (6, 60), (7, 60), (8, 120), (9, 120),
                     (10, 120), (10, 240), (11, 120), (11, 240)]
OFF_EXPLICIT = {"val_all_horizons": False, "selection_mode": "long_horizon",
                "mixed_horizon_frac": 0.0, "mixed_horizon_s": [5.0],
                "selection_reference": [1.0, 1.0, 1.0, 1.0]}


@pytest.mark.parametrize("sched", [None, OFF_EXPLICIT])
def test_off_makes_the_pre_sched_loop_decisions(tmp_path, sched):
    calls, result, _ = _fake_fit(tmp_path, sched)
    assert [c for c in calls if c[0] == "train"] == EXPECTED_OFF
    assert [c[1:] for c in calls if c[0] == "eval"] == EXPECTED_EVAL_OFF
    # Old selection: longest eligible horizon (120 s) wins; lowest score among 120 s rows.
    rows120 = [r for r in result.val_by_stage if r["horizon_s"] == 120.0]
    best = min(rows120, key=lambda r: r["mean_position"])
    sel = result.receipt["long_horizon_selection"]
    assert sel["best"]["step"] == best["step"] and sel["best"]["horizon_s"] == 120.0
    assert sel["policy"] == "longest-eligible-horizon-then-lowest-mean-position"
    assert result.selection is None and "multi_horizon" not in sel
    assert all("mixed_horizon_s" not in r and "selection_score" not in r for r in
               result.history + result.val_by_stage)


def test_off_flags_are_bit_identical_in_a_real_fit(tmp_path):
    def run(sub, sched):
        flat = _flat(tmp_path / sub, sched, unroll_curriculum_s=(5.0, 10.0), stage_steps=(2, 2),
                     val_every=1)
        ds = _dataset()
        trainer = WmSTrainer(FlatConfig(flat), small_model(), ds, val_dataset=ds,
                             out_dir=tmp_path / sub)
        result = trainer.fit()
        return trainer.model.state_dict(), result

    a, ra = run("a", None)
    b, rb = run("b", OFF_EXPLICIT)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert [r["loss"] for r in ra.history] == [r["loss"] for r in rb.history]
    strip = lambda rows: [{k: v for k, v in r.items()} for r in rows]  # noqa: E731
    assert strip(ra.val_by_stage) == strip(rb.val_by_stage)


def test_settings_defaults_and_validation():
    s = TrainSettings()
    assert (s.val_all_horizons, s.selection_mode, s.mixed_horizon_frac) == (
        False, "long_horizon", 0.0)
    with pytest.raises(ValueError):
        TrainSettings(val_all_horizons=True)          # needs the validation schedule
    with pytest.raises(ValueError):
        TrainSettings(selection_mode="multi_horizon")
    with pytest.raises(ValueError):
        TrainSettings(selection_mode="best", select_by_long_horizon=True)
    with pytest.raises(ValueError):
        TrainSettings(mixed_horizon_frac=1.5)
    with pytest.raises(ValueError):
        TrainSettings(select_by_long_horizon=True, selection_reference=(1.0,))
    TrainSettings(select_by_long_horizon=True, val_all_horizons=True,
                  selection_mode="multi_horizon", mixed_horizon_frac=0.3)


# ------------------------------------------------------------------- validation horizons
def test_validation_horizons():
    assert selection.validation_horizons(30.0) == (30.0,)
    assert selection.validation_horizons(120.0) == (60.0, 120.0)
    f = lambda st: selection.validation_horizons(st, all_horizons=True)  # noqa: E731
    assert f(10.0) == (10.0,)
    assert f(30.0) == (10.0, 30.0)
    assert f(60.0) == (10.0, 30.0, 60.0)
    assert f(120.0) == (10.0, 30.0, 60.0, 120.0)
    assert f(45.0) == (10.0, 30.0, 45.0)


def test_val_all_horizons_logs_every_horizon(tmp_path):
    calls, result, _ = _fake_fit(tmp_path, {"val_all_horizons": True})
    evals = [c[1:] for c in calls if c[0] == "eval"]
    assert evals == [(2, 20), (3, 20), (4, 20), (4, 60), (6, 20), (6, 60), (7, 20), (7, 60),
                     (8, 20), (8, 60), (8, 120), (9, 20), (9, 60), (9, 120),
                     (10, 20), (10, 60), (10, 120), (10, 240),
                     (11, 20), (11, 60), (11, 120), (11, 240)]
    logged = [json.loads(line) for line in (tmp_path / "val_log.jsonl").read_text().splitlines()]
    assert [(r["step"], r["horizon_s"]) for r in logged] == [
        (step, unroll * 0.5) for step, unroll in evals]
    # Selection policy unchanged (long-horizon) unless multi_horizon is requested.
    assert result.receipt["long_horizon_selection"]["best"]["horizon_s"] == 120.0
    assert [c for c in calls if c[0] == "train"] == EXPECTED_OFF


# ------------------------------------------------------------------- multi-horizon score
def _rows(stage, step, values, agg=None):
    return [dict(horizon_s=h, mean_position=v, stage_s=stage, step=step, valid_targets=5,
                 **({} if agg is None else {"mean_agg_reward": agg[h]}))
            for h, v in values.items()]


def test_multi_horizon_score_and_pick():
    sel = selection.MultiHorizonSelection(references=(10.0, 50.0, 130.0, 0.0),
                                          eligible_stage_s=30.0)
    log = [(10.0, 200, {10.0: 1.0}),                                   # ineligible stage
           (30.0, 1400, {10.0: 9.0, 30.0: 50.0}),                       # .95
           (30.0, 1600, {10.0: 7.0, 30.0: 48.0}),                       # .83  <- best
           (60.0, 2200, {10.0: 8.0, 30.0: 49.0, 60.0: 130.0}),          # .9267
           (120.0, 2600, {10.0: 8.0, 30.0: 50.0, 60.0: 128.0, 120.0: 240.0}),  # 120: own=1
           (120.0, 2660, {10.0: 8.0, 30.0: 50.0, 60.0: 128.0, 120.0: 230.0})]
    chosen = [sel.consider_event(_rows(*e), stage_s=e[0], step=e[1]) for e in log]
    assert chosen == [False, True, True, False, False, False]
    scores = [e["score"] for e in sel.table]
    assert scores[1] == pytest.approx((0.9 + 1.0) / 2)
    assert scores[2] == pytest.approx((0.7 + 48 / 50) / 2)
    assert scores[3] == pytest.approx((0.8 + 49 / 50 + 1.0) / 3)
    assert scores[4] == pytest.approx((0.8 + 1.0 + 128 / 130 + 1.0) / 4)
    assert scores[5] == pytest.approx((0.8 + 1.0 + 128 / 130 + 230 / 240) / 4)
    assert not sel.table[0]["eligible"] and sel.table[0]["score"] == pytest.approx(0.1)
    assert sel.best["step"] == 1600 and sel.horizon_s == 30.0
    assert [e["selected"] for e in sel.table] == [False, False, True, False, False, False]
    summary = sel.summary()
    assert summary["best_step"] == 1600 and summary["own_first_position"]["120"] == 240.0
    # The pre-SCHED policy on the same log always takes the final stage.
    old = selection.LongHorizonSelection()
    for e in log:
        for row in _rows(*e):
            old.consider(row)
    assert old.best["step"] == 2660 and old.best["horizon_s"] == 120.0


def test_multi_horizon_agg_reward_weight_and_bad_rows():
    agg_a, agg_b = {10.0: 1.0, 30.0: 2.0}, {10.0: 3.0, 30.0: 6.0}
    sel = selection.MultiHorizonSelection(references=(10.0, 50.0, 130.0, 0.0),
                                          agg_reward_weight=1.0)
    assert sel.consider_event(_rows(30.0, 1, {10.0: 9.0, 30.0: 50.0}, agg_a), stage_s=30.0, step=1)
    # better positions but 3x the aggregate-reward error -> loses with weight 1
    assert not sel.consider_event(_rows(30.0, 2, {10.0: 8.0, 30.0: 45.0}, agg_b),
                                  stage_s=30.0, step=2)
    assert sel.table[1]["score"] == pytest.approx((0.8 + 0.9) / 2 + 3.0)
    nan_rows = _rows(60.0, 3, {10.0: float("nan"), 30.0: float("nan")})
    assert not sel.consider_event(nan_rows, stage_s=60.0, step=3)
    assert sel.table[-1]["score"] == float("inf") and not sel.table[-1]["eligible"]
    odd = _rows(60.0, 4, {11.5: 1.0})                    # not a selection horizon
    assert not sel.consider_event(odd, stage_s=60.0, step=4)
    assert selection.eligible_stage_s(STAGES, 30.0) == 30.0
    assert selection.eligible_stage_s((5.0, 10.0), 30.0) == 10.0


def test_multi_horizon_fit_selects_and_reports(tmp_path):
    calls, result, _ = _fake_fit(tmp_path, {"val_all_horizons": True,
                                            "selection_mode": "multi_horizon",
                                            "selection_reference": [10.0, 30.0, 60.0, 0.0]})
    table = result.selection["table"]
    assert [e["step"] for e in table] == [2, 3, 4, 6, 7, 8, 9, 10, 11]
    eligible = [e for e in table if e["eligible"]]
    assert all(e["stage_s"] >= 30.0 for e in eligible) and len(eligible) == 7
    best = min(eligible, key=lambda e: e["score"])
    assert result.selection["best_step"] == best["step"] and best["selected"]
    assert sum(e["selected"] for e in table) == 1
    sel = result.receipt["long_horizon_selection"]
    assert sel["policy"] == "multi-horizon-reference-normalised-mean"
    assert sel["best"]["step"] == best["step"] and sel["multi_horizon"]["table"] == table
    ckpt = torch.load(tmp_path / "wm_s_best.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == best["step"] and ckpt["selection"]["selected"]
    logged = [json.loads(line) for line in (tmp_path / "val_log.jsonl").read_text().splitlines()]
    assert all("selection_score" in r for r in logged)
    assert {r["step"] for r in logged if r["selection"]} <= {e["step"] for e in table}
    # per-stage checkpoints are still written
    for stage in (10, 30, 60, 120):
        assert (tmp_path / f"wm_s_stage{stage}.pt").is_file()


# ------------------------------------------------------------------- mixed horizons
def test_mixed_schedule_rate_and_stages():
    mix = selection.MixedHorizonSchedule(0.3, (10.0, 20.0), 30.0, seed=101)
    assert all(mix.draw(10.0) is None for _ in range(100))
    draws = [mix.draw(30.0) for _ in range(20000)]
    short = [d for d in draws if d is not None]
    assert abs(len(short) / len(draws) - 0.3) < 0.015
    assert set(short) == {10.0, 20.0} and abs(short.count(10.0) / len(short) - 0.5) < 0.03
    assert selection.MixedHorizonSchedule(0.0, (10.0,), 30.0, seed=1).draw(120.0) is None
    only_short = selection.MixedHorizonSchedule(1.0, (10.0, 60.0), 30.0, seed=1)
    assert {only_short.draw(30.0) for _ in range(50)} == {10.0}   # never >= the stage


def test_mixed_batches_rate_in_fit(tmp_path):
    calls, result, _ = _fake_fit(tmp_path, {"mixed_horizon_frac": 0.3},
                                 stage_steps=(100, 300, 150, 150), select_by_long_horizon=False)
    hist = result.history
    by_stage = {st: [r for r in hist if r["stage_s"] == st] for st in STAGES}
    assert all("mixed_horizon_s" not in r for r in by_stage[10.0])
    full = {30.0: 60, 60.0: 120, 120.0: 240}
    n_short = 0
    for st, rows in by_stage.items():
        if st == 10.0:
            continue
        short = [r for r in rows if "mixed_horizon_s" in r]
        n_short += len(short)
        assert all(r["unroll_steps"] == round(r["mixed_horizon_s"] / 0.5) for r in short)
        assert {r["mixed_horizon_s"] for r in short} == {10.0, 20.0}
        assert all(r["unroll_steps"] == full[st] for r in rows if "mixed_horizon_s" not in r)
    assert abs(n_short / 600 - 0.3) < 0.06
    # same BPTT chunk setting for short batches
    assert {c[2] for c in calls if c[0] == "train"} == {8}


def test_mixed_and_multi_real_fit(tmp_path):
    """End to end with real compute: short batches train, multi-horizon picks wm_s_best."""
    flat = _flat(tmp_path, {"val_all_horizons": True, "selection_mode": "multi_horizon",
                            "selection_horizons_s": [5.0, 10.0], "selection_reference": [0.0, 0.0],
                            "selection_min_stage_s": 10.0, "mixed_horizon_frac": 1.0,
                            "mixed_horizon_s": [5.0], "mixed_horizon_min_stage_s": 10.0},
                 unroll_curriculum_s=(5.0, 10.0), stage_steps=(1, 2), val_every=1)
    ds = _dataset()
    result = WmSTrainer(FlatConfig(flat), small_model(), ds, val_dataset=ds,
                        out_dir=tmp_path).fit()
    assert [r["unroll_steps"] for r in result.history] == [10, 10, 10]
    assert [r.get("mixed_horizon_s") for r in result.history] == [None, 5.0, 5.0]
    assert all(r["finite"] for r in result.history)
    assert result.selection["best_step"] in (2, 3) and (tmp_path / "wm_s_best.pt").is_file()
    assert [e["horizons_s"] for e in result.selection["table"]] == [[5.0], [5.0, 10.0],
                                                                     [5.0, 10.0]]


# ------------------------------------------------------------------- config
def test_v4_sched_config_differs_from_v3_only_in_sched_keys():
    v3 = load_train_config("configs/c4_wm_v3.toml").to_legacy_flat()
    v4 = load_train_config("configs/c4_wm_v4_sched.toml").to_legacy_flat()
    diff = {k for k in set(v3) | set(v4) if v3.get(k) != v4.get(k)}
    assert diff == set(V4_SCHED_DIFF)
    assert {k: v4[k] for k in diff} == V4_SCHED_DIFF
    assert TrainConfig.from_legacy_flat(v4).to_legacy_flat() == v4
    s = TrainSettings.from_flat(v4)
    assert s.select_by_long_horizon and s.mixed_horizon_s == (10.0, 20.0)
    assert s.selection_reference == (7.0356, 49.4889, 128.7179, 0.0)


def test_old_receipts_keep_their_flat_keys():
    flat = TrainConfig().to_legacy_flat()
    for key in ("val_all_horizons", "selection_mode", "mixed_horizon_frac", "mixed_horizon_s",
                "selection_reference", "selection_horizons_s"):
        assert key not in flat
    v3 = load_train_config("configs/c4_wm_v3.toml").to_legacy_flat()
    assert TrainConfig.from_legacy_flat(v3).to_legacy_flat() == v3
