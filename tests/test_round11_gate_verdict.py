"""Synthetic contract tests for the round-11 smoke verdict (round 10b + the structural courtesy fix)."""
import copy
from dataclasses import asdict
import json

import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import save_checkpoint
from wms.model.follower_courtesy import expand_head_inputs
from wms.physics.config import PhysicsConfig
from wms.train.round11_gate import check_checkpoints, check_init_report, check_log

from test_round10b_gate_verdict import FREEZE, make_rows as make_rows_10b

HEAD0 = "follower_courtesy.net.0.weight"


def make_rows():
    rows = make_rows_10b()
    for row in rows:
        row.update(follower_courtesy_contrast_n_controls=4000, follower_courtesy_contrast_shift_mean=-0.2,
                   follower_courtesy_contrast_neff_median=30.0, cutin_reaction_n_partnered_controls_excluded=12)
    return rows


@pytest.fixture
def run(tmp_path):
    def go(rows, *, text="ok\n", exit_code=0):
        log, smoke = tmp_path / "train_log.jsonl", tmp_path / "smoke.log"
        log.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in rows), encoding="utf-8")
        smoke.write_text(text, encoding="utf-8")
        return check_log(log, smoke, train_exit_code=exit_code)
    return go


def test_valid_round11_log_passes(run):
    result = run(make_rows())
    assert result["ok"], result["errors"]
    assert result["metrics"]["contrast_n_controls"] == 16000
    assert result["metrics"]["partnered_controls_excluded"] == 48


def test_round10b_failures_still_fail(run):
    rows = make_rows()
    for row in rows:
        row["follower_courtesy_n_rows"] = 0
    assert not run(rows)["ok"]
    assert not run(make_rows(), text="Traceback (most recent call last)")["ok"]


@pytest.mark.parametrize("field", ("follower_courtesy_contrast_n_controls", "follower_courtesy_contrast_shift_mean",
                                   "cutin_reaction_n_partnered_controls_excluded"))
@pytest.mark.parametrize("value", (None, float("nan"), float("inf"), True, "1"))
def test_missing_or_nonfinite_r11_telemetry_fails(run, field, value):
    rows = make_rows()
    rows[1][field] = value
    assert not run(rows)["ok"]


def test_no_contrast_controls_fails(run):
    rows = make_rows()
    for row in rows:
        row["follower_courtesy_contrast_n_controls"] = 0
    assert not run(rows)["ok"]


def test_init_report_modes():
    heads = ["follower_courtesy.net.0.bias", HEAD0, "follower_courtesy.net.4.weight"]
    check_init_report({"new_in_model": [], "expanded": [HEAD0], "shape_mismatch": [], "unused_from_checkpoint": [],
                       "follower_courtesy_init": "checkpoint"}, "checkpoint")
    check_init_report({"new_in_model": heads, "expanded": [], "shape_mismatch": [], "unused_from_checkpoint": [],
                       "follower_courtesy_init": "zero"}, "zero")
    bad = [
        ({"new_in_model": [], "expanded": [HEAD0], "follower_courtesy_init": "checkpoint"}, "zero"),
        ({"new_in_model": heads, "expanded": [], "follower_courtesy_init": "zero"}, "checkpoint"),
        ({"new_in_model": [], "expanded": [], "follower_courtesy_init": "checkpoint"}, "checkpoint"),
        ({"new_in_model": ["temporal.0.weight"], "expanded": [HEAD0], "follower_courtesy_init": "checkpoint"},
         "checkpoint"),
        ({"new_in_model": [], "expanded": [HEAD0, "lane_v5.x"], "follower_courtesy_init": "checkpoint"},
         "checkpoint"),
        ({"new_in_model": [], "expanded": [HEAD0], "shape_mismatch": ["x"], "follower_courtesy_init": "checkpoint"},
         "checkpoint"),
        ({"new_in_model": [], "expanded": [HEAD0], "follower_courtesy_init": "checkpoint"}, "other"),
    ]
    for report, expect in bad:
        with pytest.raises(ValueError):
            check_init_report(report, expect)


@pytest.fixture
def smoke(tmp_path):
    base = {"rollout_mode": "hybrid", "lane_model": "hazard_v5", "merge_decision": "first_passage",
            "merge_passage_law": "hybrid", "merge_pivot_state_st": True, "follower_courtesy": True}
    common = {"physics": {"ramp_lane_end_stop": True},
              "curriculum": {"unroll_s": [120.0, 300.0], "stage_steps": [2, 2]},
              "optim": {"batch_size": 4, "batch_size_per_stage": [4, 4]}}
    cfg10b = TrainConfig.from_dict({"model": base, **common})
    cfg = TrainConfig.from_dict({"model": {**base, "follower_courtesy_version": 2}, **common})
    physics = PhysicsConfig(ramp_lane_end_stop=True)
    torch.manual_seed(0)
    m10b = WmSModel(ModelSpec.from_mapping(cfg10b.to_legacy_flat()), physics=physics)
    with torch.no_grad():
        for p in m10b.follower_courtesy.parameters():
            p.add_(0.1)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    model.load_state_dict(expand_head_inputs(m10b.state_dict(), model.state_dict()))
    init = tmp_path / "init.pt"
    save_checkpoint(init, m10b, cfg10b)
    rows = [{"step": i + 1, "stage_s": stage, "loss": 0.5} for i, stage in enumerate((120, 120, 300, 300))]
    out = tmp_path / "smoke"
    out.mkdir()
    (out / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")

    def save(mutate=None):
        trained = copy.deepcopy(model)
        if mutate:
            with torch.no_grad():
                mutate(trained)
        for stage, step in ((120, 2), (300, 4)):
            save_checkpoint(out / f"wm_s_stage{stage}.pt", trained, cfg, extra={"stage_s": stage, "step": step})
        save_checkpoint(out / "wm_s.pt", trained, cfg, history=rows)
        return out, rows, init
    return save


def first(model, prefix):
    return next(p for n, p in model.named_parameters() if n.startswith(prefix))


def move_head(m):
    dict(m.named_parameters())["follower_courtesy.net.2.weight"].add_(0.01)


def test_temporal_and_head_may_change(smoke):
    def move(m):
        first(m, "temporal").add_(0.5)
        move_head(m)
    reports = check_checkpoints(*smoke(move), FREEZE)
    assert len(reports) == 3 and all(r["head_moved"] and r["head_from_init"] for r in reports)


def test_head_unchanged_from_round10b_init_fails(smoke):
    with pytest.raises(ValueError, match="did not move"):
        check_checkpoints(*smoke(lambda m: first(m, "temporal").add_(0.5)), FREEZE)


def test_unchanged_round10_set_fails(smoke):
    with pytest.raises(ValueError, match="no round-10 trainable tensor moved"):
        check_checkpoints(*smoke(move_head), FREEZE)


@pytest.mark.parametrize("prefix", ("merge_pivot", "residual_net", "driver_encoder"))
def test_frozen_or_pivot_tensor_change_refuses(smoke, prefix):
    def move(m):
        first(m, "temporal").add_(0.5)
        move_head(m)
        first(m, prefix).add_(1e-3)
    with pytest.raises(ValueError, match="stay frozen"):
        check_checkpoints(*smoke(move), FREEZE)
