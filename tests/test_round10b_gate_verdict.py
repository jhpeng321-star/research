"""Synthetic contract tests for the round-10b smoke verdict (round 10 + learned follower courtesy)."""
import copy
from dataclasses import asdict
import json

import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import save_checkpoint
from wms.physics.config import PhysicsConfig
from wms.train.round10b_gate import HEAD_LAST_WEIGHT, check_checkpoints, check_init_report, check_log

FREEZE = ("driver_encoder", "driver_updater", "lane_gate_", "history_proj", "k_embedding",
          "lane_v5", "merge_core", "merge_pivot", "residual_net")


def make_rows():
    rows = []
    for index, stage in enumerate((120, 120, 300, 300)):
        rows.append({
            "step": index + 1, "stage_s": stage, "loss": 1.0 - 0.1 * index,
            "merge_pivot_on_objective_disabled": 1,
            "merge_pivot_tf_positive_mass_events": 2, "merge_pivot_tf_zero_mass_events": 0,
            "merge_pivot_tf_scored_seconds": 2.0, "merge_pivot_tf_scored_cars": 2,
            "merge_pivot_tf_excluded_motion": 0, "merge_pivot_tau": 0.5, "merge_pivot_w": 0.25,
            "merge_pivot_grad_norm": 0.0,
            "cuda_peak_mb": 100.0 * (index + 1),
            "cutin_reaction": 0.9, "cutin_reaction_event_loss": 0.6, "cutin_reaction_control_loss": 0.3,
            "cutin_reaction_n_events": 3, "cutin_reaction_n_event_rows": 9, "cutin_reaction_n_control_rows": 100,
            "follower_courtesy": 0.4, "follower_courtesy_loss_raw": 0.2, "follower_courtesy_n_rows": 500,
            "follower_courtesy_shared_replay": 1,
        })
    return rows


@pytest.fixture
def run(tmp_path):
    def go(rows, *, text="ok\n", exit_code=0):
        log, smoke = tmp_path / "train_log.jsonl", tmp_path / "smoke.log"
        log.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in rows), encoding="utf-8")
        smoke.write_text(text, encoding="utf-8")
        return check_log(log, smoke, train_exit_code=exit_code)
    return go


def test_valid_round10b_log_passes(run):
    result = run(make_rows())
    assert result["ok"], result["errors"]
    assert result["metrics"]["follower_courtesy_n_rows"] == 2000
    assert result["metrics"]["follower_courtesy_shared_replay"] == 4


def test_round10_failures_still_fail(run):
    rows = make_rows()
    for row in rows:
        row["cutin_reaction_n_events"] = 0
    assert not run(rows)["ok"]
    assert not run(make_rows(), text="Traceback (most recent call last)")["ok"]


def test_no_courtesy_rows_or_zero_loss_fails(run):
    rows = make_rows()
    for row in rows:
        row["follower_courtesy_n_rows"] = 0
    assert not run(rows)["ok"]
    rows = make_rows()
    for row in rows:
        row["follower_courtesy"] = 0.0
    assert not run(rows)["ok"]


@pytest.mark.parametrize("field", ("follower_courtesy", "follower_courtesy_loss_raw", "follower_courtesy_n_rows"))
@pytest.mark.parametrize("value", (None, float("nan"), float("inf"), True, "1", -1.0))
def test_nonfinite_or_missing_courtesy_fails(run, field, value):
    rows = make_rows()
    rows[2][field] = value
    assert not run(rows)["ok"]


def test_init_report_allows_only_the_head_as_new():
    head = ["follower_courtesy.net.0.weight", "follower_courtesy.net.4.weight"]
    check_init_report({"new_in_model": head, "shape_mismatch": [], "unused_from_checkpoint": []})
    for bad in ({"new_in_model": [], "shape_mismatch": [], "unused_from_checkpoint": []},
                {"new_in_model": head + ["temporal.0.weight"], "shape_mismatch": [], "unused_from_checkpoint": []},
                {"new_in_model": head, "shape_mismatch": ["x"], "unused_from_checkpoint": []},
                {"new_in_model": head, "shape_mismatch": [], "unused_from_checkpoint": ["y"]}):
        with pytest.raises(ValueError):
            check_init_report(bad)


@pytest.fixture
def smoke(tmp_path):
    base = {"rollout_mode": "hybrid", "lane_model": "hazard_v5", "merge_decision": "first_passage",
            "merge_passage_law": "hybrid", "merge_pivot_state_st": True}
    common = {"physics": {"ramp_lane_end_stop": True},
              "curriculum": {"unroll_s": [120.0, 300.0], "stage_steps": [2, 2]},
              "optim": {"batch_size": 4, "batch_size_per_stage": [4, 4]}}
    cfg9 = TrainConfig.from_dict({"model": base, **common})
    cfg = TrainConfig.from_dict({"model": {**base, "follower_courtesy": True}, **common})
    physics = PhysicsConfig(ramp_lane_end_stop=True)
    torch.manual_seed(0)
    model9 = WmSModel(ModelSpec.from_mapping(cfg9.to_legacy_flat()), physics=physics)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    model.load_state_dict(model9.state_dict(), strict=False)
    init = tmp_path / "init.pt"
    save_checkpoint(init, model9, cfg9)
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
    dict(m.named_parameters())[HEAD_LAST_WEIGHT].add_(0.01)


def test_temporal_and_head_may_change(smoke):
    def move(m):
        first(m, "temporal").add_(0.5)
        move_head(m)
    reports = check_checkpoints(*smoke(move), FREEZE)
    assert len(reports) == 3 and all(r["head_moved"] for r in reports)


def test_untrained_head_fails_the_final_checkpoint(smoke):
    with pytest.raises(ValueError, match="still exactly zero"):
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
