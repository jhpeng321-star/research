"""Synthetic JSONL contract tests for the round-10 smoke verdict (frozen merge_pivot, cut-in reaction loss)."""
import json

import pytest

from wms.train.round10_gate import check_log


def make_rows():
    rows = []
    for index, stage in enumerate((120, 120, 300, 300)):
        rows.append({
            "step": index + 1, "stage_s": stage, "loss": 1.0 - 0.1 * index,
            "merge_pivot_on_objective_disabled": 1,
            "merge_pivot_tf_positive_mass_events": 2, "merge_pivot_tf_zero_mass_events": 0,
            "merge_pivot_tf_scored_seconds": 2.0, "merge_pivot_tf_scored_cars": 2,
            "merge_pivot_tf_excluded_motion": 0, "merge_pivot_tau": 0.5, "merge_pivot_w": 0.25,
            "merge_pivot_grad_norm": 0.0,  # frozen in round 10
            "cuda_peak_mb": 100.0 * (index + 1),
            "cutin_reaction": 0.9, "cutin_reaction_event_loss": 0.6, "cutin_reaction_control_loss": 0.3,
            "cutin_reaction_n_events": 3, "cutin_reaction_n_event_rows": 9, "cutin_reaction_n_control_rows": 100,
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


def test_valid_round10_log_passes_with_zero_pivot_gradient(run):
    result = run(make_rows())
    assert result["ok"], result["errors"]
    assert result["metrics"]["cuda_peak_mb"] == 400.0


def test_traceback_or_exit_code_fails(run):
    assert not run(make_rows(), text="Traceback (most recent call last)")["ok"]
    assert not run(make_rows(), exit_code=1)["ok"]


@pytest.mark.parametrize("field", ("cutin_reaction_n_events", "cutin_reaction_n_event_rows",
                                    "cutin_reaction_n_control_rows"))
def test_no_cutin_events_fails(run, field):
    rows = make_rows()
    for row in rows:
        row[field] = 0
    assert not run(rows)["ok"]


@pytest.mark.parametrize("field", ("cutin_reaction", "cutin_reaction_event_loss", "cutin_reaction_control_loss", "loss"))
@pytest.mark.parametrize("value", (None, float("nan"), float("inf"), True, "1"))
def test_nonfinite_cutin_or_loss_fails(run, field, value):
    rows = make_rows()
    rows[1][field] = value
    assert not run(rows)["ok"]


def test_zero_mass_flag_and_steps(run):
    rows = make_rows()
    rows[2]["merge_pivot_tf_zero_mass_events"] = 1
    assert not run(rows)["ok"]
    rows = make_rows()
    rows[0]["merge_pivot_on_objective_disabled"] = 0
    assert not run(rows)["ok"]
    assert not run(make_rows()[:3])["ok"]
    rows = make_rows()
    for row in rows:
        row["stage_s"] = 120
    assert not run(rows)["ok"]


# ---- checkpoint comparison against the round-9 init checkpoint -------------------------------------------------
import copy
from dataclasses import asdict

import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import save_checkpoint
from wms.physics.config import PhysicsConfig
from wms.train.round10_gate import check_checkpoints

FREEZE = ("driver_encoder", "driver_updater", "lane_gate_", "history_proj", "k_embedding",
          "lane_v5", "merge_core", "merge_pivot", "residual_net")


@pytest.fixture
def smoke(tmp_path):
    cfg = TrainConfig.from_dict({
        "model": {"rollout_mode": "hybrid", "lane_model": "hazard_v5", "merge_decision": "first_passage",
                  "merge_passage_law": "hybrid", "merge_pivot_state_st": True},
        "physics": {"ramp_lane_end_stop": True},
        "curriculum": {"unroll_s": [120.0, 300.0], "stage_steps": [2, 2]},
        "optim": {"batch_size": 4, "batch_size_per_stage": [4, 4]},
    })
    physics = PhysicsConfig(ramp_lane_end_stop=True)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    init = tmp_path / "init.pt"
    save_checkpoint(init, model, cfg)
    rows = [{"step": i + 1, "stage_s": stage, "loss": 0.5} for i, stage in enumerate((120, 120, 300, 300))]
    out = tmp_path / "smoke"
    out.mkdir()
    (out / "physics.json").write_text(__import__("json").dumps(asdict(physics)), encoding="utf-8")

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


def test_only_temporal_and_gate_scalars_may_change(smoke):
    def move(m):
        first(m, "temporal").add_(0.5)
    reports = check_checkpoints(*smoke(move), FREEZE)
    assert len(reports) == 3 and all(r["changed_tensors"] >= 1 for r in reports)


def test_unchanged_trainable_set_fails_final_checkpoint(smoke):
    with pytest.raises(ValueError, match="no trainable tensor moved"):
        check_checkpoints(*smoke(), FREEZE)


@pytest.mark.parametrize("prefix", ("merge_pivot", "residual_net", "driver_encoder"))
def test_frozen_or_pivot_tensor_change_refuses(smoke, prefix):
    def move(m):
        first(m, "temporal").add_(0.5)
        first(m, prefix).add_(1e-3)
    with pytest.raises(ValueError, match="stay frozen"):
        check_checkpoints(*smoke(move), FREEZE)
