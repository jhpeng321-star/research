"""Synthetic contract tests for the round-12 smoke verdict (round 9 frozen + base calibration + R11 courtesy head)."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import save_checkpoint, sha256_file
from wms.physics.config import PhysicsConfig
from wms.train import round12_gate
from wms.train.round12_gate import NEW_KEYS, check_checkpoints, check_init_report, check_log

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/c4_wm_v10round12.toml"
FREEZE = ("driver_encoder.history_proj", "driver_encoder.query", "driver_encoder.interaction_proj",
          "driver_encoder.history_gru", "driver_updater", "lane_gate_", "history_proj", "k_embedding",
          "lane_v5", "merge_core", "merge_pivot", "residual_net.lane")
MEANS = {"accel": 1.4, "decel": 2.2, "time_headway": 1.3, "min_gap": 2.1, "v0_ratio": 1.0}


def make_rows():
    rows = []
    for index, stage in enumerate((120, 120, 300, 300)):
        rows.append({
            "step": index + 1, "stage_s": stage, "loss": 1.0 - 0.1 * index,
            "merge_pivot_on_objective_disabled": 1,
            "merge_pivot_tf_positive_mass_events": 2, "merge_pivot_tf_zero_mass_events": 0,
            "merge_pivot_tf_scored_seconds": 2.0, "merge_pivot_tf_scored_cars": 2,
            "merge_pivot_tf_excluded_motion": 0, "merge_pivot_tau": 0.5, "merge_pivot_w": 0.25,
            "merge_pivot_grad_norm": 0.0,  # frozen
            "cuda_peak_mb": 100.0 * (index + 1),
            "follower_courtesy": 0.4, "follower_courtesy_loss_raw": 0.2, "follower_courtesy_n_rows": 500,
            "follower_courtesy_shared_replay": 0,
            "follower_courtesy_contrast_n_controls": 4000, "follower_courtesy_contrast_shift_mean": -0.2,
            "follower_courtesy_head_min": -1.9, "follower_courtesy_head_max": 1.6,
            "follower_courtesy_head_mean": 0.15,
            "follower_courtesy_replay_filled_extra_rows": 7 * index,
            "driver_prior": 0.01, "driver_prior_means": dict(MEANS, accel=1.4 + 0.01 * index),
        })
    return rows


@pytest.fixture
def run(tmp_path):
    def go(rows, *, text="ok\n", exit_code=0):
        log, smoke = tmp_path / "train_log.jsonl", tmp_path / "smoke.log"
        log.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in rows), encoding="utf-8")
        smoke.write_text(text, encoding="utf-8")
        return check_log(log, smoke, train_exit_code=exit_code, accel_bounds=(-9.0, 3.0))
    return go


def test_valid_round12_log_passes(run):
    result = run(make_rows())
    assert result["ok"], result["errors"]
    m = result["metrics"]
    assert m["replay_filled_extra_rows_total"] == 42
    assert m["contrast_n_controls"] == 16000 and m["follower_courtesy_n_rows"] == 2000
    assert m["driver_prior_means_first"]["accel"] == 1.4
    assert m["driver_prior_means_last"]["accel"] == pytest.approx(1.43)
    assert m["head_range_bounds"] == [-9.0, 3.0]


def test_zero_replay_total_is_reported_not_failed(run):
    rows = make_rows()
    for row in rows:
        row["follower_courtesy_replay_filled_extra_rows"] = 0
    result = run(rows)
    assert result["ok"], result["errors"]
    assert result["metrics"]["replay_filled_extra_rows_total"] == 0


def test_merge_and_courtesy_failures_still_fail(run):
    for key, value in (("merge_pivot_tf_zero_mass_events", 1), ("loss", float("nan")),
                       ("merge_pivot_tau", 0.0), ("follower_courtesy", None)):
        rows = make_rows()
        rows[1][key] = value
        assert not run(rows)["ok"], key
    rows = make_rows()
    for row in rows:
        row["follower_courtesy_contrast_n_controls"] = 0
    assert not run(rows)["ok"]
    rows = make_rows()
    for row in rows:
        row["follower_courtesy_n_rows"] = 0
    assert not run(rows)["ok"]
    rows = make_rows()
    rows[2]["follower_courtesy_head_mean"] = -37.6
    assert any("saturated head" in e for e in run(rows)["errors"])
    assert not run(make_rows(), text="Traceback (most recent call last)")["ok"]
    assert not run(make_rows(), exit_code=1)["ok"]
    assert not run(make_rows()[:3])["ok"]


@pytest.mark.parametrize("key,value", (("cutin_reaction", 0.9), ("cutin_reaction_n_events", 3),
                                       ("cutin_reaction_n_partnered_controls_excluded", 12),
                                       ("cutin_reaction", None), ("cutin_reaction", float("nan"))))
def test_cutin_term_present_fails(run, key, value):
    rows = make_rows()
    rows[2][key] = value
    result = run(rows)
    assert not result["ok"] and any("must be absent" in e for e in result["errors"])


def test_cutin_exact_zero_is_tolerated(run):
    rows = make_rows()
    for row in rows:
        row["cutin_reaction"] = 0.0
    assert run(rows)["ok"]


@pytest.mark.parametrize("value", ("missing", None, float("nan"), -1, 1.5, True, "3"))
def test_missing_replay_telemetry_fails(run, value):
    rows = make_rows()
    if value == "missing":
        del rows[1]["follower_courtesy_replay_filled_extra_rows"]
    else:
        rows[1]["follower_courtesy_replay_filled_extra_rows"] = value
    result = run(rows)
    assert not result["ok"] and any("fix D" in e for e in result["errors"])


def test_replay_telemetry_not_required_without_courtesy_rows(run):
    rows = make_rows()
    rows[0]["follower_courtesy_n_rows"] = 0
    del rows[0]["follower_courtesy_replay_filled_extra_rows"]
    assert run(rows)["ok"]


@pytest.mark.parametrize("key,value", (
    ("accel", 0.3), ("accel", 4.0), ("accel", 4.5), ("decel", 0.8), ("decel", 6.0),
    ("time_headway", 0.4), ("time_headway", 3.0), ("min_gap", 0.5), ("min_gap", 5.0),
    ("v0_ratio", 0.6), ("v0_ratio", 1.4), ("v0_ratio", float("nan")), ("decel", None), ("min_gap", "missing")))
def test_driver_means_at_or_outside_bound_fail(run, key, value):
    rows = make_rows()
    if value == "missing":
        del rows[3]["driver_prior_means"][key]
    else:
        rows[3]["driver_prior_means"][key] = value
    result = run(rows)
    assert not result["ok"] and any("driver_prior_means" in e for e in result["errors"])


@pytest.mark.parametrize("value", ("missing", {}, None, [1.0]))
def test_driver_means_missing_fail(run, value):
    rows = make_rows()
    if value == "missing":
        del rows[0]["driver_prior_means"]
    else:
        rows[0]["driver_prior_means"] = value
    assert not run(rows)["ok"]


def good_init(**over):
    report = {"source": "/x/wm_s_stage300.pt", "sha256": "abc", "loaded": ["a", "b"], "expanded": [],
              "new_in_model": list(reversed(NEW_KEYS)), "shape_mismatch": [], "unused_from_checkpoint": [],
              "follower_courtesy_init": "zero",
              "driver_param_init": {"mode": "centre", "zeroed_rows": [0, 1, 2, 3, 4],
                                    "tensor": "driver_encoder.head.2", "centre_logits": [0.1, -0.9, -0.1, 0.4, 0.6]},
              "residual_accel_reinit": {"tensor": "residual_net.accel.2", "zeroed_rows": [0], "abs_sum_before": 3.0}}
    report.update(over)
    return report


def test_init_report_contract():
    check_init_report(good_init(), init_keys=["b", "a"], init_sha256="abc")
    assert len(NEW_KEYS) == 9


@pytest.mark.parametrize("over", (
    {"shape_mismatch": ["x"]}, {"unused_from_checkpoint": ["x"]}, {"expanded": ["follower_courtesy.net.0.weight"]},
    {"new_in_model": list(NEW_KEYS)[:-1]}, {"new_in_model": [*NEW_KEYS, "temporal.0.weight"]},
    {"new_in_model": None}, {"follower_courtesy_init": "checkpoint"}, {"follower_courtesy_init": None},
    {"driver_param_init": None}, {"driver_param_init": {"mode": "checkpoint", "zeroed_rows": [0, 1, 2, 3, 4],
                                                         "tensor": "driver_encoder.head.2",
                                                         "centre_logits": [0.0] * 5}},
    {"driver_param_init": {"mode": "centre", "zeroed_rows": [0, 1, 2, 3], "tensor": "driver_encoder.head.2",
                           "centre_logits": [0.0] * 5}},
    {"driver_param_init": {"mode": "centre", "zeroed_rows": [0, 1, 2, 3, 4], "tensor": "driver_encoder.head.2",
                           "centre_logits": [0.0] * 4}},
    {"residual_accel_reinit": None}, {"residual_accel_reinit": {"tensor": "residual_net.accel.2", "zeroed_rows": []}},
    {"loaded": ["a"]}, {"sha256": "other"},
))
def test_bad_init_report_variants_fail(over):
    with pytest.raises(ValueError):
        check_init_report(good_init(**over), init_keys=["a", "b"], init_sha256="abc")


@pytest.fixture(scope="module")
def models():
    base = {"rollout_mode": "hybrid", "lane_model": "hazard_v5", "merge_decision": "first_passage",
            "merge_passage_law": "hybrid", "merge_pivot_state_st": True}
    common = {"physics": {"ramp_lane_end_stop": True},
              "curriculum": {"unroll_s": [120.0, 300.0], "stage_steps": [2, 2]},
              "optim": {"batch_size": 4, "batch_size_per_stage": [4, 4]}}
    cfg9 = TrainConfig.from_dict({"model": base, **common})
    cfg = TrainConfig.from_dict({"model": {**base, "follower_courtesy": True, "follower_courtesy_version": 2,
                                           "driver_param_mode": "physical"}, **common})
    physics = PhysicsConfig(ramp_lane_end_stop=True)
    torch.manual_seed(0)
    m9 = WmSModel(ModelSpec.from_mapping(cfg9.to_legacy_flat()), physics=physics)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in m9.state_dict().items() if k in own and own[k].shape == v.shape},
                          strict=False)
    with torch.no_grad():   # the trainer's R12 re-init
        model.driver_encoder.head[-1].weight[:5].zero_()
        model.driver_encoder.head[-1].bias[:5].zero_()
        model.residual_net.accel[-1].weight[0].zero_()
        model.residual_net.accel[-1].bias[0].zero_()
    return m9, cfg9, model, cfg, physics


@pytest.fixture
def smoke(tmp_path, models):
    m9, cfg9, model, cfg, physics = models
    init = tmp_path / "init.pt"
    save_checkpoint(init, m9, cfg9)
    report = good_init(sha256=sha256_file(init), loaded=sorted(m9.state_dict()))
    report["driver_param_init"]["centre_logits"] = model.driver_encoder.centre_logits.detach().tolist()
    check_init_report(report, init_keys=m9.state_dict().keys(), init_sha256=sha256_file(init))
    rows = [{"step": i + 1, "stage_s": stage, "loss": 0.5} for i, stage in enumerate((120, 120, 300, 300))]
    out = tmp_path / "smoke"
    out.mkdir()
    (out / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")

    def save(mutate=train):
        trained = copy.deepcopy(model)
        with torch.no_grad():
            mutate(trained)
        for stage, step in ((120, 2), (300, 4)):
            save_checkpoint(out / f"wm_s_stage{stage}.pt", trained, cfg, extra={"stage_s": stage, "step": step})
        save_checkpoint(out / "wm_s.pt", trained, cfg, history=rows)
        return out, rows, init, FREEZE, report
    return save


def param(m, name):
    return dict(m.named_parameters())[name]


def train(m, skip=()):
    """A plausible trained state: every re-initialised/new trainable piece moved."""
    if "head" not in skip:
        param(m, "follower_courtesy.net.4.weight").add_(0.01)
    if "centre" not in skip:
        param(m, "driver_encoder.centre_logits").add_(0.01)
    if "accel" not in skip:
        param(m, "residual_net.accel.2.bias")[0] += 0.01
    if "rows" not in skip:
        param(m, "driver_encoder.head.2.weight")[:5] += 0.01
    elif "rows_partial" in skip:
        param(m, "driver_encoder.head.2.weight")[:4] += 0.01
    next(p for n, p in m.named_parameters() if n.startswith("temporal")).add_(0.5)


def test_trained_checkpoints_pass(smoke):
    reports = check_checkpoints(*smoke())
    assert len(reports) == 3
    assert all(r["head_moved"] and r["centre_moved"] and r["accel_row0_moved"] and all(r["driver_head_rows_moved"])
               for r in reports)


@pytest.mark.parametrize("skip,match", ((("head",), "courtesy head"), (("centre",), "centre"),
                                        (("accel",), "row 0"), (("rows",), "rows"),
                                        (("rows", "rows_partial"), r"rows \[4\]")))
def test_untrained_reinit_pieces_fail(smoke, skip, match):
    with pytest.raises(ValueError, match=match):
        check_checkpoints(*smoke(lambda m: train(m, skip)))


@pytest.mark.parametrize("name", ("merge_pivot", "driver_encoder.history_proj", "residual_net.lane",
                                  "lane_v5", "driver_encoder.query"))
def test_frozen_or_pivot_tensor_change_fails(smoke, name):
    def move(m):
        train(m)
        next(p for n, p in m.named_parameters() if n.startswith(name)).add_(1e-3)
    with pytest.raises(ValueError, match="stay frozen"):
        check_checkpoints(*smoke(move))


def test_physical_bounds_must_equal_constants(smoke):
    def move(m):
        train(m)
        m.driver_encoder.physical_upper[1] += 0.5
    with pytest.raises(ValueError, match="physical_lower/upper"):
        check_checkpoints(*smoke(move))


def test_history_and_stage_step_mismatch_fail(smoke):
    out, rows, init, freeze, report = smoke()
    with pytest.raises(ValueError, match="history"):
        check_checkpoints(out, [*rows[:3], {**rows[3], "loss": 0.4}], init, freeze, report)
    with pytest.raises(ValueError, match="stage/step"):
        check_checkpoints(out, [*rows, {"step": 5, "stage_s": 300, "loss": 0.5}], init, freeze, report)


def test_cli_writes_report_and_fails_closed(tmp_path, capsys, smoke):
    out, rows, init, _, report = smoke()
    (out / "INIT_FROM.json").write_text(json.dumps(report), encoding="utf-8")
    (out / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in make_rows()), encoding="utf-8")
    smoke_log = tmp_path / "smoke.log"
    smoke_log.write_text("ok\n", encoding="utf-8")
    args = ["--smoke-dir", str(out), "--smoke-log", str(smoke_log), "--train-exit-code", "0",
            "--init-checkpoint", str(init), "--config", str(CONFIG)]
    # checkpoint history = 4 bare rows != the 4 log rows: the checkpoint stage must fail closed
    assert round12_gate.main([*args, "--report", str(tmp_path / "a.json")]) == 3
    assert "smoke: bad" in capsys.readouterr().out
    result = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert not result["ok"] and any("checkpoint/init check failed" in e for e in result["errors"])
    # the report is never overwritten
    assert round12_gate.main([*args, "--report", str(tmp_path / "a.json")]) == 3
    assert "bad report" in capsys.readouterr().out


def test_cli_pass(tmp_path, capsys, models):
    m9, cfg9, model, cfg, physics = models
    rows = make_rows()
    out = tmp_path / "smoke"
    out.mkdir()
    (out / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")
    init = tmp_path / "init.pt"
    save_checkpoint(init, m9, cfg9)
    report = good_init(sha256=sha256_file(init), loaded=sorted(m9.state_dict()))
    report["driver_param_init"]["centre_logits"] = model.driver_encoder.centre_logits.detach().tolist()
    trained = copy.deepcopy(model)
    with torch.no_grad():
        train(trained)
    for stage, step in ((120, 2), (300, 4)):
        save_checkpoint(out / f"wm_s_stage{stage}.pt", trained, cfg, extra={"stage_s": stage, "step": step})
    save_checkpoint(out / "wm_s.pt", trained, cfg, history=rows)
    (out / "INIT_FROM.json").write_text(json.dumps(report), encoding="utf-8")
    (out / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    smoke_log = tmp_path / "smoke.log"
    smoke_log.write_text("ok\n", encoding="utf-8")
    code = round12_gate.main(["--smoke-dir", str(out), "--smoke-log", str(smoke_log), "--train-exit-code", "0",
                              "--init-checkpoint", str(init), "--config", str(CONFIG),
                              "--report", str(tmp_path / "ok.json")])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert printed.strip() == "smoke: ok"
    result = json.loads((tmp_path / "ok.json").read_text(encoding="utf-8"))
    assert result["ok"] and len(result["checkpoints"]) == 3
