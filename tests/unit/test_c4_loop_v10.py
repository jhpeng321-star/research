"""scripts/c4/loop_v10.py: the closed v10 Dreamer-loop orchestration (plan/dry-run/rollback).

No SUMO, no GPU, no ssh: every test runs the stage planner, the dry-run writer and the pure
decision helpers locally.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("loop_v10", REPO / "scripts" / "c4" / "loop_v10.py")
LV = importlib.util.module_from_spec(_spec)
sys.modules["loop_v10"] = LV
_spec.loader.exec_module(LV)

CFG = LV.load_config(REPO / "configs" / "c4_dreamer_loop_v10loop.json")


def _paths(tmp_path, k=1):
    return LV._paths(tmp_path, k, CFG)


def test_config_seed_block_is_disjoint_and_approved():
    lo, hi = CFG["seed"]["block"]
    assert lo == 969000 and hi == 969199
    assert not (lo <= LV.EVAL_SEEDS[1] and hi >= LV.EVAL_SEEDS[0])
    assert not (lo <= LV.CHECK_SEEDS[1] and hi >= LV.CHECK_SEEDS[0])
    assert any(a <= lo and hi <= b for a, b in LV.APPROVED_BLOCKS)
    assert [LV.CHECK_SEEDS[0], LV.CHECK_SEEDS[1]] in [list(x) for x in CFG["seed"]["exclude"]]


def test_reserved_seeds_are_refused():
    for seed in (964000, 964199, 969824, 969835, 4107):
        try:
            LV.check_seed(seed)
        except ValueError:
            continue
        raise AssertionError(f"seed {seed} should be refused")
    LV.check_seed(969000)


def test_allocate_skips_used_and_reserved():
    seeds = LV.allocate_seeds(CFG, used=[969000, 969001], n=3)
    assert seeds == [969002, 969003, 969004]
    assert not (set(seeds) & {LV.EVAL_SEEDS[0], LV.CHECK_SEEDS[0], 969000, 969001})


def test_plan_covers_every_stage():
    text = LV.plan_text(CFG)
    for name in (*LV.ITER0_STAGES, *LV.ITER_STAGES):
        assert name in text, name


def test_dry_run_exercises_all_stages_without_executing(tmp_path):
    result = LV.dry_run(CFG, tmp_path)
    names = {row["stage"] for row in result["records"]}
    assert set(LV.ITER0_STAGES) <= names
    assert set(LV.ITER_STAGES) <= names
    assert result["summary"]["executed_external_commands"] == 0
    assert result["summary"]["stage_count"] == len(result["records"])
    rows = [json.loads(line) for line in (tmp_path / "dry_run.jsonl").read_text().splitlines()]
    assert len(rows) == len(result["records"])
    assert (tmp_path / "dry_run_summary.json").exists()
    assert result["summary"]["rollback_self_test"]["action"] == "revert"


def test_pass_through_keys_are_verbatim(tmp_path):
    paths = _paths(tmp_path)
    stages = LV.plan_iteration(CFG, 1, paths, LV.allocate_seeds(CFG, n=2), 200)
    wm = next(s for s in stages if s["name"] == "wm_finetune")
    for key in CFG["wm"]["extra_set"]:
        assert key in wm["cmd"], key
    assert "WMS_DATA_V6_POLICY" in wm["env"]
    iter0 = LV.plan_iteration0(CFG, LV._paths(tmp_path, 0, CFG), 200)
    cmd0 = next(s for s in iter0 if s["name"] == "wm_finetune_iter0")["cmd"]
    assert "model.lane_learnable=true" in cmd0
    assert "data.mixed_replay.prioritised=true" not in cmd0


def test_policy_config_is_entrance_zone_native():
    assert CFG["policy"]["config"] == "configs/c4_policy_v10entzone_v5m.toml"
    assert CFG["collect"]["entrance_zone_native"] is True
    iter0 = LV.plan_iteration0(CFG, LV._paths(Path("."), 0, CFG), 200)
    policy = next(s for s in iter0 if s["name"] == "policy_update_iter0")
    assert "--resume" not in policy["cmd"]
    stages = LV.plan_iteration(CFG, 1, LV._paths(Path("."), 1, CFG), LV.allocate_seeds(CFG, n=2), 200)
    policy = next(s for s in stages if s["name"] == "policy_update")
    assert "--resume" in policy["cmd"]


def test_rollback_rule_keeps_and_halves():
    rule = CFG["rollback"]
    prev = {"sumo_total_D_delta": 0.0, "wm60_hdv_abs_ds": 10.0}
    keep = LV.decide_rollback(prev, dict(prev), rule, 200)
    assert keep["action"] == "keep" and keep["next_policy_updates"] == 200
    bad = {"sumo_total_D_delta": 0.05, "wm60_hdv_abs_ds": 11.5}
    reverted = LV.decide_rollback(prev, bad, rule, 200)
    assert reverted["action"] == "revert" and reverted["next_policy_updates"] == 100
    assert len(reverted["reasons"]) == 2
    floored = LV.decide_rollback(prev, {"sumo_total_D_delta": 1.0, "wm60_hdv_abs_ds": 99.0}, rule, 60)
    assert floored["next_policy_updates"] == rule["min_policy_updates"]


def test_sample_counter_tracks_mappo_fairness():
    records = [{"collect": {"episodes": 50, "sim_s": 60000.0, "horizon_s": 900},
                "budget": {"mappo_control_dt_s": 0.5}}]
    counter = LV.sample_counter(records)
    assert counter["train_episodes"] == 50
    assert counter["policy_sim_s"] == 60000.0
    assert counter["control_samples"] == 120000
    assert "MAPPO fairness" in counter["note"]


def test_budget_is_positive_and_iteration0_has_no_collection():
    b0 = LV.budget(CFG, 0)
    b1 = LV.budget(CFG, 1)
    assert b0["collect_episodes"] == 0 and b0["check_episodes"] == 0
    assert b0["gpu_hours_total"] > 0
    assert b1["collect_episodes"] == 50
    assert b1["check_episodes"] == 12
    assert b1["sim_s"] == 50 * (300 + 900)
    assert b1["gpu_hours_total"] > b0["gpu_hours_total"]


def test_wm_sanity_accepts_and_rejects(tmp_path):
    run = tmp_path / "wm"
    run.mkdir()
    (run / "train_log.jsonl").write_text(json.dumps({"step": 1, "loss": 1.0}) + "\n")
    (run / "val_log.jsonl").write_text("".join(
        json.dumps({"step": 10, "horizon_s": h, "mean_position": 10.0}) + "\n" for h in (60.0, 120.0)))
    ref = tmp_path / "ref.jsonl"
    ref.write_text("".join(
        json.dumps({"step": 5, "horizon_s": h, "mean_position": 10.0}) + "\n" for h in (60.0, 120.0)))
    assert LV.wm_sanity(run, ref, tolerance=0.30)["ok"] is True
    (run / "val_log.jsonl").write_text(json.dumps({"step": 10, "horizon_s": 60.0, "mean_position": 14.0}) + "\n")
    verdict = LV.wm_sanity(run, ref, tolerance=0.30)
    assert verdict["ok"] is False and "worse by" in verdict["reasons"][0]


def test_execute_refused_without_guard():
    assert LV.execute(CFG) == 3


def test_execute_logs_iteration_with_injected_runner(tmp_path):
    cfg = deepcopy(CFG)
    cfg["execute_guard"]["allow"] = True
    cfg["iterations"] = 0
    cfg["loop_root"] = str(tmp_path / "loop")
    seen = []

    def runner(stage, config):
        seen.append(stage["name"])
        return 0

    assert LV.execute(cfg, runner=runner) == 0
    assert set(LV.ITER0_STAGES) <= set(seen)
    row = json.loads((tmp_path / "loop" / "iterations.jsonl").read_text().splitlines()[0])
    assert row["iteration"] == 0 and row["seed_block"] == cfg["seed"]["block"]


def test_plan_document_has_required_sections():
    doc = REPO / "docs" / "C4_V10LOOP_PLAN.md"
    assert doc.exists(), "docs/C4_V10LOOP_PLAN.md must exist"
    text = doc.read_text(encoding="utf-8")
    for heading in ("Seed block", "Stages and commands", "Budget", "Rollback", "Dry-run"):
        assert heading.lower() in text.lower(), heading
