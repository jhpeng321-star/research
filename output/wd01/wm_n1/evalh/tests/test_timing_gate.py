"""Timing-gate tests (CPU, --tiny): throughput, extrapolation, loss decrease."""
from __future__ import annotations

import json

import pytest

from evalh import timing_gate as T


def test_tiny_gate_runs_on_cpu(tmp_path):
    cfg = T.make_config(tiny=True)
    result = T.run_gate(cfg, device="cpu", tiny=True, warmup=1, timed_steps=2,
                        loss_steps=25)
    assert result["schema"] == "wm_n1_timing_gate/1"
    assert result["throughput"]["steps_per_sec"] > 0
    assert result["throughput"]["rollout_samples_per_sec"] > 0
    assert isinstance(result["passed"], bool)
    assert result["loss_decrease"]["decreased"] is True
    assert result["checks"]["within_gate"] is True
    assert result["thresholds"]["budget_hours_max"] == T.BUDGET_HOURS_MAX
    assert "WM_N1_DESIGN.md:226" in result["thresholds"]["citation"]


def test_extrapolation_monotonic_in_throughput():
    cfg = T.make_config()
    slow = T.extrapolate(cfg, dict(steps_per_sec=1.0))
    fast = T.extrapolate(cfg, dict(steps_per_sec=1000.0))
    assert slow["total_hours"] > fast["total_hours"]
    assert fast["total_hours"] <= T.BUDGET_HOURS_MAX
    assert fast["total_hours"] == pytest.approx(
        fast["train_hours"] * (1.0 + T.ROLLOUT_FRACTION) + T.EVAL_HOURS)


def test_extrapolation_infeasible_on_zero_throughput():
    cfg = T.make_config()
    r = T.extrapolate(cfg, dict(steps_per_sec=0.0))
    assert r["feasible"] is False
    assert r["total_hours"] == float("inf")


def test_gate_cli_writes_json(tmp_path):
    out = tmp_path / "TIMING_GATE.json"
    rc = T.main(["--out", str(out), "--device", "cpu", "--tiny", "--warmup",
                 "1", "--timed-steps", "2", "--loss-steps", "25"])
    assert rc in (0, 1)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tiny"] is True
    assert "passed" in data and "extrapolation" in data
