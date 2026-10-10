from __future__ import annotations

import json

import pytest

from output.wd01.wm_n1.wms import gate_s2
from output.wd01.wm_n1.wms.check400 import DRIVER_PARAM_BOUNDS

STAGES = ["10", "30", "60", "120", "300"]


def _driver_stats():
    return {name: {"min": lo, "median": lo, "max": lo}
            for name, (lo, _hi) in DRIVER_PARAM_BOUNDS.items()}


def _row(step: int, stage_s: float, **overrides):
    row = {
        "step": step, "stage_s": stage_s, "finite": True, "gate_monotone": True,
        "gate_g0": 0.0, "gate_curve": [0.0, 0.1, 0.2], "valid_targets": 8,
        "valid_hdv_slots_t0": 4, "driver_param_stats": _driver_stats(),
        "loss": 1.0, "position": 0.5, "speed": 0.2, "lane": 0.1, "seconds": 0.5,
        "traffic": 0.05, "entrants": 1, "projection_active_total": 0.0,
        "conservation_metric": 0.0,
    }
    row.update(overrides)
    return row


def _write_log(path, stages=STAGES, per_stage=2, **overrides):
    rows = []
    step = 0
    for stage in stages:
        for _ in range(per_stage):
            step += 1
            rows.append(_row(step, float(stage), **overrides))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_gate_accepts_every_stage(tmp_path):
    log = _write_log(tmp_path / "train_log.jsonl")
    result = gate_s2.check(log, stages=STAGES, steps_per_stage=2)
    assert result["status"] == "passed"
    assert result["stages"] == [10, 30, 60, 120, 300]


def test_gate_rejects_missing_stage(tmp_path):
    log = _write_log(tmp_path / "train_log.jsonl", stages=["10", "30"])
    with pytest.raises(SystemExit):
        gate_s2.check(log, stages=STAGES, steps_per_stage=2)


def test_gate_rejects_wrong_record_count(tmp_path):
    log = _write_log(tmp_path / "train_log.jsonl", per_stage=3)
    with pytest.raises(SystemExit):
        gate_s2.check(log, stages=STAGES, steps_per_stage=2)


def test_gate_rejects_out_of_bounds_driver_param(tmp_path):
    log = _write_log(tmp_path / "train_log.jsonl")
    lines = log.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["driver_param_stats"]["accel"]["max"] = 99.0
    lines[0] = json.dumps(row)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        gate_s2.check(log, stages=STAGES, steps_per_stage=2)


def test_gate_rejects_nonfinite_term(tmp_path):
    log = _write_log(tmp_path / "train_log.jsonl")
    lines = log.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["traffic"] = float("nan")
    lines[0] = json.dumps(row)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        gate_s2.check(log, stages=STAGES, steps_per_stage=2)


def test_gate_rejects_absent_loss_family(tmp_path):
    log = _write_log(tmp_path / "train_log.jsonl")
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("projection_active_total")
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        gate_s2.check(log, stages=STAGES, steps_per_stage=2)
