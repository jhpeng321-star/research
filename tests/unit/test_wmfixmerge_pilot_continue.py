"""MAIN 11:52 continuation helpers (determinism record + relayed panel schema)."""
import json

import pytest

from tools.wmfixmerge.pilot import HORIZONS, decide
from tools.wmfixmerge.pilot_continue import (is_pilot_run, step0_panel_runs,
                                             validate_step_panel, wait_for_step_panels)


def test_pilot_run_marker_check(tmp_path):
    tmp_path.joinpath("RUN.json").write_text(
        json.dumps(dict(pid=1, pilot="S-WMFIX-PILOT option (b), MAIN 10:40")))
    assert is_pilot_run(tmp_path) is True
    tmp_path.joinpath("RUN.json").write_text(json.dumps(dict(pid=1, pilot="other")))
    assert is_pilot_run(tmp_path) is False


def panel_doc(step, mae120, mae300, rows=24):
    return dict(step=step, seconds=1710.0,
                panel={str(h): dict(paired_J_MAE=mae120 if h == 120 else mae300,
                                    per_group={str(g): float(g) for g in range(12)},
                                    rows=rows) for h in HORIZONS})


def test_identical_step0_panels_give_zero_noise(tmp_path):
    for name in ("PANEL_step0_r0.json", "PANEL_step0_r1.json"):
        tmp_path.joinpath(name).write_text(json.dumps(panel_doc(0, 283.25, 2185.25)))
    runs, files = step0_panel_runs(tmp_path)
    assert [p.name for p in files] == ["PANEL_step0_r0.json", "PANEL_step0_r1.json"]
    assert runs == {120: [283.25, 283.25], 300: [2185.25, 2185.25]}
    decision = decide(runs, {10: {120: 200.0, 300: 1500.0}})
    assert decision["sample_sd"] == {120: 0.0, 300: 0.0}
    assert decision["passed"] is True  # strict improvement with SD 0


def test_disagreeing_step0_panels_refuse(tmp_path):
    tmp_path.joinpath("PANEL_step0_r0.json").write_text(json.dumps(panel_doc(0, 283.25, 2185.25)))
    tmp_path.joinpath("PANEL_step0_r1.json").write_text(json.dumps(panel_doc(0, 283.25, 2189.0)))
    with pytest.raises(ValueError, match="determinism unproven"):
        step0_panel_runs(tmp_path)


def test_single_step0_panel_refuses(tmp_path):
    tmp_path.joinpath("PANEL_step0_r0.json").write_text(json.dumps(panel_doc(0, 283.25, 2185.25)))
    with pytest.raises(ValueError, match="fewer than two"):
        step0_panel_runs(tmp_path)


@pytest.mark.parametrize("mutate", [dict(step=7), dict(drop_horizon=True), dict(rows=12)])
def test_bad_panel_schema_rejected(tmp_path, mutate):
    doc = panel_doc(10, 200.0, 1500.0)
    if mutate.get("drop_horizon"):
        doc["panel"].pop("300")
    if "rows" in mutate:
        doc["panel"]["120"]["rows"] = mutate["rows"]
    if "step" in mutate:
        doc["step"] = mutate["step"]
    with pytest.raises(ValueError):
        validate_step_panel(doc, 10)


def test_wait_for_step_panels_collects_relayed_receipts(tmp_path, monkeypatch):
    for tag in (10, 20, 30):
        tmp_path.joinpath(f"PANEL_step{tag}.json").write_text(
            json.dumps(panel_doc(tag, 200.0 + tag, 1500.0 + tag)))
    monkeypatch.setattr("time.sleep", lambda s: None)
    evals = wait_for_step_panels(tmp_path, deadline=0.0)
    assert evals == {10: {120: 210.0, 300: 1510.0},
                     20: {120: 220.0, 300: 1520.0},
                     30: {120: 230.0, 300: 1530.0}}


def test_wait_for_step_panels_times_out_with_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(TimeoutError, match="10,20,30"):
        wait_for_step_panels(tmp_path, deadline=0.0)
