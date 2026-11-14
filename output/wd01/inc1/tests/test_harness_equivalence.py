"""A1.5 harness-equivalence smoke tests (zero SUMO, synthetic rows)."""
from __future__ import annotations

from inc1 import harness_equivalence as HE


def _rows(speeds, *, steps=10, teleports=(), collisions=()):
    rows = []
    for k in range(steps):
        vehs = {f"v{i}": {"lane_id": "1191554636_0", "lane_position_m": 100.0 + k,
                          "speed_mps": v} for i, v in enumerate(speeds)}
        rows.append({"time_s": 300.0 + k, "vehicles": vehs,
                     "starting_teleports": list(teleports) if k == 0 else [],
                     "ending_teleports": [], "collisions": list(collisions)
                     if k == 0 else []})
    return rows


def test_identical_tracks_pass():
    rows = _rows([20.0, 15.0])
    verdict = HE.evaluate(rows, rows)
    assert verdict["verdict"] == "PASS"
    assert verdict["identical_trajectories"] is True
    assert verdict["teleports"] == 0 and verdict["collisions"] == 0


def test_delay_within_half_percent_passes():
    runner = _rows([20.0, 15.0])
    driver = _rows([20.0, 14.99])
    verdict = HE.evaluate(driver, runner)
    assert verdict["identical_trajectories"] is False
    assert verdict["delay_rel_diff"] <= 0.005
    assert verdict["verdict"] == "PASS"


def test_delay_outside_half_percent_fails():
    runner = _rows([20.0, 15.0])
    driver = _rows([20.0, 10.0])
    verdict = HE.evaluate(driver, runner)
    assert verdict["delay_rel_diff"] > 0.005
    assert verdict["verdict"] == "FAIL"


def test_teleport_or_collision_fails():
    rows = _rows([20.0])
    assert HE.evaluate(rows, rows)["verdict"] == "PASS"
    tp = _rows([20.0], teleports=("v0",))
    assert HE.evaluate(tp, rows)["verdict"] == "FAIL"
    col = _rows([20.0], collisions=("v0",))
    assert HE.evaluate(rows, col)["verdict"] == "FAIL"


def test_verdict_file_round_trip(tmp_path):
    driver_dir = tmp_path / "driver"
    runner_dir = tmp_path / "runner"
    driver_dir.mkdir()
    runner_dir.mkdir()
    for name, speeds in ((driver_dir, [20.0, 15.0]),
                         (runner_dir, [20.0, 15.0])):
        (name / "physics_steps.jsonl").write_text(
            "\n".join(__import__("json").dumps(r) for r in _rows(speeds)) + "\n",
            encoding="utf-8")
    payload = HE.compare_run_dirs(driver_dir, runner_dir,
                                  out_dir=tmp_path / "out")
    assert payload["verdict"] == "PASS"
    assert HE.verdict_is_pass(tmp_path / "out" / HE.VERDICT_NAME) is True
    assert HE.verdict_is_pass(tmp_path / "missing.json") is False
