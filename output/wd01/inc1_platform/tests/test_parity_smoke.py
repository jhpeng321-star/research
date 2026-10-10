"""inc1_platform parity_smoke tests: the comparison logic against synthetic
physics rows and a FAKE runner.  ZERO SUMO here; the real PARITY_INPUTS.json
worlds are never opened (only their names/schema rules are pinned)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
for _p in (str(PKG), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parity_smoke as PS  # noqa: E402
import fake_parity_runner  # noqa: E402

WORLDS = ("w8659m2700p10t420", "w8670m2700p10t300")

MAIN = ("1487606178", "1487606178_0")          # phys lane 1, lim 31.29
RAMP = ("11553048#1", "11553048#1_0")          # phys lane 2, lim 22.22
JUNCTION = (":j12", ":j12_0")                  # internal lane (skipped)


def veh(road, lane, pos, speed):
    return {"road_id": road, "lane_id": lane, "lane_position_m": pos,
            "speed_mps": speed}


def row(t, vehicles, colliding=None, teleports=None):
    return {"time_s": t, "vehicles": vehicles,
            "colliding_ids": colliding or [],
            "starting_teleports": teleports or []}


def cruise_rows(t0, ds_a=15.0, ds_b=10.0, drift=0.0):
    """4 half-steps on (t0, t0+2]: one mainline car (30 m/s), one ramp car
    (20 m/s); positions consistent with speeds (pos += v*DT per step)."""
    rows = []
    for k in (1, 2, 3, 4):
        t = t0 + k * PS.DT
        rows.append(row(t, {
            "carA": veh(MAIN[0], MAIN[1],
                        100.0 + (ds_a - drift) * k, 30.0),
            "carB": veh(RAMP[0], RAMP[1],
                        50.0 + (ds_b - drift) * k, 20.0)}))
    return rows


def slow_rows(t0, drift=0.0):
    """Congested case: carA 10 m/s, carB 5 m/s (large, stable delay)."""
    rows = []
    for k in (1, 2, 3, 4):
        t = t0 + k * PS.DT
        rows.append(row(t, {
            "carA": veh(MAIN[0], MAIN[1], 100.0 + (5.0 - drift) * k, 10.0),
            "carB": veh(RAMP[0], RAMP[1], 50.0 + (2.5 - drift) * k, 5.0)}))
    return rows


def window(rows, t0=0.0, horizon=2.0):
    return PS.rows_to_window(iter(rows), "w", t0, horizon)


# ------------------------------------------------------- pure comparison

def test_identical_runs_pass_with_zero_deltas():
    rows = cruise_rows(0.0)
    res = PS.compare_windows(window(rows), window(rows))
    assert res.passed
    assert res.max_abs_ds_m == 0.0 and res.max_abs_dv_mps == 0.0
    assert res.delay_diff_pct == 0.0
    assert res.vehicle_ticks_compared == 8


def test_position_delta_reported_but_within_gate():
    res = PS.compare_windows(window(slow_rows(0.0)),
                             window(slow_rows(0.0, drift=0.01)))
    assert res.passed                       # tiny drift, delay within 2%
    assert res.max_abs_ds_m == pytest.approx(0.04)
    assert res.max_abs_dv_mps == 0.0
    assert res.road_mismatch_vehicle_ticks == 0


def test_systematic_slowdown_fails_delay_gate():
    # platform 1 m/step behind -> more delay than the recorded run
    res = PS.compare_windows(window(cruise_rows(0.0)),
                             window(cruise_rows(0.0, drift=1.0)))
    assert not res.passed
    assert any("delay diff" in f for f in res.failures)
    assert abs(res.delay_diff_pct) > PS.DELAY_TOL_PCT


def test_delay_diff_sign_is_platform_minus_laptop():
    res = PS.compare_windows(window(cruise_rows(0.0)),
                             window(cruise_rows(0.0, drift=1.0)))
    assert res.delay_diff_pct > 0.0         # platform slower -> worse


def test_collision_fails_and_is_reported_per_id():
    bad = cruise_rows(0.0)
    bad[2]["colliding_ids"] = ["carB"]
    res = PS.compare_windows(window(cruise_rows(0.0)), window(bad))
    assert not res.passed
    assert res.platform_collision_events == 1
    assert any("platform collisions" in f and "carB" in f
               for f in res.failures)


def test_teleport_fails():
    bad = cruise_rows(0.0)
    bad[1]["starting_teleports"] = ["carC"]
    res = PS.compare_windows(window(cruise_rows(0.0)), window(bad))
    assert not res.passed
    assert res.platform_teleport_events == 1
    assert any("teleports" in f and "carC" in f for f in res.failures)


def test_laptop_side_events_are_reported_only():
    ref = cruise_rows(0.0)
    ref[2]["colliding_ids"] = ["carB"]
    ref[3]["starting_teleports"] = ["carD"]
    res = PS.compare_windows(window(ref), window(cruise_rows(0.0)))
    assert res.passed                       # PASS rule is platform-side
    assert res.laptop_collision_events == 1
    assert res.laptop_teleport_events == 1
    assert res.platform_collision_events == 0


def test_road_mismatch_counted_and_position_skipped():
    moved = cruise_rows(0.0)
    moved[1]["vehicles"]["carA"]["road_id"] = "OTHER"
    moved[1]["vehicles"]["carA"]["lane_id"] = "OTHER_0"
    res = PS.compare_windows(window(cruise_rows(0.0)), window(moved))
    assert res.road_mismatch_vehicle_ticks == 1
    assert res.vehicle_ticks_compared == 8


def test_world_mismatch_refuses():
    lap = PS.rows_to_window(iter(cruise_rows(0.0)), "w1", 0.0, 2.0)
    plat = PS.rows_to_window(iter(cruise_rows(0.0)), "w2", 0.0, 2.0)
    with pytest.raises(ValueError, match="world mismatch"):
        PS.compare_windows(lap, plat)


def test_rows_outside_window_are_ignored():
    rows = cruise_rows(0.0) + [row(2.5, {"carA": veh(*MAIN, 0.0, 0.0)})]
    win = window(rows, t0=0.0, horizon=2.0)
    assert sorted(win.ticks) == [1, 2, 3, 4]


# --------------------------------------------------- geometry and delay

def test_delay_step_matches_frozen_convention():
    prev = (MAIN[0], MAIN[1], 100.0, 30.0)
    cur = (MAIN[0], MAIN[1], 115.0, 30.0)
    assert PS.lane_phys(*MAIN) == 1
    assert PS.phys_s(MAIN[0], MAIN[1], 100.0) == pytest.approx(2819.66)
    assert PS.delay_step(prev, cur) == pytest.approx(0.5 - 15.0 / 31.29)


def test_ramp_limit_switches_at_378_66():
    assert PS.segment_limit(2, 100.0) == pytest.approx(22.22)
    assert PS.segment_limit(2, 378.66) == pytest.approx(31.29)
    assert PS.segment_limit(2, 400.0) == pytest.approx(31.29)


def test_internal_lane_rows_are_skipped():
    prev = (JUNCTION[0], JUNCTION[1], 10.0, 20.0)
    cur = (JUNCTION[0], JUNCTION[1], 25.0, 20.0)
    assert PS.lane_phys(JUNCTION[0], JUNCTION[1]) is None
    assert PS.delay_step(prev, cur) is None


def test_mean_delay_per_vehicle_divides_by_tracked():
    win = window(cruise_rows(0.0))
    mean, n = PS.mean_delay_per_vehicle(win)
    assert n == 2                            # carA + carB, both mapped
    total = (0.5 - 15.0 / 31.29) * 3 + (0.5 - 10.0 / 22.22) * 3
    assert mean == pytest.approx(total / 2)
    assert mean > 0.0


# --------------------------------------------------------- input guards

def _make_tree(tmp_path):
    root = tmp_path / "laptop"
    for w in WORLDS:
        write_physics(root / "runs" / "campaign_stage1_chunk4" /
                      f"{w}__native" / "run" / "physics_steps.jsonl",
                      cruise_rows(0.0))
        for rel, body in (
                (f"{w}__base/saved_state/state.xml", "<state/>\n"),
                (f"{w}__base/saved_state/STATE_MANIFEST.json", "{}\n"),
                (f"{w}__native/run/scenario/scenario.sumocfg", "<cfg/>\n"),
                (f"{w}__native/run/scenario/requests.rou.xml", "<rou/>\n")):
            p = root / "runs" / "campaign_stage1_chunk4" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
    return root


def write_physics(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(root: Path) -> dict:
    worlds = []
    for w in WORLDS:
        files = []
        for rel in (
                f"runs/campaign_stage1_chunk4/{w}__base/saved_state/state.xml",
                f"runs/campaign_stage1_chunk4/{w}__base/saved_state/"
                f"STATE_MANIFEST.json",
                f"runs/campaign_stage1_chunk4/{w}__native/run/scenario/"
                f"scenario.sumocfg",
                f"runs/campaign_stage1_chunk4/{w}__native/run/scenario/"
                f"requests.rou.xml",
                f"runs/campaign_stage1_chunk4/{w}__native/run/"
                f"physics_steps.jsonl"):
            files.append({"path": rel, "sha256": _sha(root / rel)})
        worlds.append(dict(world=w, split="TRAIN", t0_s=0.0,
                           state_xml=files[0]["path"],
                           scenario_dir=f"runs/campaign_stage1_chunk4/"
                                        f"{w}__native/run/scenario",
                           reference_physics=files[4]["path"],
                           files=files))
    return {"schema": "inc1_parity_inputs/1", "worlds": worlds,
            "purpose": "test"}


def _write_inputs(tmp_path, doc) -> Path:
    p = tmp_path / "PARITY_INPUTS.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_sha_verification_accepts_clean_tree(tmp_path):
    root = _make_tree(tmp_path)
    inputs = _write_inputs(tmp_path, _inputs(root))
    PS.verify_inputs(PS.load_inputs(inputs), root)   # must not raise


def test_sha_verification_refuses_tampered_file(tmp_path):
    root = _make_tree(tmp_path)
    doc = _inputs(root)
    doc["worlds"][0]["files"][0]["sha256"] = "0" * 64
    inputs = _write_inputs(tmp_path, doc)
    with pytest.raises(PS.InputRefused, match="sha256 mismatch"):
        PS.verify_inputs(PS.load_inputs(inputs), root)


def test_forbidden_reference_name_refused(tmp_path):
    root = _make_tree(tmp_path)
    doc = _inputs(root)
    doc["worlds"][0]["reference_physics"] = "runs/x/suffix_summary.jsonl"
    inputs = _write_inputs(tmp_path, doc)
    with pytest.raises(PS.InputRefused, match="forbidden input file"):
        PS.load_inputs(inputs)


def test_non_train_split_refused(tmp_path):
    root = _make_tree(tmp_path)
    doc = _inputs(root)
    doc["worlds"][0]["split"] = "HOLDOUT"
    inputs = _write_inputs(tmp_path, doc)
    with pytest.raises(PS.InputRefused, match="only TRAIN worlds"):
        PS.load_inputs(inputs)


def test_exactly_two_worlds_enforced(tmp_path):
    root = _make_tree(tmp_path)
    doc = _inputs(root)
    doc["worlds"] = doc["worlds"][:1]
    inputs = _write_inputs(tmp_path, doc)
    with pytest.raises(PS.InputRefused, match="exactly 2"):
        PS.load_inputs(inputs)


# ------------------------------------------------------ CLI and runner

def test_cli_without_runner_simulates_nothing(tmp_path, capsys):
    root = _make_tree(tmp_path)
    inputs = _write_inputs(tmp_path, _inputs(root))
    code = PS.main(["--inputs", str(inputs), "--laptop-root", str(root),
                    "--horizon", "2.0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "ZERO SUMO" in out
    assert not (tmp_path / "platform").exists()
    assert fake_parity_runner.CALLS == []


def test_fake_runner_runs_both_worlds_and_passes(tmp_path):
    root = _make_tree(tmp_path)
    inputs = _write_inputs(tmp_path, _inputs(root))
    run_dir = tmp_path / "platform"
    report_path = tmp_path / "parity_report.json"
    code = PS.main(["--inputs", str(inputs), "--laptop-root", str(root),
                    "--runner", "fake_parity_runner:fake_runner",
                    "--run-dir", str(run_dir),
                    "--platform-root", str(run_dir),
                    "--horizon", "2.0", "--out", str(report_path)])
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] and len(report["worlds"]) == 2
    assert len(fake_parity_runner.CALLS) == 2
    for call, w in zip(fake_parity_runner.CALLS, WORLDS):
        assert w in str(call["scenario_dir"])
        assert call["duration_s"] == pytest.approx(2.0)
    for w in report["worlds"]:
        assert w["ticks_compared"] == 4
        assert w["max_abs_ds_m"] == 0.0
        assert w["platform_collision_events"] == 0


def test_fake_runner_refuses_overwrite(tmp_path):
    root = _make_tree(tmp_path)
    inputs = _write_inputs(tmp_path, _inputs(root))
    run_dir = tmp_path / "platform"
    argv = ["--inputs", str(inputs), "--laptop-root", str(root),
            "--runner", "fake_parity_runner:fake_runner",
            "--run-dir", str(run_dir), "--platform-root", str(run_dir),
            "--horizon", "2.0"]
    assert PS.main(argv) == 0
    with pytest.raises(PS.InputRefused, match="refusing to overwrite"):
        PS.main(argv)


def test_runner_spec_must_be_module_function():
    with pytest.raises(ValueError, match="module:function"):
        PS.load_runner("not_a_spec")


def test_report_file_refuses_overwrite(tmp_path):
    root = _make_tree(tmp_path)
    inputs = _write_inputs(tmp_path, _inputs(root))
    out = tmp_path / "report.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(PS.InputRefused, match="refusing to overwrite"):
        PS.main(["--inputs", str(inputs), "--laptop-root", str(root),
                 "--platform-root", str(root), "--horizon", "2.0",
                 "--out", str(out)])


def test_real_parity_inputs_schema_matches_loader():
    """The committed PARITY_INPUTS.json passes the loader rules (names,
    TRAIN labels, no forbidden file names; the referenced research-dense
    files are NOT opened here)."""
    real = PKG / "PARITY_INPUTS.json"
    doc = json.loads(real.read_text(encoding="utf-8"))
    worlds = doc["worlds"]
    assert [w["world"] for w in worlds] == list(WORLDS)
    assert doc["horizon_s"] == 60.0
    for w in worlds:
        assert w["split"] == "TRAIN"
        PS.check_input_path(w["reference_physics"], w["world"], w["split"])
        for entry in w["files"]:
            PS.check_input_path(entry["path"], w["world"], w["split"])
