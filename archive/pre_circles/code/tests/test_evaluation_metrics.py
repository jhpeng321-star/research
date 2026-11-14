"""Tests for the unified evaluation metrics (Task 6: fair evaluation).

Covers the new metric definitions shared by the All-HDV pure-SUMO path and
the TraCI policy path: hard braking from FCD, statistic-output parsing,
driver-type distribution from tripinfo, the unified ``evaluate_run`` record,
and ``run_sumo_evaluation`` CLI construction (seed / stats / collision output).
"""

import subprocess
from pathlib import Path

import pytest

from evaluation import (
    evaluate_hard_braking,
    evaluate_run,
    evaluate_type_distribution,
    parse_sumo_stats,
    run_sumo_evaluation,
)


SAMPLE_STATS = """<?xml version="1.0" encoding="UTF-8"?>
<statistics xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/statistic_file.xsd">
    <performance clockBegin="470324.12" clockEnd="470324.26" clockDuration="0.14" traciDuration="0.00" realTimeFactor="7133.33" vehicleUpdatesPerSecond="173600.00" personUpdatesPerSecond="0.00" begin="0.00" end="963.00" duration="963.00"/>
    <vehicles loaded="426" inserted="426" running="0" waiting="0"/>
    <teleports total="1" jam="1" yield="0" wrongLane="0"/>
    <safety collisions="2" emergencyStops="0" emergencyBraking="3"/>
    <vehicleTripStatistics count="420" routeLength="1569.26" speed="28.50" duration="55.01" waitingTime="0.04" timeLoss="2.91" departDelay="0.41" departDelayWaiting="-1.00" totalTravelTime="23436.00" totalDepartDelay="174.33"/>
</statistics>
"""

# 10 arrived trips: 1 aggressive, 7 normal, 2 cautious.
SAMPLE_TRIPINFO = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="v0" depart="0.00" departLane="1191554636_0" arrival="50.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="aggressive_car" vaporized=""/>
    <tripinfo id="v1" depart="1.00" departLane="1191554636_1" arrival="51.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="normal_car" vaporized=""/>
    <tripinfo id="v2" depart="2.00" departLane="1191554636_0" arrival="52.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="normal_car" vaporized=""/>
    <tripinfo id="v3" depart="3.00" departLane="11553048#1_0" arrival="53.00" duration="50.00" routeLength="900.00" timeLoss="1.00" vType="normal_car" vaporized=""/>
    <tripinfo id="v4" depart="4.00" departLane="1191554636_1" arrival="54.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="normal_car" vaporized=""/>
    <tripinfo id="v5" depart="5.00" departLane="1191554636_0" arrival="55.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="normal_car" vaporized=""/>
    <tripinfo id="v6" depart="6.00" departLane="11553048#1_0" arrival="56.00" duration="50.00" routeLength="900.00" timeLoss="1.00" vType="normal_car" vaporized=""/>
    <tripinfo id="v7" depart="7.00" departLane="1191554636_1" arrival="57.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="normal_car" vaporized=""/>
    <tripinfo id="v8" depart="8.00" departLane="1191554636_0" arrival="58.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="cautious_car" vaporized=""/>
    <tripinfo id="v9" depart="9.00" departLane="1191554636_1" arrival="59.00" duration="50.00" routeLength="1500.00" timeLoss="1.00" vType="cautious_car" vaporized=""/>
</tripinfos>
"""

# Vehicle A brakes -6 m/s^2 between t=0 and t=1 (one hard-braking event).
SAMPLE_FCD = """<?xml version="1.0" encoding="UTF-8"?>
<fcd-export>
    <timestep time="0.00">
        <vehicle id="A" x="0" y="0" speed="20.00" lane="e_0" pos="0.00"/>
    </timestep>
    <timestep time="1.00">
        <vehicle id="A" x="0" y="0" speed="14.00" lane="e_0" pos="17.00"/>
    </timestep>
</fcd-export>
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# statistic-output parsing
# ---------------------------------------------------------------------------
def test_parse_sumo_stats(tmp_path):
    m = parse_sumo_stats(_write(tmp_path, "stats.xml", SAMPLE_STATS))
    assert m["arrived"] == 420
    assert m["duration_s"] == 963.0
    assert m["teleports"] == 1
    assert m["collisions"] == 2
    assert m["mean_speed"] == 28.50
    assert m["mean_time_loss"] == 2.91
    assert m["mean_duration"] == 55.01


# ---------------------------------------------------------------------------
# Hard braking from FCD (deceleration < -4.5 m/s^2)
# ---------------------------------------------------------------------------
def test_hard_braking_counts_events_per_vehicle_step(tmp_path):
    fcd = """<?xml version="1.0" encoding="UTF-8"?>
<fcd-export>
    <timestep time="0.00">
        <vehicle id="A" x="0" y="0" speed="20.00" lane="e_0" pos="0.00"/>
        <vehicle id="B" x="0" y="0" speed="20.00" lane="e_0" pos="50.00"/>
    </timestep>
    <timestep time="1.00">
        <vehicle id="A" x="0" y="0" speed="15.00" lane="e_0" pos="17.00"/>
        <vehicle id="B" x="0" y="0" speed="19.60" lane="e_0" pos="69.00"/>
    </timestep>
    <timestep time="2.00">
        <vehicle id="A" x="0" y="0" speed="12.00" lane="e_0" pos="30.00"/>
        <vehicle id="B" x="0" y="0" speed="19.00" lane="e_0" pos="88.00"/>
        <vehicle id="C" x="0" y="0" speed="1.00" lane="e_0" pos="5.00"/>
    </timestep>
    <timestep time="3.00">
        <vehicle id="A" x="0" y="0" speed="12.00" lane="e_0" pos="42.00"/>
        <vehicle id="B" x="0" y="0" speed="18.50" lane="e_0" pos="107.00"/>
        <vehicle id="C" x="0" y="0" speed="3.00" lane="e_0" pos="7.00"/>
    </timestep>
</fcd-export>
"""
    m = evaluate_hard_braking(_write(tmp_path, "fcd.xml", fcd))
    # A: -5 at t=1 (event), -3 at t=2 (no), 0 at t=3 (no).
    # B: -0.4/-0.6/-0.5 (never). C: first seen at t=2 -> no baseline, no event.
    assert m["threshold"] == -4.5
    assert m["total_events"] == 1
    assert m["events_by_timestep"] == {1.0: 1}


def test_hard_braking_uses_actual_timestep_dt(tmp_path):
    fcd = """<?xml version="1.0" encoding="UTF-8"?>
<fcd-export>
    <timestep time="0.00">
        <vehicle id="A" x="0" y="0" speed="20.00" lane="e_0" pos="0.00"/>
    </timestep>
    <timestep time="0.50">
        <vehicle id="A" x="0" y="0" speed="17.00" lane="e_0" pos="9.00"/>
    </timestep>
</fcd-export>
"""
    m = evaluate_hard_braking(_write(tmp_path, "fcd.xml", fcd))
    # -3 m/s over 0.5 s = -6 m/s^2 -> one event despite the small speed drop.
    assert m["total_events"] == 1


def test_hard_braking_empty_fcd(tmp_path):
    fcd = """<?xml version="1.0" encoding="UTF-8"?>
<fcd-export>
</fcd-export>
"""
    m = evaluate_hard_braking(_write(tmp_path, "fcd.xml", fcd))
    assert m["total_events"] == 0
    assert m["events_by_timestep"] == {}


# ---------------------------------------------------------------------------
# Driver-type distribution from tripinfo
# ---------------------------------------------------------------------------
def test_evaluate_type_distribution(tmp_path):
    dist = evaluate_type_distribution(_write(tmp_path, "trips.xml", SAMPLE_TRIPINFO))
    assert dist["total"] == 10
    assert dist["counts"] == {
        "aggressive_car": 1,
        "normal_car": 7,
        "cautious_car": 2,
    }
    assert dist["fractions"]["aggressive_car"] == pytest.approx(0.10)
    assert dist["fractions"]["normal_car"] == pytest.approx(0.70)
    assert dist["fractions"]["cautious_car"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Unified run record (shared by All-HDV and policy paths)
# ---------------------------------------------------------------------------
def test_evaluate_run_with_stats(tmp_path):
    m = evaluate_run(
        _write(tmp_path, "trips.xml", SAMPLE_TRIPINFO),
        _write(tmp_path, "fcd.xml", SAMPLE_FCD),
        stats_path=_write(tmp_path, "stats.xml", SAMPLE_STATS),
    )
    assert m["travel_time"]["all"]["count"] == 10
    assert "ttc" in m
    # arrived comes from the tripinfo file itself, not from the stats count.
    assert m["arrived"] == 10
    assert m["sim_seconds"] == 963.0
    assert m["throughput_veh_per_hour"] == pytest.approx(10.0 / (963.0 / 3600.0))
    assert m["mean_speed"] == pytest.approx((8 * 1500.0 / 50.0 + 2 * 900.0 / 50.0) / 10.0)
    assert m["mean_time_loss"] == pytest.approx(1.0)
    assert m["hard_braking"]["total_events"] == 1
    # safety counters fall back to the statistic-output when not overridden.
    assert m["collisions"] == 2
    assert m["teleports"] == 1
    assert m["type_counts"]["normal_car"] == 7
    assert m["type_fractions"]["normal_car"] == pytest.approx(0.7)


def test_evaluate_run_traci_overrides_beat_stats(tmp_path):
    m = evaluate_run(
        _write(tmp_path, "trips.xml", SAMPLE_TRIPINFO),
        _write(tmp_path, "fcd.xml", SAMPLE_FCD),
        stats_path=_write(tmp_path, "stats.xml", SAMPLE_STATS),
        collisions=5,
        teleports=0,
        sim_seconds=100.0,
    )
    assert m["collisions"] == 5
    assert m["teleports"] == 0
    assert m["sim_seconds"] == 100.0
    assert m["throughput_veh_per_hour"] == pytest.approx(360.0)


def test_evaluate_run_without_stats_falls_back_to_last_arrival(tmp_path):
    m = evaluate_run(
        _write(tmp_path, "trips.xml", SAMPLE_TRIPINFO),
        _write(tmp_path, "fcd.xml", SAMPLE_FCD),
    )
    assert m["sim_seconds"] == 59.0
    assert m["collisions"] is None
    assert m["teleports"] is None


# ---------------------------------------------------------------------------
# run_sumo_evaluation CLI construction
# ---------------------------------------------------------------------------
def test_run_sumo_evaluation_seed_and_extra_outputs(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, cwd, check):
        calls.append({"cmd": cmd, "cwd": cwd, "check": check})

    monkeypatch.setattr(subprocess, "run", fake_run)
    trip = tmp_path / "tripinfos.xml"
    fcd = tmp_path / "fcd.xml"
    stats = tmp_path / "stats.xml"
    coll = tmp_path / "collisions.xml"
    run_sumo_evaluation(
        sumo_binary="sumo",
        sumo_config=Path("osm.light.sumocfg"),
        project_root=tmp_path,
        tripinfo_output=trip,
        fcd_output=fcd,
        seed=43,
        stats_output=stats,
        collision_output=coll,
    )
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--seed") + 1] == "43"
    assert cmd[cmd.index("--statistic-output") + 1] == str(stats)
    assert cmd[cmd.index("--collision-output") + 1] == str(coll)
    assert cmd[cmd.index("--tripinfo-output") + 1] == str(trip)
    assert cmd[cmd.index("--fcd-output") + 1] == str(fcd)
    assert calls[0]["cwd"] == str(tmp_path)


def test_run_sumo_evaluation_defaults_have_no_seed(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))
    run_sumo_evaluation(
        sumo_binary="sumo",
        sumo_config=Path("cfg.sumocfg"),
        project_root=tmp_path,
        tripinfo_output=tmp_path / "t.xml",
        fcd_output=tmp_path / "f.xml",
    )
    cmd = calls[0]
    assert "--seed" not in cmd
    assert "--statistic-output" not in cmd
    assert "--collision-output" not in cmd
