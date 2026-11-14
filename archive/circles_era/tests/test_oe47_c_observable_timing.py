"""OE-47 Branch C tests (zero simulation; read-only on frozen inputs).

Covers the frozen replay contract in output/oe47_c_observable_timing_v1/
SPEC_FREEZE.md: replay integrity at t=600 against the executed binding,
strictly-earlier first-feasible filtering, selector determinism, and coverage
counters including the age-1 s carryover missingness rule.
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_shepherd.reporting import CAVReport  # noqa: E402
from traffic_shepherd.sensing import SensorFrame  # noqa: E402

_SPEC = ROOT / "output/oe47_c_observable_timing_v1/replay_timing.py"
_spec = importlib.util.spec_from_file_location("oe47c_replay_timing", _SPEC)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

LANE0, LANE1 = rt.LANES


def _report(vehicle_id, tick, *, lane_id, distance, speed=25.0):
    frame = SensorFrame(
        vehicle_id=vehicle_id, timestamp_s=float(tick), speed_mps=float(speed),
        acceleration_mps2=0.0, allowed_speed_mps=31.29, lane_index=0,
        distance_to_merge_m=float(distance), front=None, rear=None,
        front_coverage_m=100.0, rear_coverage_m=100.0, lane_id=lane_id,
        position_m=float(distance))
    return CAVReport(frame=frame, sent_at_s=float(tick), owner="native")


def _pair(tick, *, dist_b=1500.0):
    """Two fresh reports, one per required lane, far upstream with clear margins.

    Equal distances and speeds keep the registered coordination law compact in
    the plan model, so the pair is admitted when both members are eligible.
    """
    return [_report("cav.a", tick, lane_id=LANE0, distance=1500.0),
            _report("cav.b", tick, lane_id=LANE1, distance=dist_b)]


def _row(report):
    """Telemetry-shaped rsu_received_report row (zero-delay current frame)."""
    payload = report.payload()
    return dict(kind="rsu_received_report",
                received_at_s=payload["frame"]["timestamp_s"], **payload)


# ---------------------------------------------------------------------------
# 1. Replay integrity: one group reproduces executed feasibility at 600
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s90004():
    return rt.evaluate_group(90004)


def test_replay_reproduces_feasibility_at_600(s90004):
    binding = json.loads((rt.RUNS / "bindings" / "s90004_h20.json").read_text(encoding="utf-8"))
    assert s90004["integrity"]["evidence_identical_at_600"] is True
    assert s90004["integrity"]["selection_sha256_match"] is True
    assert s90004["feasible_at_600"] is True
    assert s90004["replay_600"]["member_ids"] == list(binding["selection"]["selected"]["member_ids"])
    assert s90004["replay_600"]["selection_sha256"] == binding["selection"]["selection_sha256"]


def test_census_matches_selector_candidates_at_600(s90004):
    """Stage-1 census equals the executed candidate comprehension of select_cohort."""
    executed = json.loads(
        (rt.RUNS / "bindings" / "s90004_h20.json").read_text(encoding="utf-8"))
    assert s90004["replay_600"]["candidates_by_lane"] == executed["selection"]["candidates_by_lane"]
    assert s90004["replay_600"]["candidate_count"] == executed["selection"]["candidate_count"]


# ---------------------------------------------------------------------------
# 2. Strictly-earlier filtering of the first-feasible scan
# ---------------------------------------------------------------------------

def test_first_feasible_is_strictly_earliest():
    """The frozen scan stops at the earliest feasible tick, never a later one."""
    def fresh_at(tick):
        # No pair before 105: member B sits below the 200 m recruitment distance.
        return _pair(tick, dist_b=180.0 if tick < 105 else 1500.0)

    first = None
    for tick in range(100, 131):
        count = rt.census(fresh_at(tick), float(tick))
        if not count["pair_exists"]:
            continue
        if rt.admit(fresh_at(tick), float(tick))["feasible"]:
            first = tick
            break
    assert first == 105
    # Admission is stable here, so any later stop would be wrong.
    for tick in (106, 120):
        assert rt.admit(fresh_at(tick), float(tick))["feasible"] is True


def test_scan_on_synthetic_stream_uses_gap_ticks_and_stops_early():
    """Silent control ticks still get a fresh set; the stop stays strictly earliest."""
    rows = []
    for tick in range(100, 112):
        if tick in (104, 108):
            continue  # fully silent control ticks
        rows.extend(_row(r) for r in _pair(tick, dist_b=180.0 if tick < 106 else 1500.0))
    first = None
    seen_silent = set()
    ticks_seen = set()
    for tick, fresh in rt.stream_fresh_sets(rows):
        ticks_seen.add(tick)
        if not (100 <= tick <= 111):
            continue
        if tick in (104, 108):
            seen_silent.add(tick)
        if not rt.census(fresh, float(tick))["pair_exists"]:
            continue
        if rt.admit(fresh, float(tick))["feasible"]:
            first = tick
            break
    # Gap ticks are yielded (with carryover reports), never skipped, up to the
    # early stop; the silent tick 104 was evaluated before the stop at 106.
    assert 104 in seen_silent
    assert ticks_seen == set(range(100, 107))
    # Feasible first at 106 (B crosses 200 m); the silent tick 104 produced no
    # spurious earlier stop, and the scan stopped before the later gap tick 108.
    assert first == 106


# ---------------------------------------------------------------------------
# 3. Selector determinism
# ---------------------------------------------------------------------------

def test_selector_determinism_under_input_order():
    fresh = _pair(600) + [_report(f"cav.extra{i}", 600, lane_id=(LANE0, LANE1)[i % 2],
                                  distance=1600.0 + 10 * i, speed=24.0)
                          for i in range(6)]
    baseline = rt.admit(fresh, 600.0)
    assert baseline["feasible"] is True
    for seed in (1, 2, 3):
        shuffled = list(fresh)
        random.Random(seed).shuffle(shuffled)
        again = rt.admit(shuffled, 600.0)
        assert again["selection_sha256"] == baseline["selection_sha256"]
        assert again["member_ids"] == baseline["member_ids"]
        assert again["target_speed_mps"] == baseline["target_speed_mps"]


# ---------------------------------------------------------------------------
# 4. Coverage counters and missingness rules
# ---------------------------------------------------------------------------

def test_age_one_carryover_and_dropout():
    rows = []
    for tick in range(10, 16):
        rows.append(_row(_report("cav.a", tick, lane_id=LANE0, distance=1500.0)))
        if tick != 13:  # cav.b is silent exactly once
            rows.append(_row(_report("cav.b", tick, lane_id=LANE1, distance=1520.0)))
    observed = dict(rt.stream_fresh_sets(rows))
    # cav.b missing at 13 is still observed at 13 via its age-1.0 s tick-12 frame.
    b13 = next(r for r in observed[13] if r.frame.vehicle_id == "cav.b")
    assert b13.frame.timestamp_s == 12.0
    assert len(observed[13]) == 2
    assert len(observed[14]) == 2
    # Two consecutive silent ticks drop a reporter (age > 1 s): silence a at 14-15.
    rows2 = []
    for tick in range(10, 16):
        if tick not in (14, 15):
            rows2.append(_row(_report("cav.a", tick, lane_id=LANE0, distance=1500.0)))
        rows2.append(_row(_report("cav.b", tick, lane_id=LANE1, distance=1520.0)))
    observed2 = dict(rt.stream_fresh_sets(rows2))
    ids_15 = {r.frame.vehicle_id for r in observed2[15]}
    assert ids_15 == {"cav.b"}  # a's latest frame (13) is age 2 s: dropped
    a14 = next(r for r in observed2[14] if r.frame.vehicle_id == "cav.a")
    assert a14.frame.timestamp_s == 13.0  # one silent tick: still carried


def test_coverage_window_counters():
    rows = []
    for tick in range(10, 16):
        rows.append(_row(_report("cav.a", tick, lane_id=LANE0, distance=1500.0)))
        rows.append(_row(_report("cav.b", tick, lane_id=LANE1, distance=1520.0)))
    observed = dict(rt.stream_fresh_sets(rows))
    window = [t for t in observed if 10 <= t <= 15]
    assert len(window) == 6
    assert all(observed[t] for t in window)
    counts = [rt.census(observed[t], float(t)) for t in window]
    assert all(c["n_fresh"] == 2 and c["pair_exists"] for c in counts)


def test_no_candidate_pair_is_distinguished_from_no_reports():
    count = rt.census([], 300.0)
    assert count["n_fresh"] == 0 and count["pair_exists"] is False
    fresh_only_lane0 = [_report("cav.a", 300, lane_id=LANE0, distance=1500.0)]
    count = rt.census(fresh_only_lane0, 300.0)
    assert count["n_fresh"] == 1
    assert count["eligible_by_lane"][LANE0] == 1
    assert count["eligible_by_lane"][LANE1] == 0
    assert count["pair_exists"] is False
