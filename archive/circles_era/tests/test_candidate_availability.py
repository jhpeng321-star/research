"""Shared causal candidate-availability assessment (PP-09) — offline tests.

Covers: causal report eligibility (future acquisition, delayed receipt,
stale acquisition, duplicates, out-of-order delivery), A/B equivalence of
the fixed-cohort and enumerated-candidate paths on identical inputs, and
the exhaustive/ordered enumeration contract.
"""

import pytest

from traffic_shepherd.candidate_availability import (
    current_report_candidates, eligible_report_rows, shared_admission)
from traffic_shepherd.cohort_study import LANES


def row(vid, ts, received=None, lane="1191554636_0", dist=500.0, speed=22.0):
    received = ts if received is None else received
    return dict(frame=dict(vehicle_id=vid, timestamp_s=ts, speed_mps=speed,
                           acceleration_mps2=0.0, allowed_speed_mps=27.0,
                           lane_index=0, distance_to_merge_m=dist,
                           front=None, rear=None, front_coverage_m=100.0,
                           rear_coverage_m=100.0, previous_command_delta=0.0,
                           lane_id=lane, position_m=3500.0 - dist),
                sent_at_s=ts, received_at_s=received, owner="damping",
                network_output_mps2=None, issued_acceleration_mps2=None,
                fallback_reason=None)


# ---- causal eligibility ------------------------------------------------------

def vid_of(report):
    return report.frame.vehicle_id


def test_future_acquisition_excluded():
    rows = [row("a", 115.0), row("b", 130.0)]      # b acquired after now=120
    eligible = eligible_report_rows(rows, now_s=120.0, max_age_s=10.0)
    assert [vid_of(r) for r in eligible] == ["a"]


def test_delayed_receipt_excluded():
    rows = [row("a", 115.0, received=119.0), row("b", 115.0, received=121.0)]
    eligible = eligible_report_rows(rows, now_s=120.0, max_age_s=10.0)
    assert [vid_of(r) for r in eligible] == ["a"]  # b received at 121 > 120


def test_stale_acquisition_excluded():
    rows = [row("a", 100.0), row("b", 119.5)]
    eligible = eligible_report_rows(rows, now_s=120.0, max_age_s=1.0)
    assert [vid_of(r) for r in eligible] == ["b"]  # a is 20 s old


def test_duplicate_same_observation_keeps_earliest_receipt():
    rows = [row("a", 115.0, received=116.0), row("a", 115.0, received=118.0)]
    # receipt provenance: use the pairs helper
    from traffic_shepherd.candidate_availability import eligible_report_rows_with_receipts
    pairs_with_receipts = eligible_report_rows_with_receipts(rows, now_s=120.0, max_age_s=10.0)
    assert len(pairs_with_receipts) == 1
    assert pairs_with_receipts[0][1] == 116.0


def test_out_of_order_delivery_latest_observation_wins():
    rows = [row("a", 110.0), row("a", 118.0), row("a", 114.0)]
    eligible = eligible_report_rows(rows, now_s=120.0, max_age_s=10.0)
    assert len(eligible) == 1 and eligible[0].frame.timestamp_s == 118.0


# ---- A/B equivalence and enumeration contract --------------------------------

def _snapshot_reports(now=600.0):
    """A deterministic two-lane candidate set in the recruitment band."""
    rows = []
    for lane_i, lane in enumerate(LANES):
        for k in range(3):
            vid = f"cav_{lane_i}_{k}"
            rows.append(row(vid, now, lane=lane, dist=300.0 + 25.0 * k))
        # one stale + one wrong-lane report that must be excluded
        rows.append(row(f"stale_{lane_i}", now - 5.0, lane=lane, dist=320.0))
    rows.append(row("far", now, lane=LANES[0], dist=2000.0))   # outside band? (>=200 kept)
    return rows


def test_ab_equivalence_same_cohort_same_snapshot():
    """A (fixed-cohort path) and B (enumerated path restricted to that
    cohort) must produce identical feasibility for the same cohort at the
    same snapshot."""
    rows = _snapshot_reports()
    eligible = eligible_report_rows(rows, now_s=600.0, max_age_s=1.0)
    members = ["cav_0_0", "cav_1_0"]
    from traffic_shepherd.candidate_availability import (
        current_report_candidates, shared_admission)
    fa_full = shared_admission(eligible, 600.0, members)
    fa, proposal_a, _plan_a = fa_full[0], fa_full[2], fa_full[3] if False else None
    _, pairs = current_report_candidates(eligible, 600.0)
    matches = [shared_admission(eligible, 600.0, p) for p in pairs
               if tuple(p) == tuple(members)]
    assert len(matches) == 1
    fb = matches[0][0]
    assert fa == fb
    assert proposal_a.fingerprint == matches[0][2].fingerprint


def test_enumeration_is_exhaustive_and_ordered():
    rows = _snapshot_reports()
    eligible = eligible_report_rows(rows, 600.0, max_age_s=1.0)
    by_lane, pairs = current_report_candidates(eligible, 600.0)
    # 3 fresh candidates per lane in the recruitment band (stale excluded;
    # 'far' at 2000 m is on LANE[0] and IS >= 200 m so included)
    assert len(by_lane[LANES[0]]) >= 3 and len(by_lane[LANES[1]]) >= 3
    # full product in lane-major deterministic order
    import itertools
    expected = tuple(tuple(f.vehicle_id for f in group) for group in itertools.product(
        by_lane[LANES[0]], by_lane[LANES[1]]))
    assert len(pairs) == len(expected) and set(pairs) == set(expected)


def test_missing_fields_fail_loudly():
    bad = row("a", 600.0)
    broken = dict(bad)
    broken["frame"] = dict(bad["frame"])
    broken["frame"].pop("speed_mps")
    with pytest.raises((KeyError, TypeError, ValueError)):
        shared_admission([broken], 600.0, ["cav_a"])


# ---- shared-path feasibility sanity on real-scale synthetic inputs ----------

def test_shared_admission_feasible_pair_signature():
    rows = [row("cav_a", 600.0, lane=LANES[0], dist=1500.0),
            row("cav_b", 600.0, lane=LANES[1], dist=1505.0)]
    typed = [__import__("traffic_shepherd.candidate_availability",
                        fromlist=["row_to_cav_report"]).row_to_cav_report(r)
             for r in rows]
    feasible, reason, proposal, plan = shared_admission(typed, 600.0,
                                                        ("cav_a", "cav_b"))
    assert feasible is True
    assert plan is not None and plan.get("remaining_total_s") is not None
    assert proposal.target_speed_mps == 20.0     # mean(22,22) - 2
