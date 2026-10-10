"""Shared causal candidate-availability assessment (PP-09; zero simulation).

One assessment path for BOTH candidate comparisons:
  A. the cohort fixed by the executed pilot (bound from the damping
     reference's t=600 snapshot), and
  B. candidates enumerated from current acquired reports.

The path is the wall_ppo/2 activation admission contract:
  make_proposal (target = mean reported member speed - 2, capped 24 m/s,
  required lanes from the map) -> assess_formation (approved gate) ->
  remaining_plan with readiness_count=0 (the pilot's activation-admission
  assumption; continuity readiness is a runtime quantity and is not
  assumed at admission).

Causal report availability: a report is eligible at decision time `now`
only if it was ACQUIRED at or before now (frame.timestamp_s <= now), within
the acquisition-age limit, RECEIVED at or before now (received_at_s <= now),
and its receipt is not older than the receipt-age limit. The latest eligible
report per CAV wins (deterministic: highest acquisition timestamp, then
highest receipt time, then vehicle id). Duplicate rows for the same
observation are deduplicated on (vehicle_id, frame.timestamp_s).

NOTE: this shared path differs from the PP-08 v1 audit, where the fixed
cohort was checked with assess_formation + remaining_plan(readiness_count=0)
but the search reused select_cohort's internal feasibility
(readiness_count=int(settled)). The v1 audit results are superseded by the
shared-path recomputation; the difference is quantified in OLD_VS_NEW.
"""
from itertools import product

from traffic_shepherd.formation import make_proposal, assess_formation
from traffic_shepherd.lifecycle_wall import LifecycleProfile, remaining_plan
from traffic_shepherd.cohort_study import LANES, MINIMUM_RECRUITMENT_DISTANCE_M
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


def row_to_cav_report(row):
    """Reconstruct the typed CAVReport from a telemetry receipt row."""
    frame = row["frame"]
    front = frame.get("front")
    rear = frame.get("rear")
    if front is not None:
        front = NeighborMeasurement(gap_m=front["gap_m"],
                                    relative_speed_mps=front["relative_speed_mps"])
    if rear is not None:
        rear = NeighborMeasurement(gap_m=rear["gap_m"],
                                   relative_speed_mps=rear["relative_speed_mps"])
    sf = SensorFrame(vehicle_id=frame["vehicle_id"],
                     timestamp_s=frame["timestamp_s"],
                     speed_mps=frame["speed_mps"],
                     acceleration_mps2=frame["acceleration_mps2"],
                     allowed_speed_mps=frame["allowed_speed_mps"],
                     lane_index=frame["lane_index"],
                     distance_to_merge_m=frame.get("distance_to_merge_m"),
                     front=front, rear=rear,
                     front_coverage_m=frame.get("front_coverage_m", 0.0),
                     rear_coverage_m=frame.get("rear_coverage_m", 0.0),
                     previous_command_delta=frame.get("previous_command_delta", 0.0),
                     lane_id=frame.get("lane_id"), position_m=frame.get("position_m"))
    return CAVReport(sf, row["sent_at_s"], row.get("owner", "damping"),
                     row.get("network_output_mps2"),
                     row.get("issued_acceleration_mps2"),
                     row.get("fallback_reason"))

PROFILE = LifecycleProfile()


def _eligible_pairs(rows, now_s, *, max_age_s=1.0, receipt_age_s=None):
    """Causally eligible (report, received_at_s) pairs at now_s. Latest
    eligible report per CAV wins (highest acquisition timestamp, then
    highest receipt time, then vehicle id)."""
    if receipt_age_s is None:
        receipt_age_s = max_age_s
    best = {}
    for row in rows:
        frame = row["frame"]
        acquired = frame["timestamp_s"]
        received = row["received_at_s"]
        vid = frame["vehicle_id"]
        if acquired > now_s + 1e-9 or received > now_s + 1e-9:
            continue
        if now_s - acquired > max_age_s + 1e-9:
            continue
        if now_s - received > receipt_age_s + 1e-9:
            continue
        key = (vid, acquired)
        prior = best.get(vid)
        if prior is None:
            best[vid] = (key, received, row)
            continue
        p_key, p_received, _ = prior
        if key > p_key or (key == p_key and received < p_received):
            best[vid] = (key, received, row)
    return [(row_to_cav_report(v[2]), v[1]) for v in sorted(best.values())]


def eligible_report_rows(rows, now_s, *, max_age_s=1.0):
    return [rep for rep, _recv in
            _eligible_pairs(rows, now_s, max_age_s=max_age_s)]


def eligible_report_rows_with_receipts(rows, now_s, *, max_age_s=1.0):
    return _eligible_pairs(rows, now_s, max_age_s=max_age_s)


def shared_admission(reports, now_s, member_ids, *, profile=None,
                     required_lanes=None):
    """wall_ppo/2 activation admission for ONE candidate cohort.

    profile / required_lanes: explicit wall profile and map lanes for the
    assessment. When omitted, the versioned PP-09 audit defaults apply
    (LifecycleProfile / cohort_study.LANES) — the runtime selector MUST
    pass the policy's own profile and lanes so search and execution
    assess the identical offer (PP-12 repair; PP-11 silently assessed a
    different profile).

    Returns (feasible, reason, proposal, plan). A cohort whose members lack
    current reports is infeasible ("bound vehicles unavailable"), not an
    error — matching the runtime activation gate's treatment of missing
    member reports.
    """
    use_profile = profile if profile is not None else LifecycleProfile()
    use_lanes = tuple(required_lanes) if required_lanes is not None else LANES
    try:
        proposal = make_proposal(reports, now_s=now_s, required_lanes=use_lanes,
                                 member_ids=tuple(member_ids), profile=use_profile)
    except ValueError as exc:
        return False, "proposal_unavailable:" + str(exc), None, None
    assessment = assess_formation(proposal, reports, now_s=now_s)
    if not assessment.approved:
        return False, "assessment:" + assessment.reason, proposal, None
    plan = remaining_plan(proposal, reports, now_s=now_s, previous_commands={},
                          readiness_count=0)
    return bool(plan["feasible"]), plan.get("reason"), proposal, plan


def current_report_candidates(reports, now_s):
    """Exhaustive, uncapped enumeration over received candidates in the
    recruitment band on the required lanes. Deterministic order: lanes in
    registered order, candidates within a lane sorted by
    (distance_to_merge_m, vehicle_id). The full product is evaluated; there
    is no pair cap (declared bound: complete enumeration)."""
    by_lane = {lane: [] for lane in LANES}
    for report in reports:
        f = report.frame
        if (f.lane_id in by_lane and f.distance_to_merge_m is not None
                and f.distance_to_merge_m >= MINIMUM_RECRUITMENT_DISTANCE_M):
            by_lane[f.lane_id].append(f)
    for lane in LANES:
        by_lane[lane].sort(key=lambda f: (f.distance_to_merge_m, f.vehicle_id))
    return by_lane, [tuple(f.vehicle_id for f in group)
                     for group in product(*(by_lane[lane] for lane in LANES))]


def search_current_reports(reports, now_s, member_filter=None):
    """Exhaustive current-report candidate search with the SHARED admission.

    member_filter: optional callable(member_ids) restricting which enumerated
    pairs are assessed (used for comparison A on the fixed cohort). Returns
    per-pair results in deterministic enumeration order plus the best
    feasible candidate under the registered ranking
    (remaining_total_s, member_ids).
    """
    by_lane, all_pairs = current_report_candidates(reports, now_s)
    results = []
    best = None
    for member_ids in all_pairs:
        if member_filter is not None and not member_filter(member_ids):
            continue
        feasible, reason, proposal, plan = shared_admission(reports, now_s, member_ids)
        entry = dict(member_ids=list(member_ids), feasible=feasible, reason=reason,
                     remaining_total_s=(plan or {}).get("remaining_total_s"))
        results.append(entry)
        if feasible and (best is None or
                         (entry["remaining_total_s"], tuple(member_ids)) <
                         (best["remaining_total_s"], tuple(best["member_ids"]))):
            best = entry
    return dict(pairs=len(all_pairs), assessed=len(results), results=results,
                best=best, by_lane_counts={lane: len(g) for lane, g in by_lane.items()})
