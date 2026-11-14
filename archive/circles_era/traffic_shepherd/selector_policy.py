"""Causal current-report cohort selector for the PPO policy (wall_ppo/3).

Versioned module (PP-11, repaired in PP-12). Duration semantics remain
EXACTLY wall_ppo/2 — this module only changes WHO may be activated, never
how long a committed wall holds.

Selection contract (deterministic, declared before any use):
  - Causal eligibility at decision time now: acquired <= now, acquisition
    age <= max_report_age_s, receipt <= now (enforced by the RSU buffer) and
    receipt age <= max_report_age_s; latest eligible report per CAV
    (highest acquisition timestamp, then vehicle id). Future, delayed,
    stale and duplicate rows are excluded by construction (regression-
    tested). PP-12 repair: the eligibility filter now runs INSIDE the
    search — the caller can no longer bypass it by passing raw reports.
  - Enumeration: EXHAUSTIVE over eligible received candidates in the
    recruitment band (>= MINIMUM_RECRUITMENT_DISTANCE_M) on the SUPPLIED
    lanes, full product, lane-major deterministic order. No pair cap.
  - Feasibility: the SHARED admission path under the SUPPLIED profile
    (make_proposal -> assess_formation -> remaining_plan readiness_count=0)
    — identical to the fixed-cohort path and to the runtime execution gate.
    PP-12 repair: profile and required_lanes are REQUIRED keywords. The
    PP-11 signature accepted a profile parameter and ignored it (the search
    silently assessed a different profile and hardcoded lanes); passing
    them is now mandatory so search and execution provably assess the same
    offer.
  - Ranking: (remaining_total_s, member_ids) ascending — the registered
    minimum-predicted-maneuver rule. No thresholds, no outcome-based
    ranking.
  - The top-ranked FEASIBLE pair is offered to the actor as the specific
    proposal for this tick.

COMPATIBILITY: causal_selector/1 (PP-11) accepted raw reports and an
ignored profile keyword; that version is superseded. No checkpoint was
ever trained or saved under either selector version, so no checkpoint
reinterpretation is possible. The PP-09 audit entry point
(candidate_availability.search_current_reports) is untouched versioned
audit code with its own documented defaults.
"""
from itertools import product

from traffic_shepherd.candidate_availability import shared_admission
from traffic_shepherd.cohort_study import MINIMUM_RECRUITMENT_DISTANCE_M

SELECTOR_VERSION = "causal_selector/2"

DEFAULT_MAX_REPORT_AGE_S = 1.0


def causal_eligible_reports(reports, now_s, *, max_report_age_s=DEFAULT_MAX_REPORT_AGE_S):
    """Causally eligible CAVReports at now_s: acquired <= now within the
    acquisition-age limit, receipt <= now within the receipt-age limit
    (CAVReport.sent_at_s is the receipt-time proxy recorded at the RSU;
    the RSU buffer separately enforces its own receipt window). Latest
    eligible report per CAV wins (highest acquisition timestamp, then
    vehicle id). Future, stale and duplicate rows are dropped."""
    best = {}
    for report in reports:
        frame = report.frame
        acquired = frame.timestamp_s
        received = getattr(report, "sent_at_s", acquired)
        if (acquired > now_s + 1e-9 or received > now_s + 1e-9
                or now_s - acquired > max_report_age_s + 1e-9
                or now_s - received > max_report_age_s + 1e-9):
            continue
        key = (acquired, received)
        prior = best.get(frame.vehicle_id)
        if prior is None or (key, frame.vehicle_id) > (prior[0], frame.vehicle_id):
            best[frame.vehicle_id] = (key, report)
    return [best[vid][1] for vid in sorted(best)]


def search_current_proposals(reports, now_s, *, profile, required_lanes,
                             max_report_age_s=DEFAULT_MAX_REPORT_AGE_S):
    """Exhaustive shared-admission search over current acquired reports.

    profile and required_lanes are REQUIRED and are used for every assessed
    candidate — the same objects the execution-time admission will use.

    Returns (best_member_ids, best_reason, assessed_count) where
    best_member_ids is None when no pair is feasible. Ties broken by
    member_ids (deterministic). best_reason is the refusal reason of the
    first infeasible assessed pair (diagnostic only).
    """
    eligible = causal_eligible_reports(reports, now_s,
                                       max_report_age_s=max_report_age_s)
    by_lane = {lane: [] for lane in required_lanes}
    for report in eligible:
        f = report.frame
        if (f.lane_id in by_lane and f.distance_to_merge_m is not None
                and f.distance_to_merge_m >= MINIMUM_RECRUITMENT_DISTANCE_M):
            by_lane[f.lane_id].append(f)
    best_ids, best_key, best_reason = None, None, None
    assessed = 0
    for group in product(*(by_lane[lane] for lane in required_lanes)):
        member_ids = tuple(f.vehicle_id for f in group)
        assessed += 1
        feasible, reason, _proposal, plan = shared_admission(
            eligible, now_s, member_ids, profile=profile,
            required_lanes=required_lanes)
        if not feasible:
            if best_reason is None:
                best_reason = reason
            continue
        key = ((plan or {}).get("remaining_total_s"), member_ids)
        if best_key is None or key < best_key:
            best_ids, best_key = member_ids, key
    return best_ids, best_reason, assessed
