"""Proposal selection and finite CAV-only feature packets; no fitted gate claim.

Receipt times are explicit. Vehicle IDs, absolute decision time and experiment
metadata bind evidence but are never model inputs. Missing neighbours remain
missing; coverage is per reporter and is never summed into a traffic census.
"""
from dataclasses import asdict
from itertools import product
import hashlib
import json
from math import isfinite
from statistics import mean, pvariance

from .formation import FormationProfile, assess_formation, make_proposal
from .reporting import CAVReport


SCHEMA = "cav_proposal_features/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def latest_received(history, *, now_s, maximum_age_s=1.):
    """Reject future packets, then retain the newest observed packet per CAV."""
    if not isfinite(now_s) or not isfinite(maximum_age_s) or min(now_s, maximum_age_s) < 0:
        raise ValueError("invalid observation clock")
    latest = {}
    for receipt, report in history:
        if not isinstance(report, CAVReport):
            raise TypeError("only acquired CAVReport packets")
        if not isfinite(receipt) or not report.sent_at_s <= receipt <= now_s:
            raise ValueError("future or invalid receipt")
        if not 0 <= now_s-report.frame.timestamp_s <= maximum_age_s:
            continue
        key = report.frame.vehicle_id
        old = latest.get(key)
        if old and old[1].frame.timestamp_s == report.frame.timestamp_s and old[1] != report:
            raise ValueError("conflicting report identity")
        if old is None or old[1].frame.timestamp_s < report.frame.timestamp_s:
            latest[key] = (receipt, report)
    return tuple(latest[k] for k in sorted(latest))


def select_proposal(history, *, now_s, required_lanes, recruitment_band_m=(200., 1900.),
                    profile=None, maximum_members_per_lane=16):
    """One fixed report-based ranking at any traffic state, without outcome access.

    Prefer locally admissible pairs, then estimated delivery runway slack, then
    smaller initial span. Retain refusals and enumeration truncation explicitly.
    No initial 50 m cutoff and no congestion prerequisite are applied.
    """
    p = profile or FormationProfile()
    if not required_lanes or len(set(required_lanes)) != len(required_lanes):
        raise ValueError("distinct static map lanes required")
    if len(required_lanes) > 2 or type(maximum_members_per_lane) is not int or not 1 <= maximum_members_per_lane <= 16:
        raise ValueError("initial two-lane bounded selection only")
    lo, hi = recruitment_band_m
    if not 0 <= lo < hi:
        raise ValueError("invalid recruitment band")
    packets = latest_received(history, now_s=now_s, maximum_age_s=p.max_report_age_s)
    reports = [r for _, r in packets]
    lanes = {lane: [] for lane in required_lanes}
    for r in reports:
        f = r.frame
        if f.lane_id in lanes and f.distance_to_merge_m is not None and lo <= f.distance_to_merge_m <= hi:
            lanes[f.lane_id].append(r)
    counts = {lane: len(rows) for lane, rows in lanes.items()}
    for rows in lanes.values():
        rows.sort(key=lambda r: (-r.frame.distance_to_merge_m, r.frame.vehicle_id))
        del rows[maximum_members_per_lane:]
    metadata = dict(reporting_cavs=len(reports), reporting_candidates_by_map_lane=counts,
        enumeration_complete=all(n <= maximum_members_per_lane for n in counts.values()),
        missing_reported_lanes=[lane for lane, n in counts.items() if not n],
        lane_occupancy_known=False, coverage_is_union=False, traffic_state_filter=False)
    if metadata["missing_reported_lanes"]:
        return None, None, dict(metadata, status="insufficient_members", evaluated_combinations=0)
    ranked = []
    for group in product(*(lanes[lane] for lane in required_lanes)):
        proposal = make_proposal(reports, now_s=now_s, required_lanes=tuple(required_lanes),
            member_ids=tuple(r.frame.vehicle_id for r in group), profile=p)
        assessment = assess_formation(proposal, reports, now_s=now_s)
        rank = (not assessment.approved, -(assessment.estimated_runway_slack_m or 0.),
                assessment.initial_or_current_span_m, proposal.member_ids)
        ranked.append((rank, proposal, assessment))
    _, proposal, assessment = min(ranked, key=lambda item: item[0])
    return proposal, assessment, dict(metadata, status=assessment.status,
        evaluated_combinations=len(ranked), selection="local_admission_then_estimated_runway_then_span_then_id")


def feature_packet(proposal, history, *, now_s, history_s=20.):
    """Bind finite model inputs to exact proposal and already received evidence."""
    if not isfinite(history_s) or history_s < proposal.profile.max_report_age_s:
        raise ValueError("invalid history window")
    history = tuple(history)
    # Validate even excluded rows; future evidence cannot silently slip through.
    latest = latest_received(history, now_s=now_s, maximum_age_s=proposal.profile.max_report_age_s)
    retained = [(t, r) for t, r in history if t >= now_s-history_s]
    reports = [r for _, r in latest]
    assessment = assess_formation(proposal, reports, now_s=now_s)
    values = dict(target_speed_mps=proposal.target_speed_mps,
                  reporting_cavs=float(len(reports)), required_lane_count=float(len(proposal.required_lanes)))
    by_id = {r.frame.vehicle_id: (t, r) for t, r in latest}
    for i, member in enumerate(proposal.members):
        prefix = f"member_{i}_"
        item = by_id.get(member.vehicle_id)
        values[prefix+"report_present"] = float(item is not None)
        base = {k: 0. for k in ("speed_mps", "acceleration_mps2", "allowed_speed_mps", "runway_m",
            "observation_age_s", "receipt_age_s", "map_lane_match", "command_present", "issued_acceleration_mps2",
            "native_owner", "damping_owner", "wall_owner")}
        for side in ("front", "rear"):
            base.update({side+"_"+key: 0. for key in ("present", "gap_m", "relative_speed_mps", "coverage_m", "closing_ttc_capped_s")})
        if item:
            receipt, r = item
            f = r.frame
            base.update(speed_mps=f.speed_mps, acceleration_mps2=f.acceleration_mps2,
                allowed_speed_mps=f.allowed_speed_mps, runway_m=f.distance_to_merge_m or 0.,
                observation_age_s=now_s-f.timestamp_s, receipt_age_s=now_s-receipt,
                map_lane_match=float(f.lane_id == member.lane_id),
                command_present=float(r.issued_acceleration_mps2 is not None),
                issued_acceleration_mps2=r.issued_acceleration_mps2 or 0.,
                native_owner=float(r.owner == "native"), damping_owner=float(r.owner == "damping"),
                wall_owner=float(r.owner == "wall"))
            for side, sign in (("front", -1.), ("rear", 1.)):
                n = getattr(f, side)
                base[side+"_coverage_m"] = getattr(f, side+"_coverage_m")
                if n:
                    closing = sign*n.relative_speed_mps
                    base.update({side+"_present": 1., side+"_gap_m": n.gap_m,
                        side+"_relative_speed_mps": n.relative_speed_mps,
                        side+"_closing_ttc_capped_s": min(30., n.gap_m/closing) if closing > 0 else 30.})
        own = [r for _, r in retained if r.frame.vehicle_id == member.vehicle_id]
        speeds, accels = [r.frame.speed_mps for r in own], [r.frame.acceleration_mps2 for r in own]
        base.update(history_samples=float(len(own)), history_speed_mean_mps=mean(speeds) if speeds else 0.,
                    history_speed_std_mps=pvariance(speeds)**.5 if speeds else 0.,
                    history_acceleration_rms_mps2=mean(a*a for a in accels)**.5 if accels else 0.)
        values.update({prefix+k: v for k, v in base.items()})
    for key in ("initial_or_current_span_m", "speed_spread_mps", "estimated_assembly_s", "estimated_runway_slack_m"):
        val = getattr(assessment, key)
        values[key] = val if val is not None else 0.
        values[key+"_present"] = float(val is not None)
    if any(not isfinite(v) for v in values.values()):
        raise ValueError("nonfinite report features")
    evidence = [dict(received_at_s=t, **r.payload()) for t, r in retained]
    binding = dict(schema=SCHEMA, proposal_sha256=proposal.fingerprint,
        evidence_sha256=digest(evidence), assessment_evidence_sha256=assessment.evidence_fingerprint,
        now_s=now_s, history_s=history_s, values=values)
    return dict(binding, row_sha256=digest(binding), proposal=asdict(proposal),
                claim="CAV_acquired_features_not_a_fitted_or_validated_gate")
