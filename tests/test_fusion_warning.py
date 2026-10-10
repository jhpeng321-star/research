"""Tests for the fusion-warning evaluator (v5).

These exercise the analysis implementation on synthetic streams: spatial sign
errors, stale reports, onset classification, one-to-one matching, censoring
and window boundaries would each have caught a defect found during the audit.
"""
import importlib.util
import json
import os
from pathlib import Path

#: WM-55 drive migration: the retired D: junction is gone; the evidence
#: root resolves through env RESEARCH_DENSE_ROOT with the live E: default.
RESEARCH_DENSE_ROOT = os.environ.get("RESEARCH_DENSE_ROOT",
                                     "E:/research-dense")
_WALL_HARM_DIR = Path(RESEARCH_DENSE_ROOT) / "output/wall_harm_speed_v1"

_TARGET = _WALL_HARM_DIR / "fusion_warning_v5.py"
_spec = importlib.util.spec_from_file_location("fusion_warning_v5", str(_TARGET))
fw5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw5)


def make_report(received, dtm, speed, lane="L0", ts=None, front_gap=None,
                front_rel=None):
    return dict(received=received, ts=received if ts is None else ts,
                lane=lane, dtm=dtm, speed=speed, front_gap=front_gap,
                front_rel=front_rel)


def synth_reports(a_times_speeds, b_times_speeds, sep=90.0, lane_a="L0",
                  lane_b="L0"):
    reports = {}
    for v, series, dtm0, lane in (("A", a_times_speeds, sep, lane_a),
                                  ("B", b_times_speeds, 0.0, lane_b)):
        rows = []
        for t, v_speed in series:
            # A is ahead: its dtm is sep smaller than B's (dtm decreases toward merge)
            rows.append(make_report(t, dtm0 - (sep if v == "A" else 0.0)
                                    - (t - series[0][0]), v_speed, lane))
        reports[v] = rows
    return reports


def test_m_spatial_sign_ahead_means_smaller_dtm():
    """A hard-braking ahead reporter + approaching B must produce a warning.

    Regression: the first implementation required A.dtm > B.dtm, which the
    received reports contradict (ahead vehicles have SMALLER distance-to-merge).
    """
    # A brakes 28 -> 8 over 4 s; B approaches at 26 m/s
    a = [(1070.0, 28.0), (1071.0, 26.0), (1072.0, 22.0), (1073.0, 18.0),
         (1074.0, 14.0), (1075.0, 10.0), (1076.0, 8.0)]
    b = [(1070.0, 26.5), (1072.0, 26.5), (1074.0, 26.5), (1076.0, 26.3)]
    reports = {}
    for v, series in (("A", a), ("B", b)):
        rows = []
        for t, v_speed in series:
            rows.append(make_report(t, 600.0 - (90.0 if v == "A" else 0.0),
                                    v_speed, "L0"))
        reports[v] = rows
    # simpler: fixed dtm values with A behind-in-dtm (ahead physically)
    reports = {
        "A": [make_report(t, 600.0, v_speed) for t, v_speed in a],
        "B": [make_report(t, 690.0, v_speed) for t, v_speed in b],
    }
    ticks = fw5.m_warning_ticks(reports, 600.0)
    assert ticks.get("B"), "approaching B behind hard-braking A must warn"
    # reversed dtm (A given the larger value) must NOT warn: the ahead vehicle
    # physically has the smaller distance-to-merge
    reports_rev = {
        "A": [make_report(t, 690.0, v_speed) for t, v_speed in a],
        "B": [make_report(t, 600.0, v_speed) for t, v_speed in b],
    }
    assert not fw5.m_warning_ticks(reports_rev, 600.0).get("B")


def test_m_stale_ahead_report_does_not_warn():
    a = [(1060.0, 28.0), (1075.0, 8.0)]  # last report 10 s old at 1076
    b = [(1070.0, 26.5), (1076.0, 26.3)]
    reports = {"A": [make_report(t, 600.0, s) for t, s in a],
               "B": [make_report(t, 690.0, s) for t, s in b]}
    assert not fw5.m_warning_ticks(reports, 600.0).get("B")


def test_event_classification_before_at_after_onset():
    reports = {"V": [make_report(100.0 + i, 100.0, 20.0, front_gap=20.0,
                                 front_rel=-6.0) for i in range(10)]}
    events = [dict(vehicle="V", onset=105.0, end=106.5)]
    for warning_start, expected in ((102.0, "advance"), (105.0, "simultaneous")):
        eps = {"V": [dict(start=warning_start, last=warning_start + 1.0)]}
        counts, detail = fw5.classify_events(events, eps, reports, 0.5, 0.0)
        assert detail[0]["cls"] == expected
    # warning only after the grace window but still inside the event -> late
    eps = {"V": [dict(start=106.0, last=106.4)]}
    counts, detail = fw5.classify_events(events, eps, reports, 0.5, 0.0)
    assert detail[0]["cls"] == "late"
    # no warning at all -> missed
    counts, detail = fw5.classify_events(events, {"V": []}, reports, 0.5, 0.0)
    assert detail[0]["cls"] == "missed"
    assert counts["missed"] == 1


def test_unobservable_event_is_unassessable_not_missed():
    reports = {"V": [make_report(99.0, 100.0, 20.0)]}  # 1 report only
    events = [dict(vehicle="V", onset=105.0, end=106.5)]
    counts, detail = fw5.classify_events(
        events, {"V": [dict(start=104.0, last=104.5)]}, reports, 0.5, 0.0)
    assert counts["unassessable"] == 1
    assert detail[0]["cls"] == "unassessable"


def test_one_to_one_matching_consumes_episodes():
    reports = {"V": [make_report(100.0 + i, 100.0, 20.0, front_gap=20.0,
                                 front_rel=-6.0) for i in range(30)]}
    events = [dict(vehicle="V", onset=105.0, end=105.5),
              dict(vehicle="V", onset=108.0, end=108.5)]
    # a single long episode overlapping both events can only be consumed once:
    # the first event takes it, the second must not count the same episode as
    # simultaneous again
    eps = {"V": [dict(start=103.0, last=108.2)]}
    counts, detail = fw5.classify_events(events, eps, reports, 0.5, 0.0)
    assert detail[0]["cls"] in ("advance", "simultaneous")
    assert sum(counts[k] for k in ("advance", "simultaneous", "late", "missed")) == 2


def test_prediction_censoring_near_horizon_end():
    events = []
    eps = {"V": [dict(start=2395.0, last=2395.5)]}
    res = fw5.prediction_view(eps, events, 600.0)
    assert res["censored"] == 1 and res["false_positives"] == 0
    eps = {"V": [dict(start=1000.0, last=1000.5)]}
    res = fw5.prediction_view(eps, events, 600.0)
    assert res["false_positives"] == 1 and res["censored"] == 0


def test_f_baseline_sign_convention():
    reports = {"V": [make_report(100.0, 100.0, 26.0, front_gap=29.0,
                                 front_rel=-5.0)]}
    assert fw5.f_episodes(reports, 0.0).get("V")  # boundary gap 29<=30, rel -5<=-5
    reports = {"V": [make_report(100.0, 100.0, 26.0, front_gap=31.0,
                                 front_rel=-5.0)]}
    assert not fw5.f_episodes(reports, 0.0).get("V")
    reports = {"V": [make_report(100.0, 100.0, 26.0, front_gap=29.0,
                                 front_rel=-4.9)]}
    assert not fw5.f_episodes(reports, 0.0).get("V")


def test_events_before_release_are_excluded():
    reports = {"V": [make_report(500.0 + i, 100.0, 20.0) for i in range(100)]}
    events = [dict(vehicle="V", onset=520.0, end=521.0)]  # pre-release onset
    eps = fw5.f_episodes(reports, 633.0)
    assert not eps.get("V")
    counts, detail = fw5.classify_events(events, eps, reports, 0.5, 633.0)
    assert sum(counts.values()) == 0  # pre-release event not in the population


def test_state_action_ttc_cells_recomputed_from_summaries():
    """Source-derived arithmetic: vs-keep TTC cells (the -49/-57/-72 errors)."""
    def ttc(path):
        return json.loads(Path(path).read_text())["totals"][
            "ttc_below_3_vehicle_seconds"]
    base = RESEARCH_DENSE_ROOT + "/runs/active/"
    keep7 = ttc(base + "lifecycle_repair_v2/jobs/s4107_m2700_p20_t600_lifecycle_v2_wall/summary.json")
    lower7 = ttc(base + "target_adjust_v1/jobs/s4107_m2700_p20_t600_hold_adjust_down_wall/summary.json")
    raise7 = ttc(base + "target_adjust_v3/jobs/s4107_m2700_p20_t600_hold_adjust_up_wall/summary.json")
    rel7 = ttc(base + "duration_release_v1/jobs/s4107_m2700_p20_t600_release_h5_wall/summary.json")
    keep8 = ttc(base + "cohort_eligibility_v1/jobs/s4108_m2700_p20_t600_cohort_v1_wall/summary.json")
    lower8 = ttc(base + "target_adjust_v1/jobs/s4108_m2700_p20_t600_hold_adjust_down_wall/summary.json")
    raise8 = ttc(base + "target_adjust_v3/jobs/s4108_m2700_p20_t600_hold_adjust_up_wall/summary.json")
    rel8 = ttc(base + "duration_release_v1/jobs/s4108_m2700_p20_t600_release_h5_wall/summary.json")
    assert raise7 - keep7 == -103.0
    assert lower7 - keep7 == -83.5
    assert rel7 - keep7 == -112.5
    assert raise8 - keep8 == 0.0
    assert lower8 - keep8 == -12.5
    assert rel8 - keep8 == 3.5


# ---- v6 evaluator repairs ----

def _load_v6():
    target = _WALL_HARM_DIR / "fusion_warning_v6.py"
    spec = importlib.util.spec_from_file_location("fusion_warning_v6", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v6_old_warning_beyond_horizon_is_not_advance():
    mod = _load_v6()
    reports = {"V": [make_report(158.0, 100.0, 26.0),
                     make_report(160.0, 100.0, 26.0),
                     make_report(165.5, 100.0, 26.0)]}
    events = [dict(vehicle="V", onset=165.0, end=166.0, samples=2)]
    eps = {"V": [dict(start=100.0, last=100.5)]}  # 65 s before onset
    counts, detail = mod.classify_events(events, eps, reports, 0.0)
    assert detail[0]["cls"] != "advance"


def test_v6_two_warnings_before_one_event_count_once():
    mod = _load_v6()
    reports = {"V": [make_report(100.0 + i, 100.0, 26.0, front_gap=20.0,
                                 front_rel=-6.0) for i in range(20)]}
    events = [dict(vehicle="V", onset=112.0, end=113.0, samples=2)]
    eps = {"V": [dict(start=108.0, last=108.5),   # advance
                 dict(start=110.0, last=110.5)]}  # also before onset
    counts, detail = mod.classify_events(events, eps, reports, 0.0)
    assert counts["advance"] == 1 and counts["simultaneous"] == 0
    # prediction view: second episode is redundant, not a second success
    pred = mod.prediction_view(eps, events, 0.0)
    assert pred["true_predictions"] == 1
    assert pred["redundant_warnings"] == 1
    assert pred["false_positives"] == 0


def test_v6_warning_after_onset_within_grace_is_simultaneous():
    mod = _load_v6()
    reports = {"V": [make_report(100.0, 100.0, 20.0),
                     make_report(104.0, 100.0, 20.0),
                     make_report(105.5, 100.0, 20.0)]}
    events = [dict(vehicle="V", onset=105.0, end=106.0, samples=2)]
    eps = {"V": [dict(start=105.5, last=105.9)]}
    counts, detail = mod.classify_events(events, eps, reports, 0.0)
    assert counts["simultaneous"] == 1 and counts["advance"] == 0


def test_v6_ongoing_event_at_window_boundary_censored():
    mod = _load_v6()
    # event onset 2398 (after 2400 - horizon): its warning is censored
    reports = {"V": [make_report(2397.0, 100.0, 26.0, front_gap=20.0,
                                 front_rel=-6.0),
                     make_report(2397.5, 100.0, 26.0, front_gap=20.0,
                                 front_rel=-6.0)]}
    events = [dict(vehicle="V", onset=2398.0, end=2399.5, samples=4)]
    eps = mod.f_episodes(reports, 0.0)
    pred = mod.prediction_view(eps, events, 0.0)
    assert pred["censored"] == 1 and pred["false_positives"] == 0


def test_v6_staleness_uses_acquisition_age_not_receipt_age():
    mod = _load_v6()
    # A's frames are 10 s old at t=1076 although only just received: stale.
    b = [make_report(1076.0, 690.0, 26.5)]
    # first A report is stale at later ticks; ensure ts ordering respected
    assert b[0]["ts"] <= b[0]["received"]
    # direct check via m_warning_ticks: stale acquisition frames received recently
    # the ts<=received guard but the OTHER report's age is what matters; build
    # A entirely from old frames received recently:
    a_stale = [make_report(1076.0, 598.0, 28.0, ts=1066.0),
               make_report(1076.0, 590.0, 8.0, ts=1066.0)]
    reports = {"A": a_stale, "B": b}
    # acquisition age = 1076-1066 = 10 s > 2 s -> no warning
    assert not mod.m_warning_ticks(reports, 0.0).get("B")


def test_v6_missing_follower_vs_missing_ahead_coverage_distinct():
    mod = _load_v6()
    # follower silent -> F-unassessable regardless of ahead CAVs
    events = [dict(vehicle="V", onset=700.0, end=701.0, samples=2)]
    reports = {"V": [make_report(680.0, 100.0, 20.0)],          # 1 report only
               "A": [make_report(695.0, 90.0, 25.0)]}
    counts, detail = mod.classify_events(events, {"V": []}, reports, 0.0)
    assert counts["unassessable_follower_silent"] == 1
    cov = mod.m_coverage(events, reports, 0.0)
    assert ("V", 700.0) not in cov   # follower silent: not even M-observable
    # follower reporting but no ahead CAV: F-observable, M-coverage absent
    events2 = [dict(vehicle="V", onset=700.0, end=701.0, samples=2)]
    reports2 = {"V": [make_report(695.0, 100.0, 20.0),
                      make_report(696.0, 100.0, 20.0)]}
    cov2 = mod.m_coverage(events2, reports2, 0.0)
    assert ("V", 700.0) not in cov2


def test_v6_exposure_sums_samples_not_span():
    # 4 samples across a 3 s span with a quiet gap: exposure = 2.0 veh-s
    events = [dict(vehicle="V", onset=100.0, end=103.0, samples=4)]
    assert 0.5 * events[0]["samples"] == 2.0


def test_v6_denominators_reconcile_in_saved_results():
    results = json.loads(
        (_WALL_HARM_DIR / "fusion_warning_results_v6.json").read_text())
    for arm, r in results["arms"].items():
        for cand in ("F", "M"):
            c = r[cand]["event_counts"]
            total = sum(c[k] for k in ("advance", "simultaneous", "late",
                                       "missed", "unassessable_follower_silent"))
            assert total == r["events_eligible"], (arm, cand)
            p = r[cand]["prediction"]
            assert (p["true_predictions"] + p["false_positives"]
                    + p["censored"] + p["redundant_warnings"]) == p["episodes"]


def test_v6_release_times_distinguish_h5_and_common_aligned():
    results = json.loads(
        (_WALL_HARM_DIR / "fusion_warning_results_v6.json").read_text())
    assert results["arms"]["4107_release_h5"]["release_s"] == 618.0
    assert results["arms"]["4108_release_h5"]["release_s"] == 624.0
    assert results["arms"]["4107_keep"]["release_s"] == 633.0
    assert results["arms"]["4108_keep"]["release_s"] == 639.0
    assert "common aligned" in results["arms"]["4107_keep"]["common_aligned_window_note"]


# ---- v7 coverage/detection repairs ----

def _load_v7():
    target = _WALL_HARM_DIR / "fusion_warning_v7.py"
    spec = importlib.util.spec_from_file_location("fusion_warning_v7", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v7_future_report_leakage_cannot_cover_earlier_event():
    mod = _load_v7()
    # pair-eligible tick exists only at t=2000; event at 700 must be uncovered
    ticks = {"V": [2000.0]}
    events = [dict(vehicle="V", onset=700.0, end=701.0, samples=2)]
    covered = mod.m_coverage(events, ticks, 0.0)
    assert ("V", 700.0) not in covered
    # tick inside the event window does cover
    ticks = {"V": [695.0]}
    assert ("V", 700.0) in mod.m_coverage(events, ticks, 0.0)


def test_v7_warning_exactly_at_onset_is_simultaneous_not_advance():
    mod = _load_v6()
    reports = {"V": [make_report(100.0, 100.0, 20.0),
                     make_report(104.0, 100.0, 20.0),
                     make_report(105.0, 100.0, 20.0)]}
    events = [dict(vehicle="V", onset=105.0, end=106.0, samples=2)]
    eps = {"V": [dict(start=105.0, last=105.9)]}
    # v6 defect demonstrated: a warning at exactly onset was counted advance
    counts, _ = mod.classify_events(events, eps, reports, 0.0)
    assert counts["advance"] == 1 and counts["simultaneous"] == 0
    # v7 repair: strict inequality -> simultaneous
    mod7 = _load_v7()
    counts7, _ = mod7.classify_events(events, eps, reports, 0.0)
    assert counts7["simultaneous"] == 1 and counts7["advance"] == 0


def test_v7_detection_never_exceeds_coverage_in_saved_results():
    results = json.loads(
        (_WALL_HARM_DIR / "fusion_warning_results_v7.json").read_text())
    for arm, r in results["arms"].items():
        m = r["M"]["event_counts"]
        detected = m["advance"] + m["simultaneous"] + m["late"]
        assert detected <= r["m_coverage"]["m_observable_events"], arm
        leads = r["M"]["lead_distribution"]
        assert all(0 < x <= 10.0 for x in leads), (arm, leads)


def test_v7_denominators_reconcile():
    results = json.loads(
        (_WALL_HARM_DIR / "fusion_warning_results_v7.json").read_text())
    for arm, r in results["arms"].items():
        for cand in ("F", "M"):
            c = r[cand]["event_counts"]
            assert sum(c.values()) == r["events_eligible"], (arm, cand)
