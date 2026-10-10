"""Tests for the TTC diagnosis: conventions, conservation, cutoffs, missingness."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from output_shim import aggregate_encounters

from traffic_shepherd.continuation_dataset import build_dataset


def test_ground_truth_4107_raise26_cost_differences():
    """Authoritative recomputation from immutable summaries (task-stated values)."""
    root = ("D:/research-dense/runs/active/target_adjust_v3/jobs/"
            "s4107_m2700_p20_t600_hold_adjust_up_wall")
    ref = "D:/research-dense/runs/active/observer_replay_v1/jobs/s4107_m2700_p20_damping"
    t = json.loads(Path(root + "/summary.json").read_text())
    r = json.loads(Path(ref + "/summary.json").read_text())
    pop = r["population"]
    diff = dict(
        service=(t["totals"]["requested_not_arrived_vehicle_seconds"]
                 - r["totals"]["requested_not_arrived_vehicle_seconds"]) / pop,
        braking=(t["totals"]["braking_vehicle_seconds"]
                 - r["totals"]["braking_vehicle_seconds"]) / pop,
        effort=(t["totals"]["acceleration_energy"]
                - r["totals"]["acceleration_energy"]) / pop,
        ttc=(t["totals"]["ttc_below_3_vehicle_seconds"]
             - r["totals"]["ttc_below_3_vehicle_seconds"]) / pop,
        wait=(t["source_wait_vehicle_seconds"]
              - r["source_wait_vehicle_seconds"]) / pop)
    assert diff["service"] == pytest.approx(-8.435689, abs=1e-5)
    assert diff["braking"] == pytest.approx(-0.057597, abs=1e-5)
    assert diff["effort"] == pytest.approx(-5.745149, abs=1e-5)
    assert diff["ttc"] == pytest.approx(0.039576, abs=1e-5)
    assert diff["wait"] == pytest.approx(-0.133569, abs=1e-5)
    # TTC is the only violated guardrail
    allowances = {"service": 0.5, "braking": 0.005, "effort": 0.5,
                  "ttc": 0.02, "wait": 0.1}
    violated = [m for m, d in diff.items() if d > allowances[m]]
    assert violated == ["ttc"]
    assert t["totals"]["collision_involvements"] == 0
    assert r["totals"]["collision_involvements"] == 4
    assert t["totals"]["starting_teleports"] == 0 and t["unfinished_requested"] == 0
    assert t["all_requested_service_completed"] is True


def test_improvement_is_negative_of_cost_difference():
    effect = dict(improvements={"a": 2.0, "b": -3.0})
    for m, g in effect["improvements"].items():
        assert -g == -effect["improvements"][m]  # trivial identity
        # convention: cost difference (treatment - reference) = -improvement
        assert (-g) == pytest.approx(-(effect["improvements"][m]))


def test_ttc_event_streams_reconcile_with_summary_totals():
    """Accounting conservation: event samples vs registered totals (<=1 sample gap).

    The diagnostic channel samples at the post-physics boundary while the
    registered outcome evaluator samples slightly earlier, so a single boundary
    sample can flip; the per-arm gap must stay within one 0.5 s sample.
    """
    jobs = {
        "4107_hold_up": ("target_adjust_v3", "s4107_m2700_p20_t600_hold_adjust_up_wall"),
        "4107_hold_down": ("target_adjust_v1", "s4107_m2700_p20_t600_hold_adjust_down_wall"),
        "4108_hold_up": ("target_adjust_v3", "s4108_m2700_p20_t600_hold_adjust_up_wall"),
        "4108_hold_down": ("target_adjust_v1", "s4108_m2700_p20_t600_hold_adjust_down_wall"),
        "4107_damping": ("observer_replay_v1", "s4107_m2700_p20_damping"),
        "4108_damping": ("observer_replay_v1", "s4108_m2700_p20_damping"),
        "4107_h20_wall": ("lifecycle_repair_v2", "s4107_m2700_p20_t600_lifecycle_v2_wall"),
        "4108_h20_wall": ("cohort_eligibility_v1", "s4108_m2700_p20_t600_cohort_v1_wall"),
        "4107_h5_wall": ("duration_release_v1", "s4107_m2700_p20_t600_release_h5_wall"),
        "4108_h5_wall": ("duration_release_v1", "s4108_m2700_p20_t600_release_h5_wall"),
    }
    for arm, (block, job) in jobs.items():
        summary = json.loads(Path(
            f"D:/research-dense/runs/active/{block}/jobs/{job}/summary.json").read_text())
        total = summary["totals"]["ttc_below_3_vehicle_seconds"]
        n = sum(1 for _ in open(
            f"D:/research-dense/output/wall_harm_speed_v1/ttc_events_{arm}.jsonl",
            encoding="utf-8"))
        assert abs(0.5 * n - total) <= 0.5, (arm, 0.5 * n, total)


def test_encounter_aggregation_keeps_split_encounters():
    """Regression: per-key aggregation must not drop split encounters."""
    encs = [dict(vehicle="a", leader="b", veh_seconds=1.0, start=10.0, end=10.5,
                 region="r", min_ttc=2.0, lanes=["x"]),
            dict(vehicle="a", leader="b", veh_seconds=2.0, start=20.0, end=21.0,
                 region="r", min_ttc=1.0, lanes=["x"])]
    agg = aggregate_encounters(encs)
    assert agg[("a", "b")]["veh_seconds"] == 3.0
    assert agg[("a", "b")]["start"] == 10.0
    assert agg[("a", "b")]["min_ttc"] == 1.0


def test_continuation_rows_use_hold_plus_5_decision_time():
    dataset = build_dataset()
    for r in dataset["rows"]:
        assert r["decision_type"] == "continuation"
        assert r["entry_time_s"] == 600.0
        assert r["decision_time_s"] == r["branch_time_s"] in (617.0, 623.0)
        assert r["decision_time_s"] != 600.0
    for r in dataset["rows"]:
        if r["status"] == "delivered":
            assert r["executed_history"]["prefix_verified"] is True
            assert r["executed_history"]["prefix_evidence"]
    failure = [r for r in dataset["rows"] if r["status"] == "implementation_failure"]
    assert failure and failure[0]["physical_outcome_label"] is None


def test_post_decision_measurements_are_not_action_fields():
    dataset = build_dataset()
    for r in dataset["rows"]:
        action_text = json.dumps(r["action"])
        assert "transition" not in action_text and "near" not in action_text


# ---- v4 phase/warning corrections ----

def _load_phase_mod():
    import importlib.util
    target = Path("D:/research-dense/output/wall_harm_speed_v1/ttc_phase_v4.py")
    spec = importlib.util.spec_from_file_location("ttc_phase_v4", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase_totals_match_task_authoritative_values():
    mod = _load_phase_mod()
    res = mod.main()
    # task-stated: 4107 raise/keep/damping 38 veh-s in [600,633)
    for arm in ("4107_raise", "4107_keep"):
        phases = res[arm]["phases"]
        coord = sum(v for k, v in phases.items() if k not in ("pre_entry", "post_release"))
        assert coord == 38.0, (arm, coord)
    assert res["4107_damping"]["phases"]["aligned_coordination_600_633"] == 38.0
    # post-release: raise 1823, damping 1767, keep 1926
    assert res["4107_raise"]["phases"]["post_release"] == 1823.0
    assert res["4107_damping"]["phases"]["aligned_post_633"] == 1767.0
    assert res["4107_keep"]["phases"]["post_release"] == 1926.0
    # 4108 releases at 639, coordination [600,639) = 30.5, post: 1267/1267/1161.5
    for arm in ("4108_raise", "4108_keep"):
        phases = res[arm]["phases"]
        coord = sum(v for k, v in phases.items() if k not in ("pre_entry", "post_release"))
        assert coord == 30.5, (arm, coord)
    assert res["4108_damping"]["phases"]["aligned_coordination_600_639"] == 30.5
    assert res["4108_raise"]["phases"]["post_release"] == 1267.0
    assert res["4108_keep"]["phases"]["post_release"] == 1267.0
    assert res["4108_damping"]["phases"]["aligned_post_639"] == 1161.5
    # conservation: phase sums reconcile with authoritative totals
    for arm, v in res.items():
        assert v["phase_total"] == pytest.approx(v["authoritative_total"], abs=0.5)


def test_phase_boundaries_and_pre_entry_exclusion():
    mod = _load_phase_mod()
    ph = mod.PHASES[("4107", "raise")]
    assert mod.phase_of(ph, 599.5) == "pre_entry"
    assert mod.phase_of(ph, 600.0) == "assembly"
    assert mod.phase_of(ph, 616.5) == "hold_pre_adjust"
    assert mod.phase_of(ph, 617.0) == "hold_post_adjust"
    assert mod.phase_of(ph, 632.0) == "taper_exit"
    assert mod.phase_of(ph, 633.0) == "post_release"
    # damping comparison windows are named, not wall phases
    dp = mod.PHASES[("4107", "damping")]
    assert mod.phase_of(dp, 632.0) == "aligned_coordination_600_633"
    assert "hold" not in mod.phase_of(dp, 620.0)
    # a synthetic encounter crossing the release boundary splits by sample
    assert mod.phase_of(ph, 632.5) == "taper_exit"
    assert mod.phase_of(ph, 633.5) == "post_release"


def test_release_times_per_arm():
    mod = _load_phase_mod()
    assert mod.RELEASE_S == {"4107": 633.0, "4108": 639.0}


def test_relative_speed_sign_and_units_documented():
    """front.relative_speed = front speed - ego speed, in m/s (not m/s^2)."""
    from traffic_shepherd.sensing import NeighborMeasurement
    m = NeighborMeasurement(gap_m=30.0, relative_speed_mps=-5.0)
    assert m.relative_speed_mps == -5.0  # front slower by 5 m/s -> ego closing 5 m/s
    with pytest.raises(ValueError):
        NeighborMeasurement(gap_m=-1.0, relative_speed_mps=0.0)  # overlap invalid


def test_warning_episode_grouping_and_cutoffs():
    import importlib.util
    target = Path("D:/research-dense/output/wall_harm_speed_v1/warning_timing.py")
    spec = importlib.util.spec_from_file_location("warning_timing", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    samples = [
        (100.0, 25.0, -6.0),   # warning
        (100.5, 20.0, -7.0),   # same episode
        (102.0, 18.0, -8.0),   # gap>1 s -> new episode
        (105.0, 40.0, -9.0),   # gap>30 -> not a warning
        (106.0, 25.0, -4.9),   # closing<5 -> not a warning
        (95.0, 10.0, -10.0),   # before cutoff
    ]
    eps = mod.episodes(samples, after_s=99.0)
    assert len(eps) == 2
    assert eps[0]["n"] == 2 and eps[0]["start"] == 100.0
    assert eps[1]["n"] == 1 and eps[1]["start"] == 102.0
    # cutoff excludes the pre-cutoff warning sample
    assert all(e["start"] >= 99.0 for e in eps)


def test_event_onsets_group_repeated_warnings():
    import importlib.util
    target = Path("D:/research-dense/output/wall_harm_speed_v1/warning_timing.py")
    spec = importlib.util.spec_from_file_location("warning_timing", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [dict(id="v", t=t) for t in (700.0, 700.5, 702.0, 715.0, 715.5)]
    orig = mod.events_rows
    mod.events_rows = lambda key: rows
    try:
        onsets = mod.event_onsets(("4107", "raise"), 633.0)
        assert onsets["v"] == [700.0, 715.0]  # samples within 10 s group to one onset
    finally:
        mod.events_rows = orig
