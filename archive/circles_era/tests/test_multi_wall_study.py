from output.multi_wall_probe_v1.study import preflight, score
import pytest


def test_preflight_has_six_locked_source_jobs():
    x = preflight()
    assert x["prior_sources_unchanged"] is True
    assert len(x["jobs"]) == 6
    assert {j["seed"] for j in x["jobs"]} == {90030, 90045, 90046}
    assert {j["policy_spec"]["multi_wall_mode"] for j in x["jobs"]} == {
        "same_cohort_extend20",
        "distinct_upstream_h20",
    }
    from traffic_shepherd.paired_experiments import validate_job

    for j in x["jobs"]:
        validate_job(j)


def test_preflight_preserves_source_binding():
    x = preflight()
    assert any("s90030_h20" in path for path in x["sources"])


def test_score_keeps_physical_vector():
    s = {
        "system_seconds_per_requested_vehicle": 10,
        "system_vehicle_seconds": 100,
        "population": 10,
        "totals": {
            "braking_vehicle_seconds": 2,
            "acceleration_energy": 3,
            "ttc_below_3_vehicle_seconds": 4,
            "collision_involvements": 1,
            "starting_teleports": 0,
        },
        "source_wait_vehicle_seconds": 5,
        "unfinished_requested": 0,
        "requested_due_at_horizon": 10,
        "arrived": 10,
        "all_requested_service_completed": True,
        "mechanics_only": False,
        "complete_fixed_horizon": True,
    }
    criteria = {
        "maximum_cost_increase": {
            "system_seconds_per_request": 0.5,
            "source_wait_seconds_per_request": 0.1,
            "braking_seconds_per_request": 0.005,
            "effort_per_request": 0.5,
            "ttc_seconds_per_request": 0.02,
        },
        "primary_minimum_improvement": {
            "system_seconds_per_request": 1.0,
            "braking_seconds_per_request": 0.02,
        },
    }
    x = score(s, damping_summary=s, h20_summary=s, criteria=criteria)
    assert x["collisions"] == 1 and x["unfinished"] == 0


def test_score_rejects_missing_physical_totals():
    s = {
        "population": 10,
        "unfinished_requested": 0,
        "totals": {},
        "mechanics_only": False,
        "complete_fixed_horizon": True,
    }
    with pytest.raises((KeyError, TypeError)):
        score(s, damping_summary=s, h20_summary=s, criteria={})


def test_oe59_oe62_seed_partition_registration_is_the_seeds_py_delta():
    """Seed-partition registration pin. History: OE-59 (2026-09-14) appended
    OE59_CONDITIONAL_PANEL_SEEDS (exposure receipt 5e6d2029...); CV-20
    (2026-09-20) appended the audited CV20_MULTISEED_CORPUS_SEEDS block
    (seeds 7001-7012, dated grep-audit comment in seeds.py) plus its
    gate and prospective-exclusion entry; CV-28 (2026-09-21) appended
    CV28_VALIDATION_HOLDOUT_SEEDS (seeds 7101-7102) with the same audit;
    CV-29 (2026-09-21) appended CV29_CORPUS_EXTENSION_SEEDS (7301-7306);
    CV-31 (2026-09-22) appended CV31_COVERAGE_EXTENSION_SEEDS (7531-7536,
    m2400/m2700 coverage extension, dated grep-audit comment). Bytes are
    pinned; any further change must re-register."""
    import hashlib
    sha = hashlib.sha256(open("traffic_shepherd/seeds.py", "rb").read()).hexdigest()
    assert sha == "dab2ae39b8bc7ca40d656dbd4ab87b9341f24d1fecc2f394937db9a205dab3fd"
