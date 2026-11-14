"""Synthetic exact-budget and fail-closed registration/launch regressions."""
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest


def load(name):
    return importlib.import_module("traffic_shepherd."+name)


campaign = load("duration_campaign")


def job(seed):
    return dict(id=f"s{seed}_release", seed=seed, duration_s=2400)


def test_budget_charges_full_prefix_and_only_eligible_sources():
    one = campaign.budget([job(4107)])
    two = campaign.budget([job(4107), job(4108)])
    assert one["maximum_simulated_seconds"] == 2400
    assert two["maximum_simulated_seconds"] == 4800
    assert two["maximum_attempts"] == 2
    assert two["prefix_is_charged_in_full_episode"]
    assert two["new_mechanics_seconds"] == two["automatic_retries"] == 0
    assert not two["automatic_successors"]


@pytest.mark.parametrize("jobs", [[], [job(4108)], [job(4107), job(4107)],
    [job(4107), job(9999)], [dict(job(4107), duration_s=1783)], [job(4107), job(4108), job(4109)]])
def test_budget_rejects_unregistered_seed_duplicate_or_suffix_shortcut(jobs):
    with pytest.raises(ValueError):
        campaign.budget(jobs)


@pytest.fixture
def launch(tmp_path, monkeypatch):
    assigned = job(4107)
    data = dict(jobs=[assigned], new_job_ids=[assigned["id"]], registration_sha256="bound")
    monkeypatch.setattr(campaign, "read_registration", lambda *args, **kwargs: (tmp_path, data))
    monkeypatch.setattr(campaign, "_policy", lambda: SimpleNamespace(policy_factory=lambda *args: object()))
    return tmp_path, data, assigned


def test_previous_allocation_is_never_retried(launch, monkeypatch):
    root, _, assigned = launch
    (root / "allocations").mkdir()
    (root / "allocations" / (assigned["id"]+".json")).write_text("{}")
    monkeypatch.setattr(campaign.paired, "evaluate_arm", lambda **kwargs: pytest.fail("duplicate physical launch"))
    with pytest.raises(FileExistsError):
        campaign.run_one("registration", assigned["id"])


def test_simulator_failure_preserves_full_charge_and_stops_block(launch, monkeypatch):
    root, _, assigned = launch
    def fail(**kwargs):
        raise RuntimeError("synthetic acquisition failure")
    monkeypatch.setattr(campaign.paired, "evaluate_arm", fail)
    with pytest.raises(RuntimeError):
        campaign.run_one("registration", assigned["id"])
    allocation = json.loads((root / "allocations" / (assigned["id"]+".json")).read_text())
    assert allocation["allocated_simulated_seconds"] == 2400
    assert allocation["automatic_retries"] == 0
    assert "synthetic acquisition" in json.loads((root / "STOP.json").read_text())["reason"]
    with pytest.raises(ValueError, match="stopped"):
        campaign.run_one("registration", assigned["id"])


def test_stopped_analysis_cannot_claim_complete(launch):
    root, _, _ = launch
    (root / "STOP.json").write_text("{}")
    with pytest.raises(ValueError, match="stopped"):
        campaign.analyze("registration")


def test_extra_allocation_is_not_an_implicit_second_arm(launch, monkeypatch):
    root, _, assigned = launch
    (root / "allocations").mkdir()
    (root / "allocations" / "new_seed.json").write_text("{}")
    monkeypatch.setattr(campaign.paired, "evaluate_arm", lambda **kwargs: pytest.fail("unregistered launch"))
    with pytest.raises(ValueError, match="unregistered allocation"):
        campaign.run_one("registration", assigned["id"])


def test_duration_binding_preserves_original_hold_profile_and_cohort(monkeypatch):
    from dataclasses import asdict
    from traffic_shepherd.lifecycle_wall import LifecycleProfile
    duration = load("duration_study")
    monkeypatch.setattr(campaign, "_policy", lambda: duration)
    original = dict(id="source", case_id="c", seed=4107, loading=2700, penetration=.2,
        role="exposed", arm="wall", duration_s=2400, decision_time_s=600,
        policy_spec=dict(member_ids=["a", "b"], required_lanes=["l0", "l1"], profile=asdict(LifecycleProfile())))
    copy = deepcopy(original)
    branch = dict(version=duration.VERSION, hold_offset_s=5., expected_hold_start_s=612., branch_time_s=617.,
        branch_evidence_sha256="a"*64, previous_wall_commands_sha256="b"*64,
        proposal_sha256="c"*64, source_manifest_sha256="d"*64)
    result = campaign._job(original, dict(branch_spec=branch, branch_time_s=617.))
    assert original == copy
    assert result["policy_spec"]["profile"]["hold_s"] == 20
    assert result["policy_spec"]["member_ids"] == ["a", "b"]
    assert result["snapshot_times_s"] == [600, 617]
    assert result["duration_s"] == 2400
