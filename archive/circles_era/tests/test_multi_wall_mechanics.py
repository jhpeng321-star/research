"""Bounded mechanics checks; these tests never start SUMO."""
import importlib.util
import json
from pathlib import Path

import pytest


MECHANICS_PATH = Path(__file__).parents[1] / "output/multi_wall_probe_v1/mechanics.py"
SPEC = importlib.util.spec_from_file_location("multi_wall_mechanics_under_test", MECHANICS_PATH)
MECHANICS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MECHANICS)


def _policy_row(time_s, index, members, phase, hold=None, probe=None):
    return {
        "time_s": time_s,
        "activation_index": index,
        "member_ids": list(members),
        "phase": phase,
        "wall_commands": {member: -0.1 for member in members}
        if phase in ("assembly", "hold", "taper")
        else {},
        "assembly_completed_at_s": time_s - 20 if phase in ("hold", "release") else None,
        "actual_hold_elapsed_s": hold,
        "multi_wall_probe": probe or {
            "branch": {"selected": "extend"},
            "selection": {"proposal_binding": "verified"},
        },
    }


def test_build_fixture_uses_expected_edge_and_pair_positions(tmp_path, monkeypatch):
    def fake_netconvert(command, **_kwargs):
        Path(command[command.index("--output-file") + 1]).write_text(
            "<net/>", encoding="utf-8"
        )

        class Result:
            stdout = ""
            stderr = ""

            def check_returncode(self):
                return None

        return Result()

    monkeypatch.setattr(MECHANICS.subprocess, "run", fake_netconvert)
    config = MECHANICS.build_fixture(tmp_path / "scene", "C:/sumo/bin/sumo.exe")
    scene = config.parent
    assert config.name == "scenario.sumocfg"
    assert "1191554636" in (scene / "edges.xml").read_text(encoding="utf-8")
    routes = (scene / "routes.rou.xml").read_text(encoding="utf-8")
    assert routes.count('departPos="1000"') == 2
    assert routes.count('departPos="800"') == 2
    assert 'departLane="0"' in routes and 'departLane="1"' in routes


def test_check_delivery_accepts_extend20_as_terminal_hold40():
    rows = [
        _policy_row(5, 0, ("member_a", "member_b"), "assembly"),
        _policy_row(25, 0, ("member_a", "member_b"), "hold"),
        _policy_row(65, 0, ("member_a", "member_b"), "release", hold=40),
    ]
    checks, segments = MECHANICS.check_delivery(rows, "same_cohort_extend20")
    assert all(checks.values())
    assert segments[0]["delivered_hold_s"] == 40


@pytest.mark.parametrize("terminal_phase", ["abort", "refused"])
def test_check_delivery_rejects_missing_or_failed_extend_delivery(terminal_phase):
    rows = [
        _policy_row(5, 0, ("member_a", "member_b"), "assembly"),
        _policy_row(25, 0, ("member_a", "member_b"), "hold"),
        _policy_row(65, 0, ("member_a", "member_b"), terminal_phase, hold=19),
    ]
    checks, _ = MECHANICS.check_delivery(rows, "same_cohort_extend20")
    assert not checks["full_hold40"]
    assert not checks["terminal_release"]


def test_check_delivery_accepts_distinct_relay_and_no_third_activation():
    probe = {"branch": None, "selection": {"proposal_binding": "verified"}}
    rows = [
        _policy_row(5, 0, ("member_a", "member_b"), "assembly", probe=probe),
        _policy_row(25, 0, ("member_a", "member_b"), "hold", probe=probe),
        _policy_row(45, 0, ("member_a", "member_b"), "release", hold=20, probe=probe),
        _policy_row(46, 1, ("member_c", "member_d"), "assembly", probe=probe),
        _policy_row(66, 1, ("member_c", "member_d"), "release", hold=20, probe=probe),
    ]
    checks, segments = MECHANICS.check_delivery(rows, "distinct_upstream_h20")
    assert all(checks.values())
    assert [segment["index"] for segment in segments] == [0, 1]


def test_run_rejects_unknown_job_before_budget_allocation(tmp_path, monkeypatch):
    from output.multi_wall_probe_v1 import study

    registration = tmp_path / "registration.json"
    registration.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(study, "RUNS", tmp_path / "runs")
    study.RUNS.mkdir()
    (study.RUNS / "registration.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(study, "load_registration", lambda: {})
    allocations = []

    class ShouldNotAllocate:
        def __init__(self, *_args, **_kwargs):
            allocations.append("constructed")

    monkeypatch.setattr(MECHANICS, "BudgetGuard", ShouldNotAllocate)
    with pytest.raises(ValueError, match="Unknown registered mechanics job"):
        MECHANICS.run(registration, "third_activation")
    assert allocations == []


def test_run_records_120_seconds_and_sticky_stop_on_failure(tmp_path, monkeypatch):
    from output.multi_wall_probe_v1 import study

    runs = tmp_path / "runs"
    (runs / "mechanics").mkdir(parents=True)
    registration = runs / "registration.json"
    registration.write_text("{}", encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(study, "RUNS", runs)
    monkeypatch.setattr(study, "LEDGER", ledger)
    monkeypatch.setattr(study, "CAMPAIGN", "test-campaign")
    monkeypatch.setattr(study, "CEILINGS", {"mechanics": 240})
    monkeypatch.setattr(study, "load_registration", lambda: {"registration_sha256": "abc"})
    monkeypatch.setattr(
        MECHANICS,
        "_execute",
        lambda output, _data, mode: {
            "mode": mode,
            "passed": False,
            "completed_simulated_seconds": 120,
        },
    )

    class FakeGuard:
        calls = []

        def __init__(self, _ledger):
            self.stopped = []

        def reserve(self, **kwargs):
            self.calls.append(("reserve", kwargs["amount_s"], kwargs["category"]))
            kwargs["target_dir"].mkdir(parents=True)
            ledger.write_text(
                json.dumps({"reservations": [{"id": "alloc-1", "target_dir": str(kwargs["target_dir"])}]}),
                encoding="utf-8",
            )
            return kwargs["launcher"]()

        def mark_progress(self, allocation_id, seconds):
            self.calls.append(("progress", allocation_id, seconds))

        def stop_campaign(self, campaign, reason):
            self.calls.append(("stop", campaign, reason))

    monkeypatch.setattr(MECHANICS, "BudgetGuard", FakeGuard)
    summary = MECHANICS.run(registration, "same_cohort_extend20")
    assert summary["completed_simulated_seconds"] == 120
    assert FakeGuard.calls == [
        ("reserve", 120, "mechanics"),
        ("progress", "alloc-1", 120),
        ("stop", "test-campaign", "Failed mechanics same_cohort_extend20"),
    ]
    with pytest.raises(FileExistsError, match="No retries"):
        MECHANICS.run(registration, "same_cohort_extend20")
    assert FakeGuard.calls.count(("reserve", 120, "mechanics")) == 1
