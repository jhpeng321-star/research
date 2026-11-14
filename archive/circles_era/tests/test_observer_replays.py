"""Budget and fail-closed launch tests; never starts SUMO."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from traffic_shepherd import observer_replays as replay


def test_selected_jobs_are_exact_original_five_and_unchanged():
    source = json.loads(replay.SOURCE.read_text(encoding="utf-8"))
    selected = replay.selected_jobs(source)
    assert tuple(j["id"] for j in selected) == replay.JOB_IDS
    assert sum(j["duration_s"] for j in selected) == 12000
    assert [j["arm"] for j in selected] == ["damping", "wall", "assembly_only", "damping", "wall"]
    assert all(j in source["jobs"] for j in selected)
    selected[0]["duration_s"] = 1200
    with pytest.raises(ValueError, match="identity"):
        replay.selected_jobs(source)


@pytest.fixture
def launch(tmp_path, monkeypatch):
    calls = []
    contract = dict(sources={name: dict(path=str(tmp_path / "original" / name), manifest_sha256="original") for name in replay.JOB_IDS})
    monkeypatch.setattr(replay, "read_contract", lambda _: (tmp_path, {}, contract))
    def run(*args):
        calls.append(args)
        return {"complete": True}
    monkeypatch.setattr(replay.paired, "run_one", run)
    checker = SimpleNamespace(compare_replay=lambda *a: {"exact_replay": True,
        "original_manifest_file_sha256": "original", "replay_manifest_file_sha256": "replayed"},
        validate_diagnostics=lambda *a: {"diagnostic_channel_validated": True})
    monkeypatch.setattr(replay, "sha", lambda _: "replayed")
    monkeypatch.setitem(sys.modules, "traffic_shepherd.observer_replay_evidence", checker)
    return tmp_path, calls, checker


def test_later_jobs_cannot_launch_without_first_integrity_pass(launch):
    root, calls, _ = launch
    with pytest.raises(ValueError, match="first damping"):
        replay.run_one(root / "registration.json", replay.JOB_IDS[1])
    assert calls == []
    assert not (root / "STOP.json").exists()


def test_success_is_recorded_and_failed_comparison_stops_block(launch):
    root, calls, checker = launch
    replay.run_one(root / "registration.json", replay.JOB_IDS[0])
    assert (root / "comparisons" / f"{replay.JOB_IDS[0]}.json").exists()
    checker.compare_replay = lambda *a: {"exact_replay": False, "reason": "physical mismatch"}
    with pytest.raises(ValueError, match="mismatch"):
        replay.run_one(root / "registration.json", replay.JOB_IDS[1])
    assert len(calls) == 2
    stopped = json.loads((root / "STOP.json").read_text())
    assert not stopped["automatic_retry"]
    assert not (root / "comparisons" / f"{replay.JOB_IDS[1]}.json").exists()


def test_unknown_job_has_no_allocation_or_stop_side_effect(launch):
    root, calls, _ = launch
    with pytest.raises(ValueError, match="outside"):
        replay.run_one(Path(root) / "registration.json", "new_seed")
    assert not calls and not (root / "STOP.json").exists()
