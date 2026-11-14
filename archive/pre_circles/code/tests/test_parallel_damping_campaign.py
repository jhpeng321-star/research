"""Scheduling checks use synthetic files only; no SUMO or reserved seeds run."""

from dataclasses import asdict
import hashlib
import json
from threading import Event

import pytest

from tools import parallel_damping_campaign as runner
from idm.damping_scenario import TripRequest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_root(root):
    write(root / "manifest.json", dict(source_sha256={}))
    write(root / "active_stage.json", dict(status="complete"))


def complete(root, seed, episode):
    write(root / "active_stage.json", dict(status="complete"))
    write(
        root / f"lineage_{seed}/stage_summary.json",
        dict(status="stage_complete", optimizer_seed=seed, optimized_episodes=episode),
    )
    write(
        root / f"development/optimizer_{seed}_episode_{episode:03}/summary.json",
        dict(optimizer_seed=seed, optimized_global_steps=episode * 2400),
    )


@pytest.mark.parametrize("seed", [72, 73])
def test_worker_runs_only_assigned_lineage_once_in_registered_order(tmp_path, monkeypatch, seed):
    setup_root(tmp_path)
    calls = []

    def run(root, initialization, stage, heartbeat, log_dir):
        calls.append(stage)
        complete(root, *stage)
        return 0

    monkeypatch.setattr(runner, "run_stage", run)
    args = (tmp_path, tmp_path, seed, None, lambda *args: None, tmp_path / "logs", Event())
    runner.run_worker(*args)
    runner.run_worker(*args)
    assert calls == [(seed, episode) for episode in (20, 62, 100)]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_failed_or_incomplete_stage_stops_without_retry(tmp_path, monkeypatch, exit_code):
    setup_root(tmp_path)
    calls = []

    def run(root, initialization, stage, heartbeat, log_dir):
        calls.append(stage)
        return exit_code

    monkeypatch.setattr(runner, "run_stage", run)
    with pytest.raises(RuntimeError, match="failed or is incomplete"):
        runner.run_worker(
            tmp_path, tmp_path, 72, None, lambda *args: None, tmp_path / "logs", Event()
        )
    assert calls == [(72, 20)]


def test_failure_signal_prevents_new_stage_launch(tmp_path, monkeypatch):
    setup_root(tmp_path)
    signal = Event()
    signal.set()
    calls = []
    monkeypatch.setattr(runner, "run_stage", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="another worker failed"):
        runner.run_worker(
            tmp_path, tmp_path, 72, None, lambda *args: None, tmp_path / "logs", signal
        )
    assert not calls


def test_exact_copy_reuses_identical_file_and_rejects_conflicting_bytes(tmp_path):
    source, target = tmp_path / "original", tmp_path / "copy"
    source.write_bytes(b"immutable evidence")
    expected = runner.copy_exact(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert runner.copy_exact(source, target) == expected
    target.write_bytes(b"different evidence")
    with pytest.raises(ValueError, match="conflicting existing file"):
        runner.copy_exact(source, target)
    assert target.read_bytes() == b"different evidence"
    assert source.read_bytes() == b"immutable evidence"


def test_consolidation_rejects_incomplete_worker_before_copying_or_selection(tmp_path):
    workers = {71: tmp_path, 72: tmp_path / "workers/72", 73: tmp_path / "workers/73"}
    for seed, root in workers.items():
        setup_root(root)
        for episode in (20, 62, 100):
            if (seed, episode) != (73, 100):
                complete(root, seed, episode)
    with pytest.raises(RuntimeError, match="all registered worker evaluations"):
        runner.consolidate(tmp_path, workers)
    assert not (tmp_path / "lineage_72").exists()
    assert not (tmp_path / "lineage_73").exists()
    assert not (tmp_path / "development_selection.json").exists()
    assert not (tmp_path / "parallel_execution/consolidation.json").exists()


@pytest.mark.parametrize(
    "active,pid,seed",
    [
        (dict(status="running", pid=42, optimizer_seed=71, through_episode=62), 43, 71),
        (dict(status="complete", pid=42, optimizer_seed=71, through_episode=62), 42, 71),
        (dict(status="running", pid=42, optimizer_seed=72, through_episode=62), 42, 71),
        (dict(status="running", pid=42, optimizer_seed=71, through_episode=999), 42, 71),
    ],
)
def test_adoption_rejects_mismatched_identity_before_wait_or_launch(
    tmp_path, monkeypatch, active, pid, seed
):
    setup_root(tmp_path)
    write(tmp_path / "active_stage.json", active)
    calls = []
    monkeypatch.setattr(runner, "wait_existing", lambda *args: calls.append("wait"))
    monkeypatch.setattr(runner, "run_stage", lambda *args: calls.append("launch"))
    with pytest.raises(ValueError, match="adopted PID disagrees"):
        runner.run_worker(
            tmp_path, tmp_path, seed, pid, lambda *args: None, tmp_path / "logs", Event()
        )
    assert not calls


@pytest.mark.parametrize("arm", ["training_native", "development_native", "analytical"])
@pytest.mark.parametrize(
    "field,value", [("seed", 123), ("scenario_identity", "wrong-scenario"), ("population", 999)]
)
def test_cache_rejects_mislabeled_seed_scenario_or_population(
    tmp_path, monkeypatch, arm, field, value
):
    monkeypatch.setattr(runner, "TRAINING_CASES", ((910001, 1900),))
    monkeypatch.setattr(runner, "DEVELOPMENT_CASES", ((911001, 2000),))
    requests = [TripRequest("mainline.0", 0.5, "mainline_route", "normal_car", True)]
    monkeypatch.setattr(runner, "request_schedule", lambda *args, **kwargs: requests)
    targets = {}
    for role, seed in (("training", 910001), ("development", 911001)):
        folder = tmp_path / "scenarios" / f"{role}_{seed}"
        folder.mkdir(parents=True)
        config_bytes, route_bytes = b"synthetic config", b"synthetic route"
        (folder / "scenario.sumocfg").write_bytes(config_bytes)
        (folder / "requests.rou.xml").write_bytes(route_bytes)
        identity = hashlib.sha256(
            json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
            + config_bytes
            + route_bytes
        ).hexdigest()
        summary = dict(
            seed=seed, scenario_identity=identity, population=1,
            team_terminal=True, global_control_steps=2400,
        )
        paths = [tmp_path / "references" / f"{role}_{seed}.json"]
        targets[f"{role}_native"] = paths[0]
        if role == "development":
            paths.append(tmp_path / "analytical" / f"development_{seed}.json")
            targets["analytical"] = paths[-1]
        for path in paths:
            write(path, dict(summary=summary))
            path.with_suffix(".npz").write_bytes(b"synthetic array placeholder")
    assert len(runner.cached_paths(tmp_path)) == 6
    target = targets[arm]
    corrupted = json.loads(target.read_text())
    corrupted["summary"][field] = value
    write(target, corrupted)
    with pytest.raises(ValueError, match="cached reference identity mismatch"):
        runner.cached_paths(tmp_path)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_adopted_stage_requires_success_and_completion(tmp_path, monkeypatch, exit_code):
    setup_root(tmp_path)
    write(
        tmp_path / "active_stage.json",
        dict(status="running", pid=42, optimizer_seed=71, through_episode=62),
    )
    calls = []
    monkeypatch.setattr(runner, "wait_existing", lambda *args: exit_code)
    monkeypatch.setattr(runner, "run_stage", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="adopted stage failed or is incomplete"):
        runner.run_worker(
            tmp_path, tmp_path, 71, 42, lambda *args: None, tmp_path / "logs", Event()
        )
    assert not calls
