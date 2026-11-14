import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import continue_damping_campaign as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete(root, stage):
    seed, episode = stage
    write(root / "active_stage.json", dict(status="complete"))
    write(
        root / f"lineage_{seed}/stage_summary.json",
        dict(
            status="stage_complete",
            optimizer_seed=seed,
            optimized_episodes=episode,
        ),
    )
    write(
        root / f"development/optimizer_{seed}_episode_{episode:03}/summary.json",
        dict(
            optimizer_seed=seed,
            optimized_global_steps=episode * 2400,
        ),
    )


def setup_root(root):
    write(root / "manifest.json", dict(source_sha256={}))
    write(root / "active_stage.json", dict(status="complete"))


def test_only_registered_remaining_stages_run_once(tmp_path, monkeypatch):
    setup_root(tmp_path)
    complete(tmp_path, (71, 20))
    called = []

    def run(root, initialization, stage, heartbeat):
        called.append(stage)
        complete(root, stage)
        return 0

    monkeypatch.setattr(runner, "run_stage", run)
    runner.continue_campaign(tmp_path, tmp_path, None, lambda *args: None)
    assert called == list(runner.STAGES[1:])
    runner.continue_campaign(tmp_path, tmp_path, None, lambda *args: None)
    assert called == list(runner.STAGES[1:])


@pytest.mark.parametrize("exit_code", [0, 1])
def test_nonzero_exit_or_missing_completion_stops_without_retry(tmp_path, monkeypatch, exit_code):
    setup_root(tmp_path)
    called = []

    def fail(root, initialization, stage, heartbeat):
        called.append(stage)
        return exit_code

    monkeypatch.setattr(runner, "run_stage", fail)
    with pytest.raises(RuntimeError, match="failed or is incomplete"):
        runner.continue_campaign(tmp_path, tmp_path, None, lambda *args: None)
    assert called == [(71, 20)]


def test_live_stage_requires_matching_pid_and_successful_wait(tmp_path, monkeypatch):
    setup_root(tmp_path)
    write(
        tmp_path / "active_stage.json",
        dict(
            status="running",
            pid=42,
            optimizer_seed=71,
            through_episode=20,
        ),
    )
    calls = []
    monkeypatch.setattr(runner, "run_stage", lambda *args: calls.append("unexpected"))
    with pytest.raises(RuntimeError, match="not complete"):
        runner.continue_campaign(tmp_path, tmp_path, None, lambda *args: None)
    with pytest.raises(ValueError, match="adoption disagrees"):
        runner.continue_campaign(tmp_path, tmp_path, 43, lambda *args: None)
    monkeypatch.setattr(runner, "wait_existing", lambda *args: 1)
    with pytest.raises(RuntimeError, match="adopted training stage failed"):
        runner.continue_campaign(tmp_path, tmp_path, 42, lambda *args: None)
    assert not calls


def test_foreign_pid_command_must_match_interpreter_campaign_and_stage(tmp_path):
    root, initialization = tmp_path / "campaign space", tmp_path / "initial space"
    args = [
        sys.executable,
        "-m",
        runner.MODULE,
        "--output",
        str(root),
        "--initialization",
        str(initialization),
        "--optimizer-seed",
        "71",
        "--through-episode",
        "20",
        "--resume",
    ]
    runner.verify_command(subprocess.list2cmdline(args), root, initialization, (71, 20))
    with pytest.raises(ValueError, match="expected campaign trainer"):
        runner.verify_command(subprocess.list2cmdline(args), root, initialization, (71, 62))
    with pytest.raises(ValueError, match="expected campaign trainer"):
        runner.verify_command(
            subprocess.list2cmdline(args), Path("D:/elsewhere"), initialization, (71, 20)
        )
