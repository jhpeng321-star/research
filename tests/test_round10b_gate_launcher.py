"""Execute the real round-10b Bash gate with an isolated stub training executable."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def shell_path(path):
    text = Path(path).resolve().as_posix()
    return f"/mnt/{text[0].lower()}{text[2:]}" if os.name == "nt" else text


def launch(tmp_path, *, stale=None, train_exit=0, verdict_exit=0):
    root = tmp_path / "root with spaces"
    root.mkdir()
    if stale:
        artifact = root / stale
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("prior evidence", encoding="utf-8")
    init = tmp_path / "init.pt"
    init.write_text("checkpoint", encoding="utf-8")
    fake = tmp_path / "fake-python.sh"
    fake.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$R10B_ROOT/calls"
if [[ "$*" == *wms.train.round10b_gate* ]]; then
    [[ "$*" == *"--train-exit-code 0"* ]] || exit 3
    exit "$VERDICT_EXIT"
elif [[ "$*" == *runs/smoke_r10b* ]]; then
    exit "$TRAIN_EXIT"
fi
touch "$R10B_ROOT/full-launched"
''', encoding="utf-8", newline="\n")
    script = f'''
export R10B_ROOT={shlex.quote(shell_path(root))}
export R10B_INIT={shlex.quote(shell_path(init))}
export R10B_PYTHON={shlex.quote(shell_path(fake))}
export TRAIN_EXIT={train_exit} VERDICT_EXIT={verdict_exit}
chmod +x "$R10B_PYTHON"
bash {shlex.quote(shell_path(ROOT / 'scripts/round10b_gate.sh'))} {shlex.quote(shell_path(ROOT))}
'''
    command = (["wsl.exe", "--exec"] if os.name == "nt" else [])
    command += ["bash", "--noprofile", "--norc", "-s"]
    # Bytes preserve LF on Windows and tolerate WSL's mixed-encoding diagnostics.
    result = subprocess.run(command, input=script.encode("utf-8"),
                            capture_output=True, timeout=30)
    return root, result


def test_missing_init_checkpoint_is_a_setup_failure(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    script = (f"export R10B_ROOT={shlex.quote(shell_path(root))} R10B_INIT={shlex.quote(shell_path(root / 'none.pt'))}\n"
              f"bash {shlex.quote(shell_path(ROOT / 'scripts/round10b_gate.sh'))} {shlex.quote(shell_path(ROOT))}\n")
    command = (["wsl.exe", "--exec"] if os.name == "nt" else []) + ["bash", "--noprofile", "--norc", "-s"]
    result = subprocess.run(command, input=script.encode("utf-8"), capture_output=True, timeout=30)
    assert result.returncode == 2
    assert not (root / ".round10b_gate.lock").exists()


@pytest.mark.parametrize("artifact", (
    "runs/smoke_r10b", "runs/c4_wm_v10_round10b", "smoke_r10b.log", "smoke_r10b.verdict",
    "smoke_r10b.gate.json", ".round10b_gate.lock",
))
def test_stale_artifacts_are_refused_without_overwrite(tmp_path, artifact):
    root, result = launch(tmp_path, stale=artifact)
    assert result.returncode == 4, result.stderr
    assert (root / artifact).read_text(encoding="utf-8") == "prior evidence"
    assert not (root / "calls").exists()
    assert not (root / "full-launched").exists()


def test_smoke_arguments_and_success_launch(tmp_path):
    root, result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    calls = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    smoke, verdict, full = calls
    for required in ("optim.batch_size=4", "optim.batch_size_per_stage=[4,4]",
                     "curriculum.unroll_s=[120.0,300.0]", "curriculum.stage_steps=[2,2]",
                     "runtime.device=cuda", "--mode run", "selection.val_max_windows=4",
                     "curriculum.max_hours=1.5", "configs/c4_wm_v10round10b.toml", "model.init_from="):
        assert required in smoke
    assert "wms.train.round10b_gate" in verdict and "--train-exit-code 0" in verdict
    assert "runs/c4_wm_v10_round10b" in full
    assert "curriculum.stage_steps" not in full and "selection.val_max_windows=16" in full
    assert "--init-checkpoint" in verdict
    assert (root / "full-launched").exists()
    assert (root / ".round10b_gate.lock").is_dir()


@pytest.mark.parametrize("train_exit,verdict_exit", ((7, 0), (0, 3)))
def test_smoke_or_verdict_failure_never_launches(tmp_path, train_exit, verdict_exit):
    root, result = launch(tmp_path, train_exit=train_exit, verdict_exit=verdict_exit)
    assert result.returncode == 3, result.stderr
    calls = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert f"--train-exit-code {train_exit}" in calls[1]
    assert not (root / "full-launched").exists()
