"""Execute the real round-11 Bash gate with an isolated stub training executable."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def shell_path(path):
    text = Path(path).resolve().as_posix()
    return f"/mnt/{text[0].lower()}{text[2:]}" if os.name == "nt" else text


def launch(tmp_path, *, stale=None, train_exit=0, verdict_exit=0, head_init=None):
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
printf '%s\\n' "$*" >> "$R11_ROOT/calls"
if [[ "$*" == *wms.train.round11_gate* ]]; then
    [[ "$*" == *"--train-exit-code 0"* ]] || exit 3
    exit "$VERDICT_EXIT"
elif [[ "$*" == *runs/smoke_r11* ]]; then
    exit "$TRAIN_EXIT"
fi
touch "$R11_ROOT/full-launched"
''', encoding="utf-8", newline="\n")
    script = f'''
export R11_ROOT={shlex.quote(shell_path(root))}
export R11_INIT={shlex.quote(shell_path(init))}
export R11_PYTHON={shlex.quote(shell_path(fake))}
export TRAIN_EXIT={train_exit} VERDICT_EXIT={verdict_exit}
{'' if head_init is None else 'export R11_HEAD_INIT=' + head_init}
chmod +x "$R11_PYTHON"
bash {shlex.quote(shell_path(ROOT / 'scripts/round11_gate.sh'))} {shlex.quote(shell_path(ROOT))}
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
    script = (f"export R11_ROOT={shlex.quote(shell_path(root))} R11_INIT={shlex.quote(shell_path(root / 'none.pt'))}\n"
              f"bash {shlex.quote(shell_path(ROOT / 'scripts/round11_gate.sh'))} {shlex.quote(shell_path(ROOT))}\n")
    command = (["wsl.exe", "--exec"] if os.name == "nt" else []) + ["bash", "--noprofile", "--norc", "-s"]
    result = subprocess.run(command, input=script.encode("utf-8"), capture_output=True, timeout=30)
    assert result.returncode == 2
    assert not (root / ".round11_gate.lock").exists()


@pytest.mark.parametrize("artifact", (
    "runs/smoke_r11", "runs/c4_wm_v10_round11", "smoke_r11.log", "smoke_r11.verdict",
    "smoke_r11.gate.json", ".round11_gate.lock",
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
                     "curriculum.max_hours=1.5", "configs/c4_wm_v10round11.toml", "model.init_from="):
        assert required in smoke
    assert "wms.train.round11_gate" in verdict and "--train-exit-code 0" in verdict
    assert "runs/c4_wm_v10_round11" in full
    assert "curriculum.stage_steps" not in full and "selection.val_max_windows=16" in full
    assert "--init-checkpoint" in verdict
    assert (root / "full-launched").exists()
    assert (root / ".round11_gate.lock").is_dir()


@pytest.mark.parametrize("train_exit,verdict_exit", ((7, 0), (0, 3)))
def test_smoke_or_verdict_failure_never_launches(tmp_path, train_exit, verdict_exit):
    root, result = launch(tmp_path, train_exit=train_exit, verdict_exit=verdict_exit)
    assert result.returncode == 3, result.stderr
    calls = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert f"--train-exit-code {train_exit}" in calls[1]
    assert not (root / "full-launched").exists()


@pytest.mark.parametrize("head_init,zero", ((None, "false"), ("checkpoint", "false"), ("zero", "true")))
def test_head_init_switch_reaches_smoke_full_and_verdict(tmp_path, head_init, zero):
    root, result = launch(tmp_path, head_init=head_init)
    assert result.returncode == 0, result.stderr
    smoke, verdict, full = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert f"model.follower_courtesy_init_zero={zero}" in smoke
    assert f"model.follower_courtesy_init_zero={zero}" in full
    expect = "zero" if zero == "true" else "checkpoint"
    assert f"--expect-courtesy-init {expect}" in verdict
    rule = json.loads((ROOT / "docs/r11_weight_rule.json").read_text(encoding="utf-8"))["switch_zero_init_head"]
    for call in (smoke, full):
        has = (f"loss.follower_courtesy.weight={rule['weight']}" in call
               and f"optim.lr_scale.follower_courtesy={rule['lr_scale']}" in call)
        assert has == (zero == "true")


def test_bad_head_init_is_a_setup_failure(tmp_path):
    root, result = launch(tmp_path, head_init="random")
    assert result.returncode == 2
    assert not (root / "calls").exists() and not (root / ".round11_gate.lock").exists()
