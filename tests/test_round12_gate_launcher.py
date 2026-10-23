"""Execute the real round-12 Bash gate with an isolated stub training executable."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/round12_gate.sh"


def shell_path(path):
    text = Path(path).resolve().as_posix()
    return f"/mnt/{text[0].lower()}{text[2:]}" if os.name == "nt" else text


def bash(script, timeout=30):
    command = (["wsl.exe", "--exec"] if os.name == "nt" else []) + ["bash", "--noprofile", "--norc", "-s"]
    # Bytes preserve LF on Windows and tolerate WSL's mixed-encoding diagnostics.
    return subprocess.run(command, input=script.encode("utf-8"), capture_output=True, timeout=timeout)


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
printf '%s\\n' "$*" >> "$R12_ROOT/calls"
if [[ "$*" == *wms.train.round12_gate* ]]; then
    [[ "$*" == *"--train-exit-code 0"* ]] || exit 3
    exit "$VERDICT_EXIT"
elif [[ "$*" == *runs/smoke_r12* ]]; then
    exit "$TRAIN_EXIT"
fi
touch "$R12_ROOT/full-launched"
''', encoding="utf-8", newline="\n")
    script = f'''
export R12_ROOT={shlex.quote(shell_path(root))}
export R12_INIT={shlex.quote(shell_path(init))}
export R12_PYTHON={shlex.quote(shell_path(fake))}
export TRAIN_EXIT={train_exit} VERDICT_EXIT={verdict_exit}
chmod +x "$R12_PYTHON"
bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}
'''
    return root, init, bash(script)


def test_script_bytes_are_lf_and_carry_the_contract():
    data = SCRIPT.read_bytes()
    assert b"\r" not in data
    text = data.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    for required in ("set -u -o pipefail -o noclobber", "R12_ROOT", "R12_PYTHON", "R12_INIT",
                     "runs/c4_wm_v10_round9/wm_s_stage300.pt", "runs/c4_wm_v10_round9/wm_s_best.pt",
                     'mkdir "$L/.round12_gate.lock" || exit 4', "ulimit -n 65536 || exit 2",
                     "configs/c4_wm_v10round12.toml", "wms.train.round12_gate", "selection.val_max_windows=4",
                     "selection.val_max_windows=16", "curriculum.max_hours=1.5", '"curriculum.stage_steps=[2,2]"',
                     '"curriculum.truncation_steps_per_stage=[120,60]"', '"optim.batch_size_per_stage=[4,4]"',
                     "loss.brake_follow_cav_weight=2.0", "loss.loss_share_logging=false", "runtime.device=cuda",
                     '--report "$L/smoke_r12.gate.json" > "$L/smoke_r12.verdict"', "exit 3", "exec "):
        assert required in text, required
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "follower_courtesy_init_zero" not in code and "HEAD_INIT" not in code
    assert "loss.follower_courtesy.weight" not in code and "lr_scale" not in code
    assert "cutin" not in code


def test_missing_init_checkpoint_is_a_setup_failure(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    result = bash(f"export R12_ROOT={shlex.quote(shell_path(root))} "
                  f"R12_INIT={shlex.quote(shell_path(root / 'none.pt'))}\n"
                  f"bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}\n")
    assert result.returncode == 2
    assert not (root / ".round12_gate.lock").exists()


@pytest.mark.parametrize("artifact", (
    "runs/smoke_r12", "runs/c4_wm_v10_round12", "smoke_r12.log", "smoke_r12.verdict",
    "smoke_r12.gate.json", ".round12_gate.lock",
))
def test_stale_artifacts_are_refused_without_overwrite(tmp_path, artifact):
    root, _, result = launch(tmp_path, stale=artifact)
    assert result.returncode == 4, result.stderr
    assert (root / artifact).read_text(encoding="utf-8") == "prior evidence"
    assert not (root / "calls").exists()
    assert not (root / "full-launched").exists()


def test_concurrent_lock_is_refused(tmp_path):
    root, _, first = launch(tmp_path)
    assert first.returncode == 0, first.stderr
    assert (root / ".round12_gate.lock").is_dir()
    # a second launcher on the same root finds the retained reservation (and the prior artifacts)
    fake = tmp_path / "fake-python.sh"
    result = bash(f"export R12_ROOT={shlex.quote(shell_path(root))} R12_INIT={shlex.quote(shell_path(tmp_path / 'init.pt'))} "
                  f"R12_PYTHON={shlex.quote(shell_path(fake))}\n"
                  f"bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}\n")
    assert result.returncode == 4
    assert len((root / "calls").read_text(encoding="utf-8").splitlines()) == 3


def test_smoke_arguments_and_success_launch(tmp_path):
    root, init, result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    calls = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    smoke, verdict, full = calls
    for required in ("optim.batch_size=4", "optim.batch_size_per_stage=[4,4]",
                     "curriculum.unroll_s=[120.0,300.0]", "curriculum.stage_steps=[2,2]",
                     "curriculum.truncation_steps_per_stage=[120,60]",
                     "runtime.device=cuda", "--mode run", "selection.val_max_windows=4",
                     "curriculum.max_hours=1.5", "configs/c4_wm_v10round12.toml",
                     f"model.init_from={shell_path(init)}", "loss.loss_share_logging=false",
                     "loss.brake_follow_cav_weight=2.0", f"--out {shell_path(root)}/runs/smoke_r12"):
        assert required in smoke, required
    for required in ("-m wms.train.round12_gate", f"--smoke-dir {shell_path(root)}/runs/smoke_r12",
                     f"--smoke-log {shell_path(root)}/smoke_r12.log", "--train-exit-code 0",
                     f"--init-checkpoint {shell_path(init)}", "--config configs/c4_wm_v10round12.toml",
                     f"--report {shell_path(root)}/smoke_r12.gate.json"):
        assert required in verdict, required
    assert "--expect-courtesy-init" not in verdict
    assert f"--out {shell_path(root)}/runs/c4_wm_v10_round12" in full
    assert "configs/c4_wm_v10round12.toml" in full and f"model.init_from={shell_path(init)}" in full
    assert "curriculum.stage_steps" not in full and "curriculum.max_hours" not in full
    assert "selection.val_max_windows=16" in full and "selection.val_max_windows=4" not in full
    for call in (smoke, full):
        assert "follower_courtesy_init_zero" not in call
        assert "loss.follower_courtesy.weight" not in call and "lr_scale" not in call
    assert (root / "full-launched").exists()
    assert (root / ".round12_gate.lock").is_dir()
    assert (root / "smoke_r12.log").exists() and (root / "smoke_r12.verdict").exists()


@pytest.mark.parametrize("train_exit,verdict_exit", ((7, 0), (0, 3)))
def test_smoke_or_verdict_failure_never_launches(tmp_path, train_exit, verdict_exit):
    root, _, result = launch(tmp_path, train_exit=train_exit, verdict_exit=verdict_exit)
    assert result.returncode == 3, result.stderr
    calls = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert f"--train-exit-code {train_exit}" in calls[1]
    assert not (root / "full-launched").exists()


@pytest.mark.parametrize("present,expected", ((("wm_s_stage300.pt", "wm_s_best.pt"), "wm_s_stage300.pt"),
                                              (("wm_s_best.pt",), "wm_s_best.pt")))
def test_default_init_is_round9_final_with_best_fallback(tmp_path, present, expected):
    root = tmp_path / "root"
    run = root / "runs" / "c4_wm_v10_round9"
    run.mkdir(parents=True)
    for name in present:
        (run / name).write_text("checkpoint", encoding="utf-8")
    fake = tmp_path / "fake-python.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$R12_ROOT/calls"\n',
                    encoding="utf-8", newline="\n")
    result = bash(f"export R12_ROOT={shlex.quote(shell_path(root))} R12_PYTHON={shlex.quote(shell_path(fake))}\n"
                  'chmod +x "$R12_PYTHON"\n'
                  f"bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}\n", timeout=60)
    assert result.returncode == 0, result.stderr
    smoke, verdict, full = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert f"model.init_from={shell_path(run / expected)}" in smoke
    assert f"--init-checkpoint {shell_path(run / expected)}" in verdict
    assert f"model.init_from={shell_path(run / expected)}" in full


def test_r12_init_override_wins_over_default(tmp_path):
    root = tmp_path / "root"
    run = root / "runs" / "c4_wm_v10_round9"
    run.mkdir(parents=True)
    (run / "wm_s_stage300.pt").write_text("checkpoint", encoding="utf-8")
    other = tmp_path / "other.pt"
    other.write_text("checkpoint", encoding="utf-8")
    fake = tmp_path / "fake-python.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$R12_ROOT/calls"\n',
                    encoding="utf-8", newline="\n")
    result = bash(f"export R12_ROOT={shlex.quote(shell_path(root))} R12_PYTHON={shlex.quote(shell_path(fake))} "
                  f"R12_INIT={shlex.quote(shell_path(other))}\n"
                  'chmod +x "$R12_PYTHON"\n'
                  f"bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}\n", timeout=60)
    assert result.returncode == 0, result.stderr
    calls = (root / "calls").read_text(encoding="utf-8")
    assert f"model.init_from={shell_path(other)}" in calls and "c4_wm_v10_round9" not in calls
