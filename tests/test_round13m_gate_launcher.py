"""Execute the real round-13M Bash gate with an isolated stub training executable (round-12 contract)."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/round13m_gate.sh"


def shell_path(path):
    text = Path(path).resolve().as_posix()
    return f"/mnt/{text[0].lower()}{text[2:]}" if os.name == "nt" else text


def bash(script, timeout=30):
    command = (["wsl.exe", "--exec"] if os.name == "nt" else []) + ["bash", "--noprofile", "--norc", "-s"]
    # Bytes preserve LF on Windows and tolerate WSL's mixed-encoding diagnostics.
    return subprocess.run(command, input=script.encode("utf-8"), capture_output=True, timeout=timeout)


def launch(tmp_path, *, stale=None, train_exit=0, verdict_exit=0, prior=None):
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
printf '%s\\n' "$*" >> "$R13M_ROOT/calls"
if [[ "$*" == *wms.train.round13f_gate* ]]; then
    [[ "$*" == *"--train-exit-code 0"* ]] || exit 3
    exit "$VERDICT_EXIT"
elif [[ "$*" == *runs/smoke_r13m* ]]; then
    exit "$TRAIN_EXIT"
fi
touch "$R13M_ROOT/full-launched"
''', encoding="utf-8", newline="\n")
    script = f'''
export R13M_ROOT={shlex.quote(shell_path(root))}
export R13M_INIT={shlex.quote(shell_path(init))}
export R13M_PYTHON={shlex.quote(shell_path(fake))}
export TRAIN_EXIT={train_exit} VERDICT_EXIT={verdict_exit}
{'' if prior is None else 'export R13M_PRIOR=' + shlex.quote(prior)}
chmod +x "$R13M_PYTHON"
bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}
'''
    return root, init, bash(script)


def test_script_bytes_are_lf_and_carry_the_contract():
    data = SCRIPT.read_bytes()
    assert b"\r" not in data
    text = data.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    for required in ("set -u -o pipefail -o noclobber", "R13M_ROOT", "R13M_PYTHON", "R13M_INIT", "R13M_PRIOR",
                     '--expect-prior "$PRIOR"', "loss.driver_prior.weight=0.0", "loss.driver_prior.weight=0.01",
                     "runs/c4_wm_v10_round9/wm_s_stage300.pt", "runs/c4_wm_v10_round9/wm_s_best.pt",
                     'mkdir "$L/.round13m_gate.lock" || exit 4', "ulimit -n 65536 || exit 2",
                     "configs/c4_wm_v10round13m.toml", "wms.train.round13f_gate", "selection.val_max_windows=4",
                     "selection.val_max_windows=16", "curriculum.max_hours=1.5", '"curriculum.stage_steps=[2,2]"',
                     '"curriculum.truncation_steps_per_stage=[120,60]"', '"optim.batch_size_per_stage=[4,4]"',
                     "loss.brake_follow_cav_weight=2.0", "loss.loss_share_logging=false", "runtime.device=cuda",
                     '--report "$L/smoke_r13m.gate.json" > "$L/smoke_r13m.verdict"', "exit 3", "exec "):
        assert required in text, required
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "follower_courtesy_init_zero" not in code and "HEAD_INIT" not in code
    assert "loss.follower_courtesy.weight" not in code and "lr_scale" not in code
    assert "cutin" not in code


def test_missing_init_checkpoint_is_a_setup_failure(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    result = bash(f"export R13M_ROOT={shlex.quote(shell_path(root))} "
                  f"R13M_INIT={shlex.quote(shell_path(root / 'none.pt'))}\n"
                  f"bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}\n")
    assert result.returncode == 2
    assert not (root / ".round13m_gate.lock").exists()


@pytest.mark.parametrize("artifact", (
    "runs/smoke_r13m", "runs/c4_wm_v10_round13m", "smoke_r13m.log", "smoke_r13m.verdict",
    "smoke_r13m.gate.json", ".round13m_gate.lock",
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
    assert (root / ".round13m_gate.lock").is_dir()
    # a second launcher on the same root finds the retained reservation (and the prior artifacts)
    fake = tmp_path / "fake-python.sh"
    result = bash(f"export R13M_ROOT={shlex.quote(shell_path(root))} R13M_INIT={shlex.quote(shell_path(tmp_path / 'init.pt'))} "
                  f"R13M_PYTHON={shlex.quote(shell_path(fake))}\n"
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
                     "curriculum.max_hours=1.5", "configs/c4_wm_v10round13m.toml",
                     f"model.init_from={shell_path(init)}", "loss.loss_share_logging=false",
                     "loss.brake_follow_cav_weight=2.0", f"--out {shell_path(root)}/runs/smoke_r13m"):
        assert required in smoke, required
    for required in ("-m wms.train.round13f_gate", f"--smoke-dir {shell_path(root)}/runs/smoke_r13m",
                     f"--smoke-log {shell_path(root)}/smoke_r13m.log", "--train-exit-code 0",
                     f"--init-checkpoint {shell_path(init)}", "--config configs/c4_wm_v10round13m.toml",
                     f"--report {shell_path(root)}/smoke_r13m.gate.json"):
        assert required in verdict, required
    assert "--expect-courtesy-init" not in verdict
    assert f"--out {shell_path(root)}/runs/c4_wm_v10_round13m" in full
    assert "configs/c4_wm_v10round13m.toml" in full and f"model.init_from={shell_path(init)}" in full
    assert "curriculum.stage_steps" not in full and "curriculum.max_hours" not in full
    assert "selection.val_max_windows=16" in full and "selection.val_max_windows=4" not in full
    for call in (smoke, full):
        assert "follower_courtesy_init_zero" not in call
        assert "loss.follower_courtesy.weight" not in call and "lr_scale" not in call
    assert (root / "full-launched").exists()
    assert (root / ".round13m_gate.lock").is_dir()
    assert (root / "smoke_r13m.log").exists() and (root / "smoke_r13m.verdict").exists()


# R13F default prior = off (round 12 was launched with R12_PRIOR=off; it is the paired control)
@pytest.mark.parametrize("prior,override,expect", ((None, "loss.driver_prior.weight=0.0", "off"),
                                                   ("config", None, "config"),
                                                   ("off", "loss.driver_prior.weight=0.0", "off"),
                                                   ("weak", "loss.driver_prior.weight=0.01", "weak")))
def test_prior_switch_reaches_smoke_full_and_verdict(tmp_path, prior, override, expect):
    root, _, result = launch(tmp_path, prior=prior)
    assert result.returncode == 0, result.stderr
    smoke, verdict, full = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert f"--expect-prior {expect}" in verdict
    for call in (smoke, full):
        if override is None:
            assert "loss.driver_prior" not in call
        else:
            assert f"--set {override}" in call and call.count("loss.driver_prior") == 1


@pytest.mark.parametrize("prior", ("strong", "OFF", "0.01"))
def test_bad_prior_is_a_setup_failure(tmp_path, prior):
    root, _, result = launch(tmp_path, prior=prior)
    assert result.returncode == 2
    assert b"R13M_PRIOR" in result.stderr
    assert not (root / "calls").exists() and not (root / ".round13m_gate.lock").exists()


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
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$R13M_ROOT/calls"\n',
                    encoding="utf-8", newline="\n")
    result = bash(f"export R13M_ROOT={shlex.quote(shell_path(root))} R13M_PYTHON={shlex.quote(shell_path(fake))}\n"
                  'chmod +x "$R13M_PYTHON"\n'
                  f"bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}\n", timeout=60)
    assert result.returncode == 0, result.stderr
    smoke, verdict, full = (root / "calls").read_text(encoding="utf-8").splitlines()
    assert f"model.init_from={shell_path(run / expected)}" in smoke
    assert f"--init-checkpoint {shell_path(run / expected)}" in verdict
    assert f"model.init_from={shell_path(run / expected)}" in full


def test_r13m_init_override_wins_over_default(tmp_path):
    root = tmp_path / "root"
    run = root / "runs" / "c4_wm_v10_round9"
    run.mkdir(parents=True)
    (run / "wm_s_stage300.pt").write_text("checkpoint", encoding="utf-8")
    other = tmp_path / "other.pt"
    other.write_text("checkpoint", encoding="utf-8")
    fake = tmp_path / "fake-python.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$R13M_ROOT/calls"\n',
                    encoding="utf-8", newline="\n")
    result = bash(f"export R13M_ROOT={shlex.quote(shell_path(root))} R13M_PYTHON={shlex.quote(shell_path(fake))} "
                  f"R13M_INIT={shlex.quote(shell_path(other))}\n"
                  'chmod +x "$R13M_PYTHON"\n'
                  f"bash {shlex.quote(shell_path(SCRIPT))} {shlex.quote(shell_path(ROOT))}\n", timeout=60)
    assert result.returncode == 0, result.stderr
    calls = (root / "calls").read_text(encoding="utf-8")
    assert f"model.init_from={shell_path(other)}" in calls and "c4_wm_v10_round9" not in calls



def test_round13m_script_is_the_round13f_script_up_to_names():
    old = (ROOT / "scripts/round13f_gate.sh").read_text(encoding="utf-8")
    new = SCRIPT.read_text(encoding="utf-8")

    def body(text):
        return text[text.index("set -u -o pipefail -o noclobber"):]
    back = body(new)
    for a, b in (("R13M_", "R13F_"), ("smoke_r13m", "smoke_r13f"), ("c4_wm_v10_round13m", "c4_wm_v10_round13f"),
                 (".round13m_gate.lock", ".round13f_gate.lock"),
                 ("configs/c4_wm_v10round13m.toml", "configs/c4_wm_v10round13f.toml"), ("repo_r13m", "repo_r13f"),
                 ("round-13M", "round-13F")):
        back = back.replace(a, b)
    assert back == body(old)
