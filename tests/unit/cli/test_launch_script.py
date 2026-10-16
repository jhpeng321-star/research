"""scripts/box/launch.sh: LF endings, argument handling (run with a fake interpreter)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "box" / "launch.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def launch(*args, **env):
    full_env = {"PATH": __import__("os").environ["PATH"], "SW_ROOT": "/sw", "PY": shutil.which("echo")}
    full_env.update(env)
    return subprocess.run([BASH, str(SCRIPT), *args], capture_output=True, text=True, env=full_env)


def test_script_has_lf_line_endings_only():
    assert b"\r" not in SCRIPT.read_bytes()


def test_run_forwards_mode_config_and_overrides():
    result = launch("run", "runtime.seed=7", W="/fresh/dir")
    assert result.returncode == 0
    assert "-m wms train --config" in result.stdout and "--mode run" in result.stdout
    assert "--out /fresh/dir" in result.stdout and "--set runtime.seed=7" in result.stdout
    assert "--set runtime.omp=2" in result.stdout


def test_default_mode_is_preflight_and_stage_is_refused():
    assert "--mode preflight" in launch().stdout
    refused = launch("stage")
    assert refused.returncode == 2 and "preflight or run" in refused.stderr


def test_requires_switching_root_and_positive_omp():
    assert launch(SW_ROOT="").returncode != 0
    bad = launch(OMP="0")
    assert bad.returncode == 2 and "OMP" in bad.stderr
