"""The JSON-in / JSON-out CLI (real subprocess, real stdout)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from _tiny import default_plans, tiny_state, write_tiny_checkpoint

_API_ROOT = Path(__file__).resolve().parents[2]      # output/wd01/wm_n1


def _run(args, cwd=_API_ROOT):
    return subprocess.run(
        [sys.executable, "-m", "wm_api", *args], cwd=str(cwd),
        capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli_ckpt")
    write_tiny_checkpoint(tmp, k=1)
    return tmp


def test_cli_example_flag():
    proc = _run(["score", "--example"])
    assert proc.returncode == 0, proc.stderr
    req = json.loads(proc.stdout)
    assert req["horizon_s"] == 300
    assert len(req["candidate_plans"]) == 3
    assert req["candidate_plans"][2]["wall"]["members"] == ["cav0", "cav1"]


def test_cli_score_json_out(ckpt, tmp_path):
    request = {"state": tiny_state().to_json(),
               "candidate_plans": default_plans(),
               "horizon_s": 60, "seed": 5}
    req_path = tmp_path / "req.json"
    out_path = tmp_path / "resp.json"
    req_path.write_text(json.dumps(request), encoding="utf-8")
    proc = _run(["score", "--ckpt", str(ckpt), "--request", str(req_path),
                 "--out", str(out_path), "--allow-other-horizon"])
    assert proc.returncode == 0, proc.stderr
    envelope = json.loads(proc.stdout)
    assert envelope["ok"] is True and envelope["api_version"] == "1.0"
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["api_version"] == "1.0"
    assert data["seed"] == 5
    assert len(data["plans"]) == 2
    assert data["checkpoint"]["sha_verified"] is False


def test_cli_refusal_exit_code_and_envelope(ckpt, tmp_path):
    request = {"state": tiny_state().to_json(),
               "candidate_plans": [{"actions": {"cav0": "lurch"}}],
               "horizon_s": 60}
    req_path = tmp_path / "bad.json"
    req_path.write_text(json.dumps(request), encoding="utf-8")
    proc = _run(["score", "--ckpt", str(ckpt), "--request", str(req_path),
                 "--allow-other-horizon"])
    assert proc.returncode == 2
    err = json.loads(proc.stdout)
    assert err["error"]["type"] == "UnknownActionError"
    assert err["error"]["api_version"] == "1.0"
    assert "lurch" in err["error"]["message"]
