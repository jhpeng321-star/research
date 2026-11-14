"""P12 envcheck: torch >=2.7 gate, Blackwell sm_120 arch gate, CPU fallback."""
from __future__ import annotations

from bundle import envcheck


def test_supports_min_and_window():
    assert envcheck.version_ok("2.7.0+cu128") is True
    assert envcheck.version_ok("2.12.1+cpu") is True
    assert envcheck.version_ok("2.5.1+cu121") is False
    assert envcheck.version_ok("unparseable") is False


def test_has_sm120_arch():
    assert envcheck.has_sm120_arch(["sm_90", "sm_120"]) is True
    assert envcheck.has_sm120_arch(["compute_120"]) is True
    assert envcheck.has_sm120_arch(["sm_80", "sm_90"]) is False
    assert envcheck.has_sm120_arch([]) is False


def test_evaluate_ok_on_cpu_fallback():
    env = envcheck.evaluate_environment(
        version="2.12.1+cpu", cuda_available=False, device_requested="auto")
    assert env["ok"] is True and env["device"] == "cpu"


def test_evaluate_refuses_old_torch():
    env = envcheck.evaluate_environment(
        version="2.5.1+cu121", cuda_available=True, device_requested="cuda",
        capability=(12, 0), arch_list=["sm_120"])
    assert env["ok"] is False
    assert any("2.7" in r for r in env["reasons"])


def test_evaluate_refuses_cuda_unavailable():
    env = envcheck.evaluate_environment(
        version="2.12.1", cuda_available=False, device_requested="cuda")
    assert env["ok"] is False
    assert env["checks"]["device"] is False


def test_evaluate_refuses_blackwell_without_sm120():
    env = envcheck.evaluate_environment(
        version="2.12.1+cu128", cuda_available=True, device_requested="cuda",
        capability=(12, 0), arch_list=["sm_80", "sm_90"])
    assert env["ok"] is False
    assert env["checks"]["blackwell_detected"] is True
    assert env["checks"]["blackwell_sm120_arch"] is False


def test_evaluate_accepts_ada_sm89():
    # RTX 4090/4090 D (Ada, sm_89): only Blackwell needs the sm_120 kernels,
    # so an Ada card with torch>=2.7 cu128 is accepted without an sm_120 arch.
    env = envcheck.evaluate_environment(
        version="2.12.1+cu128", cuda_available=True, device_requested="cuda",
        capability=(8, 9), arch_list=["sm_86", "sm_89"])
    assert env["ok"] is True
    assert env["blackwell"] is False
    assert env["checks"]["blackwell_sm120_arch"] is True


def test_evaluate_accepts_blackwell_with_sm120():
    env = envcheck.evaluate_environment(
        version="2.12.1+cu128", cuda_available=True, device_requested="cuda",
        capability=(12, 0), arch_list=["sm_90", "sm_120"])
    assert env["ok"] is True and env["blackwell"] is True


def test_run_selfcheck_cpu_fallback_passes():
    report = envcheck.run_selfcheck("cpu")
    assert report["ok"] is True, report
    assert report["torch"]["version"] == envcheck._torch_info()["version"]
    # amendment 5: envcheck sizes for pack_v3 (K_max=192) by default
    assert report["format"] == "v3" and report["k_max"] == 192
    assert envcheck.run_selfcheck("cpu", format="v2")["k_max"] == 128
    fb = report["forward_backward"]
    assert fb is not None and fb["ok"] is True
    assert fb["device"] == "cpu" and fb["n_grad_tensors"] > 0


def test_main_returns_zero_on_cpu(monkeypatch, capsys):
    rc = envcheck.main(["--device", "cpu", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"ok": true' in out
