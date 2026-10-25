"""R14 gate contracts (CPU only; no data, no checkpoints, no training)."""
from dataclasses import replace
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

from wms.train import round14_gate as r14


ROOT = Path(__file__).resolve().parents[1]


def test_configs_are_13e_plus_exactly_the_registered_r14_changes():
    cfgs = r14.check_configs()
    for stage, cfg in cfgs.items():
        assert cfg.physics.lane_change_secure_gap and cfg.data.replay.low_share_oversample == 1.69
        assert [s.name for s in cfg.data.c4_sources] == [s[0] for s in r14.SOURCES]
    assert cfgs["a"].model.init_from == r14.INIT_REL
    assert cfgs["b"].model.init_from == f"runs/{r14.RUN}_a/wm_s.pt"


def test_config_check_rejects_any_other_difference(monkeypatch):
    real = r14.load_train_config

    def tampered(path, overrides=None):
        cfg = real(path, overrides) if overrides is not None else real(path)
        if Path(path).name == "c4_wm_v10round14.toml":
            cfg = replace(cfg, data=replace(cfg.data, replay=replace(cfg.data.replay, low_share_oversample=2.0)))
        return cfg
    monkeypatch.setattr(r14, "load_train_config", tampered)
    with pytest.raises(ValueError, match="differs"):
        r14.check_configs()


def test_install_points_the_13e_verdicts_at_r14():
    g = r14.install()
    assert (g.TAG, g.RUN) == ("r14", "c4_wm_v10_round14")
    assert g.CONFIGS == r14.CONFIGS and g.FLAG_KEYS == r14.FLAG_KEYS
    assert g.check_init is r14.check_init and g.check_configs is r14.check_configs
    assert set(r14.R14_SOURCE) <= set(g.REQUIRED_SOURCE)
    assert all((ROOT / name).is_file() for name in r14.REQUIRED_SOURCE), "launch dependency missing"


def _report(before, stage, **over):
    courtesy = sorted(n for n in before if n.startswith("follower_courtesy.")) if stage == "b" else []
    doc = {"sha256": "x", "loaded": sorted(set(before) - set(courtesy)), "new_in_model": courtesy,
           "expanded": [], "shape_mismatch": [], "unused_from_checkpoint": [],
           "follower_courtesy_init": "zero" if stage == "b" else "checkpoint"}
    doc.update(over)
    return doc


@pytest.mark.parametrize("stage", ["a", "b"])
def test_check_init_as_is_13e_b_and_courtesy_zero_in_b(monkeypatch, stage):
    import wms.model.checkpoint as ck
    monkeypatch.setattr(ck, "sha256_file", lambda p: "x")
    before = {"merge_yield.log_decel": 0, "merge_yield.log_distance": 0, "follower_courtesy.w": 0,
              "residual_net.accel.w": 0}
    r14.check_init(_report(before, stage), before, "ck", stage)
    with pytest.raises(ValueError):
        r14.check_init(_report(before, stage, follower_courtesy_init="zero" if stage == "a" else "checkpoint"),
                       before, "ck", stage)
    with pytest.raises(ValueError, match="yield"):
        no_yield = {k: v for k, v in before.items() if not k.startswith("merge_yield")}
        r14.check_init(_report(no_yield, stage), no_yield, "ck", stage)


def test_b_midrun_requires_a_postrun_continue_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(r14, "A_POSTRUN_WAIT_S", 0)
    monkeypatch.setattr(r14, "read_json", lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(r14, "digest", lambda p: "sha-a-final")
    with pytest.raises(ValueError, match="never appeared"):
        r14.a_postrun_continue(tmp_path)
    out = tmp_path / "chk_r14/a/postrun"
    out.mkdir(parents=True)
    (out / "RESULT.json").write_text(json.dumps({"decision": "STOP", "checkpoints": {}}))
    with pytest.raises(ValueError, match="stop B"):
        r14.a_postrun_continue(tmp_path)
    (out / "RESULT.json").write_text(json.dumps(
        {"decision": "CONTINUE", "checkpoints": {"r10b": {"sha256": "sha-a-final"}}}))
    assert r14.a_postrun_continue(tmp_path)["decision"] == "CONTINUE"


def test_production_b_needs_a_midrun_only(tmp_path, monkeypatch):
    """The lesson: B launch never reads/waits for A's postrun RESULT."""
    root = tmp_path
    (root / "chk_r14/a/midrun").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(r14, "check_registration", lambda *a: {"reg": 1})
    monkeypatch.setattr(r14, "workspace_path", lambda p, exists=True: Path(p))
    monkeypatch.setattr(r14.g, "artifact_verdict", lambda *a, **k: calls.append((a[1], k)))
    monkeypatch.setattr(r14, "digest", lambda p: "d:" + Path(p).name)
    reads = []

    def read(p):
        reads.append(Path(p).as_posix())
        if Path(p).name == "smoke_r14.complete.json":
            return {"registration": {"reg": 1}, "a_sha256": "d:wm_s.pt", "b_sha256": "d:wm_s.pt"}
        return {"decision": "CONTINUE", "registration_sha256": "d:registration_r14.json",
                "checkpoints": {"r10b": {"sha256": "d:wm_s_stage120.pt"}}}
    monkeypatch.setattr(r14, "read_json", read)
    assert r14.production_ready(root, root / "ck.pt", root / "off.json", "b")["ok"]
    assert not any("postrun" in p for p in reads)
    assert any(p.endswith("chk_r14/a/midrun/RESULT.json") for p in reads)


def test_shell_syntax_and_lesson_wiring():
    text = (ROOT / "scripts/round14_gate.sh").read_text(encoding="utf-8")
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")   # WSL's bash.exe may shadow it on Windows
    bash = os.environ.get("R14_BASH") or (str(git_bash) if os.name == "nt" and git_bash.is_file()
                                          else shutil.which("bash"))
    subprocess.run([bash, "-n", str(ROOT / "scripts/round14_gate.sh")], check=True)
    assert "nvidia-smi --query-compute-apps=pid" in text and "gpu_guard" in text
    assert "postrun/RESULT.json" not in text          # never waits on A postrun
    assert "WMS_DATA_LOWPEN_NATIVE" in text and "runs/c4_wm_v10_round13e_b/wm_s.pt" in text
    for forbidden in ("pkill", "killall", "pgrep -f", "kill -9", "kill -TERM"):
        assert forbidden not in text
