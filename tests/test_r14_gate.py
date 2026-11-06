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
        assert cfg.physics.emergency_onset == "driver_b"
        assert [s.name for s in cfg.data.c4_sources] == [s[0] for s in r14.SOURCES]
    assert cfgs["a"].model.init_from == r14.INIT_REL == "runs/c4_wm_v10_round13e_b2/wm_s.pt"
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


@pytest.fixture
def restore_13e_gate():
    """install() re-points round13e_gate globals; restore them for later 13E tests."""
    g = r14.g
    saved = {k: v for k, v in vars(g).items() if not k.startswith("__")}
    variants = {k: dict(v) for k, v in g.VARIANTS.items()}
    configs = dict(g.CONFIGS)
    yield g
    for k, v in saved.items():
        setattr(g, k, v)
    g.VARIANTS.clear()
    g.VARIANTS.update(variants)
    g.CONFIGS.clear()
    g.CONFIGS.update(configs)


def test_install_points_the_13e_verdicts_at_r14(restore_13e_gate):
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


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def test_real_stop_excludes_measurement_errors():
    assert r14.real_stop({"decision": "STOP", "reasons": ["one-step MAE delta 0.2 > 0.1"]})
    assert r14.real_stop({"decision": "STOP", "reasons": []})
    assert not r14.real_stop({"decision": "STOP", "reasons": [
        "measurement error (fail closed): RuntimeError: production exited/replaced without stage120"]})
    for decision in ("CONTINUE", "ERROR", None):
        assert not r14.real_stop({"decision": decision, "reasons": ["x"]})


class _Proc:
    def __init__(self, pid, run, ticks):
        self.pid, self.run, self.ticks, self.signalled = pid, run, ticks, 0

    def is_production(self):
        return True

    def terminate(self):
        self.signalled += 1
        return "SIGTERM sent to verified production pidfd"


class _R13C:
    made = []

    @classmethod
    def ProductionProcess(cls, pid, run, ticks):
        proc = _Proc(pid, run, ticks)
        cls.made.append(proc)
        return proc


@pytest.fixture
def json_io(monkeypatch):
    monkeypatch.setattr(r14, "read_json", lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(r14, "digest", lambda p: "sha:" + Path(p).name)

    def write_once(path, doc):
        with Path(path).open("x") as stream:
            json.dump(doc, stream)
    monkeypatch.setattr(r14, "write_once", write_once)


@pytest.mark.parametrize("decision,reasons,stops", [
    ("STOP", ["full-road hard-brake share ratio 2.4 > 2"], True),
    ("STOP", ["measurement error (fail closed): RuntimeError: missing archived file"], False),
    ("ERROR", ["measurement error (fail closed): x"], False),
    ("CONTINUE", [], False)])
def test_b2_watch_signals_owned_productions_only_on_a_real_stop(tmp_path, json_io, decision, reasons, stops):
    _R13C.made = []
    for stage, pid in (("a", 4242), ("b", 4343)):
        (tmp_path / f"production_r14_{stage}.pid").write_text(f"{pid}\n")
        (tmp_path / f"production_r14_{stage}.ticks").write_text("77\n")
    _write(tmp_path / r14.B2_POSTRUN, {"decision": decision, "reasons": reasons})
    assert r14.stop_watch(tmp_path, _R13C, "b2", poll=0, max_s=5, kill_window=0) == 0
    record = json.loads((tmp_path / "chk_r14/b2_watch.json").read_text())
    assert record["records"][0]["decision"] == decision and record["records"][0]["real_stop"] is stops
    assert (tmp_path / "chk_r14" / r14.B2_STOP_MARKER).exists() is stops
    assert r14.b2_stopped(tmp_path) is stops
    if stops:
        assert sorted((p.pid, p.ticks, Path(p.run).name, p.signalled) for p in _R13C.made) == [
            (4242, 77, "c4_wm_v10_round14_a", 1), (4343, 77, "c4_wm_v10_round14_b", 1)]
        with pytest.raises(ValueError, match="b2 postrun said STOP"):
            r14.launch_guard(tmp_path, "a")
    else:
        assert _R13C.made == [] and r14.launch_guard(tmp_path, "b")["ok"]


def test_a_watch_records_errors_then_stops_b_only_on_a_real_postrun_stop(tmp_path, json_io):
    _R13C.made = []
    for stage in ("a", "b"):
        (tmp_path / f"production_r14_{stage}.pid").write_text("5000\n")
        (tmp_path / f"production_r14_{stage}.ticks").write_text("9\n")
    _write(tmp_path / "chk_r14/a/midrun/RESULT.json",
           {"decision": "STOP", "reasons": ["measurement error (fail closed): RuntimeError: gone"]})
    _write(tmp_path / "chk_r14/a/postrun/RESULT.json",
           {"decision": "STOP", "reasons": ["one-step MAE delta 0.3 > 0.1"]})
    r14.stop_watch(tmp_path, _R13C, "a", poll=0, max_s=5, kill_window=0)
    record = json.loads((tmp_path / "chk_r14/a_watch.json").read_text())
    assert [r["real_stop"] for r in record["records"]] == [False, True]
    assert [Path(p.run).name for p in _R13C.made] == ["c4_wm_v10_round14_b"]   # never A
    assert r14.launch_guard(tmp_path, "a")["ok"]
    with pytest.raises(ValueError, match="A check said STOP"):
        r14.launch_guard(tmp_path, "b")


def test_watch_times_out_recording_only(tmp_path, json_io):
    _R13C.made = []
    assert r14.stop_watch(tmp_path, _R13C, "b2", poll=0, max_s=0, kill_window=0) == 0
    record = json.loads((tmp_path / "chk_r14/b2_watch.json").read_text())
    assert record["action"].startswith("none") and _R13C.made == []


def _b2_tree(root, *, recheck="CONTINUE", parent="sha:wm_s_stage120.pt", postrun=None):
    _write(root / r14.B_RECHECK, {"decision": recheck, "checkpoints": {"r10b": {"sha256": "sha:wm_s_stage120.pt"}}})
    (root / r14.B_RUN).mkdir(parents=True, exist_ok=True)
    (root / r14.B_RUN / "wm_s_stage120.pt").write_text("x")
    _write(root / r14.B2_RUN / "INIT_FROM.json", {"sha256": parent})
    _write(root / r14.B2_RUN / "RUN_SUMMARY.json", {})
    (root / r14.INIT_REL).write_text("x")
    if postrun:
        _write(root / r14.B2_POSTRUN, {"decision": postrun, "reasons": ["rule breach"]})


@pytest.mark.parametrize("change,match", [
    ({}, None), ({"recheck": "STOP"}, "recheck is not CONTINUE"), ({"parent": "other"}, "does not resume"),
    ({"postrun": "STOP"}, "b2 postrun said STOP"), ({"postrun": "ERROR"}, None),
    ({"postrun": "CONTINUE"}, None), ({"history": [300.0] * 59}, "60-step")])
def test_init_gate_is_b2_final_with_recheck_continue(tmp_path, json_io, monkeypatch, change, match):
    from types import SimpleNamespace
    from wms.cli.assemble import build_physics
    history = change.pop("history", r14.B2_HISTORY)
    _b2_tree(tmp_path, **change)
    physics = build_physics(r14.load_train_config(r14.REFERENCE["b"]))
    monkeypatch.setattr(r14, "workspace_path", lambda p, exists=True: Path(p))
    monkeypatch.setattr(r14.g, "strict_checkpoint", lambda p: (
        {"history": [{"stage_s": s} for s in history]}, SimpleNamespace(model=SimpleNamespace(physics=physics))))
    if match is None:
        assert r14.check_13e_b(tmp_path, tmp_path / r14.INIT_REL) == "sha:wm_s.pt"
    else:
        with pytest.raises(ValueError, match=match):
            r14.check_13e_b(tmp_path, tmp_path / r14.INIT_REL)


def test_production_b_needs_no_a_check_result(tmp_path, monkeypatch):
    """Timing request: B launch reads neither A's midrun nor its postrun RESULT (they run in
    parallel with B and are watched); only the A production verdict is required."""
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
    assert not any("postrun" in p or "midrun" in p for p in reads)
    assert ("a", {"production": True}) in calls
    (root / "chk_r14/A_CHECK_STOP.json").write_text("{}")
    with pytest.raises(ValueError, match="A check said STOP"):
        r14.production_ready(root, root / "ck.pt", root / "off.json", "b")


def test_shell_syntax_and_lesson_wiring():
    text = (ROOT / "scripts/round14_gate.sh").read_text(encoding="utf-8")
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")   # WSL's bash.exe may shadow it on Windows
    bash = os.environ.get("R14_BASH") or (str(git_bash) if os.name == "nt" and git_bash.is_file()
                                          else shutil.which("bash"))
    subprocess.run([bash, "-n", str(ROOT / "scripts/round14_gate.sh")], check=True)
    assert "nvidia-smi --query-compute-apps=pid" in text and "gpu_guard" in text
    assert "postrun/RESULT.json" not in text and "midrun/RESULT.json" not in text  # never waits on A checks
    assert "WMS_DATA_LOWPEN_NATIVE" in text and "runs/c4_wm_v10_round13e_b2/wm_s.pt" in text
    assert "smoke b \"$B\" \"$INIT\" b_final" not in text           # no B re-smoke (timing request)
    assert text.index("production_${TAG}_$STAGE.ticks") < text.index("\"${GATE[@]}\" launch-guard") < text.rindex("exec ")
    assert "watch --root \"$L\" --name a" in text
    chain = (ROOT / "scripts/round14b_autochain.sh").read_text(encoding="utf-8")
    subprocess.run([bash, "-n", str(ROOT / "scripts/round14b_autochain.sh")], check=True)
    assert "--name b2" in chain and "register" in chain and "--smoke" in chain and "--chain" in chain
    assert "chk_r13e/b2/postrun" not in chain                     # never waits for the b2 postrun
    for forbidden in ("pkill", "killall", "pgrep", "kill ", "taskkill"):
        assert forbidden not in chain
    for forbidden in ("pkill", "killall", "pgrep -f", "kill -9", "kill -TERM"):
        assert forbidden not in text


def test_offline_tree_comparison_tolerates_ulps_only():
    a = {"x": [1.0, 2.0, None], "n": 3, "s": "k"}
    assert r14.same_tree(a, {"x": [1.0 + 1e-13, 2.0, None], "n": 3, "s": "k"})
    assert not r14.same_tree(a, {"x": [1.0 + 1e-6, 2.0, None], "n": 3, "s": "k"})
    assert not r14.same_tree(a, {"x": [1.0, 2.0], "n": 3, "s": "k"})
    assert not r14.same_tree(a, {"x": [1.0, 2.0, None], "n": 4, "s": "k"})
