"""Focused R13E plan/gate tests: synthetic CPU artifacts, never a training launch."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wms.cli.assemble import build_model, build_physics
from wms.config.load import load_train_config
from wms.model.checkpoint import save_checkpoint, sha256_file
from wms.train import round13e_gate as gate
from wms.train.settings import TrainSettings
from wms.train.trainer import WmSTrainer, build_optimizer
from tests.test_round12_gate_verdict import make_rows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("plan tests must never launch a trainer or initialize CUDA")
    monkeypatch.setattr(WmSTrainer, "__init__", forbidden)
    monkeypatch.setattr(WmSTrainer, "fit", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_resolved_recipes_depth_and_actual_optimizer_sets():
    configs = gate.check_configs()  # Both complete extends chains must fit cap 16.
    a, b = configs["a"], configs["b"]
    assert a.curriculum.truncation_steps_per_stage == (120,)
    assert a.optim.batch_size_per_stage == (4,)
    assert a.selection.val_max_windows == 4
    assert a.model.init_from.endswith("c4_wm_v10_round12c/wm_s.pt")
    assert b.model.init_from == a.runtime.out_dir + "/wm_s.pt"
    assert b.curriculum.truncation_steps_per_stage == (120, 60)
    assert b.optim.batch_size_per_stage == (4, 2)
    assert b.model.temporal_accel_off and not b.model.residual_accel_off
    assert not b.model.follower_courtesy_off and b.model.follower_courtesy_init_zero
    assert b.model.follower_courtesy_state_grad and b.model.follower_courtesy_intent == "pivot"
    assert b.loss.follower_courtesy.absolute_weight == 4.525
    for stage, cfg in configs.items():
        assert cfg.physics.emergency_braking and cfg.physics.secure_gap_veto
        assert cfg.runtime.out_dir == f"runs/c4_wm_v10_round13e_{stage}"
        model = build_model(cfg, build_physics(cfg))
        _, report = build_optimizer(model, lr=cfg.optim.lr,
                                    weight_decay=cfg.optim.weight_decay,
                                    freeze_prefixes=cfg.optim.freeze_prefixes, lr_scale=cfg.optim.lr_scale)
        text = "WM-S optimizer groups: " + json.dumps(report)
        names = gate.check_optimizer(text, model, cfg, stage)
        assert names == sorted(n for n, p in model.named_parameters() if p.requires_grad)
        report["trainable_tensor_names"].append("merge_pivot.mu")
        with pytest.raises(ValueError, match="exact configured set"):
            gate.check_optimizer("WM-S optimizer groups: " + json.dumps(report), model, cfg, stage)


def test_stage_a_does_not_build_prior_or_courtesy_loss():
    from wms.train.losses.compose import build_loss
    cfg = gate.check_configs()["a"]
    model = build_model(cfg, build_physics(cfg))
    # Loss composition is independent of the cross-owner TrainSettings wiring
    # for the two new plant-only flags; preflight exercises that wiring strictly.
    flat = cfg.to_legacy_flat()
    fields = TrainSettings.__dataclass_fields__
    for key in ("emergency_braking", "secure_gap_veto"):
        if key not in fields:
            flat.pop(key)
    loss = build_loss(TrainSettings.from_flat(flat), physics=model.physics)
    assert not any("Courtesy" in type(term).__name__ or "Prior" in type(term).__name__
                   for term in (*loss.chunk_terms, *loss.unroll_terms))


def a_rows(count=2):
    base = make_rows()[0]
    return [{**{k: v for k, v in base.items()
                if not k.startswith(("follower_courtesy", "driver_prior"))}, "step": i + 1}
            for i in range(count)]


def b_rows(production=False):
    base = make_rows()
    stages = [120] * 40 + [300] * 60 if production else [120, 120, 300, 300]
    return [{**{k: v for k, v in base[0 if s == 120 else 2].items()
                if not k.startswith("driver_prior")}, "step": i, "stage_s": s}
            for i, s in enumerate(stages, 1)]


@pytest.mark.parametrize("stage", ["a", "b"])
def test_log_requires_nonempty_complete_finite_evidence(stage):
    rows = a_rows() if stage == "a" else b_rows()
    assert gate.check_log(rows, "ok", stage)["ok"]
    for damaged in ([], rows[:-1], [{**rows[0], "loss": None}, *rows[1:]],
                    [{**rows[0], "loss": float("nan")}, *rows[1:]],
                    [{k: v for k, v in rows[0].items() if k != "merge_pivot_tau"}, *rows[1:]],
                    [{**rows[0], "driver_prior": 0}, *rows[1:]],
                    [{**rows[0], "merge_pivot_tf_zero_mass_events": 1}, *rows[1:]]):
        assert not gate.check_log(damaged, "ok", stage)["ok"]
    assert not gate.check_log(rows, "Traceback", stage)["ok"]
    assert not gate.check_log(rows, "ok", stage, train_exit_code=1)["ok"]
    if stage == "a":
        assert not gate.check_log([{**rows[0], "follower_courtesy": 0}, rows[1]], "ok", stage)["ok"]
        assert gate.check_log(a_rows(30), "ok", stage, production=True)["ok"]
    else:
        assert gate.check_log(b_rows(True), "ok", stage, production=True)["ok"]
        for row in rows:
            row["follower_courtesy_contrast_n_controls"] = 0
        assert not gate.check_log(rows, "ok", stage)["ok"]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Actual initializer + optimizer + checkpoint loader; no fit/backward/step."""
    assert tmp_path.resolve().is_relative_to(ROOT), "run pytest with --basetemp inside workspace"
    # Artifact tests isolate identity/freeze checks; the mandatory emission probe
    # is tested independently below, including fail-closed pre-13C behavior.
    monkeypatch.setattr(gate, "check_emissions", lambda model, stage: None)

    def make(stage="a", production=False):
        cfg = gate.check_configs()[stage]
        source = tmp_path / "wm_s.pt"
        directory = tmp_path / "smoke"
        overrides = [f"model.init_from={source}", f"runtime.out_dir={directory}",
                     "runtime.device=cuda", "optim.batch_size=4",
                     "loss.loss_share_logging=false", "loss.brake_follow_cav_weight=2.0"]
        if not production:
            overrides += ["curriculum.stage_steps=[2]" if stage == "a" else
                          "curriculum.stage_steps=[2,2]", "selection.val_max_windows=4",
                          "curriculum.max_hours=1.5"]
        if stage == "b" and not production:
            overrides += ["optim.batch_size_per_stage=[4,4]"]
        effective = load_train_config(gate.CONFIGS[stage], overrides)
        physics = build_physics(effective)
        model = build_model(effective, physics)
        before = {n: v.clone() for n, v in model.state_dict().items()}
        # Nonzero AS-IS frozen read-outs catch the unsafe old 'reset row 0' verdict.
        before["residual_net.accel.2.weight"][0].fill_(.125)
        before["follower_courtesy.net.4.weight"].fill_(.25)
        model.load_state_dict(before, strict=True)
        source_flat = cfg.to_legacy_flat()
        source_flat["physics"] = asdict(build_physics(load_train_config(
            ROOT / "configs/c4_wm_v10round12c.toml")))
        save_checkpoint(source, model, source_flat, history=a_rows(30))
        model = build_model(effective, physics)  # Trainer creates a fresh target before init.
        report = WmSTrainer._init_from_checkpoint(SimpleNamespace(
            model=model, settings=SimpleNamespace(**effective.to_legacy_flat()), out_dir=directory), str(source))
        if stage == "a":
            assert report["follower_courtesy_init"] == "checkpoint"
            assert all(gate.same_bytes(v, model.state_dict()[n]) for n, v in before.items())
        else:
            assert report["follower_courtesy_init"] == "zero"
            assert torch.count_nonzero(model.state_dict()["follower_courtesy.net.4.weight"]) == 0
            assert all(gate.same_bytes(v, model.state_dict()[n]) for n, v in before.items()
                       if not n.startswith("follower_courtesy."))
        _, optimizer = build_optimizer(model, lr=cfg.optim.lr,
                                       weight_decay=cfg.optim.weight_decay,
                                       freeze_prefixes=cfg.optim.freeze_prefixes, lr_scale=cfg.optim.lr_scale)
        text = "WM-S optimizer groups: " + json.dumps(optimizer)
        rows = a_rows(30 if production else 2) if stage == "a" else b_rows(production)
        weights = model.state_dict()
        with torch.no_grad():
            if stage == "a":
                weights["driver_encoder.head.2.bias"].add_(.01)
                weights["driver_encoder.centre_logits"].add_(.01)
            else:
                weights["residual_net.accel.2.bias"].add_(.01)
                weights["follower_courtesy.net.4.weight"].add_(.01)
        flat = effective.to_legacy_flat()
        flat["physics"] = asdict(physics)
        for name, horizon in (("wm_s_stage120.pt", 120), ("wm_s_stage300.pt", 300), ("wm_s.pt", None)):
            if stage == "a" and horizon == 300:
                continue
            metadata = {"history": rows} if horizon is None else {
                "stage_s": horizon, "step": max(r["step"] for r in rows if r["stage_s"] == horizon)}
            save_checkpoint(directory / name, model, flat, extra=metadata)
        return SimpleNamespace(cfg=cfg, directory=directory, source=source, rows=rows, text=text,
                               stage=stage, production=production)
    return make


def validate(s):
    return gate.check_checkpoints(s.directory, s.rows, s.text, s.source, s.cfg, s.stage,
                                  production=s.production)


@pytest.mark.parametrize("stage", ["a", "b"])
def test_strict_checkpoint_verdict_accepts_actual_as_is_initializer(synthetic, stage):
    s = synthetic(stage)
    reports = validate(s)
    assert len(reports) == (2 if stage == "a" else 3)
    assert all(r["frozen_bitwise"] and r["emergency_braking"] for r in reports)


@pytest.mark.parametrize("damage", ["frozen", "reset_accel", "missing", "dtype", "physics", "emission",
                                    "prior", "step", "history", "schedule", "no_movement",
                                    "secure_gap", "output", "loss", "lr", "truncation"])
def test_a_artifact_damage_is_fail_closed(synthetic, damage):
    s = synthetic()
    target = s.directory / ("wm_s.pt" if damage == "history" else "wm_s_stage120.pt")
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if damage == "frozen":
        payload["model"]["gate_g_max_logit"].add_(.1)
    elif damage == "reset_accel":
        payload["model"]["residual_net.accel.2.weight"][0].zero_()
    elif damage == "missing":
        del payload["model"]["follower_courtesy.net.0.weight"]
    elif damage == "dtype":
        key = "residual_net.accel.2.weight"
        payload["model"][key] = payload["model"][key].double()
    elif damage == "physics":
        payload["config"]["physics"]["emergency_braking"] = False
    elif damage == "secure_gap":
        payload["config"]["physics"]["secure_gap_veto"] = False
    elif damage == "output":
        payload["config"]["out_dir"] = str(s.directory / "other")
    elif damage == "loss":
        payload["config"]["brake_follow_cav_weight"] = 0
    elif damage == "lr":
        payload["config"]["lr_per_stage"] = [1.0]
    elif damage == "truncation":
        payload["config"]["truncation_steps_per_stage"] = [1]
    elif damage == "emission":
        payload["config"]["residual_accel_off"] = False
    elif damage == "prior":
        payload["config"]["driver_prior_weight"] = .01
    elif damage == "step":
        payload["step"] = 1
    elif damage == "history":
        payload["history"] = []
    elif damage == "schedule":
        payload["config"]["stage_steps"] = [30]
    elif damage == "no_movement":
        target = s.directory / "wm_s.pt"
        payload = torch.load(target, map_location="cpu", weights_only=True)
        before = torch.load(s.source, map_location="cpu", weights_only=True)["model"]
        payload["model"] = before
    torch.save(payload, target)
    with pytest.raises((ValueError, RuntimeError)):
        validate(s)


@pytest.mark.parametrize("stage", ["a", "b"])
def test_init_transform_report_is_rejected(synthetic, stage):
    s = synthetic(stage)
    path = s.directory / "INIT_FROM.json"
    report = json.loads(path.read_text())
    report["residual_accel_reinit"] = {"zeroed_rows": [0]}
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="init transform"):
        validate(s)


@pytest.mark.parametrize("damage", ["core", "temporal", "intent", "absolute"])
def test_b_keeps_exact13c_frozen_and_repair_a_rules(synthetic, damage):
    s = synthetic("b")
    target = s.directory / "wm_s_stage300.pt"
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if damage == "core":
        payload["model"]["driver_encoder.centre_logits"].add_(.01)
    elif damage == "temporal":
        payload["config"]["temporal_accel_off"] = False
    elif damage == "intent":
        payload["config"]["follower_courtesy_intent"] = "none"
    else:
        payload["config"]["follower_courtesy_absolute_weight"] = 0
    torch.save(payload, target)
    with pytest.raises(ValueError):
        validate(s)


def test_offline_report_must_be_complete_boolean_strict_and_bind_exact_init(synthetic, tmp_path, monkeypatch):
    s = synthetic()
    path = tmp_path / "offline.json"
    good = {"status": "complete", "strict_exact_weights": True, "checkpoint": str(s.source),
            "checkpoint_sha256": sha256_file(s.source)}
    for doc in ({}, {**good, "status": "pending"}, {**good, "strict_exact_weights": 1},
                {**good, "strict_exact_weights": False}, {**good, "checkpoint_sha256": "wrong"}):
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError):
            gate.check_offline(path, s.source)
    path.write_text(json.dumps(good), encoding="utf-8")
    assert gate.check_offline(path, s.source) == good
    with pytest.raises(ValueError, match="registered S-R12C FINAL SHA256"):
        gate.preflight(tmp_path, s.source, path)
    monkeypatch.setattr(gate, "R12C_SHA256", sha256_file(s.source))
    monkeypatch.setattr(TrainSettings, "from_config", lambda cfg: None)
    assert gate.preflight(tmp_path, s.source, path)["ok"]
    (tmp_path / ".round13e_gate.lock").mkdir()
    with pytest.raises(ValueError, match="prior R13E artifact"):
        gate.preflight(tmp_path, s.source, path)


def test_workspace_paths_and_signed_zero_bytes():
    with pytest.raises(ValueError, match="inside workspace"):
        gate.workspace_path(gate.operational_root().parent / "outside.json", exists=False)
    for token in gate.FORBIDDEN:
        with pytest.raises(ValueError, match="restricted"):
            gate.workspace_path(ROOT / token.upper() / "missing.json")
    assert not gate.same_bytes(torch.tensor([0.]), torch.tensor([-0.]))


def test_shell_default_and_launch_order_are_fail_closed_without_execution():
    # Static check: even launcher verification must never execute a production command.
    script = (ROOT / "scripts/round13e_gate.sh").read_text(encoding="utf-8")
    default = script.index('if [ "$MODE" = config ]')
    assert default < script.index('"${GATE[@]}" registered') < script.index("-m pytest")
    assert script.index("-m pytest") < script.index('smoke a "$A" "$CK" a')
    assert script.index('smoke a "$A" "$CK" a') < script.index('smoke b "$B" "$L/runs/smoke_r13e_a/wm_s.pt" b')
    assert script.index('"${GATE[@]}" smoke-complete') < script.index('"${GATE[@]}" production-ready')
    assert script.index('smoke b "$B" "$INIT" b_final') < script.index('exec "$PYTHON"')
    assert 'requires --offline-report' in script
    assert '--launch-a' in script and '--launch-b' in script and '--launch)' not in script
    assert 'for CHECK in midrun postrun' in script
    assert script.index('test -f "$READY"') < script.index('exec "$PYTHON"')
    assert 'kill -0 "$WAITER"' in script
    assert "pkill" not in script and "killall" not in script and "nvidia-smi" not in script


@pytest.mark.parametrize("stage", ["a", "b"])
def test_mandatory_temporal_diagnostic_fails_closed_until_13c_merge(stage):
    cfg = gate.check_configs()[stage]
    model = build_model(cfg, build_physics(cfg))
    # Actual CPU synthetic forward: current sources must not pass by omitting the
    # 13C diagnostic. The same test passes normally once 13C supplies it.
    if "diag_emit_temporal" in (ROOT / "src/wms/model/wms.py").read_text(encoding="utf-8"):
        gate.check_emissions(model, stage)
    else:
        with pytest.raises(ValueError, match="mandatory 13C temporal"):
            gate.check_emissions(model, stage)


@pytest.mark.parametrize("stage", ["a", "b"])
def test_production_checkpoint_schedule_identity(synthetic, stage):
    s = synthetic(stage, production=True)
    assert validate(s)


def test_prepare_and_source_validation_are_static_no_launch(monkeypatch):
    import inspect
    source = inspect.getsource(gate.prepare)
    assert '"git", "archive", "--format=tar", sha' in source
    assert '"git", "cat-file", "-e"' in source
    assert "copytree" not in source and "wms train" not in source
    monkeypatch.setattr(gate, "operational_root", lambda: ROOT)
    with pytest.raises(ValueError, match="prepared repo"):
        gate.check_source()
    assert gate.STOP_RULE["full_road_lane1_hdv_a_lt_minus3_share_ratio_gt"] == 2
    assert gate.STOP_RULE["all_lane1_onestep_mae_delta_gt_mps2"] == .1
    assert gate.STOP_RULE["paired_same_panel"] is True
    waiter = inspect.getsource(gate.make_waiter)
    assert "r13c.ProductionProcess(args.pid, self.run, args.start_ticks)" in waiter
    assert "super().measure(checkpoint)" in waiter
    assert "physics.{flag}=false" in waiter


@pytest.fixture
def r13c_waiter():
    """Inspect reference Git objects in memory; never read another worktree."""
    import subprocess
    import types
    local = ROOT / "scripts/round13c_waiter.py"
    if local.is_file():
        source = local.read_text(encoding="utf-8")
    else:
        source = subprocess.run(["git", "show", "c4/v10r13c:scripts/round13c_waiter.py"],
                                cwd=ROOT, capture_output=True, text=True, check=True).stdout
    module = types.ModuleType("r13c_waiter_reference")
    exec(compile(source, "<13C Git waiter reference>", "exec"), module.__dict__)
    return module


def test_registered_13c_stop_boundaries(r13c_waiter):
    stop = r13c_waiter.stop_reasons
    assert stop(.02, .01, .5, .4) == []
    assert len(stop(.02001, .01, .50001, .4)) == 2
    assert stop(0, 0, 0, 0) == []
    assert len(stop(.0001, 0, 0, 0)) == 1
    with pytest.raises(ValueError):
        stop(float("nan"), .01, .4, .4)


@pytest.mark.parametrize("stage", ["a", "b"])
def test_waiter_adapter_uses_owned_identity_and_preserves_control_physics(
        r13c_waiter, monkeypatch, tmp_path, stage):
    identities, commands = [], []
    process_type = r13c_waiter.ProductionProcess
    monkeypatch.setattr(r13c_waiter, "ProductionProcess",
                        lambda *args: (identities.append(args) or SimpleNamespace()))
    monkeypatch.setattr(r13c_waiter.Waiter, "command",
                        lambda self, script, *args: commands.append((script, args)))
    waiter = gate.make_waiter(r13c_waiter, SimpleNamespace(
        root=tmp_path, stage=stage, mode="midrun", pid=12345, start_ticks=9876))
    assert identities == [(12345, tmp_path / f"runs/c4_wm_v10_round13e_{stage}", 9876)]
    assert waiter.out == tmp_path / f"chk_r13e/{stage}/midrun"
    assert waiter.work.is_relative_to(tmp_path / "chk_r13e")
    waiter.command(ROOT / "tools/r10b_checks/run_onestep.py",
                   "--config", "c4_wm_v10round13c.toml", "--var", "r10b=candidate")
    args = commands[0][1]
    assert args[:2] == ("--config", gate.CONFIGS[stage].name)
    for arm in ("r12c", "r9"):
        for flag in ("emergency_braking", "secure_gap_veto"):
            assert f"{arm}=physics.{flag}=false" in args
    # The implementation inherited by R13E must not signal an unrelated process.
    monkeypatch.setattr(r13c_waiter, "proc_stat", lambda pid: (9999, "S"))
    monkeypatch.setattr(r13c_waiter, "pidfd_open", lambda pid: pytest.fail("unowned PID"))
    assert process_type(12345, waiter.run, 9876).terminate().startswith("not sent")


def test_production_requires_both_smokes_and_current_a_final_checks(monkeypatch, tmp_path):
    registration = {"registration": "test"}
    (tmp_path / "registration_r13e.json").write_text(json.dumps(registration), encoding="utf-8")
    calls = []
    monkeypatch.setattr(gate, "check_registration", lambda *args: registration)
    monkeypatch.setattr(gate, "artifact_verdict", lambda root, stage, ck, **kw:
                        calls.append((stage, ck, kw.get("production", False))))
    for name in ("smoke_r13e_a", "smoke_r13e_b", "c4_wm_v10_round13e_a"):
        directory = tmp_path / "runs" / name
        directory.mkdir(parents=True)
        (directory / "wm_s.pt").write_bytes(name.encode())
        (directory / "wm_s_stage120.pt").write_bytes((name + "stage120").encode())
    checkpoint = tmp_path / "runs/c4_wm_v10_round13e_a/wm_s.pt"
    receipt = {"registration": registration,
               "a_sha256": gate.digest(tmp_path / "runs/smoke_r13e_a/wm_s.pt"),
               "b_sha256": gate.digest(tmp_path / "runs/smoke_r13e_b/wm_s.pt")}
    path = tmp_path / "smoke_r13e.complete.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert gate.production_ready(tmp_path, checkpoint, path, "a")["ok"]
    assert [c[0] for c in calls] == ["a", "b"]
    assert calls[1][1] == tmp_path / "runs/smoke_r13e_a/wm_s.pt"
    receipt["b_sha256"] = "wrong"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke receipt identity"):
        gate.production_ready(tmp_path, checkpoint, path, "a")
    receipt["b_sha256"] = gate.digest(tmp_path / "runs/smoke_r13e_b/wm_s.pt")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    # Missing A check evidence is fatal before B production readiness.
    with pytest.raises(FileNotFoundError):
        gate.production_ready(tmp_path, checkpoint, path, "b")
    for mode in ("midrun", "postrun"):
        directory = tmp_path / "chk_r13e/a" / mode
        directory.mkdir(parents=True)
        selected = checkpoint.with_name("wm_s_stage120.pt") if mode == "midrun" else checkpoint
        (directory / "RESULT.json").write_text(json.dumps({
            "decision": "CONTINUE", "registration_sha256": gate.digest(tmp_path / "registration_r13e.json"),
            "checkpoints": {"r10b": {"sha256": gate.digest(selected)}}}))
    assert gate.production_ready(tmp_path, checkpoint, path, "b")["init_checkpoint"] == str(checkpoint)
    (tmp_path / "chk_r13e/a/midrun/RESULT.json").write_text('{"decision":"STOP"}')
    with pytest.raises(ValueError, match="A midrun check"):
        gate.production_ready(tmp_path, checkpoint, path, "b")
