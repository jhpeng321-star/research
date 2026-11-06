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
    gate.set_variant("main")
    yield
    gate.set_variant("main")
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
    assert a.optim.lr_scale["merge_yield"] == 1000.0 and "merge_yield" in b.optim.freeze_prefixes
    for stage, cfg in configs.items():
        assert cfg.physics.emergency_braking and cfg.physics.secure_gap_veto
        assert cfg.model.mainline_merge_yield_learnable
        assert cfg.runtime.out_dir == f"runs/c4_wm_v10_round13e_{stage}"
        model = build_model(cfg, build_physics(cfg))
        gate.check_yield_init(model)
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
        if stage == "a":
            # The real A init (S-R12C) predates R13Y: no yield tensors or flag.
            payload = torch.load(source, map_location="cpu", weights_only=True)
            payload["model"] = {k: v for k, v in payload["model"].items()
                                if not k.startswith("merge_yield.")}
            payload["config"].pop("mainline_merge_yield_learnable", None)
            torch.save(payload, source)
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


@pytest.fixture
def offline_evidence(synthetic, tmp_path, monkeypatch):
    """Producer statistics over synthetic evidence, never an operational report."""
    import copy
    import numpy as np
    from tools.r13e_checks import offline as o
    from tools.r10b_checks import run_onestep as producer
    s = synthetic()
    monkeypatch.setattr(gate, "R12C_SHA256", sha256_file(s.source))
    payload, loaded = gate.strict_checkpoint(s.source)
    path = tmp_path / "offline.json"
    cols, packed = producer.COLS + producer.BASE_COLS, {}
    for arm, offset in zip(o.ARMS, (0.2, 0.1, 0.05)):
        data = []
        for window in range(2):
            for rec, speed, gap in ((-2., 4., 10.), (0., 10., 80.), (.5, 1., 10.)):
                for slot in range(5):
                    row = dict.fromkeys(cols, 0.)
                    row.update(win=window, lane=1, slot=slot, rec=rec, fin=rec + offset,
                               v=speed, gapF=gap)
                    data.append([row[c] for c in cols])
        packed[arm] = np.array(data)
        np.savez_compressed(tmp_path / f"onestep_{arm}.npz", rows=packed[arm], cols=np.array(cols))
    pos = np.arange(241)[:, None] * 2.5
    values = o.json_safe(o.rollout_metrics(pos, np.full_like(pos, 5.), np.ones_like(pos),
                                         np.ones_like(pos, dtype=bool), np.array([False]), np.array([5.])))
    rows = [{"state": state, "replicate": rep, "seed": o.RNG_SEED + 2 * state + rep,
             "world": f"train-{state}", "key": f"snapshot-{state}", "t0_s": 0., "vehicles": 1,
             **{arm: copy.deepcopy(values) for arm in o.ARMS}}
            for state in range(48) for rep in range(2)]
    summary, estimates = {}, {}
    for arm in (*o.ARMS, *o.CONTRASTS):
        summary[arm] = {}
        for metric, value in values.items():
            value = None if value is None else (0. if arm in o.CONTRASTS else float(value))
            if value not in estimates:
                estimates[value] = o.json_safe(o._estimate(
                    [np.nan if value is None else value] * 96, np.repeat(np.arange(48), 2)))
            summary[arm][metric] = estimates[value]
    totals = {}
    for arm in o.ARMS:
        total = {key: 96 * values[key] for key in
                 ("hdv_rows", "finite_hdv_transitions", "a_lt_minus3_rows", "ttc_eligible_rows",
                  "ttc_lt_1_5_rows", "overlap_pair_steps", "invalid_hdv_rows", "bad_acceleration_rows")}
        total.update(states=48, runs=96, blowup_runs=0, blowup_states=0,
                     a_lt_minus3_share=0., ttc_lt_1_5_share=0.)
        totals[arm] = total
    good = {"status": "complete", "strict_exact_weights": True, "checkpoint": str(s.source),
            "checkpoint_sha256": sha256_file(s.source), "state_dict_sha256": o.state_digest(payload["model"]),
            "device": "cpu", "workers": 1, "threads": 1, "linux_nice": 19,
            "states": 48, "seeds_per_state": 2, "horizon_s": 120., "dt_s": .5,
            "arms": o.ARMS, "definitions": o.DEFINITIONS, "emissions": dict.fromkeys(gate.EMISSIONS, True),
            "override_only": ["emergency_braking", "secure_gap_veto"],
            "config": str(ROOT / "configs/c4_wm_v10round12c.toml"),
            "train_roots": [str(tmp_path / "train")], "train_manifest_sha256": ["1" * 64],
            "states_sha256": "2" * 64, "base_physics": asdict(loaded.model.physics),
            "one_step_windows": ["window-0", "window-1"],
            "one_step": o.json_safe(o.analyze_onestep(packed, cols)),
            "rollouts": {"rows": rows, "totals": totals, "state_cluster_run_mean": summary}}
    path.write_text(json.dumps(good), encoding="utf-8")
    return s, path, good


def test_offline_report_must_be_complete_boolean_strict_and_bind_exact_init(offline_evidence, monkeypatch, tmp_path):
    s, path, good = offline_evidence
    for doc in ({}, {**good, "status": "pending"}, {**good, "strict_exact_weights": 1},
                {**good, "strict_exact_weights": False}, {**good, "checkpoint_sha256": "wrong"},
                {k: good[k] for k in ("status", "strict_exact_weights", "checkpoint", "checkpoint_sha256")}):
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError):
            gate.check_offline(path, s.source)
    path.write_text(json.dumps(good), encoding="utf-8")
    assert gate.check_offline(path, s.source) == good
    monkeypatch.setattr(TrainSettings, "from_config", lambda cfg: None)
    assert gate.preflight(tmp_path, s.source, path)["ok"]
    (tmp_path / ".round13e_gate.lock").mkdir()
    with pytest.raises(ValueError, match="prior R13E artifact"):
        gate.preflight(tmp_path, s.source, path)


@pytest.mark.parametrize("damage", ["old_panel", "old_horizon", "one_seed", "gpu", "workers", "nice",
                                     "arm", "metrics", "pair", "seed", "snapshot", "ci", "one_step",
                                     "tensor_bytes", "totals", "npz", "provenance", "emission_type"])
def test_offline_rejects_obsolete_incomplete_and_unpaired_evidence(offline_evidence, damage):
    s, path, doc = offline_evidence
    if damage == "old_panel":
        doc["states"] = 24
    elif damage == "old_horizon":
        doc["horizon_s"] = 60.
    elif damage == "one_seed":
        doc["seeds_per_state"] = 1
    elif damage == "gpu":
        doc["device"] = "cuda"
    elif damage == "workers":
        doc["workers"] = 8
    elif damage == "nice":
        doc["linux_nice"] = 15
    elif damage == "arm":
        doc["arms"] = {"off": doc["arms"]["off"]}
    elif damage == "metrics":
        del doc["rollouts"]["rows"][0]["off"]["delay_merge_veh_s"]
    elif damage == "pair":
        doc["rollouts"]["rows"][-1] = doc["rollouts"]["rows"][0]
    elif damage == "seed":
        doc["rollouts"]["rows"][0]["seed"] += 1
    elif damage == "snapshot":
        doc["rollouts"]["rows"][0]["world"] = "another-state"
    elif damage == "ci":
        doc["rollouts"]["state_cluster_run_mean"]["emergency_minus_off"]["delay_total_veh_s"]["clusters"] = 96
    elif damage == "one_step":
        doc["one_step"]["emergency_minus_off"]["all_lane1"]["bias"]["lo"] = -999
    elif damage == "tensor_bytes":
        doc["state_dict_sha256"] = "0" * 64
    elif damage == "totals":
        doc["rollouts"]["totals"]["off"]["runs"] = 48
    elif damage == "npz":
        (path.parent / "onestep_off.npz").unlink()
    elif damage == "emission_type":
        doc["emissions"]["temporal_accel_off"] = 1
    else:
        doc["train_roots"] = []
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.check_offline(path, s.source)


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
    assert script.index('smoke a "$A" "$CK" a') < script.index('smoke b "$B" "$L/runs/smoke_${TAG}_a/wm_s.pt" b')
    assert script.index('"${GATE[@]}" smoke-complete') < script.index('"${GATE[@]}" production-ready')
    assert script.index('smoke b "$B" "$INIT" b_final') < script.index('exec "$PYTHON"')
    assert 'requires --offline-report' in script
    assert '--launch-a' in script and '--launch-b' in script and '--launch)' not in script
    assert 'for CHECK in midrun postrun' in script
    assert script.index('test -f "$READY"') < script.index('exec "$PYTHON"')
    assert 'kill -0 "$WAITER"' in script
    assert "pkill" not in script and "killall" not in script and "kill -" not in script.replace(
        'kill -0 "$WAITER"', "")
    # The GPU guard only lists compute PIDs and runs before every training start.
    assert "nvidia-smi --query-compute-apps=pid --format=csv,noheader" in script
    assert script.index("gpu_guard" + chr(10) + "    smoke a") < script.index('smoke a "$A" "$CK" a')
    last_guard = script.rindex(chr(10) + "gpu_guard" + chr(10))
    assert script.index('"${GATE[@]}" production-ready') < last_guard < script.index('exec "$PYTHON"')


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
    assert "committed_sources(sha)" in source
    assert "committed_sources(BASELINE_COMMIT, baseline=True)" in source
    assert "export_sources(sha, names, target)" in source
    assert '"git", "archive", "--format=tar", sha, "--", *names' in inspect.getsource(gate.export_sources)
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
    def run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gate.subprocess, "run", run)
    waiter = gate.make_waiter(r13c_waiter, SimpleNamespace(
        root=tmp_path, stage=stage, mode="midrun", pid=12345, start_ticks=9876))
    assert identities == [(12345, tmp_path / f"runs/c4_wm_v10_round13e_{stage}", 9876)]
    assert waiter.out == tmp_path / f"chk_r13e/{stage}/midrun"
    assert waiter.work.is_relative_to(tmp_path / "chk_r13e")
    waiter.work.mkdir(parents=True)
    waiter.command(ROOT / "tools/r10b_checks/run_onestep.py",
                   "--config", "c4_wm_v10round13c.toml", "--var", "r10b=candidate", "--workers", 8)
    command, kw = commands[0]
    assert command[:3] == ["nice", "-n", "19"]
    args = command[6:]
    assert args[:2] == ["--config", gate.CONFIGS[stage].name]
    assert args[args.index("--workers") + 1] == "4"
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and kw["env"]["WMS_JAM_WORKERS"] == "4"
    assert all(kw["env"][key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    # B1: only the as-trained round-9 context is forced flags-off; the S-R12C
    # reference keeps the candidate's switches and enables the learned yield.
    for flag in ("emergency_braking", "secure_gap_veto"):
        assert f"r9=physics.{flag}=false" in args
        assert not any(a.startswith("r12c=physics.") for a in args)
    assert "r9=model.mainline_merge_yield_learnable=false" in args
    assert "r12c=model.mainline_merge_yield_learnable=true" in args
    # The implementation inherited by R13E must not signal an unrelated process.
    monkeypatch.setattr(r13c_waiter, "proc_stat", lambda pid: (9999, "S"))
    monkeypatch.setattr(r13c_waiter, "pidfd_open", lambda pid: pytest.fail("unowned PID"))
    assert process_type(12345, waiter.run, 9876).terminate().startswith("not sent")


def test_committed_source_allowlist_excludes_names_before_blob_export(monkeypatch):
    safe = [*gate.REQUIRED_SOURCE, "src/wms/__init__.py", "src/wms/model/entry_learned.json"]
    excluded = ["tools/SEALED/payload.py", "src/wms/holdout_reader.py", "configs/heldout.toml",
                "tools/cal_split.py", "tests/chunk5.py", "scripts/j4b.py", "tools/inc1.py",
                "tools/hetdata_truth.py", "src/wms/vType.py", "tools/payload.npz", "data/train.npy"]
    raw = b"\0".join(f"100644 blob {'1' * 40}\t{name}".encode() for name in (*safe, *excluded)) + b"\0"
    monkeypatch.setattr(gate.subprocess, "check_output", lambda *a, **kw: raw)
    assert gate.committed_sources("a" * 40) == sorted(safe)
    for name in ("../escape.py", "/src/wms/x.py", "src/wms/../x.py", "src\\wms\\x.py"):
        assert not gate.source_name_allowed(name)
    raw += f"120000 blob {'2' * 40}\tsrc/wms/link.py\0".encode()
    with pytest.raises(ValueError, match="non-regular"):
        gate.committed_sources("a" * 40)


def test_missing_committed_dependency_stops_before_archive_or_mkdir(monkeypatch):
    monkeypatch.setattr(gate.subprocess, "check_output", lambda *a, **kw:
                        b"100644 blob " + b"1" * 40 + b"\tsrc/wms/__init__.py\0")
    monkeypatch.setattr(gate.subprocess, "Popen", lambda *a, **kw: pytest.fail("no archive before validation"))
    with pytest.raises(ValueError, match="missing committed launch dependencies"):
        gate.committed_sources("a" * 40)


def test_archive_exports_only_explicit_allowlist_and_rejects_extra_members(tmp_path, monkeypatch):
    import io
    import tarfile
    names = ["src/wms/__init__.py", "pyproject.toml"]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"test"))
    commands = []
    def popen(command, **kw):
        commands.append(command)
        return SimpleNamespace(stdout=io.BytesIO(buffer.getvalue()), wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr(gate.subprocess, "Popen", popen)
    files = gate.export_sources("a" * 40, names, tmp_path)
    assert set(files) == set(names)
    assert commands == [["git", "archive", "--format=tar", "a" * 40, "--", *names]]
    with pytest.raises(ValueError, match="invalid source allowlist"):
        gate.export_sources("a" * 40, ["tools/SEALED/payload.py"], tmp_path)
    assert len(commands) == 1


def test_pinned_baseline_export_validates_all_names_and_hashes_before_use(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    base = tmp_path / gate.BASELINE_DIR
    files = {}
    for name in (*gate.BASELINE_HELPERS, "src/wms/__init__.py"):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"trusted source")
        files[name] = gate.digest(path)
    doc = {"commit": gate.BASELINE_COMMIT, "directory": gate.BASELINE_DIR, "files": files}
    assert gate.check_baseline(doc) == base
    (base / "src/wms/__init__.py").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="baseline source drift"):
        gate.check_baseline(doc)
    files["tools/SEALED/payload.py"] = "0" * 64
    monkeypatch.setattr(gate, "digest", lambda *a: pytest.fail("validate all names before bytes"))
    with pytest.raises(ValueError, match="baseline allowlist"):
        gate.check_baseline(doc)


def test_prepare_from_remote_sibling_checkout_and_revalidate_gitless_archive(tmp_path, monkeypatch):
    """The host checkout is a sibling of v5; the deployed copy has no Git DB."""
    import io
    import tarfile

    source, deployment = tmp_path / "checkout", tmp_path / "v5"
    source.mkdir()
    deployment.mkdir()
    sha = "a" * 40
    names = sorted({*gate.REQUIRED_SOURCE, "src/wms/__init__.py"})
    base_names = sorted({*gate.BASELINE_HELPERS, "src/wms/__init__.py"})
    monkeypatch.setattr(gate, "ROOT", source)
    monkeypatch.setattr(gate, "REMOTE_ROOT", deployment)
    monkeypatch.setattr(gate, "sys", SimpleNamespace(platform="linux"))
    def output(command, **kw):
        if command[1] == "rev-parse":
            return sha + "\n"
        if command[1] == "ls-files":
            return b""
        assert command[1:4] == ["ls-tree", "-rz", "--full-tree"]
        selected = names if command[4] == sha else base_names
        return b"\0".join(f"100644 blob {'1' * 40}\t{name}".encode() for name in selected) + b"\0"
    monkeypatch.setattr(gate.subprocess, "check_output", output)
    checked = []
    monkeypatch.setattr(gate.subprocess, "run", lambda command, **kw: checked.append(command))
    archived = []
    def popen(command, **kw):
        assert command[:4] == ["git", "archive", "--format=tar", command[3]]
        assert command[4] == "--"
        archived.append(command[5:])
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name in command[5:]:
                payload = f"source bytes for {name}".encode()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return SimpleNamespace(stdout=io.BytesIO(buffer.getvalue()), wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr(gate.subprocess, "Popen", popen)
    result = gate.prepare()
    prepared = Path(result["repo"])
    assert prepared == deployment / f"repo_r13e_{sha[:8]}"
    assert not (prepared / ".git").exists()
    assert archived == [names, base_names]
    assert checked[0][:6] == ["git", "diff", "--quiet", "--no-ext-diff", "--no-textconv", sha]
    monkeypatch.setattr(gate, "ROOT", prepared)
    manifest = gate.check_source()
    assert manifest["commit"] == sha and manifest["baseline"]["commit"] == gate.BASELINE_COMMIT
    assert set(manifest["files"]) == set(names)
    (prepared / "tools/added.py").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="source file set changed"):
        gate.check_source()


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


def test_fallback_variant_is_the_same_recipe_without_gs_and_disjoint_artifacts():
    main = gate.check_configs()
    gate.set_variant("fb")
    assert (gate.TAG, gate.RUN, gate.GS) == ("r13e_fb", "c4_wm_v10_round13e_fb", False)
    fb = gate.check_configs()
    for stage in ("a", "b"):
        assert fb[stage].physics.secure_gap_veto is False and fb[stage].physics.emergency_braking
        assert fb[stage].model.mainline_merge_yield_learnable
        assert fb[stage].runtime.out_dir == f"runs/c4_wm_v10_round13e_fb_{stage}"
        # Everything else equals the primary recipe.
        assert fb[stage].optim == main[stage].optim and fb[stage].loss == main[stage].loss
        assert fb[stage].curriculum == main[stage].curriculum
    assert fb["b"].model.init_from == "runs/c4_wm_v10_round13e_fb_a/wm_s.pt"
    with pytest.raises(ValueError):
        gate.set_variant("other")
    gate.set_variant("main")
    assert gate.CONFIGS["a"].name == "c4_wm_v10round13e.toml"


def test_fallback_variant_waiter_uses_disjoint_paths(r13c_waiter, monkeypatch, tmp_path):
    monkeypatch.setattr(r13c_waiter, "ProductionProcess", lambda *args: SimpleNamespace())
    gate.set_variant("fb")
    waiter = gate.make_waiter(r13c_waiter, SimpleNamespace(
        root=tmp_path, stage="a", mode="postrun", pid=1, start_ticks=2))
    assert waiter.run == tmp_path / "runs/c4_wm_v10_round13e_fb_a"
    assert waiter.out == tmp_path / "chk_r13e_fb/a/postrun"


@pytest.mark.parametrize("variant", ["main", "fb"])
def test_same_flags_reference_keeps_bytes_and_copies_only_the_three_switches(tmp_path, variant):
    """B1: S-R12C with the candidate's 13E switches; identical tensors."""
    gate.set_variant(variant)
    r12c_cfg = load_train_config(ROOT / "configs/c4_wm_v10round12c.toml")
    base_model = build_model(r12c_cfg, build_physics(r12c_cfg))
    base_dir, cand_dir = tmp_path / "baseline", tmp_path / "candidate"
    base_dir.mkdir()
    cand_dir.mkdir()
    flat = r12c_cfg.to_legacy_flat()
    save_checkpoint(base_dir / "wm_s.pt", base_model, flat)
    (base_dir / "physics.json").write_text(json.dumps(asdict(build_physics(r12c_cfg))), encoding="utf-8")
    cand_cfg = gate.check_configs()["a"]
    cand_physics = build_physics(cand_cfg)
    cand_model = build_model(cand_cfg, cand_physics)
    save_checkpoint(cand_dir / "wm_s.pt", cand_model, cand_cfg.to_legacy_flat())
    (cand_dir / "physics.json").write_text(json.dumps(asdict(cand_physics)), encoding="utf-8")
    path, info = gate.derive_same_flags_reference(base_dir / "wm_s.pt", cand_dir / "wm_s.pt",
                                                  tmp_path / "reference")
    assert info["flags"] == {"physics.emergency_braking": True, "physics.secure_gap_veto": gate.GS,
                             "model.mainline_merge_yield_learnable": True}
    before = torch.load(base_dir / "wm_s.pt", map_location="cpu", weights_only=True)
    after = torch.load(path, map_location="cpu", weights_only=True)
    assert set(after["model"]) == set(before["model"])
    assert all(gate.same_bytes(v, before["model"][k]) for k, v in after["model"].items())
    changed = {k for k in set(before["config"]) | set(after["config"])
               if before["config"].get(k) != after["config"].get(k)}
    assert changed == {"mainline_merge_yield_learnable"}
    old = json.loads((base_dir / "physics.json").read_text(encoding="utf-8"))
    new = json.loads((path.parent / "physics.json").read_text(encoding="utf-8"))
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} <= {
        "emergency_braking", "secure_gap_veto"}
    assert new["emergency_braking"] is True and new["secure_gap_veto"] is gate.GS
    from wms.model.checkpoint import load_checkpoint
    loaded = load_checkpoint(path, device="cpu", strict=True)
    assert asdict(loaded.model.physics) == asdict(cand_physics)
    assert loaded.model.mainline_merge_yield_learnable
    gate.check_yield_init(loaded.model)  # absent yield tensors load at the documented init
    # Any other physics difference fails closed; the target is never reused.
    damaged = asdict(cand_physics)
    damaged["dt_s"] = 1.0
    (cand_dir / "physics.json").write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(ValueError, match="beyond the 13E switches"):
        gate.derive_same_flags_reference(base_dir / "wm_s.pt", cand_dir / "wm_s.pt", tmp_path / "ref2")
    with pytest.raises(FileExistsError):
        (cand_dir / "physics.json").write_text(json.dumps(asdict(cand_physics)), encoding="utf-8")
        gate.derive_same_flags_reference(base_dir / "wm_s.pt", cand_dir / "wm_s.pt", tmp_path / "reference")


def _bash():
    import shutil
    for candidate in (r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        if candidate and Path(candidate).is_file() and "system32" not in candidate.lower():
            return candidate
    return None


@pytest.mark.parametrize("listing,wait,code", [("527550\n", "0", 5), ("", "0", 0),
                                               (" \n", "0", 0), ("FAIL", "0", 5)])
def test_gpu_guard_refuses_while_gpu_is_held_and_never_signals(tmp_path, listing, wait, code):
    import subprocess
    bash = _bash()
    if bash is None:
        pytest.skip("no POSIX bash")
    script = (ROOT / "scripts/round13e_gate.sh").read_text(encoding="utf-8")
    body = script[script.index("gpu_guard() {"):script.index("\n}\n", script.index("gpu_guard() {")) + 3]
    fake = tmp_path / "bin"
    fake.mkdir()
    tool = fake / "nvidia-smi"
    tool.write_text("#!/bin/sh\n" + ("exit 9\n" if listing == "FAIL" else
                                     f"printf '%s' '{listing}'\n"), encoding="utf-8", newline="\n")
    tool.chmod(0o755)
    test = (f"set -eu -o pipefail\nMODE=smoke\nWAIT={wait}\n{body}\n"
            "gpu_guard\necho GUARD_PASSED\n")
    runner = tmp_path / "guard.sh"
    runner.write_text(test, encoding="utf-8", newline="\n")
    fake_posix = subprocess.run([bash, "-c", f'cd "{fake.as_posix()}" && pwd'],
                                capture_output=True, text=True, check=True).stdout.strip()
    proc = subprocess.run([bash, "-c", f'export PATH="{fake_posix}:$PATH"; bash "{runner.as_posix()}"'],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == code, proc.stderr
    assert ("GUARD_PASSED" in proc.stdout) is (code == 0)
    if listing.strip() and listing != "FAIL":
        assert "527550" in proc.stderr and "--wait" in proc.stderr


def test_check_log_allows_only_weight_zero_on_diagnostic_infinities():
    """The weight-0 on-policy diagnostic NLL is +inf in S-R12C/13C logs; losses must stay finite."""
    import math as _math
    from wms.train import round13e_gate as gate
    row = {"merge_pivot_on_diagnostic_nll": _math.inf, "loss": 1.0}
    assert "non-finite metric/loss" not in gate.check_log([row], "", "a")["errors"]
    bad = {"merge_pivot_on_diagnostic_nll": 0.5, "loss": _math.inf}
    assert "non-finite metric/loss" in gate.check_log([bad], "", "a")["errors"]
    nested = {"x": {"merge_pivot_on_nll": _math.nan}}
    assert "non-finite metric/loss" in gate.check_log([nested], "", "a")["errors"]
