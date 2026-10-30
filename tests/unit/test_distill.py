"""CPU-only WM DAgger contracts; synthetic scenes, no traffic corpus or simulator."""
from types import SimpleNamespace

import pytest
import torch

from tools.distill import law, runtime, train
from wms.control.policy import CavPolicy, RsuGroupEncoder
from wms.control.policy14 import SmokeSampler
from wms.control.train_cli import _read_config
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    # configure_execution is a CLI helper; restore its process-global bindings
    # after this fixture so the union suite still tests default-OFF execution.
    for name in ("C4_EXECUTION_PARITY", "C4_SUMO_LEADER_RULE", "C4_EXECUTION_DRIVER"):
        monkeypatch.setenv(name, __import__('os').environ.get(name, ""))
    runtime.configure_execution("configs/execparity_cav_idm_v1.json")
    torch.set_num_threads(1)
    torch.manual_seed(701)
    model = WmSModel(ModelSpec(history_hidden=8, k_hidden=8, rollout_mode="hybrid",
        history_s=4., lane_model="hazard_v5", merge_decision="first_passage",
        lane_decision="sample", merge_passage_law="hybrid")).eval().requires_grad_(False)
    starts = SmokeSampler(model)
    teacher = SimpleNamespace(policy=CavPolicy(lane_head=True, accel_max=1.5).eval(),
                              group_encoder=RsuGroupEncoder().eval())
    from wms.control.lanes import LaneRules
    teacher.lane_rules = LaneRules()
    raw = _read_config("configs/c4_distill_mappo.toml")
    trainer = train.make_trainer(model, starts, raw, teacher=teacher)
    return model, starts, teacher, trainer, raw


def batch_with_history(model, starts):
    batch = starts.sample(1)
    # Explicit synthetic observable pre-t0 history; no label or future fields.
    mask = batch.rollout_kwargs["driver_history_mask"]
    mask[:] = batch.state["present"][..., None]
    return batch


def collect(setup, drive="student", horizon=2):
    model, starts, teacher, trainer, _ = setup
    rows = []
    batch = batch_with_history(model, starts)
    roll, entries, mix = runtime.rollout(model, batch, runtime.Feedback(teacher,
        trainer.policy, trainer.group_encoder, drive=drive, record=rows.append),
        horizon_s=horizon, seed=965160, pad_max=8)
    return rows, roll, entries, mix


def test_tiny_wm_entries_and_executed_residual_are_shared(setup):
    rows, roll, _, mix = collect(setup)
    assert len(rows) == 4 and torch.isfinite(roll["speeds"]).all()
    assert sum(mix.assigned) > 0
    assert any(bool((r["mask"] & r["entrant"]).any()) for r in rows)
    for row in rows:
        action = runtime.residual(setup[3].policy, setup[3].group_encoder,
                                  runtime.detach_inputs(row["inputs"]))
        action = torch.where(row["ready"], action, 0.)
        from wms.control.residual import executed_from_residual
        torch.testing.assert_close(row["student"], executed_from_residual(row["state"], action, "base"))
    assert all(p.grad is None for p in setup[0].parameters())


def test_representability_ledger_counts_live_entrant_cavs(setup):
    from tools.distill.represent import summarize, ledger_rows
    rows, _, _, _ = collect(setup, drive="teacher")
    summary = summarize(rows)
    ledger = list(ledger_rows(rows))
    assert len(ledger) == sum(int(r["mask"].sum()) for r in rows)
    assert any(r["entrant"] for r in ledger)
    for name in ("current", "cap_relative"):
        assert summary[name]["cav_ticks"] == len(ledger)
        assert 0 <= summary[name]["expressible_fraction"] <= 1
        assert 0 <= summary[name]["expressible_deviation_fraction"] <= 1
    for r in ledger:
        assert r["delta"] == pytest.approx(r["mappo_executed"] - r["ctrl0_executed"], abs=1e-6)


def test_executed_huber_loss_decreases_on_wm_teacher_states(setup):
    rows, _, _, _ = collect(setup, drive="teacher")
    trainer = setup[3]
    rows = [train.cpu_record(r) for r in rows]
    before = sum(float(train.executed_loss(trainer, r)[0].detach()) for r in rows)
    train.fit_initial_head(trainer, rows, "current")
    optimizer = torch.optim.Adam(trainer.actor_parameters, lr=3e-4)
    for _ in range(20):
        loss = torch.stack([train.executed_loss(trainer, r)[0] for r in rows]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = sum(float(train.executed_loss(trainer, r)[0].detach()) for r in rows)
    assert after < before
    assert all(p.grad is None for p in setup[0].parameters())
    assert all(p.grad is None for p in setup[2].policy.parameters())
    metrics = train.fit_metrics(trainer, rows, "current")
    assert set(metrics["zones"]) == {"all", "merge", "ramp", "mainline", "entrants"}


def test_student_observations_cannot_read_teacher_or_driver_state(setup):
    model, starts, _, trainer, _ = setup
    state = batch_with_history(model, starts).state
    clean = runtime.student_inputs(state)
    dirty = {**state, "driver_truth": torch.randn(1, 6, 99),
             "inferred_driver_parameters": torch.randn(1, 6, 19),
             "teacher_hidden": torch.randn(1, 6, 100),
             "cmd_accel_mps2": torch.randn(1, 6), "lane_intent": torch.randn(1, 6)}
    public = runtime.student_inputs(dirty)
    for name in ("own", "reports", "report_mask", "cav_mask"):
        assert torch.equal(getattr(clean[0], name), getattr(public[0], name))
    assert torch.equal(clean[1], public[1]) and torch.equal(clean[2], public[2])
    assert torch.equal(runtime.residual(trainer.policy, trainer.group_encoder, clean),
                       runtime.residual(trainer.policy, trainer.group_encoder, public))


def test_export_full_resume_and_public_policy_loader(setup, tmp_path):
    _, _, _, trainer, raw = setup
    trainer.real_return_provider = lambda **kwargs: None  # required config, never invoked
    path = tmp_path / "distilled.pt"
    train.save_checkpoint(trainer, path, raw, {"updates": 7})
    from wms.control.policy import load_policy_components
    actor, encoder, rules, payload = load_policy_components(path)
    assert actor.residual_on_native and not actor.has_lane_head and rules is None
    assert payload["format"] == "c4_dreamer/3" and payload["distillation"]["updates"] == 7
    assert payload["update_count"] == 0
    trainer.config = __import__("wms.control.dreamer", fromlist=["DreamerConfig"]).DreamerConfig(**raw["trainer"])
    trainer.load_checkpoint(path, new_phase=True)
    for a, b in zip(trainer.policy.parameters(), actor.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("seed", [964000, 964199, 977000, 977999, -1])
def test_excluded_seed_refused(seed):
    with pytest.raises(ValueError, match="excluded"):
        runtime.check_seed(seed)


def test_both_laws_share_ctrl0_and_cap_relative_can_brake_immediately(setup):
    from wms.control.alwayscmd import executed
    state = batch_with_history(setup[0], setup[1]).state
    zero = torch.zeros_like(state["v_mps"])
    assert torch.equal(law.execute(state, zero), executed(state, zero))
    assert torch.equal(law.execute(state, zero), law.execute(state, zero, "cap_relative"))
    for value in (-1.2, -.1, .4, 1.2):
        r = torch.full_like(zero, value)
        assert torch.equal(law.execute(state, r), executed(state, r))
    # Algebra fixture with a cap dead zone, not a traffic behaviour threshold.
    fixture = {"v_mps": torch.tensor([10.]), "base_mps2": torch.tensor([1.]),
               "cap_mps2": torch.tensor([.2])}
    r = torch.tensor([-.01], requires_grad=True)
    current = law.execute(fixture, r)
    caprel = law.execute(fixture, r, "cap_relative")
    assert current.item() == pytest.approx(.2) and caprel.item() == pytest.approx(.19)
    assert torch.autograd.grad(caprel.sum(), r)[0].item() == 1.


def test_paired_k8_check_zero_student_is_exact_ctrl0(setup):
    from tools.distill.check import evaluate_start
    model, starts, teacher, trainer, _ = setup
    batch = batch_with_history(model, starts)
    paired_batch = {"vehicle_ids": torch.arange(6)[None], "present": batch.state["present"][:, None],
                    "pair_ref": torch.tensor([0]), "pair_key": ["synthetic"],
                    "vehicle_ids_episode_encoded": False}
    result = evaluate_start(model, teacher, trainer.policy, trainer.group_encoder,
                            batch, seed=140014, horizons=(1, 2), pad_max=8, paired_batch=paired_batch)
    assert len(result["draw_costs"]) == 8
    for h in ("1", "2"):
        for metric in ("t0_hdv", "fleet_source"):
            paired = result["paired"][h][metric]["distilled_minus_ctrl0"]
            assert paired["draw_dJ"] == [0.] * 8
    # Source waiting is actually charged, not silently omitted.
    assert result["draw_costs"][0]["ctrl0"]["2"]["fleet_source"] > 40.


def test_dagger_cli_resume_is_bitwise_uninterrupted(setup, tmp_path, monkeypatch):
    import copy
    model, _, teacher, _, raw = setup
    raw = copy.deepcopy(raw)
    raw["distill"].update(horizon_s=2, pad_max=8, gradient_steps=2, replay_frames=8, checkpoint_every=1)

    class Starts(SmokeSampler):
        metadata = {0: {"world": "w965000m2400p20t300"}}
        distill_provenance = {"synthetic": True}

        def sample(self, *args, **kwargs):
            batch = super().sample(*args, **kwargs)
            batch.rollout_kwargs["driver_history_mask"][:] = batch.state["present"][..., None]
            return batch

    monkeypatch.setattr(train, "_read_config", lambda path: raw)
    monkeypatch.setattr(train, "load_assets", lambda *args: (copy.deepcopy(model), copy.deepcopy(teacher), []))
    monkeypatch.setattr(train, "load_training_starts", lambda raw, wm, seed: (Starts(wm), Starts(wm)))
    monkeypatch.setattr(train, "sha256", lambda path: "synthetic")
    common = ["--wm", "synthetic", "--wm-config", "synthetic", "--mappo", "synthetic",
              "--seed", "965160", "--device", "cpu"]
    full, first, resumed = (tmp_path / name for name in ("full", "first", "resumed"))
    assert train.main([*common, "--out", str(full), "--updates", "2"]) == 0
    assert train.main([*common, "--out", str(first), "--updates", "1"]) == 0
    assert train.main([*common, "--out", str(resumed), "--updates", "2", "--resume",
                       str(first / "policy_latest.pt")]) == 0
    a = torch.load(full / "policy_latest.pt", weights_only=True)
    b = torch.load(resumed / "policy_latest.pt", weights_only=True)
    for section in ("policy", "group_encoder"):
        for key in a[section]:
            assert torch.equal(a[section][key], b[section][key]), (section, key)
    assert a["distillation"]["updates"] == b["distillation"]["updates"] == 2
    assert torch.equal(a["distillation"]["generator"], b["distillation"]["generator"])
    assert len(a["distillation"]["replay"]) == len(b["distillation"]["replay"]) == 8


def test_policy14_cli_accepts_distilled_checkpoint_as_new_phase(setup, tmp_path, monkeypatch):
    """Exercise the real CLI full-resume path, with only data/scoring/update mocked."""
    import copy
    from wms.control import policy14_cli as cli
    model, starts, _, trainer, raw = setup
    raw = copy.deepcopy(raw)
    raw["paircrit"]["enabled"] = False
    raw["horizon"]["enabled"] = False
    raw["trainer"]["real_return_mix"] = 0.
    raw["data"].pop("focus_cell_fraction", None)
    raw["policy14"].update(critic_warmup_updates=0, independent_rescore=False,
                           signal_gate_updates=10)
    path = tmp_path / "distilled.pt"
    train.save_checkpoint(trainer, path, raw, {"updates": 2})
    monkeypatch.setattr(cli, "_read_config", lambda path: raw)
    monkeypatch.setattr(cli, "load_members", lambda *args: ([model], [{"checkpoint_sha256": "synthetic",
                                                                     "config_sha256": "synthetic"}]))
    class Provider:
        path = "synthetic"
        provenance = {"source_kind": "native"}
        _returns = torch.zeros(1)

        def __call__(self, **kwargs):
            raise AssertionError("synthetic resume test must not request real returns")

    monkeypatch.setattr(cli, "NPZRealReturnProvider", lambda *args, **kwargs: Provider())
    monkeypatch.setattr(cli, "_load_starts", lambda *args, **kwargs: (starts, starts))
    monkeypatch.setattr(cli, "check_train_seeds", lambda *args: None)
    monkeypatch.setattr(cli, "load_gate_panel", lambda *args: {"starts": []})
    monkeypatch.setattr(cli, "score_gate_panel", lambda trainer, *args: [1.-trainer.update_count*.001] * 16)
    monkeypatch.setattr(cli, "score_native_panel", lambda trainer, *args: {
        "update": trainer.update_count, "passed": True,
        "delta_per_start": [0.] * 16, "mean_delta": 0.})
    monkeypatch.setattr(cli, "independent_score", lambda *args: {"J": 1.})
    monkeypatch.setattr("wms.control.nativeinit.PANEL_SHA256", __import__("hashlib").sha256(b"panel").hexdigest())
    panel = tmp_path / "panel.json"
    panel.write_bytes(b"panel")
    original_load = cli.Policy14Trainer.load_checkpoint
    loads = []

    def load(self, path, **kwargs):
        result = original_load(self, path, **kwargs)
        loads.append(kwargs)
        return result

    def update(self):
        self.update_count += 1
        return {"update": self.update_count, "actor_loss": 0., "critic_loss": 0.,
                "actor_grad_norm": 0., "critic_grad_norm": 0.}

    monkeypatch.setattr(cli.Policy14Trainer, "load_checkpoint", load)
    monkeypatch.setattr(cli.Policy14Trainer, "update", update)
    output = tmp_path / "cli_phase"
    rc = cli.main(["--config", "synthetic", "--wm-ckpt", "synthetic", "--wm-config", "synthetic",
        "--real-returns", "synthetic", "--device", "cpu", "--max-updates", "10",
        "--output-dir", str(output), "--native-init", "--resume", str(path), "--new-phase",
        "--gate-panel", str(panel), "--compact"])
    import json
    summary = json.loads((output / "RUN_SUMMARY.json").read_text())
    assert rc == 0, summary.get("traceback", summary)
    assert loads == [{"new_phase": True, "resume_allow": []}]
