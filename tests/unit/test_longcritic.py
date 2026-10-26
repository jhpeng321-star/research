"""S-LONGCRITIC offline regressions: no simulator, no traffic data."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wms.control.longcritic import (LongCriticConfig, difference_loss, difference_target,
                                    reliability_weights)
from wms.control.coopw import ExplorationConfig, exploration_plan

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/c4"))
import loop14 as loop  # noqa: E402
import loop14_fast as fast  # noqa: E402


def helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reliability_is_measured_shrunk_horizon_specific_and_sha_bound():
    doc = {"schema": "coopw_horizon_gate/1", "wm_sha256": "a", "horizons": {
        "60": {"n_groups": 96, "sign_ci95": [.9, .97], "pearson_ci95": [.7, .9]},
        "120": {"n_groups": 96, "sign_ci95": [.4, .8], "pearson_ci95": [.7, .9]},
        "300": {"n_groups": 96, "sign_ci95": [.9, .97], "pearson_ci95": [-.1, .9]}}}
    assert reliability_weights(doc, ["a"])[60] == pytest.approx(.8 * .7 ** 2)
    assert reliability_weights(doc, ["a"])[120] == 0
    assert reliability_weights(doc, ["a"])[300] == 0
    assert 180 not in reliability_weights(doc, ["a"])
    assert reliability_weights(doc, ["b"]) == {}
    assert reliability_weights(doc, ["a", "b"]) == {}


def test_no_level_leak_and_uncertainty_shrinks_difference():
    native = torch.full((2, 300, 1), -100., dtype=torch.float64)
    policy = native.clone()
    policy[:, 0] -= 1
    policy[:, 200] += 5
    a, weight = difference_target(policy, native, .8)
    b, _ = difference_target(policy - 1e6, native - 1e6, .8)
    assert torch.equal(a, b) and a.item() == pytest.approx(3.2)
    policy[0, 200] += 100
    policy[1, 200] -= 100
    noisy, noisy_weight = difference_target(policy, native, .8)
    assert noisy_weight.item() < weight.item() and abs(noisy.item()) < abs(a.item())
    single, w = difference_target(policy[:1], native[:1], .8)
    assert single.item() == w.item() == 0


class ToyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(0.))

    def forward(self, state):
        return self.slope * state["investment"]


@pytest.mark.parametrize("persistent", [False, True])
def test_delayed_payoff_changes_short_horizon_actor_only_with_longcritic(persistent):
    # Sparse real native observations cannot identify the investment derivative.
    # Cost now, repayment at t=200; actor sees only 10 s plus the learned critic.
    outcomes = {}
    for enabled in (False, True):
        critic, native = ToyValue(), ToyValue()
        states = {"investment": torch.linspace(0., 1., 16)}
        p = torch.full((2, 300, 16), -1e6)
        b = p.clone()
        p[:, 0] -= states["investment"]
        p[:, 200] += 5 * states["investment"]
        delta, weight = difference_target(p, b, .9)
        opt = torch.optim.SGD(critic.parameters(), lr=.5)
        for _ in range(50):
            loss = critic({"investment": torch.zeros(16)}).square().mean()
            if enabled:
                loss = loss + difference_loss(critic, native, states, delta, weight, 1.)
            opt.zero_grad()
            loss.backward()
            opt.step()
        theta = torch.tensor(0., requires_grad=True)
        investment = theta.sigmoid()
        actor_loss = investment - critic({"investment": investment})
        gradient, = torch.autograd.grad(actor_loss, theta)
        outcomes[enabled] = -float(gradient)
    assert outcomes[False] < 0 < outcomes[True], outcomes
    exploration = ExplorationConfig(long_persistence=persistent)
    assert exploration.persistence_at(150) == (60 if persistent else 4)


def test_persistence_and_common_random_number_pairing():
    cfg = ExplorationConfig(enabled=True, long_persistence=True)
    assert [cfg.persistence_at(u) for u in (0, 75, 150)] == [30, 45, 60]
    state = {"s_m": torch.tensor([[0., 2.]]), "present": torch.ones(1, 2, dtype=torch.bool),
             "is_cav": torch.ones(1, 2, dtype=torch.bool), "lane": torch.zeros(1, 2, dtype=torch.long)}
    a = exploration_plan(state, 150., 60, cfg.persistence_at(0), torch.Generator().manual_seed(7))
    b = exploration_plan(state, 150., 60, cfg.persistence_at(0), torch.Generator().manual_seed(7))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    gen = torch.Generator().manual_seed(7)
    first, innovation = torch.randn(1, 1, generator=gen), torch.randn(1, 1, generator=gen)
    torch.testing.assert_close(a[1][1], math.exp(-1 / 30) * first
                               + math.sqrt(1 - math.exp(-2 / 30)) * innovation)


def test_defaults_off_and_exact_production_update_tensors():
    D = helper("test_c4_dreamer")
    D._seed(781)
    legacy = D._trainer()
    D._seed(781)
    explicit = D._trainer()
    explicit.longcritic = None
    explicit.exploration = ExplorationConfig(enabled=True, rho=[.5], long_persistence=False)
    assert LongCriticConfig().enabled is False
    D._seed(907)
    a = legacy.update()
    D._seed(907)
    b = explicit.update()
    for name in ("policy", "group_encoder", "critic", "target_critic", "actor_optimizer", "critic_optimizer"):
        D._assert_equal(getattr(legacy, name).state_dict(), getattr(explicit, name).state_dict())
    assert a["actor_loss"] == b["actor_loss"]


def test_long_group_budget_matches_actual_and_reuses_only_existing_prefixes(monkeypatch):
    from traffic_shepherd.c4.sumocredit import PairExtras
    extras = PairExtras(long_group_horizon=300, long_group_time=360, credit_imagine=False)
    calls = []
    def credit(env, point, prefix, factory, seed, world, bh, **kw):
        calls.append(kw)
        return {"extra_sim_s": 300, "collisions": 0, "teleports": 0}
    monkeypatch.setattr(extras, "_credit", credit)
    assert extras.run_point(None, {"t_s": 660}, {}, policy_factory=None, seed=968150,
                            world="synthetic", branch_h=300)["sim_s"] == 0
    assert extras.run_point(None, {"t_s": 360}, {("B", 300): {}, ("POLICY", 300): {}}, policy_factory=None, seed=968150,
                            world="synthetic", branch_h=300)["sim_s"] == 300
    assert calls == [{"horizon": 300, "group_only": True}]
    assert extras.sim_s_per_point(300) == 300
    with pytest.raises(ValueError, match="prefix"):
        extras.run_point(None, {"t_s": 360}, {}, policy_factory=None, seed=968150,
                         world="synthetic", branch_h=60)
    cfg = {"collect": {"worlds_per_cycle": 13, "branch_points": 2, "branch_first": 360,
                        "branch_cadence": 300, "branch_horizon": 300},
           "fastloop": {"two_box": True}}
    baseline = loop.collect_sim_s(cfg)
    cfg["collect"]["long_group"] = {"enabled": False}
    assert loop.collect_sim_s(cfg) == baseline
    cfg["collect"]["long_group"]["enabled"] = True
    assert loop.collect_sim_s(cfg) - baseline == 3900
    assert loop.long_group_sim_s(cfg) * 2 == 7800 <= 8000
    cfg["collect"]["worlds_per_cycle"] = 27
    with pytest.raises(loop.Refused, match="8,000"):
        loop.collect_sim_s(cfg)


def test_failed_attempts_remain_charged_and_long_cap_stops_dispatch(tmp_path, monkeypatch):
    T = helper("test_fastloop")
    cfg = T.fixture(tmp_path)
    cfg["collect"].update(branch_horizon=300, long_group={"enabled": True})
    state = loop.load_state(cfg)
    reg = loop.read_json(cfg["registration"])
    reg["budget"].update(sim_s_per_cycle=1e6, sim_s_total=1e6)
    Path(cfg["registration"]).write_text(json.dumps(reg))
    state["registration_sha256"] = loop.sha256(cfg["registration"])
    loop.save_state(cfg, state)
    root = Path(cfg["loop_root"])
    monkeypatch.setattr(loop, "disk_guard", lambda *a: None)
    def failed(*args):
        raise ConnectionError("synthetic startup failure; actual zero")
    monkeypatch.setattr(fast, "_world", failed)
    for _ in range(8):
        try:
            fast.collect(loop, cfg, state, {"attempt": 1}, root)
        except RuntimeError:
            pass
    rows = [json.loads(s) for s in (root / "seed_ledger.jsonl").read_text().splitlines()]
    before = sum(r.get("long_group_sim_s_charged", 0) for r in rows)
    assert before == 7200
    with pytest.raises(loop.Refused, match="8,000"):
        fast.collect(loop, cfg, state, {"attempt": 9}, root)
    assert state["sim_s_charged"] == sum(r.get("sim_s_charged", 0) for r in rows)


def test_actual_long_arm_plan_uses_one_group_arm_and_existing_300s_ledgers(monkeypatch):
    from traffic_shepherd.c4.sumocredit import PairExtras
    extras = PairExtras(long_group_horizon=300, long_group_time=360, credit_imagine=False)
    frame = SimpleNamespace(present=np.ones(2, bool), is_cav=np.ones(2, bool),
                            vid_index=np.arange(2), lane=np.ones(2, int), s=np.array([-20., -30.]))
    env = SimpleNamespace(frame=frame, builder=SimpleNamespace(vehicle_ids=["a", "b"]))
    monkeypatch.setattr(extras, "_load", lambda *a: None)
    calls = []
    rec = {"J": 100., "road": 70., "source": 30., "collisions": 0,
           "teleports": 0, "dropped_vehicle_steps": 0}
    def run(*args):
        calls.append(args[2])
        return {**rec, "J": 80., "sim_s": args[2]}
    monkeypatch.setattr(extras, "_run_arm", run)
    output = extras.run_point(env, {"t_s": 360}, {("B", 300): rec, ("POLICY", 300): rec},
                              policy_factory=lambda: None, seed=968150, world="synthetic", branch_h=300)
    assert calls == [300] and output["sim_s"] == 300
    assert output["credit_long"]["sumo"] == {"B": 100., "ALL": 100., "GROUP": 80.}


def test_optin_config_preserves_main_native_lane_and_imagination_contract():
    from wms.control.train_cli import _read_config
    old = _read_config("configs/c4_policy14_main14_nativelane.toml")
    new = _read_config("configs/c4_policy14_longcritic.toml")
    assert {k: v for k, v in new.items() if k not in ("longcritic", "exploration")} == {
        k: v for k, v in old.items() if k != "exploration"}
    assert new["policy14"]["native_lane_change"] is True
    assert new["horizon"] == old["horizon"]


def _real_wm_trainer():
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.policy14 import Policy14Trainer, SmokeSampler, controller_factory
    from wms.control.train_cli import GroupSignal
    from wms.control.real_returns14 import RealReturnBatch, ScaledTeamCritic
    from wms.control.objective14 import authoritative_reward
    from wms.control.obs import build_observations
    from wms.control.safety import safe_accel
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    torch.manual_seed(17)
    torch.set_num_threads(4)
    model = WmSModel(ModelSpec(rollout_mode="hybrid"))
    starts = SmokeSampler(model)
    class Native:
        provenance = {"source_kind": "native", "horizon_s": np.array([300.])}
        def __call__(self, batch_size, device, update):
            return RealReturnBatch(starts.sample(batch_size, device).state,
                                   torch.full((batch_size,), -100., device=device), 1.)
    provider = Native()
    trainer = Policy14Trainer([model], CavPolicy(lane_head=False), GroupSignal(RsuGroupEncoder()),
        ScaledTeamCritic(), starts, observation_builder=build_observations,
        reward_fn=authoritative_reward, safety=safe_accel, controller_factory=controller_factory,
        config=DreamerConfig(batch_size=1, horizon=2, gamma=1., real_return_mix=1.,
                             checkpoint_imagination=False, checkpoint_every=0, validation_every=0),
        real_return_provider=provider,
        counterfactual=CounterfactualConfig(enabled=True, native_cav="hold"))
    return trainer, model, provider


def test_real_wm_longcritic_target_is_no_grad_paired_and_checkpointed(tmp_path, monkeypatch):
    from wms.control.longcritic import LongCritic
    trainer, model, provider = _real_wm_trainer()
    auxiliary = LongCritic(trainer, LongCriticConfig(enabled=True, horizons_s=[60], native_warmup_updates=1),
                           provider, ["synthetic"])
    auxiliary.weights = {60: .4}  # Synthetic evidence, never a production setting.
    auxiliary.warmup(trainer)
    trainer.longcritic = auxiliary
    config_before = trainer.config
    metrics = trainer.update()
    assert metrics["longcritic_active"] and metrics["longcritic_horizon_s"] == 60
    assert math.isfinite(metrics["longcritic_loss"])
    assert trainer.config is config_before and not trainer._training_imagination
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in auxiliary.native_critic.parameters())
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    target = tmp_path / "long.pt"
    trainer.save_checkpoint(target)
    payload = torch.load(target, weights_only=True)
    assert payload["longcritic"]["warm"] is True
    before = copy.deepcopy(auxiliary.native_critic.state_dict())
    trainer.load_checkpoint(target)
    assert all(torch.equal(v, auxiliary.native_critic.state_dict()[k]) for k, v in before.items())
