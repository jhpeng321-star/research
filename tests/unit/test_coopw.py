"""S-COOPW (COOPW.md): cooperation credit switched on, uncertainty pessimism, multi-CAV oversampling.

CPU only, file-free except reading the shipped configs.  The synthetic world has three CAVs
sharing one actor parameter theta (their common acceleration): CAVs 0 and 1 are a close group
whose JOINT action pays (k * d0 * d1, d = speed gain), every acting CAV pays a solo cost
(c * d_i), and CAV 2 is far away (alone).  Native CAVs hold speed.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wms.control.contract import MERGE_S1_M, PHYSICS_DT_S, RewardWeights
from wms.control.coopw import (ExplorationConfig, HorizonConfig, TiltSampler, UncertaintyConfig,
                               correlate, coverage, exploration_plan, multi_cav_flags, side_by_side,
                               start_features,
                               multi_cav_start, term_coefficients, term_objectives)
from wms.control.counterfactual import CounterfactualConfig, build_variants, row_coefficients
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.start_states import StartBatch

ROOT = Path(__file__).resolve().parents[2]
V0 = 8.0


@pytest.fixture(autouse=True)
def _rng():
    state = torch.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.set_rng_state(state)


def three_cavs(far=True):
    def row(values, dtype=torch.float32):
        return torch.tensor([values], dtype=dtype)
    return {"s_m": row([-20.0, -40.0, -900.0 if far else -60.0, -10.0]),
            "v_mps": row([V0, V0, V0, V0]), "a_mps2": row([0.0] * 4),
            "lane": row([1, 1, 1, 1], torch.long), "length_m": row([5.0] * 4),
            "present": row([True] * 4, torch.bool), "is_cav": row([True, True, True, False], torch.bool)}


class Sampler:
    def __init__(self, state):
        self.state = state

    def sample(self, batch_size, device="cpu"):
        return StartBatch({k: v.expand(batch_size, -1).clone() for k, v in self.state.items()},
                          {}, [{"split": "TRAIN", "synthetic": True}] * batch_size)


class CommandWM(nn.Module):
    """CAVs follow the trainer's command; HDVs follow CAV 0 with member gain ``gain``."""

    def __init__(self, gain=0.0):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.gain = gain
        self.physics = SimpleNamespace(dt_s=PHYSICS_DT_S)

    def rollout(self, state, *, n_steps, cav_controller, **kwargs):
        current = dict(state)
        traces = {key: [] for key in ("positions", "speeds", "lanes", "masks", "accel")}
        for _ in range(n_steps):
            command = cav_controller(current)
            cav = current["present"] & current["is_cav"]
            hdv_accel = self.gain * (current["v_mps"][:, :1] - current["v_mps"])
            speed = current["v_mps"] + PHYSICS_DT_S * hdv_accel
            nxt = {**current,
                   "s_m": torch.where(cav, command[..., 0], current["s_m"] + PHYSICS_DT_S * speed),
                   "v_mps": torch.where(cav, command[..., 2], speed),
                   "a_mps2": torch.where(cav, command[..., 3], hdv_accel)}
            for out, name in (("positions", "s_m"), ("speeds", "v_mps"), ("lanes", "lane"),
                              ("masks", "present"), ("accel", "a_mps2")):
                traces[out].append(nxt[name])
            current = nxt
        return {**{k: torch.stack(v, 1) for k, v in traces.items()}, "state": current}


def observations(state, report_age_s=None):
    return SimpleNamespace(own=state["v_mps"].unsqueeze(-1), reports=state["v_mps"][..., None, None],
                           report_mask=state["present"].unsqueeze(-1),
                           cav_mask=state["present"] & state["is_cav"])


class ThetaPolicy(nn.Module):
    """Deterministic shared acceleration theta for every acting CAV."""

    def __init__(self, theta=0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta))

    def forward(self, obs, signal, *, deterministic=False):
        action = torch.where(obs.cav_mask, self.theta.expand_as(obs.cav_mask.float()),
                             torch.zeros(obs.cav_mask.shape))
        return action, torch.zeros_like(action)


class ZeroGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(()))

    def forward(self, obs, state):
        return torch.zeros(obs.cav_mask.shape[0], 1)


class ConstCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, state):
        return self.bias.expand(state["v_mps"].shape[0])


def coop_reward(k, c):
    def reward(state, next_state, accel, weights):
        d = next_state["v_mps"][:, :3] - V0
        value = k * d[:, 0] * d[:, 1] - c * d.sum(-1)
        zero = torch.zeros_like(value)
        return {"reward": value, "cost_ttc": zero, "delay": zero, "throughput": zero,
                "smooth": zero, "hard_brake": zero, "fairness_delay": zero}
    return reward


def hdv_speed_reward(state, next_state, accel, weights):
    value = next_state["v_mps"][:, 3]
    zero = torch.zeros_like(value)
    return {"reward": value, "cost_ttc": zero, "delay": zero, "throughput": zero,
            "smooth": zero, "hard_brake": zero, "fairness_delay": zero}


COOP_CF = dict(enabled=True, native_cav="hold", n_individual=0, n_group=1, n_flip=1,
               cover_group_members=True, group_max_size=3, group_state_fraction=0.0,
               curriculum_updates=[0], w_all=[1.0], w_individual=[0.0], w_group=[0.0],
               w_coop=[0.0], w_team=[0.0])


def trainer_for(reward_fn, *, models=None, cf=None, state=None, **config):
    settings = dict(batch_size=1, horizon=4, gamma=1.0, lambda_=1.0, entropy_weight=0.0,
                    uncertainty_weight=0.0, checkpoint_imagination=False, imagine_entries=False,
                    checkpoint_every=0, validation_every=0, actor_lr=1e-2, critic_lr=1e-3,
                    grad_clip=1e6)
    settings.update(config)
    return DreamerTrainer(models or [CommandWM()], ThetaPolicy(), ZeroGroup(), ConstCritic(),
                          Sampler(state or three_cavs()), observation_builder=observations,
                          reward_fn=reward_fn, safety=lambda s, a: a, config=DreamerConfig(**settings),
                          reward_weights=RewardWeights(), device="cpu",
                          counterfactual=CounterfactualConfig(**{**COOP_CF, **(cf or {})}))


def objective_gradient(trainer, weights):
    """d(maximised actor objective)/d theta and the logged cooperation metrics."""
    batch = trainer.start_states.sample(1)
    variants = build_variants(batch.state, trainer.counterfactual, torch.Generator().manual_seed(0))
    imagined = trainer.imagine(batch, deterministic=True, variants=variants)
    loss, extra = trainer.actor_objective(imagined, 1.0, weights)
    (grad,) = torch.autograd.grad(-loss, [trainer.policy.theta])
    return float(grad), extra, variants


W = {"all": 1.0, "individual": 0.0, "group": 0.0, "coop": 0.0, "team": 0.0}


def calibrated_k(c=1.0, ratio=2.8):
    """Joint-payoff strength k with (joint gradient) / (one CAV's solo-cost gradient) = ratio."""
    joint, _, _ = objective_gradient(trainer_for(coop_reward(1.0, 0.0)), W)
    solo, _, _ = objective_gradient(trainer_for(coop_reward(0.0, 1.0)), W)   # = -3 x one CAV
    assert joint > 0 and solo < 0
    return ratio * c * (-solo / 3.0) / joint


# ------------------------------------------------------------------ cooperation credit
def test_synthetic_cooperation_actor_gradient_moves_toward_joint_action_only_with_coop_weights():
    k = calibrated_k()
    trainer = trainer_for(coop_reward(k, 1.0))
    off, extra, variants = objective_gradient(trainer, W)                  # loop14c: w_all only
    assert [len(g["members"]) for g in variants.groups] == [2]
    assert extra["C_G"] > 0 and extra["coop_positive_fraction"] == 1.0     # cooperation exists
    no_coop, _, _ = objective_gradient(trainer, {**W, "individual": 1.0, "group": 1.0})
    coop, _, _ = objective_gradient(trainer, {**W, "individual": 1.0, "group": 1.0, "coop": 1.0})
    assert off < 0 and no_coop < 0          # the shared cost of CAV 2 / the solo rows dominate
    assert coop > 0                         # C_G cancels the solo-row penalty on group members
    # Closed form with ratio x = 2.8 (joint / solo gradient): off x-3, no_coop 2x-7, coop 3x-7.
    unit = off / (2.8 - 3.0)
    assert no_coop == pytest.approx(unit * (2 * 2.8 - 7), rel=1e-4)
    assert coop == pytest.approx(unit * (3 * 2.8 - 7), rel=1e-4)


def test_cooperation_term_is_the_pure_interaction_gradient():
    k = calibrated_k()
    grads = []
    for c in (0.5, 2.0):
        trainer = trainer_for(coop_reward(k, c))
        with_coop, _, _ = objective_gradient(trainer, {**W, "all": 0.0, "individual": 1.0, "coop": 1.0})
        ind_only, _, _ = objective_gradient(trainer, {**W, "all": 0.0, "individual": 1.0})
        grads.append(with_coop - ind_only)
    assert grads[0] > 0 and grads[0] == pytest.approx(grads[1], rel=1e-5)   # no solo-cost part


def test_full_update_steps_theta_toward_cooperation_with_the_shipped_end_weights():
    raw = _config("c4_policy14_coop.toml")
    end = CounterfactualConfig(**raw["counterfactual"]).weights(10**6)
    k = calibrated_k()
    thetas = {}
    for name, weights in (("off", None), ("coop", end)):
        cf = {} if weights is None else dict(
            curriculum_updates=[0], w_all=[weights["all"]], w_individual=[weights["individual"]],
            w_group=[weights["group"]], w_coop=[weights["coop"]], w_team=[weights["team"]])
        trainer = trainer_for(coop_reward(k, 1.0), cf=cf)
        before = float(trainer.policy.theta.detach())
        metrics = trainer.update()
        assert math.isfinite(metrics["actor_loss"]) and metrics["C_G"] > 0
        thetas[name] = float(trainer.policy.theta.detach()) - before
    assert thetas["off"] < 0 < thetas["coop"]


def test_term_decomposition_matches_row_coefficients_and_guard():
    state = {k: v for k, v in three_cavs(far=False).items()}
    cfg = CounterfactualConfig(**{**COOP_CF, "n_flip": 3})
    variants = build_variants(state, cfg, torch.Generator().manual_seed(1))
    assert [len(g["members"]) for g in variants.groups] == [3]
    n = variants.n_rows
    unit = term_coefficients(variants, n, 1)
    weights = {"all": 1.0, "individual": 0.7, "group": 0.4, "coop": 0.6, "team": 0.3}
    combined = sum(weights[name] * unit[name] for name in unit)
    torch.testing.assert_close(combined, row_coefficients(variants, weights, n, 1))
    with pytest.raises(RuntimeError, match="anti-gaming"):
        row_coefficients(variants, {**weights, "coop": 0.8, "individual": 0.7}, n, 1)


# ------------------------------------------------------------------ shipped configuration
def _config(name):
    from wms.control.train_cli import _read_config
    return _read_config(ROOT / "configs" / name)


def test_coop_config_guard_anchor_and_flag_off_parent():
    raw = _config("c4_policy14_coop.toml")
    cfg = CounterfactualConfig(**raw["counterfactual"])
    assert cfg.cover_group_members and cfg.n_flip >= 1 and cfg.max_starts == 1
    assert cfg.curriculum_anchor == "first_update" and cfg.validation_uniform
    for x in list(cfg.curriculum_updates) + list(range(0, int(cfg.curriculum_updates[-1]) + 50)):
        w = cfg.weights(x)
        assert w["coop"] <= w["individual"] + 1e-12 and w["all"] == 1.0
    assert cfg.weights(cfg.curriculum_updates[1]) == {"individual": 0.0, "group": 0.0, "coop": 0.0,
                                                      "team": 0.0, "all": 1.0}
    unc = UncertaintyConfig(**raw["uncertainty"])
    assert unc.enabled and unc.future_samples >= 2
    assert unc.weight(0) == pytest.approx(DreamerConfig(**raw["trainer"]).uncertainty_weight)
    explore = ExplorationConfig(**raw["exploration"])
    assert explore.enabled and explore.rho_at(0) == 0.0 and 0.0 < explore.rho_at(10**6) < 1.0
    assert set(cfg.start_tilt) <= {"multi_cav", "side_by_side", "high_share", "high_demand", "congested"}
    import json
    gate = json.loads((ROOT / raw["horizon"]["gate_file"]).read_text(encoding="utf-8"))
    for name, stages in (("c4_policy14_coop.toml", 1), ("c4_policy14_coop_gpub.toml", 2)):
        horizon = HorizonConfig(**_config(name)["horizon"])
        horizon.gate_file = str(ROOT / horizon.gate_file)
        assert horizon.bind([gate["wm_sha256"]])["allowed_stage"] == stages
        assert horizon.bind(["0" * 64])["allowed_stage"] == 0              # any other WM: base horizon
    parent = _config("c4_policy14_loop14c.toml")
    assert not {"uncertainty", "exploration", "horizon"} & set(parent)
    assert CounterfactualConfig(**parent["counterfactual"]).weights(10**6) == {
        "individual": 0.0, "group": 0.0, "coop": 0.0, "team": 0.0, "all": 1.0}
    for key in ("trainer", "policy", "reward", "lanes", "data", "policy14"):
        assert raw[key] == parent[key], key
    loop = json.loads((ROOT / "configs/c4_loop14_coop_gpur.json").read_text(encoding="utf-8"))
    assert loop["policy"]["config"] == "configs/c4_policy14_coop.toml"
    assert loop["hosts"]["cpu"]["rsync_target"] == "cpuX" and "cpuX" in loop["hosts"]["cpu"]["ssh"]


def test_curriculum_anchor_first_update_and_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    cf = dict(curriculum_anchor="first_update", curriculum_updates=[0, 2, 6],
              w_all=[1.0, 1.0, 1.0], w_individual=[0.0, 0.0, 1.0], w_group=[0.0, 0.0, 1.0],
              w_coop=[0.0, 0.0, 1.0], w_team=[0.0, 0.0, 0.0])
    trainer = trainer_for(coop_reward(1.0, 1.0), cf=cf)
    trainer.update_count = 600                                  # e.g. an adopted policy0
    metrics = trainer.update()
    assert metrics["cf_weights"]["coop"] == 0.0 and trainer.coop_anchor == 600
    for _ in range(3):
        metrics = trainer.update()
    assert metrics["cf_weights"]["coop"] == pytest.approx(0.25)      # position 3 of the 2..6 ramp
    trainer.save_checkpoint(tmp_path / "p.pt")
    other = trainer_for(coop_reward(1.0, 1.0), cf=cf)
    other.load_checkpoint(tmp_path / "p.pt")
    assert other.coop_anchor == 600 and other._curriculum_position() == 4
    legacy = trainer_for(coop_reward(1.0, 1.0))
    legacy.update_count = 7
    assert legacy._curriculum_position() == 7 and legacy.coop_anchor is None


# ------------------------------------------------------------------ uncertainty pessimism
def test_higher_disagreement_lowers_the_actor_incentive():
    """Two WM members disagree on how strongly the HDV follows CAV 0; a faster CAV 0 raises the
    HDV reward in both but also their spread.  The pessimism term lowers the gradient toward the
    uncertain action, more so for larger disagreement."""
    def gradient(gap, weight):
        trainer = trainer_for(hdv_speed_reward, models=[CommandWM(0.3), CommandWM(0.3 + gap)],
                              cf={"enabled": False}, uncertainty_weight=weight)
        batch = trainer.start_states.sample(1)
        imagined = trainer.imagine(batch, deterministic=True)
        loss, _ = trainer.actor_objective(imagined, 1.0, W)
        (grad,) = torch.autograd.grad(-loss, [trainer.policy.theta])
        return float(grad), float(imagined["disagreement"].detach().mean())

    base, _ = gradient(0.2, 0.0)
    small, dis_small = gradient(0.2, 1.0)
    large, dis_large = gradient(0.6, 1.0)
    assert base > 0 and dis_large > dis_small > 0
    assert small < base
    assert (gradient(0.6, 0.0)[0] - large) > (base - small) > 0


def test_uncertainty_config_applies_futures_and_curriculum_to_training_imagination_only():
    unc = UncertaintyConfig(enabled=True, future_samples=3, curriculum_updates=[0, 10, 20],
                            weights=[0.1, 0.1, 0.5])
    trainer = trainer_for(coop_reward(1.0, 1.0), cf={"curriculum_anchor": "first_update"})
    trainer.uncertainty = unc
    batch = trainer.start_states.sample(1)
    assert trainer._imagination_config(batch) is trainer.config           # validation / rescore
    trainer.update_count, trainer.coop_anchor = 115, 100                  # position 15
    trainer._training_imagination = True
    cfg = trainer._imagination_config(batch)
    assert cfg.future_samples == 3 and cfg.uncertainty_weight == pytest.approx(0.3)
    with pytest.raises(ValueError):
        UncertaintyConfig(enabled=True, future_samples=1)


# ------------------------------------------------------------------ multi-CAV oversampling
def scene(cav_s, cav_v=20.0, ramp_s=None, slow=False):
    s = list(cav_s) + [cav_s[0] + 30.0] + ([ramp_s] if ramp_s is not None else [])
    k = len(s)
    lane = [1] * (len(cav_s) + 1) + ([2] if ramp_s is not None else [])
    v = [5.0 if slow else cav_v] * k
    return {"s_m": torch.tensor([s]), "v_mps": torch.tensor([v]), "lane": torch.tensor([lane]),
            "present": torch.ones(1, k, dtype=torch.bool),
            "is_cav": torch.tensor([[True] * len(cav_s) + [False] * (k - len(cav_s))])}


def test_multi_cav_flags_use_observable_state_only():
    cfg = CounterfactualConfig(enabled=True)
    near = scene([-150.0, -100.0], ramp_s=-80.0)
    flags = multi_cav_flags(near, cfg, horizon_s=10.0)
    assert flags == {"group": True, "near_merge": True, "congested": False, "ramp_encounter": True}
    far = scene([-1500.0, -1450.0])
    assert multi_cav_flags(far, cfg, 10.0)["group"] and not multi_cav_start(far, cfg, 10.0)
    jam = scene([-700.0, -650.0], slow=True)                     # RSU queue cells, merge out of reach
    assert multi_cav_flags(jam, cfg, 10.0)["congested"] and not multi_cav_flags(jam, cfg, 10.0)["near_merge"]
    lone = scene([-150.0, -600.0])                               # 450 m apart: no group
    assert not multi_cav_start(lone, cfg, 10.0)
    past = scene([MERGE_S1_M + 50.0, MERGE_S1_M + 80.0])
    assert not multi_cav_flags(past, cfg, 10.0)["near_merge"]


def test_side_by_side_and_start_features_are_observable():
    cfg = CounterfactualConfig(enabled=True)
    state = {"s_m": torch.tensor([[-300.0, -290.0, -200.0]]), "v_mps": torch.tensor([[20.0, 20.0, 20.0]]),
             "lane": torch.tensor([[0, 1, 1]]), "length_m": torch.full((1, 3), 5.0),
             "present": torch.ones(1, 3, dtype=torch.bool),
             "is_cav": torch.tensor([[True, True, False]])}
    assert side_by_side(state)                                   # 10 m apart, adjacent lanes
    far = {**state, "s_m": torch.tensor([[-300.0, -100.0, -200.0]])}
    assert not side_by_side(far)                                 # 195 m bumper gap > window
    same = {**state, "lane": torch.tensor([[1, 1, 1]])}
    assert not side_by_side(same)
    queue = torch.tensor([0.0, 0.0, 2500.0 / 3600.0, 0.1])
    f = start_features(state, cfg, 10.0, entry_queue=queue)
    assert f["high_demand"] and f["side_by_side"] and f["high_share"]          # 2/3 -> 0.4
    assert not start_features(state, cfg, 10.0, entry_queue=queue * 0.7)["high_demand"]   # 1750 vph
    assert not start_features(state, cfg, 10.0)["high_demand"]                          # no queue state
    share40 = {**state, "is_cav": torch.tensor([[True, False, False]])}
    assert start_features(share40, cfg, 10.0)["high_share"]                            # 1/3 -> 0.4


class FeatureSampler:
    """Uniform windows; a fraction ``rate`` carries the multi_cav feature (flag in the state)."""

    def __init__(self, rate, seed=0):
        self.rate, self.generator = rate, torch.Generator().manual_seed(seed)

    def sample(self, batch_size, device="cpu"):
        hit = bool(torch.rand((), generator=self.generator) < self.rate)
        return StartBatch({"flag": torch.tensor([[float(hit)]])}, {}, [{}])


def test_tilt_sampler_tilts_by_m_and_importance_weights_restore_the_target():
    sampler, tilt = FeatureSampler(rate=0.2), TiltSampler({"multi_cav": 1.0, "congested": 2.0})
    features = lambda batch: {"multi_cav": bool(batch.state["flag"].item())}     # noqa: E731
    hits, weights = [], []
    for update in range(6000):
        _, info = tilt.draw(sampler, 1, "cpu", features, max_tries=100,
                            generator=torch.Generator().manual_seed(update))
        hits.append(bool(info["features"]["multi_cav"]))
        weights.append(info["importance_weight"])
    # tilted share = 0.2 * 2 / (0.8 * 1 + 0.2 * 2) = 1/3
    assert sum(hits) / len(hits) == pytest.approx(1 / 3, abs=0.025)
    assert tilt.feature_counts["multi_cav"] / tilt.n == pytest.approx(0.2, abs=0.02)
    assert tilt.z == pytest.approx(1.2, abs=0.03) and tilt.ess_fraction == pytest.approx(1.44 / 1.6, abs=0.03)
    assert sum(weights) / len(weights) == pytest.approx(1.0, abs=0.04)          # E_q[p/q] = 1
    assert sum(w * h for w, h in zip(weights, hits)) / len(hits) == pytest.approx(0.2, abs=0.02)
    restored = TiltSampler({"multi_cav": 1.0, "congested": 2.0})
    restored.load_state_dict(tilt.state_dict())
    assert restored.z == tilt.z
    with pytest.raises(ValueError):
        TiltSampler({"hidden_driver_type": 1.0})


def test_validation_uniform_restores_the_target_distribution():
    trainer = trainer_for(coop_reward(1.0, 1.0),
                          cf={"start_tilt": {"multi_cav": 1.0}, "validation_uniform": True,
                              "coop_audit": True}, imagine_entries=False)
    calls = []

    class Spy(Sampler):
        def sample(self, *a, **k):
            calls.append("plain")
            return super().sample(*a, **k)

    spy = Spy(three_cavs())
    trainer._sample_starts(spy, validation=True)
    assert calls == ["plain"] and trainer._last_start_info is None and trainer._tilt_sampler.n == 0
    trainer.start_states = spy
    metrics = trainer.update()
    info = metrics["start_oversampling"]
    assert info["features"]["multi_cav"] is True and info["m"] == 2.0
    assert metrics["cf_coverage"]["n_groups"] == 1
    assert metrics["cf_coverage"]["cav_in_group_fraction"] == pytest.approx(2 / 3)


def test_coverage_and_term_audit_are_logged():
    trainer = trainer_for(coop_reward(1.0, 1.0), cf={"coop_audit": True, "term_grad_audit_every": 1})
    metrics = trainer.update()
    assert set(metrics["cf_term_grad_norms"]) >= {"all", "individual", "group", "coop"}
    assert metrics["cf_term_grad_norms"]["coop"] > 0
    batch = trainer.start_states.sample(1)
    variants = build_variants(batch.state, trainer.counterfactual)
    assert coverage(variants, batch.state)["rows_total"] == variants.n_rows
    imagined = trainer.imagine(batch, deterministic=True, variants=variants)
    terms, values = term_objectives(imagined, 1.0, 1.0)
    assert set(terms) == set(values) and all(math.isfinite(v) for v in values.values())


# ------------------------------------------------------------------ correlated exploration
def test_correlated_noise_statistics_and_joint_action_probability():
    """Group members: corr(eps_i, eps_j) = rho, marginals N(0,1), P(both > 0) rises from 1/4 to
    1/4 + asin(rho) / (2 pi); outside groups nothing changes; the shared part persists."""
    state = {"s_m": torch.tensor([[-20.0, -40.0, -900.0, -10.0]]),
             "lane": torch.ones(1, 4, dtype=torch.long), "present": torch.ones(1, 4, dtype=torch.bool),
             "is_cav": torch.tensor([[True, True, True, False]])}
    gen = torch.Generator().manual_seed(5)
    rho, eps_all = 0.5, []
    for draw in range(3000):
        gid, shared = exploration_plan(state, 150.0, 6, 4.0, gen)
        eps = torch.randn(6, 1, 4, generator=gen)
        eps_all.append(torch.stack([correlate(eps[t], gid, shared[t], rho) for t in range(6)]))
        if draw == 0:
            assert gid.tolist() == [[0, 0, -1, -1]]
            torch.testing.assert_close(correlate(eps[0], gid, shared[0], 0.0), eps[0])
            assert torch.equal(correlate(eps[0], gid, shared[0], rho)[0, 2:], eps[0][0, 2:])
    e = torch.cat(eps_all)[:, 0]                                       # [draws*6, K]
    corr = torch.corrcoef(e[:, :3].T)
    assert corr[0, 1] == pytest.approx(rho, abs=0.03)
    assert abs(corr[0, 2]) < 0.03 and e[:, 0].std() == pytest.approx(1.0, abs=0.03)
    joint = float(((e[:, 0] > 0) & (e[:, 1] > 0)).float().mean())
    assert joint == pytest.approx(0.25 + math.asin(rho) / (2 * math.pi), abs=0.015)
    assert float(((e[:, 0] > 0) & (e[:, 2] > 0)).float().mean()) == pytest.approx(0.25, abs=0.015)
    lag = torch.stack(eps_all)[:, :, 0, 0]                             # persistence of member 0
    lag_corr = torch.corrcoef(torch.stack([lag[:, :-1].reshape(-1), lag[:, 1:].reshape(-1)]))[0, 1]
    assert lag_corr == pytest.approx(rho * math.exp(-1 / 4.0), abs=0.04)


class NoisePolicy(ThetaPolicy):
    supports_noise = True

    def forward(self, obs, signal, *, deterministic=False, noise=None):
        eps = noise.accel if (noise is not None and not deterministic) else torch.zeros(obs.cav_mask.shape)
        action = torch.where(obs.cav_mask, self.theta + 0.1 * eps, torch.zeros_like(eps))
        return action, torch.zeros_like(action)


def test_exploration_keeps_crn_pairing_and_only_touches_training_imagination():
    def actions(rho, training, deterministic=False, seed=3):
        trainer = trainer_for(coop_reward(1.0, 1.0))
        trainer.policy = NoisePolicy()
        trainer.exploration = ExplorationConfig(enabled=True, rho=[rho])
        torch.manual_seed(seed)
        batch = trainer.start_states.sample(1)
        variants = build_variants(batch.state, trainer.counterfactual, torch.Generator().manual_seed(0))
        trainer._training_imagination = training
        imagined = trainer.imagine(batch, deterministic=deterministic, variants=variants)
        return imagined["actions"][0], variants                      # [H,N,K]

    off, variants = actions(0.0, True)
    on, _ = actions(0.6, True)
    untouched, _ = actions(0.6, False)
    torch.testing.assert_close(untouched, off)                       # validation / rescore path
    torch.testing.assert_close(actions(0.6, True, deterministic=True)[0],
                               actions(0.0, True, deterministic=True)[0])
    assert not torch.equal(on[:, 0, :2], off[:, 0, :2])               # group CAVs correlated
    torch.testing.assert_close(on[:, 0, 2], off[:, 0, 2])             # lone CAV unchanged
    for (start, slot), row in variants.individual_rows.items():       # CRN exact across rows
        torch.testing.assert_close(on[:, row, slot], on[:, start, slot])
    group = variants.groups[0]
    torch.testing.assert_close(on[:, group["row"], :2], on[:, 0, :2])


# ------------------------------------------------------------------ horizon curriculum
def test_horizon_gate_binding_is_fail_closed(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    gate = tmp_path / "gate.json"
    doc = {"schema": "coopw_horizon_gate/1", "wm_sha256": "abc",
           "horizons": {"10": {"verdict": "BASE"}, "30": {"verdict": "PASS"}, "60": {"verdict": "FAIL"}}}
    gate.write_text(json.dumps(doc))
    h = HorizonConfig(enabled=True, horizons_s=[10, 30, 60], curriculum_updates=[0, 50, 100],
                      gate_file=str(gate), entry_pad_max=[32, 64, 96])
    assert h.stage(500) == 0                                          # unbound: base horizon
    assert h.bind(["zzz"])["allowed_stage"] == 0                      # other WM checkpoint
    assert h.bind(["abc"])["allowed_stage"] == 1
    assert [h.stage(x) for x in (0, 49, 50, 500)] == [0, 0, 1, 1]
    assert h.pad_cap(30.0) == 64 and h.pad_cap(10.0) == 32
    doc["horizons"]["60"]["verdict"] = "PASS"
    gate.write_text(json.dumps(doc))
    assert h.bind(["abc"])["allowed_stage"] == 2 and h.stage(100) == 2
    missing = HorizonConfig(enabled=True, horizons_s=[10, 30], curriculum_updates=[0, 1],
                            gate_file=str(tmp_path / "none.json"))
    assert missing.bind(["abc"])["allowed_stage"] == 0


def test_horizon_stage_applies_to_medium_share_training_imagination_only():
    h = HorizonConfig(enabled=True, horizons_s=[4, 8], curriculum_updates=[0, 1])
    h.allowed_stage = 1
    trainer = trainer_for(coop_reward(1.0, 1.0), horizon=4)
    trainer.horizon_curriculum = h
    trainer.update_count = 5
    batch = trainer.start_states.sample(1)                            # 3 of 4 present are CAVs
    assert trainer._imagination_config(batch).horizon == 4
    trainer._training_imagination = True
    assert trainer._imagination_config(batch).horizon == 8
    low = Sampler({**three_cavs(), "is_cav": torch.tensor([[True, False, False, False]]),
                   "present": torch.tensor([[True] * 4])}).sample(1)
    low.state["s_m"] = torch.tensor([[-20.0, -40.0, -60.0, -80.0] + [-100.0] * 8])[:, :4]
    many = {k: torch.cat([v] * 3, 1) for k, v in low.state.items()}   # 3 CAVs of 12 -> 25 %
    assert trainer._imagination_config(StartBatch(many)).horizon == 8
    sparse = {k: v.clone() for k, v in many.items()}
    sparse["is_cav"][:] = False
    sparse["is_cav"][0, 0] = True                                      # 1 of 12 <= 10 %
    assert trainer._imagination_config(StartBatch(sparse)).horizon == 4
