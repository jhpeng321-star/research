"""S-ZEROSTART (2026-10-30): zero-start of candidate A's native-residual head (CPU only).

Pins: (1) the init zeroes the head's weight AND bias (trunk kept) and clears only the head's
Adam state; (2) the initial residual is exactly 0 for every CAV even under stochastic requests
and multiplicative persistent exploration; (3) the executed acceleration at residual 0 is
exactly the shared CTRL0 law ``min(base, cap)`` (alwayscmd.executed == sumo_idm.executed_accel
at 0); (4) the dead-zone facts that motivated the restart (a positive residual beyond the
safe-speed cap has zero effect and zero gradient; braking residuals are live); (5) the candA
config binds the zero start and the phase-relative exploration schedule.
"""
from pathlib import Path
from types import SimpleNamespace

import math

import pytest
import torch

from wms.control import alwayscmd, sumo_idm
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder
from wms.control.train_cli import GroupSignal
from wms.control.zerostart import SCHEMA, zero_residual_output

ROOT = Path(__file__).resolve().parents[2]


def scene():
    s = torch.tensor([[30., 5., 90., -400., -420., 0.]])
    return dict(s_m=s, v_mps=s.new_tensor([[5., 2., 8., 12., 9., 0.]]),
                a_mps2=torch.zeros_like(s), length_m=torch.full_like(s, 5.),
                lane=torch.tensor([[2, 1, 1, 0, 0, -1]]),
                present=torch.tensor([[True, True, True, True, True, False]]),
                is_cav=torch.tensor([[True, False, False, True, False, False]]))


def loaded_trainer(trunk_trainable=True):
    """A C1-like actor: nonzero head (positive latent), trunk trainable, Adam state present."""
    torch.manual_seed(7)
    policy = CavPolicy(lane_head=False, residual_on_native=True, accel_max=1.2)
    policy.configure_native_training(group=True, trunk=trunk_trainable)
    encoder = GroupSignal(RsuGroupEncoder())
    with torch.no_grad():
        policy.residual_head.weight.normal_(0., 0.05)
        policy.residual_head.bias.fill_(0.03)
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=5e-5)
    st = scene()
    obs = build_observations(st)
    out = policy(obs, encoder(obs, st["s_m"], None))[0]
    out.sum().backward()
    optimizer.step()  # populates Adam state for every parameter
    optimizer.zero_grad(set_to_none=True)
    return SimpleNamespace(policy=policy, group_encoder=encoder, actor_optimizer=optimizer), st


def test_zero_start_zeroes_output_layer_only_and_clears_head_adam_state():
    trainer, st = loaded_trainer()
    trunk_before = {n: p.detach().clone() for n, p in trainer.policy.named_parameters()
                    if not n.startswith("residual_head.")}
    head = list(trainer.policy.residual_head.parameters())
    assert all(p in trainer.actor_optimizer.state for p in head)
    other = next(p for n, p in trainer.policy.named_parameters() if n == "own_encoder.0.weight")
    other_state = {k: v.clone() for k, v in trainer.actor_optimizer.state[other].items()}
    receipt = zero_residual_output(trainer)
    assert receipt["schema"] == SCHEMA
    assert receipt["before"]["weight_abs_max"] > 0 and receipt["after"]["weight_abs_max"] == 0.0
    assert receipt["after"]["bias"] == [0.0] and receipt["adam_state_cleared_tensors"] == 2
    assert torch.equal(trainer.policy.residual_head.weight, torch.zeros_like(trainer.policy.residual_head.weight))
    assert torch.equal(trainer.policy.residual_head.bias, torch.zeros(1))
    assert all(p not in trainer.actor_optimizer.state for p in head)
    for n, p in trainer.policy.named_parameters():       # trunk/encoders kept bitwise
        if not n.startswith("residual_head."):
            assert torch.equal(p, trunk_before[n]), n
    for k, v in other_state.items():                       # unrelated Adam state kept
        assert torch.equal(trainer.actor_optimizer.state[other][k], v)


def test_initial_residual_is_exactly_zero_even_with_exploration_noise():
    trainer, st = loaded_trainer()
    obs = build_observations(st)
    signal = trainer.group_encoder(obs, st["s_m"], None)
    assert trainer.policy(obs, signal, deterministic=True)[0].abs().max() > 0   # C1-like start
    zero_residual_output(trainer)
    from wms.control.policy import PolicyNoise
    trainer.policy.residual_persistent_exploration = True
    for deterministic in (True, False):
        noise = PolicyNoise(torch.randn(1, 6), torch.zeros(1, 6, 3))
        out = trainer.policy(obs, signal, deterministic=deterministic, noise=noise)[0]
        assert torch.equal(out, torch.zeros_like(out))


def _geometry(n=64, seed=3):
    g = torch.Generator().manual_seed(seed)
    v = 5. + 20. * torch.rand(n, generator=g, dtype=torch.float64)
    vl = (v + 3. * torch.randn(n, generator=g, dtype=torch.float64)).clamp_min(0.)
    gap = 4. + 60. * torch.rand(n, generator=g, dtype=torch.float64)
    allowed = torch.full_like(v, 31.29)
    has = torch.rand(n, generator=g) > 0.2
    return v, vl, gap, allowed, has


def test_executed_acceleration_at_zero_residual_is_the_shared_ctrl0_law():
    v, vl, gap, allowed, has = _geometry()
    zero = torch.zeros_like(v)
    ex = sumo_idm.executed_accel(v, vl, gap, allowed, zero, -3., 1.2, has_leader=has)
    base = sumo_idm.base_accel(v, vl, gap, allowed, has_leader=has)
    cap = sumo_idm.safe_speed_cap(v, vl, gap, allowed, has_leader=has)
    ref = torch.maximum(torch.minimum(base, cap), -v / 0.5)
    assert torch.equal(ex, ref)
    state = dict(v_mps=v, lane=torch.ones(v.shape, dtype=torch.long), s_m=torch.zeros_like(v))
    # the imagination wrapper evaluates the same shared function
    ex2 = sumo_idm.executed_accel_scalar(float(v[0]), float(vl[0]), float(gap[0]), float(allowed[0]),
                                         0.0, -3., 1.2, has_leader=bool(has[0]))
    assert ex2 == float(sumo_idm.executed_accel(v[:1], vl[:1], gap[:1], allowed[:1], zero[:1], -3., 1.2,
                                                has_leader=has[:1])[0])


def test_dead_zone_positive_residual_is_inert_and_braking_is_live():
    v, vl, gap, allowed, has = _geometry()
    has = torch.ones_like(has)           # car-following rows: cap (single minGap) < base
    r = torch.zeros_like(v, requires_grad=True)
    base = sumo_idm.base_accel(v, vl, gap, allowed, has_leader=has)
    cap = sumo_idm.safe_speed_cap(v, vl, gap, allowed, has_leader=has)
    assert bool((cap <= base + 1e-12).all())
    zero = sumo_idm.executed_accel(v, vl, gap, allowed, torch.zeros_like(v), -3., 1.2, has_leader=has)
    pos = sumo_idm.executed_accel(v, vl, gap, allowed, torch.full_like(v, 0.05), -3., 1.2, has_leader=has)
    assert torch.equal(pos, zero)                       # a positive residual changes nothing
    ex = sumo_idm.executed_accel(v, vl, gap, allowed, r, -3., 1.2, has_leader=has)
    ex.sum().backward()
    strict = cap < base - 1e-9
    assert bool(strict.any()) and bool((r.grad[strict] == 0).all())   # zero gradient in the dead zone
    brake = sumo_idm.executed_accel(v, vl, gap, allowed, torch.full_like(v, -2.5), -3., 1.2, has_leader=has)
    assert bool((brake < zero - 1e-9).any())            # a large braking residual is live


def test_candA_config_binds_zero_start_and_phase_relative_exploration():
    from wms.control.train_cli import _read_config
    from wms.control.coopw import ExplorationConfig
    raw = _read_config(ROOT / "configs/c4_policy14_candA.toml")
    assert raw["policy14"]["zero_residual_output"] is True
    exploration = ExplorationConfig(**raw["exploration"])
    assert exploration.enabled is True and exploration.long_persistence is True
    assert exploration.persistence_curriculum_updates == [0, 150]      # unchanged
    assert exploration.persistence_seconds == [30.0, 60.0]
    first_update = 180                                                   # C1 resumes at update_count 180
    assert exploration.rho_at(first_update) == pytest.approx(0.10)
    assert exploration.rho_at(first_update + 150) == pytest.approx(0.50)
    assert 0.10 < exploration.rho_at(first_update + 75) < 0.50
    assert exploration.rho_at(first_update) > 0.0                        # active from phase update 0


# --------------------------------------------------------------- 2026-10-30 01:00 amendments
def _cluster_state(positions):
    s = torch.tensor([positions], dtype=torch.float32)
    k = s.shape[1]
    return dict(s_m=s, v_mps=torch.full_like(s, 10.), a_mps2=torch.zeros_like(s),
                length_m=torch.full_like(s, 5.), lane=torch.ones(1, k, dtype=torch.long),
                present=torch.ones(1, k, dtype=torch.bool), is_cav=torch.ones(1, k, dtype=torch.bool))


def test_uniform_group_size_covers_two_to_six_and_legacy_default_is_unchanged():
    from wms.control.counterfactual import CounterfactualConfig, build_variants
    state = _cluster_state([0., 40., 80., 120., 160., 200., 240., 280.])   # one 8-CAV chain
    base = dict(enabled=True, n_group=1, n_flip=3, cover_group_members=True, max_starts=1,
                group_radius_m=150.0, group_max_size=6)
    sizes = set()
    for seed in range(300):
        v = build_variants(state, CounterfactualConfig(**base, group_size_sampling="uniform"),
                           torch.Generator().manual_seed(seed))
        (group,) = v.groups
        sizes.add(len(group["members"]))
        # every member keeps its individual row (C_G needs every D_i); the group row exists
        assert all((0, m) in v.individual_rows for m in group["members"])
        assert len(set(group["members"])) == len(group["members"])
    assert sizes == {2, 3, 4, 5, 6}
    # legacy: always the full min(component, group_max_size) window
    legacy = {len(build_variants(state, CounterfactualConfig(**base), torch.Generator().manual_seed(s))
                  .groups[0]["members"]) for s in range(50)}
    assert legacy == {6}
    # a component smaller than the drawn size is clamped to the component
    small = _cluster_state([0., 50., 100.])
    assert {len(build_variants(small, CounterfactualConfig(**base, group_size_sampling="uniform"),
                               torch.Generator().manual_seed(s)).groups[0]["members"]) for s in range(100)} == {2, 3}
    with pytest.raises(ValueError):
        CounterfactualConfig(group_size_sampling="bogus")


def test_candA_config_binds_group_size_and_100m_exploration_radius():
    from wms.control.train_cli import _read_config
    from wms.control.counterfactual import CounterfactualConfig
    raw = _read_config(ROOT / "configs/c4_policy14_candA.toml")
    cf = CounterfactualConfig(**raw["counterfactual"])
    assert (cf.group_min_size, cf.group_max_size, cf.group_size_sampling) == (2, 6, "uniform")
    assert (cf.n_group, cf.max_starts, cf.n_flip) == (1, 1, 0)   # one group per start; flip rows retired
    assert raw["exploration"]["radius_m"] == 100.0


def test_exploration_noise_is_correlated_within_100m_chains_only():
    from wms.control.coopw import correlate, exploration_plan
    # chain 0-60-120 (links 60 m), gap 280 m, pair 400-460; and a 140 m link 700-840
    state = _cluster_state([0., 60., 120., 400., 460., 700., 840.])
    gid, shared = exploration_plan(state, 100.0, 4, 4.0, torch.Generator().manual_seed(1))
    g = gid[0].tolist()
    assert g[0] == g[1] == g[2] >= 0 and g[3] == g[4] >= 0 and g[3] != g[0]
    assert g[5] == -1 and g[6] == -1                       # 140 m link: NOT correlated at 100 m
    gid150, _ = exploration_plan(state, 150.0, 4, 4.0, torch.Generator().manual_seed(1))
    assert gid150[0, 5] == gid150[0, 6] >= 0               # the old 150 m radius would have grouped them
    # empirical correlation of the transformed standard normals
    rho, n = 0.5, 20000
    gen = torch.Generator().manual_seed(5)
    eps = torch.randn(n, 7, generator=gen)
    gid_b = gid.expand(n, -1)
    common = torch.randn(n, 2, generator=gen)              # one shared draw per (start, group) per decision
    mixed = correlate(eps, gid_b, common, rho)
    c = torch.corrcoef(mixed.T)
    assert abs(float(c[0, 1]) - rho) < 0.03 and abs(float(c[1, 2]) - rho) < 0.03
    assert abs(float(c[3, 4]) - rho) < 0.03
    assert abs(float(c[0, 3])) < 0.03 and abs(float(c[5, 6])) < 0.03   # different groups / ungrouped
    assert all(abs(float(mixed[:, i].std()) - 1.0) < 0.03 for i in range(7))  # marginals stay N(0,1)


# --------------------------------------------------- 2026-10-30 01:05 credit fix (no flip terms)
def _shipped_cf():
    from wms.control.train_cli import _read_config
    from wms.control.counterfactual import CounterfactualConfig
    raw = _read_config(ROOT / "configs/c4_policy14_candA.toml")
    return CounterfactualConfig(**raw["counterfactual"])


def test_candA_credit_has_no_flip_rows_and_only_nonnegative_coefficients():
    from wms.control.counterfactual import (KIND_FLIP, CounterfactualConfig, build_variants,
                                            row_coefficients)
    cfg = _shipped_cf()
    assert cfg.n_flip == 0 and cfg.w_team == [0.0, 0.0]
    weights = cfg.weights(10**6)
    assert weights == {"individual": 8.9, "group": 9.9, "coop": 8.9, "team": 0.0, "all": 1.0}
    state = _cluster_state([0., 30., 60., 90., 120., 150.])
    for seed in range(40):
        v = build_variants(state, cfg, torch.Generator().manual_seed(seed))
        assert not bool((v.row_kind == KIND_FLIP).any()) and all(not g["flips"] for g in v.groups)
        c = row_coefficients(v, weights, v.n_rows, 1)
        assert bool((c >= -1e-9).all())                          # nothing rewards making a team worse
        (group,) = v.groups
        assert c[group["row"]] == pytest.approx(9.9 + 8.9)       # group effect + synergy weights
        for member in group["members"]:
            assert abs(float(c[v.individual_rows[(0, member)]])) < 1e-6   # 8.9 - 8.9: solo effect cancels
        assert float(c[:1].sum()) == pytest.approx(1.0)          # D_all row
    # the retired configuration put -w_team on every flip row
    legacy = CounterfactualConfig(**{**cfg.__dict__, "n_flip": 3, "w_team": [10.3, 10.3]})
    v = build_variants(state, legacy, torch.Generator().manual_seed(0))
    cl = row_coefficients(v, legacy.weights(10**6), v.n_rows, 1)
    assert bool((cl < 0).any()) and bool((v.row_kind == KIND_FLIP).any())


def test_candA_objective_is_all_plus_group_plus_synergy_and_updates_finitely():
    """Objective = w_all*D_all + (w_group)*D_G + w_coop*C_G (+ w_ind*(D_i not in G)); one real
    DreamerTrainer.update() on the synthetic cooperation world runs with the shipped credit config."""
    import math as _math
    from tests.unit.test_coopw import calibrated_k, coop_reward, trainer_for
    cfg = _shipped_cf()
    w = cfg.weights(10**6)
    k = calibrated_k()
    cf = dict(curriculum_updates=[0], n_flip=0, group_max_size=3, group_size_sampling="max",
              w_all=[w["all"]], w_individual=[w["individual"]], w_group=[w["group"]],
              w_coop=[w["coop"]], w_team=[w["team"]], n_individual=0)
    trainer = trainer_for(coop_reward(k, 1.0), cf=cf)
    before = float(trainer.policy.theta.detach())
    metrics = trainer.update()
    assert _math.isfinite(metrics["actor_loss"]) and metrics["C_G"] > 0
    assert float(trainer.policy.theta.detach()) != before
    # uniform group sizes with n_flip = 0 also run (size drawn 2..3 on a three-CAV component)
    cf["group_size_sampling"] = "uniform"
    cf["group_min_size"] = 2
    trainer2 = trainer_for(coop_reward(k, 1.0), cf=cf)
    assert _math.isfinite(trainer2.update()["actor_loss"])


def test_candA_objective_identity_synergy_decomposition_and_registered_comparison():
    r"""Exact shipped objective, per start (D = row return - base return):

        1.0*D_all + 9.9*D_G + 8.9*C_G + 8.9*sum_i D_i,   C_G = D_G - sum_{i in G} D_i   (synergy)
      = 1.0*D_all + 18.8*D_G + 0*sum_{i in G} D_i + 8.9*sum_{i not in G} D_i

    The registered weights (w_team 10.3, n_flip 3) additionally had the team term
    10.3*sum_i M_i = 10.3*(3*D_G - sum_flip D_{G\i}) for |G| >= 3: +30.9 on D_G and -10.3 on each
    flip row (the negative sub-team rows).  Only that team term is removed."""
    from wms.control.counterfactual import CounterfactualConfig, build_variants, decompose, row_coefficients
    cfg = _shipped_cf()
    w = cfg.weights(10**6)
    state = _cluster_state([0., 30., 60., 90., 120., 150., 700.])    # 6-chain + one far solo CAV
    for seed in range(30):
        v = build_variants(state, cfg, torch.Generator().manual_seed(seed))
        g = torch.Generator().manual_seed(100 + seed)
        j = torch.randn(v.n_rows, generator=g, dtype=torch.float64) * 10.
        c = row_coefficients(v, w, v.n_rows, 1, dtype=torch.float64)
        base = j[v.base_row[0]]
        lhs = float((c * (j - base)).sum())
        d = decompose(j, v)
        (group,) = v.groups
        members = {m for m in group["members"]}
        d_group = d["D_G"][0]
        sum_member = sum(j[v.individual_rows[(0, m)]] - base for m in members)
        sum_other = sum(j[row] - base for (s, slot), row in v.individual_rows.items() if slot not in members)
        c_g = d["C_G"][0]
        assert c_g == pytest.approx(float(d_group - sum_member))
        explicit = (1.0 * d["D_all"][0] + 9.9 * d_group + 8.9 * c_g
                    + 8.9 * (float(sum_member) + float(sum_other)))
        assert lhs == pytest.approx(explicit, rel=1e-9, abs=1e-8)
        reduced = 1.0 * d["D_all"][0] + 18.8 * d_group + 8.9 * float(sum_other)
        assert lhs == pytest.approx(reduced, rel=1e-9, abs=1e-8)
    # registered (retired) objective: same plus 10.3*(3*D_G - sum_flip D_{G\i}) when |G| >= 3
    legacy = CounterfactualConfig(**{**cfg.__dict__, "n_flip": 3, "w_team": [10.3, 10.3]})
    big = [x for x in range(40) if len(build_variants(state, legacy, torch.Generator().manual_seed(x))
                                       .groups[0]["members"]) >= 3][0]
    v = build_variants(state, legacy, torch.Generator().manual_seed(big))
    cl = row_coefficients(v, legacy.weights(10**6), v.n_rows, 1, dtype=torch.float64)
    (group,) = v.groups
    assert float(cl[group["row"]]) == pytest.approx(9.9 + 8.9 + 3 * 10.3)    # 49.7
    assert all(float(cl[r]) == pytest.approx(-10.3) for r in group["flips"].values())


def test_authority_binds_entry_config_chain_for_the_long300_variant():
    import importlib.util
    spec = importlib.util.spec_from_file_location("canda_authority", ROOT / "tools/candA/authority.py")
    authority = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(authority)
    dense = authority.effective_config(ROOT, "configs/c4_policy14_candA.toml")
    long = authority.effective_config(ROOT, "configs/c4_policy14_candA_long300.toml")
    assert dense["long_credit"]["enabled"] is False and long["long_credit"]["enabled"] is True
    for cfg in (dense, long):          # everything else is inherited identically
        assert cfg["policy14"]["zero_residual_output"] is True and cfg["execution"]["leader_rule"] is True
        assert cfg["counterfactual"]["n_flip"] == 0 and cfg["exploration"]["radius_m"] == 100.0
    assert long["long_credit"]["mode"] == "gain_only" and long["long_credit"]["cct2_file"].endswith("_mix.json")
    chain = authority.resolve_chain(ROOT, "configs/c4_policy14_candA_long300.toml")
    assert chain[0].endswith("candA_long300.toml") and all(c in chain for c in authority.CONFIG_CHAIN)
    assert authority._config_long_credit_enabled(ROOT, "configs/c4_policy14_candA_long300.toml") is True
    assert authority._config_long_credit_enabled(ROOT) is False


# --------------------------------------------- 03:25 amendment: ADDITIVE exploration (m/s^2)
def _following_geometry(k=6):
    v = torch.tensor([[12., 9., 14., 13., 10., 0.]], dtype=torch.float32)[:, :k]
    vl = v - 2.0
    gap = torch.full_like(v, 80.0)
    return v, vl, gap, torch.full_like(v, 31.29), torch.ones_like(v, dtype=torch.bool)


def test_additive_exploration_gives_nonzero_head_gradient_at_zero_init_but_latent_mode_does_not():
    from wms.control.policy import PolicyNoise
    gen = torch.Generator().manual_seed(11)

    def head_grad(additive, std):
        trainer, st = loaded_trainer()
        zero_residual_output(trainer)
        policy, enc = trainer.policy, trainer.group_encoder
        policy.residual_additive_exploration = bool(additive)
        policy.residual_persistent_exploration = not additive
        obs = build_observations(st)
        signal = enc(obs, st["s_m"], None)
        eps = std * torch.randn(1, 6, generator=gen) if additive else torch.randn(1, 6, generator=gen)
        r = policy(obs, signal, deterministic=False, noise=PolicyNoise(eps, torch.zeros(1, 6, 3)))[0]
        v, vl, gap, allowed, has = _following_geometry()
        ex = sumo_idm.executed_accel(v, vl, gap, allowed, r, -3., 1.2, has_leader=has)
        (-ex).sum().backward()
        return float(policy.residual_head.weight.grad.abs().sum()), r

    g_add, r_add = head_grad(True, 0.5)
    assert float(r_add.abs().max()) > 0                       # perturbation executed from a zero head
    assert g_add > 0.0                                        # non-zero gradient at zero init
    g_lat, r_lat = head_grad(False, 0.5)
    assert float(r_lat.abs().max()) == 0.0 and g_lat == 0.0   # the retired latent noise was inert


def test_additive_noise_statistics_std_schedule_correlation_radius_and_persistence():
    from wms.control.coopw import (ExplorationConfig, additive_noise, exploration_plan,
                                   own_exploration_plan)
    cfg = ExplorationConfig(enabled=True, curriculum_updates=[180, 330], rho=[0.10, 0.50],
                            mode="additive_accel", radius_m=100.0, additive_correlation=0.5,
                            long_persistence=True)
    assert cfg.rho_at(180) == pytest.approx(0.10) and cfg.rho_at(330) == pytest.approx(0.50)
    assert cfg.rho_at(255) == pytest.approx(0.30)
    # 0-60-120 chain, 400-460 pair, 700 / 840 (140 m link, not grouped at 100 m)
    positions = [0., 60., 120., 400., 460., 700., 840.]
    n, steps, p = 6000, 6, 4.0
    state1 = _cluster_state(positions)
    state = {k: v.expand(n, -1).contiguous() for k, v in state1.items()}
    gen = torch.Generator().manual_seed(2)
    gid, shared = exploration_plan(state, cfg.radius_m, steps, p, gen)
    own = own_exploration_plan(n, 7, steps, p, gen)
    assert gid[0].tolist()[:5] == [0, 0, 0, 1, 1] and gid[0, 5] == -1 and gid[0, 6] == -1
    for std in (0.10, 0.30, 0.50):
        e = torch.stack([additive_noise(gid, shared[t], own[t], std, 0.5) for t in range(steps)])  # [D,B,K]
        flat = e.reshape(-1, 7)
        assert all(abs(float(flat[:, i].std()) - std) < 0.03 * std + 0.004 for i in range(7))   # std = schedule
        assert abs(float(flat.mean())) < 0.01
    e = torch.stack([additive_noise(gid, shared[t], own[t], 0.5, 0.5) for t in range(steps)])
    flat = e.reshape(-1, 7)
    c = torch.corrcoef(flat.T)
    assert abs(float(c[0, 1]) - 0.5) < 0.03 and abs(float(c[1, 2]) - 0.5) < 0.03 and abs(float(c[3, 4]) - 0.5) < 0.03
    assert abs(float(c[0, 3])) < 0.03 and abs(float(c[5, 6])) < 0.03 and abs(float(c[0, 5])) < 0.03   # radius 100 m
    for slot in (0, 5):                                       # persistence: lag-1 autocorrelation exp(-1/p)
        a, b = e[:-1, :, slot].reshape(-1), e[1:, :, slot].reshape(-1)
        assert abs(float(torch.corrcoef(torch.stack([a, b]))[0, 1]) - math.exp(-1 / p)) < 0.03
    with pytest.raises(ValueError):
        ExplorationConfig(mode="bogus")


def test_additive_exploration_in_imagination_is_crn_exact_training_only_and_in_mps2():
    from tests.unit.test_coopw import ThetaPolicy, coop_reward, trainer_for
    from wms.control.coopw import ExplorationConfig
    from wms.control.counterfactual import build_variants

    class AddPolicy(ThetaPolicy):
        supports_noise = True

        def forward(self, obs, signal, *, deterministic=False, noise=None):
            eps = noise.accel if (noise is not None and not deterministic) else torch.zeros(obs.cav_mask.shape)
            return torch.where(obs.cav_mask, self.theta + eps, torch.zeros_like(eps)), torch.zeros_like(eps)

    def actions(std, training, deterministic=False, seed=3):
        trainer = trainer_for(coop_reward(1.0, 1.0), cf=dict(group_max_size=3))
        trainer.policy = AddPolicy(theta=0.0)
        trainer.policy.residual_additive_exploration = True    # as set by policy14_cli for additive mode
        trainer.exploration = ExplorationConfig(enabled=True, rho=[std], mode="additive_accel", radius_m=100.0)
        torch.manual_seed(seed)
        batch = trainer.start_states.sample(1)
        variants = build_variants(batch.state, trainer.counterfactual, torch.Generator().manual_seed(0))
        trainer._training_imagination = training
        return trainer.imagine(batch, deterministic=deterministic, variants=variants)["actions"][0], variants

    off, variants = actions(0.0, True)
    on, _ = actions(0.4, True)
    untouched, _ = actions(0.4, False)
    torch.testing.assert_close(untouched, off)                                  # validation / rescore path
    assert float(off.abs().max()) == 0.0                                        # no exploration -> no eps in the residual
    torch.testing.assert_close(actions(0.4, True, deterministic=True)[0],
                               actions(0.0, True, deterministic=True)[0])
    assert not torch.equal(on[:, 0, :2], off[:, 0, :2])                         # close CAVs perturbed
    assert not torch.equal(on[:, 0, 2], off[:, 0, 2])                           # lone CAV perturbed too (own component)
    assert float(on[:, 0, :3].std()) < 0.8                                      # m/s^2 scale (std 0.4, not latent units)
    for (start, slot), row in variants.individual_rows.items():                 # CRN exact across rows
        torch.testing.assert_close(on[:, row, slot], on[:, start, slot])
