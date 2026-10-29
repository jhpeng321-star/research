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
