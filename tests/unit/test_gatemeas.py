"""CPU-only synthetic receipts for opt-in gate measurement."""
import importlib.util
from pathlib import Path

import torch
from types import SimpleNamespace
from dataclasses import replace

from wms.control import gated300
from wms.control.counterfactual import KIND_CTRL0, build_variants

spec = importlib.util.spec_from_file_location('gm_fixture', Path(__file__).with_name('test_gated300.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_ctrl0_hook_and_long_pairing_toy():
    t = fixture.trainer()
    v = build_variants(t.start_states.batch.state, t.counterfactual,
                       torch.Generator().manual_seed(4))
    rng = torch.get_rng_state()
    extended = gated300.add_ctrl0_reference(v)
    assert torch.equal(rng, torch.get_rng_state())
    assert gated300.add_ctrl0_reference(extended) is extended
    assert extended.base_row == v.base_row
    refs = gated300.reference_index(extended)
    assert (extended.row_kind[refs] == KIND_CTRL0).all()
    # Learned-native has cost 500; CTRL0 and zero actor cost 300. Executor-only
    # improvement must disappear under CTRL0 pairing, with baseline detached.
    rewards = torch.full((8, 2, extended.n_rows), -150.)
    rewards[..., v.base_row[0]] = -250.
    rewards.requires_grad_()
    imagined = dict(rewards=rewards, components={'delay': -rewards}, variants=extended,
                    n_base=1, states=[[{}] * 3] * 8)
    t._training_imagination = True
    t.gated300 = dict(H_long=300, world_model_ids=t.world_model_ids,
                     config=dict(ctrl0_reference=True))
    actual = gated300.prefix(t, imagined, 1)
    assert torch.equal(actual['gated300_long_difference'][..., 0], torch.zeros(8, 1))
    t.gated300['config']['ctrl0_reference'] = False
    old = gated300.prefix(t, imagined, 1)
    assert (old['gated300_long_difference'][..., 0] == 200).all()
    actual['gated300_long_difference'][..., 0].sum().backward()
    assert (rewards.grad[..., 0] == 1).all()
    assert not rewards.grad[..., refs[0]].any()


def test_explicit_default_off_bitwise_update():
    left, right = fixture.trainer(), fixture.trainer()
    gated300.install(right, dict(ctrl0_reference=False), ['a' * 64])
    torch.manual_seed(119)
    expected = left.update()
    rng = torch.get_rng_state()
    torch.manual_seed(119)
    actual = right.update()
    for key in expected.keys() - {'wall_time_s'}:
        fixture.fixture.assert_tree_close(expected[key], actual[key], atol=0, rtol=0)
    for x, y in zip(left.actor_parameters, right.actor_parameters):
        assert torch.equal(x, y) and torch.equal(x.grad, y.grad)
    fixture.fixture.assert_tree_close(left.actor_optimizer.state_dict(), right.actor_optimizer.state_dict(), 0, 0)
    assert torch.equal(rng, torch.get_rng_state())


def test_bridge_ctrl0_uses_exact_zero_residual_law(monkeypatch):
    from wms.control.bridge import make_cav_controller
    from wms.control.policy import RsuGroupEncoder
    from wms.model.cav import WM_LANE_TARGET_KEY, EXEC_CAP_KEY
    from wms.control import execution
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    calls = []
    def executed(state, residual, native, *args):
        calls.append(residual.clone())
        return 2. + residual
    monkeypatch.setattr(execution, 'executed_residual', executed)
    def policy(obs, signal):
        return torch.zeros_like(obs.cav_mask, dtype=torch.float32), torch.zeros_like(obs.cav_mask, dtype=torch.float32)
    policy.residual_on_native = True
    policy.ctrl0_rows = torch.tensor([False, True])
    state = fixture.fixture.scene(2, 3)
    state[WM_LANE_TARGET_KEY] = state['lane'].clone()
    state[EXEC_CAP_KEY] = torch.full_like(state['s_m'], 2.)
    bridge = make_cav_controller(policy, RsuGroupEncoder(hidden_dim=8, n_heads=2),
        lambda state, command, **kw: command, native=lambda s: torch.zeros_like(s['s_m']),
        native_lane_change=True, act_mask=torch.zeros_like(state['present']))
    out = bridge(state)
    assert len(calls) == 2 and not calls[0].any()
    assert (out[1, :, 3] == 2).all()
    assert (out[0, :, 3] == 0).all()


def test_scoring_receipt_raw_ctrl0_separate_from_gated_composite():
    from wms.control.policy14 import independent_score
    t = fixture.trainer()
    t.gated300 = dict(config=dict(score_training_horizon=True, w_long=1., mode='gain_only'))
    batch = t.start_states.batch
    def imagine(batch, variants, **kw):
        variants = gated300.add_ctrl0_reference(variants)
        count = variants.n_rows
        dense = torch.full((8, 2, count), 100.)
        dense[..., 0] = 110.  # Dense actor cost 220 vs CTRL0 200 (harm).
        long_j = torch.full((count,), 1000.)
        long_j[0] = 800.  # Long raw gain 200, gate accepts it.
        improvement = torch.full((8, 1, count), 200.)
        return dict(n_base=1, variants=variants, components=dict(delay=dense,
            source_wait=torch.zeros_like(dense), hdv_hard_brake_seconds=torch.zeros_like(dense)),
            imagination_horizon_s=90., gated300_long_difference=improvement,
            gated300_horizon_s=300, gated300_long_J=long_j, gated300_reference='CTRL0')
    t.imagine = imagine
    rng = torch.get_rng_state()
    score = independent_score(t, batch)
    assert score['J_raw_dense'] == 220. and score['CTRL0_J_dense'] == 200.
    assert score['raw_gain_vs_CTRL0_dense'] == -20.
    assert score['J_raw_long'] == 800. and score['CTRL0_J_long'] == 1000.
    assert score['raw_gain_vs_CTRL0_long'] == 200.
    assert score['composite_gated_score'] == score['J'] == 20.
    assert score['long_credit_reference'] == 'CTRL0' and not score['critic_tail_included']
    assert torch.equal(rng, torch.get_rng_state())


def test_scoring_phase_freezes_curriculum_but_training_keeps_advancing():
    t = fixture.trainer()
    t.horizon_curriculum = SimpleNamespace(horizons_s=[60, 90], stage=lambda p: int(p >= 10))
    t.gated300 = dict(config=dict(score_training_horizon=True))
    t.config = replace(t.config, horizon=5)
    with gated300.scoring(t):
        assert t._imagination_config(t.start_states.batch).horizon == 60
    t.update_count = 20
    with gated300.scoring(t):
        assert t._imagination_config(t.start_states.batch).horizon == 60
    t._training_imagination = True
    assert t._imagination_config(t.start_states.batch).horizon == 90
