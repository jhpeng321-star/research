"""CPU-only synthetic receipts for opt-in gate measurement."""
import importlib.util
from pathlib import Path

import torch
from types import SimpleNamespace
from dataclasses import replace
import pytest
import json

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
    gated300.install(right, dict(ctrl0_reference=False, k16=False, score_training_horizon=False), ['a' * 64])
    right.counterfactual = replace(right.counterfactual, headroom_starts=False)
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
    def executed(state, residual, native, *args, **kwargs):
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


def test_headroom_uses_only_observable_initial_state_and_bounded_pressure():
    from wms.control.coopw import headroom_features, HeadroomSampler
    from wms.control.contract import SPEED_LIMIT_MPS
    t = fixture.trainer()
    state = fixture.fixture.scene(1, 3)
    state['s_m'][:] = torch.tensor([-50., -25., 0.])
    state['v_mps'][:] = SPEED_LIMIT_MPS
    free = headroom_features(state, t.counterfactual, 90)
    state['v_mps'][:] = 10.
    congested = headroom_features(state, t.counterfactual, 90)
    ramp = headroom_features(state, t.counterfactual, 90,
                            entry_queue=torch.tensor([0., 4., 0., 0., 0., 0.]))
    assert free['headroom'] == 0.
    assert 0 < congested['headroom'] < ramp['headroom'] <= 1.
    sampler = HeadroomSampler()
    assert 1. == sampler.multiplier(free) < sampler.multiplier(ramp) <= 2.
    changed = dict(state, unused_future_return=torch.full((1, 3), 1e9))
    assert headroom_features(changed, t.counterfactual, 90) == congested


def test_headroom_sampler_training_only_and_checkpointable():
    from wms.control.coopw import HeadroomSampler
    t = fixture.trainer()
    t.counterfactual = replace(t.counterfactual, headroom_starts=True, validation_uniform=True)
    t._tilt_sampler = HeadroomSampler()
    t._sample_starts(t.start_states)
    assert t._last_start_info['sampler'] == 'observable_headroom/1'
    assert 1 <= t._last_start_info['m'] <= 2
    saved = t._tilt_sampler.state_dict()
    t._sample_starts(t.start_states, validation=True)
    assert t._tilt_sampler.state_dict() == saved
    restored = HeadroomSampler()
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved


def test_k16_gate_and_joint_q10_gradient_and_cost_receipt():
    from wms.control.antiexploit import tail_credit
    t = fixture.trainer()
    v = build_variants(t.start_states.batch.state, t.counterfactual)
    samples = torch.arange(16.).reshape(16, 1, 1).expand(-1, 1, v.n_rows).clone() + 100.
    samples.requires_grad_()
    gated, on, _, _ = gated300.gate(samples, 43.91, gain_only=True, sample_count=16)
    assert on.all()  # Same strict 87.82 median margin, q10 positive.
    coefficients = torch.zeros(v.n_rows)
    coefficients[0] = 1.
    selected, receipt = tail_credit(dict(n_base=1, variants=v), gated, coefficients,
                                    enabled=True, sample_count=16)
    loss = (selected.mean((0, 1)) * coefficients).sum()
    assert loss == torch.quantile(samples[:, 0, 0], .1)
    loss.backward()
    assert samples.grad[1, 0, 0] == .5 and samples.grad[2, 0, 0] == .5
    assert samples.grad.sum() == 1 and receipt['antiexploit_futures'] == 16
    with pytest.raises(ValueError):
        gated300.gate(samples, 43.91)
    cost = gated300.cost_estimate(9, ctrl0_reference=True, k16=True)
    assert cost['forward_row_sim_seconds'] == 48000
    assert cost['forward_and_host_cache_factor_vs_native_K8'] == pytest.approx(20 / 9)


@pytest.mark.parametrize('memory', [False, True])
def test_k16_full_imagination_cpu_no_critic(tmp_path, monkeypatch, memory):
    monkeypatch.setattr(gated300, 'STEP_S', 60.)
    t = fixture.trainer(memory=memory)
    t.config = replace(t.config, horizon=1)
    fixture.enable_gain_only(t, fixture.cct2_authority(tmp_path, 300), k16=True)
    t.target_critic.forward = lambda *a: pytest.fail('K16 long credit called critic')
    t._training_imagination = True
    v = build_variants(t.start_states.batch.state, t.counterfactual)
    imagined = t.imagine(t.start_states.batch, variants=v, future_seed=11)
    assert imagined['n_futures'] == imagined['gated300_futures'] == 16
    assert imagined['gated300_long_difference'].shape == (16, 1, v.n_rows)
    assert imagined['rewards'].shape[:2] == (16, 1)
    components = {}
    loss, _ = t.actor_objective(imagined, 1., t.counterfactual.weights(0), gradient_components=components)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in t.actor_parameters)


@pytest.mark.parametrize('memory', [False, True])
@pytest.mark.parametrize('k16', [False, True])
def test_real_wm_ctrl0_zero_identity_crn_entrants_and_cpu_update(tmp_path, monkeypatch, memory, k16):
    from functools import partial
    from wms.control.bridge import make_cav_controller
    from wms.control.dreamer import DreamerConfig
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.policy14 import Policy14Trainer, independent_score
    from wms.model.entries import EntranceHistory
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    monkeypatch.setenv('C4_SUMO_LEADER_RULE', '1')
    monkeypatch.setenv('C4_EXECUTION_DRIVER', str(Path(__file__).resolve().parents[2] /
                                               'configs/execparity_cav_idm_v1.json'))
    monkeypatch.setattr(gated300, 'STEP_S', 60.)
    torch.manual_seed(71)
    starts = fixture.fixture.Starts(1, 4)
    starts.batch.state['is_cav'][:, -1] = False
    hist = EntranceHistory.defaults()
    hist.rate[:] = .5
    starts.batch.entry_feat = torch.tensor(hist.to_features()).float()[None]
    cfg = DreamerConfig(batch_size=1, horizon=1, imagine_entries=True, entry_pad_max=2, cav_exec_cap='idm',
        entrant_control=True, native_residual_train_group=True, native_residual_train_trunk=True,
        future_samples=8, pessimistic_credit=True, deterministic_crn=True,
        imagination_microbatch=int(memory), imagination_checkpoint_segment=2 if memory else 0,
        imagination_offload_futures=memory, imagination_offload_saved_tensors=memory)
    model = WmSModel(ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid',
        grad_decay_per_step=.9, lane_model='hazard_v5', merge_decision='first_passage',
        lane_decision='sample', merge_passage_law='hybrid'))
    policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True)
    cf = fixture.trainer().counterfactual
    cf.native_cav = 'head'
    def reward(*args):
        r = fixture.fixture.queue_reward(*args)
        return dict(r, hdv_hard_brake_seconds=torch.zeros_like(r['reward']))
    t = Policy14Trainer([model], policy, RsuGroupEncoder(hidden_dim=8, n_heads=2),
        fixture.fixture.Critic(), starts, observation_builder=build_observations,
        reward_fn=reward, safety=lambda state, command, **kw: command,
        controller_factory=partial(make_cav_controller, native_lane_change=True,
            native_ramp_merge='wm_law', symmetric_exec_cap=True), config=cfg, counterfactual=cf)
    fixture.enable_gain_only(t, fixture.cct2_authority(tmp_path, 300, entrant_control=True),
                            ctrl0_reference=True, score_training_horizon=True, k16=k16)
    t.target_critic.forward = lambda *a: pytest.fail('CTRL0 long credit called critic')
    measurements = []
    original_imagine = t.imagine
    def measured(*args, **kwargs):
        out = original_imagine(*args, **kwargs)
        measurements.append(out)
        return out
    t.imagine = measured
    # The zero actor and reference have identical execution even after entries.
    score = independent_score(t, starts.batch, native_reference=True)
    assert score['raw_dJ_vs_CTRL0_dense'] == 0.
    assert score['raw_dJ_vs_CTRL0_long'] == 0.
    assert score['long_gated_gain'] == 0.
    from wms.control.counterfactual import KIND_BASE
    imagined = measurements[-1]
    treatment = imagined['variants'].row_kind != KIND_BASE
    assert not imagined['gated300_long_difference'][..., treatment].any()
    t.imagine = original_imagine
    # Nonzero actor exercises the complete autograd/replay row geometry too.
    with torch.no_grad():
        policy.residual_head.bias.fill_(-.1)
    metrics = t.update()
    assert metrics['entries'] and metrics['critic_loss'] == 0
    assert torch.isfinite(torch.tensor(metrics['actor_loss']))


def test_interim_prepare_is_zero_charge_and_hash_bound(tmp_path, monkeypatch, capsys):
    from tools.candA import interim_a
    checkpoint = tmp_path / 'candidate.pt'
    checkpoint.write_bytes(b'checkpoint identity only; not parsed in prepare')
    monkeypatch.setattr(interim_a, 'run', lambda *a: pytest.fail('prepare attempted simulation'))
    assert interim_a.main([str(checkpoint)]) == 0
    text = capsys.readouterr().out
    assert interim_a.digest(checkpoint) in text and '14,400 sim-s' in text
    plan = interim_a.plan()
    assert len(plan['worlds']) == 8 and len({w['seed'] for w in plan['worlds']}) == 8
    assert plan['episodes'] * (plan['warmup_s'] + plan['horizon_s']) == plan['cap_sim_s'] == 14400
    assert plan['role'] == 'evaluation_only' and plan['arms'] == ['ours', 'CTRL0']
    assert len({(w['demand_vph'], w['penetration']) for w in plan['worlds']}) == 4


def test_interim_requires_exposure_audit_and_appended_checkpoint_registration(tmp_path, monkeypatch):
    from tools.candA import interim_a
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(dict(schema='interim_exposure_audit/1', passed=False)))
    with pytest.raises(ValueError, match='exposure audit'):
        interim_a.validate_audit(audit)
    monkeypatch.setattr(interim_a, 'validate_audit', lambda path: {})
    checkpoint, registration = tmp_path / 'candidate.pt', tmp_path / 'registration.md'
    checkpoint.write_bytes(b'checkpoint')
    registration.write_text('not the registered checkpoint hash')
    output = tmp_path / 'uncreated'
    with pytest.raises(ValueError, match='append the exact'):
        interim_a.run(checkpoint, output, audit, registration, None, None)
    assert not output.exists()


def test_300s_gate_keeps_strict_sigma_margin_and_q10_rule():
    margin = 2 * gated300.SIGMA[300]
    assert margin == 87.82
    samples = torch.tensor([[margin, 100., -1.]] * 2 + [[margin, 100., 200.]] * 6)
    _, on, _, _ = gated300.gate(samples, gated300.SIGMA[300], gain_only=True)
    assert on.tolist() == [False, True, False]


def test_old_checkpoint_long_receipt_defaults_remain_restorable(tmp_path):
    t = fixture.trainer()
    fixture.enable_gain_only(t, fixture.cct2_authority(tmp_path, 300))
    old = dict(t.gated300, config=dict(t.gated300['config']))
    old['config'].pop('ctrl0_reference')
    old['config'].pop('k16')
    other = fixture.trainer()
    gated300.checkpoint_restore(other, dict(gated300=old))
    assert other.gated300['H_long'] == 300
    assert not other.gated300['config']['ctrl0_reference'] and not other.gated300['config']['k16']
