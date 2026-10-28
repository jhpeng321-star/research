"""CPU, synthetic traffic only: signs, exact OFF, actual continuation/VJP/resume."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.gated300.cct1 import compute
from tools.gated300.cct2 import compute as compute_cct2
from wms.control import gated300
from wms.control.counterfactual import CounterfactualConfig, build_variants, row_coefficients
from wms.control.dreamer import _rng_state, _restore_rng
from wms.control.policy14 import independent_score

spec = importlib.util.spec_from_file_location('gated300_fixture', Path(__file__).with_name('test_longmem.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def authority(tmp_path, hc=120, **extra):
    doc = dict(wm_sha256='a' * 64, entrant_control=False, deterministic_crn=True,
               rows=[dict(world=i, dJ={str(h): [-200., -100.] if h <= hc else [0., -100.]
                                      for h in gated300.SIGMA}) for i in range(4)])
    result = compute(doc, panel='pairs300')
    result.update(extra)
    path = tmp_path / f'WM_CCT1_{hc}_{len(extra)}.json'
    path.write_text(json.dumps(result))
    return str(path)


def trainer(memory=False):
    t = fixture.trainer(horizon=5, batch=1, slots=3, width=8, counterfactual=True,
                        future_samples=8, pessimistic_credit=True, deterministic_crn=True,
                        micro=1 if memory else 0, segment=8 if memory else 0, offload=memory)
    t.world_models = t.world_models[:1]
    t.world_models[0].encode_drivers = lambda state, **kw: (
        None, None, {'v0': torch.full_like(state['v_mps'], 30.)})
    t.world_model_ids = t.world_model_ids[:1]
    t.long_credit_requested = True  # candidate-A's existing critic-free dense objective
    def scaled_reward(*args):
        out = fixture.reward(*args)
        return {**out, 'reward': out['reward'] * 1000, 'delay': out['delay'] * 1000}
    t.reward_fn = scaled_reward
    t.counterfactual = CounterfactualConfig(enabled=True, max_starts=1, n_individual=0,
        n_group=1, n_flip=3, group_min_size=3, group_max_size=3, cover_group_members=False,
        curriculum_updates=[0], w_individual=[0.], w_coop=[0.], w_group=[18.8],
        w_team=[10.3], w_all=[1.])
    return t


def enable(t, path, **options):
    return gated300.install(t, dict(enabled=True, cct1_file=path, **options), ['a' * 64])


def cct2_authority(tmp_path, hg=120, **extra):
    rows = [dict(world=i, dJ={str(h): ([-200., -100.] if h == hg else [0., -100.])
                              for h in gated300.SIGMA}) for i in range(12)]
    document = dict(entrant_control=False, deterministic_crn=True, rows=rows)
    result = compute_cct2(dict(role='pairs300', wm_sha256='a' * 64, **document),
                          dict(role='validation', wm_sha256='a' * 64, **document))
    assert result['H_G'] == hg
    result.update(extra)
    path = tmp_path / f'WM_CCT2_{hg}_{len(extra)}.json'
    path.write_text(json.dumps(result))
    return str(path)


def enable_gain_only(t, path, **options):
    return gated300.install(t, dict(enabled=True, mode='gain_only', cct2_file=path, **options),
                            ['a' * 64])


def test_sign_gate_both_tails_strict_threshold_and_only_on_gradients():
    # Columns: benefit, harm, below threshold, benefit with bad q10, harm with bad q90.
    samples = torch.tensor([[100., -100., 82.44, -10., 10.]] * 2
                           + [[200., -200., 82.44, 200., -200.]] * 6, requires_grad=True)
    result, on, agreement, median = gated300.gate(samples, 41.22)
    assert on.tolist() == [True, True, False, False, False]
    assert agreement.tolist() == [True, True, True, False, False]
    assert median[0] > 0 and median[1] < 0  # cost difference signs are the opposite
    result.sum().backward()
    assert torch.equal(samples.grad[:, :2], torch.ones(8, 2))
    assert torch.count_nonzero(samples.grad[:, 2:]) == 0


def test_gain_only_gate_keeps_confident_gain_direction_only():
    # Columns: gain, harm, exactly at margin, gain with bad q10, harm with bad q90.
    samples = torch.tensor([[100., -100., 82.44, -10., 10.]] * 2
                           + [[200., -200., 82.44, 200., -200.]] * 6, requires_grad=True)
    result, on, agreement, _ = gated300.gate(samples, 41.22, gain_only=True)
    assert on.tolist() == [True, False, False, False, False]  # harm gets no credit
    assert agreement.tolist() == [True, False, True, False, False]  # margin keeps sign only
    assert gated300.gate(samples, 41.22)[1].tolist() == [True, True, False, False, False]
    result.sum().backward()
    assert torch.equal(samples.grad[:, :1], torch.ones(8, 1))
    assert torch.count_nonzero(samples.grad[:, 1:]) == 0


@pytest.mark.parametrize('mode', ['disabled', 'off', 'missing', 'hg0'])
def test_gain_only_off_and_untrusted_are_bitwise_identical_updates(tmp_path, mode):
    baseline, actual = trainer(), trainer()
    if mode == 'off':
        gated300.install(actual, dict(enabled=False, mode='gain_only',
                                      cct2_file=str(tmp_path / 'unused.json')), ['a' * 64])
    elif mode == 'missing':
        enable_gain_only(actual, str(tmp_path / 'missing.json'))
    else:
        enable_gain_only(actual, cct2_authority(tmp_path, hg=0))
    assert not gated300.active(actual)
    torch.manual_seed(119)
    expected = baseline.update()
    rng = torch.get_rng_state()
    torch.manual_seed(119)
    result = actual.update()
    for key in expected.keys() - {'wall_time_s'}:
        fixture.assert_tree_close(expected[key], result[key], atol=0, rtol=0)
    for x, y in zip(baseline.actor_parameters, actual.actor_parameters):
        assert torch.equal(x, y) and torch.equal(x.grad, y.grad)
    fixture.assert_tree_close(baseline.actor_optimizer.state_dict(), actual.actor_optimizer.state_dict(), 0, 0)
    assert torch.equal(rng, torch.get_rng_state())


def test_gain_only_binds_cct2_authority_and_credits_predicted_gains(tmp_path):
    t = trainer()
    original = t.reward_fn
    def flipped(*args):  # Treated rows then predict gain instead of harm.
        out = original(*args)
        return {**out, 'reward': -out['reward']}
    t.reward_fn = flipped
    path = cct2_authority(tmp_path, 120)
    receipt = enable_gain_only(t, path)
    assert receipt['H_G'] == 120 and receipt['H_long'] == 120 and gated300.active(t)
    assert receipt['artifact_sha256'] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    t.config = replace(t.config, horizon=90)
    t._training_imagination = True
    torch.manual_seed(29)
    result = t.update()
    assert result['gated300_rows_on_share'] > 0 and result['gated300_long_credit'] != 0
    assert result['gated300_horizon_s'] == 120


def test_gain_only_binding_refuses_wrong_wm_panels_and_tampered_h_g(tmp_path):
    t = trainer()
    for extra, message in [({'wm_sha256': 'b' * 64}, 'authority'),
                           ({'panels': ['pairs300']}, 'authority'),
                           ({'policy_authorized': False}, 'authority'),
                           ({'schema': 'gated300_cct2/2'}, 'authority'),
                           ({'H_G': 300}, 'largest passing'),
                           ({'entrant_control': True}, 'entrant')]:
        with pytest.raises(ValueError, match=message):
            enable_gain_only(t, cct2_authority(tmp_path, 120, **extra))
    with pytest.raises(ValueError, match='mode'):
        gated300.install(t, dict(enabled=True, mode='both'), ['a' * 64])
    enable_gain_only(t, cct2_authority(tmp_path))
    t.world_model_ids = ['different']
    with pytest.raises(ValueError, match='WM changed'):
        gated300.active(t)


@pytest.mark.parametrize('mode', ['absent', 'off', 'missing', 'hc0', 'hc90'])
def test_default_off_and_untrusted_are_bitwise_identical_updates(tmp_path, mode):
    baseline, actual = trainer(), trainer()
    if mode == 'off':
        gated300.install(actual, dict(enabled=False), ['a' * 64])
    elif mode == 'missing':
        enable(actual, str(tmp_path / 'missing.json'))
    elif mode.startswith('hc'):
        enable(actual, authority(tmp_path, int(mode[2:])))
    torch.manual_seed(119)
    expected = baseline.update()
    rng = torch.get_rng_state()
    torch.manual_seed(119)
    result = actual.update()
    for key in expected.keys() - {'wall_time_s'}:
        fixture.assert_tree_close(expected[key], result[key], atol=0, rtol=0)
    for x, y in zip(baseline.actor_parameters, actual.actor_parameters):
        assert torch.equal(x, y) and torch.equal(x.grad, y.grad)
    fixture.assert_tree_close(baseline.actor_optimizer.state_dict(), actual.actor_optimizer.state_dict(), 0, 0)
    assert torch.equal(rng, torch.get_rng_state())


def test_binding_refuses_wrong_wm_informational_and_tampered_prefix(tmp_path):
    t = trainer()
    for extra, message in [({'wm_sha256': 'b' * 64}, 'authority'),
                           ({'panel': 'calibration'}, 'authority'),
                           ({'H_C': 300}, 'prefix'),
                           ({'entrant_control': True}, 'entrant')]:
        with pytest.raises(ValueError, match=message):
            enable(t, authority(tmp_path, **extra))
    enable(t, authority(tmp_path))
    t.world_model_ids = ['different']
    with pytest.raises(ValueError, match='WM changed'):
        gated300.active(t)


@pytest.mark.parametrize('horizon', [120, 300])
def test_actual_prefix_reused_no_second_rollout_and_no_critic(tmp_path, horizon):
    t = trainer()
    t.config = replace(t.config, horizon=90)
    enable(t, authority(tmp_path, horizon))
    calls = []
    original = t.world_models[0].rollout
    def counted(*args, **kwargs):
        calls.append(kwargs['n_steps'])
        return original(*args, **kwargs)
    t.world_models[0].rollout = counted
    t.target_critic.forward = lambda *a: pytest.fail('long credit called a value function')
    t._training_imagination = True
    batch = t.start_states.sample(1, 'cpu')
    variants = build_variants(batch.state, t.counterfactual)
    with torch.no_grad():
        data = t.imagine(batch, variants=variants, future_seed=11)
    assert len(calls) == 8 and calls == [2 * horizon] * 8
    assert data['rewards'].shape[1] == 90 and data['gated300_horizon_s'] == horizon
    torch.testing.assert_close(data['returns'], data['rewards'].flip((1,)).cumsum(1).flip((1,)))
    assert data['gated300_long_difference'].shape == (8, 1, variants.n_rows)
    t._training_imagination = False


def test_equivalent_six_row_gate_loss_and_gradients():
    # Cancellation of individual rows occurs BEFORE nonlinear gating. Coop/team
    # coefficients have the identical 49.7/-10.3 representation in either preset.
    state = fixture.scene(1, 3)
    full = CounterfactualConfig(enabled=True, max_starts=1, n_individual=0, n_group=1,
        n_flip=3, group_min_size=3, group_max_size=3, curriculum_updates=[0],
        w_individual=[8.9], w_coop=[8.9], w_group=[9.9], w_team=[10.3], w_all=[1.])
    reduced = replace(full, cover_group_members=False, w_individual=[0.], w_coop=[0.], w_group=[18.8])
    parameter = torch.tensor(1., requires_grad=True)
    losses, grads = [], []
    for config in (full, reduced):
        v = build_variants(state, config, torch.Generator().manual_seed(4))
        # A physical row's return depends only on acting slots, not storage index.
        values = (v.act_mask.float() * torch.tensor([100., 200., 300.])).sum(-1)
        samples = parameter * values.expand(8, 1, -1)
        samples = samples - samples[..., [v.base_row[int(s)] for s in v.row_start]].detach()
        coeff = row_coefficients(v, config.weights(0), v.n_rows, 1, device='cpu', dtype=samples.dtype)
        owner = SimpleNamespace(gated300=dict(config=dict(w_long=1.)))
        loss, _ = gated300.actor_term(owner, dict(n_base=1, variants=v,
            gated300_long_difference=samples, gated300_horizon_s=120), coeff, samples, 1.)
        losses.append(loss.detach())
        grads.append(torch.autograd.grad(loss, parameter)[0])
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-6, atol=1e-4)


def test_longmem_exact_full_k_gate_update_and_resume(tmp_path, monkeypatch):
    # Short recurrent graph for exhaustive replay/resume equality; the independent
    # continuation test above exercises the real 240-physics-step geometry.
    monkeypatch.setattr(gated300, 'STEP_S', 30.)
    baseline, memory = trainer(), trainer(memory=True)
    baseline.config = replace(baseline.config, horizon=2)
    memory.config = replace(memory.config, horizon=2)
    path = authority(tmp_path)
    enable(baseline, path)
    enable(memory, path)
    torch.manual_seed(29)
    expected = baseline.update()
    assert expected['gated300_rows_on_share'] > 0
    assert expected['gated300_long_credit'] != 0
    expected_rng = torch.get_rng_state()
    torch.manual_seed(29)
    result = memory.update()
    for key in ('actor_loss', 'critic_loss', 'actor_grad_norm', 'gated300_long_credit',
                'gated300_rows_on_share', 'gated300_long_credit_share'):
        fixture.assert_tree_close(expected[key], result[key])
    for x, y in zip(baseline.actor_parameters, memory.actor_parameters):
        fixture.assert_tree_close(x, y)
        fixture.assert_tree_close(x.grad, y.grad)
    assert torch.equal(expected_rng, torch.get_rng_state())
    checkpoint = tmp_path / 'resume.pt'
    memory.save_checkpoint(checkpoint)
    resumed = trainer(memory=True)
    resumed.load_checkpoint(checkpoint)
    assert resumed.gated300 == memory.gated300
    rng = _rng_state(memory.device)
    left = memory.update()
    _restore_rng(rng, resumed.device)
    right = resumed.update()
    for key in ('actor_loss', 'actor_grad_norm', 'gated300_long_credit', 'returns'):
        assert left[key] == right[key]
    for x, y in zip(memory.actor_parameters, resumed.actor_parameters):
        assert torch.equal(x, y)
    fixture.assert_tree_close(memory.actor_optimizer.state_dict(), resumed.actor_optimizer.state_dict(), 0, 0)


def test_matched_scoring_resolves_horizon_and_restores_flags_rng(tmp_path):
    t = trainer()
    path = authority(tmp_path)
    enable(t, path, score_training_horizon=True)
    t.horizon_curriculum = SimpleNamespace(horizons_s=[10, 90], stage=lambda p: 1)
    original = t.reward_fn
    def reward(*a):
        r = original(*a)
        return {**r, 'source_wait': r['reward'] * 0, 'hdv_hard_brake_seconds': r['reward'] * 0}
    t.reward_fn = reward
    batch = t.start_states.sample(1, 'cpu')
    rng = torch.get_rng_state()
    result = independent_score(t, batch, native_reference=True)
    assert result['score_horizon_s'] == 90 and result['long_horizon_s'] == 120
    assert result['J'] == result['J_dense'] - result['long_gated_gain']
    assert result['native_relative_gain'] == pytest.approx(result['native_J'] - result['J'])
    assert not t._training_imagination and not t._gated300_scoring
    assert torch.equal(rng, torch.get_rng_state())
    enable(t, path, score_training_horizon=False)
    legacy = independent_score(t, batch)
    assert 'long_horizon_s' not in legacy and 'score_horizon_s' not in legacy


def test_looplong_configure_keeps_separate_long_horizon_config(tmp_path, monkeypatch):
    from wms.control.looplong import configure
    from wms.control.coopw import HorizonConfig
    t = trainer()
    t.update_count = 180
    path = authority(tmp_path)
    monkeypatch.setattr(HorizonConfig, 'bind', lambda self, wm: setattr(
        self, 'allowed_stage', len(self.horizons_s) - 1))
    configure(t, dict(flags=dict(auto_horizon=True), horizon_s=90, previous_horizon_s=90,
                      gate_file='synthetic_gate.json', wm=dict(sha256='a' * 64)),
              dict(long_credit=dict(enabled=True, cct1_file=path)), tmp_path)
    assert t.gated300['H_long'] == 120
    assert gated300.rollout_config(t, t.config).horizon == 5  # scoring/training not active
    t._training_imagination = True
    assert t._imagination_config(t.start_states.batch).horizon == 90
    assert gated300.rollout_config(t, t.config).horizon == 120


def test_resume_binding_change_refused_and_explicit_off_takes_precedence(tmp_path):
    original = trainer()
    path = authority(tmp_path)
    enable(original, path)
    checkpoint = tmp_path / 'bound.pt'
    original.save_checkpoint(checkpoint)
    doc = json.loads(Path(path).read_text())
    doc['wm_sha256'] = 'b' * 64
    Path(path).write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='authority'):
        trainer().load_checkpoint(checkpoint)
    off = trainer()
    off._gated300_explicit = True
    gated300.install(off, {'enabled': False}, ['a' * 64])
    off.load_checkpoint(checkpoint)
    assert not gated300.active(off)


def test_production_residual_actor_k8_entrants_full_modules_cpu(tmp_path, monkeypatch):
    from functools import partial
    from wms.control.bridge import make_cav_controller
    from wms.control.dreamer import DreamerConfig
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.policy14 import Policy14Trainer
    from wms.model.entries import EntranceHistory
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    monkeypatch.setattr(gated300, 'STEP_S', 30.)
    torch.manual_seed(71)
    starts = fixture.Starts(1, 4)
    starts.batch.state['is_cav'][:, -1] = False
    hist = EntranceHistory.defaults()
    hist.rate[:] = .5
    starts.batch.entry_feat = torch.tensor(hist.to_features()).float()[None]
    cfg = DreamerConfig(batch_size=1, horizon=2, imagine_entries=True, entry_pad_max=2,
        source_wait_grad=True, entrant_control=True, native_residual_train_group=True,
        native_residual_train_trunk=True, future_samples=8, pessimistic_credit=True,
        deterministic_crn=True, imagination_microbatch=1, imagination_checkpoint_segment=3,
        imagination_offload_futures=True, imagination_offload_saved_tensors=True)
    model = WmSModel(ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid',
        grad_decay_per_step=.9, lane_model='hazard_v5', merge_decision='first_passage',
        lane_decision='sample', merge_passage_law='hybrid'))
    policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True)
    with torch.no_grad():
        policy.residual_head.weight.normal_(std=.01)
    def synthetic_reward(*args):
        out = fixture.queue_reward(*args)
        return {**out, 'reward': out['reward'] * 1e6}
    cf = trainer().counterfactual
    cf.native_cav = 'head'
    t = Policy14Trainer([model], policy, RsuGroupEncoder(hidden_dim=8, n_heads=2),
        fixture.Critic(), starts, observation_builder=build_observations,
        reward_fn=synthetic_reward, safety=lambda state, command, **kw: command,
        controller_factory=partial(make_cav_controller, native_lane_change=True,
            native_ramp_merge='wm_law', symmetric_exec_cap=True), config=cfg, counterfactual=cf)
    path = authority(tmp_path, entrant_control=True)
    enable(t, path)
    result = t.update()
    assert result['gated300_rows_on_share'] > 0 and result['gated300_long_credit'] != 0
    assert result['entries'] and result['critic_loss'] == 0
    assert all(value > 0 for value in result['actor_module_grad_norms'].values())
    assert torch.isfinite(torch.tensor(result['actor_loss']))
