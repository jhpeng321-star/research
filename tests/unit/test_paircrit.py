"""Offline metastability, paired-target, hazard and production-wiring contracts."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.breakdown import (BreakdownHead, calibrate, event_labels,
    event_reliability, fit_head, fit_otsu_rule, merge_speed, reliability_diagram, risk_loss)
from wms.control.contract import MERGE_S0_M
from wms.control.paircrit import (BreakdownConfig, PairCritic, PairCriticConfig,
    PairedProvider, TailValue, breakdown_gate, install, wm_event_frequencies)
from wms.control.real_returns14 import ScaledTeamCritic


def state(speed):
    v = torch.as_tensor(speed, dtype=torch.float32).reshape(-1, 1)
    return {'s_m': torch.full_like(v, MERGE_S0_M + 1), 'v_mps': v,
        'a_mps2': torch.zeros_like(v), 'length_m': torch.full_like(v, 5),
        'lane': torch.ones_like(v, dtype=torch.long), 'present': torch.ones_like(v, dtype=torch.bool),
        'is_cav': torch.ones_like(v, dtype=torch.bool), 'entry_backlog': torch.zeros(len(v), 2)}


def test_off_install_preserves_objects_and_rng():
    torch.manual_seed(17)
    trainer = SimpleNamespace(critic=ScaledTeamCritic())
    original, rng = trainer.critic, torch.get_rng_state().clone()
    install(trainer, PairCriticConfig(), BreakdownConfig(), [])
    assert trainer.critic is original and torch.equal(rng, torch.get_rng_state())
    assert not PairCriticConfig().enabled and not BreakdownConfig().enabled


def test_native_rule_is_deterministic_and_events_censor_initially_low_starts():
    native = np.r_[np.linspace(.1, 2., 40), np.linspace(18., 25., 40)]
    a, b = fit_otsu_rule(native, 'native'), fit_otsu_rule(native[::-1], 'native')
    assert a == b
    rule = {'boundary_mps': a['boundary_mps']}
    times = np.array([0, 60, 120, 300])
    v = np.array([[20, 1, 20], [20, 1, np.nan], [1, 1, 20], [1, 1, 20]])
    y, m, initial = event_labels(v, times, rule)
    assert y[0].tolist() == [0, 1, 1] and m[0].all()
    assert initial.tolist() == [False, True, False]
    assert not m[1].any() and not m[2].any()
    # Empty sensors are missing observations, never speed zero/breakdown.
    empty = state([0.])
    empty['present'][:] = False
    assert torch.isnan(merge_speed(empty)).all()


def test_near_identical_metastable_states_pairing_and_head_recover_order():
    torch.manual_seed(20261026)
    torch.set_num_threads(2)
    # An adversarial finite sample of bimodal futures: good happened to collapse,
    # bad happened not to. The latent future is unobservable at the saved start.
    action = np.r_[-np.ones(16), np.ones(16)]
    x = np.c_[np.ones(32), action]
    u = np.r_[np.full(16, .2), np.full(16, .8)]
    native = 40000 * (u < .5)
    policy = 40000 * (u < (.5 + .1 * action)) + 1000 * action
    absolute = np.linalg.lstsq(x, policy, rcond=None)[0]
    paired = np.linalg.lstsq(x, policy - native, rcond=None)[0]
    assert absolute[1] < 0 < paired[1]  # absolute ranks bad action as better
    # Same observable state up to 0.002 m/s; no hidden outcome is a model input.
    states = state(20. + .001 * action)
    head = BreakdownHead()
    future_u = (np.arange(1000) + .5) / 1000
    native_p = np.array([.2, .3, .5])
    policy_p = native_p + .1 * action[:, None]
    labels = np.stack([np.broadcast_to(native_p, (32, 3)), policy_p], 1)
    # Frequencies explicitly come from K shared draws, not independent arms.
    frequencies = (future_u[:, None, None, None] < labels[None]).mean(0)
    fit_head(head, states, torch.tensor(frequencies).float(), torch.ones(32, 2, 3, dtype=torch.bool), updates=200)
    with torch.no_grad():
        prediction = head(states)
    assert torch.all(prediction[:16, 1] < prediction[16:, 1])
    assert torch.all((prediction[:16, 1] - prediction[:16, 0]) < 0)
    assert torch.all((prediction[16:, 1] - prediction[16:, 0]) > 0)
    assert (prediction.diff(dim=-1) >= 0).all()
    assert float((prediction - torch.tensor(frequencies)).square().mean()) < .002
    # Exercise the actual advantage module and differentiable actor tail, using
    # the learned observable risk feature. The zero absolute backbone stays zero.
    critic = TailValue(ScaledTeamCritic(value_scale=1000.), paired=True, risk=head)
    critic.base.requires_grad_(False)
    optimizer = torch.optim.Adam(critic.projection.parameters(), lr=.02)
    target = torch.tensor(policy - native)
    for _ in range(200):
        loss = ((critic.cost_advantage(states) - target) / 1000.).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    fitted = critic.cost_advantage(states)
    assert fitted[:16].mean() < 0 < fitted[16:].mean()
    probe = state([20.])
    probe['v_mps'].requires_grad_()
    gradient, = torch.autograd.grad(-critic(probe).sum(), probe['v_mps'])
    assert gradient.item() > 0  # actor descent chooses the lower-risk action


def test_paired_value_has_correct_reward_sign_and_horizon_separation():
    value = TailValue(ScaledTeamCritic(entry_backlog=True), paired=True)
    with torch.no_grad():
        value.base.head[-1].bias.copy_(torch.tensor([2., 4., 8.]))
    s = state([10., 20.])
    assert torch.equal(value(s), torch.tensor([-8., -8.], dtype=torch.float64))
    assert torch.equal(value.cost_advantage(s, 60), torch.tensor([2., 2.], dtype=torch.float64))
    value.zero_grad()
    value.cost_advantage(s, 60).sum().backward()
    assert value.base.head[-1].bias.grad.tolist() == [2., 0., 0.]


def test_risk_uses_only_observable_sensors_and_has_calibrated_monotonic_outputs():
    head = BreakdownHead()
    a = state([10., 20.])
    b = {**a, 'entry_backlog': torch.full((2, 2), 1e6), 'driver_truth': torch.ones(2, 1)}
    assert torch.equal(head(a), head(b))
    y = torch.tensor([[[0., 1., 1.], [1., 1., 1.]], [[0., 0., 1.], [0., 1., 1.]]])
    calibrate(head, a, y, torch.ones_like(y))
    assert (head(a).diff(dim=-1) >= 0).all()
    diag = reliability_diagram(head(a).detach().numpy(), y.numpy(), np.ones(y.shape, bool))
    assert sum(b['n'] for b in diag['bins']) == y.numel()


def test_wm_event_frequencies_share_rollouts_and_missing_reliability_is_zero():
    states = [state([20. if t < 100 else 1., 20. if t < 300 else 1.])
              for t in range(301)]
    imagined = {'states': [states, copy.deepcopy(states)], 'imagination_horizon_s': 300}
    freq, counts = wm_event_frequencies(imagined, {'boundary_mps': 5.})
    # Native first, policy second; event at100s and300s for policy/native.
    assert freq.tolist() == [[[0., 0., 1.], [0., 1., 1.]]]
    assert counts.min() == 2
    assert event_reliability({}, ['wm'], {}) == [0., 0., 0.]
    head = BreakdownHead()
    target = torch.ones(1, 2, 3)
    loss = risk_loss(head, state([20.]), target, torch.ones_like(target), [0., 0., 0.])
    loss.backward()
    assert all(p.grad is None or not p.grad.any() for p in head.parameters())


def test_production_real_wm_paired_update_checkpoint_and_native_zero_tail(tmp_path):
    spec = importlib.util.spec_from_file_location('longtests', Path(__file__).with_name('test_longcritic.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer, model, _ = module._real_wm_trainer()
    trainer.critic = TailValue(trainer.critic, paired=True)
    trainer.target_critic = copy.deepcopy(trainer.critic).requires_grad_(False)
    trainer.critic_optimizer = torch.optim.Adam(trainer.critic.parameters(), lr=.001)
    class Pairs:
        sha256 = 'synthetic'
        def sample(self, size, device, update):
            return trainer.start_states.sample(size, device).state, torch.full((size,), 8., device=device)
    trainer.paircrit = PairCritic(trainer, PairCriticConfig(enabled=True), Pairs(), ['synthetic'])
    trainer.real_return_provider = lambda **kw: pytest.fail('absolute labels leaked into paired fit')
    trainer.paircrit.warmup(trainer, 1)
    metrics = trainer.update()
    assert np.isfinite(metrics['critic_loss']) and 'paircrit_real_mse' in metrics
    assert all(p.grad is None for p in model.parameters())
    checkpoint = tmp_path / 'paired.pt'
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload['paircrit']['absolute_baseline'] is False
    trainer.load_checkpoint(checkpoint)
    # Enabled real WM CRNs use the inherited no-grad target engine; native tail zero.
    trainer.paircrit.engine.weights = {60: .4}
    target, _ = trainer.paircrit.engine.targets(trainer, trainer.start_states.sample(1))
    assert target is not None and not target[1].requires_grad
    assert 'paircrit_wm_loss' in trainer.update()
    result = trainer.imagine(trainer.start_states.sample(1))
    assert result['returns'].shape[-1] >= 1
    from wms.control.counterfactual import build_variants, KIND_BASE
    batch = trainer.start_states.sample(1)
    variants = build_variants(batch.state, trainer.counterfactual)
    result = trainer.imagine(batch, variants=variants)
    assert not result['values'][..., variants.row_kind == KIND_BASE].any()


def test_real_pair_contract_and_world_partition():
    # The offline real export is optional in a source-only checkout.
    root = Path('/root/autodl-tmp/c4/paircrit20261026')
    if not (root / 'PAIRING.json').exists():
        pytest.skip('real TRAIN export absent')
    from wms.control.real_returns14 import NPZRealReturnProvider
    policy, native = [NPZRealReturnProvider(root / f'{name}_returns.npz') for name in ('policy', 'native')]
    provider = PairedProvider(policy, native, root / 'PAIRING.json', policy.provenance['policy_checkpoint'])
    assert provider.fit_indices.numel() == 20 and len(provider.development_seeds) == 3
    with pytest.raises(ValueError, match='bound current-policy'):
        PairedProvider(policy, native, root / 'PAIRING.json', 'wrong')
    native._states['v_mps'][0, 0] += 1
    with pytest.raises(ValueError, match='states differ'):
        PairedProvider(policy, native, root / 'PAIRING.json', policy.provenance['policy_checkpoint'])


def test_breakdown_gate_has_zero_native_relative_margin():
    class Risk:
        def __call__(self, states):
            return torch.tensor([[[.1, .2, .3], [.1, .2, .4]]])
    assert not breakdown_gate(Risk(), {})['passed']


def test_risk_feature_training_and_deployment_checkpoint_roundtrip(tmp_path):
    from wms.control.breakdown import rule_sha256
    from wms.control.policy import load_policy_components, PolicyExecutor
    from wms.control.obs import build_observations
    spec = importlib.util.spec_from_file_location('longtests', Path(__file__).with_name('test_longcritic.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer, _, _ = module._real_wm_trainer()
    head = BreakdownHead()
    rule = fit_otsu_rule(np.r_[np.arange(10.), np.arange(20., 30.)], 'synthetic')
    head.boundary_mps.fill_(rule['boundary_mps'])
    doc = {'schema': 'paircrit_risk/1', 'split': 'TRAIN', 'head': head.state_dict(),
        'rule': rule, 'rule_sha256': rule_sha256(rule), 'fit_worlds': [968129],
        'calibration_worlds': [968130], 'policy_checkpoint_sha256': 'synthetic',
        'calibration_qualified': False}
    head_file = tmp_path / 'risk.pt'
    torch.save(doc, head_file)
    install(trainer, PairCriticConfig(), BreakdownConfig(enabled=True, checkpoint=str(head_file)), [])
    original = copy.deepcopy(trainer.group_encoder.risk.state_dict())
    trainer.update()
    for key, value in original.items():
        assert torch.allclose(value, trainer.group_encoder.risk.state_dict()[key], equal_nan=True)
    checkpoint = tmp_path / 'feature.pt'
    trainer.save_checkpoint(checkpoint)
    policy, group, rules, _ = load_policy_components(checkpoint)
    s = trainer.start_states.sample(1).state
    obs = build_observations(s)
    assert torch.equal(group(obs, s), trainer.group_encoder(obs, s))
    exe = PolicyExecutor(policy, group, lane_rules=rules)
    assert np.isfinite(exe(s)[np.isfinite(exe(s))]).all()


def test_loop_command_pairing_flags_and_default_off_identity(tmp_path):
    spec = importlib.util.spec_from_file_location('fasttests', Path(__file__).with_name('test_fastloop.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.fixture(tmp_path)
    loop = module.loop
    state = loop.load_state(cfg)
    before = loop.policy_cmd(cfg, state, tmp_path)
    cfg['policy']['paircrit'] = False
    assert loop.policy_cmd(cfg, state, tmp_path) == before
    cfg['policy']['paircrit'] = True
    state['native_real_returns'] = 'native.npz'
    with pytest.raises(loop.Refused, match='current paired cohort'):
        loop.policy_cmd(cfg, state, tmp_path)
    state['pairing_receipt'] = 'pairing.json'
    argv, _ = loop.policy_cmd(cfg, state, tmp_path)
    assert argv[argv.index('--paircrit-receipt') + 1] == 'pairing.json'
    assert argv[argv.index('--paircrit-policy-sha256') + 1] == state['policy']['sha256']


def test_real_pair_exporter_independently_verifies_saved_start_arrays(tmp_path):
    from wms.control.paircrit_export import export_receipt
    root = Path('/root/autodl-tmp/c4/paircrit20261026')
    if not (root / 'policy_returns.receipt.json').exists():
        pytest.skip('real TRAIN receipt absent')
    doc = export_receipt('/root/main14_loop/data/pairs_c00.a1', root / 'policy_returns.npz',
                         root / 'native_returns.npz', tmp_path / 'pairing.json')
    assert len(doc['pairs']) == 26


def test_paired_overlap_binds_both_current_cohort_arms(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('fasttests', Path(__file__).with_name('test_fastloop.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg, loop, fast = module.fixture(tmp_path), module.loop, module.fast
    state = loop.load_state(cfg)
    cfg['policy']['paircrit'] = True
    with pytest.raises(loop.Refused, match='completed current-policy'):
        fast.overlap(loop, cfg, state, Path(cfg['loop_root']))
    converted = {}
    for key in ('returns', 'native_returns', 'pairing_receipt'):
        path = tmp_path / (key + '.fixture')
        path.write_bytes(key.encode())
        converted[key] = str(path)
    state['cycles'][-1]['phases']['convert']['receipt'] = converted
    seen = {}
    def run(cfg, snapshot, ph, root):
        seen.update(snapshot)
        return {}
    monkeypatch.setattr(loop, 'run_policy', run)
    doc = fast.overlap(loop, cfg, state, Path(cfg['loop_root']))
    assert seen['real_returns'] == converted['returns']
    assert seen['native_real_returns'] == converted['native_returns']
    assert seen['pairing_receipt'] == converted['pairing_receipt']
    assert doc['dependencies']['pairing_receipt_sha256'] == loop.sha256(converted['pairing_receipt'])


def test_deployment_veto_requires_candidate_bound_qualified_risk(tmp_path):
    from wms.control.nativeinit import panel_gate, verify_collection_gate
    checkpoint = tmp_path / 'policy.pt'
    checkpoint.write_bytes(b'synthetic')
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = {**panel_gate([1.] * 16, [2.] * 16, update=1), 'checkpoint_sha256': sha}
    path = checkpoint.with_name('native_gate_result.json')
    path.write_text(json.dumps(receipt))
    policy = {'checkpoint': str(checkpoint), 'breakdown_required': True}
    with pytest.raises(ValueError, match='missing required breakdown'):
        verify_collection_gate(policy)
    receipt['breakdown'] = {'horizons_s': [60, 120, 300], 'policy_minus_native': [[0.] * 3] * 16,
        'passed': True, 'calibration_qualified': True, 'rule_sha256': 'synthetic',
        'source_policy_sha256': 'another_actor'}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='breakdown native-relative'):
        verify_collection_gate(policy)
    receipt['breakdown']['source_policy_sha256'] = sha
    path.write_text(json.dumps(receipt))
    assert verify_collection_gate(policy)['passed']
