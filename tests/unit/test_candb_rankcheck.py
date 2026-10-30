"""Ranking gate pairing, controller bridges and fail-closed launch binding."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.candB import rankcheck as rank


def test_world_bootstrap_does_not_treat_futures_or_repeated_starts_as_worlds():
    rows = []
    for world, value in [('w963100m2400p20t300', 10.), ('w963100m2400p20t300', 10.),
                         ('w963101m2400p20t300', 30.)]:
        rows.append(dict(metadata=[dict(world=world)], arms={arm: dict(
            J90=[value]*8, J300=[value]*8, objective_loss=value,
            dense_credit=value, long_credit=value) for arm in rank.ARMS}))
    summary, worlds = rank.summarize(rows)
    assert worlds == 2
    np.testing.assert_array_equal(summary['J300']['arms']['mean'], [20.]*len(rank.ARMS))
    np.testing.assert_array_equal(summary['J300']['vs_CTRL0']['hi'], np.zeros(len(rank.ARMS)))


def test_owner_r2_gate_and_lane_on_evidence_cannot_substitute():
    n = len(rank.ARMS)
    summary = dict(J300=dict(vs_CTRL0=dict(hi=[1.]*n)))
    assert rank.ranking_gate(summary)[0] is False
    evidence = dict(measured_real_sumo=True, arms={'PI': dict(
        reference='CTRL0', native_lanes=True, upper95_dJ=-1.)})
    summary['J300']['vs_CTRL0']['hi'][rank.ARMS.index('PI')] = -.1
    assert rank.ranking_gate(summary, evidence)[0] is True
    evidence['arms']['PI']['native_lanes'] = False
    assert rank.ranking_gate(summary, evidence)[0] is False
    summary['J300']['vs_CTRL0']['hi'][rank.ARMS.index('MAPPO_native_lanes')] = 0.
    assert rank.ranking_gate(summary)[0] is True


def test_receipt_refuses_mix_config_runtime_and_source_drift(monkeypatch):
    source = 'src/wms/control/policy.py'
    monkeypatch.setattr(rank, 'runtime_sources', lambda: [source])
    raw = dict(trainer=dict(entrant_learned_mix=True))
    doc = dict(schema=rank.SCHEMA, complete=True, passed=True, n_starts=16, K=8,
        horizons_s=[90, 300], mix_connected=True, config_sha256=rank.binding(raw),
        wm_sha256='wm', wm_config_sha256='cfg', source_manifest={source: rank.sha256(source)},
        runtime_contract=dict(mix=True))
    assert rank.verify_receipt(doc, raw, wm='wm', wm_config='cfg') is doc
    with pytest.raises(ValueError, match='mismatched'):
        rank.verify_receipt({**doc, 'measure_only': True}, raw, wm='wm', wm_config='cfg')
    with pytest.raises(ValueError, match='mismatched'):
        rank.verify_receipt(doc, dict(trainer=dict(entrant_learned_mix=False)), wm='wm', wm_config='cfg')
    monkeypatch.setattr(rank, 'runtime_contract', lambda trainer: dict(mix=False))
    with pytest.raises(ValueError, match='trained configuration'):
        rank.verify_receipt(doc, raw, wm='wm', wm_config='cfg', trainer=object())
    doc['source_manifest'][source] = 'changed'
    with pytest.raises(ValueError, match='source manifest mismatch'):
        rank.verify_receipt(doc, raw, wm='wm', wm_config='cfg')


def test_measurement_uses_requested_loss_but_cannot_authorize_training(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    authority = tmp_path / 'old_cct2.json'
    authority.write_text('{"policy_authorized": false}')
    trainer = SimpleNamespace(config=SimpleNamespace(entrant_learned_mix=True,
        future_samples=8, pessimistic_credit=True, deterministic_crn=True),
        policy=SimpleNamespace(residual_reference='ctrl0'), world_model_ids=['wm'])
    section = dict(enabled=True, mode='gain_only', cct2_file=str(authority),
        ctrl0_reference=True, require_objective_match=True, estimator='antithetic')
    receipt = rank.install_measurement(trainer, section)
    assert receipt['H_long'] == 300 and receipt['config']['enabled'] is True
    assert receipt['measure_only'] is True and receipt['policy_authorized'] is False
    assert rank.gated300.uses_ctrl0(trainer)
    assert receipt['artifact_sha256'] == rank.sha256(authority)
    trainer.config.entrant_learned_mix = False
    with pytest.raises(ValueError, match='identity'):
        rank.install_measurement(trainer, section)


def test_checkpoint_driver_normalization_does_not_allow_parameter_drift(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    current, original = tmp_path/'driver.json', tmp_path/'original.json'
    current.write_bytes(b'{\n"parameter": 1\n}\n')
    original.write_bytes(current.read_bytes().replace(b'\n', b'\r\n'))
    monkeypatch.setenv('C4_EXECUTION_DRIVER', str(current))
    monkeypatch.setattr(rank, 'load_policy_components', lambda *a, **k:
        rank.os.environ['C4_EXECUTION_DRIVER'])
    assert rank.load_initial_components(current, 'cpu', original) == str(original)
    assert rank.os.environ['C4_EXECUTION_DRIVER'] == str(current)
    original.write_bytes(b'{\r\n"parameter": 2\r\n}\r\n')
    with pytest.raises(ValueError, match='beyond CRLF'):
        rank.load_initial_components(current, 'cpu', original)


@pytest.mark.parametrize('arm', rank.ARMS[:5])
def test_native_lane_comparators_use_live_bridge_and_exact_ctrl0(arm, monkeypatch):
    from wms.control.policy14 import SmokeSampler
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.lanes import LaneRules
    from wms.model import WmSModel, ModelSpec
    from wms.model.cav import EXEC_CAP_KEY, WM_LANE_TARGET_KEY
    from wms.control.alwayscmd import executed
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    monkeypatch.setenv('C4_SUMO_LEADER_RULE', '1')
    model = WmSModel(ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid'))
    state = SmokeSampler(model).sample(1).state
    state[EXEC_CAP_KEY] = torch.full_like(state['s_m'], 1.2)
    state[WM_LANE_TARGET_KEY] = state['lane'].clone()
    teacher = SimpleNamespace(policy=CavPolicy(), group_encoder=RsuGroupEncoder(), lane_rules=LaneRules())
    policy = SimpleNamespace(ctrl0_rows=torch.ones(1, dtype=torch.bool))
    callback = rank.comparator_factory(arm, model, teacher)(policy, None, None,
        act_mask=state['is_cav'], native=lambda st: torch.full_like(st['s_m'], .2))
    with torch.no_grad():
        out = callback(state)
    assert torch.isfinite(out).all()
    assert torch.equal(out[..., 1], state['lane'].float())
    assert bool(callback.native_lane_mask(state).any())
    if arm == 'CTRL0':
        expected = executed(state, torch.zeros_like(state['s_m']), residual_reference='ctrl0')
        mask = state['present'] & state['is_cav']
        torch.testing.assert_close(out[..., 3][mask], expected[mask], atol=2e-6, rtol=0)


def test_rankcheck_refuses_small_panels_before_loading_assets():
    with pytest.raises(SystemExit):
        rank.main(['--n-starts', '15', *[part for key in (
            'wm', 'wm-config', 'mappo', 'current-policy', 'cct2', 'horizon-gate',
            'boundary', 'pairing-receipt', 'out') for part in ('--'+key, 'unused')]])


@pytest.mark.parametrize('count', [8, 4])
def test_comparators_flow_through_actual_k8_actor_objective(tmp_path, monkeypatch, count):
    from functools import partial
    from tests.unit.test_gated300 import cct2_authority, enable_gain_only
    from tests.unit.test_longmem import Starts, Critic, queue_reward
    from wms.control.bridge import make_cav_controller
    from wms.control.dreamer import DreamerConfig
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.policy14 import Policy14Trainer
    from wms.control.counterfactual import CounterfactualConfig, build_variants, KIND_BASE
    from wms.control.train_cli import GroupSignal
    from wms.control.lanes import LaneRules
    from wms.model import WmSModel, ModelSpec
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    monkeypatch.setenv('C4_SUMO_LEADER_RULE', '1')
    monkeypatch.setenv('C4_EXECUTION_DRIVER', 'configs/execparity_cav_idm_v1.json')
    monkeypatch.setattr(rank.gated300, 'STEP_S', 60.)
    torch.set_num_threads(1)
    model = WmSModel(ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid',
        lane_model='hazard_v5', merge_decision='first_passage', lane_decision='sample',
        merge_passage_law='hybrid'))
    starts = Starts(1, 4)
    policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False,
                       residual_on_native=True, residual_reference='ctrl0')
    trainer = Policy14Trainer([model], policy, GroupSignal(RsuGroupEncoder(hidden_dim=8, n_heads=2)),
        Critic(), starts, observation_builder=rank.build_observations, reward_fn=queue_reward,
        safety=lambda st, command, **kw: command,
        controller_factory=partial(make_cav_controller, native_lane_change=True,
                                  native_ramp_merge='wm_law', symmetric_exec_cap=True),
        config=DreamerConfig(horizon=1, cav_exec_cap='idm', future_samples=8,
                            pessimistic_credit=True, deterministic_crn=True),
        counterfactual=CounterfactualConfig(enabled=True, max_starts=1, native_cav='head'))
    enable_gain_only(trainer, cct2_authority(tmp_path, 300), ctrl0_reference=True,
                     estimator='antithetic')
    trainer._training_imagination = True
    teacher = SimpleNamespace(policy=CavPolicy(hidden_dim=8, n_heads=2),
        group_encoder=RsuGroupEncoder(hidden_dim=8, n_heads=2), lane_rules=LaneRules())
    for arm in rank.ARMS[:5]:
        trainer.controller_factory = rank.comparator_factory(arm, model, teacher)
        with torch.no_grad(), rank.measurement_prefix(count):
            image = trainer.imagine(starts.batch, variants=build_variants(
                starts.batch.state, trainer.counterfactual), deterministic=True, _memory_replay=True)
            loss, _ = trainer.actor_objective(image, 1., trainer.counterfactual.weights(0))
            image['rankcheck_J90'] = image['rankcheck_J300'] = -image['rewards'].sum(1)
            zero = rank.ctrl0_image(image)
            zero_loss, _ = trainer.actor_objective(zero, 1., trainer.counterfactual.weights(0))
        assert image['n_futures'] == count and torch.isfinite(loss)
        assert torch.isfinite(zero_loss)
        if arm == 'CTRL0':
            assert torch.equal(image['returns'], zero['returns'])
            assert torch.equal(loss, zero_loss)
            treatment = image['variants'].row_kind != KIND_BASE
            assert not image['gated300_long_difference'][..., treatment].any()


def test_k4_plans_are_prefix_of_original_k8_and_context_restores(monkeypatch):
    from wms.control import dreamer
    from wms.control.futures import FuturesConfig, SampledFutures
    cfg = FuturesConfig(n_samples=8, driver_noise_scale=0, accel_noise=True)
    state = {'s_m': torch.zeros(1, 3)}
    model = SimpleNamespace()
    original = SampledFutures(cfg, 191).plan([model], state, {}, 10)[0]
    factory = dreamer.SampledFutures
    with rank.measurement_prefix(4):
        reduced = dreamer.SampledFutures(
            rank.replace(cfg, n_samples=4), 191).plan([model], state, {}, 10)[0]
        assert len(reduced) == 4
        for left, right in zip(original, reduced):
            assert torch.equal(left['accel_noise'], right['accel_noise'])
    assert dreamer.SampledFutures is factory


@pytest.mark.parametrize('controller_id', [1, 2])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_host_sensor_frames_preserve_published_controller_commands(controller_id, device, monkeypatch):
    from wms.train.cav_drive import TimelineControllers
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA comparison runs on the scoring host')
    monkeypatch.setenv('WMS_SPEED_FAST_CAV_DRIVE', '1')
    model = SimpleNamespace(physics=SimpleNamespace(dt_s=.5, speed_limit_mps=31.29),
        physics_accel=lambda st: torch.full_like(st['v_mps'], 1.2))
    ids = torch.full((2, 12, 6), controller_id, dtype=torch.long)
    timeline = dict(controller=ids, acceleration_mps2=torch.zeros_like(ids).float(),
                    target_speed_mps=torch.zeros_like(ids).float())
    direct = TimelineControllers(model, timeline)
    copied = rank.host_sensor_timeline(model, timeline)
    state = dict(s_m=torch.tensor([[0., 25., 70., 115., 150., 180.]]*2, device=device),
        v_mps=torch.tensor([[10., 9., 12., 14., 8., 13.]]*2, device=device),
        lane=torch.zeros(2, 6, dtype=torch.long, device=device),
        present=torch.ones(2, 6, dtype=torch.bool, device=device),
        is_cav=torch.ones(2, 6, dtype=torch.bool, device=device),
        length_m=torch.full((2, 6), 5., device=device),
        a_mps2=torch.zeros(2, 6, device=device))
    for tick in range(12):
        if tick == 6:
            state['present'][:, 1] = False
        original, optimized = direct(state), copied(state)
        assert torch.equal(original, optimized), (controller_id, device, tick)
        state['s_m'], state['v_mps'], state['a_mps2'] = (
            original[..., 0], original[..., 2], original[..., 3])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_measurement_pair_bias_chunking_preserves_full_attention_and_restores(device):
    from wms.model.residual_net import HdvConfig, _KinAttention
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA comparison runs on scoring host')
    torch.manual_seed(965140)
    cfg = HdvConfig()
    module = _KinAttention(cfg).to(device).eval()
    values = torch.randn(2, 37, cfg.d_model, device=device)
    mask = torch.ones(2, 37, dtype=torch.bool, device=device)
    geo = dict(s=torch.randn(2, 37, device=device)*50,
               lane=torch.randint(0, 3, (2, 37), device=device),
               v=torch.rand(2, 37, device=device)*25)
    original = module.bias.forward
    with torch.no_grad():
        dense = module(values, mask, geo)
        with rank.measurement_bias_rows(module, 8):
            chunked = module(values, mask, geo)
        for expected, actual in zip(dense, chunked):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert module.bias.forward == original
    with pytest.raises(ValueError, match='no-grad'):
        with rank.measurement_bias_rows(module, 8):
            module(values, mask, geo)
    assert module.bias.forward == original
