"""CPU-only: real WM newborn wiring, CRN identity and segmented replay safety."""
import copy
from dataclasses import replace

import pytest
import torch

from tests.unit.test_keyerr import make_trainer
from tools.errsrc.inference import LearnedMix
from wms.control.dreamer import DreamerConfig
from wms.control.entrant_mix import make_sampler
from wms.control.imagination_memory import segmented_rollout
from wms.model.driver_encoder import PARAMETER_NAMES


class Births:
    def __init__(self):
        self.tick = 0

    def insert(self, state):
        newborn = torch.zeros_like(state['present'])
        if self.tick in (1, 3):
            newborn[:, 3] = True
            state['present'][:, 3] = True
            state['is_cav'][:, 3] = False
            state['lane'][:, 3] = 0
            state['s_m'][:, 3] = -50.
            state['v_mps'][:, 3] = 6.
        self.tick += 1
        return newborn


def inputs(monkeypatch):
    t = make_trainer(monkeypatch, exits=False, memory=False)
    model = t.world_models[0]
    state = copy.deepcopy(t.start_states.batch.state)
    state['present'][:, 3] = False
    _, embedding, params = model.encode_drivers(state)
    observed = torch.zeros((*state['present'].shape, 2), dtype=torch.bool)
    observed[:, :2] = True
    # Distinct observable pools vs unobserved padding. No driver truth or type file.
    params = {k: v.clone() for k, v in params.items()}
    for k in params:
        params[k][:, 0] *= .9
        params[k][:, 1] *= 1.1
    return model, state, params, embedding, observed


def test_real_rollout_assigns_birth_settings_before_first_action_and_persists(monkeypatch):
    model, state, params, embedding, observed = inputs(monkeypatch)
    mix = LearnedMix(params, observed, 71, [0])
    seen = []
    step = model.step

    def record(current, **kwargs):
        seen.append({k: v.clone() for k, v in kwargs['driver_params'].items()})
        return step(current, **kwargs)

    monkeypatch.setattr(model, 'step', record)
    with torch.no_grad():
        result = model.rollout(state, n_steps=4, driver_params=params,
            driver_embedding=embedding, entries=Births(), entrant_driver_sampler=mix, lane_seed=9)
    assert mix.assigned == [2]
    for name in params:
        assert torch.equal(seen[0][name], params[name])
        assert torch.equal(seen[1][name][:, :3], params[name][:, :3])
        assert torch.equal(seen[1][name][:, 3], seen[2][name][:, 3])
        assert seen[1][name][0, 3] in params[name][0, :2]
    expected = torch.stack([seen[-1][k] for k in PARAMETER_NAMES], -1)
    assert torch.equal(result['driver_parameters'], expected)


def test_default_off_is_exact_and_consumes_no_additional_rng(monkeypatch):
    assert DreamerConfig().entrant_learned_mix is False
    model, state, params, embedding, _ = inputs(monkeypatch)
    results, rngs = [], []
    for kwargs in ({}, {'entrant_driver_sampler': None}):
        torch.manual_seed(99)
        with torch.no_grad():
            results.append(model.rollout(copy.deepcopy(state), n_steps=4,
                driver_params=params, driver_embedding=embedding, entries=Births(), **kwargs))
        rngs.append(torch.get_rng_state())
    for k in ('positions', 'speeds', 'lanes', 'masks', 'accel'):
        assert torch.equal(results[0][k], results[1][k])
    assert torch.equal(*rngs)
    with pytest.raises(ValueError, match='boolean'):
        DreamerConfig(entrant_learned_mix=1)


def test_mix_keys_pair_rows_despite_different_birth_ticks_and_preserve_rng():
    params = {'v0': torch.tensor([[20., 30., 1.], [20., 30., 1.]])}
    observed = torch.tensor([[[1], [1], [0]], [[1], [1], [0]]], dtype=torch.bool)
    mix = LearnedMix(params, observed, 31, [0, 0])
    rng = torch.get_rng_state()
    state = {'lane': torch.zeros(2, 3, dtype=torch.long)}
    first = mix(state, torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.bool), params)
    second = mix(state, torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.bool), first)
    assert first['v0'][0, 2] == second['v0'][1, 2]
    assert torch.equal(rng, torch.get_rng_state())


def test_make_sampler_requires_observed_history():
    with pytest.raises(ValueError, match='pre-t0'):
        make_sampler(None, {}, {}, 0, [0])
    with pytest.raises(ValueError, match='observed pre-t0 inferred'):
        make_sampler(None, {}, {'driver_params': {'v0': torch.ones(1, 2)},
            'history_mask': torch.zeros(1, 2, 2, dtype=torch.bool)}, 0, [0])


def test_segmented_mix_forward_and_gradients_match_unsegmented(monkeypatch):
    outputs, gradients, counts = [], [], []
    for segment in (0, 2):
        model, state, params, embedding, observed = inputs(monkeypatch)
        state['v_mps'] = state['v_mps'].clone().requires_grad_()
        mix = LearnedMix(params, observed, 71, [0])
        kwargs = dict(driver_params=params, driver_embedding=embedding, entries=Births(),
                      entrant_driver_sampler=mix,
                      lane_uniform=torch.full((1, 4, 4), .5))
        if segment:
            out = segmented_rollout(model, state, 4, segment, None, kwargs)
        else:
            out = model.rollout(state, n_steps=4, **kwargs)
        out['speeds'].sum().backward()
        outputs.append(out['speeds'].detach())
        gradients.append(state['v_mps'].grad)
        counts.append(mix.assigned)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    torch.testing.assert_close(*gradients, rtol=1e-5, atol=1e-5)
    assert counts == [[2], [2]]


def test_mix_cannot_reuse_an_unqualified_cct2_authority(tmp_path):
    from tests.unit.test_gated300 import trainer, enable_gain_only, cct2_authority
    t = trainer()
    t.config = replace(t.config, entrant_learned_mix=True)
    with pytest.raises(ValueError, match='mix-qualified'):
        enable_gain_only(t, cct2_authority(tmp_path))
    receipt = enable_gain_only(t, cct2_authority(tmp_path, learned_mix_qualified=True))
    assert receipt['H_long'] == 120


def test_mix_signature_is_explicit_and_preset_does_not_change_existing_launch():
    from wms.control.antiexploit import inference_signature
    from wms.control.train_cli import _read_config
    assert 'entrant_learned_mix' not in inference_signature(DreamerConfig())
    assert inference_signature(DreamerConfig(entrant_learned_mix=True))['entrant_learned_mix']
    existing = _read_config('configs/c4_policy14_candA_longgrad_skip.toml')
    optin = _read_config('configs/c4_policy14_candA_setupaudit.toml')
    assert existing['trainer'].get('entrant_learned_mix', False) is False
    assert existing['long_credit']['score_training_horizon'] is False
    assert optin['trainer']['entrant_learned_mix'] is True
    assert optin['long_credit']['score_training_horizon'] is True
    assert optin['long_credit']['require_objective_match'] is True
    assert optin['run']['gpu_memory_fraction'] <= .4


def test_audit_native_background_matches_candidate_bridge_bitwise(monkeypatch):
    from tools.setupaudit.measure import native_background_motion
    from wms.control.bridge import make_cav_controller
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.model.cav import WM_LANE_TARGET_KEY, EXEC_CAP_KEY
    from tests.unit.test_zerostart import scene
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    state = scene()
    state[WM_LANE_TARGET_KEY] = state['lane'].clone()
    state[EXEC_CAP_KEY] = torch.full_like(state['v_mps'], .3)
    def native(st):
        return torch.full_like(st['v_mps'], .361234567)
    controller = make_cav_controller(
        CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True),
        RsuGroupEncoder(hidden_dim=8, n_heads=2), lambda st, a, **kw: a,
        native=native, native_lane_change=True, native_ramp_merge='wm_law',
        act_mask=torch.zeros_like(state['present']))
    with torch.no_grad():
        expected = controller(state)
        result = native_background_motion(state, expected[..., 1], native)
    mask = state['present'] & state['is_cav'] & (state['lane'] >= 0)
    assert torch.equal(result[mask], expected[mask])


def test_objective_guard_is_off_by_default_and_refuses_population_k_mismatch(tmp_path):
    from tests.unit.test_gated300 import trainer, enable_gain_only, cct2_authority
    from wms.control.gated300 import objective_contract
    t = trainer()
    path = cct2_authority(tmp_path)
    legacy = enable_gain_only(t, path)
    assert 'require_objective_match' not in legacy['config']
    refused = enable_gain_only(t, path, require_objective_match=True)
    assert refused['H_long'] == 0 and refused['reason'] == 'unqualified objective contract'
    contract = objective_contract(t)
    path = cct2_authority(tmp_path, objective_contract={**contract, 'futures': 1})
    assert enable_gain_only(t, path, require_objective_match=True)['H_long'] == 0
    path = cct2_authority(tmp_path, objective_contract=contract)
    assert enable_gain_only(t, path, require_objective_match=True)['H_long'] == 120


def test_cct2_builder_never_pools_mismatched_objective_populations():
    from tools.gated300.cct2 import compute
    contract = {'population': 'all_present_and_future_requested', 'futures': 8}
    pairs = dict(role='pairs300', wm_sha256='a' * 64, rows=[], objective_contract=contract)
    validation = dict(role='validation', wm_sha256='a' * 64, rows=[])
    with pytest.raises(ValueError, match='objective contract'):
        compute(pairs, validation)
    validation['objective_contract'] = contract
    assert compute(pairs, validation)['objective_contract'] == contract
