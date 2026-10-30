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


def test_comparators_flow_through_actual_k8_actor_objective(tmp_path, monkeypatch):
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
        with torch.no_grad():
            image = trainer.imagine(starts.batch, variants=build_variants(
                starts.batch.state, trainer.counterfactual), deterministic=True, _memory_replay=True)
            loss, _ = trainer.actor_objective(image, 1., trainer.counterfactual.weights(0))
        assert image['n_futures'] == 8 and torch.isfinite(loss)
        if arm == 'CTRL0':
            treatment = image['variants'].row_kind != KIND_BASE
            assert not image['gated300_long_difference'][..., treatment].any()
