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
