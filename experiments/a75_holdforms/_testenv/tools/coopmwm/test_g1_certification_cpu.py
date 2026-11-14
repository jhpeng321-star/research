from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tools.coopmwm import g1_certification as cert, moves
from tools.coopmwm.acceleration import Policy


def rows():
    return [dict(vid=v, s_m=s, v_mps=5., lane=lane, is_cav=cav)
        for v, s, lane, cav in [('ramp', moves._ZONE_S0+30., moves._RAMP_LANE, True),
            ('main', moves._ZONE_S0+35., moves._MAIN_LANE, True),
            ('hdv', moves._ZONE_S0+33., moves._MAIN_LANE, False),
            ('remote', moves._ZONE_S0+900., moves._MAIN_LANE, True)]]


def macro():
    # Registered minimum hold/onset/depth, not test-specific behavioural bounds.
    return moves.Move('ramp_assist', ('main', 'ramp'), int(moves._ONSET_LO),
        int(moves._HOLD_LO), moves._DEPTH_LO)


def test_keyed_crn_is_population_and_order_invariant_and_seed_distinct():
    original = cert.innovation('a'*64, 310, ['main', 'ramp'])
    expanded = cert.innovation('a'*64, 310, ['new', 'ramp', 'main'])
    assert {v: expanded[v] for v in original} == original
    assert original != cert.innovation('b'*64, 310, list(original))
    assert original != cert.innovation('a'*64, 311, list(original))


def test_local_view_is_causal_sensor_only_and_never_includes_hdv_as_actor():
    frame = dict(time_s=300, vehicles=rows())
    local = cert.local_view(frame, macro(), 301, 4)
    assert local['observed_at_s'] == 300
    assert 'hdv' not in local['acquirable_cav_ids']
    assert 'remote' not in {r['vid'] for r in local['rows']}
    with pytest.raises(ValueError, match='exactly one second'):
        cert.local_view(frame, macro(), 302, 4)
    with pytest.raises(ValueError, match='causal local view'):
        cert.local_view(frame, macro(), 301, 1)


@pytest.mark.parametrize('selected', [[], ['main'], ['ramp'], ['main', 'ramp']])
def test_fixed_macro_changes_only_selected_cav_raw_requests_and_keeps_prefix(selected):
    move = macro()
    def apply(rs, commands, now):
        desired = {r['vid']: dict(requested_raw=.1, acceleration_residual=.1)
            for r in rs if r['is_cav']}
        controller.events.append(dict(time_s=now, requests=desired))
        return desired
    controller = SimpleNamespace(policy=Policy(), events=[], apply=apply)
    spec = dict(macro=move.to_dict(), selected_actors=selected, decision_s=301,
        seed=dict(rng_sha256='a'*64))
    cert.install_macro(controller, spec, dict(t0=300))
    commands = {r['vid']: 0. for r in rows() if r['is_cav']}
    prefix = controller.apply(rows(), commands, 300)
    assert all(v['requested_raw'] == .1 for v in prefix.values())
    actual = controller.apply(rows(), commands, 301+move.onset)
    for v in commands:
        if v in selected:
            assert actual[v]['requested_raw'] == pytest.approx(.1-move.depth)
            assert actual[v]['without_library_raw'] == .1
        else:
            assert actual[v] == prefix[v]
    assert 'hdv' not in actual
    released = controller.apply(rows(), commands, 301+move.onset+move.hold)
    assert all(v['requested_raw'] == .1 for v in released.values())


def test_keyed_policy_keeps_native_stochastic_law_and_does_not_advance_shared_rng():
    base = Policy()
    keyed = cert.KeyedPolicy(base, 'a'*64)
    keyed.now = 310
    context = dict(members=['main', 'ramp'], x=[[0.]*29]*2, rsu=[0.]*12)
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    context_before = deepcopy(context)
    sample, memory = keyed.sample(context, generator, False, {'main': .2})
    assert torch.equal(before, generator.get_state())
    assert context == context_before
    assert sample['own_action'] != [[0.], [0.]]
    assert set(memory) == {'main', 'ramp'}
    subset = dict(context, members=['ramp'], x=[[0.]*29])
    other, _ = keyed.sample(subset, generator, False, {})
    assert sample['g1_crn']['innovations']['ramp'] == other['g1_crn']['innovations']['ramp']
    assert sample['own_action'][1] == other['own_action'][0]
    with pytest.raises(ValueError, match='stochastic'):
        keyed.sample(context, generator, True)
