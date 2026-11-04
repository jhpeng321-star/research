"""T2G contracts using synthetic sealed bank records and paired CPU WM stubs.

No checkpoint, bank file, physics process, remote dispatch or GPU is used.
"""
import copy
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from . import exact_grade as xg
from . import mppi_search as ms
from . import plan_augment as pa
from . import verify_perseed as vp
from .protocol import digest


def registration(**changes):
    # Synthetic TEST registration, not proposed production hyperparameters.
    c = dict(enabled=True, seed=42, attempts_per_plan=6, parameter_fraction=.05,
             partner_swap_probability=1., spot_check_fraction=0., wm_samples=4,
             horizon_s=300, bounds=(-1.2, 1.2), variety={})
    c.update(changes)
    return c


def seal(plan, arm='B', admitted=True, cooperative=True):
    ex = dict(plan=copy.deepcopy(plan), plan_id=plan['id'], seed_index=1,
              admitted=admitted, cooperative=cooperative)
    return xg._seal(dict(arm=arm, t0=420, verify_mode=pa.ALL8, bank_rule='per_seed_v1',
                         per_seed_v1=dict(examples=[ex])), 'confirmation_sha256')


def source(arm='B'):
    rows = [dict(vid=v, is_cav=True, s_m=-40.-i*5, v_mps=20., lane=lane,
                 allowed_speed_mps=30., observed_at_s=419.)
            for i, (v, lane) in enumerate([('a', 0), ('b', 1), ('c', 1)])]
    rows.append(dict(rows[-1], vid='human', is_cav=False))
    P = np.zeros((2, 120))
    profile = P.copy()
    profile[:, 10:50] = -.4
    profile[:, 50:55] = .2
    plan = ms.build_plan(dict(id='g', members=['a', 'b']), P, None, 120,
                         (-1.2, 1.2), 'sampled', 'test', 'SOURCE', absolute=profile)
    plan['variety_family'] = 'wall'
    return dict(confirmation=seal(plan, arm), example_index=0,
                parameters=dict(onset_s=10., hold_s=40., depth=.6),
                context=dict(ego_id='a', t_s=420., rows=rows,
                             policy_traj={v: [0.]*120 for v in ('a', 'b', 'c')}))


class WM:
    def __init__(self, group=80., solo=90., bad=None):
        self.group, self.solo, self.bad, self.calls = group, solo, bad, []

    def __call__(self, context, plans, *, samples, horizon_s):
        self.calls.append(copy.deepcopy(dict(context=context, plans=plans, samples=samples,
                                             horizon_s=horizon_s)))
        result = {}
        for i, p in enumerate(plans):
            cost = 100. if i == 0 else self.group if i == 1 else self.solo
            result[p['id']] = (cost+np.arange(samples)).tolist()
        if self.bad == 'missing':
            result.pop(plans[1]['id'])
        elif self.bad == 'nan':
            result[plans[1]['id']][0] = float('nan')
        elif self.bad == 'short':
            result[plans[1]['id']].pop()
        return result


def run(src=None, arm='B', wm=None, **changes):
    return pa.augment([src or source(arm)], arm=arm, registration=registration(**changes),
                      score=wm or WM())


def assert_sealed(batch):
    assert batch['receipt_sha256'] == digest({k: v for k, v in batch.items() if k != 'receipt_sha256'})


@pytest.mark.parametrize('cfg', [None, {}, {'enabled': False}])
def test_default_off_never_consumes_source_or_calls_scorer(cfg):
    def forbidden_sources():
        raise AssertionError('off must not consume sources')
        yield
    score = Mock(side_effect=AssertionError('off must not score'))
    batch = pa.augment(forbidden_sources(), arm='B', registration=cfg, score=score)
    assert not batch['enabled'] and not batch['sumo_dispatched']
    assert batch['examples'] == batch['spot_checks'] == []
    score.assert_not_called()
    assert_sealed(batch)


@pytest.mark.parametrize('changes', [dict(extra_key=True), dict(wm_samples=1),
    dict(wm_samples=4.5), dict(seed=True), dict(attempts_per_plan=0),
    dict(parameter_fraction=0), dict(parameter_fraction=1.1),
    dict(partner_swap_probability=-.1), dict(spot_check_fraction=float('nan')),
    dict(horizon_s=0), dict(bounds=(-1.2, 1.21)), dict(bounds=(1.2, -1.2)),
    dict(variety={'variety_made_up': 1}),
    dict(variety={'variety_hold_window_s': (90, 20)}), dict(enabled='yes')])
def test_invalid_registration_refused(changes):
    with pytest.raises(ValueError):
        pa.config(registration(**changes))


def test_enabled_requires_every_registration_setting_and_scorer():
    with pytest.raises(ValueError, match='registered keys'):
        pa.config(dict(enabled=True))
    with pytest.raises(ValueError, match='scorer'):
        pa.augment([], arm='B', registration=registration())
    with pytest.raises(ValueError, match='B and R'):
        pa.augment([], arm='A')


def test_small_registered_perturbations_are_executable_and_source_is_untouched():
    src, wm = source(), WM()
    before = copy.deepcopy(src)
    batch = run(src, wm=wm)
    assert src == before
    assert len(batch['examples']) == 6
    for ex in batch['examples']:
        p = ex['plan']
        assert p['members'] == ['a', 'c']  # one same-lane visible partner swap
        assert p['duration_s'] == 120 and p['variety_family'] == 'wall'
        ms.validate_plan(p, (-1.2, 1.2))
        for key, window in pa.WINDOWS.items():
            lo, hi = batch['registration']['variety'][window]
            assert lo <= ex['parameters'][key] <= hi
            assert abs(ex['parameters'][key]-src['parameters'][key]) <= .05*(hi-lo)+1e-12
        assert ex['wm']['group_gain'] == 20
        assert ex['wm']['synergy'] == 10
        assert ex['wm']['horizon_s'] == 300
        assert not ex['sumo_verified'] and ex['verification'] == 'wm_only'
        assert p['profile'] != src['confirmation']['per_seed_v1']['examples'][0]['plan']['profile']
    assert len({e['id'] for e in batch['examples']}) == len(batch['examples'])
    assert_sealed(batch)
    assert json.loads(json.dumps(batch))['examples'] == batch['examples']


def test_scorer_receives_real_base_group_and_each_solo_on_full_horizon():
    wm = WM()
    run(wm=wm, attempts_per_plan=1)
    call, = wm.calls
    assert call['horizon_s'] == 300 and call['samples'] == 4
    base, group, solo0, solo1 = call['plans']
    assert np.count_nonzero(base['profile']) == 0
    np.testing.assert_array_equal(solo0['profile'][0], group['profile'][0])
    np.testing.assert_array_equal(solo1['profile'][1], group['profile'][1])
    assert not np.any(solo0['profile'][1]) and not np.any(solo1['profile'][0])
    assert set(call['context']) == {'ego_id', 't_s', 'rows', 'policy_traj'}


@pytest.mark.parametrize('group,solo', [(100., 110.), (101., 120.), (80., 80.), (80., 70.)])
def test_no_gain_or_no_synergy_is_rejected(group, solo):
    batch = run(wm=WM(group, solo))
    assert batch['examples'] == [] and batch['counts']['wm_kept'] == 0
    assert batch['counts']['wm_scored'] > 0


@pytest.mark.parametrize('bad', ['missing', 'nan', 'short'])
def test_incomplete_or_nonfinite_wm_predictions_fail_closed(bad):
    assert not run(wm=WM(bad=bad))['examples']


def test_best_solo_is_by_expected_cost_not_samplewise_or_first_solo():
    def score(context, plans, **kw):
        return {p['id']: values for p, values in zip(plans, [
            [100]*4, [80]*4, [90]*4, [60, 60, 110, 110]])}
    batch = run(wm=score)
    assert batch['examples'][0]['wm']['synergy'] == 5.
    assert batch['examples'][0]['wm']['best_solo_gain'] == 15.


@pytest.mark.parametrize('admitted,cooperative', [(False, True), (True, False)])
def test_only_verified_cooperative_sources(admitted, cooperative):
    src = source()
    p = src['confirmation']['per_seed_v1']['examples'][0]['plan']
    src['confirmation'] = seal(p, admitted=admitted, cooperative=cooperative)
    with pytest.raises(ValueError, match='admitted cooperative'):
        run(src)


def test_source_receipt_tampering_and_cross_arm_bank_refused():
    src = source()
    src['confirmation']['arm'] = 'R'
    with pytest.raises(ValueError, match='mismatch'):
        run(src)
    with pytest.raises(ValueError, match='own bank'):
        run(source('R'))


@pytest.mark.parametrize('mutation', ['future', 'missing_member', 'wrong_time', 'wrong_centre', 'bad_parameters'])
def test_observable_and_source_contract_enforced(mutation):
    src = source()
    if mutation == 'future':
        src['context']['rows'][1]['observed_at_s'] = 420
    elif mutation == 'missing_member':
        src['context']['rows'][1]['is_cav'] = False
    elif mutation == 'wrong_time':
        src['context']['t_s'] = 421
    elif mutation == 'wrong_centre':
        src['context']['policy_traj']['a'][0] = .5
    else:
        src['parameters']['depth'] = 10.
    with pytest.raises(ValueError):
        run(src)


def test_never_shorten_registered_lookahead():
    with pytest.raises(ValueError, match='look-ahead'):
        run(horizon_s=119)


def test_extra_fields_not_forwarded_and_hdv_not_commanded():
    src, wm = source(), WM()
    src['context']['private_unobservable_extra'] = 'not a model input'
    src['context']['rows'][0]['private_unobservable_extra'] = 'not a model input'
    batch = run(src, wm=wm)
    for ex in batch['examples']:
        assert 'human' not in ex['plan']['members']
    for call in wm.calls:
        assert 'private_unobservable_extra' not in call['context']
        assert all('private_unobservable_extra' not in r for r in call['context']['rows'])


def test_no_partner_swap_keeps_ego_and_members():
    batch = run(partner_swap_probability=0.)
    assert all(ex['plan']['members'] == ['a', 'b'] for ex in batch['examples'])


def test_B_R_identical_procedure_own_banks_and_repeatable():
    b, r = run(), run(arm='R')
    assert [x['plan'] for x in b['examples']] == [x['plan'] for x in r['examples']]
    assert [x['parameters'] for x in b['examples']] == [x['parameters'] for x in r['examples']]
    assert b['counts'] == r['counts']
    assert run() == b
    assert all(x['arm'] == 'R' for x in r['examples'])


def test_spot_fraction_uses_separate_rng_and_pending_not_trainable():
    base = run()
    batch = run(spot_check_fraction=.25)
    assert len(batch['spot_checks']) == 2  # ceil(.25 * 6)
    assert len(batch['examples']) == 4
    assert {x['id'] for x in batch['examples']+batch['spot_checks']} == {x['id'] for x in base['examples']}
    assert all(x['verification'] == 'pending_all8' for x in batch['spot_checks'])
    assert not batch['sumo_dispatched']
    assert batch == run(spot_check_fraction=.25)


def test_resolve_accepts_only_actual_admitted_cooperative_matching_plan():
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    original = copy.deepcopy(batch)
    checker = Mock(side_effect=lambda request: seal(request['plan']))
    result = pa.resolve_spot_checks(batch, checker)
    assert batch == original
    checker.assert_called_once()
    assert not result['spot_checks']
    ex, = result['examples']
    assert ex['sumo_verified'] and ex['verification'] == 'all8'
    assert ex['verified_example']['plan'] == ex['plan']
    assert result['counts']['sumo_verified_examples'] == 1
    assert_sealed(result)


@pytest.mark.parametrize('kind', ['not_cooperative', 'not_admitted', 'wrong_plan'])
def test_failed_spot_check_is_dropped(kind):
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    def checker(request):
        p = copy.deepcopy(request['plan'])
        if kind == 'wrong_plan':
            p['id'] = 'another plan'
        return seal(p, admitted=kind != 'not_admitted', cooperative=kind != 'not_cooperative')
    result = pa.resolve_spot_checks(batch, checker)
    assert result['examples'] == [] and result['spot_checks'] == []
    assert result['spot_check_results'][0]['accepted_examples'] == 0


def test_spot_check_cross_arm_and_tampered_confirmation_refused():
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    with pytest.raises(ValueError, match='arm mismatch'):
        pa.resolve_spot_checks(batch, lambda req: seal(req['plan'], arm='R'))
    def tampered(req):
        rec = seal(req['plan'])
        rec['arm'] = 'R'
        return rec
    with pytest.raises(ValueError, match='mismatch'):
        pa.resolve_spot_checks(batch, tampered)
    batch['spot_checks'][0]['plan']['id'] = 'edited'
    with pytest.raises(ValueError, match='identity mismatch'):
        pa.resolve_spot_checks(batch, Mock())


def test_spot_check_quote_includes_both_seeds_later_harm_and_solos():
    request = run(attempts_per_plan=1, spot_check_fraction=1.)['spot_checks'][0]
    quote = pa.spot_check_quote(request, 420)
    assert quote['all8_seed2_sim_s'] > 0
    assert quote['all8_extended_sim_s'] > 0
    assert quote['all8_solo_sim_s'] > 0
    assert quote['all8_total_sim_s'] > quote['first_seed_sim_s']


@pytest.mark.parametrize('arm,proposer', [('B', 'wm'), ('R', 'random')])
def test_explicit_adapter_delegates_to_existing_charged_all8_machinery(arm, proposer):
    req = run(arm=arm, attempts_per_plan=1, spot_check_fraction=1.)['spot_checks'][0]
    snap = {'t0': 420}
    runner, health = Mock(), Mock()
    first, confirmed = {'first': 'receipt'}, {'confirmed': 'receipt'}
    with patch.object(vp, '_verify', return_value=first) as verify, \
         patch.object(vp, 'confirm_all8', return_value=confirmed) as confirm:
        result = pa.verify_all8(req, snapshot=snap, root='synthetic-root', run_root='single-ledger',
                                runner_for=runner, phase='registered-augment', health=health)
    assert result == confirmed
    args = verify.call_args.args
    assert args[0] == 'augment_first_all8' and args[2] == [req['plan']]
    assert args[5:8] == ('single-ledger', 'registered-augment', proposer)
    assert args[-2:] == (arm, health)
    assert confirm.call_args.args[2:4] == (first, snap)
    assert confirm.call_args.kwargs['later_harm_seed1'] is True
    assert confirm.call_args.kwargs['later_harm_seed2'] is True
    assert confirm.call_args.kwargs['candidates'] == [req['plan']]
    runner.assert_not_called()  # no runner/physics in the CPU test


def test_adapter_wrong_snapshot_time_refused_before_dispatch():
    req = run(attempts_per_plan=1, spot_check_fraction=1.)['spot_checks'][0]
    with patch.object(vp, '_verify') as verify:
        with pytest.raises(ValueError, match='snapshot time'):
            pa.verify_all8(req, snapshot={'t0': 600}, root='synthetic-root', run_root='single-ledger',
                          runner_for=Mock(), phase='registered-augment')
    verify.assert_not_called()


def test_duplicate_source_does_not_double_exposure_for_fixed_parameter_windows():
    src = source()
    c = registration(attempts_per_plan=3, variety={
        'variety_onset_window_s': (10., 10.), 'variety_hold_window_s': (40., 40.),
        'variety_depth_window': (.6, .6)})
    batch = pa.augment([src, copy.deepcopy(src)], arm='B', registration=c, score=WM())
    assert len(batch['examples']) == 1  # same physical partner replacement every time
    assert batch['counts']['wm_scored'] == 1


def test_feasibility_erased_perturbation_never_reaches_scorer():
    src = source()
    plan = src['confirmation']['per_seed_v1']['examples'][0]['plan']
    # At rest, an entirely braking residual issues as exactly the base zero action.
    for row in src['context']['rows']:
        row['v_mps'] = 0.
    profile = np.full((2, 120), -.4)
    plan = ms.build_plan(dict(id='g', members=['a', 'b']), np.zeros((2, 120)), None,
                         120, (-1.2, 1.2), 'sampled', 'test', absolute=profile)
    src['confirmation'] = seal(plan)
    wm = Mock(side_effect=AssertionError('erased perturbation must not be scored'))
    batch = run(src, wm=wm)
    assert not batch['examples'] and batch['counts']['wm_scored'] == 0


def test_partner_uses_its_own_base_policy_not_replaced_members_base():
    src, wm = source(), WM()
    src['context']['policy_traj']['c'] = [.5]*120
    run(src, wm=wm, attempts_per_plan=1)
    call, = wm.calls
    base, group = call['plans'][:2]
    assert group['members'] == ['a', 'c']
    assert base['profile'][1][:20] == [.5]*20
    assert base['profile'][1][20:] == [0.]*100  # inherited allowed-speed feasibility clamp
    assert 0 < group['profile'][1][15] < .5  # own base plus source braking residual


def test_resolved_batch_is_idempotent_and_never_checks_twice():
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    result = pa.resolve_spot_checks(batch, lambda req: seal(req['plan']))
    checker = Mock(side_effect=AssertionError('no repeated checks'))
    assert pa.resolve_spot_checks(result, checker) == result
    assert result['sumo_dispatched']
    checker.assert_not_called()


def test_unchecked_examples_are_not_relabelled_when_spot_checks_complete():
    batch = run(spot_check_fraction=.25)
    wm_only = copy.deepcopy(batch['examples'])
    result = pa.resolve_spot_checks(batch, lambda req: seal(req['plan']))
    assert result['examples'][:len(wm_only)] == wm_only
    assert result['counts']['sumo_verified_examples'] == 2
    assert len(result['examples']) == 6


def test_scorer_mutation_cannot_rewrite_plan_or_source():
    src = source()
    before = copy.deepcopy(src)
    wm = WM()
    def mutating(context, plans, **kwargs):
        result = wm(context, plans, **kwargs)
        context['policy_traj']['a'][0] = 999
        plans[1]['profile'][0][0] = 999
        return result
    batch = run(src, wm=mutating, attempts_per_plan=1)
    assert src == before
    assert batch['examples'][0]['plan']['profile'][0][0] != 999
