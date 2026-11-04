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
from . import replay_bank as rb
from . import sumo_verify as sv
from .protocol import digest


def registration(**changes):
    # Synthetic TEST registration, not proposed production hyperparameters.
    c = dict(enabled=True, seed=42, attempts_per_plan=6, parameter_fraction=.05,
             partner_swap_probability=1., spot_check_fraction=0., wm_samples=4,
             horizon_s=300, bounds=(-1.2, 1.2), variety={})
    c.update(changes)
    return c


def snapshot(arm='B'):
    # Identity projection only: no simulator, checkpoint or traffic truth is loaded.
    return dict(episode=dict(id=f'train_{arm}_7', arm=arm, world=dict(seed=7)),
                policy=dict(state_dict_sha256='actor-7'), sumo=dict(noise_seed=11),
                t0=420, end_s=1200, rng=dict(generator_state_sha256='rng-7'),
                prefixes={}, samples={}, committed_plans=[])


def first_receipt(plan, snap):
    specs = sv.build_branches(snap, [plan], 'wm' if snap['episode']['arm'] == 'B' else 'random')
    return xg._seal(dict(verification_id=specs[0]['verification_id'],
                         snapshot_sha256=sv.snapshot_sha(snap), episode=snap['episode'],
                         t0=snap['t0'], candidate_ids=[plan['id']], candidate_plan_sha256=[digest(plan)],
                         proposer=specs[0]['proposer'], status='accepted'), 'receipt_sha256')


def seal(plan, arm='B', admitted=True, cooperative=True, snap=None, seeds=(1,)):
    snap = snap or snapshot(arm)
    first = first_receipt(plan, snap)
    vid = first['verification_id']
    examples = []
    for k in seeds:
        examples.append(dict(plan=copy.deepcopy(plan), plan_id=plan['id'], plan_sha256=digest(plan),
            seed_index=k, candidate_index=0, admitted=admitted, cooperative=cooperative,
            advantage_veh_s=10.*k, synergy=dict(lower95_one_sided_veh_s=float(k)),
            tape=dict(branches_dir=f'synthetic/seed{k}/branches', verification_id=vid,
                      seed_index=k, branch=dict(branch='cand1', result_sha256=digest(dict(seed=k))),
                      window=dict(start_s=snap['t0'], end_s=snap['t0']+plan['duration_s']+120))))
    record = dict(arm=arm, t0=snap['t0'], verification_id=vid,
                         verify_mode=pa.ALL8, bank_rule='per_seed_v1',
                         first_seed=dict(verification_id=vid, receipt_sha256=first['receipt_sha256']),
                         seed2=dict(verification_id=vid, salt=vid,
                                    snapshot_sha256=sv.snapshot_sha(rb.replicate(snap, vid))),
                         candidates=[dict(index=0, plan_id=plan['id'], plan_sha256=digest(plan))],
                         per_seed_v1=dict(examples=examples))
    extras = {str(k): dict(seed_index=k, salt=vp.extra_seed_salt(vid, k), candidates=[0],
                          verification_id=vid,
                          snapshot_sha256=sv.snapshot_sha(rb.replicate(snap, vp.extra_seed_salt(vid, k))))
              for k in seeds if k >= 3}
    if extras:
        record.update(extra=extras, extra_seeds=max(seeds)-2)
    return xg._seal(record, 'confirmation_sha256')


def evidence(request, *, snap=None, **kwargs):
    snap = snap or request['verification_provenance']['snapshot_identity']
    arm = kwargs.pop('arm', request['arm'])
    return dict(confirmation=seal(request['plan'], arm=arm, snap=snap, **kwargs),
                first_seed_receipt=first_receipt(request['plan'], snap),
                source_confirmation_sha256=request['source_confirmation_sha256'])


def source(arm='B', duration=120):
    rows = [dict(vid=v, is_cav=True, s_m=-40.-i*5, v_mps=20., lane=lane,
                 allowed_speed_mps=30., observed_at_s=419.)
            for i, (v, lane) in enumerate([('a', 0), ('b', 1), ('c', 1)])]
    rows.append(dict(rows[-1], vid='human', is_cav=False))
    P = np.zeros((2, duration))
    profile = P.copy()
    profile[:, 10:50] = -.4
    profile[:, 50:55] = .2
    plan = ms.build_plan(dict(id='g', members=['a', 'b']), P, None, duration,
                         (-1.2, 1.2), 'sampled', 'test', 'SOURCE', absolute=profile)
    plan['variety_family'] = 'wall'
    return dict(confirmation=seal(plan, arm), example_index=0,
                verification=dict(snapshot_identity=snapshot(arm), first_seed_receipt=first_receipt(plan, snapshot(arm))),
                parameters=dict(onset_s=10., hold_s=40., depth=.6),
                context=dict(ego_id='a', t_s=420., rows=rows,
                             policy_traj={v: [0.]*duration for v in ('a', 'b', 'c')}))


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


@pytest.mark.parametrize('duration,horizon', [(120, 119), (120, 120), (120, 299), (300, 300), (300, 419)])
def test_never_shorten_registered_lookahead(duration, horizon):
    with pytest.raises(ValueError, match='look-ahead'):
        run(source(duration=duration), horizon_s=horizon)


@pytest.mark.parametrize('duration,horizon', [(120, 300), (300, 420), (300, 600)])
def test_authoritative_release_horizon_shared_by_base_group_and_all_solos(duration, horizon):
    wm = WM()
    batch = run(source(duration=duration), wm=wm, horizon_s=horizon, attempts_per_plan=1)
    call, = wm.calls
    assert horizon >= ms.final_horizon(duration)
    assert call['horizon_s'] == batch['examples'][0]['wm']['horizon_s'] == horizon
    assert len(call['plans']) == 4  # one paired call covers base, group and both solos
    assert all(p['duration_s'] == duration for p in call['plans'])


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


def test_visible_hdv_queue_position_and_speed_reach_scorer_without_command_eligibility():
    src, before, after = source(), WM(), WM()
    run(src, wm=before, attempts_per_plan=1)
    src['context']['rows'][-1].update(s_m=-41., v_mps=0., lane=1)
    run(src, wm=after, attempts_per_plan=1)
    assert before.calls[0]['context'] != after.calls[0]['context']
    human, = [r for r in after.calls[0]['context']['rows'] if not r['is_cav']]
    assert human['s_m'] == -41. and human['v_mps'] == 0. and human['lane'] == 1
    assert 'human' not in after.calls[0]['context']['policy_traj']
    for call in before.calls+after.calls:
        assert all('human' not in p['members'] for p in call['plans'])


@pytest.mark.parametrize('time', [420., 421., float('nan')])
def test_hdv_observation_timestamps_are_validated(time):
    src = source()
    src['context']['rows'][-1]['observed_at_s'] = time
    with pytest.raises(ValueError):
        run(src)


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
    checker = Mock(side_effect=evidence)
    result = pa.resolve_spot_checks(batch, checker)
    assert batch == original
    checker.assert_called_once()
    assert not result['spot_checks']
    ex, = result['examples']
    assert ex['sumo_verified'] and ex['verification'] == 'all8'
    assert ex['verified_example']['plan'] == ex['plan']
    assert result['counts']['sumo_verified_examples'] == 1
    assert_sealed(result)


@pytest.mark.parametrize('kind', ['not_cooperative', 'not_admitted'])
def test_failed_spot_check_is_dropped(kind):
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    def checker(request):
        return evidence(request, admitted=kind != 'not_admitted', cooperative=kind != 'not_cooperative')
    result = pa.resolve_spot_checks(batch, checker)
    assert result['examples'] == [] and result['spot_checks'] == []
    assert result['spot_check_results'][0]['accepted_examples'] == 0


def test_spot_check_cross_arm_and_tampered_confirmation_refused():
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    with pytest.raises(ValueError, match='arm mismatch'):
        pa.resolve_spot_checks(batch, lambda req: evidence(req, arm='R'))
    def tampered(req):
        result = evidence(req)
        result['confirmation']['arm'] = 'R'
        return result
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
    snap = req['verification_provenance']['snapshot_identity']
    runner, health = Mock(), Mock()
    first = first_receipt(req['plan'], snap)
    confirmed = seal(req['plan'], arm=arm, snap=snap)
    with patch.object(vp, '_verify', return_value=first) as verify, \
         patch.object(vp, 'confirm_all8', return_value=confirmed) as confirm:
        result = pa.verify_all8(req, snapshot=snap, root='synthetic-root', run_root='single-ledger',
                                runner_for=runner, phase='registered-augment', health=health)
    assert result == dict(confirmation=confirmed, first_seed_receipt=first,
                          source_confirmation_sha256=req['source_confirmation_sha256'])
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
    src['verification']['first_seed_receipt'] = first_receipt(plan, snapshot())
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
    result = pa.resolve_spot_checks(batch, evidence)
    checker = Mock(side_effect=AssertionError('no repeated checks'))
    assert pa.resolve_spot_checks(result, checker) == result
    assert result['sumo_dispatched']
    checker.assert_not_called()


def test_unchecked_examples_are_not_relabelled_when_spot_checks_complete():
    batch = run(spot_check_fraction=.25)
    wm_only = copy.deepcopy(batch['examples'])
    result = pa.resolve_spot_checks(batch, evidence)
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


@pytest.mark.parametrize('field', ['t0', 'episode', 'seed', 'actor', 'crn', 'prefix'])
def test_foreign_decision_evidence_rejected_even_with_same_arm_and_plan(field):
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    def checker(req):
        snap = copy.deepcopy(req['verification_provenance']['snapshot_identity'])
        mutate_snapshot(snap, field)
        return evidence(req, snap=snap)
    with pytest.raises(ValueError, match='(time|identity|snapshot|decision)'):
        pa.resolve_spot_checks(batch, checker)


def mutate_snapshot(snap, field):
    if field == 't0':
        snap['t0'] = 900
    elif field == 'episode':
        snap['episode']['id'] = 'foreign-episode'
    elif field == 'seed':
        snap['episode']['world']['seed'] = 8
    elif field == 'actor':
        snap['policy']['state_dict_sha256'] = 'foreign-actor'
    elif field == 'crn':
        snap['rng']['generator_state_sha256'] = 'foreign-crn'
    else:
        snap['prefixes']['419'] = 'foreign-state'


@pytest.mark.parametrize('field', ['t0', 'episode', 'seed', 'actor', 'crn', 'prefix'])
def test_adapter_checks_full_bound_snapshot_before_dispatch(field):
    req = run(attempts_per_plan=1, spot_check_fraction=1.)['spot_checks'][0]
    snap = copy.deepcopy(req['verification_provenance']['snapshot_identity'])
    mutate_snapshot(snap, field)
    with patch.object(vp, '_verify') as verify, patch.object(vp, 'confirm_all8') as confirm:
        with pytest.raises(ValueError, match='snapshot'):
            pa.verify_all8(req, snapshot=snap, root='synthetic-root', run_root='single-ledger',
                          runner_for=Mock(), phase='registered-augment')
    verify.assert_not_called()
    confirm.assert_not_called()


@pytest.mark.parametrize('field', ['snapshot', 'receipt', 'source_hash', 'plan', 'seed2_salt', 'seed2_snapshot', 'tape'])
def test_provenance_and_own_seed_evidence_fail_closed(field):
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    def checker(req):
        result = evidence(req, seeds=(1, 2))
        rec, first = result['confirmation'], result['first_seed_receipt']
        if field == 'snapshot':
            first['snapshot_sha256'] = 'foreign-snapshot'
        elif field == 'receipt':
            rec['first_seed']['receipt_sha256'] = 'foreign-receipt'
        elif field == 'source_hash':
            result['source_confirmation_sha256'] = 'foreign-source'
        elif field == 'plan':
            rec['per_seed_v1']['examples'][0]['plan']['profile'][0][0] = 1.
        elif field == 'seed2_salt':
            rec['seed2']['salt'] = 'foreign-salt'
        elif field == 'seed2_snapshot':
            rec['seed2']['snapshot_sha256'] = 'foreign-snapshot'
        else:
            rec['per_seed_v1']['examples'][1]['tape'] = copy.deepcopy(rec['per_seed_v1']['examples'][0]['tape'])
        first.pop('receipt_sha256')
        result['first_seed_receipt'] = xg._seal(first, 'receipt_sha256')
        rec.pop('confirmation_sha256')
        result['confirmation'] = xg._seal(rec, 'confirmation_sha256')
        return result
    with pytest.raises(ValueError):
        pa.resolve_spot_checks(batch, checker)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_source_seed_snapshot_is_bound_and_not_forwarded_to_wm(seed):
    src, wm = source(), WM()
    p = src['confirmation']['per_seed_v1']['examples'][0]['plan']
    src['confirmation'] = seal(p, seeds=(seed,))
    batch = run(src, wm=wm, attempts_per_plan=1, spot_check_fraction=1.)
    req, = batch['spot_checks']
    vid = src['confirmation']['verification_id']
    expected = snapshot() if seed == 1 else rb.replicate(snapshot(), vid if seed == 2 else vp.extra_seed_salt(vid, seed))
    assert sv.snapshot_sha(req['verification_provenance']['snapshot_identity']) == sv.snapshot_sha(expected)
    assert set(wm.calls[0]['context']) == {'ego_id', 't_s', 'rows', 'policy_traj'}
    result = pa.resolve_spot_checks(batch, lambda r: evidence(r, seeds=(1, 2)))
    assert [e['seed_index'] for e in result['examples']] == [1, 2]
    for ex in result['examples']:
        k = ex['seed_index']
        verified = ex['verified_example']
        assert verified['tape']['seed_index'] == k
        assert verified['tape']['branch']['result_sha256'] == digest(dict(seed=k))
        assert verified['advantage_veh_s'] == 10.*k
        assert verified['synergy']['lower95_one_sided_veh_s'] == float(k)


@pytest.mark.parametrize('field', ['actor', 'seed', 'crn', 'prefix'])
def test_source_identity_must_match_authoritative_sealed_first_receipt(field):
    src = source()
    mutate_snapshot(src['verification']['snapshot_identity'], field)
    wm = Mock()
    with pytest.raises(ValueError, match='(snapshot|identity)'):
        run(src, wm=wm)
    wm.assert_not_called()


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_adapter_evidence_resolves_on_own_source_seed_with_mocked_runners(arm):
    src = source(arm)
    plan = src['confirmation']['per_seed_v1']['examples'][0]['plan']
    src['confirmation'] = seal(plan, arm=arm, seeds=(2,))
    batch = run(src, arm=arm, attempts_per_plan=1, spot_check_fraction=1.)
    req, = batch['spot_checks']
    snap = rb.replicate(snapshot(arm), src['confirmation']['verification_id'])
    runner = Mock(side_effect=AssertionError('CPU contract must not dispatch physics'))
    first = first_receipt(req['plan'], snap)
    confirmed = seal(req['plan'], arm=arm, snap=snap, seeds=(1, 2))
    with patch.object(vp, '_verify', return_value=first), patch.object(vp, 'confirm_all8', return_value=confirmed):
        result = pa.resolve_spot_checks(batch, lambda request: pa.verify_all8(
            request, snapshot=snap, root='synthetic-root', run_root='single-ledger',
            runner_for=runner, phase='registered-augment'))
    assert [e['verified_example'] for e in result['examples']] == confirmed['per_seed_v1']['examples']
    runner.assert_not_called()


def test_source_confirmation_must_reference_supplied_authoritative_receipt():
    src = source()
    record = src['confirmation']
    record['first_seed']['receipt_sha256'] = 'foreign-first-receipt'
    record.pop('confirmation_sha256')
    src['confirmation'] = xg._seal(record, 'confirmation_sha256')
    with pytest.raises(ValueError, match='decision identity'):
        run(src)


def test_wrong_confirmation_time_rejected_even_if_first_receipt_matches_source():
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    def checker(req):
        result = evidence(req)
        record = result['confirmation']
        record['t0'] = 900
        record.pop('confirmation_sha256')
        result['confirmation'] = xg._seal(record, 'confirmation_sha256')
        return result
    with pytest.raises(ValueError, match='decision time'):
        pa.resolve_spot_checks(batch, checker)


def test_corrupt_bound_snapshot_rejected_before_checker_dispatch():
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    batch['spot_checks'][0]['verification_provenance']['snapshot_sha256'] = 'wrong-snapshot'
    batch.pop('receipt_sha256')
    batch['receipt_sha256'] = digest(batch)
    checker = Mock()
    with pytest.raises(ValueError, match='snapshot identity'):
        pa.resolve_spot_checks(batch, checker)
    checker.assert_not_called()


def reseal_evidence(result):
    """Keep the receipt chain digest-valid while testing semantic mismatches."""
    first, rec = result['first_seed_receipt'], result['confirmation']
    first.pop('receipt_sha256')
    result['first_seed_receipt'] = xg._seal(first, 'receipt_sha256')
    rec['first_seed']['receipt_sha256'] = result['first_seed_receipt']['receipt_sha256']
    rec.pop('confirmation_sha256')
    result['confirmation'] = xg._seal(rec, 'confirmation_sha256')
    return result


@pytest.mark.parametrize('changes', [dict(status='failed'), dict(status='budget_blocked'),
                                     dict(proposer='random')])
def test_all8_examples_cannot_override_incomplete_or_wrong_arm_first_decision(changes):
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    def checker(req):
        result = evidence(req)
        result['first_seed_receipt'].update(changes)
        return reseal_evidence(result)
    with pytest.raises(ValueError, match='first-seed decision'):
        pa.resolve_spot_checks(batch, checker)


@pytest.mark.parametrize('accepted_seeds', [(2,), (3,), (1, 2, 3)])
def test_only_each_accepted_seeds_own_tape_and_labels_survive(accepted_seeds):
    batch = run(attempts_per_plan=1, spot_check_fraction=1.)
    returned = []
    def checker(req):
        result = evidence(req, seeds=(1, 2, 3))
        for ex in result['confirmation']['per_seed_v1']['examples']:
            ex['admitted'] = ex['cooperative'] = ex['seed_index'] in accepted_seeds
        returned.extend(copy.deepcopy(result['confirmation']['per_seed_v1']['examples']))
        return reseal_evidence(result)
    resolved = pa.resolve_spot_checks(batch, checker)
    assert [ex['seed_index'] for ex in resolved['examples']] == list(accepted_seeds)
    assert [ex['verified_example'] for ex in resolved['examples']] == [
        ex for ex in returned if ex['seed_index'] in accepted_seeds]
    assert resolved['counts']['sumo_verified_examples'] == len(accepted_seeds)
    assert_sealed(resolved)


@pytest.mark.parametrize('where', ['source', 'spot_check'])
def test_candidate_id_must_match_plan_even_when_plan_hash_matches(where):
    def foreign_candidate(record, first, snap):
        first['candidate_ids'] = ['foreign-candidate']
        vid = sv.verification_id(snap['episode']['id'], snap['t0'], [dict(id='foreign-candidate')])
        first['verification_id'] = vid
        first.pop('receipt_sha256')
        first = xg._seal(first, 'receipt_sha256')
        record['verification_id'] = vid
        record['first_seed'].update(verification_id=vid, receipt_sha256=first['receipt_sha256'])
        record['candidates'][0]['plan_id'] = 'foreign-candidate'
        record['per_seed_v1']['examples'][0]['tape']['verification_id'] = vid
        record.pop('confirmation_sha256')
        return xg._seal(record, 'confirmation_sha256'), first

    src = source()
    if where == 'source':
        src['confirmation'], src['verification']['first_seed_receipt'] = foreign_candidate(
            src['confirmation'], src['verification']['first_seed_receipt'], snapshot())
        with pytest.raises(ValueError, match='candidate identity'):
            run(src)
    else:
        batch = run(src, attempts_per_plan=1, spot_check_fraction=1.)
        def checker(req):
            result = evidence(req)
            result['confirmation'], result['first_seed_receipt'] = foreign_candidate(
                result['confirmation'], result['first_seed_receipt'],
                req['verification_provenance']['snapshot_identity'])
            return result
        with pytest.raises(ValueError, match='candidate identity'):
            pa.resolve_spot_checks(batch, checker)
