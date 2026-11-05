"""G1 meaningful CPU contracts: no simulator, process launcher or external data."""
from copy import deepcopy
import json
import pytest

from . import executable_labels as el, moves, bank_breadth as breadth
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401


def sha(value):
    return el.digest(value)


def source(arm='B'):
    plan = dict(members=['a', 'b'], duration_s=41,
        perturbation_profile=[[0.]+[-.5]*40]*2, variety_family='wall')
    receipt = el.seal(dict(verification_id='native_'+arm, t0=300,
        chosen_plan=plan, window=dict(start_s=300, end_s=420),
        episode=dict(role='train', arm=arm, world=dict(seed=986000))), 'receipt_sha256')
    example = dict(example_id=arm+'#s1', seed_index=1)
    return receipt, example, plan


def lesson(arm='B', move=None, identity=0, **changes):
    receipt, example, plan = source(arm)
    macro = move or moves.Move('wall', ('a', 'b'), 0, 40, .5)
    local = dict(self='a', observed_at_s=300, acquirable_cav_ids=['a', 'b'],
        rows=[dict(vid='a', s_m=-1500., v_mps=10., lane=0, is_cav=True),
              dict(vid='b', s_m=-1510., v_mps=10., lane=1, is_cav=True)],
        offered_moves=[macro.to_dict()])
    values = dict(arm=arm, receipt=receipt, example=example, plan=plan,
        snapshot_sha256=sha(['snapshot', arm, identity]), prefix_sha256=sha('prefix'),
        base_policy_sha256=sha('base'), runtime_sha256=sha('runtime'), local=local,
        decision_s=301, window=dict(start_s=300, end_s=420), move=macro)
    values.update(changes)
    if 'window' in changes and 'receipt' not in changes:
        values['receipt'] = el.seal(dict(receipt, window=changes['window']), 'receipt_sha256')
    return el.bind_lesson(**values)


def seeds():
    return [dict(verification_seed=i, snapshot_sha256=sha(['restart', i]), rng_sha256=sha(['rng', i])) for i in (1, 2)]


def job(arm='B', **kwargs):
    return el.job_spec(lesson(arm, **kwargs), seeds())


def results(spec, delays=None, clipped=False, asynchronous=False):
    delays = delays or dict(group=1., base=5., solo0=3., solo1=4.)
    out = []
    for branch in spec['branches']:
        n = branch['window']['end_s']-branch['window']['start_s']
        audit = []
        if branch['branch'] == 'group':
            for i, vid in enumerate(spec['lesson']['actors']):
                t = spec['lesson']['decision_s']+(i if asynchronous else 0)
                state = dict(s_m={v: -1500. for v in spec['lesson']['macro']['members']})
                offset = moves.offsets(moves.Move.from_dict(spec['lesson']['macro']),
                    t-spec['lesson']['decision_s'], state)[vid]
                issued = 0. if clipped else offset
                audit.append(dict(time_s=t, vid=vid, observable_state=state,
                    library_move_offset_mps2=offset, requested_raw=offset, without_library_raw=0.,
                    issued_safe=issued, without_library_issued_safe=0., library_issued_delta_mps2=issued))
        out.append(el.seal(dict(status='graded', branch_sha256=branch['branch_sha256'],
            window=branch['window'], prefix_matched=True, crn_aligned_through_s=branch['window']['end_s'],
            actual_sim_s=branch['sim_s'], delay_increments=[delays[branch['branch']]]*n,
            safety=dict.fromkeys(el.SAFETY, 0), library_actuation_audit=audit), 'result_sha256'))
    return out


def reseal(result):
    return el.seal(result, 'result_sha256')


def test_binding_is_actual_macro_roles_local_view_and_native_fidelity():
    bound = lesson()
    assert bound['actors'] == ['a', 'b']
    assert bound['roles'] == dict(a='upstream_holder', b='upstream_holder')
    assert bound['fidelity']['exact_native_profile']
    projected = lesson(move=moves.Move('wall', ('a', 'b'), 1, 40, .5))
    assert projected['fidelity']['squared_offset_error'] == 1.
    assert not projected['fidelity']['exact_native_profile']
    assert projected['lesson_sha256'] != bound['lesson_sha256']
    assert el.prioritize([projected, bound])[0] == bound


@pytest.mark.parametrize('change', ['hidden', 'hdv', 'owned', 'stale', 'far', 'downstream', 'not_offered', 'arrival'])
def test_binding_refuses_unexecutable_visibility_and_geometry(change):
    local = deepcopy(lesson()['local'])
    if change == 'hidden': local['rows'].pop()
    if change == 'hdv': local['rows'][1]['is_cav'] = False
    if change == 'owned': local['acquirable_cav_ids'] = ['a']
    if change == 'stale': local['observed_at_s'] = 299
    if change == 'far': local['rows'][1]['s_m'] = -5000.
    if change == 'downstream': local['rows'][0]['s_m'] = 0.
    if change == 'not_offered': local['offered_moves'] = []
    if change == 'arrival':
        for r in local['rows']: r['s_m'] = -150.
    with pytest.raises(ValueError):
        lesson(local=local)


def test_post_t0_full_horizon_and_immutable_source_required():
    with pytest.raises(ValueError, match='post-t0'):
        lesson(decision_s=300)
    with pytest.raises(ValueError, match='full macro'):
        lesson(window=dict(start_s=300, end_s=320))
    receipt, example, plan = source()
    receipt['episode']['role'] = 'diagnostic'
    with pytest.raises(ValueError, match='TRAIN'):
        lesson(receipt=receipt)
    receipt, _, _ = source()
    receipt['t0'] = 299
    with pytest.raises(ValueError, match='receipt_sha256'):
        lesson(receipt=receipt)
    with pytest.raises(ValueError, match='full registered window'):
        lesson(receipt=source()[0], window=dict(start_s=300, end_s=400))


def test_job_binds_group_base_and_every_member_solo_same_full_window_crn():
    spec = job()
    assert len(spec['branches']) == 8
    assert spec['sim_s'] == 8*420  # includes all replay, not merely 120-second outcome window
    for i in (0, 4):
        group, base, a, b = spec['branches'][i:i+4]
        assert [r['selected_actors'] for r in (group, base, a, b)] == [['a', 'b'], [], ['a'], ['b']]
        assert all(r['macro'] == group['macro'] and r['seed'] == group['seed'] and
            r['engine'] == group['engine'] and r['window'] == dict(start_s=300, end_s=420) for r in (base, a, b))
        assert group['simulate_from_s'] == 0 and group['simulate_through_s'] == 420
        assert group['decision_s'] == 301
    duplicate = seeds(); duplicate[1]['rng_sha256'] = duplicate[0]['rng_sha256']
    with pytest.raises(ValueError, match='two distinct'):
        el.job_spec(lesson(), duplicate)


def test_admits_exact_executable_positive_synergy_and_reports_total_separately():
    spec = job(); report = el.admit(spec, results(spec))
    assert report['admitted'] and not report['reasons']
    stats = report['statistics']
    assert stats['synergy_mean_veh_s'] == 240.
    assert stats['total_benefit_mean_veh_s'] == 480.
    assert stats['synergy_ci95'] == [240., 240.]
    assert all(p['joint_seconds'] == 1 and p['passed'] for p in report['physical'])


@pytest.mark.parametrize('group,solo,reason', [(1., 1., 'CI95'), (2., 1., 'CI95')])
def test_native_positive_legacy_fields_cannot_admit_zero_or_negative_executable(group, solo, reason):
    spec = job()
    res = results(spec, dict(group=group, base=100., solo0=solo, solo1=4.))
    report = el.admit(spec, res)
    assert not report['admitted'] and reason in ' '.join(report['reasons'])
    assert report['statistics'] is not None


@pytest.mark.parametrize('case', ['missing_solo', 'duplicate', 'short_window', 'short_series', 'wrong_crn',
    'prefix', 'undercharge', 'nonfinite', 'edited', 'native_hash', 'ungraded', 'missing_safety', 'effect_mismatch'])
def test_incomplete_misaligned_or_native_ancestor_evidence_never_admits(case):
    spec = job(); res = results(spec)
    if case == 'missing_solo': res.pop()
    elif case == 'duplicate': res[-1] = res[0]
    else:
        row = deepcopy(res[0])
        if case == 'short_window': row['window'] = dict(start_s=300, end_s=419)
        if case == 'short_series': row['delay_increments'].pop()
        if case == 'wrong_crn': row['crn_aligned_through_s'] = 419
        if case == 'prefix': row['prefix_matched'] = False
        if case == 'undercharge': row['actual_sim_s'] = 120
        if case == 'nonfinite': row['delay_increments'][0] = float('nan')
        if case == 'native_hash': row['branch_sha256'] = sha('native plan job')
        if case == 'ungraded': row['status'] = 'planned'
        if case == 'missing_safety': row['safety'].pop('hard_brake')
        if case == 'effect_mismatch': row['library_actuation_audit'][0]['library_issued_delta_mps2'] = .1
        if case == 'edited': row['prefix_matched'] = False
        res[0] = row if case in ('edited', 'nonfinite') else reseal(row)
    report = el.admit(spec, res)
    assert not report['admitted'] and 'evidence' in ' '.join(report['reasons'])


@pytest.mark.parametrize('clipped,asynchronous', [(True, False), (False, True)])
def test_two_members_or_requested_offsets_are_not_joint_physical_execution(clipped, asynchronous):
    spec = job(); report = el.admit(spec, results(spec, clipped=clipped, asynchronous=asynchronous))
    assert not report['admitted']
    assert 'physically' in ' '.join(report['reasons'])


def test_gap_open_has_one_actuator_and_cannot_be_joint_certificate():
    macro = moves.Move('gap_open', ('a', 'b'), 0, 40, .5)
    local = lesson()['local']; local['rows'][0].update(s_m=10., lane=1)
    local['rows'][1].update(s_m=0., lane=2); local['offered_moves'] = [macro.to_dict()]
    bound = lesson(move=macro, local=local)
    assert bound['actors'] == ['a'] and bound['roles']['b'] == 'context_only'
    spec = el.job_spec(bound, seeds())
    assert not el.admit(spec, results(spec))['admitted']


def test_unsafe_positive_synergy_and_negative_second_seed_stay_diagnostics():
    spec = job(); res = results(spec)
    res[0]['safety']['hard_brake'] = 1; res[0] = reseal(res[0])
    assert 'safety failed' in ' '.join(el.admit(spec, res)['reasons'])
    res = results(spec)
    res[4]['delay_increments'] = [10.]*120; res[4] = reseal(res[4])
    assert not el.admit(spec, res)['admitted']


def test_all_solo_reselection_and_uncertainty_not_selected_solo_or_min_per_tick():
    # Solo0/1 alternate winning blocks. Neither pointwise minima nor choosing a
    # favorable solo before resampling is the registered full-window estimand.
    spec = job(); res = results(spec)
    a = [0.]*60+[4.]*60; b = [4.]*60+[0.]*60
    for i in (0, 4):
        res[i]['delay_increments'] = [1.9]*120
        res[i+2]['delay_increments'] = a; res[i+3]['delay_increments'] = b
    res = [reseal(r) for r in res]
    report = el.admit(spec, res)
    assert report['statistics']['synergy_mean_veh_s'] == pytest.approx(12.)
    assert report['statistics']['synergy_ci95'][0] < 0
    assert not report['admitted']


def test_quote_exact_cap_prefix_cost_symmetry_and_no_duplicate_contexts():
    quote = el.budget_quote()
    assert quote['preferred_two_actor_sim_s'] == 153600
    assert quote['worst_case_sim_s'] == quote['cap_sim_s'] == 230400
    assert quote['executed_sim_s'] == 0 and not quote['dispatched']
    b, r = job('B'), job('R')
    assert el.budget_quote([b, r])['planned_sim_s'] == 6720
    with pytest.raises(ValueError, match='equal B/R'):
        el.budget_quote([b])
    with pytest.raises(ValueError, match='duplicate context'):
        el.budget_quote([b, r, b, r])
    long_r = job('R', window=dict(start_s=300, end_s=500))
    with pytest.raises(ValueError, match='effort must match'):
        el.budget_quote([b, long_r])


def test_four_actor_full_episode_batch_hits_cap_and_requires_all_four_solos():
    batch = []
    for arm in ('B', 'R'):
        for identity in range(8):
            macro = moves.Move('wall', ('a', 'b', 'c', 'd'), 0, 40, .5)
            receipt, example, plan = source(arm)
            plan = dict(plan, members=list(macro.members), perturbation_profile=plan['perturbation_profile']*2)
            receipt = el.seal(dict(receipt, window=dict(start_s=300, end_s=1200), chosen_plan=plan), 'receipt_sha256')
            local = lesson()['local']
            local['rows'].extend([dict(vid=v, s_m=-1520.-i*10, v_mps=10., lane=i%2, is_cav=True)
                                  for i, v in enumerate(('c', 'd'))])
            local.update(acquirable_cav_ids=list(macro.members), offered_moves=[macro.to_dict()])
            bound = lesson(arm, macro, identity=identity, receipt=receipt, example=example, plan=plan,
                local=local, window=receipt['window'])
            batch.append(el.job_spec(bound, seeds()))
    quote = el.budget_quote(batch)
    assert quote['planned_sim_s'] == quote['cap_sim_s'] == 230400
    assert quote['branches'] == 192
    assert [b['branch'] for b in batch[0]['branches'][:6]] == ['group', 'base', 'solo0', 'solo1', 'solo2', 'solo3']
    with pytest.raises(ValueError, match='at most eight'):
        el.budget_quote(batch+[job('B', identity=9), job('R', identity=9)])


def manifest_value(negative_r=False):
    b, r = job('B'), job('R')
    return el.manifest([dict(job=b, results=results(b)), dict(job=r, results=results(r, clipped=negative_r))])


def test_manifest_keeps_negatives_recomputes_and_binds_source_not_boolean(tmp_path):
    value = manifest_value(negative_r=True)
    assert value['admitted'] == 1 and value['diagnostics'] == 1
    path = tmp_path/'G1_MANIFEST.json'; path.write_text(json.dumps(value), encoding='utf-8')
    assert el.read_manifest(path) == value
    receipt, example, plan = source()
    chosen = el.bank_certificate(value, arm='B', receipt=receipt, example=example, plan=plan,
        snapshot_sha256=lesson()['snapshot_sha256'])
    assert chosen == lesson()
    changed_plan = dict(plan, duration_s=42)
    assert el.bank_certificate(value, arm='B', receipt=receipt, example=example, plan=changed_plan,
        snapshot_sha256=lesson()['snapshot_sha256']) is None
    for key in ('prefix_sha256', 'base_policy_sha256', 'runtime_sha256'):
        assert el.bank_certificate(value, arm='B', receipt=receipt, example=example, plan=plan,
            snapshot_sha256=lesson()['snapshot_sha256'], **{key: sha('wrong binding')}) is None
    short_receipt = dict(receipt, window=dict(start_s=300, end_s=400))
    assert el.bank_certificate(value, arm='B', receipt=short_receipt, example=example, plan=plan,
        snapshot_sha256=lesson()['snapshot_sha256']) is None
    value['entries'][1]['admission']['admitted'] = True
    path.write_text(json.dumps(el.seal(value, 'manifest_sha256')), encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        el.read_manifest(path)


def test_f3_visit_gate_shared_by_practice_and_trigger_and_a_bypasses(monkeypatch, tmp_path):
    from . import track2_backend as p1, practice_starts as ps, practice_weighting as pw, sumo_verify as sv
    value = manifest_value(negative_r=True)
    monkeypatch.setattr(el, 'read_manifest', lambda path: value)
    config = dict(run_root=str(tmp_path), training_seed=1, executable_label_manifest='CPU',
        executable_label_runtime_sha256=sha('runtime'))
    monkeypatch.setattr(breadth, 'inputs', lambda cfg: [dict(run_root=str(tmp_path))])
    monkeypatch.setattr(ps, 'examples', lambda *args: [])
    monkeypatch.setattr(ps, '_links', lambda *args: {})
    monkeypatch.setattr(pw, 'augmented_examples', lambda *a, **k: ([], None))
    class Bank:
        def bank_pairs(self, root, arm, seed, **kwargs):
            receipt, example, plan = source(arm)
            kwargs['example_consumer'](receipt=receipt, example=example, plan=plan,
                pairs=[('sample', 1.)], trace=tmp_path/'CPU_TRACE.json')
    class View:
        def __enter__(self): return Bank()
        def __exit__(self, *args): pass
    monkeypatch.setattr(p1, '_bank_view', lambda cfg: View())
    monkeypatch.setattr(breadth, 'verified_restart', lambda root, arm, *args: (dict(arm=arm), {}))
    monkeypatch.setattr(sv, 'snapshot_sha', lambda snapshot: lesson(snapshot['arm'])['snapshot_sha256'])
    monkeypatch.setattr(el, 'restart_binding', lambda snapshot: dict(snapshot_sha256=lesson(snapshot['arm'])['snapshot_sha256'],
        prefix_sha256=sha('prefix'), base_policy_sha256=sha('base')))
    got = []
    for arm in ('A', 'B', 'R'):
        breadth.visit(config, arm, lambda **row: got.append(row))
    assert len(got) == 1
    assert got[0]['cfg']['_executable_lesson']['arm'] == 'B'
    # Unconfigured behavior still visits both native arms, no certification read.
    monkeypatch.setattr(el, 'read_manifest', lambda path: pytest.fail('off must not read manifest'))
    got.clear()
    for arm in ('B', 'R'):
        breadth.visit(dict(run_root=str(tmp_path), training_seed=1), arm, lambda **row: got.append(row))
    assert len(got) == 2 and all('_executable_lesson' not in row['cfg'] for row in got)


def causal_replay(cfg, arm, moving=True):
    from tools.coopnative.policy import encode
    from .track2_backend import EpisodePolicy
    from .v2v_intent import IntentPolicy
    bound = lesson(arm)
    rows = [dict(row, length_m=5.) for row in bound['local']['rows']]
    frames = {300: dict(time_s=300, vehicles=rows)}
    for t in (301, 302):
        frames[t] = dict(time_s=t, vehicles=[dict(row,
            s_m=row['s_m']+(t-300)*10. if moving else row['s_m'],
            v_mps=row['v_mps']+(t-300) if moving else row['v_mps']) for row in rows])
    pairs = [(dict(intent_link=dict(t=t), context=encode(frames[t]['vehicles'])), 1.) for t in frames]
    return EpisodePolicy(IntentPolicy(), cfg, arm), bound, pairs, frames


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('moving', [False, True])
def test_real_observer_certified_time_proposer_features_and_macro(cfg, arm, moving):
    from tools.coopnative.policy import encode
    policy, bound, pairs, frames = causal_replay(cfg, arm, moving)
    before = deepcopy(frames)
    got = list(breadth.observable_contexts(policy, source(arm)[2], pairs, frames,
        t0=300, certification=bound))
    assert len(got) == 1
    context, label, reader, _, time = got[0]
    assert time == 301 and context['members'] == ['a'] and label.to_dict() == bound['macro']
    assert reader.now == bound['local']['observed_at_s'] == 300
    assert reader.local_states['a']['rows'] == frames[300]['vehicles']
    assert reader.local_states['a']['acquirable_cav_ids'] == ['a', 'b']
    expected = encode(frames[300]['vehicles'])
    assert context['x'] == [expected['x'][expected['members'].index('a')]]
    assert context['trigger_v2v']['a'] == reader.local_intents['a']
    # Both local sensor and partner speed/position inputs must use the bound view.
    assert context['trigger_v2v']['a'][0] == -10.
    assert context['trigger_v2v']['a'][2] == 10.
    if moving:
        assert context['x'] != [pairs[1][0]['context']['x'][0]]
    assert frames == before and policy.now is None  # no trace/policy mutation
    assert list(breadth.observable_contexts(policy, source(arm)[2], pairs[2:], frames,
        t0=300, certification=bound)) == []


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('case', ['position', 'speed', 'missing_previous', 'stale_frame',
    'stale_certificate', 'missing_actor', 'inactive_actor', 'unavailable'])
def test_real_observer_rejects_mismatched_or_stale_certified_evidence(cfg, arm, case):
    policy, bound, pairs, frames = causal_replay(cfg, arm)
    if case == 'position':
        # Current geometry must not pass as the actual t-1 evidence.
        frames[300]['vehicles'] = deepcopy(frames[301]['vehicles'])
    elif case == 'speed':
        frames[300]['vehicles'][1]['v_mps'] += 1.
    elif case == 'missing_previous':
        del frames[300]
    elif case == 'stale_frame':
        frames[300]['time_s'] = 299
    elif case == 'stale_certificate':
        bound = el.seal(dict(bound, local=dict(bound['local'], observed_at_s=299)), 'lesson_sha256')
    elif case == 'missing_actor':
        frames[301]['vehicles'].pop()
    elif case == 'inactive_actor':
        pairs[1][0]['context']['members'] = ['a']
    elif case == 'unavailable':
        bound = el.seal(dict(bound, local=dict(bound['local'], acquirable_cav_ids=['a'])), 'lesson_sha256')
    assert list(breadth.observable_contexts(policy, source(arm)[2], pairs, frames,
        t0=300, certification=bound)) == []


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_certified_candidate_availability_does_not_expand_to_all_visible_cavs(cfg, arm):
    import numpy as np
    from tools.coopnative.policy import encode
    from . import feasible_proposals
    policy, bound, pairs, frames = causal_replay(cfg, arm)
    extra = dict(vid='c', s_m=-1520., v_mps=10., lane=1, is_cav=True)
    local = dict(bound['local'], rows=bound['local']['rows']+[extra])
    bound = lesson(arm, local=local)  # c is visible, but reserved by another macro
    frames[300]['vehicles'].append(dict(extra, length_m=5.))
    frames[301]['vehicles'].append(dict(extra, length_m=5., s_m=-1510., v_mps=11.))
    pairs[1][0]['context'] = encode(frames[301]['vehicles'])
    got = list(breadth.observable_contexts(policy, source(arm)[2], pairs, frames,
        t0=300, certification=bound))
    assert len(got) == 1
    _, _, reader, _, _ = got[0]
    assert 'c' in reader.visible['a']
    state = reader.local_states['a']
    assert state['acquirable_cav_ids'] == ['a', 'b']
    proposals = feasible_proposals.candidates(state, np.random.default_rng(123), 32)
    assert proposals and all('c' not in move.members for move in proposals)


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('evidence', ['matched', 'mismatched', 'stale'])
def test_enabled_moving_certificates_reach_real_trigger_learning(cfg, monkeypatch, arm, tmp_path, evidence):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import torch
    from . import track2_backend as p1, practice_starts as ps, practice_weighting as pw
    from . import bank_v2v, launch_production as launch
    path = tmp_path/'G1_MANIFEST.json'
    path.write_text(json.dumps(manifest_value()), encoding='utf-8')
    config = dict(cfg, executable_label_manifest=str(path), executable_label_runtime_sha256=sha('runtime'))
    policy, _, _, _ = causal_replay(config, arm)
    monkeypatch.setattr(breadth, 'inputs', lambda config: [dict(run_root=str(tmp_path))])
    monkeypatch.setattr(ps, 'examples', lambda *args: [])
    monkeypatch.setattr(ps, '_links', lambda *args: {})
    monkeypatch.setattr(pw, 'augmented_examples', lambda *a, **k: ([], None))
    monkeypatch.setattr(breadth, 'verified_restart', lambda root, arm, *args: (dict(arm=arm), {}))
    monkeypatch.setattr(el, 'restart_binding', lambda snapshot: dict(
        snapshot_sha256=lesson(snapshot['arm'])['snapshot_sha256'],
        prefix_sha256=sha('prefix'), base_policy_sha256=sha('base')))
    class Bank:
        def bank_pairs(self, root, requested_arm, seed, **kwargs):
            receipt, example, plan = source(requested_arm)
            _, _, pairs, _ = causal_replay(config, requested_arm)
            kwargs['example_consumer'](receipt=receipt, example=example, plan=plan,
                pairs=pairs, trace=tmp_path/(requested_arm+'_TRACE.json'))
    @contextmanager
    def bank_view(config):
        yield Bank()
    monkeypatch.setattr(p1, '_bank_view', bank_view)
    frames = causal_replay(config, arm)[3]
    if evidence == 'mismatched':
        frames[300]['vehicles'] = deepcopy(frames[301]['vehicles'])
    elif evidence == 'stale':
        frames[300]['time_s'] = 299
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path: frames)
    base_before = deepcopy(policy.layer.base.state_dict())
    head_before = deepcopy(policy.trigger.trigger_head.state_dict())
    runtime = SimpleNamespace(cfg=config, policy=policy, arm=arm)
    if evidence != 'matched':
        with pytest.raises(ValueError, match='nonempty observable verified-plan trigger supervision required'):
            launch.bank_move_supervision(runtime)
        assert all(torch.equal(v, policy.trigger.trigger_head.state_dict()[k]) for k, v in head_before.items())
        return
    report = launch.bank_move_supervision(runtime)
    assert report['examples'] == 1 and report['family_counts'] == {'wall': 1}
    assert report['contexts'][0]['time_s'] == 301 and report['contexts'][0]['members'] == ['a']
    assert report['final_ce'] < report['initial_ce'] and any(p['accepted'] for p in report['passes'])
    assert any(not torch.equal(v, policy.trigger.trigger_head.state_dict()[k]) for k, v in head_before.items())
    assert all(torch.equal(v, policy.layer.base.state_dict()[k]) for k, v in base_before.items())
    assert all(p.grad is None for p in policy.layer.base.parameters())


def test_frozen_restart_rechecks_certificate(monkeypatch):
    from . import track2_backend as p1, replay_bank as rb, sumo_verify as sv
    receipt, example, plan = source()
    binding = dict(seed_index=1)
    lookup = dict(binding_version=1, run_root='.', receipt=receipt, example=example, plan=plan,
        verification_binding=binding, bank_cfg=dict(executable_label_manifest='CPU', executable_label_runtime_sha256=sha('runtime')),
        executable_lesson_sha256=lesson()['lesson_sha256'])
    class View:
        def __enter__(self): return rb
        def __exit__(self, *args): pass
    monkeypatch.setattr(p1, '_bank_view', lambda cfg: View())
    monkeypatch.setattr(rb, 'bank_decisions', lambda *a, **k: [receipt])
    monkeypatch.setattr(breadth, 'verified_restart', lambda *a: ({}, binding))
    monkeypatch.setattr(sv, 'snapshot_sha', lambda snapshot: lesson()['snapshot_sha256'])
    monkeypatch.setattr(el, 'restart_binding', lambda snapshot: dict(snapshot_sha256=lesson()['snapshot_sha256'],
        prefix_sha256=sha('prefix'), base_policy_sha256=sha('base')))
    monkeypatch.setattr(el, 'read_manifest', lambda path: manifest_value())
    assert breadth.load_verified_restart('B', lookup) == {}
    lookup['executable_lesson_sha256'] = sha('different macro')
    with pytest.raises(ValueError, match='certificate changed'):
        breadth.load_verified_restart('B', lookup)
