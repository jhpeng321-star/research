"""T2B CPU contracts; no simulator, worker dispatch, remote connection or model loading."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.coopmwm import dense_teacher as dt
from tools.coopmwm.protocol import digest


def observation():
    # Hidden extras are deliberate sentinels: dependencies must never receive them.
    return dict(ego_id='a', time_s=600, global_state='forbidden', rows=[
        dict(vid='a', s_m=100., v_mps=8., lane='m1', is_cav=True, driver_type='forbidden'),
        dict(vid='b', s_m=90., v_mps=9., lane='m1', is_cav=True),
        dict(vid='h', s_m=80., v_mps=7., lane='m1', is_cav=False)])


def move(family='gap_open', depth=0.2, members=('a', 'b')):
    return dict(family=family, members=list(members), onset=5., hold=40., depth=depth)


class Moves:
    """Only the registered T1A seam, pending its independent landing."""
    FAMILIES = ('ease_off', 'gap_open', 'wall', 'ramp_assist', 'none')
    Move = staticmethod(lambda family, members, onset, hold, depth:
                        SimpleNamespace(family=family, members=members, onset=onset,
                                        hold=hold, depth=depth))

    @staticmethod
    def offsets(selected, t):
        actors = selected.members[:1] if selected.family in ('ease_off', 'gap_open') else selected.members
        return {v: -selected.depth if selected.onset <= t < selected.onset+selected.hold
                and v in actors else 0. for v in selected.members}

    @staticmethod
    def candidates(obs_state, rng, n):
        assert 'global_state' not in obs_state
        assert all('driver_type' not in r for r in obs_state['rows'])
        return [move(Moves.FAMILIES[i % 4], float(rng.uniform(0.1, 0.3))) for i in range(n)]


def teacher(random=False, **kwargs):
    cfg = dict(dense_teacher=True, dense_teacher_random=random, dense_teacher_candidates=3,
               dense_teacher_fraction=0.)
    cfg.update(kwargs.pop('cfg', {}))
    if not random:
        kwargs.setdefault('scorer', lambda obs, c: [dict(p_better=.9, gain=i+1.)
                                                  for i in range(len(c))])
    return dt.make_teacher(cfg, moves_mod=Moves, **kwargs)


def label(t=None, step=0):
    return (t or teacher()).label_state('episode_B_1', step, 600+step, 'a', observation())


def bank_label(none=False, arm='B'):
    obs = observation()
    obs['ego_id'] = 'c1'
    obs['rows'][0]['vid'], obs['rows'][1]['vid'] = 'c1', 'c2'
    stub = SimpleNamespace(FAMILIES=Moves.FAMILIES, candidates=lambda obs, rng, n:
                           [move(depth=.6, members=('c1', 'c2')) for _ in range(n)])
    t = dt.WMDenseTeacher(stub, scorer=lambda obs, c:
                         [dict(p_better=.1 if none else .9, gain=1)]*len(c))
    if arm == 'R':
        t = dt.RandomTeacher(stub)
        t.choice_rng = np.random.default_rng(7)
    return t.label_state(f'warm_u005_{arm}_7', 0, 600, 'c1', obs)


def dense_setup(tmp_path, gains, k=2, arm='B'):
    """Real all8 fixture, with plans built BEFORE dispatch from an independent base schedule."""
    from tools.coopmwm import test_verify_perseed_cpu as fixture
    lb = bank_label(arm=arm)
    snap = fixture.snapshot(600)
    snap['episode'] = dict(snap['episode'], id=lb['episode'], arm=arm)
    def plan(pid, gain):
        return dt.prepare_verification_plan(lb, snap, [[.3]*120]*2,
                                            moves_mod=Moves, plan_id=pid)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixture, 'plan', plan)
        result = fixture.setup(tmp_path, gains, k=k, arm=arm)
    return (*result, lb)


def test_config_off_removes_all_teacher_keys():
    cfg = dict(ordinary=123, dense_teacher=False, dense_teacher_seed=7)
    assert dt.split_config(cfg) is None
    assert cfg == dict(ordinary=123)
    assert dt.plan_config() is None
    cfg = dict(dense_teacher=True, dense_teacher_fraction=.5)
    assert dt.plan_config(cfg)['dense_teacher_fraction'] == .5
    assert cfg == dict(dense_teacher=True, dense_teacher_fraction=.5)


@pytest.mark.parametrize('key,value', [
    ('dense_teacher_typo', 1), ('dense_teacher', 'false'), ('dense_teacher_random', 1),
    ('dense_teacher_fraction', -.1), ('dense_teacher_fraction', float('nan')),
    ('dense_teacher_candidates', 0), ('dense_teacher_candidates', 2.5),
    ('dense_teacher_buckets', 1), ('dense_teacher_p_star', 1.1),
    ('dense_teacher_cadence_s', 0), ('dense_teacher_cadence_s', float('inf')),
    ('dense_teacher_extra_seeds', -1), ('dense_teacher_seed', -1)])
def test_invalid_config_refused_even_off(key, value):
    with pytest.raises(ValueError):
        dt.plan_config({key: value})
    with pytest.raises(ValueError):
        dt.make_teacher({key: value}, moves_mod=Moves)


def test_factory_off_and_random_scorer_refused():
    with pytest.raises(ValueError, match='OFF'):
        dt.make_teacher({}, moves_mod=Moves)
    with pytest.raises(ValueError, match='random teacher'):
        dt.make_teacher(dict(dense_teacher=True, dense_teacher_random=True),
                        moves_mod=Moves, scorer=lambda *_: [])


def test_wm_gate_then_gain_and_candidate_order():
    scores = [dict(p_better=.99, gain=1), dict(p_better=.8, gain=10),
              dict(p_better=.79, gain=100)]
    lb = label(teacher(scorer=lambda *_: scores))
    assert lb['candidate_class'] == 2 and lb['scores'] == scores[1]
    assert lb['target']['members'] == ['a', 'b']
    assert label(teacher(scorer=lambda *_: [dict(p_better=.8, gain=2)]*3))['candidate_class'] == 1
    lb = label(teacher(scorer=lambda *_: [dict(p_better=.799, gain=100)]*3))
    assert lb['move'] is None and lb['candidate_class'] == lb['target']['action_class'] == 0


@pytest.mark.parametrize('scores', [[], [dict(p_better=1.1, gain=1)]*3,
                                    [dict(p_better=.9, gain=float('nan'))]*3])
def test_bad_scores_refused(scores):
    with pytest.raises(ValueError):
        label(teacher(scorer=lambda *_: scores))


def test_dependencies_only_receive_observables():
    seen = []
    def score(obs, cands):
        seen.append(obs)
        assert obs == dict(dt.observable_state(observation()), cav='a', self='a')
        return [dict(p_better=.9, gain=1)]*len(cands)
    lb = label(teacher(scorer=score))
    assert seen and 'forbidden' not in json.dumps(lb)


def test_identical_candidate_distribution_across_arms_and_repeatable_random_labels():
    b, r, r2 = teacher(), teacher(random=True), teacher(random=True)
    draws = []
    for i in range(80):
        bl, rl, rl2 = label(b, i), label(r, i), label(r2, i)
        assert bl['candidates'] == rl['candidates'] == rl2['candidates']
        assert bl['support_sha256'] == rl['support_sha256']
        assert rl == rl2
        assert rl['candidate_class'] == rl['draw']
        draws.append(rl['draw'])
    assert set(draws) == {0, 1, 2, 3}


def test_none_and_empty_support_do_not_score():
    stub = SimpleNamespace(FAMILIES=Moves.FAMILIES, candidates=lambda *_: [])
    t = dt.WMDenseTeacher(stub, scorer=lambda *_: pytest.fail('empty support scored'))
    assert label(t)['candidate_class'] == 0
    stub.candidates = lambda *_: [move('none')]
    assert label(t)['candidate_class'] == 0


@pytest.mark.parametrize('members', [('a', 'h'), ('a', 'invisible'), ('b',), ('a', 'a')])
def test_members_are_visible_cavs_including_self(members):
    stub = SimpleNamespace(FAMILIES=Moves.FAMILIES, candidates=lambda *_: [move(members=members)])
    with pytest.raises(ValueError, match='candidate'):
        label(dt.WMDenseTeacher(stub, scorer=lambda *_: []))


def test_move_attribute_seam_and_encoder():
    encoder = dt.TargetEncoder(('gap_open',), buckets=2,
                               windows=dict(onset_s=(0, 10), hold_s=(0, 80), depth=(0, 1)))
    target = encoder.encode(SimpleNamespace(**move(depth=1.)))
    assert target == dict(action_class=8, family='gap_open', family_index=0,
                          buckets=[1, 1, 1], members=['a', 'b'])
    assert encoder.n_classes == 9 and encoder.encode(None)['action_class'] == 0
    with pytest.raises(ValueError):
        encoder.encode(move('unknown'))


def test_ce_stable_and_exact_candidate_members_preserved():
    assert dt.cross_entropy([10000., 10000.], 0) == pytest.approx(np.log(2))
    assert dt.softmax([10000., 10001.]).sum() == pytest.approx(1)
    t = teacher()
    lb = label(t)
    ds = dt.build_dataset([lb], t.encoder)
    row = ds['targets'][0]
    assert row['candidate_class'] == 3
    assert row['candidates'][row['candidate_class']-1] == lb['move']
    assert row['members'] == lb['move']['members']
    assert row['obs_state'] == lb['obs_state']
    assert ds['counts']['wm'] == ds['counts']['moves'] == 1


@pytest.mark.parametrize('logits,index', [([], 0), ([[1, 2]], 0), ([1, float('nan')], 0),
                                       ([1, 2], -1), ([1, 2], 2), ([1, 2], .5)])
def test_bad_ce_inputs_refused(logits, index):
    with pytest.raises(ValueError):
        dt.cross_entropy(logits, index)


def test_dagger_current_rollout_snapshots_cadence_dedup_and_incremental_labels():
    buffer = dt.DAggerBuffer(cadence_s=60)
    obs = observation()
    assert not buffer.add('ep', 59, 59., 'a', obs)
    assert not buffer.add('ep', 60, 60.5, 'a', obs)
    assert buffer.add('ep', 60, 60., 'a', obs)
    assert not buffer.add('ep', 60, 60., 'a', obs)
    obs['rows'][0]['v_mps'] = 1000.
    t = teacher()
    first = buffer.label(t)
    assert first[0]['obs_rows'][0]['v_mps'] == 8.
    assert buffer.label(t) == first
    assert buffer.add('ep', 120, 120., 'a', obs)
    assert buffer.label(t)[0] is first[0]
    assert len(buffer.label(t)) == 2
    assert dt.at_decision_point(1., .5) and not dt.at_decision_point(1.1, .5)


@pytest.mark.parametrize('cadence', [0, -1, float('inf')])
def test_bad_cadence_refused(cadence):
    with pytest.raises(ValueError):
        dt.DAggerBuffer(cadence)


def test_verification_fraction_deterministic_order_independent_and_boundaries():
    labels = [dict(id=str(i)) for i in range(1000)]
    first = {r['id'] for r in dt.select_for_verification(labels, .25)}
    assert 210 <= len(first) <= 290
    assert first == {r['id'] for r in dt.select_for_verification(labels[::-1], .25)}
    assert not dt.select_for_verification(labels, 0)
    assert len(dt.select_for_verification(labels, 1)) == len(labels)


def test_dataset_seal_and_byte_stability(tmp_path):
    t = teacher()
    ds = dt.build_dataset([label(t)], t.encoder)
    p = tmp_path/'dense.json'
    dt.save_dataset(ds, p)
    before = p.read_bytes()
    assert dt.load_dataset(p) == ds
    dt.save_dataset(ds, p)
    assert p.read_bytes() == before
    blob = json.loads(before)
    blob['targets'][0]['candidate_class'] = 0
    p.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match='seal'):
        dt.load_dataset(p)


def test_quote_is_all_candidates_admitted_upper_bound():
    from tools.coopmwm import verify_perseed as vp
    lb = bank_label()
    dt.select_for_verification([lb], 1.)
    q = dt.verification_quote([lb], k=3, duration_s=120, members=2, extra_seeds=1)
    expected = vp.quote_decision(600, [120]*3, [['a', 'b']]*3,
                                 {1: [0, 1, 2], 2: [0, 1, 2]}, extra_seeds=1)
    assert q['all8_total_with_extra_sim_s'] == expected['all8_total_with_extra_sim_s']
    assert q['n_selected'] == 1
    assert dt.verification_quote([], 3)['all8_total_sim_s'] == 0


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_selected_all8_end_to_end_and_no_redispatch(tmp_path, arm):
    # Existing deterministic plant exercises real all8 seed-2, later-harm, solo, sealed-record
    # and ledger paths. It never invokes SUMO or a subprocess.
    from tools.coopmwm.test_verify_perseed_cpu import _budget
    _budget(tmp_path)
    gains = {('p1', 1): .6, ('p1', 2): .6, ('p1', 1, 'solo'): .1, ('p1', 2, 'solo'): .1}
    root, _, plans, receipt, runner_for, _, lb = dense_setup(tmp_path, gains, k=2, arm=arm)
    t = teacher()
    assert lb['source'] == ('wm' if arm == 'B' else 'random')
    verifier = dt.All8Verifier(root, arm, runner_for, str(root), 'bank_verification_ext')
    calls = []
    def dispatch(row):
        calls.append(row['id'])
        return receipt, 0, plans[0]
    out = dt.verify_selected([lb], 1., dispatch, verifier)
    assert calls == [lb['id']]
    assert out['sealed'] == out['replicated'] == 1
    assert lb['verified'] and lb['verified_replicated'] and lb['verified_cooperative']
    assert lb['verified_admitted_seeds'] == [1, 2]
    again = dt.verify_selected([lb], 1., dispatch, verifier)
    assert out == again and calls == [lb['id']]
    ds = dt.build_dataset([lb], t.encoder)
    assert ds['counts']['selected'] == ds['counts']['verified'] == ds['counts']['replicated'] == 1


def test_all8_first_seed_rejected_second_seed_wins_and_none_label(tmp_path):
    from tools.coopmwm.test_verify_perseed_cpu import _budget
    _budget(tmp_path)
    root, snap, plans, receipt, runner_for, _, lb = dense_setup(tmp_path, {('p2', 2): .7}, k=2)
    assert receipt['status'] == 'rejected'
    dt.select_for_verification([lb], 1.)
    dt.bind_receipt(lb, receipt, 1, plans[1])
    none = bank_label(none=True)
    dt.prepare_verification_plan(none, snap)
    dt.select_for_verification([none], 1.)
    dt.bind_receipt(none, receipt, None)
    verifier = dt.All8Verifier(root, 'B', runner_for, str(root), 'bank_verification_ext')
    verifier.flush()
    out = verifier.consume([lb, none])
    assert out['sealed'] == 2
    assert lb['verified_admitted_seeds'] == [2] and not lb['verified_replicated']
    assert none['verified'] and none['verified_flip'] and not none['verified_replicated']


def test_binding_rejects_unselected_wrong_identity_and_wrong_move(tmp_path):
    _, _, plans, receipt, _, _, lb = dense_setup(tmp_path, {}, k=1)
    with pytest.raises(ValueError, match='selected'):
        dt.bind_receipt(lb, receipt, 0, plans[0])
    dt.select_for_verification([lb], 1.)
    with pytest.raises(ValueError, match='sealed receipt'):
        dt.bind_receipt(lb, receipt, 0, dict(plans[0], id='different'))
    with pytest.raises(ValueError, match='outside receipt'):
        dt.bind_receipt(lb, receipt, 4, plans[0])
    lb['move']['depth'] = .99
    with pytest.raises(ValueError, match='labelled move'):
        dt.bind_receipt(lb, receipt, 0, plans[0])
    broken = copy.deepcopy(receipt)
    broken['status'] = 'changed'
    with pytest.raises(ValueError):
        dt.bind_receipt(lb, broken, 0, plans[0])


def test_no_selected_labels_never_dispatch_or_flush():
    verifier = SimpleNamespace(flush=lambda: pytest.fail('nothing selected'),
                               consume=lambda _: pytest.fail('nothing selected'))
    out = dt.verify_selected([label()], 0., lambda _: pytest.fail('dispatch'), verifier)
    assert out['labels'] == 0


def test_lazy_planner_import_seam(monkeypatch):
    from tools import coopmwm
    from dataclasses import dataclass
    @dataclass
    class Config:
        n_candidates: int = 32
        horizon_s: int = 600
    calls = []
    def scorer(record, rng, *, rollout, calibration, config, calibrate, move_api):
        assert not {'cav', 'self'} & set(record)
        assert config.horizon_s == 600
        candidates = move_api.candidates(record, rng, config.n_candidates)
        assert len(candidates) == config.n_candidates == 3
        assert rollout == 'cpu-rollout' and calibration == 'fitted'
        calls.append(candidates)
        return dict(scores=[dict(p_calibrated=.9, expected_gain=i+1)
                            for i in range(len(candidates))])
    monkeypatch.setattr(coopmwm, 'planner_online', SimpleNamespace(score_moves=scorer, Config=Config),
                        raising=False)
    api = SimpleNamespace(FAMILIES=Moves.FAMILIES, offsets=lambda *_: {})
    kwargs = dict(rollout='cpu-rollout', calibration='fitted', move_api=api)
    t = teacher(scorer=None, planner_kwargs=kwargs)
    lb = label(t)
    assert calls == [lb['candidates']] and lb['candidate_class'] == 3
    with pytest.raises(ValueError, match='explicit CPU rollout'):
        dt.planner_scorer()


def test_recorded_trigger_table_labels_and_ce_targets_without_redrawing():
    api = SimpleNamespace(FAMILIES=Moves.FAMILIES,
                          candidates=lambda *_: pytest.fail('recorded table was redrawn'))
    t = dt.WMDenseTeacher(api, scorer=lambda obs, c: [dict(p_better=.9, gain=2)]*len(c))
    table = [None, move(), move(depth=.3), None]
    lb = t.label_state('ep', 0, 600, 'a', observation(), candidates=table)
    batch = SimpleNamespace(context=dict(members=['a']), tables=(tuple(table),),
                            mask=np.asarray([[True, True, True, False]]))
    assert dt.trigger_targets([lb], batch) == [1]
    reordered = [None, table[2], table[1], None]
    batch.tables = (tuple(reordered),)
    with pytest.raises(ValueError, match='support differs'):
        dt.trigger_targets([lb], batch)
    batch.tables = (tuple(table),)
    batch.mask[0, 1] = False
    with pytest.raises(ValueError, match='masked'):
        dt.trigger_targets([lb], batch)


def test_dagger_captures_recorded_table_and_move_parameter_aliases():
    table = [None, dict(family='gap_open', members=['a', 'b'], onset_s=5, hold_s=40, depth=.2)]
    buf = dt.DAggerBuffer(60)
    buf.add('ep', 0, 600, 'a', observation(), candidates=table)
    table[1]['depth'] = .8
    lb = buf.label(teacher())[0]
    assert lb['move']['depth'] == .2
    assert lb['move']['onset'] == 5 and lb['move']['hold'] == 40


def test_planner_belief_payload_and_history_preserved_without_hidden_rows():
    obs = observation()
    obs.update(group=['a', 'b'], commands=dict(a=.1, b=.2), policy_residuals=dict(a=0, b=0),
               policy_state={'actor.test': [1, 2]}, committed_offsets=[dict(b=-.1)])
    obs['history'] = [dict(time_s=600, rows=copy.deepcopy(obs['rows']), entrance_counts=[1, 2])]
    result = dt.observable_state(obs)
    assert result['policy_state'] == obs['policy_state']
    assert result['commands'] == obs['commands']
    assert 'forbidden' not in json.dumps(result)


def test_nonpositive_expected_gain_always_labels_none():
    assert label(teacher(scorer=lambda *_: [dict(p_better=1., gain=0)]*3))['move'] is None


@pytest.mark.parametrize('failure', ['selected_both', 'selected_seed2', 'policy_seed2'])
def test_failed_measurements_never_certify_move_or_none(tmp_path, monkeypatch, failure):
    from tools.coopmwm import test_verify_perseed_cpu as fixture
    fixture._budget(tmp_path)
    original = fixture.SeedPlant.run
    def run(plant, spec, snap):
        result = original(plant, spec, snap)  # persist specs/tapes, then inject branch failure
        seed, plan = fixture.seed_of(snap), spec['plan']
        selected = plan is not None and plan['id'] == 'p1'
        fail = ((failure == 'selected_both' and selected)
                or (failure == 'selected_seed2' and selected and seed == 2)
                or (failure == 'policy_seed2' and plan is None and seed == 2))
        if fail:
            raise RuntimeError('injected measurement failure')
        return result
    monkeypatch.setattr(fixture.SeedPlant, 'run', run)
    root, snap, plans, receipt, runner_for, _, lb = dense_setup(tmp_path, {})
    assert receipt['status'] == 'rejected'  # other candidate completed despite p1 failure
    none = bank_label(none=True)
    dt.prepare_verification_plan(none, snap)
    for row in (lb, none):
        dt.select_for_verification([row], 1.)
    dt.bind_receipt(lb, receipt, 0, plans[0])
    dt.bind_receipt(none, receipt, None)
    if failure == 'selected_both':
        assert lb['verification_seed_status']['1']['branches'][1]['status'] == 'failed'
    verifier = dt.All8Verifier(root, 'B', runner_for, str(root), 'bank_verification_ext')
    verifier.flush()
    out = verifier.consume([lb, none])
    assert out['sealed'] == 2 and out['verified'] == 0
    assert out['missing'] == out['incomplete'] == 2
    assert out['missing_seed_measurements'] == (4 if failure == 'selected_both' else 2)
    for row in (lb, none):
        assert not row['verified'] and not row['verified_replicated']
        assert row['verification_status'] == 'incomplete'
        bad = row['verification_seed_status']['2']
        assert not bad['measured'] and bad['examples'][0]['status'] == 'data_missing'
        assert not dt.verify_selected([row], 1., lambda _: pytest.fail('implicit retry'),
                                      verifier)['verified']
    counts = dt.build_dataset([lb, none], teacher().encoder)['counts']
    assert counts['selected'] == counts['incomplete'] == 2 and counts['verified'] == 0


def test_measured_rejections_are_completed_verification(tmp_path):
    from tools.coopmwm.test_verify_perseed_cpu import _budget
    _budget(tmp_path)
    root, snap, plans, receipt, runner_for, _, lb = dense_setup(tmp_path, {})
    none = bank_label(none=True)
    dt.prepare_verification_plan(none, snap)
    dt.select_for_verification([lb, none], 1.)
    dt.bind_receipt(lb, receipt, 0, plans[0])
    dt.bind_receipt(none, receipt, None)
    verifier = dt.All8Verifier(root, 'B', runner_for, str(root), 'bank_verification_ext')
    verifier.flush()
    out = verifier.consume([lb, none])
    assert out['verified'] == out['sealed'] == 2 and out['incomplete'] == out['missing'] == 0
    assert all(r['verified'] and not r['verified_replicated'] for r in (lb, none))
    assert all(s['measured'] for r in (lb, none) for s in r['verification_seed_status'].values())


@pytest.mark.parametrize('defect', ['absent_example', 'no_advantage', 'nonfinite_advantage',
                                  'ungraded_tape', 'no_later_harm', 'no_reference'])
def test_incomplete_seed2_record_cannot_certify_or_retain_stale_success(tmp_path, monkeypatch, defect):
    from tools.coopmwm import verify_perseed as vp
    from tools.coopmwm.test_verify_perseed_cpu import _budget
    _budget(tmp_path)
    root, snap, plans, receipt, runner_for, _, lb = dense_setup(tmp_path, {})
    none = bank_label(none=True)
    dt.prepare_verification_plan(none, snap)
    dt.select_for_verification([lb, none], 1.)
    dt.bind_receipt(lb, receipt, 0, plans[0])
    dt.bind_receipt(none, receipt, None)
    verifier = dt.All8Verifier(root, 'B', runner_for, str(root), 'bank_verification_ext')
    verifier.flush()
    assert verifier.consume([lb, none])['verified'] == 2
    records = copy.deepcopy(vp.records(root, 'B'))
    rec = records[lb['verification_id']]
    examples = rec['per_seed_v1']['examples']
    selected = next(e for e in examples if e['candidate_index'] == 0 and e['seed_index'] == 2)
    if defect == 'absent_example':
        examples.remove(selected)
    elif defect == 'no_advantage':
        selected.pop('advantage_veh_s')
    elif defect == 'nonfinite_advantage':
        selected['advantage_veh_s'] = float('nan')
    elif defect == 'ungraded_tape':
        selected['tape']['branch']['status'] = 'failed'
    elif defect == 'no_later_harm':
        selected['lower95_one_sided_veh_s'] = 1.
        selected['later_harm_missing'] = True
    else:
        rec['seed2']['J_policy_veh_s'] = None
    monkeypatch.setattr(vp, 'records', lambda *_: records)
    out = verifier.consume([lb, none])
    assert out['verified'] == 0 and out['incomplete'] == out['missing_seed_measurements'] == 2
    assert not lb['verified'] and not none['verified']
    assert lb['verification_seed_status']['1']['measured']
    assert not lb['verification_seed_status']['2']['measured']


def test_explicit_recovery_uses_new_receipt_and_preserves_failures(tmp_path, monkeypatch):
    from tools.coopmwm import test_verify_perseed_cpu as fixture, replay_bank as rb, sumo_verify as sv
    fixture._budget(tmp_path)
    original = fixture.SeedPlant.run
    def run(plant, spec, snap):
        result = original(plant, spec, snap)
        if spec['plan'] is not None and spec['plan']['id'] == 'p1' and fixture.seed_of(snap) == 2:
            raise RuntimeError('selected seed 2 failed')
        return result
    monkeypatch.setattr(fixture.SeedPlant, 'run', run)
    root, snap, plans, receipt, runner_for, _, lb = dense_setup(tmp_path, {})
    verifier = dt.All8Verifier(root, 'B', runner_for, str(root), 'bank_verification_ext')
    out = dt.verify_selected([lb], 1., lambda _: (receipt, 0, plans[0]), verifier)
    assert out['incomplete'] == 1
    old_vid = lb['verification_id']
    before = copy.deepcopy(lb)
    with pytest.raises(ValueError, match='fresh verification id'):
        dt.verify_selected([lb], 1., lambda _: pytest.fail('normal redispatch'), verifier,
                           recover_incomplete=lambda _: (receipt, 0, plans[0]))
    assert lb == before
    def recovery(row):
        plan = dt.prepare_verification_plan(row, snap, [[.3]*120]*2,
                                            moves_mod=Moves, plan_id='p1_retry')
        bank = rb.bank_root(root, 'B')
        work = bank/'sumo_verification'
        new = sv.verify(snap, [plan], fixture.SeedPlant(work, {}),
                        sv.local_charge(tmp_path/'recovery_charge'), 'wm', root=work)
        (bank/'decisions'/'retry.json').write_text(json.dumps(dict(
            state_id='s1', verification_id=new['verification_id'], status=new['status'], policy_update=5)))
        return new, 0, plan
    result = dt.verify_selected([lb], 1., lambda _: pytest.fail('normal redispatch'), verifier,
                                recover_incomplete=recovery)
    assert result['verified'] == 1 and result['incomplete'] == 0
    assert lb['verification_id'] != old_vid
    history = lb['verification_history'][0]
    assert history['verification_id'] == old_vid
    assert not history['verification_seed_status']['2']['measured']
    dt.verify_selected([lb], 1., lambda _: pytest.fail('redispatch'), verifier,
                       recover_incomplete=lambda _: pytest.fail('completed recovery'))


def nearest_box_projection(plan):
    """Contract-faithful T1A seam: least-squares single-actor box on registered windows.

    The test has a fixed family/member support; optimize onset, hold and depth over
    the entire allowed grid, rather than returning the teacher's requested move.
    """
    offsets = np.asarray(plan['profile'])-np.asarray(plan['policy_profile'])
    windows = dt.default_windows()
    best = None
    for onset in range(int(windows['onset_s'][0]), int(windows['onset_s'][1])+1):
        for hold in range(int(windows['hold_s'][0]), int(windows['hold_s'][1])+1):
            signal = np.zeros(max(offsets.shape[1], onset+hold))
            signal[:offsets.shape[1]] = offsets[0]
            depth = float(np.clip(-signal[onset:onset+hold].mean(), *windows['depth']))
            fit = np.zeros_like(signal)
            fit[onset:onset+hold] = -depth
            error = float(np.square(signal-fit).sum())
            choice = (error, onset, hold, depth)
            if best is None or choice < best:
                best = choice
    _, onset, hold, depth = best
    return dict(family=plan['family'], members=plan['members'], onset=onset, hold=hold, depth=depth)


def test_same_nearest_move_different_executed_profile_is_rejected(tmp_path):
    from tools.coopmwm.exact_grade import _seal
    root, snap, plans, receipt, _, _, lb = dense_setup(tmp_path, {}, k=1)
    original = plans[0]
    changed = copy.deepcopy(original)
    changed['profile'][0][10] += .1
    changed['profile'][0][11] -= .1
    changed['perturbation_profile'][0][10] += .1
    changed['perturbation_profile'][0][11] -= .1
    projected = nearest_box_projection(changed)
    wanted = nearest_box_projection(original)
    assert projected['family'] == wanted['family'] == lb['move']['family']
    assert projected['members'] == wanted['members'] == lb['move']['members']
    assert [projected[k] for k in ('onset', 'hold', 'depth')] == pytest.approx(
        [lb['move'][k] for k in ('onset', 'hold', 'depth')])
    assert digest(original) != digest(changed)
    changed_receipt = _seal(dict(receipt, candidate_plan_sha256=[digest(changed)]), 'receipt_sha256')
    dt.select_for_verification([lb], 1.)
    with pytest.raises(ValueError, match='exact labelled move'):
        dt.bind_receipt(lb, changed_receipt, 0, changed,
                        moves_mod=SimpleNamespace(from_plan=nearest_box_projection))
    dt.bind_receipt(lb, receipt, 0, original)  # the exact plan still binds


@pytest.mark.parametrize('field', ['policy_profile', 'perturbation_profile', 'dense_execution',
                                 'duration_s', 'members', 'overlap_truncated_plan_ids'])
def test_execution_metadata_tampering_rejected_even_with_valid_receipt_seal(tmp_path, field):
    from tools.coopmwm.exact_grade import _seal
    _, _, plans, receipt, _, _, lb = dense_setup(tmp_path, {}, k=1)
    changed = copy.deepcopy(plans[0])
    if field in ('policy_profile', 'perturbation_profile'):
        changed[field][0][0] += .1
    elif field == 'dense_execution':
        changed[field]['policy_sha256'] = 'another-policy'
    elif field == 'duration_s':
        changed[field] -= 1
    elif field == 'members':
        changed[field].reverse()
    else:
        changed[field] = ['another-plan']
    r = _seal(dict(receipt, candidate_plan_sha256=[digest(changed)]), 'receipt_sha256')
    dt.select_for_verification([lb], 1.)
    with pytest.raises(ValueError, match='exact labelled move'):
        dt.bind_receipt(lb, r, 0, changed)


def test_execution_contract_full_offsets_base_identity_and_preparation(tmp_path):
    from tools.coopmwm.exact_grade import _seal
    _, snap, plans, receipt, _, _, lb = dense_setup(tmp_path, {}, k=1)
    contract = lb['verification_execution']
    assert contract['duration_s'] == 120
    assert contract['offset_profile'][0] == [0.]*5+[-.6]*40+[0.]*75
    assert contract['offset_profile'][1] == [0.]*120
    assert np.asarray(plans[0]['profile']) == pytest.approx(
        np.asarray(contract['base_profile'])+np.asarray(contract['offset_profile']))
    dt.select_for_verification([lb], 1.)
    changed = _seal(dict(receipt, snapshot_sha256='another-base-policy-snapshot'), 'receipt_sha256')
    with pytest.raises(ValueError, match='snapshot/base policy'):
        dt.bind_receipt(lb, changed, 0, plans[0])
    with pytest.raises(ValueError, match='full labelled move'):
        dt.prepare_verification_plan(lb, snap, [[.3]*44]*2, moves_mod=Moves)
    with pytest.raises(ValueError, match='already frozen'):
        dt.prepare_verification_plan(lb, snap, [[.4]*120]*2, moves_mod=Moves)
    another_snapshot = copy.deepcopy(snap)
    another_snapshot['policy']['state_dict_sha256'] = 'another-policy'
    with pytest.raises(ValueError, match='already frozen'):
        dt.prepare_verification_plan(lb, another_snapshot, [[.3]*120]*2, moves_mod=Moves)
    missing = copy.deepcopy(lb)
    missing.pop('verification_execution')
    with pytest.raises(ValueError, match='before dispatch'):
        dt.bind_receipt(missing, receipt, 0, plans[0])
