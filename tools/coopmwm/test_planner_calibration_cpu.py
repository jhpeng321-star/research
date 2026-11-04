"""T1C CPU contract tests: actual verifier receipts, fake plant, no simulation."""
import copy
from dataclasses import asdict, replace
import json
import xml.etree.ElementTree as ET

import pytest

from tools.coopmwm import planner_calibration as pc
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.exact_grade import _seal
from tools.coopmwm.test_sumo_verify_cpu import FakePlant, plan, snapshot

CONTRACT = pc.GateContract(300, 'team_delay_vs', 'MAPPO-only', 'p')


def evidence(snap, candidate, samples=None, *, contract=CONTRACT):
    samples = samples if samples is not None else [('a', 10., 9.), ('b', 10., 11.)]
    samples = [list(s) for s in samples]
    return _seal(dict(schema=pc.EVIDENCE_SCHEMA, paired=True, n_samples=len(samples), samples=samples,
                      p_better=sum(s[1] > s[2] for s in samples)/len(samples),
                      candidate_id=candidate['id'], snapshot_sha256=sv.snapshot_sha(snap),
                      start_s=snap['t0'], duration_s=candidate['duration_s'], contract=asdict(contract)),
                 'evidence_sha256')


def receipt(root, seed=986000, *, reject=False, replica=False, fail=(), fallback=False,
            probability=True, tail_s=180, plant=None, probability_samples=None):
    snap = snapshot(world=seed)
    if replica:
        snap['rng'] = dict(snap['rng'], replication_salt='independent-post-t0')
    plans = [plan('win', gain=-1. if reject else 1.), plan('loss', gain=-1.), plan('tie', gain=0.)]
    # WM predictions can be wrong; labels must come from branch J, never predictions.
    plans[1]['saving_vs_policy'] = 3.
    if probability:
        for p in plans:
            p['wm_probability_evidence'] = evidence(snap, p, (probability_samples or {}).get(p['id']))
    if fallback:
        for p in plans:
            p.pop('saving_vs_policy')
    specs = sv.build_branches(snap, plans, tail_s=tail_s)[1:]
    r = sv.verify(snap, plans, plant or FakePlant(fail=fail), lambda *args: None, root=root, tail_s=tail_s)
    for i, spec in enumerate(specs, 1):
        p = root/'branches'/r['verification_id']/f'cand{i}.spec.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec), encoding='utf-8')
    return root/'verifications'/f'{r["verification_id"]}.json', r, specs


def test_actual_receipt_all_candidates_and_strict_ties(tmp_path):
    path, r, specs = receipt(tmp_path/'sumo_verification')
    rows, skipped = pc.examples_from_receipt(r, specs, [986000])
    assert [x.wm_gain_veh_s for x in rows] == [1., 3., 0.]
    assert [x.realised_gain_veh_s for x in rows] == [10., -10., 0.]
    assert [x.better for x in rows] == [1, 0, 0]
    assert skipped == {}
    loaded, report = pc.load_verified_records([path, path], [986000])
    assert loaded == [replace(x, replica='unregistered-direct') for x in rows]
    assert len(report['sources']) == 1


def test_rejected_decisions_are_not_filtered(tmp_path):
    _, r, specs = receipt(tmp_path/'sumo_verification', reject=True)
    assert r['status'] == 'rejected' and r['grade_veh_s'] is None
    rows, _ = pc.examples_from_receipt(r, specs, [986000])
    assert len(rows) == 3 and [x.better for x in rows] == [0, 0, 0]


def test_ungraded_and_legacy_wrong_comparator_never_imputed(tmp_path):
    _, r, specs = receipt(tmp_path/'missing', fail=('cand2',))
    rows, skipped = pc.examples_from_receipt(r, specs, [986000])
    assert len(rows) == 2 and skipped == {'ungraded_candidate': 1}
    _, r, specs = receipt(tmp_path/'fallback', fallback=True)
    rows, skipped = pc.examples_from_receipt(r, specs, [986000])
    assert rows == [] and skipped == {'missing_own_policy_prediction': 3}


@pytest.mark.parametrize('seed', [962000, 962999, 964000, 964199, 977000, 977999])
def test_eval_ranges_refused_before_any_read(seed, monkeypatch):
    def no_read(*args, **kwargs):
        pytest.fail('input opened before seed allowlist validation')
    monkeypatch.setattr(pc, '_read', no_read)
    with pytest.raises(ValueError, match='evaluation/sealed'):
        pc.load_verified_records(['not-needed.json'], [seed])


@pytest.mark.parametrize('seed', [961999, 963000, 964200, 976999, 978000, 986000])
def test_seed_range_boundaries(seed):
    assert pc.development_seed(seed) == seed


@pytest.mark.parametrize('part', pc.BLOCKED_PATH_PARTS)
def test_forbidden_paths_refused_without_read(part, monkeypatch, tmp_path):
    def no_read(*args, **kwargs):
        pytest.fail('forbidden path was opened')
    monkeypatch.setattr(pc, '_read', no_read)
    with pytest.raises(ValueError, match='forbidden'):
        pc.load_verified_records([tmp_path/part/'record.json'], [986000])


@pytest.mark.parametrize('role', ['check', 'eval', 'final_test', 'sealed', None])
def test_non_development_roles_refused_even_if_seed_allowlisted(tmp_path, role):
    _, r, specs = receipt(tmp_path/'sumo_verification')
    r['episode']['role'] = role
    r = _seal(r, 'receipt_sha256')
    with pytest.raises(ValueError, match='development'):
        pc.examples_from_receipt(r, specs, [986000])


def test_receipt_seed_must_be_allowlisted_and_eval_is_always_refused(tmp_path):
    _, r, specs = receipt(tmp_path/'sumo_verification')
    with pytest.raises(ValueError, match='development'):
        pc.examples_from_receipt(r, specs, [986001])
    r['episode']['world']['seed'] = 964100
    r = _seal(r, 'receipt_sha256')
    with pytest.raises(ValueError, match='evaluation/sealed'):
        pc.examples_from_receipt(r, specs, [986000])


def test_integrity_and_prediction_identity(tmp_path):
    _, r, specs = receipt(tmp_path/'sumo_verification')
    bad = copy.deepcopy(r)
    bad['predicted_saving'][0] = 999.
    with pytest.raises(ValueError, match='receipt_sha256'):
        pc.examples_from_receipt(bad, specs, [986000])
    bad = _seal(bad, 'receipt_sha256')
    with pytest.raises(ValueError, match='prediction differs'):
        pc.examples_from_receipt(bad, specs, [986000])
    bad_specs = copy.deepcopy(specs)
    bad_specs[0]['plan']['saving_vs_policy'] = 999.
    bad_specs[0] = _seal(bad_specs[0], 'branch_sha256')
    with pytest.raises(ValueError, match='identity mismatch'):
        pc.examples_from_receipt(r, bad_specs, [986000])


def test_all8_links_full_receipts_not_admitted_examples(tmp_path):
    p1, r1, _ = receipt(tmp_path/'sumo_verification')
    _, r2, _ = receipt(tmp_path/'confirm_seed2_all8', replica=True)
    record = dict(schema=pc.ALL8_SCHEMA, verify_mode='all8_both_seeds',
                  first_seed=dict(verification_id=r1['verification_id'], receipt_sha256=r1['receipt_sha256'],
                                  branches_dir=str(tmp_path/'sumo_verification'/'branches')),
                  seed2=dict(verification_id=r2['verification_id'], receipt_sha256=r2['receipt_sha256'],
                             root=str(tmp_path/'confirm_seed2_all8')),
                  per_seed_v1=dict(examples=[{'admitted': True}]))
    p = tmp_path/'all8.json'
    p.write_text(json.dumps(_seal(record, 'confirmation_sha256')), encoding='utf-8')
    rows, report = pc.load_verified_records([p1, p], [986000])
    assert len(rows) == 6 and len(report['sources']) == 2
    assert sum(x.better for x in rows) == 2
    record['seed2']['receipt_sha256'] = 'wrong'
    p.write_text(json.dumps(_seal(record, 'confirmation_sha256')), encoding='utf-8')
    with pytest.raises(ValueError, match='linked receipt hash'):
        pc.load_verified_records([p], [986000])


def test_isotonic_exact_pava_ties_monotonicity_and_roundtrip():
    model = pc.fit_isotonic([0., 0., 1., 2., 3.], [0, 1, 0, 1, 1])
    assert model.predict_probability(0.) == pytest.approx(1/3)
    assert model.predict_probability(1.) == pytest.approx(1/3)
    assert model.predict_probability(-100.) == pytest.approx(1/3)
    assert model.predict_probability(100.) == 1.
    preds = [model.predict_probability(x/10) for x in range(-10, 50)]
    assert preds == sorted(preds)
    restored = pc.IsotonicCurve.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model
    for value in (float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            model.predict_probability(value)


def test_reliability_hand_computed_bins_endpoints_and_empty_bins():
    report = pc.reliability([0., .2, .8, 1.], [0, 1, 0, 1], n_bins=5)
    assert report['ece'] == pytest.approx(.4)
    assert report['brier'] == pytest.approx(.32)
    assert [b['count'] for b in report['bins']] == [1, 1, 0, 0, 2]
    assert report['bins'][2]['mean_probability'] is None
    assert report['bins'][-1]['mean_probability'] == .9
    assert ET.fromstring(pc.reliability_svg(report)).tag.endswith('svg')


def dataset(tmp_path):
    paths = []
    for s in range(986000, 986004):
        p1, r1, _ = receipt(tmp_path/f'world{s}'/'sumo_verification', seed=s)
        _, r2, _ = receipt(tmp_path/f'world{s}'/'seed2', seed=s, replica=True)
        paths.append(all8(tmp_path/f'world{s}', r1, r2))
    return paths, pc.load_verified_records(paths, range(986000, 986004))[0]


def all8(root, r1, r2, extra=None):
    record = dict(schema=pc.ALL8_SCHEMA, verify_mode='all8_both_seeds',
                  first_seed=dict(verification_id=r1['verification_id'], receipt_sha256=r1['receipt_sha256'],
                                  branches_dir=str(root/'sumo_verification'/'branches')),
                  seed2=dict(verification_id=r2['verification_id'], receipt_sha256=r2['receipt_sha256'],
                             root=str(root/'seed2')), extra=extra or {})
    p = root/'all8.json'
    p.write_text(json.dumps(_seal(record, 'confirmation_sha256')), encoding='utf-8')
    return p


def test_seed_split_replicas_reordering_and_validation_labels_cannot_affect_fit(tmp_path):
    _, rows = dataset(tmp_path)
    seeds = range(986000, 986004)
    fit, val = pc.split_by_seed(rows, seeds)
    assert {x.world_seed for x in fit}.isdisjoint({x.world_seed for x in val})
    a = pc.fit_calibration(rows, seeds, planner_contract=CONTRACT)
    b = pc.fit_calibration(list(reversed(rows)), seeds, planner_contract=CONTRACT)
    assert a == b
    changed = [replace(x, realised_gain_veh_s=0.)
               if x.world_seed in a['split']['validation_seeds'] else x for x in rows]
    c = pc.fit_calibration(changed, seeds, planner_contract=CONTRACT)
    assert a['model'] == c['model'] and a['validation'] != c['validation']
    assert a['fit']['n']+a['validation']['n'] == len(rows)


def test_small_or_single_class_data_refused(tmp_path):
    _, r, specs = receipt(tmp_path/'sumo_verification')
    rows, _ = pc.examples_from_receipt(r, specs, [986000])
    rows = [replace(x, population=pc.PRIMARY_POPULATION) for x in rows]
    with pytest.raises(ValueError, match='two development'):
        pc.fit_calibration(rows, [986000], planner_contract=CONTRACT)
    rows += [replace(x, world_seed=986001) for x in rows]
    rows = [replace(x, realised_gain_veh_s=0.) for x in rows]
    with pytest.raises(ValueError, match='both wins'):
        pc.fit_calibration(rows, [986000, 986001], planner_contract=CONTRACT)


def test_cli_json_svg_artifact_loading_and_no_overwrite(tmp_path):
    paths, _ = dataset(tmp_path)
    out, svg = tmp_path/'model.json', tmp_path/'reliability.svg'
    contract_path = tmp_path/'contract.json'
    contract_path.write_text(json.dumps(asdict(CONTRACT)))
    args = ['--records', *map(str, paths), '--dev-seeds', *map(str, range(986000, 986004)),
            '--output', str(out), '--reliability-svg', str(svg), '--planner-contract', str(contract_path)]
    artifact = pc.main(args)
    assert pc.load_calibrator(out, planner_contract=CONTRACT).to_dict() == pc.Calibrator.from_dict(artifact['model']).to_dict()
    assert ET.fromstring(svg.read_text()).tag.endswith('svg')
    assert artifact['provenance']['n_examples'] == 24
    with pytest.raises(ValueError, match='new file'):
        pc.main(args)
    saved = json.loads(out.read_text())
    saved['model']['curve']['probabilities'][0] = .123
    out.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='artifact_sha256'):
        pc.load_calibrator(out, planner_contract=CONTRACT)


def test_artifact_refuses_valid_hash_with_seed_leakage(tmp_path):
    _, rows = dataset(tmp_path)
    a = pc.fit_calibration(rows, range(986000, 986004), planner_contract=CONTRACT)
    a['split']['validation_seeds'] = a['split']['fit_seeds']
    p = tmp_path/'model.json'
    p.write_text(json.dumps(_seal(a, 'artifact_sha256')))
    with pytest.raises(ValueError, match='seed leakage'):
        pc.load_calibrator(p, planner_contract=CONTRACT)


def predict(model, prediction, **overrides):
    args = dict(planner_contract=CONTRACT, candidate_id=prediction.candidate_id,
                snapshot_sha256=prediction.snapshot_sha256, start_s=prediction.start_s)
    return model.predict_probability(prediction, **dict(args, **overrides))


def test_equal_mean_gains_different_paired_win_probabilities(tmp_path):
    # Both sample distributions have mean gain 1, but P(better) is .25 vs .75.
    low = [(str(i), 10., 10.-g) for i, g in enumerate([7., -1., -1., -1.])]
    high = [(str(i), 10., 10.-g) for i, g in enumerate([2., 2., 2., -2.])]
    assert sum(s[1]-s[2] for s in low) == sum(s[1]-s[2] for s in high)
    paths = []
    for seed in range(986000, 986004):
        root = tmp_path/str(seed)
        _, r1, _ = receipt(root/'sumo_verification', seed=seed,
                           probability_samples={'win': high, 'loss': low, 'tie': low})
        _, r2, _ = receipt(root/'seed2', seed=seed, replica=True,
                           probability_samples={'win': high, 'loss': low, 'tie': low})
        paths.append(all8(root, r1, r2))
    rows, _ = pc.load_verified_records(paths, range(986000, 986004))
    # Deliberately equal diagnostic mean scores: fitting must still distinguish P(better).
    rows = [replace(x, wm_gain_veh_s=1.) for x in rows]
    artifact = pc.fit_calibration(rows, range(986000, 986004), planner_contract=CONTRACT)
    model = pc.Calibrator.from_dict(artifact['model'])
    win, loss = [pc.WMProbability.from_dict(rows[i].wm_prediction) for i in (0, 1)]
    assert predict(model, win) == 1. and predict(model, loss) == 0.
    for scalar in (.8, 1., -10.):
        with pytest.raises(ValueError, match='typed WMProbability'):
            model.predict_probability(scalar, planner_contract=CONTRACT, candidate_id='win',
                                      snapshot_sha256=win.snapshot_sha256, start_s=400)
    for identity in (dict(candidate_id='other'), dict(snapshot_sha256='other'), dict(start_s=401)):
        with pytest.raises(ValueError, match='identity mismatch'):
            predict(model, win, **identity)


@pytest.mark.parametrize('mutation,match', [
    ({'paired': False}, 'paired WM'),
    ({'n_samples': 1}, 'multiple paired'),
    ({'p_better': .8}, 'differs from paired'),
    ({'p_better': 1.1}, 'outside'),
    ({'samples': [('a', 10., 9.), ('a', 10., 11.)]}, 'sample identities'),
    ({'samples': [('a', 10.), ('b', 10., 11.)]}, 'sample identities'),
    ({'samples': [('a', 10., float('inf')), ('b', 10., 11.)]}, 'JSON compliant'),
    ({'start_s': 1000}, 'shortened full-horizon'),
    ({'duration_s': 300}, 'shortened full-horizon'),
])
def test_invalid_probability_evidence_refused(mutation, match):
    e = evidence(snapshot(), plan('win'))
    e.update(mutation)
    with pytest.raises(ValueError, match=match):
        pc.WMProbability.from_dict(_seal(e, 'evidence_sha256'))


def test_recorded_probability_without_samples_requires_paired_provenance():
    e = evidence(snapshot(), plan('win'))
    e.pop('samples')
    assert pc.WMProbability.from_dict(_seal(e, 'evidence_sha256')).probability == .5


def test_old_gain_only_receipts_are_unavailable_for_gate(tmp_path):
    _, r, specs = receipt(tmp_path/'sumo_verification', probability=False)
    rows, _ = pc.examples_from_receipt(r, specs, [986000])
    assert pc.gain_diagnostic(rows)['planner_gate_available'] is False
    rows = [replace(x, population=pc.PRIMARY_POPULATION) for x in rows]
    with pytest.raises(ValueError, match='missing paired WM probability'):
        pc.fit_calibration(rows, [986000], planner_contract=CONTRACT)


def test_short_window_win_full_window_loss_never_fits(tmp_path):
    class LaterLoss(FakePlant):
        def run(self, spec, snap):
            result = super().run(spec, snap)
            if spec['plan'] is not None and spec['window']['end_s'] >= 700:
                result['final']['metrics']['team_delay_vs'] = float(
                    float.fromhex(result['final']['metrics']['team_delay_vs'])+100.).hex()
            return result

    _, short, ss = receipt(tmp_path/'short', tail_s=120, plant=LaterLoss())
    _, full, fs = receipt(tmp_path/'full', tail_s=180, plant=LaterLoss())
    short_rows, _ = pc.examples_from_receipt(short, ss, [986000])
    full_rows, _ = pc.examples_from_receipt(full, fs, [986000])
    assert short['window']['end_s'] == 640 and full['window']['end_s'] == 700
    assert short_rows[0].better == 1 and full_rows[0].better == 0
    short_rows = [replace(x, population=pc.PRIMARY_POPULATION) for x in short_rows]
    with pytest.raises(ValueError, match='full-horizon evidence'):
        pc.fit_calibration(short_rows, [986000], planner_contract=CONTRACT)
    # Redefining the planner to the shorter receipt window cannot bypass the registered horizon.
    e = copy.deepcopy(short_rows[0].wm_prediction)
    e['contract']['horizon_s'] = 240
    with pytest.raises(ValueError, match='shortened full-horizon'):
        pc.WMProbability.from_dict(_seal(e, 'evidence_sha256'))


@pytest.mark.parametrize('field,value', [('horizon_s', 360), ('policy_state_sha256', 'different')])
def test_consumer_contract_mismatch_refused_at_fit_load_and_use(tmp_path, field, value):
    _, rows = dataset(tmp_path)
    changed = replace(CONTRACT, **{field: value})
    with pytest.raises(ValueError, match='full-horizon'):
        pc.fit_calibration(rows, range(986000, 986004), planner_contract=changed)
    artifact = pc.fit_calibration(rows, range(986000, 986004), planner_contract=CONTRACT)
    path = tmp_path/'model.json'
    path.write_text(json.dumps(_seal(artifact, 'artifact_sha256')))
    with pytest.raises(ValueError, match='full-horizon'):
        pc.load_calibrator(path, planner_contract=changed)
    model = pc.load_calibrator(path, planner_contract=CONTRACT)
    with pytest.raises(ValueError, match='full-horizon'):
        predict(model, pc.WMProbability.from_dict(rows[0].wm_prediction), planner_contract=changed)


@pytest.mark.parametrize('field,value', [('objective', 'other'), ('baseline', 'other')])
def test_wrong_objective_or_baseline_refused(field, value):
    e = evidence(snapshot(), plan('win'))
    e['contract'][field] = value
    with pytest.raises(ValueError, match='objective/baseline'):
        pc.WMProbability.from_dict(_seal(e, 'evidence_sha256'))


@pytest.mark.parametrize('mutation,match', [
    ({'candidate_id': 'another'}, 'identity mismatch'),
    ({'snapshot_sha256': 'another'}, 'identity mismatch'),
    ({'start_s': 401}, 'identity mismatch'),
])
def test_identity_linked_evidence_refused_on_different_candidate(tmp_path, mutation, match):
    _, rows = dataset(tmp_path)
    e = dict(rows[0].wm_prediction, **mutation)
    rows[0] = replace(rows[0], wm_prediction=_seal(e, 'evidence_sha256'))
    with pytest.raises(ValueError, match=match):
        pc.fit_calibration(rows, range(986000, 986004), planner_contract=CONTRACT)


def test_realised_baseline_with_committed_moves_is_not_mappo_only(tmp_path):
    _, rows = dataset(tmp_path)
    rows[0] = replace(rows[0], baseline_is_unmodified_policy=False)
    with pytest.raises(ValueError, match='baseline provenance'):
        pc.fit_calibration(rows, range(986000, 986004), planner_contract=CONTRACT)


@pytest.mark.parametrize('change', ['short_window', 'minimum_horizon', 'legacy_gain_model', 'population'])
def test_hashed_incompatible_artifact_cannot_load(tmp_path, change):
    _, rows = dataset(tmp_path)
    a = pc.fit_calibration(rows, range(986000, 986004), planner_contract=CONTRACT)
    if change == 'short_window':
        a['windows'][0] = (400, 640)
    elif change == 'minimum_horizon':
        a['minimum_horizon_s'] = 420
    elif change == 'legacy_gain_model':
        a['model']['score'] = 'wm_gain_veh_s'
    else:
        a['population'] = 'outcome-selected-extra'
    p = tmp_path/'model.json'
    p.write_text(json.dumps(_seal(a, 'artifact_sha256')))
    with pytest.raises(ValueError):
        pc.load_calibrator(p, planner_contract=CONTRACT)


def test_real_all8_winner_only_extras_cannot_change_primary_population(tmp_path, monkeypatch):
    from tools.coopmwm import replay_bank as rb, verify_perseed as vp
    from tools.coopmwm.test_verify_perseed_cpu import setup, _budget, PHASE

    gains = {('p1', k): .6 for k in range(1, 7)}
    root, snap, plans, r, runner_for, _ = setup(tmp_path, gains, k=2)
    _budget(root)
    c = vp.confirm_all8(root, 'B', r, snap, runner_for, str(root), PHASE,
                        candidates=plans, branches_dir=rb.bank_root(root, 'B')/'sumo_verification'/'branches',
                        extra_seeds=4, later_harm_seed1=False, later_harm_seed2=False)
    assert len(c['extra']) == 4
    assert all(link['candidates'] == [0] for link in c['extra'].values())
    no_extras = dict(c)
    no_extras.pop('extra')
    before = tmp_path/'primary.json'
    before.write_text(json.dumps(_seal(no_extras, 'confirmation_sha256')))
    after = rb.bank_root(root, 'B')/vp.RECORDS/(r['verification_id']+'.json')
    primary, _ = pc.load_verified_records([before], [7])
    read = pc._read

    def guard(path):
        assert not any(link['kind'] in str(path) for link in c['extra'].values())
        return read(path)

    monkeypatch.setattr(pc, '_read', guard)
    with_extra, provenance = pc.load_verified_records([after], [7])
    assert with_extra == primary and len(primary) == 4
    assert pc.gain_diagnostic(with_extra)['observed_win_rate'] == .5
    assert {x.replica for x in with_extra} == {'first_seed', 'seed2'}
    assert len(provenance['excluded_replicas']) == 4
    monkeypatch.setattr(pc, '_read', read)
    extra_paths = [pc.Path(link['root'])/'verifications'/(link['verification_id']+'.json')
                   for link in c['extra'].values()]
    # Directly supplying the selected receipts, in either order, cannot reintroduce them.
    for paths in ([*extra_paths, after], [after, *extra_paths]):
        assert pc.load_verified_records(paths, [7])[0] == primary
    extras_only, _ = pc.load_verified_records(extra_paths, [7])
    with pytest.raises(ValueError, match='calibration population'):
        pc.fit_calibration(extras_only, [7], planner_contract=CONTRACT)


def test_cpu_dependency_uses_registered_observable_selection():
    from tools.coopmwm.planner_calibration_cpu_env import selection_module

    features = selection_module().features
    assert features([]) == dict(cav_count=0, density=0., cav_share=0., ramp=False,
                                queued=0, mean_speed=0., pair=[])
    rows = [dict(vid='left', s_m=-100., lane=0, is_cav=True, v_mps=4.),
            dict(vid='right', s_m=-99., lane=1, is_cav=True, v_mps=10.),
            dict(vid='ramp', s_m=75.44, lane=2, is_cav=False, v_mps=7.),
            dict(vid='outside', s_m=75.45, lane=0, is_cav=True, v_mps=0.),
            dict(vid='other_lane', s_m=-90., lane=3, is_cav=True, v_mps=0.)]
    result = features(rows)
    assert result == dict(cav_count=2, density=3/(3*175.44/1000), cav_share=2/3,
                          ramp=True, queued=1, mean_speed=7., pair=['left', 'right'])
