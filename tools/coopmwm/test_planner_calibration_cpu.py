"""T1C CPU contract tests: actual verifier receipts, fake plant, no simulation."""
import copy
from dataclasses import replace
import json
import xml.etree.ElementTree as ET

import pytest

from tools.coopmwm import planner_calibration as pc
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.exact_grade import _seal
from tools.coopmwm.test_sumo_verify_cpu import FakePlant, plan, snapshot


def receipt(root, seed=986000, *, reject=False, replica=False, fail=(), fallback=False):
    snap = snapshot(world=seed)
    if replica:
        snap['rng'] = dict(snap['rng'], replication_salt='independent-post-t0')
    plans = [plan('win', gain=-1. if reject else 1.), plan('loss', gain=-1.), plan('tie', gain=0.)]
    # WM predictions can be wrong; labels must come from branch J, never predictions.
    plans[1]['saving_vs_policy'] = 3.
    if fallback:
        for p in plans:
            p.pop('saving_vs_policy')
    specs = sv.build_branches(snap, plans)[1:]
    r = sv.verify(snap, plans, FakePlant(fail=fail), lambda *args: None, root=root)
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
    assert loaded == rows and len(report['sources']) == 1


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
    restored = pc.Calibrator.from_dict(json.loads(json.dumps(model.to_dict())))
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
    paths = [receipt(tmp_path/f'world{s}', seed=s)[0] for s in range(986000, 986004)]
    return paths, pc.load_verified_records(paths, range(986000, 986004))[0]


def test_seed_split_replicas_reordering_and_validation_labels_cannot_affect_fit(tmp_path):
    _, rows = dataset(tmp_path)
    seeds = range(986000, 986004)
    rows += [replace(x, snapshot_sha256='replica') for x in rows]
    fit, val = pc.split_by_seed(rows, seeds)
    assert {x.world_seed for x in fit}.isdisjoint({x.world_seed for x in val})
    a = pc.fit_calibration(rows, seeds)
    b = pc.fit_calibration(list(reversed(rows)), seeds)
    assert a == b
    changed = [replace(x, realised_gain_veh_s=0.)
               if x.world_seed in a['split']['validation_seeds'] else x for x in rows]
    c = pc.fit_calibration(changed, seeds)
    assert a['model'] == c['model'] and a['validation'] != c['validation']
    assert a['fit']['n']+a['validation']['n'] == len(rows)


def test_small_or_single_class_data_refused(tmp_path):
    _, r, specs = receipt(tmp_path/'sumo_verification')
    rows, _ = pc.examples_from_receipt(r, specs, [986000])
    with pytest.raises(ValueError, match='two development'):
        pc.fit_calibration(rows, [986000])
    rows += [replace(x, world_seed=986001) for x in rows]
    rows = [replace(x, realised_gain_veh_s=0.) for x in rows]
    with pytest.raises(ValueError, match='both wins'):
        pc.fit_calibration(rows, [986000, 986001])


def test_cli_json_svg_artifact_loading_and_no_overwrite(tmp_path):
    paths, _ = dataset(tmp_path)
    out, svg = tmp_path/'model.json', tmp_path/'reliability.svg'
    args = ['--records', *map(str, paths), '--dev-seeds', *map(str, range(986000, 986004)),
            '--output', str(out), '--reliability-svg', str(svg)]
    artifact = pc.main(args)
    assert pc.load_calibrator(out).to_dict() == pc.Calibrator.from_dict(artifact['model']).to_dict()
    assert ET.fromstring(svg.read_text()).tag.endswith('svg')
    assert artifact['provenance']['n_examples'] == 12
    with pytest.raises(ValueError, match='new file'):
        pc.main(args)
    saved = json.loads(out.read_text())
    saved['model']['probabilities'][0] = .123
    out.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='artifact_sha256'):
        pc.load_calibrator(out)


def test_artifact_refuses_valid_hash_with_seed_leakage(tmp_path):
    _, rows = dataset(tmp_path)
    a = pc.fit_calibration(rows, range(986000, 986004))
    a['split']['validation_seeds'] = a['split']['fit_seeds']
    p = tmp_path/'model.json'
    p.write_text(json.dumps(_seal(a, 'artifact_sha256')))
    with pytest.raises(ValueError, match='seed leakage'):
        pc.load_calibrator(p)
