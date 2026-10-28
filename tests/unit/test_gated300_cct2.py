import json

import pytest

from tools.gated300.cct1 import SIGMA
from tools.gated300.cct2 import compute, horizon_result, main


def panel(role, rows, wm='a' * 64, entrant=True):
    return {'role': role, 'wm_sha256': wm, 'entrant_control': entrant,
            'deterministic_crn': True, 'rows': rows}


def gain_rows(n, worlds=0, pred=-200., real=-100.):
    return [{'pair_key': f'w{worlds + i}m1@t0', 'dJ': {str(h): [pred, real] for h in SIGMA}}
            for i in range(n)]


def test_sign_convention_negative_is_improvement_and_zero_real_excluded():
    groups = {'w1': [[-200., -100.], [200., 100.], [-200., 0.], [-200., 50.]]}
    result = horizon_result(groups, SIGMA[120])
    assert result['n_gain'] == 3  # real dJ = 0 stays in |G_h|, leaves sign counts
    assert result['n_gain_sign'] == 2
    assert result['gain_sign_agreement'] == .5
    assert result['wrong_gain_mean_abs_real'] == 50.
    assert result['harm'] == dict(n_harm=1, n_harm_sign=1, harm_sign_agreement=1.,
                                  wrong_harm_mean_abs_real=0.)
    assert result['verdict'] == 'FAIL'  # |G_h| < 20 even though agreement would pass


def test_n_below_20_fails_even_with_perfect_agreement():
    perfect = [[-200., -100.]]
    result = horizon_result({'w1': perfect * 19}, SIGMA[120])
    assert result['gain_sign_agreement'] == 1. and result['verdict'] == 'FAIL'
    assert horizon_result({'w1': perfect * 20}, SIGMA[120])['verdict'] == 'PASS'


def test_pooling_pools_both_verdict_panels_with_informational_splits():
    pairs300 = panel('pairs300', gain_rows(10))
    validation = panel('validation', gain_rows(10, worlds=50))
    result = compute(pairs300, validation)
    pooled = result['horizons']['120']
    assert pooled['n_gain'] == 20 and pooled['verdict'] == 'PASS'
    assert result['H_G'] == 300
    for split in pooled['panels'].values():
        assert split['n_gain'] == 10 and split['verdict'] == 'FAIL'
    assert result['panels'] == ['pairs300', 'validation'] and result['policy_authorized']


def test_no_contiguous_prefix_requirement():
    rows = []
    for i in range(25):
        dJ = {'120': [0., -100.], '180': [-1., -100.], '300': [-200., -100.]}
        rows.append({'pair_key': f'w{i}m1@t0', 'dJ': dJ})
    result = compute(panel('pairs300', rows[:13]), panel('validation', rows[13:]))
    assert [result['horizons'][str(h)]['verdict'] for h in (90, 120, 180, 300)] == [
        'FAIL', 'FAIL', 'FAIL', 'PASS']
    assert result['H_G'] == 300  # 120/180 FAIL does not block a passing 300


def test_world_clusters_pool_across_panels_and_bootstrap_reproducible():
    rows = gain_rows(3)
    rows.append({'pair_key': 'w0m2@t9',
                 'dJ': {str(h): [[-200., -100.], [-200., 100.]] for h in SIGMA}})
    first = panel('pairs300', rows)
    second = panel('validation', gain_rows(2, worlds=1))  # w1/w2 shared with panel one
    result = compute(first, second)
    pooled = result['horizons']['120']
    assert pooled['n_pairs'] == 7 and pooled['n_worlds'] == 3
    assert pooled['gain_sign_agreement'] == 6 / 7
    assert pooled['panels']['validation']['n_worlds'] == 2
    assert pooled['gain_sign_ci95'] == compute(first, second)['horizons']['120']['gain_sign_ci95']


def test_fail_closed_inputs():
    good = panel('pairs300', gain_rows(12))
    holdout = panel('validation', gain_rows(12))
    for first, second, message in [
            (panel('calibration', gain_rows(12)), holdout, 'pairs300'),
            (panel('validation', gain_rows(12)), holdout, 'pairs300'),
            (good, panel('pairs300', gain_rows(12)), 'validation'),
            (good, panel('validation', gain_rows(12), wm='b' * 64), 'same exact WM'),
            (good, panel('validation', gain_rows(12), entrant=False), 'entrant'),
            (good, panel('validation', gain_rows(12) + [{'pair_key': 'w0x@t0',
             'dJ': {str(h): [float('nan'), -100.] for h in SIGMA}}]), 'finite')]:
        with pytest.raises(ValueError, match=message):
            compute(first, second)


def test_cli_writes_wm_hash_bound_authority_and_refuses_overwrite(tmp_path):
    wm = 'c' * 64
    pairs300 = tmp_path / 'pairs300.json'
    validation = tmp_path / 'validation.json'
    original = json.dumps(panel('pairs300', gain_rows(12), wm=wm))
    pairs300.write_text(original)
    validation.write_text(json.dumps(panel('validation', gain_rows(12), wm=wm)))
    output = tmp_path / f'CCT2_{wm}.json'
    assert main([str(pairs300), str(validation), '--out', str(output)]) == 0
    assert pairs300.read_text() == original  # read-only on the verdict panels
    assert json.loads(output.read_text())['H_G'] == 300
    assert json.loads(output.read_text())['sources']['pairs300']['sha256']
    with pytest.raises(FileExistsError):
        main([str(pairs300), str(validation), '--out', str(output)])
    with pytest.raises(ValueError, match='sha256'):
        main([str(pairs300), str(validation), '--out', str(tmp_path / 'CCT2_no_hash.json')])
