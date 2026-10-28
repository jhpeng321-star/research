import copy
import json

import pytest

from tools.gated300.cct1 import SIGMA, compute, horizon_pairs, horizon_result, main


def panel():
    return {'wm_sha256': 'a' * 64, 'entrant_control': True, 'threshold': .868421052631579,
            'rows': [{'pair_key': f'w{i}m1@t0',
                      'dJ': {str(h): [[-200., -100.]] for h in SIGMA}} for i in range(4)]}


def test_contiguous_horizon_and_informational_authority():
    doc = panel()
    assert compute(doc, panel='pairs300')['H_C'] == 300
    result = compute(doc, panel='calibration')
    assert result['H_C'] == 300 and not result['policy_authorized']
    for row in doc['rows']:
        row['dJ']['120'] = [200., -100.]
    result = compute(doc, panel='pairs300')
    assert result['H_C'] == 90
    assert result['horizons']['300']['verdict'] == 'PASS'


def test_strict_confidence_zero_real_and_empty_wrong():
    sigma = SIGMA[90]
    groups = {'w1': [[2 * sigma, 1.], [200., 0.], [-200., -3.], [0., -4.]]}
    result = horizon_result(groups, sigma)
    assert result['coverage'] == .5  # real=0 excluded only from sign counts
    assert result['sign_agreement'] == 1 and result['n_confident_sign'] == 1
    assert result['wrong_mean_abs_real'] == 0 and result['verdict'] == 'PASS'
    assert result['all_pairs_sign_agreement'] == 2 / 3


def test_registered_threshold_boundaries():
    # 4/5 confident signs, coverage exactly 1/4, wrong magnitude exactly 2 sigma.
    pairs = [[200., 1.]] * 4 + [[200., -2 * SIGMA[120]]] + [[0., 1.]] * 15
    result = horizon_result({'w1': pairs}, SIGMA[120])
    assert result['verdict'] == 'PASS'
    pairs[4][1] -= .01
    assert horizon_result({'w1': pairs}, SIGMA[120])['verdict'] == 'FAIL'
    assert horizon_result({'w1': [[0., 1.]]}, SIGMA[120])['sign_agreement'] is None


def test_world_clusters_and_bootstrap_reproducibility():
    rows = [{'pair_key': 'w100m1@t0', 'dJ': {'90': [200., 1.]}},
            {'pair_key': 'w100m2@t9', 'dJ': {'90': [[200., 1.], [200., -1.]]}}]
    groups = horizon_pairs(rows, 90)
    assert len(groups) == 1
    result = horizon_result(groups, SIGMA[90])
    assert result['sign_ci95'] == [2 / 3, 2 / 3]
    assert result == horizon_result(groups, SIGMA[90])


def test_missing_horizons_and_fail_closed_inputs():
    doc = panel()
    for row in doc['rows']:
        del row['dJ']['90']
    assert compute(doc, panel='pairs300')['H_C'] == 0
    doc = panel()
    doc['role'] = 'calibration'
    with pytest.raises(ValueError, match='relabel'):
        compute(doc, panel='pairs300')
    doc['rows'][0]['dJ']['90'] = [float('nan'), 1.]
    with pytest.raises(ValueError, match='finite'):
        compute(doc, panel='calibration')


def test_cli_preserves_input_and_refuses_overwrite(tmp_path):
    source = tmp_path / 'panel.json'
    output = tmp_path / 'WM_CCT1.json'
    doc = panel()
    original = copy.deepcopy(doc)
    source.write_text(json.dumps(doc))
    assert main([str(source), '--panel', 'calibration', '--out', str(output)]) == 0
    assert json.loads(source.read_text()) == original
    assert json.loads(output.read_text())['wm_sha256'] == doc['wm_sha256']
    with pytest.raises(FileExistsError):
        main([str(source), '--panel', 'calibration', '--out', str(output)])
