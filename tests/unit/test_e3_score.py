"""S-WMCOMBINED e3_score statistics layer (CPU, synthetic rows; gate-schema parity)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.wmcombined.e3_score import (  # noqa: E402
    _boot_groups, _sign, _slope, _summarize, arm_analysis)

THRESH = 0.868421052631579


HORIZONS = ('10', '30', '60', '90', '120', '180', '300')


def _rows(seed=0, scale=1.0, bias=0.0, worlds=6, arms=('coopoff_c1', 'mappo', 'ours_c1')):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(worlds):
        real = {a: float(rng.normal(-50, 120)) for a in arms}
        row = {'pair_key': f'w9730{w:02d}m1900p05t300', 'arms': len(arms),
               'arm_names': list(arms),
               'dJ': {h: [[scale*real[a] + bias, real[a]] for a in arms]
                      for h in HORIZONS}}
        rows.append(row)
    return rows


def test_summarize_schema_matches_gate_tools():
    doc = _summarize(_rows(), 'validation', '/e3data', 'sha', THRESH)
    assert doc['schema'] == 'wmcombined_e3_gate/1' and doc['groups'] == 6
    assert set(doc['horizons']) == set(HORIZONS)
    h = doc['horizons']['300']
    assert h['n_pairs'] == 18 and h['n_groups'] == 6
    assert abs(h['mean_signed_error']) < 1e-9          # scale 1, bias 0
    assert 0.9 < h['sign_agreement'] <= 1.0            # identical wm/real signs


def test_amended_and_compare_accept_the_schema():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/longloss'))
    from amended import evaluate as amended_evaluate
    from paired_statistics import bias, compare
    ref = _summarize(_rows(seed=1), 'validation', '/e3data', 'sha1', THRESH)
    cand = _summarize(_rows(seed=1, scale=1.2, bias=25.0), 'validation', '/e3data',
                      'sha2', THRESH)
    got = compare(ref, cand)
    assert got['n_worlds'] == 6 and got['n_pair_groups'] == 6
    assert got['absolute_300s_bias_reduction'] < 0     # bias added -> worse
    assert {'120', '300'} <= set(bias(cand))
    out = amended_evaluate(ref, cand)
    # scale!=1 with bias: candidate worse on (a); long sign may still match
    assert out['amended_rule']['a_pass'] is False
    assert out['metering_slope_300s']['candidate'] == pytest.approx(1.2, abs=1e-9)
    assert set(out['metering_slope_300s']['per_world_n_ge3']) <= set(range(973000, 973006))


def test_boot_groups_resamples_whole_worlds_and_slope_per_arm():
    rows = _rows(seed=2, scale=0.5)
    per_group = [r['dJ']['300'] for r in rows]
    point, ci = _boot_groups(per_group, _slope, seed=0)
    assert abs(point - 0.5) < 1e-9 and ci[0] <= 0.5 <= ci[1]
    analysis = arm_analysis(rows)
    assert set(analysis) == {'coopoff_c1', 'mappo', 'ours_c1'}
    for arm in analysis.values():
        assert arm['300']['n_pairs'] == 6
        assert abs(arm['300']['pooled_slope'] - 0.5) < 1e-9
        assert arm['300']['pooled_slope_ci95'][0] <= 0.5


def test_sign_drops_zero_reals():
    assert _sign([(1.0, 0.0), (2.0, 1.0)]) == 1.0
    assert _sign([(-1.0, 2.0), (3.0, 1.0)]) == 0.5
