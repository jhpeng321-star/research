"""Zero-SUMO unit tests of ops/a99_panel.py (A99 PANEL). Run inside the package env (A96_PART=c)."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('A96_PART', 'c')
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]

import pytest  # noqa: E402

import a99_panel as P  # noqa: E402

U0 = P.FROZEN_U0_SHA256


def test_arms_contrasts_caps():
    assert P.ARMS == ('BANK_S4_20C', 'BANK_HAND_20C', 'DUO_HOLD_20C', 'MAPPO20C')
    assert P.M.ARMS == P.ARMS and P.M.BANK_ARMS == ('BANK_S4_20C', 'BANK_HAND_20C') and P.M.HAND_RUN
    assert 'BANK_S5_20C' in P.OMITTED and 'BANK_S5_20C' not in P.ARMS
    assert [(a, r) for a, r, _ in P.contrasts()] == [
        ('BANK_S4_20C', 'MAPPO20C'), ('BANK_HAND_20C', 'MAPPO20C'), ('DUO_HOLD_20C', 'MAPPO20C'),
        ('BANK_S4_20C', 'BANK_HAND_20C'), ('BANK_S4_20C', 'DUO_HOLD_20C'), ('BANK_HAND_20C', 'DUO_HOLD_20C')]
    assert P.MECH_CAP == 9_600 and P.TOTAL_CAP == 921_600 == 4*192*1_200
    assert P.M.PREFIX['c'] == 'a99pc_' and all(v.startswith('a99p') for v in P.M.PREFIX.values())
    assert P.M.BOOTSTRAP == dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
    assert P.M.BLOCK == (984400, 984599) and P.M.seeds_of(192) == tuple(range(984400, 984592))


def _spec(kind='a32', exe=U0):
    ck = dict(kind=kind, executor_sha256=exe, comparator_kind='u0_kind', label='u0', executor_path='/u0.pt',
              binding_sha256='b0', binding_path='/b0.json', path='/x/A_checkpoint_000.pt', other=1)
    return dict(job_id='a99pc_duo_hold_20c_0000', arm='DUO_HOLD', checkpoint=ck, world=dict(seed=984400))


def test_with_exec6_only_six_fields():
    exe = dict(kind='a77_MAPPO20C', label='L', path='/m.pt', sha256=P.MAPPO20C_SHA256, binding_sha256='b1',
               binding_path='/b1.json')
    s = _spec()
    out = P.with_exec6([s], exe)[0]
    diff = P.M.spec_diff(out, s)
    assert sorted(diff) == sorted(f'checkpoint.{k}' for k, _ in P.EXEC_FIELDS)
    assert out['checkpoint']['executor_sha256'] == P.MAPPO20C_SHA256 and out['job_id'] == s['job_id']
    assert s['checkpoint']['executor_sha256'] == U0                      # input untouched
    with pytest.raises(Exception):
        P.with_exec6([_spec(exe=P.MAPPO20C_SHA256)], exe)
    with pytest.raises(Exception):
        P.with_exec6([_spec(kind='a37')], exe)


def _rec(n_cavs):
    acts = [dict(family='f1', t=10, members=[f'v{i}' for i in range(n_cavs)], release={}),
            dict(family='f2', t=20, members=['w0'], release={'w0': [5]})]
    return dict(acts=acts)


def test_string_keys_seal_roundtrip():
    """The A98 int-key bug: >= 10 CAVs engaged -> key order changes after a JSON round trip. Fixed: string keys."""
    N = P._N()
    r = _rec(12)
    eng, ov = P.engaged_hist(r), P.overlap_stats(r)
    assert all(isinstance(k, str) for k in eng) and '13' in eng
    assert all(isinstance(k, str) for k in ov['seconds_by_n_plans_active'])
    assert P.M.engaged_hist is P.engaged_hist and P.M.overlap_stats is P.overlap_stats
    v = N.seal(dict(engaged=eng, overlap=ov, rows=[dict(engaged=eng)]), 'receipt_sha256')
    back = json.loads(json.dumps(v, indent=1, sort_keys=True))
    N.verify(back, 'receipt_sha256', 'roundtrip')
    # same seconds as the unchanged a96_p20 definitions
    assert {str(k): v for k, v in P._ENG(r).items()} == eng


def _row(t, r, ci, all_pass):
    return dict(treatment=t, reference=r, kind='vs_mappo20c', groups=dict(all_pass=all_pass),
                total_minus_ref=dict(diff=sum(ci)/2, ci95=list(ci), p_two_sided=0.01))


def test_rule_verdict():
    assert P.rule_verdict(_row('A', 'B', (-5, -1), True))['verdict'] == 'SUPPORTED'
    x = P.rule_verdict(_row('A', 'B', (-5, -1), False))
    assert x['verdict'] == 'NOT SUPPORTED' and x['direction'] == 'NOT_SEPARATED' and x['delay_pass']
    assert P.rule_verdict(_row('A', 'B', (1, 5), True))['direction'] == 'REFERENCE_BETTER'
    assert P.rule_verdict(_row('A', 'B', (-1, 5), True))['direction'] == 'NOT_SEPARATED'
