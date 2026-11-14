"""Zero-SUMO tests of the A82 part-A driver (ops/a82_s3b20.py). Run on the host tree (needs the A76 / A80 / A75-D2
drivers and tools.coopmwm); temporary files only under the pytest basetemp."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a82_s3b20 as M  # noqa: E402
import a76_d2p20 as M76  # noqa: E402
import a80_s2b as M80  # noqa: E402
import a75_d2forms as D2  # noqa: E402

SYNTH = HERE/'a82'/'SYNTH_TOP_OFFER_CONDITIONS_D2800_p20.json'


@pytest.fixture(autouse=True)
def _restore():
    saved76 = {k: getattr(M76, k) for k in M.M76_NAMES}
    saved = (M80.CELL, D2.FORMS)
    yield
    for k, v in saved76.items():
        setattr(M76, k, v)
    M80.CELL, D2.FORMS = saved


def test_caps_and_constants():
    assert (M.VERIFY_CAP, M.CAP_SIM_S, M.N_MAX, M.VERIFY_N) == (345_600, 345_600, 12, 24)
    assert M.CELL == 'D2800_p20' and M.PHASE == 'holdforms_a82s3b20' and M.HORIZON == 1200
    assert M.A76_V_SEEDS == tuple(range(997200, 997224)) and M.REF_WAVE == 'A76V_COMP' and M.REF_TAG == 'a76v_comp'
    assert M.FROZEN_U0_SHA256.startswith('d9042cc5') and M.A75_SHA256.startswith('d1af7e0f')
    assert M.PREFIX['v'] == 'a82v_' and not M.PREFIX['v'].startswith(('a76', 'a80'))


def test_rules_are_the_a76_rules():
    assert 'paired percentile bootstrap 10,000, seed 20261104' in M.ADMIT_RULE
    assert 'five-group 10 s' in M.ADMIT_RULE and 'STORED A76V_COMP' in M.ADMIT_RULE
    assert M.ADMIT_RULE.split('(a37_cellbank')[0] == M76.ADMIT_RULE.split('(a37_cellbank')[0]
    assert 'other' in M.BANK_RULE and 'most negative are kept' in M.BANK_RULE
    assert M.BANK_RULE.split('per form family')[0] == M76.BANK_RULE.split('per form family')[0]


def test_point_names_exist_and_are_replaced():
    for k in M.M76_NAMES:
        assert hasattr(M76, k), k
    M._point('/nonexistent/a76root', None)
    assert M76.verify_waves is M.admit_waves and M76._load is M._load and M76._mods is M._mods_proxy
    assert M76.ADMISSION_FILE == M.ADMISSION_FILE and M76.BANK_DIR == M.BANK_DIR and M76.MECH_FILE == M.MECH_FILE
    assert M80.CELL == 'D2800_p20' and D2.FORMS == M80.FORMS and 'other' in D2.FORMS
    assert M76.admit.__code__.co_filename.endswith('a76_d2p20.py')


def test_synthetic_top_maps_losslessly_at_p20():
    raw = SYNTH.read_bytes()
    doc = M.candidates_doc(raw, M.sha_bytes(raw))
    assert doc['cell'] == 'D2800_p20' and doc['n_candidates'] == 12 and not doc['STOP']
    assert all(c['mech'] for c in doc['candidates']) and doc['mech_plans'] == [c['family'] for c in doc['candidates']]
    assert {c['form'] for c in doc['candidates']} <= set(M80.FORMS)
    assert D2.seal_doc(doc)['doc_sha256'] == doc['doc_sha256']
    bank = M76.cands_bank_doc(doc['candidates'])
    from tools.coopmwm import a75_holdforms as H
    got, refused = H.a75_bank_plans(bank, 'D2800_p20')
    assert len(got) == 12 and not any(refused.values())


def test_bad_top_refused():
    rows = json.loads(SYNTH.read_bytes())
    rows[0]['offer_condition']['situation'] = {'unconditional': False}
    with pytest.raises(SystemExit):
        M.candidates_doc(json.dumps(rows).encode(), 'x'*64)
    with pytest.raises(SystemExit):
        M.candidates_doc(json.dumps(rows[:0]).encode(), 'x'*64)
    with pytest.raises(SystemExit):
        M.candidates_doc(json.dumps(rows+rows[:1]).encode(), 'x'*64)


def test_pin_top_roundtrip(tmp_path):
    raw = SYNTH.read_bytes()
    env = {M.TOP_ENV: str(SYNTH), M.TOP_SHA_ENV: M.sha_bytes(raw)}
    out = M.pin_top(tmp_path, env)
    assert out['n'] == 12 and (tmp_path/M.TOP_COPY).read_bytes() == raw
    assert [c['family'] for c in M.candidates(tmp_path)] == [r['family'] for r in json.loads(raw)]
    with pytest.raises(Exception):
        M.pin_top(tmp_path, env)
    with pytest.raises(Exception):
        M.pin_top(tmp_path/'other', {M.TOP_ENV: str(SYNTH), M.TOP_SHA_ENV: '0'*64})
    with pytest.raises(SystemExit):
        M.top_env({})


def test_ref_report_ok():
    drop = dict(report=dict(sha256='a'*64))
    assert M.ref_report_ok({'vehicles_report.json', 'DROPPED_RAW.json'}, drop, 'a'*64, None) == []
    assert M.ref_report_ok({'vehicles_report.json', 'DROPPED_RAW.json'}, drop, 'a'*64, 'a'*64) == []
    assert M.ref_report_ok({'vehicles.jsonl.gz'}, drop, 'a'*64, None)
    assert M.ref_report_ok(set(), None, 'a'*64, None)
    assert M.ref_report_ok(set(), drop, 'b'*64, None)
    assert M.ref_report_ok(set(), drop, 'a'*64, 'c'*64)


def test_cap_check():
    assert M.cap_check(0, 0, 345_600)
    with pytest.raises(SystemExit):
        M.cap_check(2_400, 2_400, 345_600)


def test_bank_rule_with_wm_forms_after_point():
    M._point_rules()
    rows = [dict(family=f'p{i}', key=f'k{i}', draw_rank=i, form=f, PASS=True, ci95=[-10.0-i, -1.0-i],
                 per_seed={1: float(i)}) for i, f in enumerate(('other', 'zone', 'pulse', 'gap', 'zone'))]
    chosen, dup, table = D2.bank_choice(rows)
    assert len(chosen) == 4 and 'p1' not in chosen and table['other']['admitted'] == 1


def test_mech_rule_and_verdict_reused():
    assert 'first 2 verification episodes' in M.MECH_RULE and 'd9042cc5' in M.MECH_RULE
    v, f, t = M80.mech_verdict({'a': [dict(world_id=1, errors=[], exercised=[['plain']])]},
                               [('a', 1, dict(collisions=0, teleports=0))], {'a': [['plain'], ['ext_zone', 'mid']]})
    assert v == 'PASS' and t['a']['ext_zone/mid'] == M80.NOT_EXERCISED
