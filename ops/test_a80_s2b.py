"""A80 (A80-S2B) driver, zero SUMO: caps, the lossless mapping of a75ss TOP_OFFER_CONDITIONS rows onto the A75
executor (synthetic 12-row file from the a75ss writer + the real tiny end-to-end search output), refusals of every
non-lossless row, form families / tokens, receipt invariants, mechanics verdict, pin-top / candidates round trip."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a80_s2b as M  # noqa: E402
import a75_d2forms as D2  # noqa: E402
from tools.coopmwm import a75_holdforms as H  # noqa: E402

SYNTH = HERE/'a80'/'SYNTH_TOP_OFFER_CONDITIONS_D2800_p60.json'
TINY = HERE/'a80'/'E2E_TINY_TOP_OFFER_CONDITIONS_D2800_p60.json'


@pytest.fixture(autouse=True)
def _restore_d2():
    saved = {k: getattr(D2, k) for k in M.D2_NAMES}
    yield
    for k, v in saved.items():
        setattr(D2, k, v)


def rows(p=SYNTH):
    return json.loads(p.read_text(encoding='utf-8'))


def test_caps_and_constants():
    assert (M.VERIFY_CAP, M.CONFIRM_CAP, M.CAP_SIM_S) == (345_600, 115_200, 460_800)
    assert M.N_MAX*M.VERIFY_N*M.HORIZON == M.VERIFY_CAP and M.CONFIRM_N == 96 and M.MECH_WORLDS == 2
    assert M.A75_SHA256 == D2.A75_SHA256 and M.CELL == 'D2800_p60'
    assert M.cap_check(0, 0, 345_600, 'verify') and M.cap_check(0, 345_600, 115_200, 'confirm')
    with pytest.raises(SystemExit):
        M.cap_check(1_200, 1_200, 345_600, 'verify')
    with pytest.raises(SystemExit):
        M.cap_check(0, M.CAP_SIM_S, 1_200, 'confirm')
    assert 'A80_BANK_60C' in M.PRIMARY_RULE and 'a80v_' in M.DROP_RULE and 'other' in M.BANK_RULE


@pytest.mark.parametrize('path', [SYNTH, TINY])
def test_writer_rows_map_losslessly(path):
    rs = rows(path)
    cands, bad = M.map_rows(rs)
    assert not bad and len(cands) == len(rs)
    for c, r in zip(cands, rs):
        assert c['key'] == r['strategy_key'] and c['family'] == r['family'] == c['strategy']['name']
        assert H.executor_refusal(c['strategy']) is None and c['extended'] == r['extended']
        bank = H.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0)], M.CELL)
        assert bank['clean5c']['templates'][0]['offer_condition'] == r['offer_condition']
    # the whole set is one executable bank (the confirmation arm path)
    bank = H.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], M.CELL)
    got, refused = H.a75_bank_plans(bank, M.CELL)
    assert not refused and [p['family'] for p in got] and len(got) == len(cands)


def test_synthetic_fixture_covers_the_generator_forms():
    cands, _ = M.map_rows(rows())
    assert len(cands) == 12
    assert {c['trigger'] for c in cands} == {'wall', 'lanes'} and sum(c['extended'] for c in cands) == 10
    fams = {c['form'] for c in cands}
    assert {'gap', 'pulse', 'taper'} <= fams


def _bad(mut, i=0):
    rs = deepcopy(rows())
    mut(rs[i])
    return M.map_rows(rs)[1]


@pytest.mark.parametrize('mut', [
    lambda r: r['offer_condition']['roles'][0].update(new_field=1),
    lambda r: r['offer_condition']['roles'][0].pop('rank'),
    lambda r: r['offer_condition'].update(phases=[[0], [1]]),
    lambda r: r['offer_condition'].update(gate=5.0),
    lambda r: r['offer_condition'].update(start_gate='empty100'),
    lambda r: r['offer_condition'].update(situation=dict(unconditional=False)),
    lambda r: r['offer_condition'].update(extra=1),
    lambda r: r.update(strategy_key='0'*16),
    lambda r: r.update(extended=not r['extended']),
    lambda r: r.pop('score'),
    lambda r: r['offer_condition']['roles'][0].update(hold_s=901),
    lambda r: r['offer_condition']['roles'][0].update(sumo_lane=2, start='gap', gap_dx_lo=0.0, gap_dx_hi=20.0,
                                                      gap_arm_s=30),
])
def test_non_lossless_rows_refused(mut):
    assert _bad(mut)


def test_lc_role_refused_and_sizes():
    rs = deepcopy(rows())
    plain = next(r for r in rs if not r['extended'])
    plain['offer_condition']['roles'][0]['lc_target_sumo'] = 1
    s = H.spec_of(plain['family'], plain['offer_condition'])
    plain['strategy_key'] = H.strategy_key(s)
    assert any('lane-change' in why or 'refusal' in why for _, _, why in M.map_rows(rs)[1])
    assert M.map_rows([])[1] and M.map_rows(rows()+rows()[:1])[1]
    dup = rows()[:2]
    dup[1] = dup[0]
    assert M.map_rows(dup)[1]


def test_form_family_matches_d2_on_examples():
    strat, _ = D2.examples()
    got = {f: M.form_family(H.canonical(s)) for f, s in strat.items()}
    assert got == dict(ex_short='short', ex_gap='gap', ex_zone='zone', ex_pulse='pulse', ex_taper='taper',
                       ex_rolling='rolling')
    assert M.form_family(dict(zone='near', roles=[dict(hold_s=90, onset_s=0)])) == 'other'


def test_tokens_and_coverage():
    cov = M.covered_tokens()
    assert ('plain',) in cov and ('ext_trigger', 'lanes') in cov and ('gap_release', 'clear') in cov
    assert ('ext_trigger', 'wall') not in cov and ('gap_release', 'timer') not in cov
    doc = M.candidates_doc(SYNTH.read_bytes(), M.sha_bytes(SYNTH.read_bytes()))
    assert doc['n_candidates'] == 12 and doc == M.candidates_doc(SYNTH.read_bytes(), M.sha_bytes(SYNTH.read_bytes()))
    wall_ext = [c for c in doc['candidates'] if c['trigger'] == 'wall' and c['extended']]
    assert wall_ext and all(['ext_trigger', 'wall'] in c['uncovered'] and c['mech'] for c in wall_ext)
    assert all(not c['mech'] for c in doc['candidates'] if not c['extended'])
    assert doc['mech_plans'] == [c['family'] for c in doc['candidates'] if c['mech']]


def test_candidates_doc_refuses_bad_file():
    rs = rows()
    rs[3]['offer_condition']['phases'] = [[0], [1]]
    raw = json.dumps(rs).encode()
    with pytest.raises(SystemExit):
        M.candidates_doc(raw, M.sha_bytes(raw))


GAP = dict(lane=1, rank=0, v_w=12.0, dmax=1.0, hold_s=10, onset_s=0, lc_target=None, start='gap', gap_dx_lo=-10.0,
           gap_dx_hi=40.0, gap_arm_s=60, gap_release='clear', gap_events=3)
ONS = dict(lane=2, rank=0, v_w=16.0, dmax=1.0, hold_s=20, onset_s=10, lc_target=None, pulse_on_s=5, pulse_off_s=5)
SPEC = dict(name='x', trigger='wall', zone='mid', spread_m=300.0, cooldown_s=90, slots=1, gate=None, start_gate=None,
            phases=None, roles=[GAP, ONS])


def receipts():
    acts = [dict(t=400, family='x', members=['c1', 'c2'], release=dict(c1=[60, 'gap_arm_end'], c2=[30, 'hold_end']))]
    ev = [dict(family='x', vid='c1', start_kind='gap', armed_t=400, fire_t=405, end_t=410, end_why='clear',
               trigger_vid='h9', dx_m=12.0, v0=20.0, issued_s=5),
          dict(family='x', vid='c1', start_kind='gap', armed_t=400, fire_t=420, end_t=430, end_why='timer',
               trigger_vid='h7', dx_m=-5.0, v0=19.0, issued_s=10),
          dict(family='x', vid='c2', start_kind='onset', armed_t=410, fire_t=410, end_t=430, end_why='hold_end',
               trigger_vid=None, dx_m=None, v0=22.0, issued_s=10)]
    return dict(refused={}, acts=acts, holdforms=ev, activations=1)


def test_plan_checks_clean_and_exercise():
    errs, ex, st = M.plan_checks(receipts(), SPEC)
    assert errs == [] and st['activations'] == 1 and st['gap_fired'] == 2
    for t in [('ext_trigger', 'wall'), ('ext_zone', 'mid'), ('ext_roles', 'gap+onset'), ('role', 'gap', ''),
              ('role', 'onset', 'p'), ('onset_delay', 'onset'), ('gap_release', 'clear'), ('gap_events_multi', True)]:
        assert t in ex, t
    assert ('gap_release', 'timer') not in ex           # a clear role ending on its timer is not a timer-role test


@pytest.mark.parametrize('mut', [
    lambda r: r['holdforms'][0].update(dx_m=50.0),                       # outside the gap window
    lambda r: r['holdforms'][1].update(issued_s=9),                      # not every law second issued
    lambda r: r['holdforms'][2].update(issued_s=20),                     # pulse count wrong
    lambda r: r['holdforms'][2].update(fire_t=411),                      # onset event not at t + onset_s
    lambda r: r['acts'][0]['release'].update(c2=[25, 'hold_end']),       # wrong timed length
    lambda r: r['acts'][0]['release'].update(c1=[50, 'gap_arm_end']),    # released before the arm end
    lambda r: r['acts'][0]['release'].update(c1=[60, 'weird']),
    lambda r: r['acts'][0].update(members=['c1']),
    lambda r: r['holdforms'].extend([dict(r['holdforms'][0], fire_t=440, end_t=445, issued_s=5),
                                     dict(r['holdforms'][0], fire_t=450, end_t=455, issued_s=5)]),  # 4 > gap_events
    lambda r: r['holdforms'][0].update(start_kind='onset'),
    lambda r: r['acts'][0]['release'].pop('c2'),                         # never released, far from the end
    lambda r: r.update(refused=dict(bad_zone=['x'])),
])
def test_plan_checks_violations(mut):
    r = receipts()
    mut(r)
    assert M.plan_checks(r, SPEC)[0]


def test_open_session_near_end_allowed():
    r = receipts()
    r['acts'][0]['t'] = 1150
    for e in r['holdforms']:
        e['fire_t'] += 750
        e['armed_t'] += 750
        e['end_t'] += 750
    r['acts'][0]['release'].pop('c1')
    assert M.plan_checks(r, SPEC)[0] == []


def test_mech_verdict():
    rows_ = {'a': [dict(world_id=1, errors=[], exercised=[['ext_trigger', 'wall']])],
             'b': [dict(world_id=1, errors=[], exercised=[])]}
    v, f, t = M.mech_verdict(rows_, [('a', 1, dict(collisions=0, teleports=0))],
                             {'a': [['ext_trigger', 'wall']], 'b': [['gap_release', 'timer']]})
    assert v == 'PASS' and not f and t == {'a': {'ext_trigger/wall': 'exercised'},
                                           'b': {'gap_release/timer': M.NOT_EXERCISED}}
    assert M.mech_verdict(rows_, [('a', 1, dict(collisions=1, teleports=0))], {})[0] == 'FAIL'
    rows_['a'][0]['errors'] = ['x']
    assert M.mech_verdict(rows_, [], {})[0] == 'FAIL'
    assert M.mech_verdict({}, [], {})[0] == 'PASS'


def test_bank_rule_with_wm_forms():
    M._point_d2()
    assert D2.FORMS == M.FORMS and D2.core is M.core and D2.candidates is M.candidates
    seeds = {992500+i: 0.0 for i in range(3)}

    def row(f, form, rank, up, ok=True, ps=None):
        return dict(family=f, key=f, draw_rank=rank, form=form, PASS=ok, ci95=[up-5, up],
                    per_seed=ps or {k: 100.0*rank+k for k in seeds})
    rs = [row('a', 'other', 1, -3.0), row('b', 'gap', 2, -9.0), row('c', 'gap', 3, -1.0), row('d', 'pulse', 4, -2.0),
          row('e', 'zone', 5, -4.0), row('f', 'short', 6, -8.0), row('g', 'taper', 7, 1.0, ok=False)]
    chosen, dup, table = D2.bank_choice(rs)
    assert chosen == ['b', 'f', 'e', 'a'] and not dup and table['other']['in_bank'] and not table['pulse']['in_bank']


def test_pin_top_and_candidates_roundtrip(tmp_path):
    M._point_d2()
    raw = SYNTH.read_bytes()
    env = {M.TOP_ENV: str(SYNTH), M.TOP_SHA_ENV: M.sha_bytes(raw)}
    out = M.pin_top(tmp_path, env)
    assert out['n'] == 12 and (tmp_path/M.TOP_COPY).read_bytes() == raw
    cands = M.candidates(tmp_path)
    assert [c['family'] for c in cands] == [r['family'] for r in rows()]
    with pytest.raises(Exception):
        M.pin_top(tmp_path, env)                               # once
    doc = json.loads((tmp_path/M.CANDS_DOC).read_text())
    doc['candidates'][0]['form'] = 'short'
    (tmp_path/M.CANDS_DOC).write_text(json.dumps(doc))
    with pytest.raises(Exception):
        M.candidates(tmp_path)
    with pytest.raises(Exception):
        M.pin_top(tmp_path/'r2', {M.TOP_ENV: str(SYNTH), M.TOP_SHA_ENV: 'ab'*32})
    with pytest.raises(SystemExit):
        M.top_env({})


def test_pin_top_refusal_is_recorded(tmp_path):
    rs = rows()
    rs[0]['offer_condition']['roles'][0]['new_field'] = 1
    p = tmp_path/'top.json'
    p.write_text(json.dumps(rs))
    with pytest.raises(SystemExit):
        M.pin_top(tmp_path/'root', {M.TOP_ENV: str(p), M.TOP_SHA_ENV: M.sha_bytes(p.read_bytes())})
    assert (tmp_path/'root'/M.REFUSED_DOC).is_file() and not (tmp_path/'root'/M.CANDS_DOC).exists()


def test_point_d2_names_complete():
    import inspect
    src = inspect.getsource(M._point_d2)
    set_names = {n.strip() for line in src.splitlines() if line.strip().startswith('D2.')
                 for n in line.split('=')[0].replace('D2.', '').split(',')}
    assert set_names == set(M.D2_NAMES) and all(hasattr(D2, n) for n in M.D2_NAMES)
