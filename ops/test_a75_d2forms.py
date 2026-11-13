"""A75 stage D2 driver (zero SUMO): the seeded draw (strata, grids, executability, distinct keys, determinism, pin),
the bank rule, seed / cap guards, and the mechanics checks on receipts of the REAL A75 / A37 executors driven by
synthetic traffic (tools/coopmwm/test_a44_lcvariant_cpu.traffic)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a75_d2forms as M  # noqa: E402
from tools.coopmwm import a75_holdforms as H  # noqa: E402


@pytest.fixture(scope='module')
def doc():
    return M.candidates_doc()


# ------------------------------------------------------------------------------------------------ the draw

def test_draw_size_strata_and_forms(doc):
    c = doc['candidates']
    assert len(c) == M.DRAW_N == 36
    assert doc['candidates_by_form'] == dict(gap='12', short='6', zone='6', pulse='4', taper='4', rolling='4') or \
        doc['candidates_by_form'] == {'gap': 12, 'pulse': 4, 'rolling': 4, 'short': 6, 'taper': 4, 'zone': 6}
    assert {k: v for k, v in doc['candidates_by_stratum'].items()} == {s[0]: s[1] for s in M.STRATA}
    assert {x['n_roles'] for x in c} <= {1, 2, 3, 4} and 4 in {x['n_roles'] for x in c} and 1 in {x['n_roles'] for x in c}
    assert [x['draw_rank'] for x in c] == list(range(1, 37))


def test_every_candidate_executable_distinct_and_canonical(doc):
    keys = set()
    for c in doc['candidates']:
        s = c['strategy']
        assert H.executor_refusal(s) is None
        assert H.canonical(s) == s
        assert c['key'] == H.strategy_key(s) and s['name'] == f'd2_{c["key"]}' == c['family']
        keys.add(c['key'])
    assert len(keys) == 36
    bank = H.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in doc['candidates']],
                      M.CELL)
    got, refused = H.a75_bank_plans(bank, M.CELL)
    assert len(got) == 36 and not refused


def test_values_only_from_the_spec_grids(doc):
    for c in doc['candidates']:
        s, form = c['strategy'], c['form']
        assert s['trigger'] == 'lanes' and s['slots'] == 1 and s['gate'] is None and s['phases'] is None
        assert s['spread_m'] in M.SPREADS
        hmax = max(r['hold_s'] for r in s['roles'])
        assert s['cooldown_s'] in (hmax, hmax+30, 60, 120)
        assert len({(r['lane'], r['rank']) for r in s['roles']}) == len(s['roles'])
        for r in s['roles']:
            assert r['dmax'] in M.DMAX and r['onset_s'] in M.ONSETS and r['lc_target'] is None
            if r.get('start') == 'gap':
                assert r['lane'] == 1 and r['hold_s'] in M.GAP_HOLDS
                assert r['gap_dx_lo'] in M.GAP_DX_LO and r['gap_dx_hi'] in M.GAP_DX_HI
                assert r.get('gap_drop_m') in M.GAP_DROP_M and r['gap_arm_s'] in M.GAP_ARM_S
                assert r.get('gap_release', 'timer') in M.GAP_RELEASE and r.get('gap_events', 1) in M.GAP_EVENTS
                assert r.get('dv_rel') in M.DV_REL
                assert r['v_w'] in ((0.0,)+M.V_W if 'dv_rel' in r else M.V_W)
            else:
                assert r['hold_s'] in M.HOLD_GRID and r['v_w'] in M.V_W and 'dv_rel' not in r
            if 'pulse_on_s' in r:
                assert r['pulse_on_s'] in M.PULSE and r['pulse_off_s'] in M.PULSE
                assert r['hold_s'] >= r['pulse_on_s'] + r['pulse_off_s']
            for k in ('taper_in_s', 'taper_out_s'):
                assert r.get(k, 0) in M.TAPER
            assert r.get('taper_in_s', 0) + r.get('taper_out_s', 0) <= r['hold_s']
        if form == 'gap':
            assert sum(r.get('start') == 'gap' for r in s['roles']) == 1 and s['zone'] in M.GAP_ZONES
            assert all(r['lane'] in (1, 2) for r in s['roles'])
            assert c['n_roles'] == {'gap_alone': 1, 'gap_plus1': 2, 'gap_plus2': 3}[c['stratum']]
        else:
            assert not any(r.get('start') == 'gap' for r in s['roles'])
        if form == 'short':
            assert all(r['hold_s'] in M.SHORT_HOLDS for r in s['roles']) and s['zone'] in M.OLD_ZONES
        if form == 'zone':
            assert s['zone'] == c['stratum'].split('_', 1)[1]
            if s['zone'] == 'custom':
                up, down = s['zone_d']
                assert up in M.ZONE_D_UP and down in M.ZONE_D_DOWN and up - down >= M.ZONE_D_MIN_SPAN
        else:
            assert 'zone_d' not in s
        if form == 'pulse':
            assert any('pulse_on_s' in r for r in s['roles']) and s['zone'] in M.OLD_ZONES
        if form == 'taper':
            assert any(r.get('taper_in_s', 0) or r.get('taper_out_s', 0) for r in s['roles'])
        if form == 'rolling':
            on = [r['onset_s'] for r in s['roles']]
            assert on[0] == 0 and on == sorted(on) and len(set(on)) == len(on) >= 2
        if form in ('pulse', 'taper', 'rolling', 'short'):
            assert H.is_extended(s) == (form in ('pulse', 'taper'))


def test_draw_deterministic_and_stream_specific(doc):
    assert M.candidates_doc() == doc
    other = M.draw_rng(M.DRAW_SEED, 'other').random()
    assert M.draw_rng().random() != other
    assert M.seal_doc(doc)['doc_sha256'] == doc['doc_sha256']


def test_draw_writes_once_and_pin_roundtrip(tmp_path, monkeypatch):
    out = tmp_path/'C.json'
    d = M.draw(out)
    assert out.read_bytes().count(b'\r') == 0 and out.with_suffix('.csv').is_file()
    assert out.with_suffix('.sha256').read_text().split()[0] == d['doc_sha256']
    with pytest.raises(SystemExit):
        M.draw(out)
    got = M.candidates(out)
    assert [c['key'] for c in got] == [c['key'] for c in d['candidates']]
    bad = json.loads(out.read_text())
    bad['candidates'][0]['strategy']['roles'][0]['v_w'] = 30.0
    out.write_text(json.dumps(bad, indent=1, sort_keys=True)+'\n')
    with pytest.raises(Exception):
        M.candidates(out)


def test_shipped_candidate_file_is_the_draw():
    if not M.CANDS_FILE.is_file():
        pytest.skip('candidate file not drawn yet')
    assert M.CANDS_FILE.read_text(encoding='utf-8') == json.dumps(M.candidates_doc(), indent=1, sort_keys=True)+'\n'
    assert len(M.candidates()) == 36


def test_examples_bank_executes_six_forms():
    strat, bank = M.examples()
    assert tuple(strat) == M.EXAMPLE_FAMILIES
    got, refused = H.a75_bank_plans(bank, M.CELL)
    assert not refused and {p['family'] for p in got} == set(M.EXAMPLE_FAMILIES)


# ------------------------------------------------------------------------------------------------ bank rule

def _row(fam, form, rank, hi, ok, per=None):
    return dict(family=fam, form=form, draw_rank=rank, key=f'k{rank:02}', PASS=ok, ci95=[hi-10, hi],
                per_seed=per or {'1': float(rank), '2': hi})


def test_bank_rule_family_best_max4_duplicates():
    rows = [_row('a', 'gap', 1, -5, True), _row('b', 'gap', 2, -9, True), _row('c', 'short', 3, -1, True),
            _row('d', 'zone', 4, -3, True), _row('e', 'pulse', 5, -4, True), _row('f', 'taper', 6, -2, True),
            _row('g', 'rolling', 7, 3, False), _row('h', 'zone', 8, -20, True, per={'1': 4.0, '2': -3})]
    chosen, dup, table = M.bank_choice(rows)
    assert dup == {'h': 'd'}                     # same 24 results as an earlier plan: no bank slot
    assert chosen == ['b', 'e', 'd', 'f']        # family bests by CI95 upper, best 4
    assert table['gap']['best'] == 'b' and table['short']['in_bank'] is False and table['rolling']['best'] is None
    assert M.bank_choice([_row('x', 'gap', 1, 2, False)])[0] == []
    with pytest.raises(ValueError):
        M.bank_choice([_row('x', 'wall', 1, -2, True)])


def test_bank_plans_priority_order(doc):
    c = doc['candidates'][:3]
    rows = [dict(family=x['family'], ci95=[-9, hi]) for x, hi in zip(c, (-1.0, -5.0, -3.0))]
    out = M.bank_plans(c, rows, [x['family'] for x in c[:2]])
    assert [p['family'] for p in out] == [c[1]['family'], c[0]['family']]
    assert out[0]['lb95_veh_s'] == 5.0


# ------------------------------------------------------------------------------------------------ seeds / caps

def test_mech_seed_guard():
    with pytest.raises(SystemExit):
        M.mech_seed0({})
    with pytest.raises(SystemExit):
        M.mech_seed0({M.SEED_ENV: ''})
    for bad in ('964100', '962000', '977400', '989600', '997450', '12'):
        with pytest.raises(SystemExit):
            M.mech_seed0({M.SEED_ENV: bad})
    assert M.mech_seed0({M.SEED_ENV: '997400'}) == 997400


def test_block_free():
    table = ((992400, 992499, 992400), (997400, 997499, 997400))
    assert M.check_block_free(997400, table)
    with pytest.raises(ValueError):
        M.check_block_free(997400, table + ((997450, 997460, 997450),))
    with pytest.raises(ValueError):
        M.check_block_free(997500, table)


def test_caps():
    assert (M.MECH_CAP, M.VERIFY_CAP, M.CONFIRM_CAP) == (24_000, 1_036_800, 115_200)
    assert M.cap_check(0, 0, 20*1200, 'mech')
    with pytest.raises(SystemExit):
        M.cap_check(1200, 1200, 20*1200, 'mech')
    with pytest.raises(SystemExit):
        M.cap_check(0, M.CAP_SIM_S, 1200, 'confirm')
    assert M.cap_check(0, 0, 36*24*1200, 'verify')
    with pytest.raises(SystemExit):
        M.cap_check(0, 0, 36*24*1200+1, 'verify')


# ------------------------------------------------------------------------------------------------ mechanics checks

class _T44:
    """tools.coopmwm.test_a44_lcvariant_cpu, imported lazily (a host tree without it skips only these tests)."""
    def __getattr__(self, name):
        return getattr(pytest.importorskip('tools.coopmwm.test_a44_lcvariant_cpu'), name)


T44 = _T44()


def _run(fam, seed=11, arm='A75_PLAN'):
    """Drive the real executor on synthetic traffic; also return pos(vid, t) = SUMO x of the rows the executor saw."""
    from tools.coopmwm import oracle_plan as O
    strat, bank = M.examples()
    hook = H.make_hook(arm, M.CELL, bank=bank, family=fam if arm == 'A75_PLAN' else None)
    xs = {}

    def stream():
        for t, rows, cmds in T44.traffic(seed, t1=1200):
            for r in rows:
                xs[(r['vid'], t)] = float(r['s_m']) + float(O.S_OFF)
            yield t, rows, cmds
    T44.drive(hook, stream())
    return hook.receipts(), strat, (lambda vid, t: xs.get((vid, int(t))))


@pytest.mark.parametrize('fam', M.EXAMPLE_FAMILIES)
def test_form_checks_pass_on_real_executor_receipts(fam):
    r, strat, pos = _run(fam)
    s = strat[fam]
    role = s['roles'][0]
    errs = dict(ex_short=lambda: M.check_short(r, role), ex_gap=lambda: M.check_gap(r, role),
                ex_zone=lambda: M.check_zone(r, H.zone_bounds(s), pos), ex_pulse=lambda: M.check_pulse(r, role),
                ex_taper=lambda: M.check_taper(r, role), ex_rolling=lambda: M.check_rolling(r, s['roles']))[fam]()
    assert errs == [], errs[:5]


def test_bank_check_and_detects_overlap():
    r, _, _ = _run(None, arm='A75_BANK')
    assert M.check_bank(r) == []
    bad = dict(activations=1, acts=[dict(t=10, family='a', members=['c1'], release={'c1': [50, 'hold_end']}),
                                    dict(t=20, family='b', members=['c1'], release={})])
    assert M.check_bank(bad)


def test_checks_detect_broken_receipts():
    strat, _ = M.examples()
    role = strat['ex_pulse']['roles'][0]
    ev = dict(vid='c1', start_kind='onset', fire_t=100, end_t=160, end_why='hold_end', issued_s=60, armed_t=100)
    assert M.check_pulse(dict(holdforms=[ev]), role)                     # pulsed hold issued every second
    assert M.check_pulse(dict(holdforms=[dict(ev, issued_s=30)]), role) == []
    g = strat['ex_gap']['roles'][0]
    assert M.check_gap(dict(holdform_summary=dict(gap_fired=0)), g)
    open_ev = dict(vid='c1', start_kind='gap', armed_t=1150, fire_t=1195, end_t=None, end_why=None, trigger_vid='h1',
                   dx_m=0.0, issued_s=4)
    one = dict(holdform_summary=dict(gap_fired=1))
    assert M.check_gap(dict(one, holdforms=[open_ev]), g) == []                     # still running at the horizon
    assert M.check_gap(dict(one, holdforms=[dict(open_ev, fire_t=900)]), g)         # never ended
    assert M.check_gap(dict(one, holdforms=[dict(open_ev, end_t=1199, end_why='timer', dx_m=99.0)]), g)
    acts = [dict(t=1, family='ex_zone', members=['c1'], release={})]
    assert M.check_zone(dict(activations=1, acts=acts), H.zone_bounds(strat['ex_zone']), lambda v, t: 0.0)
    assert M.check_zone(dict(activations=1, acts=acts), H.zone_bounds(strat['ex_zone']), lambda v, t: None)
    lo, hi = H.zone_bounds(strat['ex_zone'])
    assert M.check_zone(dict(activations=1, acts=acts), (lo, hi), lambda v, t: (lo+hi)/2) == []
    short = strat['ex_short']['roles'][0]
    assert M.check_short(dict(activations=1, acts=[dict(t=5, family='ex_short', members=['c1'],
                                                         release={'c1': [9, 'hold_end']})]), short)
    assert M.check_short(dict(activations=0), strat['ex_short']['roles'][0])


def test_ident_row_on_a37_vs_a75_plain_plan():
    from tools.coopmwm import a37_cellbank as C
    strat, _ = M.examples()
    s = strat['ex_short']
    bank37 = C.bank_doc([dict(family='ex_short', strategy=s, lb95_veh_s=0.0)], M.CELL)
    h37 = C.make_hook('A37_PLAN', M.CELL, bank=bank37, family='ex_short')
    h75 = H.make_hook('A75_PLAN', M.CELL, bank=bank37, family='ex_short')
    assert T44.drive(h37, T44.traffic(5, t1=900)) == T44.drive(h75, T44.traffic(5, t1=900))
    rec = dict(world_id='w', J_veh_s=1.0, safety=dict(collisions=0), vehicle_log=dict(sha256='x'))
    r37, r75 = dict(rec, a37=h37.receipts()), dict(rec, a75=h75.receipts())
    row = M.ident_row(r37, r75, 'L', 'L', r37)
    assert row['PASS'], row
    assert not M.ident_row(r37, dict(r75, J_veh_s=1.5), 'L', 'L', r37)['PASS']
    assert not M.ident_row(r37, r75, 'L', 'M', r37)['PASS']


# ------------------------------------------------------------------------------------------------ host table patch

def _runner_text():
    for p in (HERE.parent/'tools'/'coopmwm'/'nm_eval_runner.py', HERE.parent/'pytools'/'tools'/'coopmwm'/'nm_eval_runner.py'):
        if p.is_file():
            return p.read_bytes().decode('utf-8')
    pytest.skip('runner not in this tree')


def test_patch_tables_d2_adds_one_free_block():
    sys.path.insert(0, str(HERE/'a75d2'/'host'))
    import patch_tables_d2 as P
    s = _runner_text()
    t = P.runner(s, 997400)
    assert 'A75D2_MECH_RANGE = (997400, 997499)' in t and P.runner(t, 997400) == t
    assert t.count('(A75D2_MECH_RANGE[0], A75D2_MECH_RANGE[1], A75D2_MECH_RANGE[0])') == 1
    ns = {}
    i0 = t.index('A33_PANEL_RANGE = ')
    i1 = t.index('\n\n', t.index('EVAL_HELDOUT_RANGES = ('))
    exec('EVAL_HELDOUT_RANGE = (993000, 993299)\n' + t[i0:i1], ns)          # the table block only
    assert M.check_block_free(997400, ns['EVAL_HELDOUT_RANGES'])
    for bad in (997450, 992400, 964100):
        with pytest.raises(SystemExit):
            P.runner(s, bad)
    with pytest.raises(SystemExit):
        P.runner(t, 997500)


def test_a75_from_a37_only_changes_the_executor_fields():
    ck = dict(kind='a37', a37_arm='A37_PLAN', a37_sha256='x', a37_bank=dict(path='p', file_sha256='s'), a37_family='f',
              executor_sha256='e', path='u0', cell_key=M.CELL)
    base = [dict(arm='A37_PLAN', job_id='a75d2v_k_0000', world_id='w', checkpoint=ck)]
    entry = dict(path='/root/a75d2/root/a75d2_banks/B.json', file_sha256='b')
    out = M.a75_from_a37(base, 'A75_PLAN', entry, family='f', tag='a75d2v_k')
    c = out[0]['checkpoint']
    assert out[0]['arm'] == 'A75_PLAN' and out[0]['job_id'] == 'a75d2v_k_0000'
    assert not any(k.startswith('a37_') for k in c) and c['kind'] == 'a75' and c['a75_sha256'] == M.A75_SHA256
    assert c['a75_bank'] == entry and c['a75_family'] == 'f' and c['executor_sha256'] == 'e' and c['path'] == 'u0'
    assert base[0]['checkpoint']['kind'] == 'a37'                       # input untouched
    assert M.a75_from_a37(base, 'A75_BANK', entry, family=None, tag='t')[0]['checkpoint']['a75_family'] is None
    assert 2*M.IDENT_N + sum(n for _, _, n in M.FORM_WAVES) == M.MECH_EPISODES == 20


def test_drop_pointer_is_the_decompressed_result(tmp_path):
    import gzip
    import hashlib
    body = b'{"x": 1}' * 100
    p = tmp_path/'RESULT.json.gz'
    with gzip.open(p, 'wb') as f:
        f.write(body)
    c = M._gz_content(p)
    assert c == [hashlib.sha256(body).hexdigest(), len(body)]
    ptr = dict(file='RESULT.json.gz', encoding='coopmwm-lossless-gzip-v1', sha256=c[0], uncompressed_bytes=c[1])
    assert M.pointer_ok(ptr, c, {'RESULT.json.gz': 'gzsha'})
    assert not M.pointer_ok(dict(ptr, sha256=M._sha(p)), c, {'RESULT.json.gz': 'gzsha'})   # the gzip-container sha
    assert not M.pointer_ok(ptr, None, {'RESULT.json.gz': 'gzsha'}) and not M.pointer_ok(ptr, c, {})
    p.write_bytes(p.read_bytes()[:20])
    assert M._gz_content(p) is None
