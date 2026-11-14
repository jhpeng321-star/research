"""E1-FIX (Amendment 74, POST HOC REPAIR), CPU side: verification addendum with reuse of the registered and the
WM-ACT episodes, admission, budget-6 banks, confirmation waves and the WM-FIX vs Random analysis. SUMO-free; every
'result' below is a synthetic fixture and everything is written under tmp."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/'tests'))
import box_driver  # noqa: E402
import cvlib as L  # noqa: E402
import e1_act as A  # noqa: E402
import e1_fix as X  # noqa: E402
import e1_order as E  # noqa: E402
import e1_pool  # noqa: E402
import stats as S  # noqa: E402
from test_e1_act import CW, SEEDS, _plan_result, _registered, _tree, _write_act_order  # noqa: E402


def _write_fix_order(out, seed, top, verdict='PASS'):
    pool = e1_pool.load_pool(out, seed)
    rest = [e['desc_hash'] for e in pool['pool'] if e['desc_hash'] not in top]
    doc = E.order_doc(X.ARM, seed, pool['doc_sha256'], list(top)+rest, dict(label=X.LABEL, g0=dict(verdict=verdict)))
    L.write_once(E.order_path(out, X.ARM, seed), L.text_of(doc))


def _with_act(tmp_path):
    """Registered chain + a collected WM-ACT addendum on seed 11 (three fresh plans verified, two admitted, its bank
    confirmed). -> everything the FIX arm may reuse."""
    out, args, plan, frozen, res, u0, confirm = _registered(tmp_path)
    reg = {sp['search_seed']: sp for sp in plan['seeds']}
    pool = e1_pool.load_pool(out, 11)
    wm, rnd = reg[11]['top']['WM'], reg[11]['top']['RANDOM']
    union = set(wm) | set(rnd)
    fresh = [e['desc_hash'] for e in pool['pool'] if e['desc_hash'] not in union][:6]
    r_only = next(h for h in rnd if h not in wm)
    _write_act_order(out, 11, [fresh[0], wm[0], r_only, fresh[1], wm[2], fresh[2]])
    _write_act_order(out, 23, reg[23]['top']['WM'])
    assert A.main(['plan']+args) == 0
    s11 = L.read_json(out/'e1'/'act'/'VERIFY_PLAN_ACT.json')['seeds'][0]
    fam = {c['desc_hash']: c['family'] for c in s11['candidates']}
    act_res = dict(u0=dict(u0), plans={fam[fresh[0]]: _plan_result(u0, 80.0), fam[fresh[1]]: _plan_result(u0, 60.0),
                                       fam[fresh[2]]: _plan_result(u0, -20.0)})
    return dict(out=out, args=args, plan=plan, frozen=frozen, res=res, u0=u0, confirm=confirm, reg=reg, wm=wm,
                rnd=rnd, fresh=fresh, r_only=r_only, act_res=act_res)


def _collect_act(f):
    out, args = f['out'], f['args']
    (out/'e1'/'act'/'VERIFY_RESULTS_ACT_seed11.json').write_text(json.dumps(f['act_res']), encoding='utf-8')
    assert A.main(['admit', '--results-dir', str(f['res'])]+args) == 0
    assert A.main(['build']+args) == 0
    fz = L.verify_seal(L.read_json(out/'e1'/'act'/'FROZEN_BANKS_ACT.json'))
    b11 = fz['banks'][0]
    conf = dict(u0=dict(f['confirm']['u0']), banks={b11['sha256']: {w: 900.0+i-100.0 for i, w in enumerate(CW)}})
    (out/'e1'/'act'/'CONFIRM_RESULTS_ACT.json').write_text(json.dumps(conf), encoding='utf-8')
    return b11, conf


def test_e1_fix_chain_reuses_registered_and_act(tmp_path, capsys):
    f = _with_act(tmp_path)
    out, args, reg, wm, fresh, u0, frozen = f['out'], f['args'], f['reg'], f['wm'], f['fresh'], f['u0'], f['frozen']
    b_act, act_conf = _collect_act(f)
    before = _tree(out)
    # seed 11: two WM-ACT-verified plans, two registered ones, two nobody verified.
    # seed 23: exactly the registered WM top six.
    top11 = [fresh[0], wm[0], fresh[3], fresh[2], f['r_only'], fresh[4]]
    _write_fix_order(out, 11, top11)
    _write_fix_order(out, 23, reg[23]['top']['WM'])

    assert X.main(['plan']+args) == 0
    fp = L.verify_seal(L.read_json(out/'e1'/'fix'/'VERIFY_PLAN_FIX.json'))
    assert fp['label'] == X.LABEL and fp['arm'] == 'WMFIX' and fp['cap_sim_s'] == 1_036_800
    s11, s23 = fp['seeds']
    assert [c['verified'] for c in s11['candidates']] == ['act', 'registered', 'fix', 'act', 'registered', 'fix']
    assert (s11['n_new'], s11['n_reused_registered'], s11['n_reused_act'], s23['n_new'], s23['n_reused']) == (2, 2, 2, 0, 6)
    assert fp['sim_s_new_plans'] == 2*24*1200 and fp['reused_from_act'] == 2 and fp['reused_candidates'] == 10
    assert s11['act_reuse']['act_results_sha256'] == L.sha_file(out/'e1'/'act'/'VERIFY_RESULTS_ACT_seed11.json')
    assert s23['waves'] == [] and s23['bank'] is None and s23['act_reuse'] is None
    assert s11['waves'][0] == reg[11]['waves'][0] and [w['kind'] for w in s11['waves']] == ['U0', 'PLAN', 'PLAN']
    assert [w['desc_hash'] for w in s11['waves'][1:]] == [fresh[3], fresh[4]]
    act_plan = L.read_json(out/'e1'/'act'/'VERIFY_PLAN_ACT.json')
    taken = {w['wave'] for d in (f['plan'], act_plan) for w in box_driver.find_waves(d, 'e1v_seed11')}
    assert not {w['wave'] for w in s11['waves'][1:]} & taken
    bank = L.read_json(out/'e1'/'fix'/'banks'/'E1_FIX_CANDS_seed11.json')
    assert sorted(p['family'] for p in L.bank_strategies(bank)) == sorted(w['family'] for w in s11['waves'][1:])
    assert len(box_driver.merge_waves([f['plan'], act_plan, fp], 'e1v_seed11')) == len(reg[11]['waves'])+3+2
    assert X.main(['has-waves', '--out', str(out), '--seed', '11']) == 0
    assert X.main(['has-waves', '--out', str(out), '--seed', '23']) == 10

    fam = {c['desc_hash']: c['family'] for c in s11['candidates']}
    fix_res = dict(u0=dict(u0), plans={fam[fresh[3]]: _plan_result(u0, 70.0), fam[fresh[4]]: _plan_result(u0, -20.0)})
    reg_res = L.read_json(f['res']/'VERIFY_RESULTS_seed11.json')
    merged = X.merged_results(s11, reg_res, f['act_res'], fix_res)
    assert merged['u0'] == u0 and set(merged['plans']) == set(fam.values())
    assert merged['plans'][fam[fresh[0]]] == f['act_res']['plans'][fam[fresh[0]]]         # reused WM-ACT episodes
    assert merged['plans'][fam[wm[0]]] == reg_res['plans'][fam[wm[0]]]                    # reused registered ones
    for bad in (dict(fix_res, u0=dict(u0, w00=u0['w00']+1e-9)), dict(fix_res, u0=dict(list(u0.items())[:-1])), None,
                dict(u0=u0, plans={})):
        with pytest.raises(L.Refused):                                                    # exact U0 equality
            X.merged_results(s11, reg_res, f['act_res'], bad)
    with pytest.raises(L.Refused):
        X.merged_results(s11, reg_res, dict(f['act_res'], u0=dict(u0, w05=0.0)), fix_res)
    (out/'e1'/'fix'/'VERIFY_RESULTS_FIX_seed11.json').write_text(json.dumps(fix_res), encoding='utf-8')

    assert X.main(['admit', '--results-dir', str(f['res'])]+args) == 0
    adm = L.verify_seal(L.read_json(out/'e1'/'fix'/'ADMISSION_FIX_seed11.json'))
    assert sorted(adm['admitted']) == sorted([fresh[0], wm[0], fresh[3]]) and adm['label'] == X.LABEL
    act_adm = L.read_json(out/'e1'/'act'/'ADMISSION_ACT_seed11.json')['verdicts']
    reg_adm = L.read_json(out/'e1'/'bank'/'ADMISSION_seed11.json')['verdicts']
    for h, src, ref in ((fresh[0], 'act', act_adm), (fresh[2], 'act', act_adm), (wm[0], 'registered', reg_adm),
                        (f['r_only'], 'registered', reg_adm)):
        v = adm['verdicts'][h]                                                            # same rule, same verdict
        assert v['verified'] == src and v['PASS'] == ref[h]['PASS'] and v['ci95'] == ref[h]['ci95']
    assert adm['verdicts'][fresh[3]]['verified'] == 'fix'

    assert X.main(['build']+args) == 0
    fz = L.verify_seal(L.read_json(out/'e1'/'fix'/'FROZEN_BANKS_FIX.json'))
    b11, b23 = fz['banks']
    regrow = {(b['search_seed'], b['arm']): b for b in frozen['banks'] if b['budget'] == 6}
    assert b11['n_admitted'] == 3 and b11['arm'] == 'WMFIX' and not b11['primary']
    assert b11['priority'] == sorted(b11['priority'], reverse=True)
    assert [p['family'] for p in L.bank_strategies(L.read_json(out/'e1'/'fix'/'banks'/b11['file']))] == b11['families']
    assert b23['sha256'] == regrow[(23, 'WM')]['sha256'] and b23['confirmed_by'].startswith('registered e1conf')
    assert b11['confirmed_by'] == 'FIX confirmation wave' and b11['sha256'] != b_act['sha256']
    ac = fz['admitted_count']
    assert ac['WMFIX'] == [3, regrow[(23, 'WM')]['n_admitted']] and ac['label'] == X.LABEL
    assert ac['RANDOM'] == [regrow[(s, 'RANDOM')]['n_admitted'] for s in SEEDS]           # registered Random, reused
    assert ac['wmfix_minus_random'] == [a-b for a, b in zip(ac['WMFIX'], ac['RANDOM'])]
    assert ac['permutation'] == S.exact_paired_permutation(ac['wmfix_minus_random'])
    cwv = fz['confirm_waves']
    assert cwv[0] == frozen['confirm_waves'][0] and [w['bank'] for w in cwv[1:]] == [b11['file']]
    assert fz['confirm_new_bank_waves'] == 1 and fz['confirm_sim_s_new_banks'] == len(CW)*1200
    assert fz['cap_sim_s'] == 691_200 and fz['cap_u0_sim_s'] == 115_200
    assert fz['act_confirmation_reused']['sha256'] == L.sha_file(out/'e1'/'act'/'CONFIRM_RESULTS_ACT.json')
    assert X.main(['has-waves', '--out', str(out), '--confirm']) == 0

    with pytest.raises(L.Refused):                                                        # not confirmed yet
        X.main(['analyse']+args)
    fix_conf = dict(u0=dict(f['confirm']['u0']), banks={b11['sha256']: {w: 900.0+i-70.0 for i, w in enumerate(CW)}})
    (out/'e1'/'fix'/'CONFIRM_RESULTS_FIX.json').write_text(json.dumps(fix_conf), encoding='utf-8')
    assert X.main(['analyse']+args) == 0
    an = L.verify_seal(L.read_json(out/'e1'/'fix'/'E1_FIX_ANALYSIS.json'))
    exp = [-70.0+30.0*regrow[(11, 'RANDOM')]['n_admitted'],
           -30.0*(regrow[(23, 'WM')]['n_admitted']-regrow[(23, 'RANDOM')]['n_admitted'])]
    assert an['wmfix_minus_random_per_seed'] == exp and an['label'] == X.LABEL and an['n_worlds'] == len(CW)
    assert an['permutation'] == S.exact_paired_permutation(exp) and an['wmfix_minus_random']['design'] == 'crossed'
    assert an['bank_minus_u0']['WMFIX_seed11']['diff'] == -70.0 and an['admitted_count'] == ac
    assert an['bank_minus_u0']['RANDOM_seed11'] == (
        dict(diff=0.0, ci95=[0.0, 0.0], pct=0.0, note='bank is U0') if regrow[(11, 'RANDOM')]['is_u0']
        else an['bank_minus_u0']['RANDOM_seed11'])
    with pytest.raises(L.Refused):                                                        # another CONF_U0
        X.analyse(fz, frozen, f['confirm'], [('FIX', dict(fix_conf, u0=dict(f['confirm']['u0'], c0=1.0)))])
    with pytest.raises(L.Refused):                                                        # one bank, two results
        X.analyse(fz, frozen, f['confirm'], [('WM-ACT', dict(u0=f['confirm']['u0'], banks={b11['sha256']: {}})),
                                             ('FIX', fix_conf)])
    # resume keeps every file; nothing registered or WM-ACT was written or changed
    for cmd in (['plan'], ['admit', '--results-dir', str(f['res'])], ['build'], ['analyse']):
        assert X.main(cmd+args) == 0
    after = _tree(out)
    assert {k: v for k, v in after.items() if k in before} == before
    new = sorted(k.replace('\\', '/') for k in after if k not in before)
    assert new and all(k.startswith('e1/fix/') or k.startswith('e1/order/ORDER_WMFIX_') for k in new)
    capsys.readouterr()


def test_e1_fix_bank_already_confirmed_by_act(tmp_path, capsys):
    """WM-FIX top six = the WM-ACT top six: nothing new to verify, the bank content is the WM-ACT bank, and its
    collected confirmation is reused (zero new sim-s)."""
    f = _with_act(tmp_path)
    out, args, reg, wm, fresh = f['out'], f['args'], f['reg'], f['wm'], f['fresh']
    b_act, act_conf = _collect_act(f)
    _write_fix_order(out, 11, [fresh[0], wm[0], f['r_only'], fresh[1], wm[2], fresh[2]])
    _write_fix_order(out, 23, reg[23]['top']['WM'])
    assert X.main(['plan']+args) == 0
    fp = L.read_json(out/'e1'/'fix'/'VERIFY_PLAN_FIX.json')
    assert fp['new_candidates'] == 0 and fp['sim_s_new_plans'] == 0 and fp['seeds'][0]['n_reused_act'] == 3
    assert X.main(['admit', '--results-dir', str(f['res'])]+args) == 0
    assert X.main(['build']+args) == 0
    fz = L.read_json(out/'e1'/'fix'/'FROZEN_BANKS_FIX.json')
    assert fz['banks'][0]['sha256'] == b_act['sha256'] and fz['banks'][0]['confirmed_by'].startswith('WM-ACT')
    assert fz['confirm_new_bank_waves'] == 0 and fz['confirm_sim_s_new_banks'] == 0
    assert X.main(['has-waves', '--out', str(out), '--confirm']) == 10
    assert X.main(['analyse']+args) == 0
    an = L.read_json(out/'e1'/'fix'/'E1_FIX_ANALYSIS.json')
    assert an['bank_minus_u0']['WMFIX_seed11']['diff'] == -100.0 and an['fix_results_sha256'] is None
    capsys.readouterr()


def test_e1_fix_plan_refusals(tmp_path, capsys):
    f = _with_act(tmp_path)
    out, args, reg, wm, fresh = f['out'], f['args'], f['reg'], f['wm'], f['fresh']
    pool = e1_pool.load_pool(out, 11)
    sp, bank = X.fix_seed_plan(wm, pool, reg[11])
    assert sp['n_new'] == 0 and bank is None and sp['waves'] == [] and sp['sim_s_new_plans'] == 0
    sp, bank = X.fix_seed_plan(fresh, pool, reg[11])                                      # the cap case: six new
    assert sp['n_new'] == 6 and sp['sim_s_new_plans'] == 6*28_800 and 6*sp['sim_s_new_plans'] == X.CAP_VERIFY_S
    with pytest.raises(L.Refused):
        X.fix_seed_plan(wm[:5], pool, reg[11])
    with pytest.raises(L.Refused):
        X.fix_seed_plan(wm[:5]+wm[:1], pool, reg[11])
    with pytest.raises(L.Refused):
        X.fix_seed_plan(wm, pool, dict(reg[11], pool_doc_sha256='0'*64))
    with pytest.raises(L.Refused):                                                        # WM-ACT ran it elsewhere
        X.fix_seed_plan([fresh[0]]+wm[:5], pool, reg[11], {fresh[0]: 'another_family'})
    # an order whose gate G0 did not pass, or another arm's order, is not accepted
    _write_fix_order(out, 23, reg[23]['top']['WM'])
    _write_fix_order(out, 11, wm, verdict='FAIL')
    with pytest.raises(L.Refused):
        X.main(['plan']+args)
    E.order_path(out, X.ARM, 11).unlink()
    L.write_once(E.order_path(out, X.ARM, 11), E.order_path(out, 'WM', 11).read_text(encoding='utf-8'))
    with pytest.raises(L.Refused):
        X.main(['plan']+args)
    # a candidate the WM-ACT addendum planned but has not collected: refuse (collect first), unless --no-act
    E.order_path(out, X.ARM, 11).unlink()
    _write_fix_order(out, 11, [fresh[0]]+wm[:5])
    with pytest.raises(L.Refused):
        X.main(['plan']+args)
    assert X.main(['plan', '--no-act', '--dry-run']+args) == 0
    # every bank empty -> only the registered CONF_U0, zero new sim-s; a bank WM-ACT confirmed is not run again
    rows = [dict(search_seed=s, is_u0=True, sha256=None, file=None, n_admitted=0) for s in SEEDS]
    assert X.fix_confirm_waves(rows, f['frozen']) == [f['frozen']['confirm_waves'][0]]
    assert X.admitted_contrast(rows, f['frozen'])['WMFIX'] == [0, 0]
    rows = [dict(search_seed=11, is_u0=False, sha256='a'*64, file='x.json', n_admitted=1)]
    assert len(X.fix_confirm_waves(rows, f['frozen'])) == 2
    assert len(X.fix_confirm_waves(rows, f['frozen'], {'a'*64})) == 1
    capsys.readouterr()
