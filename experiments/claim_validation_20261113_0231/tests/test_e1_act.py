"""WM-ACT sensitivity arm (post-registration, pre-outcome): verification addendum, admission, budget-6 banks,
confirmation waves and the WM-ACT vs Random analysis. SUMO-free; every 'result' below is a synthetic fixture and
everything is written under tmp."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import box_driver  # noqa: E402
import cvlib as L  # noqa: E402
import e1_act as A  # noqa: E402
import e1_bank  # noqa: E402
import e1_order as E  # noqa: E402
import e1_pool  # noqa: E402
import e1_verify  # noqa: E402
import stats as S  # noqa: E402

CFG = dict(search_grid=dict(v_w=[18, 19, 20, 21, 22], dmax=[1.0, 1.5], hold_s=[90, 120, 180], onset_s=[0, 20, 40]),
           onset_max=30)
SEEDS = [11, 23]
WORLDS = [f'w{i:02d}' for i in range(24)]
CW = [f'c{i}' for i in range(8)]
GROUPS = {g: 1.0 for g in S.CO_PRIMARY_GROUPS}


def _plan_result(u0, gain):
    return dict(J={w: u0[w]-gain-(i % 3) for i, w in enumerate(WORLDS)}, group_increase=GROUPS)


def _registered(tmp_path):
    """The registered E1 chain on synthetic results (pools, WM + Random orders, verification, admission, banks,
    confirmation results). WM top six = pool[11..6]; WM ranks 1 and 3 are admitted, nothing of Random's."""
    cfg = tmp_path/'cfg.json'
    cfg.write_text(json.dumps(dict(config=CFG)), encoding='utf-8')
    out = tmp_path/'out'
    args = ['--out', str(out), '--seeds']+[str(s) for s in SEEDS]
    assert e1_pool.main(['--cfg', str(cfg)]+args) == 0
    sd = tmp_path/'scores'
    sd.mkdir()
    for s in SEEDS:
        pool = e1_pool.load_pool(out, s)
        rows = [dict(desc_hash=e['desc_hash'], key=e['key'], screen_summary=dict(score=-float(e['index'])),
                     screen_activated=True, finalist=e['index'] < 12, activated=True if e['index'] < 12 else None,
                     summary=dict(score=float(e['index'])) if e['index'] < 12 else None) for e in pool['pool']]
        (sd/f'WM_SCORES_seed{s}.json').write_text(json.dumps(dict(
            mode='two_stage', pool_doc_sha256=pool['doc_sha256'], search_seed=s,
            wm_checkpoint_sha256=L.WM_CKPT_SHA256, rows=rows,
            finalist_hashes=[e['desc_hash'] for e in pool['pool'][:12]], cfg_sha256='x', wm_rollouts=0, counters=None,
            n_final_states=18, K=4, H=300, n_screen_states=4, K_screen=4, finalists=12)), encoding='utf-8')
    blocks = tmp_path/'blocks.json'
    blocks.write_text(json.dumps({'11': list(range(5, 29)), '23': list(range(105, 129))}), encoding='utf-8')
    assert E.main(['random']+args) == 0
    assert E.main(['wm-ingest', '--scores-dir', str(sd)]+args) == 0
    assert e1_verify.main(['plan', '--seed-blocks', str(blocks)]+args) == 0
    plan = L.verify_seal(L.read_json(out/'e1'/'verify'/'VERIFY_PLAN.json'))
    res = tmp_path/'res'
    res.mkdir()
    u0 = {w: 1000.0+i for i, w in enumerate(WORLDS)}
    for sp in plan['seeds']:
        plans = {c['family']: _plan_result(u0, 50.0 if c['selected_by'].get('WM') in (1, 3) else -20.0)
                 for c in sp['candidates']}
        (res/f'VERIFY_RESULTS_seed{sp["search_seed"]}.json').write_text(json.dumps(dict(u0=u0, plans=plans)),
                                                                        encoding='utf-8')
    assert e1_bank.main(['admit', '--results-dir', str(res)]+args) == 0
    assert e1_bank.main(['build', '--confirm-worlds', str(len(CW))]+args) == 0
    frozen = L.verify_seal(L.read_json(out/'e1'/'bank'/'FROZEN_BANKS.json'))
    confirm = dict(u0={w: 900.0+i for i, w in enumerate(CW)},
                   banks={b['sha256']: {w: 900.0+i-30.0*b['n_admitted'] for i, w in enumerate(CW)}
                          for b in frozen['banks'] if b['primary'] and not b['is_u0']})
    (out/'e1'/'bank'/'CONFIRM_RESULTS.json').write_text(json.dumps(confirm), encoding='utf-8')
    return out, args, plan, frozen, res, u0, confirm


def _write_act_order(out, seed, top):
    pool = e1_pool.load_pool(out, seed)
    rest = [e['desc_hash'] for e in pool['pool'] if e['desc_hash'] not in top]
    doc = E.order_doc(A.ARM, seed, pool['doc_sha256'], list(top)+rest, dict(label=A.LABEL))
    assert doc['top'] == list(top)
    L.write_once(E.order_path(out, A.ARM, seed), L.text_of(doc))


def _tree(root):
    return {str(p.relative_to(root)): L.sha_file(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def test_wm_act_addendum_chain(tmp_path, capsys):
    out, args, plan, frozen, res, u0, confirm = _registered(tmp_path)
    reg = {sp['search_seed']: sp for sp in plan['seeds']}
    before = _tree(out)
    # seed 11: WM ranks 1 and 3 (registered, admitted), one Random-only candidate (registered, rejected) and three
    # plans nobody verified. seed 23: exactly the registered WM top six (nothing new, same bank as the WM arm).
    pool = e1_pool.load_pool(out, 11)
    wm, rnd = reg[11]['top']['WM'], reg[11]['top']['RANDOM']
    union = set(wm) | set(rnd)
    fresh = [e['desc_hash'] for e in pool['pool'] if e['desc_hash'] not in union][:3]
    r_only = next(h for h in rnd if h not in wm)
    top11 = [fresh[0], wm[0], r_only, fresh[1], wm[2], fresh[2]]
    _write_act_order(out, 11, top11)
    _write_act_order(out, 23, reg[23]['top']['WM'])

    assert A.main(['plan']+args) == 0
    ap = L.verify_seal(L.read_json(out/'e1'/'act'/'VERIFY_PLAN_ACT.json'))
    assert ap['label'] == A.LABEL and ap['arm'] == 'WMACT'
    s11, s23 = ap['seeds']
    assert (s11['n_new'], s11['n_reused'], s23['n_new'], s23['n_reused']) == (3, 3, 0, 6)
    assert [c['verified'] for c in s11['candidates']] == ['addendum', 'registered', 'registered', 'addendum',
                                                           'registered', 'addendum']
    assert [c['selected_by'] for c in s11['candidates']] == [{'WMACT': i} for i in range(1, 7)]
    # exact budget: only new candidates are simulated; the registered U0 wave is listed verbatim (published -> reused)
    assert ap['sim_s_new_plans'] == s11['sim_s_new_plans'] == 3*24*1200 and s23['sim_s_new_plans'] == 0
    assert ap['sim_s_u0_if_rerun_everywhere'] == 24*1200
    assert s23['waves'] == [] and s23['bank'] is None
    assert s11['waves'][0] == reg[11]['waves'][0] and [w['kind'] for w in s11['waves']] == ['U0']+['PLAN']*3
    assert [w['desc_hash'] for w in s11['waves'][1:]] == fresh
    assert not {w['wave'] for w in s11['waves'][1:]} & {w['wave'] for w in reg[11]['waves']}
    assert s11['world_seeds'] == reg[11]['world_seeds'] == list(range(5, 29))
    bank = L.read_json(out/'e1'/'act'/'banks'/'E1_ACT_CANDS_seed11.json')
    assert sorted(p['family'] for p in L.bank_strategies(bank)) == sorted(w['family'] for w in s11['waves'][1:])
    assert L.sha_file(out/'e1'/'act'/'banks'/'E1_ACT_CANDS_seed11.json') == s11['bank']['sha256']
    # the driver sees the addendum waves of the block; merged with the registered plan the U0 wave is kept once
    ws = box_driver.find_waves(ap, 'e1v_seed11')
    assert [w['wave'] for w in ws] == [w['wave'] for w in s11['waves']]
    assert len(box_driver.merge_waves([plan, ap], 'e1v_seed11')) == len(reg[11]['waves'])+3
    assert box_driver.find_waves(ap, 'e1v_seed23') == []
    assert A.main(['has-waves', '--out', str(out), '--seed', '11']) == 0
    assert A.main(['has-waves', '--out', str(out), '--seed', '23']) == 10

    # addendum verification results (fixture): fresh[0] and fresh[1] gain, fresh[2] loses
    fam = {c['desc_hash']: c['family'] for c in s11['candidates']}
    act_res = dict(u0=dict(u0), plans={fam[fresh[0]]: _plan_result(u0, 80.0), fam[fresh[1]]: _plan_result(u0, 60.0),
                                       fam[fresh[2]]: _plan_result(u0, -20.0)})
    rp = out/'e1'/'act'/'VERIFY_RESULTS_ACT_seed11.json'
    rp.write_text(json.dumps(act_res), encoding='utf-8')
    reg_res = L.read_json(res/'VERIFY_RESULTS_seed11.json')
    merged = A.merged_results(s11, reg_res, act_res)
    assert merged['u0'] == u0 and set(merged['plans']) == set(fam.values())
    assert merged['plans'][fam[wm[0]]] == reg_res['plans'][fam[wm[0]]]            # reused row, not the addendum's
    with pytest.raises(L.Refused):                                                # another U0 = another pairing
        A.merged_results(s11, reg_res, dict(act_res, u0=dict(u0, w00=u0['w00']+1e-9)))
    with pytest.raises(L.Refused):
        A.merged_results(s11, reg_res, None)
    with pytest.raises(L.Refused):
        A.merged_results(s11, reg_res, dict(u0=u0, plans={}))
    assert A.merged_results(s23, L.read_json(res/'VERIFY_RESULTS_seed23.json'), None)['u0'] == u0

    assert A.main(['admit', '--results-dir', str(res)]+args) == 0
    adm = L.verify_seal(L.read_json(out/'e1'/'act'/'ADMISSION_ACT_seed11.json'))
    assert sorted(adm['admitted']) == sorted([fresh[0], wm[0], fresh[1], wm[2]]) and adm['label'] == A.LABEL
    reg_adm = L.read_json(out/'e1'/'bank'/'ADMISSION_seed11.json')['verdicts']
    for h in (wm[0], wm[2], r_only):                                              # same rule, same inputs, same verdict
        assert adm['verdicts'][h]['PASS'] == reg_adm[h]['PASS'] and adm['verdicts'][h]['ci95'] == reg_adm[h]['ci95']
        assert adm['verdicts'][h]['verified'] == 'registered'
    assert adm['verdicts'][fresh[0]]['verified'] == 'addendum'

    assert A.main(['build']+args) == 0
    fz = L.verify_seal(L.read_json(out/'e1'/'act'/'FROZEN_BANKS_ACT.json'))
    b11, b23 = fz['banks']
    regrow = {(b['search_seed'], b['arm']): b for b in frozen['banks'] if b['budget'] == 6}
    assert b11['n_admitted'] == 4 and b11['arm'] == 'WMACT' and not b11['primary']
    assert b11['priority'] == sorted(b11['priority'], reverse=True)               # executor order
    assert [p['family'] for p in L.bank_strategies(L.read_json(out/'e1'/'act'/'banks'/b11['file']))] == b11['families']
    # seed 23 tests the registered WM top six -> the registered WM bank content -> no new confirmation run
    assert b23['sha256'] == regrow[(23, 'WM')]['sha256'] and b23['confirmed_by'].startswith('registered e1conf')
    assert b11['confirmed_by'] == 'addendum confirmation wave'
    ac = fz['admitted_count']
    assert ac['WMACT'] == [4, regrow[(23, 'WM')]['n_admitted']]
    assert ac['RANDOM'] == [regrow[(s, 'RANDOM')]['n_admitted'] for s in SEEDS]
    assert ac['WM_registered'] == frozen['primary_admitted_count']['WM']
    assert ac['wmact_minus_random'] == [a-b for a, b in zip(ac['WMACT'], ac['RANDOM'])]
    assert ac['permutation'] == S.exact_paired_permutation(ac['wmact_minus_random']) and ac['label'] == A.LABEL
    cwv = fz['confirm_waves']
    assert cwv[0] == frozen['confirm_waves'][0] and [w['bank'] for w in cwv[1:]] == [b11['file']]
    assert fz['confirm_new_bank_waves'] == 1 and fz['confirm_sim_s_new_banks'] == len(CW)*1200
    assert len(box_driver.merge_waves([frozen, fz], 'conf')) == len(frozen['confirm_waves'])+1   # CONF_U0 kept once
    assert A.main(['has-waves', '--out', str(out), '--confirm']) == 0

    with pytest.raises(L.Refused):                                                # not confirmed yet
        A.main(['analyse']+args)
    act_conf = dict(u0=dict(confirm['u0']), banks={b11['sha256']: {w: 900.0+i-100.0 for i, w in enumerate(CW)}})
    (out/'e1'/'act'/'CONFIRM_RESULTS_ACT.json').write_text(json.dumps(act_conf), encoding='utf-8')
    assert A.main(['analyse']+args) == 0
    an = L.verify_seal(L.read_json(out/'e1'/'act'/'E1_ACT_ANALYSIS.json'))
    # fixture: a registered bank is worth -30 s per admitted plan, the new WM-ACT bank -100 s
    exp = [-100.0+30.0*regrow[(11, 'RANDOM')]['n_admitted'],
           -30.0*(regrow[(23, 'WM')]['n_admitted']-regrow[(23, 'RANDOM')]['n_admitted'])]
    assert an['wmact_minus_random_per_seed'] == exp and an['label'] == A.LABEL and an['n_worlds'] == len(CW)
    assert an['permutation'] == S.exact_paired_permutation(exp) and an['wmact_minus_random']['design'] == 'crossed'
    assert an['wmact_minus_wm_registered_per_seed'] == [-100.0+30.0*regrow[(11, 'WM')]['n_admitted'], 0.0]
    assert an['bank_minus_u0']['WMACT_seed11']['diff'] == -100.0 and an['admitted_count'] == ac
    with pytest.raises(L.Refused):                                                # another CONF_U0
        A.analyse(fz, frozen, confirm, dict(act_conf, u0=dict(confirm['u0'], c0=1.0)))
    # resume keeps every file; no registered file was written or changed (the two fixture files written above aside)
    for cmd in (['plan'], ['admit', '--results-dir', str(res)], ['build'], ['analyse']):
        assert A.main(cmd+args) == 0
    after = _tree(out)
    assert {k: v for k, v in after.items() if k in before} == before
    new = sorted(k.replace('\\', '/') for k in after if k not in before)
    assert all(k.startswith('e1/act/') or k.startswith('e1/order/ORDER_WMACT_') for k in new)
    capsys.readouterr()


def test_wm_act_plan_refusals(tmp_path, capsys):
    out, args, plan, frozen, res, u0, confirm = _registered(tmp_path)
    reg = {sp['search_seed']: sp for sp in plan['seeds']}
    pool = e1_pool.load_pool(out, 11)
    top = reg[11]['top']['WM']
    sp, bank = A.act_seed_plan(top, pool, reg[11])
    assert sp['n_new'] == 0 and bank is None and sp['waves'] == []
    with pytest.raises(L.Refused):
        A.act_seed_plan(top[:5], pool, reg[11])
    with pytest.raises(L.Refused):
        A.act_seed_plan(top, pool, dict(reg[11], pool_doc_sha256='0'*64))
    # an order of another arm / another pool is not accepted as the WM-ACT order
    L.write_once(E.order_path(out, A.ARM, 11), (E.order_path(out, 'WM', 11)).read_text(encoding='utf-8'))
    with pytest.raises(L.Refused):
        A.main(['plan', '--out', str(out), '--seeds', '11'])
    # every bank empty -> nothing to confirm: only the registered CONF_U0, zero new sim-s
    rows = [dict(search_seed=s, is_u0=True, sha256=None, file=None, n_admitted=0) for s in SEEDS]
    assert A.act_confirm_waves(rows, frozen) == [frozen['confirm_waves'][0]]
    assert A.admitted_contrast(rows, frozen)['WMACT'] == [0, 0]
    capsys.readouterr()
