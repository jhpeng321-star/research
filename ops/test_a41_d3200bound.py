"""CPU-only tests for the amendment-41 driver (fake runner; no simulator, no process pool): mechanical candidate
construction from the A37 bank f96f8c8e, the registered selection rule (eligible-A / eligible-B / STOP + ties),
stage-2 refusal without the sealed SELECTION receipt, the primary conjunction, seed-range registration and the
d3200bound_a41 ledger cap; stage 1 -> select -> stage 2 -> analyse end to end on fake records."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a41_d3200bound as D  # noqa: E402
import nm_finaleval as N  # noqa: E402
import test_a37_celltailor as T37  # noqa: E402

G5 = ('origin:L0', 'origin:L1', 'origin:L2', 'class:hdv', 'class:cav')


@pytest.fixture
def root(tmp_path):
    src = T37.mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    out = tmp_path/'root'
    for f in T37.BIND:
        (out/f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src/f, out/f)
    D.prepare(out, D.A41_DIR/D.A37_BANK_NAME)
    return out


def runtime(root, **kw):
    return D.A41Runtime(T37.FakeInner(N.prerun_plan(root), **kw), root)


# ------------------------------------------------------------------ candidate construction

def test_candidates_are_the_registered_mechanical_subsets():
    bank = D.load_a37_bank()
    fams = [r['family'] for r in bank['rows']]
    assert [f[:12] for f in fams] == list(D.P_SHORT)
    doc = D.candidates()
    assert D.build_candidates(bank) == doc                     # committed doc == mechanical rebuild
    c = {x['label']: x for x in doc['candidates']}
    assert list(c) == ['S1', 'S2', 'S3', 'S4', 'S5', 'N2', 'N3', 'N4']
    for i in range(5):
        s = c[f'S{i+1}']
        assert (s['arm'], s['family'], s['families'], s['n_plans']) == ('A37_PLAN', fams[i], [fams[i]], 1)
        assert s['bank'] == D.A37_BANK_NAME and s['bank_file_sha256'] == D.A37_BANK_SHA256
    for k in (2, 3, 4):
        n = c[f'N{k}']
        assert (n['arm'], n['family'], n['families'], n['n_plans']) == ('CELL_BANK', None, fams[:k], k)
        pb = D.prefix_bank(bank, k)
        assert [r['family'] for r in pb['rows']] == fams[:k] == [t['family'] for t in pb['clean5c']['templates']]
        assert pb['rows'] == bank['rows'][:k] and pb['schema'] == bank['schema']        # nothing changed but the cut
        assert D.sha_text(D.C37.bank_text(pb)) == n['bank_file_sha256'] == doc['banks'][n['bank']]
    assert 'N5' not in c and doc['executor']['a37_sha256'] == D.A37_DIGEST


def test_candidate_doc_and_bank_are_sealed(tmp_path, monkeypatch):
    doc = json.loads((D.A41_DIR/'CANDIDATES_A41.json').read_text(encoding='utf-8'))
    doc['candidates'][5]['families'] = doc['candidates'][5]['families'][::-1]
    (tmp_path/'CANDIDATES_A41.json').write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(N.Blocked, match='doc_sha256'):
        D.candidates(tmp_path/'CANDIDATES_A41.json')
    raw = (D.A41_DIR/D.A37_BANK_NAME).read_text(encoding='utf-8').replace('"lb95_veh_s": 9251', '"lb95_veh_s": 9252')
    (tmp_path/'bank.json').write_text(raw, encoding='utf-8')
    with pytest.raises(N.Blocked, match='f96f8c8e'):
        D.load_a37_bank(tmp_path/'bank.json')


def test_prepare_writes_the_registered_banks_once(root):
    names = sorted(p.name for p in (root/D.BANK_DIR).iterdir())
    assert names == sorted(D.candidates()['banks'])
    for n, sha in D.candidates()['banks'].items():
        assert N.file_sha(root/D.BANK_DIR/n) == sha
    assert D.prepare(root, D.A41_DIR/D.A37_BANK_NAME)['banks'][D.A37_BANK_NAME]['file_sha256'] == D.A37_BANK_SHA256
    (root/D.BANK_DIR/'A41_N2_D3200_p20.json').write_text('{}\n', encoding='utf-8')
    with pytest.raises(N.Blocked):
        D.prepare(root, D.A41_DIR/D.A37_BANK_NAME)


def test_frozen_executor_digests_unchanged():
    from tools.coopmwm import a37_cellbank as C, a32_arms as A32, learned_trigger as L
    assert C.a37_digest() == D.A37_DIGEST and C.a37_digest().startswith('0a902437')
    assert A32.a32_digest() == D.A32_DIGEST and D.A32_DIGEST.startswith('1c51f22c')
    assert L.learned_trigger_digest() == D.LT_DIGEST and D.LT_DIGEST.startswith('41fc8d1a')


# ------------------------------------------------------------------ seed ranges / panels

def test_seed_ranges_registered_and_disjoint(monkeypatch):
    from tools.coopmwm import map_scenarios as ms
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(ms)
    assert ms.seed_scope(991724) == 'heldout' and ms.world(991724, 3200, .2)['repeat'] == 24
    assert ms.seed_scope(998400) == 'heldout' and ms.world(998591, 3200, .2)['repeat'] == 191
    assert ms.seed_scope(998600) is None
    s1 = T37.use(monkeypatch, D.SELECT_SETTING)
    seeds1 = [r['world']['seed'] for r in s1['worlds']]
    assert seeds1 == list(range(991724, 991772)) and s1['cell']['cell_key'] == 'D3200_p20'
    assert s1['panel_sha256'] == N.SETTINGS_PINS[D.SELECT_SETTING]['PANEL_SHA256']
    audit = json.loads((D.A41_DIR/'SEED_AUDIT.json').read_text(encoding='utf-8'))
    assert audit['panels']['stage1_selection']['panel_sha256'] == s1['panel_sha256']
    assert audit['panels']['stage2_confirmation']['panel_sha256'] == N.SETTINGS_PINS[D.CONFIRM_SETTING]['PANEL_SHA256']


def test_confirmation_panel_is_fresh_and_disjoint(monkeypatch):
    s2 = T37.use(monkeypatch, D.CONFIRM_SETTING)
    seeds2 = [r['world']['seed'] for r in s2['worlds']]
    assert seeds2 == list(range(998400, 998592)) and s2['cell']['cell_key'] == 'D3200_p20'
    never = [(964000, 964199), (962000, 962999), (977000, 977999), (998100, 998195), (998000, 998399)]
    assert not any(lo <= s <= hi for s in seeds2 for lo, hi in never)
    assert not set(seeds2) & set(range(991700, 991800))


# ------------------------------------------------------------------ specs

def test_stage1_specs_u0_plus_eight_and_no_duo_hold(root, monkeypatch):
    panel = T37.use(monkeypatch, D.SELECT_SETTING)
    waves = D.stage1_waves(panel, root)
    assert [w[0] for w in waves] == ['A41S_U0'] + [f'A41S_{x}' for x in D.LABELS]
    ids = [s['job_id'] for _, _, specs in waves for s in specs]
    assert len(ids) == 9*48 == len(set(ids)) and all(i.startswith('a41s_') for i in ids)
    assert {s['arm'] for _, _, specs in waves for s in specs} == {'U0', 'A37_PLAN', 'CELL_BANK'}
    doc = D.candidates()
    for (name, _, specs), cand in zip(waves[1:], doc['candidates']):
        ck = specs[0]['checkpoint']
        assert specs[0]['arm'] == cand['arm'] and ck['a37_sha256'] == D.A37_DIGEST and 'bank_cell_key' not in ck
        assert ck['a37_bank'] == dict(path=f'{D.HOST_ROOT}/{D.BANK_DIR}/{cand["bank"]}',
                                      file_sha256=cand['bank_file_sha256'])
        assert ck.get('a37_family') == cand['family'] and ck['cell_key'] == 'D3200_p20'
        assert [s['world']['seed'] for s in specs] == list(range(991724, 991772))


# ------------------------------------------------------------------ selection rule

def row(label, mean_j, groups, order=None, n_plans=None):
    order = D.LABELS.index(label) if order is None else order
    n = n_plans or (1 if label.startswith('S') else int(label[1]))
    return dict(label=label, order=order, n_plans=n, mean_J=mean_j, groups=groups)


def gs(mean, upper):
    return {g: (mean, upper) for g in G5}


def test_selection_rule_eligible_a():
    rows = [row('S1', 100.0, gs(5, 12)), row('S2', 90.0, gs(5, 9.9)), row('N2', 80.0, gs(9, 10.0)),
            row('N3', 70.0, gs(11, 12))]
    v = D.selection_rule(rows)
    assert v['decision'] == 'eligible-A' and v['eligible_A'] == ['S2', 'N2'] and v['pick'] == 'N2'
    assert 'N3' not in v['eligible_B'] and 'S1' in v['eligible_B']


def test_selection_rule_eligible_b_when_no_a():
    one_high = dict(gs(4, 9), **{'origin:L2': (9.5, 13.0)})
    rows = [row('S1', 100.0, one_high), row('S3', 95.0, gs(10.0, 14)), row('N4', 50.0, gs(10.5, 14))]
    v = D.selection_rule(rows)
    assert v['decision'] == 'eligible-B' and v['eligible_A'] == [] and v['eligible_B'] == ['S1', 'S3']
    assert v['pick'] == 'S3'                                         # lowest mean J among eligible-B (not N4)


def test_selection_rule_stop():
    rows = [row(x, 10.0*i, dict(gs(3, 5), **{'origin:L2': (10.01, 15)})) for i, x in enumerate(D.LABELS)]
    v = D.selection_rule(rows)
    assert v == dict(decision='STOP', eligible_A=[], eligible_B=[], pick=None,
                     report='no bank inside the bound (no stage 2)')


def test_selection_rule_missing_group_is_ineligible():
    g = gs(1, 2)
    g['class:cav'] = None
    assert D.selection_rule([row('S1', 1.0, g)])['decision'] == 'STOP'


def test_selection_rule_ties():
    ok = gs(1, 5)
    v = D.selection_rule([row('N2', 50.0, ok), row('S4', 50.0, ok), row('S2', 50.0, ok), row('N3', 49.0, gs(11, 12))])
    assert v['pick'] == 'S2'                    # tie on J -> fewer plans (S2, S4) -> earlier in the list (S2)
    v = D.selection_rule([row('N3', 50.0, ok), row('N2', 50.0, ok)])
    assert v['pick'] == 'N2'                    # tie on J -> fewer plans
    v = D.selection_rule([row('S5', 50.0, ok), row('S1', 50.0000001, ok)])
    assert v['pick'] == 'S5'                    # strictly lower J wins over list order


# ------------------------------------------------------------------ stage 2 refusal / conjunction / budget

def test_stage2_refused_without_sealed_selection(root, monkeypatch):
    panel = T37.use(monkeypatch, D.CONFIRM_SETTING)
    for arm in D.STAGE2_ARMS:
        with pytest.raises(N.Blocked, match='stage 2 refused'):
            D.stage2_specs(arm, panel, root)
    with pytest.raises(N.Blocked, match='stage 2 refused'):
        D.stage2_waves(panel, root)
    rt = runtime(root)
    with pytest.raises(N.Blocked, match='stage 2 refused'):
        rt.reserve(None, [dict(id='a41c_u0_0000', sim_s=1200)])
    assert D.charged(root) == 0
    doc = D.candidates()
    base = dict(schema='coopmwm.a41.selection.v1', candidates_doc_sha256=doc['doc_sha256'],
                panel_sha256=N.SETTINGS_PINS[D.SELECT_SETTING]['PANEL_SHA256'])
    N.publish(root/D.SELECTION_FILE, N.seal(dict(base, decision='STOP', pick=None), 'receipt_sha256'))
    with pytest.raises(N.Blocked, match='STOP'):
        D.stage2_waves(panel, root)
    (root/D.SELECTION_FILE).unlink()
    good = N.seal(dict(base, decision='eligible-A', pick=dict(label='N2', candidate=D.candidate('N2'))), 'receipt_sha256')
    bad = dict(good, pick=dict(label='N2', candidate=D.candidate('N3')))      # seal broken
    N.publish(root/D.SELECTION_FILE, bad)
    with pytest.raises(N.Blocked):
        D.stage2_waves(panel, root)
    N.publish(root/D.SELECTION_FILE, good, replace=True)
    waves = D.stage2_waves(panel, root)
    assert [w[0] for w in waves] == ['A41C_U0', 'A41C_DUO_HOLD', 'A41C_LEARNED_TRIGGER', 'A41C_SEL']
    sel = waves[3][2]
    assert sel[0]['arm'] == 'CELL_BANK' and sel[0]['checkpoint']['a37_bank']['file_sha256'] == doc['banks']['A41_N2_D3200_p20.json']
    assert waves[1][2][0]['checkpoint']['bank_cell_key'] == 'D2800_p20' == waves[2][2][0]['checkpoint']['bank_cell_key']
    assert waves[1][2][0]['checkpoint']['a32_sha256'] == D.A32_DIGEST
    assert all(len(w[2]) == 192 for w in waves)


def cmp_row(ci_upper, all_pass):
    return dict(total_minus_ref=dict(ci95=[ci_upper-100, ci_upper]), groups=dict(all_pass=all_pass))


def test_primary_conjunction():
    for upper, gp, want in ((-1.0, True, True), (-1.0, False, False), (0.0, True, False), (5.0, False, False)):
        v = D.primary_verdict(cmp_row(upper, None), cmp_row(0, gp))
        assert v['supported'] is want and v['reading'] == ('SUPPORTED' if want else 'NOT SUPPORTED')
    assert D.primary_verdict(cmp_row(1.0, None), cmp_row(0, False))['failed_parts'] == ['a', 'b']


def test_budget_fits_and_ledger_cap(root):
    b = D.planned_budget()
    assert (b['identity'], b['stage1'], b['stage2'], b['planned']) == (4_800, 518_400, 921_600, 1_444_800)
    assert b['fits'] and b['retry_reserve'] == 115_200 and b['cap'] == 1_560_000
    rt = runtime(root)
    with pytest.raises(N.Blocked, match='one stage'):
        rt.reserve(None, [dict(id='a41s_u0_0000', sim_s=1), dict(id='a41i_p20c_u0_0000', sim_s=1)])
    rt.reserve(None, [dict(id='a41s_x_0000', sim_s=1_559_000)])
    with pytest.raises(N.Blocked, match='d3200bound_a41 cap'):
        rt.reserve(None, [dict(id='a41s_x_0001', sim_s=1_200)])
    rows = N.fe.ledger_rows(root)
    assert {r['phase'] for r in rows} == {D.A41_PHASE}


def test_end_to_end_select_confirm_analyse(root, monkeypatch):
    # the fake vehicle log has no origin:L0 vehicles; register the other four groups for this fixture only
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))
    panel = T37.use(monkeypatch, D.SELECT_SETTING)
    fams = D.candidates()['priority_order_P']
    waves = D.stage1_waves(panel, root)
    out = D.execute_with_retry(root, waves, workers=4, chunk=96, log=lambda x: None,
                               runtime=runtime(root, factors={fams[1]: .9, fams[3]: .95, 'CELL_BANK': 1.05}))
    assert out['status'] == 'PASS' and D.charged(root) == 9*48*1200
    with pytest.raises(N.Blocked, match='stage 2 refused'):
        D.require_selection(root)
    sel = D.select(root, panel)
    assert sel['decision'] in ('eligible-A', 'eligible-B') and sel['pick']['label'] == 'S2'
    assert {c['label'] for c in sel['candidates']} == set(D.LABELS) and 'DUO_HOLD' not in json.dumps(sel['sources'])
    with pytest.raises(N.Blocked, match='one shot'):
        D.select(root, panel)
    assert D.reselect(root, panel)['identical'] is True
    panel2 = T37.use(monkeypatch, D.CONFIRM_SETTING)
    waves2 = D.stage2_waves(panel2, root)
    assert waves2[3][2][0]['checkpoint']['a37_family'] == fams[1]
    out = D.execute(root, waves2, workers=4, chunk=200, log=lambda x: None,
                    runtime=runtime(root, factors={fams[1]: .97, 'LEARNED_TRIGGER': .99}))
    assert out['status'] == 'PASS' and D.charged(root) == 9*48*1200 + 4*192*1200
    res = D.analyse(root, panel2)
    assert res['selection']['pick'] == 'S2' and res['primary']['a_total_ci_upper_below_0'] is True
    assert res['primary']['supported'] is res['primary']['b_group_rule_vs_u0']
    pairs = [(c['treatment'], c['reference']) for c in res['contrasts']]
    assert pairs[:2] == [('SEL', 'DUO_HOLD'), ('SEL', 'U0')] and ('SEL', 'LEARNED_TRIGGER') in pairs
    assert 'tail_cvar90' in res['contrasts'][0] and res['bootstrap']['seed'] == 20261104
    assert res['lt_group_result'] is not None and set(res['arms']) == set(D.STAGE2_ARMS)


def test_failed_episode_is_retried_once(root, monkeypatch):
    panel = T37.use(monkeypatch, D.SELECT_SETTING)
    waves = D.stage1_waves(panel, root)[:1]
    logs = []
    out = D.execute_with_retry(root, waves, workers=4, chunk=48, log=logs.append,
                               runtime=runtime(root, fail={'a41s_u0_0003'}))
    assert out['status'] == 'PASS' and any('replay_authorized' in x for x in logs)
    # the fake failure aborts the whole 48-episode dispatch: all 48 are interrupted, charged, and replayed once (#r1)
    assert D.charged(root) == 2*48*1200 and len(N.fe.ledger_rows(root)) == 96
    assert deepcopy(len(list(root.glob('REPLAY_AUTHORIZED_*.json')))) == 1
