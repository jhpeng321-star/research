"""CPU-only tests for the amendment-50 drivers (fake runner; no simulator): the A50 drivers are the A42 / A45 code
with only the amendment names replaced (phases, caps, job ids, waves, files, host base, stop files, settings / seed
blocks, candidates pin), nothing collides with A42 / A45, the A42 / A45 modules are untouched by importing A50, and the
A50 big-bank -> diverse chain runs end to end on fake records."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D42  # noqa: E402
import a45_divp60 as E45  # noqa: E402
import a50_divp60 as A50D  # noqa: E402
import a50_fullp60 as A50  # noqa: E402
import nm_finaleval as N  # noqa: E402
import test_a37_celltailor as T37  # noqa: E402
import test_a42_bigp60 as T42  # noqa: E402

D, E = A50.D, A50D.E
N_DIV = 3


def test_a42_a45_modules_keep_their_names_after_importing_a50():
    assert (D42.A42_PHASE, D42.STOP_A42, D42.VERIFY_SETTING, D42.BANK_DIR) == \
        ('bigp60_a42', 'STOP_A42', 'a42v_D2800_p60', 'a42_banks')
    assert D42.CANDS_FILE.parent.name == 'a42' and D42.A42Runtime.phase == 'bigp60_a42' and D42.AMENDMENT == '42'
    assert D42.HOST_BASE == os.environ.get('A42_HOST_BASE', '/root/a42v')
    assert (E45.A45_PHASE, E45.D, E45.CONFIRM_WAVE, E45.A45Runtime.phase) == ('divp60_a45', D42, 'A45C_DIVERSE_BANK',
                                                                               'divp60_a45')
    assert D is not D42 and E is not E45 and E.D is D


def test_a50_names_are_disjoint_from_a42_a45():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, E.A45_PHASE, E.A45_CAP_SIM_S) == ('bigp60_a50', 720_000, 'divp60_a50', 480_000)
    assert D.A42Runtime.phase == 'bigp60_a50' and E.A45Runtime.phase == 'divp60_a50'
    assert D.HOST_BASE == os.environ.get('A50_HOST_BASE', '/root/a50v') and D.HOST_ROOT == f'{D.HOST_BASE}/root'
    assert D.HOST_BASE != D42.HOST_BASE
    assert D.SETTINGS == ('a50v_D2800_p60', 'a50c_D2800_p60') and D.CANDS_FILE == A50.HERE/'a50'/'CANDIDATES_D2800_p60.json'
    assert A50.CANDIDATES_DOC_SHA256 == D.CANDIDATES_DOC_SHA256 == '3b63698634a0ac7fc3217a1e73580c0f766211f001b0600fadd0fc48359d4ee5'  # pinned 2026-11-11
    a50 = [D.STOP_A42, E.STOP_A45, D.BANK_DIR, D.CANDS_BANK_FILE, D.BIG_BANK_FILE, D.ADMISSION_FILE, D.RESULTS_FILE,
           D.LOG_FILE, E.ADMISSION_FILE, E.CANDS_BANK, E.DIVERSE_BANK_FILE, E.RESULTS_FILE, E.LOG_FILE, D.IDENT_WAVE,
           E.CONFIRM_WAVE]
    old = [D42.STOP_A42, E45.STOP_A45, D42.BANK_DIR, D42.CANDS_BANK_FILE, D42.BIG_BANK_FILE, D42.ADMISSION_FILE,
           D42.RESULTS_FILE, D42.LOG_FILE, E45.ADMISSION_FILE, E45.CANDS_BANK, E45.DIVERSE_BANK_FILE, E45.RESULTS_FILE,
           E45.LOG_FILE, D42.IDENT_WAVE, E45.CONFIRM_WAVE]
    assert len(set(a50)) == len(a50) and not set(a50) & set(old)
    tags = [D.VERIFY_TAG, D.CONFIRM_TAG, D.IDENT_TAG, E.VERIFY_TAG, E.CONFIRM_TAG]
    assert all(t.startswith('a50') for t in tags) and len(set(tags)) == len(tags)
    assert all(t.startswith(E.JOB_PREFIX) for t in (E.VERIFY_TAG, E.CONFIRM_TAG))
    assert not any(t.startswith(E.JOB_PREFIX) for t in (D.VERIFY_TAG, D.CONFIRM_TAG, D.IDENT_TAG))
    assert D.RUNNER_RANGES_ATTR == 'A50_RANGES' and D.AMENDMENT == E.AMENDMENT == '50'
    # no amendment name may be frozen into a def-time default or class attribute of the reused code
    import inspect
    for mod in (D, E):
        for name, f in vars(mod).items():
            if inspect.isfunction(f) and f.__module__ == mod.__name__:
                vals = list(f.__defaults__ or ())+list((f.__kwdefaults__ or {}).values())
                assert not any(isinstance(v, str) and ('42' in v or '45' in v) for v in vals), name


def test_rules_and_budget_are_the_a42_a45_ones():
    assert D.PRIMARY_RULE == D42.PRIMARY_RULE and E.PRIMARY_RULE == E45.PRIMARY_RULE and E.NOT_BUILT == E45.NOT_BUILT
    assert (D.SECONDARY, D.CONTEXT, D.CONFIRM_ARMS, D.RUNNER_ARM, D.IDENT_PANELS) == \
        (D42.SECONDARY, D42.CONTEXT, D42.CONFIRM_ARMS, D42.RUNNER_ARM, D42.IDENT_PANELS)
    b, b42 = D.budget_plan(), D42.budget_plan()
    assert b == dict(b42, phase='bigp60_a50')
    e = E.budget_plan()
    assert e['phase'] == 'divp60_a50' and e['separate_from'] == dict(phase='bigp60_a50', cap_sim_s=720_000)
    assert {k: v for k, v in e.items() if k not in ('phase', 'separate_from')} == \
        {k: v for k, v in E45.budget_plan().items() if k not in ('phase', 'separate_from')}
    for name in ('admit', 'analyse', 'execute', 'primary_verdict', '_contrast', '_a37_specs', 'candidates'):
        assert getattr(D, name).__code__.co_code == getattr(D42, name).__code__.co_code
    for name in ('admit', 'analyse', 'build_diverse_bank', 'diverse_candidates', 'check_search'):
        assert getattr(E, name).__code__.co_code == getattr(E45, name).__code__.co_code


def test_runner_registers_the_a50_blocks_without_moving_older_ones():
    from tools.coopmwm import map_scenarios as ms
    from tools.coopmwm import nm_eval_runner as R
    assert R.A50_RANGES == ((992800, 992899), (996800, 996899)) and R.A42_RANGES == ((992000, 992099), (998600, 998699))
    assert R._eval_base(992823) == 992800 and R._eval_base(996895) == 996800 and R._eval_base(992899) == 992800
    assert R._eval_base(992799) is None and R._eval_base(992900) is None and R._eval_base(996799) is None
    assert R._eval_base(996900) is None and R._eval_base(992023) == 992000 and R._eval_base(998695) == 998600
    used = [(992000, 992099), (998600, 998699), (992400, 992499), (991724, 991771), (998400, 998591),
            (964000, 964199), (962000, 962999), (977000, 977999)] + [(lo, hi) for lo, hi, _ in R.EVAL_HELDOUT_RANGES
                                                                     if (lo, hi) not in R.A50_RANGES]
    for lo, hi in R.A50_RANGES:
        assert all(hi < a or lo > b for a, b in used), (lo, hi)
        assert ms.seed_scope(lo) != 'train'
    R.install_eval_ranges(ms)
    assert ms.seed_scope(992800) == 'heldout' and ms.seed_scope(996800) == 'heldout'


def test_panels_are_the_audited_a50_blocks(monkeypatch):
    v = T42.use(monkeypatch, D.VERIFY_SETTING)
    assert [r['world']['seed'] for r in v['worlds']] == list(range(992800, 992824))
    assert v['cell']['cell_key'] == 'D2800_p60' and N.SETTINGS_PINS[D.VERIFY_SETTING]['PANEL_SHA256'] == v['panel_sha256']
    c = T42.use(monkeypatch, D.CONFIRM_SETTING)
    assert [r['world']['seed'] for r in c['worlds']] == list(range(996800, 996896))
    assert N.SETTINGS_PINS[D.CONFIRM_SETTING]['PANEL_SHA256'] == c['panel_sha256']
    assert c['contract']['bootstrap'] == dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
    seeds = {r['world']['seed'] for p in (v, c) for r in p['worlds']}
    for other in ('nm_eval_trainblock_wz3to2_a42v_D2800_p60.json', 'nm_eval_panel_wz3to2_a42c_D2800_p60.json',
                  'nm_eval_panel_wz3to2_a35_D2800_p60.json'):
        assert not seeds & {r['world']['seed'] for r in N.read(Path(N.HERE)/other)['worlds']}
    audit = json.loads((A50.A50_DIR/'SEED_AUDIT.json').read_text(encoding='utf-8'))
    assert audit['verdict'] == dict(verification='UNUSED', confirmation='UNUSED')
    assert audit['panels']['verification']['panel_sha256'] == v['panel_sha256']
    assert audit['panels']['confirmation']['panel_sha256'] == c['panel_sha256']


def test_select_stamps_a50_and_candidates_refuse_unpinned_a42_or_edited(tmp_path):
    from tools.coopmwm import a42_bigbank as B
    search = tmp_path/'STRATEGY_SEARCH_D2800_p60.json'
    s = json.loads(T42.FIX.read_text(encoding='utf-8'))
    search.write_text(json.dumps(dict(s, cell=D.CELL)), encoding='utf-8')
    out = tmp_path/'a50'/'CANDIDATES_D2800_p60.json'
    doc = D.select(search, out)
    fresh = B.select(dict(s, cell=D.CELL), N.file_sha(search))
    assert doc['amendment'] == '50' and doc['doc_sha256'] == B.doc_sha(doc) and 'a50' in doc
    assert doc['candidates'] == fresh['candidates'] and doc['diverse'] == fresh['diverse']   # rule unchanged
    with pytest.raises(N.Blocked, match='exists'):
        D.select(search, out)
    with pytest.raises(N.Blocked, match='not the registered selection'):
        D.candidates(out)                                                   # unpinned -> refuse
    assert len(D.candidates(out, pinned=doc['doc_sha256'])) == 6
    got = E.check_search(search, out, pinned=doc['doc_sha256'])
    assert got['n_top'] == 6 and got['n_diverse'] == len(doc['diverse']['candidates'])
    a42 = tmp_path/'a42.json'
    D42.select(search, a42)                                                 # an A42 document is never A50's
    with pytest.raises(N.Blocked, match='cell / K'):
        D.candidates(a42, pinned=N.read(a42)['doc_sha256'])
    out.write_text(json.dumps(dict(doc, k=7)), encoding='utf-8')
    with pytest.raises(N.Blocked, match='doc_sha256 differs'):
        D.candidates(out, pinned=doc['doc_sha256'])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    src = T37.mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    out = tmp_path/'root'
    for f in T37.BIND:
        (out/f).parent.mkdir(parents=True, exist_ok=True)
        (out/f).write_bytes((src/f).read_bytes())
    from tools.coopmwm import a42_bigbank as B
    s = json.loads(T42.FIX.read_text(encoding='utf-8'))
    doc = B.select(dict(s, cell=D.CELL), 'fixture')
    top, div = doc['candidates'], doc['diverse']['candidates'][:N_DIV]
    monkeypatch.setattr(D, 'candidates', lambda *a, **k: top)
    monkeypatch.setattr(E, 'diverse_candidates', lambda *a, **k: div)
    D.prepare(out)
    E.prepare(out)
    return out, top, div


def rt50(root, **kw):
    return D.A42Runtime(T42.Fake(N.prerun_plan(root), **kw), root)


def rt50d(root, **kw):
    return E.A45Runtime(T42.Fake(N.prerun_plan(root), **kw), root)


def test_specs_use_a50_ids_host_paths_and_the_unchanged_a37_executor(setup, monkeypatch):
    from tools.coopmwm import a37_cellbank as C
    root, top, div = setup
    assert (root/'a50_banks'/'A50_CANDS_D2800_p60.json').is_file()
    assert (root/'a50_banks'/'A50DIV_CANDS_D2800_p60.json').is_file()
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    waves = D.verify_waves(panel, root)
    assert [w[0] for w in waves] == ['A50V_U0']+[f'A50V_{c["family"]}' for c in top]
    ids = [s['job_id'] for _, _, sp in waves for s in sp]
    assert len(ids) == 168 == len(set(ids)) and all(i.startswith('a50v_') for i in ids)
    ck = waves[1][2][0]['checkpoint']
    assert ck['a37_sha256'] == C.a37_digest() and ck['a37_bank']['path'] == f'{D.HOST_ROOT}/a50_banks/A50_CANDS_D2800_p60.json'
    assert [s['world']['seed'] for s in waves[0][2]] == list(range(992800, 992824))
    dw = E.verify_waves(panel, root)
    assert [w[0] for w in dw] == [f'A50DV_{c["family"]}' for c in div] and E.u0_wave(panel, root)[0] == 'A50V_U0'
    assert all(s['job_id'].startswith('a50dv_') for _, _, sp in dw for s in sp)
    assert [s['job_id'] for s in D.ident_specs(root)] == ['a50i_p20c_u0_0000', 'a50i_p20c_u0_0001',
                                                          'a50i_a35d3200_u0_0000', 'a50i_a35d3200_u0_0001']
    cpanel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    lt = D.confirm_specs('LEARNED_TRIGGER', cpanel, root)
    assert lt[0]['job_id'] == 'a50c_learned_trigger_0000' and [s['world']['seed'] for s in lt] == list(range(996800, 996896))
    with pytest.raises(N.Blocked, match='ADMISSION'):
        D.confirm_specs('BIG_BANK', cpanel, root)


def test_ledger_caps_and_job_prefixes(setup):
    root = setup[0]
    rt = rt50(root)
    with pytest.raises(N.Blocked, match='bigp60_a50 cap'):
        rt.check_capacity(720_001)
    for bad in ('a42v_x_0000', 'a45v_x_0000'):
        with pytest.raises(N.Blocked, match='job ids are a50'):
            rt.reserve(None, [dict(id=bad, sim_s=1)])
    rt.reserve(None, [dict(id='a50v_x_0000', sim_s=700_000)])
    d = rt50d(root)
    with pytest.raises(N.Blocked, match='job ids are a50d'):
        d.reserve(None, [dict(id='a50v_y_0000', sim_s=1)])
    with pytest.raises(N.Blocked, match='divp60_a50 cap'):
        d.check_capacity(480_001)
    d.reserve(None, [dict(id='a50dv_y_0000', sim_s=479_000)])
    with pytest.raises(N.Blocked, match='divp60_a50 cap'):
        d.reserve(None, [dict(id='a50dc_z_0000', sim_s=1_200)])
    assert {r['phase'] for r in N.fe.ledger_rows(root)} == {'bigp60_a50', 'divp60_a50'}
    assert E.charged_phase(root) == 479_000 and E.charged_phase(root, D.A42_PHASE) == 700_000


def test_stop_files_are_the_a50_ones(setup, monkeypatch):
    root = setup[0]
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    (root/'STOP_A42').write_text('')
    (root/'STOP_A45').write_text('')                    # A42 / A45 stop files do not stop A50
    (root/'STOP_A50').write_text('')
    out = D.execute(root, D.verify_waves(panel, root)[:1], workers=4, chunk=24, log=lambda x: None, runtime=rt50(root))
    assert out['status'] == 'STOPPED' and D.charged(root) == 0
    (root/'STOP_A50DIV').write_text('')
    assert E.run_stage(root, E.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None, panel=panel,
                       runtime=rt50d(root))['status'] == 'STOPPED'


def test_end_to_end_a50_big_then_diverse(setup, monkeypatch):
    root, top, div = setup
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))
    tf, df = [c['family'] for c in top], [c['family'] for c in div]
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    out = D.execute_with_retry(root, D.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None,
                               runtime=rt50(root, factors={tf[1]: .9}, fail={'a50v_u0_0003'}))
    assert out['status'] == 'PASS' and len(list(root.glob('REPLAY_AUTHORIZED_*.json'))) == 1
    adm = D.admit(root, panel)
    assert adm['admitted'] == [tf[1]] and adm['amendment'] == '50' and adm['schema'] == 'coopmwm.a50.admission.v1'
    assert (root/'ADMISSION_A50_D2800_p60.json').is_file() and not (root/'ADMISSION_D2800_p60.json').exists()
    assert (root/'a50_banks'/'A50_BIG_BANK_D2800_p60.json').is_file()
    cpanel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    waves = D.stage_waves('confirm', D.CONFIRM_SETTING, cpanel, root)
    assert [w[0] for w in waves] == ['A50C_U0', 'A50C_DUO_HOLD', 'A50C_LEARNED_TRIGGER', 'A50C_BIG_BANK']
    assert D.execute(root, waves, workers=4, chunk=96, log=lambda x: None,
                     runtime=rt50(root, factors=dict(CELL_BANK=.97)))['status'] == 'PASS'
    big = D.charged(root)
    retried = big-(168+4*96)*1200                    # the one charged replay round (charged-unpublished jobs)
    assert retried > 0 and retried % 1200 == 0 and big <= 720_000 and E.charged_phase(root, 'bigp60_a50') == big
    res = D.analyse(root, cpanel)
    assert res['primary']['verdict'] == 'SUPPORTED' and res['amendment'] == '50' and res['label'].startswith('amendment 50')
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    assert E.run_stage(root, E.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None, panel=panel,
                       runtime=rt50d(root, factors={df[0]: .95, df[2]: .9}))['status'] == 'PASS'
    assert E.charged_phase(root) == N_DIV*24*1200                       # U0 shared: no diverse-part U0 episode
    adm2 = E.admit(root, panel)
    assert adm2['admitted_diverse'] == [df[2], df[0]] and adm2['u0_shared_from']['wave'] == 'A50V_U0'
    assert adm2['a42_admission']['receipt_sha256'] == adm['receipt_sha256'] and adm2['schema'] == 'coopmwm.a50div.admission.v1'
    assert adm2['diverse_bank']['path'] == f'{D.HOST_ROOT}/a50_banks/A50_DIVERSE_BANK_D2800_p60.json'
    cpanel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    w = E.stage_waves('confirm', D.CONFIRM_SETTING, cpanel, root)
    assert [x[0] for x in w] == ['A50DC_DIVERSE_BANK'] and w[0][2][0]['job_id'] == 'a50dc_diverse_bank_0000'
    assert E.run_stage(root, w, workers=4, chunk=96, log=lambda x: None, panel=cpanel,
                       runtime=rt50d(root, factors=dict(CELL_BANK=.97)))['status'] == 'PASS'
    r2 = E.analyse(root, cpanel)
    assert [(c['treatment'], c['reference']) for c in r2['contrasts']] == \
        [('DIVERSE_BANK', 'DUO_HOLD'), ('DIVERSE_BANK', 'U0'), ('DIVERSE_BANK', 'BIG_BANK')]
    assert r2['primary']['verdict'] == 'SUPPORTED' and r2['amendment'] == '50'
    assert {r['phase'] for r in N.fe.ledger_rows(root)} == {'bigp60_a50', 'divp60_a50'}
    assert not list(root.glob('*A42*')) and not list(root.glob('*A45*'))


def test_cli_budget_and_check_search_routing(capsys, tmp_path):
    A50.main(['budget'])
    assert json.loads(capsys.readouterr().out)['phase'] == 'bigp60_a50'
    A50D.main(['budget'])
    assert json.loads(capsys.readouterr().out)['phase'] == 'divp60_a50'
    search = tmp_path/'STRATEGY_SEARCH_D2800_p60.json'
    s = json.loads(T42.FIX.read_text(encoding='utf-8'))
    search.write_text(json.dumps(dict(s, cell=D.CELL)), encoding='utf-8')
    out = tmp_path/'C.json'
    D.select(search, out)
    with pytest.raises(N.Blocked, match='not the registered selection'):    # unpinned A50 document
        A50.main(['check-search', '--search', str(search), '--candidates', str(out)])
