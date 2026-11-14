"""CPU-only tests for the amendment-45 driver (fake runner; no simulator): budget phase divp60_a45 (cap 480,000,
separate from bigp60_a42) and its refusal, DIVERSE_BANK construction with zero / some diverse admissions, and the
A42 -> A45 chain end to end on fake records (U0 shared, same 24 seeds, extra arm on the same 96 worlds)."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D  # noqa: E402
import a45_divp60 as E  # noqa: E402
import nm_finaleval as N  # noqa: E402
import test_a37_celltailor as T37  # noqa: E402
import test_a42_bigp60 as T42  # noqa: E402

N_DIV = 3


def fixture_doc():
    from tools.coopmwm import a42_bigbank as B
    s = json.loads(T42.FIX.read_text(encoding='utf-8'))
    return B.select(dict(s, cell=D.CELL), 'fixture')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    src = T37.mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    out = tmp_path/'root'
    for f in T37.BIND:
        (out/f).parent.mkdir(parents=True, exist_ok=True)
        (out/f).write_bytes((src/f).read_bytes())
    doc = fixture_doc()
    top, div = doc['candidates'], doc['diverse']['candidates'][:N_DIV]
    monkeypatch.setattr(D, 'candidates', lambda *a, **k: top)
    monkeypatch.setattr(E, 'diverse_candidates', lambda *a, **k: div)
    D.prepare(out)
    E.prepare(out)
    return out, top, div


def rt42(root, **kw):
    return D.A42Runtime(T42.Fake(N.prerun_plan(root), **kw), root)


def rt45(root, **kw):
    return E.A45Runtime(T42.Fake(N.prerun_plan(root), **kw), root)


def test_budget_plan_is_separate_and_capped():
    b = E.budget_plan()
    assert (b['phase'], b['cap_sim_s']) == ('divp60_a45', 480_000) and E.A45_PHASE != D.A42_PHASE
    assert b['verification_max'] == 12*24*1200 and b['confirmation'] == 96*1200
    assert b['verification_max']+b['confirmation']+b['retry_reserve'] == 480_000 and b['retry_reserve'] >= 0
    assert b['own_ledger_total_allocation'] == 720_000+480_000


def test_a45_ledger_refuses_to_exceed_its_cap(setup):
    root = setup[0]
    rt42(root).reserve(None, [dict(id='a42v_x_0000', sim_s=700_000)])     # A42 spend never counts against A45
    rt = rt45(root)
    rt.check_capacity(480_000)
    with pytest.raises(N.Blocked, match='divp60_a45 cap'):
        rt.check_capacity(480_001)
    with pytest.raises(N.Blocked, match='a45'):
        rt.reserve(None, [dict(id='a42c_y_0000', sim_s=1)])
    rt.reserve(None, [dict(id='a45v_x_0000', sim_s=479_000)])
    with pytest.raises(N.Blocked, match='divp60_a45 cap'):
        rt.reserve(None, [dict(id='a45c_y_0000', sim_s=1_200)])
    assert E.charged_phase(root) == 479_000 and E.charged_phase(root, D.A42_PHASE) == 700_000
    with pytest.raises(N.Blocked, match='left in divp60_a45'):
        E.stage_fits(root, [('w', 'episode', [{}])])
    with pytest.raises(N.Blocked, match='left in divp60_a45'):
        E.run_stage(root, [('w', 'episode', [{}])], workers=1, chunk=1, log=lambda x: None, panel=None,
                    runtime=rt45(root))


def test_build_diverse_bank_zero_and_some_admissions():
    final = json.loads(T42.FIX.read_text(encoding='utf-8'))['final']
    top = [dict(family=x['strategy']['name'], strategy=x['strategy']) for x in final[:3]]
    div = [dict(family=x['strategy']['name'], strategy=x['strategy']) for x in final[3:6]]

    def v(ok, lb):
        return dict(PASS=ok, lb95_gain_veh_s=lb)
    tv = {top[0]['family']: v(True, 5.0), top[1]['family']: v(False, -1.0), top[2]['family']: v(True, 9.0)}
    none = {d['family']: v(False, -2.0) for d in div}
    assert E.build_diverse_bank(top, tv, div, none) is None            # no diverse admission: not built
    assert E.build_diverse_bank(top, tv, div, {}) is None
    some = dict(none, **{div[1]['family']: v(True, 7.0)})
    plans = E.build_diverse_bank(top, tv, div, some)
    assert [p['family'] for p in plans] == [top[2]['family'], div[1]['family'], top[0]['family']]
    only = E.build_diverse_bank(top, {k: v(False, 0.0) for k in tv}, div, some)
    assert [p['family'] for p in only] == [div[1]['family']]


def test_verify_waves_are_the_diverse_plans_on_the_same_24_worlds(setup, monkeypatch):
    root, top, div = setup
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    waves = E.verify_waves(panel, root)
    assert [w[0] for w in waves] == [f'A45V_{c["family"]}' for c in div]
    a42 = D.verify_waves(panel, root)
    assert all([s['world']['seed'] for s in w[2]] == [s['world']['seed'] for s in a42[0][2]] for w in waves)
    assert E.u0_wave(panel, root)[0] == 'A42V_U0'
    ids = [s['job_id'] for _, _, specs in waves for s in specs]
    assert len(ids) == N_DIV*24 == len(set(ids)) and all(i.startswith('a45v_') for i in ids)
    ck = waves[0][2][0]['checkpoint']
    assert waves[0][2][0]['arm'] == 'A37_PLAN' and ck['a37_bank']['path'].endswith(E.CANDS_BANK)


def _a42_verify_admit(root, monkeypatch, factors):
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    out = D.execute_with_retry(root, D.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None,
                               runtime=rt42(root, factors=factors))
    assert out['status'] == 'PASS'
    return D.admit(root, panel), panel


def test_end_to_end_a42_then_a45(setup, monkeypatch):
    root, top, div = setup
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))
    tf, df = [c['family'] for c in top], [c['family'] for c in div]
    adm42, panel = _a42_verify_admit(root, monkeypatch, {tf[1]: .9})
    assert adm42['admitted'] == [tf[1]]
    cpanel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    with pytest.raises(N.Blocked, match='ADMISSION_A45'):
        E.confirm_specs(cpanel, root)
    # the chain order: every A42 stage (confirmation included) finishes before any A45 spend
    waves = D.stage_waves('confirm', D.CONFIRM_SETTING, cpanel, root)
    assert D.execute(root, waves, workers=4, chunk=96, log=lambda x: None,
                     runtime=rt42(root, factors=dict(CELL_BANK=.97)))['status'] == 'PASS'
    a42_spend = D.charged(root)
    assert a42_spend == 168*1200+4*96*1200
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    out = E.run_stage(root, E.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None, panel=panel,
                      runtime=rt45(root, factors={df[0]: .95, df[2]: .9}))
    assert out['status'] == 'PASS'
    assert E.charged_phase(root) == N_DIV*24*1200 and D.charged(root) == a42_spend+N_DIV*24*1200
    assert E.charged_phase(root, D.A42_PHASE) == a42_spend               # U0 shared: no a45 U0 episode
    adm = E.admit(root, panel)
    assert adm['admitted_diverse'] == [df[2], df[0]] and adm['diverse_bank_built']
    assert adm['diverse_bank_families'][0] == tf[1] and set(adm['diverse_bank_families']) == {tf[1], df[0], df[2]}
    assert adm['admitted_classes'] == {df[0]: div[0]['class_key'], df[2]: div[2]['class_key']}
    assert adm['u0_shared_from']['wave'] == 'A42V_U0'
    with pytest.raises(N.Blocked, match='one shot'):
        E.admit(root, panel)
    bank = json.loads((root/D.BANK_DIR/E.DIVERSE_BANK_FILE).read_text(encoding='utf-8'))
    assert [r['family'] for r in bank['rows']] == adm['diverse_bank_families']
    cpanel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    w45 = E.stage_waves('confirm', D.CONFIRM_SETTING, cpanel, root)
    assert [w[0] for w in w45] == ['A45C_DIVERSE_BANK'] and len(w45[0][2]) == 96
    s0 = w45[0][2][0]
    assert s0['arm'] == 'CELL_BANK' and s0['job_id'] == 'a45c_diverse_bank_0000'
    assert [s['world']['seed'] for s in w45[0][2]] == [s['world']['seed'] for s in waves[0][2]]
    out = E.run_stage(root, w45, workers=4, chunk=96, log=lambda x: None, panel=cpanel,
                      runtime=rt45(root, factors=dict(CELL_BANK=.97)))
    assert out['status'] == 'PASS' and E.charged_phase(root) == N_DIV*24*1200+96*1200
    res = E.analyse(root, cpanel)
    pairs = [(c['treatment'], c['reference']) for c in res['contrasts']]
    assert pairs == [('DIVERSE_BANK', 'DUO_HOLD'), ('DIVERSE_BANK', 'U0'), ('DIVERSE_BANK', 'BIG_BANK')]
    assert res['primary']['verdict'] == 'SUPPORTED' and res['primary']['rule'] == E.PRIMARY_RULE
    assert res['contrasts'][2]['big_bank_is_u0'] is False
    assert [p['family'] for p in res['admitted_diverse_plans']] == [df[2], df[0]]
    assert res['admitted_diverse_plans'][0]['class_key'] == div[2]['class_key']
    assert D.analyse(root, cpanel)['primary']['verdict'] == 'SUPPORTED'   # A42 reading unchanged by the A45 arm


def test_no_diverse_admission_means_no_diverse_bank_and_no_arm(setup, monkeypatch):
    root, top, div = setup
    _a42_verify_admit(root, monkeypatch, {})
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    assert E.run_stage(root, E.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None, panel=panel,
                       runtime=rt45(root))['status'] == 'PASS'
    adm = E.admit(root, panel)
    assert adm['admitted_diverse'] == [] and not adm['diverse_bank_built'] and adm['diverse_bank'] is None
    assert not (root/D.BANK_DIR/E.DIVERSE_BANK_FILE).exists() and adm['consequence'] == E.NOT_BUILT
    cpanel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    assert E.stage_waves('confirm', D.CONFIRM_SETTING, cpanel, root) == []
    res = E.analyse(root, cpanel)
    assert res['primary']['verdict'] == 'NOT SUPPORTED' and res['primary']['note'] == E.NOT_BUILT


def test_stop_file_stops_a45_before_dispatch(setup, monkeypatch):
    root = setup[0]
    (root/E.STOP_A45).write_text('')
    panel = T42.use(monkeypatch, D.VERIFY_SETTING)
    assert E.run_stage(root, E.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None, panel=panel,
                       runtime=rt45(root))['status'] == 'STOPPED'
    assert E.charged_phase(root) == 0


def test_check_search_matches_the_pinned_document(tmp_path):
    from tools.coopmwm import a42_bigbank as B
    search = tmp_path/'STRATEGY_SEARCH_D2800_p60.json'
    s = json.loads(T42.FIX.read_text(encoding='utf-8'))
    search.write_text(json.dumps(dict(s, cell=D.CELL)), encoding='utf-8')
    out = tmp_path/'CANDIDATES_D2800_p60.json'
    doc = D.select(search, out)
    assert 'diverse' in doc and doc['diverse']['amendment'] == '45'
    got = E.check_search(search, out, pinned=doc['doc_sha256'])
    assert got['n_top'] == 6 and got['n_diverse'] == len(doc['diverse']['candidates'])
    assert len(E.diverse_candidates(out, pinned=doc['doc_sha256'])) == got['n_diverse']
    other = tmp_path/'other.json'
    other.write_text(json.dumps(dict(s, cell=D.CELL, final=s['final'][:-1])), encoding='utf-8')
    with pytest.raises(N.Blocked, match='not the pinned search'):
        E.check_search(other, out, pinned=doc['doc_sha256'])
    edited = B.seal(dict(doc, diverse=dict(doc['diverse'], candidates=[])))
    out2 = tmp_path/'C2.json'
    out2.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(N.Blocked, match='diverse'):
        E.check_search(search, out2, pinned=edited['doc_sha256'])
