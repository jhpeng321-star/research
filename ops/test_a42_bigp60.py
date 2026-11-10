"""CPU-only tests for the amendment-42 driver (fake runner; no simulator, no process pool): candidate pinning / STOP,
spec construction on the mirrored bank-root bindings, the bigp60_a42 ledger cap, identity specs, verify -> admit ->
confirm -> analyse end to end on fake records (registered conjunction), BIG_BANK := U0 when nothing is admitted,
one-retry rule, budget arithmetic."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D  # noqa: E402
import nm_finaleval as N  # noqa: E402
import test_a37_celltailor as T37  # noqa: E402  (mirror discovery, FakeInner)

_FIXREL = Path('tools')/'coopmwm'/'fixtures'/'a42_r7b_STRATEGY_SEARCH_D2800_p20.json'
_ROOT = Path(__file__).resolve().parents[1]
# worktree layout: <root>/tools/...; run-host overlay layout: <root>/pytools/tools/...
FIX = next((r/_FIXREL for r in (_ROOT, _ROOT/'pytools') if (r/_FIXREL).is_file()), _ROOT/_FIXREL)


def fixture_cands():
    """Six executable r7b plans with WM feasibility > 0 (R7B's own verified picks), relabelled to D2800_p60."""
    from tools.coopmwm import a37_cellbank as C
    s = json.loads(FIX.read_text(encoding='utf-8'))
    picks = ('mut_10aa9d2905645022', 'mut_14c15871f17bf2f6', 'mut_4ee05c033859e8c2', 'mut_33ae025e556121e7',
             'mut_5a64ff4adbdfa95e', 'mut_a2b854fe94d48555')
    s = dict(s, cell=D.CELL, final=[x for x in s['final'] if x['strategy']['name'] in picks])
    return C.select_candidates(s, D.CELL, k=6)['candidates']   # row form only; rule: a42_bigbank


@pytest.fixture
def root(tmp_path, monkeypatch):
    src = T37.mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    out = tmp_path/'root'
    for f in T37.BIND:
        (out/f).parent.mkdir(parents=True, exist_ok=True)
        (out/f).write_bytes((src/f).read_bytes())
    cands = fixture_cands()
    monkeypatch.setattr(D, 'candidates', lambda *a, **k: cands)
    D.prepare(out)
    return out


def use(monkeypatch, name):
    return T37.use(monkeypatch, name)


class Fake(T37.FakeInner):
    """T37.FakeInner + A37 receipts with acts (members / releases) for the realised-group-size secondary."""

    def run(self, items, cfg, workers):
        out = super().run(items, cfg, workers)
        for (kind, spec, work), rec in zip(items, out):
            if spec['arm'] in ('A37_PLAN', 'CELL_BANK'):
                fam = spec['checkpoint'].get('a37_family') or 'mut_10aa9d2905645022'
                rec['a37'].update(activations_by_family={fam: 1}, activations=1,
                                  acts=[dict(t=400, family=fam, members=['a', 'b', 'c', 'd'], release={})])
        return out


def runtime(root, **kw):
    return D.A42Runtime(Fake(N.prerun_plan(root), **kw), root)


def test_budget_arithmetic_and_total_cap_check():
    b = D.budget_plan()
    assert (b['identity'], b['verification'], b['confirmation']) == (4_800, 201_600, 460_800)
    assert b['planned'] == 667_200 and b['retry_reserve'] == 52_800 and b['cap_sim_s'] == 720_000
    assert D.check_total_cap(35_068_000, 30_000_000)['headroom'] == 35_068_000-30_000_000-720_000
    with pytest.raises(N.Blocked, match='total_cap'):
        D.check_total_cap(35_068_000, 34_500_000)


def test_unpinned_or_stop_candidates_are_refused(tmp_path):
    from tools.coopmwm import a42_bigbank as B
    out = tmp_path/'CANDIDATES_D2800_p60.json'
    search = tmp_path/'STRATEGY_SEARCH_D2800_p60.json'
    src = json.loads(FIX.read_text(encoding='utf-8'))             # a search in which no plan ever activates -> STOP
    src = dict(src, cell=D.CELL, final=[dict(x, feasible=0.0) for x in src['final']])
    search.write_text(json.dumps(src), encoding='utf-8')
    doc = D.select(search, out)
    assert doc['STOP'] and doc['doc_sha256'] == B.doc_sha(doc)
    with pytest.raises(N.Blocked, match='exists'):
        D.select(search, out)
    with pytest.raises(N.Blocked, match='not the registered selection'):
        D.candidates(out, pinned=None)
    with pytest.raises(N.Blocked, match='STOP'):
        D.candidates(out, pinned=doc['doc_sha256'])
    edited = dict(doc, k=7)
    out.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(N.Blocked, match='doc_sha256 differs'):
        D.candidates(out, pinned=doc['doc_sha256'])


def test_panels_are_the_audited_blocks(monkeypatch):
    v = use(monkeypatch, D.VERIFY_SETTING)
    assert [r['world']['seed'] for r in v['worlds']] == list(range(992000, 992024))
    assert v['panel_sha256'].startswith('58a01184') and v['cell']['cell_key'] == 'D2800_p60'
    c = use(monkeypatch, D.CONFIRM_SETTING)
    assert [r['world']['seed'] for r in c['worlds']] == list(range(998600, 998696))
    assert c['panel_sha256'].startswith('0b42fd54') and c['contract']['bootstrap']['seed'] == 20261104
    a35 = N.read(Path(N.HERE)/'nm_eval_panel_wz3to2_a35_D2800_p60.json')
    assert not {r['world']['seed'] for r in a35['worlds']} & {r['world']['seed'] for r in c['worlds']}


def test_verify_specs_use_the_unchanged_a37_executor(root, monkeypatch):
    from tools.coopmwm import a37_cellbank as C
    panel = use(monkeypatch, D.VERIFY_SETTING)
    waves = D.verify_waves(panel, root)
    assert len(waves) == 7 and waves[0][0] == 'A42V_U0'
    ids = [s['job_id'] for _, _, specs in waves for s in specs]
    assert len(ids) == 168 == len(set(ids)) and all(i.startswith('a42v_') for i in ids)
    for name, _, specs in waves[1:]:
        ck = specs[0]['checkpoint']
        assert specs[0]['arm'] == 'A37_PLAN' and ck['kind'] == 'a37' and ck['a37_family'] == name[len('A42V_'):]
        assert ck['a37_sha256'] == C.a37_digest() and ck['a37_bank']['path'].startswith(D.HOST_ROOT)
        assert 'bank_cell_key' not in ck and ck['cell_key'] == 'D2800_p60'


def test_confirm_specs_frozen_arms_transfer_and_big_bank_needs_admission(root, monkeypatch):
    panel = use(monkeypatch, D.CONFIRM_SETTING)
    lt = D.confirm_specs('LEARNED_TRIGGER', panel, root)
    duo = D.confirm_specs('DUO_HOLD', panel, root)
    assert lt[0]['checkpoint']['bank_cell_key'] == 'D2800_p20' == duo[0]['checkpoint']['bank_cell_key']
    assert lt[0]['job_id'] == 'a42c_learned_trigger_0000' and len(lt) == 96 and len(duo) == 96
    with pytest.raises(N.Blocked, match='ADMISSION'):
        D.confirm_specs('BIG_BANK', panel, root)
    ids = [s['job_id'] for s in D.ident_specs(root)]
    assert ids == ['a42i_p20c_u0_0000', 'a42i_p20c_u0_0001', 'a42i_a35d3200_u0_0000', 'a42i_a35d3200_u0_0001']


def test_ledger_cap(root):
    rt = runtime(root)
    rt.check_capacity(720_000)
    with pytest.raises(N.Blocked, match='bigp60_a42 cap'):
        rt.check_capacity(720_001)
    rt.reserve(None, [dict(id='a42v_x_0000', sim_s=719_000)])
    with pytest.raises(N.Blocked, match='bigp60_a42 cap'):
        rt.reserve(None, [dict(id='a42c_y_0000', sim_s=1_200)])
    with pytest.raises(N.Blocked, match='a42'):
        rt.reserve(None, [dict(id='a37v_z_0000', sim_s=1)])
    assert {r['phase'] for r in N.fe.ledger_rows(root)} == {D.A42_PHASE}
    with pytest.raises(N.Blocked, match='confirmation needs'):
        D.confirm_fits(root, 4)


def _verify_and_admit(root, monkeypatch, factors, fail=()):
    panel = use(monkeypatch, D.VERIFY_SETTING)
    out = D.execute_with_retry(root, D.verify_waves(panel, root), workers=4, chunk=48, log=lambda x: None,
                               runtime=runtime(root, factors=factors, fail=fail))
    assert out['status'] == 'PASS'
    return D.admit(root, panel)


def test_end_to_end_verify_admit_confirm_analyse(root, monkeypatch):
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))
    fams = [c['family'] for c in fixture_cands()]
    adm = _verify_and_admit(root, monkeypatch, {fams[1]: .9, fams[3]: .95, fams[0]: 1.1})
    assert adm['admitted'] == [fams[1], fams[3]] and not adm['big_bank_is_u0']
    assert D.charged(root, 'a42v_') == 168*1200
    v = {x['family']: x for x in adm['verdicts']}
    assert v[fams[1]]['PASS'] and not v[fams[0]]['PASS'] and v[fams[1]]['realised_group_size']['n_held_ge3'] == 24
    with pytest.raises(N.Blocked, match='one shot'):
        D.admit(root, use(monkeypatch, D.VERIFY_SETTING))
    bank = json.loads((root/D.BANK_DIR/'A42_BIG_BANK_D2800_p60.json').read_text(encoding='utf-8'))
    assert [r['family'] for r in bank['rows']] == [fams[1], fams[3]] and {r['regime'] for r in bank['rows']} == {D.CELL}
    panel = use(monkeypatch, D.CONFIRM_SETTING)
    waves = D.stage_waves('confirm', D.CONFIRM_SETTING, panel, root)
    assert [w[0] for w in waves] == [D.confirm_wave(a) for a in D.CONFIRM_ARMS]
    assert waves[3][2][0]['arm'] == 'CELL_BANK' and waves[3][2][0]['job_id'] == 'a42c_big_bank_0000'
    D.confirm_fits(root, 4)
    out = D.execute(root, waves, workers=4, chunk=96, log=lambda x: None,
                    runtime=runtime(root, factors=dict(CELL_BANK=.97, LEARNED_TRIGGER=.99)))
    assert out['status'] == 'PASS' and D.charged(root) == 168*1200+4*96*1200
    res = D.analyse(root, panel)
    prim = res['contrasts'][0]
    assert (prim['treatment'], prim['reference']) == ('BIG_BANK', 'DUO_HOLD')
    assert res['primary']['a_pass'] and res['primary']['b_group_rule_vs_u0'] and res['primary']['verdict'] == 'SUPPORTED'
    assert [(c['treatment'], c['reference']) for c in res['contrasts']] == \
        [('BIG_BANK', 'DUO_HOLD')]+list(D.SECONDARY)+list(D.CONTEXT)
    assert set(res['groups_vs_u0']) == {'BIG_BANK', 'DUO_HOLD', 'LEARNED_TRIGGER'}
    assert res['arms']['BIG_BANK']['realised_group_size']['activations'] == 96 and 'tail_cvar90' in prim


def test_primary_conjunction_needs_both_parts():
    ok = dict(total_minus_ref=dict(ci95=[-10.0, -1.0]))
    bad = dict(total_minus_ref=dict(ci95=[-10.0, 1.0]))
    g_ok = dict(groups=dict(all_pass=True, groups={}))
    g_bad = dict(groups=dict(all_pass=False, groups={'origin:L2': dict(registered=True, increase_mean_s=12.0)}))
    assert D.primary_verdict(ok, g_ok)['supported']
    assert not D.primary_verdict(bad, g_ok)['supported']
    v = D.primary_verdict(ok, g_bad)
    assert not v['supported'] and v['b_worst'] == ('origin:L2', 12.0) and v['verdict'] == 'NOT SUPPORTED'


def test_nothing_admitted_means_big_bank_is_u0_and_no_confirmation(root, monkeypatch):
    adm = _verify_and_admit(root, monkeypatch, {})
    assert adm['admitted'] == [] and adm['big_bank_is_u0'] and adm['big_bank'] is None
    panel = use(monkeypatch, D.CONFIRM_SETTING)
    assert D.confirm_specs('BIG_BANK', panel, root) == []
    res = D.analyse(root, panel)
    assert res['primary']['verdict'] == 'NOT SUPPORTED' and 'BIG_BANK := U0' in res['primary']['note']


def test_failed_episode_is_retried_once(root, monkeypatch):
    panel = use(monkeypatch, D.VERIFY_SETTING)
    logs = []
    out = D.execute_with_retry(root, D.verify_waves(panel, root)[:1], workers=4, chunk=24, log=logs.append,
                               runtime=runtime(root, fail={'a42v_u0_0003'}))
    assert out['status'] == 'PASS' and any('replay_authorized' in x for x in logs)
    assert len(list(root.glob('REPLAY_AUTHORIZED_*.json'))) == 1


def test_identity_refs_cover_the_chosen_worlds(root):
    refs = N.read(D.C37.CANDS_DIR/'IDENTITY_REFS.json')
    assert refs['receipt_sha256'].startswith(D.IDENTITY_REFS_SHA256)
    for s in D.ident_specs(root):
        assert f'{s["job_id"].split("_")[1]}:{s["world_id"]}' in refs['worlds']
    assert deepcopy(D.IDENT_PANELS[0][0]) == 'p20c'
