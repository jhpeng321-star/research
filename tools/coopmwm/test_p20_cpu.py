"""p20 (owner REGISTRATION 2026-11-08 17:22 + low-congestion follow-up): D2400/D2800/D3200_p20 training cells for the
WM-guided (clean5c) and rule-guided (F8) runs; per-cell priority (delay / fair+delay / fairness) with prefer-both;
guidance restricted to the training cells; B = R except guidance; G4 workers configurable (6)."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json
from collections import Counter
from pathlib import Path

import pytest

from . import clean5a_restart as c5, clean5b_restart as c5b
from .test_lc_families_cpu import wz  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401

DOCS = Path(__file__).resolve().parents[2]/'clean5c'/'p20'


def _doc(name):
    return json.loads((DOCS/name).read_text(encoding='utf-8'))


def test_default_worlds_and_cells_unchanged():
    for v in (c5.VARIANT, c5b.VARIANT, c5b.VARIANT_F8, c5b.VARIANT_C):
        assert v['cell_keys'] == c5.CELL_KEYS and set(v) == set(c5.VARIANT)
    assert c5b.worlds_by_update_8() == c5b.worlds_by_update_8(c5.CELL_KEYS)
    assert c5b.VARIANT_F8['cfg_overrides']['g4_workers'] == 4            # F8T as launched


def test_p20_worlds_same_seeds_p20_cells_only():
    worlds = c5b.worlds_by_update_8(c5b.P20_CELLS)
    base = c5b.worlds_by_update_8()
    count = Counter()
    for u, rows in worlds.items():
        assert [w['seed'] for w in rows] == [w['seed'] for w in base[u]]        # same TRAIN seeds per update
        assert all(w['penetration'] == .2 and 989220 <= w['seed'] <= 989267 for w in rows)
        count.update(w['cell_key'] for w in rows)
    assert set(count) == set(c5b.P20_CELLS) and max(count.values())-min(count.values()) <= 8


def test_p20_cell_validation():
    assert c5b.P20_CELLS == ('D2400_p20', 'D2800_p20', 'D3200_p20')
    assert c5b._check_p20_cells(('D3200_p20', 'D2400_p20', 'D2800_p20')) == c5b.P20_CELLS
    for bad in (('D3200_p20', 'D2800_p20'), ('D2400_p20', 'D2800_p20', 'D3200_p60'),
                ('D2400_p20', 'D2800_p20', 'D3200_p20', 'D3200_p20')):
        with pytest.raises(ValueError, match='p20 cells'):
            c5b._check_p20_cells(bad)


def test_built_guidance_docs():
    rule, wm = _doc('GUIDANCE_P20_RULE.json'), _doc('GUIDANCE_P20_WM.json')
    assert sorted((r['regime'], r['family']) for r in rule['rows']) == [('D3200_p20', 'wall_g_duo12'),
                                                                        ('D3200_p20', 'wall_g_solo2')]
    assert all(r['fair'] and r['lb95_veh_s'] > 0 and r['arm'] == 'B' and r['p20_both'] for r in rule['rows'])
    assert [(r['regime'], r['family'], r['source']) for r in wm['rows']] == [('D3200_p20', 'mut_0584e488a8bf93c5', 'wm_bank')]
    assert wm['rows'][0]['n2_parity'] == 'identical' and wm['clean5c']['verified_cells'] == ['D3200_p20']
    assert [t['family'] for t in wm['clean5c']['templates']] == ['mut_0584e488a8bf93c5']
    for doc, is_wm, src in ((rule, False, 'rule'), (wm, True, 'wm')):
        assert c5b.check_p20_guidance(doc, c5b.P20_CELLS, wm=is_wm) == ('D2400_p20', 'D2800_p20')   # "none"
        assert {c: v['source'] for c, v in doc['p20']['cells'].items()} == dict(D2400_p20='none', D2800_p20='none',
                                                                                D3200_p20=src)
        assert {c: v['priority'] for c, v in doc['p20']['cells'].items()} == c5b.P20_PRIORITY
    with pytest.raises(ValueError, match='LOWCAV'):
        c5b.check_p20_guidance(wm, c5b.P20_CELLS, wm=False)
    with pytest.raises(ValueError, match='p20-selected'):
        c5b.check_p20_guidance(_doc('../GUIDANCE_CLEAN5C.json'), c5b.P20_CELLS, wm=True)


def _p(cell, lb, ub, worst, cvar=None):
    return dict(cell=cell, lb95_veh_s=lb, ub95_veh_s=ub, worst_group_increase_s=worst, cvar90_diff_s=cvar)


def test_p20_priority_by_congestion():
    e = c5b.p20_eligible
    assert e(_p('D3200_p20', 1., 5., 25.)) == (True, 'delay:over_fair_bound', False)       # delay priority
    assert e(_p('D3200_p20', 1., 5., 3.)) == (True, 'delay:within_fair_bound', True)
    assert e(_p('D3200_p20', -1., 5., -3.))[:2] == (False, 'lb95<=0')
    assert e(_p('D2800_p20', 1., 5., 9.)) == (True, 'fair_and_delay', True)               # fair AND cuts delay
    assert e(_p('D2800_p20', 1., 5., 11.))[0] is False and e(_p('D2800_p20', -1., 5., -9.))[0] is False
    assert e(_p('D2400_p20', -100., 50., -1., -2.)) == (True, 'fairness', False)          # fairness priority
    assert e(_p('D2400_p20', 100., 500., -1., -2.)) == (True, 'fairness', True)
    for bad, why in ((_p('D2400_p20', 1., 5., 0., -2.), 'worst_group_not_lowered'),
                     (_p('D2400_p20', 1., 5., -1.), 'cvar90_missing'),
                     (_p('D2400_p20', 1., 5., -1., .5), 'cvar90_not_lowered'),
                     (_p('D2400_p20', -9., -1., -1., -2.), 'delay_significantly_worse'),
                     (_p('D2400_p40', 1., 5., -1., -2.), 'not_a_p20_training_cell')):
        assert e(bad)[:2] == (False, why)


def _row(cell, fam, lb, worst, cvar=None):
    return dict(arm='B', regime=cell, family=fam, baseline='mappo', n=2, mean_veh_s=lb+1., lb95_veh_s=lb,
                ub95_veh_s=lb+100., worst_group_increase_s=worst, fair=worst <= 10, source='multicrit',
                **({} if cvar is None else dict(cvar90_diff_s=cvar)))


def test_p20_select_prefer_both_fairness_admission_and_labels():
    from .fullpol_learn import FAIRNESS_ADMISSION, guidance_doc, qualifying
    rows = [_row('D3200_p20', 'a', 5., 20.), _row('D3200_p20', 'b', 1., 3.),           # b: both -> a dropped
            _row('D2800_p20', 'c', 2., 12.),                                          # over bound: none
            _row('D2400_p20', 'd', -50., -2., -1.), _row('D3200_p60', 'e', 9., 0.)]   # d: fairness admission
    doc = guidance_doc(rows, sources=[dict(name='m', sha256='a'*64)], provenance='p')
    out = c5b.p20_select(doc, c5b.P20_CELLS, note='n')
    assert [(r['regime'], r['family']) for r in out['rows']] == [('D2400_p20', 'd'), ('D3200_p20', 'b')]
    assert out['p20']['rejected'] == {'D3200_p20/a': dict(reason='not_both_while_both_available', source='multicrit'),
                                      'D2800_p20/c': dict(reason='over_fair_bound', source='multicrit')}
    d = out['rows'][0]
    assert d['admission'] == FAIRNESS_ADMISSION and d['fair'] and 'admission' not in out['rows'][1]
    assert {c: v['source'] for c, v in out['p20']['cells'].items()} == dict(D2400_p20='rule', D2800_p20='none',
                                                                            D3200_p20='rule')
    assert c5b.check_p20_guidance(out, c5b.P20_CELLS, wm=True) == ('D2800_p20',)
    assert out['source']['sha256'] != doc['source']['sha256'] and len(doc['rows']) == 5      # input untouched
    # learner: the flag admits the fairness target, after every lb95 > 0 family; unflagged rows unchanged
    table = dict(d=d, x=dict(family='x', lb95_veh_s=1., mean_veh_s=1.), y=dict(family='y', lb95_veh_s=-1., mean_veh_s=9.))
    assert qualifying(table) == ['x', 'd']
    with pytest.raises(ValueError, match='no eligible rows'):
        c5b.p20_select(doc, c5b.P20_CELLS, families=('zz',), note='n')


def test_bank_eligible_fn_default_unchanged_and_cvar_hook(tmp_path):
    from . import fullpol_bank as fb
    from .fullpol_learn import guidance_doc
    groups = {g: -1. for g in fb.GROUPS}
    bank = tmp_path/'bank.jsonl'
    bank.write_text('\n'.join(json.dumps(dict(cell='D2400_p20', family='m', members=['a', 'b'], seed=s,
        diff_vs_u0=d, cell_groups_vs_u0=groups, cvar90_vs_u0=-3., executor='plant', n2_parity_verdict='identical',
        template=dict(members=['@0', '@1']), template_sha256='t', offer_condition={})) for s, d in ((1, 50.), (2, -20.))))
    fallback = guidance_doc([], sources=[dict(name='f', sha256='b'*64)], provenance='f')
    base = fb.build([bank], fallback=fallback, provenance='p')
    assert base['rows'] == [] and base['clean5c']['rejected']['D2400_p20/m']['reason'] == 'lb95<=0'
    summary, _ = fb.plans(fb.load_rows(bank))
    assert summary[('D2400_p20', 'm')]['cvar90_diff_s'] == -3.
    p20 = fb.build([bank], fallback=fallback, provenance='p', eligible_fn=c5b.p20_eligible)
    assert [(r['regime'], r['family'], r['cvar90_diff_s']) for r in p20['rows']] == [('D2400_p20', 'm', -3.)]
    out = c5b.p20_select(p20, c5b.P20_CELLS, note='n')
    assert out['p20']['cells']['D2400_p20']['source'] == 'wm' and out['clean5c']['verified_cells'] == ['D2400_p20']
    assert [t['family'] for t in out['clean5c']['templates']] == ['m']


def _capture(monkeypatch):
    seen = {}
    monkeypatch.setattr(c5b.c5, 'prepare', lambda source, *, root, variant, guidance_doc, **kw: seen.update(
        v=variant, doc=guidance_doc, kw=kw, root=root))
    return seen


def test_p20_prepares_b_equals_r_code_and_g4_workers(monkeypatch):
    seen = _capture(monkeypatch)
    rule, wm = _doc('GUIDANCE_P20_RULE.json'), _doc('GUIDANCE_P20_WM.json')
    c5b.prepare_f8_p20('src', root='/r/f8u', guidance_doc=rule, contexts=[])
    f8 = seen['v']
    assert set(f8) == set(c5.VARIANT) and f8['cell_keys'] == c5b.P20_CELLS
    assert f8['guidance_missing_ok'] == ('D2400_p20', 'D2800_p20')
    assert f8['cfg_overrides'] == dict(c5b.VARIANT_F8['cfg_overrides'], g4_workers=6) and seen['kw'] == dict(contexts=[])
    assert f8['keys'] is c5b.keys_f and f8['run'] == 'clean5b'
    assert {w['cell_key'] for rows in f8['worlds']().values() for w in rows} == set(c5b.P20_CELLS)
    c5b.prepare_c_p20('src', root='/r/c3', guidance_doc=wm, g4_workers=5)
    c = seen['v']
    assert c['cell_keys'] == c5b.P20_CELLS and c['keys'] is c5b.keys_c and c['run'] == 'clean5c'
    assert c['cfg_overrides'] == c5b._c_overrides(5) and 'learn_threads' not in c['cfg_overrides']
    assert c['worlds']() == f8['worlds']()                                  # both runs: the same worlds
    for k in ('wave_workers', 'mem_available_min_gb', 'g4_mem_available_min_gb', 'arm_update_process', 'load_max'):
        assert c['cfg_overrides'][k] == f8['cfg_overrides'][k]
    with pytest.raises(ValueError, match='LOWCAV'):
        c5b.prepare_f8_p20('src', root='/r', guidance_doc=wm)                # WM bank doc into the rule run
    with pytest.raises(ValueError, match='p20-selected'):
        c5b.prepare_c_p20('src', root='/r', guidance_doc=_doc('../GUIDANCE_CLEAN5C.json'))   # p40/p60 bank


def test_p20_keys_and_check_guidance(cfg, wz):
    """The built docs pass the real keys + clean5a check_guidance at p20 (D2400/D2800_p20 = no-target cells)."""
    from .lc_families import CONFIG_KEY
    for keys, name in ((c5b.keys_f, 'GUIDANCE_P20_RULE.json'), (c5b.keys_c, 'GUIDANCE_P20_WM.json')):
        out = dict(cfg)
        out.update(keys(desc=wz, guidance_doc=_doc(name), provenance='p20 test'))
        g = c5.check_guidance(out, ('D2400_p20', 'D2800_p20'), c5b.P20_CELLS)
        assert g['regimes'] == ['D3200_p20'] and g['no_guidance_cells'] == ['D2400_p20', 'D2800_p20']
        with pytest.raises(ValueError, match='missing'):
            c5.check_guidance(out, (), c5b.P20_CELLS)
        if keys is c5b.keys_c:
            tpl = out[CONFIG_KEY]['generator_args']['templates']
            assert tpl['B'] == tpl['R'] and [t['family'] for t in tpl['B']] == ['mut_0584e488a8bf93c5']
            assert out['clean5_ev']['parity_families'] == ['mut_0584e488a8bf93c5']


def test_chain_cells_input_routes_to_p20_prepare(tmp_path, monkeypatch):
    import importlib
    from .test_clean5b_restart_cpu import _chain_control, _chain_module
    restart = importlib.import_module('tools.coopmwm.clean5b_restart')
    mod = _chain_module()
    for variant, name in (('F8', 'prepare_f8_p20'), ('C', 'prepare_c_p20')):
        root = tmp_path/variant/'root'
        source, control = _chain_control(tmp_path/variant, variant, root=str(root), cells=list(c5b.P20_CELLS),
                                         g4_workers=6)
        seen = {}

        def fake(src, **kw):
            seen.update(kw)
            Path(kw['root']).mkdir(parents=True)
            (Path(kw['root'])/'PREPARED.json').write_text(json.dumps(dict(status='STAGED_NOT_LAUNCHED')))
        for other in ('prepare', 'prepare_f', 'prepare_f8', 'prepare_c', 'prepare_f8_p20', 'prepare_c_p20'):
            monkeypatch.setattr(restart, other, fake if other == name else (lambda *a, **k: pytest.fail('wrong')))
        monkeypatch.setattr(Path, 'symlink_to', lambda self, target, target_is_directory=False: None)
        assert mod.Chain(source, control).prepare() == dict(status='STAGED_NOT_LAUNCHED')
        assert seen['cells'] == c5b.P20_CELLS and seen['g4_workers'] == 6
    source, control = _chain_control(tmp_path/'F', 'F', root=str(tmp_path/'F'/'root'), cells=list(c5b.P20_CELLS))
    with pytest.raises(ValueError, match='variant F8 or C'):
        mod.Chain(source, control).prepare()
