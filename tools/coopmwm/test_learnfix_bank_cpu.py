"""c4/learnfix: bank growth at update boundaries, demo-BC records / seed disjointness, CLEAN5C5 cells (3/3/2)."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from . import bank_growth as bg, clean5b_restart as c5b, demo_bc
from .test_lc_families_cpu import wz  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401

DOC = Path(__file__).resolve().parents[2]/'clean5c'/'c4sit'/'GUIDANCE_C4SIT_priv.json'


def test_c5_cells_and_3_3_2_pattern():
    assert c5b._check_p20_cells(['D3200_p40', 'D2800_p20', 'D3200_p20']) == c5b.C5_CELLS
    assert c5b._check_p20_cells(['D3200_p20', 'D2800_p20']) == c5b.P20_HM_CELLS
    worlds = c5b.worlds_by_update_8(c5b.cell_pattern(c5b.C5_CELLS))
    base = c5b.worlds_by_update_8()
    seen = Counter()
    for u, rows in worlds.items():
        assert Counter(w['cell_key'] for w in rows) == dict(D3200_p20=3, D2800_p20=3, D3200_p40=2)
        assert [w['seed'] for w in rows] == [w['seed'] for w in base[u]]
        seen.update((w['seed'], w['cell_key']) for w in rows)
    assert len({s for s, _ in seen}) == 48
    assert c5b.P20_PRIORITY['D3200_p40'] == 'delay'


def _native(cfg, wz):
    doc = json.loads(DOC.read_bytes())
    out = dict(cfg)
    out.update(c5b.keys_c(desc=wz, guidance_doc=doc, provenance='bank growth test'))
    out['newmap'] = dict(cell=list(c5b.P20_CELLS))
    return out, doc


def test_apply_guidance_reproduces_keys_c(cfg, wz):
    native, doc = _native(cfg, wz)
    assert bg.apply_guidance(native, doc) == native                   # same doc -> byte-identical config


def _control(tmp_path, doc_path, demo=None):
    control = dict(guidance_path=str(doc_path), guidance_sha256=hashlib.sha256(Path(doc_path).read_bytes()).hexdigest())
    if demo is not None:
        control.update(demo_path=str(demo), demo_sha256=hashlib.sha256(Path(demo).read_bytes()).hexdigest())
    path = tmp_path/'CONTROL.json'
    path.write_text(json.dumps(control))
    return path


def test_update_view_snapshot_b_freezes_r_reuses_and_logs(cfg, wz, tmp_path):
    native, doc = _native(cfg, wz)
    root = tmp_path/'run'
    root.mkdir()
    control = _control(tmp_path, DOC)
    native.update(run_root=str(root), bank_growth=dict(rule=bg.RULE, control=str(control)))
    assert bg.update_view(dict(native, bank_growth=None), 'B', 1)[1] is None           # absent: no-op
    assert bg.update_view(native, 'A', 1) == (native, None)
    with pytest.raises(ValueError, match='fail closed'):
        bg.update_view(native, 'R', 1)                                                 # R never freezes
    out_b, rec_b = bg.update_view(native, 'B', 1)
    assert rec_b['changed_vs_config'] is False and out_b == native and rec_b['update'] == 1
    # operator repoints the control between updates: a grown doc (one more row) joins at u2 only
    grown = json.loads(DOC.read_bytes())
    grown['rows'] = grown['rows']+[dict(grown['rows'][0], regime='D3200_p20')]
    gpath = tmp_path/'G2.json'
    gpath.write_text(json.dumps(grown))
    _control(tmp_path, gpath)
    out_r1, rec_r1 = bg.update_view(native, 'R', 1)                 # u1 R: B's frozen snapshot, not the new doc
    assert rec_r1['guidance_sha256'] == rec_b['guidance_sha256'] and out_r1 == out_b
    out_b2, rec_b2 = bg.update_view(native, 'B', 2)
    out_r2, _ = bg.update_view(native, 'R', 2)
    assert rec_b2['changed_vs_config'] is True and out_b2 == out_r2
    assert out_b2['clean5_ev']['regime_guidance']['rows'] == grown['rows']
    assert json.loads((root/'bank_growth'/'U002.json').read_text())['guidance_sha256'] == rec_b2['guidance_sha256']
    gpath.write_text('{}')                                          # a changed file never sneaks in
    _control(tmp_path, DOC)
    assert not (root/'bank_growth'/'U003.json').exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        bg.update_view(dict(native, bank_growth=dict(rule=bg.RULE, control=str(tmp_path/'missing.json'))), 'B', 3)  # noqa


def _trigger():
    entry = dict(family='free_program_initiate', template=dict(source=dict(variant='mut_x')))
    return dict(context=dict(members=['a', 'b']), tables=[[None, entry], [None]], mask=[[True, True], [True, False]],
                descriptors=[[[0.]*4, [0.]*4], [[0.]*4, [0.]*4]], actions=[0, 0], logp=[0., 0.], values=[0., 0.])


def test_plan_slot_and_seed_disjointness():
    assert demo_bc.plan_slot(_trigger(), 'mut_x', ['b', 'a']) == (0, 1)
    assert demo_bc.plan_slot(_trigger(), 'mut_y') is None
    assert demo_bc.assert_seeds_disjoint([989974, 991152, 989950])
    with pytest.raises(ValueError, match='held-out'):
        demo_bc.assert_seeds_disjoint([989974, 989650])


def test_settings_and_load_registration(tmp_path):
    assert demo_bc.settings({}) is None
    with pytest.raises(ValueError):
        demo_bc.settings(dict(learnfix_demo=dict(path='x')))
    doc = dict(schema=demo_bc.SCHEMA, rule=demo_bc.RULE, items=[dict(meta=dict(seed=989601))])
    p = tmp_path/'d.json'
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='held-out'):
        demo_bc.load(p)
    with pytest.raises(ValueError, match='sha256'):
        demo_bc.load(p, '0'*64)


UNION = Path('E:/research-dense/runs/active/C4-V7FT-20261017/wmverify_r3/GUIDANCE_C4SIT_priv_UNION_r3.json')


def test_d2800_only_cells_and_training_cell_view():
    assert c5b._check_p20_cells(['D2800_p20']) == c5b.D2800_CELLS
    w = c5b.worlds_by_update_8(c5b.cell_pattern(c5b.D2800_CELLS))
    assert all(len(rows) == 8 and {r['cell_key'] for r in rows} == {'D2800_p20'} for rows in w.values())
    doc = json.loads(DOC.read_bytes())
    assert c5b.cells_view(doc, c5b.P20_CELLS) is doc                       # nothing outside: unchanged
    grown = dict(doc, rows=doc['rows']+[dict(doc['rows'][0], regime='D3200_p20')])
    v = c5b.cells_view(grown, c5b.D2800_CELLS)
    assert {r['regime'] for r in v['rows']} == {'D2800_p20'} and v['cells_view']['parent_rows_sha256'] == grown['source']['sha256']
    assert c5b.check_p20_guidance(v, c5b.D2800_CELLS, wm=True) == ()
    with pytest.raises(ValueError, match='no rows'):
        c5b.cells_view(grown, ('D3200_p40',))


@pytest.mark.skipif(not UNION.exists(), reason='UNION doc (research tree) not present')
def test_union_bank_d2800_view_passes():
    doc = json.loads(UNION.read_bytes())
    v = c5b.cells_view(doc, c5b.D2800_CELLS)
    assert sorted(r['family'] for r in v['rows']) == ['duo12_v20', 'mut_006c290e03ad3d3c', 'mut_0d50e4e18041530d',
                                                      'mut_6c2bf382711cbc37']
    assert c5b.check_p20_guidance(v, c5b.D2800_CELLS, wm=True) == ()


def test_chain_coverage_episodes_and_external_final(monkeypatch):
    import importlib
    from .test_clean5b_restart_cpu import _chain_module
    mod = _chain_module()
    c4m = importlib.import_module('tools.coopmwm.clean4_restart')
    phases = dict(v2v_train=10**9, v2v_lessons=10**9, early_probes=10**9, isolated_replicated=10**9, v2v_final=0,
                  gate_bank_verification_ext=10**9, track2_smoke=10**9)
    monkeypatch.setattr(c4m, 'checked', lambda cfg: dict(phases=phases, g4_cap_sim_s=230400))
    cfg = dict(clean4_restart=dict(diagnostic_quotes=dict(probe_u002=1, probe_u004=1, replicated_u006=1,
               replicated_u012=1, replicated_u024=1), lesson_quotes={}, gate_sim_s=1),
               practice_weighting=dict(fraction=0.), clean2_smoke_reporting_quote=dict(complete_smoke_sim_s=1),
               worlds_by_update={'1': [{}]*8})
    with pytest.raises(ValueError, match='v2v_final'):
        mod.coverage(cfg)
    r = mod.coverage(dict(cfg, final_eval_external=c5b.FINAL_EXTERNAL_RULE))
    assert r['need']['v2v_final'] == 0 and r['need']['v2v_train'] == 24*3*8*1200


SMOKE = Path(__file__).resolve().parents[2]/'clean5c'/'learnfix'


def test_smoke_ppo_kl_receipt_validator_accepts_learnfix():
    """CLEAN5C5 smoke 2026-11-09 failed 'trigger-PPO row-mean KL receipt required: B': under learnfix the recorded
    B/R smoke receipts carry the choice-row rule; the validator accepts them only with learnfix registered."""
    from . import clean4_receipts as rc4, learnfix
    on = dict(learnfix=learnfix.value())
    for arm in ('B', 'R'):
        learning = json.loads((SMOKE/f'SMOKE_C5C5_{arm}_UPDATE_001.json').read_text(encoding='utf-8'))['learning']
        assert rc4.ppo_kl_receipt(on, learning, arm)['rule'] == rc4.LEARNFIX_PPO_KL_RULE
        with pytest.raises(ValueError, match='row-mean'):
            rc4.ppo_kl_receipt({}, learning, arm)                               # inherited runs: unchanged
        bad = json.loads(json.dumps(learning))
        bad['fullpol_ppo_kl']['final_kl'] = 0.05
        with pytest.raises(ValueError, match='learnfix'):
            rc4.ppo_kl_receipt(on, bad, arm)
    assert rc4.ppo_kl_receipt({}, dict(fullpol_ppo_kl=dict(rule='row_mean')), 'B')['rule'] == 'row_mean'


def test_p20_prepare_registers_learnfix2_only_on_request(monkeypatch):
    from . import learnfix
    seen = {}
    monkeypatch.setattr(c5b.c5, 'prepare', lambda source, *, root, variant, guidance_doc, **kw: seen.update(v=variant))
    doc = json.loads(DOC.read_bytes())
    c5b.prepare_c_p20('src', root='/r', guidance_doc=doc)
    assert 'learnfix2' not in seen['v']['cfg_overrides']
    c5b.prepare_c_p20('src', root='/r', guidance_doc=doc, learnfix2=True)
    assert seen['v']['cfg_overrides']['learnfix2'] == learnfix.value2()
