"""clean5b zero-physics contracts with the EXACT clean5b keys (clean4 levers + clean5b_restart.keys) on the real
N1 wz3to2 descriptor: B = R action space (clean5a N2 slate + LCEXEC free_program), programs offered but never
BC/EV targets, G4 off for programs, the 3,900,000 clean5b budget envelope, clean5a defaults unchanged."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import re

import pytest

from . import clean4_restart as c4, clean5a_restart as c5, clean5b_restart as c5b, bank_breadth as bb
from . import fullpol_ev as fe, fullpol_head as fh, fullpol_learn as fl, lc_families as lf, sparse_synergy_credit as ssc
from .exact_grade import _seal
from .test_clean5a_restart_cpu import _doc
from .test_fullpol_clean5b_cpu import PreferPrograms
from .test_lc_families_cpu import Capture, controller, e2e_rows, make, step, wz  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)

ALLOWED = {'abreast_wall', 'group_hold', 'lane_drop_initiate', 'lane_drop_join', *lf.PROGRAM_FAMILIES}


def exact(cfg, wz):
    out = c4.levers(dict(cfg, trigger_candidates=8, trigger_partner_slots=4), lane_drop_keys={}, v2v_keys={},
                    w2_keys={})
    out.update(c5b.keys(desc=wz, guidance_doc=_doc(), provenance='MAIN clean5b 23:40'))
    return out


def test_exact_keys_registered_values(cfg, wz):
    k = exact(cfg, wz)
    ev = fe.settings(k)
    assert ev['families'] == ('abreast_wall', 'group_hold', 'free_program')
    assert ev['ev_bc_steps'] == 60 and ev['ppo_kl'] == 'row_mean' and k['imitation_epochs'] == 60 and bb.ev_only(k)
    assert ev['hold_lane_pin'] is True                                          # MAIN 2026-11-07
    lc = lf.settings(k)
    assert k['lc_families']['cap'] == c5b.LC_CAP == fh.TEMPLATE_CAP == 6     # every FULLPOL template admissible
    assert k['lc_families']['generator_args']['templates'] == []
    assert lc['propose'].__name__ == 'propose' and k['group_lane_moves'] is True
    # programs are never BC/EV targets: no program BC/negatives; EV parity families are solo holds only
    learn = fl.learn_settings(k)
    assert learn['ev_bc'] and not learn.get('bc') and not learn.get('negatives') and not learn['ev_credit']
    assert all(re.fullmatch(r'wall_[gh](_solo[0-2])?', f) for f in ev['parity_families'])
    # G4 off for programs: the G4LD extension keeps exactly the clean5a families
    assert k['g4_extension']['families'] == sorted(c5.G4_FAMILIES)
    assert not set(k['g4_extension']['families']) & {'free_program', *lf.PROGRAM_FAMILIES}
    # the only clean5a -> clean5b key differences are the program slot and the LCEXEC binding
    a = c4.levers(dict(cfg, trigger_candidates=8, trigger_partner_slots=4), lane_drop_keys={}, v2v_keys={},
                  w2_keys={})
    a.update(c5.keys(desc=wz, guidance_doc=_doc(), provenance='MAIN clean5b 23:40'))
    assert set(k)-set(a) == {'lc_families'} and {x for x in a if a[x] != k[x]} == {'clean5_ev'}
    assert {x: v for x, v in a['clean5_ev'].items() if x != 'families'} == {
        x: v for x, v in k['clean5_ev'].items() if x not in ('families', 'hold_lane_pin')}


def test_b_and_r_offer_identical_slates_with_programs(cfg, wz):
    k = exact(cfg, wz)
    tables, offered = {}, {}
    for arm in ('B', 'R'):
        p = make(k, arm)
        c = controller(p)
        seen, arm_tables = set(), []
        for t in range(300, 306):
            p.last_decision = None
            sample, _ = step(p, c, e2e_rows(t), t, Capture())
            trig = sample.get('track2_trigger')
            if trig:
                arm_tables.append(trig['tables'])
                seen |= {m['family'] for table in trig['tables'] for m in table if m}
        tables[arm], offered[arm] = arm_tables, seen
    assert tables['B'] == tables['R'], 'B and R slates must be identical (only guidance differs)'
    assert offered['B'] <= ALLOWED and 'free_program_initiate' in offered['B'], offered['B']


def test_g4_skips_program_selections_under_exact_keys(cfg, wz, monkeypatch):
    k = exact(cfg, wz)
    p = make(k, 'B')
    p.trigger.trigger_head = PreferPrograms(p.trigger)
    c = controller(p)
    steps = []
    for t in range(300, 303):
        sample, _ = step(p, c, e2e_rows(t), t, Capture())
        steps.append(dict(time_s=t, sample=dict(sample, deterministic=False)))
    picked = {tab[a]['family'] for s in steps if s['sample'].get('track2_trigger')
              for tab, a in zip(s['sample']['track2_trigger']['tables'], s['sample']['track2_trigger']['actions'])
              if 0 <= int(a) < len(tab) and tab[int(a)]}
    assert picked & set(lf.PROGRAM_FAMILIES), picked
    rows = ssc.opportunities([dict(id='ep', steps=steps)], ssc.extension(k))
    assert not [r for r in rows if r.get('move', {}).get('family', '').startswith('free_program')
                or (r.get('g4ld') or {}).get('family', '').startswith('free_program')]


def _quote(total=3_850_000):
    phases = dict(v2v_train_waves=dict(base_sim_s=1, allocated_sim_s=total-230_400), g4_synergy_credit=dict(
        base_sim_s=1, allocated_sim_s=230_400))
    return _seal(dict(phases=phases, runtime_phases=dict(v2v_train=total-230_400, g4_synergy_credit=230_400),
                      total_training_sim_s=total, prior_total_sim_s=0, prior_registered_sim_s={},
                      smoke_gate={}), 'receipt_sha256')


def test_clean5b_budget_envelope_is_3_9m_and_clean5a_default_unchanged():
    q = _quote()
    approval = dict(owner_approval='run clean5b alongside clean5a', approved_sim_s=5_050_000,
                    total_cap_sim_s=34_318_000, quote_sha256=q['receipt_sha256'])
    rec = c4.budget(quote_doc=q, approval=approval, prior_runtime_sim_s=0, approved=c5b.APPROVED, rule='x')
    assert rec["approved_sim_s"] == 5_050_000 and rec["remaining_sim_s"] == 5_050_000 and rec['rule'] == 'x'
    sealed = _seal(dict({k: v for k, v in rec.items() if k != 'receipt_sha256'}, run='clean5b'), 'receipt_sha256')
    assert c4.checked(dict(clean4_budget=sealed))["approved_sim_s"] == 5_050_000
    stale = _seal(dict({k: v for k, v in rec.items() if k != 'receipt_sha256'}, run='clean5b', approved_sim_s=3_900_000),
                  'receipt_sha256')
    with pytest.raises(ValueError):
        c4.checked(dict(clean4_budget=stale))                 # a superseded clean5b envelope is refused
    with pytest.raises(ValueError):                           # quote above the clean5b envelope
        c4.budget(quote_doc=_quote(5_100_000), approval=dict(approval, approved_sim_s=5_100_000,
                  quote_sha256=_quote(5_100_000)['receipt_sha256']), prior_runtime_sim_s=0, approved=c5b.APPROVED)
    d = c4.budget(quote_doc=q, approval=dict(approval, approved_sim_s=4_000_000), prior_runtime_sim_s=0)
    assert d['approved_sim_s'] == c4.APPROVED and d['rule'].startswith('owner 14:58')    # clean5a default
    assert c5.VARIANT['approval_file'] == 'BUDGET_CLEAN4.json' and c5.VARIANT['keys'] is None
    assert c5b.VARIANT['keys'] is c5b.keys and c5b.VARIANT['requires_lc_runtime'] is True
    assert set(c5b.VARIANT) == set(c5.VARIANT)


def test_full_class_generator_never_breaks_the_slate_under_exact_keys(cfg, wz):
    """Stress: all FULLPOL classes/structured families on a denser wz3to2 scene (8 CAVs, 3 HDVs, 3 lanes) for
    40 ticks: every proposed program is a valid template (no ValueError) and only registered families appear."""
    from .test_lc_families_cpu import row
    k = exact(cfg, wz)
    p = make(k, 'B')
    c = controller(p)
    seen = set()
    for t in range(300, 340):
        dt = t-300
        rows = [row(f'c{i}', -1300+40*i+18*dt, i % 3) for i in range(8)]+[
            row(f'h{i}', -1280+55*i+17*dt, (i+1) % 3, cav=False) for i in range(3)]
        p.last_decision = None
        sample, _ = step(p, c, rows, t, Capture())
        trig = sample.get('track2_trigger')
        if trig:
            seen |= {m['family'] for table in trig['tables'] for m in table if m}
    assert seen <= ALLOWED and 'free_program_initiate' in seen, seen


def test_wave_workers_override_is_registered():
    """clean5b binds one 16-worker wave per (sequential) arm update; the backend_config override is bounded and
    absent elsewhere (clean5a byte-identical). The live value is checked by the cpuV dry prepare."""
    from . import launch_production as production
    assert c5b.VARIANT['cfg_overrides'] == dict(load_max=120, wave_workers=16, mem_available_min_gb=30)
    assert c5.VARIANT['cfg_overrides'] == {}
    src = open(production.__file__, encoding='utf-8').read()
    assert "not 1 <= cfg['wave_workers'] <= 32" in src and "out['workers'] = cfg['wave_workers']" in src


def test_wave_admission_memory_floor(monkeypatch):
    from . import runner
    import tools.coopnative.runner as prior
    monkeypatch.setattr(prior, 'admission', lambda b: None)
    free = iter([10*1024**3, 31*1024**3])
    monkeypatch.setattr(runner, '_mem_available_bytes', lambda: next(free))
    slept = []
    monkeypatch.setattr(runner.time, 'sleep', lambda s: slept.append(s))
    runner.admission(dict(load_max=120, mem_available_min_gb=30))
    assert slept == [30]                                   # waited once below the 30 GB floor
    runner.admission(dict(load_max=40))                    # absent key: inherited guard only
    for bad in (7, 161, 30.0):
        with pytest.raises(ValueError):
            runner.admission(dict(load_max=40, mem_available_min_gb=bad))


def test_clean5b_f_keys_fair_slot_guidance_and_registered_no_guidance_cell(cfg, wz):
    """clean5b-F exact keys on the REAL MULTICRIT guidance (confirm + taskb + D3200_p40): wall_g-only fair slot,
    fairness-eligible parity, pin ON, G4 = 5a families; D2400_p40 (no rows) is the registered no-guidance cell."""
    from pathlib import Path
    m = Path('E:/research-dense/runs/active/C4-V7FT-20261017/multicrit')
    files = [m/'confirm'/'MULTICRIT_confirm.json', m/'MULTICRIT_taskb.json', m/'p40'/'MULTICRIT_confirm.json']
    if not all(f.is_file() for f in files):
        pytest.skip('MULTICRIT files are on the checkout host only')
    from . import fullpol_fair as ff
    g = ff.build_guidance([str(f) for f in files], provenance='test')
    k = c4.levers(dict(cfg, trigger_candidates=8, trigger_partner_slots=4), lane_drop_keys={}, v2v_keys={},
                  w2_keys={})
    k.update(c5b.keys_f(desc=wz, guidance_doc=g, provenance='test'))
    ev = fe.settings(k)
    assert ev['fair']['solo_variants'] == ['wall_g'] and ev['fair']['reward'] and ev['hold_lane_pin'] is True
    assert sorted(ev['parity_families']) == sorted(fe.FAIR_PARITY)
    assert k['g4_extension']['families'] == sorted(c5.G4_FAMILIES) and lf.settings(k)['propose']
    fair = {(r['regime'], r['family']) for r in g['rows'] if r['fair']}
    assert ('D3200_p40', 'wall_g_solo2') not in fair and ('D3200_p40', 'wall_g_duo12') in fair
    assert ('D3200_p40', 'wall_g_solo2_gate0') in fair
    out = c5.check_guidance(k, c5b.F_NO_GUIDANCE_CELLS)
    assert out['no_guidance_cells'] == ['D2400_p40']
    with pytest.raises(ValueError, match='every training cell'):
        c5.check_guidance(k)
    assert set(c5b.VARIANT_F) == set(c5.VARIANT) and c5b.VARIANT_F['quote_file'] == 'CLEAN5BF_QUOTE.json'


def test_clean5b_f8_worlds_and_memory_overrides(wz):
    """MAIN 06:37: 8 TRAIN worlds per update (all registered seeds/cells rotate), 6 wave workers, 100 GB floor."""
    from collections import Counter
    worlds = c5b.worlds_by_update_8()
    assert sorted(worlds, key=int) == [str(u) for u in range(1, 25)]
    cells, seeds = Counter(), set()
    for rows in worlds.values():
        assert len(rows) == 8 and len({w['seed'] for w in rows}) == 8
        assert all(989220 <= w['seed'] <= 989267 for w in rows)
        cells.update(w['cell_key'] for w in rows)
        seeds |= {w['seed'] for w in rows}
    assert set(cells) == set(c5.CELL_KEYS) and len(seeds) == 48
    o = c5b.VARIANT_F8['cfg_overrides']
    assert {k: v for k, v in o.items() if k != 'arm_update_process'} == dict(
        load_max=120, wave_workers=8, mem_available_min_gb=40, g4_workers=4, g4_mem_available_min_gb=20)
    assert 41+46+o['wave_workers']*10.6 < 184 and 184-86-46 >= o['mem_available_min_gb']   # MAIN 10:4x arithmetic
    # F8S (owner 11-08): G4 with both parents resident (B 86 + R 46 + ~9 A/system): idle floor reachable, 4 one-job
    # workers at the measured G4 peak 2.6 GB stay <= 165 GB, and the G4 floor is below what is reachable.
    assert 86+46+9+o['g4_workers']*2.6 <= 165 and 184-(86+46+9) >= o['g4_mem_available_min_gb']
    assert c5b.VARIANT_F8['quote_file'] == 'CLEAN5BF8_QUOTE.json' and set(c5b.VARIANT_F8) == set(c5.VARIANT)


def test_fresh_update_process_rule_and_fail_closed(tmp_path, monkeypatch):
    """F8 memory fix: each arm update runs in a fresh child (memory returned on exit); absent key = in-process."""
    import subprocess
    from types import SimpleNamespace
    from . import launch_training as lt
    from .exact_grade import _seal
    from .protocol import digest
    assert c5b.VARIANT_F8['cfg_overrides']['arm_update_process'] == lt.FRESH_UPDATE_RULE
    assert lt.fresh_update_process({}) is False
    with pytest.raises(ValueError):
        lt.fresh_update_process(dict(arm_update_process='yes'))
    cfg = dict(run_root=str(tmp_path), arm_update_process=lt.FRESH_UPDATE_RULE)
    calls = []

    def fake_popen(command, **kw):
        calls.append(command)
        u = int(command[command.index('--update')+1])
        (tmp_path/f'B_UPDATE_{u:03}.json').write_text(json.dumps(_seal(dict(config_sha256=digest(cfg)), 'update_sha256')))
        return SimpleNamespace(pid=4242+u, wait=lambda: 0)
    import json
    monkeypatch.setattr(subprocess, 'Popen', fake_popen)
    for u in (1, 2):
        assert lt.run_update_process(cfg, 'B', u)['config_sha256'] == digest(cfg)
    assert [c[c.index('--update')+1] for c in calls] == ['1', '2'] and 'update' in calls[0]
    assert calls[0][calls[0].index('--config')+1] == str(tmp_path/'CONFIG.json')        # run_root/CONFIG.json
    assert json.loads((tmp_path/'owners'/'children'/'B_u002.json').read_text())['pid'] == 4244
    import inspect
    src = inspect.getsource(lt.run_arm)
    assert 'if fresh_update_process(cfg):' in src and 'run_update_process(cfg, arm, u)' in src
    monkeypatch.setattr(subprocess, 'Popen', lambda command, **kw: SimpleNamespace(pid=9, wait=lambda: 3))
    with pytest.raises(RuntimeError, match='no automatic retry'):
        lt.run_update_process(cfg, 'R', 1)


def test_clean5c_keys_bank_guidance_reaches_ev_table(cfg, wz):
    """L5C acceptance (zero SUMO): F8S CONFIG keys + GUIDANCE_CLEAN5C -> B has EV-BC targets in all 5 guided training
    cells (D3200_p60 / D2800_p40 from the WM bank, the other 3 from the labelled F8S fallback); R none; B gets the
    bank templates, R none; parity verdicts recorded."""
    import json
    from pathlib import Path
    from . import fullpol_learn as fl, fullpol_templates as ft
    path = Path(__file__).resolve().parents[2]/'clean5c'/'GUIDANCE_CLEAN5C.json'
    g = json.loads(path.read_text(encoding='utf-8'))
    k = c4.levers(dict(cfg, trigger_candidates=8, trigger_partner_slots=4), lane_drop_keys={}, v2v_keys={},
                  w2_keys={})
    k.update(c5b.keys_c(desc=wz, guidance_doc=g, provenance='test'))
    table, receipt = fl.ev_table(k, 'B')
    assert {c: sorted(fl.qualifying(table[c])) for c in ('D3200_p60', 'D2800_p40')} == dict(
        D3200_p60=['mut_0584e488a8bf93c5'], D2800_p40=['mut_006c290e03ad3d3c'])
    for c in ('D2400_p60', 'D2800_p60', 'D3200_p40'):
        assert table[c] and all(r['source'] == 'f8s_rule' and r['fallback'] for r in table[c].values())
    assert set(c5.CELL_KEYS)-set(table) == {'D2400_p40'} and not receipt['dropped_without_parity']
    r_table, _ = fl.ev_table(k, 'R')
    assert not r_table                                                                   # R: target-free
    t = k['lc_families']['generator_args']['templates']
    assert t['R'] == t['B']                                                              # same action set
    assert [(x['family'], x['cells']) for x in t['B']] == [('mut_006c290e03ad3d3c', ['D2800_p40']),
                                                           ('mut_0584e488a8bf93c5', ['D3200_p20', 'D3200_p60'])]
    assert k['clean5c']['templates'] == [dict(family=x['family'], cells=x['cells']) for x in t['B']]
    assert all('cell' not in x['template']['source'] for x in t['B'])
    assert k['clean5c']['bank_targets'] == {'D2800_p40/mut_006c290e03ad3d3c': 'identical',
                                            'D3200_p20/mut_0584e488a8bf93c5': 'identical',
                                            'D3200_p60/mut_0584e488a8bf93c5': 'identical'}
    assert c5.check_guidance(k, c5b.C_NO_GUIDANCE_CELLS)['no_guidance_cells'] == ['D2400_p40']
    assert set(c5b.VARIANT_C) == set(c5.VARIANT) and c5b.VARIANT_C["approved"] == 4_700_000
    # the generator offers bank templates to B only (per-arm generator_args)
    calls = []
    slot = dict(classes=['verified'], templates=dict(B=[dict(template={'source': dict(variant='m')},
                                                            offer_condition={'x': 1})], R=[]))
    import numpy as np
    from . import fullpol_bank as fb
    orig = (fb.offer_ok, fb.session_ok, ft._lc, ft._view)
    try:
        fb.offer_ok = fb.session_ok = lambda *a: calls.append(1) or True
        ft._lc = lambda: type('LC', (), {'FreeProgram': type('FP', (), {'from_dict': staticmethod(lambda d: d)})})
        ft._view = lambda f: None
        st = lambda arm: dict(frame=None, rows=[], fullpol=slot, now=400, ledger={}, arm=arm)  # noqa: E731
        own = dict(vid='e', s_m=-1000., lane=0, is_cav=True)
        assert len(ft.propose(st('B'), own, np.random.default_rng(0))) == 1
        assert ft.propose(dict(st('R'), fullpol=dict(slot, templates=dict(B=[], R=[]))), own,
                          np.random.default_rng(0)) == []
        assert ft.propose(st(None), own, np.random.default_rng(0)) == []
    finally:
        fb.offer_ok, fb.session_ok, ft._lc, ft._view = orig


def test_clean5c_unique_templates_dedup():
    """Same family + template + offer condition verified in several cells -> offered once (cells listed);
    a differing offer condition stays a separate entry."""
    cond = dict(trigger='wall', zone_s=[-1700., -300.])
    tpl = lambda cell: dict(members=['@0', '@1'], source=dict(variant='m', family='m', cell=cell))  # noqa: E731
    doc = dict(clean5c=dict(templates=[
        dict(cell='D3200_p20', family='m', template=tpl('D3200_p20'), offer_condition=cond),
        dict(cell='D3200_p60', family='m', template=tpl('D3200_p60'), offer_condition=cond),
        dict(cell='D2800_p40', family='m', template=tpl('D2800_p40'), offer_condition=dict(cond, zone_s=[-1, 0]))]))
    out = c5b.unique_templates(doc)
    assert [(x['family'], x['cells']) for x in out] == [('m', ['D3200_p20', 'D3200_p60']), ('m', ['D2800_p40'])]
    assert all('cell' not in x['template']['source'] for x in out)
    assert doc['clean5c']['templates'][0]['template']['source']['cell'] == 'D3200_p20'     # input untouched


def _chain_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/'ops'/'clean5b_chain.py'
    spec = importlib.util.spec_from_file_location('clean5b_chain_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chain_control(tmp_path, variant, **extra):
    import json
    control, source = tmp_path/'control', tmp_path/'source'
    control.mkdir(parents=True)
    source.mkdir(parents=True)
    (source/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(commit='0'*40, files={})))
    inputs = dict(contexts_path=str(tmp_path/'ctx.json'), guidance_path=str(tmp_path/'g.json'), prior_runtime_sim_s=0,
                  host='cpuZ', provenance='test', **extra)
    if variant is not None:
        inputs['variant'] = variant
    (control/'PREPARE_INPUTS.json').write_text(json.dumps(inputs))
    for name in ('ctx.json', 'g.json'):
        (tmp_path/name).write_text(json.dumps(dict(name=name)))
    for name in ('CLEAN5B_QUOTE.json', 'CLEAN5BF_QUOTE.json', 'CLEAN5BF8_QUOTE.json', 'CLEAN5C_QUOTE.json',
                 'BUDGET_CLEAN5B.json', 'BUDGET_CLEAN5C.json'):
        (control/name).write_text(json.dumps(dict(name=name, receipt_sha256='f'*64)))
    return source, control


def test_chain_variant_files_existing_variants_unchanged(tmp_path):
    mod = _chain_module()
    want = {None: ('CLEAN5B_QUOTE.json', 'BUDGET_CLEAN5B.json', c5b.ROOT),
            'F': ('CLEAN5BF_QUOTE.json', 'BUDGET_CLEAN5B.json', c5b.ROOT),
            'F8': ('CLEAN5BF8_QUOTE.json', 'BUDGET_CLEAN5B.json', c5b.ROOT),
            'C': ('CLEAN5C_QUOTE.json', 'BUDGET_CLEAN5C.json', c5b.C_ROOT)}
    for k, (variant, (quote, approval, root)) in enumerate(want.items()):
        source, control = _chain_control(tmp_path/str(k), variant)
        chain = mod.Chain(source, control)
        assert (chain.quote_file, chain.approval_file, chain.root.as_posix()) == (quote, approval, root)
    source, control = _chain_control(tmp_path/'bad', 'X')
    with pytest.raises(ValueError, match='unknown clean5b variant'):
        mod.Chain(source, control)


def test_chain_variant_c_zero_sumo_dry_run(tmp_path, monkeypatch):
    """Variant C dry run (zero SUMO): BUDGET binds CLEAN5C_QUOTE + BUDGET_CLEAN5C; PREPARE dispatches prepare_c with
    the PREPARE_INPUTS guidance/contexts and the CLEAN5C files; nothing else is called."""
    import json
    from pathlib import Path
    import importlib
    clean4_budget = importlib.import_module('tools.coopmwm.clean4_budget')     # the modules the chain imports
    restart = importlib.import_module('tools.coopmwm.clean5b_restart')
    mod = _chain_module()
    root = tmp_path/'root'
    source, control = _chain_control(tmp_path, 'C', root=str(root))
    seen = {}
    monkeypatch.setattr(clean4_budget, 'approved', lambda approval, quote: seen.update(budget=(approval['name'],
                        quote['name'])) or dict(ok=True))

    def fake_prepare(src, **kw):
        seen['prepare'] = dict(source=str(src), **{k: kw[k] for k in ('contexts', 'guidance_doc', 'root')},
                               quote=kw['quote_doc']['name'], approval=kw['approval']['name'])
        Path(kw['root']).mkdir()
        (Path(kw['root'])/'PREPARED.json').write_text(json.dumps(dict(status='STAGED_NOT_LAUNCHED')))
    for name in ('prepare', 'prepare_f', 'prepare_f8'):
        monkeypatch.setattr(restart, name, lambda *a, **k: pytest.fail('wrong prepare variant'))
    monkeypatch.setattr(restart, 'prepare_c', fake_prepare)
    monkeypatch.setattr(Path, 'symlink_to', lambda self, target, target_is_directory=False: seen.update(link=str(target)))
    chain = mod.Chain(source, control)
    assert chain.budget() == dict(ok=True) and seen['budget'] == ('BUDGET_CLEAN5C.json', 'CLEAN5C_QUOTE.json')
    assert chain.prepare() == dict(status='STAGED_NOT_LAUNCHED')
    assert seen['prepare'] == dict(source=str(source), contexts=dict(name='ctx.json'), guidance_doc=dict(name='g.json'),
                                   root=root, quote='CLEAN5C_QUOTE.json', approval='BUDGET_CLEAN5C.json')
    assert seen['link'] == str(control) and 'f'*64 in (root/'AUTHORITY.md').read_text(encoding='utf-8')


def test_validate_levers_routes_clean5c_to_validate_clean5a(monkeypatch):
    """REV-5BF: a clean5c smoke must run the clean5a/5b/5c receipt checks (incl. the clean5c branch)."""
    import importlib
    rc4 = importlib.import_module('tools.coopmwm.clean4_receipts')
    seen = []
    monkeypatch.setattr(rc4, 'validate_clean5a', lambda cfg, updates, fixes=None: seen.append(cfg['newmap']['run']) or 'ok')
    for run in ('clean5a', 'clean5b', 'clean5c'):
        assert rc4.validate_levers(dict(newmap=dict(run=run)), []) == 'ok'
    assert seen == ['clean5a', 'clean5b', 'clean5c']


def test_host_receipt_cpuw_constant_never_reads(monkeypatch):
    """cpuW binds N6 HOST_IDENTITY_cpuW_r2 by its receipt sha constant; no file (sealed root) is opened."""
    import builtins
    from pathlib import Path
    from . import launch_training
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: opened.append(a[0]) or real_open(*a, **k))
    for name in ('read_bytes', 'read_text', 'exists', 'is_file', 'open'):
        monkeypatch.setattr(Path, name, lambda self, *a, **k: opened.append(str(self)) or pytest.fail('read '+str(self)))
    monkeypatch.setattr(launch_training, 'read', lambda p: pytest.fail('read '+str(p)))
    out = c5.host_receipt('cpuW')
    assert out == dict(host='cpuW', receipt='/home/ubuntu/nm_finaleval_clean5a/HOST_IDENTITY_cpuW_r2.json',
                       sha256='862be772f10af898af1e95ba5ae34a782ba25f78175443ad69dfa7943a74091a', verdict='IDENTICAL',
                       bytes='4699/4699 HASHES_OK', rule='N6 HOST_IDENTITY_cpuW_r2 receipt_sha256 constant; file not read')
    assert not [p for p in opened if 'nm_finaleval' in str(p)]
    with pytest.raises(KeyError):
        c5.host_receipt('cpuX')
