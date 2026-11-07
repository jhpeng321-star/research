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
    approval = dict(owner_approval='run clean5b alongside clean5a', approved_sim_s=4_360_000,
                    total_cap_sim_s=34_318_000, quote_sha256=q['receipt_sha256'])
    rec = c4.budget(quote_doc=q, approval=approval, prior_runtime_sim_s=0, approved=c5b.APPROVED, rule='x')
    assert rec["approved_sim_s"] == 4_360_000 and rec["remaining_sim_s"] == 4_360_000 and rec['rule'] == 'x'
    sealed = _seal(dict({k: v for k, v in rec.items() if k != 'receipt_sha256'}, run='clean5b'), 'receipt_sha256')
    assert c4.checked(dict(clean4_budget=sealed))["approved_sim_s"] == 4_360_000
    stale = _seal(dict({k: v for k, v in rec.items() if k != 'receipt_sha256'}, run='clean5b', approved_sim_s=3_900_000),
                  'receipt_sha256')
    with pytest.raises(ValueError):
        c4.checked(dict(clean4_budget=stale))                 # a superseded clean5b envelope is refused
    with pytest.raises(ValueError):                           # quote above the clean5b envelope
        c4.budget(quote_doc=_quote(4_400_000), approval=dict(approval, approved_sim_s=4_400_000,
                  quote_sha256=_quote(4_400_000)['receipt_sha256']), prior_runtime_sim_s=0, approved=c5b.APPROVED)
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
    assert not r_table
    t = k['lc_families']['generator_args']['templates']
    assert t['R'] == [] and [x['family'] for x in t['B']] == [x['family'] for x in g['clean5c']['templates']]
    assert k['clean5c']['bank_targets'] == {'D2800_p40/mut_006c290e03ad3d3c': 'identical',
                                            'D3200_p20/mut_0584e488a8bf93c5': 'identical',
                                            'D3200_p60/mut_0584e488a8bf93c5': 'identical'}
    assert c5.check_guidance(k, c5b.C_NO_GUIDANCE_CELLS)['no_guidance_cells'] == ['D2400_p40']
    assert set(c5b.VARIANT_C) == set(c5.VARIANT) and c5b.VARIANT_C['approved'] == 3_600_000
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
        assert ft.propose(st('R'), own, np.random.default_rng(0)) == []
        assert ft.propose(st(None), own, np.random.default_rng(0)) == []
    finally:
        fb.offer_ok, fb.session_ok, ft._lc, ft._view = orig
