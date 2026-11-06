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
        x: v for x, v in k['clean5_ev'].items() if x != 'families'}


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
    approval = dict(owner_approval='run clean5b alongside clean5a', approved_sim_s=3_900_000,
                    total_cap_sim_s=34_318_000, quote_sha256=q['receipt_sha256'])
    rec = c4.budget(quote_doc=q, approval=approval, prior_runtime_sim_s=0, approved=c5b.APPROVED, rule='x')
    assert rec['approved_sim_s'] == 3_900_000 and rec['remaining_sim_s'] == 3_900_000 and rec['rule'] == 'x'
    sealed = _seal(dict({k: v for k, v in rec.items() if k != 'receipt_sha256'}, run='clean5b'), 'receipt_sha256')
    assert c4.checked(dict(clean4_budget=sealed))['approved_sim_s'] == 3_900_000
    as5a = _seal(dict({k: v for k, v in rec.items() if k != 'receipt_sha256'}, run='clean5a'), 'receipt_sha256')
    with pytest.raises(ValueError):
        c4.checked(dict(clean4_budget=as5a))                  # 3.9M record cannot pass as a 4M clean5a record
    with pytest.raises(ValueError):                           # quote above the clean5b envelope
        c4.budget(quote_doc=_quote(3_950_000), approval=dict(approval, approved_sim_s=3_950_000,
                  quote_sha256=_quote(3_950_000)['receipt_sha256']), prior_runtime_sim_s=0, approved=c5b.APPROVED)
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
