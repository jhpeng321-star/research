"""CPU contracts for the clean3 restart budget, G1 go/no-go rule and chain checks."""
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

from tools.coopmwm import clean3_restart as c3
from tools.coopmwm.exact_grade import verify_sealed

ACCEPTED = {k: 'a'*64 for k in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'G1', 'G2', 'G4', 'G5')}
PREFIXES = dict(clean1='b'*64, clean2='c'*64)


def rows(name, total, size=1200):
    out, i = [], 0
    while total > 0:
        out.append(dict(id=f'{name}{i}', phase='x', sim_s=min(size, total))); total -= size; i += 1
    return out


def budget(g1=38400, mining=0):
    return c3.build(clean1_rows=rows('a', 1248480, 480), clean2_rows=rows('b', 1849200),
        g1_rows=rows('g', g1), mining_rows=rows('m', mining), prefixes=PREFIXES, accepted=ACCEPTED)


def test_quote_matches_registered_phases_and_fits_owner_cap():
    assert sum(c3.PHASES.values())+c3.G1_CAP+c3.G4_CAP == c3.QUOTE_TOTAL == 3_680_400
    value = budget()
    verify_sealed(value, 'receipt_sha256')
    assert value['protected_prior_sim_s'] == 1248480+1849200+960+38400+500000
    assert value['remaining_sim_s'] == 7_300_000-value['protected_prior_sim_s']
    assert sum(value['phases'].values())+c3.G4_CAP <= value['remaining_sim_s']
    assert value['plan_fix_reserved_sim_s'] == 0


def test_budget_refuses_overspend_and_missing_acceptance():
    with pytest.raises(ValueError, match='G1 certification exceeded'):
        budget(g1=c3.G1_CAP+1200)
    with pytest.raises(ValueError, match='mining exceeded'):
        budget(mining=501_600)
    with pytest.raises(ValueError, match='acceptance'):
        c3.build(clean1_rows=[], clean2_rows=[], g1_rows=[], mining_rows=[], prefixes=PREFIXES,
                 accepted={k: v for k, v in ACCEPTED.items() if k != 'G5'})
    with pytest.raises(ValueError, match='exceeds remaining owner cap'):
        c3.build(clean1_rows=rows('a', 1248480+30000), clean2_rows=rows('b', 1849200),
                 g1_rows=[], mining_rows=[], prefixes=PREFIXES, accepted=ACCEPTED)


def test_checked_recomputes_and_detects_tampering():
    cfg = dict(clean3_budget=budget())
    assert c3.checked(cfg)['phases'] == c3.PHASES
    total, prior = c3.native_budget(cfg)
    assert total['total'] == c3.CAP and prior == cfg['clean3_budget']['protected_prior_sim_s']
    cfg['clean3_budget']['phases'] = dict(c3.PHASES, v2v_train=1)
    with pytest.raises(ValueError):
        c3.checked(cfg)


def manifest(admitted):
    entries = []
    for arm, family, ok in admitted:
        entries.append(dict(job=dict(lesson=dict(arm=arm, macro=dict(family=family))),
                            admission=dict(admitted=ok)))
    return dict(entries=entries, admitted=sum(ok for *_, ok in admitted),
                diagnostics=sum(not ok for *_, ok in admitted), manifest_sha256='d'*64)


@pytest.mark.parametrize('rows_,decision', [
    ([('B', 'wall', True), ('R', 'wall', True)], 'GO'),
    ([('B', 'wall', True), ('R', 'wall', False)], 'NO-GO'),
    ([('B', 'wall', True), ('R', 'gap_open', True)], 'NO-GO'),
    ([], 'NO-GO')])
def test_g1_go_nogo_rule(tmp_path, rows_, decision):
    value = c3.go_nogo(manifest(rows_), work=tmp_path, runtime_sha256='e'*64, charged=0, branches=0)
    assert value['decision'] == decision
    assert (tmp_path/'CLEAN3_GO_NOGO.json').is_file()
    verify_sealed(value, 'receipt_sha256')


def test_reservation_limit_allows_clean3_cap_only_with_registration(tmp_path):
    from tools.coopmwm import track2_backend as p1
    cfg = dict(budget=dict(total=c3.CAP, phases=dict(c3.PHASES)), prior_charged_sim_s=3_686_000,
               run_root=str(tmp_path))
    with pytest.raises(ValueError, match='owner-cap'):
        p1.reservation_limits(cfg)
    cfg['clean3_budget'] = {}
    root, cap, phases = p1.reservation_limits(cfg)
    assert cap == c3.CAP-3_686_000 and phases == c3.PHASES


def chain_path():
    # Repository layout: <repo>/ops; deployed source layout: <src>/pytools/tools/... and <src>/ops.
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[3]):
        if (parent/'ops'/'clean3_chain.py').is_file():
            return parent/'ops'/'clean3_chain.py'
    raise FileNotFoundError('ops/clean3_chain.py')


def chain_module():
    path = chain_path()
    spec = importlib.util.spec_from_file_location('clean3_chain', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def coverage_inputs():
    clean2 = [dict(id=f't2c_warm_u{u:03}_A_{i}', phase='v2v_train', sim_s=1200)
              for u in range(1, 19) for i in range(60)]
    diag = dict(probe_u002=22740, probe_u004=22740, replicated_u006=47920,
                replicated_u012=47920, replicated_u024=47920)
    lessons = {f'lesson_u{u:03}': 8760 for u in (6, 12, 18, 24)}
    return clean2, diag, lessons


def test_chain_coverage_asserts_full_u1_u24_schedule():
    chain = chain_module()
    cfg = dict(clean3_budget=budget())
    clean2, diag, lessons = coverage_inputs()
    result = chain.coverage(cfg, clean2, gate_sim_s=15160, diagnostic_quotes=diag, lesson_quotes=lessons)
    assert result['need']['v2v_train'] == 24*72000 and result['need']['v2v_final'] == 4*96*1200
    with pytest.raises(ValueError, match='u1-u24'):
        chain.coverage(cfg, clean2, gate_sim_s=15160, diagnostic_quotes=diag,
                       lesson_quotes={k: 20000 for k in lessons})
    heavy = clean2+[dict(id=f't2c_warm_u001_B_x{i}', phase='v2v_train', sim_s=1200) for i in range(10)]
    with pytest.raises(ValueError, match='u1-u24'):
        chain.coverage(cfg, heavy, gate_sim_s=15160, diagnostic_quotes=diag, lesson_quotes=lessons)
    with pytest.raises(ValueError, match='measured'):
        chain.coverage(cfg, clean2[:60], gate_sim_s=15160, diagnostic_quotes=diag, lesson_quotes=lessons)


def test_chain_sequence_fails_closed_and_orders_steps():
    chain = chain_module()
    seen = []
    def status(step, state, detail=None):
        seen.append((step, state))
    def boom():
        raise ValueError('x')
    with pytest.raises(ValueError):
        chain.sequence([('A', lambda: 1), ('B', boom), ('C', lambda: 3)], status)
    assert seen == [('A', 'RUNNING'), ('A', 'PASS'), ('B', 'RUNNING'), ('B', 'FAILED')]


def test_chain_step_order_is_registered():
    text = chain_path().read_text()
    order = ['QUALIFIED', 'G1_CERTIFY', 'MINED_WAIT', 'PREPARE', 'SMOKE', 'GATE', 'PHASE_COVERAGE', 'LAUNCH']
    positions = [text.index(f"('{step}'") for step in order]
    assert positions == sorted(positions)


def test_cutoff_is_local_wall_clock():
    chain = chain_module()
    now = datetime(2026, 11, 6, 4, 0).astimezone()
    assert chain.deadline_unix('06:30', now) - now.timestamp() == 2.5*3600


def test_lever_receipts_use_native_registration_and_hard_checks(tmp_path, monkeypatch):
    """Raw CONFIG lacks trigger_candidates; G2 registration is checked on the native config."""
    from copy import deepcopy
    from tools.coopmwm import clean3_receipts as receipts, composite_health, launch_production as production
    from tools.coopmwm import coordination_opportunities as co, launch_training as training
    from tools.coopmwm.exact_grade import _seal
    go = _seal(dict(decision='NO-GO', runtime_sha256='e'*64, executable_joint_labels_ci95_low_positive=0), 'receipt_sha256')
    (tmp_path/'g1').mkdir()
    (tmp_path/'g1'/'CLEAN3_GO_NOGO.json').write_text(__import__('json').dumps(go))
    raw = c3.levers(dict(trigger_head=True))
    raw.update(run_root=str(tmp_path), clean3_restart=dict(g1_root=str(tmp_path/'g1'),
               g1_go_nogo_sha256=go['receipt_sha256']))
    assert 'trigger_candidates' not in raw
    def native(cfg, *, root=None, smoke=False):
        out = deepcopy(cfg); out.update(trigger_candidates=8)
        out['budget'] = dict(phases=dict(g4_synergy_credit=230400))
        if smoke:
            out['g4_synergy_credit'] = False
        return out
    monkeypatch.setattr(production, 'backend_config', native)
    monkeypatch.setattr(composite_health, 'report_only', lambda root, arm: True)
    monkeypatch.setattr(training, 'write', lambda path, value: None)
    proposals = dict(coordination_opportunities=dict(rule=co.RULE, decisions=10, offered_joint=4,
        offered_joint_rate=.4, graph_edges=3, disjoint_pairs=1, event_decisions=1),
        proposed_by_family=dict(yield_lane=0, open_gap_merge=2, wall=3))
    updates = [dict(arm='A', metrics={}, update_sha256='a'),
               dict(arm='B', metrics=dict(move_proposals=proposals), update_sha256='b'),
               dict(arm='R', metrics=dict(move_proposals=deepcopy(proposals)), update_sha256='r')]
    pre = receipts.validate_levers(raw, updates)
    assert pre['status'] == 'LEVER_PRECHECK_ONLY' and pre['G1']['decision'] == 'NO-GO'
    done = receipts.validate_levers(raw, updates, dict(receipt_sha256='f'*64))
    assert done['status'] == 'LEVER_RECEIPTS_PASS'
    updates[2]['metrics']['move_proposals']['proposed_by_family'].pop('yield_lane')
    with pytest.raises(ValueError, match='G5'):
        receipts.validate_levers(raw, updates)
    updates[2]['metrics']['move_proposals'] = dict(proposals, coordination_opportunities=None)
    with pytest.raises(ValueError, match='G2'):
        receipts.validate_levers(raw, updates)
    bound = dict(raw, executable_label_manifest=str(tmp_path/'m.json'))
    with pytest.raises(ValueError, match='NO-GO'):
        receipts.validate_levers(bound, updates[:2]+[dict(updates[1], arm='R')])
