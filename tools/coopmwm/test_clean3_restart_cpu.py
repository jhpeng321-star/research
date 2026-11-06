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
    order = ['QUALIFIED', 'G1_CERTIFY', 'MINED_WAIT', 'CONFIRM_WAIT', 'PREPARE', 'SMOKE', 'GATE', 'PHASE_COVERAGE', 'LAUNCH']
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
    raw.update(run_root=str(tmp_path), mined_moves=str(tmp_path/'mined.jsonl'),
               clean3_restart=dict(g1_root=str(tmp_path/'g1'), g1_go_nogo_sha256=go['receipt_sha256']))
    from tools.coopmwm import clean3_mined
    monkeypatch.setattr(clean3_mined, 'smoke_check', lambda cfg, updates, artifacts: dict(B={}, R={}))
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


def mined_fixture(tmp_path, monkeypatch, rows, annotated=None, base=None):
    base = base or {}
    import json as _json
    from types import SimpleNamespace
    from tools.coopmwm import bank_breadth, group_lane_bank, group_lane
    from tools.coopmwm.protocol import digest
    plans = {}
    for r in rows:
        plan = dict(id=r['verification_id'], family=r.pop('_family'), members=r.pop('_members'))
        r['plan_sha256'] = digest(plan); plans[(r['arm'], r['verification_id'])] = plan
    path = tmp_path/'MINED_MOVES.jsonl'
    path.write_text(''.join(_json.dumps(r)+chr(10) for r in rows))
    if annotated is not None:
        (tmp_path/'MINED_MOVES_ANNOTATED.jsonl').write_text(''.join(
            _json.dumps(dict(arm=r['arm'], verification_id=r['verification_id'], plan_sha256=r['plan_sha256'],
                             _synergy_lower95=annotated(r)))+chr(10) for r in rows))
    monkeypatch.setattr(group_lane_bank, 'inputs', lambda cfg: [])
    def visit(cfg, arm, consumer):
        row = _json.loads(Path(cfg['mined_moves']).read_text())
        plan = plans.get((arm, row['verification_id']))
        if plan is not None:
            consumer(cfg=cfg, receipt=dict(verification_id=row['verification_id']), example=None,
                     plan=plan, pairs=[1], trace=None)
    monkeypatch.setattr(bank_breadth, 'visit', visit)
    monkeypatch.setattr(bank_breadth, 'mined_move', lambda plan: SimpleNamespace(
        family=plan['family'], members=tuple(range(plan['members'])), actors=tuple(range(plan['members']))))
    monkeypatch.setattr(bank_breadth, 'practice_pool', lambda cfg, arm: [dict(family=f) for f in base.get(arm, ())])
    return path


def mrow(arm, vid, family, members, ci=None, veh=True):
    r = dict(schema='coopmwm-mined-move-v1', arm=arm, run_root='/abs/root', verification_id=vid,
             _family=family, _members=members)
    if ci is not None:
        r['synergy_lower95'] = ci
        if veh:
            r['synergy_veh_s'] = {k: 10. for k in ci}
    return r


def test_walls_only_rule_counts_per_seed_verified_joint_examples(tmp_path, monkeypatch):
    # One 4-CAV wall with real synergy on seeds 1 and 3, plus one more seed -> 3 per-seed examples.
    rows = [mrow('B', 'w1', 'wall', 4, {'1': .5, '2': -.1, '3': .2}), mrow('B', 'w2', 'wall', 3, {'1': .4})]
    path = mined_fixture(tmp_path, monkeypatch, rows, base=dict(R=['wall', 'gap_open'], B=['gap_open']))
    out = c3.admit_mined(path, dict(run_root=str(tmp_path)), tmp_path/'adm.jsonl')
    assert out['ready'] and out['joint_verified_examples'] == 3 and out['families'] == ['wall']


def test_walls_only_rule_refuses_insufficient_or_unsupported_examples(tmp_path, monkeypatch):
    rows = [mrow('B', 'w1', 'wall', 4, {'1': .5, '3': .2}), mrow('B', 'w2', 'wall', 1, {'1': .4})]
    out = c3.admit_mined(mined_fixture(tmp_path, monkeypatch, rows, base=dict(R=['wall'])), {}, tmp_path/'a.jsonl')
    assert not out['ready'] and out['joint_verified_examples'] == 2          # single-actor row is not joint
    (tmp_path/'b').mkdir()
    rows = [mrow('B', 'w1', 'wall', 4, {'1': .5, '2': .1, '3': .2})]
    out = c3.admit_mined(mined_fixture(tmp_path/'b', monkeypatch, rows, base=dict(R=['gap_open'])), {}, tmp_path/'b'/'a.jsonl')
    assert not out['ready'] and out['admitted'] == 0                           # R has no wall support
    (tmp_path/'c').mkdir()
    rows = [mrow('B', 'w1', 'wall', 4, {'1': .5, '2': .1, '3': .2}, veh=False)]
    out = c3.admit_mined(mined_fixture(tmp_path/'c', monkeypatch, rows, base=dict(R=['wall'])), {}, tmp_path/'c'/'a.jsonl')
    assert not out['ready'] and out['joint_verified_examples'] == 0           # no measured synergy -> no G4 row
    (tmp_path/'d').mkdir()
    rows = [mrow('B', 'w1', 'wall', 4), mrow('B', 'w2', 'wall', 4), mrow('B', 'w3', 'wall', 4)]
    out = c3.admit_mined(mined_fixture(tmp_path/'d', monkeypatch, rows, base=dict(R=['wall']),
                         annotated=lambda r: {'1': .2}), {}, tmp_path/'d'/'a.jsonl')
    assert not out['ready']                                                    # annotated CI but no veh_s


def test_lever_receipts_refuse_launch_without_admitted_mined_moves(tmp_path, monkeypatch):
    from tools.coopmwm import clean3_receipts as receipts, composite_health, launch_production as production
    from tools.coopmwm import coordination_opportunities as co, launch_training as training
    from tools.coopmwm.exact_grade import _seal
    from copy import deepcopy
    go = _seal(dict(decision='NO-GO', runtime_sha256='e'*64, executable_joint_labels_ci95_low_positive=0), 'receipt_sha256')
    (tmp_path/'g1').mkdir(); (tmp_path/'g1'/'CLEAN3_GO_NOGO.json').write_text(__import__('json').dumps(go))
    raw = c3.levers(dict(trigger_head=True))
    raw.update(run_root=str(tmp_path), clean3_restart=dict(g1_root=str(tmp_path/'g1'), g1_go_nogo_sha256=go['receipt_sha256']))
    def native(cfg, *, root=None, smoke=False):
        out = deepcopy(cfg); out.update(trigger_candidates=8); out['budget'] = dict(phases=dict(g4_synergy_credit=230400))
        if smoke: out['g4_synergy_credit'] = False
        return out
    monkeypatch.setattr(production, 'backend_config', native)
    monkeypatch.setattr(composite_health, 'report_only', lambda root, arm: True)
    proposals = dict(coordination_opportunities=dict(rule=co.RULE, decisions=1, offered_joint=1, offered_joint_rate=1.,
        graph_edges=1, disjoint_pairs=1, event_decisions=0), proposed_by_family=dict(yield_lane=0, open_gap_merge=0))
    ups = [dict(arm=a, metrics=dict(move_proposals=proposals), update_sha256=a) for a in 'BR']
    with pytest.raises(ValueError, match='mined'):
        receipts.validate_levers(raw, ups)


def test_practice_plan_smoke_priority_puts_mined_families_first(tmp_path, monkeypatch):
    from tools.coopmwm import bank_breadth
    pools = {a: [dict(context_id=f'{a}{f}{i}', family=f, example_id=f'{a}{f}{i}', mined=f in ('yield_lane',))
                 for f in ('gap_open', 'wall', 'yield_lane') for i in range(2)] for a in 'BR'}
    monkeypatch.setattr(bank_breadth, 'matched_pools', lambda cfg: pools)
    monkeypatch.setattr(bank_breadth, 'inputs', lambda cfg: [])
    plain = bank_breadth.practice_plan(tmp_path, 'B', 1, cfg={}, fraction=.2, repeat_cap=2, count=1, training_seed=1)
    assert plain['rows'][0]['family'] == 'gap_open'
    smoke = bank_breadth.practice_plan(tmp_path, 'B', 1, cfg=dict(practice_priority_mined=True), fraction=.2,
                                       repeat_cap=2, count=1, training_seed=1)
    assert smoke['rows'][0]['family'] == 'yield_lane' and smoke['rows'][0]['mined']


def test_mined_demonstration_tape_and_credit_rows(tmp_path, monkeypatch):
    from tools.coopmwm import clean3_mined as cm
    from tools.coopmwm.protocol import digest
    import json as _json
    plan = dict(id='p', members=['c1', 'c2'], duration_s=3, profile=[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    tape = dict(steps=[dict(time_s=100+t, sample=dict(absolute_accels=dict(c1=plan['profile'][0][t],
                c2=plan['profile'][1][t]))) for t in range(3)])
    assert cm.tape_check(plan, tape, 100)['passed']
    bad = _json.loads(_json.dumps(tape)); bad['steps'][1]['sample']['absolute_accels']['c2'] = 0.
    with pytest.raises(ValueError, match='verified profile'):
        cm.tape_check(plan, bad, 100)
    job = tmp_path/'jobs'/'ep1'; job.mkdir(parents=True)
    from tools.coopmwm import accel_storage
    monkeypatch.setattr(accel_storage, 'read', lambda path: tape)
    episodes = [dict(id='ep1', steps=[], practice=dict(verification_id='v1', example_id='v1#s1', seed_index=1, t0=100,
        plan=plan, breadth=dict(family='wall', context_id='x'), demonstration=dict(plan_sha256=digest(plan))))]
    demos = cm.demonstrations(episodes, tmp_path)
    assert demos[0]['executed']['passed']
    mined = tmp_path/'m.jsonl'
    mined.write_text(_json.dumps(dict(arm='B', verification_id='v1', plan_sha256=digest(plan),
        verified_synergy=dict(veh_s={'1': 50.0, '2': 30.0}, lower95={'1': 5.0, '2': 1.0})))+chr(10))
    rows = cm.credit_rows(dict(mined_moves=str(mined)), 'B', demos)
    assert len(rows) == 1 and rows[0]['measured_synergy_veh_s'] == 40.0 and rows[0]['used_in_ppo'] is False
    assert rows[0]['normalized_advantage'] == pytest.approx(40.0/10000.)
    usage = cm.usage(dict(mined_moves=str(mined)), 'B', episodes, dict(bank_trigger=dict(
        contexts=[dict(verification_id='v1', example_id='v1#s1', family='wall')], context_final_ce=[0.25])), tmp_path)
    assert usage['imitation'] == {'wall|v1#s1': 1} and usage['practice_starts'] == {'wall|v1#s1': 1}
    assert usage['executed_joint_moves'] == {'wall|v1#s1': 1} and usage['imitation_fit_ce'] == 0.25


def test_mined_smoke_check_a_to_d_and_fairness(monkeypatch):
    from tools.coopmwm import clean3_mined as cm, bank_breadth, launch_production as production
    monkeypatch.setattr(production, 'backend_config', lambda cfg, **k: dict(cfg))
    found = {a: [dict(verification_id=f'v{a}', plan_sha256='s', example_id=f'v{a}#s1', executable=True,
                      family='yield_lane')] for a in 'BR'}
    monkeypatch.setattr(cm, 'examples', lambda native, arm: (found[arm], {(f'v{arm}', 's')}))
    monkeypatch.setattr(bank_breadth, 'practice_pool', lambda native, arm, **k: [dict(context_id=f'c{arm}',
        verification_id=f'v{arm}', example_id=f'v{arm}#s1', mined=True, demonstrate=True, family='yield_lane')])
    def update(a, passed=True, credit=1, ctx=None):
        return dict(arm=a, learning=dict(bank_trigger=dict(contexts=[dict(verification_id=f'v{a}', example_id=f'v{a}#s1')])),
            practice=dict(practice=1, episodes=[dict(breadth=dict(context_id=ctx or f'c{a}', family='yield_lane'))]),
            clean3_mined_usage=dict(demonstrations=[dict(id='e', family='yield_lane', executed=dict(passed=passed, kind='g5_lane_and_profile'))],
                                    g4_credit=[{}]*credit))
    out = cm.smoke_check({}, [update('B'), update('R')], 'x')
    assert out['B']['practice_starts'] == 1 and out['R']['g4_credit'] == 1
    with pytest.raises(ValueError, match=r'\(c\)'):
        cm.smoke_check({}, [update('B', passed=False), update('R')], 'x')
    with pytest.raises(ValueError, match=r'\(d\)'):
        cm.smoke_check({}, [update('B', credit=0), update('R')], 'x')
    with pytest.raises(ValueError, match=r'\(b\)'):
        cm.smoke_check({}, [update('B', ctx='other'), update('R')], 'x')
    found['R'] = []  # walls-only: R exposed on its own (non-mined) verified examples
    monkeypatch.setattr(cm, 'examples', lambda native, arm: (found[arm], {(f'v{arm}', 's')} if arm == 'B' else set()))
    assert cm.smoke_check({}, [update('B'), update('R', credit=0)], 'x')['R']['g4_credit'] == 0
    found['B'][0]['executable'] = False
    with pytest.raises(ValueError, match=r'\(a\)'):
        cm.smoke_check({}, [update('B'), update('R')], 'x')


def test_mined_move_binds_acceleration_wall_plans_unpatched(monkeypatch):
    from tools.coopmwm import bank_breadth, moves
    plan = dict(id='w', members=['a', 'b', 'c', 'd'], duration_s=3,
                profile=[[0.1]*3, [0.2]*3, [0.3]*3, [0.4]*3])
    seen = []
    monkeypatch.setattr(moves, 'from_plan', lambda p: seen.append(p) or 'wall-move')
    assert bank_breadth.mined_move(plan) == 'wall-move' and seen == [plan]
    with pytest.raises(ValueError, match='members'):
        bank_breadth.mined_move(dict(plan, members=[]))


def test_walls_only_rule_counts_b_examples_only(tmp_path, monkeypatch):
    rows = [mrow('R', 'r1', 'wall', 4, {'1': .5, '2': .1, '3': .2})]
    out = c3.admit_mined(mined_fixture(tmp_path, monkeypatch, rows, base=dict(B=['wall'])), {}, tmp_path/'a.jsonl')
    assert out['admitted'] == 1 and not out['ready'] and out['joint_verified_examples'] == 0


def test_failed_clean3_root_spend_is_counted_once():
    value = c3.build(clean1_rows=rows('a', 1248480, 480), clean2_rows=rows('b', 1849200), g1_rows=[],
                     mining_rows=[], prefixes=PREFIXES, accepted=ACCEPTED, failed_rows=rows('f', 3600))
    assert value['spent']['failed'] == 3600
    assert value['protected_prior_sim_s'] == 1248480+1849200+3600+960+500000
    assert c3.checked(dict(clean3_budget=value))['spent']['failed'] == 3600


def test_supervision_candidates_respect_observable_slots_with_g5_rows():
    """Offline trigger supervision uses the online acquirability contract (r8 prepare fault)."""
    import numpy as np
    from tools.coopmwm import feasible_proposals
    rows = [dict(vid=f'c{i}', s_m=500.+15*i, v_mps=20., lane=i % 2, is_cav=True, length_m=5.) for i in range(7)]
    state = dict(self='c0', rows=rows, acquirable_cav_ids=['c0', 'c1', 'c2', 'c3'])
    for move in feasible_proposals.candidates(state, np.random.default_rng(0), 8):
        assert set(move.members) <= {'c0', 'c1', 'c2', 'c3'}


def test_amended_check_a_reports_non_observable_examples_without_failing(monkeypatch):
    from tools.coopmwm import clean3_mined as cm, bank_breadth, launch_production as production
    monkeypatch.setattr(production, 'backend_config', lambda cfg, **k: dict(cfg))
    found = [dict(verification_id='v', plan_sha256='s', example_id=f'v#s{i}', seed_index=i, executable=True,
                  family='wall') for i in (1, 5)]
    monkeypatch.setattr(cm, 'examples', lambda native, arm: (found, {('v', 's')}) if arm == 'B' else ([], set()))
    monkeypatch.setattr(bank_breadth, 'practice_pool', lambda native, arm, **k: [dict(context_id='c'+arm+e['example_id'],
        verification_id='v', example_id=e['example_id'], mined=arm == 'B', demonstrate=True, family='wall') for e in found])
    def update(a, ctx):
        return dict(arm=a, learning=dict(bank_trigger=dict(contexts=ctx)),
            practice=dict(practice=1, episodes=[dict(breadth=dict(context_id='c'+a+'v#s1', family='wall'))]),
            clean3_mined_usage=dict(demonstrations=[dict(id='e', family='wall',
                executed=dict(passed=True, kind='profile'))], g4_credit=[{}]))
    out = cm.smoke_check(dict(demonstrate_families=['wall']),
                         [update('B', [dict(verification_id='v', example_id='v#s1')]), update('R', [])], 'x')
    obs = out['B']['observability']
    assert obs['observable'] == 1 and obs['non_observable'] == 1
    assert obs['by_seed'] == {'1': dict(observable=1, non_observable=0), '5': dict(observable=0, non_observable=1)}
