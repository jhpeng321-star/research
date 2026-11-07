"""S-WMVERIFY re-plan (owner 2026-11-07) CPU tests (no SUMO, no GPU):

* flag off = 0eee867c: the LCEXEC executor's decisions replayed on recorded stage A/B plant tapes equal the deployed
  code's replay (lcgate/out, which reproduced A_check.json) - opt-in: WMVERIFY_TAPES=<hosttapes dir>,
  WMVERIFY_REPLAY_REF=<lcgate out dir> [WMVERIFY_REPLAY_EPISODES=all|A];
* lcexec_ungated: passes ONLY gap-test refusals (adjacency / missing lane / edge guard / bad sensing still refuse),
  logs would_withhold, restores the gate; worker refuses it outside a pilot root;
* check: 'held' is never an executed lane change;
* B2 (hold continuation) and pilot plans / registration blocks: digests reproducible, B2 = plant arms + u0 + incumbent
  only on superseded never-charged seeds, pilot ids distinct from every charged id, pairing, phase cap, never pooled.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from tools.coopmwm import imitreplay as ir
from tools.coopmwm import map_descriptor as md
from tools.coopmwm import regimeu0 as R
from tools.wzprop import strategy_hook as SH
from tools.wzprop import strategy_search as SS
from tools.wzprop import wmverify as V
from tools.wzprop import wmverify_hook as VH
from tools.wzprop.test_wmverify_cpu import _fake_physics, _registration, inc, lane_spec, solo2, wz  # noqa: F401

CELL = 'D3200_p60'
ARMS_V3 = Path('E:/research-dense/runs/active/C4-V7FT-20261017/wmc_strategy_v3/ARMS_D3200_p60.json')
ARMS_V3_SHA = 'a37d9f8dcee07dd7541d346679a926a8ecdb2acdae443269b94bcbee3966104d'
K_GROUPS = dict(wm=4, wm_fairer=4, structured=8, random=8)
LCEXEC_ARMS = {'lc01_g22', 'rand_425b8bf3216d446f', 'g2_lc01', 'g2_gapopen12', 'rand_bb89ed03daa33a63',
               'rand_25071e07bf236408', 'rand_dd63b48dc43a3e2a', 'rand_52ad2d7c0477e2f1', 'rand_023a4cc1d9dd4737'}
PILOT_ARMS = ['g2_gapopen12', 'rand_bb89ed03daa33a63', 'lc01_g22']
REG_A_JOBS = '4ae95484c833d4d1abb75de7a0ad9b3a0dda8b54e6777227ed2fb0410c31aea7'    # REGISTRATION.md stage A / B
REG_B_JOBS = '2a629f5acc6cd33ab11841fdba2c97fe796f5645eb70addee69557e048561df3'    # (generated on cpuQ, 0eee867c)
# stage A (seed 989950) gate_withheld per lcexec arm = A_check.json lane_receipts (cpuQ, 0eee867c)
A_CHECK_WITHHELD = {'g2_gapopen12': 193, 'g2_lc01': 104, 'lc01_g22': 184, 'rand_023a4cc1d9dd4737': 21,
                    'rand_25071e07bf236408': 0, 'rand_425b8bf3216d446f': 15, 'rand_52ad2d7c0477e2f1': 27,
                    'rand_bb89ed03daa33a63': 270, 'rand_dd63b48dc43a3e2a': 20}


# ------------------------------------------------------------------------------------------ gap-only semantics
def _row(vid, s, lane, v=20.0, cav=True, length=5.0):
    return dict(vid=vid, s_m=float(s), lane=lane, v_mps=float(v), length_m=float(length), is_cav=cav)


def _cases(frame):
    drop = frame.drop_s_m
    ego = _row('c1', -300.0, 1)
    blocker = _row('h1', -297.0, 2, cav=False)              # beside the ego in the target lane: gap test fails
    near_drop = _row('c1', drop - 15.0, 1)                    # inside v*1 s + edge_guard of the drop
    past = _row('c1', drop + 30.0, 1)                         # closing lane (WM 2) no longer exists here
    return dict(
        ok=('c1', 2, [ego], (True, 'ok'), False),
        hold=('c1', 1, [ego, blocker], (True, 'hold'), False),
        gap_only=('c1', 2, [ego, blocker], (False, V.VH.GAP_REASON), True),
        gap_from_closing=('c2', 1, [_row('c2', -300.0, 2), _row('h2', -298.0, 1, cav=False)],
                          (False, V.VH.GAP_REASON), True),
        not_adjacent=('c1', 0, [_row('c1', -300.0, 2), blocker], (False, 'not_adjacent'), False),
        no_target_lane=('c1', 2, [past], (False, 'no_target_lane'), False),
        edge_guard=('c1', 0, [near_drop], (False, V.VH.GAP_REASON), False),
        edge_and_gap=('c1', 0, [near_drop, _row('h1', drop - 14.0, 0, cav=False)], (False, V.VH.GAP_REASON), False),
        bad_sensing=('c1', 2, [ego, _row('h9', -280.0, 7, cav=False)], (False, 'missing_observable_gap_or_cav'),
                     False),
        not_cav=('h1', 1, [blocker, ego], (False, 'missing_observable_gap_or_cav'), False))


def test_gap_only_failure_classifies_only_the_gap_test(wz):
    from tools.coopmwm import lc_families as lf, lc_program as lp
    frame = lp.frame_from_descriptor(wz)
    for name, (vid, target, rows, verdict, gap_only) in _cases(frame).items():
        got, flag = VH.gap_only_failure(lf.lane_gate, vid, target, rows, frame, 3)
        assert got == verdict and flag is gap_only, name
        assert lf.lane_gate(vid, target, rows, frame, 3) == verdict, name       # the gate itself is unchanged


def test_ungated_executor_passes_gap_failed_requests_only(wz, monkeypatch):
    from tools.coopmwm import lc_families as lf, lc_program as lp
    frame = lp.frame_from_descriptor(wz)
    cases = _cases(frame)
    original = lf.lane_gate
    seen = {}

    def fake_commands(self, now, active, **kw):          # Protocol.commands' only gate use: lane_gate(...)
        for name, (vid, target, rows, _v, _g) in cases.items():
            seen[name] = lf.lane_gate(vid, target, rows, self.frame, self.n_lanes)
        return {}
    monkeypatch.setattr(lf.Protocol, 'commands', fake_commands)
    gated = VH.LcexecExecutor(lane_spec(), None)
    gated._init()
    gated.proto.commands(600, [])
    assert {k: v for k, v in seen.items()} == {k: c[3] for k, c in cases.items()}     # flag off: the gate as is
    exe = VH.LcexecUngatedExecutor(lane_spec(), None)
    exe._init()
    assert exe.proto.commands(601, []) == {}
    for name, (_vid, _t, _rows, verdict, gap_only) in cases.items():
        assert seen[name] == ((True, 'ok') if gap_only else verdict), name
    assert lf.lane_gate is original                       # restored after the call
    rc = exe.receipts()['ungated']
    assert rc['would_withhold'] == 2 and rc['gate_calls']['passed_gap_only'] == 2
    assert [(x['t'], x['vid'], x['source'], x['target'], x['old_verdict']) for x in rc['would_withhold_log']] == [
        (601, 'c1', 1, 2, 'would_withhold'), (601, 'c2', 2, 1, 'would_withhold')]
    assert rc['gate_calls']['withheld_not_adjacent'] == 1 and rc['gate_calls']['withheld_no_target_lane'] == 1
    assert rc['gate_calls'][f'withheld_{V.VH.GAP_REASON}'] == 2                 # edge guard (+gap) still refuses
    assert rc['gate_calls']['withheld_missing_observable_gap_or_cav'] == 2
    seen.clear()
    gated.proto.commands(602, [])                         # the gated executor of the same process is untouched
    assert seen == {k: c[3] for k, c in cases.items()}


def test_executor_registry_defaults_unchanged():
    assert VH.resolve_executor(lane_spec()) == 'lcexec' and VH.resolve_executor(inc()) == 'plant'
    assert VH.EXECUTORS['lcexec'] is VH.LcexecExecutor and VH.PILOT_EXECUTORS == ('lcexec_ungated',)
    assert VH.EXECUTORS['lcexec_ungated'].needs_lc_runtime and V.DEFAULT_LANE_EXECUTOR == 'lcexec'
    assert VH.make_hook(lane_spec()).executor == 'lcexec'
    assert VH.make_hook(lane_spec(), executor='lcexec_ungated').exe.name == 'lcexec_ungated'


def test_ungated_hook_runs_on_fake_plant(wz):
    from tools.wzprop import wmverify_fake as F
    a, b = VH.make_hook(lane_spec(), executor='lcexec'), VH.make_hook(lane_spec(), executor='lcexec_ungated')
    F.run_episode(a, seed=989952, seconds=700)
    F.run_episode(b, seed=989952, seconds=700)
    ra, rb = a.receipts(), b.receipts()
    assert rb['executor'] == 'lcexec_ungated' and 'ungated' in rb['protocol'] and 'ungated' not in ra['protocol']
    assert rb['activations'] >= 1 and rb['protocol']['ungated']['would_withhold'] >= 0


# ------------------------------------------------------------------------------------------ check: held != executed
def _side(status, acts=3, executor='lcexec', completed=0):
    return dict(executor=executor, receipts=dict(activations=acts, lc_ticks=10, lc_vehicle_status=status,
                                                 lc_completed=completed,
                                                 protocol=dict(sessions=acts, n_refusals=0, gate_withheld={})))


def test_held_is_not_an_executed_lane_change():
    side = {(989950, 'held_only'): _side({'held': 71}), (989950, 'issued'): _side({'issued': 2, 'held': 5},
                                                                                   completed=1),
            (989951, 'held_only'): _side({'held': 11}), (989950, 'never'): _side({}, acts=0),
            (989950, 'plant_arm'): _side({'held': 9}, executor='plant'),
            (989951, 'pilot_ungated'): _side({'issued': 4}, executor='lcexec_ungated')}
    lanes = V.lane_receipts(side)
    assert set(lanes) == {'held_only', 'issued', 'never', 'pilot_ungated'}
    assert lanes['held_only']['executed_lane_changes'] == 0 and lanes['held_only']['held_requests'] == 82
    assert lanes['issued']['executed_lane_changes'] == 2 and lanes['issued']['completed_lane_changes'] == 1
    warned = {w['arm'] for w in V.lane_warnings(lanes) if w['name'] == 'lane_program_without_executed_lane_requests'}
    assert warned == {'held_only'}                         # 0eee867c counted held as executed: no warning


# ------------------------------------------------------------------------------------------ recorded-tape replay
def _replay(tape, spec, executor):
    """lcgate/replay.py's decision trace (receipts, ticks with lane requests, per-member commands/gate) with the
    CURRENT code, uninstrumented."""
    with gzip.open(tape, 'rt', encoding='utf-8') as f:
        d = json.load(f)
    core = SH.StrategyCore(spec, 300)
    exe = VH.EXECUTORS[executor](spec, core)
    lc_ticks, ticks, member = 0, [], {}
    for st in d['steps']:
        t = int(st['time_s'])
        rows = st['rows']
        tick = exe.step(rows, st['commands'], t)
        if tick:
            lc_ticks += 1
            ticks.append(dict(t=t, lanes=tick.get('lanes'), lane_mode=tick.get('lane_mode'),
                              n_accel=len(tick.get('accel') or {}), native=tick.get('native')))
        by = {r['vid']: r for r in rows}
        for vid, c in ((exe.proto.last_commands if exe.proto is not None else {}) or {}).items():
            r = by.get(vid)
            member.setdefault(vid, []).append(dict(
                t=t, el=c.get('el'), status=c.get('status'), commands=c.get('commands'), gate=c.get('gate'),
                lane=None if r is None else int(r['lane']), s=None if r is None else round(float(r['s_m']), 1),
                v=None if r is None else round(float(r['v_mps']), 2)))
    out = dict(activations=len(core.acts), lc_ticks=lc_ticks, receipts=exe.receipts(), acts=core.acts,
               member_trace=member, ticks_with_lanes=[x for x in ticks if x['lanes']])
    return json.loads(json.dumps(out, default=str))


def _episodes():
    tapes, ref = os.environ.get('WMVERIFY_TAPES'), os.environ.get('WMVERIFY_REPLAY_REF')
    if not tapes or not ref:
        return []
    which = os.environ.get('WMVERIFY_REPLAY_EPISODES', 'A')
    out = []
    for p in sorted(Path(ref).glob('[AB]*.json')):
        ep, arm = p.stem.split('_', 1)
        if which != 'all' and ep != which:
            continue
        root, seed = ('A', 989950) if ep == 'A' else ('B', int(ep[1:]))
        tape = Path(tapes) / root / f'{CELL}_s{seed}_{arm}' / 'plant' / 'RESULT.json.gz'
        if tape.exists():
            out.append(pytest.param(tape, p, arm, id=f'{ep}_{arm}'))
    return out


@pytest.mark.parametrize('tape,ref,arm', _episodes() or [pytest.param(None, None, None, marks=pytest.mark.skip(
    reason='opt-in: WMVERIFY_TAPES + WMVERIFY_REPLAY_REF (recorded stage A/B plant tapes)'))])
def test_flag_off_replay_identity_and_ungated_on_recorded_tapes(tape, ref, arm, monkeypatch):
    if not ARMS_V3.exists():
        pytest.skip('ARMS v3 not present')
    from tools.coopmwm.lc_writer_harness import MAP_DIR
    monkeypatch.delenv(md.ENV, raising=False)
    md._reset_for_tests()
    md.bind(md.load(MAP_DIR))
    try:
        spec = next(s for g in ('wm', 'wm_fairer', 'structured', 'random')
                    for s in json.loads(ARMS_V3.read_text(encoding='utf-8'))[CELL][g] if s['name'] == arm)
        assert VH.resolve_executor(spec) == 'lcexec'
        want = json.loads(Path(ref).read_text(encoding='utf-8'))
        got = _replay(tape, spec, 'lcexec')
        for k in ('activations', 'lc_ticks', 'receipts', 'acts', 'member_trace', 'ticks_with_lanes'):
            assert got[k] == want[k], (arm, k)                    # flag off: byte-identical decisions
        withheld = sum(got['receipts']['gate_withheld'].values())
        if Path(ref).stem.startswith('A_'):
            assert withheld == A_CHECK_WITHHELD[arm]
        gap_only = sum(1 for g in want['gate_log'] if not g['ok'] and g['reason'] == VH.GAP_REASON
                       and not g.get('edge_block'))
        un = _replay(tape, spec, 'lcexec_ungated')                # open loop on the gated rows
        assert un['receipts']['ungated']['would_withhold'] == gap_only
        assert sum(un['receipts']['gate_withheld'].values()) == withheld - gap_only
        assert un['activations'] == got['activations'] and un['receipts']['sessions'] == got['receipts']['sessions']
        assert len(un['ticks_with_lanes']) >= len(got['ticks_with_lanes'])
    finally:
        md._reset_for_tests()


# ------------------------------------------------------------------------------------------ B2 / pilot plans
def _arms_v3():
    if not ARMS_V3.exists():
        pytest.skip('ARMS v3 not present')
    assert ir.sha_file(ARMS_V3) == ARMS_V3_SHA
    return json.loads(ARMS_V3.read_text(encoding='utf-8'))


def test_b2_and_pilot_registration_blocks_v3_arms():
    doc = _arms_v3()
    specs = {CELL: V.candidates(doc, CELL, V.GROUPS, K_GROUPS)}
    lc = {s['name'] for _src, s in specs[CELL] if VH.resolve_executor(s) == 'lcexec'}
    assert lc == LCEXEC_ARMS and len(specs[CELL]) == 24
    for span, digest, planned in (('989950-989950', REG_A_JOBS, 33_600), ('989951-989957', REG_B_JOBS, 220_800)):
        gr = V.build_groups(V.parse_cells([f'{CELL}:{span}']), specs, R.world_of, parity_seeds=1)
        assert (V.jobs_digest(gr), sum(j['sim_s'] for g in gr for j in g['jobs'])) == (digest, planned)
    b2_cells = V.parse_cells([f'{CELL}:989954-989957'])
    t1, p1 = V.b2_registration_text(cells=b2_cells, specs_by_cell=specs, cap=280_000, arms_sha256=ARMS_V3_SHA,
                                    prior_committed=129_600)
    t2, p2 = V.b2_registration_text(cells=b2_cells, specs_by_cell=specs, cap=280_000, arms_sha256=ARMS_V3_SHA,
                                    prior_committed=129_600)
    assert t1 == t2 and p1 == p2 == 1200 * (1 + 4 * (15 + 2)) == 82_800
    kept = V.select_executors(specs, ('plant',))
    gr = V.build_groups(b2_cells, kept, R.world_of, parity_seeds=0)
    jobs = [j for g in gr for j in g['jobs']]
    assert V.registration_check(t1, cap=280_000, planned=p1, jobs_sha256=V.jobs_digest(gr), arms_sha256=ARMS_V3_SHA,
                                module_sha256=V.module_shas())
    assert not {j['arm'] for j in jobs} & LCEXEC_ARMS and {j['executor'] for j in jobs} == {'u0', 'regimeu0', 'plant'}
    assert V.PARITY_ARM not in {j['arm'] for j in jobs} and len(jobs) == 1 + 4 * 17
    assert {j['world']['seed'] for j in jobs[1:]} == {989954, 989955, 989956, 989957}
    assert 'excludes the lane-program (lcexec) arms' in t1 and '129,600 sim-s + this root 82,800' in t1
    # pilot: 3 lcexec arms x 989951-989952, ids distinct from every stage B id
    p_cells = V.parse_cells([f'{CELL}:989951-989952'])
    pil = V.pilot_specs(specs, PILOT_ARMS)
    pt, pp = V.pilot_registration_text(cells=p_cells, pilot_by_cell=pil, cap=280_000, arms_sha256=ARMS_V3_SHA,
                                       prior_committed=129_600 + 82_800, selection_note='x')
    assert pp == 1200 * (1 + 3 * 2) == 8_400 and pt == V.pilot_registration_text(
        cells=p_cells, pilot_by_cell=pil, cap=280_000, arms_sha256=ARMS_V3_SHA, prior_committed=212_400,
        selection_note='x')[0]
    pg = V.build_pilot_groups(p_cells, pil, R.world_of)
    assert V.registration_check(pt, cap=280_000, planned=pp, jobs_sha256=V.jobs_digest(pg), arms_sha256=ARMS_V3_SHA,
                                module_sha256=V.module_shas())
    b_all = V.build_groups(V.parse_cells([f'{CELL}:989951-989957']), specs, R.world_of, parity_seeds=1)
    b_ids = {j['id'] for g in b_all for j in g['jobs']}
    pj = [j for g in pg[1:] for j in g['jobs']]
    assert len(pj) == 6 and not {j['id'] for j in pj} & b_ids and all(j['id'].endswith('_ungated') for j in pj)
    assert all(j['pilot']['gated_job'] in b_ids and j['pilot']['u0_job'] in b_ids for j in pj)
    assert {j['executor'] for j in pj} == {'lcexec_ungated'} and '220,800 sim-s <= cap 280,000' in pt
    with pytest.raises(ValueError, match='lcexec'):
        V.pilot_specs(specs, ['wall_g_alt21'])


def _plan_b(tmp_path, reg_root, arms, cap):
    cells = V.parse_cells([f'{CELL}:989950-989951'])
    specs = {CELL: V.candidates(json.loads(arms.read_text()), CELL)}
    reg, _planned = _registration(tmp_path / 'regB', cells, specs, cap, parity_seeds=0, arms=arms)
    return V.plan(tmp_path / 'B', regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap, registration_md=reg,
                  parity_seeds=0)


def test_e2e_b2_hold_continuation_and_ungated_pilot(tmp_path, monkeypatch, wz):
    from tools.wzprop.test_wmverify_cpu import _e2e_inputs
    from tools.wzprop import wmverify_bank as BK
    reg_root, arms = _e2e_inputs(tmp_path)              # candidates: wmv_lc_closing (lcexec) + g_v18 (plant)
    cap = 40_000
    (tmp_path / 'regB').mkdir()
    body_b = _plan_b(tmp_path, reg_root, arms, cap)
    root_b = tmp_path / 'B'
    _fake_physics(monkeypatch)
    for g in body_b['groups'][:2]:                       # identity + seed 989950 run; seed 989951 never charged
        for j in g['jobs']:
            assert ir.charge(root_b, [(j['id'], j['sim_s'], g['stage'])])
            assert not V.worker(root_b, j['id'])['error']
    spent_b = ir.spent(root_b)
    doc = json.loads(arms.read_text())
    specs = {CELL: V.candidates(doc, CELL)}
    b2_cells = V.parse_cells([f'{CELL}:989951-989951'])
    text, planned = V.b2_registration_text(cells=b2_cells, specs_by_cell=specs, cap=cap,
                                           arms_sha256=ir.sha_file(arms), prior_committed=spent_b)
    reg2 = tmp_path / 'REG_B2.md'
    reg2.write_text('# reg\n\n' + text, encoding='utf-8')
    kw = dict(regimeu0_root=reg_root, arms_path=arms, cells=b2_cells, cap=cap, registration_md=reg2, parity_seeds=0,
              prior_roots=[root_b], only_executors=('plant',))
    with pytest.raises(ValueError, match='prior root'):                     # no supersede flag
        V.plan(tmp_path / 'B2x', **kw)
    with pytest.raises(ValueError, match='prior root'):                     # B not STOPPED
        V.plan(tmp_path / 'B2y', supersede_uncharged=True, **kw)
    (root_b / 'STOP').write_text('owner pause')
    with pytest.raises(ValueError, match='phase total'):
        V.plan(tmp_path / 'B2z', supersede_uncharged=True, **dict(kw, cap=spent_b + planned - 1200))
    body2 = V.plan(tmp_path / 'B2', supersede_uncharged=True, **kw)
    root2 = tmp_path / 'B2'
    jobs2 = [j for g in body2['groups'] for j in g['jobs']]
    assert [j['arm'] for j in jobs2] == ['u0', 'u0', 'wall_g_solo2', 'g_v18'] and body2['planned_sim_s'] == planned
    assert body2['supersedes'][0]['groups'][0]['seed'] == 989951 and body2['only_executors'] == ['plant']
    assert body2['phase_budget']['total'] == spent_b + planned
    assert V.superseded_groups([root_b, root2]) == {(body_b['plan_sha256'], 989951)}
    for g in body2['groups']:
        for j in g['jobs']:
            assert ir.charge(root2, [(j['id'], j['sim_s'], g['stage'])])
            assert not V.worker(root2, j['id'])['error']
    col = V.collect([root_b, root2])
    assert sorted((r['seed'], Path(r['root']).name) for r in col['rows']) == [(989950, 'B'), (989951, 'B2')]
    chk = V.checks([root_b, root2], col)
    names = {c['name']: c for c in chk['checks']}
    assert names['superseded_groups_never_charged']['ok']
    assert names['crn_no_divergence_before_first_activation']['ok']
    # ---- pilot: the lcexec arm on seed 989950, paired with B's charged + DONE jobs
    p_cells = V.parse_cells([f'{CELL}:989950-989950'])
    pil = V.pilot_specs(specs, ['wmv_lc_closing'])
    pt, pp = V.pilot_registration_text(cells=p_cells, pilot_by_cell=pil, cap=cap, arms_sha256=ir.sha_file(arms),
                                       prior_committed=spent_b + planned)
    regp = tmp_path / 'REG_P.md'
    regp.write_text('# reg\n\n' + pt, encoding='utf-8')
    pkw = dict(regimeu0_root=reg_root, arms_path=arms, cells=p_cells, cap=cap, registration_md=regp,
               prior_roots=[root_b, root2], pilot_arms=['wmv_lc_closing'])
    with pytest.raises(ValueError, match='lcexec'):
        V.pilot_specs(specs, ['g_v18'])
    with pytest.raises(ValueError, match='not planned in a prior root'):
        V.plan(tmp_path / 'Px', **dict(pkw, prior_roots=[root2]))
    bp = V.plan(tmp_path / 'P', **pkw)
    root_p = tmp_path / 'P'
    pj = [j for g in bp['groups'] for j in g['jobs']]
    assert [j['id'] for j in pj] == [V.IDENTITY_JOB, f'{CELL}_s989950_wmv_lc_closing_ungated']
    assert pj[1]['executor'] == 'lcexec_ungated' and bp['pilot']['pairs'][0]['gated_job']['id'] == \
        f'{CELL}_s989950_wmv_lc_closing' and bp['families'] == {} and pp == bp['planned_sim_s'] == 2400
    prior_ids = {c['id'] for r in (root_b, root2) for c in ir.charges(r)} - {V.IDENTITY_JOB}
    assert pj[1]['id'] not in prior_ids
    # the pilot executor never runs outside a pilot root
    bad = json.loads((root2 / 'PLAN.json').read_text())
    bad['groups'][-1]['jobs'][-1]['executor'] = 'lcexec_ungated'
    (tmp_path / 'X').mkdir()
    (tmp_path / 'X' / 'PLAN.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='pilot root'):
        V.worker(tmp_path / 'X', bad['groups'][-1]['jobs'][-1]['id'])
    for g in bp['groups']:
        for j in g['jobs']:
            assert ir.charge(root_p, [(j['id'], j['sim_s'], g['stage'])])
            assert not V.worker(root_p, j['id'])['error']
    side = json.loads((root_p / 'jobs' / pj[1]['id'] / 'WMVERIFY.json').read_text())
    assert side['executor'] == 'lcexec_ungated' and 'ungated' in side['receipts']['protocol']
    idj = ir._read(root_p / 'jobs' / V.IDENTITY_JOB / 'SUMMARY.json.gz')['totals']['team_delay_vs']
    monkeypatch.setattr(R, 'IDENTITY', dict(R.IDENTITY, expected_team_delay_vs=idj))
    rep = V.pilot_report(root_p, [root_b, root2])
    assert rep['integrity']['ok'], [c for c in rep['integrity']['checks'] if not c['ok']]
    assert rep['pooled_with_gated'] is False and 'NEVER pooled' in rep['label']
    ep = rep['episodes'][0]
    assert ep['status'] == 'ok' and ep['diff_vs_gated'] == pytest.approx(ep['J'] - ep['J_gated'])
    assert {'diff_vs_u0', 'diff_vs_wall_g_solo2'} <= set(ep)
    s = rep['summary']['wmv_lc_closing_ungated']
    assert s['n'] == 1 and s['lane']['seeds'] == 1 and s['lane_gated']['seeds'] == 1
    assert 'would_withhold' in s['lane'] and 'would_withhold' not in s['lane_gated']
    # never pooled: main collect / analyze / bank refuse the pilot root
    with pytest.raises(ValueError, match='PILOT'):
        V.collect([root_b, root_p])
    with pytest.raises(ValueError, match='PILOT'):
        _bank_refuses(BK, root_p)
    out = tmp_path / 'pilot_out'
    V.main(['pilot', '--pilot-root', str(root_p), '--prior-root', str(root_b), str(root2), '--out', str(out)])
    assert json.loads((out / 'PILOT_UNGATED.json').read_text())['schema'] == V.SCHEMA + '.pilot_ungated'
    with pytest.raises(FileExistsError):
        V.write_pilot(out, root_p, [root_b, root2])


def _bank_refuses(BK, root_p):
    """wmverify_bank.build on a pilot root (with a minimal integrity-clean analysis dir): refused."""
    import hashlib
    adir = root_p.parent / 'fake_analysis'
    adir.mkdir(exist_ok=True)
    allowed = json.dumps(dict(cells={})).encode()
    (adir / 'ANALYSIS_ALLOWED.json').write_bytes(allowed)
    (adir / 'VERIFY.json').write_text(json.dumps(dict(analysis_written=True, integrity=dict(ok=True),
                                                      analysis_allowed_sha256=hashlib.sha256(allowed).hexdigest())))
    return BK.build(adir, [root_p])
