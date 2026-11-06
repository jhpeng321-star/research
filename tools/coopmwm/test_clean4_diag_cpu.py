"""CPU tests for the read-only CLEAN4 per-update diagnostic (tools.coopmwm.clean4_diag).

Fixtures: fixtures/clean4_diag/clean2_root holds REAL sealed clean2 receipts (cpuV
/home/ubuntu/coopmwm20261105_clean2: {A,B,R}_UPDATE_00{1,2}.json gzipped byte-for-byte,
LEARNING_MONITOR_u001.json); fixtures/clean4_diag/clean2_tapes holds two TRIMMED real clean2
job tapes (trigger steps/fields only). The synthetic root covers the W1/N2 family names
(wall_initiate, wall_join, closing_pair, abreast_wall, lane_drop_initiate*, lane_drop_join),
clean3 mined usage and G4 credit, which clean2 did not log.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tools.coopmwm import clean4_diag as cd

FIXTURES = Path(__file__).resolve().parent/'fixtures'/'clean4_diag'
NL = cd.NL


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.is_dir(), None if p.is_dir() else _sha(p),
                                            p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*'))}


def materialize_clean2(dest):
    root = dest/'clean2_run'
    root.mkdir()
    for path in (FIXTURES/'clean2_root').iterdir():
        if path.name.endswith('.json.gz'):
            (root/path.name[:-3]).write_bytes(gzip.decompress(path.read_bytes()))
        else:
            shutil.copyfile(path, root/path.name)
    return root


def raw(root, name):
    return json.loads((root/name).read_text(encoding='utf-8'))


def write_tape(root, episode_id, value, *, pointer, sha=None):
    folder = root/'jobs'/episode_id
    folder.mkdir(parents=True)
    payload = json.dumps(value, sort_keys=True).encode()
    if pointer:
        (folder/'RESULT.json.gz').write_bytes(gzip.compress(payload, mtime=0))
        (folder/'RESULT.json').write_text(json.dumps(dict(encoding=cd.TAPE_ENCODING, file='RESULT.json.gz',
            sha256=sha or hashlib.sha256(payload).hexdigest(), uncompressed_bytes=len(payload))))
    else:
        (folder/'RESULT.json').write_bytes(payload)


def expected_delay(root, arm, u):
    receipt, reference = raw(root, f'{arm}_UPDATE_{u:03}.json'), raw(root, f'A_UPDATE_{u:03}.json')
    practice = {e['id'] for e in receipt['practice']['episodes']}
    a_practice = {e['id'] for e in reference['practice']['episodes']}
    a_rows = {r['world']['seed']: r for r in reference['physical'] if r['id'] not in a_practice}
    diffs = [r['totals']['team_delay_vs']-a_rows[r['world']['seed']]['totals']['team_delay_vs']
             for r in receipt['physical'] if r['id'] not in practice and r['world']['seed'] in a_rows]
    return sum(diffs)/len(diffs), len(diffs)


# ------------------------------------------------------------------ real clean2 receipts


def test_clean2_real_receipts_logged_and_not_logged(tmp_path):
    root = materialize_clean2(tmp_path)
    before = snapshot(root)
    reports = {u: cd.diagnose_update(root, u) for u in (1, 2)}
    for u, report in reports.items():
        assert report['complete'] and report['line'].startswith(f'CLEAN4 diag u{u} ')
        assert 'not logged' in report['line'] and '\n' not in report['line']
        for arm in cd.ARMS:
            assert report['arms'][arm]['seal_ok'] is True
        for arm in ('B', 'R'):
            receipt = raw(root, f'{arm}_UPDATE_{u:03}.json')
            a = report['arms'][arm]
            im = a['imitation']
            # (1) clean2 logged the trigger-bank CE and the effective-space imitation.
            assert im['bank_move_ce'] == receipt['learning']['bank_move_ce']
            assert im['bank_move_ce_initial'] == receipt['learning']['bank_trigger']['initial_ce']
            assert im['bank_move_examples'] == receipt['learning']['bank_trigger']['examples']
            epochs = receipt['learning']['bank']['epochs']
            assert im['bank_imitation_loss_before'] == epochs[0]['awr_loss_before']
            assert im['bank_imitation_loss_after'] == epochs[-1]['awr_loss_after']
            assert isinstance(im['bank_imitation_action_error_after_mps2'], float)
            # clean2 had no per-context CE and no mined usage block.
            assert im['bank_move_ce_by_family'] == NL
            assert im['mined_imitation_fit_ce'] == NL and im['mined_imitation_fit_contexts'] == NL
            # (2) overall fire rate from metrics.move_proposals; no tapes -> split not logged.
            proposals = receipt['metrics']['move_proposals']
            tr = a['trigger']
            assert tr['all']['rate'] == proposals['selected_non_none']/proposals['decisions'] > 0
            assert tr['all']['source'] == 'metrics.move_proposals'
            for kind in ('lesson_moment', 'non_lesson', 'practice', 'practice_mined', 'ordinary'):
                assert tr[kind] == NL
            assert tr['tapes'] == f"0/{len(receipt['physical'])}"
            assert tr['mined_trigger_rate_by_family'] == NL
            # (3) proposals/accepted totals logged; W1/N2 protocol + per-type split not logged.
            gm = a['group_moves']
            assert gm['proposed_by_family'] == proposals['proposed_by_family']
            assert gm['accepted_moves_total'] == receipt['execution']['accepted_moves'] > 0
            assert gm['selected_by_family'] == {f: n for f, n in receipt['metrics']['trigger_counts'].items()
                                                if f != 'none'}
            for key in ('initiated', 'joined', 'v2v_wall', 'lane_drop', 'mined_executed_joint_moves',
                        'mined_joint_selected'):
                assert gm[key] == NL, key
            # (4) clean2 had no G4.
            for key in ('receipt', 'envelope', 'mined', 'plan'):
                assert a['g4'][key] == NL
            # (5) team delay vs A, ordinary episodes paired by world seed.
            td = a['team_delay']
            mean, n = expected_delay(root, arm, u)
            assert td['paired_n'] == n == 10 and td['diff_vs_a_veh_s'] == pytest.approx(mean)
            assert td['practice_excluded'] == len(receipt['practice']['episodes']) == 6
            assert 'positive = more delay than A = worse' in td['sign_convention']
        a_arm = reports[u]['arms']['A']
        assert a_arm['imitation']['bank_move_ce'] == NL and a_arm['trigger']['all'] == NL
        assert a_arm['group_moves']['proposed_by_family'] == NL
    assert reports[1]['monitor']['seal_ok'] is True
    assert {a: r['action'] for a, r in reports[1]['monitor']['arms'].items()} == dict(
        A='CONTINUE', B='CONTINUE', R='CONTINUE')
    assert reports[2]['monitor'] == NL   # only u001's monitor is in the fixture
    assert reports[1]['g4_plan'] == NL and reports[1]['track2_diagnostics'] == NL
    assert snapshot(root) == before      # nothing written under the run root


def test_cli_status_file_idempotent_and_never_inside_root(tmp_path, capsys):
    root = materialize_clean2(tmp_path)
    before = snapshot(root)
    status = tmp_path/'out'/'CLEAN4_DIAG.log'
    assert cd.main([str(root), '--status-file', str(status), '--no-tapes']) == 0
    lines = status.read_text(encoding='utf-8').splitlines()
    assert [line.split(' | ')[0] for line in lines] == ['CLEAN4 diag u1', 'CLEAN4 diag u2']
    assert cd.main([str(root), '--status-file', str(status), '--no-tapes']) == 0
    assert cd.main([str(root), '--status-file', str(status), '--update', '2', '--no-tapes']) == 0
    assert status.read_text(encoding='utf-8').splitlines() == lines
    state = json.loads((tmp_path/'out'/'CLEAN4_DIAG.log.state.json').read_text())
    assert state['roots'] == {str(root.resolve()): [1, 2]}
    capsys.readouterr()
    assert cd.main([str(root), '--update', '1', '--no-tapes']) == 0   # stdout only
    assert capsys.readouterr().out.startswith('CLEAN4 diag u1 ')
    assert cd.main([str(root), '--update', '1', '--no-tapes', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['arms']['B']['receipt'] == 'present'
    with pytest.raises(ValueError, match='outside the run root'):
        cd.main([str(root), '--status-file', str(root/'diag.log'), '--no-tapes'])
    assert snapshot(root) == before


def test_trimmed_real_clean2_tapes_split_lesson_vs_ordinary(tmp_path):
    root = materialize_clean2(tmp_path)
    tapes = {p.name[:-len('.json.gz')]: json.loads(gzip.decompress(p.read_bytes()))
             for p in (FIXTURES/'clean2_tapes').glob('*.json.gz')}
    ordinary, practice = 't2c_warm_u001_B_986000', 't2c_warm_u001_p04_B_986026'
    write_tape(root, ordinary, tapes[ordinary], pointer=True)
    write_tape(root, practice, tapes[practice], pointer=False)
    before = snapshot(root)
    report = cd.diagnose_update(root, 1)
    tr = report['arms']['B']['trigger']
    assert tr['tapes'] == '2/16'
    def proposals(tape, predicate=lambda s: True):
        rows = [s['sample']['track2_trigger']['proposals'] for s in tape['steps'] if predicate(s)]
        return sum(r['selected_non_none'] for r in rows), sum(r['decisions'] for r in rows)
    # Independent check: the per-step F2 proposal receipts recorded in the same tapes.
    assert (tr['ordinary']['non_none'], tr['ordinary']['decisions']) == proposals(tapes[ordinary])
    assert tr['ordinary']['episodes'] == 1 and tr['practice']['episodes'] == 1
    assert (tr['practice']['non_none'], tr['practice']['decisions']) == proposals(tapes[practice])
    t0 = tapes[practice]['practice']['t0']
    first = min(s['time_s'] for s in tapes[practice]['steps'] if s['time_s'] >= t0)
    assert (tr['lesson_moment']['non_none'], tr['lesson_moment']['decisions']) == proposals(
        tapes[practice], lambda s: s['time_s'] == first)
    assert tr['lesson_moment']['decisions'] > 0
    assert tr['non_lesson']['decisions'] == (tr['ordinary']['decisions']+tr['practice']['decisions']
                                             - tr['lesson_moment']['decisions'])
    assert tr['practice_mined'] == NL    # clean2: no clean3_mined_usage
    gm = report['arms']['B']['group_moves']
    assert '2/16' in gm['source'] and gm['joined'] == {}
    assert set(gm['initiated']) <= {'wall', 'ramp_assist'}
    assert gm['v2v_wall'] == NL and gm['lane_drop'] == NL   # W1/N2 off in clean2
    assert report['arms']['R']['trigger']['ordinary'] == NL   # R tapes absent
    assert 'lesson=' in report['line'] and 'tapes=2/16' in report['line']
    assert snapshot(root) == before


# ------------------------------------------------------------------ synthetic W1/N2 + clean3 + G4


def _seal(value, key='update_sha256'):
    return dict(value, **{key: cd.digest(value)})


def _world(seed):
    return dict(cell=0, demand=2000, penetration=0.1, repeat=0, seed=seed)


def _phys(episode_id, seed, delay):
    return dict(id=episode_id, world=_world(seed), ttc_events=0,
                totals=dict(team_delay_vs=float(delay), collisions=0.0, teleports=0.0, hard_brake=0.0))


def _step(time_s, picks, *, executed=False, proposals=None):
    """picks: [(vid, move-or-None, accepted)] -> one recorded track2_trigger step."""
    members = [vid for vid, _, _ in picks]
    tables = [[None, move] if move is not None else [None] for _, move, _ in picks]
    actions = [1 if move is not None else 0 for _, move, _ in picks]
    accepted = {vid: bool(ok) for vid, move, ok in picks if move is not None}
    return dict(time_s=time_s, sample=dict(behaviour=dict(executed=executed), deterministic=False,
        track2_trigger=dict(context=dict(members=members), tables=tables, actions=actions,
                            accepted=accepted, proposals=proposals or {})))


LD = dict(offered_join=1, selected_initiate=1, accepted_initiate=1, selected_join=1, accepted_join=1)
WALL = dict(offered_join=2, selected_initiate=1, accepted_initiate=1, selected_join=1, accepted_join=1)


@pytest.fixture
def synthetic_n2_root(tmp_path):
    root = tmp_path/'synthetic_clean4'
    (root/'g4').mkdir(parents=True)
    (root/'updates').mkdir()
    families = ['none', 'ease_off', 'wall', 'gap_open', 'ramp_assist', 'closing_pair', 'abreast_wall',
                'lane_drop_initiate', 'lane_drop_join', 'wall_initiate', 'wall_join']
    counts = dict.fromkeys(families, 0)
    counts.update(none=3, ease_off=1, closing_pair=2, abreast_wall=1, lane_drop_initiate=1,
                  lane_drop_join=1, wall_initiate=1, wall_join=1)
    proposals = dict(rule='f2-feasible-balanced-v1', decisions=11, selected_non_none=8, rejected_non_none=1,
                     family_checks=40, infeasible_family_checks=10,
                     proposed_by_family={f: 2 for f in families[1:]})
    common = dict(update=1, config_sha256='0'*64, smoke=False, safety=dict(status='checked'))
    a = _seal(dict(common, arm='A', learning=dict(bank=dict(status='A_no_bank'), ppo=[]),
        metrics=dict(decisions=0, trigger_counts=dict.fromkeys(cd.ARMS, 0)), execution=dict(accepted_moves=0),
        practice=dict(episodes=[], practice=0), physical=[_phys('u1_A_1', 1, 100), _phys('u1_A_2', 2, 200)]))
    b_learning = dict(bank_move_ce=1.5, bank_trigger=dict(initial_ce=1.9, final_ce=1.5, examples=3,
        family_counts=dict(closing_pair=2, wall_initiate=1),
        contexts=[dict(family='closing_pair'), dict(family='wall_initiate'), dict(family='closing_pair')],
        context_final_ce=[1.0, 2.0, 3.0]),
        bank=dict(epochs=[dict(accepted=True, awr_loss_before=5.0, awr_loss_after=4.0),
                          dict(accepted=False, awr_loss_before=4.0, awr_loss_after=4.0)],
                  action_error_before_mps2=0.3, action_error_after_mps2=0.25, kl=0.01, pairs_total=7),
        g4_synergy_credit=dict(enabled=True, measured_starts=2, eligible_starts=5, unsampled_starts=3,
            rows=[dict(synergy_veh_s=10.0, advantage=0.1), dict(synergy_veh_s=-4.0, advantage=-0.04)]))
    mined = dict(arm='B', imitation_fit_ce=0.7, imitation_fit_contexts=2, imitation={'closing_pair|ex1#s1': 2},
                 practice_starts={'closing_pair|ex1#s1': 1}, executed_joint_moves={'closing_pair|ex1#s1': 1},
                 trigger_rate_by_family={'none': 0.5, 'closing_pair:joint': 0.25, 'wall_initiate': 0.25},
                 trigger_decisions=8, g4_credit=[dict(measured_synergy_veh_s=5.0, normalized_advantage=0.05,
                                                      used_in_ppo=False)])
    practice = dict(practice=1, episodes=[dict(id='u1_p01_B_3', t0=300, example_id='ex1#s1',
        verification_id='ex1', seed_index=1, breadth=dict(family='closing_pair'))])
    b = _seal(dict(common, arm='B', learning=b_learning, clean3_mined_usage=mined, practice=practice,
        metrics=dict(decisions=11, trigger_counts=counts, move_proposals=proposals),
        execution=dict(accepted_moves=7),
        physical=[_phys('u1_B_1', 1, 90), _phys('u1_B_2', 2, 230), _phys('u1_p01_B_3', 3, 999)]))
    r = _seal(dict(common, arm='R', learning=dict(bank_move_ce=1.6), practice=dict(episodes=[], practice=0),
        metrics=dict(decisions=11, trigger_counts=counts, move_proposals=proposals),
        execution=dict(accepted_moves=6),
        physical=[_phys('u1_R_1', 1, 120), _phys('u1_R_x_inc1', 2, 260), _phys('u1_R_corrupt', 4, 50)]))
    for arm, value in (('A', a), ('B', b), ('R', r)):
        (root/f'{arm}_UPDATE_001.json').write_text(json.dumps(value, sort_keys=True))
    (root/'g4'/'ENVELOPE_001.json').write_text(json.dumps(dict(rows=[
        dict(start=dict(arm='B'), synergy_veh_s=10.0, advantage=0.1),
        dict(start=dict(arm='B'), synergy_veh_s=-4.0, advantage=-0.04),
        dict(start=dict(arm='R'), synergy_veh_s=-7.0, advantage=-0.07)])))
    plan = dict(update=1, quote_sim_s=4*1200*3, starts=[dict(arm='B'), dict(arm='B'), dict(arm='R')], jobs=[])
    (root/'CHARGES.jsonl').write_text(json.dumps(dict(id='u1_B_1', phase='train', sim_s=1200))+'\n'
                                      + json.dumps(dict(id='g4_x', phase='g4', sim_s=0, g4_plan=plan))+'\n'
                                      + '{"id": "partial-tail')
    ld_init = dict(family='lane_drop_initiate', members=['c1'], kind='closing_pair')
    ld_join = dict(family='lane_drop_join', members=['c3'], kind='abreast_wall')
    write_tape(root, 'u1_B_1', dict(id='u1_B_1', steps=[
        _step(300, [('c1', ld_init, True), ('c2', dict(family='closing_pair', members=['c2', 'c3']), False),
                    ('c3', None, False)], proposals=dict(lane_drop=LD, v2v_wall=WALL)),
        _step(360, [('c1', dict(family='wall_initiate', members=['c1']), True),
                    ('c2', dict(family='wall_join', members=['c2']), True), ('c3', ld_join, True)],
              proposals=dict(lane_drop=LD, v2v_wall=WALL)),
        dict(time_s=361, sample=dict(own_action=[[0.0]]))]), pointer=False)
    write_tape(root, 'u1_B_2', dict(id='u1_B_2', steps=[
        _step(300, [('c4', dict(family='abreast_wall', members=['c4', 'c5']), True),
                    ('c5', dict(family='ease_off', members=['c5']), True)])]), pointer=True)
    write_tape(root, 'u1_p01_B_3', dict(id='u1_p01_B_3', practice=dict(t0=300), steps=[
        _step(301, [('c1', dict(family='closing_pair', members=['c1', 'c2']), True), ('c2', None, False)]),
        _step(361, [('c1', dict(family='ease_off', members=['c1']), True)], executed=True),
        _step(421, [('c2', None, False)])]), pointer=True)
    write_tape(root, 'u1_R_corrupt', dict(id='u1_R_corrupt', steps=[]), pointer=True, sha='f'*64)
    (root/'jobs'/'u1_R_x_inc1').mkdir()   # forbidden token: must never be opened
    (root/'jobs'/'u1_R_x_inc1'/'RESULT.json').write_text('{}')
    return root


def test_synthetic_w1_n2_families_clean3_and_g4(synthetic_n2_root):
    root = synthetic_n2_root
    before = snapshot(root)
    report = cd.diagnose_update(root, 1)
    b, r = report['arms']['B'], report['arms']['R']
    assert b['seal_ok'] is True and r['seal_ok'] is True and report['complete']
    # (1)
    im = b['imitation']
    assert im['bank_move_ce'] == 1.5 and im['bank_move_ce_initial'] == 1.9
    assert im['bank_move_ce_by_family'] == {'closing_pair': dict(mean_ce=2.0, contexts=2),
                                            'wall_initiate': dict(mean_ce=2.0, contexts=1)}
    assert (im['bank_imitation_loss_before'], im['bank_imitation_loss_after']) == (5.0, 4.0)
    assert im['bank_imitation_epochs_accepted'] == 1
    assert (im['mined_imitation_fit_ce'], im['mined_imitation_fit_contexts'], im['mined_imitation_contexts']) == (0.7, 2, 2)
    assert r['imitation']['bank_move_ce'] == 1.6 and r['imitation']['bank_move_ce_initial'] == NL
    # (2) lesson moment = first trigger tick at/after t0 of the practice episode
    tr = b['trigger']
    assert tr['all']['rate'] == 8/11 and tr['tapes'] == '3/3'
    assert (tr['lesson_moment']['non_none'], tr['lesson_moment']['decisions']) == (1, 2)
    assert tr['lesson_moment']['lesson_family_selected'] == 1
    assert (tr['practice']['non_none'], tr['practice']['decisions'], tr['practice']['demo_decisions']) == (1, 3, 1)
    assert tr['practice_mined']['decisions'] == 3 and tr['practice_mined']['episodes'] == 1
    assert (tr['ordinary']['non_none'], tr['ordinary']['decisions'], tr['ordinary']['episodes']) == (7, 8, 2)
    assert (tr['non_lesson']['non_none'], tr['non_lesson']['decisions']) == (7, 9)
    assert tr['mined_trigger_rate_by_family']['closing_pair:joint'] == 0.25
    # (3) by move type; single-actor ease_off is not a group move
    gm = b['group_moves']
    assert gm['initiated'] == {'lane_drop_initiate:closing_pair': dict(selected=1, accepted=1),
                               'closing_pair': dict(selected=2, accepted=1),
                               'wall_initiate': dict(selected=1, accepted=1),
                               'abreast_wall': dict(selected=1, accepted=1)}
    assert gm['joined'] == {'wall_join': dict(selected=1, accepted=1),
                            'lane_drop_join:abreast_wall': dict(selected=1, accepted=1)}
    assert gm['lane_drop'] == {k: 2*v for k, v in LD.items()}
    assert gm['v2v_wall'] == {k: 2*v for k, v in WALL.items()}
    assert gm['by_episode_kind']['practice']['initiated'] == {'closing_pair': dict(selected=1, accepted=1)}
    assert gm['accepted_moves_total'] == 7 and gm['mined_executed_joint_moves'] == {'closing_pair': 1}
    assert gm['mined_joint_selected'] == {'closing_pair': 2}
    assert gm['proposed_by_family']['lane_drop_initiate'] == 2
    # R: no readable tape -> receipt fallback (protocol families by name, acceptance not logged)
    rg = r['group_moves']
    assert rg['initiated'] == {'wall_initiate': dict(selected=1, accepted=NL),
                               'lane_drop_initiate': dict(selected=1, accepted=NL)}
    assert rg['joined'] == {'wall_join': dict(selected=1, accepted=NL),
                            'lane_drop_join': dict(selected=1, accepted=NL)}
    assert rg['lane_drop'] == NL and rg['mined_executed_joint_moves'] == NL
    assert r['trigger']['ordinary'] == NL and r['trigger']['tapes'] == '0/3'
    errors = {e['id']: e['error'] for e in r['tape_errors']}
    assert 'digest mismatch' in errors['u1_R_corrupt'] and 'forbidden' in errors['u1_R_x_inc1']
    # (4) G4 sign convention: positive = group beat best solo
    assert b['g4']['receipt']['count'] == 2 and b['g4']['receipt']['sum'] == 6.0
    assert b['g4']['receipt']['sign'] == '+' and b['g4']['receipt']['n_neg'] == 1
    assert b['g4']['envelope']['sum'] == 6.0 and b['g4']['mined']['sum'] == 5.0
    assert b['g4']['mined']['used_in_ppo'] == [False] and b['g4']['plan']['starts'] == 2
    assert r['g4']['receipt'] == NL and r['g4']['mined'] == NL
    assert (r['g4']['envelope']['count'], r['g4']['envelope']['sum'], r['g4']['envelope']['sign']) == (1, -7.0, '-')
    assert report['g4_plan']['starts'] == dict(B=2, R=1)
    # (5) practice excluded; paired by seed; positive = more delay than A = worse
    td = b['team_delay']
    assert (td['mean_veh_s'], td['a_mean_veh_s'], td['diff_vs_a_veh_s']) == (160.0, 150.0, 10.0)
    assert td['paired_n'] == 2 and td['practice_excluded'] == 1
    assert r['team_delay']['per_seed_diff_veh_s'] == {'1': 20.0, '2': 60.0} and r['team_delay']['unpaired_n'] == 1
    line = report['line']
    for text in ('CLEAN4 diag u1 ', 'lane_drop_initiate:closing_pair:1/1', 'wall_join:1/1',
                 'lane_drop_join:abreast_wall:1/1', 'lesson=0.500(1/2)', 'delay-A=10 veh-s',
                 'sign=+', 'sign=-', 'wall_initiate:1/not logged'):
        assert text in line, text
    assert snapshot(root) == before


def test_forbidden_root_and_incomplete_update(tmp_path, synthetic_n2_root):
    bad = tmp_path/'holdout_run'
    bad.mkdir()
    with pytest.raises(cd.ForbiddenPath):
        cd.diagnose_update(bad, 1)
    (synthetic_n2_root/'R_UPDATE_001.json').rename(tmp_path/'R_moved.json')
    report = cd.diagnose_update(synthetic_n2_root, 1, tapes=False)
    assert not report['complete'] and 'INCOMPLETE receipts=AB' in report['line']
    assert report['arms']['B']['trigger']['tapes'] == NL
    status = tmp_path/'s.log'
    assert cd.main([str(synthetic_n2_root), '--status-file', str(status), '--no-tapes']) == 0
    assert not status.exists()   # incomplete updates are never appended or marked reported
