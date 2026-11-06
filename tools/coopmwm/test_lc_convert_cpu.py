"""LCEXEC-CV: universal verified-row -> free_program converter, TraCI parity over ALL verified rows, labels (CPU).

* fixtures: 9 verified row files (gz copies) + verbatim probe code, every byte sha256-pinned;
* conversion: every row of every source converts (0 NotRepresentable), registered, round-trips, never a
  legacy projection; dithers excluded; refusals carry a reason;
* parity: per row the log-consistent stream reproduces the row's OWN log on the verbatim probe Arm
  (log_check) and the probe's TraCI writes == lc_program executors + the real LCEXEC env writer, plus
  3 seeded random streams per row; the only allowed differences are the documented writer receipts
  (``writer_guard`` = edge-end deferral, ``writer_refused`` = non-adjacent request), listed explicitly;
* labels: lcexec-tick-label-v1 per member per control second of the log-consistent run.
"""
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools.coopmwm import lc_convert as cv, lc_parity as par, lc_program as lp, moves

HERE = Path(__file__).resolve().parent
FIX = HERE/'fixtures'/'lcexec'
ROW_COUNTS = dict(NEWMAP_MOVES_B=27, NEWMAP_MOVES_R=3, PLANGEN_MOVES_B=18, PLANGEN_MOVES_R=15, PLANGEN2_MOVES_B=24,
                  PLANGEN2_MOVES_R=18, COOPKINDS_MOVES_B=18, LOWCAV_MOVES_B=12, LCMOVES_MOVES_B=3)
PROBE_SHA = {                                   # sha256 of the probe sources (runs/active/C4-V7FT-20261017)
    'newmap2/nm2_core.pysrc': 'ad6b4bd63426ca41f6ebf8b974db48eea508532f076981f652139de15fa20abb',
    'plangen/nm2_core.pysrc': 'ad6b4bd63426ca41f6ebf8b974db48eea508532f076981f652139de15fa20abb',
    'lowcav/lc_core.pysrc': '287a53b3d3d8d083228072203a835e7aeca3d8610ae975bb37aec9973a7ce6b2',
    'plangen2/pg2_core.pysrc': 'fb00fe48f01dff96ef777fe053ff739f57873d936016e6c1b6f926c59b2960f9',
    'coopkinds/ck_core.pysrc': '003311cb5debc09d2cf86d0b049854e2e967f2b85c46ae39b990958d1eb27a02',
    'lcmoves/lcm_core.pysrc': '48aaae384210e1bb97e9a2c028761e01ba7fd6d4c632d70946b007319cc330db',
    'lcmoves/lcm_families.pysrc': '4df417f5f33f506011a3a1abdf1cfd5d29df96bb9521bcbcf6013343fe8d2c77',
    'lcmoves/lcm_plan.pysrc': '94ebbc933cdeceeb5e433574d52f1a3f0c192fccb0f61b2273bf7fe9cfedfb90',
}
POPULATION_SHA = '6af5ee232dd99f5ae0c142fdaf1b7de50f23f04d58f70def61482b76f4b2e6c6'
# Known writer differences on the LOG streams (lc_runtime edge-end deferral, v*2 + 10 m of the closing
# lane end): COOPKINDS zipper zip_merge changeLane(1) in its last seconds before the fallback.
LOG_WRITER_GUARD = {
    'ckv_ck_D2800_p60_s989400_t880_B_zipper_d0190fb5#r1': [(42, 'main.563'), (43, 'main.563')],
    'ckv_ck_D2800_p60_s989400_t880_B_zipper_d0190fb5#r2': [(42, 'main.563'), (43, 'main.563')],
    'ckv_ck_D2800_p60_s989400_t880_B_zipper_d0190fb5#r3': [(43, 'main.563'), (44, 'main.563')],
}
WRITER_CLASSES = {'writer_guard': 'deferred_edge_end', 'writer_refused': 'refused_not_allowed'}


@pytest.fixture(scope='module')
def desc():
    return par._wz_desc()


@pytest.fixture(scope='module')
def frame(desc):
    return lp.frame_from_descriptor(desc)


# ------------------------------------------------------------------ fixtures
def test_fixture_bytes_match_the_manifest_and_the_probe_sources():
    man = par.manifest()
    assert man['schema'] == 'lcexec-cv-fixtures-v1'
    assert {k: v['rows'] for k, v in man['rows'].items()} == ROW_COUNTS
    assert set(man['empty_rows']) == {'COOPKINDS_MOVES_R', 'LCMOVES_MOVES_R'}
    assert all(v['bytes'] == 0 for v in man['empty_rows'].values())
    for name, meta in man['rows'].items():
        buf = (FIX/'rows'/f'{name}.jsonl.gz').read_bytes()
        assert hashlib.sha256(buf).hexdigest() == meta['gz_sha256']
        assert hashlib.sha256(gzip.decompress(buf)).hexdigest() == meta['sha256']
        assert len(par.fixture_rows(name)) == ROW_COUNTS[name]
    for rel, meta in man['probes'].items():
        raw = (FIX/'probes'/rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta['sha256']
        want = POPULATION_SHA if rel.endswith('population.pysrc') else PROBE_SHA[rel]
        assert meta['sha256'] == want, rel
    assert set(PROBE_SHA) <= set(man['probes'])
    # N9's fixture copies are the same verified bytes
    for name in ('COOPKINDS_MOVES_B', 'PLANGEN2_MOVES_B', 'PLANGEN2_MOVES_R'):
        n9 = (HERE/'fixtures'/'n9_probe_rows'/f'{name}.jsonl').read_bytes()
        assert hashlib.sha256(n9).hexdigest() == man['rows'][name]['sha256']


@pytest.mark.parametrize('source', sorted(par.CORES))
def test_probe_code_loads_verbatim_without_sumo(source):
    before = set(sys.modules)
    p = par.probe(source)
    assert p.Arm.__name__ == 'Arm' and callable(p.positions) and p.X_B == 2696.0
    assert (p.families is not None) == (source == 'lcmoves')
    new = set(sys.modules) - before
    assert not {m for m in new if m.split('.')[0] in ('traci', 'libsumo', 'sumolib')}
    assert 'population' not in new and 'lcm_core' not in new          # namespaced, nothing leaked
    assert str(FIX) not in ' '.join(sys.path)


# ------------------------------------------------------------------ conversion
def test_every_row_of_every_source_converts_exactly(desc, frame):
    counts, paths = {}, {}
    for name, idx, row in par.all_rows():
        program = cv.from_mined_row(row, desc, file=f'{name}.jsonl')
        assert isinstance(program, lp.FreeProgram) and lp.check_registered(program) == []
        assert list(program.members) == row['members'] and program.join_el == (0,)*len(row['members'])
        assert program.duration_s == row['duration_s'] and program.map == 'wz3to2'
        assert lp.FreeProgram.from_dict(json.loads(json.dumps(program.to_dict()))) == program
        src = program.source
        for k in ('file', 'arm', 'verification_id', 'plan_sha256', 'family', 'variant', 'r_kind', 't0', 'world_seed'):
            assert k in src
        assert src['verification_id'] == row['verification_id'] and src['t0'] == row['t0']
        assert src['world_seed'] == row['world']['seed'] and src['file'] == f'{name}.jsonl'
        roles = [m.get('role') or 'hold' for m in cv.member_specs(row)]
        assert [p.role for p in program.programs] == roles
        dither = [d['id'] for d in row.get('dither') or []]
        assert src['dither_excluded'] == dither and not set(dither) & set(program.members)
        counts[name] = counts.get(name, 0)+1
        paths[src['path']] = paths.get(src['path'], 0)+1
    assert counts == ROW_COUNTS
    assert paths == {'lc_convert_probe': 36, 'lc_roles': 138-36}


def test_zero_legacy_projection(desc, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('moves.from_plan must never be used by the converter')
    monkeypatch.setattr(moves, 'from_plan', forbidden)
    for name, _, row in par.all_rows():
        cv.from_mined_row(row, desc, file=name)
    src = Path(cv.__file__).read_text(encoding='utf-8')
    assert 'moves' not in cv.__dict__ and 'import moves' not in src and 'from .moves' not in src


def _row(name, idx=0):
    return copy.deepcopy(par.fixture_rows(name)[idx][1])


@pytest.mark.parametrize('mutate,match', [
    (lambda r: r.update(schema='other'), 'mined-move'),
    (lambda r: r['move'].update(map='merge'), 'map'),
    (lambda r: r['move']['members'][0].update(role='teleport'), 'role'),
    (lambda r: r['move']['members'].pop(), 'member spec'),
    (lambda r: r.update(duration_s=0), 'duration'),
    (lambda r: r['move']['members'][0]['params'].pop('v_w'), 'member spec'),
    (lambda r: r.update(verification_id='zz_unknown'), 'source'),
])
def test_refusals_carry_a_reason(desc, mutate, match):
    row = _row('NEWMAP_MOVES_B')
    mutate(row)
    with pytest.raises(cv.NotRepresentable, match=match) as exc:
        cv.from_mined_row(row, desc)
    assert exc.value.reason


def test_refusals_for_lane_rows_and_coopkinds_hold(desc):
    ck = _row('COOPKINDS_MOVES_B')
    move = json.loads(ck['move'])
    move['members'][0]['role'] = 'hold'
    ck['move'] = json.dumps(move)
    with pytest.raises(cv.NotRepresentable, match='ck_core hold'):
        cv.from_mined_row(ck, desc)
    lb = next(r for _, _, r in par.all_rows() if r['family'] == 'lane_balance' and r['arm'] == 'B'
              and r['verification_id'].startswith('ckv'))
    lb = copy.deepcopy(lb)
    lb['commands'][lb['members'][0]][0][2] = 1                 # changeLane to a lane it never targets
    with pytest.raises(cv.NotRepresentable, match='lane-family row') as exc:
        cv.from_mined_row(lb, desc)
    assert 'changeLane' in exc.value.reason
    with pytest.raises(cv.NotRepresentable, match='differs'):
        cv.from_mined_row(_row('NEWMAP_MOVES_B'), desc, file='LCMOVES_MOVES_B.jsonl')


def test_row_meta_identity_without_outcomes():
    row = _row('LCMOVES_MOVES_B', 1)
    meta = cv.row_meta(row, 'x/LCMOVES_MOVES_B.jsonl.gz')
    assert meta['key'] == f"{row['verification_id']}#r2" and meta['probe'] == 'lcmoves'
    assert meta['file'] == 'LCMOVES_MOVES_B.jsonl.gz' and meta['dither_excluded'] == ['main.198']
    assert not {'synergy_veh_s', 'advantage_veh_s', 'profile', 'safety'} & set(meta)


# ------------------------------------------------------------------ comparison unit
def test_normalisations_and_classification():
    probe = [(0, 'c', 'slowDown', 10.0), (1, 'c', 'setSpeed', -1.0), (1, 'c', 'setLaneChangeMode', 1621),
             (3, 'd', 'setSpeed', -1.0), (4, 'e', 'slowDown', 5.0), (6, 'f', 'changeLane', (1, 2.0))]
    ours = [(0, 'c', 'slowDown', 10.0+1e-12), (1, 'c', 'setSpeed', -1.0), (1, 'c', 'setLaneChangeMode', 1621),
            (4, 'e', 'slowDown', 5.0), (5, 'e', 'setSpeed', -1.0)]
    receipts = {6: {'f': dict(status='deferred_edge_end')}}
    out = par.compare(probe, ours, receipts)
    # N1: d never slowed (dropped); N2: e handed back at 5 == probe slowDown expiry; f: writer guard
    assert out == [dict(el=6, vid='f', kind='changeLane', probe=[1, 2.0], ours=None, cls='writer_guard',
                        reason='env edge-end deferral (v*duration + 10 m of the lane end)')]
    out = par.compare(probe[:3], ours[:2])
    assert [(m['kind'], m['cls']) for m in out] == [('setLaneChangeMode', 'mismatch')]
    out = par.compare([(0, 'c', 'slowDown', 10.0)], [(0, 'c', 'slowDown', 10.001)])
    assert len(out) == 2 and {m['cls'] for m in out} == {'mismatch'}
    # N2 never hides a probe command at that second
    out = par.compare([(0, 'c', 'slowDown', 1.0), (1, 'c', 'slowDown', 2.0)], [(0, 'c', 'slowDown', 1.0),
                                                                              (1, 'c', 'setSpeed', -1.0)])
    assert sorted(m['kind'] for m in out) == ['setSpeed', 'slowDown']


# ------------------------------------------------------------------ parity over ALL rows
def _check_runs(key, log, randoms):
    lc = log['log_check']
    assert lc['ok'], (key, lc['diffs'][:3])
    assert lc['accel_max_err'] <= par.A_TOL
    plain = [m for m in log['mismatches'] if m['cls'] == 'mismatch']
    assert plain == [], (key, plain[:3])
    guard = [(m['el'], m['vid']) for m in log['mismatches'] if m['cls'] == 'writer_guard']
    assert guard == LOG_WRITER_GUARD.get(key, []), (key, guard)
    assert all(m['cls'] == 'writer_guard' and m['kind'] == 'changeLane' for m in log['mismatches'])
    assert len(randoms) >= 3
    for r in randoms:
        bad = [m for m in r['mismatches'] if m['cls'] == 'mismatch']
        assert bad == [], (key, r['seed'], bad[:3])
        for m in r['mismatches']:
            assert m['kind'] == 'changeLane' and m['ours'] is None and m['cls'] in WRITER_CLASSES


@pytest.mark.parametrize('name', sorted(ROW_COUNTS))
def test_parity_log_and_random_streams_for_every_row(name, desc, frame):
    for idx, row in par.fixture_rows(name):
        program = cv.from_mined_row(row, desc, file=f'{name}.jsonl')
        key = cv.row_key(row)
        log = par.run(row, program, frame=frame, stream='log', desc=desc, file=f'{name}.jsonl')
        assert log['n_ticks'] == row['duration_s']+1 and log['probe_calls']
        randoms = [par.run(row, program, frame=frame, stream='random', seed=s, desc=desc, file=f'{name}.jsonl')
                   for s in range(3)]
        _check_runs(key, log, randoms)
        if name == 'LCMOVES_MOVES_B':                        # persistent lane hold: ours re-issues it
            assert log['receipts_count'].get('held', 0) > 0
            assert log['passed']


def test_dithers_are_excluded_from_both_sides(desc, frame):
    row = _row('LCMOVES_MOVES_B', 1)
    program = cv.from_mined_row(row, desc)
    res = par.run(row, program, frame=frame, stream='log', desc=desc)
    vids = {c[1] for c in res['probe_calls']} | {c[1] for c in res['our_calls']}
    assert 'main.198' not in vids and vids <= set(row['members'])
    assert res['passed'] and res['log_check']['ok']


def test_parity_detects_a_wrong_program(desc, frame):
    """Non-vacuity: a perturbed member program (v_w) is caught by the log-stream parity."""
    import dataclasses
    row = _row('NEWMAP_MOVES_B')
    program = cv.from_mined_row(row, desc)
    p0 = program.programs[0]
    bad = dataclasses.replace(p0, lon=dataclasses.replace(p0.lon, v_w=p0.lon.v_w-1.0))
    wrong = dataclasses.replace(program, programs=(bad,)+program.programs[1:])
    res = par.run(row, wrong, frame=frame, stream='log', desc=desc)
    assert not res['passed'] and {m['cls'] for m in res['mismatches']} == {'mismatch'}
    assert {m['kind'] for m in res['mismatches']} == {'slowDown'}


# ------------------------------------------------------------------ labels
LABEL_KEYS = {'schema', 'vid', 't', 'el', 'session_id', 'role', 'lon', 'accel_mps2', 'lane_target', 'lane_duration_s',
              'lc_mode', 'tau', 'event', 'release_reason', 'source'}


@pytest.mark.parametrize('name,idx', [('NEWMAP_MOVES_B', 0), ('COOPKINDS_MOVES_B', 3), ('LCMOVES_MOVES_B', 1),
                                      ('PLANGEN2_MOVES_B', 0), ('PLANGEN2_MOVES_R', 15), ('LOWCAV_MOVES_B', 0)])
def test_labels_schema_and_consistency_with_the_row(desc, frame, name, idx):
    row = par.fixture_rows(name)[idx][1]
    labels = cv.labels(row, desc, file=f'{name}.jsonl')
    assert labels and all(set(lab) == LABEL_KEYS and lab['schema'] == lp.LABEL_SCHEMA for lab in labels)
    assert {lab['session_id'] for lab in labels} == {row['verification_id']}
    assert {lab['source'] for lab in labels} == {f'{name}.jsonl'} and {lab['vid'] for lab in labels} == set(row['members'])
    for vid in row['members']:
        mine = [lab for lab in labels if lab['vid'] == vid]
        assert [lab['el'] for lab in mine] == list(range(len(mine)))
        assert all(lab['t'] == row['t0']+lab['el'] for lab in mine)
        rel = row['release_el'].get(vid)
        releases = [lab['el'] for lab in mine if lab['event'] == 'release']
        assert releases == ([] if rel is None else [rel])
        assert mine[-1]['el'] == (row['duration_s'] if rel is None else rel)
        assert all(lab['lon'] in ('accel', 'native', 'base') for lab in mine)
    stream = par.build_stream(row, cv.from_mined_row(row, desc), frame=frame, desc=desc)
    for vid, acc in stream.expected_accel.items():
        got = {lab['el']: lab['accel_mps2'] for lab in labels
               if lab['vid'] == vid and lab['lon'] == 'accel' and lab['el'] < row['duration_s']}
        assert set(got) == set(acc) and all(abs(got[el]-a) <= par.A_TOL for el, a in acc.items())
    wm_cmds = {(el, vid) for el, vid, kind, _ in par._row_commands(row, set(row['members'])) if kind == 'changeLane'}
    lane_lab = {(lab['el'], lab['vid']) for lab in labels if lab['lane_target'] >= 0}
    assert wm_cmds <= lane_lab


# ------------------------------------------------------------------ CLI + report
def test_cli_convert_and_labels(tmp_path):
    src = FIX/'rows'/'LCMOVES_MOVES_B.jsonl.gz'
    rows = tmp_path/'LCMOVES_MOVES_B.jsonl.gz'
    rows.write_bytes(src.read_bytes())
    assert cv.main(['convert', str(rows), str(tmp_path/'conv.jsonl')]) == 0
    recs = [json.loads(x) for x in (tmp_path/'conv.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(recs) == 3 and all(r['status'] == 'converted' and r['registered_violations'] == [] for r in recs)
    assert lp.FreeProgram.from_dict(recs[0]['program']).members == tuple(recs[0]['meta']['members'])
    assert cv.main(['labels', str(rows), str(tmp_path/'lab.jsonl')]) == 0
    labs = [json.loads(x) for x in (tmp_path/'lab.jsonl').read_text(encoding='utf-8').splitlines()]
    assert labs and {lab['schema'] for lab in labs} == {lp.LABEL_SCHEMA}


def test_committed_parity_report_is_for_these_fixtures():
    rep = json.loads((HERE.parents[1]/'LCEXEC_PARITY.json').read_text(encoding='utf-8'))
    man = par.manifest()
    assert rep['schema'] == 'lcexec-parity-report-v1' and rep['n_random'] >= 3
    assert rep['fixtures']['rows'] == {k: v['sha256'] for k, v in man['rows'].items()}
    assert rep['fixtures']['probes'] == {k: v['sha256'] for k, v in man['probes'].items()}
    t = rep['totals']
    assert t['rows'] == t['converted'] == t['log_check_pass'] == sum(ROW_COUNTS.values())
    assert rep['n_other_mismatches'] == 0
    assert t['parity_log_pass']+t['parity_log_writer_only'] == t['rows']
    assert t['parity_log_writer_only'] == len(LOG_WRITER_GUARD)
    assert t['parity_random_pass']+t['parity_random_writer_only'] == t['rows']
