"""harden-2: MAIN decisions (a)-(d) after the 19:xx review (CPU only; no SUMO/GPU/ssh).

(a) one failed/stopped/non-CRN exact grade -> per-grade receipt + alert + exclusion, no global STOP;
(b) one failed TRAIN episode -> receipt, never replayed, fresh charged replacement on the next
    unused TRAIN world of the same cell; STOP only on 3 consecutive or > 20 % of a wave;
(c) learned-search startup gate fails loudly (never READY-then-silently-decline);
(d) a budget-blocked SUMO verification always writes an alert.
"""
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.protocol import digest
from tools.coopmwm.test_sumo_verify_cpu import FakePlant, plan, snapshot


def grade(gid, status='graded', crn=True):
    return dict(grade_id=gid, status=status, reason=None if status == 'graded' else 'sumo crashed',
                noise=dict(crn_verified_t0=crn), receipt_sha256='h'+gid)


def test_a_failed_grade_gets_receipt_alert_and_exclusion_not_global_stop(tmp_path):
    from tools.coopmwm import exact_runtime as xr
    receipts = [grade('ok'), grade('f', 'failed'), grade('s', 'stopped'), grade('n', crn=False)]
    for r in receipts:
        if xr.failed_grade(r):
            assert xr.record_failed_grade(tmp_path, r)
    assert not xr.record_failed_grade(tmp_path, receipts[1])          # idempotent, never replayed
    assert [r['grade_id'] for r in xr.usable(tmp_path, receipts)] == ['ok']
    v = json.loads((tmp_path/'failed_grades'/'f.json').read_text())
    assert v['excluded_from_summaries'] and v['no_imputation'] and not v['global_stop']
    assert sorted(p.stem for p in (tmp_path/'grade_alerts').iterdir()) == ['f', 'n', 's']
    assert not list(tmp_path.glob('*STOP*'))
    src = Path(xr.__file__).read_text()
    assert 'exact-grade engineering fault' not in src


def test_a_unrecorded_failure_is_still_excluded_from_evidence(tmp_path):
    from tools.coopmwm import exact_runtime as xr
    assert [r['grade_id'] for r in xr.usable(tmp_path, [grade('x', 'failed'), grade('y')])] == ['y']


def train_pool():
    return [dict(seed=986000+6*c+r, cell=c, repeat=r) for c in range(4) for r in range(6)]


def run_wave(tmp_path, seeds, fail):
    from tools.coopmwm import accel_runner as ar
    by = {w['seed']: w for w in train_pool()}
    jobs = [dict(id=f'warm_u004_B_{s}', path=str(tmp_path/'jobs'/f'warm_u004_B_{s}'), world=by[s], arm='B',
                 noise_seed=int(digest(dict(seed=ar.SEED, world=s, wave='warm_u004'))[:15], 16)) for s in seeds]
    charged, submitted = [], []

    def episode(job):
        submitted.append(job)
        if job['world']['seed'] in fail:
            raise fail[job['world']['seed']]
        return dict(id=job['id'], world=job['world'], arm='B')
    with patch.object(ar, 'worlds', lambda role: train_pool()), \
            patch.object(ar, 'reserve', lambda root, js, phase: charged.extend(j['id'] for j in js)), \
            ThreadPoolExecutor(max_workers=1) as pool:
        out = ar.run_train_jobs(tmp_path, jobs, 'train', 'warm_u004', {j['world']['seed'] for j in jobs},
                                set(), pool.submit, episode)
    return out, charged, submitted


def test_b_one_failed_episode_is_closed_and_replaced_from_same_cell_fresh_seed(tmp_path):
    from tools.coopmwm import accel_runner as ar
    seeds = [986000+6*c+r for c in range(4) for r in (0, 1)]          # 8 jobs, cells 0-3
    out, charged, submitted = run_wave(tmp_path, seeds, {986001: RuntimeError('sumo segfault')})
    assert charged == ['warm_u004_B_986002']                          # next unused repeat, same cell, charged
    assert sorted(e['world']['seed'] for e in out) == sorted(set(seeds)-{986001}|{986002})
    assert [j['id'] for j in submitted].count('warm_u004_B_986001') == 1   # claim never replayed
    new = next(j for j in submitted if j['id'] == 'warm_u004_B_986002')
    assert new['replaces'] == 'warm_u004_B_986001'
    assert new['noise_seed'] == int(digest(dict(seed=ar.SEED, world=986002, wave='warm_u004'))[:15], 16)
    receipt = json.loads((tmp_path/'jobs'/'warm_u004_B_986001'/'EPISODE_FAILED.json').read_text())
    assert receipt['replacement'] == 'warm_u004_B_986002' and not receipt['claims_replayed']


def test_b_replacement_shortfall_is_recorded_when_cell_exhausted(tmp_path):
    seeds = [986000+r for r in range(6)]+[986006+r for r in range(6)]+[986012+r for r in range(6)]  # 18 jobs
    out, charged, _ = run_wave(tmp_path, seeds, {986003: RuntimeError('traci closed')})
    receipt = json.loads((tmp_path/'jobs'/'warm_u004_B_986003'/'EPISODE_FAILED.json').read_text())
    assert receipt['replacement_shortfall'] and charged == [] and len(out) == 17


def test_b_three_consecutive_failures_escalate(tmp_path):
    from tools.coopmwm import accel_runner as ar
    seeds = [986000+6*c+r for c in range(4) for r in range(4)]        # 16 jobs: 3 failures <= 20 %
    boom = RuntimeError('sumo segfault')
    with pytest.raises(ar.EpisodeFailureEscalation):
        run_wave(tmp_path, seeds, {986000: boom, 986001: boom, 986002: boom})


def test_b_more_than_20_percent_escalates(tmp_path):
    from tools.coopmwm import accel_runner as ar
    seeds = [986000, 986006, 986012, 986018, 986001]                  # 5 jobs; 2 failures = 40 %
    boom = RuntimeError('traci closed')
    with pytest.raises(ar.EpisodeFailureEscalation):
        run_wave(tmp_path, seeds, {986000: boom, 986012: boom})


def test_b_global_stop_inside_an_episode_still_propagates(tmp_path):
    with pytest.raises(ValueError, match='durable warm STOP'):
        run_wave(tmp_path, [986000, 986006], {986000: ValueError('durable warm STOP')})
    assert not (tmp_path/'jobs'/'warm_u004_B_986000'/'EPISODE_FAILED.json').exists()


def test_b_health_pause_propagates_without_replacement(tmp_path):
    from tools.coopmwm.warm_health import HealthPause
    with pytest.raises(HealthPause):
        run_wave(tmp_path, [986000, 986006], {986000: HealthPause('registered health pause PAUSE_SAFETY.json')})


def gate_fixture():
    rows = [dict(vid=v, s_m=1000.-20*i, v_mps=20., lane=0, is_cav=True) for i, v in enumerate(('a', 'b'))]
    record = dict(state_id='0'*64, world_id=986000, time_s=400, rows=rows, group=['a', 'b'],
                  commands={'a': 0., 'b': 0.}, policy_residuals={'a': 0., 'b': 0.})
    from tools.coopmwm.mppi_search import group_components
    return record, group_components(rows)[0]


def test_c_learned_startup_gate_fails_loudly_on_warm_start_type_error():
    from tools.coopmwm import mppi_search, plan_service as ps
    record, group = gate_fixture()
    def warm(r, g, pt, b, reg):
        return dict(config=dict(top_k=8), warm_start=dict(mean=[0.]))
    problems = ps.learned_gate_problems(mppi_search.search, warm, record, group, [[0.]*600]*2, (-3., 2.),
                                        'ordinary', None)
    if 'warm_start' not in inspect.signature(mppi_search.search).parameters:
        assert problems and 'TypeError' in problems[0]
    else:
        assert problems                                                 # malformed warm start still caught
    with pytest.raises(ValueError, match='startup gate failed'):
        ps.learned_gate(dict(learned_search=dict(checkpoint='x')), problems)
    ps.learned_gate(dict(), ['ignored without a learned binding'])


def test_c_learned_gate_passes_conformant_config_and_flags_small_top_k():
    from tools.coopmwm import mppi_search, plan_service as ps
    record, group = gate_fixture()
    def ok(r, g, pt, b, reg):
        return dict(config=dict(top_k=8))
    assert ps.learned_gate_problems(mppi_search.search, ok, record, group, [[0.]*600]*2, (-3., 2.),
                                    'ordinary', None) == []
    def plain(r, g, pt, b, reg):
        return dict()
    assert any('top_k=4' in p for p in ps.learned_gate_problems(
        mppi_search.search, plain, record, group, [[0.]*600]*2, (-3., 2.), 'ordinary', None))


def test_c_preflight_on_this_tree_blocks_learned_activation():
    problems = sv.resume_preflight(dict(k=8, workers=9, phase='sumo_verification_k8'),
                                   dict(top_k=8), learned=True)
    assert any('learned-search hook' in p for p in problems)            # mppi_runtime not integrated yet


def eight():
    return [plan(c, gain=g) for c, g in zip('abcdefgh', (1., 2., .5, -1., 3., 5., 0., 4.))]


def test_d_budget_blocked_verification_writes_alert(tmp_path):
    from tools.coopmwm.exact_grade import BudgetBlocked

    def charge(vid, specs):
        raise BudgetBlocked('phase sumo_verification_k8 cap')
    r = sv.verify(snapshot(), eight(), FakePlant(), charge, root=tmp_path)
    assert r['status'] == 'budget_blocked' and r['charged_sim_s'] == 0
    alerts = list((tmp_path/'alerts').glob('BUDGET_BLOCKED_*.json'))
    assert len(alerts) == 1 and 'sumo_verification_k8' in json.loads(alerts[0].read_text())['phase_reason']
