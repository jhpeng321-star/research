"""Recovery invariants against the actual interrupted ledger; no SUMO/GPU."""
import json
from pathlib import Path

import pytest

from tools.c1switch.resume import (DROPPED, TOTAL, amended_plan, imported_rows,
                                   reusable_branch, tasks, verify_pick)


@pytest.fixture
def audit():
    return json.loads((Path(__file__).resolve().parents[1]/'tools/c1switch/evidence/V7_INTERRUPTION_AUDIT.json').read_text())


def test_exact_amended_panel(audit):
    p = amended_plan(audit)
    assert p['projected_total_sim_s'] == TOTAL == 499260
    assert p['per_arm']['random']['projected_total_sim_s'] == 11700
    assert len(p['complete_jobs']) == 241
    assert len(tasks()) == 125
    assert [w['seed'] for a,w in tasks() if a=='random'] == list(range(969112,969125))
    assert DROPPED == (969125,969126,969127)
    assert p['prior_gpu_s']+p['remaining_gpu_s'] == 14400
    assert not p['dispatch_authorized']


def test_partial_debits_preserved_without_identifier_collision(audit):
    rows = imported_rows(audit)
    assert sum(r['sim_s'] for r in rows) == 227310
    assert sum(r['sim_s'] for r in rows if '_abandoned_v7_' in r['id']) == 11100
    assert len({r['id'] for r in rows}) == len(rows)
    assert sum(r['sim_s'] for r in rows if r['id'].startswith('wm_')) == 17550
    assert sum(r['sim_s'] for r in rows if r['id'].startswith('bb_')) == 150300


def test_no_trim_after_random_started(audit):
    # Admission detects unexpected accounting before proposing any new compute.
    audit['cpuQ']['accounting'].append(dict(id='random_s969127_episode_x',sim_s=1))
    with pytest.raises(ValueError):
        amended_plan(audit)


def test_pick_replay_requires_history_prefix_costs_and_choice():
    row = dict(j=0,pick='native',incumbent='native',belief_sha256='a',prefix={'t':300},predicted_costs={'native':1})
    verify_pick(row,dict(row))
    for k in row:
        with pytest.raises(ValueError,match='committed pick mismatch'):
            verify_pick(row,dict(row,**{k:None}))


def test_reused_branch_must_match_live_state():
    row = dict(prefix={'t':300,'tape':'abc'},end={},**{'60':{},'90':{}})
    assert reusable_branch(row,dict(prefix=row['prefix'])) is row
    with pytest.raises(ValueError,match='prefix'):
        reusable_branch(row,dict(prefix={'t':300,'tape':'different'}))


def test_obsolete_owner_cannot_admit(audit):
    audit['cpuQ']['own_pids']['123'] = True
    with pytest.raises(ValueError,match='alive'):
        amended_plan(audit)


def test_deferred_truth_equals_immediate_toy_and_reuses_success(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tests.test_c1switch import ReplayToy
    from tools.c1switch import resume
    from tools.c1switch.design import CONTROLLERS, worlds
    from tools.c1switch.episode import episode, measure
    from tools.c1switch.storage import write
    world = worlds('main')[0]
    predictor = SimpleNamespace(predict=lambda *args:{c:float(c!='pi') for c in CONTROLLERS})
    immediate = episode(ReplayToy(),'wm',world,predictor=predictor,
        measurement=lambda schedule,prefix:measure(lambda c:ReplayToy,schedule,prefix))
    for d in immediate['decisions']:
        if 'branches' in d:
            for b in d['branches'].values():
                b.pop('belief')  # Real main branch output excludes inference payload.
            c = 'native'
            write(tmp_path/'jobs'/f'wm_s{world["seed"]}_j{d["j"]}_{c}'/'RESULT.json',d['branches'][c])
    made = []
    def factory(world,arm,ident,inputs):
        (tmp_path/'jobs'/ident).mkdir()
        b = ReplayToy();made.append(b)
        return b
    monkeypatch.setattr(resume,'ROOT',tmp_path)
    monkeypatch.setattr(resume,'backend',factory)
    monkeypatch.setattr(resume,'Predictor',lambda arm:predictor)
    resume.run_episode(world,'wm',None)
    specs = list((tmp_path/'specs').glob('*.json'))
    assert len(specs)==12  # Three native branches already succeeded: no repeats.
    assert not (tmp_path/'jobs'/f'wm_s{world["seed"]}_episode'/'RESULT.json').exists()
    for p in specs:
        resume.run_branch(json.loads(p.read_text()),None)
    resume.finalize('wm',world)
    got = json.loads((tmp_path/'jobs'/f'wm_s{world["seed"]}_episode'/'RESULT.json').read_text())
    assert got == immediate
    assert len(made)==13 and all(b.closed for b in made)
