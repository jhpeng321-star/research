"""N1 sanity job: zero-physics plan and the registered decision rule (no SUMO)."""
import json
from pathlib import Path

import pytest

from tools.coopmwm import map_descriptor as md, newmap_sanity as ns

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO/'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'


@pytest.fixture
def wz(monkeypatch):
    md._reset_for_tests()
    monkeypatch.setenv(md.ENV, str(SPEC))
    yield
    md._reset_for_tests()


def test_plan_registers_paired_train_worlds_budget_and_rule(tmp_path, wz):
    body = ns.plan(tmp_path/'r', runtime='/home/ubuntu/e3final', checkpoint='/x/policy_00169.pt', commit='abc')
    assert [w['seed'] for w in body['worlds']] == [989596, 989597, 989598, 989599]
    assert all(w['map'] == 'wz3to2' for w in body['worlds']) and body['arms'] == ['mappo', 'native']
    assert body['budget'] == dict(jobs=8, sim_s_per_job=1200, total_sim_s=9600)
    assert body['episode'] == dict(warmup_s=300, horizon_s=901, stop_s=1200, control_window_s=[300, 1200])
    assert json.loads((tmp_path/'r'/'SANITY_PLAN.json').read_text())['plan_sha256'] == body['plan_sha256']
    with pytest.raises(ValueError):
        ns.plan(tmp_path/'r', runtime='r', checkpoint='c')                 # never overwrite
    b3 = ns.plan(tmp_path/'u', runtime='r', checkpoint='c', u0='/x/A_checkpoint_000.pt')
    assert b3['budget']['total_sim_s'] == 3*4*1200 and 'clean3_u0' in b3['arms']


def test_plan_refuses_a_merge_process(tmp_path, monkeypatch):
    md._reset_for_tests()
    monkeypatch.delenv(md.ENV, raising=False)
    try:
        with pytest.raises(ValueError):
            ns.plan(tmp_path/'r', runtime='r', checkpoint='c')
    finally:
        md._reset_for_tests()


def _results(root, body, mappo, native):
    for job in body['jobs']:
        if job['arm'] not in ('mappo', 'native'):
            continue
        totals = dict(mappo if job['arm'] == 'mappo' else native)
        p = Path(job['path'])
        p.mkdir(parents=True)
        (p/'RESULT.json').write_text(json.dumps(dict(totals=totals, max_present=200, actual_sim_s=1200.)))


def test_decision_rule(tmp_path, wz):
    ok = dict(team_delay_vs=1000., collisions=0., teleports=0., dropped_vehicle_steps=0., ttc_below_3_vs=5., hard_brake=5.)
    nat = dict(team_delay_vs=980., collisions=0., teleports=0., dropped_vehicle_steps=0., ttc_below_3_vs=4., hard_brake=4.)
    body = ns.plan(tmp_path/'a', runtime='r', checkpoint='c')
    _results(tmp_path/'a', body, ok, nat)
    s = ns.summarize(tmp_path/'a')
    assert s['decision']['base'] == 'mappo_u0' and all(s['decision']['checks'].values())
    body = ns.plan(tmp_path/'b', runtime='r', checkpoint='c')
    _results(tmp_path/'b', body, dict(ok, team_delay_vs=1100.), nat)         # +12% delay
    assert ns.summarize(tmp_path/'b')['decision']['base'] == 'native'
    body = ns.plan(tmp_path/'c', runtime='r', checkpoint='c')
    _results(tmp_path/'c', body, dict(ok, collisions=1.), nat)
    assert ns.summarize(tmp_path/'c')['decision']['checks']['safety'] is False
    body = ns.plan(tmp_path/'d', runtime='r', checkpoint='c')
    _results(tmp_path/'d', body, ok, nat)
    import shutil
    shutil.rmtree(Path(body['jobs'][0]['path']))                             # one mappo episode missing
    assert ns.summarize(tmp_path/'d')['decision']['base'] == 'native'
