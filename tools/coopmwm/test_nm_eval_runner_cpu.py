"""CPU-only tests for tools.coopmwm.nm_eval_runner (fake plants; no SUMO, no frozen runtime)."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from tools.coopmwm import map_descriptor as md, map_scenarios as ms, nm_eval_runner as runner

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'ops'))
import nm_finaleval as nm  # noqa: E402

SPEC = REPO/nm.MAP_SPEC_REL


@pytest.fixture
def bound(monkeypatch):
    """The N1 process map = wz3to2 (COOPMWM_MAP_SPEC), reset afterwards."""
    monkeypatch.setenv(md.ENV, str(SPEC))
    md._reset_for_tests()
    yield
    md._reset_for_tests()


@pytest.fixture
def env(tmp_path, bound, monkeypatch):
    panel = nm.load_panel()
    root, cfg = nm.build_fake_root(tmp_path/'dry', panel, 'mappo_u0')
    cfg = dict(cfg, checkpoint='/frozen/mappo_d9042cc5.pt', warm_history=True, accel_bounds=[-3.0, 1.5])
    from tools.coopmwm import launch_production as production
    monkeypatch.setattr(production, 'backend_config', lambda c: dict(c, translated=True))
    monkeypatch.setattr(production, 'checkpoint_weights', lambda c, arm, update: {'arm': arm, 'update': update})
    monkeypatch.setattr(production, 'load_policy', lambda weights, c: object())
    from tools.coopmwm import replay_bank
    monkeypatch.setattr(replay_bank, 'ttc_events', lambda trace, start, end: 2)
    return dict(panel=panel, root=root, cfg=cfg, tmp=tmp_path)


class FakeController:
    def __init__(self):
        self.events, self.active_plans = [], []

    def apply(self, rows, commands, now):
        return None


def fake_plant(calls, moves_at):
    """accel_plant.episode stand-in: records the job, drives the hook with track2 move samples."""
    def episode(job, *, controller_hook=None, policy_loader=None):
        calls.append(job)
        policy_loader(job['weights'])
        controller = FakeController()
        controller_hook(controller, job, {})
        active = []
        for t in range(300, job['stop']):
            for start, family, members in moves_at:
                if start == t:
                    active.append(dict(move=dict(family=family, members=members, onset=0, hold=60, depth=1.0),
                                       start=t, until=t+60))
            active = [a for a in active if a['until'] > t]
            controller.events.append(dict(time_s=t, sample=dict(track2=dict(active_moves=deepcopy(active)))))
            controller_hook.after_step(t, [])
        return dict(totals=dict(team_delay_vs=54321.5, collisions=0, teleports=0, hard_brake=4))
    return episode


def test_runner_declares_the_contract():
    assert runner.MAP == 'wz3to2' and runner.NATIVE is True and callable(runner.episode)
    assert not hasattr(runner, 'decision_branch')            # per-decision branches not built: never faked
    assert nm.Runtime().runner_status() == dict(ok=True, native=True, decisions=False, reason=None)


def test_policy_episode_runs_the_registered_job_and_logs_decisions(env, monkeypatch):
    from tools.coopmwm import accel_plant
    calls = []
    moves = [(320, 'abreast_wall', ['main.1', 'main.2', 'main.3']), (500, 'closing_pair', ['main.4', 'main.5']),
             (600, 'wall', ['main.6', 'main.7']), (700, 'ease_off', ['main.8'])]
    monkeypatch.setattr(accel_plant, 'episode', fake_plant(calls, moves))
    specs = nm.episode_specs('B', env['panel'], env['cfg'], env['root'], 'mappo_u0')
    subset, other = specs[0], specs[1]
    record = runner.episode(subset, env['cfg'], env['tmp']/'work'/subset['job_id'])
    job = calls[-1]
    assert (job['role'], job['sim_s'], job['stop'], job['deterministic'], job['guided'], job['v2v']) == \
        ('final_test', 1200, 1200, True, False, True)
    assert job['world'] == ms.world(989600, 3200, 0.6) and job['noise_seed'] == 989600 and job['arm'] == 'B'
    assert job['checkpoint'] == '/frozen/mappo_d9042cc5.pt' and job['weights'] == {'arm': 'B', 'update': 24}
    assert record['J_veh_s'] == 54321.5 and record['safety'] == dict(collisions=0, teleports=0, hard_brakes=4, ttc_events=2)
    assert [r['t_s'] for r in record['decision_log']] == [320, 500, 600]
    assert record['decision']['t_s'] == 320 and record['decision']['move'] == 'abreast_wall'
    assert record['decision']['state_sha256'] is None
    checked = nm.check_episode(record, subset)                 # the launcher accepts it as is
    assert checked['decision_log'][1]['move'] == 'closing_pair'
    record = runner.episode(other, env['cfg'], env['tmp']/'work'/other['job_id'])
    assert 'decision' not in record and len(record['decision_log']) == 3
    nm.check_episode(record, other)
    a = nm.episode_specs('A', env['panel'], env['cfg'], env['root'], 'mappo_u0')[0]
    record = runner.episode(a, env['cfg'], env['tmp']/'work'/a['job_id'])
    assert 'decision_log' not in record and 'decision' not in record
    nm.check_episode(record, a)


def test_mappo_arm_swaps_only_the_executor_and_reverifies_it(env, monkeypatch):
    from tools.coopmwm import accel_plant
    calls = []
    monkeypatch.setattr(accel_plant, 'episode', fake_plant(calls, []))
    final_root = env['tmp']/'dry'/'final_root'
    nm.register(root=env['root'], final_root=final_root, runtime=nm.FakeRuntime(), panel=env['panel'])
    binding = nm.bind_fake_mappo(env['tmp']/'dry', final_root)
    spec = nm.episode_specs('MAPPO', env['panel'], env['cfg'], env['root'], 'mappo_u0', binding)[0]
    record = runner.episode(spec, env['cfg'], env['tmp']/'work'/spec['job_id'])
    assert calls[-1]['checkpoint'] == binding['path'] and calls[-1]['weights'] == {'arm': 'A', 'update': 0}
    assert calls[-1]['arm'] == 'A' and nm.check_episode(record, spec)['J_veh_s'] == 54321.5
    Path(binding['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='bound sha256'):
        runner.episode(spec, env['cfg'], env['tmp']/'work'/'again')
    assert len(calls) == 1


def test_refusals_happen_before_any_plant_call(env, monkeypatch):
    from tools.coopmwm import accel_plant
    calls = []
    monkeypatch.setattr(accel_plant, 'episode', fake_plant(calls, []))
    spec = nm.episode_specs('R', env['panel'], env['cfg'], env['root'], 'mappo_u0')[3]
    for mutate, match in ((lambda s: s.update(world=ms.world(989200, 3200, 0.6)), 'not allowed'),
                          (lambda s: s.update(window=dict(start_s=0, end_s=1800)), '1200'),
                          (lambda s: s.update(noise_seed=1), 'noise seed'),
                          (lambda s: s.update(deterministic=False), 'deterministic'),
                          (lambda s: s['checkpoint'].update(file_sha256='0'*64), 'changed after registration'),
                          (lambda s: s.update(arm='LANEON'), 'unknown evaluation arm')):
        bad = deepcopy(spec)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            runner.episode(bad, env['cfg'], env['tmp']/'work'/'x')
    assert not calls


def test_process_bound_to_another_map_is_refused(env, monkeypatch):
    monkeypatch.delenv(md.ENV)
    md._reset_for_tests()
    spec = nm.episode_specs('A', env['panel'], env['cfg'], env['root'], 'mappo_u0')[0]
    with pytest.raises(ValueError, match='process map'):
        runner.episode(spec, env['cfg'], env['tmp']/'work'/'x')


def fake_native_episode(job):
    """Stands in for tools.coopnative.plant.episode; its globals are replaced by the runner."""
    approval_line(job['approval_line'])  # noqa: F821
    setup(job['runtime'])  # noqa: F821
    guard(job['world'], job['role'])  # noqa: F821
    NATIVE_JOBS.append(job)
    return dict(metrics={'900': dict(team_delay_vs=61000.25, collisions=0, teleports=0, hard_brake=7)})


NATIVE_JOBS = []


def test_native_episode_uses_the_native_plant_with_the_map_binding(env, monkeypatch):
    from tools.coopnative import plant as native
    from tools.coopmwm import map_runtime, protocol
    installed = []
    monkeypatch.setattr(native, 'episode', fake_native_episode)
    monkeypatch.setattr(native, 'setup', lambda runtime: installed.append(('setup', runtime)))
    monkeypatch.setattr(protocol, 'install_seed_scope', lambda: installed.append(('seed_scope',)))
    monkeypatch.setattr(map_runtime, 'install', lambda job: installed.append(('map', job['world']['map'])) or 'desc')
    monkeypatch.setattr(map_runtime, 'patch_globals', lambda ns, job: installed.append(('rows',)))
    spec = nm.episode_specs('NATIVE', env['panel'], env['cfg'], env['root'], 'mappo_u0')[5]
    record = runner.episode(spec, env['cfg'], env['tmp']/'work'/spec['job_id'])
    job = NATIVE_JOBS[-1]
    assert job['arm'] == 'native' and job['role'] == 'final_test' and job['sim_s'] == job['stop'] == 1200
    assert job['world'] == spec['world'] and job['noise_seed'] == spec['world']['seed']
    assert [x[0] for x in installed] == ['setup', 'seed_scope', 'map', 'rows']
    assert record['J_veh_s'] == 61000.25 and record['safety'] == dict(collisions=0, teleports=0, hard_brakes=7, ttc_events=2)
    assert nm.check_episode(record, spec)['decision'] is None
    monkeypatch.setattr(map_runtime, 'install', lambda job: None)
    with pytest.raises(ValueError, match='map binding'):
        runner.episode(spec, env['cfg'], env['tmp']/'work'/'native2')
    with pytest.raises(ValueError, match='native checkpoint marker'):
        runner.episode(dict(spec, checkpoint=dict(kind='mappo')), env['cfg'], env['tmp']/'work'/'native3')


def test_mappo_load_check_hashes_then_loads(tmp_path):
    import torch
    path = tmp_path/'m.pt'
    torch.save(dict(format='c4_dreamer/3', policy={'actor.0.weight': torch.zeros(2, 33)}), path)
    sha = nm.file_sha(path)
    out = runner.mappo_load_check(path, sha)
    assert out['status'] == 'loaded' and out['format'] == 'c4_dreamer/3' and 'policy' in out['keys']
    with pytest.raises(ValueError, match='does not hash'):
        runner.mappo_load_check(path, '0'*64)
    assert json.dumps(out)
