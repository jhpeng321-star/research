"""CPU-only tests for tools.coopmwm.nm_eval_runner (fake plants; no SUMO, no frozen runtime)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from tools.coopmwm import map_descriptor as md, map_scenarios as ms, nm_eval_runner as runner

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'ops'))
import nm_finaleval as nm  # noqa: E402

SPEC = REPO/nm.MAP_SPEC_REL


def log_frame(ids=('main.0', 'main.1', 'main.2'), cav=(True, False, True)):
    import numpy as np
    n = len(ids)
    return types.SimpleNamespace(present=np.ones(n, bool), s=np.array([-900.0-50*i for i in range(n)]),
                                 v=np.full(n, 20.0), a=np.full(n, -0.5), lane=np.arange(n) % 3,
                                 lane_carry=np.arange(n) % 3, junction=np.zeros(n, bool), is_cav=np.array(cav),
                                 vid_index=np.arange(n), length_true=np.full(n, 4.3))


class LogEnv:
    """Read-only frame/builder surface the ground-truth vehicle log uses."""
    def __init__(self, t):
        self.now, self.frame, self.api = float(t), log_frame(), None
        self.builder = types.SimpleNamespace(vehicle_ids=['main.0', 'main.1', 'main.2'])


def plant_tick(t):
    """What the real plant does each control second: tools.coopfree.backend.tick(b, ...)."""
    from tools.coopfree import backend as cf
    cf.tick(types.SimpleNamespace(env=LogEnv(t)))


def observations(env):
    """Stand-in for tools.coopnative.plant.observations (the runner wraps it with the vehicle log)."""
    return []


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
    built = []
    monkeypatch.setattr(runner, 'eval_policy', lambda native, arm, update, seed: built.append((arm, update, seed)) or object())
    from tools.coopmwm import replay_bank
    from tools.coopfree import backend as cf
    monkeypatch.setattr(replay_bank, 'ttc_events', lambda trace, start, end: 2)
    monkeypatch.setattr(cf, 'tick', lambda b, controller=None, trace=None: [])     # the plant step (fake)
    return dict(panel=panel, root=root, cfg=cfg, tmp=tmp_path, built=built)


FAKE_MAPPO_SHA = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'


def _frozen_setup(runtime, checkpoint):
    """Pin structure of the frozen setup (globals sha / MAPPO_SHA / CALLS are injected)."""
    if sha(checkpoint) != MAPPO_SHA:  # noqa: F821
        raise ValueError('frozen source/checkpoint hash mismatch')
    CALLS.append((runtime, str(checkpoint)))  # noqa: F821


@pytest.fixture
def leverprobe(monkeypatch):
    """Stand-in for the frozen tools.leverprobe.backend (cpuV only): the same pin structure as
    git 1542367d setup(): ``sha(checkpoint) != MAPPO_SHA`` -> 'frozen source/checkpoint hash mismatch'."""
    module = types.ModuleType('tools.leverprobe.backend')
    module.MAPPO_SHA = FAKE_MAPPO_SHA
    module.calls = []

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    module.sha = sha
    module.setup = types.FunctionType(_frozen_setup.__code__, dict(sha=sha, MAPPO_SHA=FAKE_MAPPO_SHA, CALLS=module.calls,
                                                                   __builtins__=__builtins__), 'setup')
    package = types.ModuleType('tools.leverprobe')
    package.backend = module
    monkeypatch.setitem(sys.modules, 'tools.leverprobe', package)
    monkeypatch.setitem(sys.modules, 'tools.leverprobe.backend', module)
    return module


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
            plant_tick(t)
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
    assert nm.Runtime().runner_status() == dict(ok=True, native=True, decisions=False, lane=True, rule=True, parity=True,
                                                reason=None)
    assert runner.MAPPO_LANE is True and runner.RULE is True and runner.rule_digest() == nm.RULE_SHA256


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
    assert env['built'][-1] == ('B', 24, 989600)               # policy built like training, noise seed = world
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


def test_mappo_arm_swaps_only_the_executor_and_reverifies_it(env, monkeypatch, leverprobe):
    from tools.coopmwm import accel_plant
    calls = []
    plant = fake_plant(calls, [])

    def episode(job, **kw):
        import tools.leverprobe.backend as lp                  # what bind_runtime calls
        lp.setup('/home/ubuntu/e3final', job['checkpoint'])
        return plant(job, **kw)
    monkeypatch.setattr(accel_plant, 'episode', episode)
    final_root = env['tmp']/'dry'/'final_root'
    nm.register(root=env['root'], final_root=final_root, runtime=nm.FakeRuntime(), panel=env['panel'])
    binding = nm.bind_fake_mappo(env['tmp']/'dry', final_root, 'MAPPO_SPEED')
    u0 = nm.bind_fake_mappo(env['tmp']/'dry', final_root, 'U0', residual=env['root']/'A_checkpoint_000.pt')
    spec = nm.episode_specs('MAPPO_SPEED', env['panel'], env['cfg'], env['root'], 'mappo_u0', binding, u0)[0]
    original_setup = leverprobe.setup
    record = runner.episode(spec, env['cfg'], env['tmp']/'work'/spec['job_id'])
    assert calls[-1]['checkpoint'] == binding['path'] and calls[-1]['weights'] == {'arm': 'A', 'update': 0}
    assert calls[-1]['arm'] == 'A' and nm.check_episode(record, spec)['J_veh_s'] == 54321.5
    assert record['executor_sha256_loaded'] == binding['sha256'] != FAKE_MAPPO_SHA     # (b) bound new sha loads
    assert leverprobe.setup is original_setup and leverprobe.MAPPO_SHA == FAKE_MAPPO_SHA  # training pin restored
    with pytest.raises(ValueError, match='hash mismatch'):
        leverprobe.setup('/rt', binding['path'])            # outside the context the pin is d9042cc5 again
    with pytest.raises(nm.Blocked, match='never silent'):
        nm.check_episode(dict(record, executor_sha256_loaded=FAKE_MAPPO_SHA), spec)
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
    for t in range(300, 1200):
        observations(LogEnv(t))  # noqa: F821  (replaced by the runner's logged wrapper)
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
    monkeypatch.setattr(runner, 'closing_lane_regime', lambda trace: dict(closing_lane_speed_mps=7.5, n_obs=40))
    spec = nm.episode_specs('NATIVE', env['panel'], env['cfg'], env['root'], 'mappo_u0')[5]
    record = runner.episode(spec, env['cfg'], env['tmp']/'work'/spec['job_id'])
    job = NATIVE_JOBS[-1]
    assert job['arm'] == 'native' and job['role'] == 'final_test' and job['sim_s'] == job['stop'] == 1200
    assert job['world'] == spec['world'] and job['noise_seed'] == spec['world']['seed']
    assert [x[0] for x in installed] == ['setup', 'seed_scope', 'map', 'rows']
    assert record['J_veh_s'] == 61000.25 and record['safety'] == dict(collisions=0, teleports=0, hard_brakes=7, ttc_events=2)
    assert nm.check_episode(record, spec)['decision'] is None and record['regime']['closing_lane_speed_mps'] == 7.5
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


def test_comparator_setup_swaps_only_the_pin(tmp_path, leverprobe):
    frozen = tmp_path/'frozen.pt'
    other = tmp_path/'n7_it6.pt'
    third = tmp_path/'unbound.pt'
    other.write_bytes(b'n7 snapshot')
    third.write_bytes(b'something else')
    frozen_sha, other_sha = FAKE_MAPPO_SHA, hashlib.sha256(b'n7 snapshot').hexdigest()
    original = leverprobe.setup
    with runner.comparator_setup(frozen_sha) as setup:          # (a) d9042cc5 path unchanged
        assert setup is original and leverprobe.setup is original
    with runner.comparator_setup(other_sha) as setup:           # (b) the bound sha loads
        assert leverprobe.setup is setup and setup is not original
        assert setup.__code__ is original.__code__              # same function code, only the pin differs
        setup('/rt', other)
        with pytest.raises(ValueError, match='hash mismatch'):  # (c) an unbound sha still raises
            setup('/rt', third)
    assert leverprobe.setup is original and leverprobe.calls == [('/rt', str(other))]
    with pytest.raises(ValueError, match='hash mismatch'):      # (c) training path keeps the d9042cc5 pin
        original('/rt', other)
    assert frozen.exists() is False


def test_bound_executor_refuses_mismatched_or_tampered_bindings(tmp_path):
    from tools.coopmwm.exact_grade import _seal
    exe = tmp_path/'exe.pt'
    exe.write_bytes(b'exe')
    sha = hashlib.sha256(b'exe').hexdigest()
    record = _seal(dict(kind='retrained_wz3to2', label='x', path=str(exe), sha256=sha), 'binding_sha256')
    path = tmp_path/'MAPPO_COMPARATOR.json'
    path.write_text(json.dumps(record))
    ck = dict(binding_path=str(path), binding_sha256=record['binding_sha256'], executor_sha256=sha, executor_path=str(exe))
    assert runner.bound_executor(ck) == (str(exe), sha)
    with pytest.raises(ValueError, match='bound record'):
        runner.bound_executor(dict(ck, executor_sha256='0'*64))
    path.write_text(json.dumps(dict(record, sha256='1'*64)))   # edited after sealing
    with pytest.raises(ValueError, match='mismatch'):
        runner.bound_executor(ck)


def test_comparator_labels_name_the_loaded_sha_and_never_a_substitute():
    kind, label = nm.mappo_label('6de2509f37fe3debdaa84b56a5ddd1df9c886381440f36bc3bf6fc605d7c7a0c')
    assert kind == 'retrained_wz3to2_it168' and 'S-MAPPOWZ it168' in label and '6de2509f37fe' in label
    kind, label = nm.mappo_label('ab'*32, 'MAPPO_LANE')
    assert kind == 'lane_retrained_wz3to2_final' and 'S-MAPPOWZL final' in label
    assert nm.U0_LABEL.startswith('MAPPO u0 transferred')          # d9042cc5 is the U0 arm, never MAPPO_SPEED


def test_closing_lane_regime_reads_only_the_registered_rows(tmp_path):
    import gzip
    trace = tmp_path/'observable.jsonl.gz'
    rows = [dict(time_s=299, vehicles=[dict(lane=2, s_m=0., v_mps=1.)]),
            dict(time_s=300, vehicles=[dict(lane=2, s_m=-100., v_mps=6.), dict(lane=1, s_m=-100., v_mps=30.),
                                        dict(lane=2, s_m=-400., v_mps=30.)]),
            dict(time_s=359, vehicles=[dict(lane=2, s_m=70., v_mps=8.)]),
            dict(time_s=360, vehicles=[dict(lane=2, s_m=0., v_mps=99.)])]
    with gzip.open(trace, 'wt', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')
    out = runner.closing_lane_regime(trace)
    assert out['closing_lane_speed_mps'] == pytest.approx(7.) and out['n_obs'] == 2


class FakeMemory:
    present = None


class FakeLaneExecutor:
    memory, dwell = FakeMemory(), FakeMemory()

    def __call__(self, state, with_lanes=True):
        import numpy as np
        cmd = np.array([[.4, np.nan, -.2]])        # slot 1 = HDV (uncommanded)
        lanes = np.array([[1, -1, 0]])
        return cmd, lanes


class FakeEnv:
    def __init__(self):
        import numpy as np
        self.now, self.frame, self.steps = 300., log_frame(('c1', 'h1', 'c2')), []
        self.builder, self.api = types.SimpleNamespace(vehicle_ids=['c1', 'h1', 'c2']), None
        assert list(self.frame.vid_index) == [0, 1, 2] and np.all(self.frame.present)

    def state(self):
        return dict(is_cav=[True, False, True], present=[True, True, True])

    def slot_vehicle_ids(self):
        return ['c1', 'h1', 'c2']

    def step(self, cmd, lanes=None):
        self.steps.append((cmd.copy(), None if lanes is None else lanes.copy()))
        self.now += 1
        return None, dict(done=False, interval=dict(speed_deficit_vehicle_seconds=2., source_wait_vehicle_seconds=1.),
                          lane_commands={'c1': 1})


class FakeBackend:
    made = []

    def __init__(self, work, world, sim_s, checkpoint):
        import hashlib as h
        self.env, self.previous, self.command_digest = FakeEnv(), None, h.sha256()
        self.totals = dict(team_delay_vs=0., road_delay_vs=0., source_wait_vs=0., collisions=0., teleports=0.,
                           ttc_below_3_vs=0., hard_brake=0.)
        self.actual, self.checkpoint = 900., checkpoint
        FakeBackend.made.append(self)

    def observe(self, env):
        pass

    def close(self):
        pass


def test_mappo_lane_episode_steers_cavs_only_through_the_env_lane_path(env, monkeypatch, tmp_path):
    from tools.coopmwm import accel_plant
    from tools.coopfree import backend as cf
    from tools.coopmwm import rerun_tracks_backend
    final_root = env['tmp']/'dry'/'final_root'
    nm.register(root=env['root'], final_root=final_root, runtime=nm.FakeRuntime(), panel=env['panel'])
    binding = nm.bind_fake_mappo(env['tmp']/'dry', final_root, 'MAPPO_LANE')
    spec = nm.episode_specs('MAPPO_LANE', env['panel'], env['cfg'], env['root'], 'mappo_frozen', binding)[0]
    plant = types.SimpleNamespace(Backend=FakeBackend, exact=lambda x: x.tobytes().hex())   # as leverprobe.exact
    bound = []
    monkeypatch.setattr(accel_plant, 'bind_runtime', lambda job: bound.append(job) or plant)
    monkeypatch.setattr(runner, 'load_lane_executor', lambda path, sha: (FakeLaneExecutor(), dict(file_sha256=sha)))
    monkeypatch.setattr(cf, 'observations', lambda b: [dict(vid='c1', is_cav=True), dict(vid='h1', is_cav=False),
                                                      dict(vid='c2', is_cav=True)])
    monkeypatch.setattr(rerun_tracks_backend, 'safety', lambda result, trace: dict(collisions=0, teleports=0,
                                                                                    hard_brakes=0, ttc_events=0))
    record = runner.episode(spec, env['cfg'], tmp_path/'work'/spec['job_id'])
    b = FakeBackend.made[-1]
    assert bound[-1]['role'] == 'final_test' and bound[-1]['checkpoint'] == '/frozen/mappo_d9042cc5.pt'
    assert b.checkpoint == '/frozen/mappo_d9042cc5.pt' and len(b.env.steps) == 900       # 300..1200 s
    assert all(lanes is not None and list(lanes.reshape(-1)) == [1, -1, 0] for _, lanes in b.env.steps)
    assert record['J_veh_s'] == 900*3. and record['executor_sha256_loaded'] == binding['sha256']
    assert record['lane_request_ticks'] == 900 and nm.check_episode(record, spec)['J_veh_s'] == 2700.
    with pytest.raises(ValueError, match='non-CAV'):
        runner.cav_only(dict(is_cav=[True, False]), [[.1, .2]], [[0, -1]])
    with pytest.raises(ValueError, match='lane command for a non-CAV'):
        runner.cav_only(dict(is_cav=[True, False]), [[.1, float('nan')]], [[0, 1]])


class RuleController(FakeController):
    def apply(self, rows, commands, now):
        return {}


def test_rule_arm_runs_the_exact_s_regimeu0_rule_on_the_u0_episode(env, monkeypatch, leverprobe):
    from tools.coopmwm import accel_plant, regimeu0
    calls, applied = [], []
    rows = [dict(vid=f'main.{lane}', is_cav=True, s_m=-1000.0+10*lane, v_mps=25.0, lane=lane) for lane in (0, 1, 2)]
    rows.append(dict(vid='main.h', is_cav=False, s_m=-995.0, v_mps=20.0, lane=1))

    def episode(job, *, controller_hook=None, policy_loader=None):
        import tools.leverprobe.backend as lp
        lp.setup('/home/ubuntu/e3final', job['checkpoint'])
        calls.append(job)
        policy_loader(job['weights'])
        controller = RuleController()
        controller_hook(controller, job, {})
        commands = {r['vid']: 0.5 for r in rows if r['is_cav']}
        for t in (300, 301, 391):
            applied.append((t, controller.apply(rows, commands, t), dict(getattr(controller, 'g5_lane_targets', {}))))
            controller_hook.after_step(t, rows)
        for t in range(300, 1200):
            plant_tick(t)
        return dict(totals=dict(team_delay_vs=50000.0, collisions=0, teleports=0, hard_brake=1))
    monkeypatch.setattr(accel_plant, 'episode', episode)
    final_root = env['tmp']/'dry'/'final_root'
    nm.register(root=env['root'], final_root=final_root, runtime=nm.FakeRuntime(), panel=env['panel'])
    u0 = nm.bind_fake_mappo(env['tmp']/'dry', final_root, 'U0', residual=env['root']/'A_checkpoint_000.pt')
    spec = nm.episode_specs('RULE', env['panel'], env['cfg'], env['root'], 'mappo_u0', None, u0)[0]
    record = runner.episode(spec, env['cfg'], env['tmp']/'work'/spec['job_id'])
    job = calls[-1]
    assert job['checkpoint'] == u0['path'] and job['weights'] == {'arm': 'A', 'update': 0} and job['arm'] == 'A'
    assert env['built'][-1] == ('A', 0, 989600)                 # the U0 policy, built like the training A arm
    # solo2: only the CAV in SUMO lane 2 (WM lane 0) holds: a = clip(15 - 25, -1.5, +1) = -1.5, lane kept
    t, desired, lanes = applied[0]
    assert desired == {'main.0': dict(requested_raw=-1.5, acceleration_residual=-2.0, regime_rule='wall_h_solo2')}
    assert lanes == {'main.0': 0} and 'main.h' not in desired
    assert applied[1][1]['main.0']['requested_raw'] == -1.5        # held while active
    assert record['rule'] == dict(strategy='wall_h_solo2', activations=2, issued=3, lane_held_steps=3, members=2)
    assert record['rule_sha256_loaded'] == nm.RULE_SHA256 and record['executor_sha256_loaded'] == u0['sha256']
    checked = nm.check_episode(record, spec)
    assert checked['rule']['activations'] == 2 and 'decision_log' not in record
    monkeypatch.setattr(regimeu0, 'WALL_H', dict(regimeu0.WALL_H, v_w=16.0))  # any rule change is refused
    with pytest.raises(ValueError, match='differs from its registration'):
        runner.episode(spec, env['cfg'], env['tmp']/'work'/'tampered')
    assert len(calls) == 1


# ---------------------------------------------------------------- F11 parity traces

def _parity_states(t, k=4):
    """Shared fake truth for both paths: CAV slots 0..k-1 at time t (s, v, lane) and the MAPPO command."""
    return [dict(vid=f'main.{j}', s=-1500.0+40*j+20.0*(t-300), v=20.0+0.1*j, lane=j % 3, a=round(0.1*j-0.2, 4))
            for j in range(k)]


def _bind_parity(env):
    final_root = env['tmp']/'dry'/'final_root'
    nm.register(root=env['root'], final_root=final_root, runtime=nm.FakeRuntime(), panel=env['panel'])
    u0 = nm.bind_fake_mappo(env['tmp']/'dry', final_root, 'U0', residual=env['root']/'A_checkpoint_000.pt')
    speed = nm.bind_fake_mappo(env['tmp']/'dry', final_root, 'MAPPO_SPEED')
    return nm.parity_specs('MAPPO_SPEED', env['panel'], env['cfg'], env['root'], 'mappo_u0', speed, u0), speed


def test_parity_plant_path_records_the_mappo_speed_episode(env, monkeypatch, leverprobe):
    from tools.coopmwm import accel_plant
    calls = []

    def episode(job, *, controller_hook=None, policy_loader=None):
        import tools.leverprobe.backend as lp
        lp.setup('/home/ubuntu/e3final', job['checkpoint'])
        calls.append(job)
        policy_loader(job['weights'])
        controller = RuleController()
        controller_hook(controller, job, {})
        for t in range(300, job['stop']):
            truth = _parity_states(t)
            rows = [dict(vid=x['vid'], is_cav=True, s_m=x['s'], v_mps=x['v'], lane=x['lane']) for x in truth]
            rows.append(dict(vid='main.h', is_cav=False, s_m=-1600.0, v_mps=20.0, lane=0))
            controller.apply(rows, {x['vid']: x['a'] for x in truth}, float(t))
            controller_hook.after_step(t, rows)
        return dict(totals=dict(team_delay_vs=1.0, collisions=0, teleports=0, hard_brake=0))
    monkeypatch.setattr(accel_plant, 'episode', episode)
    specs, speed = _bind_parity(env)
    record = runner.parity(specs[0], env['cfg'], env['tmp']/'work'/specs[0]['job_id'])
    job = calls[-1]
    assert (job['role'], job['sim_s'], job['stop'], job['checkpoint']) == ('check', 420, 420, speed['path'])
    assert job['world'] == ms.world(989598, 3200, 0.6) and job['weights'] == {'arm': 'A', 'update': 0}
    trace = record['trace']
    assert len(trace) == 120 and trace[0]['t'] == 300 and set(trace[0]['cav']) == {f'main.{j}' for j in range(4)}
    assert trace[0]['cav']['main.1'] == [-1460.0, 20.1, 1, -0.1, None] and 'main.h' not in trace[0]['cav']
    checked = nm.check_parity(record, specs[0])
    assert checked['executor_sha256_loaded'] == speed['sha256'] and checked['safety'] == dict(collisions=0, teleports=0)
    with pytest.raises(ValueError, match='not allowed for role'):        # held-out worlds are refused for parity
        runner.parity(dict(specs[0], world=ms.world(989600, 3200, 0.6)), env['cfg'], env['tmp']/'work'/'heldout')


class FakeTrainEnv:
    def __init__(self, spec):
        self.spec, self.t, self.closed, self.steps = spec, None, False, []

    @property
    def now(self):
        return float(self.t)

    def _state(self):
        import numpy as np
        truth = _parity_states(self.t)
        pad = 2                                                  # one empty slot and one HDV slot
        s = np.array([x['s'] for x in truth]+[0.0, -1600.0])[None]
        v = np.array([x['v'] for x in truth]+[0.0, 20.0])[None]
        lane = np.array([x['lane'] for x in truth]+[0, 0])[None]
        return dict(s_m=s, v_mps=v, lane=lane, present=np.array([[True]*len(truth)+[False, True]]),
                    is_cav=np.array([[True]*len(truth)+[False]*pad]))

    def reset(self, *, seed, demand_vph, penetration, warmup_s, horizon_s):
        assert (seed, demand_vph, penetration, warmup_s, horizon_s) == (989598, 3200, 0.6, 300.0, 120.0)
        self.t = 300
        return self._state()

    def slot_vehicle_ids(self):
        return [f'main.{j}' for j in range(4)]+[None, 'main.h']

    def step(self, accel, lanes=None):
        self.steps.append((self.t, lanes))
        self.t += 1
        return self._state(), dict(interval=dict(collision_involvements=0, starting_teleports=0), done=False)

    def close(self):
        self.closed = True


def test_parity_train_path_uses_the_n7_training_env(env, monkeypatch):
    import numpy as np
    specs, speed = _bind_parity(env)
    built = []

    class Exe:
        def reset(self):
            built.append('reset')

        def __call__(self, state, with_lanes=False):
            a = np.full(state['s_m'].shape, np.nan)
            for j, x in enumerate(_parity_states(int(round(float(state['s_m'][0, 0]+1500.0)/20.0))+300)):
                a[0, j] = x['a']
            return a, None
    sanity = types.ModuleType('tools.nm_mappo.sanity_eval')
    sanity.load_cfg = lambda path: {'lanes': {'dwell_s': 5}}
    sanity.resolve = lambda raw, fake: ({'env_name': 'tools.nm_mappo.n1_adapter.make_env', 'env_kwargs': {}},
                                        types.SimpleNamespace(check_seeds=lambda seeds: built.append(('seeds', seeds))),
                                        [(0.6, 3200)])
    sanity.load_actor = lambda path, lanes: (built.append(('actor', path, lanes)) or Exe(), 'native')
    rollout = types.ModuleType('wms.control.mappo.rollout')
    envs = []
    rollout.build_env = lambda spec: envs.append(FakeTrainEnv(spec)) or envs[-1]
    monkeypatch.setitem(sys.modules, 'tools.nm_mappo', types.ModuleType('tools.nm_mappo'))
    monkeypatch.setitem(sys.modules, 'tools.nm_mappo.sanity_eval', sanity)
    monkeypatch.setitem(sys.modules, 'wms.control.mappo.rollout', rollout)
    train = specs[1]
    assert train['path'] == 'train' and train['train_config_sha256'] == nm.file_sha(train['train_config'])
    value = runner.parity_train_trace(train)
    assert built[:2] == [('seeds', [989598]), ('actor', speed['path'], {'dwell_s': 5})] and envs[0].closed
    assert len(value['trace']) == 120 and value['executor_sha256_loaded'] == speed['sha256']
    assert all(lanes is None for _, lanes in envs[0].steps)
    plant = dict(trace=[dict(t=t, n=5, cav={x['vid']: [x['s'], x['v'], x['lane'], x['a'], None] for x in _parity_states(t)})
                        for t in range(300, 420)])
    assert nm.parity_compare(plant, value)['verdict'] == 'IDENTICAL'
    Path(train['train_config']).write_text('# edited after binding\n')
    with pytest.raises(ValueError, match='differs from its binding'):
        runner.parity_train_trace(train)



def test_every_arm_writes_the_evaluation_only_ground_truth_vehicle_log(env, monkeypatch):
    import gzip
    from tools.coopmwm import accel_plant
    monkeypatch.setattr(accel_plant, 'episode', fake_plant([], []))
    spec = nm.episode_specs('B', env['panel'], env['cfg'], env['root'], 'mappo_u0')[1]
    work = env['tmp']/'work'/spec['job_id']
    record = runner.episode(spec, env['cfg'], work)
    log = record['vehicle_log']
    assert log['ticks'] == 900 and log['vehicles'] == 3 and log['sha256'] == runner._sha(work/'vehicles.jsonl.gz')
    assert nm.check_episode(record, spec)['vehicle_log'] == log
    with gzip.open(work/'vehicles.jsonl.gz', 'rt') as f:
        lines = [json.loads(x) for x in f]
    heads = [x for x in lines if 'veh' in x]
    ticks = [x for x in lines if 't' in x]
    assert [h['id'] for h in heads] == ['main.0', 'main.1', 'main.2'] and [h['cav'] for h in heads] == [True, False, True]
    assert set(heads[0]) == {'veh', 'id', 'cav', 'origin', 'depart_delay_s', 'first_t'}      # no vType / driver truth
    assert [x['t'] for x in ticks] == list(range(300, 1200)) and set(ticks[0]) == {'t', 'k', 's', 'v', 'a', 'lane'}
    assert ticks[0]['k'] == [0, 1, 2] and ticks[0]['a'] == [-0.5]*3 and ticks[0]['lane'] == [0, 1, 2]
    with pytest.raises(nm.Blocked, match='vehicle log'):
        nm.check_episode(dict(record, vehicle_log=dict(log, ticks=899)), spec)
    with pytest.raises(nm.Blocked, match='vehicle log'):
        nm.check_episode({k: v for k, v in record.items() if k != 'vehicle_log'}, spec)
