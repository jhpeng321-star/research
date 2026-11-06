"""CPU/fake tests for the extra frozen lane-on MAPPO aecd7d70 u24 baseline.

No SUMO, GPU or ssh. Real C4 exports (tiny CavPolicy) drive the real deployment
executor; the frozen plant, process pool and A/B/R episodes are fakes. Covers: the
SHA-pinned adapter, the same paired worlds/seeds as A/B/R, lane commands only to
CAVs through the env lane path, the receipt schema, charging the budgeted phase, the
paired CIs against each arm, and byte-identity of the existing u24 evaluation when
the option is off.
"""
from copy import deepcopy
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pytest
import torch

from . import launch_evaluation as le, launch_training, protocol, rerun_cpus as rc
from . import laneon_eval as lb
from .accel_storage import write as accel_write
from .clean3_restart import LANE_ON_BASELINE, PHASES
from .exact_grade import _seal, verify_sealed
from .protocol import digest

COUNTERS = list(rc.FINAL_EVALUATION['safety_counters'])
OFFSET = {'A': 4.0, 'B': 2.0, 'R': 3.0}


# ---------------------------------------------------------------- fixtures / fakes

def export_checkpoint(path, *, lane_head=True, trainer='mappo', fmt='c4_dreamer/3', bias=None):
    """A real C4 MAPPO export in the aecd7d70 payload schema (or a deliberately wrong one)."""
    from wms.control.lanes import LaneRules
    from wms.control.mappo.agent import export_payload
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    torch.manual_seed(11)
    payload = export_payload(CavPolicy(lane_head=lane_head), RsuGroupEncoder(), LaneRules(),
                             iteration=170)
    payload.update(lane_head=lane_head, trainer=trainer, format=fmt)
    if bias is not None:
        payload['policy']['lane_head.2.bias'] = torch.tensor(bias)
    torch.save(payload, path)
    return Path(path)


def worlds96():
    return protocol.worlds('train', 0)+protocol.worlds('train', 1)[:24]


def registration96():
    return rc.final_world_registration(dict(final_evaluation=dict(role='train',
        worlds=[dict(world_id=str(w['seed']), world=deepcopy(w)) for w in worlds96()])))


def J(world, arm):
    return 1000.0+world['seed'] % 17+OFFSET.get(arm, 1.0)


def prefix(world):
    return dict(t_s=300.0, state_sha256=hashlib.sha256(str(world['seed']).encode()).hexdigest())


class Done:
    def __init__(self, value, error):
        self.value, self.error = value, error

    def result(self, timeout=None):
        if self.error is not None:
            raise self.error
        return self.value


class InlinePool:
    """ProcessPoolExecutor stand-in: runs each submitted job immediately in-process."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args):
        try:
            return Done(fn(*args), None)
        except Exception as error:
            return Done(None, error)


def fake_arm_episode(spec, cfg, work):
    accel_write(Path(work)/'RESULT.json', dict(prefixes={'300': prefix(spec['world'])}))
    return dict({k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world', 'checkpoint', 'window',
                                                'final_worlds_sha256')},
                status='graded', J_veh_s=J(spec['world'], spec['arm']),
                safety=dict(collisions=0, teleports=0, hard_brakes=1, ttc_events=2))


def fake_laneon_episode(calls, *, fail_world=None, prefix_fn=prefix):
    def run(spec, cfg, work):
        calls.append((spec, Path(work)))
        if spec['world_id'] == fail_world:
            raise ValueError('fake plant fault')
        return dict({k: deepcopy(spec[k]) for k in lb.IDENTITY_KEYS}, status='graded',
                    J_veh_s=J(spec['world'], 'laneon'),
                    safety=dict(collisions=0, teleports=0, hard_brakes=3, ttc_events=1),
                    lane_commands=dict(issued=5, completed=4, noop=2), lane_request_ticks=7,
                    actual_sim_s=1200.0, warmup_prefix_sha256=digest(prefix_fn(spec['world'])),
                    executor=dict(fake=True), result_ref=str(Path(work)/'RESULT.json'))
    return run


def make_root(path):
    root = Path(path)
    (root/'evaluation').mkdir(parents=True)
    (root/'evaluation'/'LESSON_u024.json').write_text('{}')    # u24 lessons: out of scope
    launch_training.write(root/'ISOLATED_REGISTRATION.json', dict(final_evaluation=dict(
        role='train', worlds=[dict(world_id=str(w['seed']), world=deepcopy(w)) for w in worlds96()])))
    for update in (8, 24):
        for arm in ('A', 'B', 'R'):
            (root/f'{arm}_checkpoint_{update:03}.pt').write_bytes(f'{arm}-u{update}'.encode())
    return root


def make_cfg(root, *, registered=True, phases=None):
    cfg = dict(run_root=str(root), final_evaluation=deepcopy(rc.FINAL_EVALUATION),
               prior_charged_sim_s=960,
               budget=dict(total=5_000_000, phases=phases or dict(
                   v2v_lessons=40_800, v2v_interim=398_400, v2v_final=PHASES['v2v_final'])))
    if registered:
        cfg['extra_final_baselines'] = [dict(LANE_ON_BASELINE)]
    return cfg


@pytest.fixture
def plant(monkeypatch):
    """Fake process pool/episodes; the real reserve/ledger, harness and analysis run."""
    monkeypatch.setattr(le.production, 'backend_config', lambda cfg: cfg)
    monkeypatch.setattr(launch_training, 'health', lambda root: None)
    monkeypatch.setattr(le, 'ProcessPoolExecutor', InlinePool)
    monkeypatch.setattr(le, 'episode', fake_arm_episode)
    monkeypatch.setattr(lb, 'ProcessPoolExecutor', InlinePool)
    calls = []
    monkeypatch.setattr(lb, 'episode', fake_laneon_episode(calls))
    return calls


@pytest.fixture
def staged(tmp_path):
    """A staged C4 export; tests redirect the aecd7d70 pin to its SHA with pin()."""
    folder = tmp_path/'staged'
    folder.mkdir()
    return export_checkpoint(folder/'mappo_v5_final.pt')


def pin(monkeypatch, path):
    monkeypatch.setattr(lb, 'CHECKPOINT_SHA256', rc.sha(path))


def ledger_rows(root):
    path = Path(root)/'CHARGES.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------- 1. adapter: checkpoint pin

def test_adapter_loads_the_pinned_checkpoint_and_refuses_a_wrong_sha(tmp_path, monkeypatch):
    assert lb.CHECKPOINT_SHA256 == 'aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564'
    assert lb.CHECKPOINT_SHA256.startswith(LANE_ON_BASELINE['commit'])
    path = export_checkpoint(tmp_path/'export.pt')
    sha = rc.sha(path)
    executor, meta = lb.load_executor(path, expected_sha256=sha)
    assert executor.deterministic is True
    assert any(k.startswith('lane_head.') for k in executor.policy.state_dict())
    assert meta == dict(format='c4_dreamer/3', trainer='mappo', lane_head=True, iteration=170,
                        file_sha256=sha, deterministic=True,
                        loader='wms.control.policy.load_policy_fn', lane_change_mode=513)
    with pytest.raises(ValueError, match='SHA differs'):
        lb.load_executor(path)                             # production pin: aecd7d70 only
    tampered = tmp_path/'tampered.pt'
    tampered.write_bytes(path.read_bytes()+b'\0')
    with pytest.raises(ValueError, match='SHA differs'):
        lb.load_executor(tampered, expected_sha256=sha)
    for kwargs, match in ((dict(lane_head=False), 'lane head'), (dict(trainer='dreamer'), 'mappo'),
                          (dict(fmt='c4_dreamer/2'), 'c4_dreamer/3')):
        other = export_checkpoint(tmp_path/f'other_{len(match)}.pt', **kwargs)
        with pytest.raises(ValueError, match=match):
            lb.load_executor(other, expected_sha256=rc.sha(other))
    monkeypatch.setenv('C4_POLICY_STOCHASTIC', '1')
    with pytest.raises(ValueError, match='deterministic'):
        lb.load_executor(path, expected_sha256=sha)
    monkeypatch.delenv('C4_POLICY_STOCHASTIC')
    entry = dict(LANE_ON_BASELINE)
    assert lb.staged_checkpoint(entry, path, expected_sha256=sha) == path
    with pytest.raises(ValueError, match='SHA differs'):
        lb.staged_checkpoint(entry, path)
    with pytest.raises(ValueError, match='absolute'):
        lb.staged_checkpoint(entry, 'relative/export.pt', expected_sha256=sha)
    with pytest.raises(ValueError, match='registered binding'):
        lb.staged_checkpoint(dict(entry, checkpoint=str(tampered)), path, expected_sha256=sha)
    # registration slot: exactly the clean3_restart metadata (+ optional checkpoint binding)
    assert lb.registered_entry(dict(extra_final_baselines=[dict(entry)])) == entry
    for bad, match in (([], 'not registered'), ([entry, entry], 'more than once'),
                       ([dict(entry, episodes=95)], 'differs'), ([dict(entry, phase='x')], 'differs')):
        with pytest.raises(ValueError, match=match):
            lb.registered_entry(dict(extra_final_baselines=bad))
    # a wrong-SHA staged file is refused before ANY reservation / ledger write
    root = make_root(tmp_path/'run')
    with pytest.raises(ValueError, match='SHA differs'):
        lb.run(make_cfg(root), str(path))
    assert not (root/'CHARGES.jsonl').exists() and not (root/lb.FAILED).exists()


# ---------------------------------------------------------------- 2. same paired worlds/seeds

def test_specs_use_the_same_paired_worlds_and_seeds_as_the_arms(tmp_path):
    from . import planner_eval
    registration = registration96()
    contract = deepcopy(rc.FINAL_EVALUATION)
    checkpoints = {a: dict(path=f'/policies/{a}_024.pt', file_sha256='a'*64, update=24,
                           components=contract['checkpoint_components'][a]) for a in ('A', 'B', 'R')}
    arms = planner_eval.prepare_track2(registration=registration, checkpoints=checkpoints,
                                       contract=contract)
    pairs = lb.specs(registration, tmp_path/'aecd7d70.pt')
    assert len(pairs) == len(arms)//3 == 96
    assert sum(s['sim_s'] for s, _ in pairs) == lb.QUOTE_BASE_SIM_S == 115_200
    for arm in ('A', 'B', 'R'):
        mine = [(s['world_id'], s['world'], s['sim_s'], s['window'], s['final_worlds_sha256'])
                for s, _ in pairs]
        theirs = [(s['world_id'], s['world'], s['sim_s'], s['window'], s['final_worlds_sha256'])
                  for s, _ in arms if s['arm'] == arm]
        assert mine == theirs                                     # same worlds, same order
    assert [s['world']['seed'] for s, _ in pairs] == [w['seed'] for w in worlds96()]
    assert all(s['arm'] == lb.NAME and s['lane_change_mode'] == 513 and
               s['checkpoint']['file_sha256'] == lb.CHECKPOINT_SHA256 for s, _ in pairs)
    reference = {(s['arm'], s['world_id']): dict(s, status='graded') for s, _ in arms}
    check = lb.same_worlds(pairs, reference, contract)
    assert check['worlds'] == 96 and check['seeds'] == sorted(w['seed'] for w in worlds96())
    moved = deepcopy(reference)
    moved[('B', pairs[5][0]['world_id'])]['world'] = dict(pairs[5][0]['world'], seed=1)
    with pytest.raises(ValueError, match='differs from the paired arm world'):
        lb.same_worlds(pairs, moved, contract)
    missing = {k: v for k, v in reference.items() if k != ('R', pairs[0][0]['world_id'])}
    with pytest.raises(ValueError, match='differs from the paired arm world'):
        lb.same_worlds(pairs, missing, contract)
    short = _seal(dict(registration, worlds=registration['worlds'][:95]), 'final_worlds_sha256')
    with pytest.raises(ValueError, match='96'):
        lb.specs(short, tmp_path/'aecd7d70.pt')


def test_run_uses_exactly_the_paired_u24_worlds_and_refuses_a_different_panel(
        tmp_path, monkeypatch, plant, staged):
    pin(monkeypatch, staged)
    root = make_root(tmp_path/'run')
    cfg = make_cfg(root)
    paired = le.scheduled(cfg, 24)
    value = lb.run(cfg, str(staged))
    assert [spec['world'] for spec, _ in plant] == [r['world'] for r in paired['registration']['worlds']]
    assert [spec['world_id'] for spec, _ in plant] == \
        [r['world_id'] for r in paired['results'] if r['arm'] == 'A']
    assert value['same_worlds']['seeds'] == sorted(w['seed'] for w in worlds96())
    assert value['analysis']['warmup_prefix_check'] == dict(
        rule='t=300 Backend fingerprint digest equal across lane-on and A/B/R',
        compared=96, matched=96, mismatched=[], unavailable=[])
    # a lane-on world whose physical warmup prefix differs is reported, never hidden
    other = make_root(tmp_path/'other')
    cfg2 = make_cfg(other)
    le.scheduled(cfg2, 24)
    odd = worlds96()[3]['seed']
    monkeypatch.setattr(lb, 'episode', fake_laneon_episode([], prefix_fn=lambda w: dict(
        prefix(w), state_sha256='0'*64) if w['seed'] == odd else prefix(w)))
    check = lb.run(cfg2, str(staged))['analysis']['warmup_prefix_check']
    assert check['mismatched'] == [str(odd)] and check['matched'] == 95
    # a PAIRED receipt on a different panel is refused before any lane-on charge
    third = make_root(tmp_path/'third')
    cfg3 = make_cfg(third)
    le.scheduled(cfg3, 24)
    receipt = launch_training.read(third/lb.PAIRED)
    (third/lb.PAIRED).unlink()
    swapped = deepcopy(receipt['registration'])
    swapped['worlds'][0], swapped['worlds'][1] = swapped['worlds'][1], swapped['worlds'][0]
    launch_training.write(third/lb.PAIRED, _seal(dict(receipt, registration=_seal(
        swapped, 'final_worlds_sha256')), 'receipt_sha256'))
    with pytest.raises(ValueError, match='panel differs'):
        lb.run(cfg3, str(staged))
    assert not any(r['id'].startswith('laneon_') for r in ledger_rows(third))


# ---------------------------------------------------------------- 3. lane commands: CAVs only

class FakeFrame:
    def __init__(self, vid_index):
        self.vid_index = np.asarray(vid_index)


class FakeEnv:
    """Toy direct-mode env exposing the C4 state/step contract (no SUMO)."""
    def __init__(self, s, lane, cav, present, now=300.0):
        k = len(s)
        self.now, self.k = now, k
        self.s, self.lane = list(s), list(lane)
        self.cav, self.present = list(cav), list(present)
        self.frame = FakeFrame([i if p else -1 for i, p in enumerate(present)])
        self.steps = []

    def slot_vehicle_ids(self):
        return [f'{"cav" if c else "hdv"}{i}' for i, c in enumerate(self.cav)]

    def state(self):
        k = self.k
        return {'s_m': torch.tensor([self.s], dtype=torch.float32),
                'v_mps': torch.full((1, k), 20.0), 'a_mps2': torch.zeros(1, k),
                'lane': torch.tensor([self.lane]), 'present': torch.tensor([self.present]),
                'is_cav': torch.tensor([self.cav]) & torch.tensor([self.present]),
                'length_m': torch.full((1, k), 5.0)}

    def step(self, accel, **kwargs):
        lanes = kwargs.get('lanes')
        self.steps.append(dict(accel=np.asarray(accel).copy(), kwargs=sorted(kwargs),
                               lanes=None if lanes is None else np.asarray(lanes).copy()))
        issued = 0
        for i, target in enumerate(np.asarray(lanes).reshape(-1)):
            if target >= 0 and self.cav[i] and self.present[i]:   # env: CAV slots only
                self.lane[i], issued = int(target), issued+1
        self.s = [x+20.0 for x in self.s]
        self.now += 1.0
        return self.state(), dict(done=False, lane_commands={'issued': issued},
            interval=dict(speed_deficit_vehicle_seconds=1.5, source_wait_vehicle_seconds=0.5,
                          braking_vehicle_seconds=0.0))


class FakeBackend:
    def __init__(self, env):
        self.env, self.previous = env, None
        self.command_digest = hashlib.sha256()
        self.totals = dict.fromkeys(('team_delay_vs', 'road_delay_vs', 'source_wait_vs',
                                     'collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake'), 0.)
        self.actual = 300.0
        self.traci = types.SimpleNamespace(vehicle=types.SimpleNamespace(getLength=lambda vid: 5.0))

    def observe(self, env):
        self.actual = float(env.now)

    def fingerprint(self):
        return dict(t_s=float(self.env.now), lanes=list(self.env.lane))

    def close(self):
        pass


def fake_observations(env):
    ids = env.slot_vehicle_ids()
    return [dict(vid=ids[i], s_m=env.s[i], v_mps=20.0, lane=env.lane[i], is_cav=env.cav[i])
            for i in range(env.k) if env.present[i]]


def fake_plant_module():
    module = types.ModuleType('tools.leverprobe.backend')
    module.calls = []
    module.observations = fake_observations
    module.exact = lambda cmd: [None if not math.isfinite(x) else round(float(x), 9)
                                for x in np.asarray(cmd).reshape(-1)]
    module.setup = lambda runtime, checkpoint: module.calls.append(('setup', runtime, checkpoint))

    def backend(path, world, sim_s, checkpoint):
        module.calls.append(('Backend', Path(path), world, sim_s, checkpoint))
        module.env = FakeEnv([-500.0, -950.0, -300.0, -700.0], [0, 1, 1, 1],
                             cav=[True, True, False, True], present=[True, True, True, False])
        return FakeBackend(module.env)
    module.Backend = backend
    return module


def test_lane_commands_reach_only_cavs_through_the_env_lane_path(tmp_path):
    from tools.coopfree.protocol import canonical
    # Biased 3-way head (stay -5, left/right +5): every feasible CAV asks to change lane.
    path = export_checkpoint(tmp_path/'biased.pt', bias=[-5.0, 5.0, 5.0])
    executor, _ = lb.load_executor(path, expected_sha256=rc.sha(path))
    plant = fake_plant_module()
    b = plant.Backend(tmp_path, {}, 1200, 'speed-only.pt')
    tape = io.StringIO()
    observed = types.SimpleNamespace(observations=lambda b: [dict(r, length_m=5.0)
                                                             for r in fake_observations(b.env)])
    requested, counts = lb.tick(b, executor, tape, plant=plant, cf=observed, canonical=canonical)
    first = b.env.steps[0]
    assert first['kwargs'] == ['lanes']                       # env lane path, no lane_actors
    assert first['lanes'].tolist() == [[1, 0, -1, -1]]        # CAV targets; HDV/absent stay
    assert np.isfinite(first['accel'][0, :2]).all() and np.isnan(first['accel'][0, 2:]).all()
    assert requested == 2 and counts == {'issued': 2}
    assert b.totals['team_delay_vs'] == 2.0
    row = json.loads(tape.getvalue())
    hdv = [v for v in row['vehicles'] if not v['is_cav']]
    assert hdv and all('lane_target' not in v and 'issued_acceleration_mps2' not in v for v in hdv)
    assert [v['lane_target'] for v in row['vehicles'] if v['is_cav']] == [1, 0]
    # a stay-only tick (dwell) still passes a lanes tensor: 513 stays on for commanded CAVs
    lb.tick(b, executor, None, plant=plant, cf=observed, canonical=canonical)
    assert b.env.steps[1]['lanes'] is not None and (b.env.steps[1]['lanes'] == -1).all()
    # the adapter refuses any command on a non-CAV/absent slot before the env sees it
    state = b.env.state()
    for cmd, lanes, match in (([[1., 1., np.nan, np.nan]], [[-1, -1, 0, -1]], 'lane command'),
                              ([[1., 1., np.nan, np.nan]], [[-1, -1, -1, 2]], 'lane command'),
                              ([[1., 1., 0.5, np.nan]], [[-1, -1, -1, -1]], 'non-CAV slot'),
                              ([[1., 1., np.nan, np.nan]], [[7, -1, -1, -1]], 'lane grid'),
                              ([[1., 1., np.nan, np.nan]], [[0.5, -1, -1, -1]], 'integers'),
                              ([[1., 1., np.nan]], [[-1, -1, -1]], 'slot count'),
                              ([[np.inf, 1., np.nan, np.nan]], [[-1, -1, -1, -1]], 'infinite')):
        with pytest.raises(ValueError, match=match):
            lb.cav_only(state, np.asarray(cmd), np.asarray(lanes))
    # slot reuse: a new vehicle in a slot never inherits command memory / lane dwell
    b.env.frame = FakeFrame([0, 7, 2, -1])
    lb.tick(b, executor, None, plant=plant, cf=observed, canonical=canonical)
    assert b.previous.tolist() == [0, 7, 2, -1]


def test_episode_runs_the_frozen_plant_with_lanes_every_tick(tmp_path, monkeypatch):
    path = export_checkpoint(tmp_path/'export.pt')
    pin(monkeypatch, path)
    plant = fake_plant_module()
    package = types.ModuleType('tools.leverprobe')
    package.backend = plant
    monkeypatch.setitem(sys.modules, 'tools.leverprobe', package)
    monkeypatch.setitem(sys.modules, 'tools.leverprobe.backend', plant)
    guarded = []
    monkeypatch.setattr(protocol, 'guard', lambda world, role: guarded.append((world['seed'], role)))
    monkeypatch.setattr(protocol, 'install_seed_scope', lambda: guarded.append('scope'))
    monkeypatch.delenv('COOPMWM_G5_RUNTIME', raising=False)
    spec = lb.specs(registration96(), path)[0][0]
    cfg = dict(runtime='/frozen/e3final', checkpoint='/frozen/speed_only_d9042cc5.pt')
    with pytest.raises(ValueError, match='COOPMWM_G5_RUNTIME'):
        lb.episode(spec, dict(cfg, g5_runtime=dict(flag='COOPMWM_G5_RUNTIME')), tmp_path/'w0')
    assert not (tmp_path/'w0').exists()
    work = tmp_path/'evaluation'/'laneon_u024_0000'
    result = lb.episode(spec, cfg, work)
    # the SAME plant binding as the A/B/R final episodes (runtime + their checkpoint arg)
    assert plant.calls[0] == ('setup', cfg['runtime'], cfg['checkpoint'])
    assert plant.calls[1] == ('Backend', work, spec['world'], 1200, cfg['checkpoint'])
    assert guarded == ['scope', (spec['world']['seed'], 'train')]
    steps = plant.env.steps
    assert len(steps) == 900 and all(s['kwargs'] == ['lanes'] and s['lanes'] is not None
                                     for s in steps)
    assert all((s['lanes'][0, 2:] == -1).all() for s in steps)       # HDV/absent: never
    assert result['status'] == 'graded' and result['J_veh_s'] == 900*2.0
    assert set(result['safety']) == set(COUNTERS)
    assert result['safety']['collisions'] == result['safety']['teleports'] == 0
    assert {k: result[k] for k in lb.IDENTITY_KEYS} == {k: spec[k] for k in lb.IDENTITY_KEYS}
    assert result['warmup_prefix_sha256'] == digest(dict(t_s=300.0, lanes=[0, 1, 1, 1]))
    assert result['executor']['file_sha256'] == rc.sha(path)
    assert (work/'CLAIM.json').is_file() and (work/'RESULT.json').is_file()
    record = lb.make_receipt(spec, result, job_id='laneon_u024_0000', phase='v2v_final',
                             safety_counters=COUNTERS)
    assert record['lane_request_ticks'] == result['lane_request_ticks']


# ---------------------------------------------------------------- 4. receipt schema

def test_receipt_schema_binds_identity_and_refuses_tampering(tmp_path):
    pairs = lb.specs(registration96(), tmp_path/'aecd7d70.pt')
    spec, other = pairs[0][0], pairs[1][0]
    result = fake_laneon_episode([])(spec, {}, tmp_path/'w')
    record = lb.make_receipt(spec, result, job_id='laneon_u024_0000', phase='v2v_final',
                             safety_counters=COUNTERS)
    assert set(record) == set(lb.RECEIPT_KEYS) and record['schema'] == lb.RECEIPT_SCHEMA
    verify_sealed(record, 'receipt_sha256')
    assert record['seed'] == spec['world']['seed'] and record['charged_sim_s'] == 1200
    assert record['checkpoint_sha256'] == lb.CHECKPOINT_SHA256 and record['lane_change_mode'] == 513
    assert record['charged_phase'] == 'v2v_final' and record['job_id'] == 'laneon_u024_0000'
    lb.validate_receipt(record, spec, safety_counters=COUNTERS)
    with pytest.raises(ValueError, match='receipt_sha256 mismatch'):
        lb.validate_receipt(dict(record, J_veh_s=record['J_veh_s']-1), safety_counters=COUNTERS)
    body = {k: v for k, v in record.items() if k != 'receipt_sha256'}
    with pytest.raises(ValueError, match='keys differ'):
        lb.validate_receipt(_seal(dict(body, extra=1), 'receipt_sha256'), safety_counters=COUNTERS)
    with pytest.raises(ValueError, match='full registered window'):
        lb.validate_receipt(_seal(dict(body, window=dict(start_s=0, end_s=600)), 'receipt_sha256'),
                            safety_counters=COUNTERS)
    with pytest.raises(ValueError, match='exceeds its charge'):
        lb.validate_receipt(_seal(dict(body, actual_sim_s=1200.5), 'receipt_sha256'),
                            safety_counters=COUNTERS)
    with pytest.raises(ValueError, match='differs from its job'):
        lb.validate_receipt(record, other, safety_counters=COUNTERS)
    for change, match in ((dict(safety=dict(collisions=0)), 'safety keys'),
                          (dict(safety=dict.fromkeys(COUNTERS, -1)), 'nonnegative'),
                          (dict(J_veh_s=float('nan')), 'finite'),
                          (dict(lane_commands={'issued': 1.5}), 'lane_commands'),
                          (dict(status='failed'), 'graded'),
                          (dict(world=other['world']), 'binding differs'),
                          (dict(checkpoint=dict(spec['checkpoint'], file_sha256='0'*64)),
                           'binding differs')):
        with pytest.raises(ValueError, match=match):
            lb.make_receipt(spec, dict(result, **change), job_id='laneon_u024_0000',
                            phase='v2v_final', safety_counters=COUNTERS)


# ---------------------------------------------------------------- 5. option off: byte-identical

def test_existing_u24_evaluation_is_byte_identical_when_the_option_is_off(tmp_path, monkeypatch,
                                                                          plant, staged):
    pin(monkeypatch, staged)
    assert 'laneon' not in inspect.getsource(le)          # no hook in the existing evaluation
    root = tmp_path/'run'

    def evaluate(registered):
        if root.exists():
            shutil.rmtree(root)
        make_root(root)
        cfg = make_cfg(root, registered=registered)
        le.scheduled(cfg, 24)
        return cfg, (root/lb.PAIRED).read_bytes(), (root/'CHARGES.jsonl').read_bytes()

    _, off_paired, off_ledger = evaluate(registered=False)
    assert plant == [] and not (root/lb.TARGET).exists()
    assert not any(r['id'].startswith('laneon_') for r in ledger_rows(root))
    assert sorted(p.name for p in (root/'evaluation').iterdir() if p.is_file()) ==         ['LESSON_u024.json', 'PAIRED_u024.json']
    cfg, on_paired, on_ledger = evaluate(registered=True)
    assert on_paired == off_paired and on_ledger == off_ledger   # registration alone changes nothing
    assert plant == []                                           # option off: never executed
    lb.run(cfg, str(staged))                                     # option on, afterwards
    assert (root/lb.PAIRED).read_bytes() == off_paired           # primary receipt untouched
    assert (root/'CHARGES.jsonl').read_bytes().startswith(off_ledger)   # append-only
    assert le.scheduled(cfg, 24) == json.loads(off_paired)       # existing reader unchanged


# ---------------------------------------------------------------- 6. charge phase + paired CIs

def test_run_charges_its_budgeted_phase_and_reports_paired_cis_against_each_arm(
        tmp_path, monkeypatch, plant, staged):
    pin(monkeypatch, staged)
    root = make_root(tmp_path/'run')
    cfg = make_cfg(root)
    with pytest.raises(ValueError, match='not complete'):
        lb.run(cfg, str(staged))                         # never before the u24 paired receipt
    assert ledger_rows(root) == []
    paired = le.scheduled(cfg, 24)
    before = ledger_rows(root)
    assert len(before) == 288 and {r['phase'] for r in before} == {'v2v_final'}
    quote = lb.dry(cfg, str(staged))
    assert quote['new_sim_s'] == 0 and quote['fits'] and quote['quoted_sim_s'] == 115_200
    assert quote['charge']['phase'] == 'v2v_final' and quote['charge']['allocation_sim_s'] == 133_200
    assert ledger_rows(root) == before
    value = lb.run(cfg, str(staged))
    verify_sealed(value, 'receipt_sha256')
    rows = ledger_rows(root)
    assert rows[:288] == before
    new = rows[288:]
    assert [r['id'] for r in new] == [f'laneon_u024_{i:04}' for i in range(96)]
    assert {r['phase'] for r in new} == {'v2v_final'} and sum(r['sim_s'] for r in new) == 115_200
    assert value['charge']['phase'] == 'v2v_final' and value['charge']['merged_into_v2v_final']
    assert value['new_sim_s'] == 115_200 and value['descriptive'] is True
    assert value['paired_receipt_sha256'] == paired['receipt_sha256']
    assert len(value['receipts']) == 96
    for record, (spec, _) in zip(value['receipts'], lb.specs(paired['registration'], staged)):
        lb.validate_receipt(record, spec, safety_counters=COUNTERS)
    analysis = value['analysis']
    verify_sealed(analysis, 'analysis_sha256')
    assert analysis['gated'] is False and analysis['descriptive'] is True
    assert analysis['comparisons'] == [f'{lb.NAME}-{a}' for a in ('A', 'B', 'R')]
    for arm in ('A', 'B', 'R'):
        ci = analysis['paired_ci'][f'{lb.NAME}-{arm}']
        assert ci['n'] == 96 and ci['mean'] == pytest.approx(1.0-OFFSET[arm])
        assert ci['low'] == pytest.approx(ci['mean']) == pytest.approx(ci['high'])
        assert analysis['safety_paired_ci'][f'{lb.NAME}-{arm}']['hard_brakes']['mean'] == 2.0
    assert analysis['bootstrap'] == rc.FINAL_EVALUATION['bootstrap']
    assert analysis['safety'][lb.NAME] == dict(collisions=0, teleports=0, hard_brakes=288, ttc_events=96)
    assert analysis['safety']['A']['ttc_events'] == 192
    assert analysis['lane_commands'] == dict(issued=480, completed=384, noop=192)
    assert lb.run(cfg, str(staged)) == value and len(ledger_rows(root)) == 384   # idempotent
    # a registration carrying the quote phase key is charged under that key
    split = make_root(tmp_path/'split')
    cfg2 = make_cfg(split, phases=dict(v2v_lessons=40_800, v2v_interim=398_400,
                                       v2v_final=398_400, **{lb.QUOTE_PHASE: 133_200}))
    le.scheduled(cfg2, 24)
    assert lb.run(cfg2, str(staged))['charge']['phase'] == lb.QUOTE_PHASE
    assert {r['phase'] for r in ledger_rows(split)[288:]} == {lb.QUOTE_PHASE}
    # no allocation for the baseline (v2v_final without its 133200) -> refused, nothing charged
    bare = make_root(tmp_path/'bare')
    cfg3 = make_cfg(bare, phases=dict(v2v_lessons=40_800, v2v_interim=398_400, v2v_final=398_400))
    le.scheduled(cfg3, 24)
    with pytest.raises(ValueError, match='no registered budget allocation'):
        lb.run(cfg3, str(staged))
    assert len(ledger_rows(bare)) == 288


def test_run_never_retries_and_records_a_failed_attempt(tmp_path, monkeypatch, plant, staged):
    pin(monkeypatch, staged)
    root = make_root(tmp_path/'run')
    cfg = make_cfg(root)
    le.scheduled(cfg, 24)
    # A/B/R charges must precede the baseline (it can never crowd out the primary)
    rows = (root/'CHARGES.jsonl').read_text().splitlines()
    (root/'CHARGES.jsonl').write_text('\n'.join(rows[1:])+'\n')
    with pytest.raises(ValueError, match='A/B/R u24 charges'):
        lb.run(cfg, str(staged))
    (root/'CHARGES.jsonl').write_text('\n'.join(rows)+'\n')
    failing = str(worlds96()[10]['seed'])
    monkeypatch.setattr(lb, 'episode', fake_laneon_episode([], fail_world=failing))
    with pytest.raises(ValueError, match='incomplete lane-on baseline'):
        lb.run(cfg, str(staged))
    failed = launch_training.read(root/lb.FAILED)
    verify_sealed(failed, 'receipt_sha256')
    assert failed['errors'] == [dict(job='laneon_u024_0010', error="ValueError('fake plant fault')")]
    assert failed['charged_sim_s'] == 115_200 and not (root/lb.TARGET).exists()
    assert sum(r['id'].startswith('laneon_') for r in ledger_rows(root)) == 96   # charged, kept
    with pytest.raises(ValueError, match='never retry'):
        lb.run(cfg, str(staged))
    assert sum(r['id'].startswith('laneon_') for r in ledger_rows(root)) == 96
