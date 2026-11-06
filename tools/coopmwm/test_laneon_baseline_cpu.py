"""CPU/fake tests for the registered EXTRA frozen lane-on MAPPO aecd7d70 baseline.

No SUMO, GPU or ssh: the adapter is exercised against fabricated C4 exports and
fake envs/waves. Covers the W4 requirements: SHA-bound executable path on the
SAME paired worlds, lane commands only for CAVs, own-phase charging, receipt
schema and a real consumer/dispatch test through launch_evaluation.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from . import clean3_restart
from . import laneon_baseline as lb
from . import launch_evaluation as le
from . import protocol, rerun_cpus as rc
from .exact_grade import verify_sealed


def export_checkpoint(path, *, lane_head=True, trainer='mappo', fmt='c4_dreamer/3',
                      bias=None):
    """A real C4 export payload in the aecd7d70 schema (or a deliberately wrong one)."""
    from wms.control.lanes import LaneRules
    from wms.control.mappo.agent import export_payload
    from wms.control.policy import CavPolicy, RsuGroupEncoder

    torch.manual_seed(11)
    policy = CavPolicy(lane_head=lane_head)
    payload = export_payload(policy, RsuGroupEncoder(), LaneRules(), iteration=170)
    if not lane_head:
        payload['lane_head'] = False
    if trainer != 'mappo':
        payload['trainer'] = trainer
    if fmt != 'c4_dreamer/3':
        payload['format'] = fmt
    if bias is not None:
        payload['policy']['lane_head.2.bias'] = torch.tensor(bias)
    torch.save(payload, path)
    return Path(path)


def worlds96():
    return protocol.worlds('train', 0)+protocol.worlds('train', 1)[:24]


def registration96():
    return rc.final_world_registration(dict(final_evaluation=dict(role='train',
        worlds=[dict(world_id=str(w['seed']), world=deepcopy(w)) for w in worlds96()])))


def state(s, lane, v=None, cav=None, present=None):
    k = len(s)
    v = [20.0]*k if v is None else v
    cav = [True]*k if cav is None else cav
    present = [True]*k if present is None else present
    return {'s_m': torch.tensor([s], dtype=torch.float32),
            'v_mps': torch.tensor([v], dtype=torch.float32),
            'a_mps2': torch.zeros(1, k), 'lane': torch.tensor([lane]),
            'present': torch.tensor([present]),
            'is_cav': torch.tensor([cav]) & torch.tensor([present]),
            'length_m': torch.full((1, k), 5.0)}


class FakeEnv:
    def __init__(self, frame):
        self.frame, self.steps = frame, []

    def state(self):
        return self.frame

    def step(self, accel, lanes=None, fallback='native'):
        import numpy as np
        lanes = None if lanes is None else np.asarray(lanes)
        self.steps.append((np.asarray(accel).copy(),
                           None if lanes is None else lanes.copy()))
        if lanes is not None:                     # SUMO executes feasible requests
            for i, target in enumerate(lanes.reshape(-1)):
                if target >= 0:
                    self.frame['lane'][0, i] = int(target)
        return self.frame, dict(interval={}, lane_commands={'issued': 2}, done=False)


# ---------------------------------------------------------------- adapter: load / refuse

def test_load_executor_accepts_sha_bound_export_and_refuses_wrong_sha(tmp_path):
    path = export_checkpoint(tmp_path/'mappo_v5_final.pt')
    sha = rc.sha(path)
    executor, meta = lb.load_executor(path, expected_sha256=sha)
    assert executor.has_lane_head and executor.deterministic
    assert executor.lane_rules.dwell_s == 3.0 and executor.accel_max == 2.6
    assert meta == dict(format='c4_dreamer/3', trainer='mappo', lane_head=True,
                        iteration=170, file_sha256=sha, accel_min_mps2=-3.0,
                        accel_max_mps2=2.6, lane_change_mode=lb.LANE_CHANGE_MODE)
    # the production pin is the exact aecd7d70 export SHA (feasibility S-LANEMAPPO)
    assert lb.CHECKPOINT_SHA256 == 'aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564'
    with pytest.raises(ValueError, match='SHA differs'):
        lb.load_executor(path, expected_sha256='0'*64)
    (tmp_path/'tampered.pt').write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError, match='SHA differs'):
        lb.load_executor(tmp_path/'tampered.pt', expected_sha256=sha)
    with pytest.raises(ValueError, match='SHA differs'):
        lb.staged_checkpoint(dict(checkpoint=str(tmp_path/'tampered.pt')))
    with pytest.raises(ValueError, match='SHA differs'):
        lb.staged_checkpoint(dict(checkpoint=str(path)))  # default pin: aecd7d70 only


def test_load_executor_refuses_lane_stripped_or_foreign_exports(tmp_path):
    stripped = export_checkpoint(tmp_path/'speed_only.pt', lane_head=False)
    with pytest.raises(ValueError, match='lane head'):
        lb.load_executor(stripped, expected_sha256=rc.sha(stripped))
    foreign = export_checkpoint(tmp_path/'dreamer.pt', trainer='dreamer')
    with pytest.raises(ValueError, match='mappo'):
        lb.load_executor(foreign, expected_sha256=rc.sha(foreign))
    old = export_checkpoint(tmp_path/'old.pt', fmt='c4_dreamer/2')
    with pytest.raises(ValueError, match='c4_dreamer/3'):
        lb.load_executor(old, expected_sha256=rc.sha(old))


# ---------------------------------------------------------------- adapter: same worlds

def test_specs_use_the_same_paired_worlds_windows_and_panel_as_the_arms(tmp_path):
    from . import planner_eval
    registration = registration96()
    entry = dict(clean3_restart.LANE_ON_BASELINE, checkpoint=str(tmp_path/'aecd7d70.pt'))
    pairs = lb.specs(registration, entry)
    contract = deepcopy(rc.FINAL_EVALUATION)
    checkpoints = {a: dict(path=f'/policies/{a}_024.pt', file_sha256='a'*64, update=24,
                           components=contract['checkpoint_components'][a]) for a in ('A', 'B', 'R')}
    arms = planner_eval.prepare_track2(registration=registration, checkpoints=checkpoints,
                                       contract=contract)
    assert len(pairs) == len(arms)//3 == lb.EPISODES
    assert sum(spec['sim_s'] for spec, _ in pairs) == 96*1200 == contract['sim_s']//3
    by_world = {}
    for spec, _ in arms:
        by_world.setdefault(spec['world_id'], {})[spec['arm']] = spec
    for spec, _ in pairs:
        a = by_world[spec['world_id']]['A']
        assert spec['world'] == a['world'] and spec['world_id'] == a['world_id']
        assert spec['sim_s'] == a['sim_s'] == 1200
        assert spec['window'] == a['window'] == dict(start_s=0, end_s=1200)
        assert spec['final_worlds_sha256'] == a['final_worlds_sha256']
        assert spec['branch'] == 'shared_final'
        assert spec['baseline'] == spec['arm'] == lb.NAME
        assert spec['lane_change_mode'] == 513
        assert spec['checkpoint'] == dict(path=entry['checkpoint'],
                                          file_sha256=lb.CHECKPOINT_SHA256)
    with pytest.raises(ValueError, match='96'):
        from .exact_grade import _seal
        lb.specs(_seal(dict(registration, worlds=registration['worlds'][:2]),
                       'final_worlds_sha256'), entry)


# ---------------------------------------------------------------- adapter: CAV-only lanes

def test_lane_targets_flow_through_the_env_lane_path_for_cavs_only(tmp_path):
    # Biased 3-way head: stay -5, left/right +5 -> every feasible CAV changes lane.
    path = export_checkpoint(tmp_path/'biased.pt', bias=[-5.0, 5.0, 5.0])
    executor, _ = lb.load_executor(path, expected_sha256=rc.sha(path))
    frame = state([-500.0, -950.0, -300.0, -700.0], [0, 1, 1, 1],
                  cav=[True, True, False, False], present=[True, True, True, False])
    env = FakeEnv(frame)
    controller = lb.LaneOnController(executor)
    info = controller.step(env)
    accel, lanes = env.steps[0]
    assert lanes is not None                      # lanes tensor supplied EVERY tick
    assert lanes.tolist() == [[1, 0, -1, -1]]     # CAV targets only; HDV/absent stay
    assert math.isnan(accel[0, 2]) and math.isnan(accel[0, 3])
    assert controller.ticks == 1 and controller.lane_commands['issued'] == 2
    assert info['lane_commands'] == {'issued': 2}
    # an all-stay tick still keeps the policy lane regime on (tensor, not None)
    controller.step(env)
    assert env.steps[1][1] is not None and (env.steps[1][1] == -1).all()


def test_controller_refuses_lane_commands_on_non_cav_slots_or_off_grid_targets():
    class BadExecutor:
        def reset(self):
            pass

        def act_with_lane(self, frame):
            import numpy as np
            return np.array([[1., 2., 3.]]), np.array([[0, 1, 1]])
    with pytest.raises(ValueError, match='non-CAV'):
        lb.LaneOnController(BadExecutor()).step(FakeEnv(state([-500.0, -400.0, -300.0],
                                                              [0, 1, 1], cav=[True, True, False])))
    class OffGrid(BadExecutor):
        def act_with_lane(self, frame):
            import numpy as np
            return np.array([[1., np.nan, np.nan]]), np.array([[7, -1, -1]])
    with pytest.raises(ValueError, match='lane grid'):
        lb.LaneOnController(OffGrid()).step(FakeEnv(state([-500.0, -400.0], [0, 1],
                                                          cav=[True, False])))


# ---------------------------------------------------------------- receipts + analysis

def spec0(entry_path='/staged/mappo_v5_final.pt'):
    registration = registration96()
    return lb.specs(registration, dict(clean3_restart.LANE_ON_BASELINE,
                                       checkpoint=entry_path))[0][0]


def outcome(J=54123.5):
    return dict(J_veh_s=J, safety=dict(collisions=0, teleports=0, hard_brakes=2,
                                       ttc_events=1), lane_commands=dict(issued=9, completed=7),
                executor='tools.coopmwm.laneon_baseline', result_ref='work/RESULT.json')


def test_receipt_schema_seals_and_refuses_tampering_or_unknown_keys():
    spec = spec0()
    record = lb.make_receipt(spec, outcome())
    lb.validate_receipt(record, spec)
    verify_sealed(record, 'receipt_sha256')
    assert record['schema'] == lb.RECEIPT_SCHEMA and record['baseline'] == lb.NAME
    assert record['phase'] == lb.PHASE and record['checkpoint_sha256'] == lb.CHECKPOINT_SHA256
    assert record['lane_change_mode'] == 513 and record['episode_end_s'] == 1200
    assert record['seed'] == spec['world']['seed']
    tampered = dict(record, J_veh_s=record['J_veh_s']-1.0)
    with pytest.raises(ValueError, match='receipt_sha256 mismatch'):
        lb.validate_receipt(tampered, spec)
    with pytest.raises(ValueError, match='unregistered lane-on outcome keys'):
        lb.make_receipt(spec, dict(outcome(), extra='x'))
    with pytest.raises(ValueError, match='safety keys'):
        lb.make_receipt(spec, dict(outcome(), safety={'collisions': 0}))
    with pytest.raises(ValueError, match='nonnegative int'):
        lb.make_receipt(spec, dict(outcome(), safety=dict(collisions=0, teleports=0,
                                                          hard_brakes=-1, ttc_events=0)))
    with pytest.raises(ValueError, match='lane_commands'):
        lb.make_receipt(spec, dict(outcome(), lane_commands={'issued': 1.5}))
    other = lb.make_receipt(spec, outcome())
    from .exact_grade import _seal
    forged = _seal({k: v for k, v in other.items() if k != 'receipt_sha256'} | dict(extra=1),
                   'receipt_sha256')
    with pytest.raises(ValueError, match='receipt keys'):
        lb.validate_receipt(forged, spec)
    second = lb.specs(registration96(), dict(clean3_restart.LANE_ON_BASELINE,
                                             checkpoint='/staged/aecd7d70.pt'))[1][0]
    with pytest.raises(ValueError, match='world_id'):
        lb.validate_receipt(other, second)


def test_analysis_pairs_descriptively_against_every_arm_and_refuses_incomplete_panels():
    registration = registration96()
    entry = dict(clean3_restart.LANE_ON_BASELINE, checkpoint='/staged/aecd7d70.pt')
    pairs = lb.specs(registration, entry)
    laneon = [dict({k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world', 'checkpoint',
                                                   'window', 'final_worlds_sha256', 'baseline',
                                                   'lane_change_mode')},
                   status='graded', J_veh_s=1000.0+int(spec['world']['seed'] % 7)+1,
                   safety=dict(collisions=0, teleports=0, hard_brakes=3, ttc_events=2),
                   lane_commands=dict(issued=1)) for spec, _ in pairs]
    arm_records = []
    for spec, _ in pairs:
        for arm, offset in (('A', 4), ('B', 2), ('R', 3)):
            arm_records.append(dict(arm=arm, world_id=spec['world_id'],
                world=deepcopy(spec['world']), checkpoint={'file_sha256': 'a'*64},
                window=dict(start_s=0, end_s=1200),
                final_worlds_sha256=spec['final_worlds_sha256'], status='graded',
                J_veh_s=1000.0+int(spec['world']['seed'] % 7)+offset,
                safety=dict(collisions=0, teleports=0, hard_brakes=1, ttc_events=1)))
    contract = deepcopy(rc.FINAL_EVALUATION)
    analysis = lb.analyse(registration=registration, records=laneon,
                          arm_records=arm_records, contract=contract)
    assert analysis['descriptive'] is True and analysis['phase'] == lb.PHASE
    assert analysis['comparisons'] == sorted(f'{lb.NAME}-{a}' for a in ('A', 'B', 'R'))
    for arm in contract['arms']:
        ci = analysis['paired_ci'][f'{lb.NAME}-{arm}']
        assert ci['n'] == 96 and ci['low'] <= ci['high']
        assert ci['mean'] == pytest.approx(1.0-offset_mean(arm))
        assert ci['native']['n_worlds'] == 96
    assert analysis['safety'][lb.NAME]['hard_brakes'] == 96*3
    assert analysis['safety']['A']['ttc_events'] == 96
    verify_sealed(analysis, 'analysis_sha256')
    with pytest.raises(ValueError, match='incomplete lane-on panel'):
        lb.analyse(registration=registration, records=laneon[:-1], arm_records=arm_records,
                   contract=contract)
    with pytest.raises(ValueError, match='arm records incomplete'):
        lb.analyse(registration=registration, records=laneon,
                   arm_records=arm_records[:-1], contract=contract)
    wrong = [dict(laneon[0], lane_change_mode=512)]+laneon[1:]
    with pytest.raises(ValueError, match='binding differs'):
        lb.analyse(registration=registration, records=wrong, arm_records=arm_records,
                   contract=contract)
    foreign = [dict(laneon[0], checkpoint={'file_sha256': '0'*64})]+laneon[1:]
    with pytest.raises(ValueError, match='checkpoint SHA'):
        lb.analyse(registration=registration, records=foreign, arm_records=arm_records,
                   contract=contract)


def offset_mean(arm):
    return {'A': 4, 'B': 2, 'R': 3}[arm]


# ---------------------------------------------------------------- registration + budget

def test_registered_entry_binds_the_quoted_baseline_and_its_own_phase(tmp_path):
    assert lb.PHASE == 'v2v_final_laneon_mappo_aecd7d70'
    assert lb.PHASE_ALLOCATED_SIM_S == 133_200 == math.ceil(96*1200*1.15/1200)*1200
    phases = clean3_restart.PHASES
    assert phases['v2v_final'] == 398_400          # A/B/R base 3*96*1200 with 15% margin
    assert phases[lb.PHASE] == 133_200             # lane-on base 96*1200 with 15% margin
    assert phases['v2v_final'] != phases[lb.PHASE]  # never merged into the A/B/R phase
    assert sum(phases.values())+clean3_restart.G1_CAP+clean3_restart.G4_CAP == \
        clean3_restart.QUOTE_TOTAL == 3_680_400
    accepted = {name: 'a'*64 for name in
                ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'G1', 'G2', 'G4', 'G5')}
    budget = clean3_restart.build(clean1_rows=[], clean2_rows=[], g1_rows=[],
                                  mining_rows=[], prefixes={}, accepted=accepted)
    assert budget['phases'][lb.PHASE] == 133_200 and budget['phases']['v2v_final'] == 398_400
    entry = dict(clean3_restart.LANE_ON_BASELINE, checkpoint=str(tmp_path/'aecd7d70.pt'))
    assert lb.registered_entry(dict(extra_final_baselines=[entry])) == entry
    with pytest.raises(ValueError, match='not registered'):
        lb.registered_entry(dict(extra_final_baselines=[]))
    with pytest.raises(ValueError, match='checkpoint path'):
        lb.registered_entry(dict(extra_final_baselines=[dict(clean3_restart.LANE_ON_BASELINE)]))
    with pytest.raises(ValueError, match='registration differs'):
        lb.registered_entry(dict(extra_final_baselines=[
            dict(entry, sha256='0'*64)]))
    with pytest.raises(ValueError, match='registration differs'):
        lb.registered_entry(dict(extra_final_baselines=[entry, entry]))


# ---------------------------------------------------------------- consumer / dispatch

def laneon_cfg(root, checkpoint, registered=True):
    cfg = dict(run_root=str(root), final_evaluation=deepcopy(rc.FINAL_EVALUATION))
    if registered:
        cfg['extra_final_baselines'] = [dict(clean3_restart.LANE_ON_BASELINE,
                                             checkpoint=str(checkpoint))]
    return cfg


def paired_root(tmp_path, name='run'):
    root = tmp_path/name
    (root/'evaluation').mkdir(parents=True)
    registration = dict(role='train', worlds=[dict(world_id=str(w['seed']),
                                                   world=deepcopy(w)) for w in worlds96()])
    (root/'ISOLATED_REGISTRATION.json').write_text(json.dumps(
        dict(final_evaluation=registration)))
    (root/'evaluation'/'LESSON_u024.json').write_text('{}')   # lessons run elsewhere
    for update in (8, 24):
        for arm in ('A', 'B', 'R'):
            (root/f'{arm}_checkpoint_{update:03}.pt').write_bytes(f'{arm}-u{update}'.encode())
    return root


def fake_wave_factory(calls):
    def fake_wave(execution, pairs, phase, prefix, fixed=False):
        calls.append((phase, prefix, fixed, len(pairs)))
        out = []
        for spec, _ in pairs:
            row = dict({k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world',
                                                       'checkpoint', 'window',
                                                       'final_worlds_sha256')},
                       status='graded',
                       J_veh_s=1000.0+int(spec['world']['seed'] % 7)+{'A': 4, 'B': 2,
                                                                      'R': 3}.get(spec['arm'], 1),
                       safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0))
            if fixed == 'laneon':
                row.update(baseline=spec['baseline'],
                           lane_change_mode=spec['lane_change_mode'],
                           lane_commands=dict(issued=0),
                           executor='fake', result_ref='work/RESULT.json')
            out.append(row)
        return out
    return fake_wave


def test_scheduled_u24_consumes_the_baseline_on_the_same_worlds_and_own_phase(
        tmp_path, monkeypatch):
    root = paired_root(tmp_path)
    staged = tmp_path/'staged_aecd7d70.pt'
    staged.write_bytes(b'frozen-export')
    cfg = laneon_cfg(root, staged)
    calls = []
    monkeypatch.setattr(le, '_wave', fake_wave_factory(calls))
    monkeypatch.setattr(lb, 'staged_checkpoint', lambda entry: staged)
    value = le.scheduled(cfg, 24)
    assert calls == [('v2v_final', 'paired_u024', True, 288),
                     (lb.PHASE, 'laneon_u024', 'laneon', 96)]
    verify_sealed(value, 'receipt_sha256')
    assert value['baseline']['name'] == lb.NAME and value['update'] == 24
    assert value['lane_change_mode'] == 513
    assert len(value['receipts']) == 96
    for record in value['receipts']:
        lb.validate_receipt(record)
    analysis = value['analysis']
    assert analysis['comparisons'] == sorted(f'{lb.NAME}-{a}' for a in ('A', 'B', 'R'))
    for arm in ('A', 'B', 'R'):
        ci = analysis['paired_ci'][f'{lb.NAME}-{arm}']
        assert ci['n'] == 96 and ci['mean'] == pytest.approx(1.0-offset_mean(arm))
    verify_sealed(analysis, 'analysis_sha256')
    # idempotent: the durable receipts are read back, no further waves
    assert le.scheduled(cfg, 24)['receipt_sha256'] == value['receipt_sha256']
    assert len(calls) == 2
    # interim u8 never runs the extra baseline
    interim_root = paired_root(tmp_path, 'interim')
    interim = laneon_cfg(interim_root, staged)
    calls.clear()
    assert le.scheduled(interim, 8)['update'] == 8
    assert calls == [('v2v_interim', 'paired_u008', True, 288)]
    assert not (interim_root/'evaluation'/'LANEON_u024.json').exists()
    # unregistered configurations keep the unchanged contract
    plain_root = paired_root(tmp_path, 'plain')
    plain = laneon_cfg(plain_root, staged, registered=False)
    calls.clear()
    assert le.scheduled(plain, 24)['update'] == 24
    assert calls == [('v2v_final', 'paired_u024', True, 288)]
    assert not (plain_root/'evaluation'/'LANEON_u024.json').exists()


def test_laneon_final_requires_the_registered_checkpoint_before_any_charge(
        tmp_path, monkeypatch):
    root = paired_root(tmp_path)
    staged = tmp_path/'staged_aecd7d70.pt'
    staged.write_bytes(b'not-the-export')
    cfg = laneon_cfg(root, staged)
    calls = []
    monkeypatch.setattr(le, '_wave', fake_wave_factory(calls))
    with pytest.raises(ValueError, match='SHA differs'):
        le.scheduled(cfg, 24)
    assert [c[:3] for c in calls] == [('v2v_final', 'paired_u024', True)]
    assert not (root/'evaluation'/'LANEON_u024.json').exists()


def test_wave_dispatches_the_laneon_worker_and_reserves_its_own_phase(
        tmp_path, monkeypatch):
    from . import launch_training

    class FakeFuture:
        def __init__(self, value):
            self.value = value

        def result(self, timeout=None):
            return self.value

    submitted = []

    class FakePool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            submitted.append((fn, args))
            spec = args[0]
            return FakeFuture(dict(
                {k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world', 'checkpoint',
                                                'window', 'final_worlds_sha256', 'baseline',
                                                'lane_change_mode')},
                status='graded', J_veh_s=1.0,
                safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0),
                lane_commands=dict(issued=0)))

    reserved = []
    monkeypatch.setattr(le.production, 'backend_config', lambda c: c)
    monkeypatch.setattr(le.p1, '_reserve', lambda cfg, jobs, phase: reserved.append((jobs, phase)))
    monkeypatch.setattr(launch_training, 'health', lambda root: None)
    monkeypatch.setattr(le, 'ProcessPoolExecutor', FakePool)
    monkeypatch.setattr(le, 'episode', lambda *a: pytest.fail('A/B/R worker must not run'))
    registration = registration96()
    entry = dict(clean3_restart.LANE_ON_BASELINE, checkpoint='/staged/aecd7d70.pt')
    pairs = lb.specs(registration, entry)
    cfg = laneon_cfg(tmp_path/'run', entry['checkpoint'])
    results = le._wave(cfg, pairs, lb.PHASE, 'laneon_u024', fixed='laneon')
    assert len(results) == 96 and all(r['status'] == 'graded' for r in results)
    jobs, phase = reserved[0]
    assert phase == lb.PHASE and len(jobs) == 96
    assert [job['id'] for job in jobs] == [f'laneon_u024_{i:04}' for i in range(96)]
    assert all(job['sim_s'] == 1200 for job in jobs)
    assert sum(job['sim_s'] for job in jobs) == 115_200
    for (fn, args), job in zip(submitted, jobs):
        spec, native, work = args
        assert fn is lb.episode            # the SHA-bound lane-on worker, never the arm one
        assert work == Path(cfg['run_root'])/'evaluation'/job['id']
        assert Path(spec['checkpoint']['path']) == Path(entry['checkpoint'])
        assert spec['checkpoint']['file_sha256'] == lb.CHECKPOINT_SHA256
