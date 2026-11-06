"""G4 branch workers bind the frozen runtime exactly like training-wave workers.

clean3 (2026-11-06 12:25, root clean3_r6) faulted in u1: the B arm owner ran the G4 branch
IN-PROCESS, where ArmRuntime.collect's seed scope had already imported traffic_shepherd.c4
from the owner's PYTHONPATH copy, so g5_runtime.install refused with the guard message
below. Here the cpuV frozen runtime is faked on disk and leverprobe's setup is faked with
the real prepend order; accel_plant.bind_runtime, the seed scope/guard, g5_runtime.install
(its hash and package-path guard), the production pools and run_branch are all real.
No SUMO, no physics.
"""
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from . import accel_plant, accel_runner, g4_runtime, g5_runtime, runner, v2v_runtime
from . import sparse_synergy_credit as credit

FAULT = 'traffic_shepherd.c4 must resolve to the hash-verified frozen runtime'
REPO = Path(__file__).resolve().parents[2]


class Bound(Exception):
    """Raised at the first policy construction, i.e. right after the runtime binding."""


def make_layout(tmp):
    def package(root, mark):
        c4 = root/'traffic_shepherd'/'c4'
        c4.mkdir(parents=True)
        (root/'traffic_shepherd'/'__init__.py').write_text(f'MARK = {mark!r}\n')
        (root/'traffic_shepherd'/'seeds.py').write_text(
            'def require_development_case(seed, loading, role):\n    return None\n')
        (c4/'__init__.py').write_text(f'MARK = {mark!r}\n')
        (c4/'seeds.py').write_text('def check_case(seed, demand_vph):\n    return None\n')
        (c4/'env.py').write_text(f'MARK = {mark!r}\n')
    frozen, owner = tmp/'frozen', tmp/'owner_pythonpath'
    package(frozen/'runtime', 'frozen')
    (frozen/'runtime'/'src').mkdir()
    package(owner, 'owner')
    overlay = tmp/'g5_overlay.pysrc'
    overlay.write_text("MARK = 'frozen+g5'\n")
    run = tmp/'run'
    run.mkdir()
    (run/'OWNER_APPROVAL.json').write_text(json.dumps(dict(approved=True)))
    return dict(frozen=str(frozen), owner=str(owner), overlay=str(overlay), run_root=str(run))


def install_fake_runtime(layout):
    """Worker/subprocess side: fake frozen runtime + leverprobe setup (real prepend order)."""
    os.environ[g5_runtime.FLAG] = '1'
    g5_runtime.BASE_SHA256 = g5_runtime.normalized_sha(Path(layout['frozen'])/g5_runtime.BASE_REL)
    g5_runtime.OVERLAY = Path(layout['overlay'])
    g5_runtime.OVERLAY_SHA256 = g5_runtime.normalized_sha(layout['overlay'])
    backend = types.ModuleType('tools.leverprobe.backend')

    def setup(runtime, checkpoint):
        sys.path[:0] = [str(Path(runtime)/'runtime'/'src'), str(Path(runtime)/'runtime')]
    backend.setup = setup
    package = types.ModuleType('tools.leverprobe')
    package.backend = backend
    sys.modules['tools.leverprobe'], sys.modules['tools.leverprobe.backend'] = package, backend

    def first_policy(*args, **kwargs):
        raise Bound(signature())
    v2v_runtime.from_weights = first_policy
    runner.admission = lambda cfg: None  # /proc load gate, not a binding step


def signature():
    import traffic_shepherd.c4 as c4
    env = sys.modules.get('traffic_shepherd.c4.env')
    seeds = sys.modules.get('traffic_shepherd.c4.seeds')
    return dict(c4=str(Path(list(c4.__path__)[0]).resolve()),
        seeds=seeds and str(Path(seeds.__file__).resolve()),
        env_overlay=getattr(env, 'G5_RUNTIME_OVERLAY_SHA256', None), env_mark=getattr(env, 'MARK', None),
        front=[str(Path(p).resolve()) for p in sys.path[:2]], pid=os.getpid())


def frozen_signature(layout):
    runtime = Path(layout['frozen']).resolve()/'runtime'
    return dict(c4=str(runtime/'traffic_shepherd'/'c4'), seeds=str(runtime/'traffic_shepherd'/'c4'/'seeds.py'),
        env_overlay=g5_runtime.normalized_sha(layout['overlay']), env_mark='frozen+g5',
        front=[str(runtime/'src'), str(runtime)])


def native_job(layout, name):
    from .accel_protocol import worlds
    return dict(id=name, path=str(Path(layout['run_root'])/'jobs'/name), world=worlds('train')[0],
                role='train', runtime=layout['frozen'], checkpoint='mappo.pt')


def bind_in_worker(layout, job):
    install_fake_runtime(layout)
    accel_plant.bind_runtime(job)
    return signature()


def training_worker(job):
    """Child of accel_runner.wave's own pool: the REAL accel_plant.episode."""
    install_fake_runtime(job['probe_layout'])
    try:
        accel_plant.episode(job)
    except Bound as bound:
        return dict(id=job['id'], world=job['world'], arm=job['arm'], binding=bound.args[0])
    raise AssertionError('training worker did not reach policy construction')


def branch_worker(job, cfg):
    """Child of G4's production pool: the REAL g4_runtime.run_branch (real capture load)."""
    install_fake_runtime(cfg['probe_layout'])
    try:
        g4_runtime.run_branch(job, cfg)
    except Bound as bound:
        return dict(binding=bound.args[0])
    raise AssertionError('G4 branch worker did not reach policy construction')


OWNER_SCRIPT = r'''
import json, sys
layout = json.loads(sys.argv[1])
sys.path.insert(0, layout['owner'])  # cpuV owner PYTHONPATH: runtime_src precedes e3final
from tools.coopmwm import test_g4_worker_binding_cpu as t
print('RESULT ' + json.dumps(t.owner_process(layout)))
'''


def owner_process(layout):
    """An arm owner: collect's seed scope imports traffic_shepherd before any G4 branch."""
    from .accel_protocol import install_seed_scope
    install_fake_runtime(layout)
    install_seed_scope()  # ArmRuntime.collect, in the owner process
    import traffic_shepherd.c4 as c4
    owner_c4 = str(Path(list(c4.__path__)[0]).resolve())
    job = native_job(layout, 'g4_probe')
    # Fixed path: the same binding in a worker of G4's production (spawned) pool.
    with g4_runtime.worker_pool(dict(workers=1)) as pool:
        spawned = pool.submit(bind_in_worker, layout, job).result(timeout=600)
    # Faulted path: the binding inside the owner process itself.
    try:
        accel_plant.bind_runtime(job)
        in_process = None
    except ValueError as error:
        in_process = str(error)
    return dict(owner_c4=owner_c4, in_process_error=in_process, spawned=spawned)


def test_clean3_fault_reproduced_in_owner_process_and_fixed_by_spawned_g4_worker(tmp_path):
    layout = make_layout(tmp_path)
    env = dict(os.environ)
    env.pop(g5_runtime.FLAG, None)
    env['PYTHONPATH'] = os.pathsep.join([str(REPO)]+[p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p])
    result = subprocess.run([sys.executable, '-B', '-c', OWNER_SCRIPT, json.dumps(layout)], cwd=str(REPO),
                            env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-4000:]
    value = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith('RESULT ')))
    assert value['owner_c4'] == str((Path(layout['owner'])/'traffic_shepherd'/'c4').resolve())
    assert value['in_process_error'] == FAULT  # exactly the clean3 u1 FAULT.json error
    spawned = value['spawned']
    spawned.pop('pid')
    assert spawned == frozen_signature(layout)


def capture_job(layout):
    """A real immutable G4 capture on disk and its branch job (load_capture verifies all hashes)."""
    from .accel_protocol import worlds
    track2 = dict(g4_synergy_credit=dict(enabled=True, registration=credit.registration(),
                                         phase_id='g4-binding-cpu', first_update=1))
    episode_id = 'probe_u001_B_episode'
    payload = dict(time_s=600, job=dict(id=episode_id, world=worlds('train')[0], arm='B',
            runtime=layout['frozen'], checkpoint='mappo.pt', noise_seed=1, accel_bounds=[-3., 2.6],
            warm_history=False, weights={'actor.0.weight': torch.zeros(2, 2)}, track2_config=track2,
            track2_state={'layer': torch.ones(3)}),
        prefixes={'300': dict(t_s=300)}, samples={}, generator_state=torch.Generator().get_state(),
        memory=None, reports={}, previous_frame=None, intent_board={}, trigger_intent_board={})
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    opportunity = credit.digest(['binding-probe'])
    path = Path(layout['run_root'])/'jobs'/episode_id/'g4_captures'/f'{opportunity}.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    digest = g4_runtime.state_digest
    capture = dict(stage='pre_action', time_s=600, snapshot_sha256=g4_runtime._bytes_sha(data),
        prefix_sha256=credit.digest(payload['prefixes']),
        crn_sha256=digest({k: payload[k] for k in ('generator_state', 'memory', 'previous_frame',
                                                    'intent_board', 'trigger_intent_board')}),
        policy_sha256=digest(payload['job']['track2_state']),
        base_policy_sha256=digest(payload['job']['weights']))
    job = dict(start=dict(episode_id=episode_id, opportunity_id=opportunity, arm='B', time_s=600,
                          capture=capture), binding=credit.settings(track2), verification_seed=1)
    return dict(job, job_sha256=credit.digest(job)), payload['job']['world']


def test_g4_branch_worker_and_training_worker_share_pool_and_runtime_binding(tmp_path, monkeypatch):
    layout = make_layout(tmp_path)
    monkeypatch.delenv(g5_runtime.FLAG, raising=False)
    pools = []

    class Recorded(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            pools.append((args, kwargs['max_workers'], kwargs['mp_context'].get_start_method()))
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(accel_runner, 'ProcessPoolExecutor', Recorded)
    monkeypatch.setattr(g4_runtime, 'ProcessPoolExecutor', Recorded)
    workers = 2
    # Training wave: real accel_runner.wave job construction, pool and submission path.
    monkeypatch.setattr(accel_runner, 'reserve', lambda root, jobs, phase: None)
    monkeypatch.setattr(accel_runner, 'ledger', lambda root: [])
    monkeypatch.setattr(runner, 'admission', lambda cfg: None)
    monkeypatch.setattr(accel_plant, 'episode', training_worker)
    job, world = capture_job(layout)
    binding = dict(workers=workers, runtime=layout['frozen'], checkpoint='mappo.pt',
                   accel_bounds=[-3., 2.6], gpu_deadline=9999999999)
    trained = accel_runner.wave(Path(layout['run_root']), binding, {'B': torch.nn.Linear(1, 1)}, [world],
                                'train', 'probe_u001', 'train', False, dict(v2v=True, probe_layout=layout))
    # G4: the production dispatch -> spawned pool -> real run_branch (real capture load).
    monkeypatch.setattr(g4_runtime, 'run_branch', branch_worker)
    cfg = dict(run_root=layout['run_root'], workers=workers, probe_layout=layout)
    branched = g4_runtime.dispatch(cfg, [job])
    # Identical worker path: the same spawned pool construction for both.
    assert pools == [((), workers, 'spawn'), ((), workers, 'spawn')]
    signatures = [trained[0]['binding'], branched[0]['binding']]
    assert all(s.pop('pid') != os.getpid() for s in signatures)  # ran in worker processes
    # Identical binding, and it is the hash-verified frozen runtime + bound G5 env.
    assert signatures[0] == signatures[1] == frozen_signature(layout)


def test_repeated_binding_in_a_reused_worker_is_idempotent(tmp_path, monkeypatch):
    layout = make_layout(tmp_path)
    monkeypatch.delenv(g5_runtime.FLAG, raising=False)
    job = native_job(layout, 'g4_probe')
    with g4_runtime.worker_pool(dict(workers=1)) as pool:
        first = pool.submit(bind_in_worker, layout, job).result(timeout=600)
        second = pool.submit(bind_in_worker, layout, job).result(timeout=600)
    assert first['pid'] == second['pid']  # pool-reused process, like training waves
    assert first == second


def test_production_coordinator_defaults_to_spawned_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(g4_runtime.tb, 'reservation_limits', lambda cfg: (None, 0, {}))
    coordinator = g4_runtime.Coordinator(dict(workers=3))
    assert coordinator.worker is None  # measure() then uses dispatch, never run_branch in-process
    monkeypatch.setattr(g4_runtime, 'worker_pool', lambda cfg: calls.append(cfg) or pytest.fail('no jobs'))
    assert g4_runtime.dispatch(dict(workers=3), []) == [] and not calls


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
