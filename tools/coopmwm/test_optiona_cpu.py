"""Option A CPU tests (no SUMO, no ssh, no GPU): early R start, measure-time equivalence guard, atomic TTC cache,
RemotePool with a fake transport (ssh drop, timeout, landing mismatch, existing remote dir, dead worker, setup
error -> RemoteWaveLost FAULT without replacement or extra charge), replica ledger refusal, STOP/PAUSE mirroring,
cleanup only after LEARNING_MONITOR + re-verify, and flags-off identity."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch

from . import accel_runner, g4_runtime as g4, launch_training as launch
from . import practice_starts as ps, remote_wave as rwave, remote_worker as rw, safety_diagnosis as sd
from . import track2_backend as tb
from .exact_grade import _seal
from .test_g4_runtime_cpu import toy  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401
from .v2v_runner import patched

EARLY = g4.EARLY_ENV


# ------------------------------------------------------------------ helpers

def _norm(value, roots):
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        return ['tensor', str(t.dtype), list(t.shape), hashlib.sha256(t.numpy().tobytes()).hexdigest()]
    if isinstance(value, np.ndarray):
        return ['ndarray', str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {(k if isinstance(k, str) else 'k:'+repr(k)): _norm(v, roots) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_norm(v, roots) for v in value]
    if isinstance(value, str):
        for root in roots:
            value = value.replace(root, '<ROOT>')
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('unexpected job value type '+type(value).__name__)


def _job_bytes(jobs, root):
    return json.dumps(_norm(jobs, [root]), sort_keys=True).encode()


@pytest.fixture
def practice_toy(toy, monkeypatch):  # noqa: F811
    """The real paired launcher (toy physics) with practice ON and 0-row plans (live u1 shape)."""
    cfg = toy.cfg
    cfg.update(practice_starts=True, practice_repeat_cap=2, runtime='fake', checkpoint='fake',
               accel_bounds=[-3., 2.6], gpu_deadline=0)
    counts, captured, gate = [], [], dict(event=None)

    def plan_for_update(root, arm, update, *, count, **kwargs):
        path = ps.plan_path(root, arm, update)
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        counts.append((arm, int(count)))
        record = _seal(dict(arm=arm, update=int(update), rows=[], breadth={}, count=int(count)),
                       'practice_plan_sha256')
        launch.write(path, record)
        return record
    monkeypatch.setattr(ps, 'plan_for_update', plan_for_update)
    toy_wave = tb._wave

    def recording(cfg_, policies, worlds, ident, extras, **kwargs):
        if 'R' in policies:
            with patched(accel_runner, dict(ledger=lambda r: [])):
                jobs, _ = accel_runner.wave_jobs(Path(cfg_['run_root']), cfg_, policies, worlds, 'train', ident,
                                                 'track2_train', False, extras)
            captured.append(dict(jobs=jobs, root=cfg_['run_root'],
                                 b_input=g4.input_path(cfg_, 'B', 1).exists()))
            out = toy_wave(cfg_, policies, worlds, ident, extras, **kwargs)
            if gate['event'] is not None:
                gate['event'].set()
            return out
        if gate['event'] is not None:
            # early order proof: B does not even collect until R has collected
            assert gate['event'].wait(120), 'early R never collected before B'
        return toy_wave(cfg_, policies, worlds, ident, extras, **kwargs)
    monkeypatch.setattr(tb, '_wave', recording)
    toy.counts, toy.captured, toy.gate = counts, captured, gate
    return toy


def _run_pair(toy):  # noqa: F811
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {a: pool.submit(toy.update, a) for a in ('B', 'R')}
        return {a: f.result(timeout=300) for a, f in futures.items()}


# ------------------------------------------------------------------ 1. early R start

def test_early_vs_serial_r_inputs_byte_identical(practice_toy, monkeypatch, tmp_path_factory):
    toy, cfg = practice_toy, practice_toy.cfg
    # Both orders run in the SAME run-root path (job dicts and G4 capture payloads embed it, as on cpuZ).
    root = Path(cfg['run_root'])
    initial = tmp_path_factory.mktemp('optiona_initial')/'root'
    shutil.copytree(root, initial)
    monkeypatch.delenv(EARLY, raising=False)
    _run_pair(toy)
    serial = dict(counts=list(toy.counts), captured=list(toy.captured))
    assert not g4.early_receipt_path(cfg, 1).exists()
    assert serial['captured'] and all(c['b_input'] for c in serial['captured'])   # serial: R after B_INPUT
    serial_root = tmp_path_factory.mktemp('optiona_serial')/'root'
    shutil.copytree(root, serial_root)
    for child in list(root.iterdir()):
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(initial, root, dirs_exist_ok=True)
    early_root = root
    toy.counts.clear(), toy.captured.clear()
    monkeypatch.setenv(EARLY, '1')
    toy.gate['event'] = threading.Event()
    _run_pair(toy)
    early = dict(counts=list(toy.counts), captured=list(toy.captured))
    assert not any(c['b_input'] for c in early['captured'])                       # early: before B_INPUT
    assert sorted(serial['counts']) == sorted(early['counts']) == [('B', 0), ('R', 0)]
    receipt = g4.read(g4.early_receipt_path(cfg, 1))
    assert receipt['count'] == 0 and receipt['b_plan_file_sha256'] == g4.file_sha(g4.b_plan_path(cfg, 1))
    # job ids, worlds, noise seeds and whole job dicts (no wall-clock fields exist) are byte-identical
    left = [_job_bytes(c['jobs'], c['root']) for c in serial['captured']]
    right = [_job_bytes(c['jobs'], c['root']) for c in early['captured']]
    assert left == right
    ids = [[(j['id'], j['world']['seed'], j['noise_seed']) for j in c['jobs']] for c in early['captured']]
    assert ids == [[(j['id'], j['world']['seed'], j['noise_seed']) for j in c['jobs']] for c in serial['captured']]
    for arm in ('R', 'B'):
        assert (serial_root/'g4'/f'{arm}_INPUT_001.json').read_bytes() == \
            (early_root/'g4'/f'{arm}_INPUT_001.json').read_bytes()
    assert (serial_root/'g4'/'ENVELOPE_001.json').read_bytes() == (early_root/'g4'/'ENVELOPE_001.json').read_bytes()


def _sealed_b_plan(cfg, rows):
    record = _seal(dict(arm='B', update=1, rows=[dict(family='f', example_id=f'e{i}') for i in range(rows)],
                        breadth={}), 'practice_plan_sha256')
    g4.publish(g4.b_plan_path(cfg, 1), record)
    return record


def test_early_count_matches_serial_count(toy, monkeypatch):  # noqa: F811
    cfg = toy.cfg
    cfg['worlds_by_update']['1'] = [dict(seed=s, cell=0) for s in (7, 8, 9, 10)]
    _sealed_b_plan(cfg, 2)
    g4.publish(g4.input_path(cfg, 'B', 1), dict(binding=g4.credit.settings(cfg), update=1, arm='B',
        episodes=[dict(practice=dict(slot='p1')), dict(practice=dict(slot='p2')), dict(id='a'), dict(id='b')]))
    monkeypatch.delenv(EARLY, raising=False)
    assert g4.paired_practice_count(cfg, 1) == 2 and not g4.early_receipt_path(cfg, 1).exists()
    monkeypatch.setenv(EARLY, '1')
    assert g4.paired_practice_count(cfg, 1) == 2
    assert g4.paired_practice_count(cfg, 1) == 2          # idempotent receipt
    assert g4.read(g4.early_receipt_path(cfg, 1))['ordinary'] == 2


@pytest.mark.parametrize('case', ['practice_count', 'ordinary_count', 'plan_changed'])
def test_count_mismatch_faults_before_any_ledger_write(toy, monkeypatch, case):  # noqa: F811
    cfg, root = toy.cfg, Path(toy.cfg['run_root'])
    cfg['worlds_by_update']['1'] = [dict(seed=s, cell=0) for s in (7, 8)]
    monkeypatch.setenv(EARLY, '1')
    _sealed_b_plan(cfg, 1)
    assert g4.paired_practice_count(cfg, 1) == 1
    b = [dict(id='p', practice=dict(slot='p1')), dict(id='o', world=dict(seed=7, cell=0))]
    if case == 'practice_count':
        b = [dict(id='o', world=dict(seed=7, cell=0)), dict(id='o2', world=dict(seed=8, cell=0))]
    elif case == 'ordinary_count':
        b = [dict(id='p', practice=dict(slot='p1'))]
    else:
        path = g4.b_plan_path(cfg, 1)
        path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='early R|B practice plan changed'):
        g4.Coordinator(cfg).measure(1, dict(B=b, R=[]))
    assert not (root/'CHARGES.jsonl').exists() and not g4._envelope_path(cfg, 1).exists()


def test_guard_passes_matching_counts_and_is_noop_without_receipt(toy, monkeypatch):  # noqa: F811
    cfg = toy.cfg
    cfg['worlds_by_update']['1'] = [dict(seed=s, cell=0) for s in (7, 8)]
    assert g4.early_guard(cfg, 1, dict(B=[], R=[])) is None          # flags off / serial: unchanged
    monkeypatch.setenv(EARLY, '1')
    _sealed_b_plan(cfg, 1)
    g4.paired_practice_count(cfg, 1)
    assert g4.early_guard(cfg, 1, dict(B=[dict(practice=dict(slot='p')), dict(id='o')], R=[]))['count'] == 1


def test_flags_off_launcher_waits_for_b_input(monkeypatch):
    monkeypatch.delenv(EARLY, raising=False)
    assert g4.early_r(dict(practice_starts=True), 1) is False


# ------------------------------------------------------------------ TTC cache

def _ttc_root(tmp_path):
    job = tmp_path/'jobs'/'warm_u001_A_7'
    job.mkdir(parents=True)
    (job/'RESULT.json').write_text(json.dumps(dict(world=dict(seed=7), metrics={'900': dict(
        collisions=0, teleports=0, hard_brake=1)})))
    (job/'observable.jsonl.gz').write_bytes(b'')
    return tmp_path


@pytest.mark.parametrize('flag', [True, False])
def test_concurrent_ttc_cache_writes(tmp_path, monkeypatch, flag):
    root = _ttc_root(tmp_path)
    cache = root/'ttc_cache'
    cache.mkdir()
    from . import replay_bank
    barrier = threading.Barrier(6)

    def slow(path, start, end):
        barrier.wait(10)
        return 3
    monkeypatch.setattr(replay_bank, 'ttc_events', slow)
    if flag:
        monkeypatch.setenv(EARLY, '1')
    else:
        monkeypatch.delenv(EARLY, raising=False)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(sd.episode, root, 'warm_u001_A_7', True, cache) for _ in range(6)]
        outcomes = []
        for f in futures:
            try:
                outcomes.append(f.result(timeout=60)['ttc_below_1p5_events'])
            except FileExistsError:
                outcomes.append('exists')
    files = sorted(p.name for p in cache.iterdir())
    assert len(files) == 1 and not files[0].startswith('.')
    if flag:
        assert outcomes == [3]*6          # atomic + idempotent: every concurrent writer succeeds
    else:
        assert outcomes.count(3) == 1 and outcomes.count('exists') == 5   # unchanged exclusive-create contract


def test_ttc_cache_existing_different_bytes_fail_closed(tmp_path):
    path = tmp_path/'entry.json'
    sd.cache_write(path, dict(a=1))
    sd.cache_write(path, dict(a=1))
    with pytest.raises(ValueError, match='different bytes'):
        sd.cache_write(path, dict(a=2))
    assert [p.name for p in tmp_path.iterdir()] == ['entry.json']


# ------------------------------------------------------------------ RemotePool (fake transport)

class FakeTransport:
    """remote_worker's REAL ops against a replica directory; only ssh/rsync and the detached process are faked."""
    host, python = 'fake-cpuV', sys.executable

    def __init__(self, zroot, vroot):
        self.zroot, self.vroot = Path(zroot), Path(vroot)
        self.calls, self.fail, self.behaviour, self.delay = [], {}, {}, {}
        self.polls, self.started, self.corrupt = 0, {}, False
        self.on_poll = None

    def call(self, op, root, *, job=None, payload=None, data=None, timeout=None):
        assert Path(root) == self.zroot
        self.calls.append(op)
        if self.fail.get(op):
            raise rwave.RemoteWaveLost(f'remote {op} transport error: ssh exit 255 (connection reset)')
        try:
            if op == 'check':
                return rw.op_check(self.vroot, payload)
            if op == 'admit':
                rw._charges_dir(self.vroot)
                return dict(disk_free_gb=20., mem_available_gb=80., load1=1.)
            if op == 'start':
                return rw.op_start(self.vroot, payload, data, spawn=self._spawn)
            if op == 'status':
                self.polls += 1
                if self.on_poll:
                    self.on_poll(self)
                self._advance()
                return rw.op_status(self.vroot, payload,
                                    alive=lambda record: self.behaviour.get(record['ident']) != 'dead')
            if op == 'manifest':
                return rw.op_manifest(self.vroot, job)
            if op == 'cleanup':
                return rw.op_cleanup(self.vroot, payload)
            if op == 'abort':
                return dict(abort={i: 'sigterm' for i in payload['ids']})
        except rw.Refused as error:
            raise rwave.RemoteWaveLost(f'remote {op} refused (exit 3): {error}') from None
        raise AssertionError(op)

    def _spawn(self, root, ident, header):
        self.started[ident] = 0
        return 4242, None

    def _advance(self):
        for ident in list(self.started):
            self.started[ident] += 1
            stage = self.vroot/rw.STAGE/ident
            if (stage/'DONE.json').exists() or (stage/'ERROR.json').exists():
                continue
            kind = self.behaviour.get(ident, 'done')
            if kind in ('running', 'dead') or self.started[ident] <= self.delay.get(ident, 0):
                # PID.json carries the ident for the fake liveness check
                record = json.loads((stage/'PID.json').read_text())
                if 'ident' not in record:
                    (stage/'PID.json').write_text(json.dumps(dict(record, ident=ident)))
                continue
            job = pickle.loads((stage/'JOB.pkl').read_bytes())
            if kind == 'setup_error':
                rw._publish_bytes(stage/'ERROR.json', rw._json_bytes(dict(phase='setup', type='Refused',
                    module='tools.coopmwm.remote_worker', args=['replica CHARGES.jsonl must be a directory'],
                    message='replica CHARGES.jsonl must be a directory')))
                continue
            rw.op_run(self.vroot, ident, episode=lambda j: fake_episode(j, self.vroot, kind), job_root=self.zroot)

    def pull_tree(self, root, rel, *, exclude=(), timeout=None):
        if self.fail.get('rsync'):
            raise rwave.RemoteWaveLost('rsync landing exit 12: connection unexpectedly closed')
        src, dst = self.vroot/rel, Path(root)/rel
        for path in sorted(src.rglob('*')):
            name = path.relative_to(src).as_posix()
            if not path.is_file() or ('/'+name) in exclude:
                continue
            target = dst/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                data = path.read_bytes()
                if self.corrupt and name.endswith('observable.jsonl.gz'):
                    data += b'x'
                target.write_bytes(data)


def fake_episode(job, vroot, kind):
    """accel_plant.episode's on-disk outputs, under the replica's identical path (here: vroot)."""
    from .accel_storage import write
    path = vroot/'jobs'/job['id']
    path.mkdir(parents=True)
    write(path/'CLAIM.json', dict(pid=1, id=job['id'], charged_sim_s=job['sim_s']))
    (path/'observable.jsonl.gz').write_bytes(b'tape-'+job['id'].encode())
    (path/'g4_captures').mkdir()
    (path/'g4_captures'/'opp1.pt').write_bytes(b'capture-'+job['id'].encode())
    if kind == 'episode_error':
        raise ValueError('SUMO episode failed (episode-local)')
    if kind == 'stop':
        raise ValueError('durable warm STOP')
    result = dict(id=job['id'], world=job['world'], arm=job['arm'], steps=[], metrics={'900': dict(collisions=0.)},
                  totals=dict(team_delay_vs=1.5), actual_sim_s=1200, charged_sim_s=job['sim_s'])
    write(path/'RESULT.json', result)
    return result


@pytest.fixture
def remote(tmp_path, monkeypatch):
    zroot, vroot, src = tmp_path/'Z'/'run', tmp_path/'V'/'run', tmp_path/'src'
    for root in (zroot, vroot):
        root.mkdir(parents=True)
        (root/'CONFIG.json').write_text('{"config": 1}\n')
    (vroot/'CHARGES.jsonl').mkdir()                 # the replica ledger is a DIRECTORY
    src.mkdir()
    (src/'a.py').write_text('x = 1\n')
    (src/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(files={'a.py': rw.file_sha(src/'a.py')})))
    transport = FakeTransport(zroot, vroot)
    monkeypatch.setenv(rwave.REMOTE_ENV, '172.17.0.15')
    monkeypatch.setattr(rwave.SshTransport, 'from_env', classmethod(lambda cls, source_root: transport))
    monkeypatch.setattr(rwave, 'POLL_S', 0.01)
    ledger_rows = []

    def reserve(root, jobs, phase):
        ids = {r['id'] for r in ledger_rows}
        assert not ids & {j['id'] for j in jobs}, 'duplicate charge'
        ledger_rows.extend(dict(id=j['id'], sim_s=j['sim_s'], phase=phase) for j in jobs)
    monkeypatch.setattr(accel_runner, 'reserve', reserve)
    monkeypatch.setattr(accel_runner, 'ledger', lambda root: list(ledger_rows))
    from . import runner
    monkeypatch.setattr(runner, 'admission', lambda b: None)
    monkeypatch.setattr(accel_runner, 'replacement_world', lambda world, used, taken, arm, ident:
                        dict(seed=900+int(world['seed']), cell=world['cell']))
    b = dict(workers=8, load_max=64, source_root=str(src), run_root=str(zroot))

    def jobs(*seeds, arm='R'):
        return [dict(id=f'ns_warm_u001_{arm}_{s}', path=str(zroot/'jobs'/f'ns_warm_u001_{arm}_{s}'),
                     world=dict(seed=s, cell=0), role='train', arm=arm, sim_s=1200, queue_root=str(zroot),
                     update=1, noise_seed=s, weights={'w': torch.ones(2)}) for s in seeds]

    def wave(job_list):
        monkeypatch.setattr(accel_runner, 'wave_jobs', lambda *a, **k: (list(job_list), []))
        return accel_runner.wave(zroot, b, {'R': None}, [j['world'] for j in job_list], 'train', 'ns_warm_u001',
                                 'track2_train', False, dict(update=1))
    return type('Remote', (), dict(zroot=zroot, vroot=vroot, src=src, t=transport, ledger=ledger_rows, b=b,
                                   jobs=staticmethod(jobs), wave=staticmethod(wave)))


def test_remote_wave_lands_results_pointer_last_and_records_manifests(remote):
    jobs = remote.jobs(7, 8)
    remote.t.delay = {jobs[0]['id']: 2}
    results = remote.wave(jobs)
    assert [r['id'] for r in results] == [j['id'] for j in jobs]
    for job in jobs:
        local, replica = remote.zroot/'jobs'/job['id'], remote.vroot/'jobs'/job['id']
        assert rw.tree_manifest(local) == rw.tree_manifest(replica)
        landed = json.loads((remote.zroot/rw.STAGE/job['id']/'LANDED.json').read_text())
        assert landed['manifest'] == rw.tree_manifest(local) and landed['update'] == 1
        sent = json.loads((remote.zroot/rw.STAGE/job['id']/'SENT.json').read_text())
        staged = (remote.vroot/rw.STAGE/job['id']/'JOB.pkl').read_bytes()
        assert sent['sha256'] == hashlib.sha256(staged).hexdigest()
        assert pickle.loads(staged)['id'] == job['id']
    assert [r['id'] for r in remote.ledger] == [j['id'] for j in jobs]      # charged once, on cpuZ only
    assert remote.t.calls[0] == 'check'                                       # identity before anything else


@pytest.mark.parametrize('failure', ['ssh_drop', 'timeout', 'landing_mismatch', 'existing_remote_dir',
                                     'dead_worker', 'setup_error', 'rsync_drop'])
def test_remote_failures_fault_without_replacement_or_extra_charge(remote, monkeypatch, failure):
    jobs = remote.jobs(7, 8)
    if failure == 'ssh_drop':
        remote.t.on_poll = lambda t: t.fail.__setitem__('status', True) if t.polls >= 2 else None
        remote.t.behaviour[jobs[0]['id']] = 'running'
    elif failure == 'timeout':
        monkeypatch.setenv(rwave.JOB_TIMEOUT_ENV, '0.05')
        remote.t.behaviour[jobs[0]['id']] = 'running'
    elif failure == 'landing_mismatch':
        remote.t.corrupt = True
    elif failure == 'existing_remote_dir':
        (remote.vroot/'jobs'/jobs[1]['id']).mkdir(parents=True)
    elif failure == 'dead_worker':
        remote.t.behaviour[jobs[0]['id']] = 'dead'
    elif failure == 'setup_error':
        remote.t.behaviour[jobs[0]['id']] = 'setup_error'
    else:
        remote.t.fail['rsync'] = True
    with pytest.raises(rwave.RemoteWaveLost):
        remote.wave(jobs)
    assert [r['id'] for r in remote.ledger] == [j['id'] for j in jobs]     # no extra charge, no replacement
    assert not list(remote.zroot.glob('jobs/*/EPISODE_FAILED.json'))
    assert list((remote.zroot/rw.STAGE).glob('REMOTE_WAVE_LOST_*.json'))
    assert not accel_runner.episode_local(rwave.RemoteWaveLost('x'))


@pytest.mark.parametrize('what', ['config_sha256', 'signals', 'source tree'])
def test_remote_identity_mismatch_refused_before_reserve(remote, what):
    if what == 'config_sha256':
        (remote.vroot/'CONFIG.json').write_text('{"config": 2}\n')
    elif what == 'signals':
        (remote.vroot/'STOP.json').write_text('{"stale": "mirror"}\n')     # e.g. left over from an earlier FAULT
    else:
        (remote.src/'a.py').write_text('x = 2\n')       # one shared path in the test: the manifest check fails
    with pytest.raises(rwave.RemoteWaveLost, match=what):
        remote.wave(remote.jobs(7))
    assert remote.ledger == [] and remote.t.calls == ['check']


def test_replica_ledger_write_refused(remote):
    from . import storage
    with pytest.raises(OSError):
        storage.reserve(remote.vroot, [dict(id='x', sim_s=1)], 'track2_train')
    with pytest.raises(OSError):
        (remote.vroot/'CHARGES.jsonl').open('a').close()
    shutil.rmtree(remote.vroot/'CHARGES.jsonl')
    (remote.vroot/'CHARGES.jsonl').write_text('')
    with pytest.raises(rwave.RemoteWaveLost, match='CHARGES.jsonl'):
        remote.wave(remote.jobs(7))
    assert remote.ledger == []
    with pytest.raises(rw.Refused):
        rw.op_start(remote.vroot, dict(id='j', size=0, sha256=hashlib.sha256(b'').hexdigest()), b'')


def test_remote_episode_error_keeps_the_unchanged_replacement_rule(remote):
    jobs = remote.jobs(7, 8, 9, 10, 11)        # one failure in five stays inside the registered 20% tolerance
    remote.t.behaviour[jobs[0]['id']] = 'episode_error'
    results = remote.wave(jobs)
    failed = json.loads((remote.zroot/'jobs'/jobs[0]['id']/'EPISODE_FAILED.json').read_text())
    assert failed['error'] == "ValueError('SUMO episode failed (episode-local)')"
    assert failed['replacement'] == 'ns_warm_u001_R_907'
    assert sorted(r['id'] for r in results) == sorted([j['id'] for j in jobs[1:]]+['ns_warm_u001_R_907'])
    assert json.loads((remote.zroot/rw.STAGE/jobs[0]['id']/'LANDED.json').read_text())['status'] == 'episode_error'
    assert [r['id'] for r in remote.ledger] == [j['id'] for j in jobs]+['ns_warm_u001_R_907']   # charged replacement


def test_stop_and_pause_propagation(remote):
    jobs = remote.jobs(7)
    ident = jobs[0]['id']
    remote.t.delay = {ident: 3}

    def on_poll(t):
        if t.polls == 1:
            (remote.zroot/'STOP.json').write_bytes(b'{"error": "owner stop"}\n')
            (remote.vroot/'PAUSE_SAFETY.json').write_bytes(b'{"rule": "SAFETY"}\n')
        if t.polls == 3:
            assert (remote.vroot/'STOP.json').read_bytes() == (remote.zroot/'STOP.json').read_bytes()
            assert (remote.vroot/'STOP.json').stat().st_mtime_ns == (remote.zroot/'STOP.json').stat().st_mtime_ns
            t.behaviour[ident] = 'stop'    # the remote worker's per-tick health guard sees the mirrored STOP
    remote.t.on_poll = on_poll
    with pytest.raises(ValueError, match='durable warm STOP'):
        remote.wave(jobs)
    assert (remote.zroot/'PAUSE_SAFETY.json').read_bytes() == b'{"rule": "SAFETY"}\n'       # pulled back
    assert not list(remote.zroot.glob('jobs/*/EPISODE_FAILED.json'))                    # stop: no replacement
    assert [r['id'] for r in remote.ledger] == [ident]


def test_cleanup_only_after_learning_monitor_and_reverify(remote):
    jobs = remote.jobs(7, 8)
    remote.wave(jobs)
    cfg = dict(run_root=str(remote.zroot))
    with pytest.raises(ValueError, match='LEARNING_MONITOR'):
        rwave.cleanup_after_monitor(cfg, 1, transport=remote.t)
    assert all((remote.vroot/'jobs'/j['id']).exists() for j in jobs)
    (remote.zroot/'LEARNING_MONITOR_u001.json').write_text('{}')
    tampered = remote.zroot/'jobs'/jobs[0]['id']/'observable.jsonl.gz'
    original = tampered.read_bytes()
    tampered.write_bytes(original+b'!')
    with pytest.raises(ValueError, match='no longer matches'):
        rwave.cleanup_after_monitor(cfg, 1, transport=remote.t)
    assert all((remote.vroot/'jobs'/j['id']).exists() for j in jobs)
    tampered.write_bytes(original)
    # disk_prune may delete consumed captures: accepted only with the recorded identical sha
    capture = remote.zroot/'jobs'/jobs[1]['id']/'g4_captures'/'opp1.pt'
    (remote.zroot/'pruned').mkdir()
    (remote.zroot/'pruned'/'PRUNE_u001.json').write_text(json.dumps(dict(captures=[dict(
        path=f'jobs/{jobs[1]["id"]}/g4_captures/opp1.pt', sha256=rw.file_sha(capture))])))
    capture.unlink()
    out = rwave.cleanup_after_monitor(cfg, 1, transport=remote.t)
    assert set(out['cleaned'].values()) == {'deleted'}
    assert not any((remote.vroot/'jobs'/j['id']).exists() for j in jobs)
    assert not any((remote.vroot/rw.STAGE/j['id']).exists() for j in jobs)
    assert rwave.cleanup_after_monitor(cfg, 1, transport=remote.t)['cleaned'] == {}


def test_flags_off_wave_uses_the_local_pool_only(remote, monkeypatch):
    monkeypatch.delenv(rwave.REMOTE_ENV)
    monkeypatch.setattr(rwave, 'pool_for', lambda *a, **k: (_ for _ in ()).throw(AssertionError('remote used')))
    used = []

    class LocalPool:
        def __init__(self, max_workers, mp_context):
            used.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, job):
            from concurrent.futures import Future
            f = Future()
            f.set_result(dict(id=job['id'], world=job['world'], arm=job['arm']))
            return f
    monkeypatch.setattr(accel_runner, 'ProcessPoolExecutor', LocalPool)
    results = remote.wave(remote.jobs(7))
    assert used == [8] and results[0]['id'] == 'ns_warm_u001_R_7' and remote.t.calls == []


def test_remote_pool_used_for_r_train_waves_only(remote):
    assert rwave.remote_wave_jobs(remote.jobs(7, arm='A'), 'train') is False
    with pytest.raises(rwave.RemoteWaveLost):
        rwave.remote_wave_jobs(remote.jobs(7)+remote.jobs(8, arm='A'), 'train')
    with pytest.raises(rwave.RemoteWaveLost):
        rwave.remote_wave_jobs([dict(remote.jobs(7)[0], sumo_verify=dict(root='x'))], 'train')


def test_ssh_transport_errors_become_remote_wave_lost(tmp_path, monkeypatch):
    key, known = tmp_path/'k', tmp_path/'kh'
    key.write_text('k'), known.write_text('h')
    transport = rwave.SshTransport('172.17.0.15', key, known, source_root='/src')
    argv = transport.ssh_base()
    for option in ('BatchMode=yes', 'StrictHostKeyChecking=yes', f'UserKnownHostsFile={known}', 'IdentitiesOnly=yes',
                   'ControlMaster=no', 'ForwardAgent=no', 'RequestTTY=no'):
        assert option in argv
    assert '-T' in argv

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired('ssh', 1)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(rwave.RemoteWaveLost, match='timed out'):
        transport.call('status', '/run', payload={})
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 255, b'', b'reset'))
    with pytest.raises(rwave.RemoteWaveLost, match='no result line'):
        transport.call('status', '/run', payload={})
    line = (rw.MARKER+json.dumps(dict(error='remote job directory already exists: j'))+'\n').encode()
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 3, line, b''))
    with pytest.raises(rwave.RemoteWaveLost, match='already exists'):
        transport.call('start', '/run', payload={}, data=b'')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 23, b'', b'io error'))
    with pytest.raises(rwave.RemoteWaveLost, match='exit 23'):
        transport.pull_tree(tmp_path, 'jobs/j')


def test_remote_worker_run_refuses_foreign_job_and_reports_setup(tmp_path):
    root = tmp_path/'run'
    (root/'CHARGES.jsonl').mkdir(parents=True)
    job = dict(id='j', path=str(tmp_path/'elsewhere'/'jobs'/'j'), queue_root=str(root), arm='R')
    data = pickle.dumps(job)
    header = dict(id='j', size=len(data), sha256=hashlib.sha256(data).hexdigest())
    rw.op_start(root, header, data, spawn=lambda r, i, h: (1, None))
    with pytest.raises(rw.Refused, match='staging already exists'):
        rw.op_start(root, header, data, spawn=lambda r, i, h: (1, None))
    assert rw.op_run(root, 'j', episode=lambda j: pytest.fail('must not run')) == 1
    error = json.loads((root/rw.STAGE/'j'/'ERROR.json').read_text())
    assert error['phase'] == 'setup' and 'does not belong' in error['message']
    with pytest.raises(rw.Refused, match='sha256'):
        rw.op_start(root, dict(header, id='k', sha256='0'*64), data)
    os.environ.pop(EARLY, None)
