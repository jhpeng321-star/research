"""Option A (operator 2026-11-07): run arm R's TRAIN episode wave on a second host (cpuV); everything else on cpuZ.

Env-gated: COOPMWM_REMOTE_R=<private ip> plus COOPMWM_REMOTE_R_KEY (dedicated key) and
COOPMWM_REMOTE_R_KNOWN_HOSTS (pinned known_hosts). Unset: accel_runner.wave never imports this module.

Stays on cpuZ (the R parent): reserve (before admission, unchanged), the CHARGES.jsonl ledger, admission
bookkeeping, EPISODE_FAILED receipts + charged same-cell replacements (run_train_jobs, unchanged), post-collect
including R_INPUT, safety, reward, G4 and training. Only accel_plant.episode(job) runs remotely, on the SAME job
dict, in a detached `nice -n 19` worker of the frozen tree deployed at the identical path (remote_worker.py).

Fail closed: any ssh/rsync error, timeout, sha mismatch, dead or missing remote worker, replica identity mismatch
or remote setup error raises RemoteWaveLost. accel_runner.episode_local treats it as NOT episode-local, so it
propagates to launch_training.main (FAULT.json + STOP.json): never a replacement, never a retry, no extra charge.
Remote episode exceptions proper are re-raised as the same exception type/args, so the unchanged episode-local rule
(and its charged replacement) applies exactly as for a local pool worker.

Landing (per finished job): sha manifest built on cpuV -> rsync -a --ignore-existing (RESULT.json pointer
excluded) -> manifest of the landed tree on cpuZ must be identical -> the pointer bytes are checked against the
manifest and published LAST (exclusive). A LANDED.json record (manifest) is kept in <root>/remote_r/<id>/ on cpuZ.
cpuV copies are deleted only after LEARNING_MONITOR_u<u> exists and the landed copies re-verify (cleanup_after_monitor).
"""
from concurrent.futures import Future
import base64
import builtins
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import threading
import time

from . import remote_worker as rw

REMOTE_ENV = 'COOPMWM_REMOTE_R'
KEY_ENV = 'COOPMWM_REMOTE_R_KEY'
KNOWN_ENV = 'COOPMWM_REMOTE_R_KNOWN_HOSTS'
USER_ENV = 'COOPMWM_REMOTE_R_USER'
ENVSH_ENV = 'COOPMWM_REMOTE_R_ENVSH'
JOB_TIMEOUT_ENV = 'COOPMWM_REMOTE_R_JOB_TIMEOUT_S'
DEFAULT_ENVSH = '/home/ubuntu/cpus_env/env.sh'
DISK_MIN_GB = 12          # owner approval (chat ~14:55): cpuV free-disk floor
MEM_MIN_GB = 40           # same MemAvailable floor as the cpuZ waves
POLL_S = 20.
CALL_TIMEOUT_S = 120.
RSYNC_TIMEOUT_S = 1800.
ADMISSION_TIMEOUT_S = 3600.
DEFAULT_JOB_TIMEOUT_S = 4*3600.
FORWARD_PREFIXES = ('COOPMWM_', 'C4NM_', 'OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_', 'KMP_', 'TORCH_')
FORWARD_KEYS = ('PYTHONHASHSEED', 'SUMO_HOME', 'PYTHONPATH')
# Job keys whose worker writes leave jobs/<id> (or are not arm-R TRAIN episodes): never sent remotely.
REFUSED_KEYS = ('native_reference', 'sumo_verify', 'ls_collect', 'collect')


class RemoteWaveLost(RuntimeError):
    """Transport/identity/landing failure of a remote R wave: a FAULT, never episode-local."""


def enabled():
    return bool(os.environ.get(REMOTE_ENV))


def forwarded_env(environ=None):
    environ = os.environ if environ is None else environ
    return {k: v for k, v in sorted(environ.items()) if not k.startswith(REMOTE_ENV)
            and (k.startswith(FORWARD_PREFIXES) or k in FORWARD_KEYS)}


def remote_wave_jobs(jobs, role):
    """True: an arm-R TRAIN wave for the remote pool; False: run locally (other arms, e.g. A references)."""
    arms = {j['arm'] for j in jobs}
    if 'R' not in arms:
        return False
    if arms != {'R'} or role != 'train':
        raise RemoteWaveLost('remote R wave must be an arm-R TRAIN wave only (fail closed before reserve)')
    bad = sorted({k for j in jobs for k in REFUSED_KEYS if j.get(k)})
    if bad:
        raise RemoteWaveLost('remote R wave refuses job keys that write outside the job dir: '+','.join(bad))
    return True


def _write_new(path, value):
    rw._publish_bytes(path, rw._json_bytes(value))


# ------------------------------------------------------------------ transport

class SshTransport:
    """Plain subprocess ssh/rsync with explicit options (no multiplexing, no agent, no forwarding, no pty)."""

    def __init__(self, host, key, known_hosts, *, source_root, python=None, user='ubuntu', envsh=DEFAULT_ENVSH):
        if not host or not key or not known_hosts:
            raise RemoteWaveLost(f'{REMOTE_ENV}, {KEY_ENV} and {KNOWN_ENV} are all required')
        for path in (key, known_hosts):
            if not Path(path).is_file():
                raise RemoteWaveLost('remote R key/known_hosts file missing: '+str(path))
        self.host, self.key, self.known, self.user = host, str(key), str(known_hosts), user
        self.source_root, self.python, self.envsh = str(source_root), python or sys.executable, envsh

    @classmethod
    def from_env(cls, source_root):
        return cls(os.environ.get(REMOTE_ENV), os.environ.get(KEY_ENV), os.environ.get(KNOWN_ENV),
                   source_root=source_root, user=os.environ.get(USER_ENV, 'ubuntu'),
                   envsh=os.environ.get(ENVSH_ENV, DEFAULT_ENVSH))

    def ssh_base(self):
        return ['ssh', '-T', '-i', self.key, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile={self.known}',
                '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
                '-o', 'ServerAliveCountMax=3', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
                '-o', 'ForwardAgent=no', '-o', 'ClearAllForwardings=yes', '-o', 'RequestTTY=no']

    def target(self):
        return f'{self.user}@{self.host}'

    def call(self, op, root, *, job=None, payload=None, data=None, timeout=CALL_TIMEOUT_S):
        command = (f'cd {shlex.quote(self.source_root)} && . {shlex.quote(self.envsh)} >/dev/null 2>&1 && '
                   f'exec {shlex.quote(self.python)} -B -m tools.coopmwm.remote_worker {op} '
                   f'--root {shlex.quote(str(root))}' + (f' --job {shlex.quote(job)}' if job else ''))
        stdin = (json.dumps(payload or {}, sort_keys=True).encode()+b'\n'+data if data is not None
                 else json.dumps(payload or {}, sort_keys=True).encode())
        try:
            done = subprocess.run(self.ssh_base()+[self.target(), command], input=stdin, capture_output=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RemoteWaveLost(f'remote {op} timed out after {timeout:.0f} s') from None
        except OSError as error:
            raise RemoteWaveLost(f'remote {op} transport error: {error!r}') from None
        return parse_result(op, done.returncode, done.stdout, done.stderr)

    def pull_tree(self, root, rel, *, exclude=(), timeout=RSYNC_TIMEOUT_S):
        local = Path(root)/rel
        local.mkdir(parents=True, exist_ok=True)
        command = ['rsync', '-a', '--ignore-existing', *[f'--exclude={x}' for x in exclude],
                   '-e', shlex.join(self.ssh_base()), f'{self.target()}:{Path(root, rel).as_posix()}/',
                   f'{local.as_posix()}/']
        try:
            done = subprocess.run(command, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RemoteWaveLost(f'rsync landing of {rel} timed out') from None
        except OSError as error:
            raise RemoteWaveLost(f'rsync landing of {rel} failed: {error!r}') from None
        if done.returncode != 0:
            raise RemoteWaveLost(f'rsync landing of {rel} exit {done.returncode}: '
                                 +done.stderr.decode(errors='replace')[-400:])


def parse_result(op, code, stdout, stderr=b''):
    lines = [line for line in stdout.decode(errors='replace').splitlines() if line.startswith(rw.MARKER)]
    if len(lines) != 1:
        raise RemoteWaveLost(f'remote {op} exit {code}: no result line; '+stderr.decode(errors='replace')[-400:])
    value = json.loads(lines[0][len(rw.MARKER):])
    if code != 0 or 'error' in value:
        raise RemoteWaveLost(f'remote {op} refused (exit {code}): {value.get("error")}')
    return value


# ------------------------------------------------------------------ replica identity

def replica_inputs(root, jobs, environ=None, source_root=None):
    """Absolute files/dirs a worker reads that must be byte-identical on both hosts (CONFIG via config_sha256)."""
    environ = os.environ if environ is None else environ
    paths = set()
    for job in jobs:
        for key in ('runtime', 'checkpoint'):
            value = job.get(key)
            if isinstance(value, str) and os.path.isabs(value):
                paths.add(value)
    trunk = environ.get('COOPMWM_FROZEN_TRUNK')
    if trunk:
        paths.add(trunk.split('|', 1)[0])
    spec = environ.get('COOPMWM_MAP_SPEC')
    if spec:
        paths.add(str(Path(spec).parent))
    source = None if source_root is None else os.path.normpath(str(source_root))
    for entry in (environ.get('PYTHONPATH') or '').split(os.pathsep):
        if entry and os.path.isabs(entry) and os.path.normpath(entry) != source:
            paths.add(entry)     # forwarded to the worker: must exist byte-identical on the replica
    return sorted(paths)


def verify_replica(transport, root, jobs, *, source_root, envsh, local_check=None):
    payload = dict(source_root=str(source_root), inputs=replica_inputs(root, jobs, source_root=source_root),
                   envsh=envsh)
    local = (local_check or rw.op_check)(root, payload, replica=False)
    remote = transport.call('check', root, payload=payload)
    if not remote.get('charges_is_dir'):
        raise RemoteWaveLost('replica CHARGES.jsonl is not a directory (replica ledger must refuse writes)')
    if local.get('charges_is_dir'):
        raise RemoteWaveLost('cpuZ CHARGES.jsonl is a directory: refusing to run a wave from a replica root')
    if remote.get('source_files_bad'):
        raise RemoteWaveLost('replica source tree differs from its manifest: '+','.join(remote['source_files_bad'][:5]))
    # 'signals': the replica's control files (STOP/FAULT/PAUSE/release...) must equal cpuZ's at every wave start, so a
    # stale mirrored STOP or an unpulled remote pause can never leak into the next wave (operator clears the replica).
    for key in ('source_manifest_sha256', 'config_sha256', 'inputs', 'envsh_sha256', 'versions', 'ld_preload',
                'signals'):
        if local.get(key) != remote.get(key):
            raise RemoteWaveLost(f'replica identity mismatch: {key}')
    if any(v is None for v in remote['inputs'].values()):
        raise RemoteWaveLost('replica input missing')
    return dict(local=local, remote=remote)


def _reconstruct(error):
    """The remote episode exception as the same type/args where possible (episode_local reads type + text)."""
    name, module, args = error.get('type'), error.get('module'), error.get('args') or [error.get('message', '')]
    if module == 'builtins':
        kind = getattr(builtins, name, None)
        if isinstance(kind, type) and issubclass(kind, Exception):
            try:
                return kind(*args)
            except Exception:
                pass
    if name == 'HealthPause':
        from .warm_health import HealthPause
        return HealthPause(*args)
    if name == 'EpisodeFailureEscalation':
        from .accel_runner import EpisodeFailureEscalation
        return EpisodeFailureEscalation(*args)
    return RuntimeError(f'{module}.{name}: {error.get("message", "")}')


def local_manifest(directory):
    return rw.tree_manifest(directory)


# ------------------------------------------------------------------ the pool

class RemotePool:
    """Executor-shaped (submit/__enter__/__exit__) pool injected for arm R's TRAIN wave only."""

    def __init__(self, root, b, jobs, *, transport=None, source_root=None, poll_s=None, job_timeout=None,
                 admission_timeout=None, clock=time.monotonic, sleep=time.sleep, audit_open=False,
                 envsh=None):
        self.root, self.b = Path(root), b
        self.workers = int(b['workers'])
        self.load_max = b.get('load_max')
        self.source_root = Path(source_root or b.get('source_root') or os.getcwd())
        if not (self.source_root/'SOURCE_MANIFEST.json').is_file():
            raise RemoteWaveLost('no SOURCE_MANIFEST.json in the local source root '+str(self.source_root))
        self.envsh = envsh or os.environ.get(ENVSH_ENV, DEFAULT_ENVSH)
        if not Path(self.envsh).is_file():
            self.envsh = None
        self.transport = transport or SshTransport.from_env(self.source_root)
        self.poll_s, self.clock, self.sleep = POLL_S if poll_s is None else poll_s, clock, sleep
        self.job_timeout = float(job_timeout or os.environ.get(JOB_TIMEOUT_ENV) or DEFAULT_JOB_TIMEOUT_S)
        self.admission_timeout = ADMISSION_TIMEOUT_S if admission_timeout is None else admission_timeout
        self.audit_open = audit_open
        self.lock, self.wake = threading.Lock(), threading.Event()
        self.queue, self.running, self.lost, self.closing, self.aborting = [], {}, None, False, False
        self.thread, self.landed = None, {}
        # Identity BEFORE the caller reserves anything (accel_runner.wave builds the pool before reserve).
        self.identity = verify_replica(self.transport, self.root, jobs, source_root=self.source_root,
                                       envsh=self.envsh)

    # executor protocol ------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        self.shutdown(wait=True, aborting=kind is not None)
        return False

    def submit(self, fn, job):
        from .accel_plant import episode
        if fn is not episode:   # the remote worker calls exactly this function on the same job dict
            raise TypeError('RemotePool runs accel_plant.episode only')
        future = Future()
        with self.lock:
            if self.lost is not None:
                future.set_exception(self.lost)
                return future
            self.queue.append((job, future))
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name='optiona-remote-pool', daemon=True)
                self.thread.start()
        self.wake.set()
        return future

    def shutdown(self, wait=True, *, aborting=False):
        with self.lock:
            self.closing = True
            self.aborting = aborting
            if aborting:
                # never start a queued job after the wave failed; the reservation stays spent (no retry)
                for job, future in self.queue:
                    if not future.done():
                        future.set_exception(RemoteWaveLost('remote R job not started: wave aborted '+job['id']))
                self.queue = []
        self.wake.set()
        if wait and self.thread is not None:
            self.thread.join()

    # worker loop -------------------------------------------------------------
    def _loop(self):
        try:
            while True:
                with self.lock:
                    if self.lost is not None:
                        return
                    if self.closing and not self.queue and not self.running:
                        return
                self._start_queued()
                if self.running:
                    self._poll()
                with self.lock:
                    idle = self.closing and not self.queue and not self.running
                if idle:
                    return
                self.wake.wait(self.poll_s)
                self.wake.clear()
        except RemoteWaveLost as error:
            self._lose(error)
        except BaseException as error:   # anything unexpected in the transport loop is a lost wave too
            self._lose(RemoteWaveLost(f'remote pool loop failed: {error!r}'))

    def _lose(self, error):
        with self.lock:
            self.lost = error
            pending = [f for _, f in self.queue]+[r['future'] for r in self.running.values()]
            ids = [j['id'] for j, _ in self.queue]+list(self.running)
            running = list(self.running)
            self.queue, self.running = [], {}
        try:
            _write_new(self.root/rw.STAGE/f'REMOTE_WAVE_LOST_{time.time_ns()}.json', dict(error=repr(error),
                       jobs=ids, host=getattr(self.transport, 'host', None), unix=time.time(),
                       action='FAULT; no replacement, no retry, no extra charge'))
        except OSError:
            pass
        if running:
            try:   # best effort: our own remote workers, by recorded PID/start ticks
                self.transport.call('abort', self.root, payload=dict(ids=running), timeout=30)
            except Exception:
                pass
        for future in pending:
            if not future.done():
                future.set_exception(error)

    def _admit(self):
        started = self.clock()
        while True:
            value = self.transport.call('admit', self.root)
            ok = (value['disk_free_gb'] >= DISK_MIN_GB and value['mem_available_gb'] is not None
                  and value['mem_available_gb'] >= MEM_MIN_GB
                  and (self.load_max is None or value['load1'] is None or value['load1'] <= self.load_max))
            if ok:
                return value
            if self.clock()-started > self.admission_timeout:
                raise RemoteWaveLost(f'remote admission not met for {self.admission_timeout:.0f} s: {value}')
            self.sleep(30)

    def _start_queued(self):
        while True:
            with self.lock:
                if (self.closing and self.aborting) or not self.queue \
                        or len(self.running) >= self.workers:
                    return
                job, future = self.queue.pop(0)
            if Path(job['path']) != self.root/'jobs'/job['id'] or Path(job['queue_root']) != self.root:
                raise RemoteWaveLost('remote job path must be <run root>/jobs/<id> (identical on both hosts)')
            if Path(job['path']).exists():
                raise RemoteWaveLost('local job directory already exists before remote start: '+job['id'])
            self._admit()
            data = pickle.dumps(job, protocol=4)
            header = dict(id=job['id'], sha256=hashlib.sha256(data).hexdigest(), size=len(data),
                          env=forwarded_env(), python=getattr(self.transport, 'python', sys.executable),
                          source_root=str(self.source_root), audit_open=bool(self.audit_open))
            stage = self.root/rw.STAGE/job['id']
            _write_new(stage/'SENT.json', dict(header, update=job.get('update'), unix=time.time(),
                                               host=getattr(self.transport, 'host', None)))
            self.transport.call('start', self.root, payload=header, data=data)
            with self.lock:
                self.running[job['id']] = dict(job=job, future=future, started=self.clock())

    def _signals(self):
        out = {}
        for name in rw.signal_names(self.root):
            path = self.root/name
            out[name] = dict(b64=base64.b64encode(path.read_bytes()).decode(), mtime_ns=path.stat().st_mtime_ns)
        return out

    def _poll(self):
        with self.lock:
            ids = list(self.running)
        value = self.transport.call('status', self.root, payload=dict(ids=ids, signals=self._signals()))
        for name, b64 in sorted(value.get('pullback', {}).items()):
            if not fnmatch.fnmatchcase(name, 'PAUSE_*.json') or '/' in name or '\\' in name:
                raise RemoteWaveLost('unexpected pull-back file '+name)
            target = self.root/name
            if not target.exists():
                try:
                    rw._publish_bytes(target, base64.b64decode(b64))   # a remote registered health pause
                except FileExistsError:
                    pass
        for ident in ids:
            state = value['jobs'].get(ident, {}).get('state')
            record = self.running[ident]
            if state == 'running':
                if self.clock()-record['started'] > self.job_timeout:
                    raise RemoteWaveLost(f'remote job {ident} exceeded {self.job_timeout:.0f} s')
                continue
            if state == 'done':
                result = self._land(record['job'], expect_result=True)
                self._resolve(ident, result=result)
            elif state == 'error':
                error = value['jobs'][ident]['error']
                if error.get('phase') != 'episode':
                    raise RemoteWaveLost(f'remote worker {ident} failed before the episode: {error.get("message")}')
                self._land(record['job'], expect_result=False)
                self._resolve(ident, error=_reconstruct(error))
            else:
                raise RemoteWaveLost(f'remote job {ident} is {state or "unknown"} without a result')

    def _resolve(self, ident, *, result=None, error=None):
        with self.lock:
            record = self.running.pop(ident)
        if error is None:
            record['future'].set_result(result)
        else:
            record['future'].set_exception(error)

    def _land(self, job, *, expect_result):
        ident = job['id']
        value = self.transport.call('manifest', self.root, job=ident)
        manifest, pointer_b64 = value['manifest'], value.get('pointer_b64')
        if expect_result and (pointer_b64 is None or 'RESULT.json' not in manifest):
            raise RemoteWaveLost(f'remote job {ident} finished without its RESULT pointer')
        if not expect_result and pointer_b64 is not None:
            raise RemoteWaveLost(f'remote job {ident} reported an error but has a RESULT pointer')
        rel = f'jobs/{ident}'
        self.transport.pull_tree(self.root, rel, exclude=('/RESULT.json',))
        landed = local_manifest(self.root/rel)
        expected = {k: v for k, v in manifest.items() if k != 'RESULT.json'}
        if landed != expected:
            raise RemoteWaveLost(f'landing sha manifest mismatch for {ident}')
        result = None
        if expect_result:
            pointer = base64.b64decode(pointer_b64)
            if hashlib.sha256(pointer).hexdigest() != manifest['RESULT.json']:
                raise RemoteWaveLost(f'RESULT pointer sha mismatch for {ident}')
            try:
                rw._publish_bytes(self.root/rel/'RESULT.json', pointer)        # the pointer LAST
            except FileExistsError:
                raise RemoteWaveLost(f'local RESULT pointer already exists for {ident}') from None
            if local_manifest(self.root/rel) != manifest:
                raise RemoteWaveLost(f'landed tree changed while publishing {ident}')
            from .accel_storage import read
            result = read(self.root/rel/'RESULT.json')   # verifies the gzip payload sha (resume adoption path)
            if result.get('id') != ident or result.get('arm') != job['arm']:
                raise RemoteWaveLost(f'landed RESULT does not belong to {ident}')
        _write_new(self.root/rw.STAGE/ident/'LANDED.json', dict(id=ident, update=job.get('update'),
            status='result' if expect_result else 'episode_error', manifest=manifest, opened=value.get('opened'),
            host=getattr(self.transport, 'host', None), unix=time.time()))
        self.landed[ident] = manifest
        return result


def pool_for(root, b, jobs, role, *, transport=None):
    """accel_runner.wave hook: a RemotePool for an arm-R TRAIN wave when COOPMWM_REMOTE_R is set, else None."""
    if not enabled() or not jobs or not remote_wave_jobs(jobs, role):
        return None
    return RemotePool(root, b, jobs, transport=transport)


# ------------------------------------------------------------------ cpuV cleanup

def _pruned_shas(root, update):
    out = {}
    for path in sorted((Path(root)/'pruned').glob(f'PRUNE_u{int(update):03}*.json')):
        for row in json.loads(path.read_text(encoding='utf-8')).get('captures', ()):
            out[Path(row['path']).as_posix()] = row['sha256']
    return out


def reverify_landed(root, record):
    """The landed copy on cpuZ still equals its landing manifest; files deleted by disk_prune count only when
    pruned/PRUNE_u<u>*.json recorded the same sha256."""
    root = Path(root)
    rel = f'jobs/{record["id"]}'
    current = local_manifest(root/rel)
    pruned = _pruned_shas(root, record['update']) if record.get('update') is not None else {}
    for name, sha in record['manifest'].items():
        have = current.get(name)
        if have == sha:
            continue
        if have is None and pruned.get(f'{rel}/{name}') == sha:
            continue
        raise ValueError(f'landed copy of {record["id"]} no longer matches its landing manifest: {name}')
    # cpuZ-side receipts written into the job dir AFTER landing by the unchanged parent (run_train_jobs / resume)
    extra = set(current)-set(record['manifest'])-{'EPISODE_FAILED.json', 'EPISODE_FAILED_RESUMED.json'}
    if extra:
        raise ValueError(f'landed copy of {record["id"]} has unrecorded files: {sorted(extra)[:3]}')


def cleanup_after_monitor(cfg, update, *, transport=None):
    """Delete the replica's copies of update ``update``'s landed R jobs, only after LEARNING_MONITOR_u<u> exists and
    every landed copy re-verifies on cpuZ. Idempotent (CLEANED.json per job)."""
    root = Path(cfg['run_root'])
    if not (root/f'LEARNING_MONITOR_u{int(update):03}.json').exists():
        raise ValueError(f'remote cleanup of u{update} needs LEARNING_MONITOR_u{int(update):03}.json first')
    records = []
    for path in sorted((root/rw.STAGE).glob('*/LANDED.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('update') == int(update) and not (path.parent/'CLEANED.json').exists():
            records.append(record)
    if not records:
        return dict(update=int(update), cleaned={})
    for record in records:
        reverify_landed(root, record)
    if transport is None:
        transport = SshTransport.from_env(cfg.get('source_root') or os.getcwd())
    value = transport.call('cleanup', root, payload=dict(expected={r['id']: r['manifest'] for r in records}),
                           timeout=600)
    outcome = value['cleanup']
    bad = sorted(i for i, state in outcome.items() if state not in ('deleted', 'absent'))
    for record in records:
        if outcome.get(record['id']) in ('deleted', 'absent'):
            _write_new(root/rw.STAGE/record['id']/'CLEANED.json', dict(id=record['id'], update=int(update),
                       remote=outcome[record['id']], unix=time.time()))
    if bad:
        raise RemoteWaveLost('replica copies differ from the landed manifests; not deleted: '+','.join(bad[:5]))
    return dict(update=int(update), cleaned=outcome, admit=value.get('admit'))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Option A remote R wave maintenance (operator)')
    parser.add_argument('action', choices=['cleanup'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--update', type=int, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps(cleanup_after_monitor(cfg, args.update), sort_keys=True))


if __name__ == '__main__':
    main()
