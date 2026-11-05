"""F6: one persistent cpuV SSH master and a user-wide, bounded admission gate.

Run this module on Linux/WSL. Importing it does no I/O. Commands are never
retried: a lost stream can mean that a remote operation already completed.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import time

HOST = 'cpuV'
MIN_INTERVAL_S = 2.0
GATE_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 20
CONTROL_PERSIST_S = 600


def is_cpuv(host):
    return str(host).lower() == HOST.lower()


def command(host, remote_command, *, timeout=None, legacy=None):
    """Route cpuV through the gate, including Popen/streaming callers.

    Other hosts retain their original argv. The child wrapper inherits stdin,
    stdout and stderr, so scripts, JSON RPC and binary streams remain intact.
    """
    if is_cpuv(host):
        return [sys.executable, '-B', str(Path(__file__).resolve()),
                '--timeout', str(timeout if timeout is not None else 0), 'run', '--', remote_command]
    return legacy if legacy is not None else ['ssh', '-o', 'BatchMode=yes', host, remote_command]


def rsync_shell(host):
    if is_cpuv(host):
        return shlex.join([sys.executable, '-B', str(Path(__file__).resolve()), '--timeout', '0', 'rsh'])
    return 'ssh -o BatchMode=yes -o ConnectTimeout=20'


def _require_posix():
    if os.name != 'posix':
        raise RuntimeError('cpuV requires Linux/WSL OpenSSH ControlMaster; native Windows SSH is refused')


def _directory(path):
    path = Path(path)
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError('SSH state directory must be a real private directory')
    if os.name == 'posix' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError('SSH state directory must be owned by this user with mode 0700')
    return path


class Gate:
    """Cross-process lock held through the command; at most one session per client.

    State survives failed clients/commands, so separate agents cannot bypass the
    rate by constructing a new Transport. OS locks are released on process exit.
    The Windows lock implementation is for CPU tests; network use is POSIX-only.
    """
    def __init__(self, directory, *, clock=time.monotonic, wall=time.time, sleep=time.sleep):
        self.directory = Path(directory)
        self.clock, self.wall, self.sleep = clock, wall, sleep

    @contextmanager
    def claim(self):
        _directory(self.directory)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
        with os.fdopen(os.open(self.directory / 'gate.lock', flags, 0o600), 'r+b') as lock:
            if os.name == 'nt':
                import msvcrt
                if os.fstat(lock.fileno()).st_size == 0:
                    lock.write(b'0')
                    lock.flush()
                def acquire():
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                def release():
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                def acquire():
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                def release():
                    fcntl.flock(lock, fcntl.LOCK_UN)
            deadline = self.clock() + GATE_TIMEOUT_S
            while True:
                try:
                    acquire()
                    break
                except OSError as error:
                    if error.errno not in (11, 13, 35, 36):
                        raise
                    if self.clock() >= deadline:
                        raise TimeoutError('cpuV SSH admission gate timed out') from error
                    self.sleep(0.05)
            try:
                yield self
            finally:
                release()

    def admit(self):
        """Call only under claim(), immediately before an SSH network attempt."""
        path = self.directory / 'last_attempt.json'
        if path.exists():
            last = float(json.loads(path.read_text(encoding='utf-8'))['unix'])
            now = self.wall()
            if not math.isfinite(last) or not math.isfinite(now) or now < last:
                raise ValueError('invalid/backwards cpuV SSH admission clock; refusing connection')
            remaining = last + MIN_INTERVAL_S - now
            deadline = self.clock() + GATE_TIMEOUT_S
            while remaining > 0:
                if self.clock() >= deadline:
                    raise TimeoutError('cpuV SSH rate admission timed out')
                self.sleep(remaining)
                current = self.wall()
                if not math.isfinite(current) or current < now:
                    raise ValueError('invalid/backwards cpuV SSH admission clock')
                now = current
                remaining = last + MIN_INTERVAL_S - now
        now = self.wall()
        if not math.isfinite(now):
            raise ValueError('invalid cpuV SSH admission clock')
        # Write before spawning, including attempts that fail during connect.
        with path.open('w', encoding='utf-8') as stream:
            json.dump(dict(unix=now), stream)
            stream.flush()
            os.fsync(stream.fileno())


class Transport:
    def __init__(self, directory=None):
        _require_posix()
        # Shared by all worktrees/agents for this OS user, never by arm/run/PID.
        self.directory = Path(directory) if directory is not None else Path('/tmp') / f'c4-cpuv-{os.getuid()}'
        self.socket = self.directory / 'master.sock'
        if len(os.fsencode(self.socket)) > 100:
            raise ValueError('cpuV ControlPath exceeds portable Unix socket length')
        self.gate = Gate(self.directory)

    def _options(self, *, master=False):
        values = [f'ControlMaster={"yes" if master else "no"}',
                  f'ControlPersist={CONTROL_PERSIST_S}', f'ControlPath={self.socket}',
                  'BatchMode=yes', f'ConnectTimeout={CONNECT_TIMEOUT_S}',
                  'ConnectionAttempts=1', 'ServerAliveInterval=15', 'ServerAliveCountMax=3',
                  'StrictHostKeyChecking=yes', 'ForwardAgent=no', 'ForwardX11=no',
                  'ClearAllForwardings=yes', 'RequestTTY=no']
        if not master:
            # OpenSSH otherwise silently opens a fresh TCP connection if its
            # master dies between check and use. No direct fallback is allowed.
            values.insert(0, 'ProxyCommand=false')
        return [part for value in values for part in ('-o', value)]

    def _check(self):
        return subprocess.run(['ssh', *self._options(), '-O', 'check', HOST],
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=5).returncode == 0

    def _master(self):
        if self._check():
            return
        self.gate.admit()
        # A background ControlPersist process may retain inherited descriptors.
        # PIPE/communicate would then wait for its entire 600-second lifetime.
        with tempfile.TemporaryFile(dir=self.directory) as errors:
            try:
                subprocess.run(['ssh', *self._options(master=True), '-M', '-N', '-f', HOST],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                               timeout=CONNECT_TIMEOUT_S + 10, check=True)
            except subprocess.CalledProcessError as error:
                errors.seek(0)
                error.stderr = errors.read()
                raise
        if not self._check():
            raise RuntimeError('cpuV SSH master did not publish a usable control socket; no fallback')

    def run(self, remote_command, *, timeout=180, **kwargs):
        if not isinstance(remote_command, str) or not remote_command:
            raise ValueError('one nonempty remote command string is required')
        with self.gate.claim():
            self._master()
            self.gate.admit()
            return subprocess.run(['ssh', *self._options(), '-T', HOST, remote_command],
                                  timeout=timeout, **kwargs)

    def copy(self, local, remote, *, upload, timeout=180, **kwargs):
        """SCP/SFTP uses exactly the same master, admission gate and no-fallback rule."""
        if not remote.startswith('/'):
            raise ValueError('cpuV copy requires an absolute remote path')
        local = str(Path(local).absolute())
        pair = [local, HOST + ':' + remote] if upload else [HOST + ':' + remote, local]
        with self.gate.claim():
            self._master()
            self.gate.admit()
            return subprocess.run(['scp', *self._options(), '--', *pair], timeout=timeout, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=180, help='command seconds; 0 inherits caller lifetime')
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('run').add_argument('command', nargs=argparse.REMAINDER)
    sub.add_parser('rsh').add_argument('command', nargs=argparse.REMAINDER)
    for mode in ('put', 'get'):
        copy = sub.add_parser(mode)
        copy.add_argument('local')
        copy.add_argument('remote')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error('--timeout must be nonnegative and finite')
    timeout = args.timeout or None
    if args.mode in ('run', 'rsh'):
        words = args.command[1:] if args.command[:1] == ['--'] else args.command
        if args.mode == 'rsh':
            if not words or not is_cpuv(words[0]):
                parser.error('rsync remote shell requires the cpuV alias, without SSH options')
            words = words[1:]
        if not words:
            parser.error('run requires a remote command')
        remote = words[0] if len(words) == 1 else shlex.join(words)
    try:
        transport = Transport()
        result = (transport.run(remote, timeout=timeout) if args.mode in ('run', 'rsh') else
                  transport.copy(args.local, args.remote, upload=args.mode == 'put', timeout=timeout))
        return result.returncode
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print('cpuV SSH refused/failed; remote outcome may be unknown: ' + str(error), file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.decode(errors='replace'), file=sys.stderr)
        return 255


if __name__ == '__main__':
    sys.exit(main())
