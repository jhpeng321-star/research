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
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid

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
        budget = _deadline(timeout or None)
        return [sys.executable, '-B', str(Path(__file__).resolve()),
                '--timeout', str(timeout if timeout is not None else 0),
                *(['--deadline', str(budget)] if budget is not None else []),
                'run', '--', remote_command]
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
    rate by constructing a new Transport. A durable active-session reservation
    also excludes callers after the wrapper dies, until its supervisor reaps the
    actual session. An interrupted supervisor fails closed (operator recovery).
    The Windows lock implementation is for CPU tests; network use is POSIX-only.
    """
    def __init__(self, directory, *, clock=time.monotonic, wall=time.time, sleep=time.sleep):
        self.directory = Path(directory)
        self.clock, self.wall, self.sleep = clock, wall, sleep

    @contextmanager
    def claim(self, *, deadline=None):
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
            limit = self.clock() + GATE_TIMEOUT_S
            if deadline is not None:
                limit = min(limit, deadline)
            while True:
                try:
                    acquire()
                    break
                except OSError as error:
                    if error.errno not in (11, 13, 35, 36):
                        raise
                    if self.clock() >= limit:
                        raise TimeoutError('cpuV SSH admission gate timed out') from error
                    self.sleep(0.05)
            try:
                while (self.directory / 'active.json').exists():
                    if self.clock() >= limit:
                        raise TimeoutError('cpuV SSH active session still owns admission; operator recovery may be required')
                    self.sleep(0.05)
                _remaining(deadline, self.clock)
                yield self
            finally:
                release()

    def admit(self, *, deadline=None):
        """Call only under claim(), immediately before an SSH network attempt."""
        path = self.directory / 'last_attempt.json'
        if path.exists():
            last = float(json.loads(path.read_text(encoding='utf-8'))['unix'])
            now = self.wall()
            if not math.isfinite(last) or not math.isfinite(now) or now < last:
                raise ValueError('invalid/backwards cpuV SSH admission clock; refusing connection')
            remaining = last + MIN_INTERVAL_S - now
            limit = self.clock() + GATE_TIMEOUT_S
            if deadline is not None:
                limit = min(limit, deadline)
            while remaining > 0:
                if self.clock() >= limit:
                    raise TimeoutError('cpuV SSH rate admission timed out')
                self.sleep(min(remaining, max(0, limit - self.clock())))
                current = self.wall()
                if not math.isfinite(current) or current < now:
                    raise ValueError('invalid/backwards cpuV SSH admission clock')
                now = current
                remaining = last + MIN_INTERVAL_S - now
        _remaining(deadline, self.clock)
        now = self.wall()
        if not math.isfinite(now):
            raise ValueError('invalid cpuV SSH admission clock')
        # Write before spawning, including attempts that fail during connect.
        with path.open('w', encoding='utf-8') as stream:
            json.dump(dict(unix=now), stream)
            stream.flush()
            os.fsync(stream.fileno())


def _deadline(timeout, clock=time.monotonic):
    if timeout is None:
        return None
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('timeout must be nonnegative and finite')
    return clock() + timeout


def _remaining(deadline, clock=time.monotonic, cap=None):
    remaining = None if deadline is None else deadline - clock()
    if remaining is not None and remaining <= 0:
        raise TimeoutError('cpuV SSH end-to-end deadline expired')
    return cap if remaining is None else remaining if cap is None else min(cap, remaining)


def _parent_watch(pid):
    """Hold an identity-stable parent handle on Windows; POSIX uses reparenting."""
    if os.name != 'nt':
        return lambda: os.getppid() != pid, lambda: None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return lambda: True, lambda: None
    return lambda: kernel.WaitForSingleObject(handle, 0) != 258, lambda: kernel.CloseHandle(handle)


def _stop_session(child):
    """Reap the owned session/group, never the separate persistent master."""
    def send(sig):
        try:
            if os.name == 'posix':
                os.killpg(child.pid, sig)
            elif child.poll() is None:
                child.terminate() if sig == signal.SIGTERM else child.kill()
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # Also stop an scp-spawned ssh, even if the scp group leader exited first.
    send(signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
    child.wait()


def _session_supervisor(active):
    """Internal local child: survives wrapper death, owns cleanup/reservation.

    The reservation is published BEFORE spawning this supervisor. Even death in
    that launch window cannot release admission. Only successful reaping clears
    it; no PID probing or elapsed-time heuristic can admit an overlapping session.
    """
    active = Path(active)
    cfg = json.loads(active.read_text(encoding='utf-8'))
    cancelled = False
    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, cancel)
    dead, close = _parent_watch(cfg['parent'])
    child = None
    result = dict(returncode=255, timeout=False)
    try:
        # A wrapper cancelled during startup must not dispatch a remote mutation.
        if dead() or Path(cfg['cancel']).exists() or cancelled:
            return 255
        if cfg['deadline'] is not None and time.monotonic() >= cfg['deadline']:
            result['timeout'] = True
            return 255
        child = subprocess.Popen(cfg['argv'], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
                                 start_new_session=os.name == 'posix')
        while True:
            expired = cfg['deadline'] is not None and time.monotonic() >= cfg['deadline']
            if dead() or cancelled or Path(cfg['cancel']).exists() or expired:
                result['timeout'] = expired
                _stop_session(child)
                break
            try:
                result['returncode'] = child.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
        return result['returncode']
    finally:
        # If cleanup itself fails, keep active.json: admission must fail closed.
        if child is not None and child.poll() is None:
            _stop_session(child)
        if not dead():
            Path(cfg['result']).write_text(json.dumps(result), encoding='utf-8')
        close()
        active.unlink()
        Path(cfg['cancel']).unlink(missing_ok=True)


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

    def _owned_run(self, argv, *, timeout, **kwargs):
        """A local supervisor owns the real child; this wrapper never kills it.

        communicate() has no competing outer timeout. Cancellation requests
        cleanup; abrupt wrapper death is detected independently by the supervisor.
        """
        directory = self.directory.resolve()
        token = uuid.uuid4().hex
        result_path, cancel_path = directory / (token + '.result'), directory / (token + '.cancel')
        active = directory / 'active.json'
        cfg = dict(argv=argv, parent=os.getpid(), deadline=_deadline(timeout),
                   result=str(result_path), cancel=str(cancel_path))
        input_data = kwargs.pop('input', None)
        check = kwargs.pop('check', False)
        if input_data is not None:
            if kwargs.get('stdin') is not None:
                raise ValueError('stdin and input arguments may not both be used')
            kwargs['stdin'] = subprocess.PIPE
        if kwargs.pop('capture_output', False):
            if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
                raise ValueError('stdout and stderr arguments may not be used with capture_output')
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Explicit standard handles preserve redirected streams through both
        # local process hops, including Windows CPU lifecycle tests.
        kwargs.setdefault('stdin', sys.stdin)
        kwargs.setdefault('stdout', sys.stdout)
        kwargs.setdefault('stderr', sys.stderr)
        # Exclusive creation plus the surrounding gate prevent reservation races.
        with active.open('x', encoding='utf-8') as stream:
            json.dump(cfg, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Never clear a reservation from the wrapper, even if process creation
        # raises: interruption can occur after the OS has already started it.
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                    '_session', str(active)], **kwargs)
        try:
            try:
                stdout, stderr = process.communicate(input_data)
            except BaseException:
                cancel_path.touch()
                process.communicate()  # supervisor terminates/reaps before returning
                raise
            if not result_path.exists():
                raise RuntimeError('cpuV SSH supervisor failed; active reservation retained; operator recovery required')
            report = json.loads(result_path.read_text(encoding='utf-8'))
            if report['timeout']:
                raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr)
            result = subprocess.CompletedProcess(argv, report['returncode'], stdout, stderr)
            if check:
                result.check_returncode()
            return result
        finally:
            result_path.unlink(missing_ok=True)
            cancel_path.unlink(missing_ok=True)

    def _check(self, deadline):
        return self._owned_run(['ssh', *self._options(), '-O', 'check', HOST],
                               stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=_remaining(deadline, self.gate.clock, cap=5)).returncode == 0

    def _master(self, deadline):
        if self._check(deadline):
            return
        self.gate.admit(deadline=deadline)
        # A background ControlPersist process may retain inherited descriptors.
        # PIPE/communicate would then wait for its entire 600-second lifetime.
        with tempfile.TemporaryFile(dir=self.directory) as errors:
            try:
                self._owned_run(['ssh', *self._options(master=True), '-M', '-N', '-f', HOST],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                                timeout=_remaining(deadline, self.gate.clock, cap=CONNECT_TIMEOUT_S + 10), check=True)
            except subprocess.CalledProcessError as error:
                errors.seek(0)
                error.stderr = errors.read()
                raise
        if not self._check(deadline):
            raise RuntimeError('cpuV SSH master did not publish a usable control socket; no fallback')

    def run(self, remote_command, *, timeout=180, **kwargs):
        if not isinstance(remote_command, str) or not remote_command:
            raise ValueError('one nonempty remote command string is required')
        deadline = _deadline(timeout, self.gate.clock)
        with self.gate.claim(deadline=deadline):
            self._master(deadline)
            self.gate.admit(deadline=deadline)
            return self._owned_run(['ssh', *self._options(), '-T', HOST, remote_command],
                                   timeout=_remaining(deadline, self.gate.clock), **kwargs)

    def copy(self, local, remote, *, upload, timeout=180, **kwargs):
        """SCP/SFTP uses exactly the same master, admission gate and no-fallback rule."""
        if not remote.startswith('/'):
            raise ValueError('cpuV copy requires an absolute remote path')
        local = str(Path(local).absolute())
        pair = [local, HOST + ':' + remote] if upload else [HOST + ':' + remote, local]
        deadline = _deadline(timeout, self.gate.clock)
        with self.gate.claim(deadline=deadline):
            self._master(deadline)
            self.gate.admit(deadline=deadline)
            return self._owned_run(['scp', *self._options(), '--', *pair],
                                   timeout=_remaining(deadline, self.gate.clock), **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=180, help='end-to-end seconds; 0 inherits caller lifetime')
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
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
        if args.deadline is not None:
            if not math.isfinite(args.deadline):
                raise ValueError('deadline must be finite')
            timeout = _remaining(args.deadline, cap=timeout)
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
    if len(sys.argv) == 3 and sys.argv[1] == '_session':
        sys.exit(_session_supervisor(sys.argv[2]))
    sys.exit(main())
