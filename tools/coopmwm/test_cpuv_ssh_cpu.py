"""F6 CPU-only tests: fake SSH, real cross-process admission and no physics."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from . import cpuv_ssh as ssh


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original = subprocess.Popen
    def guarded(argv, *args, **kwargs):
        assert Path(argv[0]).stem.lower() not in ('ssh', 'scp', 'sftp', 'rsync', 'sumo', 'sumo-gui')
        return original(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', guarded)


class Clock:
    def __init__(self):
        self.now = 100.0
    def time(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def transport(tmp_path, monkeypatch, runner):
    # Exercise protocol construction on Windows without running native SSH.
    monkeypatch.setattr(ssh, '_require_posix', lambda: None)
    monkeypatch.setattr(ssh.Transport, '_owned_run', lambda self, argv, **kwargs: runner(argv, **kwargs))
    # A short relative path also exercises the Unix socket path limit on Windows.
    monkeypatch.chdir(tmp_path)
    client = ssh.Transport('pool')
    clock = Clock()
    client.gate = ssh.Gate(client.directory, clock=clock.time, wall=clock.time, sleep=clock.sleep)
    return client, clock


def ready(*args, **kwargs):
    return subprocess.CompletedProcess(args[0], 0, stdout=b'receipt', stderr=b'')


def option(argv, key):
    return next(value.split('=', 1)[1] for value in argv if value.startswith(key + '='))


def test_first_use_creates_master_and_reuses_socket(tmp_path, monkeypatch):
    checks = iter([1, 0, 0])
    calls = []
    client = None
    def runner(argv, **kwargs):
        calls.append((argv, kwargs, clock.time()))
        return subprocess.CompletedProcess(argv, next(checks) if '-O' in argv else 0,
                                           stdout=b'receipt', stderr=b'')
    client, clock = transport(tmp_path, monkeypatch, runner)
    assert client.run('printf receipt', input=b'input', capture_output=True).stdout == b'receipt'
    # New client/gate object must still use the same persisted rate and socket.
    second = ssh.Transport(client.directory)
    second.gate = ssh.Gate(client.directory, clock=clock.time, wall=clock.time, sleep=clock.sleep)
    second.run('second')
    masters = [(argv, stamp) for argv, _, stamp in calls if '-M' in argv]
    sessions = [(argv, kwargs, stamp) for argv, kwargs, stamp in calls if '-T' in argv]
    assert len(masters) == 1 and len(sessions) == 2
    assert [masters[0][1], sessions[0][2], sessions[1][2]] == [100, 102, 104]
    for argv, _, _ in calls:
        assert option(argv, 'ControlPath') == str(client.socket)
        assert option(argv, 'ControlPersist') == '600'
        assert option(argv, 'ConnectionAttempts') == '1'
        assert option(argv, 'BatchMode') == 'yes'
        assert option(argv, 'StrictHostKeyChecking') == 'yes'
        assert argv[-1] == 'cpuV' or argv[-2] == 'cpuV'
    assert option(masters[0][0], 'ControlMaster') == 'yes'
    master_kwargs = next(kwargs for argv, kwargs, _ in calls if '-M' in argv)
    assert master_kwargs['stdout'] == subprocess.DEVNULL
    assert master_kwargs['stderr'] != subprocess.PIPE
    for argv, _, _ in sessions:
        assert option(argv, 'ControlMaster') == 'no'
        assert option(argv, 'ProxyCommand') == 'false'
    assert sessions[0][1]['input'] == b'input'
    assert sessions[0][1]['capture_output'] is True


@pytest.mark.parametrize('failure', ['master_exit', 'master_timeout', 'missing_socket', 'lost_session'])
def test_failures_never_replay_and_keep_admission(tmp_path, monkeypatch, failure):
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        if '-O' in argv:
            return subprocess.CompletedProcess(argv, 0 if failure == 'lost_session' else 1)
        if '-M' in argv and failure == 'master_exit':
            raise subprocess.CalledProcessError(255, argv, stderr=b'connection closed')
        if '-M' in argv and failure == 'master_timeout':
            raise subprocess.TimeoutExpired(argv, 30)
        return subprocess.CompletedProcess(argv, 255 if failure == 'lost_session' else 0)
    client, clock = transport(tmp_path, monkeypatch, runner)
    if failure == 'lost_session':
        assert client.run('one-shot mutation').returncode == 255
        assert len([argv for argv in calls if '-T' in argv]) == 1
        assert not any('-M' in argv for argv in calls)
    else:
        with pytest.raises((subprocess.SubprocessError, RuntimeError)):
            client.run('one-shot mutation')
        assert len([argv for argv in calls if '-M' in argv]) == 1
        assert not any('-T' in argv for argv in calls)
    stamp = json.loads((client.directory / 'last_attempt.json').read_text())['unix']
    assert stamp == clock.time() == 100
    # Failed commands release the lock; next attempt cannot skip the interval.
    with client.gate.claim():
        client.gate.admit()
    assert clock.time() == 102


@pytest.mark.parametrize('upload', [True, False])
def test_copy_uses_same_master_and_no_direct_fallback(tmp_path, monkeypatch, upload):
    runner = Mock(side_effect=ready)
    client, _ = transport(tmp_path, monkeypatch, runner)
    client.copy('local file', '/remote file', upload=upload)
    argv = runner.call_args.args[0]
    assert argv[0] == 'scp'
    assert option(argv, 'ControlMaster') == 'no'
    assert option(argv, 'ProxyCommand') == 'false'
    assert option(argv, 'ControlPath') == str(client.socket)
    remote = 'cpuV:/remote file'
    assert argv[-1 if upload else -2] == remote
    with pytest.raises(ValueError, match='absolute'):
        client.copy('local', 'relative', upload=upload)


def test_common_runtime_routing_preserves_bytes_and_legacy_hosts(monkeypatch):
    from tools.coopbandit import ops
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, stdout='receipt', stderr=''))
    monkeypatch.setattr(ops.subprocess, 'run', runner)
    monkeypatch.setitem(ops.PYTHON, 'cpuV', '/qualified/python')
    assert ops.remote('cpuV', 'print(1)', timeout=91) == 'receipt'
    argv = runner.call_args.args[0]
    assert argv[:5] == [sys.executable, '-B', str(Path(ssh.__file__).resolve()), '--timeout', '91']
    assert argv[5] == '--deadline' and float(argv[6]) > time.monotonic()
    assert argv[7:9] == ['run', '--'] and argv[-1].startswith('/qualified/python -c ')
    assert runner.call_args.kwargs == dict(input='print(1)', capture_output=True, text=True, timeout=None)
    ops.remote('cpuQ', 'print(1)', timeout=91)
    assert runner.call_args.args[0][:4] == ['ssh', '-o', 'BatchMode=yes', 'cpuQ']
    assert ops.command('cpuv', 'read receipt') == ssh.command('cpuV', 'read receipt')


def test_rpc_and_rsync_are_routed_only_for_cpuv():
    from . import cpus_checking
    argv = cpus_checking.ssh_command('cpuV', 'rpc')
    assert argv[:6] == ssh.command('cpuV', 'rpc', timeout=600)[:6]
    assert argv[7:] == ['run', '--', 'rpc']
    assert cpus_checking.ssh_command('cpuQ', 'rpc') == [
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'cpuQ', 'rpc']
    assert shlex.split(cpus_checking.rsync_shell('cpuV')) == [
        sys.executable, '-B', str(Path(ssh.__file__).resolve()), '--timeout', '0', 'rsh']
    assert cpus_checking.rsync_shell('cpuQ') == 'ssh -o BatchMode=yes -o ConnectTimeout=20'


@pytest.mark.parametrize('words, expected', [
    (['run', '--', 'printf "%s" value'], 'printf "%s" value'),
    (['run', '--', 'printf', '%s', 'space value'], "printf %s 'space value'"),
    (['rsh', 'cpuV', 'rsync', '--server', 'space path'], "rsync --server 'space path'"),
])
def test_cli_command_boundaries_and_exit_status(monkeypatch, words, expected):
    client = Mock()
    client.run.return_value.returncode = 7
    monkeypatch.setattr(ssh, 'Transport', Mock(return_value=client))
    assert ssh.main(words) == 7
    client.run.assert_called_once_with(expected, timeout=180)


def test_streaming_and_rsync_preserve_external_lifetime(monkeypatch):
    client = Mock()
    client.run.return_value.returncode = 0
    monkeypatch.setattr(ssh, 'Transport', Mock(return_value=client))
    assert ssh.command('cpuV', 'stream')[3:7] == ['--timeout', '0', 'run', '--']
    assert ssh.main(['--timeout', '0', 'rsh', 'cpuV', 'rsync', '--server']) == 0
    client.run.assert_called_once_with('rsync --server', timeout=None)


@pytest.mark.parametrize('words', [['run'], ['rsh', 'cpuQ', 'rsync'],
                                  ['--timeout', 'nan', 'run', '--', 'true']])
def test_cli_rejects_invalid_requests_before_transport(monkeypatch, words):
    constructor = Mock()
    monkeypatch.setattr(ssh, 'Transport', constructor)
    with pytest.raises(SystemExit):
        ssh.main(words)
    constructor.assert_not_called()


def test_windows_native_fails_before_any_subprocess(monkeypatch):
    # Patch only the platform value used by the guard, without changing pathlib.
    monkeypatch.setattr(ssh, 'os', SimpleNamespace(name='nt'))
    runner = Mock()
    monkeypatch.setattr(subprocess, 'run', runner)
    with pytest.raises(RuntimeError, match='Linux/WSL'):
        ssh.Transport()
    runner.assert_not_called()


def test_corrupt_or_backwards_rate_state_fails_closed(tmp_path):
    directory = tmp_path / 'pool'
    clock = Clock()
    gate = ssh.Gate(directory, clock=clock.time, wall=clock.time, sleep=clock.sleep)
    with gate.claim():
        gate.admit()
    for text in ('broken', '{"unix": NaN}', '{"unix": 101}'):
        (directory / 'last_attempt.json').write_text(text)
        with gate.claim(), pytest.raises(ValueError):
            gate.admit()


def test_gate_releases_after_exception_and_has_bounded_wait(tmp_path):
    directory = tmp_path / 'pool'
    clock = Clock()
    first = ssh.Gate(directory)
    other = ssh.Gate(directory, clock=clock.time, wall=clock.time, sleep=clock.sleep)
    with first.claim(), pytest.raises(TimeoutError, match='gate timed out'):
        with other.claim():
            pytest.fail('second lock admitted during live command')
    with pytest.raises(RuntimeError):
        with first.claim():
            raise RuntimeError('failed command')
    with other.claim():
        pass


def test_real_processes_share_gate_and_attempt_spacing(tmp_path):
    # Only local Python subprocesses; no SSH executable can be reached here.
    directory = tmp_path / 'shared'
    code = '''
import sys,time,json
from pathlib import Path
from tools.coopmwm import cpuv_ssh as s
s.MIN_INTERVAL_S = 0.15
gate=s.Gate(Path(sys.argv[1]))
with gate.claim():
    gate.admit()
    begin=time.time()
    time.sleep(0.05)
    with (gate.directory/'events.jsonl').open('a') as f:
        f.write(json.dumps([begin,time.time()])+'\\n')
'''
    root = Path(__file__).resolve().parents[2]
    children = [subprocess.Popen([sys.executable, '-B', '-c', code, str(directory)],
                                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    for child in children:
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, (stdout, stderr)
    rows = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    for previous, current in zip(rows, rows[1:]):
        assert current[0] >= previous[1]
        assert current[0] - previous[0] >= 0.14


LOCAL_WRAPPER = r'''
import json,os,sys,time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.coopmwm import cpuv_ssh as s
s._require_posix = lambda: None
s.MIN_INTERVAL_S = 0
client = s.Transport('pool')
def master(deadline):
    time.sleep(float(sys.argv[4]))
client._master = master
owned = client._owned_run
child = r"""
import json,os,signal,sys,time
from pathlib import Path
events=Path(sys.argv[1])
with events.open('a') as stream:
    stream.write(json.dumps(dict(pid=os.getpid(), event='start'))+'\n')
    stream.flush()
if os.name == 'posix':
    def stopping(sig, frame):
        # Keep the real child alive through a visible cleanup interval.
        time.sleep(.35)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stopping)
if sys.argv[2] == 'echo':
    sys.stdout.buffer.write(sys.stdin.buffer.read())
    sys.stdout.buffer.flush()
else:
    time.sleep(30)
"""
def local(argv, **kwargs):
    assert argv[0] in ('ssh', 'scp')
    return owned([sys.executable, '-B', '-c', child, str(Path('events.jsonl').resolve()),
                  sys.argv[5]], **kwargs)
client._owned_run = local
timeout = None if sys.argv[2] == 'none' else float(sys.argv[2])
try:
    if sys.argv[3] == 'copy':
        client.copy('local', '/remote', upload=True, timeout=timeout)
    else:
        client.run('never sent to a network', timeout=timeout)
except (TimeoutError, s.subprocess.TimeoutExpired):
    sys.exit(255)
'''


def local_argv(timeout='none', mode='run', setup=0, behavior='sleep'):
    return [sys.executable, '-B', '-c', LOCAL_WRAPPER, str(Path(__file__).resolve().parents[2]),
            str(timeout), mode, str(setup), behavior]


def wait_for_child(tmp_path, wrapper):
    events = tmp_path / 'events.jsonl'
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if events.exists() and events.read_text().strip():
            rows = [json.loads(line) for line in events.read_text().splitlines()]
            assert len(rows) == 1  # The operation must never be replayed.
            return rows[0]['pid']
        assert wrapper.poll() is None, wrapper.communicate()
        time.sleep(.02)
    pytest.fail('local fake session did not start')


def child_watch(pid):
    if os.name == 'nt':
        return ssh._parent_watch(pid)
    def dead():
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            return True
    return dead, lambda: None


def assert_exclusion_until_reaped(tmp_path, dead, monkeypatch):
    monkeypatch.setattr(ssh, 'GATE_TIMEOUT_S', .03)
    gate = ssh.Gate(tmp_path / 'pool')
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            with gate.claim():
                assert dead(), 'admission released while actual command was alive'
                break
        except TimeoutError:
            pass
    else:
        pytest.fail('supervisor did not reap session and clear admission')
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('cancellation', ['terminate', 'kill', 'outer_timeout'])
@pytest.mark.parametrize('mode', ['run', 'copy'])
def test_real_wrapper_cancellation_keeps_gate_until_session_reaped(tmp_path, monkeypatch, cancellation, mode):
    wrapper = subprocess.Popen(local_argv(mode=mode), cwd=tmp_path,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dead = close = None
    try:
        pid = wait_for_child(tmp_path, wrapper)
        dead, close = child_watch(pid)
        assert not dead()
        # The gate excludes a second process before cancellation too.
        monkeypatch.setattr(ssh, 'GATE_TIMEOUT_S', .03)
        with pytest.raises(TimeoutError):
            with ssh.Gate(tmp_path / 'pool').claim():
                pytest.fail('overlapping command admitted')
        if cancellation == 'outer_timeout':
            # This is exactly subprocess.run's outer-timeout cancellation path.
            with pytest.raises(subprocess.TimeoutExpired):
                wrapper.communicate(timeout=.02)
            wrapper.kill()
        else:
            getattr(wrapper, cancellation)()
        wrapper.wait(timeout=5)
        assert_exclusion_until_reaped(tmp_path, dead, monkeypatch)
        wrapper.communicate(timeout=5)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=5)
        if dead is not None and not dead():
            # Only our local test process; never launch SSH for cleanup.
            if os.name == 'posix':
                os.kill(pid, 9)
            else:
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32')
                kernel.OpenProcess.restype = wintypes.HANDLE
                handle = kernel.OpenProcess(1, False, pid)
                if handle:
                    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
                    kernel.TerminateProcess(handle, 1)
                    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                    kernel.CloseHandle(handle)
        if close is not None:
            close()


@pytest.mark.parametrize('caller', ['ops', 'rpc'])
def test_real_rpc_deadline_allows_inner_cleanup_and_never_replays(tmp_path, monkeypatch, caller):
    # Real wrapper + supervisor + local child. Master/setup consumes part of the
    # deadline; a competing outer timeout would kill the wrapper during cleanup.
    from tools.coopbandit import ops
    from . import cpus_checking
    monkeypatch.chdir(tmp_path)
    argv = local_argv(timeout=5, setup=.5)
    if caller == 'ops':
        monkeypatch.setitem(ops.PYTHON, 'cpuV', '/unused/python')
        monkeypatch.setattr(ops, 'command', lambda host, command, timeout: argv)
        invoke = lambda: ops.remote('cpuV', 'payload', timeout=5)
        error = RuntimeError
    else:
        monkeypatch.setattr(cpus_checking, 'ssh_command', lambda host, command, timeout: argv)
        cfg = dict(q_host='cpuV', q_pythonpath='/unused', q_python='/unused/python', q_control='/unused')
        invoke = lambda: cpus_checking.rpc(cfg, 'read', timeout=5)
        error = cpus_checking.RpcError
    with pytest.raises(error):
        invoke()
    rows = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    dead, close = child_watch(rows[0]['pid'])
    try:
        assert dead(), 'RPC returned before the owned session was reaped'
        with ssh.Gate(tmp_path / 'pool').claim():
            pass
    finally:
        close()


def test_supervised_binary_input_output_and_exit_status(tmp_path):
    payload = bytes(range(128))  # Includes NUL; input is not stored in reservation.
    result = subprocess.run(local_argv(timeout=10, behavior='echo'), cwd=tmp_path,
                            input=payload, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout == payload
    assert not (tmp_path / 'pool' / 'active.json').exists()


def test_actual_subprocess_outer_timeout_reaps_owned_local_session(tmp_path, monkeypatch):
    # Exercise subprocess.run's real kill/communicate timeout machinery as an
    # external RPC owner would, not just a mocked timeout kwarg.
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(local_argv(), cwd=tmp_path, capture_output=True, timeout=4)
    rows = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    dead, close = child_watch(rows[0]['pid'])
    try:
        assert_exclusion_until_reaped(tmp_path, dead, monkeypatch)
    finally:
        close()


def test_supervisor_spawn_interruption_keeps_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(ssh, '_require_posix', lambda: None)
    monkeypatch.chdir(tmp_path)
    client = ssh.Transport('pool')
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('could have happened after OS process creation')
    with client.gate.claim():
        monkeypatch.setattr(subprocess, 'Popen', interrupted)
        with pytest.raises(KeyboardInterrupt):
            client._owned_run(['local-test-only'], timeout=None)
    assert (client.directory / 'active.json').exists()
    monkeypatch.setattr(ssh, 'GATE_TIMEOUT_S', .03)
    with pytest.raises(TimeoutError, match='active session'):
        with client.gate.claim():
            pytest.fail('spawn interruption released uncertain session admission')


def test_unfinished_reservation_fails_closed_even_without_wrapper_lock(tmp_path, monkeypatch):
    # Covers abrupt death between durable reservation and supervisor creation,
    # as well as death of the supervisor itself: never guess that a session ended.
    directory = tmp_path / 'pool'
    directory.mkdir(mode=0o700)
    (directory / 'active.json').write_text('partially written reservation')
    monkeypatch.setattr(ssh, 'GATE_TIMEOUT_S', .03)
    with pytest.raises(TimeoutError, match='active session'):
        with ssh.Gate(directory).claim():
            pytest.fail('uncertain session lifetime admitted another operation')


def test_deadline_includes_gate_master_and_rate_waits(tmp_path, monkeypatch):
    calls = []
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return ready(argv, **kwargs)
    client, clock = transport(tmp_path, monkeypatch, runner)
    first = ssh.Gate(client.directory)
    with first.claim(), pytest.raises(TimeoutError):
        client.run('no gate budget', timeout=.1)
    assert calls == []
    def slow_master(deadline):
        clock.sleep(.8)
    monkeypatch.setattr(client, '_master', slow_master)
    client.run('remaining command budget', timeout=1)
    assert calls[-1][1]['timeout'] == pytest.approx(.2)
    calls.clear()
    with pytest.raises(TimeoutError, match='rate admission'):
        client.run('no rate budget', timeout=1)
    assert calls == []


def test_expired_wrapper_startup_deadline_never_constructs_transport(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(ssh, 'Transport', constructor)
    assert ssh.main(['--deadline', str(time.monotonic() - 1), 'run', '--', 'mutation']) == 255
    constructor.assert_not_called()


@pytest.mark.skipif(os.name != 'posix', reason='Unix permissions/socket symlinks are POSIX-only')
def test_unsafe_shared_directory_refused(tmp_path):
    public = tmp_path / 'public'
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='0700'):
        with ssh.Gate(public).claim():
            pytest.fail('public directory admitted')
    link = tmp_path / 'link'
    link.symlink_to(public, target_is_directory=True)
    with pytest.raises(ValueError, match='real private directory'):
        with ssh.Gate(link).claim():
            pytest.fail('symlink admitted')
