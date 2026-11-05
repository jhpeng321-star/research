"""F6 CPU-only tests: fake SSH, real cross-process admission and no physics."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
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
    monkeypatch.setattr(subprocess, 'run', runner)
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
    assert argv[:6] == [sys.executable, '-B', str(Path(ssh.__file__).resolve()), '--timeout', '91', 'run']
    assert argv[6] == '--' and argv[-1].startswith('/qualified/python -c ')
    assert runner.call_args.kwargs == dict(input='print(1)', capture_output=True, text=True, timeout=91)
    ops.remote('cpuQ', 'print(1)', timeout=91)
    assert runner.call_args.args[0][:4] == ['ssh', '-o', 'BatchMode=yes', 'cpuQ']
    assert ops.command('cpuv', 'read receipt') == ssh.command('cpuV', 'read receipt')


def test_rpc_and_rsync_are_routed_only_for_cpuv():
    from . import cpus_checking
    assert cpus_checking.ssh_command('cpuV', 'rpc') == ssh.command('cpuV', 'rpc', timeout=600)
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
