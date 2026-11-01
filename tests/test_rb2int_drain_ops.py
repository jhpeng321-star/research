"""Exercise intervention guards with a fake /proc tree and intercepted pidfd signals."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from tools.rb2int.drain_ops import CONTROL, EXPECTED, remote_code
from tools.rb2int.final_ops import REMOTE_ROOT


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root, control, proc = (tmp_path/name for name in ('root', 'control', 'proc'))
    (control/'tools/rb2int').mkdir(parents=True)
    source = control/'tools/rb2int/final_worker.py'
    source.write_text('# frozen source\n')
    root.mkdir()
    expected = EXPECTED['gpuA']
    (root/'SUPERVISOR_PROCESS.json').write_text(json.dumps(dict(expected, host='gpuA', root=str(root))))
    (root/'LAUNCH_ONCE.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
    (proc/'sys/kernel/random').mkdir(parents=True)
    (proc/'sys/kernel/random/boot_id').write_text(expected['boot_id'])

    def process(pid, ticks, state='S'):
        directory = proc/str(pid)
        directory.mkdir(exist_ok=True)
        fields = [str(pid), '(python)', state]+['0']*19
        fields[21] = ticks
        (directory/'stat').write_text(' '.join(fields))
        return directory

    supervisor = process(expected['pid'], expected['start_ticks'])
    (supervisor/'cmdline').write_bytes(str(source).encode()+b'\0'+str(root).encode()+b'\0')
    child = dict(pid=987654, start_ticks='12345', boot_id=expected['boot_id'])
    process(child['pid'], child['start_ticks'])
    job = root/'jobs/v5_969599'
    job.mkdir(parents=True)
    (job/'PROCESS.json').write_text(json.dumps(child))
    calls = []
    monkeypatch.setattr(signal, 'SIGSTOP', 19, raising=False)
    monkeypatch.setattr(os, 'pidfd_open', lambda pid: calls.append(('open', pid)) or 999, raising=False)
    monkeypatch.setattr(os, 'close', lambda fd: calls.append(('close', fd)))

    def send(fd, sig):
        calls.append(('signal', fd, sig))
        process(expected['pid'], expected['start_ticks'], 'T')

    monkeypatch.setattr(signal, 'pidfd_send_signal', send, raising=False)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='12345, 99'))
    code = remote_code('gpuA').replace(repr(REMOTE_ROOT), repr(str(root))).replace(repr(CONTROL), repr(str(control)))
    code = code.replace('/proc/', proc.as_posix()+'/')
    return SimpleNamespace(root=root, proc=proc, code=code, calls=calls, child=child, process=process)


def test_pause_only_verified_supervisor_preserves_child(sandbox):
    exec(compile(sandbox.code, '<drain-test>', 'exec'), {})
    signals = [call for call in sandbox.calls if call[0] == 'signal']
    assert signals == [('signal', 999, signal.SIGSTOP)]
    record = json.loads((sandbox.root/'OWNER_1735_DRAIN_APPLIED.json').read_text())
    assert record['supervisor_state'] == 'T'
    assert record['new_admission_ceiling'] == 0
    assert record['evaluate_children'][0]['state'] == 'S'
    assert not record['evaluate_children'][0]['signal_sent']
    assert record['redispatched_tasks'] == 0


@pytest.mark.parametrize('failure', ['pid_reused', 'boot_changed', 'prior_intent', 'already_paused'])
def test_identity_and_duplicate_guards_send_no_signal(sandbox, failure):
    expected = EXPECTED['gpuA']
    if failure == 'pid_reused':
        sandbox.process(expected['pid'], '99999')
    elif failure == 'boot_changed':
        (sandbox.proc/'sys/kernel/random/boot_id').write_text('other-boot')
    elif failure == 'prior_intent':
        (sandbox.root/'OWNER_1735_DRAIN_INTENT.json').write_text('{}')
    else:
        sandbox.process(expected['pid'], expected['start_ticks'], 'T')
    with pytest.raises(AssertionError):
        exec(compile(sandbox.code, '<drain-test>', 'exec'), {})
    assert not any(call[0] == 'signal' for call in sandbox.calls)
