"""Run the real hold payload against fake /proc and intercepted exact pidfds."""
import hashlib
import json
import os
import signal

import pytest

from tests.test_rb2int_drain_ops import sandbox  # noqa: F401
from tools.rb2int.drain_ops import CONTROL, EXPECTED
from tools.rb2int.final_ops import REMOTE_ROOT
from tools.rb2int.hold_ops import PROCESSES, hold_code


@pytest.fixture
def hold_sandbox(sandbox, monkeypatch):  # noqa: F811 - imported pytest fixture
    current, start, late, late_start = PROCESSES['gpuA']
    identities = []
    for stage, pid, ticks in (('owner1735_control',current,start),
                              ('owner_late_control',late,late_start)):
        folder = sandbox.root/stage
        folder.mkdir()
        source = folder/'capped_worker.py'
        source.write_text('# frozen coordinator\n')
        record = dict(pid=pid,start_ticks=ticks,boot_id=EXPECTED['gpuA']['boot_id'],
                      host='gpuA',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        (folder/'PROCESS.json').write_text(json.dumps(record))
        proc = sandbox.process(pid,ticks)
        (proc/'cmdline').write_bytes(str(source).encode()+b'\0'+str(sandbox.root).encode()+b'\0')
        identities.append((pid,ticks))
    sandbox.identities = identities
    monkeypatch.setattr(os,'pidfd_open',lambda pid:pid)

    def send(fd, sig):
        sandbox.calls.append(('signal',fd,sig))
        sandbox.process(fd,dict(identities)[fd],'T')

    monkeypatch.setattr(signal,'pidfd_send_signal',send)
    sandbox.code = hold_code('gpuA').replace(repr(REMOTE_ROOT),repr(str(sandbox.root)))
    sandbox.code = sandbox.code.replace(repr(CONTROL),repr(str(sandbox.root/'control')))
    sandbox.code = sandbox.code.replace('/proc/',sandbox.proc.as_posix()+'/')
    return sandbox


def test_hold_pauses_both_admission_owners_without_touching_evaluate(hold_sandbox):
    box = hold_sandbox
    exec(compile(box.code,'<hold-test>','exec'),{})
    signals = [call for call in box.calls if call[0]=='signal']
    assert signals == [('signal',PROCESSES['gpuA'][2],signal.SIGSTOP),
                       ('signal',PROCESSES['gpuA'][0],signal.SIGSTOP)]
    result = json.loads((box.root/'OWNER_1755_HOLD_APPLIED.json').read_text())
    assert result['new_gpu_admissions']==0 and not result['automatic_resume']
    assert result['evaluate_signals']==0 and result['peer_signals']==0
    assert result['evaluate_children'][0]['process']==box.child
    assert result['evaluate_children'][0]['state']=='S'
    assert all(p['state']=='T' for p in result['paused'])


@pytest.mark.parametrize('failure',['reused_current','wrong_boot','bad_source','duplicate_hold'])
def test_invalid_owner_or_duplicate_hold_sends_no_signal(hold_sandbox,failure):
    box = hold_sandbox
    if failure=='reused_current':
        box.process(PROCESSES['gpuA'][0],'wrong-start')
    elif failure=='wrong_boot':
        (box.proc/'sys/kernel/random/boot_id').write_text('wrong-boot')
    elif failure=='bad_source':
        (box.root/'owner1735_control/capped_worker.py').write_text('# changed')
    else:
        (box.root/'OWNER_1755_HOLD_INTENT.json').write_text('{}')
    with pytest.raises(AssertionError):
        exec(compile(box.code,'<hold-test>','exec'),{})
    assert not any(call[0]=='signal' for call in box.calls)
