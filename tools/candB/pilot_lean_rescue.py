"""Schedule a tested CPython pending call on the exact owned startup trainer.

The pending call runs with the interpreter GIL at a Python safe point; gdb never
executes Python directly on an arbitrary paused CUDA/allocator stack.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def schedule(pid, code):
    commands = ['gdb', '-nx', '-batch', '-p', str(pid),
        '-ex', 'set pagination off',
        '-ex', 'set $pilot_code = (char*)strdup('+json.dumps(code)+')',
        '-ex', 'call (int)Py_AddPendingCall((int (*)(void*))PyRun_SimpleString, (void*)$pilot_code)',
        '-ex', 'detach']
    began = time.monotonic()
    result = subprocess.run(commands, capture_output=True, text=True, timeout=20)
    doc = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        attach_seconds=time.monotonic()-began, pid=pid, code=code)
    print(json.dumps(doc), flush=True)
    if result.returncode or '= 0' not in result.stdout:
        raise RuntimeError('pending-call scheduling failed; inspect exact receipt before retry')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('test', 'install'))
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    assert root.is_dir() and 'EXPLORATORY-LEAN' in root.name
    if args.action == 'test':
        marker = root / 'pilot_pending_CPU_ACK.json'
        assert not marker.exists(), 'test already finished; recover receipt'
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        try:
            time.sleep(1)
            receipt = schedule(child.pid, f"__import__('pathlib').Path({str(marker)!r}).write_text('pending call CPU PASS')")
            for _ in range(50):
                if marker.exists():
                    break
                time.sleep(.1)
            assert marker.read_text() == 'pending call CPU PASS'
        finally:
            child.terminate()
            child.wait()
    else:
        marker = root / 'pilot_EXPLORATORY-LEAN' / 'pilot_gate_overlay_INSTALLED.json'
        assert not marker.exists(), 'already installed; never duplicate'
        owner = json.loads((root / 'TRAINER_OWNERSHIP.json').read_text())
        proc = Path('/proc') / str(owner['pid'])
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        assert stat[19] == str(owner['start_ticks'])
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert '-m tools.candB.pilot_lean ' in command and str(root) in command
        assert (root / 'pilot_pending_CPU_ACK.json').exists(), 'test pending mechanism first'
        receipt = schedule(owner['pid'], 'from tools.candB.pilot_lean_gate import install; install()')
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(.1)
        assert marker.exists(), 'scheduled, ACK pending; never inject twice blindly'
        receipt['ACK'] = json.loads(marker.read_text())
    receipt.update(action=args.action, overlay_sha256=hashlib.sha256(
        Path(__file__).with_name('pilot_lean_gate.py').read_bytes()).hexdigest())
    with (root / f'pilot_pending_{args.action}_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=1)


if __name__ == '__main__':
    main()
