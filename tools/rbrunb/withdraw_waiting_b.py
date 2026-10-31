"""One-time withdrawal of our unlaunched B supervisor after confirmed API failure."""
import json
import watch


def main():
    claim = json.loads((watch.SHARED / 'WM-PHYS-B.claim').read_text())
    assert claim['owner'] == watch.OWNER
    assert claim['manifest']['commit'] == 'b0f66943ec300594fc220d3779fb482c26534b1c'
    print(watch.ssh(r'''
import ast, ctypes, fcntl, json, os, signal, time
from pathlib import Path
r = Path('/root/rbeval20261031/runb/WM-PHYS-B')
admission = (r.parent.parent/'RB_GPU_DISPATCH.lock').open('a')
fcntl.flock(admission, fcntl.LOCK_EX)
assert not (r/'DISPATCH.json').exists(), 'Never withdraw running inference here'
assert not list(r.glob('LAUNCHED_*.json'))
assert not list(r.glob('shard*'))
assert not (r/'FAILED.json').exists()
launch = json.loads((r/'LAUNCH.json').read_text())
assert launch['pid'] == 728893
receipts = list(r.glob('SUPERVISOR_728893_*.json'))
assert len(receipts) == 1
owned = json.loads(receipts[0].read_text())
assert owned['owner'] == 'S-RB-RUNB' and owned['pid'] == 728893
assert owned['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p = Path('/proc/728893')
assert os.uname().machine == 'x86_64'
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.syscall(434, 728893, 0)  # pidfd_open, including Python <3.9
assert fd >= 0, ('pidfd_open failed', ctypes.get_errno())
assert (p/'stat').read_text().split()[21] == owned['start_ticks']
cmd = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode()
assert launch['argv'][6] in cmd and str(r) in cmd and 'supervise.py' in cmd
bridge = Path(owned['manifest']['runtime_root'])/'src/wms/control/bridge.py'
tree = ast.parse(bridge.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_cav_controller')
assert 'policy_native_lanes' not in [a.arg for a in fn.args.args + fn.args.kwonlyargs]
assert fn.args.kwarg is None
note = dict(owner='S-RB-RUNB', state='WITHDRAWN_BEFORE_INFERENCE',
            reason='Frozen five-arm MAPPO callback requires unsupported policy_native_lanes; A confirmed constructor failure; immutable B shares exact bridge',
            pid=728893, start_ticks=owned['start_ticks'], tasks=0, owned_worker_pids=[],
            original_launch=launch, bridge=str(bridge), unix=time.time())
with (r/'WITHDRAWAL_REQUEST.json').open('x') as f:
    json.dump(note, f, indent=2); f.flush(); os.fsync(f.fileno())
assert libc.syscall(424, fd, signal.SIGTERM, ctypes.c_void_p(), 0) == 0, ('pidfd_send_signal failed', ctypes.get_errno())
for _ in range(100):
    if not p.exists() or (p/'stat').read_text().split()[2] == 'Z':
        break
    time.sleep(.1)
else:
    raise RuntimeError('Owned waiter has not exited; no terminal receipt written')
os.close(fd)
assert not (r/'DISPATCH.json').exists()
note.update(error=note['reason'], owned_pids=[], exited=True, automatic_retry=False)
with (r/'FAILED.json').open('x') as f:
    json.dump(note, f, indent=2); f.flush(); os.fsync(f.fileno())
print(json.dumps(note))
'''))


if __name__ == '__main__':
    main()
