"""Supervise the memory-only legacy reference continuation; no new broker/grader."""
import argparse
import ctypes
import gc
import os
from pathlib import Path
import signal
import time

from . import replay_start as rs
from .storage import read, write


def release_heap():
    """Called only after all previous-episode aliases have been dropped."""
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)


def verify_identity(receipt, component, arm=None):
    assert rs.alive(receipt), 'owner exited or identity changed'
    p = Path('/proc', str(receipt['pid']))
    argv = (p / 'cmdline').read_bytes().split(b'\0')
    args = [x.decode() for x in argv if x]
    assert args[1:] == ['-B', '-u', '-m', 'tools.coopmwm.replay_start',
                        receipt['run_root'], component] + (['--arm', arm] if arm else [])
    return p


def watch(root):
    import fcntl
    root = Path(root)
    v = rs.config(root)
    registration = read(root / 'memory_REGISTERED.json')
    assert registration['binding_sha256'] == rs.sha(root / rs.NAME)
    assert registration['binding_sha256'] in (root / 'memory_AUTHORITY.md').read_text()
    rs.accounting(v)
    control = Path(v['control_root'])
    lock = (control / 'memory_WATCH.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    write(control / 'memory_WATCH_OWNER.json', rs.owner())
    children = []
    for a in ('B', 'R'):
        d = v['adopted_graders'][a]
        verify_identity(d, 'grade', a)
        children.append(d)
        write(control / ('grade_' + a + '_DISPATCH.json'), d)
    for a in ('B', 'R'):
        resume = v['resume_updates'][a]
        assert rs.sha(root / f'{a}_checkpoint_{resume:03}.pt') == v['checkpoints'][a]
        assert not (root / f'{a}_checkpoint_{resume+1:03}.pt').exists()
        from .warm_health import assert_running
        assert_running(root, a)
        with (root / f'WARM_{a}.claim').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children.append(rs.spawn(root, v, 'arm', a))
    write(root / v['start_receipt_name'], dict(unix=time.time(), children=children,
        resume_updates=v['resume_updates'], checkpoints=v['checkpoints'], final_update=24,
        precharged_unattempted=v['precharged_unattempted'],
        amendment_sha256=v['amendment_sha256'], legacy_recipe=True))
    while time.time() < v['gpu_deadline']:
        rs.bridge_once(root, v)
        updates = {a: max(int(p.stem.rsplit('_', 1)[1]) for p in
            root.glob(a + '_UPDATE_[0-9][0-9][0-9].json')) for a in ('B', 'R')}
        dead = [d for d in children if not rs.alive(d) and not rs.completed(control, d)]
        rs.atomic(control / 'memory_STATUS.json', dict(unix=time.time(), updates=updates,
            children=children, dead_children=dead, alert=bool(dead), single_existing_broker=True))
        if dead:
            raise RuntimeError('reference child exited; preserve claims and diagnose')
        if all(u >= 24 for u in updates.values()):
            write(control / 'memory_COMPLETE.json', dict(unix=time.time(), updates=updates))
            return
        time.sleep(5)
    raise ValueError('registered deadline')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    v = rs.config(a.root)
    from .nullfix_geometry import install
    s = Path(v['source_root'])
    install(str(s / 'FROZEN_GEOMETRY_SOURCE.py'), str(s / 'FROZEN_NETWORK.net.xml.gz'))
    try:
        watch(a.root)
    except BaseException as error:
        import traceback
        write(Path(v['control_root']) / 'memory_WATCH_FAILED.json', dict(
            unix=time.time(), error=repr(error), traceback=traceback.format_exc(), physics_retry=False))
        raise


if __name__ == '__main__':
    main()
