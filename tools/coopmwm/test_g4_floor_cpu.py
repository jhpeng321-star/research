"""clean5b-F8S contracts (owner 2026-11-08 "fix + fresh relaunch", "never stall again"; F8R G4 floor deadlock):
one-job G4 workers (maxtasksperchild=1) at most g4_workers at a time, the G4-only memory floor, admission wait
alert files (10 min, removed when the wait clears) and the 30-min admission-deadlock FAULT."""
import json
import os

import pytest

from . import g4_runtime as g4, runner


@pytest.fixture
def no_prior(monkeypatch):
    import tools.coopnative.runner as prior
    monkeypatch.setattr(prior, 'admission', lambda b: None)
    monkeypatch.setattr(runner.time, 'sleep', lambda s: None)


def _clock(monkeypatch, step):
    t = iter(range(0, 10**7, step))
    monkeypatch.setattr(runner.time, 'time', lambda: next(t))


def _job_pid(job, cfg):
    import time
    time.sleep(0.05)
    if job.get('fail'):
        raise ValueError('boom '+str(job['i']))
    return dict(i=job['i'], pid=os.getpid())


def test_g4_settings_validation():
    assert g4.g4_settings(dict(workers=8)) == (None, None)                 # absent = inherited pool and floor
    assert g4.g4_settings(dict(g4_workers=4, g4_mem_available_min_gb=20)) == (4, 20)
    for bad in (dict(g4_workers=0), dict(g4_workers=4.0), dict(g4_workers=33),
                dict(g4_mem_available_min_gb=7), dict(g4_mem_available_min_gb=20.0), dict(g4_mem_available_min_gb=161)):
        with pytest.raises(ValueError):
            g4.g4_settings(bad)


def test_fresh_dispatch_one_job_per_worker_in_order(monkeypatch):
    """Every job runs in its own process (memory returned after each job), results keep the plan order."""
    monkeypatch.setattr(g4, 'run_branch', _job_pid)
    jobs = [dict(i=i) for i in range(7)]
    out = g4.dispatch(dict(workers=8, g4_workers=3), jobs)
    assert [r['i'] for r in out] == list(range(7))
    assert len({r['pid'] for r in out}) == 7 and os.getpid() not in {r['pid'] for r in out}


def test_fresh_dispatch_concurrency_cap(monkeypatch):
    live, peak = [0], [0]
    real = g4.worker_pool

    class Counting:
        def __init__(self, cfg, workers=None):
            assert workers == 1
            self.pool = real(cfg, workers=1)
            live[0] += 1
            peak[0] = max(peak[0], live[0])

        def submit(self, *a):
            return self.pool.submit(*a)

        def shutdown(self, **k):
            self.pool.shutdown(**k)
            live[0] -= 1

    monkeypatch.setattr(g4, 'worker_pool', Counting)
    monkeypatch.setattr(g4, 'run_branch', _job_pid)
    g4.dispatch(dict(workers=8, g4_workers=2), [dict(i=i) for i in range(6)])
    assert peak[0] == 2 and live[0] == 0


def test_fresh_dispatch_failure_stops_new_starts_and_raises(monkeypatch):
    started = []
    real = g4.worker_pool

    def counting(cfg, workers=None):
        started.append(1)
        return real(cfg, workers=workers)

    monkeypatch.setattr(g4, 'worker_pool', counting)
    monkeypatch.setattr(g4, 'run_branch', _job_pid)
    jobs = [dict(i=0, fail=True)]+[dict(i=i) for i in range(1, 10)]
    with pytest.raises(ValueError, match='boom 0'):
        g4.dispatch(dict(workers=8, g4_workers=2), jobs)
    assert len(started) < len(jobs)                        # not-yet-started jobs are cancelled (reservation stays spent)


def test_inherited_pool_unchanged_without_g4_workers(monkeypatch):
    called = []
    monkeypatch.setattr(g4, '_dispatch_fresh', lambda *a: called.append(a))
    monkeypatch.setattr(g4, 'run_branch', _job_pid)
    out = g4.dispatch(dict(workers=2), [dict(i=i) for i in range(3)])
    assert not called and [r['i'] for r in out] == [0, 1, 2]


def test_run_branch_admits_at_the_g4_floor(monkeypatch):
    seen = []
    monkeypatch.setattr(g4.tb, '_approved', lambda cfg: None)

    class Admitted(Exception):
        pass

    def admit(b):
        seen.append(b['mem_available_min_gb'])
        raise Admitted

    monkeypatch.setattr(runner, 'admission', admit)
    for cfg, floor in ((dict(mem_available_min_gb=40, g4_mem_available_min_gb=20), 20), (dict(mem_available_min_gb=40), 40)):
        with pytest.raises(Admitted):
            g4.run_branch(dict(), cfg)
        assert seen[-1] == floor


def test_memory_wait_alert_written_after_10_min_and_removed_when_cleared(tmp_path, monkeypatch, no_prior):
    _clock(monkeypatch, 60)
    avail = iter([10*1024**3]*15+[41*1024**3])
    seen = []

    def mem():
        seen.append(sorted(p.name for p in tmp_path.glob('MEM_FLOOR_WAIT_*.json')))
        return next(avail)

    monkeypatch.setattr(runner, '_mem_available_bytes', mem)
    runner.admission(dict(load_max=40, mem_available_min_gb=40, run_root=str(tmp_path)))
    assert seen[5] == [] and seen[-1] == [f'MEM_FLOOR_WAIT_{os.getpid()}.json']    # appears only after > 10 min
    assert not list(tmp_path.glob('MEM_FLOOR_WAIT_*.json'))                           # removed when the wait clears
    assert not (tmp_path/'FAULT.json').exists()


def test_memory_wait_alert_content(tmp_path, monkeypatch, no_prior):
    _clock(monkeypatch, 700)
    avail = iter([30*1024**3, 30*1024**3, 50*1024**3])
    snap = []
    monkeypatch.setattr(runner, '_mem_available_bytes', lambda: next(avail))
    monkeypatch.setattr(runner, '_mem_total_bytes', lambda: 184*1024**3)      # hermetic: no /proc reads
    monkeypatch.setattr(runner, '_resident_bytes', lambda root: 100*1024**3)  # floor reachable -> no FAULT
    monkeypatch.setattr(runner.time, 'sleep', lambda s: snap.extend(
        json.loads(p.read_text()) for p in tmp_path.glob('MEM_FLOOR_WAIT_*.json')))
    runner.admission(dict(load_max=40, mem_available_min_gb=40, run_root=str(tmp_path)))
    assert snap and snap[0]['floor_gb'] == 40 and snap[0]['mem_available_gb'] == 30.0 and snap[0]['pid'] == os.getpid()


def test_disk_wait_alert_after_10_min_and_removed(tmp_path, monkeypatch, no_prior):
    _clock(monkeypatch, 60)
    free = iter([1*1024**3]*12+[70*1024**3])
    seen = []

    def disk(path):
        seen.append(sorted(p.name for p in tmp_path.glob('DISK_FLOOR_WAIT_*.json')))
        return next(free)

    monkeypatch.setattr(runner, '_disk_free_bytes', disk)
    monkeypatch.setenv('COOPMWM_DISK_MIN_GB', '60')
    runner.admission(dict(load_max=40, run_root=str(tmp_path)))
    assert seen[-1] == [f'DISK_FLOOR_WAIT_{os.getpid()}.json']
    assert not list(tmp_path.glob('DISK_FLOOR_WAIT_*.json'))


def _deadlock_env(tmp_path, monkeypatch, resident_gb):
    _clock(monkeypatch, 60)
    monkeypatch.setattr(runner, '_mem_available_bytes', lambda: 36*1024**3)
    monkeypatch.setattr(runner, '_mem_total_bytes', lambda: 184*1024**3)
    monkeypatch.setattr(runner, '_resident_bytes', lambda root: int(resident_gb*1024**3))


def test_admission_deadlock_faults_after_30_min(tmp_path, monkeypatch, no_prior):
    """The F8R state: B 86 + R 46 + 8 waiting workers x 2.6 -> 184 - 153 = 31 GB < 40: FAULT, not an endless sleep."""
    _deadlock_env(tmp_path, monkeypatch, 86+46+1+8*2.6)
    with pytest.raises(RuntimeError, match='admission deadlock'):
        runner.admission(dict(load_max=40, mem_available_min_gb=40, run_root=str(tmp_path)))
    fault = json.loads((tmp_path/'FAULT.json').read_text())
    assert fault['error'] == 'admission deadlock' and fault['floor_gb'] == 40 and fault['waited_s'] > 1800
    assert fault['mem_total_gb'] == 184 and not list(tmp_path.glob('MEM_FLOOR_WAIT_*.json'))


def test_reachable_floor_keeps_waiting_without_fault(tmp_path, monkeypatch, no_prior):
    """184 - (86 + 46 + 1) = 51 >= 40: memory held by RUNNING jobs will return, so it is not a deadlock."""
    _deadlock_env(tmp_path, monkeypatch, 86+46+1)
    n = [0]

    def mem():
        n[0] += 1
        return 36*1024**3 if n[0] < 100 else 45*1024**3

    monkeypatch.setattr(runner, '_mem_available_bytes', mem)
    runner.admission(dict(load_max=40, mem_available_min_gb=40, run_root=str(tmp_path)))
    assert not (tmp_path/'FAULT.json').exists() and n[0] == 100


def test_existing_fault_is_not_overwritten(tmp_path, monkeypatch, no_prior):
    (tmp_path/'FAULT.json').write_text('{"error": "earlier"}')
    _deadlock_env(tmp_path, monkeypatch, 150)
    with pytest.raises(RuntimeError, match='admission deadlock'):
        runner.admission(dict(load_max=40, mem_available_min_gb=40, run_root=str(tmp_path)))
    assert json.loads((tmp_path/'FAULT.json').read_text()) == {'error': 'earlier'}


def test_resident_counts_root_processes_and_waiters(tmp_path, monkeypatch):
    (tmp_path/'MEM_FLOOR_WAIT_11.json').write_text('{}')
    (tmp_path/'MEM_FLOOR_WAIT_bad.json').write_text('{}')
    monkeypatch.setattr(runner, '_root_processes', lambda root: [7, 11])
    monkeypatch.setattr(runner, '_rss_bytes', lambda pid: {7: 5, 11: 3}[pid])
    assert runner._resident_bytes(tmp_path) == 8                          # waiter 11 counted once
