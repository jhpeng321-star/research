"""Owner17:35 operational continuation: drain, preserve claims, then admit at2/1.

No evaluate process is signalled. The original CPU dispatcher is retired only
after every recorded child has exited and every prior claim is complete K8.
"""
import argparse
import ctypes
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

CAPS = {'gpuA': 2, 'gpuR': 1}
CONTROL = Path('/root/autodl-tmp/c4/rb2int20261101/phys_code_d0df3d66e80a')


def priority(job):
    """Outcome-blind stable amendment: remaining reserved v5 first, then old order."""
    return 0 if job['model'] == 'FULL-v5' and 969432 <= job['seed'] <= 969439 else 1


def recovery_plan(root, jobs, complete, alive):
    """Validate the entire old claim set before writing or dispatching anything."""
    by_id = {job['id']: job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError('duplicate registered job')
    claimed, active, recovered = set(), [], []
    for directory in sorted((root/'jobs').iterdir()):
        if not directory.is_dir():
            continue
        if directory.name not in by_id:
            raise ValueError('unregistered claim; preserve it')
        job = by_id[directory.name]
        claimed.add(job['id'])
        claim = json.loads((directory/'CLAIM_ONCE.json').read_text())
        process = json.loads((directory/'PROCESS.json').read_text())
        if claim['job'] != job or process['job'] != job:
            raise ValueError('claimed job differs from registered plan')
        if alive(process):
            active.append(job['id'])
            continue
        if not complete(directory/'tasks', job['seed']):
            raise ValueError('exited incomplete claim: never retry or redispatch')
        exit_path = directory/'EXIT.json'
        if exit_path.exists():
            if not json.loads(exit_path.read_text())['complete_K8']:
                raise ValueError('prior failed claim: never retry')
        else:
            recovered.append((directory, job, process))
    pending = sorted((job for job in jobs if job['id'] not in claimed), key=priority)
    return active, recovered, pending, sorted(claimed)


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    if platform.machine() != 'x86_64':
        raise ValueError('pidfd fallback requires x86_64')
    result = ctypes.CDLL(None, use_errno=True).syscall(424, fd, sig, 0, 0)
    if result < 0:
        raise OSError(ctypes.get_errno(), 'own dispatcher pidfd signal')


def retire_drained(root, original, fw):
    """Bind the original stopped dispatcher, never a child or process group."""
    if fw.incarnation(original['pid']) != {k: original[k] for k in ('pid', 'start_ticks', 'boot_id')}:
        raise ValueError('original dispatcher identity changed')
    if Path(f"/proc/{original['pid']}/stat").read_text().split()[2] != 'T':
        raise ValueError('original dispatcher must remain paused until drain')
    if hasattr(os, 'pidfd_open'):
        fd = os.pidfd_open(original['pid'])
    else:
        if platform.machine() != 'x86_64':
            raise ValueError('pidfd fallback requires x86_64')
        fd = ctypes.CDLL(None, use_errno=True).syscall(434, original['pid'], 0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), 'own dispatcher pidfd_open')
    try:
        if fw.incarnation(original['pid']) != {k: original[k] for k in ('pid', 'start_ticks', 'boot_id')}:
            raise ValueError('original dispatcher changed after pidfd binding')
        fw.save(root/'OWNER_1735_DISPATCHER_RETIRE_INTENT.json', dict(
            process=original, evaluate_children_live=0, complete_prior_claims=True,
            signals=['SIGTERM', 'SIGCONT'], target='drained own CPU dispatcher only'))
        pidfd_signal(fd, signal.SIGTERM)
        pidfd_signal(fd, signal.SIGCONT)
    finally:
        os.close(fd)
    deadline = time.monotonic()+30
    while fw.alive(original):
        if time.monotonic() >= deadline:
            raise TimeoutError('original dispatcher has not exited; no successor dispatch')
        time.sleep(.1)


def run(root, host, fw):
    import fcntl
    spec = fw.load(root/'SPEC.json')
    if spec['host'] != host or Path(spec['control']) != CONTROL:
        raise ValueError('frozen control/host differs')
    for name, expected in spec['files'].items():
        if fw.sha(root/name) != expected:
            raise ValueError('frozen execution metadata changed')
    launch = fw.load(root/'LAUNCH_ONCE.json')
    if fw.sha(CONTROL/'tools/rb2int/final_worker.py') != launch['source_sha256']:
        raise ValueError('original supervisor source changed')
    applied = fw.load(root/'OWNER_1735_DRAIN_APPLIED.json')
    original = fw.load(root/'SUPERVISOR_PROCESS.json')
    if applied['process'] != {k: original[k] for k in ('pid', 'start_ticks', 'boot_id')}:
        raise ValueError('drain identity differs')
    # A separate coordinator lock prevents two waiting continuations. The
    # dispatch lock stays with the original until its verified drained exit.
    own = (root/'OWNER_1735_COORDINATOR.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fw.save(root/'OWNER_1735_CAPPED_START_ONCE.json', dict(
        **fw.incarnation(os.getpid()), source_sha256=fw.sha(__file__),
        cap=CAPS[host], no_evaluate_signals=True, original=original,
        started_at=datetime.now().astimezone().isoformat()))
    plan, release = fw.load(root/'RUN_PLAN.json'), fw.load(root/'RELEASE.json')
    jobs = [job for job in plan['jobs'] if job['host'] == host]
    while True:
        active, recovered, pending, claimed = recovery_plan(root, jobs, fw.completed_world, fw.alive)
        print('DRAIN', host, 'active', active, 'prior_claims', len(claimed), flush=True)
        if not active:
            break
        if not fw.alive(original):
            raise ValueError('original dispatcher exited unexpectedly while draining')
        time.sleep(15)
    retire_drained(root, original, fw)
    lock = (root/'SUPERVISOR.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Revalidate after acquiring dispatch ownership; prior claims never enter
    # pending and never reach Popen, including any failed/incomplete claim.
    active, recovered, pending, claimed = recovery_plan(root, jobs, fw.completed_world, fw.alive)
    if active:
        raise ValueError('unexpected active child after drain')
    for directory, job, process in recovered:
        fw.save(directory/'EXIT.json', dict(job=job, process=process, returncode=None,
            complete_K8=True, completion_error=None, recovered_from_verified_complete=True,
            finished_at=datetime.now().astimezone().isoformat()))
    fw.save(root/'OWNER_1735_CAPPED_HANDOVER.json', dict(
        process=fw.incarnation(os.getpid()), original=original, original_exited=True,
        dispatch_lock_acquired=True, claimed_jobs_preserved=claimed,
        pending_jobs=[job['id'] for job in pending], cap=CAPS[host],
        priority='remaining reserved FULL-v5 first; otherwise original stable order',
        plan_sha256=fw.sha(root/'RUN_PLAN.json'), source_sha256=fw.sha(__file__),
        handover_at=datetime.now().astimezone().isoformat(), redispatched_tasks=0))
    running, completed, failed = {}, list(claimed), []
    cutoff = datetime.fromisoformat(plan['dispatch_cutoff']).timestamp()
    with (root/'OWNER_1735_capped_status.jsonl').open('x') as journal:
        while pending or running:
            for key, entry in list(running.items()):
                child, job, directory, process = entry
                if child.poll() is None:
                    continue
                good = child.returncode == 0 and fw.completed_world(directory/'tasks', job['seed'])
                fw.save(directory/'EXIT.json', dict(job=job, process=process,
                    returncode=child.returncode, complete_K8=good, completion_error=None,
                    finished_at=datetime.now().astimezone().isoformat()))
                (completed if good else failed).append(job['id'])
                del running[key]
                print('WORLD_COMPLETE' if good else 'WORLD_FAILED', job['id'], flush=True)
            resources = None
            while pending and len(running) < CAPS[host] and not failed and time.time() < cutoff:
                admitted, resources = fw.admission(root, plan['memory_gib_per_worker'])
                if not admitted:
                    break
                job = pending[0]
                directory = root/'jobs'/job['id']
                if directory.exists():
                    raise ValueError('existing claim: never duplicate or retry')
                directory.mkdir()
                fw.save(directory/'CLAIM_ONCE.json', dict(job=job, resources=resources,
                    release_sha256=fw.sha(root/'RELEASE.json'), automatic_retry=False,
                    operational_amendment='owner17:35 capped continuation'))
                panel = root/'panels'/(job['model']+'.json')
                argv = fw.child_argv(CONTROL, directory, job, release, panel)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
                with (directory/'worker.log').open('x') as log:
                    child = subprocess.Popen(argv, env=env, cwd=CONTROL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                process = dict(**fw.incarnation(child.pid), argv=argv, job=job)
                fw.save(directory/'PROCESS.json', process)
                running[job['id']] = (child, job, directory, process)
                pending.pop(0)
                print('WORLD_START', job['id'], child.pid, flush=True)
            status = dict(time=datetime.now().astimezone().isoformat(), cap=CAPS[host],
                completed_world_jobs=completed, failed_world_jobs=failed,
                running={key: entry[3] for key, entry in running.items()},
                pending_world_jobs=[job['id'] for job in pending],
                cutoff_reached=time.time() >= cutoff, resource_wait=resources,
                efficacy_values_read=False)
            journal.write(json.dumps(status)+'\n')
            journal.flush()
            if not running and (failed or time.time() >= cutoff):
                break
            if pending or running:
                time.sleep(30)
    final = dict(host=host, completed_world_jobs=completed, failed_world_jobs=failed,
        unstarted_world_jobs=[job['id'] for job in pending], cutoff_reached=time.time() >= cutoff,
        complete=not pending and not failed, forecast_efficacy_values_read=False,
        final_time=datetime.now().astimezone().isoformat(), cap=CAPS[host])
    fw.save(root/('PANEL_COMPLETE.json' if final['complete'] else 'PANEL_TERMINAL_INCOMPLETE.json'), final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--host', choices=tuple(CAPS), required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('frozen_rb2int_final_worker', CONTROL/'tools/rb2int/final_worker.py')
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    try:
        run(Path(args.root), args.host, fw)
    except Exception as error:
        fw.save(Path(args.root)/'OWNER_1735_CAPPED_STOP.json', dict(
            error=repr(error), automatic_retry=False, evaluate_process_signals=False))
        raise


if __name__ == '__main__':
    main()
