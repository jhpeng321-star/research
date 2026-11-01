"""Bounded exclusive world-job supervisor; never reads forecast efficacy values."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CONTROL = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CONTROL), str(CONTROL/'src')]

from tools.rb2int.admission import memory_snapshot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=1, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def incarnation(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().split()
    return dict(pid=pid, start_ticks=fields[21], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def alive(process):
    try:
        return incarnation(process['pid']) == {k:process[k] for k in ('pid','start_ticks','boot_id')} and Path(f"/proc/{process['pid']}/stat").read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def completed_world(out, seed):
    path = out/'COMPLETE.json'
    if not path.is_file():
        return False
    record = load(path)
    expected = {f'w{seed}_k{k}.json' for k in range(8)}
    if record['tasks'] != 8 or set(record['hashes']) != expected:
        raise ValueError('world completion must contain exactly its complete K8')
    if record['identity_sha256'] != sha(out/'IDENTITY.json'):
        raise ValueError('world identity differs at completion')
    if any(sha(out/name) != value for name,value in record['hashes'].items()):
        raise ValueError('world completion hash differs')
    return True


def admission(root, memory_gib):
    resources = dict(load=os.getloadavg(), memory=memory_snapshot(),
                     disk_free_bytes=__import__('shutil').disk_usage(root).free)
    if resources['load'][0] >= os.cpu_count()/2:
        return False, dict(resources, wait='physical CPU load')
    cg = Path('/sys/fs/cgroup')
    maximum, stat = cg/'cpu.max', cg/'cpu.stat'
    if maximum.is_file() and maximum.read_text().split()[0] != 'max':
        quota, period = map(int, maximum.read_text().split())
        capacity = quota/period
        def usage():
            return int(dict(line.split() for line in stat.read_text().splitlines())['usage_usec'])
        begin, before = time.monotonic(), usage()
        time.sleep(1)
        used = (usage()-before)/1e6/(time.monotonic()-begin)
        resources.update(cpu_capacity=capacity, cpu_used=used)
        if used+1 >= .8*capacity:
            return False, dict(resources, wait='effective CPU quota')
    if resources['memory']['effective_available_bytes'] <= 8*2**30:
        return False, dict(resources, wait='effective memory')
    if resources['disk_free_bytes'] <= 512*2**20:
        return False, dict(resources, wait='own output disk')
    query = subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True)
    if query.returncode:
        return False, dict(resources, wait='GPU query failed', stderr=query.stderr)
    free, util = map(int, query.stdout.strip().split(','))
    resources.update(gpu_free_mib=free, gpu_utilization=util)
    if free <= (memory_gib+8)*1024:
        return False, dict(resources, wait='GPU cap plus shared headroom')
    return True, resources


def child_argv(control, root, job, release, panel):
    catalog = release['catalog'][job['model']]
    binding = catalog['binding']
    worlds = load(panel)['worlds']
    # Each child owns one complete world, enabling dynamic load balance without
    # importing two different frozen model codebases into one Python process.
    index = worlds.index(job['seed'])
    return ['nice','-n','19','/root/miniconda3/bin/python','-B','-u',
        str(control/'tools/rb2int/evaluate.py'),'--variant',binding['variant'],
        '--runtime-root',binding['runtime_root'],'--wm-config',binding['config'],
        '--wm','/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt',
        '--model-name',catalog['identity_name'],'--registered-panel',str(panel),
        '--out',str(root/'tasks'),'--device','cuda','--memory-gib','5',
        '--shard',str(index),'--shards',str(len(worlds))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--host', choices=('gpuA','gpuR'), required=True)
    args = parser.parse_args()
    root = Path(args.root)
    spec = load(root/'SPEC.json')
    if spec['host'] != args.host or Path(spec['control']) != CONTROL:
        raise ValueError('supervisor host/control source differs')
    for name, expected in spec['files'].items():
        if sha(root/name) != expected:
            raise ValueError('immutable final execution metadata changed')
    plan, release = load(root/'RUN_PLAN.json'), load(root/'RELEASE.json')
    if release['status'] != 'READY' or not release['exposure_audit_passed']:
        raise ValueError('final release required')
    import fcntl
    lock = (root/'SUPERVISOR.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(root/'PANEL_START_ONCE.json',dict(**incarnation(os.getpid()),
        plan_sha256=sha(root/'RUN_PLAN.json'),release_sha256=sha(root/'RELEASE.json'),
        source_sha256=sha(__file__),automatic_retry=False,started_at=datetime.now().astimezone().isoformat()))
    pending = [j for j in plan['jobs'] if j['host'] == args.host]
    slots, cutoff = plan['slots'][args.host], datetime.fromisoformat(plan['dispatch_cutoff']).timestamp()
    running, completed, failed = {}, [], []
    cutoff_reached = False
    (root/'jobs').mkdir()
    with (root/'status.jsonl').open('x') as journal:
        while pending or running:
            for key, entry in list(running.items()):
                child, job, job_root, process = entry
                if child.poll() is None:
                    continue
                completion_error = None
                try:
                    good = completed_world(job_root/'tasks', job['seed']) if child.returncode == 0 else False
                except Exception as error:
                    good, completion_error = False, repr(error)
                exit_record = dict(job=job,process=process,returncode=child.returncode,
                    complete_K8=good,completion_error=completion_error,
                    finished_at=datetime.now().astimezone().isoformat())
                save(job_root/'EXIT.json',exit_record)
                (completed if good else failed).append(job['id'])
                del running[key]
                print('WORLD_COMPLETE' if good else 'WORLD_FAILED',job['id'],flush=True)
            if time.time() >= cutoff:
                cutoff_reached = True
            last_resources = None
            while pending and len(running) < slots and not failed and not cutoff_reached:
                admitted, last_resources = admission(root,plan['memory_gib_per_worker'])
                if not admitted:
                    break  # resource waiting is not a scientific retry.
                job = pending[0]
                job_root = root/'jobs'/job['id']
                if job_root.exists():
                    raise ValueError('preserve an existing world claim; never duplicate a task')
                job_root.mkdir()
                save(job_root/'CLAIM_ONCE.json',dict(job=job,resources=last_resources,
                    release_sha256=sha(root/'RELEASE.json'),automatic_retry=False))
                panel = root/'panels'/(job['model']+'.json')
                argv = child_argv(CONTROL,job_root,job,release,panel)
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                    PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
                with (job_root/'worker.log').open('x') as log:
                    child = subprocess.Popen(argv,env=env,cwd=CONTROL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                process = dict(**incarnation(child.pid),argv=argv,job=job)
                save(job_root/'PROCESS.json',process)
                running[job['id']] = (child,job,job_root,process)
                pending.pop(0)
                print('WORLD_START',job['id'],child.pid,flush=True)
            status = dict(time=datetime.now().astimezone().isoformat(),
                completed_world_jobs=completed,failed_world_jobs=failed,
                running={key:entry[3] for key,entry in running.items()},
                pending_world_jobs=[j['id'] for j in pending],cutoff_reached=cutoff_reached,
                resource_wait=last_resources, efficacy_values_read=False)
            journal.write(json.dumps(status)+'\n');journal.flush()
            if not running and (failed or cutoff_reached):
                break
            if pending or running:
                time.sleep(30)
    final = dict(host=args.host,completed_world_jobs=completed,failed_world_jobs=failed,
        unstarted_world_jobs=[j['id'] for j in pending],cutoff_reached=cutoff_reached,
        complete=not pending and not failed,forecast_efficacy_values_read=False,
        final_time=datetime.now().astimezone().isoformat())
    save(root/('PANEL_COMPLETE.json' if final['complete'] else 'PANEL_TERMINAL_INCOMPLETE.json'),final)
    print(json.dumps(final),flush=True)


if __name__ == '__main__':
    main()
