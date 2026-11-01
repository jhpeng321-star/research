"""Read-only checkpoint/exposure audit; never dispatches or signals a process."""
import argparse
import json
from pathlib import Path

from tools.rbint.remote import ssh
from tools.rbeval.core import digest, load, save_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert args.output.parent.is_dir() and not args.output.exists()
    shards = []
    for shard, pid in ((0, 501141), (1, 500921)):
        folder = root/f'docs/rb2break/devk8_pause_s{shard}'
        snapshot = load(folder/'SNAPSHOT.json')
        for name, expected in snapshot['hashes'].items():
            assert Path(name).name == name
            assert digest(folder/name) == expected, name
        dispatch = load(folder/'DISPATCH.json')
        pause = load(folder/'GPU_PAUSE_1735.json')
        assert dispatch['owner'] == pause['owner'] == 'S-RB2-BREAK'
        assert dispatch['pid'] == pid and pause['process'] == dispatch
        assert (folder/'STOP').exists() and not (folder/'COMPLETE.json').exists()
        identity_sha = digest(folder/'IDENTITY.json')
        tasks = []
        for path in sorted(folder.glob('w*_k*.json')):
            task = load(path)
            assert 969420 <= task['seed'] <= 969431 and 0 <= task['future'] < 8
            assert task['identity_sha256'] == identity_sha
            assert len(task['episodes']) == 5
            tasks.append(dict(name=path.name, seed=task['seed'], future=task['future'], sha256=digest(path)))
        for name, expected in pause['completed_hashes'].items():
            assert digest(folder/name) == expected, 'pre-pause receipt changed'
        assert 'AssertionError: cooperative stop' in (folder/'worker.log').read_text()
        shards.append(dict(shard=shard, pid=pid, source=snapshot['source'],
            snapshot_sha256=digest(folder/'SNAPSHOT.json'), tasks=tasks,
            completed_tasks=len(tasks), complete_k8_worlds=[], qualification=False))
    remote = json.loads(ssh('gpuA', '''import json,os,subprocess,time
from pathlib import Path
root=Path('/root/rbeval20261031/rb2break')
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def status(path):
 r=json.loads(path.read_text())
 if not isinstance(r,dict) or r.get('owner')!='S-RB2-BREAK' or 'pid' not in r: return None
 proc=Path('/proc')/str(r['pid'])/'stat'
 fields=proc.read_text().split() if proc.exists() else []
 alive=bool(fields and fields[2]!='Z' and fields[21]==r['start_ticks'] and boot==r['boot_id'])
 return dict(receipt=str(path),pid=r['pid'],alive=alive)
workers=[status(root/name/'DISPATCH.json') for name in ('devk8_native2_s0','devk8_native2_s1')]
queues=[row for path in sorted(root.glob('*QUEUE*.json')) if (row:=status(path)) is not None]
assert not any(row['alive'] for row in workers+queues), 'owned worker/queue still active'
gpu=subprocess.check_output(['nvidia-smi'],text=True)
print(json.dumps(dict(unix=time.time(),local_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
 boot_id=boot,load=os.getloadavg(),workers=workers,queues=queues,nvidia_smi=gpu)))
''', timeout=120))
    receipt = dict(schema='rb2break-gpu-checkpoint/1', owner='S-RB2-BREAK',
        reason='MAIN 2026-11-01 17:35 GPU priority',
        resume_requires='MAIN appends S-RB2-BREAK RESUME',
        no_new_gpu_jobs=True, no_signals=True, no_deletions=True,
        source_head='5947c801ba93', development_only=True,
        shards=shards, completed_tasks=sum(s['completed_tasks'] for s in shards),
        complete_k8_worlds=[], qualification=False, resource_audit=remote,
        source_sha256=digest(__file__))
    save_new(args.output, receipt)
    print(json.dumps(dict(receipt=str(args.output), completed_tasks=receipt['completed_tasks'],
        complete_k8_worlds=[], workers_exited=True, queues_exited=True)))


if __name__ == '__main__':
    main()
