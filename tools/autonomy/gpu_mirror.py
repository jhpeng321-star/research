"""Detached gpuA pull mirror: existing gpuA->collector SSH, no laptop/key relay."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

R = Path('/root/autodl-tmp/c4/restart20261026')
BASE = '/home/ubuntu/c4/main14/restart20261026'
TOKENS = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
          'hetdata_truth', 'vtype', 'assignment')


def main():
    while True:
        doc = {'unix': time.time(), 'pid': os.getpid(), 'sid': os.getsid(0), 'boxes': {}}
        for box in ('cpuP', 'cpuQ'):
            remote = '''import pathlib,json
b=pathlib.Path(%r);paths=[]
for p in (b/'work').glob('w*/C4-LOOP14-COOPON-20261026/c*'):
 if any(t in str(p).lower() for t in %r): continue
 n=p.name.split('.')[0][1:]
 if not n.isdigit() or int(n)<2: continue
 for name in ('artifact','collect/branches'):
  q=p/name
  if q.is_dir(): paths.append(str(q.relative_to(b)))
s=b/'autonomy/SPOT_STATUS.json'
print(json.dumps({'paths':paths,'watcher':json.loads(s.read_text()) if s.exists() else None}))
''' % (BASE, TOKENS)
            ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
            try:
                p = subprocess.run([*ssh, box, '/home/ubuntu/miniconda3/bin/python -B -c ' + shlex.quote(remote)],
                    capture_output=True, text=True, timeout=30)
                assert p.returncode == 0, p.stderr[-1000:]
                result = json.loads(p.stdout)
                result.update(reachable=True, mirrors=[])
                root = R / 'spot_mirror' / box
                root.mkdir(parents=True, exist_ok=True)
                (root / 'WATCHER_HEARTBEAT.json').write_text(json.dumps(result['watcher'], indent=2))
                for rel in result['paths']:
                    if any(t in rel.lower() for t in TOKENS) or '..' in Path(rel).parts:
                        raise ValueError('restricted mirror path refused')
                    import shutil
                    if shutil.disk_usage(R).free < 3 * 1024**3:
                        result['pause'] = 'gpuA disk floor3GiB'
                        break
                    dst = root / rel
                    dst.mkdir(parents=True, exist_ok=True)
                    cmd = ['rsync', '-a', '--partial', '--timeout=30', '-e', shlex.join(ssh)]
                    cmd += ['--exclude=*' + t + '*' for t in TOKENS]
                    cmd += ['--include=*/', '--include=*.json', '--include=*.jsonl',
                            '--include=*.npz', '--include=*.npy', '--exclude=*',
                            box + ':' + BASE + '/' + rel + '/', str(dst) + '/']
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                    result['mirrors'].append({'source': rel, 'exit': p.returncode, 'error': p.stderr[-400:]})
                doc['boxes'][box] = result
            except Exception as error:
                doc['boxes'][box] = {'reachable': False, 'error': repr(error), 'collection_action': 'peer or pause'}
        temp = R / 'autonomy/MIRROR_STATUS.tmp'
        temp.write_text(json.dumps(doc, indent=2))
        temp.replace(R / 'AUTONOMY_MIRROR_STATUS.json')
        time.sleep(20)


if __name__ == '__main__':
    main()
