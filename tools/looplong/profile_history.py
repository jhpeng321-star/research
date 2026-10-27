"""Read-only cycle-1 WM/check timing receipt; never starts remote work."""
import json
from pathlib import Path
import shlex
import subprocess

SOURCE = '''import json,pathlib
root=pathlib.Path('/root/autodl-tmp/c4/restart20261026/arm/loop')
p=root/'wm/c01/train_log.jsonl'
rows=[json.loads(x) for x in p.read_text().splitlines()]
out={'wm_steps':len(rows),'wm_optimizer_seconds':sum(x.get('seconds',0) for x in rows),
     'wm_first_step':{k:v for k,v in rows[0].items() if k in ('seconds','loss','stage_s','n_chunks','follower_courtesy')},
     'wm_mean_step_seconds':sum(x.get('seconds',0) for x in rows)/len(rows)}
check=root/'checks/c01/ACCEPT.json'
if check.is_file():
 d=json.loads(check.read_text()); out['acceptance']={k:d[k] for k in ('wall_s','decision') if k in d}
print(json.dumps(out))
'''


if __name__ == '__main__':
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpuA',
        shlex.join(['/root/miniconda3/bin/python', '-B', '-c', SOURCE])],
        capture_output=True, text=True, timeout=60, check=True)
    doc = json.loads(proc.stdout)
    Path(__file__).with_name('CYCLE1_TIMING.json').write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))
