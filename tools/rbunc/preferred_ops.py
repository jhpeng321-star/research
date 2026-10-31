"""Exclusive preferred-posterior probe; preserve old panels and all failed jobs."""
from pathlib import Path
import argparse
import json

from remote import PYTHON, REMOTE, ROOT, ssh, stage

OUT = REMOTE + '/preferred_v1'
FIT = '/home/ubuntu/rbfit20261101_addendum/shrinkage_v1'
LANE = '/home/ubuntu/rbfit20261101_addendum/lane_v2'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('inspect', 'stage', 'dispatch', 'status', 'fetch'))
    p.add_argument('--code')
    a = p.parse_args()
    if a.command == 'stage':
        print(stage())
    elif a.command == 'inspect':
        print(ssh(f'''
from pathlib import Path
import os,json,subprocess
old=Path({REMOTE!r})/'addendum_v2'
terminal=json.loads((old/'probe_TERMINAL.json').read_text())
assert all(r['rc']==0 for r in terminal['results'])
gpu=subprocess.run(['sh','-c','command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(load=os.getloadavg(),cpus=os.cpu_count(),old_terminal=terminal,
 target_exists=Path({OUT!r}).exists(),fit=Path({FIT!r}).is_dir(),lane=Path({LANE!r}).is_dir(),
 gpu_status=gpu.returncode,gpu=gpu.stdout)))
'''))
    elif a.command == 'dispatch':
        assert a.code and a.code.startswith(REMOTE+'/code/')
        print(ssh(f'''
from pathlib import Path
import os,json,subprocess,time,hashlib
old=Path({REMOTE!r})/'addendum_v2'
terminal=json.loads((old/'probe_TERMINAL.json').read_text())
assert all(r['rc']==0 for r in terminal['results'])
assert os.getloadavg()[0]<24 and os.cpu_count()>=32,'insufficient free CPU'
assert Path({FIT!r}).is_dir() and Path({LANE!r}).is_dir()
root=Path({OUT!r});assert root.parent.is_dir();root.mkdir(exist_ok=False)
raw=(old/'observed_arrivals.json').read_bytes()
with (root/'observed_arrivals.json').open('xb') as f:f.write(raw)
with (root/'probe_DISPATCH.json').open('x') as f:json.dump(dict(start_unix=time.time(),
 code={a.code!r},fit={FIT!r},lane={LANE!r},load=os.getloadavg(),
 arrival_sha256=hashlib.sha256(raw).hexdigest(),old_terminal=terminal),f)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 CUDA_VISIBLE_DEVICES='',PYTHONPATH={a.code!r}+'/src:'+{a.code!r})
with (root/'probe.log').open('x') as log:
 child=subprocess.Popen([{PYTHON!r},'-B',{a.code!r}+'/tools/rbunc/fitted_panel.py',
  '--out',str(root),'--fit-directory',{FIT!r},'--lane-directory',{LANE!r}],
  cwd={a.code!r},env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=child.pid,code={a.code!r})
 with (root/'probe_PID.json').open('x') as f:json.dump(receipt,f)
 print(json.dumps(receipt))
'''))
    elif a.command == 'status':
        print(ssh(f'''
from pathlib import Path
import os,json
root=Path({OUT!r})
print(json.dumps(dict(load=os.getloadavg(),complete=[s for s in range(969400,969432)
 if (root/f'fitted_w{{s}}.json').exists()])))
for name in ['probe.log','probe_TERMINAL.json']+[f'probe_worker{{i}}.log' for i in range(4)]:
 p=root/name
 if p.exists():print(name,p.read_text())
'''))
    else:
        text = ssh(f'''
from pathlib import Path
import json
root=Path({OUT!r})
terminal=json.loads((root/'probe_TERMINAL.json').read_text())
assert all(r['rc']==0 for r in terminal['results'])
records=[json.loads((root/f'fitted_w{{s}}.json').read_text()) for s in range(969400,969432)]
receipts={{n:json.loads((root/n).read_text()) for n in
 ['probe_DISPATCH.json','probe_PID.json','probe_TERMINAL.json']+
 [f'probe_worker{{i}}_PID.json' for i in range(4)]}}
print(json.dumps(dict(records=records,receipts=receipts)))
''')
        path = ROOT/'docs/paper_results/rbunc/preferred_source_probe.json'
        assert path.parent.is_dir()
        with path.open('x', encoding='utf-8') as f:
            f.write(text)
        print('FETCHED', path, len(text))


if __name__ == '__main__':
    main()
