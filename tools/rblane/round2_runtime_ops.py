"""Launch/collect one owned CPU-only runtime smoke on an immutable stage."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from tools.rbint.remote import ssh

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('launch','status','collect'))
    ap.add_argument('--kind', default='runtime_v1')
    ap.add_argument('--runtime', default='/root/rbeval20261031/phys_code_6b4bfbd5cff8')
    args = ap.parse_args()
    import re
    if not re.fullmatch('[a-z0-9_]+', args.kind):
        raise ValueError('invalid run identity')
    receipt = ROOT/'docs/rb2lane'/(args.kind+'_DISPATCH.json')
    if args.command == 'launch':
        if receipt.exists():
            raise ValueError('immutable dispatch exists')
        code = f'''import pathlib,os,json,subprocess,time
r=pathlib.Path({args.runtime!r}); stage=json.loads((r/'INT_STAGE.json').read_text())
p=r.parent/{('rb2lane_'+args.kind)!r}
assert p.parent.is_dir() and not p.exists() and os.getloadavg()[0]<20
p.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'src')+':'+str(r)+':/root/autodl-tmp/c4/speedhead20261030/pythondeps',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.rblane.round2_runtime_smoke','--out',str(p/'SMOKE.json')]
with (p/'run.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,start_ticks=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],command=cmd,root=str(p),stage=stage,load=os.getloadavg(),cpu_only=True)
(p/'PID.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
'''
        record = json.loads(ssh('gpuA', code))
        receipt.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2))
        return
    record = json.loads(receipt.read_text())
    run = record['root']
    code = f'''import pathlib,json,os
p=pathlib.Path({run!r}); r=json.loads((p/'PID.json').read_text()); f=pathlib.Path('/proc/'+str(r['pid'])+'/stat')
alive=f.exists() and f.read_text().split()[21]==r['start_ticks'] and f.read_text().split()[2]!='Z'
result=p/'SMOKE.json'
print(json.dumps(dict(alive=alive,complete=result.is_file(),load=os.getloadavg(),progress=(p/'run.log').read_text().splitlines()[-20:])))
'''
    status = json.loads(ssh('gpuA', code))
    print(json.dumps(status, indent=2))
    if args.command == 'collect':
        target = receipt.with_name(args.kind+'_SMOKE.json')
        if status['alive'] or not status['complete'] or target.exists():
            raise ValueError('completed exited uncollected smoke required')
        subprocess.run(['scp','-q','gpuA:'+run+'/SMOKE.json',str(target)],check=True,timeout=60)
        receipt.with_name(args.kind+'_COMPLETE.json').write_text(json.dumps(dict(status,
            process=record, collected_at=time.time()),indent=2)+'\n')


if __name__ == '__main__':
    main()
