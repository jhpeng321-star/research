"""Read-only, receipt-only v7 interruption audit. Never launches simulation."""
import json

from .continuation import CPU, GPU, TOTAL
from .ops import EVIDENCE, ssh
from .storage import canonical, write


CPU_AUDIT = '''import hashlib,json,os,time
from pathlib import Path
r=Path(ROOT)
def read(p):return json.loads(p.read_text()) if p.exists() else None
rows=[json.loads(x) for x in (r/'accounting.jsonl').read_text().splitlines()]
jobs=[]
for p in sorted((r/'jobs').iterdir()):
 if not p.is_dir():continue
 start=read(p/'START.json') or {};close=read(p/'CLOSE.json');result=read(p/'RESULT.json')
 ident=p.name;charges=[x for x in rows if x['id'].startswith(ident+'_') or x['id']==ident]
 jobs.append(dict(id=ident,start=start,close=close,result_present=result is not None,
  picks=[int(x.stem.split('j')[-1]) for x in sorted(p.glob('pick_j*.json'))],
  final_pick=read(p/'pick_j9.json'),
  measurements=[int(x.stem.split('j')[-1]) for x in sorted(p.glob('measurement_j*.json'))],
  charged_sim_s=sum(x['sim_s'] for x in charges),
  branch_logs={x.name:x.read_text()[-6000:] for x in sorted(p.glob('branch_*.log')) if 'Traceback' in x.read_text()}))
receipts={str(p.relative_to(r)):read(p) for p in [r/'STOP',r/'main/STATUS.json',r/'main_DISPATCH.json']}
owners=[read(p) for p in sorted((r/'main').glob('OWN_*.json'))]
owners += [x['start'] for x in jobs]
pids=sorted({x['pid'] for x in owners if x and x.get('pid')}|{receipts['main_DISPATCH.json']['pid']})
rpc={}
for arm in ('wm','bb'):
 d=r/'rpc'/arm;req=[read(p) for p in d.glob('*.request.json')]
 rpc[arm]=dict(requests=len(req),results=len(list(d.glob('*.result.json'))),
  timing=[dict(id=x['id'],request_unix=(d/(x['request_sha256']+'.request.json')).stat().st_mtime,
   result_unix=(d/(x['request_sha256']+'.result.json')).stat().st_mtime
   if (d/(x['request_sha256']+'.result.json')).exists() else None) for x in req],
  missing_results=[x['id'] for x in req if not (d/(x['request_sha256']+'.result.json')).exists()],
  stop=read(d/'STOP'))
logs={p.name:p.read_text()[-6000:] for p in sorted((r/'main').glob('*.log')) if 'Traceback' in p.read_text()}
print(json.dumps(dict(unix=time.time(),load=os.getloadavg(),receipts=receipts,jobs=jobs,
 own_pids={str(pid):Path('/proc',str(pid)).exists() for pid in pids},rpc=rpc,logs=logs,
 log_mtimes={p.name:p.stat().st_mtime for p in (r/'main').glob('*.log')},
 accounting_sha256=hashlib.sha256((r/'accounting.jsonl').read_bytes()).hexdigest(),accounting=rows)))
'''

GPU_AUDIT = '''import json,time
from pathlib import Path
r=Path(ROOT);out={}
for arm in ('wm','bb'):
 d=r/'rpc'/arm
 names=['GPU_STARTED.json','GPU_FINISHED.json','STOP','READY.json']
 rows={n:json.loads((d/n).read_text()) for n in names if (d/n).exists()}
 pid=rows.get('READY.json',{}).get('pid')
 rows['owned_pid_alive']=bool(pid and Path('/proc',str(pid)).exists())
 rows.pop('READY.json',None)
 rows['log_tail']=(r/(arm+'_gpu.log')).read_text()[-6000:]
 out[arm]=rows
print(json.dumps(dict(unix=time.time(),arms=out)))
'''


def summarize(cpu, gpu):
    rows = cpu['accounting']
    jobs = cpu['jobs']
    episodes = [x for x in jobs if x['id'].endswith('_episode')]
    complete = [x for x in episodes if x['result_present'] and x['close'] and
                x['close']['actual_sim_s'] == x['close']['charged_sim_s'] == 900]
    partial = [x for x in episodes if x not in complete]
    partial_branches = [x for x in jobs if not x['id'].endswith('_episode') and not x['result_present']]
    # Every incomplete run must be replayed from zero. Successful branch results
    # are reusable; all abandoned episode AND branch physics remain charged.
    abandoned_episodes = sum(x['close']['actual_sim_s'] for x in partial if x['close'])
    abandoned_branches = sum(x['close']['actual_sim_s'] for x in partial_branches if x['close'])
    extra = abandoned_episodes + abandoned_branches
    charged = sum(x['sim_s'] for x in rows)
    actual = 47010 + 12450 + sum(x['close']['actual_sim_s'] for x in jobs if x['close'])
    return dict(charged_sim_s=charged, actual_closed_sim_s=actual,
                complete_episodes=[x['id'] for x in complete],
                partial_episodes=[dict(id=x['id'],close=x['close'],picks=x['picks']) for x in partial],
                completed_branches=sum(x['result_present'] for x in jobs if not x['id'].endswith('_episode')),
                abandoned_episode_sim_s=abandoned_episodes, abandoned_branch_sim_s=abandoned_branches,
                standard_extra_replay_sim_s=extra,
                standard_completion_plan_sim_s=TOTAL+extra,
                registered_plan_sim_s=TOTAL, hard_cap_sim_s=500000,
                gpu_cumulative_s={a:g['GPU_FINISHED.json']['cumulative_gpu_s'] for a,g in gpu['arms'].items()},
                all_audited_owned_pids_absent=not any(cpu['own_pids'].values()) and
                not any(g['owned_pid_alive'] for g in gpu['arms'].values()))


def main():
    cpu = ssh('cpuQ', CPU_AUDIT.replace('ROOT', repr(CPU)), timeout=90)
    gpu = ssh('gpuR', GPU_AUDIT.replace('ROOT', repr(GPU)), timeout=90)
    report = dict(cpuQ=cpu, gpuR=gpu, summary=summarize(cpu,gpu))
    write(EVIDENCE/'V7_INTERRUPTION_AUDIT.json', report)
    print(json.dumps(dict(summary=report['summary'],receipt_sha256=canonical(report)),indent=2))


if __name__ == '__main__':
    main()
