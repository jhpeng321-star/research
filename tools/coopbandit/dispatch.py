"""Recoverable transport/control plane; no physics retries or broad remote reads."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import time

from .ops import HOSTS, LOCAL, REG, remote, launch
from .protocol import DATA_CAP, TIMES, FORBIDDEN, canonical, digest, sha, worlds
from .storage import read, write

SCREEN = '/home/ubuntu/coopbandit20261031_v3'
LABEL_ROOTS = {h:c['root'] for h,c in HOSTS.items()}
LABEL_ROOTS['cpuQ'] = '/home/ubuntu/coopbandit20261031_v3_labels'
EVIDENCE = LOCAL/'tools/coopbandit/evidence'
AGGREGATE = '/home/ubuntu/coopbandit20261031_aggregate_v3'


def active_plan():
    p=EVIDENCE/'RECOVERY_PLAN.json'
    return read(p if p.exists() else EVIDENCE/'SHARDS.json')


def once(path, value):
    if Path(path).exists():
        if canonical(read(path)) != canonical(value):
            raise ValueError('immutable receipt changed '+str(path))
    else:
        write(path,value)


def status(host, root):
    return json.loads(remote(host,f'''import pathlib,json,os,shutil,time
r=pathlib.Path({root!r})
jobs=r/'jobs'; results=[p for p in jobs.glob('*/RESULT.json') if (p.parent/'CLOSE.json').exists()] if jobs.exists() else []
charges=[json.loads(s) for s in (r/'CHARGES.jsonl').read_text().splitlines()] if (r/'CHARGES.jsonl').exists() else []
charged_ids={{x['id'] for x in charges}}
closed=[json.loads(p.read_text()) for p in jobs.glob('*/CLOSE.json') if p.parent.name in charged_ids] if jobs.exists() else []
print(json.dumps(dict(host={host!r},root=str(r),unix=time.time(),results=len(results),
 result_ids=[p.parent.name for p in results],charged=sum(x['sim_s'] for x in charges),
 actual_closed=sum(x['actual_sim_s'] for x in closed),load=os.getloadavg(),disk_free=shutil.disk_usage(r).free,
 canary=(r/'CANARY.json').exists(),screen_complete=(r/'SCREEN_COMPLETE.json').exists(),
 branch_canary=(r/'BRANCH_CANARY_COMPLETE.json').exists(),branches_complete=(r/'BRANCH_COMPLETE.json').exists(),
 stop=json.loads((r/'STOP').read_text()) if (r/'STOP').exists() else None)))
'''))


def fetch(host, root, names):
    """Only explicit public receipt names, compressed in transit; no scenario files."""
    for name in names:
        if FORBIDDEN.search(name) or '..' in Path(name).parts or not name.endswith('.json'):
            raise ValueError('invalid receipt transport')
    encoded=remote(host,f'''import pathlib,json,gzip,base64
r=pathlib.Path({root!r})
names={names!r}
data={{name:(r/name).read_text() for name in names}}
print(base64.b64encode(gzip.compress(json.dumps(data).encode())).decode())
''',timeout=120).strip()
    return json.loads(gzip.decompress(base64.b64decode(encoded)))


def send(host, root, files):
    for name in files:
        if FORBIDDEN.search(name) or '..' in Path(name).parts or not name.endswith('.json'):
            raise ValueError('invalid receipt transport')
    payload=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()
    return remote(host,f'''import pathlib,json,gzip,base64,os
r=pathlib.Path({root!r});assert r.is_dir()
files=json.loads(gzip.decompress(base64.b64decode({payload!r})))
for name,text in files.items():
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_text()==text,'immutable receipt '+name
 else:
  with p.open('x') as f: f.write(text);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(files=len(files))))
''',timeout=120)


def audit_gpuA():
    seeds=[w['seed'] for w in worlds()]+list(range(981080,981092))
    source=f'''import pathlib,os,re,json
forbidden=re.compile({FORBIDDEN.pattern!r},re.I)
identity=re.compile(r'(?:^|[^0-9a-fA-F.])(?:s|w)?(?:'+ '|'.join(map(str,{seeds!r}))+r')(?![0-9a-fA-F])')
hits=[];count=0
for parent,dirs,files in os.walk('/root'):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not forbidden.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  p=pathlib.Path(parent)/name
  if forbidden.search(str(p)): continue
  count+=1
  if identity.search(name): hits.append(str(p))
print(json.dumps(dict(host='gpuA',safe_names=count,hits=hits,pass_=not hits)))
'''
    result=json.loads(remote('gpuA',source,timeout=120))
    if not result['pass_']:
        raise ValueError('gpuA exposure collision '+str(result['hits']))
    once(EVIDENCE/'GPUA_EXPOSURE.json',result)
    return result


def make_plan(selection):
    panels=selection['worlds']
    # Two light-cell worlds on marginal-disk gpuR; remaining worlds balanced by load.
    gpur={next(e['world']['seed'] for e in panels if e['world']['cell']==c) for c in (0,3)}
    assignment={}; i=0
    for entry in panels:
        w=entry['world']
        if w['seed'] in gpur:
            assignment[w['seed']]='gpuR'
        else:
            assignment[w['seed']]='cpuQ' if i%9<4 else 'gpuA'
            i+=1
    shards={h:dict(root=LABEL_ROOTS[h],jobs=[],attempts={},worlds=[]) for h in HOSTS}
    for entry in panels:
        w=entry['world']; shard=shards[assignment[w['seed']]];shard['worlds'].append(w)
        for t in TIMES:
            options=[(oid,None) for oid in ('O1','O2','O3','O4')]
            if w['seed'] in selection['solo_worlds']:
                options += [(oid,j) for oid in ('O2','O3') for j in (0,1)]
            for oid,j in options:
                ident=f's{w["seed"]}_t{t}_{oid}'+('' if j is None else f'_solo{j}')
                shard['jobs'].append([w,t,oid,j])
                shard['attempts'][ident]=dict(sim_s=t+600,phase='data')
    all_ids=[ident for s in shards.values() for ident in s['attempts']]
    if len(all_ids)!=len(set(all_ids)) or sum(a['sim_s'] for s in shards.values() for a in s['attempts'].values())+102000!=DATA_CAP:
        raise ValueError('disjoint exact global budget')
    return dict(screen_root=SCREEN,shards=shards,selection=selection,successful_data=DATA_CAP,
                abandoned_charged=2280,abandoned_actual=366,planned_live=99360,
                conservative_total=DATA_CAP+2280+99360,dispatch_source_sha256=sha(__file__))


def plan():
    current=status('cpuQ',SCREEN)
    if current['stop'] or not current['screen_complete']:
        print(canonical(current));return None
    selection=json.loads(fetch('cpuQ',SCREEN,['SELECTED.json'])['SELECTED.json'])
    result=make_plan(selection)
    once(EVIDENCE/'SHARDS.json',result)
    print(canonical(dict(plan_sha256=digest(result),hosts={h:dict(worlds=len(s['worlds']),jobs=len(s['jobs']),sim_s=sum(a['sim_s'] for a in s['attempts'].values())) for h,s in result['shards'].items()},conservative_total=result['conservative_total'])))
    return result


def install():
    allocation=active_plan()
    # cpuQ labels get a separate immutable allocation; screens retain their original one.
    r=LABEL_ROOTS['cpuQ']
    remote('cpuQ',f'''import pathlib,shutil,json
r=pathlib.Path({r!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
s=pathlib.Path({SCREEN!r})
for name in ('COOPBANDIT_DESIGN.md','BINDING.json','APPROVAL.json'):
 p=r/name
 if p.exists(): assert p.read_bytes()==(s/name).read_bytes()
 else: shutil.copyfile(s/name,p)
if not (r/'pytools').exists(): (r/'pytools').symlink_to(s/'pytools',target_is_directory=True)
print('cpuQ label source bound to successful screen runtime')
''')
    for host,shard in allocation['shards'].items():
        root=shard['root']
        files={'ALLOCATION.json':canonical(dict(host=host,attempts=shard['attempts'],global_plan_sha256=digest(allocation)))+'\n',
               'BRANCH_JOBS.json':canonical(shard['jobs'])+'\n',
               'GLOBAL_ALLOCATION.json':canonical(allocation)+'\n'}
        send(host,root,files)
        if host=='cpuQ':
            # Same filesystem: copy only explicitly named public receipts, no generated inputs.
            names=[f'jobs/s{w["seed"]}_screen/{n}' for w in shard['worlds'] for n in ('RESULT.json','CLOSE.json')]
            remote(host,f'''import pathlib,shutil
s=pathlib.Path({SCREEN!r});r=pathlib.Path({root!r})
for name in {names!r}:
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_bytes()==(s/name).read_bytes()
 else: shutil.copyfile(s/name,p)
print('baseline receipts installed')
''')
        else:
            for w in shard['worlds']:
                names=[f'jobs/s{w["seed"]}_screen/{n}' for n in ('RESULT.json','CLOSE.json')]
                send(host,root,fetch('cpuQ',SCREEN,names))
        print(canonical(dict(host=host,installed_worlds=len(shard['worlds']),plan_sha256=digest(allocation))))


def canaries():
    allocation=active_plan()
    for host,shard in allocation['shards'].items():
        root=shard['root'];config=dict(HOSTS[host]);config['root']=root
        # Explicit one-job gate before pool expansion, to limit possible cross-host drift cost.
        code=f'''import pathlib,json,os,sys,time
from tools.coopbandit.runner import init,branch,completed,job_id
from tools.coopbandit.storage import read,write,stop
r=pathlib.Path({root!r});(r/'jobs').mkdir(exist_ok=True)
write(r/'OWNED_BRANCH_CANARY.json',dict(pid=os.getpid(),unix=time.time()))
try:
 init(str(r));job=read(r/'BRANCH_JOBS.json')[0]
 if completed(r,job_id(job)) is None: branch(job)
 write(r/'BRANCH_CANARY_COMPLETE.json',dict(job_id=job_id(job),bitwise_prefix=True,unix=time.time()))
except BaseException:
 import traceback
 stop(r,traceback.format_exc());raise
'''
        output=remote(host,f'''import pathlib,subprocess,os,sys,json
r=pathlib.Path({root!r});sys.path[:0]={config['deps']!r}
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=':'.join([str(r/'pytools')]+{config['deps']!r}),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'SUPERVISOR_canary.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-c',{code!r}],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(host={host!r},root=str(r),pid=p.pid,phase='branch-canary')))
''')
        receipt=json.loads(output)
        once(EVIDENCE/f'DISPATCH_CANARY_{host}.json',receipt)
        print(canonical(receipt))


def expand():
    allocation=active_plan()
    for host,shard in allocation['shards'].items():
        current=status(host,shard['root'])
        if current['stop']:
            print(canonical(current));continue
        if not current['branch_canary']:
            print(canonical(dict(host=host,waiting_for_canary=True)));continue
        receipt=EVIDENCE/f'DISPATCH_BRANCHES_{host}.json'
        if receipt.exists():
            print(canonical(dict(host=host,already_dispatched=True)));continue
        original=HOSTS[host]['root']
        try:
            HOSTS[host]['root']=shard['root']
            result=json.loads(launch(host,'branches'))
            once(receipt,result)
            print(canonical(result))
        finally:
            HOSTS[host]['root']=original


def collect():
    allocation=active_plan()
    initial=[f'jobs/s{w["seed"]}_screen/{n}' for w in worlds() for n in ('RESULT.json','CLOSE.json')]
    initial += [f'jobs/s{next(w for w in worlds() if w["cell"]==c)["seed"]}_repeat/{n}' for c in range(12) for n in ('RESULT.json','CLOSE.json')]
    remote('cpuQ',f'''import pathlib,shutil,json
r=pathlib.Path({AGGREGATE!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
s=pathlib.Path({SCREEN!r})
for name in {initial!r}+['COOPBANDIT_DESIGN.md','SELECTED.json']:
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_bytes()==(s/name).read_bytes()
 else: shutil.copyfile(s/name,p)
print('screen receipts aggregated')
''')
    existing=set(status('cpuQ',AGGREGATE)['result_ids'])
    counts={}
    ledgers=[]
    screen_ledger=remote('cpuQ',f'import pathlib; print((pathlib.Path({SCREEN!r})/"CHARGES.jsonl").read_text())')
    ledgers.extend(json.loads(s) for s in screen_ledger.splitlines() if s.strip())
    for host,shard in allocation['shards'].items():
        current=status(host,shard['root'])
        if current['stop']:
            raise ValueError('shard STOP: '+canonical(current))
        ids=[ident for ident in current['result_ids'] if ident in shard['attempts'] and ident not in existing]
        for begin in range(0,len(ids),50):
            names=[f'jobs/{ident}/{name}' for ident in ids[begin:begin+50] for name in ('RESULT.json','CLOSE.json')]
            send('cpuQ',AGGREGATE,fetch(host,shard['root'],names))
        ledger=remote(host,f'import pathlib; p=pathlib.Path({shard["root"]!r})/"CHARGES.jsonl"; print(p.read_text() if p.exists() else "")')
        ledgers.extend(json.loads(s) for s in ledger.splitlines() if s.strip())
        counts[host]=dict(completed=sum(ident in shard['attempts'] for ident in current['result_ids']),
                          charged=current['charged'],actual_closed=current['actual_closed'],stop=current['stop'])
    if len({r['id'] for r in ledgers})!=len(ledgers):
        raise ValueError('aggregate duplicate attempt')
    result=json.loads(remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({AGGREGATE!r});s=json.loads((r/'SELECTED.json').read_text())
complete=[]
for e in s['worlds']:
 for t in (300,360,420,480,540):
  ident=f's{{e["world"]["seed"]}}_t{{t}}'
  if all((r/'jobs'/f'{{ident}}_{{o}}'/'RESULT.json').exists() for o in ('O1','O2','O3','O4')): complete.append(ident)
print(json.dumps(dict(states=len(complete),state_ids=complete)))
'''))
    progress=dict(unix=time.time(),states=result['states'],hosts=counts,
                  conservative_charged=sum(r['sim_s'] for r in ledgers)+allocation['abandoned_charged']+allocation.get('diagnostic_charged',0),
                  actual_closed=sum(x['actual_closed'] for x in counts.values())+102000+allocation['abandoned_actual']+allocation.get('diagnostic_actual',0))
    once(EVIDENCE/f'PROGRESS_{int(progress["unix"])}.json',progress)
    # Unique immutable snapshot at the first >=100-state milestone, and at completion.
    full=all(counts[h]['completed']==len(s['attempts']) for h,s in allocation['shards'].items())
    marker=EVIDENCE/('FULL_EXPORT.json' if full else 'PARTIAL_EXPORT.json')
    if (full or result['states']>=100) and not marker.exists():
        name='full' if full else f'partial{result["states"]}'
        snapshot=AGGREGATE+'/snapshots/'+name
        payload=base64.b64encode(gzip.compress(canonical(ledgers).encode())).decode()
        output=remote('cpuQ',f'''import pathlib,json,os,sys,gzip,base64,shutil
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,{SCREEN!r}+'/pytools')
from tools.coopbandit.storage import write
from tools.coopbandit.export import export
r=pathlib.Path({snapshot!r});r.parent.mkdir(exist_ok=True);r.mkdir()
a=pathlib.Path({AGGREGATE!r});(r/'jobs').symlink_to(a/'jobs',target_is_directory=True)
for n in ('COOPBANDIT_DESIGN.md','SELECTED.json'): shutil.copyfile(a/n,r/n)
ledger=json.loads(gzip.decompress(base64.b64decode({payload!r})))
with (r/'CHARGES.jsonl').open('x') as f:
 for row in ledger: f.write(json.dumps(row)+chr(10))
export(r,partial={not full!r})
if {full!r}: write(r/'DATA_COMPLETE.json',dict(states=300,successful_data_sim_s=1489200,abandoned_actual={allocation['abandoned_actual']},abandoned_charged={allocation['abandoned_charged']},diagnostic_charged={allocation.get('diagnostic_charged',0)}))
print(json.dumps(dict(snapshot=str(r),states={result['states']},partial={not full!r})))
''',timeout=120)
        receipt=json.loads(output)
        once(marker,receipt)
        print(canonical(receipt))
    print(canonical(progress))
    return progress


def analyze():
    full=(EVIDENCE/'FULL_EXPORT.json').exists()
    marker=EVIDENCE/('FULL_EXPORT.json' if full else 'PARTIAL_EXPORT.json')
    exported=read(marker)
    receipt_path=EVIDENCE/('FULL_ANALYSIS.json' if full else 'PARTIAL_ANALYSIS.json')
    if receipt_path.exists():
        print(canonical(read(receipt_path)));return
    root=exported['snapshot']
    result=json.loads(remote('cpuQ',f'''import os,sys,json
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy
sys.path[:0]=[{SCREEN!r}+'/pytools','/home/ubuntu/coopbandit20261031_cpu1/deps']
from tools.coopbandit.learn import learn
summary=learn({root!r},partial={not full!r},exam_name={'exam' if full else 'exam-partial'!r})
print(json.dumps(dict(snapshot={root!r},partial={not full!r},summary=summary)))
''',timeout=120))
    once(receipt_path,result)
    print(canonical(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('status','audit-gpuA','plan','install','canaries','expand','collect','analyze'))
    a=p.parse_args()
    if a.mode=='status':
        print(canonical(status('cpuQ',SCREEN)))
        if (EVIDENCE/'SHARDS.json').exists():
            for h,s in active_plan()['shards'].items():
                print(canonical(status(h,s['root'])))
    elif a.mode=='audit-gpuA': print(canonical(audit_gpuA()))
    else: globals()[a.mode]()
