"""Owned multi-host expansion; frozen worker/scoring and one collector retained."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

from probe import ssh, allowed
from operate import HERE, REMOTE, scp


def save(name, value):
    with (HERE/'evidence'/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=1)


def audit():
    code = f'''from pathlib import Path
import json,os,subprocess,hashlib
p=Path({REMOTE!r})
result=dict(host=os.uname().nodename,load=os.getloadavg(),cpus=os.cpu_count(),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free','--format=csv,noheader'],text=True),
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True))
if (p/'DISPATCH.json').exists():
 d=json.loads((p/'DISPATCH.json').read_text());result['dispatch']=d
 q=Path('/proc/'+str(d['pid'])+'/stat')
 result['dispatch_stat']=q.read_text() if q.exists() else None
 result['completed']=len(list((p/'panel').glob('*_h300.json')))
 result['reference']=json.loads((p/'panel/wm2_common_w0_k0_h300.json').read_text())
print(json.dumps(result))
'''
    a=json.loads(ssh('gpuA', code)); r=json.loads(ssh('gpuR', code))
    ref=a.pop('reference')
    paths=set(ref['identity']['runtime_closure']) | set(ref['identity']['dependencies'])
    paths.update(('/root/autodl-tmp/c4/speedhead20261030/tools/speedhead/wm_gpu_wmcalib.py',
                  '/root/autodl-tmp/c4/speedhead20261030/tools/speedhead/wm_worker.py',
                  '/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json',
                  '/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt',
                  '/root/wmcalib/expected_hashes.json','/root/wmcalib/data/design.json'))
    # The runtime records concrete config/candidate paths in its source.
    for seed in range(969400,969440):
        paths.update(f'/root/wmcalib/data/jobs/s{seed}_ctrl0/{n}'
                     for n in ('start.pt','start_history.npz','result.json'))
    for p in paths: allowed(p)
    manifests={}
    for host in ('gpuA','gpuR'):
        manifests[host]=json.loads(ssh(host, f'''from pathlib import Path
import json,hashlib
paths={sorted(paths)!r}
print(json.dumps({{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() if Path(n).is_file() else None for n in paths}}))
'''))
    differences={p: [manifests[h][p] for h in ('gpuA','gpuR')]
                 for p in paths if manifests['gpuA'][p] != manifests['gpuR'][p]}
    save('EXPAND_AUDIT.json',dict(gpuA=a,gpuR=r,manifests=manifests,differences=differences,
         reference_identity=ref['identity']))
    print(json.dumps(dict(gpuA=a,gpuR=r,files=len(paths),differences=differences),indent=1))


def stage():
    audit=json.loads((HERE/'evidence/EXPAND_AUDIT.json').read_text())
    binding='/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json'
    assert set(audit['differences']) <= {binding}, 'do not overwrite shared peer runtime'
    paths=['/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml',
           '/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',
           '/root/autodl-tmp/c4/setupaudit20261030/fix/configs/execparity_cav_idm_v1.json',
           '/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt']
    extra={}
    for host in ('gpuA','gpuR'):
        extra[host]=json.loads(ssh(host,f'''from pathlib import Path
import hashlib,json
b=json.loads(Path({binding!r}).read_text())
keys=('path','sha256','iteration','lane_head','lane_control','lanes_is_none','preflight_contract_sha256','action_audit')
print(json.dumps(dict(binding={{k:b[k] for k in keys}},hashes={{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() if Path(n).is_file() else None for n in {paths!r}}})))
'''))
    for path,expected in extra['gpuA']['hashes'].items():
        assert expected is not None
        if extra['gpuR']['hashes'][path] is None:
            allowed(path)
            local=HERE/'out'/('expand_input_'+Path(path).name)
            print(scp(f'gpuA:{path}',local))
            assert hashlib.sha256(local.read_bytes()).hexdigest()==expected
            print(ssh('gpuR',f'''from pathlib import Path
p=Path({path!r});assert Path('/root/autodl-tmp/c4').is_dir();assert not p.exists()
resolved=p.resolve();assert not any(x in str(resolved).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype'))
print('resolved missing input',p,'=>',resolved)
assert str(resolved).startswith(('/root/autodl-tmp/c4/','/root/wmcalib/')), 'external symlink needs explicit review'
resolved.parent.mkdir(parents=True,exist_ok=True)
print('verified absent input destination',p)
'''))
            print(scp(local,f'gpuR:{path}'))
            actual=ssh('gpuR',f'from pathlib import Path;import hashlib;print(hashlib.sha256(Path({path!r}).read_bytes()).hexdigest())').strip()
            assert actual==expected
            extra['gpuR']['hashes'][path]=actual
    assert extra['gpuA']==extra['gpuR'],'concrete model/config/export mismatch'
    save('EXPAND_INPUTS_VERIFIED.json',dict(status='PASS',matching_files=len(audit['manifests']['gpuA'])-len(audit['differences']),
         supplemental=extra,binding_difference='host/source/diagnostic metadata only; executable fields and actual checkpoint identical'))
    print(ssh('gpuR',f'''from pathlib import Path
p=Path({REMOTE!r});assert p.parent.is_dir();p.mkdir(exist_ok=True)
for name in ('evidence','panel','parity','ckpt'):(p/name).mkdir(exist_ok=True)
assert not (p/'EXPAND_ASSIGNMENT.json').exists()
print('verified task destination',p)
'''))
    for name in ('worker.py','aggregates.py','expand_runner.py'):
        print(scp(HERE/name,f'gpuR:{REMOTE}/{name}'))
    print(scp(HERE/'expand_runner.py',f'gpuA:{REMOTE}/expand_runner.py'))


def parity():
    attempt=1
    while (HERE/'evidence'/f'EXPAND_PARITY{attempt}_OWNED.json').exists() or (attempt==1 and (HERE/'evidence/EXPAND_PARITY_OWNED.json').exists()):
        attempt+=1
    code=f'''from pathlib import Path
import json,os,subprocess,time,hashlib
p=Path({REMOTE!r});dest=p/'parity';assert dest.is_dir()
for q in (p/'evidence').glob('EXPAND_PARITY*_OWNED.json'):
 old=json.loads(q.read_text());stat=Path('/proc/'+str(old['pid'])+'/stat')
 if stat.exists():
  s=stat.read_text().split(') ',1)[1].split();assert s[19]!=old['start_ticks'] or s[0]=='Z','existing parity still live'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=5120
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/'worker.py'),'--model','wm2','--world','0','--future','0','--plant','common','--seconds','300','--out','parity','--device','cuda']
with (p/'evidence/EXPAND_PARITY{attempt}.log').open('x') as log:
 child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
  env={{**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
 receipt=dict(pid=child.pid,start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split(') ',1)[1].split()[19],command=cmd,started=time.time(),role='parity only; never counted in panel')
 with (p/'evidence/EXPAND_PARITY{attempt}_OWNED.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
'''
    receipt=json.loads(ssh('gpuR',code));save(f'EXPAND_PARITY{attempt}_OWNED.json',receipt)
    print(json.dumps(receipt))


def repair_configs():
    paths=json.loads(ssh('gpuA', '''from pathlib import Path
import tomllib,json,hashlib
paths={}
def read(p):
 p=p.resolve();assert not any(x in str(p).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype'))
 b=p.read_bytes();paths[str(p)]=hashlib.sha256(b).hexdigest();v=tomllib.loads(b.decode())
 parent=v.get('extends')
 if parent:read(p.parent/parent)
read(Path('/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'))
print(json.dumps(paths))
'''))
    actual=json.loads(ssh('gpuR',f'''from pathlib import Path
import json,hashlib
print(json.dumps({{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None for p in {list(paths)!r}}}))
'''))
    copied=[]
    for path,digest in paths.items():
        allowed(path)
        if actual[path]==digest:continue
        assert actual[path] is None,'never overwrite a shared config'
        local=HERE/'out'/('expand_config_'+Path(path).name)
        print(scp(f'gpuA:{path}',local));assert hashlib.sha256(local.read_bytes()).hexdigest()==digest
        print(ssh('gpuR',f'from pathlib import Path;p=Path({path!r});assert not p.exists();assert p.parent.is_dir();print(p)'))
        print(scp(local,f'gpuR:{path}'));copied.append(path)
        assert ssh('gpuR',f'from pathlib import Path;import hashlib;print(hashlib.sha256(Path({path!r}).read_bytes()).hexdigest())').strip()==digest
    save('EXPAND_CONFIG_CLOSURE.json',dict(status='PASS',manifest=paths,copied_absent_only=copied,
         parity_attempt1='failed before inference: missing inherited config; preserved log and exact PID receipt'))
    print(json.dumps(dict(status='PASS',configs=len(paths),copied=copied)))


def verify_parity():
    target=HERE/'out/expand_parity.json'
    print(scp(f'gpuR:{REMOTE}/parity/wm2_common_w0_k0_h300.json',target))
    ref=json.loads(ssh('gpuA',f'from pathlib import Path;print(Path({(REMOTE+"/panel/wm2_common_w0_k0_h300.json")!r}).read_text())'))
    actual=json.loads(target.read_text())
    ignored={'runtime_seconds','pid'}
    assert ref.keys()==actual.keys()
    differences=[k for k in ref if k not in ignored and ref[k]!=actual[k]]
    assert not differences, 'cross-host exact parity failed: '+str(differences)
    receipt=dict(status='PASS',comparison='exact all fields except pid/runtime_seconds',
                 task='wm2_common_w0_k0_h300',worker_sha256=actual['identity']['source_sha256'],
                 reference_sha256=hashlib.sha256(json.dumps(ref,sort_keys=True).encode()).hexdigest(),
                 gpuR_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), counted_panel_tasks=0)
    save('EXPAND_PARITY_VERIFIED.json',receipt);print(json.dumps(receipt))


def diagnose_parity():
    import numpy as np
    ref=json.loads(ssh('gpuA',f'from pathlib import Path;print(Path({(REMOTE+"/panel/wm2_common_w0_k0_h300.json")!r}).read_text())'))
    actual=json.loads((HERE/'out/expand_parity.json').read_text())
    diffs=[]
    def compare(a,b,path):
        if a==b:return
        if isinstance(a,dict) and isinstance(b,dict):
            for k in a.keys()|b.keys():compare(a.get(k),b.get(k),path+'/'+k)
        elif isinstance(a,list) and isinstance(b,list):
            try:
                x=np.asarray(a);y=np.asarray(b)
                if x.shape==y.shape and x.dtype.kind in 'fiub' and y.dtype.kind in 'fiub':
                    d=np.abs(x.astype(float)-y.astype(float))
                    diffs.append(dict(path=path,shape=list(x.shape),max_abs=float(d.max()),changed=int((d>0).sum()),
                                      ref_range=[float(x.min()),float(x.max())]));return
            except (ValueError,TypeError):pass
            if len(a)==len(b):
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
            else:diffs.append(dict(path=path,ref=a,actual=b))
        else:diffs.append(dict(path=path,ref=a,actual=b))
    for key in ('identity','values'):compare(ref[key],actual[key],key)
    save('EXPAND_PARITY_DIAGNOSTIC.json',dict(status='EXACT_FAILED',differences=diffs))
    print(json.dumps(diffs,indent=1))


def environment():
    result={}
    for host in ('gpuA','gpuR'):
        result[host]=json.loads(ssh(host,'''import sys,os,json
sys.path.insert(0,'/root/autodl-tmp/c4/speedhead20261030/pythondeps')
import torch,numpy
print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_path=torch.__file__,numpy=numpy.__version__,numpy_path=numpy.__file__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),config=torch.__config__.show())))
'''))
    save('EXPAND_ENVIRONMENT.json',result);print(json.dumps(result,indent=1))


def repeat_parity():
    for host in ('gpuA','gpuR'):
        receipt=json.loads(ssh(host,f'''from pathlib import Path
import os,json,subprocess,time
p=Path({REMOTE!r});dest=p/'parity_repeat';dest.mkdir(exist_ok=True)
assert not (dest/'wm2_common_w0_k0_h300.json').exists()
assert not (p/'evidence/EXPAND_REPEAT_OWNED.json').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=5120
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/'worker.py'),'--model','wm2','--world','0','--future','0','--plant','common','--seconds','300','--out','parity_repeat','--device','cuda']
with (p/'evidence/EXPAND_REPEAT.log').open('x') as log:
 child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
  env={{**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
 r=dict(pid=child.pid,start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split(') ',1)[1].split()[19],command=cmd,started=time.time(),role='same-host repeatability diagnostic; never counted as panel task')
 with (p/'evidence/EXPAND_REPEAT_OWNED.json').open('x') as f:json.dump(r,f)
print(json.dumps(r))
'''))
        save(f'EXPAND_REPEAT_OWNED_{host}.json',receipt);print(host,json.dumps(receipt))


def compare_repeats():
    import numpy as np
    runs={}
    for host,folder,tag in (('gpuA','panel','A0'),('gpuA','parity_repeat','A1'),('gpuR','parity','R0'),('gpuR','parity_repeat','R1')):
        local=HERE/'out'/f'expand_{tag}.json'
        print(scp(f'{host}:{REMOTE}/{folder}/wm2_common_w0_k0_h300.json',local))
        runs[tag]=json.loads(local.read_text())
    result=[]
    for h in ('10','30','60','90','120','180','300'):
        row=dict(horizon=int(h))
        for a,b,label in (('A0','A1','same_A'),('R0','R1','same_R'),('A0','R0','cross')):
            av=runs[a]['values'][h];bv=runs[b]['values'][h]
            row[label]=dict(J_max_abs=float(np.max(np.abs(np.array(av['J'])-bv['J']))),
                J_relative_max=float(np.max(np.abs(np.array(av['J'])-bv['J'])/np.maximum(np.abs(av['J']),1))),
                position_max_abs=max(float(np.max(np.abs(np.array(x['s'])-y['s']))) for x,y in zip(av['endpoints'],bv['endpoints'])),
                speed_max_abs=max(float(np.max(np.abs(np.array(x['v'])-y['v']))) for x,y in zip(av['endpoints'],bv['endpoints'])),
                crossings_max_abs=float(np.max(np.abs(np.array(av['crossings'])-bv['crossings']))),
                queue_max_abs=float(np.max(np.abs(np.array(av['merge_queue'])-bv['merge_queue']))))
        result.append(row)
    save('EXPAND_REPEATABILITY.json',dict(registered_exact_parity='FAIL; no panel expansion authorized',rows=result))
    print(json.dumps(result,indent=1))


def numerical_parity():
    import copy
    import numpy as np
    ensemble=[]
    for k in range(8):
        local=HERE/'out'/f'expand_K{k}.json'
        try:value=json.loads(local.read_text())
        except FileNotFoundError:value=None
        except json.JSONDecodeError:
            backup=local.with_name(local.name+'.partial_'+hashlib.sha256(local.read_bytes()).hexdigest()[:12])
            if not backup.exists():backup.write_bytes(local.read_bytes())
            value=None
        if value is None:
            print(scp(f'gpuA:{REMOTE}/panel/wm2_common_w0_k{k}_h300.json',local),flush=True)
            value=json.loads(local.read_text())
        assert (value['world'],value['future'],value['steps'])==(0,k,600)
        ensemble.append(value)
    ref=ensemble[0];actual=json.loads((HERE/'out/expand_parity.json').read_text())
    a=copy.deepcopy(actual);r=copy.deepcopy(ref)
    # The sole metadata exception is a hash-verified symlink resolution alias.
    alias='/root/autodl-tmp/c4/longmem300/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'
    canonical='/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'
    assert a['identity']['wm_receipt'][0]['config']==alias
    assert r['identity']['wm_receipt'][0]['config']==canonical
    a['identity']['wm_receipt'][0]['config']=canonical
    for x in (a,r):
        for key in ('values','pid','runtime_seconds'):x.pop(key)
    assert a==r,'unapproved non-numeric task identity difference'
    checks=[]
    def check(name,error,bound):
        e=np.asarray(error,dtype=float);b=np.asarray(bound,dtype=float)
        checks.append(dict(name=name,passed=bool(np.all(np.isfinite(e)) and np.all(e<=b)),
                           error=e.tolist(),bound=b.tolist()))
    scalars=('source','upstream','merge','discharge','J','crossings','merge_queue','mean_merge_queue')
    for h in ('10','30','60','90','120','180','300'):
        x=ref['values'][h];y=actual['values'][h];early=int(h)<=60
        for key in scalars:
            error=np.abs(np.array(x[key])-y[key])
            if early:
                bound=0. if key in ('crossings','merge_queue','mean_merge_queue') else .001
            else:
                spread=np.ptp(np.array([z['values'][h][key] for z in ensemble]),axis=0)
                tol=0. if key in ('crossings','merge_queue') else .0001 if key=='mean_merge_queue' else .001
                bound=spread+tol
            check(h+'/'+key,error,bound)
        for key in ('speed','density','occupied'):
            error=np.abs(np.array(x['fields'][key],dtype=float)-np.array(y['fields'][key],dtype=float))
            if early:
                check(h+'/field_'+key,error.max(axis=(1,2)),.0001 if key=='speed' else 0.)
            elif key!='occupied':
                spread=np.ptp(np.array([z['values'][h]['fields'][key] for z in ensemble]),axis=0)
                check(h+'/field_'+key,error.mean(axis=(1,2)),spread.mean(axis=(1,2))+.0001)
        for i,(ex,ey) in enumerate(zip(x['endpoints'],y['endpoints'])):
            assert ex['ids']==ey['ids']
            if early:check(f'{h}/arm{i}/present',np.array(ex['present'])!=ey['present'],0.)
            present=np.ones(len(ex['ids']),dtype=bool) if early else np.array(ex['present'],dtype=bool)&np.array(ey['present'],dtype=bool)
            for key,tol in (('s',.001),('v',.0001)):
                error=np.abs(np.array(ex[key])-ey[key])[present]
                if early:bound=np.full(error.shape,tol)
                else:
                    spread=np.ptp(np.array([z['values'][h]['endpoints'][i][key] for z in ensemble]),axis=0)
                    bound=spread[present]+tol
                check(f'{h}/arm{i}/{key}',error,bound)
    passed=all(x['passed'] for x in checks)
    receipt=dict(status='PASS' if passed else 'FAIL',exact_parity='FAIL_PRESERVED',
        design_commit='51ba73ca',model_sha256=actual['identity']['model_sha256'],worker_sha256=actual['identity']['source_sha256'],
        reference_task='wm2_common_w0_k0_h300',ensemble='same-world fixed K0..K7; no recorded SUMO labels used',
        checks=checks,failures=[x['name'] for x in checks if not x['passed']],
        reference_output_sha256=hashlib.sha256((HERE/'out/expand_parity.json').read_bytes()).hexdigest())
    save('EXPAND_NUMERICAL_PARITY.json',receipt)
    print(json.dumps(dict(status=receipt['status'],checks=len(checks),failures=[x for x in checks if not x['passed']]),indent=1))
    assert passed,'prospective numerical/ensemble gate failed; no expansion'


def stage_legacy():
    old=Path('E:/research-dense/runs/active/WM-S-C1FAST-20261006/box')
    from analyze import MODEL_HASHES as HASHES
    for tag,folder in (('bbtraj','full_bb_traj'),('bbid','full_bb_id'),('historical','full_struct')):
        print(ssh('gpuR',f'''from pathlib import Path
p=Path({REMOTE!r})/'ckpt'/{tag!r};assert p.parent.is_dir();p.mkdir(exist_ok=True)
print(p)
'''),flush=True)
        for name in ('wm_s.pt','TRAIN_RECEIPT.json'):
            local=allowed(old/folder/name);digest=hashlib.sha256(local.read_bytes()).hexdigest()
            if name=='wm_s.pt':assert digest==HASHES[tag]
            remote=REMOTE+'/ckpt/'+tag+'/'+name
            actual=ssh('gpuR',f'from pathlib import Path;import hashlib;p=Path({remote!r});print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "ABSENT")').strip()
            if actual==digest:continue
            assert actual=='ABSENT','never overwrite an existing checkpoint'
            print(scp(local,f'gpuR:{remote}'),flush=True)
            assert ssh('gpuR',f'from pathlib import Path;import hashlib;print(hashlib.sha256(Path({remote!r}).read_bytes()).hexdigest())').strip()==digest
    save('EXPAND_LEGACY_STAGED.json',dict(status='PASS',models=('bbtraj','bbid','historical'),source=str(old),training=False))


def legacy_canary():
    import numpy as np
    checks=[]
    for model in ('bbtraj','bbid','historical'):
        print(ssh('gpuR',f'''from pathlib import Path
import os,subprocess,json,time
p=Path({REMOTE!r});out=p/'expansion_smoke';out.mkdir(exist_ok=True)
assert not (out/{(model+'_common_w0_k0_h10.json')!r}).exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=5120
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/'worker.py'),'--model',{model!r},'--world','0','--seconds','10','--out','expansion_smoke','--device','cuda']
with (p/'evidence'/{('EXPAND_SMOKE_'+model+'.log')!r}).open('x') as log:
 child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={{**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
 r=dict(pid=child.pid,start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split(') ',1)[1].split()[19],command=cmd,started=time.time())
 with (p/'evidence'/{('EXPAND_SMOKE_'+model+'_OWNED.json')!r}).open('x') as f:json.dump(r,f)
 code=child.wait(timeout=70)
 assert code==0,code
print(json.dumps(dict(model={model!r},returncode=code)))
'''),flush=True)
        name=model+'_common_w0_k0_h10.json'
        ref=json.loads(ssh('gpuA',f'from pathlib import Path;print(Path({(REMOTE+"/cpu_verified/"+name)!r}).read_text())'))
        actual=json.loads(ssh('gpuR',f'from pathlib import Path;print(Path({(REMOTE+"/expansion_smoke/"+name)!r}).read_text())'))
        assert ref['identity']['model_sha256']==actual['identity']['model_sha256']
        x=ref['values']['10'];y=actual['values']['10']
        errors={}
        for key in ('source','upstream','merge','discharge','J'):
            errors[key]=float(np.max(np.abs(np.array(x[key])-y[key])));assert errors[key]<=.001,(model,key,errors[key])
        for key in ('crossings','merge_queue','mean_merge_queue'):assert x[key]==y[key],(model,key)
        for key in ('density','occupied'):assert x['fields'][key]==y['fields'][key]
        errors['field_speed']=float(np.max(np.abs(np.array(x['fields']['speed'])-y['fields']['speed'])));assert errors['field_speed']<=.0001
        for key,tol in (('s',.001),('v',.0001)):
            errors[key]=max(float(np.max(np.abs(np.array(a[key])-b[key]))) for a,b in zip(x['endpoints'],y['endpoints']))
            assert errors[key]<=tol,(model,key,errors[key])
        assert all(a['ids']==b['ids'] and a['present']==b['present'] for a,b in zip(x['endpoints'],y['endpoints']))
        checks.append(dict(model=model,passed=True,max_abs=errors))
    save('EXPAND_LEGACY_CUDA_VERIFIED.json',dict(status='PASS',checks=checks,seconds=10,source='original CPU_VERIFIED paths',training=False))
    print(json.dumps(checks,indent=1))


def freeze():
    assert json.loads((HERE/'evidence/EXPAND_NUMERICAL_PARITY.json').read_text())['status']=='PASS'
    code=f'''from pathlib import Path
import os,json,signal,time
p=Path({REMOTE!r});d=json.loads((p/'DISPATCH.json').read_text());pid=d['pid']
def stat(pid):
 try:
  s=Path(f'/proc/{{pid}}/stat').read_text().split(') ',1)[1].split();return dict(state=s[0],ticks=s[19])
 except FileNotFoundError:return None
assert not (p/'COMPLETE.json').exists()
s=stat(pid);assert s and s['ticks']==d['start_ticks']
assert b'/dispatch.py' in Path(f'/proc/{{pid}}/cmdline').read_bytes()
receipt=p/'evidence/EXPAND_HANDOFF.json'
assert not receipt.exists(), 'existing handoff must be inspected'
os.kill(pid,signal.SIGSTOP)
for i in range(100):
 s=stat(pid)
 if s['state']=='T':break
 time.sleep(.05)
assert s['state']=='T' and s['ticks']==d['start_ticks']
children=[]
for q in (p/'evidence').glob('*_OWNED.json'):
 r=json.loads(q.read_text());s=stat(r['pid'])
 if s and s['ticks']==r['start_ticks'] and s['state']!='Z':children.append(r)
# Freeze may land between Popen and its receipt write. Account for every direct
# child of every stopped scheduler thread as well, so no unreceipted task races.
known={{r['pid'] for r in children}}
for q in Path(f'/proc/{{pid}}/task').glob('*/children'):
 for child in map(int,q.read_text().split()):
  if child in known:continue
  s=stat(child)
  if not s or s['state']=='Z':continue
  cmd=Path(f'/proc/{{child}}/cmdline').read_bytes().decode().split('\\0')
  children.append(dict(pid=child,start_ticks=s['ticks'],command=cmd,ownership='direct child of exact stopped owned scheduler',started=None))
  known.add(child)
r=dict(original_dispatch=d,paused_state=stat(pid),draining_children=children,at=time.time(),
 reason='disjoint cross-host scheduler handoff; original worker code and single collector unchanged')
with receipt.open('x') as f:json.dump(r,f,indent=1)
print(json.dumps(r))
'''
    r=json.loads(ssh('gpuA',code));save('EXPAND_HANDOFF.json',r);print(json.dumps(r))


def assign():
    assert json.loads((HERE/'evidence/EXPAND_LEGACY_CUDA_VERIFIED.json').read_text())['status']=='PASS'
    handoff=json.loads((HERE/'evidence/EXPAND_HANDOFF.json').read_text())
    snapshot=json.loads(ssh('gpuA',f'''from pathlib import Path
import json,hashlib
p=Path({REMOTE!r});h=json.loads((p/'evidence/EXPAND_HANDOFF.json').read_text())
def stat(pid):
 try:
  s=Path(f'/proc/{{pid}}/stat').read_text().split(') ',1)[1].split();return dict(state=s[0],ticks=s[19])
 except FileNotFoundError:return None
d=h['original_dispatch'];s=stat(d['pid']);assert s==dict(state='T',ticks=d['start_ticks'])
live=[]
for r in h['draining_children']:
 s=stat(r['pid'])
 if s and s['ticks']==r['start_ticks'] and s['state']!='Z':live.append(r)
done=[];worker=hashlib.sha256((p/'worker.py').read_bytes()).hexdigest()
for q in (p/'panel').glob('*_h300.json'):
 try:x=json.loads(q.read_text())
 except json.JSONDecodeError:continue
 assert x['identity']['source_sha256']==worker and x['steps']==600
 done.append([x['model'],x['world'],x['future'],x['plant']])
print(json.dumps(dict(live=live,done=done,worker_sha256=worker)))
'''))
    assert not any(r.get('device')=='cuda' for r in snapshot['live']), 'original CUDA children must drain before GPU lane replacement'
    roster=[]
    for world in range(40):
        for model in ('wm2','bbtraj','bbid','historical'):
            roster.extend((model,world,k,'common') for k in range(8 if model=='wm2' else 1))
        roster.append(('wm2',world,0,'production'))
    done={tuple(t) for t in snapshot['done']}
    draining=set()
    for r in snapshot['live']:
        cmd=r['command']
        assert str(REMOTE+'/worker.py') in cmd,'inspect an unrecognized direct child before assignment'
        def arg(name):return cmd[cmd.index(name)+1]
        task=(arg('--model'),int(arg('--world')),int(arg('--future')),arg('--plant'))
        assert task not in done
        draining.add(task)
    assert len(done)==len(snapshot['done']) and done <= set(roster)
    assert draining<=set(roster)
    gpu=[t for t in roster if t not in done|draining and t[0]=='wm2']
    legacy=[t for t in roster if t not in done|draining and t[0]!='wm2']
    assignments={}
    for host in ('gpuA','gpuR'):
        assignments[host]=dict(host=host,worker_sha256=snapshot['worker_sha256'],
            gpu_lanes=2 if host=='gpuA' else 4,budget_bytes=12_000_000_000 if host=='gpuA' else 8_000_000_000,
            gpu_tasks=([t for i,t in enumerate(gpu) if i%9==0] if host=='gpuA' else
                       legacy+[t for i,t in enumerate(gpu) if i%9!=0]),cpu_tasks=[],
            preserved_completed=len(done),K=8,parity='EXPAND_NUMERICAL_PARITY.json PASS; exact host parity failed and retained')
    assigned=[tuple(t) for a in assignments.values() for t in a['gpu_tasks']+a['cpu_tasks']]
    assert len(assigned)==len(set(assigned)) and set(assigned).isdisjoint(done|draining) and set(assigned)|done|draining==set(roster)
    save('EXPAND_ASSIGNMENTS.json',dict(assignments=assignments,done=sorted(done),draining_reserved=sorted(draining),handoff=handoff,task_count=len(roster)))
    for host,a in assignments.items():
        local=HERE/'out'/f'expand_assignment_{host}.json'
        with local.open('x') as f:json.dump(a,f,indent=1)
        print(scp(local,f'{host}:{REMOTE}/EXPAND_ASSIGNMENT.json'))
    print(json.dumps({h:dict(gpu=len(a['gpu_tasks']),cpu=len(a['cpu_tasks']),lanes=a['gpu_lanes']) for h,a in assignments.items()}))


def launch_hosts():
    a=json.loads((HERE/'evidence/EXPAND_ASSIGNMENTS.json').read_text())
    for host in ('gpuA','gpuR'):
        receipt=json.loads(ssh(host,f'''from pathlib import Path
import os,json,subprocess,time
p=Path({REMOTE!r});assert (p/'EXPAND_ASSIGNMENT.json').is_file()
assert not (p/'evidence/EXPAND_LAUNCH.json').exists()
assert not (p/'EXPAND_DISPATCH.json').exists()
assert json.loads((p/'EXPAND_ASSIGNMENT.json').read_text())=={a['assignments'][host]!r}
with (p/'evidence/EXPAND_DISPATCH.log').open('x') as log:
 child=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/'expand_runner.py')],
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
  env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
 r=dict(pid=child.pid,start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split(') ',1)[1].split()[19],started=time.time())
 with (p/'evidence/EXPAND_LAUNCH.json').open('x') as f:json.dump(r,f)
print(json.dumps(r))
'''))
        save(f'EXPAND_LAUNCH_{host}.json',receipt);print(host,json.dumps(receipt))


def host_state(host):
    return json.loads(ssh(host,f'''from pathlib import Path
import json,os,subprocess,hashlib
p=Path({REMOTE!r});launch=p/'evidence/EXPAND_LAUNCH.json'
r=json.loads(launch.read_text()) if launch.exists() else None
live=False
if r:
 q=Path('/proc/'+str(r['pid'])+'/stat')
 if q.exists():
  s=q.read_text().split(') ',1)[1].split();live=s[19]==r['start_ticks'] and s[0]!='Z'
q=p/'EXPAND_COMPLETE.json'
done=json.loads(q.read_text()) if q.exists() else None
names={{}}
for q in (p/'panel').glob('*_h300.json'):
 if {host!r}=='gpuR':
  attempt=p/'evidence'/('EXPAND_'+q.stem+'_attempt1.json')
  if not attempt.exists():continue
  if json.loads(attempt.read_text())['returncode']!=0:continue
 try:x=json.loads(q.read_text())
 except json.JSONDecodeError:continue
 assert x['steps']==600
 names[q.name]=hashlib.sha256(q.read_bytes()).hexdigest()
print(json.dumps(dict(alive=live,receipt=r,complete=done,outputs=names,load=os.getloadavg(),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader'],text=True))))
'''))


def publish(states):
    for name,digest in states['gpuR']['outputs'].items():
        assert name in {f'{m}_{p}_w{w}_k{k}_h300.json' for m,w,k,p in
                       json.loads((HERE/'evidence/EXPAND_ASSIGNMENTS.json').read_text())['assignments']['gpuR']['gpu_tasks']}
        if name in states['gpuA']['outputs']:
            assert states['gpuA']['outputs'][name]==digest
            continue
        local=HERE/'out'/('expand_'+name)
        print(scp(f'gpuR:{REMOTE}/panel/{name}',local),flush=True)
        assert hashlib.sha256(local.read_bytes()).hexdigest()==digest
        partial=REMOTE+'/panel/'+name+'.gpuR_publish'
        print(scp(local,f'gpuA:{partial}'),flush=True)
        print(ssh('gpuA',f'''from pathlib import Path
import hashlib,os
q=Path({partial!r});dest=q.with_name({name!r});assert hashlib.sha256(q.read_bytes()).hexdigest()=={digest!r}
assert not dest.exists();os.link(q,dest)
print('published verified gpuR',dest.name)
'''),flush=True)


def bridge():
    import msvcrt
    lock=(HERE/'evidence/EXPAND_BRIDGE.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        try:
            states={h:host_state(h) for h in ('gpuA','gpuR')}
            publish(states)
            summary={h:dict(alive=s['alive'],completed=len(s['outputs']),gpu=s['gpu'],load=s['load'],
                           success=s['complete']['success'] if s['complete'] else None) for h,s in states.items()}
            with (HERE/'evidence/EXPAND_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),hosts=summary))+'\n')
            print(json.dumps(summary),flush=True)
            if any(not s['alive'] and s['complete'] is None for s in states.values()):
                save('EXPAND_BLOCKED.json',dict(reason='expansion dispatcher absent without completion',hosts=summary));return
            if all(s['complete'] is not None for s in states.values()):
                assert all(s['complete']['success'] for s in states.values()),'failed task: inspect, no automatic retry'
                finish();return
        except (RuntimeError,subprocess.TimeoutExpired,OSError) as exc:
            print('WAIT',repr(exc),flush=True)
        time.sleep(60)


def finish():
    print(ssh('gpuA',f'''from pathlib import Path
import json,hashlib,os,signal,time
p=Path({REMOTE!r});a=json.loads((p/'EXPAND_ASSIGNMENT.json').read_text());worker=a['worker_sha256']
results=[]
for w in range(40):
 tasks=[(m,k,'common') for m in ('wm2','bbtraj','bbid','historical') for k in range(8 if m=='wm2' else 1)]+[('wm2',0,'production')]
 for m,k,plant in tasks:
  name=f'{{m}}_{{plant}}_w{{w}}_k{{k}}_h300';q=p/'panel'/(name+'.json');x=json.loads(q.read_text())
  assert (x['model'],x['world'],x['future'],x['plant'],x['steps'])==(m,w,k,plant,600)
  assert x['identity']['source_sha256']==worker
  results.append(dict(task=name,status='complete'))
assert len(results)==480 and len(list((p/'panel').glob('*_h300.json')))==480
assert not (p/'COMPLETE.json').exists()
with (p/'COMPLETE.json').open('x') as f:
 json.dump(dict(success=True,count=480,results=results,finished=time.time(),
  scheduler='disjoint gpuA2+gpuR4 expansion; prior exact-owned scheduler paused and drained; registered numerical/ensemble parity PASS, exact parity FAIL retained'),f)
 f.flush();os.fsync(f.fileno())
d=json.loads((p/'DISPATCH.json').read_text());q=Path('/proc/'+str(d['pid'])+'/stat')
if q.exists():
 s=q.read_text().split(') ',1)[1].split()
 assert s[19]==d['start_ticks'] and s[0]=='T'
 # Only the original, receipt-owned coordinator. All its children have drained.
 os.kill(d['pid'],signal.SIGTERM);os.kill(d['pid'],signal.SIGCONT)
print('COMPLETE 480/480; existing single collector will finalize')
'''),flush=True)
    save('EXPAND_FINISHED.json',dict(at=time.time(),status='480/480 complete; existing collector handles finalization'))


def launch_bridge():
    from watch import creation
    assert not (HERE/'evidence/EXPAND_BRIDGE_LAUNCH.json').exists()
    with (HERE/'evidence/EXPAND_BRIDGE.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'bridge'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
    identity=creation(child.pid);assert identity is not None
    r=dict(pid=child.pid,creation_time=identity,at=time.time(),role='scheduler publication only; existing watch remains sole analysis collector')
    save('EXPAND_BRIDGE_LAUNCH.json',r);print(json.dumps(r))


def checkpoint():
    from watch import creation
    result=dict(at=time.time(),hosts={})
    for host in ('gpuA','gpuR'):
        state=host_state(host)
        usage=json.loads(ssh(host,f'''import sys,json
from pathlib import Path
sys.path.insert(0,{REMOTE!r});import expand_runner as r
free,owned=r.memory();p=Path({REMOTE!r})
live=[]
for q in (p/'evidence').glob('EXPAND_*_OWNED.json'):
 x=json.loads(q.read_text())
 if r.incarnation(x['pid'])==x['start_ticks']:
  s=Path('/proc/'+str(x['pid'])+'/stat').read_text().split(') ',1)[1].split()
  if s[0]!='Z':live.append(dict(pid=x['pid'],ticks=x['start_ticks'],task=x.get('task'),device=x.get('device')))
print(json.dumps(dict(free_mib=free,owned_mib=owned,live=live)))
'''))
        result['hosts'][host]=dict(alive=state['alive'],completed=len(state['outputs']),load=state['load'],
                                  gpu=state['gpu'],usage=usage,receipt=state['receipt'])
    watch=json.loads((HERE/'evidence/WATCH_LAUNCH.json').read_text())
    result['collector_alive']=creation(watch['pid'])==watch['creation_time']
    result['collector_sources_unchanged']=all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==value for name,value in watch['source_hashes'].items())
    b=json.loads((HERE/'evidence/EXPAND_BRIDGE_LAUNCH.json').read_text())
    result['bridge_alive']=creation(b['pid'])==b['creation_time']
    result['expansion_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert result['collector_alive'] and result['collector_sources_unchanged'] and result['bridge_alive']
    for host in ('gpuA','gpuR'):
        assert result['hosts'][host]['alive']
        assert result['hosts'][host]['usage']['owned_mib']*2**20<=(12_000_000_000 if host=='gpuA' else 8_000_000_000)
        assert result['hosts'][host]['usage']['free_mib']>=1024
    save('EXPAND_RUNNING_'+str(int(time.time()))+'.json',result);print(json.dumps(result,indent=1))


def main():
    actions={'audit':audit,'stage':stage,'parity':parity,'repair-configs':repair_configs,'verify-parity':verify_parity,'diagnose-parity':diagnose_parity,'environment':environment,'repeat-parity':repeat_parity,'compare-repeats':compare_repeats,'numerical-parity':numerical_parity,'stage-legacy':stage_legacy,'legacy-canary':legacy_canary,'freeze':freeze,
             'assign':assign,'launch-hosts':launch_hosts,'bridge':bridge,'launch-bridge':launch_bridge,
             'checkpoint':checkpoint,'status':lambda:print(json.dumps({h:host_state(h) for h in ('gpuA','gpuR')}))}
    p=argparse.ArgumentParser();p.add_argument('action',choices=actions)
    args=p.parse_args()
    actions[args.action]()


if __name__=='__main__':main()
