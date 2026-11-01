"""Explicit immutable staging, read-only evidence intake, and one-use dispatch."""
import argparse
import json
from pathlib import Path

from tools.coopbandit.protocol import digest, sha

LOCAL = Path(__file__).resolve().parents[2]
OUT = LOCAL/'docs/paper_results/wmplan'
ROOT = '/root/autodl-tmp/wmplan20261101_v1'
BASE = '/root/autodl-tmp/distill20261101_v1'
QUALITY = '/root/autodl-tmp/wmteach_v5quality20261101_v2'
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def remote(source, timeout=60, host=None):
    compile(source,'<wmplan-remote-preflight>','exec')
    from tools.wmteach.ops import remote as call
    return call(source,timeout=timeout,host=host)


def recover():
    """Read named receipts only; never attach to or change peer processes."""
    result = {}
    for host in ('gpuA','gpuR'):
        result[host] = json.loads(remote(f'''import pathlib,json,os,shutil,subprocess,time
root=pathlib.Path({ROOT!r});q=pathlib.Path({QUALITY!r})
names=('BINDING.json','DISPATCH.json','LAUNCH_ONCE.json','COMPLETE.json','STOP.json')
print(json.dumps(dict(unix=time.time(),root_exists=root.exists(),
 receipts={{n:json.loads((root/n).read_text()) for n in names if (root/n).is_file()}},
 quality_futures=len(list((q/'tasks').glob('*/k*.json'))),
 quality_complete=(q/'QUALITY.json').is_file(),load=os.getloadavg(),
 disk=shutil.disk_usage('/root/autodl-tmp').free,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 runtime_paths={{p:pathlib.Path(p).exists() for p in (
 '/root/rbeval20261031/phys_code_22cc32f65a80',
 '/root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f',
 '/root/wmcalib/data','/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt',
 '/root/autodl-tmp/wmteach20261101_v1/deps')}})))
''',host=host))
    save('RECOVERY_1448.json',dict(hosts=result,old_local_continuation=dict(
        pid=21412,creation='2026-11-01T14:03:36',command='tools.wmplan.watch',
        action='identity-checked stop before launch-policy amendment'),new_sumo_sim_s=0))
    print(json.dumps(result))


def save(name, value):
    assert OUT.parent.is_dir()
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n')


def inspect():
    result = json.loads(remote(f'''import pathlib,json,os,shutil,time,subprocess,hashlib
base=pathlib.Path({BASE!r});q=pathlib.Path({QUALITY!r})
def load(p):return json.loads(p.read_text())
panel=load(base/'PANEL.json');binding=load(base/'BINDING.json')
assert (base/'COMPLETE.json').is_file() and not (base/'STOP.json').exists()
hashes={{}};baselines=[]
for w in panel:
 p=base/'jobs'/f's{{w["seed"]}}_mappo'/'RESULT.json'
 data=load(p);assert data['world']==w and data['actual_sim_s']==1200
 hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();baselines.append(data)
files={{n:load(q/n) for n in ('IDENTITY.json','DISPATCH.json','QUALITY.json','QUALITY_RECORDS.json','QUALITY_COMPLETE.json','FAILURE.json') if (q/n).is_file()}}
print(json.dumps(dict(unix=time.time(),load=os.getloadavg(),disk=shutil.disk_usage(base).free,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 baseline_hashes=hashes,baselines=baselines,baseline_binding=binding,panel=panel,quality=files,
 quality_futures=len(list((q/'tasks').glob('*/k*.json'))),new_sumo_sim_s=0)))
''', timeout=90))
    assert digest(result['panel']) == '72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6'
    save(f'INTAKE_{int(result["unix"])}.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in
                     ('baseline_hashes','baselines','baseline_binding','quality','panel')}))
    print(json.dumps(dict(quality_complete='QUALITY_COMPLETE.json' in result['quality'],
                         quality=result['quality'].get('QUALITY.json'))))


def register():
    from tools.wmplan.design import budget
    design = sha(LOCAL/'docs/paper_results/WMPLAN_DESIGN.md')
    text = REG.read_text(encoding='utf-8')
    owner = next(s for s in text.splitlines() if 'S-WMPLAN BUDGET APPROVED' in s)
    line = (f'- S-WMPLAN EXACT PLAN REGISTERED BEFORE LIVE DISPATCH: designSHA256{design}; '
            'owner13:45/c4-wmplan; reuse S-DISTILL DEV984000..984023/panelSHA72776ff1 '
            'and ALL24 byte-bound completed MAPPO baselines/newMAPPO0/newmechanics0. '
            'Primary expected-positive-mean/secondary inherited strictP>=0.8, K8paired '
            'complete futures/H300/cadence60/t300..960/O0..O4/executors/safety unchanged. '
            'EXACT199200upper=57600(48x1200planner)+141600(48x2950disjointcredit), '
            '800unallocated under200000cap. All executed groups reserve COMPLETE '
            'full-prefix H120 O0/joint/BOTHsolo quartet4*(t+120) before execution; '
            'otherwise explicit resource-O0, no cross-world/arm transfer/refund/retry. '
            'IMPORTANT budget-limited estimand: <=1group/world-arm and none aftert600; '
            'report proposals/executions/refusals. Reuse exact S-WMTEACH roster-only '
            'FIT-population/sourceprior/coldmemory code path; no richer-conditioning claim. '
            'Model rule BEFORE outcomes: FULL-v5 b2df7458 iff SAME12 complete H300 '
            'helpful argmax agreement AND all12 exact chosen saving>=v4 22cc32f6; '
            'reuse peer quality/no duplicate; final comparison and model/source/baseline '
            'freeze+hostCPU24dualguards before physics. GPUA immutable '
            '/root/autodl-tmp/wmplan20261101_v1, <=2CPUepisode+2creditworkers/2GPU '
            'services strict8GiB/nice19/singlethread/load20/RAM8GiB/disk2GiB; noGPUroutputs. '
            'All24 pairedJ600/J900/10000whole-worldbootstrap20261101/safety/moves/ALLCG; '
            'matched S-DISTILL/availableWMstudent, oracle different-panel descriptive only. '
            'No efficacy earlystop/retuning/peerPIDaction/prohibitedreads/deletion/newenv/push.\n')
    if line not in text:
        with REG.open('a',encoding='utf-8') as f:
            f.write(line);f.flush()
    save('REGISTRATION.json',dict(line=line,owner=owner,design_sha256=design,budget=budget()))
    print(json.dumps(dict(design_sha256=design,budget=budget(),new_sumo_sim_s=0)))


def amend():
    """Owner14:48 scheduling/host amendment, before all live outcomes."""
    from tools.wmplan.design import budget
    old = json.loads((OUT/'REGISTRATION.json').read_text())
    design = sha(LOCAL/'docs/paper_results/WMPLAN_DESIGN.md')
    line = (f'- S-WMPLAN OWNER14:48 SPEED AMENDMENT REGISTERED BEFORE LIVE: '
            f'designSHA256{design}; supersedes host/order clauses only of original{old["design_sha256"]}. '
            'PRIMARYexpected all24 worlds+completeCG+verified paired report saved before '
            'SECONDARYstrict starts; no outcome-dependent continuation. One WM service on gpuA '
            'and one on gpuR via owned persistent SSH relay; same runtime.fit_futures K8/H300 '
            'serial computation, byte-identical source/artifact/FIT32 mirror; exclusive gpuA '
            'claims/results/charges. gpuR outputs /root/wmplan20261101_v1 on root filesystem '
            '(16GiBfree), not its nearly-full autodl-tmp; missing immutable runtime files only '
            'copied after hash/conflict gates, no overwrite/deletion/newenv. Strict8GiB/process, '
            'GPUfree>12GiB and rootdisk>=2GiB startup; GPUA CPU load20/RAM8GiB gates retained. '
            'All exact199200upper/cap200000, model noninferiority rule, same24 reusedMAPPO, '
            'equal2950credit quotas, O0..O4/safety/cadence/bootstrap unchanged. Old owned '
            'local watcher21412 identity-checked stopped; no existing live dispatch or physics.\n')
    with REG.open('a',encoding='utf-8') as f:
        f.write(line);f.flush()
    save('REGISTRATION_1448.json',dict(**{k:v for k,v in old.items() if k not in ('line','design_sha256','budget')},
        line=line,design_sha256=design,original_design_sha256=old['design_sha256'],budget=budget()))
    print(json.dumps(dict(design_sha256=design,budget=budget(),new_sumo_sim_s=0)))


def compare():
    from tools.wmplan.design import model_choice
    result = json.loads(remote(f'''import pathlib,json,hashlib,time
roots={{'v4':pathlib.Path('/root/autodl-tmp/wmteach20261101_v3'),'v5':pathlib.Path({QUALITY!r})}}
out={{}}
for name,root in roots.items():
 if not (root/'QUALITY.json').is_file():
  owner=json.loads((root/'DISPATCH.json').read_text());p=pathlib.Path(f'/proc/{{owner["pid"]}}/stat')
  st=p.read_text().split() if p.is_file() else None
  alive=bool(st and st[21]==owner['start_ticks'] and st[2]!='Z' and
   pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'])
  assert alive,'peer quality ended incomplete; no duplicate, fallback, or process action'
  print(json.dumps(dict(complete=False,waiting_for=name,futures=len(list((root/'tasks').glob('*/k*.json'))),owner_alive=alive)));break
 paths=[root/'QUALITY.json',root/'QUALITY_RECORDS.json',root/'IDENTITY.json']
 assert len(list((root/'tasks').glob('*/k*.json')))==96
 out[name]=dict(records=json.loads(paths[1].read_text()),identity=json.loads(paths[2].read_text()),
  hashes={{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
else:print(json.dumps(dict(complete=True,models=out,unix=time.time())))
''',timeout=90))
    if not result['complete']:
        print(json.dumps(result))
        return
    choice = model_choice(result['models']['v4']['records'],result['models']['v5']['records'])
    save('MODEL_COMPARISON.json',dict(**result,**choice,new_sumo_sim_s=0,new_forecasts=0))
    print(json.dumps(choice))


def stage():
    from tools.wmplan.design import panel, budget
    comparison = json.loads((OUT/'MODEL_COMPARISON.json').read_text())
    registration = json.loads((OUT/'REGISTRATION_1448.json').read_text())
    assert registration['design_sha256'] == sha(LOCAL/'docs/paper_results/WMPLAN_DESIGN.md')
    model = comparison['selected']
    quality_root = QUALITY if model == 'FULL-v5' else '/root/autodl-tmp/wmteach20261101_v3'
    modules = {p.name:p.read_text(encoding='utf-8') for p in (LOCAL/'tools/wmplan').glob('*.py')}
    quality_helpers = {name:(LOCAL/'tools/wmteach'/name).read_text(encoding='utf-8')
                       for name in ('__init__.py','labels.py')}
    design = (LOCAL/'docs/paper_results/WMPLAN_DESIGN.md').read_text(encoding='utf-8')
    result = json.loads(remote(f'''import pathlib,json,hashlib,shutil
root=pathlib.Path({ROOT!r});base=pathlib.Path({BASE!r});quality=pathlib.Path({quality_root!r})
assert root.parent.is_dir() and not root.exists()
old=json.loads((base/'BINDING.json').read_text());identity=json.loads((quality/'IDENTITY.json').read_text())
files={{}}
for name,expected in old['sources'].items():
 p=base/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 files[name]=p.read_text()
external={{}}
for name,expected in identity['sources'].items():
 p=pathlib.Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
 if p.is_relative_to(quality):
  rel=p.relative_to(quality)
  if p.suffix=='.py':files['pywm/'+str(rel)]=p.read_text()
 else:external[name]=expected
for name in ('storage.py','protocol.py'):
 files['pywm/tools/coopplan/'+name]=files['pytools/tools/coopplan/'+name]
for name,text in json.loads({json.dumps(quality_helpers)!r}).items():
 files['pytools/tools/wmteach/'+name]=text
modules=json.loads({json.dumps(modules)!r})
for tree in ('pytools','pywm'):
 for name,text in modules.items():files[f'{{tree}}/tools/wmplan/{{name}}']=text
files['WMPLAN_DESIGN.md']={design!r}
files['MODEL_COMPARISON.json']={json.dumps(comparison)!r}
files['REGISTRATION.json']={json.dumps(registration)!r}
files['PANEL.json']=json.dumps({panel()!r},sort_keys=True)
files['APPROVAL.json']=json.dumps(dict(line={registration['owner']!r}))
baseline_hashes={{}}
for world in {panel()!r}:
 p=base/'jobs'/f's{{world["seed"]}}_mappo'/'RESULT.json'
 baseline_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
files['REUSED_BASELINES.json']=json.dumps(dict(hashes=baseline_hashes,
 mechanics=json.loads((base/'MECHANICS.json').read_text()),baseline_root=str(base)))
root.mkdir()
for name,text in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(text)
sources={{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files if n.endswith('.py')}}
artifacts={{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files if not n.endswith('.py')}}
binding=dict(runtime=old['runtime'],checkpoint=old['checkpoint'],deps=old['deps'],workers=2,
 load_max=20,budget={budget()!r},panel={panel()!r},baseline_root=str(base),
 model=dict(selected={model!r},comparison_sha256=artifacts['MODEL_COMPARISON.json']),
 sources=sources,artifacts=artifacts,external_sources={{**external,**baseline_hashes}},
  design_sha256=artifacts['WMPLAN_DESIGN.md'],gpu_workers=2,inference_hosts=['gpuA','gpuR'],
 wm_deps='/root/autodl-tmp/wmteach20261101_v1/deps')
with (root/'BINDING.json').open('x') as f:json.dump(binding,f,sort_keys=True)
for name in ('jobs','states','reservations','requests','responses','forecast_claims','forecasts'):(root/name).mkdir()
print(json.dumps(dict(root=str(root),binding=binding,files=len(files),new_sumo_sim_s=0)))
''',timeout=180))
    save('STAGED.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'binding'}))


def cpu():
    result = json.loads(remote(f'''import pathlib,sys,os,json,unittest
from unittest.mock import patch
root=pathlib.Path({ROOT!r});os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools'),'/root/autodl-tmp/c4/speedhead20261030/pythondeps']
from tools.wmplan.live import setup,verify
from tools.distill.live import guard
from tools.wmplan.design import panel
setup(root)
with patch('subprocess.Popen',side_effect=AssertionError('SUMO blocked during CPU gate')):
 for w in panel():guard(w)
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
  'tools.wmplan.test_cpu','tools.wmplan.test_report','tools.coopplan.test_cpu']))
assert r.wasSuccessful() and not r.skipped
from tools.coopplan.storage import write
receipt=dict(pass_=True,tests=r.testsRun,guards=24,skipped=0,new_sumo_sim_s=0)
write(root/'CPU.json',receipt)
print(json.dumps(receipt))
''',timeout=180))
    save('CPU.json',result)
    print(json.dumps(result))


def freeze():
    comparison = json.loads((OUT/'MODEL_COMPARISON.json').read_text())
    staged = json.loads((OUT/'STAGED.json').read_text())
    cpu_result = json.loads((OUT/'CPU.json').read_text())
    assert cpu_result['pass_']
    mirror = json.loads((OUT/'MIRROR_CPU_GPU_GATE.json').read_text())
    assert mirror['pass_'] and mirror['fit_worlds']==32
    b = staged['binding']
    line = (f'- S-WMPLAN FINAL MODEL/SOURCE FREEZE BEFORE LIVE OUTCOMES: '
            f'model{comparison["selected"]}; v4helpfulagreement'
            f'{comparison["v4_helpful"]["helpful_choice_agreement"]}/'
            f'v5{comparison["v5_helpful"]["helpful_choice_agreement"]}, '
            f'v4all12chosenD{comparison["v4_all"]["exact_chosen_saving"]}/'
            f'v5{comparison["v5_all"]["exact_chosen_saving"]}veh-s; '
            f'comparisonSHA256{sha(OUT/"MODEL_COMPARISON.json")}/'
            f'design{b["design_sha256"]}/sourcebinding{digest(b)}. '
            f'Existing same12 completed teacher forecasts reused/newforecasts0/newSUMO0; '
             f'hostCPU{cpu_result["tests"]}PASS/24dualguards/allbaselinebytesfrozen. '
             f'gpuRbyteidenticalmirrorSHA256{sha(OUT/"MIRROR.json")}/bootstrapPASS32FIT/newforecasts0; '
             f'privatebaseallowlistrepairSHA256{sha(OUT/"MIRROR_BASE_REPAIR.json")}/unchangedloader; '
             f'packageinitializersSHA256{sha(OUT/"MIRROR_PACKAGE_ASSETS.json")}; '
             'expected24+primaryreport before strict24; one8GiBservice each gpuA/gpuR. '
            'Original EXACT199200upper, equal2950credit quotas, resource-limited '
            'estimand and no-new-MAPPO contract retained. One-use gpuA root '
            f'{ROOT}; no model switch or live-outcome selection.\n')
    with REG.open('a',encoding='utf-8') as f:
        f.write(line);f.flush()
    receipt = dict(line=line,binding_sha256=digest(b),comparison_sha256=sha(OUT/'MODEL_COMPARISON.json'),
                   mirror_sha256=sha(OUT/'MIRROR.json'),mirror_gate_sha256=sha(OUT/'MIRROR_CPU_GPU_GATE.json'),
                   mirror_base_repair_sha256=sha(OUT/'MIRROR_BASE_REPAIR.json'),
                   mirror_package_assets_sha256=sha(OUT/'MIRROR_PACKAGE_ASSETS.json'))
    save('FINAL_FREEZE.json',receipt)
    print(remote(f'''import pathlib,json
root=pathlib.Path({ROOT!r})
with (root/'FINAL_FREEZE.json').open('x') as f:json.dump({receipt!r},f)
print(json.dumps(dict(frozen=True,new_sumo_sim_s=0)))
'''))


def launch():
    freeze_receipt = json.loads((OUT/'FINAL_FREEZE.json').read_text())
    assert freeze_receipt['mirror_sha256']==sha(OUT/'MIRROR.json')
    assert freeze_receipt['mirror_gate_sha256']==sha(OUT/'MIRROR_CPU_GPU_GATE.json')
    assert freeze_receipt['mirror_base_repair_sha256']==sha(OUT/'MIRROR_BASE_REPAIR.json')
    assert freeze_receipt['mirror_package_assets_sha256']==sha(OUT/'MIRROR_PACKAGE_ASSETS.json')
    result = json.loads(remote(f'''import pathlib,sys,os,json,subprocess,shutil,time
root=pathlib.Path({ROOT!r});assert root.is_dir()
b=json.loads((root/'BINDING.json').read_text())
assert json.loads((root/'CPU.json').read_text())['pass_']
assert json.loads((root/'FINAL_FREEZE.json').read_text())=={freeze_receipt!r}
assert not (root/'LAUNCH_ONCE.json').exists() and not list((root/'jobs').iterdir())
assert os.getloadavg()[0]<=20 and shutil.disk_usage(root).free>=2*2**30
mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0])*1024>=8*2**30
sys.path.insert(0,b['deps'][0]);import sumo
home=str(pathlib.Path(sumo.__file__).parent)
with (root/'LAUNCH_ONCE.json').open('x') as f:json.dump(dict(parent=os.getpid(),unix=time.time()),f)
common=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
owners=[]
def start(name,args,env):
 log=(root/(name+'.log')).open('x')
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u',*args],cwd=root,env=env,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 owners.append(dict(name=name,pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],
  boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),command=args))
for i in range(1):
 start('GPU'+str(i),['-m','tools.wmplan.predict',str(root),'--worker',str(i)],
  dict(common,PYTHONPATH=str(root/'pywm')+':'+b['wm_deps'],WMTEACH_BACKEND=b['model']['selected']))
start('LIVE',['-m','tools.wmplan.live',str(root)],dict(common,CUDA_VISIBLE_DEVICES='',
 PYTHONPATH=str(root/'pytools')+':'+':'.join(b['deps']),SUMO_HOME=home,PATH=home+'/bin:'+os.environ['PATH']))
receipt=dict(root=str(root),owners=owners,unix=time.time(),new_sumo_sim_s=0)
with (root/'DISPATCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''',timeout=90))
    save('DISPATCH.json',result)
    print(json.dumps(result))
    from tools.wmplan.bridge import launch as launch_bridge
    print(json.dumps(launch_bridge()))


def status(collect=False, primary=False):
    result = json.loads(remote(f'''import pathlib,json,time,hashlib
root=pathlib.Path({ROOT!r})
def load(p):return json.loads(p.read_text())
owners=[]
if (root/'DISPATCH.json').is_file():
 for r in load(root/'DISPATCH.json')['owners']:
  p=pathlib.Path(f'/proc/{{r["pid"]}}/stat');st=p.read_text().split() if p.exists() else None
  owners.append(dict(**r,state=st[2] if st else None,identity_match=bool(st and st[21]==r['start_ticks'] and
   pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id'])))
charges=[json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()] if (root/'CHARGES.jsonl').exists() else []
closed=[];incomplete=[];results={{}}
for job in (root/'jobs').iterdir():
 if (job/'CLOSE.json').is_file():closed.append(dict(id=job.name,**load(job/'CLOSE.json')))
 else:incomplete.append(job.name)
 if (job/'RESULT.json').is_file():results[job.name]=load(job/'RESULT.json')
out=dict(unix=time.time(),root=str(root),complete=(root/'COMPLETE.json').is_file(),
 primary_complete=(root/'ARM_COMPLETE_expected.json').is_file(),
 stopped=(root/'STOP.json').is_file(),owners=owners,charged=sum(r['sim_s'] for r in charges),
 actual=sum(r['actual_sim_s'] for r in closed),incomplete=incomplete,
 episodes=sum('_t' not in k for k in results),forecasts=len(list((root/'responses').glob('*.json'))),
 requests=len(list((root/'requests').glob('*.json'))),
 forecast_futures=sum(len(list(p.glob('k*.json'))) for p in (root/'forecasts').iterdir()),
 stop=load(root/'STOP.json') if (root/'STOP.json').exists() else None,
 logs={{p.name:p.read_text()[-3000:] for p in root.glob('*.log')}})
if {collect!r}:
 assert (out['primary_complete'] if {primary!r} else out['complete']) and not out['stopped'] and not incomplete and out['episodes']=={24 if primary else 48}
 import sys
 sys.path.insert(0,str(root/'pytools'))
 from tools.wmplan.live import verify
 verify(root)
 baseroot=pathlib.Path({BASE!r});panel=load(root/'PANEL.json')
 out.update(binding=load(root/'BINDING.json'),charges=charges,closed=closed,results=results,
  baselines={{str(w['seed']):load(baseroot/'jobs'/f's{{w["seed"]}}_mappo'/'RESULT.json') for w in panel}},
  forecasts={{p.stem:load(p) for p in (root/'responses').glob('*.json')}},
  observations={{p.stem:load(p) for p in (root/'requests').glob('*.json')}},
  reservations={{p.stem:load(p) for p in (root/'reservations').glob('*.json')}},
  result_hashes={{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'jobs').glob('*/RESULT.json')}})
 peer=pathlib.Path('/root/autodl-tmp/wmteach_live20261101_v1')
 out['wm_student']=None
 if (peer/'COMPLETE.json').is_file() and not (peer/'STOP.json').exists():
  pairs=[]
  for w in panel:
   s=w['seed'];pairs.append(dict(world=w,baseline=load(peer/'jobs'/f's{{s}}_mappo'/'RESULT.json'),
    student=load(peer/'jobs'/f's{{s}}_student'/'RESULT.json')))
  out['wm_student']=dict(root=str(peer),pairs=pairs)
print(json.dumps(out))
''',timeout=180))
    if collect:
        save(('PRIMARY_' if primary else '')+'RECEIPTS.json',result)
    else:
        save(f'STATUS_{int(result["unix"])}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in
                     ('binding','charges','closed','results','baselines','forecasts','observations','reservations','result_hashes','wm_student')}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['recover','inspect','register','amend','compare','stage','mirror','cpu','freeze','launch','status','collect','primary'])
    a = p.parse_args().action
    if a == 'mirror':
        from tools.wmplan.mirror import stage as mirror_stage
        mirror_stage()
    elif a == 'primary':
        status(collect=True,primary=True)
    elif a in ('status','collect'):
        status(a == 'collect')
    else:
        globals()[a]()
