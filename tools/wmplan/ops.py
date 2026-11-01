"""Explicit immutable staging, read-only evidence intake, and one-use dispatch."""
import argparse
import json
from pathlib import Path

from tools.coopbandit.protocol import digest, sha
from tools.wmteach.ops import remote

LOCAL = Path(__file__).resolve().parents[2]
OUT = LOCAL/'docs/paper_results/wmplan'
ROOT = '/root/autodl-tmp/wmplan20261101_v1'
BASE = '/root/autodl-tmp/distill20261101_v1'
QUALITY = '/root/autodl-tmp/wmteach_v5quality20261101_v2'
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


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


def compare():
    from tools.wmplan.design import model_choice
    result = json.loads(remote(f'''import pathlib,json,hashlib,time
roots={{'v4':pathlib.Path('/root/autodl-tmp/wmteach20261101_v3'),'v5':pathlib.Path({QUALITY!r})}}
out={{}}
for name,root in roots.items():
 if not (root/'QUALITY.json').is_file():
  print(json.dumps(dict(complete=False,waiting_for=name,futures=len(list((root/'tasks').glob('*/k*.json'))))));break
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
    registration = json.loads((OUT/'REGISTRATION.json').read_text())
    assert registration['design_sha256'] == sha(LOCAL/'docs/paper_results/WMPLAN_DESIGN.md')
    model = comparison['selected']
    quality_root = QUALITY if model == 'FULL-v5' else '/root/autodl-tmp/wmteach20261101_v3'
    modules = {p.name:p.read_text(encoding='utf-8') for p in (LOCAL/'tools/wmplan').glob('*.py')}
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
 design_sha256=artifacts['WMPLAN_DESIGN.md'],gpu_workers=2,
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
  'tools.wmplan.test_cpu','tools.coopplan.test_cpu']))
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
            'Original EXACT199200upper, equal2950credit quotas, resource-limited '
            'estimand and no-new-MAPPO contract retained. One-use gpuA root '
            f'{ROOT}; no model switch or live-outcome selection.\n')
    with REG.open('a',encoding='utf-8') as f:
        f.write(line);f.flush()
    receipt = dict(line=line,binding_sha256=digest(b),comparison_sha256=sha(OUT/'MODEL_COMPARISON.json'))
    save('FINAL_FREEZE.json',receipt)
    print(remote(f'''import pathlib,json
root=pathlib.Path({ROOT!r})
with (root/'FINAL_FREEZE.json').open('x') as f:json.dump({receipt!r},f)
print(json.dumps(dict(frozen=True,new_sumo_sim_s=0)))
'''))


def launch():
    freeze_receipt = json.loads((OUT/'FINAL_FREEZE.json').read_text())
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
for i in range(b['gpu_workers']):
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inspect','register','compare','stage','cpu','freeze','launch'])
    globals()[p.parse_args().action]()
