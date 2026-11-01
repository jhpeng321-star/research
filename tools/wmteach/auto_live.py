"""First-complete-policy live freeze, CPU gates and one-shot approved dispatch."""
import json
from pathlib import Path

from tools.coopbandit.protocol import sha
from tools.wmteach.ops import remote,ROOT,REMOTE

LIVE='/root/autodl-tmp/wmteach_live20261101_v1'
PEER='/root/autodl-tmp/distill20261101_v1'
SCALE='/root/autodl-tmp/wmteach_scale_v5_20261101_v2'


def prepare():
    design=(ROOT/'docs/paper_results/WMTEACH_LIVE_DESIGN.md').read_text(encoding='utf-8')
    files={'pytools/tools/wmteach/'+name:(ROOT/'tools/wmteach'/name).read_text(encoding='utf-8')
           for name in ('__init__.py','portable.py','live.py','test_live.py')}
    registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    from tools.wmteach.live import approval
    from tools.wmteach.live import budget,PANEL_SHA
    authority=approval(registry.read_text(encoding='utf-8'))
    source=f'''import pathlib,json,hashlib,sys,os,shutil
root=pathlib.Path({LIVE!r});teacher=pathlib.Path({SCALE!r});peer=pathlib.Path({PEER!r})
if root.exists():
 print(json.dumps(dict(state='STAGED',freeze=json.loads((root/'FREEZE.json').read_text()))));sys.exit()
student=teacher/'student_scale_5120'
if not (student/'COMPLETE.json').is_file() or not (teacher/'QUALITY.json').is_file():
 print(json.dumps(dict(state='WAIT_STUDENT')));sys.exit()
gate=json.loads((teacher/'HELPFUL_QUALITY.json').read_text())
if not gate['pass_']:
 print(json.dumps(dict(state='WITHHELD_TEACHER_QUALITY',quality=gate)));sys.exit()
if shutil.disk_usage(root.parent).free<2*2**30:
 print(json.dumps(dict(state='WAIT_DISK')));sys.exit()
sys.path[:0]=[str(teacher),str(teacher/'deps')]
import pickle,numpy as np,types
from tools.distill.model import predict_values,choose as original_choose
from tools.distill.data import KEYS
from tools.coopbandit.protocol import digest,sha
summary=json.loads((student/'SUMMARY.json').read_text());complete=json.loads((student/'COMPLETE.json').read_text())
assert summary['states']>=5000
original=sha(student/'policy.pkl');assert original==complete['policy_sha256']==summary['policy_sha256']
with (student/'policy.pkl').open('rb') as f:payload=pickle.load(f)
assert payload['label_horizon_s']==300 and payload['train_worlds']==list(range(969400,969432))
assert payload['teacher']=='FULL-v5_WM_only'
portable=types.ModuleType('wmteach_frozen_export');sys.modules[portable.__name__]=portable
own=json.loads({json.dumps(files)!r})
exec(compile(own['pytools/tools/wmteach/portable.py'],'<frozen-wmteach-export>','exec'),portable.__dict__)
policy=portable.export(payload)
labels=[json.loads(p.read_text()) for p in sorted((teacher/'labels').glob('*/LABEL_H300.json'))[:32]]
x=np.array([[r['features'][k] for k in KEYS] for r in labels]);av=np.array([r['available'] for r in labels])
rng=np.random.default_rng(20261101)
query=np.concatenate([x,x.mean(0)+rng.normal(size=(200,len(KEYS)))*np.maximum(x.std(0),1.)])
available=np.concatenate([av,np.ones((200,5),dtype=bool)])
expected=predict_values(payload['models'],query);actual=portable.predict(policy,query)
np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-10)
np.testing.assert_array_equal(portable.choose(policy,actual,available)[0],
 original_choose(expected,available,payload['errors'],payload['confidence']))
binding=json.loads((peer/'BINDING.json').read_text())
for name,d in {{**binding['sources'],**binding['artifacts']}}.items():assert sha(peer/name)==d,name
source_files={{name:(peer/name).read_text() for name in binding['sources']}}
source_files.update(own)
PANEL_SHA={PANEL_SHA!r}
def budget():return {budget()!r}
panel=json.loads((peer/'PANEL.json').read_text());assert digest(panel)==PANEL_SHA
exposure=json.loads((peer/'EXPOSURE.json').read_text());assert exposure['pass_']
artifacts={{'POLICY.json':json.dumps(policy,sort_keys=True),'PANEL.json':json.dumps(panel,sort_keys=True),
 'WMTEACH_LIVE_DESIGN.md':{design!r},
 'EXPOSURE.json':json.dumps(dict(pass_=True,shared_panel=True,original_pre_panel_exposure=exposure,
  authority={authority!r},panel_sha256=PANEL_SHA),sort_keys=True)}}
root.mkdir()
for name,text in {{**source_files,**artifacts}}.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(text)
for name in ('jobs','states','reservations'):(root/name).mkdir()
new_binding={{k:binding[k] for k in ('runtime','checkpoint','deps')}}
new_binding.update(workers=2,load_max=16,budget=budget(),panel=panel,
 sources={{n:sha(root/n) for n in source_files}},artifacts={{n:sha(root/n) for n in artifacts}},
 design_sha256=sha(root/'WMTEACH_LIVE_DESIGN.md'))
freeze=dict(design_sha256=new_binding['design_sha256'],original_policy_sha256=original,
 policy_sha256=sha(root/'POLICY.json'),panel_sha256=PANEL_SHA,budget=budget(),
  selected_snapshot='student_scale_5120',training_states=summary['states'],helpful_quality=gate,
  teacher_identity_sha256=sha(teacher/'IDENTITY.json'),teacher_backend=payload['teacher'],
 quality_sha256=sha(teacher/'QUALITY.json'),portable_parity_states=len(query),
 max_prediction_error=float(abs(expected-actual).max()),new_sumo_sim_s=0)
for name,value in [('BINDING.json',new_binding),('FREEZE.json',freeze),('APPROVAL.json',dict(line={authority!r}))]:
 with (root/name).open('x') as f:json.dump(value,f,sort_keys=True)
print(json.dumps(dict(state='STAGED',freeze=freeze)))
'''
    result=json.loads(remote(source,timeout=120))
    if result['state']=='STAGED':
        path=ROOT/'docs/paper_results/wmteach/LIVE_FREEZE.json'
        if not path.exists():
            with path.open('x',encoding='utf-8') as f:json.dump(result['freeze'],f,sort_keys=True);f.write('\n')
    return result


def cpu():
    return json.loads(remote(f'''import pathlib,json,os,sys,unittest
from unittest.mock import patch
root=pathlib.Path({LIVE!r})
if (root/'CPU.json').exists():print((root/'CPU.json').read_text());sys.exit()
binding=json.loads((root/'BINDING.json').read_text())
sys.path[:0]=[str(root/'pytools'),*binding['deps']]
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from tools.wmteach.live import configure
p=configure();p.setup(root)
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled during CPU gate')):
 for w in p.panel():p.guard(w)
 result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromName('tools.wmteach.test_live'))
receipt=dict(pass_=result.wasSuccessful(),tests=result.testsRun,guards=24,skips=len(result.skipped),
 new_sumo_sim_s=0,portable_parity=json.loads((root/'FREEZE.json').read_text())['portable_parity_states'])
assert receipt['pass_'] and not receipt['skips']
with (root/'CPU.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''',timeout=120).splitlines()[-1])


def preregister(freeze):
    registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    line=(f'- S-WMTEACH LIVE FROZEN BEFORE DISPATCH: designSHA256{freeze["design_sha256"]}; '
          f'FIRST complete student_scale_5120 ONLY/originalpolicySHA{freeze["original_policy_sha256"]}/'
          f'portablepolicySHA{freeze["policy_sha256"]}, H300 WM-only targets/32FITworlds/'
          f'{freeze["training_states"]}distinctstates/helpfulqualityPASS, '
          f'teacher{freeze["teacher_backend"]}/publishedb2df7458fixedBEFOREv5quality, '
          f'qualitySHA{freeze["quality_sha256"]} completed before fitting; portable original parity '
          f'{freeze["portable_parity_states"]}states. SAME S-DISTILL24world DEV984000..984023/'
          f'panelSHA{freeze["panel_sha256"]}, no new seed block/fitting on live worlds. EXACT58800upper='
          '57600paired24x2x1200+1200extraBITWISE-O0mechanics/firstbaseline reusedONCE; '
          '41200UNALLOCATED under MAIN11:48 cap100000. Efficacy-only/no new member-aloneCG branches/'
          'no action budget pruning/no superadditivity claim; same frozen MAPPO169/options/executors/'
          'safety/cadence/metrics and bothguards. All24pairedJ600/J900/worldbootstrap10000/safety/moves. '
          f'Immutable {LIVE}; CPU3tests/24guards and original shared pre-panel exposure receipts, '
          'nice19/singlethread/CUDAhidden/max2worldworkers/load16/mem8GiB/disk2GiB. '
          'One-use root/claim/no retries/refunds/peerPIDactions/deletions/push/newenv.\n')
    text=registry.read_text(encoding='utf-8')
    marker='portablepolicySHA'+freeze['policy_sha256']
    if marker not in text:
        with registry.open('a',encoding='utf-8') as f:f.write(line);f.flush()
    receipt=ROOT/'docs/paper_results/wmteach/LIVE_REGISTRATION.json'
    if not receipt.exists():
        with receipt.open('x',encoding='utf-8') as f:json.dump(dict(line=line,freeze=freeze),f,sort_keys=True);f.write('\n')


def launch():
    return json.loads(remote(f'''import pathlib,json,subprocess,sys,os,shutil,time
root=pathlib.Path({LIVE!r})
for name in ('DISPATCH_ONCE.json','LAUNCH_ONCE.json','STOP'):
 if (root/name).exists():print(json.dumps(dict(state='EXISTING',marker=name)));sys.exit()
assert not list((root/'jobs').iterdir()) and not (root/'CHARGES.jsonl').exists()
assert json.loads((root/'CPU.json').read_text())['pass_']
if os.getloadavg()[0]>=16 or shutil.disk_usage(root).free<2*2**30:
 print(json.dumps(dict(state='WAIT_RESOURCES')));sys.exit()
binding=json.loads((root/'BINDING.json').read_text());sys.path[:0]=[str(root/'pytools'),*binding['deps']]
from tools.wmteach.live import configure
p=configure();p.verify(root)
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
with (root/'LAUNCH_ONCE.json').open('x') as f:json.dump(dict(parent=os.getpid(),unix=time.time()),f)
with (root/'LIVE.log').open('x') as log:
 child=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.wmteach.live',str(root),'--workers','2'],
  cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
  env={{**os.environ,'PYTHONPATH':':'.join([str(root/'pytools'),*binding['deps']]),'CUDA_VISIBLE_DEVICES':'',
   'SUMO_HOME':home,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
receipt=dict(state='LAUNCHED',pid=child.pid,start_ticks=pathlib.Path(f'/proc/{{child.pid}}/stat').read_text().split()[21],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),budget=58800)
with (root/'PROCESS.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
'''))


def advance():
    result=prepare()
    if result['state']!='STAGED':return result
    gate=cpu();assert gate['pass_']
    preregister(result['freeze'])  # Durably append authority BEFORE launch.
    from tools.wmteach.watch import commit_owned
    commit_owned([ROOT/'docs/paper_results/wmteach/LIVE_FREEZE.json',
                  ROOT/'docs/paper_results/wmteach/LIVE_REGISTRATION.json'],
                 'wmteach: freeze first student and register approved paired live budget')
    result=launch()
    path=ROOT/'docs/paper_results/wmteach/LIVE_DISPATCH.json'
    if result['state']=='LAUNCHED' and not path.exists():
        with path.open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    if args.plan:
        from tools.wmteach.live import budget,PANEL_SHA
        print(json.dumps(dict(design_sha256=sha(ROOT/'docs/paper_results/WMTEACH_LIVE_DESIGN.md'),
            budget=budget(),panel_sha256=PANEL_SHA,policy_selection='FIRST complete student_scale_5120 only',
            live_root=LIVE,dispatch_requires='>=5000distinct states + helpful quality PASS + portable parity + CPU/guards + final registration')))
    else:print(json.dumps(advance()))
