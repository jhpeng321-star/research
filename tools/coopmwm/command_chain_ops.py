"""MAIN07:18 observable-only command-chain diagnosis and qualification."""
import argparse
import json
from . import ops
from .storage import write

CPU='/home/ubuntu/coopmwm20261102_commandcpu3'


def sources():
    """Inspect only the named runtime code, never scenario/type assignment files."""
    programs = {
        'gpuA': '''import pathlib,json,hashlib
r=pathlib.Path('/root/rbeval20261031/phys_code_22cc32f65a80')
names=['src/wms/control/alwayscmd.py','src/wms/control/bridge.py','src/wms/control/contract.py','src/wms/control/safety.py']
out={}
for name in names:
 p=r/name;raw=p.read_bytes();out[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),code=raw.decode())
print(json.dumps(out))''',
        'cpuQ': '''import pathlib,json,hashlib
r=pathlib.Path('/home/ubuntu/e3final/runtime')
names=['traffic_shepherd/damping.py','traffic_shepherd/config.py','traffic_shepherd/sensing.py','traffic_shepherd/c4/env.py','traffic_shepherd/ownership.py']
out={}
for name in names:
 p=r/name;raw=p.read_bytes();out[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),code=raw.decode())
print(json.dumps(out))'''}
    for host,code in programs.items():
        result=json.loads(ops.ssh.remote(host,code,timeout=40))
        write(ops.EVIDENCE/f'COMMAND_CHAIN_SOURCE_{host}.json',result)
        for name,v in result.items():
            if name.endswith(('alwayscmd.py','config.py')):
                print(host,name,v['sha256']);print(v['code'])


def show():
    from .storage import read
    for host,name in [('gpuA','src/wms/control/bridge.py'),('gpuA','src/wms/control/safety.py')]:
        print(host,name);print(read(ops.EVIDENCE/f'COMMAND_CHAIN_SOURCE_{host}.json')[name]['code'])


def laws():
    result=json.loads(ops.ssh.remote('gpuA','''import pathlib,json,hashlib
r=pathlib.Path('/root/rbeval20261031/phys_code_22cc32f65a80')
out={}
for name in ['src/wms/control/sumo_idm.py','src/wms/control/sumo_leader.py']:
 p=r/name;raw=p.read_bytes();out[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),code=raw.decode())
print(json.dumps(out))''',timeout=40))
    write(ops.EVIDENCE/'COMMAND_CHAIN_LAWS_gpuA.json',result)
    print(result['src/wms/control/sumo_idm.py']['code'])


def fixtures():
    from .warm_deploy import ROOT
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,gzip,time,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261102_quotacpu4/pytools')
from tools.coopmwm.accel_storage import read
cases=[]
for world in (986051,986070):
 episode=f'warm_u002_B_{{world}}';gid=f'xg_{{episode}}_t0300'
 grade=read(r/'exact_grades/receipts'/(gid+'.json'))
 dirs=[r/'jobs'/episode,r/'exact_grades/jobs'/gid/f'attempt{{grade["attempt"]}}']
 results=[read(p/'RESULT.json') for p in dirs]
 frames=[]
 for p in dirs:
  with gzip.open(p/'observable.jsonl.gz','rt') as f:
   frames.append([json.loads(line) for line in f if 300<=json.loads(line)['time_s']<=306])
 step=next(s for s in results[0]['steps'] if s['time_s']==300)
 request=next(read(p) for p in (r/'requests').glob('*.json') if read(p).get('world_id')==world and read(p).get('time_s')==300)
 cases.append(dict(world=world,grade=grade,plan=step['proposal']['plan'],record=request,
  frames=frames,steps=[[s for s in result['steps'] if 300<=s['time_s']<=306] for result in results]))
print(json.dumps(dict(unix=time.time(),cases=cases,new_sumo_sim_s=0)))''',timeout=60))
    write(ops.EVIDENCE/'COMMAND_CHAIN_NULL_FIXTURES.json',value)
    for c in value['cases']:
        print(json.dumps(dict(world=c['world'],group=c['plan']['members'],
            frames=[dict(time_s=f['time_s'],members=[r for r in f['vehicles'] if r['vid'] in c['plan']['members']]) for f in c['frames'][0]])))


def contrast():
    from .storage import read
    for c in read(ops.EVIDENCE/'COMMAND_CHAIN_NULL_FIXTURES.json')['cases']:
        proposal=c['steps'][0][0]['proposal']
        print(json.dumps(dict(world=c['world'],proposal_keys=list(proposal),
            plan={k:c['plan'].get(k) for k in ('id','origin','saving_vs_policy','saving_vs_mappo')},
            scores={k:v for k,v in proposal.items() if k in ('comparator_means','expected_saving','expected_saving_vs_policy','centre_open_loop_gap','centre_rollout','final')},
            topk=[{k:p.get(k) for k in ('id','origin','saving_vs_policy','saving_vs_mappo','p_best')} for p in proposal['topk']])))


def qualify():
    from .storage import read
    from .protocol import checked
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_centrefaircpu1.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});out={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;out[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(out))''',timeout=40))
    for name in ('command_chain','command_chain_ops','test_command_chain_cpu','plan_wm'):
        files['tools/coopmwm/'+name+'.py']=checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    stage=json.loads(ops.stage('cpuQ',CPU,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path[:0]=[{CPU!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={CPU!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={CPU!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
names=['tools.coopmwm.'+n for n in ('test_command_chain_cpu','test_mppi_runtime_cpu','test_volume_cpu','test_warm_cpu','test_mppi_search_cpu','test_committed_cpu','test_overlap_cpu')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
begin=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('CPU only/no SUMO or WM calls')):
 result=unittest.TextTestRunner(stream=sys.stdout).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={CPU!r},wall_s=time.monotonic()-begin)))''',timeout=90)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=stage['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_commandcpu3.json',v)
    assert v['pass_'] and v['sim_s']==v['skips']==0


def probe():
    from .storage import read
    from .protocol import checked
    root='/root/coopmwm20261102_commandprobe1'
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_commandcpu3.json')
    ops.stage_frozen('gpuA',root,cpu)
    files={'tools/coopmwm/command_chain_probe.py':checked(ops.LOCAL/'tools/coopmwm/command_chain_probe.py').read_text(encoding='utf-8'),
        'NULL_FIXTURES.json':checked(ops.EVIDENCE/'COMMAND_CHAIN_NULL_FIXTURES.json').read_text(encoding='utf-8')}
    ops.stage('gpuA',root,files)
    out=ops.ssh.remote('gpuA',f'''import sys,os,json
sys.path.insert(0,{root!r}+'/pytools');os.nice(19)
from tools.coopmwm.command_chain_probe import run
print(json.dumps(run({root!r}+'/NULL_FIXTURES.json')))''',timeout=90)
    v=json.loads(out);write(ops.EVIDENCE/'COMMAND_CHAIN_NULL_CALLBACK_PROBE.json',v)
    print(json.dumps(dict(v,cases=[{k:x for k,x in c.items() if k!='trace'} for c in v['cases']])))


def probe_kinematic():
    """Separate CPU-only audit; original probe and all running owners retained."""
    from .storage import read
    from .protocol import checked
    root='/root/coopmwm20261102_commandkinematic1'
    target=ops.EVIDENCE/'COMMAND_CHAIN_KINEMATIC_CALLBACK_PROBE.json'
    assert not target.exists(),'adopt completed probe rather than repeat it'
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_commandcpu3.json')
    ops.stage_frozen('gpuA',root,cpu)
    files={'tools/coopmwm/'+name+'.py':checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
        for name in ('command_chain_probe','test_command_chain_probe_cpu')}
    files['NULL_FIXTURES.json']=checked(ops.EVIDENCE/'COMMAND_CHAIN_NULL_FIXTURES.json').read_text(encoding='utf-8')
    staged=json.loads(ops.stage('gpuA',root,files))
    out=ops.ssh.remote('gpuA',f'''import sys,os,json,unittest
from unittest.mock import patch
sys.path.insert(0,{root!r}+'/pytools');os.nice(19)
from tools.coopmwm.command_chain_probe import run
suite=unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_command_chain_probe_cpu')
with patch('subprocess.Popen',side_effect=AssertionError('audit cannot launch physics or GPU jobs')):
 result=unittest.TextTestRunner(stream=sys.stderr).run(suite)
 assert result.wasSuccessful() and not result.skipped
 v=run({root!r}+'/NULL_FIXTURES.json',kinematic=True,continuous=True)
v.update(tests=result.testsRun,source_root={root!r},numerical_activation=False)
print(json.dumps(v))''',timeout=90)
    v=json.loads(out);v['staged_hashes']=staged['hashes'];write(target,v)
    print(json.dumps(dict(v,cases=[{k:x for k,x in c.items() if k not in ('trace','errors','unresolved_half_steps')} for c in v['cases']])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('sources','show','laws','fixtures','contrast','qualify','probe','probe_kinematic'))
    globals()[p.parse_args().mode]()
