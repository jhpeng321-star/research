"""MAIN07:18 observable-only command-chain diagnosis and qualification."""
import argparse
import json
from . import ops
from .storage import write


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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('sources','show','laws','fixtures','contrast'))
    globals()[p.parse_args().mode]()
