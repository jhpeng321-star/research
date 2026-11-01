"""Owner15:55 one-use engineering restart; preserve the stopped v1 root."""
import argparse
import json
import os
from pathlib import Path

from tools.wmplan import ops

OLD_ROOT = ops.ROOT
OLD_OUT = ops.OUT
ROOT = '/root/autodl-tmp/wmplan20261101_v2'
OUT = OLD_OUT/'restart_v2'
ACCOUNTING = dict(prior_charged_sim_s=3600,prior_actual_sim_s=900,
                  new_run_upper_sim_s=195840,combined_charged_upper_sim_s=199440,
                  combined_actual_upper_sim_s=196740)


def configure():
    ops.ROOT, ops.OUT = ROOT, OUT


def inspect():
    """Read only the named run receipts and recorded process identities."""
    result = {}
    for host in ('gpuA', 'gpuR'):
        receipt = json.loads((OLD_OUT/('DISPATCH.json' if host == 'gpuA' else 'GPU_R_READY.json')).read_text())
        owners = receipt['owners'] if host == 'gpuA' else [receipt]
        result[host] = json.loads(ops.remote(f'''import pathlib,json,os,shutil,subprocess
owners=json.loads({json.dumps(owners)!r});rows=[]
for owner in owners:
 p=pathlib.Path('/proc')/str(owner['pid'])/'stat';s=p.read_text().split() if p.exists() else None
 alive=bool(s and s[21]==owner['start_ticks'] and s[2]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'])
 rows.append(dict(owner=owner,alive=alive))
assert not any(r['alive'] for r in rows),'prior owned service still alive; do not dispatch'
root=pathlib.Path({ROOT!r})
print(json.dumps(dict(owners=rows,restart_exists=root.exists(),load=os.getloadavg(),disk=shutil.disk_usage('/root').free,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
''',host=host))
    OUT.mkdir(exist_ok=True)
    configure()
    ops.save('RECOVERY.json', dict(hosts=result,new_sumo_sim_s=0))
    print(json.dumps(result))


def register():
    from tools.coopbandit.protocol import sha
    text = ops.REG.read_text(encoding='utf-8')
    approval = next(s for s in text.splitlines() if '15:55 MAIN: S-WMPLAN engineering-restart amendment APPROVED' in s)
    fault = json.loads((OLD_OUT/'FAULT_RECEIPTS.json').read_text())
    assert (fault['charged'],fault['actual'],fault['complete_episodes']) == (3600,900,0)
    proposal = json.loads((OLD_OUT/'RESTART_PROPOSAL.json').read_text())
    assert all(proposal[k] == v for k,v in ACCOUNTING.items())
    configure()
    ops.save('APPROVAL.json',dict(line=approval,approved=True,**ACCOUNTING,
        smoke='first expected world984000 counts toward48, no extra simulation',
        fault_receipt_sha256=sha(OLD_OUT/'FAULT_RECEIPTS.json'),
        model='FULL-v4',new_baseline_episodes=0,one_attempt_per_new_episode=True,
        on_new_fault='durable STOP and report; no automatic retry'))
    print(json.dumps(dict(registered=True,**ACCOUNTING,new_sumo_sim_s=0)))


def stage():
    """Clone only hash-bound inputs; never mutate the stopped v1 tree."""
    approval = json.loads((OUT/'APPROVAL.json').read_text())
    assert approval['approved']
    configure()
    modules = {p.name:p.read_text(encoding='utf-8') for p in (ops.LOCAL/'tools/wmplan').glob('*.py')}
    result = json.loads(ops.remote(f'''import pathlib,json,hashlib,shutil
old=pathlib.Path({OLD_ROOT!r});root=pathlib.Path({ROOT!r})
assert root.parent.is_dir() and not root.exists()
b=json.loads((old/'BINDING.json').read_text())
assert (old/'STOP.json').is_file() and not (old/'COMPLETE.json').exists()
assert b['model']['selected']=='FULL-v4'
for name,h in {{**b['sources'],**b['artifacts'],**b['external_sources']}}.items():
 p=pathlib.Path(name) if name.startswith('/') else old/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
root.mkdir()
def put(name,data):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
for name in {{**b['sources'],**b['artifacts']}}:
 if '/tools/wmplan/' in name:continue
 put(name,(old/name).read_bytes())
modules=json.loads({json.dumps(modules)!r})
for tree in ('pytools','pywm'):
 for name,text in modules.items():put(f'{{tree}}/tools/wmplan/{{name}}',text.encode())
put('RESTART_APPROVAL.json',{json.dumps(approval)!r}.encode())
for name in ('jobs','states','reservations','requests','responses','forecast_claims','forecasts'):
 (root/name).mkdir()
reused={{}}
for p in sorted((old/'requests').glob('*.json')):
 key=p.stem;response=old/'responses'/p.name;futures=sorted((old/'forecasts'/key).glob('k*.json'))
 assert response.is_file() and len(futures)==8
 for source in [p,response,*futures]:
  name=str(source.relative_to(old));data=source.read_bytes();put(name,data)
  reused[name]=hashlib.sha256(data).hexdigest()
 put('forecast_claims/'+p.name,json.dumps(dict(reused_from=str(old),completed=True)).encode())
assert len(reused)==30
put('REUSED_FORECASTS.json',json.dumps(reused,sort_keys=True).encode())
b['restart']={ACCOUNTING!r}
b['sources']={{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for tree in ('pytools','pywm') for p in (root/tree).rglob('*.py')}}
b['artifacts'].update({{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('RESTART_APPROVAL.json','REUSED_FORECASTS.json')}})
put('BINDING.json',json.dumps(b,sort_keys=True).encode())
print(json.dumps(dict(root=str(root),binding=b,reused_K8_states=3,new_sumo_sim_s=0)))
''',timeout=180))
    ops.save('STAGED.json',result)
    # The gpuR serial teacher path is unchanged: reuse its immutable, tested mirror.
    for name in ('MIRROR.json','MIRROR_CPU_GPU_GATE.json','MIRROR_BASE_REPAIR.json',
                 'MIRROR_PACKAGE_ASSETS.json','MODEL_COMPARISON.json'):
        ops.save(name,json.loads((OLD_OUT/name).read_text()))
    print(json.dumps(dict(root=ROOT,reused_K8_states=3,new_sumo_sim_s=0)))


def cpu():
    configure()
    ops.cpu()


def freeze():
    from tools.coopbandit.protocol import digest,sha
    configure()
    cpu_result = json.loads((OUT/'CPU.json').read_text())
    b = json.loads((OUT/'STAGED.json').read_text())['binding']
    assert cpu_result['pass_'] and cpu_result['guards'] == 24
    line = (f'- S-WMPLAN OWNER15:55 ENGINEERING RESTART FROZEN BEFORE DISPATCH: '
        f'initial_rows repair e492210c; bindingSHA256{digest(b)}; '
        f'approvalSHA256{sha(OUT/"APPROVAL.json")}; FULL-v4 unchanged (4/8 vs v5 2/8); '
        'prior3600charged/900actual retained, no refund; new48episodes57600 + '
        'max48x2880twins138240 =195840newupper; combined199440charged/196740actual '
        '<=200000cap; unchanged2950/world-arm credit, at most1group/t<=600, '
        'K8/H300/60s/O0..O4/executors/safety/expected-and-strict rules. '
        'Smoke=expected984000 INCLUDED in24primary/48total, then other23expected '
        'on bothGPUs, verified committed primary report before strict24. '
        'New immutable gpuA /root/autodl-tmp/wmplan20261101_v2; gpuR reuses '
        'byte-frozen inference-only v1 mirror; all3complete K8 forecasts reused; '
        f'CPU{cpu_result["tests"]}PASS/24guards; all24bitwiseMAPPO reused/newMAPPO0. '
        'One attempt per new episode; any new fault STOP/report, no retry; v1 preserved.\n')
    receipt = dict(line=line,binding_sha256=digest(b),**ACCOUNTING,
                   mirror_sha256=sha(OUT/'MIRROR.json'),
                   mirror_gate_sha256=sha(OUT/'MIRROR_CPU_GPU_GATE.json'),
                   mirror_base_repair_sha256=sha(OUT/'MIRROR_BASE_REPAIR.json'),
                   mirror_package_assets_sha256=sha(OUT/'MIRROR_PACKAGE_ASSETS.json'))
    ops.save('FINAL_FREEZE.json',receipt)
    print(ops.remote(f'''import pathlib,json
root=pathlib.Path({ROOT!r})
with (root/'FINAL_FREEZE.json').open('x') as f:json.dump({receipt!r},f)
print(json.dumps(dict(frozen=True,new_sumo_sim_s=0)))
'''))
    # Registration text is applied with apply_patch before launch; this receipt
    # supplies the exact immutable line for the shared ledger.
    print(line)


def launch():
    configure()
    receipt = json.loads((OUT/'FINAL_FREEZE.json').read_text())
    assert receipt['line'].strip() in ops.REG.read_text(encoding='utf-8')
    os.environ.update(WMPLAN_RUN_ROOT=ROOT,WMPLAN_RECEIPT_DIR=str(OUT),
                      WMPLAN_RELAY_TAG='wmplan_v2_gpuR')
    # Set bridge configuration in this process as well as its detached child.
    from tools.wmplan import bridge
    bridge.TAG = 'wmplan_v2_gpuR'
    ops.launch()


def status():
    configure()
    ops.status()


def progress():
    configure()
    result = json.loads(ops.remote(f'''import pathlib,json,time
root=pathlib.Path({ROOT!r})
def load(p):return json.loads(p.read_text())
print(json.dumps(dict(unix=time.time(),smoke=load(root/'SMOKE_COMPLETE.json') if (root/'SMOKE_COMPLETE.json').exists() else None,
 ready=[p.name for p in root.glob('GPU_READY_*.json')],
 jobs=[dict(id=p.name,close=load(p/'CLOSE.json') if (p/'CLOSE.json').exists() else None,
 decisions=[dict(time_s=e['time_s'],action=e['executed_action']['id'],proposed=e.get('proposed_action',{{}}).get('id'))
 for e in [load(f) for f in sorted(p.glob('DECISION_*.json'))]]) for p in (root/'jobs').iterdir()],
 forecasts=[dict(key=p.name,complete=(root/'responses'/(p.name+'.json')).exists(),futures=len(list(p.glob('k*.json')))) for p in (root/'forecasts').iterdir()])))
'''))
    ops.save(f'PROGRESS_{int(result["unix"])}.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inspect','register','stage','cpu','freeze','launch','status','progress'])
    globals()[p.parse_args().action]()
