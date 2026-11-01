"""Deploy, launch once, and inspect the registered final world-job supervisors."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tools.rbint.remote import ssh
from tools.rb2int.audit import ROOT, load

REMOTE_ROOT = '/root/autodl-tmp/c4/rb2int20261101/final_v5_v1'


def deployment_files(host, control):
    base = ROOT/'docs/rb2int'
    release_raw = (base/'FINAL_RELEASE_V2.json').read_bytes()
    registration_raw = (base/'EVALUATION_REGISTRATION.json').read_bytes()
    plan_raw = (base/'FINAL_RUN_PLAN.json').read_bytes()
    release, plan = json.loads(release_raw), json.loads(plan_raw)
    if release['status'] != 'READY' or release['registration_sha256'] != hashlib.sha256(registration_raw).hexdigest() or plan['registration_sha256'] != release['registration_sha256']:
        raise ValueError('registered immutable release/plan required')
    files = {'RELEASE.json':release_raw,'REGISTRATION.json':registration_raw,'RUN_PLAN.json':plan_raw}
    own_models = sorted({j['model'] for j in plan['jobs'] if j['host'] == host})
    for model in own_models:
        root = '/root/autodl-tmp/c4/rb2int20261101/'
        if model in ('WM2','FULL-v4'):
            inputs = root+'view_new50_prefix_all58_v1'
            receipt = load(base/('INPUT_NEW50_GPUA_VIEW.json' if host == 'gpuA' else 'INPUT_NEW50_GPUR_VIEW.json'))
            worlds = list(range(969550,969600))
        else:
            inputs = root+('received_prefix_all58_v1' if host == 'gpuA' else 'export_prefix_all58_v1/inputs58')
            # These hashes come from the verified complete58 causal export.
            receipt = load(base/'INPUT_ALL58_MANIFEST.json')['build']
            worlds = plan['final_worlds']
        panel = dict(schema='rb2int-panel-inputs/1',purpose='final_comparison',worlds=worlds,
            registration_path=REMOTE_ROOT+'/REGISTRATION.json',registration_sha256=release['registration_sha256'],
            release_path=REMOTE_ROOT+'/RELEASE.json',release_sha256=hashlib.sha256(release_raw).hexdigest(),
            observations_root=inputs,observation_registry_sha256=receipt['observation_registry_sha256'],
            real_observables=inputs+'/prefix_observables.json',real_observables_sha256=receipt['prefix_observables_sha256'])
        files['panels/'+model+'.json'] = json.dumps(panel,indent=1).encode()
    spec = dict(schema='rb2int-final-host-spec/1',host=host,control=control,
        files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
        source_model_commit=release['source_model_commit'],final_outcome_values_used=False)
    files['SPEC.json'] = json.dumps(spec,indent=1).encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('deploy','launch','status'))
    parser.add_argument('--host',choices=('gpuA','gpuR'),required=True)
    parser.add_argument('--control',required=True)
    parser.add_argument('--receipt',required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'/root/autodl-tmp/c4/rb2int20261101/phys_code_[a-f0-9]{12}',args.control):
        raise ValueError('immutable committed control stage required')
    out = Path(args.receipt)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive local final operation receipt required')
    common = f'''
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path({REMOTE_ROOT!r});control=Path({args.control!r})
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):
 with path.open('x') as f: json.dump(obj,f,indent=1)
def alive(process):
 try:
  f=Path('/proc/'+str(process['pid'])+'/stat').read_text().split()
  return f[2]!='Z' and f[21]==process['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']
 except FileNotFoundError: return False
'''
    if args.action == 'deploy':
        payload = deployment_files(args.host,args.control)
        code = f'''
assert not root.exists(), 'preserve prior final deployment/claims'
assert root.parent.is_dir() and (control/'INT_STAGE.json').is_file()
assert shutil.disk_usage(root.parent).free>512*2**20
payload={payload!r}
release=json.loads(payload['RELEASE.json']);plan=json.loads(payload['RUN_PLAN.json'])
assert release['status']=='READY' and release['exposure_audit_passed']
origin=release['original_model_config'];assert sha(Path(origin['path']))==origin['sha256']
models={{j['model'] for j in plan['jobs'] if j['host']=={args.host!r}}}
for model in models:
 binding=release['catalog'][model]['binding']
 assert sha(Path(binding['config']))==binding['config_sha256']
 assert sha(Path('/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt'))==binding['wm_sha256']
 for path,expected in binding['sources'].items(): assert sha(Path(path))==expected, path
 panel=json.loads(payload['panels/'+model+'.json']);observations=Path(panel['observations_root'])
 assert sha(observations/'expected_hashes.json')==panel['observation_registry_sha256']
 assert sha(Path(panel['real_observables']))==panel['real_observables_sha256']
 assert [w['seed'] for w in json.loads((observations/'data/design.json').read_text())['worlds']]==panel['worlds']
root.mkdir();(root/'panels').mkdir()
for name,raw in payload.items():
 with (root/name).open('xb') as f: f.write(raw)
result=dict(root=str(root),host={args.host!r},control=str(control),files={{name:sha(root/name) for name in payload}},source_files_verified=True,final_efficacy_values_read=False)
save(root/'DEPLOYED.json',result);print(json.dumps(result))
'''
    elif args.action == 'launch':
        code = f'''
assert (root/'DEPLOYED.json').is_file() and not (root/'LAUNCH_ONCE.json').exists()
spec=json.loads((root/'SPEC.json').read_text());assert spec['host']=={args.host!r} and spec['control']==str(control)
for name,expected in spec['files'].items(): assert sha(root/name)==expected
entry=control/'tools/rb2int/final_worker.py';assert entry.is_file()
save(root/'LAUNCH_ONCE.json',dict(owner='S-RB2-INT',source_sha256=sha(entry),automatic_retry=False,plan_sha256=sha(root/'RUN_PLAN.json')))
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(entry),'--root',str(root),'--host',{args.host!r}]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'supervisor.log').open('x') as log:
 child=subprocess.Popen(argv,env=env,cwd=control,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
fields=Path('/proc/'+str(child.pid)+'/stat').read_text().split()
result=dict(pid=child.pid,start_ticks=fields[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=argv,root=str(root),host={args.host!r})
save(root/'SUPERVISOR_PROCESS.json',result);print(json.dumps(result))
'''
    else:
        code = '''
result=dict(root=str(root),exists=root.is_dir())
if root.is_dir():
 p=root/'SUPERVISOR_PROCESS.json'
 if p.is_file(): result['process']=json.loads(p.read_text());result['alive']=alive(result['process'])
 jobs=root/'jobs';completed=[];failed=[];running=[];partial_tasks=0
 if jobs.is_dir():
  for job in sorted(jobs.iterdir()):
   if not job.is_dir(): continue
   partial_tasks+=len(list((job/'tasks').glob('w*_k*.json')))
   if (job/'EXIT.json').is_file():
    record=json.loads((job/'EXIT.json').read_text());(completed if record['complete_K8'] else failed).append(job.name)
   elif (job/'PROCESS.json').is_file():
    process=json.loads((job/'PROCESS.json').read_text());running.append(dict(job=job.name,pid=process['pid'],alive=alive(process)))
 result.update(complete_world_jobs=completed,failed_world_jobs=failed,running=running,future_task_files=partial_tasks)
 for name in ('PANEL_COMPLETE.json','PANEL_TERMINAL_INCOMPLETE.json'):
  if (root/name).is_file(): result[name]=json.loads((root/name).read_text())
 if (root/'supervisor.log').is_file(): result['log_tail']=(root/'supervisor.log').read_text()[-3000:]
print(json.dumps(result))
'''
    result = json.loads(ssh(args.host,common+code,timeout=90))
    with out.open('x',encoding='utf-8') as stream:
        json.dump(dict(result,action=args.action,host=args.host),stream,indent=1)
    print(json.dumps(result,indent=1))


if __name__ == '__main__':
    main()
