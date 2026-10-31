"""Owned CPU-only dispatch/collection for the entrant-inclusive FIT diagnostic."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/ubuntu/miniconda3/bin/python'
INT_COMMIT = 'ff8a637f'


def remote(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'cpuQ', PYTHON+' -B -'], input=code, text=True,
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('status', 'stage', 'launch', 'poll', 'collect'))
    p.add_argument('--version', default='v3')
    args = p.parse_args()
    if not args.version.isalnum():
        p.error('version must be alphanumeric')
    header = 'import os,json,time,hashlib\nfrom pathlib import Path\n'
    header += f'root=Path({("/tmp/rb-merge-fullfleet-"+args.version)!r})\n'
    if args.mode == 'status':
        print(remote(header+"print(json.dumps(dict(load=os.getloadavg(), cpus=os.cpu_count(), old_owned_pids={p:Path('/proc/'+str(p)).exists() for p in (200540,233273)}, existing_dispatch=(root/'LAUNCH.json').exists())))"))
    elif args.mode == 'stage':
        archive = subprocess.check_output(['git', 'archive', '--format=tar', INT_COMMIT, 'src/wms',
                                           'configs/native_cav_head_v5.json',
                                           'configs/execparity_cav_idm_v1.json'],
                                           cwd='E:/agent-work/c4-rb-int')
        payload = dict(runtime_archive=base64.b64encode(archive).decode(),
            validator=base64.b64encode((ROOT/'tools/rbeval/full_fleet.py').read_bytes()).decode(),
            tests=base64.b64encode((ROOT/'tests/unit/test_rb_merge_full_fleet.py').read_bytes()).decode(),
            targets={n: base64.b64encode((Path('E:/agent-work/rb-shared/targets')/n).read_bytes()).decode()
                     for n in ('ramp_delay_world.csv', 'ramp_merges.csv')})
        code = header+f'payload=json.loads({json.dumps(payload)!r})\n'
        code += '''import base64,io,tarfile,subprocess,sys
assert root.parent.is_dir()
root.mkdir(exist_ok=False)
runtime=root/'runtime'
runtime.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload['runtime_archive']))) as tar:
    tar.extractall(runtime, filter='data')
tools=root/'tools'/'rbeval'
tools.mkdir(parents=True)
(tools/'full_fleet.py').write_bytes(base64.b64decode(payload['validator']))
test=root/'test_full_fleet.py'
test.write_bytes(base64.b64decode(payload['tests']))
targets=root/'targets'
targets.mkdir()
for name,data in payload['targets'].items():
    (targets/name).write_bytes(base64.b64decode(data))
env=dict(os.environ, PYTHONPATH=str(runtime/'src')+':'+str(root), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
result=subprocess.run([sys.executable,'-m','pytest',str(test),'-q'],cwd=root,env=env,text=True,capture_output=True)
receipt=dict(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,load=os.getloadavg(),cpus=os.cpu_count())
(root/'CPU_TESTS.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
assert result.returncode==0
preflight=subprocess.run([sys.executable,'-B',str(tools/'full_fleet.py'),'--root','/home/ubuntu/wmdata20261031/views','--runtime',str(runtime),'--targets',str(targets),'--seeds',*map(str,range(969400,969432)),'--preflight'],cwd=root,env=env,text=True,capture_output=True)
pre=dict(command=preflight.args,returncode=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr)
(root/'PREFLIGHT.json').write_text(json.dumps(pre,indent=2)+'\\n')
print(json.dumps(pre))
assert preflight.returncode==0
'''
        print(remote(code))
    elif args.mode == 'launch':
        code = header+f'python={PYTHON!r}\n'
        code += '''import subprocess
assert json.loads((root/'CPU_TESTS.json').read_text())['returncode']==0
assert json.loads((root/'PREFLIGHT.json').read_text())['returncode']==0
assert os.getloadavg()[0]<28 and os.cpu_count()>=32
assert not (root/'results').exists()
with (root/'LAUNCH.json').open('x') as f:
    json.dump(dict(state='reserved',owner='S-RB-MERGE'),f)
script=root/'tools/rbeval/full_fleet.py'
command=[python,'-B',str(script),'--root','/home/ubuntu/wmdata20261031/views','--runtime',str(root/'runtime'),'--targets',str(root/'targets'),'--output',str(root/'results'),'--seeds',*map(str,range(969400,969432)),'--horizon','300','--slots','320']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
with (root/'RUN.log').open('x') as log:
    child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
record=dict(owner='S-RB-MERGE',pid=child.pid,started_unix=time.time(),command=command,validator_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),load=os.getloadavg(),cpus=os.cpu_count(),scope='FIT32 native full fleet H300, one CPU thread; no SUMO/GPU/training')
(root/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
        print(remote(code))
    elif args.mode == 'poll':
        code = header+'''launch=json.loads((root/'LAUNCH.json').read_text())
pid=launch['pid']
result=root/'results'
rows=[s for s in range(969400,969432) if (result/(str(s)+'.json')).exists()]
print(json.dumps(dict(launch=launch,alive=Path('/proc/'+str(pid)).exists(),completed=rows,complete=(result/'COMPLETE.json').exists(),failed=json.loads((result/'FAILED.json').read_text()) if (result/'FAILED.json').exists() else None,log=(root/'RUN.log').read_text(),load=os.getloadavg())))
'''
        print(remote(code))
    else:
        code = header+'''result=root/'results'
files=[root/'LAUNCH.json',root/'CPU_TESTS.json',root/'PREFLIGHT.json',result/'IDENTITY.json',result/'COMPLETE.json',result/'FAILED.json']
files += [result/(str(s)+'.json') for s in range(969400,969432)]
print(json.dumps({p.name:dict(content=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files if p.exists()}))
'''
        data = json.loads(remote(code))
        dest = ROOT/('docs/rb_merge/fullfleet_'+args.version)
        dest.mkdir(exist_ok=False)
        for name, item in data.items():
            (dest/name).write_text(item['content'], encoding='utf-8', newline='\n')
            import hashlib
            assert hashlib.sha256((dest/name).read_bytes()).hexdigest() == item['sha256']
        (dest/'COLLECTION.json').write_text(json.dumps({n: v['sha256'] for n,v in data.items()}, indent=2)+'\n')
        print('Hash-verified collection:', dest, len(data))


if __name__ == '__main__':
    main()
