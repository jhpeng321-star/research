"""Task-only GPU metadata admission; never signals or inspects peer process inputs."""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile
import time

from tools.c1switch.design import checked, sha
from tools.c1switch.model import WM2_SHA, BB_SHA
from tools.c1switch.storage import canonical, write

from .design import CONTROLLERS, DESIGN, GPU_SECONDS, PROTOCOL, REGISTRATION
from .gate import approval, cpu_authority
from .operations import EVIDENCE, ssh


def inspect(host):
    result = ssh(host, '''import json,os,subprocess,time
from pathlib import Path
base=Path('/root/autodl-tmp/c4/speedhead20261030')
paths=[base/'tools/speedhead/wm_worker.py',base/'e3final/runtime/src',
       base/'EXPORT_BOUND.json',base/'pythondeps',Path('/root/c1switch20261031/main_v6/bb.pt')]
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(dict(unix=time.time(),gpu=gpu,load=os.getloadavg(),required_paths={str(p):p.exists() for p in paths},new_gpu_s=0,new_sim_s=0)))
''', timeout=30)
    path = EVIDENCE/('GPU_INSPECT_'+host+'_'+str(int(result['unix']))+'.json')
    write(path, result)
    print(json.dumps(result, indent=2))


def stage(host, model):
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    if not approval(text) or not cpu_authority(text):
        raise ValueError('current MAIN authority absent')
    files = sorted(Path('tools/c1switch').glob('*.py'))+sorted(Path('tools/c1switch2').glob('*.py'))
    files += [Path(DESIGN), Path('traffic_shepherd/c4/switchable_wall.py'), Path('src/wms/control/wall.py')]
    sources = {p.as_posix(): sha(p) for p in files}
    root = '/root/c1switch2_gpu_'+model+'_'+canonical(sources)[:16]
    binding = dict(protocol=PROTOCOL, arm=model, model=model, library=list(CONTROLLERS),
        history_s=60, horizon_s=60, K=8, objective='on_road_fleet_delay',
        evaluation_objective='full_team_delay_fleet_plus_source', objective_design_sha256=sha(DESIGN),
        observable_source_admissions=False, entries=False, controller_feedback=True,
        wall_start_stop=True, anonymous_seed=True, wm_checkpoint_sha256=WM2_SHA,
        bb_checkpoint_sha256=BB_SHA, checkpoint_path=root+'/bb.pt',
        service_gpu_cap_s=GPU_SECONDS[model]/2, prior_gpu_s=0)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for path in files:
            tar.add(checked(path), arcname=path.as_posix())
    encoded = base64.b64encode(buf.getvalue()).decode()
    result = ssh(host, f'''import base64,hashlib,io,json,os,tarfile
from pathlib import Path
root=Path({root!r});assert root.parent.is_dir() and not root.exists(),'preserve prior GPU stage'
for entry in os.scandir(root.parent):
 if entry.name.startswith('c1switch2_gpu_{model}_') and entry.is_dir(follow_symlinks=False):
  assert not Path(entry.path,'DISPATCH.json').exists(),'GPU allocation already claimed; never retry'
root.mkdir();stage=root/'pytools';stage.mkdir();(root/'rpc').mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r})),mode='r:gz') as tar:tar.extractall(stage,filter='data')
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==v for n,v in {sources!r}.items())
(root/'STAGED.json').write_text(json.dumps({sources!r},indent=2))
(root/'ENGINE.json').write_text(json.dumps({binding!r},indent=2))
print(json.dumps(dict(host={host!r},model={model!r},root=str(root),engine={binding!r},sources_sha256={canonical(sources)!r},new_gpu_s=0,new_sim_s=0)))
''', timeout=90)
    write(EVIDENCE/('GPU_STAGE_'+host+'_'+Path(root).name+'.json'), result)
    print(json.dumps(result, indent=2))
    return result


def cpu_probe(host, root, model):
    result = ssh(host, f'''import contextlib,io,json,os,sys
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_gpu_{model}_')
assert not (root/'CPU.json').exists()
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools'),'/root/autodl-tmp/c4/speedhead20261030/pythondeps'];os.chdir(root/'pytools')
log=io.StringIO()
try:
 with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  from tools.c1switch2.hostcheck import probe
  out=probe({model!r},str(root/'bb.pt'))
except BaseException as exc:
 import traceback
 out=dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),new_gpu_s=0,new_sim_s=0)
out['output']=log.getvalue();(root/'CPU.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''', timeout=110)
    write(EVIDENCE/('GPU_CPU_'+host+'_'+Path(root).name+'.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('output', 'attestation')}, indent=2))
    return result


def start(host, root):
    # CPU provenance comparison against the already validated original-host checkpoint.
    model = 'wm' if Path(root).name.startswith('c1switch2_gpu_wm_') else 'bb'
    cpu = json.loads((EVIDENCE/('GPU_CPU_'+host+'_'+Path(root).name+'.json')).read_text())
    original = json.loads((EVIDENCE/('MODEL_CPU_'+model+'_steps120_c1switch2_run_7e448ccdafc0c1c3.json')).read_text())
    if cpu['status'] != 'PASS':
        raise ValueError('real CPU feedback probe did not pass')
    for key in ('checkpoint_sha256', 'wm_config_sha256', 'candidate_sha256', 'mappo_sha256'):
        if cpu['attestation'][key] != original['attestation'][key]:
            raise ValueError('frozen inference provenance differs: '+key)
    result = ssh(host, f'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_gpu_{model}_')
assert json.loads((root/'CPU.json').read_text())['status']=='PASS'
assert all(hashlib.sha256((root/'pytools'/n).read_bytes()).hexdigest()==v for n,v in json.loads((root/'STAGED.json').read_text()).items())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(gpu.strip())>=7*1024,'yield to WM-PHYS memory priority'
for entry in os.scandir(root.parent):
 if entry.name.startswith('c1switch2_gpu_{model}_') and entry.is_dir(follow_symlinks=False):
  assert not Path(entry.path,'DISPATCH.json').exists(),'GPU allocation already claimed; never retry'
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root/'pytools'))
with (root/'GPU.log').open('x') as log:
 process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch2.gpuserver','--root',str(root)],cwd=root/'pytools',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(host={host!r},root=str(root),model={model!r},pid=process.pid,cap_s={GPU_SECONDS[model]/2!r},unix=time.time())
with (root/'DISPATCH.json').open('x') as stream:json.dump(out,stream,indent=2);stream.flush();os.fsync(stream.fileno())
print(json.dumps(out))
''', timeout=45)
    write(EVIDENCE/('GPU_DISPATCH_'+host+'_'+Path(root).name+'.json'), result)
    print(json.dumps(result, indent=2))
    return result


def collect(host, root):
    result = ssh(host, f'''import json,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_gpu_')
out=dict(host={host!r},root=str(root),unix=time.time())
for name in ('DISPATCH.json','rpc/GPU_STARTED.json','rpc/READY.json','rpc/STOP','rpc/GPU_FINISHED.json','CPU.json'):
 path=root/name
 if path.exists():out[name]=json.loads(path.read_text())
if 'DISPATCH.json' in out:
 proc=Path('/proc',str(out['DISPATCH.json']['pid']))
 out['pid_state']=proc.joinpath('stat').read_text().split()[2] if proc.exists() else 'EXITED'
if 'rpc/STOP' in out:out['log']=(root/'GPU.log').read_text()
print(json.dumps(out))
''', timeout=30)
    write(EVIDENCE/('GPU_PROGRESS_'+host+'_'+str(int(result['unix']))+'.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rpc/READY.json', 'CPU.json')}, indent=2))
    return result


def copy_bb(host, root):
    if host not in ('gpuA', 'gpuR') or not Path(root).name.startswith('c1switch2_gpu_bb_'):
        raise ValueError('explicit owned BB stage required')
    raw = ssh('gpuR', f'''import base64,hashlib,json,zlib
from pathlib import Path
source=Path('/root/c1switch20261031/main_v6/bb.pt')
assert hashlib.sha256(source.read_bytes()).hexdigest()=={BB_SHA!r}
out={{}}
for name,path in [('bb.pt',source),('physics.json',source.with_name('physics.json'))]:
 data=path.read_bytes();out[name]=dict(sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(zlib.compress(data)).decode())
print(json.dumps(out))
''', timeout=45)
    result = ssh(host, f'''import base64,hashlib,json,zlib
from pathlib import Path
root=Path({root!r});assert root.is_dir() and root.name.startswith('c1switch2_gpu_bb_')
out={{}}
for name,row in {raw!r}.items():
 assert name in ('bb.pt','physics.json');data=zlib.decompress(base64.b64decode(row['data']))
 assert hashlib.sha256(data).hexdigest()==row['sha256']
 with (root/name).open('xb') as stream:stream.write(data)
 out[name]=dict(sha256=row['sha256'],bytes=len(data))
print(json.dumps(dict(host={host!r},root=str(root),files=out,new_gpu_s=0,new_sim_s=0)))
''', timeout=60)
    write(EVIDENCE/('BB_COPY_'+host+'_'+Path(root).name+'.json'), result)
    print(json.dumps(result, indent=2))


def config(root, model):
    rows = {h: json.loads((EVIDENCE/('GPU_STAGE_'+h+'_'+Path(root).name+'.json')).read_text())
            for h in ('gpuA', 'gpuR')}
    if rows['gpuA']['engine'] != rows['gpuR']['engine']:
        raise ValueError('cross-GPU immutable engine specifications differ')
    path = EVIDENCE/(model.upper()+'_EXECUTION.json')
    if path.exists():
        raise ValueError('preserve existing GPU execution specification')
    out = dict(engine=rows['gpuA']['engine'], gpus={h: rows[h]['root'] for h in rows})
    write(path, out)
    print(json.dumps(dict(path=str(path), sha256=sha(path))))


def release_after_cpu_stop(config_path, host):
    """Apply the relay's existing global-stop release even if another CPU host is offline."""
    config_path = checked(config_path)
    config = json.loads(config_path.read_text())
    cpu_root = config['cpus'][host]
    proof = ssh(host, f'''import hashlib,json
from pathlib import Path
root=Path({cpu_root!r});assert root.name.startswith('c1switch2_run_')
path=root/'STOP';assert path.is_file(),'verified CPU STOP required before GPU release'
dispatch=json.loads((root/'DISPATCH_MODEL.json').read_text())
assert dispatch['arm']=={config['engine']['model']!r}
print(json.dumps(dict(root=str(root),stop=json.loads(path.read_text()),stop_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
''', timeout=30)
    releases = {}
    for gpu, root in config['gpus'].items():
        releases[gpu] = ssh(gpu, f'''import json,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_gpu_')
assert json.loads((root/'ENGINE.json').read_text())=={config['engine']!r}
dispatch=json.loads((root/'DISPATCH.json').read_text())
proc=Path('/proc',str(dispatch['pid']))
if proc.exists():
 command=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert str(root) in command and 'tools.c1switch2.gpuserver' in command,'owned GPU identity mismatch'
path=root/'rpc/COMPLETE'
if not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='verified owned CPU STOP; ordinary relay release',proof={proof!r},unix=time.time()),stream)
print(json.dumps(dict(root=str(root),release_requested=True,pid=dispatch['pid'])))
''', timeout=30)
    result = dict(unix=time.time(), model=config['engine']['model'], proof=proof,
                  releases=releases, config_sha256=sha(config_path), new_sim_s=0, new_gpu_allocations=0)
    write(EVIDENCE/('GPU_RELEASE_'+config['engine']['model']+'_'+str(int(result['unix']))+'.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('inspect', 'stage', 'cpu-probe', 'start', 'collect', 'copy-bb', 'config',
                                           'release-after-cpu-stop'))
    parser.add_argument('--host', choices=('gpuA', 'gpuR', 'cpuQ'), required=True)
    parser.add_argument('--config')
    parser.add_argument('--model', choices=('wm', 'bb'))
    parser.add_argument('--root')
    args = parser.parse_args()
    if args.command == 'inspect':
        inspect(args.host)
    elif args.command == 'stage':
        stage(args.host, args.model)
    elif args.command == 'cpu-probe':
        cpu_probe(args.host, args.root, args.model)
    elif args.command == 'start':
        start(args.host, args.root)
    elif args.command == 'copy-bb':
        copy_bb(args.host, args.root)
    elif args.command == 'config':
        config(args.root, args.model)
    elif args.command == 'release-after-cpu-stop':
        release_after_cpu_stop(args.config, args.host)
    else:
        collect(args.host, args.root)
