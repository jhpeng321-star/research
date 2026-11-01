"""Explicit own-root development dispatch. No peer signalling; no reserved task reads."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from tools.rbint.remote import ssh

PARENT = '/root/rbeval20261031'
JOB = PARENT+'/rb2break'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'launch', 'status', 'fetch', 'baseline'))
    parser.add_argument('--host', default='gpuA', choices=('gpuA', 'gpuR'))
    parser.add_argument('--runtime')
    parser.add_argument('--factor', default='baseline')
    parser.add_argument('--worlds', nargs='+', type=int, default=list(range(969420, 969432)))
    parser.add_argument('--futures', nargs='+', type=int, default=[0])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--tag', default='screen')
    args = parser.parse_args()
    if not set(args.worlds) <= set(range(969420, 969432)) or not args.worlds:
        raise ValueError('development only')
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('simple unique tag required')
    root = Path(__file__).resolve().parents[2]
    if args.action == 'stage':
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        paths = ['src/wms', 'tools/rbint', 'tools/rbeval', 'tools/rb2break', 'tools/rbfit',
            'configs/execparity_cav_idm_v1.json', 'configs/native_cav_head_v5.json', 'configs/wm_phys.toml',
            'artifacts/rbfit/shrinkage_v1', 'docs/paper_results/rbunc/observed_arrivals.json',
            'tests/test_rb2break.py']
        archive = subprocess.check_output(['git', 'archive', '--format=tar', commit, *paths], cwd=root)
        sha = hashlib.sha256(archive).hexdigest()
        dest = PARENT+'/rb2break_code_'+commit[:12]
        code = f'''import sys,io,tarfile,hashlib,json
from pathlib import Path
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=={sha!r}
parent=Path({PARENT!r});assert parent.is_dir()
dest=Path({dest!r});assert not dest.exists(), 'new immutable stage only'
dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 for member in tar.getmembers():
  assert not member.issym() and not member.islnk() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
 tar.extractall(dest,filter='data')
receipt=dict(owner='S-RB2-BREAK',commit={commit!r},archive_sha256={sha!r},runtime=str(dest))
with (dest/'STAGE.json').open('x') as f: json.dump(receipt,f,indent=1)
print(json.dumps(receipt))'''
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
            '/root/miniconda3/bin/python', '-B', '-c', shlex.quote(code)],
            input=archive, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stdout.decode()+result.stderr.decode())
        print(result.stdout.decode())
        return
    common = f'''from pathlib import Path
import json,os,subprocess,time,hashlib
parent=Path({PARENT!r});root=Path({JOB!r})
assert parent.is_dir()
def save(path,data):
 with path.open('x') as f: json.dump(data,f,indent=1)
'''
    out_name = f'{args.tag}_{args.factor}_s{args.shard}'
    if args.action == 'launch':
        if args.factor not in ('baseline', 'no_mainline_lc', 'no_merge_advice', 'depart_last',
                               'depart_first', 'population', 'point_v0', 'point_tau', 'fixed_rate'):
            raise ValueError('unregistered factor')
        if not args.runtime or not args.runtime.startswith(PARENT+'/rb2break_code_'):
            raise ValueError('own immutable runtime required')
        code = f'''
runtime=Path({args.runtime!r});assert (runtime/'STAGE.json').is_file()
resources=dict(load=os.getloadavg(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
assert resources['load'][0]<16, 'host busy'
assert int(resources['gpu'].split(',')[0])>12000, 'leave shared GPU headroom'
root.mkdir(exist_ok=True)
out=root/{out_name!r};assert not out.exists(), 'never duplicate a dispatch'
out.mkdir()
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(runtime/'tools/rb2break/run.py'),
 '--out',str(out),'--runtime-root',str(runtime),'--wm-config',str(runtime/'configs/wm_phys.toml'),
 '--model-name',{('RB2BREAK_'+args.tag+'_'+args.factor)!r},'--factor',{args.factor!r},
 '--dev-worlds',*map(str,{args.worlds!r}),'--futures',*map(str,{args.futures!r}),
 '--shard',{str(args.shard)!r},'--shards',{str(args.shards)!r},'--memory-gib','5']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',
 PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
with (out/'worker.log').open('x') as log:
 p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=runtime,start_new_session=True)
receipt=dict(owner='S-RB2-BREAK',pid=p.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],argv=argv,resources=resources,unix=time.time())
save(out/'DISPATCH.json',receipt);print(json.dumps(receipt))
'''
    elif args.action == 'status':
        code = '''
print('LOAD',os.getloadavg())
print(subprocess.check_output(['nvidia-smi'],text=True))
if root.is_dir():
 for out in sorted(root.iterdir()):
  if not out.is_dir(): continue
  print('JOB',out.name,'tasks',len(list(out.glob('w*_k*.json'))),'complete',(out/'COMPLETE.json').exists())
  dispatch=out/'DISPATCH.json'
  if dispatch.exists():
   receipt=json.loads(dispatch.read_text());proc=Path('/proc')/str(receipt['pid'])/'stat'
   alive=proc.exists() and proc.read_text().split()[2]!='Z' and proc.read_text().split()[21]==receipt['start_ticks']
   print('OWN_PID',receipt['pid'],'alive',alive)
  log=out/'worker.log'
  if log.exists(): print(log.read_text()[-3500:])
'''
    elif args.action == 'baseline':
        code = '''
source=parent/'int/WM-PHYS-FULL-v4-gpu'
assert source.is_dir()
root.mkdir(exist_ok=True)
out=root/'frozen_baseline';assert not out.exists();out.mkdir()
manifest={}
for seed in range(969420,969432):
 for future in range(8):
  name=f'w{seed}_k{future}.json'
  files=list(source.glob('shard*/'+name));assert len(files)==1
  path=files[0];raw=path.read_bytes();task=json.loads(raw)
  assert task['seed']==seed and task['future']==future
  identity=json.loads((path.parent/'IDENTITY.json').read_text())
  assert identity['model']=='WM-PHYS-FULL-v4-gpu' and identity['K']==8
  assert hashlib.sha256((path.parent/'IDENTITY.json').read_bytes()).hexdigest()==task['identity_sha256']
  with (out/name).open('xb') as f: f.write(raw)
  manifest[name]=dict(source=str(path),sha256=hashlib.sha256(raw).hexdigest())
save(out/'BASELINE.json',dict(owner='S-RB2-BREAK',development_only=True,source='FULL-v4/22cc32f6',files=manifest))
print(json.dumps(dict(tasks=len(manifest),development_only=True)))
'''
    else:
        target = root/'docs/rb2break'/args.tag
        if not target.parent.is_dir():
            raise ValueError('local evidence parent missing')
        target.mkdir(exist_ok=True)
        subprocess.run(['scp', '-r', '-o', 'BatchMode=yes', f'{args.host}:{JOB}/{out_name}', str(target)],
                       check=True, timeout=600)
        return
    print(ssh(args.host, common+code, timeout=120))


if __name__ == '__main__':
    main()
