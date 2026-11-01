"""Own one-use causal-prefix exporter and hash-verified archive transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from tools.rbint.remote import ssh
from tools.rb2int.audit import ROOT

PARENT = '/root/autodl-tmp/c4/rb2int20261101'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export', 'status', 'download', 'upload', 'view', 'manifest'))
    parser.add_argument('--host', choices=('gpuA', 'gpuR'), required=True)
    parser.add_argument('--tag', default='prefix_all58_v1')
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--local-archive')
    parser.add_argument('--archive-sha256')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]{1,40}', args.tag) or any(
            s in args.tag for s in ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1')):
        raise ValueError('explicit safe own export tag required')
    receipt = Path(args.receipt)
    if not receipt.parent.is_dir() or receipt.exists():
        raise ValueError('exclusive local operation receipt required')
    root = PARENT+'/export_'+args.tag
    if args.action == 'export':
        if args.host != 'gpuR':
            raise ValueError('recorded WMDATA2 source is gpuR; do not guess another corpus')
        registration = (ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json').read_bytes()
        metadata = (ROOT/'docs/rb2int/PANEL_METADATA.json').read_bytes()
        source = Path(__file__).with_name('prepare_inputs.py').read_bytes()
        trailer = b'''\nif __name__ == '__main__':
    _root=Path(__file__).parent
    save(_root/'EXPORT_DONE.json', dict(passed=True, archive_sha256=sha(_root/'PREFIX58.tar'),
        archive_bytes=(_root/'PREFIX58.tar').stat().st_size,
        inputs_build_sha256=sha(_root/'inputs58/INPUTS_BUILD.json')))
'''
        code = f'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
parent=Path({PARENT!r});assert parent.parent.is_dir()
assert shutil.disk_usage(parent if parent.exists() else parent.parent).free>512*2**20
parent.mkdir(exist_ok=True);root=Path({root!r});assert not root.exists(), 'preserve prior export claim/job'
assert os.getloadavg()[0]<os.cpu_count()/2
cg=Path('/sys/fs/cgroup');maximum=cg/'cpu.max';stat=cg/'cpu.stat'
if maximum.is_file() and maximum.read_text().split()[0]!='max':
 quota,period=map(int,maximum.read_text().split());capacity=quota/period
 def usage(): return int(dict(line.split() for line in stat.read_text().splitlines())['usage_usec'])
 begin=time.monotonic();before=usage();time.sleep(1);used=(usage()-before)/1e6/(time.monotonic()-begin)
 assert used+1<.8*capacity, 'wait for spare prefix-export CPU quota'
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True);assert gpu.returncode==0
root.mkdir()
def save(path,data):
 with path.open('x') as f: json.dump(data,f,indent=1)
save(root/'CLAIM_ONCE.json',dict(owner='S-RB2-INT',source_sha256={hashlib.sha256(source).hexdigest()!r},registration_sha256={hashlib.sha256(registration).hexdigest()!r},worlds=58,final_outcomes_exported=False,automatic_retry=False,cuda_hidden=True,gpu_read_only=gpu.stdout.strip()))
for name,data in [('REGISTRATION.json',{registration!r}),('METADATA.json',{metadata!r}),('EXPORT.py',{source+trailer!r})]:
 with (root/name).open('xb') as f: f.write(data)
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(root/'EXPORT.py'),'--registration',str(root/'REGISTRATION.json'),'--metadata',str(root/'METADATA.json'),'--source-jobs','/root/wmdata2/jobs','--legacy-jobs','/root/wmcalib/data/jobs','--out',str(root/'inputs58'),'--archive',str(root/'PREFIX58.tar'),'--scope','all58']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'worker.log').open('x') as log:
 child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
fields=Path('/proc/'+str(child.pid)+'/stat').read_text().split()
result=dict(root=str(root),pid=child.pid,start_ticks=fields[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=argv,source_sha256={hashlib.sha256(source).hexdigest()!r})
save(root/'PROCESS.json',result);print(json.dumps(result))
'''
        result = json.loads(ssh(args.host, code, timeout=45))
    elif args.action == 'manifest':
        base = PARENT+'/received_'+args.tag if args.host == 'gpuA' else root+'/inputs58'
        code = f'''
import hashlib,json
from pathlib import Path
root=Path({base!r});path=root/'INPUTS_BUILD.json';raw=path.read_bytes();build=json.loads(raw)
assert build['passed'] and build['final_outcomes_exported'] is False
print(json.dumps(dict(root=str(root),inputs_build_sha256=hashlib.sha256(raw).hexdigest(),build=build)))
'''
        result = json.loads(ssh(args.host, code, timeout=30))
    elif args.action == 'view':
        source = Path(__file__).with_name('prepare_inputs.py').read_text(encoding='utf-8')
        base = PARENT+'/received_'+args.tag if args.host == 'gpuA' else root+'/inputs58'
        destination = PARENT+'/view_new50_'+args.tag
        code = f'''
import json,os
os.nice(19)
namespace={{'__name__':'rb2_causal_view'}}
exec(compile({source!r},'<trusted causal view>','exec'),namespace)
result=namespace['make_view']({base!r},{destination!r},list(range(969550,969600)))
result['root']={destination!r}
print(json.dumps(result))
'''
        result = json.loads(ssh(args.host, code, timeout=45))
    elif args.action == 'status':
        code = f'''
import hashlib,json
from pathlib import Path
root=Path({root!r});result=dict(root=str(root),exists=root.is_dir())
if root.is_dir():
 process=json.loads((root/'PROCESS.json').read_text());result['process']=process
 try:
  f=Path('/proc/'+str(process['pid'])+'/stat').read_text().split()
  result['alive']=f[2]!='Z' and f[21]==process['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']
 except FileNotFoundError: result['alive']=False
 done=root/'EXPORT_DONE.json'
 if done.is_file(): result['completion']=json.loads(done.read_text())
 log=root/'worker.log'
 if log.is_file(): result['log_tail']=log.read_text()[-2500:]
print(json.dumps(result))
'''
        result = json.loads(ssh(args.host, code, timeout=30))
    else:
        archive = Path(args.local_archive) if args.local_archive else None
        expected = args.archive_sha256
        if archive is None or not archive.parent.is_dir() or not expected or not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('explicit local archive parent and expected SHA256 required')
        if args.action == 'download':
            if archive.exists():
                raise ValueError('preserve existing transfer bytes; inspect receipt before recovery')
            script = f'''import hashlib,json,sys
from pathlib import Path
root=Path({root!r});done=json.loads((root/'EXPORT_DONE.json').read_text());assert done['passed'] and done['archive_sha256']=={expected!r}
data=(root/'PREFIX58.tar').read_bytes();assert hashlib.sha256(data).hexdigest()=={expected!r};sys.stdout.buffer.write(data)'''
            with archive.open('xb') as stream:
                completed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
                    '/root/miniconda3/bin/python', '-B', '-c', shlex.quote(script)], stdout=stream, stderr=subprocess.PIPE, timeout=120)
            if completed.returncode:
                raise RuntimeError('transfer failed; partial bytes preserved: '+completed.stderr.decode())
            actual = hashlib.sha256(archive.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('causal archive transfer SHA mismatch; bytes preserved')
            result = dict(action='download', host=args.host, archive=str(archive), sha256=actual, bytes=archive.stat().st_size)
        else:
            data = archive.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('local causal archive SHA differs')
            destination = PARENT+'/received_'+args.tag
            script = f'''import hashlib,io,json,shutil,sys,tarfile
from pathlib import Path
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=={expected!r}
parent=Path({PARENT!r});assert parent.is_dir();assert shutil.disk_usage(parent).free>max(512*2**20,3*len(data))
root=Path({destination!r});assert not root.exists(), 'preserve existing input receipt';root.mkdir()
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 for m in tar.getmembers():
  assert m.isfile() and not m.issym() and not m.islnk() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 tar.extractall(root,filter='data')
build=json.loads((root/'INPUTS_BUILD.json').read_text());assert build['passed'] and build['final_outcomes_exported'] is False
for name,sha in json.loads((root/'expected_hashes.json').read_text()).items(): assert hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()==sha
receipt=dict(root=str(root),archive_sha256={expected!r},worlds=len(build['worlds']),inputs_build_sha256=hashlib.sha256((root/'INPUTS_BUILD.json').read_bytes()).hexdigest(),final_outcomes_exported=False)
with (root/'RECEIVED.json').open('x') as f: json.dump(receipt,f,indent=1)
print(json.dumps(receipt))'''
            completed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
                '/root/miniconda3/bin/python', '-B', '-c', shlex.quote(script)], input=data,
                capture_output=True, timeout=120)
            if completed.returncode:
                raise RuntimeError(completed.stderr.decode()+completed.stdout.decode())
            result = json.loads(completed.stdout)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(dict(result, action=args.action, host=args.host), stream, indent=1)
    if args.action == 'manifest':
        print(json.dumps(dict(root=result['root'],inputs_build_sha256=result['inputs_build_sha256'],
            worlds=len(result['build']['worlds']),passed=result['build']['passed']),indent=1))
    else:
        print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
