"""One-use CPU extraction and verified collection of final scoring-only targets."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from tools.rbint.remote import ssh

REMOTE = '/root/autodl-tmp/c4/rb2int20261101/factual50_v1'
CONTROL = '/root/autodl-tmp/c4/rb2int20261101/phys_code_d0df3d66e80a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('launch','status','download'))
    parser.add_argument('--receipt',required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    receipt = Path(args.receipt)
    if not receipt.parent.is_dir() or receipt.exists():
        raise ValueError('exclusive target operation receipt required')
    if args.action == 'launch':
        source = Path(__file__).with_name('factual_endpoints.py').read_bytes()
        trailer = b'''\nif __name__ == '__main__':
    _root=Path(__file__).parent
    with (_root/'EXTRACT_DONE.json').open('x') as _stream:
        json.dump(dict(passed=True,sha256=sha(_root/'FACTUAL50.json')),_stream)
'''
        code = f'''
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path({REMOTE!r});assert root.parent.is_dir() and not root.exists()
assert shutil.disk_usage(root.parent).free>512*2**20 and os.getloadavg()[0]<os.cpu_count()/2
root.mkdir()
with (root/'EXTRACT.py').open('xb') as f: f.write({source+trailer!r})
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(root/'EXTRACT.py'),'--starts','/root/autodl-tmp/c4/rb2int20261101/export_prefix_all58_v1/inputs58/data/jobs','--out',str(root/'FACTUAL50.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RB2INT_CONTROL_ROOT={CONTROL!r})
with (root/'worker.log').open('x') as log:
 child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
fields=Path('/proc/'+str(child.pid)+'/stat').read_text().split()
result=dict(pid=child.pid,start_ticks=fields[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=argv,root=str(root),cpu_only=True,source_sha256={hashlib.sha256(source).hexdigest()!r},scoring_only=True,model_inputs=False)
with (root/'PROCESS.json').open('x') as f: json.dump(result,f,indent=1)
print(json.dumps(result))
'''
        result = json.loads(ssh('gpuR',code,timeout=45))
    elif args.action == 'status':
        code = f'''
from pathlib import Path
import json
root=Path({REMOTE!r});result=dict(root=str(root),exists=root.is_dir())
if root.is_dir():
 process=json.loads((root/'PROCESS.json').read_text());result['process']=process
 try:
  f=Path('/proc/'+str(process['pid'])+'/stat').read_text().split();result['alive']=f[2]!='Z' and f[21]==process['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']
 except FileNotFoundError: result['alive']=False
 if (root/'EXTRACT_DONE.json').is_file(): result['completion']=json.loads((root/'EXTRACT_DONE.json').read_text())
 if (root/'worker.log').is_file(): result['log_tail']=(root/'worker.log').read_text()[-2500:]
print(json.dumps(result))
'''
        result = json.loads(ssh('gpuR',code,timeout=30))
    else:
        out = Path(args.out) if args.out else None
        if out is None or not out.parent.is_dir() or out.exists():
            raise ValueError('exclusive factual target output required')
        status = json.loads(ssh('gpuR',f"from pathlib import Path; import json; print((Path({REMOTE!r})/'EXTRACT_DONE.json').read_text())",timeout=30))
        if not status['passed']:
            raise ValueError('target extraction not complete')
        code = f'''from pathlib import Path
import hashlib,sys
raw=(Path({REMOTE!r})/'FACTUAL50.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=={status['sha256']!r};sys.stdout.buffer.write(raw)'''
        with out.open('xb') as stream:
            completed = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpuR',
                '/root/miniconda3/bin/python','-B','-c',shlex.quote(code)],stdout=stream,stderr=subprocess.PIPE,timeout=120)
        if completed.returncode:
            raise RuntimeError('partial target transfer preserved: '+completed.stderr.decode())
        actual = hashlib.sha256(out.read_bytes()).hexdigest()
        data = json.loads(out.read_text())
        if actual != status['sha256'] or len(data['episodes']) != 250:
            raise ValueError('factual target transfer/count differs')
        result = dict(out=str(out),sha256=actual,episodes=250,scoring_only=True,model_inputs=False)
    with receipt.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=1)
    print(json.dumps(result,indent=1))


if __name__ == '__main__':
    main()
