"""Stage immutable own code only; run CPU tests or launch one owned job."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tools.coopaudit.remote import run


def deploy(action):
    root=Path(__file__).resolve().parents[2]
    files=sorted((root/'tools/coopaudit').glob('*.py'))
    version=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()[:16]
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tf:
        for p in files:
            tf.add(p,arcname='tools/coopaudit/'+p.name,recursive=False)
    data=base64.b64encode(buffer.getvalue()).decode()
    code=f'''
import base64,io,json,os,subprocess,sys,tarfile,time
from pathlib import Path
root=Path('/root/coopaudit20261031')
assert root.parent.is_dir()
root.mkdir(exist_ok=True)
dest=root/'code_{version}'
if not dest.exists():
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode({data!r})),mode='r:gz') as tf:
        for item in tf.getmembers():
            assert item.name.startswith('tools/coopaudit/') and '..' not in item.name
        tf.extractall(dest)
env=dict(os.environ,PYTHONPATH=str(dest),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',C4_POLICY_STOCHASTIC='0',C4_POLICY_ACCEL_NOISE='0')
if {action!r}=='test':
    os.environ.update(env)
    sys.path.insert(0,str(dest))
    from tools.coopaudit import runner
    os.nice(19)
    runner.setup()
    import unittest
    suite=unittest.defaultTestLoader.loadTestsFromName('tools.coopaudit.test_cpu')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(version={version!r},tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),ok=result.wasSuccessful())
    (root/'CPU_TESTS.json').write_text(json.dumps(receipt,indent=1))
    print(json.dumps(receipt))
    assert result.wasSuccessful()
elif {action!r}=='stage':
    print(json.dumps(dict(version={version!r},code=str(dest))))
else:
    assert not (root/'STOP').exists()
    log=(root/({action!r}+'_'+str(int(time.time()))+'.log')).open('x')
    proc=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-m','tools.coopaudit.runner',{action!r}],cwd=dest,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=proc.pid,version={version!r},code=str(dest),action={action!r},unix=time.time(),proc_start_ticks=Path('/proc/'+str(proc.pid)+'/stat').read_text().split()[21])
    (root/({action!r}+'_PID.json')).write_text(json.dumps(receipt,indent=1))
    print(json.dumps(receipt))
'''
    result=run('gpuR',code,timeout=120)
    print(result.stdout,end='')
    print(result.stderr,end='')
    if result.returncode:
        raise SystemExit(result.returncode)
    out=root/'tools/coopaudit/evidence'/f'{action}_deploy.json'
    out.write_text(result.stdout)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('stage','test','canary','census','main'))
    deploy(p.parse_args().action)
