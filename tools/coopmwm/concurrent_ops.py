"""Explicit inference-only source upgrade for MAIN02:43; never launches SUMO."""
import argparse
import json
import time
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha

CPU='/home/ubuntu/coopmwm20261102_concurrentcpu4'
GPU='/root/coopmwm20261102_concurrent1'


def stage(host):
    from .warm_remote_ops import gpu_status
    old=gpu_status(host)
    assert old['alive'] and old['ready'] and not old['failed']
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_concurrentcpu4.json')
    assert cpu['pass_'] and cpu['tests']==164 and cpu['sim_s']==cpu['skips']==0
    staged=json.loads(ops.stage_frozen(host,GPU,cpu))
    extra={'INPUTS.json':ops.ssh.remote(host,"import pathlib;print(pathlib.Path('/root/coopmwm20261101_planprobe3/INPUTS.json').read_text())",timeout=30)}
    hashes=dict(staged['hashes'],**json.loads(ops.stage(host,GPU,extra))['hashes'])
    original=read(ops.EVIDENCE/f'WARM_DURABLE_STAGE_{host}.json')['binding']
    b=dict(original,hashes=hashes,concurrent_plans=True)
    ops.stage(host,GPU,{'GPU_BINDING.json':json.dumps(b,sort_keys=True)})
    value=dict(host=host,root=GPU,binding=b,cpu=cpu,adopted_original=old,new_sumo_sim_s=0)
    path=ops.EVIDENCE/f'CONCURRENT_STAGE_{host}.json';write(path,value)
    line='\nS-COOPMWM MAIN0243 SOURCE QUALIFICATION '+host+' receiptSHA'+sha(path)+': CPU164PASS/zeroSUMO; new remaining-commitment WM context and decoder require source-bound GPU parity; original READY workers preserved idle, no old inference repeated. New source adds K4H300/K8H420 background full-fast exact gates before READY; no SUMO or search-design change.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    print(json.dumps(dict(host=host,staged=True,new_sumo_sim_s=0)))


def dispatch(host):
    assert read(ops.EVIDENCE/f'CONCURRENT_STAGE_{host}.json')['root']==GPU
    from . import warm_remote_ops as old
    old.GPU=GPU
    # The same one-use detach/identity machinery, with the new stage receipt.
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,subprocess,os,time
r=pathlib.Path({GPU!r});assert r.is_dir() and not (r/'DISPATCH.json').exists() and not (r/'SPOOL_OWNER.json').exists()
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
if {host!r}=='gpuR':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.warm_spool',str(r)]
with (r/'SPOOL.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),root=str(r),host={host!r})
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=40))
    write(ops.EVIDENCE/f'CONCURRENT_DISPATCH_{host}.json',value);print(json.dumps(value))


def status(host):
    from . import warm_remote_ops as old
    old.GPU=GPU
    return old.gpu_status(host)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','dispatch','status'));p.add_argument('--host',choices=('gpuA','gpuR'),required=True)
    a=p.parse_args();globals()[a.mode](a.host)
