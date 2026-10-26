"""Verify the measured checkpoint on the delivered numerical path, without an update."""
import subprocess

SCRIPT = r'''
import hashlib,json,os,pathlib,sys
r=pathlib.Path('/root/autodl-tmp/c4/nativeinit_s1')
d=json.loads(pathlib.Path('/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/CONFIG.json').read_text())
os.environ.update(d['policy']['env'])
os.environ.update(C4_GPU_MEM_FRACTION='0.40',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
os.environ['C4_ALLOWED_ROOTS']=':'.join([str(r),d['loop_root'],*d['policy']['allowed_roots']])
runtime=r/'runtime_verified'
sys.path[:0]=[str(runtime/'src'),str(runtime)]
os.chdir(runtime)
import wms.control.policy14_cli as cli
summary=json.loads((r/'measure30_final/RUN_SUMMARY.json').read_text())
assert summary['status']=='completed'
expected=next(row for row in summary['native_curve'] if row['update']==summary['recommended_updates_per_cycle'])
original=cli.score_native_panel
def score(*args,**kwargs):
    result=original(*args,**kwargs)
    assert result==expected, 'delivered numerical path differs from measured checkpoint'
    source_names=['src/wms/control/'+n+'.py' for n in ('policy','bridge','dreamer','nativeinit','policy14','policy14_cli','confident_gate')]
    receipt={'status':'PASS','result':result,'actor_updates':0,'sumo_seconds':0,
             'checkpoint_sha256':hashlib.sha256((r/'measure30_final/policy_latest.pt').read_bytes()).hexdigest(),
             'delivered_runtime_source_sha256':{n:hashlib.sha256((runtime/n).read_bytes()).hexdigest() for n in source_names}}
    (r/'DELIVERED_RESCORE.json').write_text(json.dumps(receipt,indent=1))
    print(json.dumps(receipt))
    raise RuntimeError('expected offline rescore stop before optimizer updates')
cli.score_native_panel=score
argv=json.loads((r/'MEASURE_FINAL_PID.json').read_text())['argv'][3:]
argv[argv.index('--output-dir')+1]=str(r/'delivered_rescore')
argv += ['--resume',str(r/'measure30_final/policy_latest.pt'),'--critic-warmup-updates','0']
cli.main(argv)
assert (r/'DELIVERED_RESCORE.json').is_file()
'''

if __name__ == '__main__':
    result = subprocess.run(['ssh','-o','BatchMode=yes','gpuR','/root/miniconda3/bin/python -'],
                            input=SCRIPT.encode(),capture_output=True,check=True)
    print(result.stdout.decode())
    print(result.stderr.decode())
