"""Trace first paired update-zero divergence in the owned offline probe only."""
import subprocess

SCRIPT = r'''
import json,os,pathlib,sys
r=pathlib.Path('/root/autodl-tmp/c4/nativeinit_s1')
d=json.loads(pathlib.Path('/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/CONFIG.json').read_text())
os.environ.update(d['policy']['env'])
os.environ.update(C4_GPU_MEM_FRACTION='0.40',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
os.environ['C4_ALLOWED_ROOTS']=':'.join([str(r),d['loop_root'],*d['policy']['allowed_roots']])
sys.path[:0]=[str(r/'runtime_tests/src'),str(r/'runtime_tests')]
os.chdir(r/'runtime_tests')
import torch
import wms.control.policy14_cli as cli
from wms.control.counterfactual import build_variants
def score(trainer,val,panel,device):
    seed=965149
    torch.manual_seed(seed)
    s=panel['starts'][0]
    batch=val._batch([val._load(int(s['index']))],device)
    batch=trainer._entry_slots(batch,torch.Generator().manual_seed(seed))
    variants=build_variants(batch.state,trainer.counterfactual,torch.Generator().manual_seed(seed))
    base=variants.base_row[0]
    records=[]
    model=trainer.world_models[0]
    original=model.rollout
    def rollout(state,**kw):
        callback=kw['cav_controller']
        def traced(st):
            differences={}
            for k,v in st.items():
                if torch.is_tensor(v) and v.ndim and v.shape[0]==variants.n_rows:
                    a,b=v[0],v[base]
                    if a.is_floating_point():
                        same=torch.allclose(a,b,rtol=0,atol=0,equal_nan=True)
                        if not same: differences[k]=float(torch.nan_to_num(a-b).abs().max())
                    elif not torch.equal(a,b): differences[k]=int((a!=b).sum())
            out=callback(st)
            records.append({'tick':len(records),'state_differences':differences,
                            'motion_maxdiff':float((out[0]-out[base]).abs().max()),
                            'command_maxdiff':float((out[0,...,3]-out[base,...,3]).abs().max())})
            if hasattr(callback,'native_lane_mask'): traced.native_lane_mask=callback.native_lane_mask
            return out
        if hasattr(callback,'native_lane_mask'): traced.native_lane_mask=callback.native_lane_mask
        kw['cav_controller']=traced
        return original(state,**kw)
    model.rollout=rollout
    with torch.no_grad():
        result=trainer.imagine(batch,deterministic=True,variants=variants,future_seed=seed)
    components={k:float((v[...,0]-v[...,base]).abs().max()) for k,v in result['components'].items()
                if torch.is_tensor(v) and v.ndim==3}
    print(json.dumps({'component_maxdiff':components}))
    (r/'TRACE_ZERO.json').write_text(json.dumps({'base':base,'n_rows':variants.n_rows,'records':records},indent=1))
    print(json.dumps({'base':base,'n_rows':variants.n_rows,'records':records}))
    raise RuntimeError('trace complete: no updates requested')
cli.score_native_panel=score
argv=json.loads((r/'MEASURE_PID.json').read_text())['argv'][3:]
argv[argv.index('--output-dir')+1]=str(r/'trace0')
cli.main(argv)
'''
if __name__ == '__main__':
    import sys
    script=SCRIPT
    if '--crn' in sys.argv:
        script=script.replace('runtime_tests','runtime_measure_crn').replace("r/'trace0'", "r/'trace0_crn'")
        script=script.replace('TRACE_ZERO.json','TRACE_ZERO_CRN.json')
    result = subprocess.run(['ssh','-o','BatchMode=yes','gpuR','/root/miniconda3/bin/python -'],
                            input=script.encode(),capture_output=True,check=True)
    print(result.stdout.decode())
    print(result.stderr.decode())
