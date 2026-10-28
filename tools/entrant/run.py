"""Fixed WM2 factorial replay, keeping the original LRDIAG measurement contract."""
from dataclasses import replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HOME = Path.home()/'entrant'
VARIANTS = ('baseline', 'ramp_adapt_off', 'entrant_on', 'entrant_on_ramp_off')


def setup():
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', WMS_SPEED_COMPACT_SLOTS='1', WMS_SKIP_PIVOT_DIAG='1',
        WMS_UNIT_SEED='1', LRDIAG_TRACE_OFF='1')
    sys.path[:0] = [str(HOME/'runtime'), str(HOME/'runtime/src')]


def specs():
    root=Path.home()/'e3long/inputs'
    return {
        'mappo':dict(checkpoint=str(root/'mappo_v5_final.pt'),
            sha256='aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564'),
        'ours_c1':dict(checkpoint=str(root/'cycle1/policy_latest.pt'),
            sha256='64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c')}


def run(group, draw, variant, steps=600, panel='e3'):
    setup()
    import torch
    from tools.lrdiag import replay
    from tools import r10_attribution
    from wms.control.entrant import EntrantControllers
    replay.HOME=HOME
    old_configure=replay.configure
    old_load=r10_attribution.load_model
    old_call=EntrantControllers.__call__
    old_yielding=replay.yielding
    audit={'entrant_cav_steps':[], 'non_native_command_steps':[]}
    def load(*a, **kw):
        cfg, settings, model, report=old_load(*a, **kw)
        on=variant.startswith('entrant_on')
        return cfg,replace(settings,entrant_control=on,
            entrant_controllers=specs() if on else None),model,report
    def configure(model, ignored):
        old_configure(model,'ramp_adapt_off' if variant in
            ('ramp_adapt_off','entrant_on_ramp_off') else 'baseline')
    def audited(self, state):
        out=old_call(self,state)
        mask=state['entrant'].bool() & state['is_cav'].bool() & state['present'].bool()
        count=mask.sum(-1).cpu().tolist()
        if not audit['entrant_cav_steps']:
            audit['entrant_cav_steps']=[0]*len(count)
            audit['non_native_command_steps']=[0]*len(count)
        native=self.recorded._native_accel(state)
        diff=mask & (out[...,3] != native)
        for i,n in enumerate(count):
            audit['entrant_cav_steps'][i]+=n
            audit['non_native_command_steps'][i]+=int(diff[i].sum())
        return out
    r10_attribution.load_model=load
    replay.configure=configure
    def yielding():
        while (HOME/'YIELD').exists():
            time.sleep(15)
    replay.yielding=yielding  # marker-only coordination; never discover peer processes
    EntrantControllers.__call__=audited
    try:
        torch.manual_seed(140014+7919*group+1000003*draw)
        result=replay.run(group,draw,variant,steps,panel,
                          horizons=(10,30,60,90,120,180,240,300))
        result['controller_audit']=audit
        result['entrant_control']=variant.startswith('entrant_on')
        result['controller_specs']=specs() if result['entrant_control'] else None
        result['implementation_manifest_sha256']=hashlib.sha256((HOME/'SOURCE.json').read_bytes()).hexdigest()
        result['panel_manifest_sha256']=hashlib.sha256(
            (HOME/('e3data' if panel=='e3' else 'pairs300')/'C4_EPISODES_MANIFEST.json').read_bytes()).hexdigest()
        result['contract']='t0 recorded commands; entrant same-arm feedback; observable entry CAV distribution'
        return result
    finally:
        r10_attribution.load_model=old_load
        replay.configure=old_configure
        EntrantControllers.__call__=old_call
        replay.yielding=old_yielding


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',type=int,default=0);p.add_argument('--draw',type=int,default=0)
    p.add_argument('--variant',choices=VARIANTS,default='baseline')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--panel',choices=('e3','pairs300'),default='e3')
    p.add_argument('--repeat',action='store_true')
    a=p.parse_args()
    result=run(a.group,a.draw,a.variant,a.steps,a.panel)
    if a.repeat:
        again=run(a.group,a.draw,a.variant,a.steps,a.panel)
        # Timings alone are not scientific outputs.
        scientific=lambda r:{k:v for k,v in r.items() if k!='seconds'}
        assert scientific(result)==scientific(again),'identical weights did not rescore identically'
        result['repeat_identical']=True
    dest=HOME/'results_v4';dest.mkdir(exist_ok=True)
    path=dest/f'{a.panel}_g{a.group:02d}_{a.variant}_k{a.draw}_s{a.steps}.json'
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,allow_nan=False));tmp.replace(path)
    print(json.dumps({'result':str(path),'seconds':result['seconds'],
                      'repeat_identical':result.get('repeat_identical'), 'audit':result['controller_audit']}))


if __name__=='__main__':main()
