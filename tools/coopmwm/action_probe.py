"""Trace actual frozen WM controller inputs/outputs without SUMO or truth."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
from .storage import read,write


def run(root):
    import torch
    from tools.wmteach import runtime
    from .wm import adapted_futures,candidates,proposal_candidates,executor_adapter,cost_adapter
    rt=runtime.bootstrap()
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    record=read(root/'INPUT.json')
    from tools.coopfree.policy import requests
    from wms.control import safety
    real_safe=safety.safe_accel
    original=runtime.option_controller
    traces=[]
    def controller(mp,acts,rows):
        ids=[r['vid'] for r in rows];tick=0
        def safe(state,accel,**kw):
            out=real_safe(state,accel,**kw)
            if tick<24:
                for b,a in enumerate(acts):
                    for vid in record['group']:
                        i=ids.index(vid)
                        traces.append(dict(stage='safety',tick=tick,action=a['id'],vid=vid,
                            bounded_raw=float(accel[b,i]),safe=float(out[b,i])))
            return out
        with patch.object(safety,'safe_accel',safe):
            cb=original(mp,acts,rows)
        def wrapped(state):
            nonlocal tick
            out=cb(state)
            if tick<24:
                for b,a in enumerate(acts):
                    for vid in record['group']:
                        i=ids.index(vid)
                        traces.append(dict(stage='trajectory',tick=tick,t_s=tick*.5,action=a['id'],vid=vid,slot=i,
                            present=bool(state['present'][b,i]),is_cav=bool(state['is_cav'][b,i]),
                            s_m=float(state['s_m'][b,i]),v_mps=float(state['v_mps'][b,i]),
                            next_s_m=float(out[b,i,0]),next_v_mps=float(out[b,i,2]),
                            accel_mps2=float(out[b,i,3])))
            tick+=1
            return out
        wrapped.native_lane_mask=cb.native_lane_mask
        return wrapped
    for mode in ('credit','options'):
        acts=candidates(record) if mode=='credit' else proposal_candidates(dict(record,setting=0))
        ns=adapted_futures((60,));ns['_actions']=acts
        with patch.object(runtime,'option_controller',controller),executor_adapter():
            result=ns['futures'](rt,dict(record,setting=0),indices=(0,))
        write(root/(mode.upper()+'.json'),dict(actions=acts,traces=traces,result=result))
        means={a:fs[0]['team_delay'] for a,fs in result['futures']['60'].items()}
        print(json.dumps(dict(mode=mode,actions=[a['id'] for a in acts],means_H60=means,trace_rows=len(traces))),flush=True)
        traces=[]
    write(root/'COMPLETE.json',dict(pass_=True,new_sumo_sim_s=0))


if __name__=='__main__':run(Path(sys.argv[1]))
