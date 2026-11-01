"""Source-bound warm batched/single gate before the persistent worker is READY."""
from pathlib import Path
from unittest.mock import patch
import time
from .storage import read,write


def main(root):
    import torch
    from tools.wmteach import runtime
    from .plan_service import configured_runtime,batch_width,main as serve
    from .wm import cost_adapter
    from .plan_wm import evaluate
    from .speed2_memory import native_fixture
    from .mppi_search import group_components,sample_unscored,comparators
    binding=read(root/'GPU_BINDING.json')
    rt=configured_runtime(runtime.bootstrap,binding.get('memory_fraction',.65))
    original=rt.frozen.costs
    try:
        rt.frozen.costs=cost_adapter(original,read(root/'FROZEN_COSTS.json')['code'],horizons=(60,120,300,360,420,600))
        record,memory=native_fixture(rt)
        group=max(group_components(record['rows']),key=lambda g:g['size'])
        bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
        acts=comparators(record,600)+[sample_unscored(group,[[0.]*300]*group['size'],bounds,'ordinary',20261101,duration_s=300)]
        for k,h in ((4,120),(8,300)):
            width=batch_width(binding,k);begin=time.monotonic()
            single=evaluate(rt,record,acts,k,h,chunk_actions=1,pad_actions=width,trim=False,speed2=True,transfers=True,memory=memory)
            batch=evaluate(rt,record,acts,k,h,chunk_actions=width,pad_actions=width,trim=False,speed2=True,transfers=True,memory=memory)
            error=max(abs(a['team_delay']-b['team_delay']) for key in single for a,b in zip(single[key],batch[key]))
            draws=all(a['draw_id']==b['draw_id'] for key in single for a,b in zip(single[key],batch[key]))
            value=dict(pass_=error==0 and draws,max_cost_error=error,identical_draw_ids=draws,K=k,H=h,width=width,
                group_size=group['size'],wall_s=time.monotonic()-begin,new_sumo_sim_s=0)
            write(root/f'STARTUP_WARM_BATCH_SINGLE_K{k}_H{h}.json',value)
            if not value['pass_']:raise ValueError('warm batched/single exact parity failed')
    finally:rt.frozen.costs=original
    # Reuse this already admitted model. The normal worker then checks full vs
    # fast cold/warm trajectories and only afterwards announces READY.
    with patch.object(runtime,'bootstrap',return_value=rt):serve(root)


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
