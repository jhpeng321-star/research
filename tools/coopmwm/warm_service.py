"""Source-bound warm batched/single gate before the persistent worker is READY."""
from pathlib import Path
from unittest.mock import patch
import time
from .storage import read,write


ADOPTABLE=('STARTUP_COMMITTED_K4_H300.json','STARTUP_COMMITTED_K8_H420.json',
           'STARTUP_WARM_BATCH_SINGLE_K4_H120.json','STARTUP_WARM_BATCH_SINGLE_K8_H300.json')
NUMERICAL=tuple('pytools/tools/coopmwm/'+n+'.py' for n in ('plan_wm','speed2_runtime','speed2_transfers',
    'speed2_memory','policy_wm','mppi_search','mppi_runtime','wm','committed','overlap','plan_service',
    'plan_proposer','warm_runtime','warm_history'))+('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py',
    'pytools/tools/wmteach/runtime.py','pytools/tools/coopnative/policy.py')


def adopted_gates(root,binding):
    """Reuse completed exact gates only under source/SHA-bound engineering admission."""
    from .protocol import sha
    adopted=binding.get('adopted_startup_gates')
    if not adopted:return False
    if set(adopted['receipts'])!=set(ADOPTABLE):raise ValueError('incomplete adopted startup gate set')
    if set(adopted['numerical_hashes'])!=set(NUMERICAL):raise ValueError('incomplete adopted numerical source set')
    for name,h in adopted['receipts'].items():
        r=read(root/name)
        if sha(root/name)!=h or not r['pass_'] or r['max_cost_error']!=0.:
            raise ValueError('adopted startup receipt mismatch')
    for name,h in adopted['numerical_hashes'].items():
        if binding['hashes'].get(name)!=h:raise ValueError('adopted numerical source changed')
    return True


def main(root):
    import os
    binding=read(root/'GPU_BINDING.json')
    if adopted_gates(root,binding):
        from .plan_service import main as serve
        # Fresh process still runs all four cold/warm full-vs-fast gates before
        # READY. Only the four already completed same-source checks are adopted.
        return serve(root)
    if binding.get('expandable_segments'):os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    import torch
    from tools.wmteach import runtime
    from .plan_service import configured_runtime,batch_width,main as serve
    from .wm import cost_adapter
    from .plan_wm import evaluate
    from .speed2_memory import native_fixture
    from .mppi_search import group_components,sample_unscored,comparators
    rt=configured_runtime(runtime.bootstrap,binding.get('memory_fraction',.65))
    original=rt.frozen.costs
    try:
        rt.frozen.costs=cost_adapter(original,read(root/'FROZEN_COSTS.json')['code'],horizons=(60,120,300,360,420,600))
        record,memory=native_fixture(rt)
        group=max(group_components(record['rows']),key=lambda g:g['size'])
        bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
        acts=comparators(record,600)+[sample_unscored(group,[[0.]*300]*group['size'],bounds,'ordinary',20261101,duration_s=300)]
        if binding.get('concurrent_plans'):
            # Source-bound qualification of the NEW remaining-background branch.
            # Reuse the original observable fixture, no SUMO or new exact labels.
            from .speed2_runtime import qualify
            others=[r['vid'] for r in record['rows'] if r['is_cav'] and r['vid'] not in group['members']][:2]
            if len(others)!=2:raise ValueError('concurrent qualification requires two other sensed CAVs')
            other=dict(id='background_qualified',members=others,family='MPPI',duration_s=60,
                knots=[[-1.]*7,[.4]*7])
            context=dict(record,group=group['members'],committed_plans=[dict(plan=other,elapsed_s=20)])
            acts=acts+[dict(acts[-1],id='end_background_for_new_plan',committed_plans=[])]
            for k,h in ((4,300),(8,420)):
                width=batch_width(binding,k)
                q=dict(qualify(rt,context,acts,k,h,width,memory=memory,transfers=True),
                    context='other committed group remains for40s then policy; every comparator/candidate',new_sumo_sim_s=0)
                write(root/f'STARTUP_COMMITTED_K{k}_H{h}.json',q)
                if not q['pass_']:raise ValueError('concurrent background full/fast parity failed')
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
