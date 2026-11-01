"""Exact opt-in batching of profile writes and native newborn tensor transfers.

The frozen identity-keyed sampler is still called once per newborn, in its
original order. Read-only birth support can stay on the CPU where discrete
sampling already runs. No source admissions, randomness or arithmetic change.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch


def vector_table(actions,ids,device):
    """Bulk time-axis writes; interpolation still uses the exact Python decoder."""
    from .plans import acceleration_at
    raw=torch.zeros((len(actions),300,len(ids)),device='cpu')
    mask=torch.zeros_like(raw,dtype=torch.bool)
    residual=torch.zeros_like(mask)
    slots={v:i for i,v in enumerate(ids)}
    for b,a in enumerate(actions):
        for v,x in a.get('background_residuals',{}).items():
            if v in slots:
                raw[b,:,slots[v]]=x;mask[b,:,slots[v]]=True;residual[b,:,slots[v]]=True
        if 'knots' in a:
            values=[acceleration_at(a,t) for t in range(300)]
            for v in dict.fromkeys(v for row in values for v in row):
                if v not in slots:continue
                ts=[t for t,row in enumerate(values) if v in row]
                raw[b,ts,slots[v]]=torch.tensor([values[t][v] for t in ts],dtype=raw.dtype)
                mask[b,ts,slots[v]]=True;residual[b,ts,slots[v]]=False
        else:
            ts=[t for t in range(300) if t<a.get('window',0)]
            for v,x in a.get('residuals',{}).items():
                if v in slots:
                    raw[b,ts,slots[v]]=x;mask[b,ts,slots[v]]=True;residual[b,ts,slots[v]]=True
    return raw.to(device),mask.to(device),residual.to(device)


@contextmanager
def batched_births(cpu_pools=False):
    """Keep the native identity sampler/order; batch scalar reads and scatters."""
    from wms.control.hidden_state import BirthMix
    original_pool=BirthMix.pool_for
    def pool(mix,b,cav):
        if not hasattr(mix,'_speed2_cpu_pools'):mix._speed2_cpu_pools={}
        key=(b,bool(cav))
        if key not in mix._speed2_cpu_pools:
            mix._speed2_cpu_pools[key]=original_pool(mix,b,cav).detach().cpu()
        return mix._speed2_cpu_pools[key]
    def assign(mix,state,newborn,params,birth_ordinals=None):
        result={name:value.clone() for name,value in params.items()}
        index=newborn.nonzero()
        pairs=index.tolist()
        if not pairs:return result
        b,slot=index.unbind(-1)
        lanes=state['lane'][b,slot].detach().cpu().tolist()
        cavs=state['is_cav'][b,slot].detach().cpu().tolist()
        updates={}
        for (row,col),lane,cav in zip(pairs,lanes,cavs):
            entrance=int(lane==2)
            if birth_ordinals is None or (row,col) not in birth_ordinals:
                raise ValueError('hidden-state birth requires admission identity ordinal')
            values=mix.candidate(row,entrance,birth_ordinals[row,col],bool(cav))
            for name,value in values.items():
                updates.setdefault(name,[]).append((row,col,value))
            mix.counts[row][entrance]+=1
            mix.assigned[row]+=1
        for name,values in updates.items():
            rows,cols,data=zip(*values)
            target=result[name]
            target[torch.tensor(rows,device=target.device),torch.tensor(cols,device=target.device)]=torch.tensor(data,dtype=target.dtype,device=target.device)
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(BirthMix,'__call__',assign))
        if cpu_pools:stack.enter_context(patch.object(BirthMix,'pool_for',pool))
        yield []


@contextmanager
def fast_transfers():
    from . import plan_wm
    with patch.object(plan_wm,'profile_table',vector_table),batched_births(cpu_pools=True):
        yield
