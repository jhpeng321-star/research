"""Fuse independent inherited one-step WM rollouts; keep each source/controller private.

The producer's futures(), source draw, option executor, departure loop and cost
observer are unchanged. Only simultaneous model.rollout(n_steps=1) calls are
concatenated. Every result is cropped back before its original departure loop.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import torch


def merge(values):
    first=values[0]
    if torch.is_tensor(first):
        assert all(v.ndim==first.ndim for v in values)
        if not first.ndim:
            assert all(torch.equal(v,first) for v in values);return first
        shape=[max(v.shape[d] for v in values) for d in range(1,first.ndim)]
        padded=[]
        for v in values:
            # Entry capacity is private; copy virgin-slot defaults when padding.
            for d,n in enumerate(shape,1):
                if v.shape[d]<n:
                    assert v.shape[d]>0
                    idx=torch.arange(n,device=v.device).clamp_max(v.shape[d]-1)
                    v=v.index_select(d,idx)
            padded.append(v)
        return torch.cat(padded,0)
    if isinstance(first,dict):
        assert all(set(v)==set(first) for v in values),'heterogeneous tensor ABI'
        return {k:merge([v[k] for v in values]) for k in first}
    assert all(v==first for v in values),'non-tensor rollout ABI differs'
    return first


def sliced(value,start,end,slots=None):
    if torch.is_tensor(value):
        if not value.ndim:return value
        v=value[start:end]
        if slots is not None:
            # Frame tapes [B,T,N,...] vs current state/history [B,N,...].
            d=2 if slots[0]=='tape' else 1
            if v.ndim>d:v=v.narrow(d,0,min(slots[1],v.shape[d]))
        return v
    if isinstance(value,dict):return {k:sliced(v,start,end,slots) for k,v in value.items()}
    return value


TAPES={'positions','speeds','lanes','masks','accel','lane_logits',
       'raw_residual_mps2','lane_change_probabilities','lane_probabilities',
       'entrant_seq','sections','physics_accel_mps2'}


def unpack(value,start,end,n):
    return {k:sliced(v,start,end,('tape' if k in TAPES else 'state',n))
            for k,v in value.items()}


def crop(value,n,capacity):
    if torch.is_tensor(value):
        return value[:,:n] if value.ndim>=2 and value.shape[1]==capacity else value
    if isinstance(value,dict):return {k:crop(v,n,capacity) for k,v in value.items()}
    return value


def restore(value,capacity,template=None,tape=False):
    if torch.is_tensor(value):
        d=2 if tape else 1
        if value.ndim<=d:return value
        if value.shape[d]>=capacity:return value
        shape=list(value.shape);shape[d]=capacity
        out=(template.clone() if torch.is_tensor(template) and list(template.shape)==shape
             else torch.zeros(shape,dtype=value.dtype,device=value.device))
        out.narrow(d,0,value.shape[d]).copy_(value)
        return out
    if isinstance(value,dict):return {k:restore(v,capacity,(template or {}).get(k),tape) for k,v in value.items()}
    return value


class FusedModel:
    def __init__(self,model,count,trim_absent=False):
        self.model=model;self.count=count;self.calls={};self.results={}
        self.trim_absent=trim_absent
        self.local=threading.local();self.lock=threading.Lock()
        self.barrier=threading.Barrier(count,action=self._forward,timeout=180)

    def __getattr__(self,name):return getattr(self.model,name)

    def rollout(self,state,**kwargs):
        assert kwargs.get('n_steps')==1,'only inherited one-step calls may fuse'
        i=self.local.index
        with self.lock:self.calls[i]=(state,kwargs)
        self.barrier.wait()
        result=self.results[i]
        self.barrier.wait()  # Do not overwrite buffers before all callers consume them.
        return result

    def _forward(self):
        # Barrier action runs twice/step: first forward, then release-only.
        if getattr(self,'release',False):self.release=False;return
        self.release=True
        calls=[self.calls[i] for i in range(self.count)]
        sizes=[s['present'].shape[0] for s,_ in calls]
        offsets=[0]
        for n in sizes:offsets.append(offsets[-1]+n)
        callbacks=[kw['cav_controller'] for _,kw in calls]
        ns=[s['present'].shape[1] for s,_ in calls]
        state=merge([s for s,_ in calls])
        kwargs=merge([{k:v for k,v in kw.items() if k!='cav_controller'} for _,kw in calls])
        capacity=state['present'].shape[1]
        # Preserve the complete source draw and private virgin-slot ledger.
        # Omit universally absent trailing slots only from the N² model forward;
        # restore them BEFORE the inherited insertion/cost phase.
        n=capacity
        if self.trim_absent:
            if hasattr(self,'live_slots'):
                n=min(capacity,max(1,self.live_slots))
            else:
                live=state['present'].any(0).nonzero().flatten()
                n=max(1,int(live[-1])+1) if live.numel() else 1
        full_state,full_kwargs=state,kwargs
        if self.trim_absent:
            state=crop(state,n,capacity);kwargs=crop(kwargs,n,capacity)

        def dispatch(current,mask=False):
            outs=[]
            # The recurrent MAPPO bridge owns a FIXED virgin-slot roster. Passing
            # the compact model roster to it resets its memory/dwell whenever a
            # new entry grows the compact shape, changing the counterfactual.
            # Restore before the callback as well as before source insertion;
            # crop only the returned physical controls for the model forward.
            control=restore(current,capacity,full_state) if self.trim_absent else current
            for i,cb in enumerate(callbacks):
                a,b=offsets[i:i+2]
                part=sliced(control,a,b,('state',ns[i]))
                out=cb.native_lane_mask(part) if mask else cb(part)
                # Controllers add execution-cap/lane masks to their private state.
                if not mask:
                    for key,v in part.items():
                        if key in current:continue
                        if torch.is_tensor(v) and v.ndim>=2:
                            current[key]=torch.zeros((offsets[-1],state['present'].shape[1],*v.shape[2:]),device=v.device,dtype=v.dtype)
                    for key,v in part.items():
                        if torch.is_tensor(v) and v.ndim>=2 and key in current:
                            if self.trim_absent:v=v[:,:state['present'].shape[1]]
                            current[key][a:b,:v.shape[1]]=v
                outs.append(out)
            out=merge(outs)
            return out[:,:state['present'].shape[1]] if self.trim_absent else out
        dispatch.native_lane_mask=lambda s:dispatch(s,True)
        with torch.no_grad():result=self.model.rollout(state,cav_controller=dispatch,**kwargs)
        if self.trim_absent:
            result={k:restore(v,capacity,full_state if k=='state' else full_kwargs.get(k),k in TAPES)
                    for k,v in result.items()}
        self.results={i:unpack(result,offsets[i],offsets[i+1],ns[i]) for i in range(self.count)}


def futures(rt,records,indices,trim_absent=False):
    """Many observable windows × paired futures in each inherited model forward."""
    from types import SimpleNamespace
    from .wm import adapted_futures,proposal_candidates,executor_adapter
    proxy=FusedModel(rt.model,len(records),trim_absent=trim_absent)
    runtime=SimpleNamespace(**vars(rt));runtime.model=proxy
    actions=[proposal_candidates(r) for r in records]
    def one(i):
        proxy.local.index=i
        ns=adapted_futures((300,));ns['_actions']=actions[i]
        try:
            with torch.no_grad():return ns['futures'](runtime,dict(records[i]),indices=indices)
        except BaseException:
            import traceback,sys
            print('FUSED_WINDOW_ERROR '+str(i)+'\n'+traceback.format_exc(),file=sys.stderr,flush=True)
            proxy.barrier.abort();raise
    # Adapter is process-global: installed once, not independently by each thread.
    with executor_adapter(),ThreadPoolExecutor(max_workers=len(records)) as pool:
        return list(pool.map(one,range(len(records))))
