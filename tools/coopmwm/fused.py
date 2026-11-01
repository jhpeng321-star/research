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


class FusedModel:
    def __init__(self,model,count):
        self.model=model;self.count=count;self.calls={};self.results={}
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

        def dispatch(current,mask=False):
            outs=[]
            for i,cb in enumerate(callbacks):
                a,b=offsets[i:i+2]
                part=sliced(current,a,b,('state',ns[i]))
                out=cb.native_lane_mask(part) if mask else cb(part)
                # Controllers add execution-cap/lane masks to their private state.
                if not mask:
                    for key,v in part.items():
                        if key in current:continue
                        if torch.is_tensor(v) and v.ndim>=2:
                            current[key]=torch.zeros((offsets[-1],state['present'].shape[1],*v.shape[2:]),device=v.device,dtype=v.dtype)
                    for key,v in part.items():
                        if torch.is_tensor(v) and v.ndim>=2 and key in current:
                            current[key][a:b,:v.shape[1]]=v
                outs.append(out)
            return merge(outs)
        dispatch.native_lane_mask=lambda s:dispatch(s,True)
        with torch.no_grad():result=self.model.rollout(state,cav_controller=dispatch,**kwargs)
        self.results={i:unpack(result,offsets[i],offsets[i+1],ns[i]) for i in range(self.count)}


def futures(rt,records,indices):
    """Many observable windows × paired futures in each inherited model forward."""
    from types import SimpleNamespace
    from .wm import adapted_futures,proposal_candidates,executor_adapter
    proxy=FusedModel(rt.model,len(records))
    runtime=SimpleNamespace(**vars(rt));runtime.model=proxy
    actions=[proposal_candidates(r) for r in records]
    def one(i):
        proxy.local.index=i
        ns=adapted_futures((300,));ns['_actions']=actions[i]
        try:
            with torch.no_grad():return ns['futures'](runtime,dict(records[i]),indices=indices)
        except BaseException:
            proxy.barrier.abort();raise
    # Adapter is process-global: installed once, not independently by each thread.
    with executor_adapter(),ThreadPoolExecutor(max_workers=len(records)) as pool:
        return list(pool.map(one,range(len(records))))
