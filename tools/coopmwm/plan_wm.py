"""Acceleration decoding at the inherited WMTEACH pre-safety boundary only."""
import ast
from contextlib import contextmanager
import inspect
from types import SimpleNamespace
import torch
from .plans import acceleration_at
from .wm import adapted_futures, executor_adapter
from .fused import FusedModel


def profile_table(actions, ids, device):
    """Compile once, avoiding per-step CPU rosters or per-vehicle GPU reads."""
    raw=torch.zeros((len(actions),60,len(ids)),device=device)
    mask=torch.zeros_like(raw,dtype=torch.bool)
    residual=torch.zeros_like(mask)
    slots={v:i for i,v in enumerate(ids)}
    for b,a in enumerate(actions):
        for t in range(60):
            if 'knots' in a:
                values=acceleration_at(a,t);is_residual=False
            else:
                values=a.get('residuals',{}) if t<a.get('window',0) else {};is_residual=True
            for v,x in values.items():
                if v not in slots:continue
                raw[b,t,slots[v]]=x;mask[b,t,slots[v]]=True;residual[b,t,slots[v]]=is_residual
    return raw,mask,residual


def profile_apply(accel,tick,table,bounds):
    raw,mask,residual=table
    t=int(tick*.5)
    if t>=60:return accel
    n=raw.shape[-1]
    out=accel.clone()
    request=torch.where(residual[:,t],accel[:,:n]+raw[:,t],raw[:,t])
    out[:,:n]=torch.where(mask[:,t],request.clamp(*bounds),accel[:,:n])
    return out


@contextmanager
def plan_executor():
    """Keep original bridge, memory, lane mask, cap, and safe_accel implementation."""
    from tools.wmteach import runtime
    original=runtime.option_controller
    tree=ast.parse(inspect.getsource(original));matches=0
    outer=tree.body[0]
    # Inherited fixed targets are irrelevant to acceleration profiles.
    for i,node in enumerate(outer.body):
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='fixed' for x in node.targets):
            outer.body[i]=ast.parse("table=profile_table(candidate_actions,ids,next(mp.policy.parameters()).device)").body[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name=='injected':
            matches+=1
            node.body=ast.parse("return safe_accel(state,profile_apply(accel,tick,table,(mp.policy.accel_min,mp.policy.accel_max)),**kw)").body
    if matches!=1:raise ValueError('inherited pre-safety ABI changed')
    ast.fix_missing_locations(tree)
    ns=dict(original.__globals__,profile_table=profile_table,profile_apply=profile_apply)
    exec(compile(tree,'<coopmwm-acceleration-profile-decoder>','exec'),ns)
    runtime.option_controller=ns['option_controller']
    try:yield
    finally:runtime.option_controller=original


def evaluate(rt,record,actions,K,H,chunk_actions=4,trim=True):
    """Same source draws, CRN keys, insertions, departure loop and J reducer.

    A single compact-forward proxy retains full virgin-slot controller memory.
    Candidate chunks bound memory without changing future IDs or source capacity.
    """
    output={a['id']:[] for a in actions}
    proxy=FusedModel(rt.model,1,trim_absent=trim);proxy.local.index=0
    runtime=SimpleNamespace(**vars(rt));runtime.model=proxy
    with executor_adapter(),plan_executor():
        for start in range(0,len(actions),chunk_actions):
            subset=actions[start:start+chunk_actions]
            ns=adapted_futures((H,));ns['_actions']=subset
            with torch.no_grad():result=ns['futures'](runtime,dict(record,setting=0),indices=tuple(range(K)))
            if not result['population_audit']['finite'] or not result['population_audit']['conserved']:
                raise ValueError('imagined population accounting failed')
            for a in subset:output[a['id']]=result['futures'][str(H)][a['id']]
    return output
