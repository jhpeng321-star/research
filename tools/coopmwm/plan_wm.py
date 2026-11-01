"""Acceleration decoding at the inherited WMTEACH pre-safety boundary only."""
import ast
from contextlib import contextmanager
from contextlib import nullcontext
import inspect
from types import SimpleNamespace
import torch
from .plans import acceleration_at
from .wm import adapted_futures, executor_adapter
from .fused import FusedModel


def profile_table(actions, ids, device):
    """Compile once, avoiding per-step CPU rosters or per-vehicle GPU reads."""
    raw=torch.zeros((len(actions),300,len(ids)),device='cpu')
    mask=torch.zeros_like(raw,dtype=torch.bool)
    residual=torch.zeros_like(mask)
    slots={v:i for i,v in enumerate(ids)}
    for b,a in enumerate(actions):
        for t in range(300):
            # All unchanged CAVs retain the SAME own-action continuation; after
            # commitment the selected group rejoins that continuation as well.
            background=a.get('background_residuals',{})
            for v,x in background.items():
                if v in slots:
                    raw[b,t,slots[v]]=x;mask[b,t,slots[v]]=True;residual[b,t,slots[v]]=True
            if 'knots' in a:
                values=acceleration_at(a,t);is_residual=False
            else:
                values=a.get('residuals',{}) if t<a.get('window',0) else {};is_residual=True
            for v,x in values.items():
                if v not in slots:continue
                raw[b,t,slots[v]]=x;mask[b,t,slots[v]]=True;residual[b,t,slots[v]]=is_residual
    return raw.to(device),mask.to(device),residual.to(device)


def profile_apply(accel,tick,table,bounds):
    raw,mask,residual=table
    t=int(tick*.5)
    if t>=raw.shape[1]:return accel
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


def evaluate(rt,record,actions,K,H,chunk_actions=4,trim=True,pad_actions=False,horizons=None,speed2=False):
    """Same source draws, CRN keys, insertions, departure loop and J reducer.

    A single compact-forward proxy retains full virgin-slot controller memory.
    Candidate chunks bound memory without changing future IDs or source capacity.
    """
    hs=tuple(horizons or (H,))
    output={str(h):{a['id']:[] for a in actions} for h in hs}
    proxy=FusedModel(rt.model,1,trim_absent=trim or speed2);proxy.local.index=0
    from .speed2_runtime import ledger,compact_geometry
    runtime=SimpleNamespace(**vars(rt));runtime.model=proxy
    with (compact_geometry(proxy) if speed2 else nullcontext()),executor_adapter(),plan_executor():
        for start in range(0,len(actions),chunk_actions):
            subset=actions[start:start+chunk_actions]
            subset=[dict(a,background_residuals=record['policy_residuals']) if 'knots' in a else a for a in subset]
            real_ids=[a['id'] for a in subset]
            if pad_actions:
                subset=list(subset)
                width=4 if pad_actions is True else int(pad_actions)
                while len(subset)<width:
                    subset.append(dict(subset[-1],id=f'_pad{len(subset)}'))
            ns=adapted_futures(hs);ns['_actions']=subset
            if speed2:
                proxy.live_slots=max(1,len(record['rows']))
            with (ledger(proxy) if speed2 else nullcontext()),torch.no_grad():
                result=ns['futures'](runtime,dict(record,setting=0),indices=tuple(range(K)))
            if not result['population_audit']['finite'] or not result['population_audit']['conserved']:
                raise ValueError('imagined population accounting failed')
            for h in hs:
                for a in real_ids:output[str(h)][a]=result['futures'][str(h)][a]
    return output if horizons is not None else output[str(H)]
