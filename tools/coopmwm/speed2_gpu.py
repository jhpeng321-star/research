"""Bounded GPU profiler/experiments on immutable engineering fixtures only."""
import inspect
import os
import copy
from contextlib import contextmanager, ExitStack, nullcontext
import time
from unittest.mock import patch

import torch

from .speed2 import save, setup


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


class GraphCall:
    """Experimental fixed-shape graph of a pure deterministic callable.

    Copies preserve input/output ownership. No sampling, physics, source ledger,
    recurrent bridge, or costs are captured. Every new signature is bounded.
    """
    def __init__(self, fn):
        self.fn=fn
        self.cache={}
        self.replays=0
        self.capture_s=0.

    def __call__(self,*args,**kwargs):
        from torch.utils._pytree import tree_flatten,tree_unflatten,tree_map
        leaves,spec=tree_flatten((args,kwargs))
        key=(str(spec),tuple((tuple(x.shape),tuple(x.stride()),x.dtype,x.device)
                            if torch.is_tensor(x) else x for x in leaves))
        if key not in self.cache:
            assert len(self.cache)<4,'bounded graph cache exceeded'
            start=time.perf_counter()
            static=[x.clone() if torch.is_tensor(x) else x for x in leaves]
            a,k=tree_unflatten(static,spec)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):self.fn(*a,**k)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):out=self.fn(*a,**k)
            torch.cuda.current_stream().wait_stream(stream)
            self.cache[key]=(graph,static,out)
            self.capture_s+=time.perf_counter()-start
        graph,static,out=self.cache[key]
        for x,y in zip(static,leaves):
            if torch.is_tensor(x):x.copy_(y)
        graph.replay()
        self.replays+=1
        return tree_map(lambda x:x.clone() if torch.is_tensor(x) else x,out)


@contextmanager
def actor_graphs(rt):
    from dataclasses import fields
    from torch.utils._pytree import register_pytree_node
    import ast
    import textwrap
    import types
    from wms.control import policy
    from wms.control.contract import Observation
    register_pytree_node(Observation,
        lambda x:([getattr(x,f.name) for f in fields(x)],None),
        lambda values,context:Observation(*values))
    normal=policy.Normal
    # Normal's optional argument validation reads CUDA scalars. It is not part
    # of the deterministic arithmetic; eager reference still validates normally.
    def distribution(loc,scale,*a,**kw):
        return normal(loc,scale,*a,**dict(kw,validate_args=False))
    actor=GraphCall(rt.mp.policy.forward)
    encoder=GraphCall(rt.mp.group_encoder.forward)
    constants={}
    def normalize(values,features):
        key=('norm',tuple(features),values.device,values.dtype)
        if key not in constants:
            constants[key]=values.new_tensor([policy.FEATURE_NORMALIZATION[n] for n in features])
        return values/constants[key]
    geometry=policy.cell_geometry
    def cell_geometry(device=None,dtype=torch.float32):
        key=('geometry',device,dtype)
        if key not in constants:constants[key]=geometry(device,dtype)
        return constants[key]
    def field_scale(value):
        key=('field_scale',value.device,value.dtype)
        if key not in constants:constants[key]=value.new_tensor(policy.FIELD_SCALE)
        return constants[key]
    original=rt.mp.group_encoder._field_context
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    class Scale(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and node.func.attr=='new_tensor':
                assert isinstance(node.args[0],ast.Name) and node.args[0].id=='FIELD_SCALE'
                return ast.copy_location(ast.Call(func=ast.Name(id='cached_scale',ctx=ast.Load()),
                    args=[node.func.value],keywords=[]),node)
            return self.generic_visit(node)
    tree=Scale().visit(tree);ast.fix_missing_locations(tree)
    ns=dict(original.__func__.__globals__,cached_scale=field_scale)
    exec(compile(tree,'<speed2-cached-field-scale>','exec'),ns)
    field=types.MethodType(ns['_field_context'],rt.mp.group_encoder)
    with patch.object(policy,'Normal',distribution), \
         patch.object(policy,'normalize_features',normalize), \
         patch.object(policy,'cell_geometry',cell_geometry), \
         patch.object(rt.mp.group_encoder,'_field_context',field), \
         patch.object(rt.mp.policy,'forward',actor), \
         patch.object(rt.mp.group_encoder,'forward',encoder):
        yield [actor,encoder]


def experiment(root,H,K,width):
    """Compare an opt-in graph trial with the already-qualified eager fast path."""
    from .plan_wm import evaluate
    from .speed2_runtime import trajectory_errors
    rt,record,actions=setup(root)
    memory=getattr(rt,'speed2_memory',None)
    variant=os.environ.get('SPEED2_EXPERIMENT','actor_graphs')
    if variant=='wide_batch':
        seed=next(a for a in actions if 'knots' in a)
        actions=[]
        for i in range(2*width):
            a=copy.deepcopy(seed);a['id']=f'wide_{i:03}'
            a['knots']=[[x+.001*i for x in row] for row in a['knots']]
            actions.append(a)
    tapes=[]
    costs=rt.frozen.costs
    def capture(roll,entries,initial):
        tapes.append({k:roll[k].detach().cpu().clone() for k in
                      ('positions','speeds','lanes','masks','accel')})
        return costs(roll,entries,initial)
    rt.frozen.costs=capture
    hs=tuple(h for h in (60,120,300) if h<=H)
    def run(batch_width):
        torch.cuda.synchronize();start=time.perf_counter()
        out=evaluate(rt,record,actions,K,H,chunk_actions=batch_width,trim=False,
                     pad_actions=batch_width,horizons=hs,speed2=True,memory=memory)
        torch.cuda.synchronize()
        return out,time.perf_counter()-start
    before,old_wall=run(width)
    save(root/'EAGER_FAST.json',dict(wall_s=old_wall,H=H,K=K,width=width,new_sumo_sim_s=0))
    expected=tapes[:];tapes.clear()
    from . import plan_wm
    context=(actor_graphs(rt) if variant=='actor_graphs' else
             patch.object(plan_wm,'profile_table',vector_table) if variant=='tables' else nullcontext([]))
    with context as graphs:
        after,new_wall=run(2*width if variant=='wide_batch' else width)
    if variant!='actor_graphs':graphs=[]
    errors=trajectory_errors(expected,tapes)
    worst=max(abs(x['team_delay']-y['team_delay']) for h in before for a in before[h]
              for x,y in zip(before[h][a],after[h][a]))
    rank=lambda values:sorted(values,key=lambda a:(sum(v['team_delay'] for v in values[a]),a))
    ranking=all(rank(before[h])==rank(after[h]) for h in before)
    draws=all(x['draw_id']==y['draw_id'] for h in before for a in before[h]
              for x,y in zip(before[h][a],after[h][a]))
    steps=K*max(len(actions),width)*2*H
    result=dict(experiment=variant,pass_=worst==0 and max(errors.values())==0 and ranking and draws,
        max_cost_error_veh_s=worst,trajectory_errors=errors,identical_candidate_ranking=ranking,
        identical_draw_ids=draws,reference_wall_s=old_wall,optimized_wall_s=new_wall,
        speedup=old_wall/new_wall,reference_future_steps_per_s=steps/old_wall,
        optimized_future_steps_per_s=steps/new_wall,H=H,K=K,width=width,actions=len(actions),
        warm=memory is not None,group_size=len(record['group']),
        graph_captures=[len(g.cache) for g in graphs],graph_replays=[g.replays for g in graphs],
        graph_capture_s=[g.capture_s for g in graphs],timing_includes_trajectory_capture=True,
        experimental_only=True,new_sumo_sim_s=0)
    save(root/'EXPERIMENT.json',result)
    print(result,flush=True)


def profile(root, H, K, width):
    from .plan_wm import evaluate
    rt, record, actions = setup(root)
    memory = getattr(rt, 'speed2_memory', None)
    assert next(rt.model.parameters()).is_cuda
    calls = 0
    original = rt.model.rollout
    schedule = torch.profiler.schedule(wait=12, warmup=2, active=4, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                schedule=schedule, record_shapes=True,
                                with_stack=False) as prof:
        def step(*args, **kwargs):
            nonlocal calls
            with torch.profiler.record_function('speed2_model_step'):
                out = original(*args, **kwargs)
            calls += 1
            prof.step()
            return out
        torch.cuda.synchronize()
        start = time.perf_counter()
        with patch.object(rt.model, 'rollout', step):
            evaluate(rt, record, actions, K, H, chunk_actions=width, trim=False,
                     pad_actions=width, speed2=True, memory=memory)
        torch.cuda.synchronize()
        wall = time.perf_counter()-start
    events = list(prof.key_averages())
    cpu = sum(e.self_cpu_time_total for e in events)
    # Count device kernels only: CPU op attribution and user annotations can
    # include the same device interval and must not be added to kernel totals.
    gpu = sum(e.self_device_time_total for e in events if str(e.device_type).endswith('CUDA'))
    rows = [dict(name=e.key, count=e.count, self_cpu_ms=e.self_cpu_time_total/1000,
                 self_gpu_ms=e.self_device_time_total/1000,
                 cpu_percent=100*e.self_cpu_time_total/max(cpu, 1),
                 gpu_percent=100*e.self_device_time_total/max(gpu, 1))
            for e in events]
    prof.export_chrome_trace(str(root/'GPU_TRACE.json'))
    # Save the actual loaded physics API for later targeted pure-kernel trials.
    methods = {}
    for name in ('rollout', 'forward', 'encode_drivers'):
        fn = getattr(rt.model, name, None)
        if fn is not None:
            methods[name] = inspect.getsource(fn)
    save(root/'MODEL_API.json', methods)
    result = dict(H=H, K=K, width=width, group_size=len(record['group']),
                  warm=memory is not None, sampled_steps=4, model_calls=calls,
                  wall_s=wall, timing_is_profiled_not_speedup=True,
                  cpu_self_ms=cpu/1000, gpu_self_ms=gpu/1000,
                  cpu_top=sorted(rows, key=lambda x: -x['self_cpu_ms'])[:35],
                  gpu_top=sorted([r for e,r in zip(events,rows)
                                  if str(e.device_type).endswith('CUDA')],
                                 key=lambda x: -x['self_gpu_ms'])[:25],
                  synchronization=[r for r in rows if any(s in r['name'].lower()
                                   for s in ('synchron', '_local_scalar', 'item', 'nonzero', 'memcpy'))],
                  new_sumo_sim_s=0)
    save(root/'GPU_PROFILE.json', result)
    print(result, flush=True)
