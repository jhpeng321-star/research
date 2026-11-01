"""Thin bounded adapter around the frozen WMTEACH batched K8 code path."""
import ast
import contextlib
import copy
import inspect
import json
import math
import types
from .protocol import digest
from .credit import paired


def candidates(record):
    chosen=record['targets']; members=record['group']
    if len(members)!=2 or len(set(members))!=2:
        raise ValueError('complete two-member RSU group required')
    return [dict(id=name,window=10,targets={v:t for v,t in chosen.items() if v not in revert})
            for name,revert in [('joint',[]),*[(f'revert:{i}',[v]) for i,v in enumerate(members)],
                                ('all',members)]]


def validate(record):
    from tools.coopnative.policy import encode
    encode(record['rows'])
    allowed={'state_id','world_id','time_s','rows','targets','group'}
    if set(record)!=allowed or not 300<=record['time_s']<=1190:
        raise ValueError('observable replay whitelist/clock violation')
    ids={r['vid'] for r in record['rows'] if r['is_cav']}
    if not set(record['group'])<=ids or not set(record['targets'])<=ids:
        raise ValueError('non-CAV target/group')
    for target in record['targets'].values():
        if set(target)!={'speed','tau'} or not all(math.isfinite(v) for v in target.values()) or target['tau']<=0:
            raise ValueError('invalid setpoint')
    return record


@contextlib.contextmanager
def executor_adapter():
    """Use the original option executor; only its target/request decoder changes."""
    from tools.leverprobe import intervention
    from tools.coopfree.policy import requests
    old_targets,old_requests=intervention.targets,intervention.requests
    intervention.targets=lambda action,rows:action['targets']
    intervention.requests=lambda action,rows,fixed:(requests(fixed,rows,zone=None),'setpoint')
    try:yield
    finally:intervention.targets,intervention.requests=old_targets,old_requests


def adapted_futures():
    """Minimal AST parameterization: H90 instead of H120/H300, same K8 implementation.

    Frozen upstream sources are staged/hash-bound. Fail closed on changed structure.
    No physical rollout, sampling, costs, safety, or option executor is reimplemented.
    """
    from tools.wmteach import scale,runtime
    tree=ast.parse(inspect.getsource(scale.futures))
    counts={'steps':0,'horizons':0}
    class Horizon(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==600:
                counts['steps']+=1
                return ast.copy_location(ast.Constant(180),node)
            return node
        def visit_Tuple(self,node):
            if all(isinstance(v,ast.Constant) for v in node.elts) and [v.value for v in node.elts]==[120,300]:
                counts['horizons']+=1
                return ast.copy_location(ast.Tuple([ast.Constant(90)],ast.Load()),node)
            return self.generic_visit(node)
    tree=Horizon().visit(tree);ast.fix_missing_locations(tree)
    if counts!={'steps':1,'horizons':2}:
        raise ValueError('frozen batched K8 ABI changed')
    def source(rt,r):
        b=runtime.diagnostic_source(rt,r,0)
        # Identical population-marginal source prior used by the qualified teacher.
        # These values are inferred priors, never read from SUMO RequestLedger.
        r['backlog']=b.entry_queue[0,:2].cpu().tolist()
        r['request_total']=(b.entry_queue[0,2:4]*r['time_s']).cpu().tolist()
        return b
    namespace=dict(scale.futures.__globals__,actions=lambda rows,setting:namespace['_actions'],
                   source_for=source,BACKEND='FULL-v4')
    exec(compile(tree,'<coopmwm-frozen-batched-H90>','exec'),namespace)
    return namespace


def evaluate(rt,record):
    validate(record)
    ns=adapted_futures();ns['_actions']=candidates(record)
    adapted=dict(record,setting=0)
    with executor_adapter():
        result=ns['futures'](rt,adapted)
    audit=result['population_audit']
    if not audit['finite'] or not audit['conserved']:
        raise ValueError('WM finite/conservation failure')
    f=result['futures']['90']
    credit=paired([{a:values[k] for a,values in f.items()} for k in range(8)])
    return dict(credit=credit,replay=result,record_sha256=digest(record),horizon_s=90,
                observation_contract='roster-only/cold-memory/uniform32FIT-source-prior')
