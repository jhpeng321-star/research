"""Frozen FULL-v4 GPU adapter; no SUMO imports, calls or future truth inputs."""
import ast
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.coopbandit.protocol import digest, observable, sha
from tools.wmteach.labels import actions

PHYS = Path('/root/rbeval20261031/phys_code_22cc32f65a80')
FIX = Path('/root/autodl-tmp/c4/setupaudit20261030/fix')
BASE = Path('/root/autodl-tmp/c4')


def move(value, device='cuda'):
    if torch.is_tensor(value): return value.to(device)
    if isinstance(value, dict): return {k:move(v,device) for k,v in value.items()}
    return value


def bootstrap():
    """Called after immutable source paths and run identity have been recorded."""
    import sys
    sys.path[:0] = [str(PHYS/'src'), str(PHYS)]
    import tools
    tools.__path__.append(str(PHYS/'tools'))
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        C4_ALLOWED_ROOTS='/root/autodl-tmp:/root/wmcalib:/root/rbeval20261031',
        C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1',
        C4_EXECUTION_DRIVER=str(FIX/'configs/execparity_cav_idm_v1.json'),
        WMPHYS_VARIANT='FULL', WMS_WMSPEED_DEDUPE_MOBIL='1', WMS_WMSPEED_MOBIL_LAZY_SAFE='1',
        WMS_SPEED_PIVOT_DIAG_OFF='1', WMS_SPEED_FAST_MERGE_LANE='1', WMS_SPEED_FAST_CAV_DRIVE='1',
        WMS_SPEED_FAST_LONG_HISTORY='1', WMS_SPEED_GRAPH_CVAR='1')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from wms.control.bridge import make_cav_controller
    from wms.control.contract import MERGE_S1_M, SPEED_LIMIT_MPS
    from wms.control.start_states import StartBatch
    from wms.model.entries import build_entrance_history, _counts_counter
    from tools.rbint.recipe import load_candidate
    from tools.rbeval.core import reference
    namespace = dict(globals(), **locals(), ACTORS=[])
    for key in ('worker','costs','motion'):
        reference(key, namespace)
    frozen = SimpleNamespace(**namespace)
    namespace.update(frozen=frozen, ROOT=Path('/root/wmcalib'), DATA=Path('/root/wmcalib/data'))
    reference('starts', namespace)
    # Filter BEFORE the producer opens any per-world path. No reserved world is loaded.
    from tools.rbeval.core import load, HERE
    source = load(HERE/'references.json')['starts']['code']
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    loop = next(n for n in function.body if isinstance(n,ast.For))
    loop.body.insert(1,ast.parse('if not 969400 <= seed < 969432: continue').body[0])
    ast.fix_missing_locations(tree)
    exec(compile(tree,'<wmteach-FIT-only-starts>','exec'), namespace)
    wm = BASE/'wmcombined_wm2/wm_s.pt'
    assert sha(wm) == 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
    models, receipt = load_candidate([str(wm)],[str(PHYS/'configs/wm_phys.toml')],'cuda')
    model = models[0]
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    candidate = torch.load(BASE/'setupaudit20261030/policy_000190.pt',map_location='cpu',weights_only=False)
    from wms.control.dreamer import DreamerConfig
    cfg = replace(DreamerConfig(**candidate['config']),horizon=300,entrant_learned_mix=True)
    model.cav_exec_cap = cfg.cav_exec_cap
    batches, provenance = namespace['make_starts_wmdata'](model)
    assert len(batches)==32
    from wms.control.mappo.eval_bridge import load_policy_fn
    binding = json.loads((BASE/'speedhead20261030/EXPORT_BOUND.json').read_text())
    assert sha(binding['path'])==binding['sha256']
    mp = load_policy_fn(binding['path'],device='cuda')
    assert mp.native_lanes and not mp.residual_on_native
    mp.policy.eval(); mp.group_encoder.eval()
    return SimpleNamespace(model=model, cfg=cfg, batches=batches, provenance=provenance,
                           frozen=frozen, mp=mp, receipt=receipt, export=binding)


def roster(state, ids=None):
    result=[]
    for slot in state['present'][0].nonzero().flatten().tolist():
        result.append(dict(vid=(str(slot) if ids is None else ids[slot]),
            s_m=float(state['s_m'][0,slot]),v_mps=float(state['v_mps'][0,slot]),
            lane=int(state['lane'][0,slot]),is_cav=bool(state['is_cav'][0,slot]),
            length_m=float(state['length_m'][0,slot])))
    return result


def option_controller(mp, candidate_actions, initial_rows):
    """Same LEVERPROBE requests → policy bounds → safety → plant cap as SUMO.

    Private virgin slots are never reused. Existing group IDs remain attached to
    their slots; newborn slots cannot accidentally inherit group ownership.
    """
    from wms.control.bridge import make_cav_controller
    from wms.control.train_cli import GroupSignal
    from wms.control.safety import safe_accel
    from wms.control.alwayscmd import inferred_cap
    from wms.model.cav import EXEC_CAP_KEY
    from tools.leverprobe.intervention import targets, requests
    ids = [r['vid'] for r in initial_rows]
    fixed = [targets(a,initial_rows) for a in candidate_actions]
    raw_override = mask_override = None
    tick = 0

    def injected(state, accel, **kw):
        nonlocal raw_override, mask_override
        if tick % 2 == 0:
            raw_override=torch.zeros_like(accel); mask_override=torch.zeros_like(accel,dtype=torch.bool)
            for b,a in enumerate(candidate_actions):
                if tick*.5 >= a.get('window',0): continue
                row={k:v[b:b+1] if torch.is_tensor(v) and v.ndim and v.shape[0]==len(candidate_actions) else v
                     for k,v in state.items()}
                current=roster(row,ids+['new:'+str(i) for i in range(len(ids),accel.shape[1])])
                proposed,_=requests(a,current,fixed[b])
                for slot,vid in enumerate(ids):
                    if vid in proposed:
                        raw_override[b,slot]=max(mp.policy.accel_min,min(mp.policy.accel_max,proposed[vid]['requested_raw']))
                        mask_override[b,slot]=True
        return safe_accel(state,torch.where(mask_override,raw_override,accel),**kw)

    def deterministic(obs,signal,**kw): return mp.policy(obs,signal,deterministic=True,**kw)
    deterministic.accel_max=mp.policy.accel_max
    deterministic.residual_on_native=False
    bridge=make_cav_controller(deterministic,GroupSignal(mp.group_encoder),injected,
        lane_rules=mp.lane_rules,policy_native_lanes=True,policy_ramp_merge=False,
        symmetric_exec_cap=True,mask_forced_merge=True,native_lane_change=True,native_ramp_merge='wm_law')

    def callback(state):
        nonlocal tick
        state[EXEC_CAP_KEY]=inferred_cap(state)
        out=bridge(state)
        accel=torch.maximum(-state['v_mps']/.5,torch.minimum(out[...,3],state[EXEC_CAP_KEY]))
        speed=(state['v_mps']+.5*accel).clamp_min(0.)
        result=torch.stack((state['s_m']+.5*speed,out[...,1],speed,accel),-1)
        tick+=1
        return result
    callback.native_lane_mask=bridge.native_lane_mask
    return callback


def prepare_batch(rt, source):
    from wms.control.dreamer import with_entry_slots
    b=with_entry_slots(copy.deepcopy(source),300,960,torch.Generator().manual_seed(191000573),queue_state=True)
    b.state,b.rollout_kwargs=move(b.state),move(b.rollout_kwargs)
    return b


def fit_futures(rt, source, record, future):
    from tools.rbint.recipe import candidate_future
    from tools.rbint.departure import rollout_after_movement
    from wms.physics.batched import road_end_exit_bound
    from wms.control.nativeinit import pair_merge_streams
    acts=actions(record['rows'],record['setting'])
    source = diagnostic_source(rt,record,future) if source is None else source
    b=prepare_batch(rt,source)
    initial=rt.frozen.expand(b.state,len(acts))
    pair_merge_streams(initial,torch.zeros(len(acts),dtype=torch.long,device='cuda'),1,190+future)
    initial['exit_s_m']=torch.full_like(initial['s_m'],road_end_exit_bound(rt.model.physics))
    if record['role']=='QUALITY_ONLY':
        kw,entries,receipt=diagnostic_future(rt,b,initial,record,future)
    else:
        kw,entries,receipt=candidate_future(model=rt.model,batch=b,initial=initial,future=future,
            seed=record['world_id'],frozen=rt.frozen)
    cb=option_controller(rt.mp,acts,record['rows'])
    with torch.no_grad():
        roll=rollout_after_movement(rt.model,initial,n_steps=600,cav_controller=cb,**kw)
        score=rt.frozen.costs(roll,entries,b.entry_spec['k0'])
        audit=entries.audit_population(initial,roll)
    identity=digest(dict(world=record['world_id'],state=record['state_id'],future=future,
                         sampling=receipt))
    results={str(h):{a['id']:dict(draw_id=identity,team_delay=score[str(h)]['real_J'][i],breakdown=None)
                         for i,a in enumerate(acts)} for h in (120,300)}
    return dict(results=results,sampling=receipt,population_audit=audit,
                road_source={str(h):{a['id']:[score[str(h)]['real_road'][i],score[str(h)]['real_source'][i]]
                                     for i,a in enumerate(acts)} for h in (120,300)})


def diagnostic_source(rt,record,future):
    """Roster-only EXAM lacks queues/history. Marginalize a frozen FIT source prior.

    This deliberately weaker observation contract is explicit in EVERY receipt.
    No scenario demand, source truth, exact outcome or future tape is consulted.
    """
    donor=int(digest(dict(state=record['state_id'],future=future))[:8],16)%32
    source=copy.deepcopy(rt.batches[donor])
    rows=record['rows']; n=len(rows)
    device=source.state['s_m'].device
    def tensor(values,dtype=torch.float32): return torch.tensor([values],dtype=dtype,device=device)
    source.state=dict(s_m=tensor([r['s_m'] for r in rows]),v_mps=tensor([r['v_mps'] for r in rows]),
        lane=tensor([r['lane'] for r in rows],torch.long),present=tensor([True]*n,torch.bool),
        is_cav=tensor([r['is_cav'] for r in rows],torch.bool),length_m=tensor([r.get('length_m',5.) for r in rows]),
        a_mps2=tensor([0.]*n))
    source.rollout_kwargs={k:torch.zeros((1,n,*v.shape[2:]),dtype=v.dtype,device=device)
                           for k,v in source.rollout_kwargs.items()}
    source.metadata=[dict(seed=record['world_id'],t0=300,prior_donor=969400+donor,
                          initial_state_sha256=digest(rows))]
    return source


def diagnostic_future(rt,b,initial,record,future):
    from wms.control.hidden_state import Population,plan_hidden_state_futures
    from wms.control.fitted_hidden import IDM_ORDER
    from wms.model.driver_encoder import PARAMETER_NAMES
    from wms.physics.state import state_kin
    from tools.rbint.recipe import InstrumentedEntries
    from wms.control.arrival_fit import LearnedArrivalFit
    rows=initial['present'].shape[0]
    defaults=rt.model._default_driver_parameters(state_kin(initial),*initial['present'].shape)
    fitted=rt.model._wmphys_fitted_state
    pool=defaults[0,0].expand(len(fitted.support),-1).clone()
    for i,name in enumerate(IDM_ORDER): pool[:,PARAMETER_NAMES.index(name)]=fitted.support[:,i].to(pool)
    pop=Population([pool.clone() for _ in range(rows)],
        [defaults[i,:1].clone() for i in range(rows)],[fitted.weights.clone() for _ in range(rows)])
    donor=b.metadata[0]['prior_donor']; world=record['world_id']
    original=rt.model._wmphys_arrival_fit
    arrival=LearnedArrivalFit({**original.artifact,'worlds':{str(world):original.artifact['worlds'][str(donor)]}})
    arrival.sha256=original.sha256
    kw=rt.frozen.expand(b.rollout_kwargs,rows)
    plans=plan_hidden_state_futures(rt.model,initial,kw,future_sampling='hidden_state',n_samples=8,
        seed=190,row_keys=[world]*rows,population=pop,
        entrance_histories=[b.entry_spec['histories'][0]]*rows,queues=[b.entry_spec['queues'][0]]*rows,
        t0_s=300.,arrival_fit=arrival)
    plan=plans[future]
    process=plan.kwargs.pop('entries'); entries=InstrumentedEntries(process,rt.frozen.Entries)
    kw.update(plan.kwargs,entries=entries)
    plan.receipt.update(observation_contract='EXAM_roster_only',causal_history_available=False,
        source_queue_observed=False,source_prior_donor=donor,
        source_prior='uniform_32_FIT_observed_queue_and_arrival_posteriors',
        driver_prior='frozen_FIT_joint_population_no_saved_ID_lookup',
        controller_memory='cold_start',absolute_time_s=record['time_s'])
    return kw,entries,plan.receipt
