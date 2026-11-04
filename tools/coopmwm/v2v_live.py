"""Qualified adapters for isolated V2V roots sharing the authoritative ledger.

Bank input subdirectories are read-only views of the ongoing same-arm bank;
lesson, safety, optimizer and diagnostic outputs belong to the new run root.
"""
from pathlib import Path
import time
import types


def accounting(root, cfg):
    from . import accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    from tools.coopnative import storage
    source=Path(cfg['accounting_root'])
    for name in ('ACCOUNTING.claim','CHARGES.jsonl'):
        if (Path(root)/name).resolve() != (source/name).resolve():
            raise ValueError('V2V must use the existing ledger AND its lock')
    budget=cfg['budget']
    if budget['total'] != 17471840 or sum(budget['phases'].values())>budget['total']:
        raise ValueError('V2V registered cap/allocation differs')
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,CAP=budget['total']-SUPERSEDED_CHARGE,CAPS=budget['phases']))


def hooks(root, cfg):
    from . import replay_bank as rb, warm_health as health
    from . import bank_imitation, bank_statistics
    # Owner 2026-11-03 21:45 + audit: the V2V runtime (B and R alike) uses per_seed_v1 only.
    rule = cfg.get('bank_statistics', bank_statistics.PER_SEED)
    if rule != bank_statistics.PER_SEED:
        raise ValueError(f'V2V runtime requires bank_statistics=per_seed_v1, not {rule!r}')
    bank_statistics.install(rb, bank_rule=rule, sources=cfg.get('replay_sources'))
    bank_imitation.run=bank_imitation.run_separate
    from .v2v_runtime import Policy, load_checkpoint
    from .accel_storage import read
    original_pairs,original_quality,original_lesson=rb.bank_pairs,rb.bank_quality,rb.record_lesson

    def configure(r,arm,binding):
        if Path(r)!=Path(root) or not binding['v2v'] or binding.get('sumo_verify'):
            raise ValueError('V2V own-policy episodes only')

    def safety(r,arm,u,episodes,ignored):
        if arm=='A':return
        reference=Path(r)/f'A_UPDATE_{u:03}.json'
        while not reference.exists():
            health.assert_running(r,arm)
            if time.time()>cfg['gpu_deadline']:raise ValueError('registered resource deadline')
            time.sleep(5)
        refs=[]
        for e in episodes:
            p=Path(r)/'jobs'/f'v2v1_warm_u{u:03}_A_{e["world"]["seed"]}'/'RESULT.json'
            if not p.exists():raise ValueError('missing same-world V2V A safety reference')
            refs.append(read(p))
        health.braking(r,arm,u,episodes,refs)

    policy_factory,pairs=Policy,original_pairs
    # Owner 2026-11-04 00:55: imitation_bank_filter is ONE key for B and R (A has no bank).
    from .v2v_launch3 import imitation_filter_for
    bank_filters={a:imitation_filter_for(cfg,a) for a in ('B','R')}
    if bank_filters['B']!='all_admitted':
        def pairs(r,arm,*a,**k):
            return original_pairs(r,arm,*a,**dict(k,imitation_bank_filter=bank_filters[arm]))
    if cfg.get('v2v_intent'):
        # v2v_intent ON (identical for A/B/R): intent-head policy, intent-carrying bank pairs.
        import functools
        from .v2v_intent import IntentPolicy,settings
        s=settings(cfg)
        policy_factory=functools.partial(IntentPolicy,s['horizon'],s['loss_weight'],s['bounds'],s['detach_trunk'])
        pairs=functools.partial(pairs,v2v_intent=cfg)
    return dict(configure=configure,accounting=accounting,policy_factory=policy_factory,
        load_checkpoint=load_checkpoint,bank_pairs=pairs,bank_quality=original_quality,
        record_lesson=original_lesson,health_before_update=health.before_update,safety=safety,
        diagnostics=lambda r,a,u,p,b,l:diagnostics(r,a,u,p,b,l))


def diagnostics(root, arm, update, policy, cfg, lesson):
    if update not in cfg['diagnostic_schedule'][arm]['isolated']:return lesson
    isolated(root,arm,update,policy,cfg)
    return lesson


def isolated(root, arm, update, policy, cfg):
    """Group/solos/background and a fresh legacy-u3 reference, fully precharged."""
    import torch
    from . import replay_bank as rb, policy_synergy as ps, sumo_verify as sv
    from .exact_grade import _seal
    from .storage import read,write
    from .warm_health import assert_running
    root=Path(root);base=root/'isolated'/f'u{update:03}_{arm}'
    output=root/f'POLICY_SYNERGY_u{update:03}_{arm}_V2V_ISOLATED.json'
    if output.exists():return read(output)
    base.mkdir(parents=True,exist_ok=True)
    if (base/'CLAIM.json').exists():raise ValueError('interrupted V2V diagnostic claim; never replay')
    confirmed=rb.confirmations(root,arm)
    decisions=[r for r in rb.bank_decisions(root,arm,sources=cfg['replay_sources'].get(arm,()))
        if confirmed.get(r['verification_id'],{}).get('admitted')
        and confirmed[r['verification_id']].get('cooperation',{}).get('cooperative') is True]
    links={read(p)['verification_id']:read(p) for p in (rb.bank_root(root,arm)/'decisions').glob('*.json')}
    old=torch.load(Path(cfg['legacy_root'])/f'{arm}_checkpoint_003.pt',weights_only=False)['policy']
    new={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    write(base/'CLAIM.json',dict(update=update,arm=arm,verification_ids=[r['verification_id'] for r in decisions],
        reference='original non-V2V arm u3; not this V2V arm u3',phase='v2v_isolated'))
    rows=[]
    for receipt in decisions:
        vid=receipt['verification_id']
        snapshot=rb._decision_snapshot(root,arm,receipt,links)
        pairs=ps.specs_for(snapshot,receipt,update,arm,new,old,isolated=True)
        reference=ps.specs_for(snapshot,receipt,update,arm,old,old)[0]
        pairs.append(reference)
        renamed=[]
        for i,(spec,snap) in enumerate(pairs):
            spec=dict(spec);spec.pop('branch_sha256')
            spec.update(verification_id=f'v2viso_u{update:03}_{arm}_{vid}_{i}',branch='reference' if i==len(pairs)-1 else spec['branch'])
            renamed.append((_seal(spec,'branch_sha256'),snap))
        runner=ps.BranchRunner(base/'branches',workers=1)
        blocked=runner.admit([s for s,_ in renamed])
        if blocked:raise RuntimeError(blocked)
        rb._confirm_charge(root,'v2v_isolated',f'v2v_isolated_u{update:03}_{arm}')(vid,[s for s,_ in renamed])
        results=[]
        from .bank_v2v import enabled
        for spec,snap in renamed:
            assert_running(root,arm)
            runner.prepare([spec],snap)
            with enabled(True):results.append(sv.run_branch(spec,snap,runner))
        ref=results.pop();background=results.pop()
        if ref['status']=='graded':
            measures=ps.metrics(ref['J_veh_s'],results,receipt['chosen_plan']['members'])
            measures['background_only_delta_veh_s']=(ref['J_veh_s']-background['J_veh_s']
                if background['status']=='graded' else None)
        else:measures=dict(status='incomplete',policy_cooperation_gain_veh_s=None)
        if background['status']!='graded':measures['status']='incomplete'
        row=dict(verification_id=vid,group=receipt['chosen_plan']['members'],results=results,
            reference=ref,background=background,window=receipt['window'],
            plan_cooperation_gain_veh_s=confirmed[vid]['cooperation']['cooperation_gain_veh_s'],**measures)
        write(base/(vid+'.json'),row);rows.append(row)
    result=_seal(dict(arm=arm,update=update,entries=rows,
        status='complete' if all(r['status']=='graded' for r in rows) else 'incomplete',
        phase='v2v_isolated',v2v=True,reference='legacy u3, identical snapshot/prefix/CRN',
        rule='group new vs each solo new; all other CAVs legacy u3; background-only separately'),
        'policy_synergy_sha256')
    write(output,result);return result


def run(root, cfg, release, arm):
    from .v2v_runner import run_arm
    run_arm(root,arm,cfg,release,hooks(root,cfg))


def resource_gate(root, cfg):
    import shutil
    from .warm_health import assert_running
    for arm in ('B','R'):assert_running(root,arm)
    free=shutil.disk_usage(root).free
    return dict(ready=free>=10*1024**3,disk_free_bytes=free,
        required_free_bytes=10*1024**3,rule='MAIN15:26/15:46 launch resource gate')


def configuration(root):
    from .storage import read
    from .protocol import sha
    root=Path(root);v=read(root/'V2V_RUNTIME.json')
    if v['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():
        raise ValueError('V2V runtime authority missing')
    source=Path(v['config']['source_root']);cpu=read(source/'LS_CPU_RESULT.json')
    if sha(source/'LS_CPU_RESULT.json')!=v['cpu_result_sha256']:
        raise ValueError('V2V qualification receipt changed')
    if not cpu['pass_'] or cpu['sim_s'] or cpu['skips']:raise ValueError('V2V Linux release gate')
    # The qualification manifest contains two inherited generic assignment
    # implementation filenames. Do not reopen any assignment-named path at
    # runtime; their original metadata stays in the immutable CPU receipt.
    excluded=('pytools/traffic_shepherd/mixed_assignment.py',
              'runtime_src/traffic_shepherd/mixed_assignment.py')
    for name,h in cpu['source_hashes'].items():
        if name in excluded:continue
        if sha(source/name)!=h:raise ValueError('V2V immutable source mismatch: '+name)
    if Path(__file__).resolve()!=source/'pytools/tools/coopmwm/v2v_live.py':
        raise ValueError('V2V supervisor must run from qualified source')
    return v


def stage_arm(root, stage, arm, attempt=None):
    import copy
    from .storage import read,write
    root=Path(root);value=configuration(root);cfg=copy.deepcopy(value['config'])
    release=copy.deepcopy(value['release'])
    if stage=='smoke':
        cfg.update(qualification_stage='one_update_smoke',smoke_authority_sha256=value['amendment_sha256'],
            rehearsal_late_60min=True,stop_after_update=1,A_stop_after_update=1,
            runtime_receipt_suffix='_V2V_SMOKE')
    else:
        smoke=read(root/'V2V_REAL_SMOKES_PASS.json')
        if smoke['arms']!=['A','B','R'] or not smoke['pass_']:raise ValueError('all real smokes required')
        if stage=='resume':
            # MAIN 2026-11-04 07:35: continue this arm from its latest sealed update k at k+1.
            state=resume_state(root,cfg,arm,_attempt(attempt))
            cfg.update(resume_update=state['sealed_update'],A_resume_update=state['sealed_update'],
                runtime_receipt_suffix='_V2V_RESUME_'+attempt,
                next_not_before_s={arm:state['next_not_before_s']},resume_progress=state['progress'])
        else:
            cfg.update(resume_update=1,A_resume_update=1,runtime_receipt_suffix='_V2V_LIVE',
                next_not_before_s=smoke['next_not_before_s'])
        release.update(real_update_smokes_pass=True,smoke_arms=smoke['arms'])
    run(root,cfg,release,arm)
    write(root/f'{_label(stage,arm,attempt)}_COMPLETE.json',dict(arm=arm,stage=stage,unix=time.time(),
        update=1 if stage=='smoke' else 24))


def _label(stage, arm=None, attempt=None):
    return stage+('_'+attempt if attempt else '')+('_'+arm if arm else '')


def spawn(root, stage, arm=None, attempt=None):
    import subprocess,sys
    from .storage import read,write
    from .replay_start import owner,alive
    label=_label(stage,arm,attempt)
    root=Path(root);dispatch=root/(label+'_DISPATCH.json')
    if dispatch.exists():
        d=read(dispatch)
        if not alive(d) and not (root/(label+'_COMPLETE.json')).exists():
            raise ValueError('owned V2V child exited: '+label)
        return d
    # Every V2V dispatch uses the reviewed additive launch binding (audit B5): the launch3
    # preflight runs synchronously immediately before EACH Popen (smoke, train, test8/test24)
    # and any Refused aborts. The V2V-only (10 GiB resource_gate) route is disabled.
    cfg=configuration(root)['config']
    if not cfg.get('v2v_intent'):
        raise ValueError('V2V dispatch requires the intent launch3 binding; the V2V-only route is disabled')
    launch3_check(root,cfg,stage,arm)
    command=[sys.executable,'-B','-u','-m','tools.coopmwm.v2v_live',str(root),stage]
    if arm:command+=['--arm',arm]
    if attempt:command+=['--attempt',attempt]
    with (root/(label+'.log')).open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    d=dict(owner(child.pid),argv=command,label=label);write(dispatch,d);return d


def _require_launch3(root, cfg):
    """Audit B5: intent launch3 binding mandatory; never collide with another launcher's start."""
    if not cfg.get('v2v_intent'):
        raise ValueError('V2V watcher requires the intent launch3 binding; the V2V-only route is disabled')
    if (Path(root)/'V2V_STARTED.json').exists():
        raise ValueError('V2V_STARTED.json already exists: another launcher owns this root')


SMOKE_EPISODE_SIM_S=1200     # one registered update-1 smoke/train episode (launch3_check unit)
SMOKE_EPISODES_PER_ARM=24


def remaining_phase_costs(root, cfg):
    """Exact remaining registered phase_costs for a (re)started watcher gate.

    requote's contract: a later wave passes its exact remaining positive
    phase_costs, never the whole initial run again. A watcher can only be
    (re)started before V2V_STARTED (``_require_launch3``), i.e. while the real
    one-update smokes are still the only work that may have been charged. The
    restart quote therefore credits ONLY reusable smoke physics of THIS root:
    a v2v_train charge whose id is ``<job_namespace>_warm_u001_<A|B|R>_<seed>``,
    of exactly one registered episode (1200 sim-s), whose job directory under
    this root is closed with a result (the runner reuses it, never re-runs or
    re-charges it). Everything else stays in the full registered quote:
    foreign, void, partial or unexpected charges are NOT credited, so they keep
    counting as spent in requote and the quote can never fall below the real
    remaining work (an overdrawn or over-committed ledger is still refused).
    Nothing credited -> None -> the unchanged full quote (fresh-root behaviour
    byte-identical). Any v2v diagnostic/test phase already charged means the
    run is past the smokes: no credit (full quote, which requote refuses).
    """
    import re
    from . import v2v_launch3,storage
    namespace=cfg.get('job_namespace')
    if not isinstance(namespace,str) or not namespace:return None
    try:entries=storage.ledger(root)
    except (OSError,ValueError):return None
    pattern=re.compile(re.escape(namespace)+r'_warm_u001_([ABR])_([0-9]+)')
    credited,per_arm=set(),{}
    for entry in entries:
        if not isinstance(entry,dict):return None
        phase,ident=entry.get('phase'),entry.get('id')
        if phase in v2v_launch3.V2V_PHASES and phase!='v2v_train':return None
        if phase!='v2v_train' or not isinstance(ident,str):continue
        match=pattern.fullmatch(ident)
        if not match:continue
        if ident in credited:return None    # duplicate charge: requote refuses it
        amount=entry.get('sim_s')
        if type(amount) is not int or amount!=SMOKE_EPISODE_SIM_S:continue
        job=Path(root)/'jobs'/ident
        if not ((job/'CLOSE.json').is_file() and (job/'RESULT.json').is_file()):continue
        credited.add(ident);per_arm[match[1]]=per_arm.get(match[1],0)+1
    if not credited:return None
    if any(n>SMOKE_EPISODES_PER_ARM for n in per_arm.values()):
        raise ValueError('more reusable update-1 episodes than one registered smoke per arm')
    costs=dict(v2v_launch3.V2V_PHASES)
    costs['v2v_train']-=SMOKE_EPISODE_SIM_S*len(credited)
    if costs['v2v_train']<=0:raise ValueError('restart credit cannot consume the training quote')
    return costs


def watch_gate(root, cfg):
    """Initial launch3 gate of a (re)started watcher: the full registered quote
    on a fresh ledger; after closed smoke episodes of this root were charged,
    the full quote minus exactly those reusable episodes."""
    return launch3_check(root,cfg,phase_costs=remaining_phase_costs(root,cfg))


def launch3_check(root, cfg, stage=None, arm=None, phase_costs=None):
    """Validate a registered additive trust anchor immediately before dispatch.

    The anchor lives outside immutable V2V_RUNTIME, avoiding a circular hash
    between the runtime and the launch binding that pins that runtime.
    Only resource waits are retryable; every other failed condition is fatal.
    A stageless watcher (re)start takes the exact remaining registered
    phase_costs via ``phase_costs`` (None keeps the full frozen-slate quote);
    spawned children always keep their own registered unit cost.
    """
    from . import v2v_launch3 as launch3
    from .protocol import sha
    from .storage import read
    from .replay_start import alive
    root=Path(root)
    path=root/'LAUNCH3_AUTHORITY.json'
    authority=read(path)
    if sha(path) not in (root/'AUTHORITY.md').read_text():
        raise ValueError('launch3 additive authority is not registered')
    boundary=authority['legacy_boundary']
    if set(boundary['owners'])!={'B','R'} or set(boundary['checkpoints'])!={'B','R'}:
        raise ValueError('both legacy arm boundary proofs required')
    def check_boundary():
        for a,d in boundary['owners'].items():
            if alive(d):raise ValueError('legacy arm still running: '+a)
            saved=boundary['checkpoints'][a]
            expected=Path(cfg['legacy_root'])/f'{a}_checkpoint_{saved["update"]:03}.pt'
            if Path(saved['path'])!=expected or sha(expected)!=saved['sha256']:
                raise ValueError('legacy boundary checkpoint mismatch: '+a)
    check_boundary()
    costs=None
    if stage in ('smoke','train'):
        if arm not in ('A','B','R'):raise ValueError('launch3 training arm required')
        costs={'v2v_train':24*1200*(1 if stage=='smoke' else 23)}
    elif stage=='resume':
        # Exact remaining registered episodes of THIS resumed arm (closed reusable
        # episodes of its in-progress update credited); never a caller-supplied amount.
        if arm not in ('A','B','R'):raise ValueError('launch3 training arm required')
        need=arm_remaining(root,cfg,arm,live=False)['remaining_sim_s']
        if need<=0:raise ValueError('resumed arm has no remaining registered update: '+arm)
        costs={'v2v_train':need}
    elif stage in ('test8','test24'):
        costs={'v2v_interim' if stage=='test8' else 'v2v_final':3*96*1200}
    elif stage is None:
        costs=phase_costs
    else:
        raise ValueError('unregistered launch3 stage: '+str(stage))
    result=launch3.wait_for_preflight(authority['binding_path'],
        binding_sha256=authority['binding_sha256'],root=root,
        control=authority['control_root'],mover=authority['mover_root'],phase_costs=costs,
        wait_seconds=max(0,int(cfg['gpu_deadline']-time.time())))
    check_boundary()
    launch3.check_hold(authority['control_root'],root)
    return result


def watch(root):
    import fcntl,shutil
    from .storage import read,write
    from .protocol import sha
    from .replay_start import owner,alive,atomic
    root=Path(root);value=configuration(root);cfg=value['config']
    lock=(root/'WATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'WATCH_OWNER.json').exists():raise ValueError('adopt existing V2V watcher')
    write(root/'WATCH_OWNER.json',owner())
    children=[]
    # Audit B5 (c4/launch-final): the launch3 preflight (= v2v_launch3.before_launch: HOLD,
    # binding, deployed commit, runtime, PASS gate, resources, budget) is the ONLY dispatch
    # gate. The former V2V-only 10 GiB resource_gate route is disabled, and an existing
    # V2V_STARTED.json (another launcher) refuses before any child is spawned.
    _require_launch3(root,cfg)
    # Full frozen-slate quote and all launch prerequisites once before the
    # first smoke; each new child then checks its own remaining phase cost.
    # A restarted watcher (smokes already charged part of v2v_train) passes the
    # exact remaining quote instead; a fresh ledger keeps the full quote.
    gate=watch_gate(root,cfg)
    atomic(root/'V2V_WATCH_STATUS.json',dict(ready=True,rule='registered launch3 preflight checked',
        unix=time.time(),stage='resource_wait',children=children,preflight=bool(gate)))
    children=[spawn(root,'smoke',a) for a in ('A','B','R')]
    while not all((root/f'smoke_{a}_COMPLETE.json').exists() for a in ('A','B','R')):
        for child in children:
            if not alive(child) and not (root/(child['label']+'_COMPLETE.json')).exists():
                raise ValueError('V2V smoke failed: '+child['label'])
        atomic(root/'V2V_WATCH_STATUS.json',dict(unix=time.time(),stage='real_one_update_smokes',children=children))
        time.sleep(10)
    next_times={}
    for arm in ('A','B','R'):
        jobs=list((root/'jobs').glob('v2v1_warm_u001_'+arm+'_*/CLAIM.json'))
        if len(jobs)!=24 or not (root/f'{arm}_checkpoint_001.pt').exists():
            raise ValueError('real V2V smoke incomplete: '+arm)
        next_times[arm]=min(p.stat().st_mtime for p in jobs)+cfg['pace_s']
    write(root/'V2V_REAL_SMOKES_PASS.json',dict(pass_=True,arms=['A','B','R'],unix=time.time(),
        next_not_before_s=next_times,qualified_source=cfg['source_root'],
        smoke_checkpoints={a:sha(root/f'{a}_checkpoint_001.pt') for a in ('A','B','R')},
        simulation_reused_for_training=True,new_extra_smoke_sim_s=0))
    for arm in ('A','B','R'):
        path=root/f'WARM_{arm}_PAUSED.json'
        if read(path)['update']!=1:raise ValueError('smoke endpoint mismatch')
        saved=root/f'WARM_{arm}_PAUSED_AFTER_SMOKE.json'
        if saved.exists():raise ValueError('smoke endpoint already adopted')
        path.rename(saved)
    _require_launch3(root,cfg)
    children=[spawn(root,'train',a) for a in ('A','B','R')]
    write(root/'V2V_STARTED.json',dict(unix=time.time(),children=children,first_update=1,
        resumed_after_smoke=2,final_update=24,common_u0_sha256=cfg['common_checkpoint_sha256'],
        funding_amendment_sha256=cfg['funding_amendment_sha256']))
    # Episode tests are separate durable children; the coordinator receives only
    # public hashes and never opens outcome values during training.
    while True:
        for child in children:
            if not alive(child) and not (root/(child['label']+'_COMPLETE.json')).exists():
                raise ValueError('V2V training child exited: '+child['label'])
        for update in (8,24):
            if all((root/f'{a}_UPDATE_{update:03}.json').exists() for a in ('A','B','R')):
                spawn(root,'test'+str(update))
        if all((root/f'train_{a}_COMPLETE.json').exists() for a in ('A','B','R')):
            if all((root/f'test{u}_COMPLETE.json').exists() for u in (8,24)):break
        atomic(root/'V2V_WATCH_STATUS.json',dict(unix=time.time(),stage='training',children=children))
        time.sleep(30)
    write(root/'WATCH_COMPLETE.json',dict(unix=time.time(),outcomes_closed=True))


# ---------------------------------------------------------------- resume a started root
# MAIN 2026-11-04 07:35 (conservative path inside the reviewed watcher): a started V2V root
# whose B/R children stopped on an engineering fault continues through the SAME launch3
# preflight (v2v_launch3 unchanged): (1) ``resume_prepare`` moves - never deletes - the dead
# dispatch/failure/stop records and the stale unsealed update receipts of the resumed arms,
# with a SHA receipt; (2) ``adopt`` keeps every live child (spawn's alive-dispatch reuse),
# quotes exactly the remaining registered work and dispatches one ``resume`` child per
# stopped arm; (3) ``resume`` restarts an arm from its latest sealed {arm}_UPDATE_k /
# {arm}_checkpoint_k at k+1, reusing closed episodes of update k+1 only when their recorded
# policy outputs match checkpoint k. Every resume attempt has its own STARTED suffix.

TRAIN_FINAL_UPDATE=24
EPISODES_PER_UPDATE=24
EPISODE_SIM_S=1200
ARMS=('A','B','R')
REUSE_SAMPLES_PER_EPISODE=6
REUSE_TOLERANCE=1e-4


def _attempt(attempt):
    import re
    if not isinstance(attempt,str) or not re.fullmatch(r'r[0-9]{1,3}',attempt):
        raise ValueError('resume attempt label r<N> required')
    return attempt


def _arms(arms):
    arms=tuple(a for a in ARMS if a in set(arms or ()))
    if not arms or len(arms)!=len(set(arms or ())):raise ValueError('resumed arms must be a nonempty subset of A,B,R')
    return arms


def sealed_update(root, arm):
    """Latest k with a sealed {arm}_UPDATE_k.json (no gap) and its own regular checkpoint file."""
    root=Path(root);k=0
    while k<TRAIN_FINAL_UPDATE and (root/f'{arm}_UPDATE_{k+1:03}.json').is_file():
        checkpoint=root/f'{arm}_checkpoint_{k+1:03}.pt'
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError(f'sealed {arm} update {k+1} lacks its own checkpoint file')
        k+=1
    return k


def _train_rows(root, cfg):
    """This root's v2v_train episode charges by (update, arm); other ids are never credited."""
    import re
    from . import storage
    namespace=cfg.get('job_namespace')
    if not isinstance(namespace,str) or not namespace:raise ValueError('registered job_namespace required')
    pattern=re.compile(re.escape(namespace)+r'_warm_u([0-9]{3})_([ABR])_([0-9]+)')
    rows={}
    for entry in storage.ledger(root):
        if not isinstance(entry,dict):raise ValueError('malformed ledger entry')
        if entry.get('phase')!='v2v_train':continue
        match=pattern.fullmatch(str(entry.get('id')))
        if match:rows.setdefault((int(match[1]),match[2]),[]).append(entry)
    return rows


def arm_remaining(root, cfg, arm, live, rows=None):
    """Exact remaining registered episodes of one arm (accounting only; no outcome opened).

    remaining = (24 - k) x 24 - credited, k = latest sealed update. Only charges of update
    k+1 can be credited: a LIVE arm's charged k+1 episodes are owned by its running child
    (never re-charged); a RESUMED arm's k+1 episodes are credited only when closed with a
    result (the runner reuses them). Credit never exceeds one registered update, so the
    quote is never below the real remaining work. A resumed arm refuses charged-but-unclosed
    k+1 episodes (never retried), charges beyond k+1 and a stray k+1 checkpoint.
    """
    root=Path(root)
    if arm not in ARMS:raise ValueError('registered arm required')
    rows=_train_rows(root,cfg) if rows is None else rows
    k=sealed_update(root,arm)
    ahead={u for (u,a) in rows if a==arm and u>k}
    if ahead-{k+1}:raise ValueError(f'{arm} charges beyond update {k+1} without a sealed update {k+1}')
    current=rows.get((k+1,arm),[])
    ids=[e['id'] for e in current]
    if len(set(ids))!=len(ids):raise ValueError('duplicate train charge: '+arm)
    if not live and k<TRAIN_FINAL_UPDATE:
        stray=root/f'{arm}_checkpoint_{k+1:03}.pt'
        if stray.exists() or stray.is_symlink():
            raise ValueError(f'unsealed {stray.name} present (u3-link trap): move it aside first')
    reusable=[]
    for entry in current:
        if type(entry.get('sim_s')) is not int or entry['sim_s']!=EPISODE_SIM_S:
            raise ValueError('unexpected train episode charge '+entry['id'])
        job=root/'jobs'/entry['id']
        closed=(job/'CLOSE.json').is_file() and (job/'RESULT.json').is_file()
        if not live and not closed:
            raise ValueError(f'charged update-{k+1} episode {entry["id"]} is not closed: '
                'it can never be retried; MAIN void decision required')
        if closed:reusable.append(entry['id'])
    credited=min(EPISODES_PER_UPDATE,len(current) if live else len(reusable))
    remaining=(TRAIN_FINAL_UPDATE-k)*EPISODES_PER_UPDATE-credited if k<TRAIN_FINAL_UPDATE else 0
    return dict(arm=arm,live=bool(live),sealed_update=k,resume_update=min(k+1,TRAIN_FINAL_UPDATE),
        charged_current_update=len(current),reusable=sorted(reusable),credited_episodes=credited,
        remaining_episodes=remaining,remaining_sim_s=remaining*EPISODE_SIM_S)


def _train_snapshot(root, cfg):
    return tuple(sorted(e['id'] for v in _train_rows(root,cfg).values() for e in v))


def resume_phase_costs(root, cfg, live_arms, resume_arms):
    """Remaining registered phase_costs of a started root: v2v_train = sum over arms of the
    exact remaining episodes; every diagnostic/test phase keeps its full registered quote."""
    from . import v2v_launch3
    if set(live_arms)&set(resume_arms) or set(live_arms)|set(resume_arms)!=set(ARMS):
        raise ValueError('every arm is either live (adopted) or resumed')
    rows=_train_rows(root,cfg)
    per_arm={a:arm_remaining(root,cfg,a,a in live_arms,rows) for a in ARMS}
    costs=dict(v2v_launch3.V2V_PHASES)
    train=sum(p['remaining_sim_s'] for p in per_arm.values())
    if train>costs['v2v_train']:raise ValueError('remaining train work exceeds the registered phase')
    if train:costs['v2v_train']=train
    else:costs.pop('v2v_train')
    return costs,per_arm


def adopt_gate(root, cfg, live_arms, resume_arms, attempts=5):
    """launch3 preflight with the exact remaining quote. A live arm may charge its next
    update between the quote and the ledger read: re-quote only when this root's train
    charges changed (bounded); any other refusal is final."""
    from . import v2v_launch3
    for attempt in range(attempts):
        before=_train_snapshot(root,cfg)
        costs,per_arm=resume_phase_costs(root,cfg,live_arms,resume_arms)
        try:
            return launch3_check(root,cfg,phase_costs=costs),costs,per_arm
        except v2v_launch3.Refused:
            if attempt==attempts-1 or _train_snapshot(root,cfg)==before:raise
            time.sleep(1)


def _moved(path, dest):
    """Rename (same filesystem), never delete; SHA/identity receipt for the move."""
    import os
    from .protocol import sha
    stat=os.lstat(path);link=path.is_symlink()
    before=os.readlink(path) if link else sha(path)
    target=dest/path.name
    if target.exists() or target.is_symlink():raise ValueError('archive target exists: '+str(target))
    os.rename(path,target)
    after=os.readlink(target) if link else sha(target)
    if after!=before:raise ValueError('archived bytes changed: '+path.name)
    return dict(name=path.name,source=str(path),moved_to=str(target),symlink=link,
        **({'link_target':before} if link else {'sha256':before}),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def resume_prepare(root, attempt, resume_arms):
    """Zero physics: move the dead watcher/child records and stale unsealed receipts of the
    resumed arms into resume_<attempt>_archive/ with a SHA receipt. Refuses live owners,
    health pauses, interrupted diagnostics and stops that are not the archived failure."""
    from .storage import read,write
    from .replay_start import alive
    from .warm_health import assert_running
    root=Path(root);attempt=_attempt(attempt);resume_arms=_arms(resume_arms)
    dest=root/f'resume_{attempt}_archive'
    if dest.exists():raise ValueError('resume attempt archive already exists: '+dest.name)
    if (root/f'V2V_RESUME_{attempt}.json').exists():raise ValueError('one-use resume attempt already started')
    selected=[]
    for name in ('WATCH_OWNER.json','WATCH_DISPATCH.json'):
        if (root/name).is_file() and alive(read(root/name)):raise ValueError('live V2V watcher: '+name)
    for name in ('WATCH_OWNER.json','WATCH_DISPATCH.json','watch_FAILED.json','WATCH.log',
                 'V2V_WATCH_STATUS.json'):
        selected.append(root/name)
    selected+=sorted(root.glob('adopt_r*_FAILED.json'))+sorted(root.glob('ADOPT_r*.log'))
    failures={}
    for arm in resume_arms:
        if (root/f'train_{arm}_COMPLETE.json').exists():raise ValueError('arm already complete: '+arm)
        k=sealed_update(root,arm)
        if k>=TRAIN_FINAL_UPDATE:raise ValueError('arm has no remaining update: '+arm)
        for dispatch in [root/f'train_{arm}_DISPATCH.json']+sorted(root.glob(f'resume_r*_{arm}_DISPATCH.json')):
            if not dispatch.is_file():continue
            if alive(read(dispatch)):raise ValueError('live child of a resumed arm: '+dispatch.name)
            label=dispatch.name[:-len('_DISPATCH.json')]
            if (root/(label+'_COMPLETE.json')).exists():raise ValueError('completed child: '+label)
            selected+=[dispatch,root/(label+'_FAILED.json'),root/(label+'.log')]
        for record in sorted(root.glob(f'WARM_{arm}_FAILED_*.json')):
            error=str(read(record).get('error',''))
            if error.startswith('HealthPause'):raise ValueError('registered health pause needs a MAIN release: '+record.name)
            failures.setdefault(arm,set()).add(error)
            selected.append(record)
        if (root/'isolated'/f'u{k+1:03}_{arm}'/'CLAIM.json').exists():
            raise ValueError(f'interrupted u{k+1} {arm} diagnostic claim; never replayed')
        stray=root/f'{arm}_checkpoint_{k+1:03}.pt'
        if stray.exists() or stray.is_symlink():raise ValueError('unsealed checkpoint present: '+stray.name)
        # exclusive pre-checkpoint receipts of the unsealed update are re-written by the resume;
        # the composite safety file stays (the resume re-derives it and must find it identical).
        selected+=[root/f'{arm}_VIABILITY_{k+1:03}.json']
        if arm=='B':selected.append(root/f'B_SEARCH_LEARNING_{k+1:03}.json')
    stop=root/'WARM_GUIDED_STOP.json'
    if stop.exists():
        value=read(stop)
        if value.get('arm') not in resume_arms or str(value.get('error')) not in failures.get(value.get('arm'),()):
            raise ValueError('WARM_GUIDED_STOP.json is not the record of an archived resumed-arm failure')
        selected.append(stop)
    dest.mkdir()
    moved=[]
    for path in dict.fromkeys(selected):
        if path.exists() or path.is_symlink():moved.append(_moved(path,dest))
    for arm in resume_arms:
        assert_running(root,arm)   # any other STOP/guided stop/unreleased pause still refuses
    receipt=dict(attempt=attempt,resume_arms=list(resume_arms),unix=time.time(),moved=moved,
        sealed_updates={a:sealed_update(root,a) for a in resume_arms},new_sim_s=0,
        rule='moved, never deleted; live owners, health pauses and interrupted diagnostics refused')
    write(dest/'ARCHIVE_RECEIPT.json',receipt)
    return receipt


def adopt(root, attempt, resume_arms):
    """Watcher for a STARTED root: keep live children, resume the stopped arms, then the
    registered interim/final tests. launch3 preflight before every dispatch (unchanged)."""
    import fcntl
    from .storage import read,write
    from .protocol import sha
    from .replay_start import owner,alive,atomic
    from .warm_health import assert_running
    root=Path(root);attempt=_attempt(attempt);resume_arms=_arms(resume_arms)
    live_arms=tuple(a for a in ARMS if a not in resume_arms)
    value=configuration(root);cfg=value['config']
    if not cfg.get('v2v_intent'):
        raise ValueError('V2V watcher requires the intent launch3 binding; the V2V-only route is disabled')
    if not (root/'V2V_STARTED.json').is_file():raise ValueError('adopt mode is only for a started root; use watch')
    smoke=read(root/'V2V_REAL_SMOKES_PASS.json')
    if smoke['arms']!=['A','B','R'] or not smoke['pass_']:raise ValueError('all real smokes required')
    lock=(root/'WATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'WATCH_OWNER.json').exists():
        raise ValueError('watcher owner record present: adopt the live watcher or archive it (resume_prepare)')
    archive=root/f'resume_{attempt}_archive'/'ARCHIVE_RECEIPT.json'
    if not archive.is_file() or read(archive)['resume_arms']!=list(resume_arms):
        raise ValueError('resume_prepare receipt for this attempt and arm set required')
    plan_path=root/f'V2V_RESUME_{attempt}.json'
    if plan_path.exists():raise ValueError('one-use resume attempt already started: '+plan_path.name)
    for arm in live_arms:
        dispatch=root/f'train_{arm}_DISPATCH.json'
        if not dispatch.is_file():raise ValueError('live arm without a training dispatch: '+arm)
        if not alive(read(dispatch)) and not (root/f'train_{arm}_COMPLETE.json').exists():
            raise ValueError('adopted training child exited: '+arm)
    for arm in resume_arms:
        for dispatch in [root/f'train_{arm}_DISPATCH.json']+sorted(root.glob(f'resume_r*_{arm}_DISPATCH.json')):
            if dispatch.exists():raise ValueError('archive the previous dispatch of a resumed arm: '+dispatch.name)
        assert_running(root,arm)
    write(root/'WATCH_OWNER.json',owner())
    gate,costs,per_arm=adopt_gate(root,cfg,live_arms,resume_arms)
    write(plan_path,dict(attempt=attempt,unix=time.time(),live_arms=list(live_arms),
        resume_arms=list(resume_arms),arms=per_arm,phase_costs=costs,
        quote_total_sim_s=sum(costs.values()),archive_receipt_sha256=sha(archive),
        preflight=dict(ready=bool(gate),budget=gate.get('budget') if isinstance(gate,dict) else None),
        source_root=cfg['source_root'],runtime_sha256=sha(root/'V2V_RUNTIME.json'),
        rule='MAIN 2026-11-04 07:35: resume from the latest sealed update; live children kept; '
             'exact remaining registered quote; launch3 checks unchanged'))
    children=[spawn(root,'train',a) for a in live_arms]
    children+=[spawn(root,'resume',a,attempt) for a in resume_arms]
    atomic(root/'V2V_WATCH_STATUS.json',dict(unix=time.time(),stage='training',mode='adopt',
        attempt=attempt,children=children))
    while True:
        for child in children:
            if not alive(child) and not (root/(child['label']+'_COMPLETE.json')).exists():
                raise ValueError('V2V training child exited: '+child['label'])
        for update in (8,24):
            if all((root/f'{a}_UPDATE_{update:03}.json').exists() for a in ARMS):
                spawn(root,'test'+str(update))
        if all((root/(c['label']+'_COMPLETE.json')).exists() for c in children):
            if all((root/f'test{u}_COMPLETE.json').exists() for u in (8,24)):break
        atomic(root/'V2V_WATCH_STATUS.json',dict(unix=time.time(),stage='training',mode='adopt',
            attempt=attempt,children=children))
        time.sleep(30)
    write(root/'WATCH_COMPLETE.json',dict(unix=time.time(),outcomes_closed=True,resume_attempt=attempt))


def verify_reuse(root, arm, policy, ids, samples=REUSE_SAMPLES_PER_EPISODE, tolerance=REUSE_TOLERANCE):
    """Closed episodes are reused only when they are the policy's own episodes: their
    recorded critic values and action log-probabilities are re-derived from the sealed
    checkpoint (the same calls PPO makes on these samples)."""
    import torch
    from .accel_storage import read
    from .accel_learner import own_step
    root=Path(root);rows=[]
    for ident in ids:
        job=root/'jobs'/ident
        episode=read(job/'RESULT.json');close=read(job/'CLOSE.json')
        if episode.get('id')!=ident or episode.get('arm')!=arm or episode.get('charged_sim_s')!=EPISODE_SIM_S:
            raise ValueError('reused episode identity differs: '+ident)
        if close.get('charged_sim_s')!=EPISODE_SIM_S:raise ValueError('reused episode close differs: '+ident)
        own=[s['sample'] for s in episode['steps'] if s.get('sample') and 'logp' in s['sample']
             and own_step(s['sample'])]
        if not own:raise ValueError('reused episode has no own-policy samples: '+ident)
        picks=sorted({round(i*(len(own)-1)/max(1,samples-1)) for i in range(samples)})
        worst=0.
        with torch.no_grad():
            for i in picks:
                sample=own[i]
                value=float(policy(sample['context'])[1])
                logp=float(policy.score(sample)[0])
                worst=max(worst,abs(value-float(sample['value'])),abs(logp-float(sample['logp'])))
        if not worst<=tolerance:
            raise ValueError(f'reused episode {ident} was not produced by the sealed checkpoint '
                f'(max |diff| {worst:.3g} > {tolerance})')
        rows.append(dict(id=ident,samples=len(picks),max_abs_diff=worst))
        del episode
    return rows


def restore_progress(root, cfg, arm, sealed, first):
    """The uninterrupted training child's diagnostic state after update ``sealed``: trust
    (beta) and B search learning from the sealed receipts, evidence and the exact-diagnostic
    RNG re-derived from the closed train episodes of updates first..sealed (same calls as
    warm_runner). Policy, Adam and torch RNG come from the checkpoint itself."""
    import copy
    import numpy as np
    from .accel_learner import evidence
    from .accel_storage import read
    from .warm_protocol import SEED
    root=Path(root);namespace=cfg['job_namespace']
    running=[];rng=np.random.default_rng(SEED);count=0;missing=[]
    for u in range(first,sealed+1):
        candidates=0;found=[]
        for path in (root/'jobs').glob(f'{namespace}_warm_u{u:03}_{arm}_*/RESULT.json'):
            found.append((int(path.parent.name.rsplit('_',1)[1]),path))
        if len(found)<EPISODES_PER_UPDATE:missing.append(dict(update=u,closed_results=len(found)))
        for _,path in sorted(found):
            episode=read(path)
            running.extend(evidence([episode]))
            candidates+=sum(1 for s in episode['steps'] if s['time_s']<=600
                            and s['sample'].get('behaviour',{}).get('plan_start'))
            del episode
        if arm=='B':count+=len(rng.permutation(candidates)[:max(0,12-count)])
    trust=read(root/f'{arm}_UPDATE_{sealed:03}.json')['trust'] if sealed else 1.
    learned=read(root/f'B_SEARCH_LEARNING_{sealed:03}.json') if arm=='B' and sealed else None
    if arm=='B' and learned is None:raise ValueError('B search learning state of the sealed update required')
    return dict(running=running,beta=trust,learned=learned,
        exact_rng_state=copy.deepcopy(rng.bit_generator.state),exact_count=count),missing


def resume_state(root, cfg, arm, attempt):
    """Child side: re-verify the adopt quote, reuse proof and restored progress (one-use)."""
    from .storage import read,write
    from .v2v_runtime import load_checkpoint
    root=Path(root)
    plan=read(root/f'V2V_RESUME_{attempt}.json')
    if arm not in plan['resume_arms']:raise ValueError('arm not registered in this resume attempt: '+arm)
    info=arm_remaining(root,cfg,arm,live=False)
    quoted=plan['arms'][arm]
    if info['sealed_update']!=quoted['sealed_update'] or info['reusable']!=quoted['reusable']:
        raise ValueError('sealed update or reusable episodes changed since the adopt quote')
    k=info['sealed_update']
    if not 1<=k<TRAIN_FINAL_UPDATE:raise ValueError('resume requires a sealed update in 1..23')
    policy=load_checkpoint(root/f'{arm}_checkpoint_{k:03}.pt',config=cfg)[0]
    reuse=verify_reuse(root,arm,policy,info['reusable'])
    started=read(root/'V2V_STARTED.json')
    progress,missing=restore_progress(root,cfg,arm,k,int(started['resumed_after_smoke']))
    claims=[(root/'jobs'/i/'CLAIM.json').stat().st_mtime for i in info['reusable']]
    if claims:not_before=min(claims)    # the in-progress update's original paced start
    else:
        previous=list((root/'jobs').glob(f'{cfg["job_namespace"]}_warm_u{k:03}_{arm}_*/CLAIM.json'))
        if not previous:raise ValueError('sealed update claims required for pacing')
        not_before=min(p.stat().st_mtime for p in previous)+cfg['pace_s']
    from .protocol import digest
    write(root/f'RESUME_{attempt}_{arm}_STATE.json',dict(attempt=attempt,arm=arm,sealed_update=k,
        resume_update=k+1,reused=reuse,reused_sim_s=len(reuse)*EPISODE_SIM_S,
        progress_sha256=digest(progress),running_records=len(progress['running']),
        exact_count=progress['exact_count'],missing_results=missing,next_not_before_s=not_before,
        runtime_receipt_suffix='_V2V_RESUME_'+attempt,unix=time.time(),new_sim_s=0))
    return dict(sealed_update=k,progress=progress,next_not_before_s=not_before)


def execute_test(root,cfg,spec):
    import os,torch
    from .accel_plant import episode
    from .final_runtime import public_completion
    from .protocol import sha,digest
    from .storage import write
    torch.set_num_threads(1);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    weights=torch.load(Path(root)/f'{spec["arm"]}_checkpoint_{spec["checkpoint_update"]:03}.pt',weights_only=False)['policy']
    job=dict(spec,path=str(Path(root)/'jobs'/spec['id']),role='final_test',v2v=True,weights=weights,
        runtime=cfg['runtime'],checkpoint=cfg['checkpoint'],noise_seed=int(digest(spec)[:15],16),
        deterministic=True,guided=False,gpu_deadline=cfg['gpu_deadline'],accel_bounds=cfg['accel_bounds'])
    episode(job)
    meta=public_completion(spec,Path(job['path']))
    meta.update(checkpoint_update=spec['checkpoint_update'],checkpoint_sha256=sha(Path(root)/f'{spec["arm"]}_checkpoint_{spec["checkpoint_update"]:03}.pt'))
    write(Path(root)/'final_receipts'/(spec['id']+'.json'),meta)
    return meta


def episode_test(root, update):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from .warm_protocol import worlds
    from .storage import write
    from .accel_storage import reserve,ledger
    root=Path(root);cfg=configuration(root)['config'];accounting(root,cfg)
    if not all((root/f'{a}_UPDATE_{update:03}.json').exists() for a in ('A','B','R')):
        raise ValueError('fixed V2V checkpoints required')
    claim=root/f'test{update}_CLAIM.json'
    if claim.exists():raise ValueError('interrupted episode test; no replay')
    phase='v2v_interim' if update==8 else 'v2v_final'
    jobs=[dict(id=f'v2vtest{update:03}_{a}_{w["seed"]}',arm=a,world=w,sim_s=1200,
        phase=phase,kind='episode',checkpoint_update=update) for a in ('A','B','R') for w in worlds('final_test')]
    if {x['id'] for x in jobs}&{x['id'] for x in ledger(root)}:raise ValueError('test already charged')
    write(claim,dict(update=update,jobs=288,sim_s=345600,phase=phase,outcomes_closed=True))
    # Charge each complete episode immediately before submission, without
    # holding reservations during resource waits or retrying partial attempts.
    receipts=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for offset in range(0,len(jobs),2):
            pair=jobs[offset:offset+2]
            from .runner import admission
            admission(cfg)
            reserve(root,pair,phase)
            receipts.extend(pool.map(execute_test,[root]*len(pair),[cfg]*len(pair),pair))
    write(root/f'test{update}_COMPLETE.json',dict(update=update,unix=time.time(),jobs=len(receipts),
        phase=phase,outcomes_closed=True,public_receipt_ids=[r['id'] for r in receipts]))


if __name__=='__main__':
    import argparse,signal,traceback
    from .storage import write
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('stage',choices=('watch','smoke','train','test8','test24',
        'resume_prepare','adopt','resume'))
    parser.add_argument('--arm',choices=('A','B','R'))
    parser.add_argument('--attempt',help='resume attempt label r<N> (resume_prepare/adopt/resume)')
    parser.add_argument('--arms',help='comma-separated resumed arms (resume_prepare/adopt)')
    args=parser.parse_args()
    try:
        if args.stage=='watch':watch(args.root)
        elif args.stage=='resume_prepare':print(resume_prepare(args.root,args.attempt,(args.arms or '').split(',')))
        elif args.stage=='adopt':adopt(args.root,args.attempt,(args.arms or '').split(','))
        elif args.stage.startswith('test'):episode_test(args.root,int(args.stage[4:]))
        else:stage_arm(args.root,args.stage,args.arm,args.attempt)
    except BaseException as error:
        name=_label(args.stage,args.arm,args.attempt)
        write(args.root/(name+'_FAILED.json'),dict(unix=time.time(),error=repr(error),traceback=traceback.format_exc(),physics_retry=False))
        raise
