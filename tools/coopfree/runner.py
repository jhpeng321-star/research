"""Approval-gated, restartable, single-owned-worker exact SUMO experiments."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import time
import traceback
import numpy as np
import torch
from . import backend
from .credit import credit, update, next_sigma
from .execution import Controller
from .policy import SetPointPolicy, SIGMA
from .protocol import (budget, canonical, checked, digest, sha, worlds, TIMES, TARGET_WINDOWS, approval_line)
from .selection import select, features
from .storage import read, write, reserve, ledger, completed, admission, status


def verify(root):
    binding = read(root/'BINDING.json')
    if sha(root/'COOPFREE_DESIGN.md') != binding['design_sha256'] or binding['budget'] != budget():
        raise ValueError('design/budget binding')
    for name,expected in binding['sources'].items():
        if sha(root/'pytools'/name) != expected:
            raise ValueError('source mutation: '+name)
    approval = read(root/'APPROVAL.json')
    if approval_line(approval['owner_line']) != approval['owner_line'] or \
            approval['design_sha256'] != binding['design_sha256']:
        raise ValueError('approval binding')
    cpu = read(root/'CPU_APPROVED.json')
    if not cpu['pass_'] or cpu['skips'] or cpu['sim_s'] or cpu['sources'] != binding['sources']:
        raise ValueError('CPU certificate/source binding')
    audits = read(root/'EXPOSURE.json')
    if audits['worlds_sha256'] != digest(worlds()+worlds(True)) or not audits['pass_']:
        raise ValueError('seed exposure audit binding')
    return binding


def ensure_reserved(root, attempts):
    old = {r['id']:r for r in ledger(root)}
    present = [r for r in attempts if r['id'] in old]
    if present:
        if len(present) != len(attempts) or any(old[r['id']] != r for r in attempts):
            raise ValueError('partial or changed reservation blocks resume')
    else:
        reserve(root,attempts)


def fresh(root,binding,ident,world,seconds,phase):
    admission(root)
    verify(root)
    row = dict(id=ident,sim_s=seconds,phase=phase)
    if row not in ledger(root):
        raise ValueError('attempt not precharged')
    if completed(root,ident) is not None:
        raise ValueError('duplicate completed dispatch')
    path = root/'jobs'/ident
    path.mkdir(parents=True)
    write(path/'START.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
                               world=world,charge=seconds,unix=time.time()))
    return backend.create(path,world,seconds,binding),path


def screen(root,binding,world,repeat=False):
    ident = f's{world["seed"]}_'+('repeat' if repeat else 'screen')
    old = completed(root,ident)
    if old is not None:
        return old
    b,path = fresh(root,binding,ident,world,1140,'repeat' if repeat else 'screen')
    states,metrics,motions = {},{},{}
    try:
        def capture(context):
            if int(b.env.now) in TIMES:
                states[str(int(b.env.now))]['context'] = context
        controller = Controller(capture=capture)
        while b.env.now <= 1140:
            t = int(b.env.now)
            if t in TIMES:
                rows = backend.observations(b)
                states[str(t)] = dict(prefix=b.fingerprint(),rows=rows,features=features(rows))
            metrics[str(t)] = dict(b.totals)
            # Full observable frames at control cadence allow executed-motion comparisons.
            if t <= 570:
                motions[str(t)] = backend.motion(backend.observations(b))
            if t == 1140:
                break
            backend.tick(b,controller)
        result = dict(world=world,states=states,metrics=metrics,motions=motions,actual_sim_s=b.actual)
        if repeat:
            original = read(root/'jobs'/f's{world["seed"]}_screen'/'RESULT.json')
            if canonical(original) != canonical(result):
                raise ValueError('bitwise baseline repeat mismatch')
        write(path/'RESULT.json',result)
        return result
    finally:
        b.close()


def screens(root,binding):
    panel = worlds()
    repeats = [next(w for w in panel if w['cell'] == c) for c in range(12)]
    attempts = [dict(id=f's{w["seed"]}_screen',sim_s=1140,phase='screen') for w in panel]
    attempts += [dict(id=f's{w["seed"]}_repeat',sim_s=1140,phase='repeat') for w in repeats]
    ensure_reserved(root,attempts)
    # A repeat canary before the remainder of the screen or ANY intervention.
    screen(root,binding,panel[0])
    screen(root,binding,panel[0],True)
    records = [screen(root,binding,w) for w in panel]
    for w in repeats[1:]:
        screen(root,binding,w,True)
    return records


def zero_canary(root,binding):
    """Measure, do not assume, held-target zero-update rollout closeness to MAPPO."""
    world = next(w for w in worlds() if w['cell'] == 7)
    ident = 'zero_update_canary'
    old = completed(root,ident)
    if old is not None:
        return old
    ensure_reserved(root,[dict(id=ident,sim_s=1140,phase='train')])
    baseline = completed(root,f's{world["seed"]}_screen')
    b,path = fresh(root,binding,ident,world,1140,'train')
    try:
        prefix_gate(b,baseline['states']['300'])
        torch.manual_seed(20261101)
        controller = Controller(policy=SetPointPolicy(),deterministic=True)
        changed,compared,first_error,metrics = 0,0,[],{}
        while b.env.now < 1140:
            now = str(int(b.env.now))
            if now in baseline['motions']:
                changed += backend.motion_changed(baseline['motions'][now],backend.motion(backend.observations(b)))
                compared += 1
            audit = backend.tick(b,controller)
            if now == '300':
                first_error = [abs(r['issued_safe']-r['mappo_safe']) for r in audit]
            if b.env.now in (900,1140):
                metrics[str(int(b.env.now)-300)] = dict(b.totals)
        result = dict(world=world,actual_sim_s=b.actual,max_first_command_error=max(first_error,default=0.),
                      compared_frames=compared,motion_change_frame_share=changed/max(1,compared),
                      metrics=metrics,baseline_metrics={str(t-300):baseline['metrics'][str(t)] for t in (900,1140)},
                      note='Zero-update absolute holds approximate MAPPO; this is not a parity claim over W.')
        write(path/'RESULT.json',result)
        return result
    finally:
        b.close()


def prefix_gate(b,state):
    if canonical(b.fingerprint()) != canonical(state['prefix']) or \
            canonical(backend.observations(b)) != canonical(state['rows']):
        raise ValueError('bitwise full-prefix mismatch BEFORE intervention')


def branch(root,binding,ident,world,t0,state,sample,members,phase='train',prefix_policy=None):
    old = completed(root,ident)
    if old is not None:
        return old
    b,path = fresh(root,binding,ident,world,t0+600,phase)
    try:
        prefix_controller = Controller(policy=prefix_policy,deterministic=True)
        backend.advance(b,t0,controller=prefix_controller)
        prefix_gate(b,state)
        base = dict(b.totals)
        controller = Controller(sample=sample,start=t0,members=members)
        metrics,motions = {},{}
        requested,overrides,issued = 0,0,False
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            while b.env.now <= t0+600:
                t = int(b.env.now)
                if t-t0 in (120,300,600):
                    metrics[str(t-t0)] = {k:v-base[k] for k,v in b.totals.items()}
                if t0 <= t <= t0+sample['window']:
                    motions[str(t)] = backend.motion(backend.observations(b))
                if t == t0+600:
                    break
                audit = backend.tick(b,controller,tape)
                requested += len(audit)
                overrides += sum(r['clamped'] for r in audit)
                issued |= any(abs(r['issued_safe']-r['mappo_safe']) > 1e-6 for r in audit)
        result = dict(world=world,t0=t0,sample=sample,members=members,metrics=metrics,motions=motions,
                      execution=dict(requests=requested,overrides=overrides,issued_changed=bool(issued)),
                      prefix_sha256=digest(state['prefix']),actual_sim_s=b.actual,
                      trace_sha256=sha(path/'observable.jsonl.gz'))
        write(path/'RESULT.json',result)
        return result
    finally:
        b.close()


def save_checkpoint(path,policy,optimizer,update_id,sigma,enlargements):
    with checked(path).open('xb') as f:
        torch.save(dict(format='coopfree/1',policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                        update=update_id,sigma=sigma,enlargements=enlargements),f)
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint(root,update_id):
    p = SetPointPolicy()
    optimizer = torch.optim.Adam(p.parameters(),lr=.001)
    sigma,enlargements = list(SIGMA),0
    if update_id:
        state = torch.load(root/f'policy_u{update_id:03}.pt',map_location='cpu',weights_only=True)
        p.load_state_dict(state['policy'])
        optimizer.load_state_dict(state['optimizer'])
        sigma,enlargements = state['sigma'],state['enlargements']
    return p,optimizer,sigma,enlargements


def train(root,binding,update_id):
    if (root/f'UPDATE_{update_id:03}.json').exists():
        return read(root/f'UPDATE_{update_id:03}.json')
    if (root/f'policy_u{update_id:03}.pt').exists():
        raise ValueError('unreceipted checkpoint: reconcile without repeating optimizer step')
    # Reproducible zero-init hidden layers as well as means; identical fresh-session restart.
    torch.manual_seed(20261101)
    policy,optimizer,sigma,enlargements = load_checkpoint(root,update_id-1)
    baseline = [completed(root,f's{w["seed"]}_screen') for w in worlds()]
    if any(r is None for r in baseline):
        raise ValueError('complete screen panel required')
    planpath = root/f'PLAN_{update_id:03}.json'
    if planpath.exists():
        plan = read(planpath)
    else:
        prior = read(root/'PRIOR.json') if (root/'PRIOR.json').exists() else None
        chosen,selection = select(baseline,np.random.default_rng(20261101+update_id),prior)
        generator = torch.Generator().manual_seed(20261101+update_id)
        plan = dict(selection=selection,windows=[dict(world=r['world'],t0=r['t0'],
                    sample=policy.sample(r['context'],generator,sigma=sigma)) for r in chosen])
        write(planpath,plan)
    attempts = []
    for i,w in enumerate(plan['windows']):
        n = len(w['sample']['context']['members'])
        for arm in ['joint']+[f'solo{j}' for j in range(n)]:
            attempts.append(dict(id=f'u{update_id:03}_w{i:03}_{arm}',sim_s=w['t0']+600,phase='train'))
    ensure_reserved(root,attempts)  # Fail before any physics if 156 whole-zone credits cannot fit.
    records = []
    byseed = {r['world']['seed']:r for r in baseline}
    for i,w in enumerate(plan['windows']):
        world,t0,sample = w['world'],w['t0'],w['sample']
        base = byseed[world['seed']]
        state = base['states'][str(t0)]
        name = f'u{update_id:03}_w{i:03}'
        members = sample['context']['members']
        joint = branch(root,binding,name+'_joint',world,t0,state,sample,members)
        solos = [branch(root,binding,name+f'_solo{j}',world,t0,state,sample,[v]) for j,v in enumerate(members)]
        credits = {}
        for h in (120,300,600):
            bj = base['metrics'][str(t0+h)]['team_delay_vs']-base['metrics'][str(t0)]['team_delay_vs']
            credits[str(h)] = credit(bj,joint['metrics'][str(h)]['team_delay_vs'],
                                     [s['metrics'][str(h)]['team_delay_vs'] for s in solos])
        changed = any(backend.motion_changed(base['motions'][t],value) for t,value in joint['motions'].items())
        records.append(dict(world=world,t0=t0,sample=sample,credit=credits,
                            execution=dict(joint['execution'],motion_changed=changed)))
    report = update(policy,optimizer,records)
    adjusted,enlargements = next_sigma(sigma,report['executed_change_share'],enlargements)
    report.update(update=update_id,sigma_used=sigma,sigma_next=adjusted,enlargements=enlargements,
                  selection=plan['selection'],charged=status(root),records=records,
                  gradient_method='exact correlated joint score function; objective=2D_G from update1')
    checkpoint = root/f'policy_u{update_id:03}.pt'
    save_checkpoint(checkpoint,policy,optimizer,update_id,adjusted,enlargements)
    report['checkpoint_sha256'] = sha(checkpoint)
    write(root/f'UPDATE_{update_id:03}.json',report)
    return report


def adopt_peer(root,update_id):
    """Read only peer public export artifacts; no peer jobs are dispatched or signalled."""
    from .prior import fit
    import shutil
    snapshots = Path('/home/ubuntu/coopbandit20261031_aggregate_v3/snapshots')
    if not snapshots.is_dir():
        return False
    candidates = sorted([p for p in snapshots.iterdir() if re.fullmatch(r'full|partial\d+',p.name)],
                        key=lambda p:(p.name == 'full',int(p.name[7:]) if p.name.startswith('partial') else 0),reverse=True)
    for peer in candidates:
        exam = peer/('exam' if peer.name == 'full' else 'exam-partial')
        if (exam/'MANIFEST.json').exists() and not (root/'PRIOR.json').exists():
            manifest = read(exam/'MANIFEST.json')
            for name in ('states.jsonl','labels.csv'):
                if sha(exam/name) != manifest[name]:
                    raise ValueError('peer public EXAM hash mismatch')
            write(root/'PRIOR.json',fit(exam))
        policy = peer/'learned/policy.pkl'
        if policy.exists() and (peer/'learned/COMPLETE.json').exists() and not (root/'BANDIT.json').exists():
            proof = read(peer/'learned/COMPLETE.json')
            if sha(policy) != proof['policy_sha256']:
                raise ValueError('peer policy hash mismatch')
            dest = root/'bandit_policy.pkl'
            with dest.open('xb') as f:
                f.write(policy.read_bytes());f.flush();os.fsync(f.fileno())
            write(root/'BANDIT.json',dict(file=dest.name,sha256=sha(dest),scope='TRAIN',
                                        source=str(policy),source_complete_sha256=sha(peer/'learned/COMPLETE.json')))
        if (root/'PRIOR.json').exists() and (root/'BANDIT.json').exists():
            return True
    return False


def pipeline(root,binding):
    screens(root,binding)
    canary = zero_canary(root,binding)
    first,last = None,None
    for update_id in range(1,5):
        adopt_peer(root,update_id)
        try:
            report = train(root,binding,update_id)
        except ValueError as error:
            if 'whole-batch budget insufficient' not in str(error):
                raise
            if not (root/'BUDGET_STOP.json').exists():
                write(root/'BUDGET_STOP.json',dict(error=str(error),before_update=update_id,charged=status(root)))
            break
        first = first or update_id
        last = update_id
        # Dev once after u1 if peer policy is already ready; otherwise preserve u1 for later.
        if update_id == 1 and (root/'BANDIT.json').exists():
            from .evaluation import dev
            dev(root,binding,update_id)
    if last is None:
        write(root/'TRAIN_NOT_ADMITTED.json',dict(reason='no whole156-window batch fits',canary=canary))
        return
    # Waiting consumes no SUMO and one sleeping nice19 process. STOP is respected.
    if not (root/'BANDIT.json').exists() and not (root/'WAITING_PEER.json').exists():
        write(root/'WAITING_PEER.json',dict(reason='frozen COOPBANDIT comparator not yet exported',update=last))
    while not (root/'BANDIT.json').exists():
        if (root/'STOP.json').exists():
            return
        adopt_peer(root,last)
        if not (root/'BANDIT.json').exists():
            time.sleep(60)
    from .evaluation import dev,discovery
    for u in sorted({first,last}):
        dev(root,binding,u)
    discovery(root,binding,last)
    write(root/'COMPLETE.json',dict(first_update=first,last_update=last,charged=status(root),
                                    no_superiority_claim_without_dev_and_group_credit=True))


def run(root,phase,update_id=1):
    import fcntl
    root = checked(root)
    binding = verify(root)  # Approval first, BEFORE importing/constructing a SUMO plant.
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        write(root/f'OWNED_{phase}_{os.getpid()}_{time.time_ns()}.json',
              dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],unix=time.time()))
        backend.setup(binding)
        if phase == 'pipeline':
            pipeline(root,binding)
        elif phase == 'screen':
            screens(root,binding)
        elif phase == 'update':
            train(root,binding,update_id)
        else:
            from .evaluation import dev,discovery
            (dev if phase == 'dev' else discovery)(root,binding,update_id)
    except BaseException:
        if not (root/'STOP.json').exists():
            write(root/'STOP.json',dict(error=traceback.format_exc(),charged=status(root),unix=time.time()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--phase',required=True,choices=('screen','update','dev','discovery','pipeline'))
    p.add_argument('--update',type=int,default=1)
    a = p.parse_args()
    if a.update < 1:
        p.error('update must be positive')
    run(a.root,a.phase,a.update)
