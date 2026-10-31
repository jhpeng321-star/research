"""Paired dev evaluation and exact learned-group discovery counterfactuals."""
import gzip
import json
import math
import pickle
import numpy as np
import torch
from . import backend
from .execution import Controller
from .protocol import canonical, sha, worlds, digest
from .storage import read,write,completed,ledger


class BanditController:
    """Frozen peer 'when to cooperate' policy. Its moves never enter TRAIN actions."""
    def __init__(self, payload, cell):
        self.payload,self.cell = payload,cell
        self.until,self.plan,self.fixed = None,None,None
        self.events = []

    def apply(self,rows,commands,now):
        from tools.coopbandit.protocol import observable,option
        from tools.leverprobe.intervention import requests,targets
        if self.until is None or now >= self.until:
            setting = (self.cell+int((now-300)//30))%4
            opts = [option(o,rows,setting) for o in ('O1','O2','O3','O4')]
            x = np.asarray([[observable(rows,now,setting)[k] for k in self.payload['keys']]])
            pred = np.asarray([m.predict(x)[0] for m in self.payload['models']])
            pred = np.where([o['available'] for o in opts],pred,-np.inf)
            index = int(pred.argmax())
            self.plan = opts[index] if pred[index] > self.payload['radius'] else dict(id='O0',members=[])
            self.fixed = targets(self.plan,rows)
            self.until = now+self.plan.get('window',30)
        return requests(self.plan,rows,self.fixed)[0]


def paired_summary(records, learned='learned'):
    byarm = {(r['world']['seed'],r['arm']):r for r in records}
    seeds = sorted({r['world']['seed'] for r in records})
    rng = np.random.default_rng(20261101)
    indices = rng.integers(0,len(seeds),(4000,len(seeds)))
    out = {}
    for comparator in ('mappo','bandit'):
        result = {}
        for h in ('600','900'):
            for metric in ('team_delay_vs','collisions','teleports','ttc_below_3_vs','hard_brake'):
                diffs = np.asarray([byarm[s,learned]['metrics'][h][metric]-
                                    byarm[s,comparator]['metrics'][h][metric] for s in seeds])
                result[h+'_'+metric] = dict(mean=float(diffs.mean()),
                                            ci95=np.quantile(diffs[indices].mean(1),[.025,.975]).tolist())
        out[comparator] = result
    return dict(worlds=len(seeds),learned_minus_comparator=out,
                bootstrap='4000 paired world resamples, seed20261101; delay lower is better')


def dev_episode(root,binding,world,arm,update_id,policy,bandit):
    from .runner import fresh
    ident = f'dev_u{update_id:03}_s{world["seed"]}_{arm}'
    old = completed(root,ident)
    if old is not None:
        return old
    b,path = fresh(root,binding,ident,world,1200,'dev')
    controller = (Controller(policy=policy,deterministic=True) if arm == 'learned' else
                  BanditController(bandit,world['cell']) if arm == 'bandit' else Controller())
    events,metrics = [],{}
    try:
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            while b.env.now <= 1200:
                t = int(b.env.now)
                if t in (900,1200):
                    metrics[str(t-300)] = dict(b.totals)
                if t == 1200:
                    break
                before = len(controller.events)
                # Before executor memory/dwell transition, matching full-rerun prefix_gate.
                state = dict(prefix=b.fingerprint(),rows=backend.observations(b))
                backend.tick(b,controller,tape)
                if len(controller.events) > before:
                    events.append(dict(controller.events[-1],state=state))
        result = dict(world=world,arm=arm,update=update_id,metrics=metrics,events=events,
                      actual_sim_s=b.actual,trace_sha256=sha(path/'observable.jsonl.gz'))
        write(path/'RESULT.json',result)
        return result
    finally:
        b.close()


def dev(root,binding,update_id):
    from .runner import ensure_reserved,load_checkpoint
    # A missing comparator is explicit: never substitute O0 and label it 'bandit'.
    comparator = read(root/'BANDIT.json')
    if sha(root/comparator['file']) != comparator['sha256'] or comparator['scope'] != 'TRAIN':
        raise ValueError('frozen TRAIN-only comparator binding')
    with (root/comparator['file']).open('rb') as f:
        payload = pickle.load(f)
    policy,_,_,_ = load_checkpoint(root,update_id)
    checks = {r['id'].split('_')[1] for r in ledger(root) if r['phase'] == 'dev'}
    if f'u{update_id:03}' not in checks and len(checks) >= 2:
        raise ValueError('only two registered dev checks')
    attempts = [dict(id=f'dev_u{update_id:03}_s{w["seed"]}_{arm}',sim_s=1200,phase='dev')
                for w in worlds(True) for arm in ('mappo','learned','bandit')]
    ensure_reserved(root,attempts)
    records = [dev_episode(root,binding,w,arm,update_id,policy,payload)
               for w in worlds(True) for arm in ('mappo','learned','bandit')]
    result = paired_summary(records)
    result.update(update=update_id,checkpoint_sha256=sha(root/f'policy_u{update_id:03}.pt'),
                  comparator_sha256=comparator['sha256'])
    path = root/f'DEV_{update_id:03}.json'
    if path.exists():
        if canonical(read(path)) != canonical(result):
            raise ValueError('dev receipt mutation')
    else:
        write(path,result)
    return result


def discovery(root,binding,update_id):
    from .runner import branch,ensure_reserved,load_checkpoint
    from .credit import credit
    candidates = []
    for w in worlds(True):
        record = completed(root,f'dev_u{update_id:03}_s{w["seed"]}_learned')
        if record is None:
            raise ValueError('complete learned dev trace required')
        eligible = [e for e in record['events'] if e['time_s'] <= 540 and len(e['sample']['context']['members']) >= 2]
        if eligible:
            candidates.append((w,eligible[0]))  # First temporal event, not the largest observed effect.
    planpath = root/f'DISCOVERY_PLAN_{update_id:03}.json'
    if planpath.exists():
        plan = read(planpath)
    else:
        remaining = 91200-sum(r['sim_s'] for r in ledger(root) if r['phase'] == 'discovery')
        plan,jobs = [],0
        for w,event in candidates:
            n = len(event['sample']['context']['members'])
            cost = (n+2)*(int(event['time_s'])+600)
            if cost <= remaining and jobs+n+2 <= 80:
                plan.append(dict(world=w,event=event))
                remaining -= cost
                jobs += n+2
        write(planpath,plan)
    attempts = []
    for i,r in enumerate(plan):
        for arm in ['baseline','joint']+[f'solo{j}' for j in range(len(r['event']['sample']['context']['members']))]:
            attempts.append(dict(id=f'disc_u{update_id:03}_{i:02}_{arm}',
                                 sim_s=int(r['event']['time_s'])+600,phase='discovery'))
    ensure_reserved(root,attempts)
    policy,_,_,_ = load_checkpoint(root,update_id)
    results = []
    for i,r in enumerate(plan):
        w,event = r['world'],r['event']
        sample,t0 = event['sample'],int(event['time_s'])
        members = sample['context']['members']
        def rerun(arm,chosen):
            return branch(root,binding,f'disc_u{update_id:03}_{i:02}_{arm}',w,t0,event['state'],
                          sample,chosen,'discovery',prefix_policy=policy)
        base,joint = rerun('baseline',[]),rerun('joint',members)
        solos = [rerun(f'solo{j}',[v]) for j,v in enumerate(members)]
        credits = {str(h):credit(base['metrics'][str(h)]['team_delay_vs'],joint['metrics'][str(h)]['team_delay_vs'],
                                [s['metrics'][str(h)]['team_delay_vs'] for s in solos]) for h in (120,300,600)}
        results.append(dict(world=w,t0=t0,sample=sample,credit=credits,joint_id=f'disc_u{update_id:03}_{i:02}_joint',
                            baseline_id=f'disc_u{update_id:03}_{i:02}_baseline'))
    from .analysis import analyse
    result = dict(update=update_id,groups=results,analysis=analyse(root,results),
                  selection='first eligible temporal event/world; whole groups fitting predeclared cap',
                  limitations='budget-selected descriptive dev subset, not population cooperation proof')
    path = root/f'DISCOVERY_{update_id:03}.json'
    if not path.exists():
        write(path,result)
    return result
