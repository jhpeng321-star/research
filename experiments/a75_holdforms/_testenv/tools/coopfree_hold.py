"""OWNER01:48 continuation. New sources are separately certified; build5 is immutable.

Reuse existing joint physics, never its MAPPO-credit optimizer or mixed-background solos.
Conditional-window set-point score gradients have an action-independent HOLD baseline.
"""
import concurrent.futures as futures
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from tools.coopfree import backend
from tools.coopfree.credit import credit, next_sigma
from tools.coopfree.policy import SetPointPolicy, gaussian_log_prob
from tools.coopfree.protocol import digest, sha, worlds
from tools.coopfree.storage import read, write, ledger, completed, status
from tools.coopfree import runner

ROOT = Path('/home/ubuntu/coopfree20261101_build5')


def mean_hold(policy, sample):
    """Keep the sampled window exogenous; replace every member's targets by the mean."""
    out = copy.deepcopy(sample)
    with torch.no_grad():
        mean, _ = policy(sample['context'])
    out['action'] = mean.tolist()
    return out


def solo_hold(mean, sample, member):
    """All group members hold; only the designated member takes its explored targets."""
    if mean['context'] != sample['context'] or mean['window'] != sample['window']:
        raise ValueError('solo must share the complete HOLD context and duration')
    out = copy.deepcopy(mean)
    i = sample['context']['members'].index(member)
    out['action'][i] = list(sample['action'][i])
    return out


def exact_subset(windows, available):
    """Outcome-blind round-robin by world; whole all-member-solo groups only."""
    queues = {}
    for i, w in enumerate(windows):
        queues.setdefault(w['world']['seed'], []).append(i)
    selected = []
    while any(queues.values()):
        for seed in sorted(queues):
            if not queues[seed]:
                continue
            i = queues[seed].pop(0)
            w = windows[i]
            cost = len(w['sample']['context']['members'])*(w['t0']+600)
            if cost <= available:
                selected.append(i)
                available -= cost
    return selected


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std()),
                quantiles=np.quantile(a, [0, .25, .5, .75, 1]).tolist()) if len(a) else None


def issued_difference(root, joint_id, hold_id, until):
    def actions(ident):
        out = {}
        with gzip.open(root/'jobs'/ident/'observable.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                frame = json.loads(line)
                if frame['time_s'] >= until:
                    break
                for a in frame['actions']:
                    out[frame['time_s'], a['vid']] = a['issued_safe']
        return out
    joint, hold = actions(joint_id), actions(hold_id)
    return set(joint) != set(hold) or any(abs(joint[k]-hold[k]) > 1e-6 for k in joint)


def optimize(policy, optimizer, records):
    if len(records) < 150 or len({r['world']['seed'] for r in records}) < 12:
        raise ValueError('>=150 windows/>=12 worlds required')
    # w_ind=w_coop=w_group=1 => exact objective2DG, regardless of Di availability.
    # W was sampled independently of continuous noise. Do not score W: a W-dependent
    # baseline would bias categorical-window gradients. Duration stays uniform/exogenous.
    optimizer.zero_grad()
    scores = []
    for r in records:
        s = r['sample']
        mean, _ = policy(s['context'])
        scores.append(-gaussian_log_prob(torch.tensor(s['action']), mean, s['rho'], s['sigma'])
                      * (2*r['D_G']['600'])/10000.)
    loss = torch.stack(scores).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite loss')
    loss.backward()
    grads = [p.grad.flatten() for p in policy.parameters() if p.grad is not None]
    gradient = torch.cat(grads)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError('nonfinite gradient')
    optimizer.step()
    exact = [r for r in records if 'credit' in r]
    issued = sum(r['execution']['issued_changed'] for r in records)
    requests = sum(r['execution']['requests'] for r in records)
    out = dict(loss=float(loss.detach()), gradient_norm=float(gradient.norm()),
               windows=len(records), worlds=len({r['world']['seed'] for r in records}),
               D_G=summary([r['D_G']['600'] for r in records]), exact_credit_windows=len(exact),
               exact_credit_worlds=len({r['world']['seed'] for r in exact}),
               sum_D_i=summary([r['credit']['600']['sum_D_i'] for r in exact]),
               C_G=summary([r['credit']['600']['C_G'] for r in exact]),
               positive_C_G_share=sum(r['credit']['600']['C_G'] > 0 for r in exact)/len(exact) if exact else None,
               executed_change_share=sum(r['execution']['motion_changed'] for r in records)/len(records),
               issued_change_share=issued/len(records),
               executed_vs_issued_share=sum(r['execution']['motion_changed'] and r['execution']['issued_changed']
                                           for r in records)/issued if issued else None,
               safety_override_share=sum(r['execution']['overrides'] for r in records)/requests if requests else 0.)
    return out


def relocate_completed(root):
    """Preserving relocation of closed own tapes only; never touch an active job."""
    import shutil
    if shutil.disk_usage(root).free >= 2500*1024**2:
        return
    dest = Path('/dev/shm/coopfree20261101_observable')
    assert dest.is_dir() and shutil.disk_usage(dest).free >= 4*1024**3
    rows = []
    for job in sorted((root/'jobs').iterdir()):
        p = job/'observable.jsonl.gz'
        if p.is_symlink() or not p.is_file() or not (job/'CLOSE.json').exists() or not (job/'RESULT.json').exists():
            continue
        result = read(job/'RESULT.json')
        expected = sha(p)
        assert expected == result['trace_sha256']
        target = dest/(job.name+'.jsonl.gz')
        assert not target.exists()
        shutil.move(str(p), str(target))
        p.symlink_to(target)
        assert sha(p) == expected
        rows.append(dict(id=job.name, sha256=expected, target=str(target)))
        if shutil.disk_usage(root).free >= 2800*1024**2:
            break
    if rows:
        write(root/f'HOLD_RELOCATION_{time.time_ns()}.json', dict(files=rows, no_content_deleted=True))


WORKER = None


def worker_setup(root):
    global WORKER
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    r = Path(root)
    binding = runner.verify(r)
    backend.setup(binding)
    WORKER = r, binding


def worker_job(task):
    r, binding = WORKER
    w = task['window']
    base = completed(r, f's{w["world"]["seed"]}_screen')
    result = runner.branch(r, binding, task['id'], w['world'], w['t0'], base['states'][str(w['t0'])],
                           task['sample'], task['sample']['context']['members'])
    return dict(id=task['id'], actual_sim_s=result['actual_sim_s'])


def drain(root, tasks, workers=4):
    pending_tasks = [t for t in tasks if completed(root, t['id']) is None]
    with futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_setup, initargs=(str(root),)) as pool:
        iterator = iter(pending_tasks)
        pending = {pool.submit(worker_job, t): t for t in [next(iterator, None) for _ in range(workers)] if t}
        while pending:
            done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for f in done:
                task = pending.pop(f)
                receipt = f.result()
                assert receipt['id'] == task['id']
                print(json.dumps(dict(phase='hold_physics', **receipt)), flush=True)
                relocate_completed(root)
                following = next(iterator, None)
                if following:
                    pending[pool.submit(worker_job, following)] = following


def train(root, binding):
    if (root/'HOLD_UPDATE_001.json').exists():
        return read(root/'HOLD_UPDATE_001.json')
    assert not (root/'policy_hold_u001.pt').exists(), 'unreceipted optimizer; no repeat'
    torch.manual_seed(20261101)
    policy, optimizer, sigma, enlargements = runner.load_checkpoint(root, 0)
    original = read(root/'PLAN_001.json')
    planpath = root/'HOLD_PLAN_001.json'
    if planpath.exists():
        plan = read(planpath)
    else:
        windows = original['windows']
        holds, lookup, tasks = {}, [], []
        for i, w in enumerate(windows):
            mean = mean_hold(policy, w['sample'])
            key = digest(dict(world=w['world'], t0=w['t0'], sample=mean))
            if key not in holds:
                ident = f'hold_u001_base{i:03}'
                holds[key] = ident
                tasks.append(dict(id=ident, window=w, sample=mean))
            lookup.append(holds[key])
        holdcost = sum(t['window']['t0']+600 for t in tasks)
        remaining = 1650000-sum(r['sim_s'] for r in ledger(root) if r['phase']=='train')-holdcost
        if remaining < 0:
            raise ValueError('whole HOLD-baseline batch unaffordable')
        selected = exact_subset(windows, remaining)
        for i in selected:
            w = windows[i]
            mean = mean_hold(policy, w['sample'])
            for j, member in enumerate(w['sample']['context']['members']):
                tasks.append(dict(id=f'hold_u001_w{i:03}_solo{j}', window=w,
                                  sample=solo_hold(mean, w['sample'], member)))
        plan = dict(original_plan_sha256=sha(root/'PLAN_001.json'), baseline_ids=lookup,
                    exact_indices=selected, tasks=tasks, hold_baseline_cost=holdcost,
                    selection='outcome-blind world-round-robin; complete groups fitting remaining TRAIN cap',
                    limitation='original MAPPO-background solos are retained but excluded from exact HOLD C_G')
        write(planpath, plan)
    attempts = [dict(id=t['id'], sim_s=t['window']['t0']+600, phase='train') for t in plan['tasks']]
    runner.ensure_reserved(root, attempts)
    drain(root, plan['tasks'])
    records = []
    for i, w in enumerate(original['windows']):
        joint = completed(root, f'u001_w{i:03}_joint')
        hold = completed(root, plan['baseline_ids'][i])
        assert joint['sample'] == w['sample'] and joint['prefix_sha256'] == hold['prefix_sha256']
        dg = {str(h):hold['metrics'][str(h)]['team_delay_vs']-joint['metrics'][str(h)]['team_delay_vs'] for h in (120,300,600)}
        changed = any(backend.motion_changed(hold['motions'][t], value) for t, value in joint['motions'].items())
        # Issued/override counters are original actual safety audits. Motion change is
        # remeasured against HOLD, not the structurally different MAPPO trajectory.
        issued = issued_difference(root, f'u001_w{i:03}_joint', plan['baseline_ids'][i], w['t0']+w['sample']['window'])
        record = dict(world=w['world'], t0=w['t0'], sample=w['sample'], D_G=dg,
                      baseline_id=plan['baseline_ids'][i],
                      execution=dict(joint['execution'], motion_changed=changed, issued_changed=issued,
                                     legacy_mappo_issued_changed=joint['execution']['issued_changed']))
        base = completed(root, f's{w["world"]["seed"]}_screen')
        record['hold_minus_mappo'] = {str(h):hold['metrics'][str(h)]['team_delay_vs']-
            (base['metrics'][str(w['t0']+h)]['team_delay_vs']-base['metrics'][str(w['t0'])]['team_delay_vs']) for h in (120,300,600)}
        if i in plan['exact_indices']:
            solos = [completed(root, f'hold_u001_w{i:03}_solo{j}') for j in range(len(w['sample']['context']['members']))]
            record['credit'] = {str(h):credit(hold['metrics'][str(h)]['team_delay_vs'],
                joint['metrics'][str(h)]['team_delay_vs'], [s['metrics'][str(h)]['team_delay_vs'] for s in solos]) for h in (120,300,600)}
        records.append(record)
    report = optimize(policy, optimizer, records)
    adjusted, enlargements = next_sigma(sigma, report['executed_change_share'], enlargements)
    report.update(update=1, baseline='zero-update policy-mean HOLD; sampled W conditioned/exogenous',
                  gradient_method='continuous Gaussian score only; w_ind=w_coop=w_group=1 =>2DG; no categorical W gradient',
                  sigma_used=sigma, sigma_next=adjusted, enlargements=enlargements, records=records,
                  hold_minus_mappo={str(h):summary([r['hold_minus_mappo'][str(h)] for r in records]) for h in (120,300,600)},
                  limitation=plan['limitation'], charged=status(root))
    checkpoint = root/'policy_hold_u001.pt'
    runner.save_checkpoint(checkpoint, policy, optimizer, 1, adjusted, enlargements)
    report['checkpoint_sha256'] = sha(checkpoint)
    write(root/'HOLD_UPDATE_001.json', report)
    print(json.dumps(dict(phase='hold_optimizer_complete', checkpoint_sha256=sha(checkpoint),
                          windows=report['windows'], exact_credit_windows=report['exact_credit_windows'])), flush=True)
    return report


def dev(root, binding, update_id, policy, filename, baseline):
    from tools.coopfree.evaluation import dev_episode, paired_summary
    import pickle
    path = root/filename
    if path.exists():
        return read(path)
    comparator = read(root/'BANDIT.json')
    assert sha(root/comparator['file']) == comparator['sha256'] and comparator['scope'] == 'TRAIN'
    with (root/comparator['file']).open('rb') as f:
        payload = pickle.load(f)
    attempts = [dict(id=f'dev_u{update_id:03}_s{w["seed"]}_{arm}', sim_s=1200, phase='dev')
                for w in worlds(True) for arm in ('mappo','learned','bandit')]
    runner.ensure_reserved(root, attempts)
    records = []
    for w in worlds(True):
        for arm in ('mappo','learned','bandit'):
            records.append(dev_episode(root, binding, w, arm, update_id, policy, payload))
            relocate_completed(root)
    report = paired_summary(records)
    report.update(update=update_id, policy=baseline, comparator_sha256=comparator['sha256'])
    if update_id == 101:
        report['checkpoint_sha256'] = sha(root/'policy_hold_u001.pt')
    write(path, report)
    return report


def discovery(root, binding, policy):
    from tools.coopfree.analysis import analyse
    path = root/'HOLD_DISCOVERY_001.json'
    if path.exists():
        return read(path)
    planpath = root/'HOLD_DISCOVERY_PLAN_001.json'
    if planpath.exists():
        plan = read(planpath)
    else:
        plan = []
        available = 91200-sum(r['sim_s'] for r in ledger(root) if r['phase']=='discovery')
        torch.manual_seed(20261101)
        zero = SetPointPolicy()
        for w in worlds(True):
            record = completed(root, f'dev_u101_s{w["seed"]}_learned')
            eligible = [e for e in record['events'] if e['time_s'] <= 540 and len(e['sample']['context']['members']) >= 2]
            if not eligible:
                continue
            event = eligible[0]
            cost = (len(event['sample']['context']['members'])+2)*(int(event['time_s'])+600)
            if cost <= available:
                plan.append(dict(world=w, event=event, mean=mean_hold(zero, event['sample'])))
                available -= cost
        write(planpath, plan)
    tasks = []
    for i, r in enumerate(plan):
        event = r['event']
        sample = event['sample']
        arms = [('baseline', r['mean']), ('joint', sample)]
        arms += [(f'solo{j}', solo_hold(r['mean'], sample, member)) for j, member in enumerate(sample['context']['members'])]
        for arm, action in arms:
            tasks.append(dict(id=f'hold_disc_{i:02}_{arm}', sim_s=int(event['time_s'])+600, phase='discovery'))
    runner.ensure_reserved(root, tasks)
    groups = []
    for i, r in enumerate(plan):
        event, w = r['event'], r['world']
        sample, t0 = event['sample'], int(event['time_s'])
        def rerun(arm, action):
            value = runner.branch(root, binding, f'hold_disc_{i:02}_{arm}', w, t0, event['state'],
                                  action, sample['context']['members'], 'discovery', prefix_policy=policy)
            relocate_completed(root)
            return value
        base = rerun('baseline', r['mean'])
        joint = rerun('joint', sample)
        solos = [rerun(f'solo{j}', solo_hold(r['mean'], sample, member)) for j, member in enumerate(sample['context']['members'])]
        credits = {str(h):credit(base['metrics'][str(h)]['team_delay_vs'], joint['metrics'][str(h)]['team_delay_vs'],
                                [s['metrics'][str(h)]['team_delay_vs'] for s in solos]) for h in (120,300,600)}
        groups.append(dict(world=w, t0=t0, sample=sample, credit=credits,
                           joint_id=f'hold_disc_{i:02}_joint', baseline_id=f'hold_disc_{i:02}_baseline'))
    result = dict(update=101, groups=groups, analysis=analyse(root, groups),
                  limitations='first temporal eligible event/world; complete affordable groups; HOLD reference, learned shared prefix')
    write(path, result)
    return result


def run(root=ROOT):
    import fcntl
    root = Path(root)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    cert = read(root/'HOLD_CPU_APPROVED.json')
    assert cert['pass_'] and cert['sim_s'] == 0 and not cert['skips']
    assert sha(root/'HOLD_DESIGN.md') == cert['design_sha256']
    for name, expected in cert['sources'].items():
        assert sha(root/'pytools/tools'/name) == expected
    binding = runner.verify(root)
    with (root/'hold_supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not (root/'HOLD_GATE_PARKED.json').exists():
                if (root/'STOP.json').exists():
                    raise RuntimeError('durable STOP')
                time.sleep(2)
            from tools.coopfree_batch import identity
            parked = read(root/'HOLD_GATE_PARKED.json')
            assert parked['unfinished'] == []
            assert identity(Path('/proc/339173/stat').read_text()) == 'T'
            assert not any((p/'START.json').exists() and not (p/'CLOSE.json').exists() for p in (root/'jobs').iterdir() if p.is_dir())
            # Only this exact old supervisor, safely between plants. SIGCONT delivers
            # pending SIGTERM to a stopped process; no child/peer process is signalled.
            os.kill(339173, signal.SIGTERM)
            os.kill(339173, signal.SIGCONT)
            for _ in range(100):
                p = Path('/proc/339173/stat')
                if not p.exists() or p.read_text().split()[2] == 'Z':
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('legacy supervisor did not retire')
            write(root/'HOLD_RETIRED_LEGACY.json', dict(pid=339173, ticks='47835664', no_open_plants=True, unix=time.time()))
            # Preserve rather than remove the gate marker; normal admission is restored.
            (root/'YIELD').rename(root/'HOLD_YIELD_RETIRED.json')
            with (root/'supervisor.lock').open('a') as scientific_lock:
                fcntl.flock(scientific_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                backend.setup(binding)
                report = train(root, binding)
                policy, _, _, _ = runner.load_checkpoint(root, 0)
                state = torch.load(root/'policy_hold_u001.pt', map_location='cpu', weights_only=True)
                policy.load_state_dict(state['policy'])
                dev(root, binding, 101, policy, 'HOLD_DEV_001.json', 'HOLD-credit learned u1')
                torch.manual_seed(20261101)
                dev(root, binding, 1, SetPointPolicy(), 'HOLD_ZERO_DEV.json', 'zero-update policy-mean HOLD')
                discovery(root, binding, policy)
                write(root/'HOLD_COMPLETE.json', dict(charged=status(root), update=1, exact_credit_windows=report['exact_credit_windows']))
        except BaseException:
            if not (root/'HOLD_STOP.json').exists():
                write(root/'HOLD_STOP.json', dict(error=traceback.format_exc(), charged=status(root), unix=time.time()))
            raise


if __name__ == '__main__':
    run()
