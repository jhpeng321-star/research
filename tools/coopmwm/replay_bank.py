"""B-REPLAY (MAIN, 2026-11-02 evening): verified-plan bank + AWR imitation, no plan execution.

Training episodes are plain-policy PPO episodes (no WM query, no plan, no waiting).
A separate continuous COLLECTOR builds a bank per arm:

  current policy snapshot episode (TRAIN worlds) -> at eligible windows a decision snapshot
  (sumo_verify.capture) + non-blocking request (B: WM search top-8; R: current-policy centre
  -> 8 random candidates) -> exact SUMO branches of the 8 + 'policy continues'
  (sumo_verify.verify, same window/CRN/prefix rules) -> a sealed verification receipt.

The bank reader keeps decisions whose best candidate beats 'policy continues'
(exact advantage = J_policy - J_best > 0) and turns the chosen branch's tape into the
existing imitation pairs (exact_grade.imitation_pairs). Weights use the existing temperature
rule (uniform before the first 20 completed bank decisions, then T = max(1, RMS of those
first 20 exact advantages, both signs) frozen and persisted) and the same clip; beta
multiplies the weight. The collector reloads the newest checkpoint of its arm at every
episode, so its snapshot policy refreshes after each update.
"""
import copy
import json
from pathlib import Path
import time

import torch

from . import exact_grade as xg
from . import sumo_verify as sv
from .protocol import digest

BANK = 'replay_bank'
K = 8


def bank_root(root, arm):
    return Path(root)/BANK/arm


def latest_checkpoint(root, arm):
    found = sorted(Path(root).glob(f'{arm}_checkpoint_*.pt'))
    if not found:
        raise ValueError('no checkpoint for the collector policy')
    return found[-1], int(found[-1].stem.rsplit('_', 1)[1])


# ---------------------------------------------------------------- collector guide (inside episodes)

def collector_guide(job, generator, controller, prefixes):
    """Episode hook: snapshot + non-blocking request at eligible windows; ALWAYS policy continues.

    No generator draw is consumed, so the collector episode is the plain policy episode.
    """
    from .warm_history import ObservableHistory
    from .mppi_search import group_components
    from .exploration import regime
    from .storage import write
    history = ObservableHistory()
    root = Path(job['queue_root']);cfg = job['collect'];arm = job['arm']
    bank = bank_root(root, arm);(bank/'snapshots').mkdir(parents=True, exist_ok=True)

    def call(rows, commands, now, sample):
        if int(now) % 10 or now > cfg.get('last_window_s', 900):
            return None
        groups = group_components(rows)
        if not groups:
            return None
        pending = sum(not (root/'responses'/p.name).exists() for p in (root/'requests').glob('*.json'))
        pending += sum(not (root/'centre_responses'/p.name).exists() for p in (root/'centre_requests').glob('*.json'))
        room = max(0, int(cfg.get('fanout_factor', 2)*cfg['slots'])-pending)
        traffic = regime(rows)
        for group in groups[:room]:
            r = dict(world_id=job['world']['seed'], time_s=int(now), rows=rows, group=group['members'],
                     commands=commands, policy_residuals={v: float(a[0]) for v, a in
                                                          zip(sample['context']['members'], sample['own_action'])},
                     history=history.snapshot(now))
            if arm == 'B':
                if job.get('policy_state') is not None:
                    r['policy_state'] = job['policy_state']
                if job.get('search_learning') is not None:
                    r['search_learning'] = job['search_learning']
            r['state_id'] = digest(dict(record=r, job=job['id']))
            snap = sv.capture(controller, prefixes, job, now, sample, generator)
            meta = dict(state_id=r['state_id'], arm=arm, job=job['id'], world=job['world'], t0=int(now),
                        group=group, regime=traffic, policy_update=job['collector_policy_update'],
                        record_sha256=digest(r), unix=time.time())
            torch.save(snap, bank/'snapshots'/(r['state_id']+'.pt'))
            write(bank/'snapshots'/(r['state_id']+'.json'), meta)
            from .warm_spool import immutable
            if arm == 'B':
                # Bounded (fan-out) and admitted as blocking, as in live LS-COLLECT: the broker
                # never capacity-declines a collector search; the episode never waits for it.
                (root/'quota_admissions').mkdir(exist_ok=True)
                immutable(root/'quota_admissions'/(r['state_id']+'.json'),
                          dict(blocking=True, collection=True, unix=time.time()))
                immutable(root/'requests'/(r['state_id']+'.json'), r)
            else:
                write(root/'centre_requests'/(r['state_id']+'.json'), r)
        return None

    call.observe = lambda rows, now: history.push(now, rows)

    def prime(frames):
        for frame in frames:
            history.push(frame['time_s'], frame['rows'], frame['entrance_counts'])
    call.prime = prime
    return call


# ---------------------------------------------------------------- collector loop

def b_available_k(root):
    """available_k of every B bank decision in issue order (R matches the same count)."""
    out = []
    for p in sorted((bank_root(root, 'B')/'decisions').glob('*.json')):
        v = json.loads(p.read_text())
        out.append((v['issued_unix'], v['verification_id'], v['available_k']))
    return [dict(verification_id=vid, available_k=k) for _, vid, k in sorted(out)]


def candidates_for(root, arm, meta, bounds, k=K):
    """(candidates, proposer) or (None, reason) once the request is answered."""
    key = meta['state_id']
    if arm == 'B':
        p = Path(root)/'responses'/(key+'.json')
        if not p.exists():
            return None, 'pending'
        v = json.loads(p.read_text())
        if v.get('record_sha256') != meta['record_sha256']:
            return [], 'response identity mismatch'
        # MAIN: label_list8 top-8, NO positive-saving filter (accepted flag not required).
        if not v.get('topk') or v.get('query_error'):
            return [], 'no WM candidates'
        return list(v['topk'])[:K], 'wm'
    p = Path(root)/'centre_responses'/(key+'.json')
    if not p.exists():
        return None, 'pending'
    v = json.loads(p.read_text())
    if v.get('record_sha256') != meta['record_sha256']:
        return [], 'centre identity mismatch'
    if v.get('unavailable') or v.get('query_error') or v.get('worker_unavailable') or v.get('policy_traj') is None:
        return [], 'centre unavailable'
    seed = int(digest(dict(state=key, arm='R'))[:8], 16)
    return sv.random_candidates(meta['group'], v['policy_traj'], tuple(bounds), meta['regime'], seed, k), 'random'


def verify_pending(root, arm, bounds, runner, charge, workers, decided):
    """Verify every answered snapshot once; returns the statuses of new receipts.

    R decision i uses the available_k of B decision i (issue order); it waits for it.
    """
    from concurrent.futures import ThreadPoolExecutor
    bank = bank_root(root, arm);todo = []
    matched = b_available_k(root) if arm == 'R' else None
    used = sum(1 for _ in (bank/'decisions').glob('*.json')) if arm == 'R' else 0
    for p in sorted((bank/'snapshots').glob('*.json')):
        meta = json.loads(p.read_text())
        if meta['state_id'] in decided:
            continue
        match = None
        if arm == 'R':
            if used+len(todo) >= len(matched):
                break           # no unmatched B decision yet
            match = matched[used+len(todo)]
        cands, how = candidates_for(root, arm, meta, bounds, match['available_k'] if match else K)
        if cands is None:
            continue
        decided.add(meta['state_id'])
        if not cands:
            (bank/'skipped').mkdir(exist_ok=True)
            (bank/'skipped'/(meta['state_id']+'.json')).write_text(json.dumps(dict(meta, reason=how)))
            continue
        todo.append((meta, cands, how, match))

    def one(item):
        meta, cands, how, match = item
        issued = time.time()
        snap = torch.load(bank/'snapshots'/(meta['state_id']+'.pt'), weights_only=False)
        r = sv.verify(snap, cands, runner, charge, how, workers=workers, root=bank/'sumo_verification')
        link = dict(state_id=meta['state_id'], verification_id=r['verification_id'], status=r['status'],
                    policy_update=meta['policy_update'], arm=arm, available_k=len(cands), issued_unix=issued,
                    matched_B=match)
        (bank/'decisions').mkdir(exist_ok=True)
        (bank/'decisions'/(meta['state_id']+'.json')).write_text(json.dumps(link))
        return r['status']
    if not todo:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(len(todo), 2))) as pool:
        return list(pool.map(one, todo))


def collect(root, arm, binding, worlds, stop, episodes_per_round=2, verify_workers=None, rounds=None):
    """Continuous collector for one arm (until ``stop()`` or budget blocked)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from .accel_plant import episode
    from .accel_storage import reserve
    from .acceleration import Policy
    from .accel_learner import load
    from .policy_wm import actor_state
    root = Path(root);bank = bank_root(root, arm);bank.mkdir(parents=True, exist_ok=True)
    cfg = binding['sumo_verify'];vroot = bank/'sumo_verification'
    runner = sv.SubprocessBranchRunner(vroot/'branches', workers=cfg.get('workers', K+1),
                                       timeout_s=cfg.get('timeout_s', 3600))
    charge = sv.accel_charge(str(root), cfg.get('phase', 'sumo_verification_k8'))
    decided = set();n = 0;statuses = []
    while not stop() and (rounds is None or n < rounds):
        path, u = latest_checkpoint(root, arm)
        policy, _, _, _ = load(path)
        jobs = []
        # Episode index continues across collector restarts: a job id is never reused/retried.
        start = sum(1 for _ in (root/'jobs').glob(f'collect_{arm}_*'))
        for i in range(episodes_per_round):
            w = worlds[(start+i) % len(worlds)]
            ident = f'collect_{arm}_u{u:03}_e{start+i:05d}_{w["seed"]}'
            jobs.append(dict(id=ident, path=str(root/'jobs'/ident), world=w, role='train', arm=arm,
                             weights=policy.state_dict(), runtime=binding['runtime'], checkpoint=binding['checkpoint'],
                             noise_seed=int(digest(dict(collect=ident))[:15], 16), deterministic=False, sim_s=1200,
                             queue_root=str(root), accel_bounds=binding['accel_bounds'],
                             gpu_deadline=binding['gpu_deadline'], guided=False, beta=1.,
                             policy_state=actor_state(policy), warm_history=binding.get('warm_history', False),
                             collector_policy_update=u, b_replay=True,
                             collect=dict(slots=binding['slots'], fanout_factor=binding.get('parallel_fanout_factor', 2))))
        reserve(root, jobs, binding.get('collect_phase', 'train'))
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(episode, jobs))
        statuses += verify_pending(root, arm, binding['accel_bounds'], runner, charge, verify_workers, decided)
        n += 1
        if 'budget_blocked' in statuses:
            break
    # Drain: answered requests issued by the final round.
    deadline = time.time()+binding.get('drain_s', 0)
    while time.time() < deadline:
        statuses += verify_pending(root, arm, binding['accel_bounds'], runner, charge, verify_workers, decided)
        time.sleep(2)
    return statuses


# ---------------------------------------------------------------- bank reader (trainer side)

def bank_decisions(root, arm, upto_unix=None):
    """Completed sealed verification receipts of this arm (accepted/rejected)."""
    out = []
    for p in sorted((bank_root(root, arm)/'sumo_verification'/'verifications').glob('*.json')):
        r = json.loads(p.read_text())
        xg.verify_sealed(r, 'receipt_sha256')
        if r['status'] not in ('accepted', 'rejected'):
            continue
        if upto_unix is not None and r['timing']['finished_unix'] > upto_unix:
            continue
        out.append(r)
    return sorted(out, key=lambda r: (r['timing']['finished_unix'], r['verification_id']))


def temperature(root, arm, training_seed, decisions):
    """Existing rule on bank advantages: uniform until 20 completed decisions, then frozen RMS."""
    path = bank_root(root, arm)/f'TEMPERATURE_{arm}_{training_seed}.json'
    if path.exists():
        v = json.loads(path.read_text());xg.verify_sealed(v, 'temperature_sha256');return v
    first = decisions[:xg.N_FREEZE_TEMPERATURE]
    if len(first) < xg.N_FREEZE_TEMPERATURE:
        return dict(rule='uniform_before_freeze', arm=arm, training_seed=training_seed, n=len(first),
                    temperature_veh_s=None)
    import numpy as np
    values = [float(r['best_margin_veh_s']) for r in first]
    t = xg._seal(dict(rule='frozen_rms_first_20_bank_advantages', arm=arm, training_seed=training_seed,
                      n=len(first), temperature_veh_s=max(xg.T_FLOOR_VEH_S, float(np.sqrt(np.mean(np.square(values))))),
                      verification_ids=[r['verification_id'] for r in first], frozen_unix=time.time()),
                 'temperature_sha256')
    try:
        xg._write_new(path, t)
    except FileExistsError:
        return json.loads(path.read_text())
    return t


def bank_pairs(root, arm, training_seed, beta=1., upto_unix=None, clip=xg.WEIGHT_CLIP):
    """(pairs, summary): imitation pairs for ALL exact-advantage>0 bank entries, weight*beta."""
    decisions = bank_decisions(root, arm, upto_unix)
    T = temperature(root, arm, training_seed, decisions)
    pairs, entries, counts = [], [], dict(decisions=len(decisions), positive=0, nonpositive=0)
    for r in decisions:
        if r['status'] != 'accepted' or not r['grade_veh_s'] or r['grade_veh_s'] <= 0:
            counts['nonpositive'] += 1;continue
        counts['positive'] += 1
        plan = r['chosen_plan'];branch = r['branches'][r['chosen']+1]
        result_path = bank_root(root, arm)/'sumo_verification'/'branches'/r['verification_id']/branch['branch']/'RESULT.json'
        from .accel_storage import read
        tape = read(result_path)
        if branch.get('result_sha256') and digest(tape) != branch['result_sha256']:
            raise ValueError('bank branch tape differs from its sealed verification receipt')
        w = float(beta)*xg.weight(r['grade_veh_s'], T['temperature_veh_s'], clip)
        entry = dict(start_s=r['t0'], duration_s=int(plan['duration_s']), plan_id=plan['id'], weight=w)
        pairs.extend(xg.imitation_pairs(tape, entry))
        entries.append(dict(verification_id=r['verification_id'], grade_veh_s=r['grade_veh_s'], weight=w,
                            duration_s=entry['duration_s'], receipt_sha256=r['receipt_sha256']))
    return pairs, dict(rule='B-REPLAY bank: all exact-advantage>0 entries every update; weight x beta',
                       temperature=T, beta=float(beta), counts=counts, entries=entries, clip=clip)


def main(argv=None):
    """Live entry: python -m tools.coopmwm.replay_bank collect ROOT --arm B|R (owned, nohup on cpuQ)."""
    import argparse
    from .incident_recovery import install
    from .warm_protocol import worlds
    p = argparse.ArgumentParser();p.add_argument('cmd', choices=('collect',));p.add_argument('root', type=Path)
    p.add_argument('--arm', choices=('B', 'R'), required=True);p.add_argument('--episodes', type=int, default=4)
    a = p.parse_args(argv)
    wr, _ = install(a.root, a.arm);b = wr.verify(a.root, a.arm)
    from .plan_client import worker_slots
    b = dict(b, slots=len(worker_slots(a.root)))
    from .warm_health import assert_running

    def stop():
        assert_running(a.root, a.arm)
        return (a.root/f'REPLAY_COLLECT_{a.arm}_STOP.json').exists()
    return collect(a.root, a.arm, b, b.get('collect_worlds') or worlds('train'), stop, a.episodes)


if __name__ == '__main__':
    main()
