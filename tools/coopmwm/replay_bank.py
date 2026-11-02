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

Confirmation stage (MAIN 2026-11-02 22:32 / 2026-11-03 00:01): every first-seed positive
decision runs (a) a second independent CRN seed, (b) the extended window, (c) the safety
check; only entries passing all three are admitted. For every ADMITTED entry whose plan
moves >= 2 CAVs, per-member SOLO branches on the first seed (only that member executes its
plan row, everyone else follows 'policy continues') give the cooperation gain
advantage(group) - max_m advantage(solo m); the entry is tagged cooperative iff the gain
is > 0 (single-CAV plans: cooperative=False). The imitation batch oversamples cooperative
entries up to 50% of the imitation samples (weights unchanged); BANK_QUALITY.json carries
the cooperative count, the mean gain with CI and the group-vs-best-solo synergy twins.
"""
import json
import os
from pathlib import Path
import time

import torch

from . import exact_grade as xg
from . import sumo_verify as sv
from .protocol import digest

BANK = 'replay_bank'
K = 8
EXTENDED_TAIL_S = 600        # MAIN confirmation (b): window to min(episode end, t0+D+600)
TTC_EVENT_S = 1.5            # MAIN confirmation (c): TTC<1.5 s vehicle-second events
SAFETY_KEYS = ('collisions', 'teleports', 'hard_brake')


def bank_root(root, arm):
    return Path(root)/BANK/arm


def latest_checkpoint(root, arm):
    found = sorted(Path(root).glob(f'{arm}_checkpoint_*.pt'))
    if not found:
        raise ValueError('no checkpoint for the collector policy')
    return found[-1], int(found[-1].stem.rsplit('_', 1)[1])


def _boot_ci(values, seed=20261101):
    """95% bootstrap CI of the mean (same estimator family as bank_quality); None under 2 values."""
    import numpy as np
    if len(values) < 2:
        return None
    v = np.array(values, float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]


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

    from .mppi_search import REGISTERED, final_horizon
    # MAIN: the WM arrival model is fitted only up to the observed demand phase end
    # (arrival_fit 'forecast exceeds observed/documented fitted demand phase'):
    # admit a window only if t0 + the longest registered look-ahead stays inside it.
    max_h = max(300, int(min(REGISTERED['final_H_max'], max(REGISTERED['screen_H_min'], REGISTERED['max_duration_s']))),
                final_horizon(REGISTERED['max_duration_s']))
    phase_end = float(cfg.get('demand_phase_end_s', 1200.))

    def call(rows, commands, now, sample):
        if int(now) % 10 or now > cfg.get('last_window_s', 900):
            return None
        groups = group_components(rows)
        if not groups:
            return None
        if now+max_h > phase_end:
            (bank/'admission_exclusions').mkdir(exist_ok=True)
            write(bank/'admission_exclusions'/f'{job["id"]}_t{int(now):04d}.json',
                  dict(job=job['id'], t0=int(now), max_horizon_s=max_h, demand_phase_end_s=phase_end,
                       groups=len(groups), reason='t0 + max look-ahead outside the fitted demand phase'))
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


def verify_pending(root, arm, bounds, runner, charge, workers, decided, limit=None):
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
        if limit is not None and len(todo) >= limit:
            break       # remaining answered snapshots are verified in later rounds

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


def collect(root, arm, binding, worlds, stop, episodes_per_round=2, verify_workers=None, rounds=None,
            verify_per_round=None):
    """Continuous collector for one arm (until ``stop()`` or budget blocked)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from .accel_plant import episode
    from .accel_storage import reserve
    from .accel_learner import load
    from .policy_wm import actor_state
    root = Path(root);bank = bank_root(root, arm);bank.mkdir(parents=True, exist_ok=True)
    cfg = binding['sumo_verify'];vroot = bank/'sumo_verification'
    runner = sv.SubprocessBranchRunner(vroot/'branches', workers=cfg.get('workers', K+1),
                                       timeout_s=cfg.get('timeout_s', 3600))
    charge = sv.accel_charge(str(root), cfg.get('phase', 'sumo_verification_k8'))
    confirm_phase = cfg.get('confirm_phase', 'bank_verification_ext')    # MAIN 2026-11-02 22:32 allowance

    def runner_for(work):
        return sv.SubprocessBranchRunner(Path(work)/'branches', workers=2, timeout_s=cfg.get('timeout_s', 3600))
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
                             collect=dict(slots=binding['slots'], fanout_factor=binding.get('parallel_fanout_factor', 2),
                                          demand_phase_end_s=binding.get('demand_phase_end_s', 1200.))))
        reserve(root, jobs, binding.get('collect_phase', 'train'))
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(episode, jobs))
        statuses += verify_pending(root, arm, binding['accel_bounds'], runner, charge, verify_workers, decided,
                                   verify_per_round)
        confirm_pending(root, arm, runner_for, str(root), confirm_phase, workers=2,
                        sources=binding.get('replay_sources', {}).get(arm, ()), limit=verify_per_round)
        bank_quality(root, arm, binding.get('replay_sources', {}).get(arm, ()))
        n += 1
        if 'budget_blocked' in statuses:
            break
    # Drain: answered requests issued by the final round.
    deadline = time.time()+binding.get('drain_s', 0)
    while time.time() < deadline:
        statuses += verify_pending(root, arm, binding['accel_bounds'], runner, charge, verify_workers, decided,
                                   verify_per_round)
        time.sleep(2)
    return statuses


# ---------------------------------------------------------------- bank reader (trainer side)

def _receipts(root, arm, sources=()):
    """(receipt, branches_dir) from this bank and optional live LS-COLLECT label dirs.

    LS layout (c4/coopnative ls_collection): <src>/labels/<key>.json {'verification': receipt},
    branch tapes under <src>/branches/<verification_id>/<branch>/RESULT.json.
    """
    own = bank_root(root, arm)/'sumo_verification'
    links = {}
    for p in (bank_root(root, arm)/'decisions').glob('*.json'):
        v = json.loads(p.read_text());links[v['verification_id']] = v.get('policy_update')
    for p in sorted((own/'verifications').glob('*.json')):
        r = json.loads(p.read_text())
        u = links.get(r['verification_id'])
        yield r, own/'branches', (f'u{int(u)}' if u is not None else 'unknown')
    for src in sources:
        # MAIN: live frozen-u3 LS-COLLECT labels count as B bank entries, tagged u3.
        src = Path(src)
        for p in sorted((src/'labels').glob('*.json')):
            yield json.loads(p.read_text())['verification'], src/'branches', 'u3'


def bank_decisions(root, arm, upto_unix=None, sources=()):
    """Completed sealed verification receipts of this arm (accepted/rejected), oldest first."""
    out = []
    for r, branches, version in _receipts(root, arm, sources):
        xg.verify_sealed(r, 'receipt_sha256')
        if r['status'] not in ('accepted', 'rejected') or r['episode']['arm'] != arm:
            continue
        if upto_unix is not None and r['timing']['finished_unix'] > upto_unix:
            continue
        out.append(dict(r, _branches=str(branches), _policy_version=version))
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


def bank_pairs(root, arm, training_seed, beta=1., upto_unix=None, clip=xg.WEIGHT_CLIP, sources=(),
               max_decisions=None):
    """(pairs, summary): imitation pairs for ALL exact-advantage>0 bank entries, weight*beta.

    max_decisions (R fairness, MAIN): use at most this many verified decisions, oldest first;
    never padded from another arm's data.
    """
    everything = bank_decisions(root, arm, upto_unix, sources)
    T = temperature(root, arm, training_seed, everything)
    decisions = everything if max_decisions is None else everything[:max(0, int(max_decisions))]
    pairs, entries, counts = [], [], dict(decisions=len(decisions), positive=0, nonpositive=0)
    conf = confirmations(root, arm)
    counts.update(unconfirmed=0, not_admitted=0)
    per_entry = []
    for r in decisions:
        if r['status'] != 'accepted' or not r['grade_veh_s'] or r['grade_veh_s'] <= 0:
            counts['nonpositive'] += 1;continue
        counts['positive'] += 1
        c = conf.get(r['verification_id'])
        if c is None:
            counts['unconfirmed'] += 1;continue       # MAIN: only plans that actually work
        if not c['admitted']:
            counts['not_admitted'] += 1;continue
        plan = r['chosen_plan'];branch = r['branches'][r['chosen']+1]
        result_path = Path(r['_branches'])/r['verification_id']/branch['branch']/'RESULT.json'
        from .accel_storage import read
        tape = read(result_path)
        if branch.get('result_sha256') and digest(tape) != branch['result_sha256']:
            raise ValueError('bank branch tape differs from its sealed verification receipt')
        w = float(beta)*xg.weight(c['mean_replicated_advantage_veh_s'], T['temperature_veh_s'], clip)
        entry = dict(start_s=r['t0'], duration_s=int(plan['duration_s']), plan_id=plan['id'], weight=w)
        coop = c.get('cooperation') or {}
        per_entry.append((xg.imitation_pairs(tape, entry), coop.get('cooperative') is True,
                          coop.get('cooperation_gain_veh_s')))
        entries.append(dict(verification_id=r['verification_id'], grade_veh_s=r['grade_veh_s'], weight=w,
                            mean_replicated_advantage_veh_s=c['mean_replicated_advantage_veh_s'],
                            policy_version=r['_policy_version'], available_k=r.get('available_k', r['k']),
                            duration_s=entry['duration_s'], receipt_sha256=r['receipt_sha256'],
                            cooperative=coop.get('cooperative') is True,
                            group_size=coop.get('group_size'),
                            cooperation_gain_veh_s=coop.get('cooperation_gain_veh_s')))
    # MAIN 2026-11-03 00:01 EXPOSURE: cooperative entries must reach the policy at least as
    # often as non-cooperative ones - oversample them up to 50% of the imitation samples
    # (pair duplication; the weights themselves are unchanged).
    other = [p for ps, co, _ in per_entry if not co for p in ps]
    coop_pairs = [p for ps, co, _ in per_entry if co for p in ps]
    pairs = other+coop_pairs
    exposure = dict(cooperative_entries=sum(co for _, co, _ in per_entry),
                    non_cooperative_entries=sum(not co for _, co, _ in per_entry),
                    cooperative_samples_raw=len(coop_pairs), non_cooperative_samples=len(other),
                    duplicated_pairs=0)
    if coop_pairs and len(coop_pairs) < len(other):
        full, rem = divmod(len(other)-len(coop_pairs), len(coop_pairs))
        pairs = other+coop_pairs*(1+full)+coop_pairs[:rem]
        exposure['duplicated_pairs'] = len(pairs)-len(other)-len(coop_pairs)
    exposure['rule'] = ('cooperative entries oversampled to 50% of imitation samples, weights unchanged'
                        if exposure['duplicated_pairs']
                        else ('cooperative entries already at least half of the imitation samples'
                              if coop_pairs else 'no cooperative admitted entries; no oversampling'))
    if pairs:
        exposure['cooperative_share_of_samples'] = (len(pairs)-len(other))/len(pairs)
    gains = [g for _, _, g in per_entry if g is not None]
    import numpy as np
    exposure.update(mean_cooperation_gain_veh_s=float(np.mean(gains)) if gains else None,
                    mean_cooperation_gain_ci95=_boot_ci(gains))
    counts.update(available=len(everything), used=len(decisions),
                  cooperative=exposure['cooperative_entries'],
                  by_policy_version={v: sum(r['_policy_version'] == v for r in decisions)
                                     for v in sorted({r['_policy_version'] for r in decisions})})
    return pairs, dict(rule='B-REPLAY bank: all CONFIRMED (replication, no later harm, safety) exact-positive entries; weight(mean replicated advantage) x beta; cooperative entries oversampled to 50% of samples',
                       temperature=T, beta=float(beta), counts=counts, entries=entries, clip=clip,
                       max_decisions=max_decisions, exposure=exposure)


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



# ---------------------------------------------------------------- confirmation stage (MAIN)

def replicate(snapshot, salt):
    """Same decision state/prefix/t0 sample; an independent post-t0 noise seed (CRN across its branches)."""
    import hashlib
    g = torch.Generator().manual_seed(int(digest(dict(replication=salt))[:15], 16))
    out = dict(snapshot)
    out['generator_state'] = g.get_state().clone()
    out['rng'] = dict(snapshot['rng'], generator_state_sha256=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest(),
                      replication_salt=salt)
    return out


def ttc_events(trace_path, start, end, threshold=TTC_EVENT_S):
    """Vehicle-seconds with closing TTC < threshold in [start, end) from the branch's observable trace.

    Reuses traffic_shepherd.metrics.closing_ttc on same-lane follower/leader pairs of the
    recorded observable rows (gap = leader rear - follower front)."""
    import gzip
    from traffic_shepherd.metrics import closing_ttc
    n = 0
    with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            if not start <= x['time_s'] < end:
                continue
            lanes = {}
            for v in x.get('vehicles', []):
                lanes.setdefault(v['lane'], []).append(v)
            for rows in lanes.values():
                rows.sort(key=lambda v: v['s_m'])
                for a, b in zip(rows, rows[1:]):
                    ttc = closing_ttc(b['s_m']-a['s_m']-float(b.get('length_m', 5.)), a['v_mps'], b['v_mps'])
                    n += ttc is not None and ttc < threshold
    return float(n)


def _safety(receipt, branches_dir):
    """Per confirmation decision: candidate minus policy deltas; ok iff none is higher."""
    pol, cand = receipt['branches'][0], receipt['branches'][1]
    deltas = {k: float(cand['window'][k])-float(pol['window'][k]) for k in SAFETY_KEYS}
    w = receipt['window']
    traces = {b['branch']: Path(branches_dir)/receipt['verification_id']/b['branch']/'observable.jsonl.gz' for b in (pol, cand)}
    deltas['ttc_below_1p5_events'] = (ttc_events(traces[cand['branch']], w['start_s'], w['end_s'])
                                      - ttc_events(traces[pol['branch']], w['start_s'], w['end_s']))
    return deltas, all(v <= 0 for v in deltas.values())


def _confirm_charge(run_root, phase, kind):
    """Confirmation-branch charge in the registered extension phase (MAIN 2026-11-02 22:32).

    The ledger ids embed the attempt kind: replication and extended branches of one decision
    share the verification_id (episode/t0/candidate digest), so unqualified ids would hit the
    duplicate-charge guard and the second branch could never run."""
    inner = sv.accel_charge(run_root, phase)
    return lambda vid, specs: inner(f'{kind}_{vid}', specs)


def solo_candidate(plan, member):
    """The group plan with ONLY ``member`` executing: it keeps its own profile row, the other
    members follow 'policy continues' (the branch Controller has no guide). A distinct plan id
    gives the solo branch its own verification_id so it can never collide with the group
    decision's receipts, claims or charges."""
    rows = list(plan['members'])
    i = rows.index(member)
    out = dict(plan);out['members'] = [member]
    for key, val in plan.items():
        if isinstance(val, list) and len(val) == len(rows) and all(isinstance(x, list) for x in val):
            out[key] = [val[i]]                      # per-member matrices: keep only this row
    out['id'] = f"{plan['id']}__solo_{member}"
    return out


def _cooperation(root, arm, receipt, snapshot, runner_for, run_root, phase, workers, best, group_advantage):
    """MAIN 2026-11-03 00:01: for an ADMITTED multi-CAV plan, one SOLO verification per member
    on the first seed; cooperative iff advantage(group) > every member-only solo advantage."""
    bank = bank_root(root, arm);solo_root = bank/'confirm_solo'
    charge = _confirm_charge(run_root, phase, 'confirm_solo')
    solos = []
    for m in best['members']:
        r = sv.verify(snapshot, [solo_candidate(best, m)], runner_for(solo_root), charge,
                      receipt['proposer'], workers=workers, root=solo_root)
        solos.append(dict(member=m, verification_id=r['verification_id'], status=r['status'],
                          advantage_veh_s=(r['J_policy_veh_s']-r['J_candidates_veh_s'][0]
                                           if r['status'] in ('accepted', 'rejected') else None),
                          receipt_sha256=r['receipt_sha256']))
    advs = [s['advantage_veh_s'] for s in solos]
    complete = all(a is not None for a in advs)
    if complete:
        b = max(range(len(advs)), key=lambda i: advs[i])
        gain = group_advantage-advs[b]
        return dict(group_size=len(best['members']), solo=solos, solo_advantages_veh_s=advs,
                    best_solo_advantage_veh_s=advs[b], best_solo_member=solos[b]['member'],
                    cooperation_gain_veh_s=gain, cooperative=bool(gain > 0), complete=True,
                    rule='cooperative iff the group plan beats EVERY member-only solo (first seed)')
    return dict(group_size=len(best['members']), solo=solos, solo_advantages_veh_s=advs,
                best_solo_advantage_veh_s=None, best_solo_member=None, cooperation_gain_veh_s=None,
                cooperative=None, complete=False,
                reason='a solo branch failed/stopped; never imputed, cooperation undecided')


def confirm(root, arm, receipt, snapshot, runner_for, run_root, phase, workers=2, version=None):
    """(a) replication + (b) extended-window confirmations, then (c) admission; an ADMITTED
    multi-CAV entry additionally gets the per-member SOLO cooperation check (MAIN 00:01)."""
    bank = bank_root(root, arm);best = receipt['chosen_plan'];vid = receipt['verification_id']
    rep_root, ext_root = bank/'confirm_replication', bank/'confirm_extended'
    rep = sv.verify(replicate(snapshot, vid), [best], runner_for(rep_root),
                    _confirm_charge(run_root, phase, 'confirm_replication'), receipt['proposer'],
                    workers=workers, root=rep_root)
    ext = sv.verify(snapshot, [best], runner_for(ext_root),
                    _confirm_charge(run_root, phase, 'confirm_extended'), receipt['proposer'], workers=workers,
                    root=ext_root, tail_s=EXTENDED_TAIL_S)
    out = dict(verification_id=vid, arm=arm, policy_version=version, t0=receipt['t0'],
               duration_s=int(best['duration_s']), first_seed_advantage_veh_s=receipt['grade_veh_s'],
               replication=dict(verification_id=rep['verification_id'], status=rep['status'],
                                window=rep.get('window'), receipt_sha256=rep['receipt_sha256']),
               extended=dict(verification_id=ext['verification_id'], status=ext['status'],
                             window=ext.get('window'), receipt_sha256=ext['receipt_sha256']),
               unix=time.time())
    ok = rep['status'] in ('accepted', 'rejected') and ext['status'] in ('accepted', 'rejected')
    if ok:
        out['replication_advantage_veh_s'] = rep['J_policy_veh_s']-rep['J_candidates_veh_s'][0]
        out['extended_advantage_veh_s'] = ext['J_policy_veh_s']-ext['J_candidates_veh_s'][0]
        rs, rok = _safety(rep, rep_root/'branches')
        es, eok = _safety(ext, ext_root/'branches')
        out['safety_deltas'] = dict(replication=rs, extended=es)
        out['checks'] = dict(replication=out['replication_advantage_veh_s'] > 0,
                             no_later_harm=out['extended_advantage_veh_s'] > 0, safety=rok and eok)
        out['admitted'] = all(out['checks'].values())
        out['mean_replicated_advantage_veh_s'] = (receipt['grade_veh_s']+out['replication_advantage_veh_s'])/2
    else:
        out.update(admitted=False, checks=dict(confirmation_branches_complete=False),
                   reason='confirmation branch failed/stopped/budget; never imputed')
    if out.get('admitted'):
        if len(best.get('members', [])) >= 2:
            out['cooperation'] = _cooperation(root, arm, receipt, snapshot, runner_for, run_root, phase,
                                              workers, best, out['first_seed_advantage_veh_s'])
        else:   # single-CAV plan: cooperative=False by definition, no solo branches
            out['cooperation'] = dict(group_size=len(best.get('members', [])), solo=[], solo_advantages_veh_s=[],
                                      best_solo_advantage_veh_s=None, best_solo_member=None,
                                      cooperation_gain_veh_s=None, cooperative=False, complete=True,
                                      rule='single-CAV plan: cooperative=False by definition')
    d = bank/'confirmations';d.mkdir(parents=True, exist_ok=True)
    xg._write_new(d/(vid+'.json'), xg._seal(out, 'confirmation_sha256'))
    return out


def _ls_snapshot(src, key):
    """Live LS-COLLECT snapshot completion (same rule as ls_collection.complete_snapshot)."""
    import gzip
    import io
    from .accel_storage import read
    envelope = json.loads(gzip.decompress((Path(src)/'pending'/(key+'.json.gz')).read_bytes()))
    s = torch.load(io.BytesIO(gzip.decompress((Path(src)/'snapshots'/(envelope['snapshot']+'.pt.gz')).read_bytes())),
                   map_location='cpu', weights_only=False)
    ep = read(Path(envelope['episode_path'])/'RESULT.json');t0 = s['t0']
    assert ep['id'] == s['episode']['id'] and ep['world'] == s['episode']['world']
    assert 986000 <= ep['world']['seed'] <= 986071
    s['prefixes'] = {str(t): ep['prefixes'][str(t)] for t in range(300, t0+1)}
    s['samples'] = {str(x['time_s']): x['sample'] for x in ep['steps'] if x['time_s'] < t0}
    s['samples'][str(t0)] = s.pop('current_sample')
    return s


def confirm_pending(root, arm, runner_for, run_root, phase, workers=2, sources=(), limit=None):
    """Confirm every first-seed positive decision (own bank and, for B, live LS labels) once."""
    bank = bank_root(root, arm);done = {p.stem for p in (bank/'confirmations').glob('*.json')}
    by_vid = {}
    for p in (bank/'decisions').glob('*.json'):
        v = json.loads(p.read_text());by_vid[v['verification_id']] = v
    statuses = []
    for r in bank_decisions(root, arm, sources=sources):
        if r['status'] != 'accepted' or r['verification_id'] in done or not (r['grade_veh_s'] or 0) > 0:
            continue
        if limit is not None and len(statuses) >= limit:
            break
        if r['_policy_version'] == 'u3' and r['verification_id'] not in by_vid:
            src = Path(r['_branches']).parent
            key = next(json.loads(p.read_text())['search_id'] for p in (src/'labels').glob('*.json')
                       if json.loads(p.read_text())['verification']['verification_id'] == r['verification_id'])
            snap = _ls_snapshot(src, key)
        else:
            snap = torch.load(bank/'snapshots'/(by_vid[r['verification_id']]['state_id']+'.pt'), weights_only=False)
        receipt = {k: v for k, v in r.items() if not k.startswith('_')}
        statuses.append(confirm(root, arm, receipt, snap, runner_for, run_root, phase, workers,
                                r['_policy_version'])['admitted'])
    return statuses


def confirmations(root, arm):
    out = {}
    for p in (bank_root(root, arm)/'confirmations').glob('*.json'):
        v = json.loads(p.read_text());xg.verify_sealed(v, 'confirmation_sha256');out[v['verification_id']] = v
    return out


def bank_quality(root, arm, sources=()):
    """BANK_QUALITY.json: decisions, first-seed positives, confirmation rate, admitted, mean (95% CI), safety rejections,
    plus the cooperation/synergy record (MAIN 2026-11-03 00:01): cooperative count, mean gain (95% CI)
    and the group-vs-best-solo 'synergy twins' table."""
    import numpy as np
    decisions = bank_decisions(root, arm, sources=sources)
    conf = confirmations(root, arm)
    positives = [r for r in decisions if r['status'] == 'accepted' and (r['grade_veh_s'] or 0) > 0]
    confirmed = [conf[r['verification_id']] for r in positives if r['verification_id'] in conf]
    admitted = [c for c in confirmed if c['admitted']]
    values = np.array([c['mean_replicated_advantage_veh_s'] for c in admitted], float)
    ci = _boot_ci(values)
    coop_rec = [(c, c.get('cooperation') or {}) for c in admitted]
    cooperative = [c for c, co in coop_rec if co.get('cooperative') is True]
    gains = [co['cooperation_gain_veh_s'] for _, co in coop_rec if co.get('cooperation_gain_veh_s') is not None]
    twins = [dict(verification_id=c['verification_id'], t0=c['t0'], policy_version=c['policy_version'],
                  group_size=co.get('group_size'), first_seed_advantage_veh_s=c['first_seed_advantage_veh_s'],
                  best_solo_advantage_veh_s=co.get('best_solo_advantage_veh_s'),
                  best_solo_member=co.get('best_solo_member'),
                  cooperation_gain_veh_s=co.get('cooperation_gain_veh_s'), cooperative=co.get('cooperative'))
             for c, co in coop_rec]
    q = dict(arm=arm, unix=time.time(), decisions=len(decisions), first_seed_positives=len(positives),
             confirmed=len(confirmed), pending_confirmation=len(positives)-len(confirmed), admitted=len(admitted),
             confirmation_rate=(len(admitted)/len(confirmed) if confirmed else None),
             mean_admitted_advantage_veh_s=(float(values.mean()) if len(values) else None),
             mean_admitted_advantage_ci95=ci, ci_method='10000 bootstrap over admitted decisions, RNG20261101',
             rejections=dict(replication=sum(not c['checks'].get('replication', True) for c in confirmed),
                             later_harm=sum(not c['checks'].get('no_later_harm', True) for c in confirmed),
                             safety=sum(not c['checks'].get('safety', True) for c in confirmed),
                             incomplete=sum('confirmation_branches_complete' in c['checks'] for c in confirmed)),
             by_policy_version={v: sum(c['policy_version'] == v for c in admitted)
                                for v in sorted({c['policy_version'] for c in admitted})},
             cooperative=len(cooperative),
             cooperation_undecided=sum(co.get('cooperative') is None for _, co in coop_rec),
             mean_cooperation_gain_veh_s=(float(np.mean(gains)) if gains else None),
             mean_cooperation_gain_ci95=_boot_ci(gains),
             synergy_twins=twins)
    path = bank_root(root, arm)/'BANK_QUALITY.json';tmp = path.with_name(path.name+'.publishing')
    tmp.write_text(json.dumps(q, indent=1));os.replace(tmp, path)
    return q
