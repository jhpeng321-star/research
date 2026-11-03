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

Confirmation stage (MAIN 2026-11-02 22:32 / 2026-11-03 00:01 / 04:26): every first-seed positive
decision runs (a) a second independent CRN seed, (b) the extended window, (c) the safety check -
collisions/teleports must not increase (hard constraints) and the TOTAL of hard-brake + TTC<1.5 s
events must not increase (the old no-single-metric-may-increase verdict is recorded per decision
and counted in BANK_QUALITY for disclosure); only entries passing all three are admitted. The
per-member SOLO cooperation check runs on every decision passing replication + later-harm, as a
diagnostic even when the safety check rejects it (its twin row is published). For every ADMITTED
entry whose plan moves >= 2 CAVs, solo branches on the first seed give the cooperation gain
advantage(group) - max_m advantage(solo m); the entry is tagged cooperative iff the gain
is > 0 (single-CAV plans: cooperative=False). The imitation batch oversamples cooperative
entries up to 50% of the imitation samples (weights unchanged); BANK_QUALITY.json carries
the cooperative count, the mean gain with CI and the group-vs-best-solo synergy twins.

LESSON-LEARNED CHECK (MAIN 2026-11-03 01:22), after EVERY B and R update in B-REPLAY: on a
FIXED list of up to 12 admitted bank snapshots (the first admitted, sealed/frozen once chosen)
run ONE exact SUMO branch of the NEW policy continuing from each snapshot (same window, same
CRN prefix/generator state). recapture = (J_policy_u3 - J_new_policy) / (J_policy_u3 - J_best_plan)
per entry plus the mean with CI - how much of each verified win the policy now achieves on its
own - and the imitation action error (m/s^2) of the new policy vs the plan actions on the chosen
branch tape. LESSON_CHECK_u<k>_<arm>.json; charges lesson_u<k>-prefixed in bank_verification_ext.
"""
import copy
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
SAFETY_RULE = ('MAIN 2026-11-03 04:26: collisions/teleports must not increase (hard constraints); '
               'the TOTAL of hard-brake + TTC<1.5 s events must not increase; the old '
               'no-single-metric-may-increase verdict is kept for disclosure')
LESSON_N = 12                # MAIN 2026-11-03 01:22: fixed lesson list cap


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
        # A resumed bank shares the historical run root. Closed engineering-wave
        # requests are retained forever, but cannot consume current collection slots.
        scope = None
        if cfg.get('owned_pending_only'):
            # Independent, equal bounded admission: a full B search queue must
            # not erase every R snapshot before R's PPO episodes close.
            scope = {p.stem for p in (bank/'snapshots').glob('*.json')}
        pending = sum((scope is None or p.stem in scope) and not (root/'responses'/p.name).exists()
                      for p in (root/'requests').glob('*.json'))
        pending += sum((scope is None or p.stem in scope) and not (root/'centre_responses'/p.name).exists()
                       for p in (root/'centre_requests').glob('*.json'))
        room = max(0, int(cfg.get('fanout_factor', 2)*cfg['slots'])-pending)
        traffic = regime(rows)
        for group in groups[:room]:
            r = dict(world_id=job['world']['seed'], time_s=int(now), rows=rows, group=group['members'],
                     commands=commands, policy_residuals={v: float(a[0]) for v, a in
                                                          zip(sample['context']['members'], sample['own_action'])},
                      history=history.snapshot(now))
            if arm == 'R':
                r['mode'] = 'centre'  # centre-only rollout, never score R candidates
            elif cfg.get('search_config'):
                r['search_config'] = dict(cfg['search_config'])
                if cfg.get('niche_alternation'):
                    import fcntl
                    from .niche_collect import search_config
                    admissions = bank/'collection_admissions';admissions.mkdir(exist_ok=True)
                    with (bank/'COLLECTION_ADMISSION.claim').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        ordinal = len(list(admissions.glob('*.json')))
                        r['search_config'] = search_config(ordinal)
                        write(admissions/f'{ordinal:09}.json', dict(ordinal=ordinal,
                            job=job['id'], time_s=now, group=group['members'],
                            search_config=r['search_config']))
            if job.get('policy_state') is not None:
                # Both arms need their own CURRENT policy for the centre rollout;
                # R still never asks the WM to score its random candidates.
                r['policy_state'] = job['policy_state']
            if arm == 'B':
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
    binding=Path(root)/'B_REPLAY_START_0835_V4.json'
    if binding.exists():
        source=json.loads(binding.read_text())['collection_root']
        seen={vid for _,vid,_ in out}
        for r in bank_decisions(root,'B',sources=[source]):
            if r['verification_id'] not in seen:
                out.append((r['timing']['finished_unix'],r['verification_id'],r['k']))
                seen.add(r['verification_id'])
    return [dict(verification_id=vid, available_k=k) for _, vid, k in sorted(out)]


def centre_identity(record, expected_sha256, response):
    """Validate both signed representations of the centre-only dispatch envelope."""
    if record.get('mode')!='centre' or digest(record)!=expected_sha256:return False
    if response.get('candidate_scores',0)!=0:return False
    prepared={k:v for k,v in record.items() if k!='mode'}
    return response.get('record_sha256') in (expected_sha256,digest(prepared))


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
        request=Path(root)/'centre_requests'/(key+'.json')
        if not request.exists() or not centre_identity(json.loads(request.read_text()),meta['record_sha256'],v):
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
        folder = 'requests' if arm == 'B' else 'centre_requests'
        request = Path(root)/folder/(meta['state_id']+'.json')
        if request.exists():
            q = json.loads(request.read_text())
            gate = sv.actionable_gate(cands, q['rows'], q['commands'],
                snap['samples'][str(snap['t0'])], dict(accel_bounds=bounds))
            (bank/'actionable_gates').mkdir(exist_ok=True)
            from .storage import write
            write(bank/'actionable_gates'/(meta['state_id']+'.json'),
                dict(inactionable=gate['inactionable'], members_t0=gate['members_t0'],
                     raw_k=len(cands), available_k=len(gate['actionable'])))
            cands = gate['actionable']
            if not cands:
                (bank/'skipped').mkdir(exist_ok=True)
                write(bank/'skipped'/(meta['state_id']+'.json'), dict(meta, reason='all candidates inactionable'))
                return 'inactionable'
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
                        sources=binding.get('replay_sources', {}).get(arm, ()), limit=verify_per_round,
                        **({'verify_mode': cfg['verify_mode']} if cfg.get('verify_mode') else {}))
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
            label = json.loads(p.read_text())
            key = label['search_id']
            suffix = p.stem[len(key):]
            if not p.stem.startswith(key):
                raise ValueError('LS label filename differs from search identity')
            if suffix:
                import re
                if not re.fullmatch(r'_[A-Z0-9]+', suffix):
                    raise ValueError('invalid LS label generation')
                branches = src/'label_attempts'/suffix[1:]/'branches'
            else:
                branches = src/'branches'
            yield label['verification'], branches, 'u3'


def bank_decisions(root, arm, upto_unix=None, sources=()):
    """Completed sealed verification receipts of this arm (accepted/rejected), oldest first."""
    out = []
    for r, branches, version in _receipts(root, arm, sources):
        xg.verify_sealed(r, 'receipt_sha256')
        if r['status'] not in ('accepted', 'rejected') or r['episode']['arm'] != arm:
            continue
        if upto_unix is not None and r['timing']['finished_unix'] > upto_unix:
            continue
        extra = {}
        if version == 'u3':
            extra['_source'] = str(branches.parent.parent.parent
                if branches.parent.parent.name == 'label_attempts' else branches.parent)
        out.append(dict(r, _branches=str(branches), _policy_version=version, **extra))
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
               max_decisions=None, v2v=False, v2v_cfg=None, v2v_intent=None):
    """(pairs, summary): imitation pairs for ALL exact-advantage>0 bank entries, weight*beta.

    max_decisions (R fairness, MAIN): use at most this many admitted targets, oldest first;
    never padded from another arm's data.
    v2v (MAIN 2026-11-03, part 3/3, DEFAULT OFF = byte-identical): each pair also carries the
    partners' v2v features of that second, computed from the winning plan's recorded branch
    trace (bank_v2v.attach_pairs); pairs whose trace lacks partner commands are excluded and
    counted in summary['v2v'].
    v2v_intent (DEFAULT None = unchanged; requires v2v): binding dict with v2v_intent ON - pairs
    also carry partner-intent input slots (EMPTY here; exact_update refreshes them from the current
    intent head on the partners' own t-1 bank contexts - never recorded commands) and the plan
    members' own future plan commands as intent targets (tools/coopmwm/v2v_intent.py).
    """
    everything = bank_decisions(root, arm, upto_unix, sources)
    T = temperature(root, arm, training_seed, everything)
    decisions = everything
    pairs, entries, counts = [], [], dict(decisions=len(decisions), positive=0, nonpositive=0)
    conf = confirmations(root, arm)
    counts.update(unconfirmed=0, not_admitted=0,admitted_excess_not_used=0)
    per_entry = []
    from . import bank_v2v
    v2v_counts = dict(pairs=0, kept=0, excluded_missing_prev_second=0,
                      excluded_missing_partner_command=0, trace_missing=False)
    for r in decisions:
        if r['status'] != 'accepted' or not r['grade_veh_s'] or r['grade_veh_s'] <= 0:
            counts['nonpositive'] += 1;continue
        counts['positive'] += 1
        c = conf.get(r['verification_id'])
        if c is None:
            counts['unconfirmed'] += 1;continue       # MAIN: only plans that actually work
        if not c['admitted']:
            counts['not_admitted'] += 1;continue
        if max_decisions is not None and len(entries)>=max(0,int(max_decisions)):
            counts['admitted_excess_not_used']+=1;continue
        plan = r['chosen_plan'];branch = r['branches'][r['chosen']+1]
        result_path = Path(r['_branches'])/r['verification_id']/branch['branch']/'RESULT.json'
        from .accel_storage import read
        tape = read(result_path)
        if branch.get('result_sha256') and digest(tape) != branch['result_sha256']:
            raise ValueError('bank branch tape differs from its sealed verification receipt')
        w = float(beta)*xg.weight(c['mean_replicated_advantage_veh_s'], T['temperature_veh_s'], clip)
        entry = dict(start_s=r['t0'], duration_s=int(plan['duration_s']), plan_id=plan['id'], weight=w)
        coop = c.get('cooperation') or {}
        entry_pairs = xg.imitation_pairs(tape, entry)
        intent = None
        if v2v and v2v_intent:
            from . import v2v_intent as vi
            s = vi.settings(v2v_intent)
            entry_pairs = vi.attach_bank_targets(entry_pairs, tape, entry, s['horizon'], s['bounds'])
            intent = dict(horizon=s['horizon'], entry=f"{r['verification_id']}/{branch['branch']}")
        if v2v:
            entry_pairs, entry_counts = bank_v2v.attach_pairs(
                entry_pairs, Path(r['_branches'])/r['verification_id']/branch['branch']/'observable.jsonl.gz',
                r['t0'], v2v_cfg, **({'intent': intent} if intent else {}))
            v2v_counts = bank_v2v.merge_counts(v2v_counts, entry_counts)
        per_entry.append((entry_pairs, coop.get('cooperative') is True,
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
                   available_admitted=len(entries)+counts['admitted_excess_not_used'],used_admitted=len(entries),
                  cooperative=exposure['cooperative_entries'],
                  by_policy_version={v: sum(r['_policy_version'] == v for r in decisions)
                                     for v in sorted({r['_policy_version'] for r in decisions})})
    out = dict(rule='B-REPLAY bank: all CONFIRMED (replication, no later harm, safety) exact-positive entries; weight(mean replicated advantage) x beta; cooperative entries oversampled to 50% of samples',
                       temperature=T, beta=float(beta), counts=counts, entries=entries, clip=clip,
                        max_decisions=max_decisions, exposure=exposure)
    if v2v:
        out['v2v'] = dict(rule='each pair carries the partners v2v features of its second, from the winning '
                              'plan branch trace (positions, speeds, accelerations; last issued command at the '
                              'previous second); pairs whose trace lacks partner commands are excluded and counted',
                          stub=bank_v2v.STUB, cfg=dict(bank_v2v.DEFAULT_CFG, **(v2v_cfg or {})),
                          counts=v2v_counts)
    return pairs, out


def imitation_decision_cap(update_receipt):
    """R matches plans B actually imitated, not rejected/non-admitted verification lists."""
    weights=update_receipt.get('exact_imitation') or {}
    if isinstance(weights.get('entries'),list):return len(weights['entries'])
    counts=weights.get('counts') or (update_receipt.get('trust',{}).get('bank') or {})
    return int(counts.get('used_admitted',0))


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
    """Per confirmation decision: candidate minus policy deltas, OLD and NEW verdicts.

    OLD rule (disclosed, never deleted): no metric may increase.
    NEW rule (MAIN 2026-11-03 04:26, applied identically to B and R): (i) collisions and
    teleports must NOT increase (hard constraints); (ii) the TOTAL of safety-critical events
    (hard-brake events + TTC<1.5 s events, same windows, same metric code) must NOT increase.
    One metric improving may offset another worsening; a collision/teleport increase never can."""
    pol, cand = receipt['branches'][0], receipt['branches'][1]
    deltas = {k: float(cand['window'][k])-float(pol['window'][k]) for k in SAFETY_KEYS}
    w = receipt['window']
    traces = {b['branch']: Path(branches_dir)/receipt['verification_id']/b['branch']/'observable.jsonl.gz' for b in (pol, cand)}
    deltas['ttc_below_1p5_events'] = (ttc_events(traces[cand['branch']], w['start_s'], w['end_s'])
                                      - ttc_events(traces[pol['branch']], w['start_s'], w['end_s']))
    old_ok = all(v <= 0 for v in deltas.values())
    new_ok = (deltas['collisions'] <= 0 and deltas['teleports'] <= 0
              and deltas['hard_brake']+deltas['ttc_below_1p5_events'] <= 0)
    return deltas, old_ok, bool(new_ok)


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
        rs, rok_old, rok = _safety(rep, rep_root/'branches')
        es, eok_old, eok = _safety(ext, ext_root/'branches')
        out['safety_deltas'] = dict(replication=rs, extended=es)
        out['safety_rule'] = SAFETY_RULE
        out['safety_old_rule_ok'] = bool(rok_old and eok_old)   # disclosed alongside the new verdict
        out['checks'] = dict(replication=out['replication_advantage_veh_s'] > 0,
                             no_later_harm=out['extended_advantage_veh_s'] > 0, safety=rok and eok)
        out['admitted'] = all(out['checks'].values())
        out['mean_replicated_advantage_veh_s'] = (receipt['grade_veh_s']+out['replication_advantage_veh_s'])/2
    else:
        out.update(admitted=False, checks=dict(confirmation_branches_complete=False),
                   reason='confirmation branch failed/stopped/budget; never imputed')
    if ok and out['checks'].get('replication') and out['checks'].get('no_later_harm'):
        # MAIN 2026-11-03 04:26: the synergy check also runs as a DIAGNOSTIC on every decision
        # that passes replication + later-harm, even when the safety check rejects the entry.
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


def _decision_snapshot(root, arm, r, by_vid):
    """The decision-time snapshot of a first-seed positive decision (own bank or live LS label)."""
    bank = bank_root(root, arm)
    link = by_vid.get(r['verification_id'])
    if link is None:
        if r['_policy_version'] != 'u3':
            raise ValueError('no decision link for '+r['verification_id'])
        src = Path(r.get('_source', Path(r['_branches']).parent))
        key = next(json.loads(p.read_text())['search_id'] for p in (src/'labels').glob('*.json')
                   if json.loads(p.read_text())['verification']['verification_id'] == r['verification_id'])
        return _ls_snapshot(src, key)
    return torch.load(bank/'snapshots'/(link['state_id']+'.pt'), weights_only=False)


def confirm_pending(root, arm, runner_for, run_root, phase, workers=2, sources=(), limit=None, verify_mode=None):
    """Confirm every first-seed positive decision (own bank and, for B, live LS labels) once.

    verify_mode='all8_both_seeds' (owner 2026-11-03 22:05, default OFF): every completed decision
    gets all K candidates + policy on the second CRN seed, per-seed winners, per-(plan, seed)
    solos (tools/coopmwm/verify_perseed.py); None/'legacy' keeps this function unchanged."""
    if verify_mode is not None:
        from . import verify_perseed as vp
        if vp.mode(verify_mode) == vp.ALL8:
            return vp.confirm_pending_all8(root, arm, runner_for, run_root, phase, workers, sources, limit)
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
        snap = _decision_snapshot(root, arm, r, by_vid)
        receipt = {k: v for k, v in r.items() if not k.startswith('_')}
        statuses.append(confirm(root, arm, receipt, snap, runner_for, run_root, phase, workers,
                                r['_policy_version'])['admitted'])
    return statuses


def confirmations(root, arm):
    # Adopt the SHA-linked MAIN04:26 overlays made by the already-running
    # offline builder. New confirmations already carry the amended rule; old
    # originals remain immutable and must not silently replace their overlays.
    from .bank_safety import confirmations as amended
    return amended(root, arm)


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
    # MAIN 2026-11-03 04:26: the synergy twins table publishes every completed cooperation
    # record - admitted entries AND diagnostic rows of entries the safety check rejected
    # (replication + later-harm passers) - each tagged with its admission/safety verdicts.
    synergy = [c for c in confirmed if (c.get('cooperation') or {}).get('complete') is True]
    twins = [dict(verification_id=c['verification_id'], t0=c['t0'], policy_version=c['policy_version'],
                  admitted=c['admitted'], safety_old_rule_ok=c.get('safety_old_rule_ok'),
                  group_size=(c.get('cooperation') or {}).get('group_size'),
                  first_seed_advantage_veh_s=c['first_seed_advantage_veh_s'],
                  best_solo_advantage_veh_s=(c.get('cooperation') or {}).get('best_solo_advantage_veh_s'),
                  best_solo_member=(c.get('cooperation') or {}).get('best_solo_member'),
                  cooperation_gain_veh_s=(c.get('cooperation') or {}).get('cooperation_gain_veh_s'),
                  cooperative=(c.get('cooperation') or {}).get('cooperative'))
             for c in synergy]
    q = dict(arm=arm, unix=time.time(), decisions=len(decisions), first_seed_positives=len(positives),
             confirmed=len(confirmed), pending_confirmation=len(positives)-len(confirmed), admitted=len(admitted),
             confirmation_rate=(len(admitted)/len(confirmed) if confirmed else None),
             mean_admitted_advantage_veh_s=(float(values.mean()) if len(values) else None),
             mean_admitted_advantage_ci95=ci, ci_method='10000 bootstrap over admitted decisions, RNG20261101',
             rejections=dict(replication=sum(not c['checks'].get('replication', True) for c in confirmed),
                             later_harm=sum(not c['checks'].get('no_later_harm', True) for c in confirmed),
                             safety=sum(not c['checks'].get('safety', True) for c in confirmed),
                             safety_old_rule=sum(not c.get('safety_old_rule_ok', True) for c in confirmed
                                                 if 'safety_old_rule_ok' in c),
                             incomplete=sum('confirmation_branches_complete' in c['checks'] for c in confirmed)),
             by_policy_version={v: sum(c['policy_version'] == v for c in admitted)
                                for v in sorted({c['policy_version'] for c in admitted})},
             cooperative=len(cooperative),
             cooperation_undecided=sum(co.get('cooperative') is None for _, co in coop_rec),
             synergy_diagnostic=sum(c['admitted'] is False for c in synergy),
             mean_cooperation_gain_veh_s=(float(np.mean(gains)) if gains else None),
             mean_cooperation_gain_ci95=_boot_ci(gains),
             synergy_twins=twins)
    path = bank_root(root, arm)/'BANK_QUALITY.json'
    tmp = path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.publishing')
    tmp.write_text(json.dumps(q, indent=1));os.replace(tmp, path)
    return q


# ---------------------------------------------------------------- lesson check (MAIN 2026-11-03 01:22)

def lesson_list(root, arm, sources=()):
    """The FIXED lesson list: the first admitted bank decisions of this arm (receipt finish
    order, up to LESSON_N), sealed once chosen; every later update measures recapture on
    exactly the same entries (never re-chosen, never extended)."""
    path = bank_root(root, arm)/f'LESSON_LIST_{arm}.json'
    if path.exists():
        v = json.loads(path.read_text());xg.verify_sealed(v, 'lesson_list_sha256');return v
    conf = confirmations(root, arm)
    admitted = [r['verification_id'] for r in bank_decisions(root, arm, sources=sources)
                if r['status'] == 'accepted' and (r['grade_veh_s'] or 0) > 0
                and conf.get(r['verification_id'], {}).get('admitted')]
    if not admitted:
        return None
    rec = dict(arm=arm, rule='first admitted decisions in receipt finish order, up to 12, frozen once chosen',
               verification_ids=admitted[:LESSON_N], chosen_unix=time.time())
    xg._write_new(path, xg._seal(rec, 'lesson_list_sha256'))
    return rec


def _action_error(policy, r, best):
    """Mean |policy action - plan action| (m/s^2) of the NEW policy over the chosen branch's
    committed plan steps (the same member-index rule as the imitation loss)."""
    import numpy as np
    branch = r['branches'][r['chosen']+1]
    from .accel_storage import read
    tape = read(Path(r['_branches'])/r['verification_id']/branch['branch']/'RESULT.json')
    if branch.get('result_sha256') and digest(tape) != branch['result_sha256']:
        raise ValueError('bank branch tape differs from its sealed verification receipt')
    pairs = xg.imitation_pairs(tape, dict(start_s=r['t0'], duration_s=int(best['duration_s']),
                                          plan_id=best['id'], weight=1.))
    if getattr(policy, 'actor', None) is not None and policy.actor[0].weight.shape[1] > 33:
        from .bank_v2v import attach_pairs
        pairs, _ = attach_pairs(pairs, Path(r['_branches'])/r['verification_id']/branch['branch']/'observable.jsonl.gz', r['t0'])
    errs = []
    with torch.no_grad():
        for sample, _w in pairs:
            mean, _ = policy(sample['context'])
            act = torch.tensor(sample['action']).reshape(-1, 1)
            idx = [sample['context']['members'].index(v) for v in sample['behaviour']['proposal_members']
                   if v in sample['context']['members']]
            if idx:
                errs.append(float((mean[idx]-act[idx].reshape(-1)).abs().mean()))
    return float(np.mean(errs)) if errs else None


def lesson_policy_spec(snapshot, best, receipt):
    """Policy-only branch with the ORIGINAL slate's exact comparison window."""
    spec = sv.build_branches(snapshot, [best], receipt['proposer'])[0]
    window = dict(receipt['window'])
    assert window['start_s'] == snapshot['t0']
    assert window['start_s'] < window['end_s'] <= snapshot.get('end_s', 1200)
    assert window['metric'] == spec['window']['metric']
    spec.pop('branch_sha256')
    spec.update(window=window, sim_s=sv.branch_sim_s(window['end_s']),
                verification_id=spec['verification_id']+'_matched_window_v1')
    return sv._seal(spec, 'branch_sha256')


def lesson_metrics(entries, confirmed):
    """MAIN09:06: absolute paired savings primary, replication-qualified ratios."""
    import numpy as np
    deltas=[];ratios=[]
    for e in entries:
        if e.get('status')!='graded':continue
        delta=e['J_policy_u3_veh_s']-e['J_new_policy_veh_s']
        e['delta_veh_s']=delta;deltas.append(delta)
        c=confirmed.get(e['verification_id'],{})
        first=c.get('first_seed_advantage_veh_s');rep=c.get('replication_advantage_veh_s')
        spread=abs(first-rep) if first is not None and rep is not None else None
        denominator=e['J_policy_u3_veh_s']-e['J_best_plan_veh_s']
        eligible=spread is not None and abs(denominator)>spread
        e.update(replication_spread_veh_s=spread,recapture_eligible=eligible)
        if eligible:ratios.append(delta/denominator)
    rng=np.random.default_rng(20261101)
    ci=None
    if ratios:
        a=np.asarray(ratios,float)
        ci=list(map(float,np.quantile(np.median(a[rng.integers(0,len(a),(10000,len(a)))],axis=1),[.025,.975])))
    return dict(primary_metric='paired delta_veh_s; no ratio normalization',
        delta_mean_veh_s=float(np.mean(deltas)) if deltas else None,
        delta_ci95_veh_s=_boot_ci(deltas),share_delta_positive=sum(d>0 for d in deltas)/len(deltas) if deltas else None,
        recapture_eligible_entries=len(ratios),recapture_median=float(np.median(ratios)) if ratios else None,
        recapture_median_ci95=ci,recapture_rule='abs(first-seed gain) > abs(first-seed gain - replication gain)')


def lesson_check(root, arm, policy, run_root, update, runner_for, workers=2, sources=(),
                 phase='bank_verification_ext', adopt_entries=(),attempt_kind=None, v2v=False):
    """LESSON-LEARNED CHECK after an update (never raises; the training run must not depend on it).

    For each FROZEN lesson entry: ONE exact SUMO branch of the NEW policy continuing from the
    decision snapshot - same window, same CRN (identical prefix tape and generator state), only
    the policy weights differ. recapture = (J_policy_u3 - J_new_policy)/(J_policy_u3 - J_best_plan)
    per entry plus the mean (95% bootstrap CI); plus the imitation action error (m/s^2) on the
    bank states. Charges are lesson_u<k>-prefixed (the same entry is re-measured every update,
    so unqualified ids would hit the duplicate-charge guard from the second update on).
    v2v (DEFAULT OFF): the new-policy branch runner computes last_commands per step and the
    policy observation includes the v2v block (bank_v2v.branch_controller_class)."""
    import numpy as np
    bank = bank_root(root, arm);stamp = attempt_kind or f'lesson_u{int(update):03}'
    out = dict(arm=arm, update=int(update), unix=time.time(), phase=phase,
               rule='recapture = (J_policy_u3 - J_new_policy) / (J_policy_u3 - J_best_plan); '
                    '1.0 = the new policy achieves the whole verified win on its own',
               list_size=0, entries=[])
    try:
        lst = lesson_list(root, arm, sources)
        if lst is None:
            out['reason'] = 'no admitted entries yet; the lesson list is still empty'
            return out
        out.update(list_file=f'LESSON_LIST_{arm}.json', list_size=len(lst['verification_ids']),
                   list_rule=lst['rule'])
        by_vid = {}
        for p in (bank/'decisions').glob('*.json'):
            v = json.loads(p.read_text());by_vid[v['verification_id']] = v
        receipts = {r['verification_id']: r for r in bank_decisions(root, arm, sources=sources)}
        adopted={e['verification_id']:e for e in adopt_entries}
        runner = runner_for(bank/'lesson_check'/stamp/'branches')
        charge = _confirm_charge(run_root, phase, stamp)
        weights = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        new_sha = sv._state_sha(weights)
        from . import bank_v2v
        if v2v:
            out['v2v'] = dict(enabled=True,
                              rule='new-policy branch observations include the v2v block; '
                                   'last_commands computed per step (bank_v2v)')
        for vid in lst['verification_ids']:
            e = dict(verification_id=vid)
            r = receipts.get(vid)
            if r is None or r['status'] != 'accepted' or r.get('chosen') is None:
                e.update(status='missing_or_ungraded', recapture=None)
                out['entries'].append(e);continue
            prior=adopted.get(vid)
            if prior and prior.get('status')=='graded' and prior.get('new_policy_branch',{}).get('window')==r['window']:
                e=copy.deepcopy(prior);e['adopted_existing_branch']=True
                out['entries'].append(e)
                if e.get('recapture') is not None:out.setdefault('recaptures',[]).append(e['recapture'])
                if e.get('imitation_action_error_mps2') is not None:out.setdefault('action_errors_mps2',[]).append(e['imitation_action_error_mps2'])
                continue
            try:
                snap = _decision_snapshot(root, arm, r, by_vid)
            except Exception as err:
                e.update(status='snapshot_error', reason=repr(err), recapture=None)
                out['entries'].append(e);continue
            J_u3 = float(r['branches'][0]['J_veh_s'])
            J_best = float(r['branches'][r['chosen']+1]['J_veh_s'])
            new_snap = dict(snap)
            new_snap['weights'] = weights
            new_snap['policy'] = dict(state_dict_sha256=new_sha)
            best = r['chosen_plan']
            spec = lesson_policy_spec(new_snap, best, r)
            try:
                with bank_v2v.enabled(v2v):
                    charge(vid, [spec])        # before physics, like every branch charge
                    res = sv.run_branch(spec, new_snap, runner)
            except sv.BudgetBlocked as err:
                e.update(status='budget_blocked', reason=str(err), recapture=None)
                out['entries'].append(e);continue
            e.update(status=res['status'], J_policy_u3_veh_s=J_u3, J_best_plan_veh_s=J_best,
                     new_policy_branch=dict(verification_id=spec['verification_id'],
                                            snapshot_sha256=spec['snapshot_sha256'],
                                            window=spec['window'], reason=res.get('reason')))
            if res['status'] == 'graded':
                J_new = float(res['J_veh_s'])
                e['J_new_policy_veh_s'] = J_new
                denom = J_u3-J_best
                rec = (J_u3-J_new)/denom if denom > 0 else None
                e['recapture'] = rec
                if rec is not None:
                    out.setdefault('recaptures', []).append(rec)
                try:
                    aerr = _action_error(policy, r, best)
                except Exception as err:
                    aerr = None;e['action_error_reason'] = repr(err)
                if aerr is not None:
                    e['imitation_action_error_mps2'] = aerr
                    out.setdefault('action_errors_mps2', []).append(aerr)
            out['entries'].append(e)
        recaptures = out.get('recaptures', []);errs = out.get('action_errors_mps2', [])
        out.update(recapture_mean=(float(np.mean(recaptures)) if recaptures else None),
                   recapture_ci95=_boot_ci(recaptures), graded_entries=len(recaptures),
                   imitation_action_error_mean_mps2=(float(np.mean(errs)) if errs else None),
                   imitation_action_error_ci95=_boot_ci(errs))
        out.update(lesson_metrics(out['entries'],confirmations(root,arm)))
    except Exception as err:   # the update must survive any lesson-check failure
        out['error'] = repr(err)
    return out


def record_lesson(root, arm, policy, run_root, update, runner_for, workers=2, sources=(),
                  phase='bank_verification_ext', adopt_entries=(), receipt_suffix='',attempt_kind=None,
                  v2v=False):
    """Run the lesson check and write LESSON_CHECK_u<k>_<arm>.json (arm-suffixed: B and R share
    the run root; the check runs after every update of both arms). Atomic replace, so an exact
    resume that re-applies update k re-measures instead of failing on an existing file."""
    out = lesson_check(root, arm, policy, run_root, update, runner_for, workers, sources, phase,adopt_entries,attempt_kind,v2v)
    path = Path(root)/f'LESSON_CHECK_u{int(update):03}_{arm}{receipt_suffix}.json'
    tmp = path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(xg._seal(out, 'lesson_check_sha256'), indent=1))
    os.replace(tmp, path)
    return out
