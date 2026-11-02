"""SUMO verification of world-model top-k plans (owner option 1, MAIN 2026-11-02 10:20).

'WM proposes, exact SUMO picks': at the decision moment t0 the live episode blocks while
k+1 exact SUMO branches run from the SAME state: one per candidate (its plan profile for its
group, through the unchanged Controller -> requests -> safety -> CommandOwnership chain) and
one 'policy continues' branch. Everything else is identical in every branch:

- prefix 300..t0-1: the live episode's recorded one-second samples are replayed (the
  Controller replay path used by exact_grade); fingerprints 300..t0 must equal the live
  ones bit for bit (PrefixMismatch -> verification STOPPED, the group follows its policy,
  never imputed);
- t0: the live t0 sample (taken before any plan was applied) is reused verbatim, the AR
  noise memory and the torch generator state captured at the decision are restored, then
  ONE generator draw is consumed (the live Controller's beta draw if accepted, or the
  verification pad draw if rejected) so all k+1 branches AND the live continuation share
  the same noise stream (CRN);
- plans already committed at t0 continue their recorded profiles through the same chain;
  plans that would start after t0 are unknown at decision time and are not simulated; no
  new WM query happens inside a branch.

Choice: J = increase of cumulative ``team_delay_vs`` over [t0, t_end], t_end =
min(t0 + max candidate duration + 120, episode end 1200), the same window for all branches.
The best candidate (min J, ties by candidate order) is executed only if J_best < J_policy;
the chosen plan's exact grade is J_policy - J_chosen (positive = helped) from the same
branches. Budget: every branch is charged like an accel_plant job (t_end sim-s from t=0),
precharged in full before any physics; cost per decision = (k+1) * t_end.

SUMO is behind an injectable runner: ``SubprocessBranchRunner`` (cpuQ, nice 19, owned
PIDs only) or a fake deterministic plant in the CPU tests.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

from .branch_eval import totals_at
from .exact_grade import BudgetBlocked, PrefixMismatch, _seal, _write_new, verify_sealed
from .protocol import canonical, digest

SCHEMA = 'coopmwm/sumo_verify/v1'
EXPERIMENT = 'S-COOPMWM-SUMO-VERIFY'
WARMUP_S = 300
EPISODE_END_S = 1200
TAIL_S = 120
METRIC = 'team_delay_vs'
POLICY = 'policy'
TOP_K = 4
STATUSES = ('accepted', 'rejected', 'stopped', 'failed', 'budget_blocked')
MIN_FREE_BYTES = 2*1024**3  # the registered cpuQ '2 GiB' disk admission guard (MAIN 14:47)
PROPOSERS = ('wm', 'random')


# ---------------------------------------------------------------- specs

def verification_id(episode_id, t0, candidates):
    return f'sv_{episode_id}_t{int(t0):04d}_{digest([c["id"] for c in candidates])[:8]}'


def window_end(t0, candidates, end_s=EPISODE_END_S, tail_s=TAIL_S):
    return min(int(t0)+max(int(c['duration_s']) for c in candidates)+int(tail_s), int(end_s))


def branch_sim_s(t_end):
    # Charged like every accel_plant job: simulated seconds from t=0 to the stop (warmup included).
    return int(t_end)


def quote(t0, k=TOP_K, max_duration_s=300, end_s=EPISODE_END_S, tail_s=TAIL_S):
    """Charged sim-s for one verification decision with k candidates (+ the policy branch)."""
    return (int(k)+1)*branch_sim_s(min(int(t0)+int(max_duration_s)+int(tail_s), int(end_s)))


def snapshot_sha(snapshot):
    """Identity of the decision-time snapshot (episode, prefix tape, fingerprints, RNG state)."""
    keys = ('episode', 'policy', 'sumo', 't0', 'prefixes', 'samples', 'rng', 'committed_plans', 'end_s',
            'truncate_on_accept')
    return digest({k: snapshot.get(k) for k in keys})


def build_branches(snapshot, candidates, proposer='wm'):
    """k+1 sealed branch specs from one decision-time snapshot. Branch 0 is 'policy continues'."""
    if proposer not in PROPOSERS:
        raise ValueError('unregistered proposer')
    if not candidates:
        raise ValueError('no candidates to verify')
    t0, end = int(snapshot['t0']), int(snapshot.get('end_s', EPISODE_END_S))
    if not WARMUP_S <= t0 < end:
        raise ValueError('decision time outside the controlled episode')
    ids = [c['id'] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate candidate ids')
    committed = {v for c in snapshot.get('committed_plans', []) for v in c['plan']['members']}
    for c in candidates:
        if int(c['duration_s']) <= 0:
            raise ValueError('candidate duration must be positive')
        if set(c['members']) & committed:
            raise ValueError('candidate overlaps a committed plan group')
    t_end = window_end(t0, candidates, end)
    vid = verification_id(snapshot['episode']['id'], t0, candidates)
    snap = snapshot_sha(snapshot)
    common = dict(schema=SCHEMA, experiment=EXPERIMENT, verification_id=vid, snapshot_sha256=snap,
                  episode=snapshot['episode'], proposer=proposer,
                  window=dict(start_s=t0, end_s=t_end, tail_s=TAIL_S, metric=METRIC,
                              truncated=t_end < t0+max(int(c['duration_s']) for c in candidates)+TAIL_S),
                  committed_plan_ids=[c['plan']['id'] for c in snapshot.get('committed_plans', [])],
                  sim_s=branch_sim_s(t_end))
    out = [_seal(dict(common, branch=POLICY, index=0, plan=None), 'branch_sha256')]
    for i, c in enumerate(candidates, 1):
        out.append(_seal(dict(common, branch=f'cand{i}', index=i, plan=copy.deepcopy(c),
                              plan_sha256=digest(c)), 'branch_sha256'))
    return out


# ---------------------------------------------------------------- execution

def check_prefix(result, snapshot):
    t0 = int(snapshot['t0'])
    for t in range(WARMUP_S, t0+1):
        mine, ref = result['prefixes'].get(str(t)), snapshot['prefixes'].get(str(t))
        if mine is None or ref is None or canonical(mine) != canonical(ref):
            raise PrefixMismatch(f'branch fingerprint differs from the live episode at t={t}')
    return True


def window_J(result, start, end):
    a, b = totals_at(result, start), totals_at(result, end)
    return {k: b[k]-a[k] for k in a}


def run_branch(spec, snapshot, runner):
    """One branch -> receipt dict (status graded|stopped|failed). Never raises for physics."""
    verify_sealed(spec, 'branch_sha256')
    t0, t_end = spec['window']['start_s'], spec['window']['end_s']
    try:
        result = runner.run(spec, snapshot)
        check_prefix(result, snapshot)
        w = window_J(result, t0, t_end)
        return dict(branch=spec['branch'], index=spec['index'], status='graded', J_veh_s=w[METRIC], window=w,
                    branch_sha256=spec['branch_sha256'], result_sha256=result.get('result_sha256') or digest(result),
                    actual_sim_s=result.get('actual_sim_s'), audit=result.get('audit'),
                    continuation_prefix_sha256=result.get('continuation_prefix_sha256', {}))
    except PrefixMismatch as e:
        return dict(branch=spec['branch'], index=spec['index'], status='stopped', J_veh_s=None,
                    reason='PrefixMismatch: '+str(e), branch_sha256=spec['branch_sha256'])
    except Exception as e:  # physics/executor error: never retried or imputed
        return dict(branch=spec['branch'], index=spec['index'], status='failed', J_veh_s=None,
                    reason=repr(e), branch_sha256=spec['branch_sha256'])


def run_parallel(specs, snapshot, runner, workers=None):
    """Bounded parallel execution; results are returned in branch order (deterministic)."""
    workers = max(1, min(int(workers or len(specs)), len(specs)))
    if workers == 1:
        return [run_branch(s, snapshot, runner) for s in specs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda s: run_branch(s, snapshot, runner), specs))


def choose(branches, candidates):
    """Pure decision rule. Returns (status, chosen index or None, reason).

    MAIN (harden-int): a crashed CANDIDATE branch no longer voids the decision. If the
    'policy continues' branch is graded, the best among the successfully graded candidates
    is chosen and the missing ones are recorded (graded_candidates / missing_candidates in
    the receipt). If the policy branch failed, or any branch hit a prefix mismatch
    (integrity), or no candidate was graded, the decision is void (never imputed).
    """
    bad = [b for b in branches if b['status'] != 'graded']
    stopped = [b for b in bad if b['status'] == 'stopped']
    if stopped or branches[0]['status'] != 'graded' or len(bad) == len(branches)-1:
        status = 'stopped' if stopped else 'failed'
        return status, None, f'{status}: '+'; '.join(f'{b["branch"]}: {b.get("reason")}' for b in bad)
    J_policy = branches[0]['J_veh_s']
    graded = [i for i in range(1, len(branches)) if branches[i]['status'] == 'graded']
    best = min(graded, key=lambda i: (branches[i]['J_veh_s'], i))
    missing = '' if not bad else f' ({len(bad)} candidate branch(es) failed, excluded: '+', '.join(b['branch'] for b in bad)+')'
    if branches[best]['J_veh_s'] < J_policy:
        return 'accepted', best, 'best graded candidate beats policy continues'+missing
    return 'rejected', None, 'no graded candidate beats policy continues'+missing


def verify(snapshot, candidates, runner, charge, proposer='wm', workers=None, root=None):
    """Full verification of one decision. Returns a sealed receipt (written to root if given).

    charge(verification_id, branch_specs) must durably precharge ALL k+1 branches or raise
    BudgetBlocked (then no physics runs and the group follows its policy)."""
    started = time.time()
    specs = build_branches(snapshot, candidates, proposer)
    vid, t0 = specs[0]['verification_id'], specs[0]['window']['start_s']
    charged = sum(s['sim_s'] for s in specs)
    body = dict(schema=SCHEMA, experiment=EXPERIMENT, verification_id=vid, proposer=proposer,
                snapshot_sha256=specs[0]['snapshot_sha256'], episode=snapshot['episode'], t0=t0,
                window=specs[0]['window'], k=len(candidates), candidate_ids=[c['id'] for c in candidates],
                candidate_plan_sha256=[digest(c) for c in candidates],
                predicted_saving=[c.get('saving_vs_policy', c.get('saving_vs_mappo')) for c in candidates],
                branch_specs_sha256=[s['branch_sha256'] for s in specs],
                sign='grade = J_policy - J_chosen; positive = chosen plan helped (veh-s)')
    if root is not None:
        path = Path(root)/'verifications'/f'{vid}.json'
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            verify_sealed(previous, 'receipt_sha256')
            if any(previous.get(key) != body[key] for key in
                   ('snapshot_sha256', 'candidate_plan_sha256', 'proposer')):
                raise ValueError('verification identity changed; original receipt retained')
            return previous
        # An interrupted claim is never replayed, even if its process has exited.
        # Claim BEFORE charging/preparing/launching any branch, across processes.
        _write_new(Path(root)/'claims'/f'{vid}.json', dict(
            verification_id=vid, snapshot_sha256=body['snapshot_sha256'],
            candidate_plan_sha256=body['candidate_plan_sha256'],
            pid=os.getpid(), host=socket.gethostname(), unix=started))
    blocked = runner.admit(specs) if hasattr(runner, 'admit') else None
    if blocked:  # local resource guard: no charge, no physics, the group follows its policy
        body.update(status='failed', chosen=None, chosen_plan=None, grade_veh_s=None, J_policy_veh_s=None,
                    J_candidates_veh_s=None, branches=[], charged_sim_s=0, reason='resource guard: '+blocked)
        return _finish(body, started, root)
    try:
        charge(vid, specs)
    except BudgetBlocked as e:
        body.update(status='budget_blocked', chosen=None, chosen_plan=None, grade_veh_s=None, J_policy_veh_s=None,
                    J_candidates_veh_s=None, branches=[], charged_sim_s=0, reason='BudgetBlocked: '+str(e))
        if root is not None:  # MAIN (d): a blocked verification budget must never be silent
            alert = Path(root)/'alerts'/f'BUDGET_BLOCKED_{vid}.json'
            if not alert.exists():
                _write_new(alert, dict(alert='sumo verification budget blocked; group follows its policy',
                                       verification_id=vid, phase_reason=str(e), unix=started))
        return _finish(body, started, root)
    if hasattr(runner, 'prepare'):
        try:
            runner.prepare(specs, snapshot)
        except Exception as e:  # e.g. full disk writing SNAPSHOT.pt: one decision, never the episode
            body.update(status='failed', chosen=None, chosen_plan=None, grade_veh_s=None, J_policy_veh_s=None,
                        J_candidates_veh_s=None, branches=[], charged_sim_s=charged, actual_sim_s=0,
                        reason='prepare failed before physics: '+repr(e))
            return _finish(body, started, root)
    branches = run_parallel(specs, snapshot, runner, workers)
    status, best, reason = choose(branches, candidates)
    J_policy = branches[0]['J_veh_s']
    J = [b['J_veh_s'] for b in branches[1:]]
    ok = [i for i, x in enumerate(J) if x is not None]
    done = status in ('accepted', 'rejected')
    body.update(status=status, reason=reason, branches=branches, charged_sim_s=charged,
                branch_sim_s=specs[0]['sim_s'], n_branches=len(specs),
                actual_sim_s=sum(b.get('actual_sim_s') or 0 for b in branches),
                J_policy_veh_s=J_policy, J_candidates_veh_s=J if done else None,
                graded_candidates=len(ok), missing_candidates=[b['branch'] for b in branches[1:] if b['status'] != 'graded'],
                available_k=len(candidates),
                best_index=(min(ok, key=lambda i: (J[i], i)) if done else None),
                chosen=(best-1 if best else None), chosen_plan=(copy.deepcopy(candidates[best-1]) if best else None),
                grade_veh_s=(J_policy-J[best-1] if best else None),
                best_margin_veh_s=(J_policy-min(J[i] for i in ok) if done else None))
    return _finish(body, started, root)


def _finish(body, started, root):
    body['decision_sha256'] = digest(body)       # deterministic core (no wall times/hosts)
    r = dict(body, timing=dict(started_unix=started, finished_unix=time.time(), wall_s=time.time()-started),
             host=socket.gethostname(), pid=os.getpid())
    if r['status'] not in STATUSES:
        raise ValueError('unknown status')
    r = _seal(r, 'receipt_sha256')
    if root is not None:
        _write_new(Path(root)/'verifications'/f'{body["verification_id"]}.json', r)
    return r


def check_live_continuation(verification, live):
    """Smoke gate: exact chosen-branch/live fingerprints through the next query window.

    Reads completed TRAIN results only; no runner, charges or physics. A missing
    fingerprint is a failure, never a vacuous equivalence pass. Future searches
    can change later traffic, so this gate makes no whole-window replay claim.
    """
    verify_sealed(verification, 'receipt_sha256')
    if verification['status'] not in ('accepted','rejected'):
        raise ValueError('smoke needs a completed exact verified decision')
    if live['id'] != verification['episode']['id']:
        raise ValueError('live continuation episode identity differs')
    t0 = verification['t0']
    step = next((s for s in live['steps'] if int(s['time_s']) == t0), None)
    proposal = (step or {}).get('proposal') or {}
    behaviour = (step or {}).get('sample', {}).get('behaviour') or {}
    if verification['status']=='accepted':
        if (verification['chosen'] is None or not behaviour.get('executed') or not behaviour.get('plan_start') or
                digest(proposal.get('plan')) != digest(verification['chosen_plan'])):
            raise ValueError('live continuation did not execute the verified plan')
    elif (verification['chosen'] is not None or proposal.get('accepted') or
          behaviour.get('plan_start') or any(p.get('plan_start') for p in (step or {}).get('sample',{}).get('plan_executions',[]))):
        raise ValueError('rejected verification must continue policy without starting a plan')
    identity = behaviour.get('sumo_verification') or {}
    if identity.get('receipt_sha256') != verification['receipt_sha256']:
        raise ValueError('live continuation verification receipt differs')
    chosen = verification['branches'][verification['chosen']+1 if verification['status']=='accepted' else 0]
    if chosen['status'] != 'graded':
        raise ValueError('chosen branch was not graded')
    hashes = chosen.get('continuation_prefix_sha256') or {}
    end = min((int(t0)//10+1)*10, int(float(live['final']['t_s'])))
    if end <= t0:
        raise ValueError('no live continuation seconds')
    for t in range(t0, end+1):
        fp = live['prefixes'].get(str(t))
        if fp is None and float(live['final']['t_s']) == t:
            fp = live['final']
        if fp is None or hashes.get(str(t)) != digest(fp):
            raise PrefixMismatch(f'chosen branch differs from live continuation at t={t}')
    return dict(pass_=True, verification_id=verification['verification_id'],
                decision_status=verification['status'],continuation_branch=chosen['branch'],
                verification_receipt_sha256=verification['receipt_sha256'],
                checked_from_s=t0, checked_to_s=end, fingerprints=end-t0+1,
                new_sim_s=0, scope='chosen branch to next query window, not whole grading window')


# ---------------------------------------------------------------- budget

def local_charge(root):
    """Append-only charge ledger (tests / dry runs); one line per branch, written before physics."""
    def charge(vid, specs):
        path = Path(root)/'VERIFY_CHARGES.jsonl';path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            for s in specs:
                f.write(canonical(dict(id=f'{vid}_{s["branch"]}', phase='sumo_verification', sim_s=s['sim_s']))+'\n')
            f.flush();os.fsync(f.fileno())
    return charge


def accel_charge(run_root, phase='sumo_verification'):
    """Charge into the live run ledger (Linux, fcntl) under the registered verification phase."""
    def charge(vid, specs):
        from .warm_protocol import install_accounting
        from .accel_storage import reserve
        install_accounting(run_root)
        try:
            reserve(Path(run_root), [dict(id=f'{vid}_{s["branch"]}', sim_s=s['sim_s']) for s in specs], phase)
        except ValueError as e:
            raise BudgetBlocked(str(e)) from e
    return charge


# ---------------------------------------------------------------- arm R proposer

def random_candidates(group, centre, bounds, regime, base_seed, k=TOP_K, duration_s=None, first=None):
    """Arm R = 'random proposals + SUMO pick': k generic random plans around the same policy centre
    (mppi_search.sample_unscored, zero WM scoring). Seeds derive from base_seed only, so no extra
    live generator draws. ``first`` (the matched random plan already drawn) is kept as candidate 1."""
    from .mppi_search import sample_unscored
    out = [first] if first is not None else []
    i = 0
    while len(out) < k and i < 64*int(k):  # bounded: identical (fully clipped) draws cannot spin forever
        seed = int(digest(dict(base=int(base_seed), i=i))[:8], 16);i += 1
        p = sample_unscored(group, centre, bounds, regime, seed, duration_s=duration_s)
        if all(p['id'] != q['id'] for q in out):
            out.append(p)
    return out


# ---------------------------------------------------------------- resume preflight

def resume_preflight(cfg, search_config=None, learned=False):
    """Static checks of a job['sumo_verify'] binding before a k / proposer switch (no I/O, no SUMO).

    Returns human-readable problems (empty = consistent). Advisory: it never changes a
    registered setting; MAIN decides. ``search_config`` is the GPU binding's search_config.
    """
    import inspect
    import math
    from . import mppi_search as ms
    problems = []
    k = int(cfg.get('k', TOP_K))
    top_k = int(dict(ms.REGISTERED, **(search_config or {}))['top_k'])
    if k > top_k:
        problems.append(f'B search top_k={top_k} < verification k={k}: B verifies at most {top_k} '
                        f'candidates (fewer when savings <= 0) while R always gets {k}')
    workers = cfg.get('workers')
    if workers is not None and int(workers) < k+1:
        problems.append(f'workers={workers} < k+1={k+1}: the blocking decision runs '
                        f'{math.ceil((k+1)/int(workers))} branch waves')
    if k != TOP_K and cfg.get('phase', 'sumo_verification') != 'sumo_verification_k8':
        problems.append(f'k={k} charges must use the registered sumo_verification_k8 phase (MAIN 16:43)')
    if learned:
        params = inspect.signature(ms.search).parameters
        for name in ('warm_start', 'extra_seeds'):
            if name not in params:
                problems.append(f'mppi_search.search lacks {name}: learned kwargs raise TypeError and '
                                'query_errors declines EVERY query silently')
        try:
            from .plan_proposer import learned_search_kwargs, LEARNED_TOP_K  # noqa: F401
        except ImportError:
            problems.append('plan_proposer lacks learned_search_kwargs/LEARNED_TOP_K (c4/plan-proposer not merged)')
        else:
            if LEARNED_TOP_K < k:
                problems.append(f'LEARNED_TOP_K={LEARNED_TOP_K} < k={k}')
        from . import mppi_runtime
        if 'learned_search_kwargs' not in inspect.getsource(mppi_runtime.plan):
            problems.append('mppi_runtime.plan has no learned-search hook; its own config= keyword must be '
                            'merged with the learned config (a second config= is a TypeError)')
    return problems


# ---------------------------------------------------------------- live integration hook

def capture(controller, prefixes, job, now, sample, generator):
    """Decision-time snapshot from the live accel_plant episode (called inside the guide, before
    any plan is applied at t0 and before the Controller's beta draw)."""
    t0 = int(now)
    samples = {str(e['time_s']): copy.deepcopy(e['sample']) for e in controller.events if WARMUP_S <= e['time_s'] < t0}
    if sorted(map(int, samples)) != list(range(WARMUP_S, t0)):
        raise ValueError('live tape lacks prefix seconds')
    samples[str(t0)] = copy.deepcopy(sample)
    return dict(
        episode=dict(id=job['id'], world=job['world'], arm=job['arm'], update=job.get('update'),
                     role=job.get('role'), training_seed=job.get('training_seed')),
        policy=dict(state_dict_sha256=_state_sha(controller.policy.state_dict())),
        sumo=dict(runtime=job['runtime'], checkpoint=job['checkpoint'], warm_history=bool(job.get('warm_history')),
                  accel_bounds=job.get('accel_bounds'), noise_seed=job.get('noise_seed')),
        t0=t0, end_s=int(job.get('stop', EPISODE_END_S)),
        prefixes={str(t): copy.deepcopy(prefixes[str(t)]) for t in range(WARMUP_S, t0+1)},
        samples=samples,
        rng=dict(generator_state_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                 memory_after_t0=dict(controller.memory)),
        generator_state=generator.get_state().clone(),
        weights={k: v.clone() for k, v in controller.policy.state_dict().items()},
        committed_plans=[dict(plan=copy.deepcopy(c['plan']), start_s=c['start_s']) for c in controller.active_plans
                         if t0 < c['start_s']+c['plan']['duration_s']])


def _state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode());h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def verified_guide(guide, verifier, controller, prefixes, job, generator, k=TOP_K, random_proposer=None, log=None):
    """Wrap the live plan_client guide: BLOCKING verification at the decision moment.

    - accepted WM proposal (arm B): candidates = proposal['topk'][:k] (or the single plan);
    - arm R: candidates = random_proposer(proposal, k) (see random_candidates);
    - verifier(snapshot, candidates, proposer) -> receipt (``verify`` bound to runner/charge/root).
    Accepted -> the proposal executes the SUMO-chosen candidate (quota counts it as executed via
    guide.completed). Rejected/stopped/failed/budget_blocked -> proposal['accepted']=False, one pad
    generator draw keeps the live stream identical to the branches, group follows its policy and
    the quota reservation resolves executed=False."""
    import torch

    def call(rows, commands, now, sample):
        proposal = guide(rows, commands, now, sample)
        if proposal is None or not proposal.get('accepted'):
            return proposal
        proposer = 'random' if proposal.get('random_match') is not None else 'wm'
        if proposer == 'random':
            if random_proposer is None:
                raise ValueError('arm R verification needs a random proposer')
            candidates = random_proposer(proposal, k)
        else:
            candidates = list(proposal.get('topk') or [proposal['plan']])[:k]
        # k=8 switch: the plain search returns at most its registered top_k (4) and
        # only positive-saving finalists, so B may verify fewer than k. Record it
        # per decision; the B/R count match is a live owner check, not imputed.
        proposal['behaviour']['verify_k'] = dict(requested=int(k), available=len(candidates))
        from .exploration import bucket_beta
        b = proposal['behaviour']
        tracked_beta = bucket_beta(getattr(controller, 'beta', 1.),
                            proposal.get('option', proposal['plan'].get('family', 'MPPI')),
                            b.get('regime', 'ordinary'))
        fixed_beta = job.get('sumo_verify', {}).get('guide_beta')
        if fixed_beta is not None and (fixed_beta != 1. or job['arm'] not in ('B', 'R')):
            raise ValueError('verified guide_beta must be 1 for B/R')
        beta = tracked_beta if fixed_beta is None else 1.
        # Snapshot retains the PRE-draw state: branches consume this same one
        # draw at t0. Live consumes it here, never again in Controller.apply.
        snap = capture(controller, prefixes, job, now, sample, generator)
        follows = float(torch.rand((), generator=generator)) < beta
        b.update(beta=beta, q=b.get('sample_probability', 1.)*beta,
                  trust_draw_before_verification=True, trust_follow=follows,
                  tracked_trust_beta=tracked_beta, trust_reporting_only=fixed_beta is not None)
        proposal['sumo_trust_drawn'] = True
        if not follows:
            proposal.update(accepted=False, reason='pre-verification trust draw: own policy')
            b.update(proposal_accepted=False, rejection=proposal['reason'])
            return proposal
        ended = (proposal.get('overlap_recheck') or {}).get('truncated_plan_ids') or []
        snap['truncate_on_accept'] = sorted(ended)   # committed plans an accepted start would truncate
        r = verifier(snap, candidates, proposer)
        b['sumo_verification'] = dict(verification_id=r['verification_id'], status=r['status'],
                                      receipt_sha256=r['receipt_sha256'], chosen=r['chosen'],
                                      grade_veh_s=r['grade_veh_s'], J_policy_veh_s=r['J_policy_veh_s'],
                                      J_candidates_veh_s=r['J_candidates_veh_s'], charged_sim_s=r['charged_sim_s'],
                                      wall_s=r['timing']['wall_s'])
        if log is not None:
            log(r)
        if r['status'] == 'accepted':
            plan = r['chosen_plan']
            proposal.update(plan=plan, sumo_verified=True, expected_saving=plan.get('saving_vs_mappo'),
                            exact_grade_veh_s=r['grade_veh_s'])
            if plan.get('overlap_truncated_plan_ids') is not None and proposal.get('overlap_recheck'):
                # Live truncation follows the chosen candidate's own recheck (as in its branch).
                proposal['overlap_recheck'] = dict(proposal['overlap_recheck'],
                                                   truncated_plan_ids=list(plan['overlap_truncated_plan_ids']),
                                                   per_candidate=True)
            b.update(proposal_members=plan['members'], duration_s=plan['duration_s'],
                     expected_saving=plan.get('saving_vs_mappo'), group_size=len(plan['members']))
            return proposal
        proposal.update(accepted=False, sumo_verified=False, reason=f'sumo_verify {r["status"]}: {r.get("reason")}')
        b.update(proposal_accepted=False, rejection=proposal['reason'])
        return proposal

    for name in ('completed', 'observe', 'prime'):
        if hasattr(guide, name):
            setattr(call, name, getattr(guide, name))
    return call


def install(controller, job, prefixes, generator, runner=None, charge=None):
    """accel_plant.episode hook (guarded by job['sumo_verify']): wraps controller.guide in place."""
    if not job.get('sumo_verify') or controller.guide is None:
        return controller
    cfg = job['sumo_verify']
    root = Path(cfg['root'])
    k = int(cfg.get('k', TOP_K))
    runner = runner or SubprocessBranchRunner(root/'branches', workers=cfg.get('workers', k+1),
                                              timeout_s=cfg.get('timeout_s', 3600))
    charge = charge or accel_charge(cfg['run_root'], cfg.get('phase', 'sumo_verification'))
    rp = None
    if job['arm'] == 'R':
        def rp(proposal, k):
            ctx = proposal['verify_context']
            return random_candidates(ctx['group'], ctx['centre'], ctx['bounds'], ctx['regime'],
                                     int(digest(proposal['plan'])[:8], 16), k, ctx.get('duration_s'), proposal['plan'])
    verifier = lambda snap, cands, proposer: verify(snap, cands, runner, charge, proposer,
                                                    workers=cfg.get('workers'), root=root)
    controller.guide = verified_guide(controller.guide, verifier, controller, prefixes, job, generator,
                                      k=k, random_proposer=rp)
    return controller


# ---------------------------------------------------------------- real executor (cpuQ)

def verify_controller_class():
    from .acceleration import Controller

    class _OneShot:
        def __init__(self, policy, sample, memory):
            self.policy, self.pending = policy, (sample, memory)

        def __call__(self, *a, **kw):
            return self.policy(*a, **kw)

        def sample(self, *a, **kw):
            if self.pending is not None:
                s, m = self.pending;self.pending = None
                return copy.deepcopy(s), dict(m)
            return self.policy.sample(*a, **kw)

    class VerifyController(Controller):
        """Replays the live prefix; at t0 reuses the live sample, restores memory/generator, starts the
        candidate (or nothing for 'policy'); committed plans continue; no guide after t0."""

        def __init__(self, policy, generator, snapshot, plan):
            super().__init__(policy, generator, deterministic=False, guide=None, beta=1., replay=None, branch=None)
            self.snap, self.cand, self.t0 = snapshot, plan, int(snapshot['t0'])
            self.base_policy = policy
            self.audit = dict(replayed_prefix_steps=0, live_steps=0, candidate=(plan or {}).get('id'))

        def apply(self, rows, commands, now):
            t = int(now)
            if t < self.t0:
                self.replay = {str(t): self.snap['samples'][str(t)]}
                try:
                    return super().apply(rows, commands, now)
                finally:
                    self.replay = None
                    self.audit['replayed_prefix_steps'] += 1
            if t == self.t0:
                self.generator.set_state(self.snap['generator_state'])
                state_sha = hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest()
                if state_sha != self.snap['rng']['generator_state_sha256']:
                    raise ValueError('verification generator snapshot SHA mismatch')
                self.audit.update(crn_verified_t0=True, noise_mode_used='crn')
                own = (self.cand or {}).get('overlap_truncated_plan_ids')   # MAIN: per-candidate truncation
                ended = set(own if own is not None else (self.snap.get('truncate_on_accept') or [])) if self.cand is not None else set()
                self.active_plans = [dict(plan=copy.deepcopy(c['plan']), start_s=c['start_s'])
                                     for c in self.snap['committed_plans'] if c['plan']['id'] not in ended]
                if self.cand is not None:
                    self.active_plans.append(dict(plan=copy.deepcopy(self.cand), start_s=self.t0))
                self.policy = _OneShot(self.base_policy, self.snap['samples'][str(t)], self.snap['rng']['memory_after_t0'])
                try:
                    desired = super().apply(rows, commands, now)
                finally:
                    self.policy = self.base_policy
                import torch
                torch.rand((), generator=self.generator)   # beta / pad draw (same in every branch and live)
            else:
                desired = super().apply(rows, commands, now)
            self.audit['live_steps'] += 1
            return desired

    return VerifyController


class LocalBranchRunner:
    """In-process adapter onto accel_plant.episode (one isolated process per branch in production)."""

    def __init__(self, work_root):
        self.work_root = Path(work_root)

    def run(self, spec, snapshot):
        from . import accel_plant
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        cls = verify_controller_class();holder = {}
        def factory(policy, generator, deterministic=False, guide=None, beta=1., replay=None, branch=None):
            holder['c'] = cls(policy, generator, snapshot, spec['plan'])
            return holder['c']
        ep, sumo = snapshot['episode'], snapshot['sumo']
        job = dict(id=f'{spec["verification_id"]}_{spec["branch"]}',
                   path=str(self.work_root/spec['verification_id']/spec['branch']),
                   world=ep['world'], role=ep.get('role') or 'train', arm=ep['arm'], weights=snapshot['weights'],
                   runtime=sumo['runtime'], checkpoint=sumo['checkpoint'], noise_seed=sumo.get('noise_seed') or 0,
                   deterministic=True, guided=False, health_guards=False, sim_s=t_end, stop=t_end,
                   warm_history=sumo['warm_history'], accel_bounds=sumo['accel_bounds'],
                   branch=dict(time_s=t0, takeover_s=t0, revert=[],
                               prefixes={str(t): snapshot['prefixes'][str(t)] for t in range(WARMUP_S, t0+1)}))
        original = accel_plant.Controller
        accel_plant.Controller = factory
        try:
            result = accel_plant.episode(job)
        except ValueError as e:
            if 'prefix mismatch' in str(e):
                raise PrefixMismatch(str(e)) from e
            raise
        finally:
            accel_plant.Controller = original
        continuation = {}
        for t in range(t0, min((int(t0)//10+1)*10, t_end)+1):
            fp = result['final'] if t == t_end else result['prefixes'][str(t)]
            continuation[str(t)] = digest(fp)
        return dict(prefixes={str(t): result['prefixes'][str(t)] for t in range(WARMUP_S, t0+1)},
                     final=result['final'], actual_sim_s=result.get('actual_sim_s'),
                     result_sha256=digest(result), audit=holder['c'].audit,
                     continuation_prefix_sha256=continuation)


class SubprocessBranchRunner:
    """One subprocess per branch (nice 19 on POSIX), PIDs logged; only our own Popen objects are ever
    waited on or (on timeout) terminated. Never kills by name."""

    def __init__(self, work_root, workers=TOP_K+1, timeout_s=3600., python=None, min_free_bytes=MIN_FREE_BYTES):
        self.work_root, self.workers, self.timeout_s = Path(work_root), workers, float(timeout_s)
        self.python = python or sys.executable
        self.min_free_bytes = int(min_free_bytes)

    def admit(self, specs):
        """Existing cpuQ 2 GiB free-disk admission guard, checked before any charge."""
        import shutil
        self.work_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.work_root).free
        if free < self.min_free_bytes:
            return f'free disk {free} B < registered cpuQ admission guard {self.min_free_bytes} B'
        return None

    def prepare(self, specs, snapshot):
        """Write the shared decision snapshot once (before the parallel branches start)."""
        import torch
        d = self.work_root/specs[0]['verification_id'];d.mkdir(parents=True, exist_ok=True)
        with (d/'SNAPSHOT.pt').open('xb') as f:
            torch.save(snapshot, f);f.flush();os.fsync(f.fileno())
        with (d/'SNAPSHOT.sha256').open('x') as f:
            f.write(hashlib.sha256((d/'SNAPSHOT.pt').read_bytes()).hexdigest()+'\n')

    def run(self, spec, snapshot):
        import subprocess
        d = self.work_root/spec['verification_id']
        snap_path = d/'SNAPSHOT.pt'
        if not snap_path.exists():
            self.prepare([spec], snapshot)
        spec_path = d/f'{spec["branch"]}.spec.json';out = d/f'{spec["branch"]}.result.json'
        if not spec_path.exists():
            _write_new(spec_path, spec)
        nice = (lambda: os.nice(max(0, 19-os.nice(0)))) if os.name == 'posix' else None
        p = subprocess.Popen([self.python, '-m', 'tools.coopmwm.sumo_verify', 'run-branch', '--snapshot', str(snap_path),
                              '--spec', str(spec_path), '--out', str(out), '--work', str(self.work_root)],
                             preexec_fn=nice)
        with (self.work_root/'WORKERS.jsonl').open('a', encoding='utf-8') as f:
            f.write(canonical(dict(pid=p.pid, branch=spec['branch'], verification_id=spec['verification_id'],
                                   host=socket.gethostname(), unix=time.time()))+'\n')
        try:
            code = p.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            p.terminate();p.wait()
            raise RuntimeError(f'branch {spec["branch"]} exceeded {self.timeout_s} s (own pid {p.pid} terminated)')
        r = json.loads(out.read_text(encoding='utf-8'))
        if r.get('error'):
            if r.get('prefix_mismatch'):
                raise PrefixMismatch(r['error'])
            raise RuntimeError(r['error'])
        if code != 0:
            raise RuntimeError(f'branch worker exit {code}')
        return r


def main(argv=None):
    import argparse
    import torch
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    rb = sub.add_parser('run-branch')
    rb.add_argument('--snapshot', required=True);rb.add_argument('--spec', required=True)
    rb.add_argument('--out', required=True);rb.add_argument('--work', required=True)
    q = sub.add_parser('quote')
    q.add_argument('--t0', type=int, required=True);q.add_argument('--k', type=int, default=TOP_K)
    q.add_argument('--max-duration', type=int, default=300)
    a = ap.parse_args(argv)
    if a.cmd == 'quote':
        print(quote(a.t0, a.k, a.max_duration));return 0
    spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    snapshot = torch.load(a.snapshot, weights_only=False)
    try:
        r = LocalBranchRunner(a.work).run(spec, snapshot);code = 0
    except PrefixMismatch as e:
        r = dict(error=str(e), prefix_mismatch=True);code = 3
    except Exception as e:
        r = dict(error=repr(e));code = 4
    _write_new(a.out, r)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
