"""Option ``coop_objective`` for the MPPI/niche plan search (default OFF).

Finds MORE COOPERATIVE and MORE VARIED multi-CAV plans without seeding the
search with any known or hand-designed behaviour family. With the option off
``mppi_search`` never imports this module and its output is byte-identical.

1. Synergy-aware WM score. For a candidate group plan the WM also scores, for
   EVERY member, the SOLO variant: only that member follows its profile row,
   all other CAVs (the other group members included) keep 'policy continues'
   (the same rule as ``replay_bank.solo_candidate`` used for the SUMO synergy
   check). All gains are paired against 'policy' of the SAME WM call::

       g      = mean_k(policy - group)                       (group gain)
       s_max  = max_i mean_k(policy - solo_i)                (best solo gain)
       synergy = g - s_max
       score  = g + coop_lambda * synergy                    (higher is better)

   Cost cap: solo rollouts are made only for the top ``coop_solo_cap``
   candidates of a call by group gain (default 16). When the cap covers the
   whole call the solos are batched INTO the candidate call; otherwise one
   extra call (candidates' solos + the two comparators) follows the screen
   call. Candidates outside the cap get the MINIMUM synergy computed in that
   call (pessimistic, flagged ``synergy_imputed``; the imputed share is
   logged) and are finalist-ineligible: if fewer measured candidates than
   finalists exist (or, with niches, a niche has none), one extension call
   computes the solos of the best imputed ones by group gain.
   The final K re-score batches the finalists' solos with the finalists.
2. Behavioural diversity of the finalists (the eight plans sent to exact SUMO
   in the live label-list8 configuration) and of the top-k shortlist. The
   descriptor of a plan is built from the per-member deviation = ISSUED plan
   command (after the feasibility clamp) minus the ISSUED policy command (the
   WM policy forecast under the same clamp), never the raw perturbation (the
   clamp distorts it): per member [participates (>= coop_active_min_s seconds
   with |dev| > coop_active_accel), onset s (first such second), mean over the
   first ``coop_early_s`` s, mean over the plan, mean |dev|], members in group
   order (role is fixed inside one search). Who participates, onset
   timing/order and magnitude spread vary even when every member is slower
   than the policy. Features are z-normalised over the
   searched pool (std 0 -> 1). Selection keeps the top-2 by score (with
   niches > 1: the best of each niche first, the niche comparison's rule),
   then greedy max-min (farthest-point) among the measured candidates at or
   above their median score, then fills by score. No behavioural categories
   are hand-set.
3. Novelty vs the verified bank (``coop_novelty``; ``coop_novelty_bank_sha256``
   references a content-addressed COOP_BANK_<sha>.json resolved on the GPU host,
    never embedded in search_config; tests may pass ``coop_novelty_bank``):
    a role summary descriptor (ramp lane 2 / mainline: participation share,
    earliest onset, mean early/plan deviation, min/max plan deviation, mean |dev|,
    plus ramp-minus-mainline onset order; 15 numbers, comparable across group
    sizes; bank side from SUMO issued vs policy commands) is compared with the bank's descriptors; the k-NN mean distance
    (features standardised over bank+candidates) is z-scored over the call's
    candidates and added to the score scaled by ``coop_novelty_weight`` x the
    std of the call's scores, so plans near already-verified kinds are
    penalised and far ones favoured, in the score's own unit.
 3b. Preflight: ``python -m tools.coopmwm.coop_search preflight --sha <sha>`` (GPU host,
    ``COOPMWM_COOP_BANK_DIR`` exported) resolves and digest-checks the bank, exiting
    non-zero on failure; the collector supervisor runs the same check at startup when
    coop_novelty is on, failing loudly before admitting any work.
4. Report per search: mean pairwise descriptor distance of the selected plans,
   number of distinct sign patterns (exact sign of each participant's plan mean)
   and of participation/onset-order patterns,
   WM-predicted group/best-solo/synergy of each.
"""
import argparse
import hashlib
import json
import math
import sys
import numpy as np

from .map_descriptor import FRAME

# Observable convention (speed2_transfers, coopfree.selection): the frame's merge lane
# (ramp on the merge map, closing lane on a lane drop; N1 map descriptor).
RAMP_LANE = FRAME.merge_lane
COOP_REGISTERED = dict(coop_objective=False, coop_lambda=1.0, coop_solo_cap=16, coop_diversity=True,
                       coop_novelty=False, coop_novelty_k=3, coop_novelty_weight=.5,
                       coop_novelty_bank=None, coop_novelty_bank_sha256=None, coop_early_s=10,
                       coop_keep_top=2, coop_active_accel=.3, coop_active_min_s=5)
# Content-addressed bank files COOP_BANK_<sha256>.json ({descriptors: [...]}) are looked up
# here (plus $COOPMWM_COOP_BANK_DIR, ':'/';'-separated); search_config carries the digest only.
BANK_DIRS = []


def bank_digest(descriptors):
    return _digest([[float(x) for x in row] for row in descriptors])


def bank_file_name(sha):
    return f'COOP_BANK_{sha}.json'


def resolve_bank(sha, dirs=None):
    """Load the bank descriptors whose digest is ``sha``; the content digest is re-verified."""
    import os
    from pathlib import Path
    roots = list(dirs or [])+list(BANK_DIRS)
    env = os.environ.get('COOPMWM_COOP_BANK_DIR')
    if env:
        roots += [r for r in env.replace(';', os.pathsep).split(os.pathsep) if r]
    for root in roots:
        f = Path(root)/bank_file_name(sha)
        if f.exists():
            desc = json.loads(f.read_text())['descriptors']
            if bank_digest(desc) != sha:
                raise ValueError(f'coop bank file {f} does not match its digest')
            return desc
    raise ValueError(f'coop novelty bank {sha} not found in {roots}')


def preflight(sha, dirs=None):
    """GPU-host startup check (CLI ``preflight --sha`` and the collector supervisor's startup):
    resolve and digest-check the novelty bank BEFORE budget is reserved or work is admitted.
    Returns the descriptors; raises when the bank file is missing or does not match ``sha``."""
    return resolve_bank(sha, dirs)


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def split_config(cfg):
    """Pop every ``coop_*`` key from ``cfg`` (in place). Returns the merged coop config, or
    None when the option is off (then cfg carries no coop key: byte-identical output)."""
    raw = {k: cfg.pop(k) for k in [k for k in cfg if k.startswith('coop_')]}
    unknown = set(raw)-set(COOP_REGISTERED)
    if unknown:
        raise ValueError(f'unregistered coop config keys {sorted(unknown)}')
    if not raw.get('coop_objective'):
        return None
    coop = dict(COOP_REGISTERED, **raw)
    if not coop['coop_lambda'] >= 0 or not math.isfinite(coop['coop_lambda']):
        raise ValueError('coop_lambda must be finite and >= 0')
    cap = coop['coop_solo_cap']
    if cap is not None and (int(cap) != cap or cap < 1):
        raise ValueError('coop_solo_cap must be a positive integer or None (no cap)')
    if int(coop['coop_novelty_k']) < 1 or int(coop['coop_keep_top']) < 0 or int(coop['coop_early_s']) < 1:
        raise ValueError('coop_novelty_k>=1, coop_keep_top>=0, coop_early_s>=1 required')
    if not coop['coop_active_accel'] >= 0 or int(coop['coop_active_min_s']) < 1:
        raise ValueError('coop_active_accel>=0 and coop_active_min_s>=1 required')
    bank, sha = coop['coop_novelty_bank'], coop['coop_novelty_bank_sha256']
    if bank is None and sha is not None and coop['coop_novelty']:
        bank = coop['coop_novelty_bank'] = resolve_bank(sha)   # never touched with novelty off
    if bank is not None and sha is not None and bank_digest(bank) != sha:
        raise ValueError('coop_novelty_bank does not match coop_novelty_bank_sha256')
    if bank is not None:
        b = np.asarray(bank, float)
        if b.ndim != 2 or (b.size and b.shape[1] != ROLE_DIM) or not np.isfinite(b).all():
            raise ValueError(f'coop_novelty_bank must be a list of {ROLE_DIM}-number role descriptors')
    # Recorded in the search config by size and digest only (the bank can be long).
    cfg.update({k: v for k, v in coop.items() if k not in ('coop_novelty_bank', 'coop_novelty_bank_sha256')})
    cfg['coop_novelty_bank'] = None if bank is None else dict(n=len(bank), sha256=bank_digest(bank))
    return coop


# --------------------------------------------------------------------------- solo variants

def solo_plan(plan, member):
    """Only ``member`` executes its row; every other CAV keeps 'policy continues'.

    Minimal action dict for the WM evaluator (no canonical perturbation, so the
    proposer RecordingEvaluator never logs it as a searched candidate);
    ``solo_of`` marks it for the K8 capture filters."""
    i = plan['members'].index(member)
    return dict(id=f"{plan['id']}__solo_{member}", members=[member], duration_s=int(plan['duration_s']),
                profile=[list(plan['profile'][i])], knot_times_s=list(plan['knot_times_s']),
                origin=plan['origin'], regime=plan['regime'], family=plan['family'],
                group_id=plan.get('group_id'), solo_of=plan['id'], solo_member=member)


def solos_for(plans):
    return [solo_plan(p, v) for p in plans for v in p['members']]


def _mean(scores, a):
    return float(np.mean([x['team_delay'] if isinstance(x, dict) else x for x in scores[a]]))


def _capped(plans, gain, cap):
    if cap is None or cap >= len(plans):
        return list(plans)
    return sorted(plans, key=lambda p: (-gain[p['id']], p['id']))[:int(cap)]


# --------------------------------------------------------------------------- descriptors

def issued_policy(P, duration, bounds, feasibility=None, members=None):
    """The policy's own issued command over [0, D): the WM policy forecast clipped to the plant
    bounds and, when resolved, the same actuator ceiling/floor clamp the plan goes through
    (``mppi_search.build_plan``/``damping.direct_target``)."""
    c = np.clip(np.asarray(P, float)[:, :duration], bounds[0], bounds[1])
    if feasibility is not None and (members is None or list(feasibility['members']) == list(members)):
        c = np.clip(c, feasibility['lo'][:, :duration], feasibility['hi'][:, :duration])
    return c


def member_features(dev, coop):
    """Per member (rows of ``dev`` = issued plan command - issued policy command, per second):
    [participates, onset_s, early mean, whole-plan mean, mean |dev|].

    active second: |dev| > coop_active_accel; participates: >= coop_active_min_s active
    seconds; onset: first active second (plan duration if none). Who participates, onset
    timing/order and magnitude spread carry the variety even when every sign is 'slower'."""
    dev = np.asarray(dev, float); D = dev.shape[1]
    active = np.abs(dev) > float(coop['coop_active_accel'])
    part = active.sum(1) >= int(coop['coop_active_min_s'])
    onset = np.where(active.any(1), active.argmax(1), D).astype(float)
    early = dev[:, :min(int(coop['coop_early_s']), D)].mean(1)
    return np.stack([part.astype(float), onset, early, dev.mean(1), np.abs(dev).mean(1)], 1)


FEATURES = ('participates', 'onset_s', 'early_mean', 'plan_mean', 'mean_abs')
ROLE_DIM = 15


def role_summary(F, ramp, max_onset=300.):
    """Group-size-free descriptor for bank novelty: per role (ramp lane 2, mainline):
    participation share, earliest onset of a participant, mean early/plan deviation,
    min/max plan deviation, mean |dev| (7 each; absent role -> 0 share, onset max_onset, 0);
    plus onset order = earliest ramp onset - earliest mainline onset (0 unless both act)."""
    F = np.asarray(F, float); ramp = np.asarray(ramp, bool); out, first = [], []
    for mask in (ramp, ~ramp):
        x = F[mask]
        if not len(x):
            out += [0., max_onset, 0., 0., 0., 0., 0.]; first.append(None); continue
        on = x[x[:, 0] > 0, 1]
        first.append(float(on.min()) if len(on) else None)
        out += [float(x[:, 0].mean()), first[-1] if first[-1] is not None else max_onset,
                float(x[:, 2].mean()), float(x[:, 3].mean()), float(x[:, 3].min()), float(x[:, 3].max()),
                float(x[:, 4].mean())]
    out.append(first[0]-first[1] if None not in first else 0.)
    return np.array(out)


class Context:
    """Per-search state: group roles, issued policy commands, descriptor cache, coop records."""

    def __init__(self, group, coop, policy=None, bounds=None, feasibility=None):
        self.coop, self.members = coop, list(group['members'])
        lanes = {r['vid']: int(r['lane']) for r in group.get('rows', [])}
        self.ramp = np.array([lanes.get(v) == RAMP_LANE for v in self.members], bool)
        self.policy, self.bounds, self.feasibility = policy, bounds, feasibility
        self.info, self._desc, self._role = {}, {}, {}
        self.bank = None if coop['coop_novelty_bank'] is None else np.asarray(coop['coop_novelty_bank'], float)
        self.solo_rollouts = dict(screen=0, final=0)
        self.extra_calls, self.extended, self.stage_of = 0, 0, {}

    def deviation(self, plan):
        """Issued (post-clamp) plan command minus the issued policy command, members x D."""
        if self.policy is None:   # no centre supplied: fall back to the stored perturbation
            return np.asarray(plan['perturbation_profile'], float)
        D = int(plan['duration_s'])
        return np.asarray(plan['profile'], float)-issued_policy(self.policy, D, self.bounds,
                                                                 self.feasibility, plan['members'])

    def features(self, plan):
        if plan['id'] not in self._desc:
            self._desc[plan['id']] = member_features(self.deviation(plan), self.coop)
        return self._desc[plan['id']]

    def descriptor(self, plan):
        """Per-member features (members in group order, see ``member_features``), flattened."""
        return self.features(plan).reshape(-1)

    def role_descriptor(self, plan):
        if plan['id'] not in self._role:
            self._role[plan['id']] = role_summary(self.features(plan), self.ramp)
        return self._role[plan['id']]


def plan_role_descriptor(plan, lanes, policy_commands=None, coop=None):
    """Role descriptor of a stored (e.g. SUMO-verified bank) plan for ``coop_novelty_bank``.

    Preferred input: ``plan['issued']`` / ``policy_commands`` = the commands actually issued
    under the plan and what the policy outputs at that same state (members x seconds, e.g.
    issued_safe and mappo_safe from the SUMO branch observable). Fallback: plan ``profile``
    minus ``policy_commands``; last resort ``perturbation_profile`` (distorted by the clamp)."""
    coop = dict(COOP_REGISTERED, **(coop or {}))
    cmd = plan.get('issued', plan.get('profile'))
    if cmd is not None and policy_commands is not None:
        cmd = np.asarray(cmd, float); pol = np.asarray(policy_commands, float)
        T = min(cmd.shape[1], pol.shape[1]); dev = cmd[:, :T]-pol[:, :T]
    else:
        dev = np.asarray(plan['perturbation_profile'], float)
    ramp = [int(lanes[v]) == RAMP_LANE for v in plan['members']]
    return role_summary(member_features(dev, coop), ramp)


def _zscore_cols(X, ref=None):
    ref = X if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd == 0] = 1.
    return (X-mu)/sd


def novelty_distance(cands, bank, k):
    """Mean distance to the k nearest bank descriptors; features standardised over bank+cands."""
    if bank is None or not len(bank):
        return np.zeros(len(cands))
    both = np.vstack([cands, bank])
    Z = _zscore_cols(both); C, B = Z[:len(cands)], Z[len(cands):]
    D = np.sqrt(((C[:, None, :]-B[None, :, :])**2).sum(-1))
    kk = min(int(k), D.shape[1])
    return np.sort(D, axis=1)[:, :kk].mean(1)


# --------------------------------------------------------------------------- scoring

def score_call(scores, plans, solo_scored, ctx, stage):
    """Coop score (higher better) of ``plans`` from WM ``scores`` of one or two calls.

    ``solo_scored``: {plan id: scores dict holding its solos and 'policy'} for capped plans.
    Returns the array of scores in ``plans`` order and records per-candidate info."""
    coop = ctx.coop; lam = float(coop['coop_lambda'])
    pol = _mean(scores, 'policy')
    g = np.array([pol-_mean(scores, p['id']) for p in plans])
    syn = np.full(len(plans), np.nan); best = [None]*len(plans); solo = [None]*len(plans)
    for i, p in enumerate(plans):
        s = solo_scored.get(p['id'])
        if s is None:
            continue
        sp = _mean(s, 'policy')
        gains = {v: sp-_mean(s, f"{p['id']}__solo_{v}") for v in p['members']}
        solo[i] = gains; best[i] = max(gains.values()); syn[i] = g[i]-best[i]
    computed = ~np.isnan(syn)
    # Pessimistic: an uncomputed candidate gets the LOWEST synergy computed in this call
    # (never credited with cooperation it was not measured to have).
    imputed = float(np.min(syn[computed])) if computed.any() else 0.
    syn_used = np.where(computed, syn, imputed)
    score = g+lam*syn_used
    nov = np.zeros(len(plans))
    if coop['coop_novelty'] and ctx.bank is not None and len(ctx.bank):
        d = novelty_distance(np.stack([ctx.role_descriptor(p) for p in plans]), ctx.bank, coop['coop_novelty_k'])
        if d.std() > 0 and score.std() > 0:
            nov = float(coop['coop_novelty_weight'])*float(score.std())*(d-d.mean())/d.std()
    total = score+nov
    for i, p in enumerate(plans):
        ctx.info.setdefault(stage, {})[p['id']] = dict(
            group_gain=float(g[i]), best_solo_gain=best[i], solo_gains=solo[i],
            synergy=float(syn_used[i]), synergy_imputed=not bool(computed[i]),
            novelty=float(nov[i]), score=float(total[i]))
        if stage != 'final':
            ctx.stage_of[p['id']] = stage
    return total


def extend_solos(record, evaluate, base, pool, niche_of, n, K, H, ctx):
    """Finalist eligibility needs MEASURED solos. If fewer than ``n`` finalist-ELIGIBLE screened
    candidates have computed solos (or, with niches, a niche has none), ONE extra screen-K/H call
    computes the solos of the best imputed candidates by group gain (each uncovered niche's best
    first). Eligibility follows ``_coop_finalists``: measured AND not a clipped policy duplicate.
    Updates ``ctx.info``/``pool`` in place; returns the extra action-futures."""
    info = {pid: ctx.info[st][pid] for pid, st in ctx.stage_of.items()}
    eligible = {pid for pid in pool
                if not info[pid]['synergy_imputed'] and not pool[pid][1].get('policy_duplicate')}
    imputed = [pid for pid in pool if info[pid]['synergy_imputed'] and not pool[pid][1].get('policy_duplicate')]
    if not imputed:
        return 0
    by_gain = sorted(imputed, key=lambda i: (-info[i]['group_gain'], i))
    need = []
    if niche_of is not None:
        covered = {niche_of[pid] for pid in eligible}
        for k in sorted({niche_of[i] for i in imputed}-covered):
            need.append(next(i for i in by_gain if niche_of[i] == k))
    computed = len(eligible)
    for pid in by_gain:
        if computed+len(need) >= n:
            break
        if pid not in need:
            need.append(pid)
    if not need:
        return 0
    plans = [pool[pid][1] for pid in need]
    solos = solos_for(plans)
    s = evaluate(record, base+solos, K, H)
    ctx.solo_rollouts['extension'] = ctx.solo_rollouts.get('extension', 0)+len(solos); ctx.extra_calls += 1
    lam = float(ctx.coop['coop_lambda']); sp = _mean(s, 'policy')
    for p in plans:
        x = info[p['id']]
        gains = {v: sp-_mean(s, f"{p['id']}__solo_{v}") for v in p['members']}
        syn = x['group_gain']-max(gains.values())
        score = x['group_gain']+lam*syn+x['novelty']
        x.update(best_solo_gain=max(gains.values()), solo_gains=gains, synergy=float(syn),
                 synergy_imputed=False, solo_extension=True, score=float(score))
        pool[p['id']] = (-float(score), p)
    ctx.extended = len(plans)
    return (len(base)+len(solos))*K


def measured(pool, ctx):
    """Pool ids whose screen solos were actually computed (finalist-eligible)."""
    return [pid for pid in pool if not ctx.info[ctx.stage_of[pid]][pid]['synergy_imputed']]


def screen(record, evaluate, base, cands, scores, K, H, ctx, stage):
    """Coop scores for one screen call. If the solos were batched into ``scores`` (cap covers
    the call) they are used directly; otherwise one extra call scores the capped solos.
    Returns (rel cost array, lower better, for the MPPI/CEM update; extra futures)."""
    cap = ctx.coop['coop_solo_cap']; extra = 0
    if cap is None or cap >= len(cands):
        solo_scored = {p['id']: scores for p in cands}
        ctx.solo_rollouts['screen'] += sum(len(p['members']) for p in cands)
    else:
        pol = _mean(scores, 'policy')
        gain = {p['id']: pol-_mean(scores, p['id']) for p in cands}
        top = _capped(cands, gain, cap)
        solos = solos_for(top)
        s = evaluate(record, base+solos, K, H)
        extra = (len(base)+len(solos))*K
        ctx.solo_rollouts['screen'] += len(solos); ctx.extra_calls += 1
        solo_scored = {p['id']: s for p in top}
    return -score_call(scores, cands, solo_scored, ctx, stage), extra


def batched_solos(cands, ctx):
    """Solos to batch INTO a screen call (only when the cap covers the whole call)."""
    cap = ctx.coop['coop_solo_cap']
    return solos_for(cands) if cap is None or cap >= len(cands) else []


def final_solos(finalists, rel_of, ctx):
    """Final-stage solos batched with the finalists; capped to the top-N finalists by screen score."""
    gain = {p['id']: -rel_of[p['id']] for p in finalists}
    return solos_for(_capped(finalists, gain, ctx.coop['coop_solo_cap']))


def final_scores(scores, finalists, solos, ctx):
    with_solo = {s['solo_of'] for s in solos}
    ctx.solo_rollouts['final'] += len(solos)
    return dict(zip([p['id'] for p in finalists],
                    score_call(scores, finalists, {i: scores for i in with_solo}, ctx, 'final')))


# --------------------------------------------------------------------------- diversity

def select_diverse(items, n, ctx, keep_top=None, niche_of=None):
    """items: (id, score higher-better, plan). Seeds: the top ``keep_top`` by score, or with
    ``niche_of`` the best of each niche (best-score order, the niche comparison's rule); then
    greedy max-min over normalised descriptors among items with score >= median, then fill
    by score."""
    order = sorted(items, key=lambda t: (-t[1], t[0]))
    if n <= 0:
        return []
    if niche_of is not None:
        seeds, seen = [], set()
        for t in order:
            if niche_of[t[0]] not in seen:
                seen.add(niche_of[t[0]]); seeds.append(t)
        seeds = seeds[:n]
    else:
        keep = int(ctx.coop['coop_keep_top'] if keep_top is None else keep_top)
        seeds = order[:min(keep, n)]
    # Descriptors of every eligible item are computed in both modes, so the report's
    # pool normalisation is identical with and without diversity.
    X = _zscore_cols(np.stack([ctx.descriptor(t[2]) for t in order])) if order else None
    if not ctx.coop['coop_diversity']:
        taken = {t[0] for t in seeds}
        return [t[0] for t in seeds]+[t[0] for t in order if t[0] not in taken][:n-len(seeds)]
    idx = {t[0]: i for i, t in enumerate(order)}
    chosen = list(seeds)
    med = float(np.median([t[1] for t in order])) if order else 0.
    rest = [t for t in order if t[1] >= med and t[0] not in {c[0] for c in chosen}]
    while len(chosen) < n and rest:
        if chosen:
            S = X[[idx[c[0]] for c in chosen]]
            def key(t):
                return (-float(np.sqrt(((S-X[idx[t[0]]])**2).sum(1)).min()), -t[1], t[0])
        else:
            def key(t):
                return (-t[1], t[0])
        t = min(rest, key=key)
        chosen.append(t); rest.remove(t)
    taken = {c[0] for c in chosen}
    chosen += [t for t in order if t[0] not in taken][:n-len(chosen)]
    return [t[0] for t in chosen]


def report(plans, ctx, stage):
    """Item 4: spread of the selected plans in descriptor space plus WM-predicted coop numbers."""
    if not plans:
        return dict(n=0, mean_pairwise_descriptor_distance=None, distinct_sign_patterns=0, plans=[])
    D = np.stack([ctx.descriptor(p) for p in plans])
    pool = np.stack([v.reshape(-1) for v in ctx._desc.values()])
    Z = _zscore_cols(D, pool)
    pairs = [float(np.sqrt(((Z[i]-Z[j])**2).sum())) for i in range(len(plans)) for j in range(i+1, len(plans))]
    Fs = [ctx.features(p) for p in plans]
    # Exact sign of each member's plan-mean deviation, 0 for a non-participant.
    signs = {tuple(int(np.sign(f[3])) if f[0] else 0 for f in F) for F in Fs}
    # Participation mask + onset order of the participants (ties share a rank).
    orders = {tuple(int(np.searchsorted(np.unique(F[F[:, 0] > 0, 1]), f[1])) if f[0] else -1 for f in F)
              for F in Fs}
    info = ctx.info.get(stage, {})
    return dict(n=len(plans), mean_pairwise_descriptor_distance=float(np.mean(pairs)) if pairs else 0.,
                distinct_sign_patterns=len(signs), distinct_participation_onset_orders=len(orders),
                descriptor='per member [' + ', '.join(FEATURES) + '] of issued plan command minus issued '
                           'policy command (post-clamp); z-normalised over the searched pool; sign pattern = '
                           'exact sign of plan mean per participant (0 = not participating)',
                plans=[dict(id=p['id'], descriptor=[float(x) for x in ctx.descriptor(p)],
                            **{k: info.get(p['id'], {}).get(k) for k in
                               ('group_gain', 'best_solo_gain', 'synergy', 'synergy_imputed', 'novelty', 'score')})
                       for p in plans])


# --------------------------------------------------------------------------- cost

def cost_multiplier(members, config=None):
    """WM compute with coop_objective on / off, counted as action x future (and x horizon s)."""
    from . import mppi_search as ms
    cfg = dict(ms.REGISTERED, **{k: v for k, v in (config or {}).items() if not k.startswith('coop_')})
    coop = dict(COOP_REGISTERED, **{k: v for k, v in (config or {}).items() if k.startswith('coop_')})
    durations = ms.allowed_durations(cfg['durations_s'], cfg['max_duration_s'])
    sc = ms.scaling(members, ms.canonical_knots(durations[-1]), cfg['top_k'], cfg)
    N, F, it = sc['population'], sc['finalists'], cfg['iterations']
    Ks, Kf = cfg['screen_K'], cfg['final_K']
    Hs = int(min(cfg['final_H_max'], max(cfg['screen_H_min'], durations[-1])))
    Hf = ms.final_horizon(durations[-1], cfg)
    cap = coop['coop_solo_cap']
    if cap is None or cap >= N:
        screen_extra = N*members*Ks
    else:
        screen_extra = (int(cap)*members+2)*Ks
    final_extra = min(F, F if cap is None else int(cap))*members*Kf
    base_f = it*(N+2)*Ks+(F+2)*Kf
    coop_f = base_f+it*screen_extra+final_extra
    base_s = it*(N+2)*Ks*Hs+(F+2)*Kf*Hf
    coop_s = base_s+it*screen_extra*Hs+final_extra*Hf
    return dict(members=members, population=N, finalists=F, iterations=it, solo_cap=cap,
                base_action_futures=base_f, coop_action_futures=coop_f,
                multiplier_action_futures=coop_f/base_f, multiplier_action_future_seconds=coop_s/base_s,
                screen_H=Hs, final_H_worst=Hf)


# --------------------------------------------------------------------------- CLI

def _cli():
    p = argparse.ArgumentParser(prog='python -m tools.coopmwm.coop_search')
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cost', help='GPU cost multiplier for group sizes 2..6')
    c.add_argument('--config', default='{"iterations":2,"screen_K":2,"top_k":8,"finalists_per_k":1}')
    c.add_argument('--solo-cap', type=int, default=16)
    e = sub.add_parser('enable', help='write the additive collector config LS_COOP_SEARCH_CONFIG.json')
    e.add_argument('collection'); e.add_argument('--lambda', dest='lam', type=float, default=1.)
    e.add_argument('--solo-cap', type=int, default=16); e.add_argument('--no-diversity', action='store_true')
    e.add_argument('--alternate', action='store_true', help='coop on every other admission pair (2x2 with niches)')
    e.add_argument('--bank-descriptors', help='JSON from bank-descriptors; enables novelty')
    e.add_argument('--novelty-k', type=int, default=3); e.add_argument('--novelty-weight', type=float, default=.5)
    b = sub.add_parser('bank-descriptors', help='role descriptors of verified bank plans')
    b.add_argument('plans', help='JSON list of {plan:{members, issued|profile|perturbation_profile}, '
                                 'lanes:{vid:lane}, policy_commands} (issued/policy = SUMO issued_safe/mappo_safe)')
    b.add_argument('--out', required=True)
    pf = sub.add_parser('preflight', help='resolve and digest-check the novelty bank on THIS host '
                                          '(exit 1 on failure; run at supervisor startup, before admissions)')
    pf.add_argument('--sha', required=True, help='coop_novelty_bank_sha256 of the bank')
    a = p.parse_args()
    from pathlib import Path
    if a.cmd == 'cost':
        cfg = dict(json.loads(a.config), coop_solo_cap=a.solo_cap)
        print(json.dumps([cost_multiplier(m, cfg) for m in range(2, 7)], indent=1))
    elif a.cmd == 'preflight':
        try:
            desc = preflight(a.sha)
        except (ValueError, OSError) as error:   # missing file, digest mismatch, corrupt JSON
            print(json.dumps(dict(ok=False, sha=a.sha, bank=bank_file_name(a.sha), error=str(error))),
                  file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps(dict(ok=True, sha=a.sha, bank=bank_file_name(a.sha), n=len(desc))))
    elif a.cmd == 'bank-descriptors':
        rows = json.loads(Path(a.plans).read_text())
        out = [[float(x) for x in plan_role_descriptor(r['plan'], r['lanes'], r.get('policy_commands'))] for r in rows]
        fallback = sum(r.get('policy_commands') is None for r in rows)
        with Path(a.out).open('x') as f:
            json.dump(dict(descriptors=out, n=len(out), source=str(a.plans), features=ROLE_DIM,
                           perturbation_fallback=fallback), f)
        print(json.dumps(dict(n=len(out), out=a.out, perturbation_fallback=fallback)))
    else:
        coop = dict(coop_objective=True, coop_lambda=a.lam, coop_solo_cap=a.solo_cap,
                    coop_diversity=not a.no_diversity)
        bank_note = None
        if a.bank_descriptors:
            desc = json.loads(Path(a.bank_descriptors).read_text())['descriptors']
            sha = bank_digest(desc)
            bank = Path(a.collection)/bank_file_name(sha)
            if not bank.exists():
                with bank.open('x') as f:
                    json.dump(dict(descriptors=desc, sha256=sha, source=str(a.bank_descriptors)), f)
            # search_config carries the digest only; GPU hosts resolve it from BANK_DIRS or
            # $COOPMWM_COOP_BANK_DIR (copy this file there before admissions start).
            coop.update(coop_novelty=True, coop_novelty_k=a.novelty_k, coop_novelty_weight=a.novelty_weight,
                        coop_novelty_bank_sha256=sha)
            bank_note = dict(bank_file=str(bank), sha256=sha, n=len(desc))
            BANK_DIRS.append(str(Path(a.collection)))
        probe = dict(coop); split_config(probe)        # validate (and resolve the bank) before writing
        target = Path(a.collection)/'LS_COOP_SEARCH_CONFIG.json'
        with target.open('x') as f:
            json.dump(dict(coop_search=coop, coop_alternation=bool(a.alternate)), f)
        print(json.dumps(dict(written=str(target), config=probe, bank=bank_note)))


if __name__ == '__main__':
    _cli()
