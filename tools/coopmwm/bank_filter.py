"""STATISTICAL BANK FILTER (option ``bank_filter='statistical'``, DEFAULT OFF).

B-REPLAY admitted every confirmed (replication, no-later-harm, safety) exact-positive
bank entry for imitation and tagged it cooperative iff the recorded first-seed
cooperation gain was > 0. Near-zero advantages (e.g. +0.0005 veh-s) and ~zero-gain
synergy labels therefore reach the imitation batch and dilute it.

This module is a READER-SIDE statistical rule, applied only where ``replay_bank.bank_pairs``
is called with ``bank_filter='statistical'`` (binding key ``bank_filter``; absent/None/False
keeps every existing code path byte-identical). It never writes bank artifacts and never
re-runs physics; it only re-reads the sealed receipts and the recorded branch traces:

1. IMITATION ADMISSION: an admitted entry stays in the imitation batch only if its
   replicated advantage is significantly > 0:
   - aggregate gate: every recorded replicate advantage (first seed, confirmation
     replication) must be > 0 - this is the existing admission rule, re-checked;
   - significance (paired bootstrap over vehicles across the replicate seeds): for each
     replicate seed the paired per-vehicle road speed-deficit differences of the two
     branches (policy continues vs chosen plan) are reconstructed from the recorded
     observable traces over the replicate's window; the per-vehicle values of all
     replicates are POOLED and the one-sided 95% bootstrap lower bound of the mean
     (10000 resamples, 5th percentile, numpy default_rng(20261101)) must be > 0;
   - aggregate fallback (only when NO replicate has a readable trace): the same
     one-sided bootstrap bound over the >= 2 recorded replicate aggregates must be > 0.
     No hand-set behavioural thresholds are used anywhere - only these statistical rules.
   The per-vehicle quantity is the plant's own registered speed-deficit integrand
   (traffic_shepherd.c4.env SPEED_LIMIT_MPS; env.step accumulates 0.5 * sum(max(0, 1 -
   v/SPEED_LIMIT_MPS)) per half-second substep, so one tick contributes
   max(0, 1 - v/SPEED_LIMIT_MPS) veh-s per vehicle at its start-of-tick observable speed).
   It reconstructs the ROAD component of team_delay_vs; the source-wait component is a
   source-side aggregate with no per-vehicle observable, which is why the bound is used
   as a significance test alongside the recorded aggregate gate, not as a replacement.

2. COOPERATIVE LABEL: cooperative only if the synergy is significantly > 0 by the same
   method:
   - primary: paired per-vehicle deficits of the first-seed GROUP branch vs the recorded
     BEST SOLO branch (same seed, same window); one-sided 95% lower bound > 0;
   - fallback (no usable trace pair): the recorded aggregate gain (group minus best solo)
     must be > 0 - the existing rule;
   - group-minus-sum-of-solos (available when every solo branch completed): the recorded
     aggregate C_G = group advantage - sum(solo advantages) (the registered credit
   convention, tools/coopfree/credit.py) must additionally be > 0. No physical
   sum-of-solos branch exists, so this contrast has no per-vehicle bootstrap; the
   aggregate requirement is the only form in which it is observable (disclosed).

3. REPORTING: every kept/dropped entry is reported with its reason and bounds in the
   bank-pairs summary under ``bank_filter``; the 50% cooperative oversampling applies to
   the FILTERED cooperative set (weights unchanged, pair duplication only, as before).
"""
from pathlib import Path

from . import exact_grade as xg

RULE = 'statistical'
N_BOOT = 10000
SEED = xg.BOOT_SEED          # 20261101, the registered bootstrap RNG family
LB_QUANTILE = 0.05           # one-sided 95% lower bound = 5th percentile of bootstrap means


def one_sided_lb95(values, seed=SEED, n_boot=N_BOOT):
    """One-sided 95% bootstrap lower bound of the mean: the 5th percentile of 10000
    bootstrap resample means (numpy default_rng(seed)); None when fewer than 2 values."""
    import numpy as np
    v = [float(x) for x in values]
    if len(v) < 2:
        return None
    rng = np.random.default_rng(int(seed))
    boots = np.asarray(v)[rng.integers(0, len(v), (int(n_boot), len(v)))].mean(1)
    return float(np.quantile(boots, LB_QUANTILE))


def speed_limit_mps():
    """The plant's registered deficit speed limit (traffic_shepherd.c4.env SPEED_LIMIT_MPS)."""
    from traffic_shepherd.c4.env import SPEED_LIMIT_MPS
    return float(SPEED_LIMIT_MPS)


def deficit_by_vehicle(trace_path, start_s, end_s, limit_mps=None):
    """{vid: speed-deficit veh-s} over the ticks of [start_s, end_s) of one branch's
    observable.jsonl.gz trace (max(0, 1 - v_mps/SPEED_LIMIT_MPS) per recorded second and
    vehicle, the plant's own road-delay integrand at the start-of-tick speed).
    None when the trace is missing/unreadable (the rule then degrades to the aggregate
    fallback; it never imputes)."""
    limit = speed_limit_mps() if limit_mps is None else float(limit_mps)
    out = {}
    try:
        import gzip
        import json
        with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
            for line in f:
                x = json.loads(line)
                if not float(start_s) <= float(x['time_s']) < float(end_s):
                    continue
                for v in x.get('vehicles', []):
                    d = max(0., 1.-float(v['v_mps'])/limit)
                    out[v['vid']] = out.get(v['vid'], 0.)+d
    except (OSError, EOFError, ValueError):
        return None
    return out


def paired_differences(trace_a, trace_b, start_s, end_s, limit_mps=None):
    """{vid: deficit_a - deficit_b} over the vehicles BOTH traces record inside the window
    (paired seconds and vehicles; seconds where a vehicle is missing on one side are
    skipped, never imputed). None when either trace is missing."""
    a = deficit_by_vehicle(trace_a, start_s, end_s, limit_mps)
    b = deficit_by_vehicle(trace_b, start_s, end_s, limit_mps)
    if a is None or b is None:
        return None
    return {v: a[v]-b[v] for v in sorted(set(a) & set(b))}


# ---------------------------------------------------------------- decision rules (pure)

def advantage_verdict(replicates, seed=SEED):
    """Imitation-admission verdict for one entry. ``replicates`` is a list of
    ``dict(name=..., aggregate_veh_s=float|None, paired={vid: d}|None)`` (one per
    replicate seed). Rule (documented, no hand-set thresholds):
      1. aggregate gate: every recorded replicate aggregate must be > 0;
      2. significance: pooled one-sided 95% bootstrap lower bound of the per-vehicle
         paired differences over the replicates that have them must be > 0; when no
         replicate has paired vehicle data, the bound over the >= 2 replicate aggregates
         must be > 0 instead.
    Returns dict(keep, reason, bound, n_paired, pooled_mean, aggregates)."""
    aggregates = [float(r['aggregate_veh_s']) for r in replicates if r.get('aggregate_veh_s') is not None]
    if any(a <= 0 for a in aggregates):
        return dict(keep=False, reason='a replicate advantage is not > 0', bound=None,
                    n_paired=0, pooled_mean=None, aggregates=aggregates)
    pooled = []
    for r in replicates:
        if r.get('paired'):
            pooled.extend(float(v) for v in r['paired'].values())
    if pooled:
        bound = one_sided_lb95(pooled, seed)
        mean = sum(pooled)/len(pooled)
        if bound is None:
            return dict(keep=False, reason='paired vehicle data has fewer than 2 paired vehicles', bound=None,
                        n_paired=len(pooled), pooled_mean=mean, aggregates=aggregates)
        return dict(keep=bool(bound > 0),
                    reason=('paired-vehicle bootstrap lower bound > 0' if bound > 0
                            else 'paired-vehicle bootstrap lower bound <= 0'),
                    bound=bound, n_paired=len(pooled), pooled_mean=mean, aggregates=aggregates)
    bound = one_sided_lb95(aggregates, seed)
    if bound is None:
        return dict(keep=False, reason='significance uncomputable: single replicate aggregate, no vehicle data',
                    bound=None, n_paired=0, pooled_mean=None, aggregates=aggregates)
    return dict(keep=bool(bound > 0),
                reason=('replicate-aggregate bootstrap lower bound > 0' if bound > 0
                        else 'replicate-aggregate bootstrap lower bound <= 0'),
                bound=bound, n_paired=0, pooled_mean=(sum(aggregates)/len(aggregates) if aggregates else None),
                aggregates=aggregates)


def cooperation_verdict(paired_best_solo, recorded_gain, complete, sum_synergy, seed=SEED):
    """Cooperative-label verdict for one entry. ``paired_best_solo`` is the {vid: d} map of
    per-vehicle ADVANTAGE differences (solo deficit minus group deficit = the per-vehicle
    form of group-minus-best-solo; positive = the group helps more), None when no trace
    pair is readable; ``recorded_gain`` the recorded aggregate gain (group minus best solo);
    ``complete`` whether every solo branch completed; ``sum_synergy`` the recorded
    aggregate group-minus-sum-of-solos synergy when complete else None. Rule:
      - primary: one-sided 95% bootstrap lower bound of the paired per-vehicle values > 0;
      - fallback (no usable paired data): the recorded aggregate gain must be > 0;
      - where the sum-of-solos synergy is available (complete solos) it must also be > 0.
    Returns dict(cooperative, bound, reason, sum_synergy)."""
    if paired_best_solo:
        values = [float(v) for v in paired_best_solo.values()]
        bound = one_sided_lb95(values, seed)
        if bound is not None:
            primary, reason = bound > 0, ('group-minus-best-solo bootstrap lower bound > 0' if bound > 0
                                          else 'group-minus-best-solo bootstrap lower bound <= 0')
        else:
            primary, reason = bool(recorded_gain is not None and recorded_gain > 0), \
                'fewer than 2 paired vehicles; recorded aggregate gain rule'
    else:
        bound, primary = None, bool(recorded_gain is not None and recorded_gain > 0)
        reason = 'no readable trace pair; recorded aggregate gain rule'
    if primary and complete and sum_synergy is not None and sum_synergy <= 0:
        return dict(cooperative=False, bound=bound, reason=reason+'; sum-of-solos synergy <= 0',
                    sum_synergy=float(sum_synergy))
    return dict(cooperative=bool(primary), bound=bound, reason=reason,
                sum_synergy=(float(sum_synergy) if sum_synergy is not None else None))


# ---------------------------------------------------------------- bank-side glue (read-only)

def _branch_receipt(root, arm, kind, vid, sha_key='receipt_sha256'):
    p = Path(root)/'replay_bank'/arm/kind/'verifications'/f'{vid}.json'
    if not p.exists():
        return None
    import json
    r = json.loads(p.read_text(encoding='utf-8'))
    xg.verify_sealed(r, sha_key)
    return r


def entry_verdict(root, arm, r, c, limit_mps=None):
    """Read-only statistical verdict for one admitted bank entry (receipt ``r``,
    confirmation ``c``): imitation keep/drop with reason and bounds, and the statistical
    cooperative label. Never writes anything; unreadable traces degrade to the documented
    aggregate rules."""
    vid = r['verification_id']
    first_start, first_end = float(r['window']['start_s']), float(r['window']['end_s'])
    branches = Path(r['_branches'])/vid
    first = dict(name='first_seed', aggregate_veh_s=float(r['grade_veh_s']), paired=None)
    pol = r['branches'][0]['branch'];chosen = r['branches'][r['chosen']+1]['branch']
    first['paired'] = paired_differences(branches/pol/'observable.jsonl.gz',
                                         branches/chosen/'observable.jsonl.gz',
                                         first_start, first_end, limit_mps)
    replicates = [first]
    rep = c.get('replication') or {}
    if rep.get('verification_id') and rep.get('window') and c.get('replication_advantage_veh_s') is not None:
        rr = _branch_receipt(root, arm, 'confirm_replication', rep['verification_id'])
        paired = None
        if rr is not None:
            rpol, rchosen = rr['branches'][0]['branch'], rr['branches'][rr['chosen']+1]['branch']
            paired = paired_differences(
                Path(root)/'replay_bank'/arm/'confirm_replication'/'branches'/rep['verification_id']/rpol/'observable.jsonl.gz',
                Path(root)/'replay_bank'/arm/'confirm_replication'/'branches'/rep['verification_id']/rchosen/'observable.jsonl.gz',
                float(rep['window']['start_s']), float(rep['window']['end_s']), limit_mps)
        replicates.append(dict(name='replication', aggregate_veh_s=float(c['replication_advantage_veh_s']),
                               paired=paired))
    adv = advantage_verdict(replicates)
    co = c.get('cooperation') or {}
    gain = co.get('cooperation_gain_veh_s')
    paired_co = None
    if gain is not None and co.get('best_solo_member') is not None:
        best = next((s for s in co.get('solo', []) if s.get('member') == co['best_solo_member']), None)
        if best is not None and best.get('verification_id'):
            sr = _branch_receipt(root, arm, 'confirm_solo', best['verification_id'])
            if sr is not None:
                schosen = sr['branches'][sr['chosen']+1]['branch']
                # gain = advantage(group) - advantage(solo) = deficit(solo) - deficit(group)
                paired_co = paired_differences(
                    Path(root)/'replay_bank'/arm/'confirm_solo'/'branches'/best['verification_id']/schosen/'observable.jsonl.gz',
                    branches/chosen/'observable.jsonl.gz',
                    first_start, min(first_end, float(sr['window']['end_s'])), limit_mps)
    advs = co.get('solo_advantages_veh_s') or []
    complete = bool(co.get('complete')) and all(a is not None for a in advs) and gain is not None
    sum_synergy = None
    if complete and co.get('best_solo_advantage_veh_s') is not None:
        # C_G = group advantage - sum(solo advantages); group advantage = gain + best solo
        sum_synergy = float(gain)+float(co['best_solo_advantage_veh_s'])-sum(float(a) for a in advs)
    coop = cooperation_verdict(paired_co, gain, complete, sum_synergy)
    return dict(verification_id=vid, keep=adv['keep'], reason=adv['reason'], bound=adv['bound'],
                n_paired=adv['n_paired'], pooled_mean=adv['pooled_mean'],
                replicate_aggregates_veh_s=adv['aggregates'], first_seed_traces_readable=first['paired'] is not None,
                cooperation=dict(recorded_cooperative=co.get('cooperative'), cooperative=coop['cooperative'],
                                 bound=coop['bound'], reason=coop['reason'], sum_synergy_veh_s=coop['sum_synergy']))
