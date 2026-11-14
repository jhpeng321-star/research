"""Statistics for the claim-validation experiments (zero SUMO). Convention everywhere:

    diff = treatment - reference          (negative = the treatment has less delay)
    pct  = 100 * (mean treatment - mean reference) / mean reference   (difference of means over the reference mean)

Each function mirrors or cites the registered implementation so numbers stay comparable with the paper tables:

    paired_bootstrap      ops/clean3_finaleval.paired_stats + tools/coopmwm/planner_eval._paired_ci
                          (numpy default_rng(seed).integers resamples of whole worlds, np.quantile percentile CI)
    unc_boot_ci / unc_admit   tools/coopmwm/a37_cellbank.boot_ci / unc_verdict (the admission rule of the original
                          four-plan bank: random.Random(20261106), 10,000 resamples, order-statistic CI)
    cvar90 / paired_cvar90    ops/nm_finaleval.cvar90 / _paired_tail (vehicles present in both runs)
    hard_brake_vs         ops/nm_finaleval.vehicle_log_report accel.hard_brake_vs (logged a < -3.0 m/s^2, 1 Hz rows)
    control_effort        tools/coopmwm/a42_bigbank.realised_sizes + ops/a67_plans_on_60c.activation_table
New here (no registered implementation exists): seed x world two-way bootstrap, exact paired sign-flip permutation
test over the six search seeds, Holm adjustment.

CLI:  stats.py selftest | stats.py contrast --treatment T.json --reference R.json   (JSON = {world: value})
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')   # the registered panel contract
UNC_BOOT, UNC_BOOT_SEED = 10000, 20261106                                            # a37_cellbank constants
GROUP_DELAY_MAX_S = 10.0                                                             # nm_finaleval.GROUP_DELAY_MAX_S
CO_PRIMARY_GROUPS = ('origin:L0', 'origin:L1', 'origin:L2', 'class:hdv', 'class:cav')
HARD_BRAKE_MPS2 = -3.0
TAIL_LEVEL = 0.90


# ------------------------------------------------------------------------------------------------ paired bootstrap
def paired_bootstrap(treatment, reference, bootstrap=None, pct=True):
    """World-level paired percentile bootstrap (whole worlds resampled; the same resamples for diff and pct)."""
    b = dict(BOOTSTRAP, **(bootstrap or {}))
    t, c = np.asarray(treatment, float), np.asarray(reference, float)
    if t.shape != c.shape or t.ndim != 1 or len(t) < 2:
        raise ValueError('paired_bootstrap needs two equal-length vectors of at least 2 worlds')
    alpha = 1-b['level']
    d = t-c
    idx = np.random.default_rng(b['seed']).integers(0, len(d), (b['reps'], len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    p = min(1., 2*min(np.sum(boot <= 0)+1, np.sum(boot >= 0)+1)/(b['reps']+1))
    out = dict(n=len(d), mean_treatment=float(t.mean()), mean_reference=float(c.mean()), diff=float(d.mean()),
               ci95=[float(lo), float(hi)], p_two_sided=float(p), worlds_lower=int(np.sum(d < 0)),
               worlds_higher=int(np.sum(d > 0)), convention='diff = treatment - reference; negative = less delay')
    if pct:
        mt, mc = t[idx].mean(1), c[idx].mean(1)
        if c.mean() != 0 and np.all(mc != 0):
            pb = 100*(mt-mc)/mc
            plo, phi = np.quantile(pb, [alpha/2, 1-alpha/2])
            out.update(pct=float(100*(t.mean()-c.mean())/c.mean()), pct_ci95=[float(plo), float(phi)])
        else:
            out.update(pct=None, pct_ci95=[None, None])
    return out


def paired_by_world(treatment, reference, **kw):
    """{world: value} dicts -> paired_bootstrap on the common, sorted world set (refuses a mismatch)."""
    if set(treatment) != set(reference):
        raise ValueError('treatment and reference cover different worlds')
    ws = sorted(treatment)
    return dict(paired_bootstrap([treatment[w] for w in ws], [reference[w] for w in ws], **kw), worlds=ws)


# ------------------------------------------------------------------------------------------------ admission
def unc_boot_ci(diffs, n=UNC_BOOT, seed=UNC_BOOT_SEED):
    """a37_cellbank.boot_ci verbatim."""
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k))/k for _ in range(n))
    return means[int(.025*n)], means[int(.975*n)-1]


def groups_allowed(group_increase):
    """{group: mean per-vehicle delay increase vs the reference (s), labels from the REFERENCE run}: all five
    registered groups present and each <= +10 s (nm_finaleval.co_primary_groups all_pass)."""
    return all(g in group_increase and group_increase[g] is not None and float(group_increase[g]) <= GROUP_DELAY_MAX_S
               for g in CO_PRIMARY_GROUPS)


def unc_admit(plan_by_world, ref_by_world, group_increase, n_worlds=24):
    """a37_cellbank.unc_verdict: PASS iff the 95% CI upper bound of (plan - reference) total delay < 0 AND every
    one of the five group means <= +10 s. Worlds are paired in sorted order (as the registered rule sorts seeds)."""
    ws = sorted(ref_by_world)
    if sorted(plan_by_world) != ws or len(ws) != n_worlds:
        raise ValueError(f'admission needs the same {n_worlds} paired worlds')
    diffs = [float(plan_by_world[w])-float(ref_by_world[w]) for w in ws]
    lo, hi = unc_boot_ci(diffs)
    allowed = groups_allowed(group_increase or {})
    present = {g: float(group_increase[g]) for g in CO_PRIMARY_GROUPS if (group_increase or {}).get(g) is not None}
    worst = max(present.items(), key=lambda x: x[1], default=(None, None))
    return dict(n=len(ws), mean=sum(diffs)/len(diffs), ci95=[lo, hi], allowed=allowed, worst_group=worst[0],
                worst_group_inc=worst[1], PASS=bool(hi < 0 and allowed), lb95_gain_veh_s=-hi,
                benefit=sum(d < 0 for d in diffs)/len(diffs),
                rule='(plan - U0) total delay CI95 upper < 0 AND each of the five group means <= +10 s (U0 labels)')


# ------------------------------------------------------------------------------------------------ two-way bootstrap
def two_way_bootstrap(diff_matrix, reps=10000, seed=20261104, level=.95, crossed=True):
    """Seed x world bootstrap of the mean of ``diff_matrix[s][w]`` (rows = search seeds, columns = worlds; each
    cell = WM bank - Random bank on that world). crossed=True: every seed's banks ran on the SAME worlds, so seeds
    and worlds are resampled independently and crossed (rectangular matrix required). crossed=False: worlds are
    nested in seeds (each seed has its own worlds), seeds are resampled and then worlds within each drawn seed."""
    rows = [np.asarray(r, float) for r in diff_matrix]
    rng = np.random.default_rng(seed)
    S = len(rows)
    if S < 2 or any(len(r) < 2 for r in rows):
        raise ValueError('two_way_bootstrap needs >= 2 seeds and >= 2 worlds')
    if crossed:
        if len({len(r) for r in rows}) != 1:
            raise ValueError('crossed design needs the same worlds for every seed')
        M = np.vstack(rows)
        W = M.shape[1]
        boot = np.empty(reps)
        for i in range(reps):
            boot[i] = M[np.ix_(rng.integers(0, S, S), rng.integers(0, W, W))].mean()
        point = float(M.mean())
    else:
        boot = np.empty(reps)
        for i in range(reps):
            boot[i] = np.mean([rows[s][rng.integers(0, len(rows[s]), len(rows[s]))].mean()
                               for s in rng.integers(0, S, S)])
        point = float(np.mean([r.mean() for r in rows]))
    alpha = 1-level
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    return dict(n_seeds=S, n_worlds=[len(r) for r in rows], design='crossed' if crossed else 'nested', diff=point,
                ci95=[float(lo), float(hi)], reps=reps, seed=seed,
                convention='diff = treatment - reference; negative = less delay')


# ------------------------------------------------------------------------------------------------ permutation / Holm
def exact_paired_permutation(diffs):
    """Exact paired sign-flip test of mean(diffs) = 0 over all 2^n assignments (n = 6 seed pairs -> 64).
    p_two_sided = P(|mean*| >= |mean obs|); p_less = P(mean* <= mean obs) (treatment better = negative)."""
    d = [float(x) for x in diffs]
    n = len(d)
    if not 1 <= n <= 20:
        raise ValueError('exact test supports 1..20 pairs')
    obs = sum(d)
    tol = 1e-12*max(1.0, sum(abs(x) for x in d))
    ge = le = 0
    for signs in itertools.product((1, -1), repeat=n):
        s = sum(x*g for x, g in zip(d, signs))
        ge += abs(s) >= abs(obs)-tol
        le += s <= obs+tol
    total = 2**n
    return dict(n=n, assignments=total, mean=obs/n, p_two_sided=ge/total, p_less=le/total,
                min_p_two_sided=2/total, note='sign flips of the per-seed paired differences; exact, no resampling')


def holm(pvalues):
    """Holm step-down adjusted p-values for {name: p} (two primaries in these experiments)."""
    names = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m, running, out = len(names), 0.0, {}
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m-i)*float(pvalues[k])))
        out[k] = running
    return out


# ------------------------------------------------------------------------------------------------ tails / braking
def cvar90(delays):
    """nm_finaleval.cvar90: mean delay of the ceil(0.10 n) most-delayed vehicles."""
    d = np.sort(np.asarray(delays, float))[::-1]
    return float(d[:max(1, math.ceil((1-TAIL_LEVEL)*len(d)-1e-9))].mean())


def paired_cvar90(per_vehicle_t, per_vehicle_ref):
    """One world: CVaR90 of each run over the vehicles present in BOTH runs (nm_finaleval._paired_tail).
    per_vehicle_* = {vehicle id: delay_s} or the report form {vehicle id: [delay_s, origin, is_cav]}."""
    def delay(x):
        return float(x[0]) if isinstance(x, (list, tuple)) else float(x)
    ids = sorted(set(per_vehicle_t) & set(per_vehicle_ref))
    if not ids:
        raise ValueError('no common vehicles')
    return dict(n_common=len(ids), treatment=cvar90([delay(per_vehicle_t[v]) for v in ids]),
                reference=cvar90([delay(per_vehicle_ref[v]) for v in ids]))


def hard_brake_vs(accels, threshold=HARD_BRAKE_MPS2):
    """Hard-brake vehicle-seconds: number of logged 1 Hz vehicle rows with a < -3.0 m/s^2."""
    return int(np.sum(np.asarray(accels, float) < threshold))


def iter_log_ticks(path):
    """The per-second rows {t, k, s, v, a, lane} of a runner vehicle log (vehicles.jsonl.gz; nm_eval_runner
    VehicleLog: header rows carry 'veh', tick rows carry 't'; logged for 300 <= t < 1200)."""
    with gzip.open(Path(path), 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if 't' in row:
                yield row


def log_hard_brake_vs(path, t0=None, t1=None):
    return sum(hard_brake_vs(r['a']) for r in iter_log_ticks(path)
               if (t0 is None or r['t'] >= t0) and (t1 is None or r['t'] < t1))


def log_prefix_digest(path, t_end):
    """sha256 of the DECOMPRESSED log content strictly before ``t_end`` (headers of vehicles first seen before
    t_end + tick rows). Used by the E2 replay to prove the three branches share one prefix; the gzip container sha
    is not comparable across runs (identity receipts A67/A68: vehicle_log_equal False with J diff 0)."""
    import hashlib
    h = hashlib.sha256()
    with gzip.open(Path(path), 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            t = row.get('t', row.get('first_t'))
            if t is not None and t >= t_end:
                if 't' in row:
                    break
                continue
            h.update(json.dumps(row, sort_keys=True).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ control effort
def control_effort(acts, specs, horizon_s=1200):
    """Actual control effort of one episode from the executor receipts (record['a37']['acts']): each act =
    dict(t, family, members=[vid...], release={vid: [elapsed_s, why]}); specs = {family: strategy}; members are in
    role order (a42_bigbank.realised_sizes). A member HELD iff it was never released or released after its role
    onset; its hold time = min(release elapsed, onset + hold, horizon - t) - onset (>= 0). A group is FULL iff every
    planned member held, else TRUNCATED."""
    hold_vs, cavs, full, trunc, by_family = 0.0, set(), 0, 0, {}
    for a in acts or []:
        roles = specs[a['family']]['roles']
        if len(roles) != len(a['members']):
            raise ValueError(f'{a["family"]}: members do not match the plan roles')
        rel = a.get('release') or {}
        held = 0
        for vid, role in zip(a['members'], roles):
            on, end = int(role['onset_s']), int(role['onset_s'])+int(role['hold_s'])
            r = rel.get(vid)
            stop = min(end, horizon_s-int(a['t'])) if r is None else min(float(r[0]), end, horizon_s-int(a['t']))
            if r is None or float(r[0]) > on:
                held += 1
                cavs.add(vid)
            hold_vs += max(0.0, stop-on)
        full += held == len(roles)
        trunc += held != len(roles)
        by_family[a['family']] = by_family.get(a['family'], 0)+1
    return dict(activations=len(acts or []), activations_by_family=by_family, hold_vehicle_seconds=hold_vs,
                unique_cavs=len(cavs), full_groups=full, truncated_groups=trunc)


def planned_resource(strategy):
    """Planned (descriptor-level) resource of one activation: vehicles and the sum of hold durations."""
    return dict(vehicles=len(strategy['roles']), hold_s_sum=sum(int(r['hold_s']) for r in strategy['roles']))


# ------------------------------------------------------------------------------------------------ CLI
def selftest():
    x = [5.0, 3.0, 2.5, 8.0, 1.0, 0.5, 4.0, 7.0]           # delays are positive (pct needs a non-zero reference)
    z = paired_bootstrap(x, x)
    assert z['diff'] == 0.0 and z['ci95'] == [0.0, 0.0] and z['pct'] == 0.0 and z['pct_ci95'] == [0.0, 0.0], z
    p = exact_paired_permutation([-1, -2, -3, -4, -5, -6])
    assert p['p_two_sided'] == 2/64 and p['p_less'] == 1/64, p
    assert holm(dict(a=0.01, b=0.04)) == dict(a=0.02, b=0.04)
    return dict(selftest='PASS')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('selftest', 'contrast'))
    p.add_argument('--treatment')
    p.add_argument('--reference')
    a = p.parse_args(argv)
    if a.command == 'selftest':
        print(json.dumps(selftest()))
        return 0
    t = json.loads(Path(a.treatment).read_text(encoding='utf-8'))
    r = json.loads(Path(a.reference).read_text(encoding='utf-8'))
    print(json.dumps(paired_by_world(t, r), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
