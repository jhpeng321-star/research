"""Paired branch test for the sealed S-COOPMWM final panel (co-primary #1 and #2).

Design (registered before any final-test outcome is opened):
- On every sealed final-test world, at fixed branch moments T (default six,
  evenly spaced in [300, 900] s so a 300 s continuation ends by 1200 s), every
  arm continues for H=300 s from the SAME traffic state.
- The common state is reached by an EXACT rerun of one shared prefix policy
  (frozen speed-only MAPPO169, i.e. the zero-residual update0 adapter) from the
  native warmup to T. Each arm's run is gated bitwise against the MAPPO
  reference episode fingerprints for every second 300..T; mismatch -> STOP.
- MAPPO's own branch outcome IS the reference episode (MAPPO prefix + MAPPO
  continuation), so the MAPPO arm needs no extra physics when the final-test
  MAPPO169 episode (accel_plant RESULT with per-second fingerprints) is reused.
- Metric J_b = team_delay_vs(T+H) - team_delay_vs(T): on-road speed deficit plus
  source waiting (same J as tools/coopnative comparator/evaluation), veh-s.
  Paired difference d = J(treatment) - J(comparator); negative = improvement.
  % delay reduction = 100 * (sum J_comp - sum J_treat) / sum J_comp; positive = improvement.
- Inference: world-cluster percentile bootstrap (10,000 whole-world resamples,
  numpy default_rng(20261101)); a world carries all its branches. Two-sided
  CI-inversion p-values; Holm over the registered co-primary family.

The SUMO execution itself is NOT implemented here: an injectable runner executes
jobs. `AccelPlantRunner` adapts the existing exact-rerun machinery
(tools.coopmwm.accel_plant.episode) and only adds a takeover wrapper that
suppresses the arm's residual requests before T (exact zero abstains -> bitwise
MAPPO physics, CPU-tested upstream).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math

import numpy as np

SCHEMA = 'coopmwm/branch_eval/v1'
EXPERIMENT = 'S-COOPMWM-BRANCH'
WARMUP_S = 300
EPISODE_END_S = 1200
HORIZON_S = 300
N_BRANCHES = 6
BOOT_SEED = 20261101
N_BOOT = 10000
ALPHA = 0.05
PREFIX_POLICY = 'MAPPO169'
MAPPO_SHA = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
FINAL_PANEL_SHA = 'd74f685ef75a819248bbe7bb0bef89cf0295037ff3b03b334b13ccec75c40bcc'
PRIMARY_ARMS = ('MAPPO', 'A', 'B')
LATER_ARMS = ('R', 'native')
# Owner addendum 2026-11-01 22:58: demand 2400-2700 x CAV share 0.2-0.4 (inclusive).
FOCUS = dict(demand=(2400, 2700), penetration=(0.2, 0.4))
FOCUS_SOURCES = (
    'c4-coopbandit docs/paper_results/COOPBANDIT.md (high-demand .2/.4 cells; dense-rule and helpful-state concentration)',
    'c4-coopplan docs/paper_results/COOPPLAN.md (SUMO-oracle planner J600/J900 gains, pooled36)',
)
SCOPES = ('branch', 'focus')
CO_PRIMARY_CONTRASTS = (('B', 'MAPPO'), ('B', 'A'))
SECONDARY_CONTRASTS = (('A', 'MAPPO'),)
LATER_CONTRASTS = (('B', 'R'), ('B', 'native'), ('A', 'native'), ('MAPPO', 'native'))
METRIC = 'team_delay_vs'
SAFETY = ('collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake', 'road_delay_vs', 'source_wait_vs')


class PrefixMismatch(ValueError):
    """An arm's run did not reproduce the shared prefix bit-for-bit."""


def canonical(value):
    # Identical to tools.coopnative.protocol.canonical (registered panel digests).
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


# ---------------------------------------------------------------- plan

def branch_times(n=N_BRANCHES, warmup_s=WARMUP_S, end_s=EPISODE_END_S, horizon_s=HORIZON_S):
    """Evenly spaced integer seconds in [warmup, end-horizon], inclusive both ends."""
    last = end_s - horizon_s
    if n < 1 or last < warmup_s:
        raise ValueError('no valid branch time')
    if n == 1:
        return [warmup_s]
    span = last - warmup_s
    if span % (n - 1):
        raise ValueError('branch grid must be integer seconds')
    return [warmup_s + i * span // (n - 1) for i in range(n)]


def validate_times(times, warmup_s=WARMUP_S, end_s=EPISODE_END_S, horizon_s=HORIZON_S):
    if not times or list(times) != sorted(set(times)):
        raise ValueError('branch times must be strictly increasing and unique')
    for t in times:
        if type(t) is not int or t < warmup_s or t + horizon_s > end_s:
            raise ValueError(f'branch time {t!r} outside [{warmup_s}, {end_s - horizon_s}]')
    return True


def final_panel():
    """The 96 sealed final-test worlds (metadata only; never opens any run data)."""
    from .accel_protocol import worlds
    ws = worlds('final_test')
    if digest(ws) != FINAL_PANEL_SHA or len(ws) != 96:
        raise ValueError('final panel differs from the registered SHA')
    return ws


def in_focus(world, focus=FOCUS):
    lo_d, hi_d = focus['demand']
    lo_p, hi_p = focus['penetration']
    return lo_d <= world['demand'] <= hi_d and lo_p - 1e-12 <= world['penetration'] <= hi_p + 1e-12


def focus_worlds(worlds, focus=FOCUS):
    return [w for w in worlds if in_focus(w, focus)]


def holm_family(arms):
    """Registered confirmatory family: {B-MAPPO, B-A} x {all branches, focus stratum}."""
    fam = [dict(id=f'{scope}:{t}-{c}', scope=scope, treatment=t, comparator=c)
           for scope in SCOPES for t, c in CO_PRIMARY_CONTRASTS]
    for h in fam:
        if h['treatment'] not in arms or h['comparator'] not in arms:
            raise ValueError('co-primary arm missing from plan')
    return fam


def budget_quote(n_worlds, times, arms, horizon_s=HORIZON_S, episode_end_s=EPISODE_END_S,
                 reuse_mappo_reference=True):
    """Exact-rerun sim-s, full prefixes charged (warmup included), no saved-state discount.

    Each non-MAPPO branch job runs fresh SUMO 0..T+H (the exact machinery has no
    prefix cache; prefixes shared across branches/arms cannot be reused because
    every job restarts SUMO). MAPPO's branch outcomes come from one MAPPO
    reference episode per world (0..max T+H); that episode is already part of the
    final-test charge when reused, otherwise it is charged here.
    """
    per_world_branch = sum(t + horizon_s for t in times)
    run_arms = [a for a in arms if a != 'MAPPO']
    ref_len = max(times) + horizon_s
    reference = 0 if reuse_mappo_reference else n_worlds * ref_len
    per_arm = n_worlds * per_world_branch
    total = per_arm * len(run_arms) + reference
    naive = n_worlds * per_world_branch * len(arms)
    restore = n_worlds * len(times) * len(arms) * horizon_s
    return dict(
        worlds=n_worlds, branch_times=list(times), arms=list(arms), horizon_s=horizon_s,
        formula='sum_arms!=MAPPO worlds*sum_T(T+H) + [worlds*(maxT+H) if MAPPO reference not reused]',
        per_world_per_arm_sim_s=per_world_branch, per_arm_sim_s=per_arm,
        mappo_reference_sim_s=reference, reuse_mappo_reference=reuse_mappo_reference,
        total_sim_s=total,
        naive_every_arm_rerun_sim_s=naive,
        prefix_reuse_saving_sim_s=naive - total,
        saved_state_restore_alternative_sim_s=restore,
        saved_state_note=('NOT this design: restoring a saved SUMO state is physically exact but not a '
                          'bitwise rerun (S-ENSPROBE insertion-cache/CF-LC memory residual); would need a '
                          'separate amendment'),
        episode_end_s=episode_end_s)


def build_plan(worlds=None, arms=PRIMARY_ARMS, later_arms=LATER_ARMS, n_branches=N_BRANCHES,
               times=None, horizon_s=HORIZON_S, focus=FOCUS, boot_seed=BOOT_SEED,
               n_boot=N_BOOT, alpha=ALPHA, reuse_mappo_reference=True, note=None):
    worlds = final_panel() if worlds is None else [dict(w) for w in worlds]
    seeds = [w['seed'] for w in worlds]
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate world')
    times = list(times) if times is not None else branch_times(n_branches, horizon_s=horizon_s)
    validate_times(times, horizon_s=horizon_s)
    arms = list(arms)
    if arms[0] != 'MAPPO' or len(set(arms)) != len(arms):
        raise ValueError('MAPPO (prefix policy and comparator) must be the first arm')
    fam = holm_family(arms)
    fw = focus_worlds(worlds, focus)
    plan = dict(
        schema=SCHEMA, experiment=EXPERIMENT,
        panel=dict(n_worlds=len(worlds), worlds=worlds, panel_sha256=digest(worlds),
                   registered_final_panel_sha256=FINAL_PANEL_SHA),
        branches=dict(times_s=times, per_world=len(times), horizon_s=horizon_s, warmup_s=WARMUP_S,
                      episode_end_s=EPISODE_END_S,
                      rule=('fixed evenly spaced grid in [warmup, end-horizon], identical for every world, '
                            'fixed before any outcome; no state/outcome-dependent selection')),
        prefix=dict(policy=PREFIX_POLICY, checkpoint_sha256=MAPPO_SHA,
                    executor='tools.coopmwm.accel_plant.episode, zero-residual update0 Policy(), deterministic',
                    mechanism=('every arm reruns SUMO from t=0 under the SAME code path; before T its '
                               'residual requests are suppressed (exact zero abstains -> MAPPO commands); '
                               'fingerprints for every second 300..T must equal the MAPPO reference episode '
                               'bit-for-bit, else STOP (no retry)'),
                    fairness=('the prefix policy is the comparator itself and is independent of A/B/R; every '
                              'arm faces the identical physical state, MAPPO executor memory and roster at T; '
                              'A/B/R are residuals on MAPPO, so MAPPO-induced states are on their support; '
                              'MAPPO is not favoured by state selection because times are fixed a priori')),
        arms=arms, later_arms=list(later_arms),
        arm_execution=dict(MAPPO='derived from the per-world MAPPO reference episode (0 new sim-s if reused)',
                           A='fixed update24 checkpoint, deterministic, no guidance',
                           B='fixed update24 checkpoint, deterministic, no guidance (guidance is training-only)',
                           R='fixed update24 checkpoint, deterministic (later, same plan)',
                           native='CAVs released to SUMO native at T (later; needs a native-takeover runner)'),
        metric=dict(name='paired 300 s team delay', key=METRIC,
                    definition='J_b = team_delay_vs(T+H) - team_delay_vs(T); road speed deficit + source waiting, veh-s',
                    difference='treatment - comparator (negative = improvement)',
                    percent_reduction='100*(sum J_comp - sum J_treat)/sum J_comp over the resampled worlds (positive = improvement)',
                    secondary=list(SAFETY)),
        analysis=dict(
            unit='world cluster (all branches of a world resampled together)',
            estimator='mean over all world x branch pairs of d_b (complete panel => mean of world means)',
            bootstrap=dict(resamples=n_boot, rng=f'numpy.random.default_rng({boot_seed}).integers(0,n,(B,n)) per scope',
                           seed=boot_seed, interval='percentile 95%'),
            p_value='two-sided CI inversion: min(1, 2*min((#boot<=0)+1, (#boot>=0)+1)/(B+1))',
            co_primary=dict(
                one='paired branch test, all worlds (co-primary #1)',
                two='same paired branch metric restricted to the focus stratum worlds (co-primary #2)',
                family=[h['id'] for h in fam],
                multiplicity=f'Holm step-down over the 4-hypothesis family, FWER alpha={alpha} two-sided'),
            focus=dict(definition=focus, n_worlds=len(fw), seeds=[w['seed'] for w in fw],
                       sources=list(FOCUS_SOURCES),
                       secondary='full-panel J600/J900 episode results stay registered secondary'),
            secondary_contrasts=[f'{t}-{c}' for t, c in SECONDARY_CONTRASTS],
            later_contrasts=[f'{t}-{c}' for t, c in LATER_CONTRASTS],
            descriptive=['per branch time', 'per cell', 'safety deltas'],
            completeness='every world x branch x arm exactly once; incomplete panel => no inference'),
        rng=dict(bootstrap_seed=boot_seed, sumo_seed='world seed (unchanged final-test arrivals/noise)',
                 policy_noise_seed='sha256(world,T,arm)[:15] recorded; deterministic arms draw no physical noise',
                 branch_times='not random'),
        budget=budget_quote(len(worlds), times, arms, horizon_s, reuse_mappo_reference=reuse_mappo_reference),
        budget_with_later_arms=budget_quote(len(worlds), times, arms + list(later_arms), horizon_s,
                                            reuse_mappo_reference=reuse_mappo_reference),
        note=note)
    plan['plan_sha256'] = plan_sha(plan)
    return plan


def plan_sha(plan):
    return digest({k: v for k, v in plan.items() if k != 'plan_sha256'})


def verify_plan(plan):
    if plan.get('schema') != SCHEMA or plan.get('plan_sha256') != plan_sha(plan):
        raise ValueError('plan SHA mismatch (edited after registration?)')
    validate_times(plan['branches']['times_s'], horizon_s=plan['branches']['horizon_s'])
    return True


# ---------------------------------------------------------------- jobs

def noise_seed(world, t, arm):
    return int(digest(dict(experiment=EXPERIMENT, world=world['seed'], t=t, arm=arm))[:15], 16)


def jobs(plan, arms=None):
    """One job per world x branch x arm; MAPPO jobs derive from the reference episode."""
    verify_plan(plan)
    arms = list(arms or plan['arms'])
    unknown = set(arms) - set(plan['arms']) - set(plan['later_arms'])
    if unknown:
        raise ValueError(f'unregistered arms {sorted(unknown)}')
    H = plan['branches']['horizon_s']
    reuse = plan['budget']['reuse_mappo_reference']
    out = []
    for w in plan['panel']['worlds']:
        for t in plan['branches']['times_s']:
            for arm in arms:
                derived = arm == 'MAPPO'
                out.append(dict(id=f'branch_{arm}_{w["seed"]}_t{t}', world=w, arm=arm,
                                branch_time_s=t, horizon_s=H, stop_s=t + H,
                                mode='derive_from_reference' if derived else 'run',
                                sim_s=0 if derived else t + H,
                                prefix_policy=plan['prefix']['policy'],
                                noise_seed=noise_seed(w, t, arm), deterministic=True,
                                plan_sha256=plan['plan_sha256']))
    refs = [dict(id=f'branch_ref_MAPPO_{w["seed"]}', world=w, arm='MAPPO', mode='reference',
                 stop_s=max(plan['branches']['times_s']) + H,
                 sim_s=0 if reuse else max(plan['branches']['times_s']) + H,
                 plan_sha256=plan['plan_sha256']) for w in plan['panel']['worlds']]
    return refs, out


def job_sim_s(refs, branch_jobs):
    return sum(j['sim_s'] for j in refs) + sum(j['sim_s'] for j in branch_jobs)


# ---------------------------------------------------------------- execution

def decode_metrics(fingerprint):
    """accel_plant fingerprints store totals as float.hex() strings (leverprobe exact())."""
    m = fingerprint['metrics']
    return {k: (float.fromhex(v) if isinstance(v, str) else float(v)) for k, v in m.items()}


def totals_at(result, t):
    """Cumulative totals at the START of second t (before its tick)."""
    fp = result['prefixes'].get(str(t))
    if fp is None and result.get('final') is not None and float(result['final']['t_s']) == float(t):
        fp = result['final']
    if fp is None:
        raise ValueError(f'no fingerprint at t={t}')
    return decode_metrics(fp)


def check_prefix(result, reference, t_branch, warmup_s=WARMUP_S):
    """Bitwise identical fingerprints for every second warmup..T (inclusive)."""
    for t in range(warmup_s, t_branch + 1):
        mine, ref = result['prefixes'].get(str(t)), reference['prefixes'].get(str(t))
        if mine is None or ref is None or canonical(mine) != canonical(ref):
            raise PrefixMismatch(f'prefix differs from {PREFIX_POLICY} reference at t={t}')
    return True


def window(result, t, horizon_s):
    start, end = totals_at(result, t), totals_at(result, t + horizon_s)
    return {k: end[k] - start[k] for k in start}


def execute(plan, runner, arms=None, references=None):
    """Reference semantics (serial). Production integrations must keep their own
    precharge/receipt/admission layer and call the same check_prefix/window."""
    refs, branch_jobs = jobs(plan, arms)
    references = dict(references or {})
    records = []
    for ref_job in refs:
        seed = ref_job['world']['seed']
        if seed not in references:
            references[seed] = runner.reference(ref_job)
    for job in branch_jobs:
        seed, t, H = job['world']['seed'], job['branch_time_s'], job['horizon_s']
        ref = references[seed]
        if job['mode'] == 'derive_from_reference':
            result = ref
        else:
            result = runner.run(job, ref)
            check_prefix(result, ref, t)
            if canonical(totals_at(result, t)) != canonical(totals_at(ref, t)):
                raise PrefixMismatch('cumulative totals differ at the branch moment')
        w = window(result, t, H)
        records.append(dict(seed=seed, branch_time_s=t, arm=job['arm'], J=w[METRIC], window=w,
                            job_id=job['id'], prefix_pass=True))
    return records


class TakeoverControllerMixin:
    """Suppress all residual requests strictly before branch['takeover_s'].

    The wrapped Controller still observes every prefix step (its event log keeps one
    action per second, as accel_plant requires) but issues {} -> exact MAPPO commands.
    """
    def apply(self, rows, commands, now):
        desired = super().apply(rows, commands, now)
        if self.branch and now < self.branch['takeover_s']:
            event = self.events[-1]
            event['suppressed_requests'] = event['requests']
            event['requests'] = {}
            event['takeover_suppressed'] = True
            return {}
        return desired


@contextmanager
def takeover_controller():
    """Scoped swap of accel_plant's Controller name; restored on exit. Use in an isolated worker."""
    from . import accel_plant
    from .acceleration import Controller
    original = accel_plant.Controller
    accel_plant.Controller = type('TakeoverController', (TakeoverControllerMixin, Controller), {})
    try:
        yield accel_plant.Controller
    finally:
        accel_plant.Controller = original


class AccelPlantRunner:
    """Adapter onto the existing exact-rerun executor. Not exercised on this host (no SUMO)."""
    def __init__(self, root, runtime, checkpoint, weights, role='final_test', extras=None,
                 reference_results=None):
        self.root, self.runtime, self.checkpoint = root, runtime, checkpoint
        self.weights, self.role, self.extras = weights, role, dict(extras or {})
        self.reference_results = reference_results or {}

    def _job(self, job, arm, stop, branch=None):
        from pathlib import Path
        return dict(id=job['id'], path=str(Path(self.root) / 'jobs' / job['id']), world=job['world'],
                    role=self.role, arm=arm, weights=self.weights[arm], runtime=self.runtime,
                    checkpoint=self.checkpoint, noise_seed=job.get('noise_seed', 0), deterministic=True,
                    guided=False, sim_s=stop, stop=stop, branch=branch, **self.extras)

    def reference(self, ref_job):
        seed = ref_job['world']['seed']
        if seed in self.reference_results:
            return self.reference_results[seed]()
        from .accel_plant import episode
        return episode(self._job(ref_job, 'MAPPO', ref_job['stop_s']))

    def run(self, job, reference):
        if job['arm'] == 'native':
            raise NotImplementedError('native takeover needs a native-release runner (plant.episode has no branch)')
        t = job['branch_time_s']
        branch = dict(time_s=t, takeover_s=t, revert=[],
                      prefixes={str(s): reference['prefixes'][str(s)] for s in range(WARMUP_S, t + 1)})
        from .accel_plant import episode
        with takeover_controller():
            return episode(self._job(job, job['arm'], job['stop_s'], branch))


# ---------------------------------------------------------------- analysis

def _table(records, plan, arms):
    seeds = [w['seed'] for w in plan['panel']['worlds']]
    times = plan['branches']['times_s']
    J = {}
    for r in records:
        key = (r['seed'], r['branch_time_s'], r['arm'])
        if key in J:
            raise ValueError(f'duplicate record {key}')
        J[key] = float(r['J'])
    for s in seeds:
        for t in times:
            for a in arms:
                if (s, t, a) not in J:
                    raise ValueError(f'incomplete panel: missing {(s, t, a)}')
                if not math.isfinite(J[s, t, a]):
                    raise ValueError('nonfinite J')
    extra = {k for k in J if k[0] not in seeds or k[1] not in times}
    if extra:
        raise ValueError('records outside the registered panel/branch grid')
    return {a: np.array([[J[s, t, a] for t in times] for s in seeds]) for a in arms}


def cluster_bootstrap(treat, comp, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    """treat/comp: worlds x branches arrays. Resample whole worlds."""
    treat, comp = np.asarray(treat, float), np.asarray(comp, float)
    n = len(treat)
    if n < 2:
        raise ValueError('need at least two worlds')
    d = treat - comp
    per_world_d, per_world_c, per_world_t = d.sum(1), comp.sum(1), treat.sum(1)
    count = np.full(n, d.shape[1], float)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    boot_mean = per_world_d[idx].sum(1) / count[idx].sum(1)
    cs, ts = per_world_c[idx].sum(1), per_world_t[idx].sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        boot_pct = np.where(cs != 0, 100 * (cs - ts) / cs, np.nan)
    mean = float(d.mean())
    pct = float(100 * (comp.sum() - treat.sum()) / comp.sum()) if comp.sum() != 0 else float('nan')
    lo, hi = np.quantile(boot_mean, [alpha / 2, 1 - alpha / 2])
    plo, phi = np.nanquantile(boot_pct, [alpha / 2, 1 - alpha / 2])
    p = min(1., 2 * min((np.sum(boot_mean <= 0) + 1), (np.sum(boot_mean >= 0) + 1)) / (n_boot + 1))
    return dict(n_worlds=n, n_branches=int(d.size), mean_diff_vs=mean, ci95=[float(lo), float(hi)],
                p_two_sided=float(p), mean_treatment_J=float(treat.mean()), mean_comparator_J=float(comp.mean()),
                pct_delay_reduction=pct, pct_ci95=[float(plo), float(phi)],
                worlds_improved=int(np.sum(d.mean(1) < 0)))


def holm(pvalues, alpha=ALPHA):
    """Holm step-down. pvalues: dict id->p. Returns id->(adjusted p, reject)."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m, running, out, stop = len(items), 0., {}, False
    for i, (k, p) in enumerate(items):
        running = max(running, min(1., (m - i) * p))
        reject = not stop and p <= alpha / (m - i)
        if not reject:
            stop = True
        out[k] = dict(p=float(p), p_holm=float(running), reject=bool(reject))
    return out


def analyze(records, plan, contrasts=None):
    verify_plan(plan)
    n_boot, seed = plan['analysis']['bootstrap']['resamples'], plan['analysis']['bootstrap']['seed']
    alpha = ALPHA
    fam = holm_family(plan['arms'])
    contrasts = list(contrasts or (list(CO_PRIMARY_CONTRASTS) + list(SECONDARY_CONTRASTS)))
    missing = {(h['treatment'], h['comparator']) for h in fam} - set(contrasts)
    if missing:
        raise ValueError(f'co-primary contrasts must be analysed: {sorted(missing)}')
    arms = sorted({a for c in contrasts for a in c})
    table = _table(records, plan, arms)
    worlds = plan['panel']['worlds']
    focus_mask = np.array([in_focus(w, plan['analysis']['focus']['definition']) for w in worlds])
    if focus_mask.sum() < 2:
        raise ValueError('focus stratum needs >=2 worlds')
    results = {}
    for scope in SCOPES:
        mask = np.ones(len(worlds), bool) if scope == 'branch' else focus_mask
        for t, c in contrasts:
            results[f'{scope}:{t}-{c}'] = dict(scope=scope, treatment=t, comparator=c,
                                               **cluster_bootstrap(table[t][mask], table[c][mask], n_boot, seed, alpha))
    adj = holm({h['id']: results[h['id']]['p_two_sided'] for h in fam}, alpha)
    for k, v in adj.items():
        results[k].update(holm=v, co_primary=True)
    times = plan['branches']['times_s']
    per_time = {f'{t}-{c}@{T}': float((table[t][:, i] - table[c][:, i]).mean())
                for t, c in contrasts for i, T in enumerate(times)}
    return dict(plan_sha256=plan['plan_sha256'], alpha=alpha, family=[h['id'] for h in fam],
                n_focus_worlds=int(focus_mask.sum()), results=results, per_branch_time_mean_diff=per_time,
                sign='difference = treatment - comparator (negative = improvement); pct positive = improvement')


def write_plan(path, plan):
    from pathlib import Path
    Path(path).write_text(json.dumps(plan, indent=1, sort_keys=True), encoding='utf-8')
    return plan['plan_sha256']


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Build the S-COOPMWM paired branch-test plan (no SUMO).')
    p.add_argument('--out', required=True)
    p.add_argument('--branches', type=int, default=N_BRANCHES)
    p.add_argument('--no-mappo-reuse', action='store_true')
    a = p.parse_args()
    plan = build_plan(n_branches=a.branches, reuse_mappo_reference=not a.no_mappo_reuse)
    print(write_plan(a.out, plan), plan['budget']['total_sim_s'], plan['budget_with_later_arms']['total_sim_s'])
