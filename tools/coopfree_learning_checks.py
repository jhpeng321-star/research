"""Read-only OWNER02:21 checks; never dispatch physics or count replay as learning.

Uses only published exact HOLD-credit updates and paired observable DEV metrics.
All effect slopes are descriptive, not significance tests or causal learning claims.
"""
import math
import random


METRICS = ('team_delay_vs', 'collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake')


def quantile(values, p):
    values = sorted(values)
    x = (len(values)-1)*p
    lo = int(x)
    hi = min(lo+1, len(values)-1)
    return values[lo]*(hi-x)+values[hi]*(x-lo) if hi != lo else values[lo]


def paired_hold(learned, hold):
    """Pair worlds BEFORE bootstrap; never subtract marginal confidence intervals."""
    def index(rows):
        out = {r['world']['seed']: r for r in rows}
        if len(out) != len(rows) or not out:
            raise ValueError('nonempty unique worlds required')
        return out
    a, b = index(learned), index(hold)
    if a.keys() != b.keys() or any(a[s]['world'] != b[s]['world'] for s in a):
        raise ValueError('exact same DEV worlds required')
    seeds = sorted(a)
    rng = random.Random(20261101)
    draws = [[rng.randrange(len(seeds)) for _ in seeds] for _ in range(4000)]
    out = {}
    for h in ('600', '900'):
        for metric in METRICS:
            diffs = [a[s]['metrics'][h][metric]-b[s]['metrics'][h][metric] for s in seeds]
            if not all(math.isfinite(x) for x in diffs):
                raise ValueError('nonfinite DEV metric')
            samples = [sum(diffs[i] for i in indices)/len(seeds) for indices in draws]
            out[h+'_'+metric] = dict(mean=sum(diffs)/len(seeds),
                ci95=[quantile(samples, .025), quantile(samples, .975)])
    return dict(worlds=len(seeds), seeds=seeds, learned_minus_hold=out,
                bootstrap='4000 paired world resamples; Python Random seed20261101; lower delay is better')


def slope(values):
    n = len(values)
    center = (n-1)/2
    return sum((i-center)*v for i, v in enumerate(values))/sum((i-center)**2 for i in range(n))


def learning_check(updates, train_charged, train_cap=1650000, paired=None, dev=None):
    """A completed fresh batch counts once. Missing results remain explicit."""
    updates = sorted(updates, key=lambda r: r['update'])
    ids = [r['update'] for r in updates]
    if len(set(ids)) != len(ids) or any(i < 1 for i in ids):
        raise ValueError('unique positive update IDs required')
    for r in updates:
        if not r['baseline'].startswith('zero-update policy-mean HOLD'):
            raise ValueError('legacy MAPPO-credit update excluded')
        if r['windows'] < 150 or r['worlds'] < 12:
            raise ValueError('incomplete batch cannot count as update')
    if not 0 <= train_charged <= train_cap:
        raise ValueError('TRAIN accounting outside cap')
    latest = updates[-1] if updates else None
    remaining = train_cap-train_charged
    # Even joint-only 150 windows at earliest registered t300 require 135000s.
    # Real fresh updates also need HOLD and all-member solos: this is a lower bound.
    joint_floor = 150*(300+600)
    out = dict(owner='02:21', completed_updates=ids, next_scheduled_check=((max(ids, default=0)//3)+1)*3,
        train_remaining_sim_s=remaining, fresh_update_joint_only_lower_bound=joint_floor,
        further_update_budget_blocked=remaining < joint_floor,
        status='awaiting first corrected update', trend=None, recommended_phase_changes=[],
        dev_vs_mappo=dev['learned_minus_comparator']['mappo'] if dev else None,
        dev_vs_hold=paired['learned_minus_hold'] if paired else None,
        interpretation='TRAIN exploration effects are not DEV learned-policy improvements; slopes descriptive only')
    if not latest:
        return out
    keys = ('executed_change_share', 'issued_change_share', 'executed_vs_issued_share',
            'safety_override_share', 'D_G', 'sum_D_i', 'C_G', 'positive_C_G_share',
            'exact_credit_windows', 'exact_credit_worlds', 'windows', 'worlds', 'sigma_used', 'sigma_next')
    out['latest'] = {k: latest[k] for k in keys}
    out['status'] = 'insufficient fresh updates for six-update trend'
    contiguous = ids == list(range(1, len(ids)+1))
    if len(updates) >= 6 and contiguous:
        last = updates[-6:]
        if all(r['C_G'] is not None for r in last):
            dg = slope([r['D_G']['mean'] for r in last])
            cg = slope([r['C_G']['mean'] for r in last])
            small = latest['executed_change_share'] <= .5
            stalled = dg <= 0 and cg <= 0
            out['trend'] = dict(updates=[r['update'] for r in last], D_G_slope=dg, C_G_slope=cg,
                executed_change_small=small, both_effects_nonincreasing=stalled,
                caveat='C_G uses disclosed exact subset; state-mix changes can affect descriptive slopes')
            out['status'] = 'phase fix required' if small or stalled else 'no registered phase-fix trigger'
            if small or stalled:
                out['recommended_phase_changes'] = [
                    'Calibrate shared Gaussian scale against measured execution; retain safety and log before/after.',
                    'Adopt only observable helpful-state evidence from public COOPBANDIT/COOPPLAN; keep free action space.',
                    'Prospectively compare group ES perturbations by effect/gradient SNR per charged sim-s.',
                    'Register new phase, CPU/source binding and budget before physics; retain all completed arms.']
        else:
            out['status'] = 'six-update trend unavailable: missing exact C_G'
    return out


def render(check):
    lines = ['### OWNER02:21 learning self-check',
        '- Completed corrected fresh updates: %s; next scheduled check u%d.' %
        (check['completed_updates'], check['next_scheduled_check']),
        '- Status: '+check['status']+'.',
        '- TRAIN remaining: %d sim-s; one fresh >=150-window joint-only lower bound: %d sim-s (HOLD/solos extra).' %
        (check['train_remaining_sim_s'], check['fresh_update_joint_only_lower_bound'])]
    if check['further_update_budget_blocked']:
        lines.append('- **Updates3/6/9 are budget-blocked. Prospective owner phase/budget amendment required; no replay or automatic dispatch.**')
    if check['dev_vs_hold']:
        for h in ('600', '900'):
            value = check['dev_vs_hold'][h+'_team_delay_vs']
            lines.append('- DEV learned-minus-HOLD J%s: %.3f vehicle-s, paired world-bootstrap95%%CI %s.' %
                         (h, value['mean'], value['ci95']))
    else:
        lines.append('- DEV learned-minus-HOLD J600/J900: pending both complete paired panels; no marginal-CI subtraction.')
    if check['trend']:
        t = check['trend']
        lines.append('- Six-update descriptive slopes: D_G %.3f; C_G %.3f vehicle-s/update.' % (t['D_G_slope'], t['C_G_slope']))
    lines += ['- '+s for s in check['recommended_phase_changes']]
    return '\n'.join(lines)
