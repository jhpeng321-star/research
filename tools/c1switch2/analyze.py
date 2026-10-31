"""Complete-panel paired world bootstrap; best fixed is reselected in each draw."""
import math
import random
import statistics

from tools.c1switch.storage import canonical

from .design import ARMS, CONTROLLERS, worlds


def primary(rows, *, include_phys=False, draws=2000):
    arms = ARMS+(('wm_phys',) if include_phys else ())
    seeds = [w['seed'] for w in worlds()]
    indexed = {}
    for row in rows:
        key = (row['arm'], row['world']['seed'])
        if key in indexed:
            raise ValueError('duplicate episode result')
        if row['world'] not in worlds() or len(row['schedule']) != 30:
            raise ValueError('wrong world/incomplete decisions')
        value = row['J600']['team_delay_vs']
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid J600')
        indexed[key] = row
    if set(indexed) != {(a, s) for a in arms for s in seeds}:
        raise ValueError('complete paired panel required; no survivor-only bootstrap')
    for seed in seeds:
        # First decision is still the common native warmup in every arm.
        prefixes = [canonical(indexed[a, seed]['decisions'][0]['prefix']) for a in arms]
        if len(set(prefixes)) != 1:
            raise ValueError('unpaired warmup')
    def contrast(sample, arm, baseline):
        if baseline == 'best_fixed':
            baseline = min(CONTROLLERS, key=lambda c: statistics.fmean(
                indexed[c, s]['J600']['team_delay_vs'] for s in sample))
        return statistics.fmean(indexed[arm, s]['J600']['team_delay_vs']-
                                indexed[baseline, s]['J600']['team_delay_vs'] for s in sample)
    rng = random.Random(0)
    samples = [rng.choices(seeds, k=len(seeds)) for _ in range(draws)]
    result = {}
    for arm in ('wm',)+(('wm_phys',) if include_phys else ()):
        for baseline in ('fleet_ppo', 'best_fixed'):
            distribution = sorted(contrast(s, arm, baseline) for s in samples)
            result[arm+' minus '+baseline] = dict(mean=contrast(seeds, arm, baseline),
                lo=distribution[int(.025*draws)], hi=distribution[min(draws-1, int(.975*draws))],
                n_worlds=len(seeds), units='vehicle-seconds', negative_is_better=True)
    return result


def regret(branches, chosen, joint=None):
    if set(branches) != set(CONTROLLERS):
        raise ValueError('complete six-way exact branch set required')
    if not isinstance(chosen, dict) and chosen not in CONTROLLERS:
        raise ValueError('fleet action required for library oracle regret')
    prefixes = {canonical(row['prefix']) for row in branches.values()}
    if len(prefixes) != 1:
        raise ValueError('unpaired branch prefixes')
    costs = {c: row['60']['team_delay_vs'] for c, row in branches.items()}
    if any(not math.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError('invalid exact branch delay')
    oracle = min(costs, key=costs.get)
    if isinstance(chosen, dict):
        if joint is None or canonical(joint['prefix']) not in prefixes:
            raise ValueError('per-CAV oracle gap requires its exact held-joint-action branch')
        value = joint['60']['team_delay_vs']
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid joint delay')
        return dict(oracle=oracle, costs=costs, signed_joint_oracle_gap=value-costs[oracle])
    return dict(oracle=oracle, costs=costs, regret=costs[chosen]-costs[oracle])
