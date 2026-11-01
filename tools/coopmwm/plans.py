"""Observable group acceleration plans, inherited family seeds, CEM and CRN Thompson."""
import copy
import math
import numpy as np
from .protocol import digest

SEARCH = dict(population=32, iterations=3, elite=8, finalists=8,
              screen_K=4, screen_H=120, final_K=8, final_H=300,
              knot_s=10, max_duration_s=60)


def validate(plan, bounds):
    if len(plan['members']) not in (2, 3) or len(set(plan['members'])) != len(plan['members']):
        raise ValueError('complete observable group of two or three required')
    if plan['duration_s'] not in range(10, 61, 10):
        raise ValueError('unregistered plan duration')
    a = np.asarray(plan['knots'], dtype=float)
    if a.shape != (len(plan['members']), 7) or not np.isfinite(a).all():
        raise ValueError('invalid acceleration profile')
    if (a < bounds[0]).any() or (a > bounds[1]).any():
        raise ValueError('plan outside inherited acceleration limits')
    return plan


def acceleration_at(plan, elapsed):
    if not 0 <= elapsed < plan['duration_s']:
        return {}
    j = min(5, int(elapsed//10)); f = (elapsed-10*j)/10
    return {v:float((1-f)*k[j]+f*k[j+1]) for v,k in zip(plan['members'], plan['knots'])}


def group(rows):
    from tools.coopbandit.protocol import group as inherited
    # Preserve inherited geometric group availability. Extend to a third observed
    # CAV only in future registered arms; current two-member search is within 2--3.
    return inherited(rows)


def family_seeds(rows, members, commands, bounds):
    from tools.wmteach.labels import actions
    from tools.leverprobe.intervention import targets, requests
    out = []
    for setting in range(4):
        for option in actions(rows, setting):
            if option['id'] == 'O0' or not set(option['members']) <= set(members):
                continue
            current = copy.deepcopy(rows); fixed = targets(option, rows)
            profiles = [[] for _ in members]
            for t in range(61):
                requested, _ = requests(option, current, fixed)
                raw = {v:float(np.clip(requested[v]['requested_raw'] if v in requested else commands[v], *bounds)) for v in members}
                if t % 10 == 0:
                    for v,k in zip(members, profiles):k.append(raw[v])
                for r in current:
                    a = raw.get(r['vid'], 0.)
                    r['v_mps'] = max(0., r['v_mps']+a)
                    r['s_m'] += r['v_mps']
            p = dict(id=f"{option['id']}_s{setting}", members=members,
                duration_s=min(60, max(10, int(math.ceil(option.get('window', 60)/10))*10)),
                knots=profiles, family=option['id'], origin='inherited', setting=setting)
            out.append(validate(p, bounds))
    return out


def classify(knots, seeds):
    if not seeds:
        raise ValueError('no inherited available family descriptor')
    # Nearest registered family prototype, no behavioural cutoff or hand labels.
    return min(seeds, key=lambda s:(float(np.square(np.asarray(s['knots'])-knots).mean()),s['id']))['family']


def population(record, rng, bounds, seeds, count, mean=None, std=None, durations=None):
    members = record['group']; shape = (len(members), 7)
    mean = np.full(shape, (bounds[0]+bounds[1])/2) if mean is None else mean
    std = np.full(shape, (bounds[1]-bounds[0])/math.sqrt(12)) if std is None else std
    out = []
    for _ in range(count):
        knots = np.clip(rng.normal(mean, std), *bounds)
        duration = int(rng.choice(np.arange(10,61,10), p=durations))
        p = dict(members=members, knots=knots.tolist(), duration_s=duration,
            family=classify(knots, seeds), origin='sampled')
        p['id'] = 'P'+digest(p)[:16]
        out.append(validate(p,bounds))
    return out


def baseline_actions(record):
    return [dict(id='policy', window=60, residuals=record['policy_residuals']),
            dict(id='MAPPO', window=0, residuals={})]


def thompson(record, plans, futures, rng):
    """One bootstrap of CRN future indices, shared by every candidate/comparator."""
    costs = {a:np.array([x['team_delay'] for x in fs]) for a,fs in futures.items()}
    if any(len(v) != 8 or not np.isfinite(v).all() for v in costs.values()):
        raise ValueError('complete finite K8 rescore required')
    ids = rng.integers(0,8,8)
    sampled = {a:float(v[ids].mean()) for a,v in costs.items()}
    means = {a:float(v.mean()) for a,v in costs.items()}
    winner = min(plans, key=lambda p:(sampled[p['id']],p['id']))
    saving = min(sampled['policy'],sampled['MAPPO'])-sampled[winner['id']]
    ranks = sorted(plans, key=lambda p:(means[p['id']],p['id']))
    return dict(accepted=saving>0, plan=winner, option=winner['family'],
        predicted_rank=next(i+1 for i,p in enumerate(ranks) if p['id']==winner['id']),
        expected_saving=min(means['policy'],means['MAPPO'])-means[winner['id']],
        sampled_saving=saving, bootstrap_indices=ids.tolist(), means=means,
        record_sha256=digest(record), futures=futures, search=SEARCH,
        known_family=winner['origin']=='inherited')


def search(record, bounds, evaluate):
    """Evaluation callback reuses frozen WMTEACH paired futures and option bridge."""
    seeds = family_seeds(record['rows'],record['group'],record['commands'],bounds)
    if not seeds:
        return dict(accepted=False, reason='no available inherited family', record_sha256=digest(record))
    rng = np.random.default_rng(int(digest(record)[:15],16))
    bases = baseline_actions(record); mean=std=durations=None; pool=[]; searched=0
    for iteration in range(SEARCH['iterations']):
        candidates = (seeds if iteration==0 else elites)+population(record,rng,bounds,seeds,
            SEARCH['population']-(len(seeds) if iteration==0 else len(elites)),mean,std,durations)
        scores = evaluate(bases+candidates,SEARCH['screen_K'],SEARCH['screen_H'])
        ranked = sorted(candidates,key=lambda p:(np.mean([f['team_delay'] for f in scores[p['id']]]),p['id']))
        elites = ranked[:SEARCH['elite']]; pool.extend(candidates); searched += len(candidates)
        data=np.array([p['knots'] for p in elites]);mean=data.mean(0);std=data.std(0)
        # A zero empirical variance is permitted: no hand-set exploration floor.
        durations=np.array([sum(p['duration_s']==d for p in elites) for d in range(10,61,10)],float)
        durations/=durations.sum()
    finalists=ranked[:SEARCH['finalists']]
    scores=evaluate(bases+finalists,SEARCH['final_K'],SEARCH['final_H'])
    result=thompson(record,finalists,scores,rng)
    result.update(candidates_searched=searched,iterations=SEARCH['iterations'],
                  imagined_futures=searched*SEARCH['screen_K']+(len(finalists)+2)*SEARCH['final_K'])
    return result
