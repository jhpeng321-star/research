"""TRAIN-derived observable sampling, with a separate immutable prior input."""
import numpy as np
from .protocol import TARGET_WINDOWS, ZONE, digest
from tools.coopmwm.map_descriptor import FRAME


def features(rows):
    zone = [r for r in rows if ZONE[0] <= r['s_m'] <= ZONE[1]]
    return dict(cav_count=sum(r['is_cav'] for r in zone),density=len(zone)/.52632,
                ramp=any(r['lane'] == FRAME.merge_lane for r in zone))


def contexts(screens):
    out = []
    for screen in screens:
        for clock,state in screen['states'].items():
            if state['context']['members']:
                out.append(dict(world=screen['world'],t0=int(clock),**state))
    for cell in range(12):
        subset = [r for r in out if r['world']['cell'] == cell]
        if not subset:
            continue
        limits = {k:float(np.quantile([r['features'][k] for r in subset],.75))
                  for k in ('cav_count','density')}
        for r in subset:
            f = r['features']
            r['dense'] = bool(f['ramp'] and f['cav_count'] >= 2 and
                              all(f[k] >= limits[k] for k in limits))
            r['quantiles'] = limits
    return out


def select(screens, generator, prior=None):
    """Every eligible world represented first; remaining draws 2:1 dense/ordinary."""
    candidates = contexts(screens)
    seeds = sorted({r['world']['seed'] for r in candidates})
    if len(seeds) < 12:
        raise ValueError('fewer than 12 eligible TRAIN worlds')
    # Prior is a signed TRAIN-only contextual fit; no action family enters the policy.
    if prior is not None:
        if prior.get('scope') != 'TRAIN' or set(prior['coefficients'])-{'cav_count','density','ramp'}:
            raise ValueError('prior must be observable TRAIN-only coefficients')
        if not prior.get('source_sha256') or prior.get('fit_worlds') is None:
            raise ValueError('prior provenance missing')
    def draw(pool):
        weights = np.ones(len(pool))
        if prior is not None:
            scores = np.asarray([sum(r['features'][k]*v for k,v in prior['coefficients'].items())
                                 for r in pool])
            # Smooth priorities retain support on every ordinary observable state.
            scores -= scores.max()
            weights = np.exp(np.maximum(scores,-20))
        return pool[int(generator.choice(len(pool),p=weights/weights.sum()))]
    selected = [draw([r for r in candidates if r['world']['seed'] == seed]) for seed in seeds]
    dense = [r for r in candidates if r['dense']]
    ordinary = [r for r in candidates if not r['dense']]
    for i in range(TARGET_WINDOWS-len(selected)):
        pool = dense if i%3 != 2 else ordinary
        selected.append(draw(pool or candidates))
    return selected,dict(worlds=len(seeds),dense_available=len(dense),ordinary_available=len(ordinary),
                         dense_selected=sum(r['dense'] for r in selected),
                         prior_sha256=digest(prior) if prior is not None else None)
