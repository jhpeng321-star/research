"""Paired WM labels and matched exact diagnostics (no fitting on exact outcomes)."""
from statistics import fmean
import math

from tools.coopbandit.protocol import observable, option
from tools.coopplan.planner import decide
from tools.coopplan.predictor import Prediction, Outcome

OPTIONS = ('O0', 'O1', 'O2', 'O3', 'O4')


def actions(rows, setting):
    return [a for name in OPTIONS for a in [option(name, rows, setting)]
            if name == 'O0' or a['available']]


def label(record, futures):
    candidates = actions(record['rows'], record['setting'])
    prediction = Prediction({a: tuple(Outcome(**f) for f in fs) for a, fs in futures.items()},
                            False, 'WM-PHYS-FULL-v4')
    prediction.validate(candidates, 8)
    paired = decide(prediction)
    means = paired['expected_delay']
    effects = {a: means['O0']-v for a,v in means.items()}
    chosen = max(effects, key=lambda a: (effects[a], -OPTIONS.index(a)))
    return dict(state_id=record['state_id'], world_id=record['world_id'],
                role=record['role'], time_s=record['time_s'], setting=record['setting'],
                horizon_s=record['horizon_s'], features=observable(record['rows'],record['time_s'],record['setting']),
                available=[a in effects for a in OPTIONS], effects=effects,
                selected=chosen, paired_gate=paired, K=8,
                future_ids=[f.draw_id for f in prediction.actions['O0']])


def ranks(values):
    return [sum(other < v for other in values) + (sum(other == v for other in values)-1)/2
            for v in values]


def correlation(x, y):
    mx, my = fmean(x), fmean(y)
    numerator = sum((a-mx)*(b-my) for a,b in zip(x,y))
    denominator = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return numerator/denominator if denominator else None


def compare(wm, exact):
    if (wm['state_id'], wm['horizon_s']) != (exact['state_id'], exact['horizon_s']):
        raise ValueError('exact labels must match decision state and horizon')
    if set(wm['effects']) != set(exact['effects']):
        raise ValueError('diagnostic requires the same complete available candidate set')
    names = [a for a in OPTIONS if a in wm['effects']]
    y, p = [exact['effects'][a] for a in names], [wm['effects'][a] for a in names]
    if not all(math.isfinite(v) for v in y+p) or y[0] != 0 or p[0] != 0:
        raise ValueError('finite effects and exact O0 zero required')
    best = max(names, key=lambda a: (exact['effects'][a], -OPTIONS.index(a)))
    choice, gated = wm['selected'], wm['paired_gate']['selected']
    sign = [(p[i] > 0)-(p[i] < 0) == (y[i] > 0)-(y[i] < 0) for i in range(1,len(names))]
    return dict(state_id=wm['state_id'], world_id=wm['world_id'], horizon_s=wm['horizon_s'],
                candidates=len(names), choice=choice, exact_choice=best,
                agreement=choice == best, gated_agreement=gated == best,
                exact_saving=exact['effects'][choice], exact_gated_saving=exact['effects'][gated],
                oracle_saving=exact['effects'][best], regret=exact['effects'][best]-exact['effects'][choice],
                sign_matches=sum(sign), sign_count=len(sign),
                within_state_spearman=correlation(ranks(p),ranks(y)) if len(names)>1 else None,
                predicted_D=p, exact_D=y)


def quality(rows, expected):
    if not rows:
        raise ValueError('quality must precede distillation; no matched states')
    return dict(states=len(rows), expected_states=expected, worlds=len({r['world_id'] for r in rows}),
                choice_agreement=fmean(r['agreement'] for r in rows),
                gated_choice_agreement=fmean(r['gated_agreement'] for r in rows),
                exact_chosen_saving=fmean(r['exact_saving'] for r in rows),
                exact_oracle_saving=fmean(r['oracle_saving'] for r in rows),
                exact_regret=fmean(r['regret'] for r in rows),
                sign_agreement=(sum(r['sign_matches'] for r in rows)/sum(r['sign_count'] for r in rows)
                                if sum(r['sign_count'] for r in rows) else None),
                mean_within_state_spearman=(fmean(r['within_state_spearman'] for r in rows
                    if r['within_state_spearman'] is not None)
                    if any(r['within_state_spearman'] is not None for r in rows) else None),
                meaning='conditional exact-state diagnostic; not a live policy comparison')
