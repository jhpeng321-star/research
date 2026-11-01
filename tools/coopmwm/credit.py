"""Member-reversion algebra, fixed ramp, and data-fitted scale (CPU only)."""
import math
import numpy as np


def algebra(joint, reverted, all_reverted):
    if not reverted or not all(math.isfinite(x) for x in (joint, all_reverted, *reverted)):
        raise ValueError('complete finite member-reversion twins required')
    di = [float(x-joint) for x in reverted]
    dg = float(all_reverted-joint)
    return dict(D_i=di, D_G=dg, C_G=dg-sum(di))


def ramp(update):
    if type(update) is not int or update < 1:
        raise ValueError('positive integer update required')
    return min(1., (update-1)/8.)


def bonus(credit, update, scale):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('nonnegative learned scale required')
    return ramp(update)*scale*(sum(credit['D_i'])+.5*credit['C_G'])


def paired(futures):
    """Rows are K CRN futures, with joint/member-reversions/all-reversion."""
    if len(futures) != 8:
        raise ValueError('K8 required')
    credits = []
    for f in futures:
        if len({v['draw_id'] for v in f.values()}) != 1:
            raise ValueError('unpaired CRN')
        names = sorted(k for k in f if k.startswith('revert:'))
        credits.append(algebra(f['joint']['team_delay'],
                              [f[k]['team_delay'] for k in names],f['all']['team_delay']))
    return dict(D_G=float(np.mean([c['D_G'] for c in credits])),
                C_G=float(np.mean([c['C_G'] for c in credits])),
                D_i=np.mean([c['D_i'] for c in credits],axis=0).tolist())


def wilson(hits, n):
    if not n:
        return [0.,1.]
    z=1.959963984540054; p=hits/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-w,c+w]


def calibration(records, draws=10000):
    """Whole-world bootstrap; exact outcomes used only here, never in PPO."""
    if len(records)!=12 or len({r['world']['cell'] for r in records})!=12:
        raise ValueError('complete registered twelve-cell calibration required')
    def vectors(rs):
        return [v for r in rs for v in [*r['wm']['D_i'],r['wm']['C_G']]], [
            v for r in rs for v in [*r['exact']['D_i'],r['exact']['C_G']]]
    def fit(rs):
        x,y=map(np.asarray,vectors(rs)); den=float(x@x)
        return max(0.,float(x@y)/den) if den else 0.
    result={}
    g=np.random.default_rng(20261101)
    indices=g.integers(0,len(records),(draws,len(records)))
    for kind in ('D_i','C_G'):
        pairs=[(x,y) for r in records for x,y in (zip(r['wm'][kind],r['exact'][kind])
               if kind=='D_i' else [(r['wm'][kind],r['exact'][kind])])]
        nonzero=[(x,y) for x,y in pairs if x!=0 and y!=0]
        hits=sum((x>0)==(y>0) for x,y in nonzero); n=len(nonzero)
        def agreement(rs):
            ps=[(x,y) for r in rs for x,y in (zip(r['wm'][kind],r['exact'][kind])
                if kind=='D_i' else [(r['wm'][kind],r['exact'][kind])]) if x!=0 and y!=0]
            return sum((x>0)==(y>0) for x,y in ps)/len(ps) if ps else 0.
        boot=[agreement([records[i] for i in ix]) for ix in indices]
        result[kind]=dict(sign_agreement=hits/n if n else None,n=n,ties=len(pairs)-n,
                          ci95=np.quantile(boot,[.025,.975]).tolist(),wilson_ci95=wilson(hits,n))
    samples=[fit([records[i] for i in ix]) for ix in indices]
    scale=fit(records)
    result.update(scale=scale,scale_ci95=np.quantile(samples,[.025,.975]).tolist(),
                  pass_=scale>0 and all(result[k]['ci95'][0]>.5 for k in ('D_i','C_G')),
                  windows=12,bootstrap='whole-world',seed=20261101)
    return result


def multihorizon(records,draws=10000):
    from .protocol import HORIZONS
    checks={str(h):calibration([dict(world=r['world'],wm=r['wm'][str(h)],
                exact=r['exact'][str(h)]) for r in records],draws) for h in HORIZONS}
    eligible=[str(h) for h in HORIZONS if checks[str(h)]['pass_']]
    # Both individual and interaction signs must clear chance; use the weaker
    # measured agreement, then normalize across ALL admitted horizons.
    raw={h:min(checks[h][k]['sign_agreement'] for k in ('D_i','C_G')) for h in eligible}
    weights={h:v/sum(raw.values()) for h,v in raw.items()}
    primary='300' if '300' in eligible else (max(eligible,key=lambda h:raw[h]) if eligible else None)
    return dict(pass_=bool(eligible),horizons=checks,weights=weights,primary_horizon_s=int(primary) if primary else None,
                rule='both world-bootstrap sign CI lower>.5; weights=min(D_i,C_G) measured agreement normalized',
                windows=len(records),bootstrap='whole-world',seed=20261101)


def multibonus(credits,update,gate):
    if not gate['pass_']:raise ValueError('unqualified WM credit')
    return sum(w*bonus(credits[h],update,gate['horizons'][h]['scale']) for h,w in gate['weights'].items())


def noise(futures):
    """Conditional WM estimates in veh-s², same action and K8 futures.

    A frozen critic baseline is constant within a window, so its subtraction
    leaves joint-return advantage variance unchanged. This is model-estimated
    conditional noise, not an empirical SUMO PPO variance measurement.
    """
    cs=[algebra(f['joint']['team_delay'],[f[k]['team_delay'] for k in sorted(f) if k.startswith('revert:')],
                f['all']['team_delay']) for f in futures]
    return dict(team_return_advantage_variance=float(np.var([-f['joint']['team_delay'] for f in futures],ddof=1)),
                paired_credit_variance=float(np.var([sum(c['D_i'])+.5*c['C_G'] for c in cs],ddof=1)),
                unit='veh-s squared',K=len(futures),estimator='conditional WM K8; fixed critic baseline')
