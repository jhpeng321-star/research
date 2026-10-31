"""Complete-panel, preregistered world-cluster analysis. Never an inference input."""
import json

from .design import BUDGET, CONTROLLERS, measurement_indices, worlds
from .episode import ARMS
from .storage import canonical, write


def primary(episodes, regrets, fixed_on_wm):
    import numpy as np
    # Arrays [world, arm/decision/controller]. Reselect the population-best fixed
    # inside each paired world resample; no post-hoc choice of a comparison clock.
    fixed = np.asarray(episodes,float)
    delta = np.asarray(regrets,float)[:,0]-np.asarray(regrets,float)[:,1]
    contrasts = np.asarray(fixed_on_wm,float)
    def score(ix):
        best = int(fixed[ix].mean(0).argmin())
        return np.array([delta[ix].mean(),contrasts[ix,best].mean()]),best
    estimate,best = score(np.arange(len(fixed)))
    draws = np.random.default_rng(0).integers(0,len(fixed),(2000,len(fixed)))
    boot = np.stack([score(ix)[0] for ix in draws])
    all_fixed_boot = contrasts[draws].mean(1)
    return dict(best_fixed=CONTROLLERS[best],worlds=len(fixed),bootstrap=2000,seed=0,
        wm_minus_bb_regret60=dict(mean=float(estimate[0]),ci95=np.quantile(boot[:,0],[.025,.975]).tolist()),
        wm_minus_best_fixed_regret60=dict(mean=float(estimate[1]),ci95=np.quantile(boot[:,1],[.025,.975]).tolist()),
        wm_minus_each_fixed_regret60={c:dict(mean=float(contrasts[:,j].mean()),
            ci95=np.quantile(all_fixed_boot[:,j],[.025,.975]).tolist()) for j,c in enumerate(CONTROLLERS)})


def rank(values):
    # Exact midranks, matching Spearman's definition; numerical tie EPS belongs
    # to controller selection, not a new rank binning convention.
    return [1+sum(y<x for y in values)+.5*(sum(y==x for y in values)-1) for x in values]


def cell_interval(values):
    """Within-cell world resampling; n1 degeneracy is disclosed, not hidden."""
    import numpy as np
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    if not len(values):return None
    draws=np.random.default_rng(0).integers(0,len(values),(2000,len(values)))
    return np.quantile(values[draws].mean(1),[.025,.975]).tolist()


def summarize(root):
    import numpy as np
    panel = worlds('main')
    binding = json.loads((root/'BINDING.json').read_text()) if (root/'BINDING.json').exists() else {}
    total = binding.get('total_sim_s',478410)
    retained = binding.get('retained_superseded_sim_s',0)
    recovery = total in (499260,499680) and binding.get('authorization')=='MAIN23:25'
    abandoned = binding.get('retained_abandoned_sim_s',0)
    valid_recovery = recovery and retained==12450 and (total,abandoned) in ((499260,11100),(499680,17820))
    if (total,retained) not in ((478410,0),(490860,12450)) and not valid_recovery:
        raise ValueError('unregistered continuation accounting')
    rows = {a:[] for a in ARMS}
    ledger = [json.loads(x) for x in (root/'accounting.jsonl').read_text().splitlines()]
    closes = [json.loads(p.read_text()) for p in root.glob('jobs/*/CLOSE.json')]
    if sum(x['sim_s'] for x in ledger) != total or 47010+retained+abandoned+sum(x['actual_sim_s'] for x in closes) != total:
        raise ValueError('complete-panel charged/actual accounting required')
    if any(x['actual_sim_s']!=x['charged_sim_s'] for x in closes):
        raise ValueError('job actual/charged mismatch')
    expected_budget = {a:BUDGET[a] for a in ARMS}
    if recovery:
        expected_budget.update(wm=175560,bb=180540,random=11700)
        if total==499680:expected_budget.update(wm=178380,bb=184440,random=5400)
        if binding['per_arm'] != expected_budget:
            raise ValueError('recovery per-arm plan mismatch')
    for arm in ARMS:
        if sum(x['sim_s'] for x in ledger if x['id'].startswith(arm+'_')) != expected_budget[arm]:
            raise ValueError('per-arm accounting mismatch')
        for world in panel:
            dropped = range(969118,969128) if total==499680 else (969125,969126,969127)
            if recovery and arm=='random' and world['seed'] in dropped:
                rows[arm].append(None)
                continue
            p = root/'jobs'/f'{arm}_s{world["seed"]}_episode'
            r = json.loads((p/'RESULT.json').read_text())
            if r['world'] != world or len(r['decisions']) != 10:
                raise ValueError('episode pairing mismatch')
            rows[arm].append(r)
    for i in range(16):
        if len({canonical(rows[a][i]['decisions'][0]['prefix']) for a in ARMS if rows[a][i] is not None}) != 1:
            raise ValueError('paired warmup mismatch')
    fixed = np.array([[rows[a][i]['J600']['team_delay_vs'] for a in CONTROLLERS] for i in range(16)])
    regrets = np.zeros((16,2));contrasts = np.zeros((16,5))
    decision_rows = []
    for ai,arm in enumerate(('wm','bb')):
        for i,world in enumerate(panel):
            measured = [d for d in rows[arm][i]['decisions'] if 'branches' in d]
            if [d['j'] for d in measured] != measurement_indices(world['seed']):
                raise ValueError('registered measurement subset mismatch')
            for d in measured:
                costs = np.array([d['branches'][c]['60']['team_delay_vs'] for c in CONTROLLERS])
                chosen = CONTROLLERS.index(d['pick'])
                regrets[i,ai] += (costs[chosen]-costs.min())/3
                if arm == 'wm':contrasts[i] += (costs[chosen]-costs)/3
                pred = [d['predicted_costs'][c] for c in CONTROLLERS]
                for clock in ('60','90'):
                    true = [d['branches'][c][clock]['team_delay_vs'] for c in CONTROLLERS]
                    pr,tr = np.array(rank(pred)),np.array(rank(true))
                    rho = float(np.corrcoef(pr,tr)[0,1]) if pr.std()>0 and tr.std()>0 else None
                    for c in CONTROLLERS:
                        if d['branches'][c]['prefix'] != d['prefix']:
                            raise ValueError('branch/live prefix mismatch')
                    pdir = np.sign(np.where(np.abs(np.array(pred)-pred[0])<=1e-6,0,np.array(pred)-pred[0]))
                    tdir = np.sign(np.where(np.abs(np.array(true)-true[0])<=1e-6,0,np.array(true)-true[0]))
                    eligible = (pdir[1:]!=0)&(tdir[1:]!=0)
                    decision_rows.append(dict(seed=world['seed'],arm=arm,j=d['j'],clock=clock,
                        pick=d['pick'],regret=true[chosen]-min(true),spearman=rho,
                        direction_correct=int(((pdir[1:]==tdir[1:])&eligible).sum()),
                        direction_n=int(eligible.sum()),direction_tied=int((~eligible).sum())))
    values = {a:np.array([r['J600']['team_delay_vs'] if r is not None else np.nan for r in rows[a]]) for a in ARMS}
    secondary = {}
    for a in ('wm','bb','random'):
        targets = {c:values[c] for c in CONTROLLERS}
        targets['best_fixed_in_hindsight'] = fixed.min(1)
        valid = np.isfinite(values[a])
        arm_draws = np.random.default_rng(0).integers(0,int(valid.sum()),(2000,int(valid.sum())))
        secondary[a] = {b:dict(mean=float((values[a]-v)[valid].mean()),worlds=int(valid.sum()),
            ci95=np.quantile((values[a]-v)[valid][arm_draws].mean(1),[.025,.975]).tolist()) for b,v in targets.items()}
    for a in ('wm','bb'):
        valid = np.isfinite(values['random'])
        rd = np.random.default_rng(0).integers(0,int(valid.sum()),(2000,int(valid.sum())))
        diff = (values[a]-values['random'])[valid]
        secondary[a]['random'] = dict(mean=float(diff.mean()),worlds=int(valid.sum()),ci95=np.quantile(diff[rd].mean(1),[.025,.975]).tolist())
    safety = {a:{k:sum(r['J600'][k] for r in rows[a] if r is not None) for k in
                   ('collisions','teleports','ttc_below_3_vs','hard_brake')} for a in ARMS}
    cells = []
    for p,d in sorted({(w['penetration'],w['demand']) for w in panel}):
        ix = [i for i,w in enumerate(panel) if (w['penetration'],w['demand'])==(p,d)]
        cells.append(dict(penetration=p,demand=d,n=len(ix),J600={a:float(np.nanmean(v[ix])) if np.isfinite(v[ix]).any() else None for a,v in values.items()},
                          arm_n={a:int(np.isfinite(v[ix]).sum()) for a,v in values.items()},
                          J600_ci95={a:cell_interval(v[ix]) for a,v in values.items()},
                          regret60_ci95={a:cell_interval(regrets[ix,j]) for j,a in enumerate(('wm','bb'))},
                          regret60={a:float(regrets[ix,j].mean()) for j,a in enumerate(('wm','bb'))}))
    accuracy = {}
    for arm in ('wm','bb'):
        accuracy[arm] = {}
        for clock in ('60','90'):
            selected = [r for r in decision_rows if r['arm']==arm and r['clock']==clock]
            rho = [r['spearman'] for r in selected if r['spearman'] is not None]
            n = sum(r['direction_n'] for r in selected)
            accuracy[arm][clock] = dict(spearman_mean=float(np.mean(rho)) if rho else None,
                spearman_n=len(rho),direction_accuracy=sum(r['direction_correct'] for r in selected)/n if n else None,
                direction_n=n,direction_tied=sum(r['direction_tied'] for r in selected))
    result = dict(primary=primary(fixed,regrets,contrasts),secondary_episode_paired=secondary,
        episode_mean={a:float(np.nanmean(v)) for a,v in values.items()},safety=safety,cells=cells,
        episode_n={a:int(np.isfinite(v).sum()) for a,v in values.items()},
        switching_mean={a:sum(r['switching_frequency'] for r in rows[a] if r is not None)/sum(r is not None for r in rows[a]) for a in ARMS},
        measured_decisions=decision_rows,ranking_direction=accuracy,charged_sim_s=total,actual_sim_s=total,
        retained_superseded_sim_s=retained,retained_abandoned_sim_s=abandoned,
        caveats=['predicted current-belief fleet objective vs true fleet+source evaluation',
                 'WM/BB on-policy decision states differ; world/clock paired',
                  'two exploratory primary contrasts; no multiplicity adjustment',
                  'single-world cells have degenerate empirical bootstrap intervals, not population uncertainty bounds',
                  'native CAV plant/lane forecast approximate; entries OFF and mix dormant'])
    write(root/'RESULTS.json',result)
    return result
