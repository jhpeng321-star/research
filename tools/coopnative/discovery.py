"""Evaluation-only detector-selected exact twins, never called by PPO.update."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from .evaluation import behaviour, exact_credit
from .storage import write, read, ledger, reserve
from .protocol import CAPS


def discover(root,binding,policy,records):
    if (root/'DISCOVERY.json').exists():
        return read(root/'DISCOVERY.json')
    from .runner import admission
    from .plant import episode
    groups, omissions = [], []
    remaining = CAPS['discovery']-sum(r['sim_s'] for r in ledger(root) if r['phase']=='discovery')
    for record in sorted(records,key=lambda r:r['world']['seed']):
        detected = behaviour(record,root/'jobs'/record['id']/'observable.jsonl.gz')['detectors']['events']
        # Fixed temporal grouping: union detected participants within the existing action window.
        windows = {}
        for event in detected:
            t = int(event['start_s']//10)*10
            if 300 <= t <= 600:
                windows.setdefault(t,set()).update(event['members'])
        for t,members in sorted(windows.items()):
            step = next(s for s in record['steps'] if s['time_s']==t)
            members = sorted(members & set(step['sample']['context']['members']))
            if len(members)<2:
                continue
            cost=(len(members)+2)*(t+600)
            if cost>remaining:
                omissions.append(dict(world=record['world'],time_s=t,members=members,cost=cost))
                continue
            groups.append(dict(world=record['world'],time_s=t,members=members,record=record,cost=cost))
            remaining-=cost
            break  # First affordable complete group/world, never selected by D_G or C_G.
    plan=dict(groups=groups,omissions=omissions)
    write(root/'DISCOVERY_PLAN.json',plan)
    outputs=[]
    for index,group in enumerate(groups):
        t,members,record=group['time_s'],group['members'],group['record']
        branch=dict(time_s=t,members=members,
                    samples={str(s['time_s']):s['sample'] for s in record['steps'] if s['time_s']<=t},
                    prefixes={str(s['time_s']):s['prefix'] for s in record['steps'] if s['time_s']<=t})
        jobs=[]
        for arm,active in [('baseline',[]),('joint',members)]+[(f'solo{i}',[vid]) for i,vid in enumerate(members)]:
            ident=f'discovery{index:03}_{arm}'
            jobs.append(dict(id=ident,path=str(root/'jobs'/ident),world=group['world'],role='dev',arm='learned',
                approval_line=binding['approval_line'],
                weights={k:v.detach().cpu() for k,v in policy.state_dict().items()},runtime=binding['runtime'],
                noise_seed=0,sim_s=t+600,stop=t+600,branch=dict(branch,active=active)))
        reserve(root,jobs,'discovery')
        admission(binding)
        with ProcessPoolExecutor(max_workers=binding['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(episode,jobs))
        credits={str(h):exact_credit(results[0]['metrics']['branch'+str(h)]['team_delay_vs'],
            results[1]['metrics']['branch'+str(h)]['team_delay_vs'],
            [r['metrics']['branch'+str(h)]['team_delay_vs'] for r in results[2:]]) for h in (120,300,600)}
        outputs.append(dict(world=group['world'],time_s=t,members=members,credit=credits,
                            ids=[j['id'] for j in jobs],prefix_pass=True))
    result=dict(groups=outputs,omissions=omissions,selection='first chronological affordable detector group/world; no outcome exclusions')
    write(root/'DISCOVERY.json',result)
    return result
