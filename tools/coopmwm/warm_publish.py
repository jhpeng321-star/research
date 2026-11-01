"""Append-only publication of warm TRAIN/dev receipts, never final-panel data."""
import json
from datetime import datetime,timezone
from . import ops
from .storage import read,write
from .protocol import checked,digest
from .warm_deploy import ROOT,PRIOR


def number(value):return 'NA' if value is None else f'{value:.4g}'


def describe(name,value):
    out=[f'\n### Warm continuation — {name}\n']
    if '_UPDATE_' in name:
        r=value['proposal_routing'];out.append(f'- {value["arm"]} u{value["update"]}: queries/accepted/executed '
            f'{r.get("queries",0)}/{r.get("accepted",0)}/{r.get("executed",0)}; '
            f'wall {number(value.get("wall_s"))}s; next beta {number(value.get("trust",{}).get("beta"))}. SUMO-only reward.\n')
        out.append(f'- Executions by type/regime: `{json.dumps(r.get("by_bucket",{}),sort_keys=True)}`.\n')
    elif '_CHECK_' in name or '_DEV_' in name:
        for contrast,c in value.get('comparisons',{}).items():
            metrics=c['metrics'];parts=[]
            for h in ('600','900'):
                m=metrics[h+'_team_delay_vs'];base=None
                if contrast.endswith('MAPPO') and value.get('cells'):
                    values=[x['mappo'][h]['team_delay_vs'] for x in value['cells'].values() if 'mappo' in x]
                    if values:base=sum(values)/len(values)
                percent=f'; delay reduction {-100*m["mean"]/base:.3g}%' if base else ''
                parts.append(f'J{h} Δ={m["mean"]:.3f} [{m["ci95"][0]:.3f},{m["ci95"][1]:.3f}] veh-s{percent}')
            out.append(f'- {contrast}: '+ '; '.join(parts)+'. Negative Δ is better; 95% whole-world bootstrap.\n')
        out.append('- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.\n')
    elif '_VIABILITY_' in name:
        v=value['viability'];overall=v['overall']
        out.append(f'- {value["arm"]} u{value["update"]}: executed-plan vs own-action SUMO H300 critic residual '
            f'{number(overall.get("mean"))} veh-s, 95% CI `{overall.get("ci95")}`; '
            f'new/known-like plans {v.get("new_plans",0)}/{v.get("inherited_plans",0)}.\n')
        out.append('- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.\n')
    elif name.startswith('ACCEL_EXACT_PLAN_'):
        out.append(f'- Exact executed-plan twins: `{json.dumps(value.get("credits",{}),sort_keys=True)}`.\n')
    else:out.append(f'- `{json.dumps(value,sort_keys=True)}`\n')
    out.append(f'- Receipt: `coopmwm/warm_live/{name}`.\n')
    return ''.join(out)


def publish():
    local=ops.EVIDENCE/'warm_live';local.mkdir(exist_ok=True)
    known=[p.name for p in local.glob('*.json')]
    result=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.accel_storage import read,charged
known=set(json.loads({json.dumps(known)!r}));out={{}}
patterns=('*_UPDATE_*.json','*_CHECK_*.json','*_DEV_*.json','*_VIABILITY_*.json','ACCEL_EXACT_PLAN_*.json','WARM_*_FAILED.json','WARM_*_PAUSED.json','STOP.json')
for pattern in patterns:
 for p in r.glob(pattern):
  if '_RECORDS_' in p.name or p.name in known:continue
  out[p.name]=read(p)
p=pathlib.Path({PRIOR!r})/'A_CHECK_004.json';key='A_CHECK_PRESERVED_004.json'
if key not in known and p.is_file():out[key]=read(p)
print(json.dumps(dict(receipts=out,charged_sim_s=charged(r))))''',timeout=45))
    doc=checked(ops.LOCAL/'docs/paper_results/COOPMWM.md');current=doc.read_text(encoding='utf-8')
    changed=[];text=''
    for name,value in sorted(result['receipts'].items()):
        target=local/name;write(target,value);changed.append(str(target.relative_to(ops.LOCAL)))
        marker=f'<!-- warm-receipt:{name}:{digest(value)} -->'
        if marker not in current:text+=describe(name,value)+marker+'\n'
    if text:
        with doc.open('a',encoding='utf-8') as f:
            f.write('\n## Warm live publication — '+datetime.now(timezone.utc).isoformat()+'\n')
            f.write(f'- Conservatively charged {result["charged_sim_s"]}/6500000 sim-s.\n'+text)
        changed.append(str(doc.relative_to(ops.LOCAL)))
    print(json.dumps(dict(published=len(result['receipts']),charged_sim_s=result['charged_sim_s'])))
    return changed


if __name__=='__main__':publish()
