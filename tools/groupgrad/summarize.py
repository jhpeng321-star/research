"""Compact production-update evidence; no traffic replay or model execution."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def distribution(xs):
    xs = [float(x) for x in xs if x is not None]
    return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'max':max(xs)} if xs else None


def main():
    logs = json.loads((ROOT/'logs.json').read_text())
    result = {}
    for phase, doc in logs.items():
        rows = doc['rows']
        audit = [r for r in rows if r.get('cf_term_grad_norms')]
        cover = [r.get('cf_coverage',{}) for r in rows]
        result[phase] = dict(n=len(rows),first=rows[0],last=rows[-1],
            scale=distribution([r.get('return_scale') for r in rows]),
            actor_grad_norm=distribution([r.get('actor_grad_norm') for r in rows]),
            group_update_fraction=sum(x.get('n_groups',0)>0 for x in cover)/len(rows),
            team_update_fraction=sum(x.get('team_groups',0)>0 for x in cover)/len(rows),
            audited_terms=audit,
            synergy_to_all_unit_norm=distribution([r['cf_term_grad_norms'].get('coop',0)/
                r['cf_term_grad_norms']['all'] for r in audit if r['cf_term_grad_norms'].get('all',0)>0]),
            actor_credit_components={key:distribution([r.get('actor_grad_norms',{}).get(key) for r in rows])
                                     for key in ('reward','critic','uncertainty','constraint','entropy')})
    (ROOT/'log_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:v for x,v in d.items() if x not in ('first','last','audited_terms')}
                      for k,d in result.items()},indent=2))
    for phase,d in result.items():
        print(phase, 'updates',d['first']['update'],d['last']['update'],
              'first weights',d['first'].get('cf_weights'),'last weights',d['last'].get('cf_weights'))
    canonical = [r for r in logs['c01']['rows'] if r['update']<=170]+logs['c01.a3']['rows']
    assert sorted(r['update'] for r in canonical)==list(range(31,181))
    audit = [r for r in canonical if r.get('cf_term_grad_norms',{}).get('all',0)>0]
    small = dict(n_updates=len(canonical),audited_updates=[r['update'] for r in audit],
        synergy_to_all_unit_norm=distribution([r['cf_term_grad_norms']['coop']/
                                               r['cf_term_grad_norms']['all'] for r in audit]),
        n_group_updates=sum(r.get('cf_coverage',{}).get('n_groups',0)>0 for r in canonical),
        n_team_updates=sum(r.get('cf_coverage',{}).get('team_groups',0)>0 for r in canonical))
    result['canonical_c01']=small
    (ROOT/'log_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CANONICAL_C01',json.dumps(small))


if __name__=='__main__':
    main()
