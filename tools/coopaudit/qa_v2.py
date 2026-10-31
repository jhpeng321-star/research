"""Independent final numerical/provenance QA on the complete immutable export."""
import json
from pathlib import Path
import numpy as np
from tools.coopaudit.accounting import write
from tools.coopaudit.design import sha


def main():
    evidence=Path(__file__).resolve().parent/'evidence'
    root=evidence/'v2_complete'
    summary=json.loads((evidence/'v2_analysis/SUMMARY.json').read_text())
    collection=json.loads((root/'COLLECTION.json').read_text())
    for p in collection['files']:
        assert sha(root/p['path'])==p['sha256']
    assert sha(root/'receipts.tgz')==collection['sha256']
    assert collection['scientific_files']==json.loads((evidence/'V2_CANARY_VERIFIED.json').read_text())['scientific_files']
    inventory=json.loads((evidence/'RECOVERY_INVENTORY.json').read_text())
    eligible=[]
    for item in inventory['census']:
        old=item['frozen']
        w=old['world']
        observable=[r for r in old['observations'] if r['is_cav'] and not r['junction'] and
                    r['lane'] in (0,1,2) and -150<=r['s']<=75.44]
        observable.sort(key=lambda r:(abs(r['s']),r['vid']))
        group=[r['vid'] for r in observable[:2]] if len(observable)>=2 else []
        r=json.loads((root/'jobs'/f"v2_s{w['seed']}"/'result.json').read_text())
        assert r['frozen']['group']==group
        if group:
            eligible.append((w,r))
    assert len(eligible)==21
    checked_contrasts=0
    for outcome in ('300','600','local'):
        columns=[]
        for c in ('mappo','wall','sumocf'):
            def cost(r,key):
                return r['effects'][key]['local_cohort_delay_vs'] if outcome=='local' else r['metrics'][key][outcome]['team_delay_vs']
            # Independent four-cost algebra, not the production interaction helper.
            col=np.array([cost(r,c+'_ONLY:0')+cost(r,c+'_ONLY:1')-
                          cost(r,c+'_GROUP')-cost(r,'native_B') for w,r in eligible])
            columns.append(col)
            reported=next(x for x in summary['summary'] if x['cohort']=='eligible21' and
                          x['controller']==c and x['outcome']==outcome and x['metric']=='C_G')
            assert np.isclose(reported['mean'],col.mean(),atol=1e-6,rtol=0)
            checked_contrasts+=len(col)
        values=np.stack(columns,1)
        rng=np.random.default_rng(20261031)
        samples=np.zeros((5000,3))
        for cell in sorted({w['cell'] for w,r in eligible}):
            block=values[[w['cell']==cell for w,r in eligible]]
            idx=rng.integers(0,len(block),size=(5000,len(block)))
            samples+=block[idx].mean(1)*len(block)/21
        for i,c in enumerate(('mappo','wall','sumocf')):
            reported=next(x for x in summary['summary'] if x['cohort']=='eligible21' and
                          x['controller']==c and x['outcome']==outcome and x['metric']=='C_G')
            lo,hi=np.quantile(samples[:,i],[.025,.975])
            assert abs(lo-reported['lo'])<1e-6 and abs(hi-reported['hi'])<1e-6
    for r in summary['summary']:
        if r['cohort']=='all28':
            assert r['mean'] is r['lo'] is r['hi'] is None and r['n_worlds']==21
    final=json.loads((evidence/'V2_FINAL_VERIFIED.json').read_text())
    assert final['ok'] and final['charged_sim_s']==summary['charged_sim_s']==243780
    assert final['actual_total_sim_s']==summary['actual_total_sim_s']==243480
    assert final['ledger_sha256']==sha(root/'accounting.jsonl')
    for safety in summary['safety'].values():
        assert safety['collisions']==safety['teleports']==safety['dropped_vehicle_steps']==0
    report=Path(__file__).resolve().parents[2]/'docs/paper_results/COOPAUDIT.md'
    assert len(report.read_text(encoding='utf-8').splitlines())<=60
    assert 'COMPLETE' in report.read_text(encoding='utf-8')
    receipt=dict(ok=True,status='PASS',additional_sumo_sim_s=0,checked_exact_credits=checked_contrasts,
                 files_verified=len(collection['files']),all_world_CI_undefined_disclosed=True,
                 eligible_world_bootstrap_independently_recomputed=True,
                 source_identity=collection['scientific_files'],summary_sha256=sha(evidence/'v2_analysis/SUMMARY.json'),
                 report_sha256=sha(report),final_process_and_budget_verified=True)
    write(evidence/'V2_QA.json',receipt)
    print(json.dumps(receipt,indent=1))


if __name__=='__main__':
    main()
