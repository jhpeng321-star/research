"""Prepare the reviewed corrected design without releasing HOLD or running SUMO."""
import json
from pathlib import Path
from tools.coopaudit.accounting import write
from tools.coopaudit.design import select_group, sha, worlds, MAPPO_SHA
from tools.coopaudit.controllers import wall_roster

EVIDENCE = Path(__file__).resolve().parent/'evidence'
UPSTREAM = Path('E:/agent-work/c4-sumocf')


def main():
    inventory = json.loads((EVIDENCE/'RECOVERY_INVENTORY.json').read_text())
    passed = UPSTREAM/'tools/sumocf/evidence/ARTIFACT_PASS.json'
    doc = json.loads(passed.read_text())
    assert doc['status']=='PASS' and doc['branch_mode']=='rerun' and doc['saved_state_rejected']
    assert (doc['selected_group'],doc['selected_window_s']) == (2,60)
    policy_sha = 'e4b51f64fba54249792237131c2a10a6cbc5d1286c386c2c8dc9374b2d398d8a'
    assert inventory['checkpoint_sha256'] == policy_sha
    panels=[]
    for w,item in zip(worlds(),inventory['census']):
        frozen=item['frozen']
        assert frozen['world']==w
        group=select_group(frozen['observations'],2)
        panels.append(dict(world=w,group=group,wall_roster=wall_roster(frozen['observations'],group),
                           census_sha256=item['sha256'],
                           initial_state_sha256=frozen['initial_state_sha256'],crn_sha256=frozen['crn_sha256'],
                           charge_sim_s=900*(2+3*(len(group)+1)) if group else 0))
    selected=dict(group_size=2,window_s=60,probe_grid=7,nominal_probe_rho=.5,
                  branch_mode='full_same_seed_prefix_rerun',namespace='v2',
                  source_commit='658b09a3',sumocf_table_sha256=sha(UPSTREAM/'tools/sumocf/evidence/ARTIFACT_PROBE.json'),
                  artifact_pass_sha256=sha(passed),artifact_document_sha256=sha(UPSTREAM/'docs/paper_results/ARTIFACT_CHECK.md'),
                  coordination_source_sha256=sha(UPSTREAM/'tools/sumocf/coordination.py'),
                  checkpoint=inventory['checkpoint'],checkpoint_sha256=policy_sha,checkpoint_update=15,
                  checkpoint_rule='final completed update, fixed before fresh audit outcomes; no checkpoint selection',
                  mappo_sha256=MAPPO_SHA,controllers=['mappo','wall','sumocf'],panel=panels,
                  retained_charged_sim_s=21480,canary_charge_sim_s=14400,
                  main_charge_sim_s=sum(p['charge_sim_s'] for p in panels),
                  storage_root='/root/autodl-tmp/coopaudit20261101_v2',max_workers=6)
    selected['total_admitted_sim_s']=21480+14400+selected['main_charge_sim_s']
    assert selected['total_admitted_sim_s']<=500000
    write(EVIDENCE/'CORRECTED_DESIGN.json',selected)
    print(json.dumps({k:v for k,v in selected.items() if k!='panel'},indent=1))
    print(json.dumps(dict(eligible=sum(bool(p['group']) for p in panels),
                         eligible_high_share=sum(bool(p['group']) and p['world']['cell']<4 for p in panels),
                         wall_feasible=sum(bool(p['wall_roster']) for p in panels))))


if __name__=='__main__':
    main()
