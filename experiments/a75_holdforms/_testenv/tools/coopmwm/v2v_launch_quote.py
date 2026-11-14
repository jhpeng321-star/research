"""MAIN14:06 exact three-arm schedule, with only the authorised cost fallbacks."""
from copy import deepcopy
from .v2v_budget_quote import quote


def calculate(lessons, cooperative, *, reference_reuse=False, drop_R_u8=False,
              drop_B_early=False):
    empty=dict(A=[],B=[],R=[])
    base=quote(lessons=dict(empty,B=lessons['B']),
        cooperative=dict(empty,B=cooperative['B'],R=cooperative['R']),
        reference_reuse=reference_reuse)
    b_updates=[12,24] if drop_B_early else [6,12,18,24]
    lesson_B=base['phase_costs']['lessons']['B']//24*len(b_updates)
    synergy_B=base['phase_costs']['policy_synergy']['B']//24*len(b_updates)
    synergy_R=base['phase_costs']['policy_synergy']['R']//24*2
    interim=345600-(115200 if drop_R_u8 else 0)
    phases=dict(v2v_train=2073600,v2v_interim=interim,v2v_final=345600,
        v2v_lessons=lesson_B,v2v_isolated=synergy_B+synergy_R)
    total=sum(phases.values())
    return dict(authority='MAIN2026-11-03 14:06',arms=['A-v2v','B-v2v','R-v2v'],
        common_start='SAME original MAPPO u0, zero-init V2V',first_update=1,final_update=24,
        phase_costs=phases,additional_quote_sim_s=total,cap_if_registered_sim_s=14500000+total,
        within_ceiling=total<=3000000,reference_reuse=reference_reuse,
        reference_rule='SHA-qualified identical baseline only' if reference_reuse else 'fresh reference conservatively quoted; qualified reuse can save charges',
        diagnostics=dict(A=dict(lessons=[],isolated=[]),B=dict(lessons=b_updates,isolated=b_updates),
                         R=dict(lessons=[],isolated=[12,24])),
        interim_arms=['A','B'] if drop_R_u8 else ['A','B','R'],final_arms=['A','B','R'],
        final_paired_branches=False,
        dropped=(['R-v2v u8 test'] if drop_R_u8 else [])+(['B-v2v u6/u18 lesson and isolated checks'] if drop_B_early else []),
        slate_counts=dict(B_lessons=len(lessons['B']),B_cooperative=len(cooperative['B']),R_cooperative=len(cooperative['R'])),
        growth='current cooperative slate projected at registered check updates; future growth requires phase room before any branch',
        is_full_run_upper_bound=False,phase_registration_complete=False,new_sim_s=0,new_inference=0)


def select(lessons, cooperative):
    """Cost only, not outcomes: full config -> drop R-u8 -> drop B-u6/u18."""
    attempts=[]
    for r,b in ((False,False),(True,False),(True,True)):
        q=calculate(lessons,cooperative,drop_R_u8=r,drop_B_early=b)
        attempts.append(deepcopy(q))
        if q['within_ceiling']:
            return dict(selected=q,attempts=attempts,launch_cost_gate=True,new_sim_s=0)
    return dict(selected=None,attempts=attempts,launch_cost_gate=False,
        shortfall_sim_s=attempts[-1]['additional_quote_sim_s']-3000000,new_sim_s=0)


def from_reports(reports):
    lesson=reports['lesson_B']
    if lesson['arm']!='B' or any(e['status']!='graded' for e in lesson['entries']):
        raise ValueError('complete B lesson metadata required')
    lessons=dict(B=[dict(end_s=e['new_policy_branch']['window']['end_s']) for e in lesson['entries']])
    cooperative={}
    for arm in ('B','R'):
        v=reports['synergy_'+arm]
        if v['arm']!=arm or any(e['status']!='graded' for e in v['entries']):
            raise ValueError('complete own-arm synergy metadata required')
        cooperative[arm]=[dict(end_s=e['window']['end_s'],members=len(e['group'])) for e in v['entries']]
    return select(lessons,cooperative)


def live_metadata():
    """Cost EVERY admitted cooperative bank state, even if later diagnostics failed."""
    import json
    from .ls_validity_ops import remote
    from .ls_collect_ops import ROOT, COLLECTION
    return json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,time
sys.path.insert(0,'/home/ubuntu/coopmwm20261103_compositereleasecpu1/pytools')
from tools.coopmwm.protocol import sha,digest
r=Path({ROOT!r});c=Path({COLLECTION!r});lesson=r/'LESSON_CHECK_u011_B.json';l=json.loads(lesson.read_text())
assert all(e['status']=='graded' for e in l['entries'])
out=dict(unix=time.time(),lessons=dict(B=[dict(end_s=e['new_policy_branch']['window']['end_s']) for e in l['entries']]),cooperative={{}},source_sha256=dict(lesson_B=sha(lesson)),states={{}})
legacy={{}}
for p in (c/'labels').glob('*.json'):
 x=json.loads(p.read_text())['verification']
 if x['status'] in ('accepted','rejected'):legacy[x['verification_id']]=(x,sha(p))
for a in ('B','R'):
 p=r/'replay_bank'/a/'BANK_QUALITY.json';quality=json.loads(p.read_text());out['source_sha256']['quality_'+a]=sha(p)
 wanted=[x for x in quality['synergy_twins'] if x.get('admitted') and x.get('cooperative')]
 assert len(wanted)==quality['cooperative'];states=[]
 for x in wanted:
  vid=x['verification_id'];p=r/'replay_bank'/a/'sumo_verification/verifications'/(vid+'.json')
  if p.exists():v=json.loads(p.read_text());h=sha(p)
  else:assert a=='B';v,h=legacy[vid]
  assert 986000<=v['episode']['world']['seed']<=986071
  assert v['receipt_sha256']==digest({{k:z for k,z in v.items() if k!='receipt_sha256'}})
  members=len(v['chosen_plan']['members']);assert members==x['group_size']
  states.append(dict(verification_id=vid,end_s=int(v['window']['end_s']),members=members,source_sha256=h))
 out['states'][a]=states;out['cooperative'][a]=[dict(end_s=x['end_s'],members=x['members']) for x in states]
print(json.dumps(out))''',timeout=45))


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    from .protocol import checked,sha
    from .storage import read,write
    p=argparse.ArgumentParser();p.add_argument('--update',type=int,default=11)
    p.add_argument('--out',required=True);p.add_argument('--live',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]/'docs/paper_results/coopmwm'
    paths=dict(lesson_B=root/f'LESSON_CHECK_u{a.update:03}_B.json',
        synergy_B=root/f'POLICY_SYNERGY_u{a.update:03}_B.json',
        synergy_R=root/f'POLICY_SYNERGY_u{a.update:03}_R.json')
    if a.live:
        metadata=live_metadata();result=select(metadata['lessons'],metadata['cooperative'])
        result['metadata']=metadata
        result['source_contract']='ALL current admitted cooperative own-bank states; diagnostic failures never remove states from the quote'
    else:
        result=from_reports({k:read(checked(v)) for k,v in paths.items()})
        result['source_sha256']={k:sha(v) for k,v in paths.items()}
    target=checked(a.out)
    if target.exists():assert read(target)==result
    else:write(target,result)
    print(json.dumps(dict(receipt=str(target),sha256=sha(target),selected=result['selected'],
        launch_cost_gate=result['launch_cost_gate'],new_sim_s=0)))
