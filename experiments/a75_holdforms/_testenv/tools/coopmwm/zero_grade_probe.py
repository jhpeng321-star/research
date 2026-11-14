"""MAIN06:58 read-only diagnosis of the two named TRAIN zero grades.

Reads existing RESULT/observable tapes only. No SUMO, WM calls, grading retries,
learning changes, or traversal of scenario directories.
"""
import inspect
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT


def analyse(root, world):
    import gzip
    import hashlib
    import json
    from pathlib import Path
    from itertools import zip_longest

    assert world in (986051,986070)
    root=Path(root);episode=f'warm_u002_B_{world}'
    gid=f'xg_{episode}_t0300'
    def read(path):
        v=json.loads(path.read_text())
        if v.get('encoding')=='coopmwm-lossless-gzip-v1':
            assert v['file']=='RESULT.json.gz'
            raw=gzip.decompress((path.parent/v['file']).read_bytes())
            assert hashlib.sha256(raw).hexdigest()==v['sha256']
            return json.loads(raw)
        return v
    receipt=read(root/'exact_grades/receipts'/(gid+'.json'))
    assert receipt['grade_id']==gid and receipt['status']=='graded'
    assert receipt['grade_veh_s']==0 and receipt['arm']=='B' and receipt['update']==2
    assert receipt['prefix']['pass_'] and receipt['noise']['crn_verified_t0']
    op=root/'jobs'/episode
    bp=root/'exact_grades/jobs'/gid/f'attempt{receipt["attempt"]}'
    original,branch=read(op/'RESULT.json'),read(bp/'RESULT.json')
    start,end=receipt['window']['start_s'],receipt['window']['end_s']
    ost={int(s['time_s']):s for s in original['steps']}
    bst={int(s['time_s']):s for s in branch['steps']}
    plan=ost[start]['proposal']['plan'];members=plan['members']
    out=dict(world=world,grade_id=gid,receipt=receipt,plan_id=plan['id'],
        origin=plan.get('origin'),saving_vs_policy=plan.get('saving_vs_policy'),
        saving_vs_mappo=plan.get('saving_vs_mappo'),
        max_abs_generic_knot_perturbation=max(abs(x) for row in plan['perturbation_knots'] for x in row),
        plan_member_rows_at_start=[r for r in ost[start]['rows'] if r['vid'] in members],
        members={},without_executed_plan_steps=sum(bool(s['sample'].get('behaviour',{}).get('executed')) for s in branch['steps']),
        per_second_reward_differences=0,max_abs_reward_difference=0.,trace_rows=0,
        vehicle_roster_differences=0,position_speed_lane_differences=0,
        issued_command_differences=0,prefix_observable_differences=0,
        accounting=dict(J_with=receipt['J_with_veh_s'],J_without=receipt['J_without_veh_s']))
    for vid in members:
        out['members'][vid]=dict(raw_request_differences=0,max_abs_raw_request_difference=0.,
            sensed_seconds=0,first_sensed_s=None,last_sensed_s=None,
            issued_command_differences=0,max_abs_issued_command_difference=0.,
            max_abs_position_difference=0.,max_abs_speed_difference=0.,first_commands=[])
    for t in range(start,end):
        a,b=ost[t],bst[t]
        d=abs(a['reward']-b['reward'])
        out['per_second_reward_differences']+=int(d!=0)
        out['max_abs_reward_difference']=max(out['max_abs_reward_difference'],d)
        out['prefix_observable_differences']+=int(original['prefixes'][str(t)]['observable']!=branch['prefixes'][str(t)]['observable'])
        for vid in members:
            row=out['members'][vid]
            if vid not in a['commands'] or vid not in b['commands']:continue
            ar=a['requests'].get(vid,{}).get('requested_raw',a['commands'][vid])
            br=b['requests'].get(vid,{}).get('requested_raw',b['commands'][vid])
            delta=abs(ar-br)
            row['raw_request_differences']+=int(delta!=0)
            row['max_abs_raw_request_difference']=max(row['max_abs_raw_request_difference'],delta)
            if len(row['first_commands'])<6:
                row['first_commands'].append(dict(time_s=t,with_raw=ar,without_raw=br))
    with gzip.open(op/'observable.jsonl.gz','rt') as af,gzip.open(bp/'observable.jsonl.gz','rt') as bf:
        for al,bl in zip_longest(af,bf):
            if bl is None:break  # Original lasts1200s; branch ends at registered window.
            assert al is not None,'branch exceeds original tape'
            a,b=json.loads(al),json.loads(bl)
            assert a['time_s']==b['time_s']
            t=a['time_s']
            if not start<=t<end:continue
            out['trace_rows']+=1
            av,bv=({r['vid']:r for r in q['vehicles']} for q in (a,b))
            out['vehicle_roster_differences']+=int(set(av)!=set(bv))
            for vid in av.keys()&bv.keys():
                x,y=av[vid],bv[vid]
                out['position_speed_lane_differences']+=int(any(x[k]!=y[k] for k in ('s_m','v_mps','lane')))
                issued=x.get('issued_acceleration_mps2')!=y.get('issued_acceleration_mps2')
                out['issued_command_differences']+=int(issued)
                if vid not in members:continue
                row=out['members'][vid];row['sensed_seconds']+=1
                if row['first_sensed_s'] is None:row['first_sensed_s']=t
                row['last_sensed_s']=t
                row['issued_command_differences']+=int(issued)
                xa,ya=x.get('issued_acceleration_mps2'),y.get('issued_acceleration_mps2')
                if xa is not None and ya is not None:
                    row['max_abs_issued_command_difference']=max(row['max_abs_issued_command_difference'],abs(xa-ya))
                for key,field in (('s_m','max_abs_position_difference'),('v_mps','max_abs_speed_difference')):
                    row[field]=max(row[field],abs(x[key]-y[key]))
    assert out['trace_rows']==end-start,'incomplete observable tape'
    return out


def collect():
    source=inspect.getsource(analyse)
    result=json.loads(ops.ssh.remote('cpuQ',source+f'\nimport json,time\nprint(json.dumps(dict(unix=time.time(),new_sumo_sim_s=0,WM_calls=0,learning_changes=False,probes=[analyse({ROOT!r},w) for w in (986051,986070)])))',timeout=60))
    write(ops.EVIDENCE/'ZERO_GRADE_DIAGNOSIS_986051_986070.json',result)
    print(json.dumps(result));return result


def register():
    path=ops.EVIDENCE/'ZERO_GRADE_DIAGNOSIS_986051_986070.json'
    value=read(path)
    for p in value['probes']:
        assert p['origin']=='centre' and p['saving_vs_policy']==0
        assert p['without_executed_plan_steps']==0
        assert p['trace_rows']==420 and p['position_speed_lane_differences']==0
        assert p['vehicle_roster_differences']==p['per_second_reward_differences']==0
        assert p['prefix_observable_differences']==0
    line='\nS-COOPMWM MAIN0658 ZERO-GRADE DIAGNOSIS evidenceSHA'+sha(path)+': worlds986051/986070 Bu2 t300 exact0 confirmed from independent original/without-plan RESULT and420-second observable tapes; policy-centre candidates with zero generic knot perturbation/saving-vs-policy0, WM saving-vs-MAPPO164.137479/26.224529; raw and issued requests differ but sensed motion/rosters/rewards identical, members present only1-4seconds then depart; without branches execute no plans, prefix/CRN PASS. No grading artifact found; retain zeros in exact trust evidence, exclude from positive-only imitation/library by existing rule. No new thresholds/search selection, physics/inference, retries or budget change.\n'
    registry=checked(REGISTRY)
    if sha(path) not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line)
    print(json.dumps(dict(registered=True,evidence_sha256=sha(path),new_sumo_sim_s=0)))


def classify():
    evidence=ops.EVIDENCE/'ZERO_GRADE_DIAGNOSIS_986051_986070.json'
    probes=read(evidence)['probes']
    snapshot=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r})/'exact_grades/receipts'
g=[json.loads(p.read_text()) for p in r.glob('*.json')]
g=[v for v in g if v['arm']=='B' and v.get('training_seed')==20261101 and v['status']=='graded']
print(json.dumps(dict(unix=time.time(),grades=[dict(grade_id=v['grade_id'],grade=v['grade_veh_s']) for v in g])))''',timeout=40))
    flags={p['grade_id']:dict(observed_physical_null=True,
        raw_requests_identical=not any(r['raw_request_differences'] for r in p['members'].values()),
        pre_ownership_safe_commands_identical=p['issued_command_differences']==0,
        post_ownership_records_persisted=False,
        classification='observed physical-null plan; pre-ownership requests differ; final ownership speed clipping can still erase differences') for p in probes}
    value=dict(snapshot,flags=flags,probe_evidence_sha256=sha(evidence),new_sumo_sim_s=0,
        inspected=len(flags),inspected_pre_ownership_command_identical=0,
        confirmed_physical_nulls=len(flags),exact_graded_B_plans=len(snapshot['grades']),
        confirmed_physical_null_fraction=len(flags)/len(snapshot['grades']),
        qualification='case-driven diagnosis of named exact zeros; fraction is a descriptive confirmed lower bound among current exact-graded B plans, not population rate; raw/safe command trace is before CommandOwnership and cannot establish final issued action equality',
        learning='counted as executed; exact-zero trust evidence retained; no imitation/library under existing positive-only rule')
    target=ops.EVIDENCE/'ZERO_GRADE_NOOP_CLASSIFICATION.json';write(target,value)
    line='\nS-COOPMWM MAIN0658 NOOP REPORT evidenceSHA'+sha(target)+': two named centre plans flagged observed_physical_null, counted executed; pre-ownership safe commands differ in both, final ownership records not persisted so strict final-issued-action no-op cannot be established from these tapes. Confirmed physical-null fraction'+str(value['confirmed_physical_nulls'])+'/'+str(value['exact_graded_B_plans'])+' at'+str(snapshot['unix'])+' (descriptive/case-driven, no outcome filtering); trust zeros retained, imitation/library excluded by unchanged rule; no new physics/inference.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    print(json.dumps(value));return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('collect','register','classify'),default='collect',nargs='?')
    globals()[parser.parse_args().mode]()
