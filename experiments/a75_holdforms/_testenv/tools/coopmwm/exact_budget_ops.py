"""Register latest exact-grade liability/control; no inference/physics dispatch."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,sha,REGISTRY
from .warm_deploy import ROOT
from .warm_protocol import budget
from .resume_gate import check_budget


def register():
    target=ops.EVIDENCE/'MAIN0224_EXACT_BUDGET_REGISTRATION.json'
    assert not target.exists(),'adopt existing budget registration'
    plan=budget();check_budget(plan)
    assert (plan['total'],plan['allocated'],plan['unallocated'])==(7500000,7473440,26560)
    assert plan['phases']['exact_plan_grading']==1000000
    data=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({ROOT!r});assert r.is_dir()
for arm in ('B','R'):
 for suffix in ('','_V3','_V4'):
  assert not (r/f'WARM_{{arm}}_DISPATCH{{suffix}}.json').exists()
  assert not (r/f'WARM_{{arm}}_STARTED{{suffix}}.json').exists()
p=r/'AWR_BINDING.json'
print(json.dumps(dict(AWR=json.loads(p.read_text()),AWR_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),B_R_unstarted=True)))''',timeout=35))
    value=dict(budget=plan,pre_outcome=data,new_sumo_sim_s=0,
        exact_grade='same deterministic episode prefix; policy without focal plan vs observed with plan; source-inclusive team delay over duration+120s; PrefixMismatch STOP/no imputation',
        learning='ONLY exact grades feed plan AWR/trust/library/covariance; ungraded queued; own-action PPO and all-step SUMO critic retained; baseline-only AWR temperature unchanged',
        reserve='1000000 sim-s; quote ~1000sim-s/plan, expected720 minimum B/R starts; reserve complete rerun before attempt; extra starts also charged; insufficient reserve STOP',
        runtime='delegated exact grader must be live before first B update; asynchronous CPU nice19, report lag; source-bound integration/CPU release gates still pending')
    amendment=ops.EVIDENCE/'MAIN0224_EXACT_BUDGET_AMENDMENT.json'
    write(amendment,value);h=sha(amendment)
    line=('\nS-COOPMWM MAIN02:24 OWNER EVERY-PLAN EXACT SUMO GRADING / CAP7500000 REGISTERED amendmentSHA'+h+
        ': before any B/R outcome; latest owner cap7500000 supersedes all earlier caps. '
        'Preserved6473440 allocation + exact_plan_grading1000000 =7473440; free26560 not auto-spent; '
        'fixed A/B48,R24/TRAIN4/dev16/final96 and protected A/B branches unchanged, R branches deferred. '
        'EVERY executed B/R plan graded by exact same-state deterministic prefix rerun without focal plan, '
        'current rollout policy own actions, duration+120s source-inclusive SUMO delay against actual with-plan window; '
        'PrefixMismatch STOP/logged, never imputed. ONLY exact grades determine plan imitation/trust/library/covariance; '
        'ungraded queued, no critic plan estimate; own-step PPO and all-step SUMO critic retained. '
        'AWR exactgain/10000 with frozenT4.822766875237093, min(20,exp(A/T)), positive only. '
        'Expected720 minimum starts at ~1000sim-s/grade is an estimate, complete prefix+horizon reservation before each grade, '
        'failed charges retained, extra starts consume same reserve, insufficient complete-grade budget STOP. '
        'Async nice19 CPU grader live before first B update, lag reported; delegated implementation/integration gates pending, '
        'this budget registration alone authorizes no launch. MAIN02:28 verified-PID resource reclamation recorded; '
        'MAIN02:30 measured capacity sizing pending; no new SUMO in registration.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
for arm in ('B','R'):assert not (r/f'WARM_{{arm}}_DISPATCH_V3.json').exists() and not (r/f'WARM_{{arm}}_DISPATCH_V4.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
with (r/'MAIN0224_EXACT_BUDGET.json').open('x') as f:json.dump(json.loads({json.dumps(dict(budget=plan,amendment_sha256=h))!r}),f,sort_keys=True)
print('registered, no physics')''',timeout=35)
    write(target,dict(line=line,amendment_sha256=h,budget=plan,new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,amendment_sha256=h,budget=plan,new_sumo_sim_s=0)))


def register_control():
    target=ops.EVIDENCE/'MAIN0236_CONTROL_REGISTRATION.json'
    assert not target.exists(),'adopt existing control registration'
    budget_registration=read(ops.EVIDENCE/'MAIN0224_EXACT_BUDGET_REGISTRATION.json')
    value=dict(budget_amendment_sha256=budget_registration['amendment_sha256'],
        screen=dict(K=4,H=300,conditional_reduction=False),
        final=dict(K=8,horizon='min(600,max(300,duration_s+120))'),
        endpoints=dict(A=48,B=48,R=24),owner_review_pause=False,
        health_pauses=True,grading_blocks_own_ppo=False,new_sumo_sim_s=0,
        cpu_budget_transport_tests=dict(tests=11,pass_=True,skips=0,sim_s=0))
    amendment=ops.EVIDENCE/'MAIN0236_CONTROL_AMENDMENT.json'
    write(amendment,value);h=sha(amendment)
    line=('\nS-COOPMWM MAIN02:34/02:36 CONTROL REGISTERED amendmentSHA'+h+
        ': before any B/R outcome, conditional cheaper screen REVOKED. K4/H300 screen ALWAYS; '
        'K8 Hfinal=min(600,max(300,D+120)) unchanged. No horizon reduction under any throughput condition. '
        'Auto-continue A/B48,R24 under registered health pauses; u4 viability publication is not an owner-review stop. '
        'Own-action PPO never waits for grading; ungraded plans queued, never estimated or imitated; '
        'only exact SUMO grades feed plan AWR/trust/library/covariance. Delegated grader live before first B update. '
        'After fixed training endpoints the registered once-only final evaluation/analysis remains required. '
        'CPU budget/transport contracts11PASS/zeroSUMO; preserved allocation7473440/cap7500000/free26560.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
for arm in ('B','R'):
 for suffix in ('','_V3','_V4'):
  assert not (r/f'WARM_{{arm}}_DISPATCH{{suffix}}.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
with (r/'MAIN0236_CONTROL.json').open('x') as f:json.dump(json.loads({json.dumps(dict(value,amendment_sha256=h))!r}),f,sort_keys=True)
print('registered, no physics')''',timeout=35)
    write(target,dict(line=line,amendment_sha256=h,control=value,new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,amendment_sha256=h,new_sumo_sim_s=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('budget','control'),nargs='?',default='budget')
    a=p.parse_args();register() if a.mode=='budget' else register_control()
