"""Prospective metadata-only registration of the owner2354 resolved full plan."""
from pathlib import Path
import json
from . import ops
from .protocol import checked,sha,REGISTRY,digest
from .storage import read,write
from .warm_protocol import budget,batch_worlds
from .branch_eval import build_plan,plan_sha


def register():
    target=ops.EVIDENCE/'WARM2354_REGISTRATION.json'
    assert not target.exists(),'registration already exists; inspect receipt'
    registry=checked(REGISTRY);text=registry.read_text(encoding='utf-8')
    authorities=[line for line in text.splitlines() if any(line.startswith('2026-11-01 '+t+' MAIN:') for t in ('22:58','23:15','23:30','23:54'))]
    assert len(authorities)==4
    b=budget();assert b['allocated']==6473440 and b['unallocated']==26560
    sources=[]
    for path,start,end in [('E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md',7,23),
                           ('E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md',73,102)]:
        p=checked(path);lines=p.read_text(encoding='utf-8').splitlines()
        sources.append(dict(path=str(p),sha256=sha(p),lines=[start,end],excerpt=lines[start-1:end]))
    plan=build_plan(times=[300,500,700,900],later_arms=(),note='MAIN23:15/23:54: native episode-only; R branches deferred; fixed A/B48')
    plan['arm_execution'].update(A='fixed update48, deterministic, no guidance',B='fixed update48, deterministic, no guidance',native='episode-level only; no branch arm',R='branches deferred; episode checkpoint24')
    plan['analysis']['focus']['sources']=sources
    plan['plan_sha256']=plan_sha(plan)
    value=dict(budget=b,authorities=authorities,branches=plan,
        focus=dict(fraction=.5,definition='demand2400/2700 x share.2/.4',sources=sources,
            evidence_limit='existing opportunity/oracle evidence motivates owner hypothesis; no established stratum treatment effect'),
        training_batches={str(u):batch_worlds(u) for u in range(1,49)},
        A_preservation='A u1..4/check/optimizer/RNG retained; new focus sampling prospective fromu5',
        checks='TRAIN every4/dev every16, reporting only; final A/B48,R24',
        release='B/R MAPPOu0 only after source-bound CPU/GPU warm/MPPI qualification and numeric throughput registration',
        branch_reuse='one-world T300+300 equivalence check600sim-s; mismatch STOP; extra115200 needs MAIN approval',
        discovery='200000sim-s; complete n+2 prefixes reserved before exact twins; no partial group',new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM2354_AMENDMENT.json',value)
    h=sha(ops.EVIDENCE/'WARM2354_AMENDMENT.json')
    line=('\nS-COOPMWM MAIN23:54 FULL PLAN REGISTERED amendmentSHA'+h+
        ': owner cap6500000, allocated6473440/free26560; A/B48 R24 fixed finals; TRAIN4/dev16; discovery200000; '
        'R branch345600 deferred, R episode115200 retained; A/B branch691200 atT300/500/700/900,H300; '
        'MAPPO reuse equivalence600 precharged, mismatch STOP, extra115200 needs MAIN approval. '
        'Two Holm families per comparator B-MAPPO/B-A over{all-branch,focus},two-sidedalpha.05; '
        'focus demand2400/2700 x share.2/.4, exact source lines/SHA in amendment. '
        'Identical prospective50%focus TRAIN sampling/arm, existing72worlds only; A completedu1..4/check retained; '
        'B/R warmu0 restart remains qualification-gated. No newSUMO/no final outcome access.\n')
    # Append only, preserving every concurrent task's existing registration.
    with registry.open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(amendment_sha256=h,line=line,new_sumo_sim_s=0))
    print(line.strip())


if __name__=='__main__':register()
