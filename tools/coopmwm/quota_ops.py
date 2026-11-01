"""MAIN05:48 outcome-blind admission registration and boundary handover."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_quotacpu1'


def preregister():
    path=ops.EVIDENCE/'MAIN0548_ADMISSION_PLAN.json'
    if not path.exists():
        write(path,dict(authority='MAIN2026-11-02 05:48, full290line brief',
            admission='per update Nexec10: if executed+unresolved potentially executable queries<10, eligible decisions query at p=.95 and wait even without a free slot; otherwise p=.95 only with a free slot, else0; serialize reservations; record the predicate and realised probability',
            broker='preserve quota-blocking requests; adopt all existing assigned inference without recomputation; ordinary overflow declines remain explicit',
            first_update='retain all24 current B episodes; apply u1 even if fewer than10 starts with explicit engineering deviation, no extra episodes; later shortage is governed by two-consecutive-update health pause',
            pipeline='independent R(u) consumes final B(u) quota while B(u+1) collects; exact count/regime/group-size/duration matching; no library for R',
            handover='register and CPU qualify first; freeze only receipt-owned B/R supervisor parents before optimizer boundary if necessary; existing child SUMO and GPU work continues; wait all24 B RESULTs, then terminate old parents by verified identity and adopt complete results with checkpoint0 Adam/RNG; no physics replay',
            invariants='same warm fast-path MPPI, K4H300/K8Hfinal, populations, beta/exact-only grading, guards, endpoints, original GPU deadline; no learned proposer activation',
            budget=dict(cap=7500000,allocated=7495040,free=4960,additional_sim_s=0),
            no_outcome_selection=True))
    h=sha(path)
    line='\nS-COOPMWM MAIN0548 QUOTA-AWARE ADMISSION REGISTERED amendmentSHA'+h+': '+json.dumps(read(path),sort_keys=True)+'\n'
    registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line)
    reply=ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});p=r/'AUTHORITY.md'
if {h!r} not in p.read_text():
 with p.open('a') as f:f.write({line!r})
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_sumo_sim_s=0)))''',timeout=40)
    value=json.loads(reply);write(ops.EVIDENCE/'MAIN0548_ADMISSION_PREREGISTRATION.json',value);print(reply)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preregister',));a=p.parse_args();globals()[a.mode]()
