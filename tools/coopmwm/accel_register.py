"""Metadata-only enlarged-panel exposure audit and prospective registration."""
import argparse
import json
from pathlib import Path
from . import ops
from .accel_protocol import worlds,budget,EXTRA_START,digest,sha,REGISTRY,checked
from .storage import write,read


def audit(mode):
    # Audit only the NEW block. Original36 retain the original zero-exposure
    # receipts; their metadata registration is deliberately not an exposure.
    seeds=list(range(EXTRA_START,EXTRA_START+60))
    original=ops.plan_seeds
    ops.plan_seeds=lambda:seeds
    try:
        if mode=='protocols':
            # Separate prospective receipt names; never overwrite the originals.
            original_write=ops.__dict__.get('EVIDENCE')
            from .protocol import registry_conflicts,FORBIDDEN
            import re
            forbidden=re.compile(FORBIDDEN.pattern+r'|hetdata_truth|vehicle_params|driver_params|vtype',re.I)
            files={};hits=[]
            reg=checked(REGISTRY)
            conflicts=registry_conflicts(reg.read_text(encoding='utf-8'),set(seeds))
            if conflicts:hits.append(dict(path=str(reg),lines=conflicts))
            for base in Path('E:/agent-work').iterdir():
                if not base.is_dir() or base.is_symlink() or forbidden.search(str(base)):continue
                for pattern in ('tools/*/protocol.py','tools/*/design.py','tools/*/registration*.json','docs/paper_results/*DESIGN.md','*REGISTRATION*.md','protocol*.json'):
                    for p in base.glob(pattern):
                        if not p.is_file() or forbidden.search(str(p)):continue
                        # This task's prospective design is not prior exposure.
                        if base==ops.LOCAL and p.name=='COOPMWM_DESIGN.md':continue
                        lines=registry_conflicts(p.read_text(encoding='utf-8',errors='replace'),set(seeds))
                        files[str(p)]=sha(p)
                        if lines:hits.append(dict(path=str(p),lines=lines))
            r=dict(pass_=not hits,hits=hits,files_scanned=len(files),files_sha256=digest(files),registry_sha256=sha(reg),seeds_sha256=digest(seeds),new_sumo_sim_s=0)
            write(ops.EVIDENCE/'ACCEL96_PROTOCOL_AUDIT.json',r);print(json.dumps(r))
        else:
            # Reuse the established filename scanner with a private evidence
            # directory so originals cannot be clobbered.
            destination=ops.EVIDENCE/'accel96_audit'
            destination.mkdir(exist_ok=True)
            prior=ops.EVIDENCE;ops.EVIDENCE=destination
            try:ops.audit_host(mode)
            finally:ops.EVIDENCE=prior
    finally:ops.plan_seeds=original


def register():
    audits=[read(ops.EVIDENCE/'accel96_audit'/f'FINAL_EXPOSURE_{h}.json') for h in ('cpuQ','gpuA','gpuR')]
    p=read(ops.EVIDENCE/'ACCEL96_PROTOCOL_AUDIT.json')
    assert p['pass_'] and all(x['pass_'] for x in audits)
    ws=worlds('final_test')
    assert len(ws)==len({w['seed'] for w in ws})==96
    assert all(sum(w['cell']==c for w in ws)==8 for c in range(12))
    panel=dict(worlds=ws,panel_sha256=digest(ws),arms=['MAPPO169','native','A_u24','B_u24','R_u24_later'],
        access='sealed; no outcome access until primary A/B fixedu24; each arm once; no reruns/tuning/selection',
        original36_metadata_sha256=digest(ws[:36]),precharged_primary_sim_s=460800,
        precharged_R_sim_s=115200,seed2='dropped under owner19:38 budget-first rule',
        analysis='10000 paired world bootstrap RNG20261101 95CI B-MAPPO/B-A/B-R/A-MAPPO J600/J900; percell/safety; detector-first complete exact twins')
    write(ops.EVIDENCE/'FINAL_TEST96_METADATA.json',panel)
    quote=dict(budget(),audits=audits,protocol_audit=p,new_sumo_sim_s=0)
    write(ops.EVIDENCE/'ACCEL96_BUDGET.json',quote)
    line=(f'\nS-COOPMWM OWNER19:34/38/44 ACCELERATION RESTART AND96WORLD PANEL REGISTERED BEFORE NEW PHYSICS/OUTCOMES: '
          f'old A u1..19/B u1..3 superseded; all811640sim-s retained including6unstarted reservations. '
          f'A/B/R common1s acceleration residual MAPPO169 d9042cc5 zero==MAPPO, original safe_accel/CommandOwnership; '
          f'AR1 exp(-1/5), spatialshared.5, sigma.5m/s2; SUMO-only J/sourcewaiting. '
          f'B committed2member acceleration plans10..60s,10sknots; mediumCEM32x3,elite8,K4H120screen,top8K8H300rescore; '
          f'bootstrap-Thompson sharedK8indices, accept only positive vs policycontinued AND MAPPO; '
          f'R sameplanbounds matched count/regime/duration, random proposer; inheritedEBtype-regimetrust/nohandthresholds. '
          f'NEW60worlds{EXTRA_START}..{EXTRA_START+59}/5percell plus989600..989635=96/8percell; '
          f'panelSHA256{digest(ws)}, original36 unchanged; cpuP unreachable/excluded; registry/protocol/{p["files_scanned"]}files '
          f'and cpuQ/gpuA/gpuR safe-name audits allZEROexposures. '
          f'fixedfinalu24, once primary MAPPO/native/A/B then R; no selection/tuning/rerun/peeking. '
          f'Exactcap6000000 allocation{quote["allocated"]},free{quote["unallocated"]},phases{json.dumps(quote["phases"],sort_keys=True)}. '
          f'Seed2A/Bdropped: retaining them costs{quote["with_seed2_total"]}>6000000; seed3 previouslydropped. '
          f'CheckTRAIN986072..986083 every4updates,dev984000..984023 every8; MAPPOcache and12TRAINcontrolbytes reused. '
          f'PrimarycontrastsB-MAPPO/B-A/B-R/A-MAPPO,J600/J900,pairedworldbootstrap95CI10000/RNG20261101,percell/safety. '
          f'Earlyviability perupdate selectedplan-vs-ownH300criticSUMOresidual bytype/rank/newness,worldCI; '
          f'exacttwins randomregistered12executedBstarts (RNG20261101, firstu1complete episodes thenu2etc until12), '
          f'allmembers/reversions,joint,all atH60/120/300/600;240000discoverycap includes later detector-first testtwins. '
          f'E2diagnosticonly retained. Qualityfirst/noharddeadline, hotaddsame-searchworkers; '
          f'GPUcapacity/Nexec numericalbinding and finalCPU required before restart; allarmsheld. '
          f'ReceiptsACCEL96_BUDGET.json SHA{sha(ops.EVIDENCE/"ACCEL96_BUDGET.json")},FINAL_TEST96_METADATA.json.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'ACCEL96_REGISTRATION.json',dict(line=line,new_sumo_sim_s=0))
    print(line.strip())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('protocols','cpuQ','gpuA','gpuR','register'));a=p.parse_args()
    register() if a.mode=='register' else audit(a.mode)
