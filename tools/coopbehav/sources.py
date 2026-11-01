"""Adapters for published live-task handoffs; reads results/traces, never simulator state.

Builds an index under this task's analysis root from closed per-job result.json
files. In-progress gzip traces are never consumed. The audit's initial cropped
schema is reported as blocked rather than filled with guessed geometry/actions.
"""
import argparse
import hashlib
from pathlib import Path
import re

import numpy as np

from .io import check_seed, read_json, safe
from .report import write_json


def build_source_index(kind,root,out):
    root=safe(root); out=safe(out); out.mkdir(parents=True,exist_ok=True)
    status=dict(kind=kind,root=str(root),exists=root.is_dir(),complete_jobs=0,pairs=0,blocked=[])
    pairs=[]; provenance=[]
    jobs=safe(root/"jobs")
    if jobs.is_dir():
        for raw in sorted(jobs.iterdir()):
            try: jobdir=safe(raw)
            except ValueError: continue
            if not jobdir.is_dir(): continue
            result_path=safe(jobdir/"result.json")
            if not result_path.is_file(): continue
            result=read_json(result_path)
            if kind=="coopaudit":
                # Full-prefix v2 adds issued requests, but retains the members+
                # ramp-only crop. Neither version supplies slot/follower geometry.
                status["complete_jobs"]+=1
                corrected=result.get("branch_mode")=="full_prefix_rerun"
                reason="Audit crop lacks length_m/is_cav and mainline neighbours"
                if not corrected: reason+=", with non-wall executed commands unavailable in v1"
                status["blocked"].append(dict(job=jobdir.name,
                    branch_mode=result.get("branch_mode","unspecified"),reason=reason))
                continue
            if kind!="sumocf": raise ValueError("Unknown source adapter")
            job=result.get("job",{})
            if "seed" not in job: continue
            check_seed(job["seed"])
            status["complete_jobs"]+=1
            if "credit" not in result or result.get("missing_group"):
                status["blocked"].append(dict(job=jobdir.name,reason="Canary/no complete eligible group credit"))
                continue
            group=list(map(str,result["group"]))
            branches={frozenset(map(str,b["members"])):b for b in result["branches"]}
            if len(group)<2 or frozenset() not in branches or frozenset(group) not in branches:
                status["blocked"].append(dict(job=jobdir.name,reason="Incomplete branch roster")); continue
            b=branches[frozenset()]; g=branches[frozenset(group)]
            if any(frozenset([j]) not in branches for j in group):
                status["blocked"].append(dict(job=jobdir.name,reason="Missing solo branch")); continue
            dg=float(b["J600"])-float(g["J600"])
            solo=sum(float(b["J600"])-float(branches[frozenset([j])]["J600"]) for j in group)
            cg=dg-solo
            credit=result["credit"]
            scale=max(abs(float(v["J600"])) for v in branches.values())
            tolerance=np.finfo(float).eps*scale*(len(group)+2)
            if any(abs(float(credit[k])-v)>tolerance for k,v in (("D_G",dg),("sum_D_i",solo),("C_G",cg))):
                raise ValueError(f"Source credit does not reconcile: {jobdir.name}")
            paths={}
            valid=True
            for name,record in (("factual",g),("reference",b)):
                path=safe(record["trace"])
                expected=safe(root/"traces"/jobdir.name)
                if path.parent.resolve()!=expected.resolve() or not re.fullmatch(r"b\d{3}\.jsonl\.gz",path.name):
                    raise ValueError("Unexpected source trace path")
                if not path.is_file(): valid=False; break
                paths[name]=str(path)
            if not valid:
                status["blocked"].append(dict(job=jobdir.name,reason="Closed result trace not yet present")); continue
            pairs.append(dict(id=jobdir.name,seed=job["seed"],arm="sumocf_"+job.get("checkpoint_sha256","unknown")[:12],
                kind="policy_mean_intervention",members=group,**paths,C_G=cg,D_G=dg,sum_D_i=solo,
                credit_outcome="team_delay_600s",branch=dict(t0_s=job["t0"],horizon_s=job["window"]+30),
                intervention_window_s=job["window"],trace_scope="full_observable_fleet",
                position_dtype_epsilon=float(np.finfo(np.float32).eps),
                trace_limitations=["1 Hz control frames, not 0.5 s physics frames",
                                   "Recorded response stops at intervention end +30 s, credit horizon is600s"],
                source_initial_state_sha256=result.get("initial_state_sha256")))
            pairs[-1]["source_design"]={k:job[k] for k in ("size","rho","sample","checkpoint_sha256") if k in job}
            provenance.append(dict(job=jobdir.name,result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest()))
    index=dict(ready=bool(pairs),source=kind,pairs=pairs,provenance=provenance)
    # Deterministic index contents: no changing timestamps in the index digest.
    path=out/"COOPBEHAV_INDEX.json"; write_json(path,index)
    status["pairs"]=len(pairs); status["index"]=str(path)
    write_json(out/"SOURCE_STATUS.json",status)
    return status


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=["sumocf","coopaudit"],required=True)
    p.add_argument("--root",required=True); p.add_argument("--out",required=True)
    a=p.parse_args()
    print(build_source_index(a.kind,a.root,a.out))
