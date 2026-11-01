"""Verify published COOPAUDIT v2 crops without inventing missing geometry.

Reads only trace paths explicitly inventoried in COLLECTION.json, binds their
SHA256s, and cross-checks production rows/clocks against TRACE_MANIFEST.csv.
This is a coverage receipt, not a detector report or a simulation dispatcher.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

from .io import check_seed, read_json, safe
from .report import write_json


TRACE = re.compile(r"jobs/v2_(?P<canary>canary_)?s(?P<seed>\d+)/traces/"
                   r"(?P<arm>native|mappo|wall|sumocf)_(?:B(?:_repeat)?|GROUP(?:_repeat)?|ONLY_\d+|DISABLED)\.jsonl\.gz")
REQUIRED = ("t", "vid", "s", "v", "a", "lane", "member", "ramp_observed", "requested_accel")


def inventory_path(root, name):
    # Portable export paths use both separators. No absolute/escaping inputs.
    rel=PurePosixPath(name.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or ":" in str(rel):
        raise ValueError("Inventory path escapes source root")
    path=safe(root.joinpath(*rel.parts))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Inventory path escapes source root")
    return path, str(rel)


def inspect_trace(path):
    rows=0; lo=math.inf; hi=-math.inf; previous=-math.inf
    fields=Counter(); members=Counter(); commands=0; keys=set(); unrelated=0
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            if any(k not in row for k in REQUIRED):
                raise ValueError(f"Incomplete audit row: {path}")
            t=float(row["t"])
            if not all(math.isfinite(float(row[k])) for k in ("t","s","v","a")) or t<previous:
                raise ValueError(f"Invalid audit clocks/kinematics: {path}")
            if t!=previous: keys.clear()
            if str(row["vid"]) in keys: raise ValueError(f"Duplicate identity at clock: {path}")
            keys.add(str(row["vid"])); previous=t
            for flag in ("member","ramp_observed"):
                if not isinstance(row[flag],bool): raise ValueError(f"Invalid audit flags: {path}")
            if row["requested_accel"] is not None:
                if not math.isfinite(float(row["requested_accel"])):
                    raise ValueError(f"Invalid issued request: {path}")
                commands+=1
            rows+=1; lo=min(lo,t); hi=max(hi,t); fields.update(row.keys())
            if row["member"]: members[str(row["vid"])]+=1
            unrelated+=not row["member"] and not row["ramp_observed"]
    if not rows: raise ValueError(f"Empty audit trace: {path}")
    return dict(rows=rows,min_t=lo,max_t=hi,field_rows=dict(sorted(fields.items())),
                member_identities=sorted(members),non_null_issued_request_rows=commands,
                nonmember_nonramp_rows=unrelated)


def inspect_audit(collection, manifest):
    collection=safe(collection); manifest=safe(manifest); root=collection.parent
    published=read_json(collection); selected={}
    # Validate the complete selected roster before opening any trajectory file.
    for entry in published["files"]:
        name=entry["path"].replace("\\", "/")
        match=TRACE.fullmatch(name)
        if match is None: continue
        check_seed(match["seed"])
        path,name=inventory_path(root,name)
        if name in selected: raise ValueError("Duplicate collection trace")
        selected[name]=(path,entry,match)
    expected={}
    with manifest.open(newline="",encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            check_seed(row["seed"]); _,name=inventory_path(root,row["file"])
            match=TRACE.fullmatch(name)
            if match is None or match["canary"] or int(match["seed"])!=int(row["seed"]):
                raise ValueError("Invalid production manifest trace")
            if name in expected: raise ValueError("Duplicate manifest trace")
            expected[name]=row
    production={name for name,(_,_,m) in selected.items() if not m["canary"]}
    if not selected or production!=set(expected):
        raise ValueError("Collection and production manifest rosters disagree")
    records=[]; totals=Counter(); arms={}; worlds=set()
    for name,(path,entry,match) in sorted(selected.items()):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=entry["sha256"] or (name in expected and digest!=expected[name]["sha256"]):
            raise ValueError(f"Trace SHA256 mismatch: {name}")
        result=inspect_trace(path)
        if name in expected:
            e=expected[name]
            if (result["rows"],result["min_t"],result["max_t"])!=(int(e["rows"]),float(e["min_t"]),float(e["max_t"])):
                raise ValueError(f"Trace row/clock mismatch: {name}")
            worlds.add(int(match["seed"]))
        cohort="canary" if match["canary"] else "production"
        key=cohort+"/"+match["arm"]
        arm=arms.setdefault(key,dict(files=0,rows=0,non_null_issued_request_rows=0))
        arm["files"]+=1
        for field in ("rows","non_null_issued_request_rows"):
            arm[field]+=result[field]; totals[field]+=result[field]
        for field in ("length_m","is_cav","controller"):
            totals[field+"_rows"]+=result["field_rows"].get(field,0)
        totals[cohort+"_files"]+=1
        totals["nonmember_nonramp_rows"]+=result["nonmember_nonramp_rows"]
        records.append(dict(file=name,sha256=digest,seed=int(match["seed"]),cohort=cohort,**result))
    return dict(source="COOPAUDIT full-prefix v2",kind="coverage_only",hashes_verified=True,
        collection_sha256=hashlib.sha256(collection.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        production_worlds=len(worlds),totals=dict(totals),arms=arms,files=records,
        detector_eligible=False,detector_counts=None,
        blockers=["No observed vehicle lengths: bumper gaps and overlapping bodies are not identifiable",
                  "Members-and-ramp-only crop: true mainline leaders and follower chains are not identifiable",
                  "No per-row is_cav/controller fields: fleet exposure and normalized release status are unavailable"],
        interpretation="Unavailable geometry is missing evidence, not zero events. requested_accel is an issued-safe request, not realised acceleration. No v1 credits are pooled.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection",required=True); p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True); a=p.parse_args()
    result=inspect_audit(a.collection,a.manifest)
    write_json(safe(a.out),result)
    print(json.dumps({k:result[k] for k in ("kind","hashes_verified","production_worlds","totals","detector_eligible")}))
