"""Reconcile derived reports and assemble a descriptive, source-separated comparison."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np

from .detectors import BEHAVIOURS
from .io import read_json, safe
from .report import write_json


def distribution(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return dict(n=len(x),min=float(x.min()),median=float(np.median(x)),max=float(x.max())) if len(x) else None


def verify_report(path):
    root=safe(path); summary=read_json(root/"summary.json"); complete=read_json(root/"COMPLETE.json")
    assert complete["summary_sha256"]==hashlib.sha256((root/"summary.json").read_bytes()).hexdigest()
    episodes=read_json(root/"episodes.json"); inputs=read_json(root/"inputs.json")
    assert len(episodes)==len(inputs)==summary["episodes"]==complete["episodes"]
    assert len({r["episode"] for r in episodes})==len(episodes)
    with gzip.open(root/"events.jsonl.gz","rt",encoding="utf-8") as stream:
        import json
        events=[json.loads(line) for line in stream]
    assert len(events)==summary["events"]==complete["events"]
    with (root/"members.csv").open(newline="",encoding="utf-8") as stream:
        members=list(csv.DictReader(stream))
    records={r["episode"]:r for r in inputs}
    by_arm={}
    for arm,s in summary["controllers"].items():
        ee=[e for e in events if e["arm"]==arm]; mm=[m for m in members if m["arm"]==arm]
        eps=[e for e in episodes if e["arm"]==arm]
        assert len(eps)==s["episodes"]
        assert np.isclose(sum(float(m["exposure_cav_s"]) for m in mm),s["exposure_cav_s"],rtol=1e-12,atol=0)
        for b in BEHAVIOURS:
            target=s["behaviours"][b]; selected=[e for e in ee if e["behaviour"]==b]
            if target["count"] is None: assert not selected; continue
            assert len(selected)==target["count"]
            assert sum(len(e["members"]) for e in selected)==target["member_event_count"]
            assert sum(int(m[b]) for m in mm if m[b])==target["member_event_count"]
        for e in ee:
            assert e["start_s"]<=e["end_s"]
            assert set(e["members"])
        geometry={}
        for b in ("yield_to_merge","gap_creation"):
            selected=[e for e in ee if e["behaviour"]==b]
            geometry[b]=dict(opening_m=distribution([e["opening_m"] for e in selected]),
                headway_change_s=distribution([e.get("headway_change_s") for e in selected]),
                ballistic_braking_contribution_m=distribution([e.get("ballistic_braking_contribution_m") for e in selected]),
                braking_contribution_over_opening=distribution([e["ballistic_braking_contribution_m"]/e["opening_m"]
                    for e in selected if e.get("ballistic_braking_contribution_m") is not None and e["opening_m"]>0]),
                end_reasons=dict(Counter(e["end_reason"] for e in selected)))
        cells=Counter((records[e["episode"]]["meta"].get("penetration"),
                       records[e["episode"]]["meta"].get("demand_vph")) for e in eps)
        by_arm[arm]=dict(**s,geometry=geometry,
            unique_worlds=len({e["seed"] for e in eps}),
            cells=[dict(penetration=p,demand_vph=d,count=n) for (p,d),n in cells.items()])
    discovery=read_json(root/"discovery.json"); signatures=read_json(root/"signatures.json")
    gain_seeds={records[str(r["id"])]["meta"]["seed"] for r in signatures if r["C_G"]>0}
    digest={p.name:hashlib.sha256(safe(p).read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    return dict(source=summary["source"],path=str(root),verified=True,episodes=len(episodes),events=len(events),
                controllers=by_arm,positive_credit_records=discovery["counts"]["ranked"],
                positive_credit_unique_worlds=len(gain_seeds),positive_credit_seeds=sorted(gain_seeds),
                discovery=discovery,output_sha256=digest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",action="append",required=True); p.add_argument("--out",required=True)
    a=p.parse_args()
    result=dict(interpretation="Descriptive source-separated rates; repeated probe designs are not independent worlds",
                reports=[verify_report(path) for path in a.report])
    write_json(a.out,result)
    print({r["source"]:dict(verified=r["verified"],episodes=r["episodes"],events=r["events"],
                           positive_credit_unique_worlds=r["positive_credit_unique_worlds"]) for r in result["reports"]})


if __name__=="__main__": main()
