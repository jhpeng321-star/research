import csv
import gzip
import hashlib
import json

import pytest

from tools.coopbehav.audit_coverage import inspect_audit
from tools.coopbehav.sources import build_source_index


def export(tmp_path, seed=969650):
    name=f"jobs/v2_s{seed}/traces/mappo_GROUP.jsonl.gz"
    trace=tmp_path/name; trace.parent.mkdir(parents=True)
    rows=[dict(t=t,vid="member",s=-10.,v=1.,a=-1.,lane=1,member=True,
               ramp_observed=False,requested_accel=command)
          for t,command in [(300.,None),(300.5,-1.)]]
    with gzip.open(trace,"wt") as f:
        for row in rows: f.write(json.dumps(row)+"\n")
    sha=hashlib.sha256(trace.read_bytes()).hexdigest()
    collection=tmp_path/"COLLECTION.json"
    collection.write_text(json.dumps(dict(files=[dict(path=name,sha256=sha)])))
    manifest=tmp_path/"TRACE_MANIFEST.csv"
    with manifest.open("w",newline="") as f:
        writer=csv.writer(f); writer.writerow(["seed","file","sha256","rows","min_t","max_t"])
        writer.writerow([seed,name.replace("/","\\"),sha,2,300,300.5])
    return collection,manifest,trace


def test_audit_coverage_checks_all_rows_but_does_not_invent_geometry(tmp_path):
    collection,manifest,_=export(tmp_path)
    result=inspect_audit(collection,manifest)
    assert result["hashes_verified"] and result["production_worlds"]==1
    assert result["totals"]["non_null_issued_request_rows"]==1
    assert result["totals"]["length_m_rows"]==0
    assert result["detector_counts"] is None and not result["detector_eligible"]
    assert result["files"][0]["rows"]==2


def test_audit_coverage_rejects_corrupted_published_trace(tmp_path):
    collection,manifest,trace=export(tmp_path)
    trace.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        inspect_audit(collection,manifest)


def test_audit_coverage_rejects_manifest_roster_mismatch(tmp_path):
    collection,manifest,_=export(tmp_path)
    collection.write_text('{"files": []}')
    with pytest.raises(ValueError,match="rosters disagree"):
        inspect_audit(collection,manifest)


def test_audit_coverage_checks_seed_before_opening_missing_trace(tmp_path):
    collection,manifest,_=export(tmp_path)
    collection.write_text(json.dumps(dict(files=[dict(
        path="jobs/v2_s962100/traces/mappo_GROUP.jsonl.gz",sha256="unused")])))
    with pytest.raises(ValueError,match="Restricted seed"):
        inspect_audit(collection,manifest)


def test_corrected_audit_status_does_not_repeat_superseded_command_blocker(tmp_path):
    root=tmp_path/"audit"; job=root/"jobs"/"v2_s969650"; job.mkdir(parents=True)
    (job/"result.json").write_text('{"branch_mode":"full_prefix_rerun"}')
    status=build_source_index("coopaudit",root,tmp_path/"out")
    blocked=status["blocked"][0]
    assert blocked["branch_mode"]=="full_prefix_rerun"
    assert "length_m" in blocked["reason"] and "commands" not in blocked["reason"]
    assert status["pairs"]==0
