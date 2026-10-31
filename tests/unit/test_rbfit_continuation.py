"""Durable continuation must preserve user edits and prevent duplicate jobs."""
from datetime import datetime,timedelta
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pytest


@pytest.fixture
def watcher():
    tools = Path(__file__).resolve().parents[2]/"tools/rbfit"
    sys.path.insert(0,str(tools))
    spec = importlib.util.spec_from_file_location("rbfit_watch",tools/"watch_lc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_raw_receipt_never_dispatches_duplicate(watcher,monkeypatch):
    monkeypatch.setattr(watcher,"inspect",lambda job:{"claimed":True})
    def forbidden(*args,**kwargs):
        raise AssertionError("An existing receipt must prohibit another dispatch")
    monkeypatch.setattr(watcher,"remote",forbidden)
    job = {"root":watcher.BOUNDARIES,"name":"lane_boundaries_v4"}
    state = {"jobs":[job]}
    watcher.dispatch_raw(job,state)
    watcher.dispatch_raw(job,state)
    assert len(state["jobs"])==2
    assert state["jobs"][1]["name"]=="lane_raw_v5"


def test_unfamiliar_index_blocks_before_artifact_or_handoff_write(watcher,monkeypatch,tmp_path):
    monkeypatch.setattr(watcher,"REPO",tmp_path)
    monkeypatch.setattr(watcher,"command",lambda *args,**kwargs:"user-staged.py\n")
    with pytest.raises(RuntimeError,match="Unfamiliar staged"):
        watcher.publish({"publication":"lane_raw_v5"},{})
    assert not (tmp_path/"artifacts").exists()


def test_uncommitted_handoff_is_not_overwritten(watcher,monkeypatch,tmp_path):
    monkeypatch.setattr(watcher,"REPO",tmp_path)
    def cmd(argv,**kwargs):
        return "" if "--cached" in argv else "HANDOFF.md\n"
    monkeypatch.setattr(watcher,"command",cmd)
    handoff = tmp_path/"HANDOFF.md"
    handoff.write_text("user work\n")
    with pytest.raises(RuntimeError,match="Uncommitted handoff"):
        watcher.publish({"publication":"lane_raw_v5"},{})
    assert handoff.read_text()=="user work\n"


def test_exclusive_watcher_receipt_prevents_restart(watcher,monkeypatch,tmp_path):
    now = datetime.now().astimezone()
    monkeypatch.setattr(watcher,"RECEIPTS",tmp_path)
    receipt = tmp_path/"LC_WATCH_0128_process.json"
    receipt.write_text("existing ownership receipt\n")
    monkeypatch.setattr(sys,"argv",["watch_lc.py","--launch","--until",(now+timedelta(hours=2)).isoformat(),
        "--last-start",(now+timedelta(hours=1)).isoformat()])
    def forbidden(*args,**kwargs):
        raise AssertionError("Existing watcher receipt must prevent Popen")
    monkeypatch.setattr(watcher.subprocess,"Popen",forbidden)
    with pytest.raises(FileExistsError):
        watcher.main()
    assert receipt.read_text()=="existing ownership receipt\n"


def test_successful_publication_keeps_candidate_identity(watcher,monkeypatch,tmp_path):
    repo = tmp_path/"fit"
    (repo/"artifacts/rbfit").mkdir(parents=True)
    (repo/"docs/paper_results").mkdir(parents=True)
    shared = tmp_path/"rb-shared/fit"
    shared.mkdir(parents=True)
    (repo/"HANDOFF.md").write_text("# Handoff\n")
    (repo/"docs/paper_results/RBFIT.md").write_text("- LC continuation: prior\n")
    monkeypatch.setattr(watcher,"REPO",repo)
    population = {"lc2013_source_sha256":"a"*64,"snapshot_version":"v2"}
    payloads = {name:b"{}" for name in watcher.ARTIFACTS}
    payloads["lc2013_population.json"] = json.dumps(population).encode()
    aggregate = {"expected_event_count":2.,"events":1,"event_brier":.1}
    verification = {"status":"PASS","drivers":1,"aggregate":{
        "raw_physics_shrunk_posterior":aggregate,"likelihood_shrunk_posterior":aggregate},
        "artifact_sha256":{name:hashlib.sha256(payloads[name]).hexdigest() for name in watcher.ARTIFACTS if name!="VERIFIED.json"}}
    payloads["VERIFIED.json"] = json.dumps(verification).encode()
    calls = []
    def cmd(argv,**kwargs):
        calls.append(argv)
        if argv[0]=="scp":
            target = Path(argv[-1])
            if target.name=="process.json":
                target.write_text('{"pid":1}')
            else:
                for name,payload in payloads.items():
                    (target/name).write_bytes(payload)
        return "abc123\n" if argv[:2]==["git","rev-parse"] else ""
    monkeypatch.setattr(watcher,"command",cmd)
    state = {}
    watcher.publish({"root":"/own","name":"lane_fit","publication":"lane_source_abc123"},state)
    commit = next(call for call in calls if call[:2]==["git","commit"])
    assert commit[-1].endswith("lane_source_abc123")
    assert "`lane_source_abc123`" in (repo/"HANDOFF.md").read_text()
    assert "`lane_source_abc123/`" in (repo/"docs/paper_results/RBFIT.md").read_text()
    assert "lane_events.json" not in commit[-1]


def test_os_singleton_rejects_parallel_watcher_and_releases(watcher,monkeypatch,tmp_path):
    monkeypatch.setattr(watcher,"RECEIPTS",tmp_path)
    namespace = "rbfit_test_"+uuid.uuid4().hex
    release = watcher.acquire_singleton(namespace)
    try:
        with pytest.raises(RuntimeError,match="Another continuation"):
            watcher.acquire_singleton(namespace)
    finally:
        release()
    watcher.acquire_singleton(namespace)()


def test_subprocess_text_uses_utf8_independent_of_windows_locale(watcher,monkeypatch):
    calls = []
    def run(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0,stdout="\u4ea4\u901a",stderr="")
    monkeypatch.setattr(watcher.subprocess,"run",run)
    assert watcher.command(["git","status"])=="\u4ea4\u901a"
    assert calls[0]["encoding"]=="utf-8"
