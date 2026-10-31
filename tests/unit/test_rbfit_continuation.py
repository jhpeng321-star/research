"""Durable continuation must preserve user edits and prevent duplicate jobs."""
from datetime import datetime,timedelta
import importlib.util
from pathlib import Path
import sys

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
