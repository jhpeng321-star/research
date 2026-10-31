import json

from tools.coopbehav.poll import poll_once


def test_poller_observes_readiness_reports_once_and_detects_document_change(tmp_path,monkeypatch):
    doc=tmp_path/"DESIGN.md"; doc.write_text("Trace handoff pending.\n")
    index=tmp_path/"COOPBEHAV_INDEX.json"; index.write_text('{"ready":false,"pairs":[]}')
    out=tmp_path/"poll"
    config=dict(documents=[str(doc)],indices=[str(index)])
    first=poll_once(config,out)
    assert first["indices"][0]["state"]=="producer_not_ready"
    calls=[]
    def report(views,target,path,label):
        calls.append(path); target.mkdir(parents=True)
        (target/"COMPLETE.json").write_text("{}")
        return dict(episodes=2)
    monkeypatch.setattr("tools.coopbehav.poll.run_report",report)
    index.write_text(json.dumps(dict(ready=True,pairs=[],source="test")))
    doc.write_text("Trace handoff now published.\n")
    second=poll_once(config,out)
    third=poll_once(config,out)
    assert second["documents"][0]["sha256"]!=first["documents"][0]["sha256"]
    assert len(second["reports"])==1
    assert third["indices"][0]["state"]=="complete" and len(calls)==1


def test_completed_previous_snapshot_is_published_when_live_index_has_advanced(tmp_path,monkeypatch):
    old,new="a"*64,"b"*64
    monkeypatch.setattr("tools.coopbehav.poll.remote_poll",lambda host,path:
        dict(state="running",sha256=new,path=path))
    monkeypatch.setattr("tools.coopbehav.poll.report_inventory",lambda host:
        [dict(state="complete",sha256=old),dict(state="running",sha256=new)])
    calls=[]
    def publish(config,entry):
        calls.append(entry["sha256"]); entry.update(verified=True,commit="commit-id")
    monkeypatch.setattr("tools.coopbehav.poll.publish",publish)
    config=dict(remote_indices=[dict(host="gpuR",path="/analysis/COOPBEHAV_INDEX.json")])
    poll_once(config,tmp_path/"poll")
    poll_once(config,tmp_path/"poll")
    assert calls==[old]
    assert json.loads((tmp_path/"poll"/"PUBLISHED.json").read_text())["gpuR:"+old]=="commit-id"
