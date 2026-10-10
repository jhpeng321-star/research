"""Budget, scientific-scope and execution-isolation tests. No SUMO is launched."""
from copy import deepcopy
import json

import pytest

from traffic_shepherd import escalation_study as study
from traffic_shepherd import workflow
from tests.test_active_workflow import fake_runtime as existing_runtime_fixture
from tests.test_escalation_evidence import run


@pytest.fixture
def fake_runtime(tmp_path,monkeypatch):
    return existing_runtime_fixture.__wrapped__(tmp_path,monkeypatch)


@pytest.fixture
def identity(monkeypatch):
    value = dict(config_sha256="config",model_sha256="frozen-model",active_source_sha256={"source.py":"fixed"},
                 network_sha256="network",scenario_template_sha256="template",sumo_binary_sha256="sumo")
    monkeypatch.setattr(study,"current_identity",lambda: deepcopy(value))
    return value


def test_complete_plan_has_all_dependencies_and_no_congestion_requirement():
    plan = study.load_plan()
    assert len(plan["stages"]) == 12
    assert not plan["scope"]["congestion_required"]
    assert not plan["scope"]["damping_must_outperform_idm"]
    assert plan["scope"]["penetrations"] == [.05,.1,.2,.3]
    assert all(stage["automated_tests"] for stage in plan["stages"])


def test_same_seed_load_pairs_and_finite_nonoverlapping_panels():
    plan = study.load_plan()
    pilot = study.panel_jobs(plan,"pilot")
    extension = study.panel_jobs(plan,"penetration_extension")
    assert len(pilot) == 8 and len(extension) == 24
    assert {j["id"] for j in pilot}.isdisjoint({j["id"] for j in extension})
    assert sum(j["duration"] for j in pilot) == 19200
    for seed in (4107,4108):
        assert {j["loading"] for j in pilot if j["seed"] == seed} == {2000,2700}
    assert all(j["role"] == "exposed" for j in pilot + extension)


def test_registration_is_readiness_not_execution(tmp_path,identity,monkeypatch):
    monkeypatch.setattr(study,"evaluate",lambda **_: pytest.fail("registration must not run SUMO"))
    result = study.register(tmp_path / "panel")
    assert result["simulations_started"] == 0 and result["jobs"] == 8
    report = study.analyze(result["registration"])
    assert report["status"] == "incomplete" and len(report["pending"]) == 8
    with pytest.raises(FileExistsError):
        study.register(tmp_path / "panel")


def test_one_explicit_job_no_successor_and_no_retry(tmp_path,identity,monkeypatch):
    path = study.register(tmp_path / "panel")["registration"]
    calls = []
    monkeypatch.setattr(study,"evaluate",lambda **kwargs: calls.append(kwargs) or {"status":"synthetic"})
    job_id = study.panel_jobs(study.load_plan(),"pilot")[0]["id"]
    study.run_one(path,job_id)
    assert len(calls) == 1 and calls[0]["collect_traffic"] is True
    with pytest.raises(FileExistsError):
        study.run_one(path,job_id)
    with pytest.raises(ValueError,match="unknown"):
        study.run_one(path,"s912001_sealed")
    assert len(calls) == 1


def test_failed_start_remains_charged_and_cannot_silently_retry(tmp_path,identity,monkeypatch):
    path = study.register(tmp_path / "panel")["registration"]
    def fail(**kwargs):
        raise RuntimeError("synthetic launch failure")
    monkeypatch.setattr(study,"evaluate",fail)
    job = study.panel_jobs(study.load_plan(),"pilot")[0]["id"]
    with pytest.raises(RuntimeError):
        study.run_one(path,job)
    with pytest.raises(FileExistsError):
        study.run_one(path,job)
    assert study.analyze(path)["failures"] == [job]


def test_source_changes_and_registration_tampering_fail_before_launch(tmp_path,identity,monkeypatch):
    path = study.register(tmp_path / "panel")["registration"]
    identity["model_sha256"] = "changed"
    with pytest.raises(ValueError,match="changed after"):
        study.read_registration(path)
    content = json.loads(study.Path(path).read_text())
    content["jobs"][0]["seed"] = 912001
    study.Path(path).write_text(json.dumps(content))
    with pytest.raises(ValueError,match="content changed"):
        study.read_registration(path)


def save_run(root, job):
    root.mkdir(parents=True)
    manifest, summary = run(job["controller_name"],system=20. if job["controller_name"] == "damping" else 10.)
    manifest.update(seed=job["seed"], mainline_vph=job["loading"], penetration=job["penetration"])
    manifest["scenario"] = {"network_sha256":"network"}
    summary["seed"] = job["seed"]
    (root / "telemetry.jsonl").write_text("")
    with (root / "intervals.jsonl").open("w") as stream:
        for t in range(2400):
            row = {key:value/2400 for key,value in summary["totals"].items()}
            row.update(start_s=t,end_s=t+1,requested_not_arrived_vehicle_seconds=summary["system_seconds_per_requested_vehicle"]*100/2400)
            stream.write(json.dumps(row)+"\n")
    study.write_json(root / "summary.json",summary)
    manifest.update(telemetry_sha256=study.sha(root/"telemetry.jsonl"),intervals_sha256=study.sha(root/"intervals.jsonl"),
                    summary_sha256=study.sha(root/"summary.json"))
    study.write_json(root / "manifest.json",manifest)


def test_complete_poor_damping_panel_is_characterized_not_rejected(tmp_path,identity):
    path = study.register(tmp_path / "panel")["registration"]
    root, data = study.read_registration(path)
    for job in data["jobs"]:
        save_run(root / "jobs" / job["id"],job)
    report = study.analyze(path)
    assert report["status"] == "characterized" and len(report["paired_effects"]) == 4
    assert not report["baseline_performance_pass_required"]
    assert all(row["improvements"]["system_seconds_per_request"] == -10. for row in report["paired_effects"])


def test_corruption_and_incomplete_measurement_cannot_become_evidence(tmp_path,identity):
    job = study.panel_jobs(study.load_plan(),"pilot")[0]
    save_run(tmp_path/"run",job)
    (tmp_path/"run/intervals.jsonl").write_text("")
    with pytest.raises(ValueError,match="artifact"):
        study.verified_run(tmp_path/"run")
    manifest = json.loads((tmp_path/"run/manifest.json").read_text())
    manifest["intervals_sha256"] = study.sha(tmp_path/"run/intervals.jsonl")
    study.write_json(tmp_path/"run/manifest.json",manifest)
    with pytest.raises(ValueError,match="horizon"):
        study.verified_run(tmp_path/"run")


def test_offline_density_collection_does_not_enter_reports_or_change_commands(tmp_path,fake_runtime,monkeypatch):
    class Observer:
        def __init__(self,*args):
            pass
        def sample(self,*,now_s):
            return dict(timestamp_s=now_s,channel="offline_evaluation_only",true_density=999.)
    monkeypatch.setattr(workflow,"OfflineTrafficObserver",Observer)
    def execute(path,collect):
        fake_runtime.update(time=0.,writes=[],closed=False)
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name="damping",
                          output=path,duration=2,collect_traffic=collect)
        return (path/"telemetry.jsonl").read_text(), list(fake_runtime["writes"])
    without = execute(tmp_path/"without",False)
    with_density = execute(tmp_path/"with",True)
    assert without == with_density
    assert "true_density" not in with_density[0]
    assert "999" in (tmp_path/"with/traffic.jsonl").read_text()
