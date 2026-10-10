"""No-SUMO CV-15 runner tests at registration and budget boundaries."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cv17_runner", ROOT / "output/controller_verification_v17/runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PREP_SPEC = importlib.util.spec_from_file_location(
    "cv17_prepare", ROOT / "output/controller_verification_v17/prepare_registration.py"
)
prepare = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(prepare)


def canonical():
    return prepare.build_registration(ROOT)


def temp_registration(tmp_path):
    reg = canonical()
    reg["output_root"] = str((tmp_path / "jobs").resolve())
    reg["ledger_path"] = str((tmp_path / "ledger.json").resolve())
    for job in reg["jobs"].values():
        out = tmp_path / "jobs" / Path(job["output_dir"]).name
        job["output_dir"] = str(out.resolve())
        job["run_dir"] = str((out / "run").resolve())
    return reg


def allow_temp_flow(monkeypatch, reg):
    monkeypatch.setattr(
        runner,
        "_validate_registration",
        lambda _registration, _root, job, _sumo_binary=None: reg["jobs"][job],
    )
    monkeypatch.setattr(
        runner, "verify_run", lambda _run, _expected: {"integrity_pass": True, "safety_pass": True}
    )
    requests = json.loads(Path(reg["jobs"]["retain"]["expected"]["requests_path"]).read_text())
    cav = next(row["vehicle_id"] for row in requests if row["is_cav"])
    vehicle = SimpleNamespace(
        getIDList=lambda: [cav],
        getSpeedMode=lambda _v: 31,
        getRoadID=lambda _v: "edge",
        getLaneID=lambda _v: "edge_0",
        getLanePosition=lambda _v: 1.0,
        getPosition=lambda _v: (1.0, 2.0),
        getSpeed=lambda _v: 10.0,
        getAcceleration=lambda _v: 0.0,
        getTypeID=lambda _v: "idm",
        slowDown=lambda *_args: None,
        setSpeed=lambda *_args: None,
    )
    simulation = SimpleNamespace(getDepartedIDList=lambda: [], getArrivedIDList=lambda: [])
    monkeypatch.setitem(
        sys.modules, "traci", SimpleNamespace(vehicle=vehicle, simulation=simulation)
    )


def write_artifacts(run):
    run.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "manifest.json": {},
        "summary.json": {},
        "telemetry.jsonl": "",
        "intervals.jsonl": "",
        "physics_steps.jsonl": "",
        "writer_trace.jsonl": "",
    }.items():
        (run / name).write_text(
            json.dumps(value) if name.endswith(".json") else value, encoding="utf-8"
        )


def test_canonical_registration_has_exact_roster_and_no_launch_side_effect():
    reg = canonical()
    assert reg["campaign"] == "cv17_shared_prefix"
    assert reg["cumulative_potential_seconds"] == 9600
    assert reg["prior_campaign"] == "cv15_shared_prefix"
    assert len(reg["prior_budget_sha256"]) == 64
    assert reg["cases"] == ["retain", "native", "pi-external-u"]
    assert reg["desired_speed_mps"] == 31.29
    assert reg["budget"]["ceilings"]["total_seconds"] == 7200
    assert {j["loading"] for j in reg["jobs"].values()} == {2700}
    assert reg["jobs"]["retain"]["expected"]["suffix"] == "damping"
    assert reg["jobs"]["native"]["expected"]["suffix"] == "native"
    assert reg["jobs"]["pi-external-u"]["expected"]["suffix"] == "pi-external-u"
    assert Path(reg["jobs"]["retain"]["expected"]["model_path"]).is_file()


def test_unmodified_canonical_registration_passes_preflight(monkeypatch):
    reg = canonical()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: str(prepare.SUMO))
    spec = runner._validate_registration(reg, ROOT, "retain", str(prepare.SUMO))
    assert spec["loading"] == 2700 and spec["run_dir"].endswith("retain\\run")


@pytest.mark.parametrize(
    "field,value",
    [("budget", {"amount_s": 1}), ("ledger_path", "relative.json"), ("desired_speed_mps", 10.0)],
)
def test_canonical_preflight_rejects_mutation_before_budget(field, value, monkeypatch):
    reg = canonical()
    reg[field] = value
    called = []
    monkeypatch.setattr(runner, "BudgetGuard", lambda *_: called.append(True))
    with pytest.raises((ValueError, KeyError)):
        runner._validate_registration(reg, ROOT, "retain", str(runner.shutil.which("sumo")))
    assert called == []


def test_frozen_input_mutation_rejected_before_budget(monkeypatch):
    reg = canonical()
    reg["frozen_files"][next(iter(reg["frozen_files"]))] = "changed"
    called = []
    monkeypatch.setattr(runner, "BudgetGuard", lambda *_: called.append(True))
    with pytest.raises(ValueError):
        runner._validate_registration(reg, ROOT, "retain", str(runner.shutil.which("sumo")))
    assert called == []


def test_failed_first_job_charges_full_reservation_and_actual_progress(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)

    def fake_evaluate(**kwargs):
        write_artifacts(Path(kwargs["output"]))
        kwargs["progress_callback"](0.5)
        raise RuntimeError("mock physics failure")

    monkeypatch.setattr(runner.workflow, "evaluate", fake_evaluate)
    vehicle = sys.modules["traci"].vehicle
    original_hooks = (vehicle.slowDown, vehicle.setSpeed)
    with pytest.raises(RuntimeError, match="mock physics"):
        runner.run_job(reg, job="retain", root=ROOT)
    assert (vehicle.slowDown, vehicle.setSpeed) == original_hooks
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["reservations"][0]["amount_s"] == 2400
    assert data["reservations"][0]["completed_simulated_s"] == 0.5
    assert data["campaigns"][reg["campaign"]]["stopped"] is True
    with pytest.raises(RuntimeError, match="Second job"):
        runner.run_job(reg, job="native", root=ROOT)


def test_success_uses_real_shared_prefix_contract_and_census(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)
    called = {}

    def fake_evaluate(**kwargs):
        called.update(kwargs)
        write_artifacts(Path(kwargs["output"]))
        for reached in (0.5, 2400.0):
            kwargs["progress_callback"](reached)
        return {"status": "complete", "controller": "damping"}

    monkeypatch.setattr(runner.workflow, "evaluate", fake_evaluate)
    receipt = runner.run_job(reg, job="retain", root=ROOT)
    assert receipt["execution_status"] == "complete"
    assert receipt["summary"] == {"status": "complete", "controller": "damping"}
    assert receipt["reservation_id"].startswith("cv17_shared_prefix:characterization:")
    assert called["controller_name"] == "damping"
    assert called["desired_speed_mps"] is None
    assert called["seed"] == 4108 and called["loading"] == 2700
    row = json.loads((tmp_path / "jobs/retain/run/physics_steps.jsonl").read_text().splitlines()[0])
    assert len(row["live_ids"]) == 1 and row["speed_modes"] == {row["live_ids"][0]: 31}
    assert set(row["vehicles"]) == set(row["live_ids"])
    census = row["vehicles"][row["live_ids"][0]]
    assert census == dict(
        road_id="edge",
        lane_id="edge_0",
        lane_position_m=1.0,
        position_xy=[1.0, 2.0],
        speed_mps=10.0,
        acceleration_mps2=0.0,
        type_id="idm",
        speed_mode=31,
    )


def test_repeat_job_directory_is_refused(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)
    out = Path(reg["jobs"]["retain"]["output_dir"])
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner.run_job(reg, job="retain", root=ROOT)


def test_empty_accepted_hash_map_cannot_admit_second_job(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)
    first = Path(reg["jobs"]["retain"]["output_dir"])
    first.mkdir(parents=True)
    (first / "accepted_review.json").write_text(
        json.dumps(
            dict(
                accepted=True,
                job="retain",
                registration_sha256=runner.registration_digest(reg),
                artifact_sha256={},
            )
        )
    )
    assert runner._review_passed(reg, "native") is False


def test_prefix_mismatch_stops_after_charging_native(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)

    def execute(**kwargs):
        write_artifacts(Path(kwargs["output"]))
        kwargs["progress_callback"](2400.0)
        return {}

    monkeypatch.setattr(runner.workflow, "evaluate", execute)
    monkeypatch.setattr(
        runner, "verify_prefix", lambda *a: dict(integrity_pass=False, failures=["changed prefix"])
    )
    assert runner.run_job(reg, job="retain", root=ROOT)["successor_allowed"]
    receipt = runner.run_job(reg, job="native", root=ROOT)
    assert not receipt["successor_allowed"]
    assert not receipt["prefix_review"]["integrity_pass"]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert sum(r["amount_s"] for r in ledger["reservations"]) == 4800
    assert ledger["campaigns"][reg["campaign"]]["stopped"]
    with pytest.raises(RuntimeError):
        runner.run_job(reg, job="pi-external-u", root=ROOT)
    assert len(json.loads((tmp_path / "ledger.json").read_text())["reservations"]) == 2


def test_third_allocation_rehashes_retained_not_only_native(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)

    def execute(**kwargs):
        write_artifacts(Path(kwargs["output"]))
        kwargs["progress_callback"](2400.0)
        return {}

    monkeypatch.setattr(runner.workflow, "evaluate", execute)
    monkeypatch.setattr(runner, "verify_prefix", lambda *a: dict(integrity_pass=True, failures=[]))
    runner.run_job(reg, job="retain", root=ROOT)
    runner.run_job(reg, job="native", root=ROOT)
    assert runner._review_passed(reg, "pi-external-u")
    (Path(reg["jobs"]["retain"]["run_dir"]) / "summary.json").write_text("{}changed")
    assert not runner._review_passed(reg, "pi-external-u")

