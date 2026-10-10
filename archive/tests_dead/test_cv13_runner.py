"""No-SUMO CV-13 runner tests at registration and budget boundaries."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cv13_runner", ROOT / "output/controller_verification_v13/runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PREP_SPEC = importlib.util.spec_from_file_location("cv13_prepare", ROOT / "output/controller_verification_v13/prepare_registration.py")
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
    monkeypatch.setattr(runner, "_validate_registration",
                        lambda _registration, _root, job, _sumo_binary=None: reg["jobs"][job])
    monkeypatch.setattr(runner, "verify_run",
                        lambda _run, _expected: {"integrity_pass": True, "safety_pass": True})
    monkeypatch.setattr(runner, "verify_references", lambda _refs: {})
    requests = json.loads(Path(reg["jobs"]["load2000"]["expected"]["requests_path"]).read_text())
    cav = next(row["vehicle_id"] for row in requests if row["is_cav"])
    vehicle = SimpleNamespace(getIDList=lambda: [cav], getSpeedMode=lambda _v: 31)
    simulation = SimpleNamespace(getDepartedIDList=lambda: [], getArrivedIDList=lambda: [])
    monkeypatch.setitem(sys.modules, "traci", SimpleNamespace(vehicle=vehicle, simulation=simulation))


def write_artifacts(run):
    run.mkdir(parents=True, exist_ok=True)
    for name, value in {"manifest.json": {}, "summary.json": {},
                        "telemetry.jsonl": "", "intervals.jsonl": "",
                        "physics_steps.jsonl": ""}.items():
        (run / name).write_text(json.dumps(value) if name.endswith(".json") else value,
                                encoding="utf-8")


def test_canonical_registration_has_exact_roster_and_no_launch_side_effect():
    reg = canonical()
    assert reg["campaign"] == "cv13_pi_characterization"
    assert reg["cases"] == ["load2000", "load2700"]
    assert reg["desired_speed_mps"] == 31.29
    assert reg["budget"]["ceilings"]["total_seconds"] == 4800
    assert {j["loading"] for j in reg["jobs"].values()} == {2000, 2700}


def test_unmodified_canonical_registration_passes_preflight(monkeypatch):
    reg = canonical()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: str(prepare.SUMO))
    monkeypatch.setattr(runner, "verify_references", lambda _refs: {})
    spec = runner._validate_registration(reg, ROOT, "load2000", str(prepare.SUMO))
    assert spec["loading"] == 2000 and spec["run_dir"].endswith("load2000\\run")


@pytest.mark.parametrize("field,value", [("budget", {"amount_s": 1}),
                                           ("ledger_path", "relative.json"),
                                           ("desired_speed_mps", 10.0)])
def test_canonical_preflight_rejects_mutation_before_budget(field, value, monkeypatch):
    reg = canonical()
    reg[field] = value
    called = []
    monkeypatch.setattr(runner, "BudgetGuard", lambda *_: called.append(True))
    with pytest.raises((ValueError, KeyError)):
        runner._validate_registration(reg, ROOT, "load2000", str(runner.shutil.which("sumo")))
    assert called == []


def test_frozen_input_mutation_rejected_before_budget(monkeypatch):
    reg = canonical()
    reg["frozen_files"][next(iter(reg["frozen_files"]))] = "changed"
    called = []
    monkeypatch.setattr(runner, "BudgetGuard", lambda *_: called.append(True))
    with pytest.raises(ValueError):
        runner._validate_registration(reg, ROOT, "load2000", str(runner.shutil.which("sumo")))
    assert called == []


def test_failed_first_job_charges_full_reservation_and_actual_progress(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)

    def fake_evaluate(**kwargs):
        write_artifacts(Path(kwargs["output"]))
        kwargs["progress_callback"](0.5)
        raise RuntimeError("mock physics failure")

    monkeypatch.setattr(runner.workflow, "evaluate", fake_evaluate)
    with pytest.raises(RuntimeError, match="mock physics"):
        runner.run_job(reg, job="load2000", root=ROOT)
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["reservations"][0]["amount_s"] == 2400
    assert data["reservations"][0]["completed_simulated_s"] == 0.5
    assert data["campaigns"][reg["campaign"]]["stopped"] is True
    with pytest.raises(RuntimeError, match="Second job"):
        runner.run_job(reg, job="load2700", root=ROOT)


def test_success_uses_real_pi_workflow_contract_and_census(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)
    called = {}

    def fake_evaluate(**kwargs):
        called.update(kwargs)
        write_artifacts(Path(kwargs["output"]))
        for reached in (0.5, 2400.0):
            kwargs["progress_callback"](reached)
        return {"status": "complete", "controller": "pi-external-u"}

    monkeypatch.setattr(runner.workflow, "evaluate", fake_evaluate)
    receipt = runner.run_job(reg, job="load2000", root=ROOT)
    assert receipt["execution_status"] == "complete"
    assert receipt["summary"] == {"status": "complete", "controller": "pi-external-u"}
    assert receipt["reservation_id"].startswith("cv13_pi_characterization:characterization:")
    assert called["controller_name"] == "pi-external-u"
    assert called["desired_speed_mps"] == 31.29
    assert called["seed"] == 4108 and called["loading"] == 2000
    row = json.loads((tmp_path / "jobs/load2000/run/physics_steps.jsonl").read_text().splitlines()[0])
    assert len(row["live_ids"]) == 1 and row["speed_modes"] == {row["live_ids"][0]: 31}


def test_repeat_job_directory_is_refused(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)
    out = Path(reg["jobs"]["load2000"]["output_dir"])
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner.run_job(reg, job="load2000", root=ROOT)


def test_empty_accepted_hash_map_cannot_admit_second_job(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)
    first = Path(reg['jobs']['load2000']['output_dir'])
    first.mkdir(parents=True)
    (first/'accepted_review.json').write_text(json.dumps(dict(accepted=True, job='load2000',
        registration_sha256=runner.registration_digest(reg), artifact_sha256={})))
    assert runner._review_passed(reg, 'load2700') is False





