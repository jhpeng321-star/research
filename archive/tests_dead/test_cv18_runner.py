"""No-SUMO CV-18 runner tests at registration and budget boundaries."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd.budget_guard import BlockedByStop

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cv18_runner", ROOT / "output/controller_verification_v18/runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PREP_SPEC = importlib.util.spec_from_file_location(
    "cv18_prepare", ROOT / "output/controller_verification_v18/prepare_registration.py"
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
    requests = json.loads(
        Path(reg["jobs"]["4107-retain"]["expected"]["requests_path"]).read_text()
    )
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
    assert reg["campaign"] == "cv18_suffix_replication"
    assert reg["cumulative_potential_seconds"] == 24000
    assert reg["prior_campaign"] == "cv17_shared_prefix"
    assert reg["prior_budget_stopped"] is False
    assert reg["prior_reserved_seconds"] == 7200
    assert len(reg["prior_budget_sha256"]) == 64
    assert reg["cases"] == [
        "4107-retain", "4107-native", "4107-pi-external-u",
        "4108-retain", "4108-native", "4108-pi-external-u",
    ]
    assert reg["desired_speed_mps"] == 31.29
    assert reg["budget"]["ceilings"]["total_seconds"] == 14400
    assert {j["loading"] for j in reg["jobs"].values()} == {2000}
    assert {j["seed"] for j in reg["jobs"].values()} == {4107, 4108}
    for cell in ("4107", "4108"):
        assert reg["jobs"][f"{cell}-retain"]["expected"]["suffix"] == "damping"
        assert reg["jobs"][f"{cell}-native"]["expected"]["suffix"] == "native"
        assert reg["jobs"][f"{cell}-pi-external-u"]["expected"]["suffix"] == "pi-external-u"
        assert Path(reg["jobs"][f"{cell}-retain"]["expected"]["model_path"]).is_file()
    assert reg["jobs"]["4107-retain"]["expected"]["seed"] == 4107
    assert reg["jobs"]["4108-retain"]["expected"]["seed"] == 4108
    assert reg["jobs"]["4107-retain"]["expected"]["requests_sha256"] != \
        reg["jobs"]["4108-retain"]["expected"]["requests_sha256"]
    assert reg["jobs"]["4107-retain"]["expected"]["scenario_identity"] != \
        reg["jobs"]["4108-retain"]["expected"]["scenario_identity"]


def test_unmodified_canonical_registration_passes_preflight(monkeypatch):
    reg = canonical()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: str(prepare.SUMO))
    spec = runner._validate_registration(reg, ROOT, "4107-retain", str(prepare.SUMO))
    assert spec["loading"] == 2000 and spec["run_dir"].endswith("4107-retain\\run")
    spec = runner._validate_registration(reg, ROOT, "4108-pi-external-u", str(prepare.SUMO))
    assert spec["expected"]["seed"] == 4108


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
        runner._validate_registration(reg, ROOT, "4107-retain", str(runner.shutil.which("sumo")))
    assert called == []


def test_frozen_input_mutation_rejected_before_budget(monkeypatch):
    reg = canonical()
    reg["frozen_files"][next(iter(reg["frozen_files"]))] = "changed"
    called = []
    monkeypatch.setattr(runner, "BudgetGuard", lambda *_: called.append(True))
    with pytest.raises(ValueError):
        runner._validate_registration(reg, ROOT, "4107-retain", str(runner.shutil.which("sumo")))
    assert called == []


def test_unknown_job_and_malformed_name_refused():
    with pytest.raises(ValueError):
        runner._split("4109-retain")
    with pytest.raises(ValueError):
        runner._split("4107-unknown")


def test_failed_first_job_charges_full_reservation_and_stops_whole_campaign(
        tmp_path, monkeypatch):
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
        runner.run_job(reg, job="4107-retain", root=ROOT)
    assert (vehicle.slowDown, vehicle.setSpeed) == original_hooks
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["reservations"][0]["amount_s"] == 2400
    assert data["reservations"][0]["completed_simulated_s"] == 0.5
    assert data["campaigns"][reg["campaign"]]["stopped"] is True
    with pytest.raises(RuntimeError):
        runner.run_job(reg, job="4107-native", root=ROOT)
    with pytest.raises(BlockedByStop):
        runner.run_job(reg, job="4108-retain", root=ROOT)
    assert len(json.loads((tmp_path / "ledger.json").read_text())["reservations"]) == 1


def test_success_uses_cell_seed_loading_and_prefix_contract(tmp_path, monkeypatch):
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
    receipt = runner.run_job(reg, job="4107-retain", root=ROOT)
    assert receipt["execution_status"] == "complete"
    assert receipt["summary"] == {"status": "complete", "controller": "damping"}
    assert receipt["reservation_id"].startswith("cv18_suffix_replication:characterization:")
    assert called["controller_name"] == "damping"
    assert called["desired_speed_mps"] is None
    assert called["seed"] == 4107 and called["loading"] == 2000
    assert called["fleet_switch"] == ("damping", "damping", 600.0) or \
        called["fleet_switch"].prefix == "damping"
    row = json.loads(
        (tmp_path / "jobs/4107-retain/run/physics_steps.jsonl").read_text().splitlines()[0]
    )
    assert len(row["live_ids"]) == 1 and row["speed_modes"] == {row["live_ids"][0]: 31}
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
    out = Path(reg["jobs"]["4107-retain"]["output_dir"])
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner.run_job(reg, job="4107-retain", root=ROOT)


def test_empty_accepted_hash_map_cannot_admit_second_job(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)
    first = Path(reg["jobs"]["4107-retain"]["output_dir"])
    first.mkdir(parents=True)
    (first / "accepted_review.json").write_text(
        json.dumps(
            dict(
                accepted=True,
                job="4107-retain",
                registration_sha256=runner.registration_digest(reg),
                artifact_sha256={},
            )
        )
    )
    assert runner._review_passed(reg, "4107-native") is False


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
    assert runner.run_job(reg, job="4107-retain", root=ROOT)["successor_allowed"]
    receipt = runner.run_job(reg, job="4107-native", root=ROOT)
    assert not receipt["successor_allowed"]
    assert not receipt["prefix_review"]["integrity_pass"]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert sum(r["amount_s"] for r in ledger["reservations"]) == 4800
    assert ledger["campaigns"][reg["campaign"]]["stopped"]
    with pytest.raises(RuntimeError):
        runner.run_job(reg, job="4107-pi-external-u", root=ROOT)
    with pytest.raises(BlockedByStop):
        runner.run_job(reg, job="4108-retain", root=ROOT)
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
    runner.run_job(reg, job="4107-retain", root=ROOT)
    runner.run_job(reg, job="4107-native", root=ROOT)
    assert runner._review_passed(reg, "4107-pi-external-u")
    (Path(reg["jobs"]["4107-retain"]["run_dir"]) / "summary.json").write_text("{}changed")
    assert not runner._review_passed(reg, "4107-pi-external-u")


def test_cells_gate_independently_but_stop_is_global(tmp_path, monkeypatch):
    reg = temp_registration(tmp_path)
    allow_temp_flow(monkeypatch, reg)

    def execute(**kwargs):
        write_artifacts(Path(kwargs["output"]))
        kwargs["progress_callback"](2400.0)
        return {}

    monkeypatch.setattr(runner.workflow, "evaluate", execute)
    monkeypatch.setattr(runner, "verify_prefix", lambda *a: dict(integrity_pass=True, failures=[]))
    runner.run_job(reg, job="4107-retain", root=ROOT)
    runner.run_job(reg, job="4107-native", root=ROOT)
    runner.run_job(reg, job="4107-pi-external-u", root=ROOT)
    # the completed 4107 cell never gate-keeps 4108, and 4108 keeps its own order
    assert runner._review_passed(reg, "4108-retain")
    assert runner._review_passed(reg, "4108-native") is False
