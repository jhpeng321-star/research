"""Synthetic regression checks; no test in this file launches a simulator."""
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from traffic_shepherd.damping import DampingDecision
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.scenarios import TripRequest
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


STAGING = Path(__file__).parent
if not (Path(__file__).parents[1] / "traffic_shepherd" / "paired_experiments.py").exists():
    spec = importlib.util.spec_from_file_location("traffic_shepherd.paired_experiments", STAGING / "paired_experiments.py")
    paired = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = paired
    spec.loader.exec_module(paired)
else:
    from traffic_shepherd import paired_experiments as paired


def report(now=1., vehicle="a"):
    return CAVReport(SensorFrame(vehicle, now, 20., 0., 30., 0,
        distance_to_merge_m=1500., front=NeighborMeasurement(40., 0.),
        front_coverage_m=100., rear_coverage_m=100., lane_id="L0", position_m=100.), now, "damping")


def job(arm="damping", *, duration=4, decision=1, name=None):
    return dict(id=name or arm, case_id="synthetic_case", seed=4107, loading=2000,
        penetration=.2, role="exposed", arm=arm, duration_s=duration, decision_time_s=decision,
        policy_spec=dict(required_lanes=["L0"], member_ids=["a"], profile={}))


def criteria():
    return dict(primary_minimum_improvement={"braking_seconds_per_request": .01},
                maximum_cost_increase={key: 0. for key in
                    ("braking_seconds_per_request", "effort_per_request", "system_seconds_per_request",
                     "ttc_seconds_per_request", "source_wait_seconds_per_request")})


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    state = dict(time=0., writes=[], closed=False, starts=0)
    def live():
        return ["a", "hdv"] if state["time"] >= .5 else []
    vehicle = SimpleNamespace(getIDList=live, getSpeedMode=lambda _: 31, getSpeed=lambda _: 20.,
        getAcceleration=lambda v: -.25 if v == "a" else 0., getLeader=lambda *_: None,
        getLaneID=lambda _: "L0", getLanePosition=lambda _: 100., getRouteID=lambda _: "route",
        getRouteIndex=lambda _: 0,
        slowDown=lambda *a: state["writes"].append((state["time"], "slowDown", *a)),
        setSpeed=lambda *a: state["writes"].append((state["time"], "setSpeed", *a)))
    api = SimpleNamespace(vehicle=vehicle, isLoaded=lambda: False, getVersion=lambda: [1, "synthetic"],
        simulationStep=lambda: state.update(time=state["time"]+.5),
        simulation=SimpleNamespace(getTime=lambda: state["time"],
            getDepartedIDList=lambda: ["a", "hdv"] if state["time"] == .5 else [],
            getArrivedIDList=lambda: [], getCollidingVehiclesIDList=lambda: [],
            getStartingTeleportNumber=lambda: 0))
    monkeypatch.setitem(sys.modules, "traci", api)
    binary = tmp_path / "sumo-placeholder"
    binary.write_bytes(b"synthetic executable identity")
    monkeypatch.setattr(paired.shutil, "which", lambda _: str(binary))
    monkeypatch.setattr(paired, "start_sumo", lambda *a, **k: state.update(starts=state["starts"]+1))
    monkeypatch.setattr(paired, "close_sumo", lambda *a: state.update(closed=True))
    monkeypatch.setattr(paired, "observation_map", lambda *a: {})
    class Actor:
        def __init__(self, config):
            self.config = config
        def decide(self, frame, *, now_s):
            return DampingDecision(-.7, -.5, 19.5, None, (20., 20., 40.))
    class Sensor:
        def __init__(self, *a):
            pass
        def capture(self, cav, *, now_s):
            return report(now_s, cav).frame
    class Observer:
        def __init__(self, *a):
            pass
        def sample(self, *, now_s):
            return dict(timestamp_s=now_s, channel="offline_evaluation_only", synthetic_full_count=len(live()))
    monkeypatch.setattr(paired, "CirclesController", Actor)
    monkeypatch.setattr(paired, "SumoLocalSensor", Sensor)
    monkeypatch.setattr(paired, "capture_batch", lambda api, ids, **kwargs:
        ({v: report(kwargs["now_s"], v).frame for v in sorted(ids)}, {},
         dict(sensor_read_requests=0, sensor_transport_queries=0)))
    monkeypatch.setattr(paired, "OfflineTrafficObserver", Observer)
    requests = [TripRequest("a", 0., "r", "normal_car", True), TripRequest("hdv", 0., "r", "normal_car", False)]
    def build(root, *a, **k):
        root.mkdir()
        scene = root / "scenario.sumocfg"
        scene.write_text("synthetic scenario")
        (root / "requests.rou.xml").write_text("synthetic routes")
        return scene, requests, dict(schema="synthetic")
    monkeypatch.setattr(paired, "build_scenario", build)
    return state


def registration():
    return dict(identity=paired.current_identity(), registration_sha256="registered_fixture",
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        communication=dict(max_report_age_s=1., history_s=10.))


def constant_policy(arm):
    def policy(context):
        if context.now_s < 1:
            return paired.PolicyOutput()
        return paired.PolicyOutput({"a": -1.} if arm == "wall" and context.now_s == 1 else {},
            "assembly" if arm == "wall" and context.now_s == 1 else "release" if arm == "wall" else "damping",
            member_ids=("a",), proposal_sha256="same-proposal", evidence_sha256="member-evidence")
    return policy


def test_callback_boundary_contains_only_received_data_and_damping_references():
    assert {f.name for f in fields(paired.PolicyInput)} == {
        "now_s", "reports", "history", "damping_commands", "damping_native_ids"}
    packet = report()
    context = paired.PolicyInput(1., (packet,), ((1., packet),), {"a": -.5})
    with pytest.raises(TypeError):
        context.damping_commands["a"] = 100
    with pytest.raises(ValueError, match="received"):
        paired.PolicyInput(0., (packet,), (), {})
    with pytest.raises(ValueError, match="history"):
        paired.PolicyInput(1., (packet,), ((2., packet),), {})
    with pytest.raises(ValueError, match="current reports"):
        paired.PolicyInput(2., (packet,), ((1., packet),), {}, frozenset({"a"}))
    with pytest.raises(ValueError, match="numeric reference"):
        paired.PolicyInput(1., (packet,), ((1., packet),), {"a": 0.}, frozenset({"a"}))


@pytest.mark.parametrize("phase,commands,members", [
    ("release", {"a": 0.}, ("a",)), ("abort", {"a": 0.}, ("a",)),
    ("assembly", {"hdv": 1.}, ("a",)), ("hold", {"a": float("nan")}, ("a",)),
    ("unknown", {}, ()), ("damping", {}, ("a", "a")),
])
def test_invalid_or_retained_command_after_release_rejected(phase, commands, members):
    with pytest.raises(ValueError):
        paired.PolicyOutput(commands, phase, member_ids=members)


@pytest.mark.parametrize("arm", ["damping", "wall", "sham"])
def test_integrated_owner_timing_measured_telemetry_and_full_accounting(tmp_path, runtime, arm):
    output = tmp_path / arm
    contexts = []
    policy = constant_policy(arm)
    def observe(context):
        contexts.append(context)
        return policy(context)
    result = paired.evaluate_arm(job=job(arm), registration=registration(), output=output, policy=observe)
    assert runtime["starts"] == 1 and runtime["closed"]
    assert result["mechanics_only"] and result["censored_service"]
    assert result["population"] == 2 and result["totals"]["acceleration_energy"] == .25
    assert result["ownership_invariants"]["hdv_commands"] == 0
    assert all(row[2] == "a" for row in runtime["writes"])
    assert contexts[1].reports[0].owner == "native"  # No command before vehicle's first report.
    assert contexts[2].reports[0].owner == ("wall" if arm == "wall" else "damping")
    rows = [json.loads(x) for x in (output / "telemetry.jsonl").read_text().splitlines()]
    responses = [x for x in rows if x["kind"] == "response"]
    assert [r["timestamp_s"] for r in responses] == [1.5, 2., 2.5, 3., 3.5, 4.]
    assert all(r["measured_acceleration_mps2"] == -.25 for r in responses)
    commands = [x for x in rows if x["kind"] == "command"]
    assert commands[0]["issued_acceleration_mps2"] == (-1. if arm == "wall" else -.5)
    assert commands[1]["issued_acceleration_mps2"] == -.5  # Identical damping after release.
    manifest, checked = paired.verified_arm(output)
    assert manifest["status"] == "complete" and checked == result
    formation = [json.loads(x) for x in (output / "formation.jsonl").read_text().splitlines()]
    assert [x["time_s"] for x in formation] == ([1., 1.5, 2., 2.5] if arm == "wall" else [1., 1.5])
    assert formation[0]["members"][0]["position_m"] == 100.
    assert formation[0]["owner_by_member"] == {"a": "wall" if arm == "wall" else "damping"}
    if arm == "wall":
        assert formation[2]["phase"] == "release" and formation[2]["owner_by_member"] == {"a": "damping"}
        assert formation[2]["wall_command_ids"] == []


def test_gentle_policy_runs_through_actual_paired_logger_and_owner(tmp_path, runtime):
    """Regression for the t600 duplicate-assignment failure, no SUMO launch.

    Exercise the actual evaluator emitter, not a reconstructed logging call.
    The original mechanics harness did not include this integration boundary.
    """
    from dataclasses import asdict
    from traffic_shepherd.gentle_wall import GentleProfile, policy_factory, IMPLEMENTATION_REVISION
    spec = job('wall', duration=4, decision=1)
    spec['policy_spec']['profile'] = asdict(GentleProfile())
    registered = registration()
    output = tmp_path/'gentle_logger'
    result = paired.evaluate_arm(job=spec,registration=registered,output=output,
                                 policy=policy_factory(spec,registered))
    rows = [json.loads(line) for line in (output/'policy.jsonl').read_text().splitlines()]
    assert result['ownership_invariants']['hdv_commands'] == 0
    assert any(r['wall_commands'] for r in rows)
    assert rows[1]['assignment'] == registered['assignment']
    assert rows[1]['implementation_revision'] == IMPLEMENTATION_REVISION
    assert rows[1]['experimental_authorization'] == registered['assignment']
    assert paired.verified_arm(output)[0]['status'] == 'complete'


def test_multiple_registered_snapshots_preserve_exact_prefixes(tmp_path, runtime):
    spec = job()
    spec["snapshot_times_s"] = [1, 3]
    output = tmp_path / "multi"
    paired.evaluate_arm(job=spec, registration=registration(), output=output, policy=constant_policy("damping"))
    snapshots = json.loads((output / "decision_snapshots.json").read_text())
    assert set(snapshots) == {"1", "3"}
    assert snapshots["1"]["predecision_command_prefix_sha256"] != snapshots["3"]["predecision_command_prefix_sha256"]
    assert snapshots["3"]["evidence"]["reports"][0]["frame"]["timestamp_s"] == 3.
    assert "synthetic_full_count" not in json.dumps(snapshots["3"]["evidence"])
    state_rows = [json.loads(x) for x in (output / "offline_state.jsonl").read_text().splitlines()]
    assert [x["timestamp_s"] for x in state_rows] == [1., 3.]
    assert {v["id"] for v in state_rows[0]["vehicles"]} == {"a", "hdv"}


def test_registered_sham_has_exact_damping_telemetry_and_physics(tmp_path, runtime):
    first = tmp_path / "damping"
    data = registration()
    paired.evaluate_arm(job=job(), registration=data, output=first, policy=constant_policy("damping"))
    runtime.update(time=0., writes=[], closed=False)
    second = tmp_path / "sham"
    paired.evaluate_arm(job=job("sham"), registration=data, output=second, policy=constant_policy("sham"))
    for filename in ("telemetry.jsonl", "intervals.jsonl", "traffic.jsonl", "decision_snapshots.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    comparison = paired.verify_pair(first, second)
    assert comparison["mechanics_only"] and comparison["effect"] is None and comparison["sham_exact"]


def test_no_benefit_criteria_allowed_only_for_no_wall_short_mechanics(tmp_path, runtime):
    paired.register(tmp_path / "short", jobs=[job(), job("sham")], hypotheses=["fixture"],
                    criteria=None, stopping_rules=["fixture"], policy_version="fixture")
    with pytest.raises(ValueError, match="criteria"):
        paired.register(tmp_path / "wall", jobs=[job(), job("wall")], hypotheses=["fixture"],
                        criteria=None, stopping_rules=["fixture"], policy_version="fixture")


def test_saved_prefix_comparison_ignores_only_new_rsu_channel(tmp_path, runtime):
    replay = tmp_path / "replay"
    paired.evaluate_arm(job=job(), registration=registration(), output=replay, policy=constant_policy("damping"))
    original = tmp_path / "original"
    original.mkdir()
    manifest = json.loads((replay / "manifest.json").read_text())
    for name in ("telemetry", "intervals", "traffic"):
        rows = [json.loads(x) for x in (replay / f"{name}.jsonl").read_text().splitlines()]
        if name == "telemetry":
            rows = [r for r in rows if r["kind"] != "rsu_received_report"]
        (original / f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        manifest[f"{name}_sha256"] = paired.sha(original / f"{name}.jsonl")
    paired.write_json(original / "manifest.json", manifest)
    result = paired.compare_saved_prefix(original, replay, duration_s=4)
    assert all(row["exact"] for row in result["comparisons"].values())
    assert not result["complete_horizon_replay"]


def test_failed_early_intervention_is_preserved_without_relaunch(tmp_path, runtime):
    output = tmp_path / "invalid"
    def early(context):
        return paired.PolicyOutput({"a": -1.}, "assembly", member_ids=("a",))
    with pytest.raises(ValueError, match="before its decision"):
        paired.evaluate_arm(job=job("wall"), registration=registration(), output=output, policy=early)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / "summary.json").exists() and runtime["closed"]
    with pytest.raises(FileExistsError):
        paired.evaluate_arm(job=job("wall"), registration=registration(), output=output, policy=early)
    assert runtime["starts"] == 1


def test_preexisting_connection_is_not_owned(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(sys.modules["traci"], "isLoaded", lambda: True)
    with pytest.raises(RuntimeError, match="pre-existing"):
        paired.evaluate_arm(job=job(), registration=registration(), output=tmp_path / "no", policy=constant_policy("damping"))
    assert runtime["starts"] == 0 and not runtime["closed"]


def test_corrupt_snapshot_fails_verification(tmp_path, runtime):
    output = tmp_path / "corrupted"
    paired.evaluate_arm(job=job(), registration=registration(), output=output, policy=constant_policy("damping"))
    (output / "decision_snapshots.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshots"):
        paired.verified_arm(output)


def test_snapshot_hash_claims_are_reconstructed_from_independent_raw_logs(tmp_path, runtime):
    output = tmp_path / "false-claimed-prefix"
    paired.evaluate_arm(job=job(), registration=registration(), output=output, policy=constant_policy("damping"))
    snapshot_path, manifest_path = output / "decision_snapshots.json", output / "manifest.json"
    snapshots, manifest = json.loads(snapshot_path.read_text()), json.loads(manifest_path.read_text())
    snapshots["1"]["predecision_command_prefix_sha256"] = "invented"
    paired.write_json(snapshot_path, snapshots)
    manifest["decision_snapshots_sha256"] = paired.sha(snapshot_path)
    manifest["predecision_command_prefix_sha256"] = "invented"
    paired.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="reconstruct"):
        paired.verified_arm(output)


def test_zero_unfinished_due_requests_does_not_mean_future_population_served(tmp_path, runtime, monkeypatch):
    build = paired.build_scenario
    def with_future(*args, **kwargs):
        scene, requests, metadata = build(*args, **kwargs)
        requests.append(TripRequest("future", 100., "r", "normal_car", False))
        return scene, requests, metadata
    monkeypatch.setattr(paired, "build_scenario", with_future)
    monkeypatch.setattr(sys.modules["traci"].simulation, "getArrivedIDList",
                        lambda: ["a", "hdv"] if runtime["time"] == 1. else [])
    summary = paired.evaluate_arm(job=job(), registration=registration(), output=tmp_path / "future",
                                  policy=constant_policy("damping"))
    assert summary["unfinished_requested"] == 0 and summary["future_requests_at_horizon"] == 1
    assert summary["censored_service"] and not summary["all_requested_service_completed"]


@pytest.mark.parametrize("field,value", [("penetration", .5), ("seed", 912001),
    ("decision_time_s", 1.5), ("duration_s", 2401), ("id", "../escape"), ("role", "training")])
def test_forbidden_case_or_budget_refused(field, value):
    selected = job()
    selected[field] = value
    with pytest.raises(ValueError):
        paired.validate_job(selected)


def register_fixture(tmp_path, runtime, jobs=None):
    return paired.register(tmp_path, jobs=jobs or [job(), job("wall")], hypotheses=["synthetic registration"],
        criteria=criteria(), stopping_rules=["stop on implementation mismatch"], policy_version="fixture/1")


def test_registration_shared_reference_and_bounded_allocation(tmp_path, runtime, monkeypatch):
    reference = job(name="reference")
    reference["snapshot_times_s"] = [1, 3]
    wall1, wall3 = job("wall", name="wall1"), job("wall", name="wall3", decision=3)
    wall1["reference_job_id"], wall3["reference_job_id"] = "reference", "reference"
    wall1["case_id"], wall3["case_id"] = "opportunity1", "opportunity3"
    root = tmp_path / "registered"
    result = register_fixture(root, runtime, [reference, wall1, wall3])
    assert result["maximum_attempts"] == 3 and result["maximum_simulated_seconds"] == 12
    assert runtime["starts"] == 0
    def fail(**kwargs):
        raise RuntimeError("synthetic charged failure")
    monkeypatch.setattr(paired, "evaluate_arm", fail)
    with pytest.raises(RuntimeError, match="charged"):
        paired.run_one(root / "registration.json", "wall1", policy_factory=lambda *a: None)
    assert json.loads((root / "allocations/wall1.json").read_text())["allocated_simulated_seconds"] == 4
    with pytest.raises(FileExistsError):
        paired.run_one(root / "registration.json", "wall1", policy_factory=lambda *a: None)
    assert paired.analyze(root / "registration.json")["allocated"] == 1


def test_source_change_and_stop_refuse_new_allocation(tmp_path, runtime, monkeypatch):
    root = tmp_path / "registered"
    register_fixture(root, runtime)
    identity = paired.current_identity()
    monkeypatch.setattr(paired, "current_identity", lambda: dict(identity, model_sha256="changed"))
    with pytest.raises(ValueError, match="register a new version"):
        paired.run_one(root / "registration.json", "wall", policy_factory=lambda *a: None)
    assert not (root / "allocations").exists()
    monkeypatch.setattr(paired, "current_identity", lambda: identity)
    (root / "STOP.json").write_text('{"reason":"fixture stop"}')
    with pytest.raises(RuntimeError, match="stopped"):
        paired.run_one(root / "registration.json", "wall", policy_factory=lambda *a: None)


def test_registration_tampering_refused(tmp_path, runtime):
    root = tmp_path / "registered"
    register_fixture(root, runtime)
    path = root / "registration.json"
    data = json.loads(path.read_text())
    data["maximum_simulated_seconds"] = 100000
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="modified"):
        paired.read_registration(path)


def test_fixed_reference_and_sham_materialize_same_proposal_without_wall_commands():
    reference_job = job()
    reference_job["snapshot_times_s"] = [1, 3]
    reference_job["policy_spec"] = dict(opportunities=[
        dict(time_s=t, required_lanes=["L0"], member_ids=["a"], profile={}) for t in (1, 3)])
    reference = paired.fixed_formation_policy(reference_job, {"registration_sha256": "fixture"})
    outputs = []
    for now in (1., 2., 3.):
        packet = report(now)
        outputs.append(reference(paired.PolicyInput(now, (packet,), ((now, packet),), {"a": -.5})))
    assert all(not output.wall_commands and output.phase == "damping" for output in outputs)
    assert outputs[0].proposal_sha256 == outputs[1].proposal_sha256
    assert outputs[2].proposal_sha256 != outputs[0].proposal_sha256
    assert outputs[0].details["proposal_materialized"]


def test_missing_member_report_is_retained_refusal_not_exception_or_imputed_space():
    treatment = paired.fixed_formation_policy(job("wall"), {"registration_sha256": "fixture"})
    refused = treatment(paired.PolicyInput(1., (), (), {}))
    assert refused.phase == "refused" and not refused.wall_commands
    assert refused.reason.startswith("proposal_unavailable:")
    assert not refused.details["proposal_materialized"] and refused.proposal_sha256
    assert treatment(paired.PolicyInput(2., (), (), {})) == refused


def test_safety_refusal_never_creates_experimental_authorization():
    treatment = paired.fixed_formation_policy(job("wall"), {"registration_sha256": "fixture"})
    packet = report()
    data = packet.payload()
    data["frame"]["front"] = NeighborMeasurement(2., -5.)
    data["frame"]["rear"] = None
    unsafe = CAVReport(SensorFrame(**data.pop("frame")), **data)
    refused = treatment(paired.PolicyInput(1., (unsafe,), ((1., unsafe),), {"a": -.5}))
    assert refused.phase == "refused" and refused.reason.startswith("unsafe_geometry:")
    assert not refused.wall_commands


def test_taper_valid_native_reference_releases_without_inventing_acceleration():
    selected = job("wall")
    selected["policy_spec"]["profile"] = dict(target_speed_reduction_mps=0., readiness_samples=2, hold_s=1.)
    policy = paired.fixed_formation_policy(selected, {"registration_sha256": "fixture"})
    for now in (1., 2.):
        packet = report(now)
        result = policy(paired.PolicyInput(now, (packet,), ((now, packet),), {"a": -.5}))
        assert result.wall_commands
    packet = report(3.)
    result = policy(paired.PolicyInput(3., (packet,), ((3., packet),), {}, frozenset({"a"})))
    assert result.phase == "release" and not result.wall_commands
    assert result.reason == "identical_damping_requests_native_explicit_handback"


def test_absent_reference_at_taper_is_not_treated_as_native():
    selected = job("wall")
    selected["policy_spec"]["profile"] = dict(target_speed_reduction_mps=0., readiness_samples=2, hold_s=1.)
    policy = paired.fixed_formation_policy(selected, {"registration_sha256": "fixture"})
    for now in (1., 2.):
        packet = report(now)
        policy(paired.PolicyInput(now, (packet,), ((now, packet),), {"a": -.5}))
    packet = report(3.)
    result = policy(paired.PolicyInput(3., (packet,), ((3., packet),), {}))
    assert result.phase == "abort" and result.reason == "missing_current_local_damping_reference"


def test_runner_derives_explicit_native_mask_from_fresh_controller_decisions(tmp_path, runtime, monkeypatch):
    class NativeActor:
        def __init__(self, config):
            self.config = config
        def decide(self, frame, *, now_s):
            return DampingDecision(fallback_reason="missing_or_uncovered_leader")
    monkeypatch.setattr(paired, "CirclesController", NativeActor)
    seen = []
    def observe(context):
        seen.append(context)
        return constant_policy("damping")(context)
    paired.evaluate_arm(job=job(), registration=registration(), output=tmp_path / "native-ref", policy=observe)
    assert seen[0].damping_native_ids == frozenset()
    assert all(c.damping_native_ids == frozenset({"a"}) and not c.damping_commands for c in seen[1:])
    assert not runtime["writes"]
