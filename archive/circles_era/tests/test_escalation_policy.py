"""Synthetic artifact/runtime checks only; no physical qualification or fitting."""
from dataclasses import FrozenInstanceError, asdict, replace
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


HERE = Path(__file__).resolve().parent
if (HERE / "escalation_policy.py").is_file():
    import traffic_shepherd
    package = ModuleType("traffic_shepherd_scheduler_stage")
    package.__path__ = [str(HERE), str(Path(traffic_shepherd.__file__).parent)]
    sys.modules[package.__name__] = package
    policy = importlib.import_module(package.__name__+".escalation_policy")
    formation = importlib.import_module(package.__name__+".formation")
    runtime = importlib.import_module(package.__name__+".paired_experiments")
    reporting = importlib.import_module(package.__name__+".reporting")
    # Formation and proposal_features use these exact report classes from their
    # runtime namespace; sensing imports are structurally shared.
    reporting = importlib.import_module("traffic_shepherd.reporting")
    # Stage package aliases preserve the production typed packet boundary.
    sys.modules[package.__name__+".reporting"] = reporting
    sys.modules[package.__name__+".proposal_features"].CAVReport = reporting.CAVReport
else:
    from traffic_shepherd import escalation_policy as policy
    from traffic_shepherd import formation, paired_experiments as runtime, reporting


def settings(times=(1, 2, 3, 4, 5, 6, 7, 8), cooldown=2, attempts=2):
    return policy.SchedulerConfig(("L0", "L1"), tuple(times), cooldown, attempts)


def profile():
    return formation.FormationProfile(target_speed_reduction_mps=0., readiness_samples=2, hold_s=1., taper_s=1.)


def packets(now, *, missing=(), lane_loss=False, rear_unsafe=False):
    reports = []
    for i, vehicle in enumerate(("a", "b")):
        if vehicle in missing:
            continue
        frame = SensorFrame(vehicle, float(now), 20., 0., 30., i, distance_to_merge_m=1850.+i,
            front=NeighborMeasurement(50., 0.), rear=NeighborMeasurement(50., 0.) if not rear_unsafe else NeighborMeasurement(2., 5.),
            front_coverage_m=100., rear_coverage_m=100., lane_id="wrong" if lane_loss and vehicle == "b" else f"L{i}",
            position_m=100.-i)
        reports.append(reporting.CAVReport(frame, float(now), "damping", -.5, -.5))
    return tuple(reports)


def context(now, **kwargs):
    reports = packets(now, **kwargs)
    # runtime.PolicyInput checks against its defining module's CAVReport.
    runtime.CAVReport = reporting.CAVReport
    return runtime.PolicyInput(float(now), reports, tuple((float(now), r) for r in reports),
                               {r.frame.vehicle_id: -.5 for r in reports})


def write(root, name, data):
    path = root / name
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    import hashlib
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def artifacts(tmp_path, monkeypatch, *, config=None, treatment=None, benefit_threshold=0., logistic=False):
    """Construct synthetic positive schema fixtures, never physical evidence."""
    config, treatment = config or settings(), treatment or profile()
    identity = dict(model_sha256="fixed-test-identity", active_source_sha256={"fixture": "source"})
    monkeypatch.setattr(policy, "current_identity", lambda: identity)
    feasibility = dict(schema="frozen_escalation_gate/1", feature_schema=policy.SCHEMA, kind="feasibility",
        method="analytical_formation", provenance_kind="qualified_analytical_rule", feature_names=[],
        required_true_features=["member_0_report_present", "member_1_report_present"],
        parameters=dict(minimum_estimated_runway_slack_m=0., maximum_estimated_assembly_s=50.))
    benefit = dict(schema="frozen_escalation_gate/1", feature_schema=policy.SCHEMA, kind="benefit",
        method="conjunction_rule", provenance_kind="preregistered_transparent_rule", feature_names=["target_speed_mps"],
        required_true_features=["member_0_report_present", "member_1_report_present"],
        parameters=dict(rules=[dict(feature="target_speed_mps", operator=">=", value=benefit_threshold)]))
    if logistic:
        benefit.update(method="logistic", provenance_kind="trained_from_offline_matched_physical_pairs",
            parameters=dict(coefficients=[0.], intercept=2., means=[20.], scales=[1.], threshold=.8))
        provenance = dict(schema="offline_logistic_fit_provenance/1", origin="offline_matched_physical_pairs",
            feature_schema=policy.SCHEMA, parameters_sha256=policy.digest(benefit["parameters"]),
            seed_partitions=dict(training=[100], calibration=[200], prospective=[300]),
            dataset_manifest=write(tmp_path, "dataset.json", dict(test_fixture=True)),
            exposure_audit=write(tmp_path, "exposure.json", dict(test_fixture=True)))
        benefit["training_provenance"] = write(tmp_path, "fit.json", provenance)
    gates = dict(feasibility=write(tmp_path, "feasibility.json", feasibility),
                 benefit=write(tmp_path, "benefit.json", benefit))
    bindings = dict(runtime_identity=identity, feature_schema=policy.SCHEMA, scheduler_config=asdict(config),
                    formation_profile=asdict(treatment), gate_sha256={k:v["sha256"] for k,v in gates.items()})
    evidence = {}
    for name, origin in policy.CHECKS.items():
        criteria = [dict(metric="synthetic_contract_check", operator=">=", limit=1.)]
        prior = write(tmp_path, f"criteria_{name}.json", dict(schema="registered_qualification_criteria/1",
            check=name, criteria=criteria))
        measured = dict(schema="escalation_qualification_evidence/1", check=name, status="passed", origin=origin,
            qualification_target_sha256=policy.digest(bindings), preregistered_criteria=prior, criteria=criteria,
            observed=dict(synthetic_contract_check=1.), supporting_artifacts=[write(tmp_path, f"support_{name}.json",
                dict(test_fixture=True, scientific_evidence=False))])
        evidence[name] = write(tmp_path, f"evidence_{name}.json", measured)
    manifest = dict(schema="escalation_gate_qualification/1", status="qualified_for_policy_execution",
        runtime_identity=identity, feature_schema=policy.SCHEMA, scheduler_config=asdict(config),
        formation_profile=asdict(treatment), gates=gates, evidence=evidence)
    manifest["qualification_sha256"] = policy.digest(manifest)
    write(tmp_path, "qualification.json", manifest)
    return tmp_path / "qualification.json"


def bundle(tmp_path, monkeypatch, **kwargs):
    return policy.QualifiedGateBundle.load(artifacts(tmp_path, monkeypatch, **kwargs))


def test_missing_qualification_abstains_and_keeps_all_opportunities(tmp_path):
    scheduler = policy.EscalationScheduler.from_manifest(settings((1, 2)), profile=profile(),
                                                        qualification_path=tmp_path / "absent.json")
    for now in (0, 1, 2):
        result = scheduler(context(now))
        assert not result.wall_commands
    assert scheduler.attempts == 0
    assert [event["reason"] for event in scheduler.events] == ["qualification_artifacts_unavailable"]*2
    assert all(not event["approval"] for event in scheduler.events)


def test_loader_freezes_rules_and_does_not_call_them_fitted_probabilities(tmp_path, monkeypatch):
    loaded = bundle(tmp_path, monkeypatch)
    with pytest.raises(FrozenInstanceError):
        loaded.sha256 = "changed"
    with pytest.raises(TypeError):
        loaded.gates["benefit"].data["parameters"]["rules"][0]["value"] = 100
    scheduler = policy.EscalationScheduler(settings(), profile=profile(), qualification=loaded)
    scheduler(context(0))
    result = scheduler(context(1))
    assert result.phase == "assembly" and result.wall_commands
    gate_rows = scheduler.events[0]["gates"]
    assert all(row["probability"] is None for row in gate_rows)
    assert gate_rows[1]["method"] == "conjunction_rule"


def test_one_active_wall_cooldown_fresh_reauthorization_and_action_budget(tmp_path, monkeypatch):
    config = settings(tuple(range(1, 12)), cooldown=2, attempts=2)
    loaded = bundle(tmp_path, monkeypatch, config=config)
    scheduler = policy.EscalationScheduler(config, profile=profile(), qualification=loaded)
    output = [scheduler(context(now)) for now in range(12)]
    assert [output[t].phase for t in (1, 2, 3, 4)] == ["assembly", "hold", "taper", "release"]
    assert not output[4].wall_commands and output[5].reason == "cooldown"
    assert output[6].phase == "assembly"
    assert output[6].proposal_sha256 != output[1].proposal_sha256
    assert scheduler.attempts == 2
    assert output[11].reason == "action_budget_exhausted"
    opportunities = [e for e in scheduler.events if e["event"] == "opportunity"]
    assert len(opportunities) == 11
    assert sum(e["approval"] for e in opportunities) == 2


@pytest.mark.parametrize("problem", ["missing", "lane", "clearance"])
def test_member_report_or_local_safety_loss_aborts_without_new_ownership(tmp_path, monkeypatch, problem):
    scheduler = policy.EscalationScheduler(settings(), profile=profile(), qualification=bundle(tmp_path, monkeypatch))
    scheduler(context(0))
    assert scheduler(context(1)).wall_commands
    kwargs = {"missing": ("b",)} if problem == "missing" else {"lane_loss": True} if problem == "lane" else {"rear_unsafe": True}
    result = scheduler(context(2, **kwargs))
    assert result.phase == "abort" and not result.wall_commands
    assert scheduler(context(3)).reason == "cooldown"


def test_cached_member_report_without_current_local_feedback_cannot_receive_command(tmp_path, monkeypatch):
    scheduler = policy.EscalationScheduler(settings(), profile=profile(), qualification=bundle(tmp_path, monkeypatch))
    scheduler(context(0))
    assert scheduler(context(1)).wall_commands
    current = context(2)
    stale = replace(current.reports[1], frame=replace(current.reports[1].frame, timestamp_s=1.), sent_at_s=1.)
    reports = current.reports[:1]+(stale,)
    missing_execution = runtime.PolicyInput(2., reports, ((2., reports[0]), (1., stale)), {"a": -.5})
    result = scheduler(missing_execution)
    assert result.phase == "abort" and not result.wall_commands
    assert result.reason == "missing_current_local_damping_reference"


def test_valid_native_feedback_is_available_and_taper_hands_back(tmp_path, monkeypatch):
    scheduler = policy.EscalationScheduler(settings(), profile=profile(), qualification=bundle(tmp_path, monkeypatch))
    scheduler(context(0))
    scheduler(context(1))
    scheduler(context(2))
    ctx = context(3)
    native = runtime.PolicyInput(3., ctx.reports, ctx.history, {"a": -.5}, frozenset({"b"}))
    result = scheduler(native)
    assert result.phase == "release" and not result.wall_commands
    assert result.reason == "identical_damping_requests_native_explicit_handback"


def test_benefit_rejection_cannot_be_overridden_by_feasible_geometry(tmp_path, monkeypatch):
    scheduler = policy.EscalationScheduler(settings(), profile=profile(),
        qualification=bundle(tmp_path, monkeypatch, benefit_threshold=100.))
    scheduler(context(0))
    result = scheduler(context(1))
    assert result.reason == "benefit:declared_rule_rejected" and not result.wall_commands
    assert scheduler.attempts == 0


def test_logistic_requires_provenance_and_uses_only_frozen_features(tmp_path, monkeypatch):
    loaded = bundle(tmp_path, monkeypatch, logistic=True)
    scheduler = policy.EscalationScheduler(settings(), profile=profile(), qualification=loaded)
    scheduler(context(0))
    scheduler(context(1))
    row = scheduler.events[0]["gates"][1]
    assert row["probability"] == pytest.approx(.8807970779778823)
    assert row["method"] == "logistic"
    assert "seed" not in loaded.gates["benefit"].data["feature_names"]
    raw = dict(schema="frozen_escalation_gate/1", feature_schema=policy.SCHEMA, kind="benefit", method="logistic",
        feature_names=["target_speed_mps"], required_true_features=[], provenance_kind="manual_score",
        parameters=dict(coefficients=[0.], intercept=2., means=[20.], scales=[1.], threshold=.8))
    with pytest.raises(ValueError, match="manually supplied"):
        policy.FrozenGate(raw, "0"*64)


def test_stale_or_different_feature_authorization_refused(tmp_path, monkeypatch):
    loaded = bundle(tmp_path, monkeypatch)
    ctx = context(1)
    proposal, assessment, _ = policy.select_proposal(ctx.history, now_s=1., required_lanes=("L0", "L1"), profile=profile())
    packet = policy.feature_packet(proposal, ctx.history, now_s=1., history_s=20.)
    authorization = formation.authorize_qualified_gates(proposal, assessment, packet=packet, qualification=loaded)
    assert authorization.source == "qualified_gate_artifacts"
    assert authorization.qualification_sha256 == loaded.sha256
    fabricated = formation.FormationAuthorization(**asdict(authorization))
    with pytest.raises(ValueError, match="verified gate factory"):
        formation.FormationController(proposal, fabricated)
    wrong = dict(packet, now_s=2.)
    with pytest.raises(ValueError, match="altered"):
        formation.authorize_qualified_gates(proposal, assessment, packet=wrong, qualification=loaded)
    wrong_proposal = replace(proposal, target_speed_mps=19.)
    with pytest.raises(ValueError, match="same proposal"):
        loaded.decisions(wrong_proposal, assessment, packet)


def test_fresh_gate_authorization_does_not_survive_changed_activation_packets(tmp_path, monkeypatch):
    loaded = bundle(tmp_path, monkeypatch)
    ctx = context(1)
    proposal, assessment, _ = policy.select_proposal(ctx.history, now_s=1., required_lanes=("L0", "L1"), profile=profile())
    packet = policy.feature_packet(proposal, ctx.history, now_s=1., history_s=20.)
    authorization = formation.authorize_qualified_gates(proposal, assessment, packet=packet, qualification=loaded)
    controller = formation.FormationController(proposal, authorization)
    changed = (replace(ctx.reports[0], owner="native"), ctx.reports[1])
    result = controller.step(changed, now_s=1., damping_commands={"a": -.5, "b": -.5})
    assert result.phase == "abort" and not result.commands


def test_manual_callback_or_qualification_object_cannot_activate():
    with pytest.raises(ValueError, match="manual scores"):
        policy.QualifiedGateBundle({}, {})
    with pytest.raises(TypeError, match="manual"):
        policy.EscalationScheduler(settings(), qualification=lambda packet: .99)


def test_tampered_or_unqualified_artifacts_fail_loading(tmp_path, monkeypatch):
    path = artifacts(tmp_path, monkeypatch)
    (tmp_path / "benefit.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        policy.QualifiedGateBundle.load(path)


def test_scientific_failure_is_not_fixed_by_status_passed_assertion(tmp_path, monkeypatch):
    path = artifacts(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    measured_path = tmp_path / "evidence_incremental_benefit_physical.json"
    measured = json.loads(measured_path.read_text())
    measured["observed"]["synthetic_contract_check"] = 0.
    manifest["evidence"]["incremental_benefit_physical"] = write(tmp_path, measured_path.name, measured)
    manifest["qualification_sha256"] = policy.digest({k:v for k,v in manifest.items() if k != "qualification_sha256"})
    write(tmp_path, path.name, manifest)
    with pytest.raises(ValueError, match="numerical qualification"):
        policy.QualifiedGateBundle.load(path)


def test_changed_runtime_or_scheduler_requires_new_qualification(tmp_path, monkeypatch):
    path = artifacts(tmp_path, monkeypatch)
    loaded = policy.QualifiedGateBundle.load(path)
    with pytest.raises(ValueError, match="scheduler differs"):
        policy.EscalationScheduler(settings(cooldown=3), profile=profile(), qualification=loaded)
    monkeypatch.setattr(policy, "current_identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="identities"):
        policy.QualifiedGateBundle.load(path)


def test_duplicate_time_and_inconsistent_receipt_history_rejected():
    scheduler = policy.EscalationScheduler(settings(), profile=profile())
    scheduler(context(0))
    with pytest.raises(ValueError, match="once"):
        scheduler(context(0))
    scheduler = policy.EscalationScheduler(settings(), profile=profile())
    ctx = context(1)
    wrong = runtime.PolicyInput(1., ctx.reports, (), ctx.damping_commands)
    with pytest.raises(ValueError, match="receipt history"):
        scheduler(wrong)


def test_absence_of_member_reports_has_distinct_opportunity_reason():
    scheduler = policy.EscalationScheduler(settings(), profile=profile())
    scheduler(context(0))
    result = scheduler(context(1, missing=("b",)))
    assert result.reason == "insufficient_members" and not result.wall_commands
    assert scheduler.events[0]["selection"]["missing_reported_lanes"] == ("L1",)
