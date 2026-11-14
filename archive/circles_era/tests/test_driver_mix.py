"""No-SUMO contract tests for the OE-43 driver-mix adapter."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).parents[1]
SCENARIO_PATH = ROOT / "output" / "driver_mix_v1" / "scenario.py"
spec = importlib.util.spec_from_file_location("driver_mix_scenario", SCENARIO_PATH)
scenario = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenario)


sys.path.insert(0, str(SCENARIO_PATH.parent))
study_spec = importlib.util.spec_from_file_location("driver_mix_study", SCENARIO_PATH.parent / "study.py")
study = importlib.util.module_from_spec(study_spec)
study_spec.loader.exec_module(study)
campaign_spec = importlib.util.spec_from_file_location("driver_mix_campaign", SCENARIO_PATH.parent / "campaign.py")
campaign = importlib.util.module_from_spec(campaign_spec)
campaign_spec.loader.exec_module(campaign)


@dataclass(frozen=True)
class Request:
    vehicle_id: str
    depart_s: float
    route: str
    driver_type: str
    is_cav: bool


def test_replacement_preserves_identity_schedule_routes_cav_and_non_aggressive():
    source = [Request("a", 1.25, "mainline_route", "aggressive_car", True),
              Request("b", 2.5, "ramp_route", "cautious_car", False),
              Request("c", 4.0, "mainline_route", "normal_car", True)]
    changed = scenario.replace_aggressive_with_normal(source)
    assert [item.vehicle_id for item in changed] == [item.vehicle_id for item in source]
    assert [(item.depart_s, item.route, item.is_cav) for item in changed] == \
           [(item.depart_s, item.route, item.is_cav) for item in source]
    assert [item.driver_type for item in changed] == ["normal_car", "cautious_car", "normal_car"]


def test_builder_rewrites_only_explicit_assignments_and_preserves_default_artifacts(tmp_path):
    source = [Request("a", 1.25, "mainline_route", "aggressive_car", True),
              Request("b", 2.5, "ramp_route", "cautious_car", False)]
    def original(root, *args, **kwargs):
        root = Path(root)
        root.mkdir(parents=True)
        route = ET.Element("routes")
        ET.SubElement(route, "vType", id="aggressive_car", accel="3")
        for item in source:
            ET.SubElement(route, "vehicle", id=item.vehicle_id, type=item.driver_type)
        route_path = root / "requests.rou.xml"
        route_path.write_text(ET.tostring(route, encoding="unicode"), encoding="utf-8")
        scene = root / "scenario.sumocfg"
        scene.write_text("<configuration/>", encoding="utf-8")
        return scene, source, {"schema": "native_damping_explicit_requests/1", "population": 2}
    scene, changed, manifest = scenario.make_builder(original)(tmp_path / "scene", 1, 2000,
                                                                penetration=.2)
    assert [r.driver_type for r in changed] == ["normal_car", "cautious_car"]
    types = [v.attrib["type"] for v in ET.parse(scene.parent / "requests.rou.xml").getroot().findall("vehicle")]
    assert types == ["normal_car", "cautious_car"]
    assert manifest["schema"] == scenario.SCHEMA
    assert manifest["replacement_counts"] == {"aggressive_original": 1, "aggressive_replaced": 1,
                                                "normal_final": 1, "cautious_final": 1}


def test_builder_scope_restores_paired_builder_after_success_and_failure():
    original = study.paired.build_scenario
    with study.driver_mix_builder_scope() as selected:
        assert study.paired.build_scenario is selected
    assert study.paired.build_scenario is original
    try:
        with study.driver_mix_builder_scope():
            raise RuntimeError("synthetic failure")
    except RuntimeError:
        pass
    assert study.paired.build_scenario is original


def test_binding_is_derived_from_modified_damping_snapshot(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    damping = run_root / "s90030_damping"
    damping.mkdir(parents=True)
    (damping / "manifest.json").write_text("{}", encoding="utf-8")
    (damping / "decision_snapshots.json").write_text(
        '{"600": {"predecision_report_evidence_sha256": "evidence"}}', encoding="utf-8")
    monkeypatch.setattr(campaign, "RUNS", run_root)
    monkeypatch.setattr(campaign.pe, "verified_arm", lambda path: ({"registration_sha256": "reg"}, {}))
    monkeypatch.setattr(campaign, "load_registration", lambda: {
        "registration_sha256": "reg",
        "factory": {"implementation_hashes": {"scenario.py": "s", "study.py": "t"}},
    })
    selected = {"member_ids": ["a", "b"], "proposal_sha256": "proposal"}
    selection = {"selected": selected, "selection_sha256": "selection"}
    monkeypatch.setattr(campaign.cohort_study, "snapshot_selection", lambda snapshot: selection)
    monkeypatch.setattr(campaign.pe, "validate_job", lambda job: None)
    campaign.bind(90030)
    binding = campaign.read(run_root / "bindings" / "s90030_h20.json")
    assert binding["job"]["reference_job_id"] == "s90030_damping"
    assert binding["job"]["policy_spec"]["member_ids"] == ["a", "b"]
    assert binding["selection"] == selection


def test_run_requires_and_passes_persistent_budget_block(tmp_path, monkeypatch):
    output_root = tmp_path / "runs"
    captured = {}
    registration = {"factory": {"implementation_hashes": {}}, "registration_sha256": "reg"}
    monkeypatch.setattr(campaign, "RUNS", output_root)
    monkeypatch.setattr(campaign, "LEDGER", tmp_path / "budget.json")
    monkeypatch.setattr(campaign, "load_registration", lambda: registration)
    monkeypatch.setattr(campaign, "DampingPolicy", lambda job, data: "policy")

    class Guard:
        def __init__(self, path):
            captured["ledger_path"] = path
        def mark_progress(self, reservation_id, seconds):
            captured["progress"] = (reservation_id, seconds)
        def stop_campaign(self, *args):
            raise AssertionError("synthetic run should not stop")
    monkeypatch.setattr(campaign, "BudgetGuard", Guard)

    class FakeStudy:
        @staticmethod
        def evaluate_arm(**kwargs):
            assert kwargs["budget"]["amount_s"] == 2400
            assert kwargs["budget"]["campaign"] == campaign.CAMPAIGN
            captured["budget"] = kwargs["budget"]
            return {"status": "complete"}
    monkeypatch.setitem(sys.modules, "study", FakeStudy)
    real_read = campaign.read
    monkeypatch.setattr(campaign, "read", lambda path: {
        "reservations": [{"id": "oe43:damping:1", "target_dir": str(output_root / "s90030_damping")}]
    } if Path(path) == campaign.LEDGER else real_read(path))
    campaign.run(90030, "damping")
    assert captured["budget"]["ceilings"]["total_seconds"] == 9600
    assert captured["progress"] == ("oe43:damping:1", 2400)
