"""Registration draft/finalize/check round-trip (E:/ paths pinned)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaign_walls_v1 import plan_walls_pilot as plan_mod
from campaign_walls_v1 import walls_pilot_registration as reg


@pytest.fixture()
def registered_package(source_bundle, tmp_path):
    """A tmp package with a real plan + registration, and COMPLETE native
    sibling receipts beside the fake chunk bases."""
    pkg = tmp_path / "pkg" / "campaign_walls_v1"
    plan_dir = pkg / "results_plan"
    plan = plan_mod.build_plan(
        source_bundle["plans"]["chunk2"], source_bundle["plans"]["chunk3"],
        bases_roots=source_bundle["bases_roots"])
    plan_path = plan_mod.write_plan(plan, plan_dir)
    # every selected world's native sibling gets a COMPLETE receipt
    for world in plan["worlds"]:
        runs_root = Path(world["native_sibling"]["runs_root"])
        receipt_dir = runs_root / world["native_sibling"]["run"]
        receipt_dir.mkdir(parents=True, exist_ok=True)
        (receipt_dir / "RUN_RECEIPT.json").write_text(
            json.dumps({"schema": "wm41_branch_receipt/1",
                        "run": world["native_sibling"]["run"],
                        "exit_status": "complete"}) + "\n",
            encoding="utf-8")
    paths = {
        "registration": plan_dir / "WALLS_PILOT_REGISTRATION.md",
        "digest": plan_dir / "WALLS_PILOT_REGISTRATION_DIGEST.json",
        "record": plan_dir / "WALLS_PILOT_REGISTRATION_RECORD.json",
        "plan": plan_path,
    }
    return {"pkg": pkg, "paths": paths, "plan": plan,
            "source_bundle": source_bundle}


def test_draft_then_check_round_trip(registered_package):
    result = reg.draft(plan_path=registered_package["paths"]["plan"],
                       package=registered_package["pkg"],
                       paths=registered_package["paths"])
    assert result["registration_kind"] == "draft"
    for key in ("registration", "digest", "record"):
        assert Path(result[key]).is_file()
    verified = reg.check(registered_package["paths"]["record"],
                         package=registered_package["pkg"],
                         verify_gate_files=False)
    assert verified["verified"] is True
    text = Path(result["registration"]).read_text(encoding="utf-8")
    assert "{registered_utc}" not in text and "WM-59" in text
    assert "144,000" in text and "45 GB" in text
    record = json.loads(Path(result["record"]).read_text(encoding="utf-8"))
    assert record["declared_workers"] == 10
    assert record["disk_preflight_min_free_gb"] == 45.0
    assert record["cav_speed_mode"] == 27
    assert record["simulated_seconds_total"] == 144000.0
    assert len(record["worlds_table"]) == 60
    # E:/ paths pinned
    assert record["launch_paths"]["runs_root"].startswith("E:/")
    assert record["launch_paths"]["authorization"].startswith("E:/")
    assert record["smoke"]["verdict_path"].startswith("E:/")
    assert record["gate_paths"]["wm41_verdict"].startswith("E:/")
    commands = record["commands"]
    assert commands["smoke"].startswith("C:/Users/jpeng/Miniconda3")
    assert "launch_walls.py launch" in commands["launch"]
    assert commands["launch_resume"].endswith("--resume")


def test_finalize_is_write_once_and_check_round_trips(registered_package):
    reg.draft(plan_path=registered_package["paths"]["plan"],
              package=registered_package["pkg"],
              paths=registered_package["paths"])
    final = reg.finalize(plan_path=registered_package["paths"]["plan"],
                         package=registered_package["pkg"],
                         paths=registered_package["paths"])
    assert final["registration_kind"] == "final"
    assert reg.check(final["record"], package=registered_package["pkg"],
                     verify_gate_files=False)["verified"] is True
    with pytest.raises(reg.WallsRegistrationError, match="FINAL"):
        reg.finalize(plan_path=registered_package["paths"]["plan"],
                     package=registered_package["pkg"],
                     paths=registered_package["paths"])
    with pytest.raises(reg.WallsRegistrationError, match="FINAL"):
        reg.draft(plan_path=registered_package["paths"]["plan"],
                  package=registered_package["pkg"],
                  paths=registered_package["paths"])


def test_check_refuses_a_changed_plan(registered_package):
    reg.draft(plan_path=registered_package["paths"]["plan"],
              package=registered_package["pkg"],
              paths=registered_package["paths"])
    plan_path = Path(registered_package["paths"]["plan"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["plan_seed"] = 999999
    plan_path.write_text(json.dumps(payload, indent=1) + "\n",
                         encoding="utf-8")
    with pytest.raises(reg.WallsRegistrationError, match="plan sha256"):
        reg.check(registered_package["paths"]["record"],
                  package=registered_package["pkg"], verify_gate_files=False)


def test_check_refuses_changed_or_missing_base_state(registered_package):
    reg.draft(plan_path=registered_package["paths"]["plan"],
              package=registered_package["pkg"],
              paths=registered_package["paths"])
    world = registered_package["plan"]["worlds"][0]
    state_xml = (Path(world["bases_root"]) / world["base_run"]
                 / "saved_state" / "state.xml")
    state_xml.write_text("<tampered/>", encoding="utf-8")
    with pytest.raises(reg.WallsRegistrationError, match="state.xml digest"):
        reg.check(registered_package["paths"]["record"],
                  package=registered_package["pkg"], verify_gate_files=False)


def test_check_requires_complete_native_sibling_receipts(registered_package):
    reg.draft(plan_path=registered_package["paths"]["plan"],
              package=registered_package["pkg"],
              paths=registered_package["paths"])
    world = registered_package["plan"]["worlds"][0]
    receipt = (Path(world["native_sibling"]["runs_root"])
               / world["native_sibling"]["run"] / "RUN_RECEIPT.json")
    receipt.unlink()
    with pytest.raises(reg.WallsRegistrationError, match="native sibling"):
        reg.check(registered_package["paths"]["record"],
                  package=registered_package["pkg"], verify_gate_files=False)
    receipt.write_text(json.dumps({"exit_status": "failed"}) + "\n",
                       encoding="utf-8")
    with pytest.raises(reg.WallsRegistrationError, match="not complete"):
        reg.check(registered_package["paths"]["record"],
                  package=registered_package["pkg"], verify_gate_files=False)


def test_gate_files_verified_when_requested(registered_package, monkeypatch):
    # point the PINNED gate paths at a missing file: check(verify_gate_files=
    # True) refuses; the default check() call inside draft/finalize skips
    # the filesystem probe (the real gates are read at launch time)
    monkeypatch.setattr(reg, "WM41_VERDICT", "E:/missing/EXACTNESS_VERDICT.json")
    reg.draft(plan_path=registered_package["paths"]["plan"],
              package=registered_package["pkg"],
              paths=registered_package["paths"])
    with pytest.raises(reg.WallsRegistrationError, match="gate file missing"):
        reg.check(registered_package["paths"]["record"],
                  package=registered_package["pkg"], verify_gate_files=True)


def test_base_state_digests_cover_all_sixty_worlds(registered_package):
    record_rows = reg.worlds_record(registered_package["plan"])
    assert len(record_rows) == 60
    for row in record_rows:
        assert row["base_state"]["state_xml_sha256"]
        assert row["base_state"]["state_manifest_sha256"]
        assert row["base_state"]["saved_state_id"]
        assert row["bases_root"].endswith(("chunk2", "chunk3"))
        assert row["native_sibling_run"].endswith("__native")
