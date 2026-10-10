"""Frozen wall-law v2.2 patch-SHA gate tests (zero SUMO)."""
from __future__ import annotations

import json

import pytest

from inc1 import wall_v22_freeze as F


def test_frozen_v22_copies_match_patch_shas():
    record = F.verify_frozen_v22_patches()
    assert record["all_match"] is True
    assert {row["file"] for row in record["files"]} == {
        "common.py", "wall_request.py", "wall_group.py", "wall_driver.py",
        "sensing.py"}
    for row in record["files"]:
        assert row["actual_lf_sha256"] == row["expected_lf_sha256"]
        assert row["match"] is True


def test_tampered_patch_shas_refuses(tmp_path):
    manifest = json.loads(F.PATCH_SHAS_PATH.read_text(encoding="utf-8"))
    manifest["post"]["wall_group.py"]["lf_sha256"] = "0" * 64
    bad = tmp_path / "PATCH_SHAS.json"
    bad.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(F.FreezeError, match="mismatch"):
        F.verify_frozen_v22_patches(patch_shas_path=bad)


def test_missing_copy_refuses(tmp_path):
    manifest = json.loads(F.PATCH_SHAS_PATH.read_text(encoding="utf-8"))
    bad = tmp_path / "PATCH_SHAS.json"
    bad.write_text(json.dumps(manifest), encoding="utf-8")
    empty = tmp_path / "post"
    empty.mkdir()
    with pytest.raises(F.FreezeError, match="copy is missing"):
        F.verify_frozen_v22_patches(patch_shas_path=bad, post_dir=empty)


def test_missing_manifest_refuses(tmp_path):
    with pytest.raises(F.FreezeError, match="PATCH_SHAS.json is missing"):
        F.verify_frozen_v22_patches(patch_shas_path=tmp_path / "absent.json")


def test_runner_verifies_freeze_at_start(tmp_path):
    from inc1 import constants as C
    from inc1 import gates as G
    from inc1 import plan as P
    from inc1 import runner as R

    plan = P.build_plan(seeds=C.seed_block(), include_scenario_binding=False)
    runner = R.IncidentRunner(plan, tmp_path, machine_id="t", platform="linux",
                              gate_info={"registration_lf_sha256":
                                         C.REGISTRATION_LF_SHA},
                              budget=G.BudgetCounter(tmp_path / "b.jsonl"))
    assert runner.wall_v22_freeze["all_match"] is True
