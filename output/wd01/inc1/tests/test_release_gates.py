"""A1.6 release-gate tests: inc1_hook sha + freeze addendum + PASS verdict."""
from __future__ import annotations

import json

import pytest

from inc1 import gates as G
from inc1 import harness_equivalence as HE
from inc1 import wall_v22_freeze as F


def test_inc1_hook_patch_verifies():
    record = F.verify_inc1_hook_patch()
    assert record["match"] is True
    assert record["actual_lf_sha256"] == record["expected_lf_sha256"]
    assert F.FREEZE_ADDENDUM_PATH.is_file()


def test_tampered_inc1_hook_sha_refuses(tmp_path):
    manifest = json.loads(F.PATCH_SHAS_PATH.read_text(encoding="utf-8"))
    manifest["inc1_hook"]["wall_driver.py"]["patched_lf_sha256"] = "0" * 64
    bad = tmp_path / "PATCH_SHAS.json"
    bad.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(F.FreezeError, match="sha mismatch"):
        F.verify_inc1_hook_patch(patch_shas_path=bad)


def test_missing_addendum_refuses(tmp_path):
    with pytest.raises(F.FreezeError, match="ADDENDUM_INC1"):
        F.verify_inc1_hook_patch(addendum_path=tmp_path / "absent.json")


def test_inc1_hook_patch_crlf_copy_passes(tmp_path):
    """The recorded ``diff_sha256`` is the LF sha, so a CRLF checkout of the
    hook patch (this Windows ``core.autocrlf=true`` tree) must still verify."""
    crlf = tmp_path / "hook_crlf.diff"
    text = F.HOOK_PATCH_PATH.read_text(encoding="utf-8")
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert b"\r\n" in crlf.read_bytes()
    record = F.verify_inc1_hook_patch(hook_patch_path=crlf)
    assert record["match"] is True
    assert record["diff_sha256"] == record["expected_diff_sha256"]


def test_inc1_hook_patch_content_change_refused(tmp_path):
    changed = tmp_path / "hook_changed.diff"
    text = F.HOOK_PATCH_PATH.read_text(encoding="utf-8")
    changed.write_text(text + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(F.FreezeError, match="INC1 hook patch sha"):
        F.verify_inc1_hook_patch(hook_patch_path=changed)


def test_release_gates_pass_with_verdict(tmp_path):
    verdict = HE.write_verdict(tmp_path, {
        "schema": HE.SCHEMA, "verdict": "PASS"})
    record = G.check_inc1_release_gates(verdict_path=verdict)
    assert record["harness_equivalence_pass"] is True
    assert record["inc1_hook"]["match"] is True


def test_release_gates_refuse_without_pass_verdict(tmp_path):
    fail = HE.write_verdict(tmp_path, {
        "schema": HE.SCHEMA, "verdict": "FAIL"})
    with pytest.raises(G.GateError, match="not PASS"):
        G.check_inc1_release_gates(verdict_path=fail)
    with pytest.raises(G.GateError, match="not PASS"):
        G.check_inc1_release_gates(verdict_path=tmp_path / "absent.json")
