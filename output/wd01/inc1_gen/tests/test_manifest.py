"""INC1_GENERATORS.json writer + gate tests (A2.6, zero SUMO)."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from inc1_gen import manifest as M


def test_build_manifest_binds_module_shas_and_addendum_lf_sha():
    info = M.build_manifest()
    assert info["schema"] == M.MANIFEST_SCHEMA
    # every package module file is hashed
    files = M.module_files()
    assert set(info["files"]) == set(files)
    assert M.GENERATOR_MODULE_REL in info["files"]
    for rel, path in files.items():
        # text modules are hashed with the LF convention (CRLF-safe)
        expect = M.file_sha256(path)
        assert info["files"][rel] == expect, rel
    # the registration addendum LF sha
    raw = M.ADDENDUM_PATH.read_bytes()
    expect_lf = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert info["registration_addendum_lf_sha256"] == expect_lf
    assert info["registration_addendum"] == \
        "REGISTRATION_WM_N1_INC1_ADDENDUM_2.md"
    # the specific A2.6 gate key points at the shared generator module
    assert info["generator_module"] == M.GENERATOR_MODULE_REL
    assert info["generator_module_sha256"] == \
        info["files"][M.GENERATOR_MODULE_REL]


def test_lf_sha_normalises_crlf(tmp_path):
    crlf = tmp_path / "doc.md"
    crlf.write_bytes(b"a\r\nb\r\n")
    lf = tmp_path / "doc_lf.md"
    lf.write_bytes(b"a\nb\n")
    assert M.lf_sha256(crlf) == M.lf_sha256(lf)
    assert M.raw_sha256(crlf) != M.raw_sha256(lf)


def test_file_sha256_is_crlf_safe_for_text_only(tmp_path):
    """A CRLF checkout of a text module verifies against the LF-recorded sha;
    a binary file keeps its raw sha (LF normalisation would be wrong)."""
    text = tmp_path / "m.py"
    text.write_bytes(b"x = 1\r\ny = 2\r\n")
    crlf_lf = tmp_path / "m_crlf.py"
    crlf_lf.write_bytes(b"x = 1\r\ny = 2\r\n")
    plain = tmp_path / "m_plain.py"
    plain.write_bytes(b"x = 1\ny = 2\n")
    assert M.file_sha256(text) == M.lf_sha256(text)
    assert M.file_sha256(text) == M.file_sha256(plain)
    assert M.raw_sha256(text) != M.raw_sha256(plain)
    blob = tmp_path / "weights.bin"
    blob.write_bytes(b"\x00\x01\r\n\x02")
    assert M.file_sha256(blob) == M.raw_sha256(blob)


def test_manifest_verifies_a_crlf_checkout(tmp_path):
    """The A2.6 manifest gate must pass when the package is checked out with
    CRLF line endings (Windows core.autocrlf=true)."""
    pkg = tmp_path / "inc1_gen"
    pkg.mkdir()
    for path in M.PACKAGE_ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        (pkg / path.name).write_bytes(text.replace("\n", "\r\n").encode())
    out = tmp_path / "INC1_GENERATORS.json"
    M.write_manifest(out, package_root=pkg)
    assert M.verify_manifest(out, package_root=pkg)["ok"]


def test_write_once_then_verify(tmp_path):
    out = tmp_path / "INC1_GENERATORS.json"
    M.write_manifest(out)
    with pytest.raises(FileExistsError):
        M.write_manifest(out)
    verdict = M.verify_manifest(out)
    assert verdict["ok"], verdict["detail"]


def test_verify_detects_a_tampered_module(tmp_path):
    pkg = tmp_path / "inc1_gen"
    pkg.mkdir()
    for path in M.PACKAGE_ROOT.glob("*.py"):
        shutil.copy2(path, pkg / path.name)
    out = tmp_path / "INC1_GENERATORS.json"
    M.write_manifest(out, package_root=pkg)
    assert M.verify_manifest(out, package_root=pkg)["ok"]
    # one flipped byte in the generator module -> hard mismatch
    gen = pkg / "generators.py"
    gen.write_text(gen.read_text(encoding="utf-8") + "\n# tampered\n",
                   encoding="utf-8")
    verdict = M.verify_manifest(out, package_root=pkg)
    assert not verdict["ok"]
    assert "mismatch" in verdict["detail"]


def test_verify_detects_a_missing_module(tmp_path):
    pkg = tmp_path / "inc1_gen"
    pkg.mkdir()
    for path in M.PACKAGE_ROOT.glob("*.py"):
        shutil.copy2(path, pkg / path.name)
    out = tmp_path / "INC1_GENERATORS.json"
    M.write_manifest(out, package_root=pkg)
    # a scan root missing one module (no file is deleted) -> hard mismatch
    partial = tmp_path / "inc1_gen_partial"
    partial.mkdir()
    for path in pkg.glob("*.py"):
        if path.name != "adapters.py":
            shutil.copy2(path, partial / path.name)
    verdict = M.verify_manifest(out, package_root=partial)
    assert not verdict["ok"]
    assert "module set" in verdict["detail"]


def test_build_manifest_refuses_a_missing_addendum(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.build_manifest(addendum_path=tmp_path / "nope.md")


def test_recorded_manifest_matches_the_live_package():
    """The committed INC1_GENERATORS.json gates the launch (A2.6)."""
    if not M.DEFAULT_MANIFEST_PATH.is_file():
        pytest.fail("INC1_GENERATORS.json has not been recorded")
    verdict = M.verify_manifest(M.DEFAULT_MANIFEST_PATH)
    assert verdict["ok"], verdict["detail"]
    recorded = json.loads(M.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert recorded["schema"] == M.MANIFEST_SCHEMA
