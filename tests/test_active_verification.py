import hashlib
import json

from traffic_shepherd.verification import active_import_errors, documentation_errors, verify_archive


def test_detects_archived_imports_including_dynamic(tmp_path):
    package = tmp_path / "traffic_shepherd"
    package.mkdir()
    (package / "bad.py").write_text("from env import bad\nimport torch\n__import__('idm.old')\n")
    assert len(active_import_errors(tmp_path)) == 3
    (package / "bad.py").write_text("from .config import load_config\n")
    assert active_import_errors(tmp_path) == []


def test_documentation_relative_and_archive_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("[index](docs/INDEX.md) [web](https://example.com)")
    (tmp_path / "AGENTS.md").write_text("[missing](no-file.md)")
    (tmp_path / "docs/INDEX.md").write_text("[readme](../README.md)")
    assert len(documentation_errors(tmp_path)) == 1


def test_archive_audit_catches_same_size_corruption(tmp_path):
    archive = tmp_path / "archive/pre_circles"
    archive.mkdir(parents=True)
    artifact = archive / "saved"
    artifact.write_bytes(b"old")
    data = {"status": "relocated_with_explicit_frozen_reader_aliases", "file_count": 1, "files": [{"original_path": "saved", "current_path":
            "archive/pre_circles/saved", "bytes": 3, "sha256": hashlib.sha256(b"old").hexdigest()}]}
    (archive / "manifest.json").write_text(json.dumps(data))
    assert verify_archive(all_hashes=True, root=tmp_path)["errors"] == []
    artifact.write_bytes(b"new")
    assert len(verify_archive(all_hashes=True, root=tmp_path)["errors"]) == 1


def test_incomplete_or_empty_archive_cannot_pass(tmp_path):
    archive = tmp_path / "archive/pre_circles"
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text(json.dumps({"status": "in_progress", "files": [], "file_count": 0}))
    assert len(verify_archive(root=tmp_path)["errors"]) == 2
