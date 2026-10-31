"""Read-only reporting automation: no SSH or simulation in these checks."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "selmappo2_watch", Path(__file__).absolute().parents[1] / "scripts/selmappo2_watch.py")
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def test_publish_preserves_narrative_and_archives_previous_live_report(tmp_path, monkeypatch):
    paper = tmp_path / "SELMAPPO2.md"
    paper.write_text("# Registered narrative\n\nSource receipts remain.\n", encoding="utf-8")
    monkeypatch.setattr(watcher, "PAPER", paper)
    monkeypatch.setattr(watcher.ops, "EVIDENCE", tmp_path)
    collection = tmp_path / "COLLECTION.json"
    collection.write_text("{}\n")

    def publish(path, parsed, provenance, peer):
        path.write_text("# SELMAPPO2 results\n\n" + parsed["text"] +
                        "\n\n![curve](selmappo2_curve_example.svg)\n", encoding="utf-8")

    monkeypatch.setattr(watcher.report, "publish", publish)
    first = watcher.publish(collection, dict(text="first complete update"))
    second = watcher.publish(collection, dict(text="second complete update"))
    text = paper.read_text(encoding="utf-8")
    assert text.startswith("# Registered narrative\n\nSource receipts remain.")
    assert text.count(watcher.MARKER) == 1
    assert "first complete update" not in text and "second complete update" in text
    assert "](selmappo2/selmappo2_curve_example.svg)" in text
    assert "first complete update" in first.read_text(encoding="utf-8")
    assert "second complete update" in second.read_text(encoding="utf-8")
    assert len(list(tmp_path.glob("PAPER_BEFORE_*.md"))) == 2


def test_final_handoff_refuses_intermediate_before_reading_any_files(tmp_path):
    with pytest.raises(ValueError, match="verified complete final169"):
        watcher.final_handoff(tmp_path / "absent.json", dict(state="PARTIAL", checkpoint_sha256=None))
    assert not list(tmp_path.iterdir())


def test_launch_lock_is_exclusive_and_prevents_second_process_before_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher.ops, "EVIDENCE", tmp_path)
    monkeypatch.setattr(watcher, "PAPER", Path(__file__).absolute())
    monkeypatch.setattr(watcher, "PEER", tmp_path)
    lock = tmp_path / "WATCH_ONCE.json"
    monkeypatch.setattr(watcher, "LOCK", lock)
    watcher.ops.exclusive_json(lock, dict(consumed=True))
    monkeypatch.setattr(watcher.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("duplicate watcher was spawned"))
    with pytest.raises(FileExistsError):
        watcher.launch()
    assert watcher.ops.read_json(lock) == dict(consumed=True)


def test_evaluation_requires_explicit_byte_pin_and_frozen_panel(tmp_path, monkeypatch):
    manifest = tmp_path / "PEER_EVALUATION_MANIFEST.json"
    monkeypatch.setattr(watcher, "MANIFEST", manifest)
    assert watcher.peer_evaluation({}) is None
    watcher.ops.exclusive_json(manifest, dict(schema="selmappo2.peer_manifest/1",
                                            path=str(tmp_path / "EVAL.json"), sha256="a" * 64))
    seen = []

    def evaluation(path, sha, parsed):
        seen.append((path, sha, parsed))
        return dict(contract_sha256="0" * 64)

    monkeypatch.setattr(watcher.report, "evaluation", evaluation)
    with pytest.raises(ValueError, match="frozen PANEL_v2"):
        watcher.peer_evaluation(dict(state="COMPLETE"))
    assert seen[0][1] == "a" * 64
