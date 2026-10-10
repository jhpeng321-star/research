from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import fetch_data


class FakeRclone:
    def __init__(self, payload: dict[str, bytes], returncode: int = 0):
        self.payload = payload
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if self.returncode == 0:
            destination = Path(command[3])
            for relative, content in self.payload.items():
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        return type("Result", (), {"returncode": self.returncode, "stdout": "", "stderr": "fake failure"})()


class FetchDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "data_cache"
        self.payload = {"part/a.bin": b"abc", "note.txt": b"dataset"}
        self.entry = {
            "drive_path": "gdrive:research-archive/datasets/example",
            "size_bytes": sum(map(len, self.payload.values())),
            "md5_manifest": {
                name: hashlib.md5(value, usedforsecurity=False).hexdigest()
                for name, value in self.payload.items()
            },
            "local_path": "data_cache/example",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_registry_empty_is_valid_but_get_unknown_fails(self):
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"schema_version": 1, "datasets": {}}), encoding="utf-8")
        self.assertEqual(fetch_data.main(["--registry", str(registry), "list"]), 0)
        self.assertEqual(fetch_data.main(["--registry", str(registry), "get", "example"]), 2)

    def test_get_downloads_verifies_then_status_and_verify_pass(self):
        fake = FakeRclone(self.payload)
        with patch.object(fetch_data.subprocess, "run", side_effect=fake):
            result = fetch_data._fetch("example", self.entry, self.cache, 1024, "fake-rclone")
        self.assertIn("fetched and verified", result)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0][1:3], ["copy", self.entry["drive_path"]])
        target = self.cache / "example"
        self.assertTrue(fetch_data._inspect_tree(target, self.entry["md5_manifest"])[0])
        self.assertIn("verified", fetch_data._format_status("example", self.entry, self.cache))
        self.assertIn("already present", fetch_data._fetch("example", self.entry, self.cache, 1024, "unused"))

    def test_bad_download_is_removed_only_from_cache_staging(self):
        fake = FakeRclone({"note.txt": b"wrong"})
        with patch.object(fetch_data.subprocess, "run", side_effect=fake):
            with self.assertRaises(fetch_data.DataError):
                fetch_data._fetch("example", self.entry, self.cache, 1024, "fake-rclone")
        self.assertEqual(list(self.cache.iterdir()), [])
        outside = self.root / "keep.txt"
        outside.write_text("untouched", encoding="utf-8")
        self.assertEqual(outside.read_text(encoding="utf-8"), "untouched")

    def test_failed_rclone_and_budget_refusal_do_not_change_destination(self):
        with patch.object(fetch_data.subprocess, "run", side_effect=FakeRclone({}, 1)):
            with self.assertRaises(fetch_data.DataError):
                fetch_data._fetch("example", self.entry, self.cache, 1024, "fake-rclone")
        self.assertEqual(list(self.cache.iterdir()), [])
        with patch.object(fetch_data.subprocess, "run", side_effect=FakeRclone(self.payload)):
            with self.assertRaisesRegex(fetch_data.DataError, "budget exceeded"):
                fetch_data._fetch("example", self.entry, self.cache, 1, "fake-rclone")
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_existing_invalid_data_is_preserved_and_links_are_refused(self):
        target = self.cache / "example"
        target.mkdir(parents=True)
        (target / "note.txt").write_bytes(b"keep this")
        with self.assertRaisesRegex(fetch_data.DataError, "preserving it"):
            fetch_data._fetch("example", self.entry, self.cache, 1024, "fake-rclone")
        self.assertEqual((target / "note.txt").read_bytes(), b"keep this")
        self.assertFalse(fetch_data._inspect_tree(target, self.entry["md5_manifest"])[0])

    def test_invalid_registry_paths_fail_closed(self):
        bad = dict(self.entry, local_path="data_cache/../../outside")
        with self.assertRaises(fetch_data.DataError):
            fetch_data._cache_target(self.cache, bad["local_path"])
        bad_manifest = dict(self.entry, md5_manifest={"../escape": "0" * 32})
        registry = self.root / "bad.json"
        registry.write_text(json.dumps({"schema_version": 1, "datasets": {"x": bad_manifest}}), encoding="utf-8")
        with self.assertRaises(fetch_data.DataError):
            fetch_data._load_registry(registry)


if __name__ == "__main__":
    unittest.main()
