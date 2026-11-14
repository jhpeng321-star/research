"""Tests for upload_driver scheduling/locking/splitting + an end-to-end run
against a fake rclone and fake zstd (no network, no real Drive)."""
import json, os, shutil, subprocess, sys, tempfile, time, unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts")
sys.path.insert(0, _SCRIPTS)

_BASE = tempfile.mkdtemp(prefix="dataops_test_")
os.environ["DATAOPS_WORK"] = os.path.join(_BASE, "work")
os.environ["DATAOPS_TMPDIR"] = os.path.join(_BASE, "tmp")
os.makedirs(os.environ["DATAOPS_WORK"], exist_ok=True)
os.makedirs(os.environ["DATAOPS_TMPDIR"], exist_ok=True)

import upload_driver as u  # noqa: E402


def write(p, data):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(data)


class FakeRcloneMixin:
    def _make_fakes(self, root):
        rclone = os.path.join(root, "rclone.cmd")
        zstd = os.path.join(root, "zstd.cmd")
        py = sys.executable
        write(rclone, '@echo off\r\n"%s" "%s" %%*\r\n' % (py, os.path.join(_HERE, "fake_rclone.py")))
        write(zstd, '@echo off\r\n"%s" "%s" %%*\r\n' % (py, os.path.join(_HERE, "fake_zstd.py")))
        return rclone, zstd


class TestScheduling(unittest.TestCase):
    def test_file_parts_boundaries(self):
        files = [("a", "a", 4), ("b", "b", 4), ("c", "c", 4), ("d", "d", 4)]
        parts = u.file_parts(files, 8)
        self.assertEqual(parts, [(0, 2), (2, 4)])
        out = []
        for s, e in parts:
            out.extend(files[s:e])
        self.assertEqual(out, files)

    def test_file_parts_single_oversize(self):
        files = [("a", "a", 100)]
        self.assertEqual(u.file_parts(files, 8), [(0, 1)])

    def test_space_budget_accounts_across_workers(self):
        b = u.SpaceBudget(margin=10)
        with mock.patch.object(u.shutil, "disk_usage",
                               return_value=type("D", (), {"free": 100})()):
            self.assertTrue(b.try_reserve(50))
            self.assertFalse(b.try_reserve(50))
            self.assertTrue(b.try_reserve(40))
            self.assertFalse(b.try_reserve(1))
            b.release(50)
            self.assertTrue(b.try_reserve(50))

    def test_claim_prevents_double_processing(self):
        uid = "unit_x"
        u.release_claim(uid)
        owner = "%d:test" % os.getpid()
        self.assertTrue(u.claim(uid, owner))
        self.assertFalse(u.claim(uid, owner))
        u.release_claim(uid)
        self.assertTrue(u.claim(uid, owner))
        u.release_claim(uid)

    def test_expand_queue_splits_large_leaf_into_parts(self):
        root = tempfile.mkdtemp(dir=_BASE)
        for i in range(6):
            write(os.path.join(root, "f%02d.bin" % i), "x" * 10)
        items = [{"id": "t", "name": "t", "src": root, "dest": "research-archive/t",
                  "class": "T", "note": ""}]
        units = u.expand_queue(items, excludes=(), split_bytes=25)
        self.assertGreater(len(units), 1)
        self.assertTrue(all("_part" in x for x in units))
        seen = []
        for x in units:
            seen.extend(r for r, _, _ in u.select_files(x))
        self.assertEqual(sorted(seen), sorted(os.listdir(root)))

    def test_expand_queue_recurses_subdirs(self):
        root = tempfile.mkdtemp(dir=_BASE)
        write(os.path.join(root, "sub1", "a.bin"), "a" * 20)
        write(os.path.join(root, "sub2", "b.bin"), "b" * 20)
        items = [{"id": "t", "name": "t", "src": root, "dest": "research-archive/t",
                  "class": "T", "note": ""}]
        units = u.expand_queue(items, excludes=(), split_bytes=10)
        ids = sorted(x["id"] for x in units)
        self.assertEqual(ids, ["t/sub1", "t/sub2"])


class TestEndToEndFakeRclone(FakeRcloneMixin, unittest.TestCase):
    def _build(self, tag, n_items=5, workers=4):
        base = tempfile.mkdtemp(prefix="e2e_%s_" % tag, dir=_BASE)
        work = os.path.join(base, "work")
        tmp = os.path.join(base, "tmp")
        drive = os.path.join(base, "drive")
        for d in (work, tmp, drive):
            os.makedirs(d, exist_ok=True)
        rclone, zstd = self._make_fakes(base)
        srcs = []
        for i in range(n_items):
            s = os.path.join(base, "src%d" % i)
            write(os.path.join(s, "data.bin"), ("%d" % i) * 100)
            srcs.append(s)
        q = {"schema": "dataops_queue/1", "remote": "fake",
             "items": [{"id": "item%d" % i, "name": "item%d" % i, "src": srcs[i],
                        "dest": "research-archive/out/item%d" % i, "class": "T",
                        "note": ""} for i in range(n_items)]}
        qp = os.path.join(base, "queue.json")
        write(qp, json.dumps(q))
        flog = os.path.join(base, "fake.log")
        env = dict(os.environ)
        env.update({"DATAOPS_WORK": work, "DATAOPS_TMPDIR": tmp,
                    "DATAOPS_RCLONE": rclone, "DATAOPS_ZSTD": zstd,
                    "FAKE_DRIVE_ROOT": drive, "FAKE_LOG": flog})
        return base, qp, work, tmp, drive, flog, env

    def test_concurrent_run_all_verified_no_dupes(self):
        base, qp, work, tmp, drive, flog, env = self._build("basic")
        r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "upload_driver.py"),
                            "--queue", qp, "--workers", "4"], env=env, text=True,
                           capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        recs = [f for f in os.listdir(os.path.join(work, "receipts")) if f.endswith(".json")]
        self.assertEqual(len(recs), 5)
        for f in recs:
            d = json.load(open(os.path.join(work, "receipts", f), encoding="utf-8"))
            self.assertEqual(d["check"], "passed")
        # every copyto destination was hit exactly once
        lines = [l for l in open(flog, encoding="utf-8").read().splitlines() if l]
        dests = [l.split("\t")[2] for l in lines if l.startswith("copyto")]
        self.assertEqual(len(dests), len(set(dests)))
        self.assertEqual(len(dests), 5)
        # temp archives cleaned up
        self.assertEqual(os.listdir(tmp), [])
        # a second run re-verifies nothing (all receipts done)
        r2 = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "upload_driver.py"),
                             "--queue", qp, "--workers", "4"], env=env, text=True,
                            capture_output=True)
        self.assertIn("already verified", r2.stderr + r2.stdout)

    def test_budget_defers_and_resumes(self):
        base, qp, work, tmp, drive, flog, env = self._build("budget")
        # Run with a tiny limit to prove partial progress is resumable.
        r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "upload_driver.py"),
                            "--queue", qp, "--workers", "2", "--limit", "2"], env=env,
                           text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        recs = [f for f in os.listdir(os.path.join(work, "receipts")) if f.endswith(".json")]
        self.assertEqual(len(recs), 2)
        r2 = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "upload_driver.py"),
                             "--queue", qp, "--workers", "2"], env=env, text=True,
                            capture_output=True)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        recs = [f for f in os.listdir(os.path.join(work, "receipts")) if f.endswith(".json")]
        self.assertEqual(len(recs), 5)

    def test_as_files_mode(self):
        base = tempfile.mkdtemp(prefix="e2e_files_", dir=_BASE)
        work, tmp, drive = (os.path.join(base, x) for x in ("work", "tmp", "drive"))
        for d in (work, tmp, drive):
            os.makedirs(d, exist_ok=True)
        rclone, zstd = self._make_fakes(base)
        src = os.path.join(base, "tree")
        write(os.path.join(src, "a.bin"), "aaa")
        write(os.path.join(src, "sub", "b.bin"), "bbb")
        q = {"remote": "fake", "items": [{"id": "af", "name": "af", "src": src,
             "dest": "research-archive/af", "class": "T", "note": "", "as_files": True}]}
        qp = os.path.join(base, "queue.json")
        write(qp, json.dumps(q))
        env = dict(os.environ)
        env.update({"DATAOPS_WORK": work, "DATAOPS_TMPDIR": tmp, "DATAOPS_RCLONE": rclone,
                    "DATAOPS_ZSTD": zstd, "FAKE_DRIVE_ROOT": drive,
                    "FAKE_LOG": os.path.join(base, "fake.log")})
        r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "upload_driver.py"),
                            "--queue", qp, "--workers", "2"], env=env, text=True,
                           capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        rec = json.load(open(os.path.join(work, "receipts", "af.json"), encoding="utf-8"))
        self.assertEqual(rec["check"], "passed")
        self.assertIn("rclone check", rec["check_method"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_make_tar_zst_real_zstd_threads(tmp_path):
    """Regression: '-T N' (separate arg) made zstd treat N as a file -> Errno 22."""
    import sys, tarfile
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
    import upload_driver as d
    if not os.path.exists(d.ZSTD):
        import pytest
        pytest.skip("zstd not installed")
    src = tmp_path / "a.txt"
    src.write_bytes(b"x" * 1000)
    out = str(tmp_path / "o.tar.zst")
    size = d.make_tar_zst([("a.txt", str(src), 1000)], out, zstd_threads=2)
    assert size > 0
    import subprocess
    tar = subprocess.run([d.ZSTD, "-dc", out], capture_output=True).stdout
    import io
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        assert tf.getnames() == ["a.txt"]


def test_load_receipts_matches_safe_id(tmp_path, monkeypatch):
    """Regression: ids with '/' never matched safe_id(), so done items re-uploaded."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
    import upload_driver as d
    monkeypatch.setattr(d, "RECEIPTS", str(tmp_path))
    (tmp_path / "x.json").write_text(json.dumps({"id": "output/foo", "check": "passed"}))
    assert d.safe_id("output/foo") in d.load_receipts()
