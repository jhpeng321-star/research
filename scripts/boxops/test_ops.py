#!/usr/bin/env python3
"""Isolated stdlib tests; run with python -B scripts/boxops/test_ops.py."""
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
import tarfile
import tempfile
import tomllib
import types
import unittest
from unittest import mock

sys.dont_write_bytecode = True

# Load just this sibling file, without importing any project packages.
_spec = importlib.util.spec_from_file_location("boxops_under_test", Path(__file__).with_name("ops.py"))
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)

COMMIT = "a" * 40
COLLECT = "traffic_shepherd/c4/collect.py"
SETUP = "scripts/boxops/setup_box.sh"


class IsolatedTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Fail immediately if any tested path attempts network or external work.
        for name in ("urlopen",):
            patcher = mock.patch.object(ops, name, side_effect=AssertionError("network call forbidden"))
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        for name in ("run", "check_output", "Popen"):
            patcher = mock.patch.object(
                ops.subprocess, name, side_effect=AssertionError("subprocess call forbidden")
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch("sys.stdout", new_callable=io.StringIO)
        self.stdout = patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
        return path


class ArchiveTests(IsolatedTestCase):
    def archive(self, extra=(), *, comment=COMMIT):
        archive = self.root / "runtime.tar.gz"
        headers = {} if comment is None else {"comment": comment}
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT, pax_headers=headers) as tar:
            entries = [
                (COLLECT, b"# collector fixture\r\n", 0o644, tarfile.REGTYPE, ""),
                (SETUP, b"#!/bin/sh\r\necho ready\r\n", 0o640, tarfile.REGTYPE, ""),
                ("scripts/helper.sh", b"#!/bin/sh\r\nexit 0\r\n", 0o600, tarfile.REGTYPE, ""),
                ("assets/data.txt", b"keep\r\n", 0o644, tarfile.REGTYPE, ""),
            ]
            for name, payload, mode, kind, link in entries + list(extra):
                member = tarfile.TarInfo(name)
                member.mode, member.type, member.linkname = mode, kind, link
                member.size = len(payload) if kind == tarfile.REGTYPE else 0
                tar.addfile(member, io.BytesIO(payload) if member.isfile() else None)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.write("runtime.tar.gz.sha256", f"{digest}  {archive.name}\n")
        return archive, digest

    def assert_rejected_before_staging(self, archive, message):
        target = self.root / "installed"
        before = set(self.root.iterdir())
        with self.assertRaisesRegex(ValueError, message):
            ops.extract(archive, target)
        self.assertFalse(target.exists())
        self.assertEqual(set(self.root.iterdir()), before)

    def test_verified_pax_archive_normalizes_shells_and_records_identity(self):
        archive, digest = self.archive()
        target = self.root / "installed"
        original_chmod = Path.chmod
        with mock.patch.object(Path, "chmod", autospec=True, side_effect=original_chmod) as chmod:
            ops.extract(archive, target)
        self.assertEqual((target / SETUP).read_bytes(), b"#!/bin/sh\necho ready\n")
        self.assertEqual((target / "scripts/helper.sh").read_bytes(), b"#!/bin/sh\nexit 0\n")
        self.assertEqual((target / COLLECT).read_bytes(), b"# collector fixture\r\n")
        self.assertEqual((target / "assets/data.txt").read_bytes(), b"keep\r\n")
        self.assertEqual((target / "BOXOPS_CODE_ID").read_text(), COMMIT + "\n")
        self.assertEqual((target / ".BOXOPS_ARCHIVE_SHA256").read_text(), digest + "\n")
        for name in ("setup_box.sh", "helper.sh"):
            modes = [call.args[1] for call in chmod.call_args_list if call.args[0].name == name]
            self.assertTrue(any(mode & 0o111 == 0o111 for mode in modes), modes)
            # Windows chmod cannot represent POSIX executable permission bits.
            if os.name != "nt":
                script = target / "scripts" / ("boxops/" + name if name == "setup_box.sh" else name)
                self.assertEqual(script.stat().st_mode & 0o111, 0o111)

    def test_mismatched_hash_rejected_before_writing_target(self):
        archive, _ = self.archive()
        self.write("runtime.tar.gz.sha256", f"{'0' * 64}  {archive.name}\n")
        self.assert_rejected_before_staging(archive, "SHA256 verification failed")

    def test_wrong_sidecar_filename_rejected(self):
        archive, digest = self.archive()
        self.write("runtime.tar.gz.sha256", f"{digest}  other.tar.gz\n")
        self.assert_rejected_before_staging(archive, "SHA256 verification failed")

    def test_missing_or_invalid_pax_commit_rejected(self):
        for comment in (None, "not-a-commit"):
            with self.subTest(comment=comment):
                archive, _ = self.archive(comment=comment)
                self.assert_rejected_before_staging(archive, "commit PAX header missing")

    def test_traversal_absolute_backslash_and_git_paths_rejected(self):
        for name in ("../outside.txt", "nested/../../outside.txt", "/escape.txt",
                     "nested\\escape.txt", ".git/config"):
            with self.subTest(name=name):
                archive, _ = self.archive([(name, b"bad", 0o644, tarfile.REGTYPE, "")])
                self.assert_rejected_before_staging(archive, "unsafe archive member")

    def test_symlink_and_hardlink_rejected_before_any_payload_written(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(kind=kind):
                archive, _ = self.archive([
                    ("assets/link", b"", 0o777, kind, "../outside.txt")
                ])
                self.assert_rejected_before_staging(archive, "unsafe archive member")

    def test_restricted_member_names_rejected_without_materializing_them(self):
        # These are archive metadata only: no restricted fixture path is created or read.
        for token in ops.FORBIDDEN:
            with self.subTest(token=token):
                name = f"assets/{token.upper()}/neutral.txt"
                archive, _ = self.archive([(name, b"bad", 0o644, tarfile.REGTYPE, "")])
                self.assert_rejected_before_staging(archive, "restricted path refused")

    def test_duplicate_members_rejected(self):
        archive, _ = self.archive([(COLLECT, b"replacement", 0o644, tarfile.REGTYPE, "")])
        self.assert_rejected_before_staging(archive, "unsafe archive member")

    def test_normalized_duplicate_members_rejected(self):
        alias = "traffic_shepherd/./c4/collect.py"
        archive, _ = self.archive([(alias, b"replacement", 0o644, tarfile.REGTYPE, "")])
        self.assert_rejected_before_staging(archive, "unsafe archive member")

    def test_existing_destination_preserved_without_matching_receipt(self):
        archive, _ = self.archive()
        sentinel = self.write("installed/local.txt", b"keep this content")
        target = sentinel.parent
        for marker in (None, "b" * 64):
            with self.subTest(marker=marker):
                if marker is not None:
                    self.write("installed/.BOXOPS_ARCHIVE_SHA256", marker + "\n")
                before = {p.name: p.read_bytes() for p in target.iterdir()}
                with self.assertRaisesRegex(ValueError, "destination exists"):
                    ops.extract(archive, target)
                self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, before)
                self.assertFalse((target / "traffic_shepherd").exists())

    def test_same_archive_is_idempotent_and_preserves_existing_content(self):
        archive, _ = self.archive()
        target = self.root / "installed"
        ops.extract(archive, target)
        sentinel = self.write("installed/assets/data.txt", b"local content")
        before = set(self.root.iterdir())
        ops.extract(archive, target)
        self.assertEqual(sentinel.read_bytes(), b"local content")
        self.assertEqual(set(self.root.iterdir()), before)


class BeginTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.write("repo/BOXOPS_CODE_ID", COMMIT + "\n")
        self.config = self.write("repo/baseline.toml", "seed = 7\n")
        self.weights = self.write("repo/weights.bin", b"initial weights")
        self.repo = self.root / "repo"
        self.out = self.root / "run"
        self.out.mkdir()
        self.flags = ["train", "config=baseline.toml", "init_from=weights.bin", "seed=7"]

    def test_resume_with_same_metadata_accepts_and_logs_each_start(self):
        ops.begin(self.out, self.repo, self.flags)
        first = (self.out / "RUN_SPEC.json").read_bytes()
        spec = json.loads(first)
        self.assertEqual(spec["code_commit"], COMMIT)
        self.assertEqual(spec["mode"], "train")
        self.assertEqual(spec["flags"], {"config": "baseline.toml", "init_from": "weights.bin", "seed": "7"})
        self.assertEqual(spec["input_sha256"], {
            "config": hashlib.sha256(self.config.read_bytes()).hexdigest(),
            "init_from": hashlib.sha256(self.weights.read_bytes()).hexdigest(),
        })
        ops.begin(self.out, self.repo, self.flags)
        self.assertEqual((self.out / "RUN_SPEC.json").read_bytes(), first)
        self.assertEqual((self.out / "CODE_ID.txt").read_text().count("start_utc="), 2)

    def assert_resume_refused(self, flags):
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with self.assertRaisesRegex(ValueError, "resume refused"):
            ops.begin(self.out, self.repo, flags)
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, before)

    def test_changed_flags_refuse_resume_without_altering_run(self):
        ops.begin(self.out, self.repo, self.flags)
        self.assert_resume_refused(self.flags[:-1] + ["seed=8"])

    def test_changed_mode_refuses_resume(self):
        ops.begin(self.out, self.repo, self.flags)
        self.assert_resume_refused(["evaluate"] + self.flags[1:])

    def test_changed_code_refuses_resume_without_altering_run(self):
        ops.begin(self.out, self.repo, self.flags)
        self.write("repo/BOXOPS_CODE_ID", "b" * 40 + "\n")
        self.assert_resume_refused(self.flags)

    def test_changed_config_or_weights_refuse_resume(self):
        ops.begin(self.out, self.repo, self.flags)
        for path in (self.config, self.weights):
            with self.subTest(input=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                self.assert_resume_refused(self.flags)
                path.write_bytes(original)

    def test_changed_evaluation_actor_refuses_resume(self):
        self.write("repo/actor.pt", b"initial actor")
        flags = ["eval", "controllers=native,actor.pt"]
        ops.begin(self.out, self.repo, flags)
        self.write("repo/actor.pt", b"different actor")
        self.assert_resume_refused(flags)


class PopulationTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.repo = self.root / "repo"

    def source(self, text, *, env=False):
        return self.write("repo/traffic_shepherd/c4/" + ("env.py" if env else "collect.py"), text)

    def env_source(self, supported=True):
        signature = "self, *, population='mixed'" if supported else "self, seed=1"
        self.source(f"raise RuntimeError('must not execute')\nclass CavSumoEnv:\n"
                    f"    def __init__({signature}):\n        pass\n", env=True)

    def test_collector_ast_detects_argument_without_executing_source(self):
        self.source("raise RuntimeError('must not execute')\np.add_argument('--population', default='mixed')\n")
        self.assertTrue(ops.supports_population(self.repo))
        ops.require_population(self.repo)

    def test_collector_comments_strings_and_unrelated_calls_are_not_support(self):
        self.source("# p.add_argument('--population')\nnote = '--population'\n"
                    "p.add_argument('--seed')\np.other('--population')\n")
        self.assertFalse(ops.supports_population(self.repo))
        with self.assertRaisesRegex(ValueError, "collector lacks --population"):
            ops.require_population(self.repo)

    def test_env_ast_accepts_regular_and_keyword_only_population_arguments(self):
        for signature in ("self, population='mixed'", "self, *, population='mixed'"):
            with self.subTest(signature=signature):
                self.source("\ufeffraise RuntimeError('must not execute')\nclass CavSumoEnv:\n"
                            f"    def __init__({signature}):\n        pass\n", env=True)
                self.assertTrue(ops.supports_population(self.repo, env=True))

    def test_env_ast_ignores_other_classes_methods_and_kwargs(self):
        self.source("class Other:\n    def __init__(self, population): pass\n"
                    "class CavSumoEnv:\n    def __init__(self, **kwargs): pass\n"
                    "    def reset(self, population): pass\n", env=True)
        self.assertFalse(ops.supports_population(self.repo, env=True))

    def render(self, text, population="mixed", required="1"):
        config = self.write("baseline.toml", text)
        dest = self.root / "runtime.toml"
        # Serve the lightweight TOML dependency entirely from stdlib, never repo/src.
        wms = types.ModuleType("wms")
        wms.__path__ = []
        core = types.ModuleType("wms.core")
        core.__path__ = []
        toml = types.ModuleType("wms.core.toml")
        toml.loads = tomllib.loads
        wms.core, core.toml = core, toml
        with mock.patch.dict(sys.modules, {"wms": wms, "wms.core": core, "wms.core.toml": toml}), \
                mock.patch.object(sys, "path", list(sys.path)):
            ops.mappo_config(self.repo, config, dest, population, required)
        self.assertEqual(config.read_text(), text)
        return dest, tomllib.loads(dest.read_text())

    def test_config_population_override_preserves_existing_keys_and_other_sections(self):
        self.env_source()
        text = ('# fixture config\nseed = 7\n[env]\nname = "CavSumoEnv"\n'
                'kwargs = { population = "old", seed = 11, enabled = true, rate = 0.25, '
                'labels = ["a", "b"], nested = { count = 2 } } # inline fixture\n'
                '[trainer]\nsteps = 12\n')
        dest, rendered = self.render(text, population='mixed "fleet"')
        expected = tomllib.loads(text)
        expected["env"]["kwargs"]["population"] = 'mixed "fleet"'
        self.assertEqual(rendered, expected)
        self.assertTrue(dest.read_text().startswith("# fixture config\nseed = 7\n"))
        self.assertIn('[trainer]\nsteps = 12\n', dest.read_text())

    def test_config_inserts_missing_kwargs_or_env_section(self):
        self.env_source()
        for text in ('[env]\nname = "CavSumoEnv"\n[trainer]\nsteps = 12\n',
                     '[env]\nname = "CavSumoEnv"\n', '[trainer]\nsteps = 12\n'):
            with self.subTest(text=text):
                _, rendered = self.render(text)
                expected = tomllib.loads(text)
                expected.setdefault("env", {})["kwargs"] = {"population": "mixed"}
                self.assertEqual(rendered, expected)

    def test_required_population_refuses_unsupported_env_without_writing_config(self):
        self.env_source(False)
        with self.assertRaisesRegex(ValueError, "lacks population kwarg"):
            self.render('[env]\nkwargs = { seed = 7 }\n')
        self.assertFalse((self.root / "runtime.toml").exists())

    def test_legacy_env_opt_out_preserves_kwargs_and_warns(self):
        self.env_source(False)
        text = '[env]\nkwargs = { seed = 7, enabled = true }\n[trainer]\nsteps = 12\n'
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            _, rendered = self.render(text, required="0")
        self.assertEqual(rendered, tomllib.loads(text))
        self.assertIn("legacy environment", stderr.getvalue())

    def test_invalid_required_flag_refused_before_writing_config(self):
        self.env_source()
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            self.render('[env]\nkwargs = {}\n', required="yes")
        self.assertFalse((self.root / "runtime.toml").exists())


class CollectionTests(IsolatedTestCase):
    def documents(self, complete, failed, already=None):
        self.write("run/COLLECT_SUMMARY.json", json.dumps({
            "episodes_complete": complete, "episodes_failed": failed
        }))
        self.write("run/INDEX.json", json.dumps({} if already is None else {"already_complete": already}))
        return self.root / "run"

    def test_already_complete_entries_count_toward_expected_total(self):
        out = self.documents(2, 0, ["episode-a", "episode-b"])
        ops.check_collection(out, 4)
        with self.assertRaisesRegex(ValueError, "expected 2 complete episodes, got 4"):
            ops.check_collection(out, 2)

    def test_missing_already_complete_defaults_to_empty(self):
        ops.check_collection(self.documents(2, 0), 2)

    def test_entire_collection_can_be_already_complete(self):
        ops.check_collection(self.documents(0, 0, ["episode-a", "episode-b"]), 2)

    def test_failed_episodes_refuse_even_when_complete_total_matches(self):
        with self.assertRaisesRegex(ValueError, "expected 2 complete episodes, got 2"):
            ops.check_collection(self.documents(1, 1, ["episode-a"]), 2)

    def test_conversion_excludes_incomplete_ancestors_and_partial_episodes(self):
        for relative, status in (("raw/branches/world/arm", "complete"),
                                 ("raw/branches/world.incomplete-1/arm", "complete"),
                                 ("raw/episodes/partial", "running")):
            self.write(relative + "/episode.npz", b"fixture")
            self.write(relative + "/summary.json", "{}")
            self.write(relative + "/manifest.json", json.dumps({"status": status}))
        with mock.patch.object(ops.subprocess, "run") as run:
            ops.convert(self.root / "raw", self.root / "converted")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--episodes") + 1:cmd.index("--out")],
                         [str(self.root / "raw/branches/world/arm")])
        self.assertTrue(run.call_args.kwargs["check"])


class WatchStopTests(IsolatedTestCase):
    def test_windows_watcher_refuses_before_any_pid_operation(self):
        with mock.patch.object(ops.sys, "platform", "win32"), \
                mock.patch.object(ops.os, "kill") as kill:
            with self.assertRaisesRegex(ValueError, "Linux-only"):
                ops.watch_stop(self.root, os.getpid(), 1)
        kill.assert_not_called()

    def test_invalid_spot_configuration_fails_preflight(self):
        cases = (
            {"SPOT_METADATA_URL": "https://example.invalid/notice", "SPOT_STOP_AT_UTC": ""},
            {"SPOT_METADATA_URL": "", "SPOT_STOP_AT_UTC": "2026-10-15T02:00:00"},
        )
        for env in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                with self.assertRaises(ValueError):
                    ops.check_spot(1)

    def test_local_notice_requests_stop_for_current_pid_without_network(self):
        out = self.root / "run"
        out.mkdir()
        notice = self.write("notice.txt", "local termination notice\n")
        pid = os.getpid()
        # os.kill(pid, 0) uses TerminateProcess on Windows, so never invoke it here.
        with mock.patch.dict(os.environ, {
            "SPOT_NOTICE_FILE": str(notice), "SPOT_METADATA_URL": "", "SPOT_STOP_AT_UTC": ""
        }), mock.patch.object(ops.os, "kill") as kill, \
                mock.patch.object(ops.sys, "platform", "linux"), \
                mock.patch.object(ops.time, "sleep", side_effect=AssertionError("unexpected polling")):
            ops.watch_stop(out, pid, 0.01)
        kill.assert_called_once_with(pid, 0)
        self.urlopen.assert_not_called()
        text = (out / "STOP_REQUESTED").read_text()
        self.assertTrue(text.startswith("spot notice "))
        self.assertTrue(text.endswith("\n"))
        self.assertIsNotNone(ops.datetime.fromisoformat(text.removeprefix("spot notice ").strip()).tzinfo)
        self.assertIn("STOP_REQUESTED:", self.stdout.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
