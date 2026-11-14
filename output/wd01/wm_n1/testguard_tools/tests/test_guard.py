"""Draft focused guard checks: synthetic fixtures only; no scientific data.

No cleanup fixtures, junction creation, environment escape, or real outcome opens.
Every runnable function must be selected by the coordinator's exact-node manifest.
"""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys

import pytest

import run_guarded as launcher
import wm_testguard as guard


@pytest.fixture
def forbidden_root(tmp_path):
    root = tmp_path / "synthetic_denied"
    root.mkdir()
    (root / "secret.csv").write_text("value\n1\n", encoding="utf-8")
    guard.register_forbidden_root(root)
    return root  # deliberately retained, including the denial registration


def test_startup_attestation_and_real_wrapper_coverage():
    assert guard.is_active()
    attestation = guard.assert_startup()
    assert attestation["pid"] == os.getpid()
    assert attestation["policy_sha"] == os.environ[guard.POLICY_SHA_ENV]
    guard.ensure_required_wrappers()
    assert guard.wrappers_active(require_all=True)
    guard.selfcheck()


def test_audit_checks_path_not_open_mode(tmp_path, forbidden_root):
    path = tmp_path / "allowed.txt"
    path.write_text("synthetic", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "synthetic"
    for call in (lambda: open(forbidden_root / "secret.csv", "rb"),
                 lambda: io.open(forbidden_root / "secret.csv", "r"),
                 lambda: os.open(forbidden_root / "secret.csv", os.O_RDONLY)):
        with pytest.raises(PermissionError, match="wm-testguard"):
            call()


def test_default_deny_mixed_fake_e_outcomes_and_d_lexical():
    for suffix in ("json", "jsonl", "npy", "npz", "parquet", "csv"):
        assert guard.is_protected("E:/__wm_guard_fake_uncreated__/outcome." + suffix)
    assert guard.is_protected("E:/__wm_guard_fake_uncreated__/CAL_SPLIT_A4.json")
    for path in ("D:/__fake__/x", "\\\\?\\D:\\__fake__\\x", "\\??\\D:\\__fake__\\x"):
        assert guard._lexical_deny_reason(path)
        assert guard.is_protected(path)
    # These are strings only: no D: stat, open, glob, or directory access occurs.


def test_protected_precedence_and_directory_metadata(tmp_path):
    spec = copy.deepcopy(guard._POLICY.spec)
    fake_root = "E:/__wm_guard_fake_uncreated__/repo"
    spec["roots"].append(fake_root)
    protected = os.path.join(fake_root, "runs", "approved.py")
    spec["files"][protected] = {"role": "code", "sha256": "0" * 64}
    policy = guard._Policy(spec)
    assert policy.reason(protected)
    assert policy.reason(launcher.TOOL_DIR, "metadata") is None
    assert policy.reason(launcher.TOOL_DIR)
    assert policy.reason(os.path.join(launcher.TOOL_DIR, "fake_outcome.json"))
    assert os.path.basename(launcher.TOOL_DIR) in os.listdir(os.path.dirname(launcher.TOOL_DIR))


def test_pandas_paths_columns_and_dataframe_writes(tmp_path, forbidden_root):
    import pandas as pd
    frame = pd.DataFrame({"word": ["not/a/file"], "value": [2]})
    csv = tmp_path / "frame.csv"
    frame.to_csv(csv, index=False)
    assert pd.read_csv(csv, usecols=["word"])["word"].tolist() == ["not/a/file"]
    assert "word" in frame.to_csv(index=False)
    parquet = tmp_path / "frame.parquet"
    frame.to_parquet(parquet, index=False)
    assert pd.read_parquet(parquet, columns=["value"])["value"].tolist() == [2]
    with pytest.raises(PermissionError, match="wm-testguard"):
        pd.read_csv(forbidden_root / "secret.csv")
    with pytest.raises(PermissionError, match="wm-testguard"):
        frame.to_parquet(forbidden_root / "new.parquet")


def test_numpy_paths_memmap_and_nonpath_values(tmp_path, forbidden_root):
    import numpy as np
    path = tmp_path / "array.npy"
    np.save(path, np.array([1, 2, 3]))
    assert np.load(path).tolist() == [1, 2, 3]
    strings = tmp_path / "strings.npz"
    np.savez(strings, text=np.array(["not/a/path"]))
    with np.load(strings) as saved:
        assert saved["text"].tolist() == ["not/a/path"]
    raw = tmp_path / "memmap.bin"
    raw.write_bytes(bytes(range(8)))
    mapped = np.memmap(raw, dtype="uint8", mode="r", shape=(8,))
    assert isinstance(np.memmap, type) and isinstance(mapped, np.memmap)
    assert mapped.tolist() == list(range(8))
    assert np.fromfile(raw, dtype="uint8").tolist() == list(range(8))
    for call in (lambda: np.load(forbidden_root / "secret.npy"),
                 lambda: np.fromfile(forbidden_root / "secret.bin"),
                 lambda: np.memmap(forbidden_root / "secret.bin", mode="r")):
        with pytest.raises(PermissionError, match="wm-testguard"):
            call()


def test_pyarrow_csv_parquet_feather_dataset(tmp_path, forbidden_root):
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.dataset as dataset
    import pyarrow.feather as feather
    import pyarrow.parquet as parquet
    table = pa.table({"value": [1, 2]})
    csv_path, parquet_path, feather_path = (tmp_path / name for name in ("x.csv", "x.parquet", "x.feather"))
    csv.write_csv(table, str(csv_path))
    assert csv.read_csv(str(csv_path)).column("value").to_pylist() == [1, 2]
    parquet.write_table(table, str(parquet_path))
    assert parquet.read_table(str(parquet_path), columns=["value"]).num_rows == 2
    assert parquet.read_pandas(str(parquet_path)).num_rows == 2
    assert parquet.ParquetFile(str(parquet_path)).read().num_rows == 2
    feather.write_feather(table, str(feather_path))
    assert feather.read_table(str(feather_path)).num_rows == 2
    assert dataset.dataset(str(parquet_path), format="parquet").to_table().num_rows == 2
    calls = (lambda: csv.open_csv(str(forbidden_root / "secret.csv")),
             lambda: parquet.read_table([str(parquet_path), str(forbidden_root / "secret.parquet")]),
             lambda: feather.read_table(str(forbidden_root / "secret.feather")),
             lambda: dataset.dataset(str(forbidden_root / "secret.parquet")))
    for call in calls:
        with pytest.raises(PermissionError, match="wm-testguard"):
            call()


def test_pyarrow_native_filesystem_osfile_mmap_ipc(tmp_path, forbidden_root):
    import pyarrow as pa
    import pyarrow.fs as fs
    import pyarrow.ipc as ipc
    local = fs.LocalFileSystem()
    path = str(tmp_path / "native.bin")
    with local.open_output_stream(path) as stream:
        stream.write(b"abcd")
    with local.open_input_file(path) as stream:
        assert stream.read() == b"abcd"
    with pa.OSFile(path, "rb") as stream:
        assert isinstance(stream, pa.OSFile)
        assert stream.read() == b"abcd"
    with pa.memory_map(path, "r") as stream:
        assert stream.read() == b"abcd"
    table = pa.table({"value": [3]})
    ipc_path = str(tmp_path / "table.arrow")
    with pa.OSFile(ipc_path, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    with pa.OSFile(ipc_path, "rb") as source:
        assert ipc.open_file(source).read_all().num_rows == 1
    for call in (lambda: local.open_input_file(str(forbidden_root / "secret.bin")),
                 lambda: local.open_input_stream(str(forbidden_root / "secret.bin")),
                 lambda: pa.OSFile(str(forbidden_root / "secret.bin"), "r"),
                 lambda: pa.memory_map(str(forbidden_root / "secret.bin")),
                 lambda: ipc.open_file(str(forbidden_root / "secret.arrow"))):
        with pytest.raises(PermissionError, match="wm-testguard"):
            call()


def test_unsupported_handles_filesystems_and_deep_paths(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as parquet
    with pytest.raises(PermissionError, match="unsupported"):
        parquet.read_table(pa.BufferReader(b"synthetic"))
    with pytest.raises(PermissionError, match="filesystem"):
        parquet.read_table(str(tmp_path / "fake.parquet"), filesystem=object())
    nested = str(tmp_path / "fake.csv")
    for _ in range(10):
        nested = [nested]
    with pytest.raises(PermissionError, match="nesting"):
        guard.check_paths(nested)
    guard.check_paths([str(tmp_path / "a"), (str(tmp_path / "b"),)])


def test_child_inherits_attested_guard(tmp_path):
    code = ("import os, wm_testguard as g\n"
            "print(g.assert_startup()['pid'] == os.getpid())\n"
            "try:\n"
            "    open('E:/__wm_guard_fake_uncreated__/outcome.json', 'rb')\n"
            "except PermissionError as exc:\n"
            "    print('DENIED', g.DENY_MARKER in str(exc))\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["True", "DENIED True"]


def test_child_env_removal_escape_and_other_interpreter_refused(tmp_path):
    for key in (guard.REQUIRE_GUARD_ENV, guard.POLICY_SPEC_ENV, guard.POLICY_SHA_ENV, "PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        env = os.environ.copy()
        env.pop(key, None)
        with pytest.raises(PermissionError, match="environment"):
            subprocess.run([sys.executable, "-c", "print('SHOULD_NOT_RUN')"], env=env)
    for flag in ("-S", "-I", "-E", "-uS"):
        with pytest.raises(PermissionError, match="escape"):
            subprocess.run([sys.executable, flag, "-c", "pass"])
    with pytest.raises(PermissionError, match="identical interpreter"):
        subprocess.run([str(tmp_path / "python_fake.exe"), "-c", "pass"])
    with pytest.raises(PermissionError, match="shell"):
        subprocess.run("echo synthetic", shell=True)
    with pytest.raises(PermissionError, match="D:"):
        subprocess.run([sys.executable, "-c", "pass"], cwd="D:/__fake_uncreated__")


def test_missing_startup_attestation_refuses_before_pytest(tmp_path):
    # Keep the real guard installed; simulate a missing bootstrap module reference.
    code = ("import sys\n"
            "assert 'pytest' not in sys.modules\n"
            "sys.modules.pop('wm_testguard')\n"
            "sys.argv = ['probe', '[]']\n"
            "exec(%r)\n" % launcher._CHILD_BOOTSTRAP)
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 87
    assert "bootstrap refused" in proc.stderr
    assert "sitecustomize guard missing" in proc.stderr
    assert "install(" not in launcher._PREFLIGHT_CODE


def test_exact_external_manifest_and_source_sha(tmp_path):
    module = tmp_path / "external" / "test_reviewed.py"
    module.parent.mkdir()
    module.write_text("def test_synthetic():\n    assert True\n", encoding="utf-8")
    node = str(module) + "::test_synthetic"
    payload = {"version": 3, "nodes": [node], "files": {str(module): {"role": "code", "sha256": launcher._sha256(module)}}, "pythonpath": [str(module.parent)]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sha = launcher._sha256(path)
    reviewed = launcher.load_allowlist(path, sha, review_dir=tmp_path)
    assert reviewed["nodes"] == [guard._norm(module) + "::test_synthetic"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        launcher.load_allowlist(path, "0" * 64, review_dir=tmp_path)
    module.write_text("def test_synthetic():\n    assert False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source/config SHA mismatch"):
        launcher.load_allowlist(path, sha, review_dir=tmp_path)
    for broad in (str(module.parent), str(module), str(module) + "::TestClass"):
        with pytest.raises(ValueError):
            launcher.validate_node(broad)


def test_startup_pin_recheck_detects_changed_synthetic_file(tmp_path):
    fixture = tmp_path / "reviewed_registration.md"
    fixture.write_text("# Synthetic registration 1\n", encoding="utf-8")
    spec = copy.deepcopy(guard._POLICY.spec)
    spec["files"][str(fixture)] = {"role": "config", "sha256": launcher._sha256(fixture)}
    policy = guard._Policy(spec)
    policy.verify_pins()
    fixture.write_text("# Synthetic registration 2\n", encoding="utf-8")
    with pytest.raises(guard.GuardInstallError, match="SHA256 mismatch"):
        policy.verify_pins()
    changed = copy.deepcopy(spec)
    changed["nodes"] = [changed["nodes"][0] + "_changed"]
    assert guard._Policy(changed).immutable_fingerprint() != policy.immutable_fingerprint()


def test_arbitrary_allow_paths_plugins_and_data_roles_refused(tmp_path):
    with pytest.raises(SystemExit):
        launcher.main(["--allow-path", "E:/__fake__/outcome.parquet"])
    for option in ("-p", "--collect-only", "--override-ini", "-x", "tests"):
        with pytest.raises(ValueError, match="refused pytest argument"):
            launcher.validate_pytest_extras([option])
    source = tmp_path / "test_manifest.py"
    source.write_text("def test_x(): pass\n", encoding="utf-8")
    payload = {"version": 3, "nodes": [str(source) + "::test_x"], "files": {
        str(source): {"role": "code", "sha256": launcher._sha256(source)},
        "E:/__wm_guard_fake_uncreated__/outcome.parquet": {"role": "config", "sha256": "0" * 64}}}
    manifest = tmp_path / "bad_review.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="code/config"):
        launcher.load_allowlist(manifest, launcher._sha256(manifest), review_dir=tmp_path)


def test_fresh_sandbox_and_no_destructive_cleanup(tmp_path):
    first = launcher.create_sandbox(tmp_path, "synthetic_a", [])
    second = launcher.create_sandbox(tmp_path, "synthetic_b", [])
    assert first != second
    assert not os.path.exists(launcher.basetemp_path(first))
    ini = launcher.write_pytest_ini(first)
    with pytest.raises(FileExistsError):
        launcher.write_pytest_ini(first)
    for call in (lambda: os.remove(ini), lambda: os.rmdir(second),
                 lambda: os.rename(ini, os.path.join(first, "moved.ini"))):
        with pytest.raises(PermissionError, match="disabled"):
            call()


def test_new_sandbox_aliases_preserve_protection(tmp_path, forbidden_root):
    source = tmp_path / "synthetic_source.txt"
    source.write_text("synthetic alias", encoding="utf-8")
    hardlink = tmp_path / "hardlink.txt"
    os.link(source, hardlink)
    assert hardlink.read_text(encoding="utf-8") == "synthetic alias"
    with pytest.raises(PermissionError, match="new"):
        os.link(source, hardlink)
    with pytest.raises(PermissionError, match="wm-testguard"):
        os.link(forbidden_root / "secret.csv", tmp_path / "denied_link.csv")
    with pytest.raises(PermissionError, match="wm-testguard"):
        os.symlink(forbidden_root / "secret.csv", tmp_path / "denied_symlink.csv")
    with pytest.raises(PermissionError, match="wm-testguard"):
        os.link(source, "E:/__wm_guard_fake_uncreated__/outside_link.txt")
    relative_link = tmp_path / "relative_link.txt"
    try:
        os.symlink(source.name, relative_link)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("positive symlink requires Windows privilege; hardlink checks completed")
        raise
    assert relative_link.read_text(encoding="utf-8") == "synthetic alias"


def test_tempfile_cache_confined_and_unlink_still_denied(tmp_path, monkeypatch):
    import tempfile
    expected = guard._POLICY.sandbox_tmp
    assert tempfile.tempdir == expected

    def discovery_must_not_run():
        raise AssertionError("tempfile must use the validated startup cache")

    monkeypatch.setattr(tempfile, "_get_default_tempdir", discovery_must_not_run)
    assert tempfile.gettempdir() == expected
    assert tempfile.gettempdirb() == os.fsencode(expected)
    assert guard._is_under(guard._norm(tempfile.gettempdir()), guard._POLICY.sandbox)
    retained = tmp_path / "retained_after_unlink_refusal.txt"
    retained.write_text("synthetic", encoding="utf-8")
    with pytest.raises(PermissionError, match="wm-testguard"):
        os.unlink(retained)
    assert retained.read_text(encoding="utf-8") == "synthetic"
