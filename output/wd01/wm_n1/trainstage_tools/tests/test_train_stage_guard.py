"""Pure synthetic policy tests; no registered corpus or runtime is opened."""
from __future__ import annotations

import io
import os
import stat
import sys
from types import SimpleNamespace

import pytest

import train_stage_guard as guard


def _policy(stage, *, inputs):
    return SimpleNamespace(
        stage=stage,
        spec={"stage_inputs": inputs,
              "selection_sha256": None if stage == "metadata-plan" else "a" * 64},
        sandbox=guard._lexnorm("E:/__wm_trainstage_synthetic_output__"),
        roots=(guard._lexnorm("E:/__wm_trainstage_review_source__"),),
        env_prefixes=(guard._lexnorm("C:/Users/jpeng/Miniconda3/envs/rl_driving"),),
    )


def _inputs(stage):
    fixture = "E:/__wm_trainstage_synthetic_input__"
    common = {"pack": fixture + "/pack", "synthetic_test_mode": True,
              "synthetic_root": fixture}
    if stage == "metadata-plan":
        return {**common, "plan_out": "E:/__wm_trainstage_synthetic_output__/plan"}
    if stage == "train-sidecars":
        return {**common,
                "plan_receipt": fixture + "/prior/PLAN_RECEIPT.json",
                "shards_root": fixture + "/shards",
                "selected_shards": ["chunk1/w1/native"],
                "positive_shards": ["chunk1/w1/native"],
                "identity_out": "E:/__wm_trainstage_synthetic_output__/identity",
                "sidecar_out": "E:/__wm_trainstage_synthetic_output__/sidecars"}
    return {**common,
            "plan_receipt": fixture + "/prior/PLAN_RECEIPT.json",
            "identity_dir": fixture + "/prior/identity",
            "sidecar_dir": fixture + "/prior/sidecars",
            "cpu_out": "E:/__wm_trainstage_synthetic_output__/cpu40",
            "num_workers": 0}


def test_metadata_stage_lists_only_two_pack_metadata_files():
    policy = _policy("metadata-plan", inputs=_inputs("metadata-plan"))
    paths = guard._stage_data_paths(policy)
    assert paths == {guard._lexnorm(policy.spec["stage_inputs"]["pack"] + "/" + name)
                     for name in ("manifest.json", "INDEX.json")}
    assert not any("outcomes" in path for path in paths)


def test_sidecar_stage_lists_only_selected_train_shard_files():
    policy = _policy("train-sidecars", inputs=_inputs("train-sidecars"))
    paths = guard._stage_data_paths(policy)
    assert len(paths) == 7  # two metadata, prior plan receipt, four TRAIN files
    assert paths == {guard._lexnorm(path) for path in paths}
    assert guard._lexnorm(policy.spec["stage_inputs"]["plan_receipt"]) in paths
    for name in (*guard.SHARD_COMMON_FILES, guard.SHARD_OUTCOME_FILE):
        assert guard._lexnorm("E:/__wm_trainstage_synthetic_input__/shards/"
                              "chunk1/w1/native/" + name) in paths
    assert not any("/CALIBRATION/" in path.replace("\\", "/") for path in paths)
    assert guard._lexnorm("E:/__wm_trainstage_synthetic_input__/pack/"
                          "outcomes.parquet") not in paths


def test_zero_row_shard_has_only_common_files_and_positive_subset_is_ordered():
    inputs = _inputs("train-sidecars")
    inputs["selected_shards"] = ["chunk1/w1/native", "chunk1/w2/native"]
    inputs["positive_shards"] = ["chunk1/w1/native"]
    policy = _policy("train-sidecars", inputs=inputs)
    paths = guard._stage_data_paths(policy)
    zero_shard = "E:/__wm_trainstage_synthetic_input__/shards/chunk1/w2/native/"
    assert all(guard._lexnorm(zero_shard + name) in paths
               for name in guard.SHARD_COMMON_FILES)
    assert guard._lexnorm(zero_shard + guard.SHARD_OUTCOME_FILE) not in paths
    inputs["positive_shards"] = ["chunk1/w2/native", "chunk1/w1/native"]
    with pytest.raises(guard.GuardInstallError, match="ordered positive"):
        guard._stage_data_paths(policy)


@pytest.mark.parametrize("bad", ["chunk5/w1/native", "chunk4/J4B/native",
                                  "chunk2/INC1/native", "chunk1/../native",
                                  "chunk1/w1/HOLDOUT"])
def test_sidecar_stage_refuses_protected_selection(bad):
    inputs = _inputs("train-sidecars")
    inputs["selected_shards"] = [bad]
    inputs["positive_shards"] = [bad]
    with pytest.raises(guard.GuardInstallError):
        guard._stage_data_paths(_policy("train-sidecars", inputs=inputs))


def test_cpu_stage_has_exact_pack_arrays_and_prior_sidecars():
    policy = _policy("cpu40", inputs=_inputs("cpu40"))
    paths = guard._stage_data_paths(policy)
    assert paths == {guard._lexnorm(path) for path in paths}
    assert guard._lexnorm(policy.spec["stage_inputs"]["plan_receipt"]) in paths
    assert len(paths) == 2 + 1 + len(guard.PACK_ARRAYS) + len(
        guard.IDENTITY_FILES) + len(guard.SIDECAR_FILES)
    assert guard._lexnorm("E:/__wm_trainstage_synthetic_input__/pack/"
                          "outcomes.parquet") not in paths
    assert not any("/shards/" in path.replace("\\", "/") for path in paths)
    inputs = _inputs("cpu40")
    inputs["num_workers"] = 4
    with pytest.raises(guard.GuardInstallError, match="worker count"):
        guard._stage_data_paths(_policy("cpu40", inputs=inputs))


def test_foreign_stage_key_and_wide_fixture_root_fail_closed():
    inputs = _inputs("metadata-plan")
    inputs["selected_shards"] = ["chunk1/w1/native"]
    with pytest.raises(guard.GuardInstallError, match="foreign-stage"):
        guard._stage_data_paths(_policy("metadata-plan", inputs=inputs))
    inputs = _inputs("metadata-plan")
    inputs["synthetic_root"] = "E:/"
    with pytest.raises(guard.GuardInstallError, match="fixture root"):
        guard._stage_data_paths(_policy("metadata-plan", inputs=inputs))


def test_d_drive_is_lexically_refused_without_resolution():
    for path in ("D:/research-dense/secret", "D:/rd-staging/wmn1_feat/"
                 "shards_v3/chunk5/x/outcomes.parquet"):
        assert guard._lexical_deny_reason(path)
    assert guard._protected_word("CALIBRATION")
    assert guard._protected_word("J4B")
    assert guard._protected_word("INC1")


def test_receipt_rename_is_single_new_sandbox_exception(tmp_path):
    sandbox = guard._lexnorm(str(tmp_path))
    policy = SimpleNamespace(stage="train-sidecars", sandbox=sandbox)
    source = tmp_path / "tmpabcd.tmp"
    source.write_text("synthetic receipt", encoding="utf-8")
    target = tmp_path / "TRAIN_ISOLATION_RECEIPT.json"
    assert guard._Policy.check_rename_receipt(policy, source, target)
    assert not guard._Policy.check_rename_receipt(
        SimpleNamespace(stage="cpu40", sandbox=sandbox), source, target)
    target.write_text("already exists", encoding="utf-8")
    assert not guard._Policy.check_rename_receipt(policy, source, target)


def _synthetic_policy_spec():
    fixture = "E:/__wm_trainstage_synthetic_input__"
    sandbox = "E:/tmp/wmn1_train_runs_20261002/__wm_guard_policy_test__"
    pack = fixture + "/pack"
    files = {pack + "/manifest.json": {"role": "data", "sha256": "a" * 64},
             pack + "/INDEX.json": {"role": "data", "sha256": "b" * 64}}
    return {"version": 1, "stage": "metadata-plan", "sandbox": sandbox,
            "sandbox_tmp": sandbox + "/tmp", "roots": ["E:/__wm_review_source__"],
            "e_input_roots": [fixture], "threads": 4,
            "interpreter": {"path": sys.executable, "sha256": "c" * 64},
            "helpers": {}, "files": files,
            "verified_data": {name: rec["sha256"] for name, rec in files.items()},
            "stage_inputs": {"pack": pack, "synthetic_test_mode": True,
                             "synthetic_root": fixture,
                             "plan_out": sandbox + "/plan"}}


def test_protected_config_refused_before_any_hash(monkeypatch):
    if guard._lexnorm(sys.prefix) != guard._lexnorm(
            "C:/Users/jpeng/Miniconda3/envs/rl_driving"):
        pytest.skip("policy constructor needs reviewed runtime")
    spec = _synthetic_policy_spec()
    bad = "E:/__wm_review_source__/output/wd01/wm_n1/CAL_SPLIT_A4.json"
    spec["files"][bad] = {"role": "config", "sha256": "d" * 64}
    monkeypatch.setattr(guard, "_sha256_file",
                        lambda _path: pytest.fail("hashed protected config"))
    with pytest.raises(guard.GuardInstallError, match="CAL split"):
        guard._Policy(spec, guard._lexnorm(spec["sandbox"] + "/child_spec.json"))


def test_exact_pinned_cal_source_allowed_but_other_cal_source_denied():
    if guard._lexnorm(sys.prefix) != guard._lexnorm(
            "C:/Users/jpeng/Miniconda3/envs/rl_driving"):
        pytest.skip("policy constructor needs reviewed runtime")
    spec = _synthetic_policy_spec()
    good = "E:/__wm_review_source__/output/wd01/wm_n1/evalh/cal_split.py"
    spec["files"][good] = {"role": "code", "sha256": "d" * 64}
    policy = guard._Policy(spec,
                           guard._lexnorm(spec["sandbox"] + "/child_spec.json"))
    assert policy.reason(good) is None
    other = "E:/__wm_review_source__/output/wd01/wm_n1/train/cal_split.py"
    assert policy.reason(other)


def test_source_ancestry_is_checked_before_hash(monkeypatch):
    files = {os.path.join(guard._TOOL_DIR, name):
             {"role": "code", "sha256": "a" * 64}
             for name in guard._GUARD_FILES}
    policy = SimpleNamespace(files=files, interpreter="unused",
                             spec={"interpreter": {"sha256": "b" * 64}},
                             helpers={})
    monkeypatch.setattr(guard, "_regular_single_link", lambda *_: None)
    monkeypatch.setattr(guard, "_ancestry_clean", lambda *_:
                        (_ for _ in ()).throw(guard.GuardInstallError("reparse ancestry")))
    monkeypatch.setattr(guard, "_sha256_file",
                        lambda _path: pytest.fail("hashed before ancestry check"))
    with pytest.raises(guard.GuardInstallError, match="reparse ancestry"):
        guard._Policy.verify_pins(policy)


def test_pinned_interpreter_may_be_hardlinked_but_registered_files_may_not(
        monkeypatch):
    sha = "a" * 64
    interpreter = "C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe"
    files = {os.path.join(guard._TOOL_DIR, name):
             {"role": "code", "sha256": sha} for name in guard._GUARD_FILES}
    code = os.path.join(guard._TOOL_DIR, "registered_code.py")
    config = os.path.join(guard._TOOL_DIR, "registered_config.json")
    data = "E:/__wm_trainstage_synthetic_input__/pack/manifest.json"
    files.update({code: {"role": "code", "sha256": sha},
                  config: {"role": "config", "sha256": sha},
                  data: {"role": "data", "sha256": sha}})
    policy = SimpleNamespace(files=files, interpreter=interpreter,
                             spec={"interpreter": {"sha256": sha}}, helpers={})
    linked = {interpreter}
    monkeypatch.setattr(guard, "_lstat_clean", lambda path, _what:
                        SimpleNamespace(st_mode=stat.S_IFREG,
                                        st_nlink=2 if path in linked else 1))
    monkeypatch.setattr(guard, "_ancestry_clean", lambda *_: None)
    monkeypatch.setattr(guard, "_sha256_file", lambda _path: sha)
    guard._Policy.verify_pins(policy)
    for registered in (code, config, data):
        linked.add(registered)
        with pytest.raises(guard.GuardInstallError, match="multiple hard links"):
            guard._Policy.verify_pins(policy)
        linked.remove(registered)


def test_policy_raw_hash_checked_before_json_parse(monkeypatch):
    path = "E:/tmp/wmn1_train_runs_20261002/fresh/child_spec.json"
    monkeypatch.setenv(guard.POLICY_SPEC_ENV, path)
    monkeypatch.setenv(guard.POLICY_SHA_ENV, "a" * 64)
    monkeypatch.setattr(guard, "_regular_single_link", lambda *_: None)
    monkeypatch.setattr(guard, "_ancestry_clean", lambda *_: None)
    monkeypatch.setattr(guard, "open", lambda *_args, **_kwargs: io.BytesIO(b"{bad"),
                        raising=False)
    with pytest.raises(guard.GuardInstallError, match="raw child policy SHA"):
        guard._read_pinned_spec()


def test_repeat_mkdir_of_existing_sandbox_directory_preserves_contents(
        monkeypatch, tmp_path):
    if guard._lexnorm(sys.prefix) != guard._lexnorm(
            "C:/Users/jpeng/Miniconda3/envs/rl_driving") or \
            guard._lexical_drive(str(tmp_path)) != "e:":
        pytest.skip("synthetic policy requires reviewed runtime and E: tmp")
    spec = _synthetic_policy_spec()
    spec["sandbox"] = str(tmp_path)
    spec["sandbox_tmp"] = str(tmp_path / "tmp")
    spec["stage_inputs"]["plan_out"] = str(tmp_path / "plan")
    policy = guard._Policy(spec, guard._lexnorm(str(tmp_path / "child_spec.json")))
    out = tmp_path / "plan"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(guard, "_POLICY", policy)
    guard._hook("os.mkdir", (str(out), 0o777, -1))
    out.mkdir(parents=True, exist_ok=True)
    assert marker.read_text(encoding="utf-8") == "preserved"
    with pytest.raises(PermissionError, match="sandbox writes require"):
        guard._hook("os.mkdir", (str(marker), 0o777, -1))


def test_arrow_public_uri_factory_refuses_without_native_open(monkeypatch):
    class FakeFileSystem:
        @classmethod
        def from_uri(cls, _uri):
            pytest.fail("native URI factory reached")
    class FakePyFileSystem(FakeFileSystem):
        pass
    fake = SimpleNamespace(FileSystem=FakeFileSystem,
                           LocalFileSystem=FakeFileSystem,
                           PyFileSystem=FakePyFileSystem,
                           FileSystemHandler=type("FileSystemHandler", (), {}))
    preloaded = []
    expected_preload = ["pyarrow._dataset", "pyarrow._dataset_parquet",
                        "pyarrow._dataset_orc", "pyarrow._parquet_encryption",
                        "pyarrow._dataset_parquet_encryption"]
    monkeypatch.setattr(guard.importlib, "import_module",
                        lambda name: preloaded.append(name))
    monkeypatch.setattr(guard, "_GUARDED_FS", None)
    def assign(owner, name, wrapper, _label):
        if owner is fake and name == "FileSystem":
            assert preloaded == expected_preload
        setattr(owner, name, wrapper)
    monkeypatch.setattr(guard, "_set_wrapper", assign)
    for name in ("pyarrow._fs", "pyarrow._s3fs", "pyarrow._gcsfs",
                 "pyarrow._azurefs", "pyarrow._hdfs"):
        monkeypatch.setitem(sys.modules, name, None)
    guard._patch_arrow_fs(fake)
    assert preloaded == expected_preload
    with pytest.raises(PermissionError, match="filesystem URI factory refused"):
        fake.FileSystem.from_uri("D:/protected/outcomes.parquet")
    with pytest.raises(PermissionError, match="filesystem URI factory refused"):
        fake.LocalFileSystem.from_uri("D:/protected/outcomes.parquet")


def test_pandas_parquet_uses_only_its_fresh_real_stream(monkeypatch, tmp_path):
    if guard._lexical_drive(str(tmp_path)) != "e:":
        pytest.skip("fresh synthetic sandbox must be on E:")
    target = tmp_path / "two_rows.parquet"
    target_n = guard._norm(str(target))
    policy = SimpleNamespace(sandbox=guard._norm(str(tmp_path)),
                             reason=lambda path, access="read":
                             (None if access == "read" else
                              "sandbox writes require a new destination")
                             if guard._norm(path) == target_n else "not registered")
    monkeypatch.setattr(guard, "_POLICY", policy)
    monkeypatch.setattr(guard, "_SAFE_HANDLES", {})
    wrapped = guard._wrap(lambda _table, where: where,
                          "pyarrow.parquet.write_table", 1, ("where",), "write")
    with open(target, "wb") as handle:
        def write(created=True, alias=False):
            path_or_handle = str(target) if not alias else str(tmp_path / "alias.parquet")
            filesystem = None
            handles = SimpleNamespace(handle=handle,
                                      created_handles=[handle] if created else [])
            return wrapped(None, str(target))
        monkeypatch.setitem(sys.modules, "pandas.io.parquet",
                            SimpleNamespace(PyArrowImpl=SimpleNamespace(write=write)))
        assert write() is handle
        guard._SAFE_HANDLES.clear()
        with pytest.raises(PermissionError, match="sandbox writes require"):
            write(created=False)
        with pytest.raises(PermissionError, match="sandbox writes require"):
            write(alias=True)

        class InventedNamedStream(io.RawIOBase):
            name = str(target)
        with pytest.raises(PermissionError, match="unregistered writable handle"):
            guard._check_arg(InventedNamedStream(), access="write")


def test_identity_numpy_save_accepts_only_pinned_fresh_exclusive_stream(
        monkeypatch, tmp_path):
    if guard._lexical_drive(str(tmp_path)) != "e:":
        pytest.skip("fresh synthetic sandbox must be on E:")
    import numpy as np

    out = tmp_path / "identity"
    out.mkdir()
    sandbox = guard._norm(str(tmp_path))

    def reason(path, access="read"):
        resolved = guard._norm(path)
        if not guard._is_under(resolved, sandbox):
            return "outside fresh sandbox"
        if access == "write" and os.path.lexists(resolved):
            return "sandbox writes require a new destination"
        return None

    policy = SimpleNamespace(stage="train-sidecars", sandbox=sandbox,
                             spec={"stage_inputs": {"identity_out": str(out)}},
                             reason=reason)
    monkeypatch.setattr(guard, "_POLICY", policy)
    monkeypatch.setattr(guard, "_SAFE_HANDLES", {})
    wrapped = guard._wrap(np.save, "numpy.save", 0, ("file",), "write")
    identity = SimpleNamespace(IDS_NAME="identities.npy",
                               POSITIONS_NAME="global_positions.npy")

    def build_identity_sidecar():
        name = identity.IDS_NAME
        with (out / name).open("xb") as stream:
            wrapped(stream, np.asarray([7, 11], dtype=np.int64), allow_pickle=False)
        return stream

    identity.build_identity_sidecar = build_identity_sidecar
    monkeypatch.setitem(sys.modules, "train.identity_sidecar", identity)
    closed = build_identity_sidecar()
    assert np.array_equal(np.load(out / "identities.npy", allow_pickle=False), [7, 11])
    with pytest.raises(PermissionError, match="closed write handle"):
        guard._check_arg(closed, access="write")

    # A preexisting file and a read-only stream never acquire this capability.
    with pytest.raises(FileExistsError):
        build_identity_sidecar()
    with (out / "identities.npy").open("rb") as readonly:
        with pytest.raises(PermissionError, match="unregistered writable handle"):
            wrapped(readonly, np.asarray([1]), allow_pickle=False)

    # A real exclusive stream under an aliased leaf still lacks the expected
    # identity writer frame and exact output filename.
    with (out / "alias.npy").open("xb") as alias:
        with pytest.raises(PermissionError, match="unregistered writable handle"):
            wrapped(alias, np.asarray([1]), allow_pickle=False)

    def build_aliased_identity():
        name = identity.IDS_NAME
        with (out / "other.npy").open("xb") as stream:
            wrapped(stream, np.asarray([1]), allow_pickle=False)

    identity.build_identity_sidecar = build_aliased_identity
    with pytest.raises(PermissionError, match="unregistered writable handle"):
        build_aliased_identity()

    class InventedNamedStream(io.RawIOBase):
        name = str(out / "identities.npy")
    with pytest.raises(PermissionError, match="unregistered writable handle"):
        wrapped(InventedNamedStream(), np.asarray([1]), allow_pickle=False)

    # FileIO can be given an already-open descriptor and an apparent 'x'
    # mode; mode spelling alone must never grant a fresh-file capability.
    raw = io.FileIO(os.open(out / "identities.npy", os.O_RDONLY), mode="xb",
                    closefd=True)
    with io.BufferedWriter(raw) as forged_existing:
        with pytest.raises(PermissionError, match="unsupported file handle"):
            wrapped(forged_existing, np.asarray([1]), allow_pickle=False)
