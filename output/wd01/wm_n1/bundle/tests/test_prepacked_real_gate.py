"""Queue ``convert+verify`` on an already-extracted (prepacked) v3 corpus.

GPU boxes arrive with ``$WORK/upload`` and ``$WORK/pack`` symlinked to the
extracted ``pack_v3`` tree.  The queue must verify the real ``$WORK/pack`` in
place (opaque-byte SHA256SUMS + INDEX/manifest K_max/format metadata) and must
NOT call ``corpus.pack.convert`` / overwrite the frozen symlink target.  The
timing gate must still measure the explicit real ``$WORK/pack``.

CPU only, ``WD_FORBID_D=1``; a tiny real v3 pack is built with
``bundle.e2e.write_synthetic_pack`` + corpus metadata (no raw physics, no
outcome interpretation).

Test-only compatibility (this module only): an autouse fixture swaps
``queue._save_state`` for a rename-free twin with the identical
serialization contract (the strict guarded sandbox denies ``os.replace``)
and preserves the initial seeded state bytes in an exclusive fresh backup
before the first in-test overwrite.  The timing test additionally swaps
``queue.timing_mod.write_timing_gate`` for the same kind of rename-free
twin (identical indent-1 UTF-8 body, raw-bytes sha256 return, exclusive
fresh-target write).  Every other queue behaviour (``run_stage``,
verification, guards, validation, state loading, refusals) and the
timing gate evaluation itself stay the untouched production code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from corpus.pack import TRAIN_ARRAYS
from bundle import e2e
from bundle import queue


# ------------------------------------------------------------ tiny pack -----
def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_valid_pack(root, *, k=192, fmt="v3") -> Path:
    """A minimal real v3 pack that ``corpus.verify.verify_pack`` accepts."""
    root = Path(root)
    e2e.write_synthetic_pack(root, n_runs=1, k=k, n_ticks=1, seed=0)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["format"] = fmt
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")

    arrays, files = {}, {}
    for name in TRAIN_ARRAYS:
        rel = f"{name}.npy"
        a = np.load(root / rel, mmap_mode="r")
        arrays[name] = {"file": rel, "dtype": str(a.dtype),
                        "shape": list(a.shape),
                        "n_bytes": int((root / rel).stat().st_size)}
        del a
        files[rel] = _sha(root / rel)
    files["outcomes.parquet"] = _sha(root / "outcomes.parquet")
    files["manifest.json"] = _sha(root / "manifest.json")
    index = {
        "schema": "wm_n1_corpus_pack/2", "train_schema": "wm_n1_pack/1",
        "k_max": int(k), "format": fmt, "arrays": arrays, "files": files,
        "n_ticks": 1, "n_decision": 1,
        "runs": [{
            "key": "chunk4/w1__base", "chunk": "chunk4", "batch": "chunk4",
            "world": "w1", "arm": "base", "split": "TRAIN", "t0_s": 540.0,
            "n_ticks": 1, "n_decision": 1, "tick_start": 0, "tick_end": 1,
            "dec_start": 0, "dec_end": 1, "policy": "damping"}],
    }
    (root / "INDEX.json").write_text(json.dumps(index, indent=1),
                                     encoding="utf-8")
    files["INDEX.json"] = _sha(root / "INDEX.json")
    (root / "SHA256SUMS").write_text(
        "\n".join(f"{files[rel]}  {rel}" for rel in sorted(files)) + "\n",
        encoding="utf-8")
    return root


def _tree_hashes(root) -> dict:
    return {p.name: _sha(p) for p in Path(root).iterdir() if p.is_file()}


def _seed_state(work_dir, stages):
    state = {"schema": "wm_n1_bundle_state/1", "completed": {}}
    for s in stages:
        state["completed"][s] = {"stage": s, "ok": True, "completed_utc": "seed"}
    (Path(work_dir) / "state.json").write_text(json.dumps(state),
                                               encoding="utf-8")


# ------------------------- test-only rename-free state saver (sandbox gate) --
_STATE_BACKUP_SUFFIX = ".seed-backup.json"


@pytest.fixture(autouse=True)
def _no_rename_save_state(monkeypatch):
    """Test-only finalizer replacing ONLY ``queue._save_state``.

    The strict guarded sandbox denies ``os.replace`` (rename), which the real
    ``queue._save_state`` tmp+rename write performs on every successful
    stage.  The twin keeps the exact production contract --
    ``updated_utc = queue._utc()``, the identical
    ``json.dumps(state, indent=1).encode("utf-8")`` body and a ``Path``
    return -- but writes the body straight to ``state.json`` (no tmp file,
    no rename, no removal/unlink, nothing cleaned up).  Before the first
    in-fixture overwrite of an existing state file its current bytes are
    copied ONCE to an EXCLUSIVE fresh backup
    (``state.json.seed-backup.json``, never rewritten afterwards), so the
    initial ``_seed_state`` bytes are never lost.  Nothing else is patched:
    ``run_stage``, prepacked verification, convert guards, pack/source/
    schema/checksum validation, state-load behaviour and refusal assertions
    all remain production code.
    """

    def _save_state_no_rename(work_dir, state) -> Path:
        path = queue._state_path(work_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        backup = path.with_name(path.name + _STATE_BACKUP_SUFFIX)
        if path.is_file():
            try:
                with backup.open("xb") as fh:
                    fh.write(path.read_bytes())
            except FileExistsError:
                pass
        state["updated_utc"] = queue._utc()
        body = json.dumps(state, indent=1).encode("utf-8")
        path.write_bytes(body)
        return path

    monkeypatch.setattr(queue, "_save_state", _save_state_no_rename)
    yield


# ------------------------------------------- prepacked verify in place -------
def test_prepacked_v3_verified_in_place_no_convert(tmp_path, monkeypatch):
    pack = _make_valid_pack(tmp_path / "pack")
    # upload holds the same extracted corpus (GPU box symlinks to this data)
    upload = tmp_path / "upload"
    import shutil
    shutil.copytree(pack, upload)

    def boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("convert must NOT run for a prepacked upload")

    monkeypatch.setattr(queue.corpus_pack, "convert", boom)
    before = _tree_hashes(pack)
    _seed_state(tmp_path, ["verify-upload"])
    seeded_state_bytes = (tmp_path / "state.json").read_bytes()
    rec = queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")

    assert rec["ok"] and rec["mode"] == "prepacked-verified"
    assert rec["pack_source"] == "prepacked"
    assert rec["convert"] is None
    assert rec["pack_dir"] == str(tmp_path / "pack")
    assert rec["verify"]["ok"] is True and rec["verify"]["k_max"] == 192
    assert rec["verify"]["format"] == "v3"
    assert _tree_hashes(pack) == before          # frozen pack bytes untouched
    # state contract: persisted completion via the rename-free twin
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed"]["convert+verify"]["ok"] is True
    assert state["completed"]["convert+verify"]["mode"] == "prepacked-verified"
    assert state["completed"]["verify-upload"]["completed_utc"] == "seed"
    assert state["last_stage"] == "convert+verify"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                        state["updated_utc"])
    # on-disk body is exactly the production serialization of the state
    assert state_path.read_bytes() == json.dumps(state, indent=1).encode("utf-8")
    # the patched module attr keeps the production Path return contract
    saved = queue._save_state(tmp_path, state)
    assert isinstance(saved, Path) and saved == state_path
    assert state_path.read_bytes() == json.dumps(state, indent=1).encode("utf-8")
    # initial seeded bytes preserved in the exclusive fresh backup
    backup = tmp_path / ("state.json" + _STATE_BACKUP_SUFFIX)
    assert backup.is_file() and backup.read_bytes() == seeded_state_bytes


def test_prepacked_symlink_target_not_overwritten(tmp_path, monkeypatch):
    frozen = _make_valid_pack(tmp_path / "frozen")
    try:
        os.symlink(frozen, tmp_path / "upload", target_is_directory=True)
        os.symlink(frozen, tmp_path / "pack", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")

    def boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("convert must NOT run through a symlink")

    monkeypatch.setattr(queue.corpus_pack, "convert", boom)
    before = _tree_hashes(frozen)
    _seed_state(tmp_path, ["verify-upload"])
    seeded_state_bytes = (tmp_path / "state.json").read_bytes()
    rec = queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")

    assert rec["mode"] == "prepacked-verified"
    assert _tree_hashes(frozen) == before
    # persisted completion + exclusive seeded-byte backup
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["completed"]["convert+verify"]["mode"] == "prepacked-verified"
    assert state["last_stage"] == "convert+verify"
    backup = tmp_path / ("state.json" + _STATE_BACKUP_SUFFIX)
    assert backup.is_file() and backup.read_bytes() == seeded_state_bytes


def test_prepacked_upload_requires_real_pack_target(tmp_path, monkeypatch):
    _make_valid_pack(tmp_path / "upload")
    # $WORK/pack missing/not a pack -> refuse; never convert the frozen upload
    def boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("convert must NOT run")

    monkeypatch.setattr(queue.corpus_pack, "convert", boom)
    _seed_state(tmp_path, ["verify-upload"])
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")


def test_legacy_raw_shard_upload_still_converts(tmp_path, monkeypatch):
    (tmp_path / "upload").mkdir(parents=True, exist_ok=True)
    calls = []

    def convert(in_dir, out_dir, *, workers=4):
        calls.append(("convert", str(in_dir), str(out_dir)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"packed": 1}

    def verify(pack_dir):
        calls.append(("verify", str(pack_dir)))
        return {"ok": True, "k_max": 128}

    monkeypatch.setattr(queue.corpus_pack, "convert", convert)
    monkeypatch.setattr(queue.corpus_verify, "verify_pack", verify)
    _seed_state(tmp_path, ["verify-upload"])
    seeded_state_bytes = (tmp_path / "state.json").read_bytes()
    rec = queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")

    assert rec["pack_source"] == "converted"
    assert rec.get("mode") != "prepacked-verified"
    assert ("convert", str(tmp_path / "upload"), str(tmp_path / "pack")) in calls
    assert ("verify", str(tmp_path / "pack")) in calls
    # persisted completion + exclusive seeded-byte backup
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["completed"]["convert+verify"]["pack_source"] == "converted"
    assert state["last_stage"] == "convert+verify"
    backup = tmp_path / ("state.json" + _STATE_BACKUP_SUFFIX)
    assert backup.is_file() and backup.read_bytes() == seeded_state_bytes


# ------------------------------- fail-closed malformed / target guards -------
def test_malformed_prepacked_upload_fails_closed_no_convert(tmp_path, monkeypatch):
    """An upload with pack markers but unreadable INDEX must FAIL CLOSED.

    The old detector returned False for an unreadable INDEX and fell through to
    ``corpus.pack.convert``; here the frozen symlink target must stay untouched
    and convert must never run.
    """
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "sentinel.bin").write_bytes(b"frozen-bytes")
    try:
        os.symlink(frozen, tmp_path / "pack", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")

    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "INDEX.json").write_text("{ this is not json", encoding="utf-8")
    (upload / "manifest.json").write_text("{}", encoding="utf-8")

    def boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("convert must NOT run on malformed prepacked data")

    monkeypatch.setattr(queue.corpus_pack, "convert", boom)
    _seed_state(tmp_path, ["verify-upload"])
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")
    assert (frozen / "sentinel.bin").read_bytes() == b"frozen-bytes"


def test_prepacked_markers_without_pack_schema_refused(tmp_path, monkeypatch):
    """Valid JSON but a non-pack INDEX schema is malformed prepacked data."""
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "INDEX.json").write_text(
        json.dumps({"schema": "not_a_pack/9"}), encoding="utf-8")
    (upload / "manifest.json").write_text("{}", encoding="utf-8")

    def boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("convert must NOT run")

    monkeypatch.setattr(queue.corpus_pack, "convert", boom)
    _seed_state(tmp_path, ["verify-upload"])
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")


def test_raw_shard_convert_refuses_existing_nonempty_pack(tmp_path, monkeypatch):
    """The raw-shard flow must not convert into an existing non-empty pack."""
    upload = tmp_path / "upload"
    upload.mkdir()
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "sentinel.bin").write_bytes(b"prior-pack")

    def boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("convert must NOT overwrite an existing pack")

    monkeypatch.setattr(queue.corpus_pack, "convert", boom)
    _seed_state(tmp_path, ["verify-upload"])
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")
    assert (pack / "sentinel.bin").read_bytes() == b"prior-pack"


def test_raw_shard_convert_refuses_symlink_pack_target(tmp_path, monkeypatch):
    """A raw-shard convert must not write through a symlink pack target."""
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "sentinel.bin").write_bytes(b"frozen-bytes")
    try:
        os.symlink(frozen, tmp_path / "pack", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    upload = tmp_path / "upload"
    upload.mkdir()

    def boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("convert must NOT write through a symlink")

    monkeypatch.setattr(queue.corpus_pack, "convert", boom)
    _seed_state(tmp_path, ["verify-upload"])
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("convert+verify", work_dir=tmp_path, device="cpu")
    assert (frozen / "sentinel.bin").read_bytes() == b"frozen-bytes"


# --------------------------------------------- timing gate real $WORK/pack ---
def test_timing_gate_measures_explicit_real_pack(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "manifest.json").write_text("{}", encoding="utf-8")
    _seed_state(tmp_path, ["verify-upload", "convert+verify"])
    seeded_state_bytes = (tmp_path / "state.json").read_bytes()
    captured = {}
    gate_dict = {"schema": "wm_n1_timing_gate/1", "passed": True,
                 "pack_source": "real-pack", "measured_model": "real",
                 "extrapolation": {"total_hours": 1.0}}

    def gate(cfg, **kwargs):
        captured.update(kwargs)
        captured["num_workers"] = cfg.num_workers
        return gate_dict

    def write_gate_no_rename(path, gate):
        """Test-only twin of ``timing.write_timing_gate`` (sandbox gate).

        Identical production contract -- ``p.parent.mkdir``, the identical
        ``json.dumps(gate, indent=1).encode("utf-8")`` body and the raw-BYTES
        sha256 return -- but the body is written straight into the fresh
        target via an exclusive ``open("xb")`` (O_CREAT|O_EXCL: no tmp file,
        no ``os.replace``; an existing target refuses with FileExistsError).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(gate, indent=1).encode("utf-8")
        with p.open("xb") as fh:
            fh.write(body)
        return hashlib.sha256(body).hexdigest()

    monkeypatch.setattr(queue.timing_mod, "run_timing_gate", gate)
    monkeypatch.setattr(queue.timing_mod, "write_timing_gate",
                        write_gate_no_rename)
    monkeypatch.setattr(queue, "TINY", False)
    rec = queue.run_stage("timing-gate", work_dir=tmp_path, device="cuda")

    assert captured["pack_dir"] == str(pack)      # explicit real $WORK/pack
    assert captured["device"] == "cuda"
    assert captured["tiny"] is False
    assert captured["num_workers"] is None        # host-scaled
    assert rec["pack_source"] == "real-pack"
    gate_path = tmp_path / queue.TIMING_GATE_FILENAME
    assert gate_path.is_file()
    # on-disk bytes are exactly the production serialization of the gate
    expected_body = json.dumps(gate_dict, indent=1).encode("utf-8")
    assert gate_path.read_bytes() == expected_body
    # receipt sha256 is the raw-BYTES sha256 of exactly those bytes
    assert rec["sha256"] == hashlib.sha256(expected_body).hexdigest()
    assert rec["sha256"] == hashlib.sha256(gate_path.read_bytes()).hexdigest()
    # the exclusive fresh write refuses an already-existing target
    with pytest.raises(FileExistsError):
        queue.timing_mod.write_timing_gate(gate_path, gate_dict)
    assert gate_path.read_bytes() == expected_body    # refusal wrote nothing
    # persisted completion (seeded stages intact) + exclusive seeded backup
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["completed"]["timing-gate"]["ok"] is True
    assert state["completed"]["verify-upload"]["completed_utc"] == "seed"
    assert state["completed"]["convert+verify"]["completed_utc"] == "seed"
    assert state["last_stage"] == "timing-gate"
    backup = tmp_path / ("state.json" + _STATE_BACKUP_SUFFIX)
    assert backup.is_file() and backup.read_bytes() == seeded_state_bytes
