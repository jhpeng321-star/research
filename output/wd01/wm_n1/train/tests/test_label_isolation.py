"""TRAIN label isolation tests (synthetic only; no D:, no real pack).

Proves, with instrumented parquet opens:

* building the sidecar selects TRAIN from pack METADATA first and opens
  ONLY the TRAIN shard ``outcomes.parquet`` sources (mixed pack outcomes
  and CALIBRATION shards are never opened);
* a mixed-split corpus pack opened as TRAIN without a validated sidecar
  refuses BEFORE any label read -- including when its INDEX is missing,
  empty or carries incomplete per-run identity (no synthetic/legacy
  mixed-pack fallback) -- and sidecar mode never opens the mixed table or
  any excluded shard;
* a wholly single-split TRAIN pack (real-writer or manifest-only) keeps
  the original mixed-table behavior, because the manifest proves no
  excluded outcomes exist;
* validation is portable: with ``shards_root`` omitted the recorded
  logical ``<chunk>/<world>/<arm>`` shard identity is still derived from
  metadata and reconciled against the receipt (never skipped);
* every ``outcomes.parquet`` file (source shard and sidecar) is guarded
  as a regular, non-symlink file at its exact selected location BEFORE it
  is read or hashed, and sidecar output can never sit inside or symlink
  into the pack/source trees;
* values, dtypes, row-order mapping and the loss-target mapping are
  byte-identical to the authorized mixed read on a TRAIN-only twin pack,
  and receipt-recorded dtypes are enforced on load;
* tampered packs/sidecars/receipts, missing receipts, wrong-split (CAL)
  sidecars and drifted row coverage all fail closed, and a failed receipt
  write preserves its temporary failure receipt;
* unknown split spellings and every chunk-5 spelling variant are refused
  conservatively;
* pickling and DataLoader parity survive sidecar mode.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from model import constants as C
from packsynth import make_runs, write_idea_a_pack, write_pack
from train import label_isolation as LI
from train.dataset import (OUTCOME_SUPERVISION_EXCLUDED_ARMS, CorpusRefusal,
                           MemmapCorpus, PairAwareBatchSampler, collate,
                           epoch_batches, map_outcome_targets)

TAUS = [float(t) for t in C.TAUS]


def _mixed_runs():
    return [
        dict(chunk="chunk4", world="w1", arm="base", policy="damping",
             split="TRAIN", t0_s=540.0, n_ticks=3, n_dec=2),
        dict(chunk="chunk4", world="w1", arm="native", policy="native",
             split="TRAIN", t0_s=540.0, n_ticks=3, n_dec=2),
        dict(chunk="chunk4", world="w2", arm="native", policy="native",
             split="TRAIN", t0_s=600.0, n_ticks=2, n_dec=2),
        dict(chunk="chunk2", world="c1", arm="native", policy="native",
             split="CALIBRATION", t0_s=300.0, n_ticks=2, n_dec=2),
    ]


def _train_runs():
    return _mixed_runs()[:3]


class ParquetSpy(list):
    """Recorded parquet read paths; ``reset`` scopes it to one operation.

    ``pd.read_parquet`` dispatches into ``pyarrow.parquet.read_table``, so
    the pyarrow wrapper stays silent while a pandas read is in flight (one
    logical open = one record).
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_pd = False

    def reset(self) -> "ParquetSpy":
        self.clear()
        return self


@pytest.fixture
def parquet_spy(monkeypatch):
    """Record every parquet read attempted in-process."""
    calls = ParquetSpy()
    orig_pd = pd.read_parquet

    def spy_pd(path, *a, **k):
        calls.append(os.fspath(path))
        calls._in_pd = True
        try:
            return orig_pd(path, *a, **k)
        finally:
            calls._in_pd = False

    monkeypatch.setattr(pd, "read_parquet", spy_pd)
    try:
        import pyarrow.parquet as pq
        orig_pq = pq.read_table

        def spy_pq(path, *a, **k):
            if not calls._in_pd:
                try:
                    calls.append(os.fspath(path))
                except TypeError:
                    calls.append(f"<{type(path).__name__}:"
                                 f"{getattr(path, 'name', '')}>")
            return orig_pq(path, *a, **k)

        monkeypatch.setattr(pq, "read_table", spy_pq)
    except ImportError:                                   # pragma: no cover
        pass
    return calls


#: These tests keep exercising the PRODUCTION receipt writer (see the
#: autouse fixture below for why): its contract is exactly that a failed
#: write preserves the temporary receipt, and its failure path never
#: reaches the guard-denied finalize, so it runs cleanly under the strict
#: no-rename guard.
_PRODUCTION_RECEIPT_WRITER_TESTS = frozenset({
    "test_failed_receipt_write_preserves_temporary_receipt"})


@pytest.fixture(autouse=True)
def no_rename_atomic_json_writers(monkeypatch, request):
    """Test-only: finalize the synthetic fixtures' atomic JSON writers by
    EXCLUSIVE new-target creation instead of ``os.replace``.

    The strict wm-testguard forbids rename/delete/cleanup, and exactly two
    production atomic writers sit on this module's fixture path:

    * ``corpus._util.write_json_atomic`` -- the synthetic shard
      ``IDEA_A_DONE.json`` receipts and the packed ``manifest.json`` /
      ``INDEX.json``;
    * ``train.label_isolation._write_json_atomic`` -- the sidecar
      receipt (train/label_isolation.py line 679 via line 143).

    The test-only replacements keep IDENTICAL payload bytes, call
    signature, ``Path`` return value and never-delete retention, but
    write the JSON body straight into the always-brand-new target with
    exclusive creation (``open(..., "x")``): no tempfile handoff, no
    unlink, no rename/replace, no rmtree anywhere.  Production sources
    and the guard stay untouched; the one behavioral delta -- an
    existing target is refused instead of silently overwritten -- cannot
    fire here because every target is a fresh path under the test's
    private ``tmp_path``.  Tests named in
    ``_PRODUCTION_RECEIPT_WRITER_TESTS`` opt out so they keep pinning
    the production writer itself.
    """
    if request.node.name in _PRODUCTION_RECEIPT_WRITER_TESTS:
        yield
        return
    from corpus import _util as CU
    from train import label_isolation as LI

    def corpus_write(path, obj):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(obj, indent=1, default=CU._json_default)
        with open(p, "x", encoding="utf-8") as fh:
            fh.write(body)
        return p

    def receipt_write(path, obj):
        p = Path(path)            # no parent mkdir: as in production
        body = json.dumps(obj, indent=1, sort_keys=True)
        with open(p, "x", encoding="utf-8") as fh:
            fh.write(body)
        return p

    monkeypatch.setattr(CU, "write_json_atomic", corpus_write)
    monkeypatch.setattr(LI, "_write_json_atomic", receipt_write)
    yield


@pytest.fixture
def mixed_pack(tmp_path, monkeypatch):
    """A REAL-writer mixed pack (3 TRAIN + 1 CAL run) via corpus.pack."""
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    root = tmp_path / "data"
    pack = write_idea_a_pack(root, _mixed_runs(), k=8)
    return dict(pack=Path(pack), shards=root / "shards")


@pytest.fixture
def mixed_env(mixed_pack, parquet_spy, tmp_path):
    """``mixed_pack`` plus its built sidecar; the spy is reset after the
    build so each test only sees its own operation."""
    env = dict(mixed_pack)
    env["sidecar"] = tmp_path / "sidecar"
    env["receipt"] = LI.build_sidecar(env["pack"], env["shards"],
                                      env["sidecar"])
    env["spy"] = parquet_spy.reset()
    return env


def _receipt(sidecar) -> dict:
    return json.loads((Path(sidecar) / LI.RECEIPT_NAME).read_text(
        encoding="utf-8"))


def _zero_row_train_pack(tmp_path):
    """Build a fresh real-writer TRAIN pack with one absent zero-row shard."""
    from corpus import pack as PK

    source = tmp_path / "source"
    write_idea_a_pack(source, _train_runs(), k=8)
    shards = tmp_path / "zero_shards"
    shards.mkdir()
    source_shards = source / "shards"
    zero_rel = Path("chunk4") / "w1" / "native"
    for entry in sorted(source_shards.rglob("*")):
        rel = entry.relative_to(source_shards)
        target = shards / rel
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif rel == zero_rel / "outcomes.parquet":
            continue  # absent from birth, matching the production writer
        elif rel == zero_rel / "DONE.json":
            done = json.loads(entry.read_text(encoding="utf-8"))
            done["n_outcome_rows"] = 0
            done["outputs"].pop("outcomes.parquet")
            done["shard_dir"] = str(shards / zero_rel)
            target.write_text(json.dumps(done), encoding="utf-8")
        elif rel == Path("INDEX.json"):
            index = json.loads(entry.read_text(encoding="utf-8"))
            for rec in index["runs"].values():
                rel_shard = Path(rec["shard_dir"]).relative_to(source_shards)
                rec["shard_dir"] = str(shards / rel_shard)
            target.write_text(json.dumps(index), encoding="utf-8")
        else:
            target.write_bytes(entry.read_bytes())
    pack = tmp_path / "zero_pack"
    PK.pack_shards(shards, pack, workers=1)
    return pack, shards, zero_rel


def test_zero_row_train_source_is_absent_and_receipt_keeps_identity(tmp_path,
                                                                     parquet_spy):
    pack, shards, zero_rel = _zero_row_train_pack(tmp_path)
    sidecar = tmp_path / "zero_sidecar"
    spy = parquet_spy.reset()
    receipt = LI.build_sidecar(pack, shards, sidecar)
    assert len(receipt["runs"]) == 3
    zero = next(run for run in receipt["runs"] if run["shard_rel"] == zero_rel.as_posix())
    assert zero["split"] == "TRAIN" and zero["n_outcome_rows"] == 0
    assert zero["outcomes_sha256"] is None
    assert all("zero_shards/chunk4/w1/native/outcomes.parquet" not in path.replace("\\", "/")
               for path in spy)
    assert not (shards / zero_rel / "outcomes.parquet").exists()
    LI._validate(pack, sidecar, shards_root=shards, rehash_sources=True)
    assert len(LI.load_train_outcomes(pack, sidecar)) == 10


@pytest.mark.parametrize("fault", ["file", "hash_key", "count"])
def test_zero_row_train_done_or_file_drift_refuses_before_parquet(tmp_path,
                                                                   parquet_spy, fault):
    shard_rel = Path("chunk4") / "w1" / "native"
    root = tmp_path / fault
    shard = root / shard_rel
    shard.mkdir(parents=True)
    done = dict(schema="wm_n1_features_index/1", batch="chunk4",
                chunk="chunk4", world="w1", arm="native", split="TRAIN",
                t0_s=540.0, input_sha256="synthetic-input", n_ticks=3,
                n_outcome_rows=0, shard_dir=str(shard), outputs={})
    if fault == "file":
        (shard / "outcomes.parquet").write_bytes(b"synthetic stray bytes")
    elif fault == "hash_key":
        done["outputs"]["outcomes.parquet"] = "0" * 64
    else:
        done["n_outcome_rows"] = 1
    (shard / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    run = dict(key="chunk4/w1__native", batch="chunk4", chunk="chunk4",
               world="w1", arm="native", split="TRAIN", t0_s=540.0,
               input_sha256="synthetic-input", n=0, tick_start=0,
               tick_end=3, shard_rel=shard_rel.as_posix())
    spy = parquet_spy.reset()
    with pytest.raises(LI.LabelIsolationError):
        LI._outcome_source(root, run)
    assert spy == []


def test_positive_train_done_hash_refuses_before_parquet(tmp_path, parquet_spy):
    root = tmp_path / "positive"
    shard = root / "chunk4" / "w1" / "base"
    shard.mkdir(parents=True)
    (shard / "outcomes.parquet").write_bytes(b"synthetic wrong SHA")
    done = dict(schema="wm_n1_features_index/1", batch="chunk4",
                chunk="chunk4", world="w1", arm="base", split="TRAIN",
                t0_s=540.0, input_sha256="synthetic-input", n_ticks=3,
                n_outcome_rows=1, shard_dir=str(shard),
                outputs={"outcomes.parquet": "0" * 64})
    (shard / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    run = dict(key="chunk4/w1__base", batch="chunk4", chunk="chunk4",
               world="w1", arm="base", split="TRAIN", t0_s=540.0,
               input_sha256="synthetic-input", n=1, tick_start=0,
               tick_end=3, shard_rel="chunk4/w1/base")
    spy = parquet_spy.reset()
    with pytest.raises(LI.LabelIsolationError, match="SHA"):
        LI._outcome_source(root, run)
    assert spy == []


def test_zero_row_receipt_rejects_nonnull_outcome_hash(tmp_path):
    pack, shards, zero_rel = _zero_row_train_pack(tmp_path)
    receipt = LI.build_sidecar(pack, shards, tmp_path / "zero_sidecar")
    zero = next(run for run in receipt["runs"] if run["shard_rel"] == zero_rel.as_posix())
    zero["outcomes_sha256"] = "0" * 64
    plan = LI.plan_sidecar(pack, shards)
    with pytest.raises(LI.LabelIsolationError, match="zero-row"):
        LI._reconcile_receipt(receipt, plan)


def _write_receipt(sidecar, receipt) -> None:
    (Path(sidecar) / LI.RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")


def _cal_shard(env) -> Path:
    return env["shards"] / "chunk2" / "c1" / "native" / "outcomes.parquet"


def _mixed_table(env) -> Path:
    return env["pack"] / "outcomes.parquet"


def _sidecar_parquet(env) -> Path:
    return env["sidecar"] / LI.SIDECAR_PARQUET_NAME


def _train_shard_paths(env) -> list:
    return [str(env["shards"] / rel / "outcomes.parquet")
            for rel in ("chunk4/w1/base", "chunk4/w1/native",
                        "chunk4/w2/native")]


# ------------------------------------------------------------- builder ----
def test_build_selects_train_metadata_first_and_never_opens_excluded(
        mixed_pack, parquet_spy, tmp_path):
    spy = parquet_spy.reset()
    LI.build_sidecar(mixed_pack["pack"], mixed_pack["shards"],
                     tmp_path / "sidecar")
    assert sorted(spy) == sorted(_train_shard_paths(mixed_pack))
    receipt = _receipt(tmp_path / "sidecar")
    assert receipt["schema"] == LI.SIDECAR_SCHEMA
    assert receipt["split"] == "TRAIN"
    # chunk2/c1 sorts FIRST, so the CAL rows occupy global ids 0..4 and the
    # sidecar coverage must start above them.
    assert [r["key"] for r in receipt["runs"]] == [
        "chunk4/w1__base", "chunk4/w1__native", "chunk4/w2__native"]
    assert [r["first_global_row"] for r in receipt["runs"]] == [5, 10, 15]
    assert receipt["selection"]["n_rows"] == 15
    assert receipt["selection"]["excluded_splits"] == {"CALIBRATION": 1}


def test_plan_sidecar_is_metadata_only(mixed_pack, parquet_spy):
    spy = parquet_spy.reset()
    plan = LI.plan_sidecar(mixed_pack["pack"], mixed_pack["shards"])
    assert spy == []
    assert plan["selection"]["n_rows"] == 15
    assert [r["first"] for r in plan["train_runs"]] == [5, 10, 15]
    assert [r["shard_rel"] for r in plan["train_runs"]] == [
        "chunk4/w1/base", "chunk4/w1/native", "chunk4/w2/native"]


def test_build_refuses_existing_output_dir(mixed_pack, tmp_path):
    sidecar = tmp_path / "sidecar"
    LI.build_sidecar(mixed_pack["pack"], mixed_pack["shards"], sidecar)
    with pytest.raises(LI.LabelIsolationError, match="never overwrite"):
        LI.build_sidecar(mixed_pack["pack"], mixed_pack["shards"], sidecar)


def test_build_refuses_shard_path_escape(mixed_pack, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()                       # existing root, shard not inside
    with pytest.raises(LI.LabelIsolationError, match="escapes"):
        LI.build_sidecar(mixed_pack["pack"], elsewhere,
                         tmp_path / "sidecar2")
    assert not (tmp_path / "sidecar2").exists()


def test_build_refuses_output_inside_pack_or_shard_tree(mixed_pack,
                                                        parquet_spy):
    env = mixed_pack
    spy = parquet_spy.reset()
    for bad in (env["pack"] / "sidecar", env["shards"] / "sidecar",
                env["shards"] / "chunk4" / "w1" / "sidecar"):
        with pytest.raises(LI.LabelIsolationError, match="inside"):
            LI.build_sidecar(env["pack"], env["shards"], bad)
    assert spy == []


def _symlinks_supported(tmp_path) -> bool:
    probe = tmp_path / "symlink_probe"
    try:
        probe.symlink_to(tmp_path / "symlink_target_missing",
                         target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    # the dangling probe symlink stays inside this test's private tmp_path
    # (pytest owns that tree); this suite never destroys fixture bytes
    return True


def test_build_refuses_symlinked_output_path(mixed_pack, tmp_path):
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks unavailable on this host")
    link = tmp_path / "sidecar_link"
    link.symlink_to(tmp_path / "nowhere", target_is_directory=True)
    with pytest.raises(LI.LabelIsolationError, match="symlink"):
        LI.build_sidecar(mixed_pack["pack"], mixed_pack["shards"], link)


def test_build_refuses_symlinked_shard_outcomes_file(mixed_pack, tmp_path,
                                                     parquet_spy):
    env = mixed_pack
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks unavailable on this host")
    src = env["shards"] / "chunk4" / "w1" / "base" / "outcomes.parquet"
    # a REAL outcomes file outside the shards tree is the symlink target
    outside = tmp_path / "outside_outcomes.parquet"
    outside.write_bytes(src.read_bytes())
    # fresh synthetic mirror of the shards tree in which the selected
    # shard's outcomes file IS a symlink from the start (byte-identical
    # copies elsewhere); the original shard file stays a regular file
    mirror = tmp_path / "shards_symlink_variant"
    for path in sorted(env["shards"].rglob("*")):
        rel = path.relative_to(env["shards"])
        target = mirror / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif rel == Path("chunk4") / "w1" / "base" / "outcomes.parquet":
            target.symlink_to(outside)
        else:
            target.write_bytes(path.read_bytes())
    assert src.is_file() and not src.is_symlink()   # original preserved
    spy = parquet_spy.reset()
    with pytest.raises(LI.LabelIsolationError, match="symlink"):
        LI.build_sidecar(env["pack"], mirror, tmp_path / "sidecar")
    assert spy == []                        # guarded BEFORE read/hash


def _pack_with_extra_run(tmp_path, extra):
    root = tmp_path / "data"
    runs = _train_runs() + [extra]
    return write_idea_a_pack(root, runs, k=8), root / "shards"


@pytest.mark.parametrize("extra", [
    dict(chunk="chunk5", world="w9", arm="native", policy="native",
         split="TEST", t0_s=100.0, n_ticks=1, n_dec=1),
    dict(chunk="chunk4", world="w9", arm="native", policy="native",
         split="HOLDOUT", t0_s=100.0, n_ticks=1, n_dec=1),
])
def test_build_refuses_chunk5_and_holdout_packs(tmp_path, extra,
                                                parquet_spy):
    pack, shards = _pack_with_extra_run(tmp_path, extra)
    spy = parquet_spy.reset()
    with pytest.raises(LI.LabelIsolationError):
        LI.plan_sidecar(pack, shards)
    with pytest.raises(LI.LabelIsolationError):
        LI.build_sidecar(pack, shards, tmp_path / "sidecar")
    assert spy == []


def test_build_refuses_overlapping_or_drifted_row_ranges(tmp_path,
                                                         parquet_spy):
    pack, shards = _pack_with_extra_run(tmp_path, _mixed_runs()[3])
    manifest = json.loads((pack / "manifest.json").read_text("utf-8"))
    # duplicate the first TRAIN item's rows into its sibling's range
    manifest["items"][1]["outcome_rows"] = list(
        manifest["items"][0]["outcome_rows"])
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    spy = parquet_spy.reset()
    with pytest.raises(LI.LabelIsolationError, match="derived global range"):
        LI.build_sidecar(pack, shards, tmp_path / "sidecar")
    assert spy == []


def test_build_refuses_nontrain_split_drift(tmp_path, parquet_spy):
    pack, shards = _pack_with_extra_run(tmp_path, _mixed_runs()[3])
    index = json.loads((pack / "INDEX.json").read_text("utf-8"))
    for rec in index["runs"]:
        if rec["key"] == "chunk2/c1__native":
            rec["split"] = "TRAIN"          # a CAL run disguised as TRAIN
    (pack / "INDEX.json").write_text(json.dumps(index, indent=1),
                                     encoding="utf-8")
    spy = parquet_spy.reset()
    with pytest.raises(LI.LabelIsolationError, match="split"):
        LI.build_sidecar(pack, shards, tmp_path / "sidecar")
    assert spy == []


# ------------------------------------------------ portable plan / identity -
def test_plan_without_shards_root_still_validates_logical_identity(
        mixed_pack, parquet_spy):
    spy = parquet_spy.reset()
    plan = LI.plan_sidecar(mixed_pack["pack"])          # NO shards_root
    assert spy == []                        # pure metadata, but identity held
    assert [r["shard_rel"] for r in plan["train_runs"]] == [
        "chunk4/w1/base", "chunk4/w1/native", "chunk4/w2/native"]


def test_plan_refuses_recorded_shard_identity_mismatch(mixed_pack):
    env = mixed_pack
    index = json.loads((env["pack"] / "INDEX.json").read_text("utf-8"))
    index["runs"][0]["shard_dir"] = str(
        env["shards"] / "chunk4" / "w2" / "base")   # wrong world/arm
    (env["pack"] / "INDEX.json").write_text(json.dumps(index, indent=1),
                                            encoding="utf-8")
    with pytest.raises(LI.LabelIsolationError, match="run identity"):
        LI.plan_sidecar(env["pack"], env["shards"])
    with pytest.raises(LI.LabelIsolationError, match="run identity"):
        LI.plan_sidecar(env["pack"])                # portable path too


def test_plan_refuses_recorded_shard_traversal(mixed_pack):
    env = mixed_pack
    index = json.loads((env["pack"] / "INDEX.json").read_text("utf-8"))
    index["runs"][0]["shard_dir"] = str(
        env["shards"] / "chunk2" / "c1" / "native" / ".." / ".." / ".."
        / "outside")
    (env["pack"] / "INDEX.json").write_text(json.dumps(index, indent=1),
                                            encoding="utf-8")
    with pytest.raises(LI.LabelIsolationError):
        LI.plan_sidecar(env["pack"], env["shards"])
    with pytest.raises(LI.LabelIsolationError):
        LI.plan_sidecar(env["pack"])


def test_load_without_shards_root_reconciles_recorded_shard_rel(
        mixed_env, monkeypatch):
    env = mixed_env
    receipt = _receipt(env["sidecar"])
    receipt["runs"][0]["shard_rel"] = "chunk4/w1/other"
    _write_receipt(env["sidecar"], receipt)
    # sidecar mode: without the env set, production correctly refuses the
    # mixed pack BEFORE sidecar validation (mixed-label refusal), so point
    # the env at the built synthetic sidecar to reach the tamper refusal
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    corpus = MemmapCorpus(env["pack"], split="TRAIN")   # no shards_root
    with pytest.raises(CorpusRefusal, match="shard_rel"):
        corpus[0]
    assert env["spy"] == []                 # zero-label-open retained


def test_load_without_shards_root_reconciles_recorded_shard_dir(
        mixed_env, monkeypatch):
    env = mixed_env
    receipt = _receipt(env["sidecar"])
    receipt["runs"][0]["shard_dir_recorded"] = str(
        env["shards"] / "chunk4" / "w2" / "base")
    _write_receipt(env["sidecar"], receipt)
    # sidecar mode: without the env set, production correctly refuses the
    # mixed pack BEFORE sidecar validation (mixed-label refusal), so point
    # the env at the built synthetic sidecar to reach the tamper refusal
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    corpus = MemmapCorpus(env["pack"], split="TRAIN")   # no shards_root
    with pytest.raises(CorpusRefusal, match="shard_dir_recorded"):
        corpus[0]
    assert env["spy"] == []                 # zero-label-open retained


# ----------------------- portable recorded source identity (lexical) ------
def _derived_shard_run(shard_dir, batch="chunk4", world="w1", arm="base"):
    """A minimal derived INDEX run record (metadata only; no pack, no IO)."""
    return dict(key=f"{batch}/{world}__{arm}", batch=batch, world=world,
                arm=arm, chunk=batch, split="TRAIN", shard_dir=shard_dir,
                n=5)


def test_recorded_windows_source_identity_is_portable_across_hosts():
    # the canonical Windows staging spelling must parse on ANY host OS
    want = "chunk4/w1/base"
    for recorded in (
            "D:/rd-staging/wmn1_feat/shards_v3/chunk4/w1/base",
            "D:\\rd-staging\\wmn1_feat\\shards_v3\\chunk4\\w1\\base",
            "//staging-srv/wmn1_feat/shards_v3/chunk4/w1/base"):
        assert LI._shard_rel_logical(_derived_shard_run(recorded)) == want
    # ... and a POSIX absolute spelling parses identically
    assert LI._shard_rel_logical(_derived_shard_run(
        "/rd-staging/wmn1_feat/shards_v3/chunk4/w1/base")) == want
    # the normalized logical identity MATCHES between the two flavors
    assert LI._shard_rel_logical(_derived_shard_run(
        "D:/rd-staging/wmn1_feat/shards_v3/chunk4/w1/base")) == \
        LI._shard_rel_logical(_derived_shard_run(
            "/rd-staging/wmn1_feat/shards_v3/chunk4/w1/base"))


@pytest.mark.parametrize("bad", [
    "", ".",                       # empty spellings
    "chunk4", "chunk4/w1",         # relative, too short
    "chunk4/w1/base/extra",        # relative, too long
    "chunk4/../w1/base",           # relative traversal
    "D:", "D:chunk4/w1/base",      # drive-relative (drive without root)
    "D:/rd-staging",               # absolute, too short to end in triple
    "D:/rd-staging/wmn1_feat/shards_v3/chunk4/w1/../base",   # traversal
    "D:/rd-staging/wmn1_feat/shards_v3/chunk4/w2/base",      # wrong world
])
def test_malformed_recorded_source_paths_are_refused(bad):
    with pytest.raises(LI.LabelIsolationError):
        LI._shard_rel_logical(_derived_shard_run(bad))


def test_windows_recorded_pack_plans_portably_without_opening_source(
        mixed_pack, parquet_spy):
    env = mixed_pack
    index = json.loads((env["pack"] / "INDEX.json").read_text("utf-8"))
    for rec in index["runs"]:
        batch, tail = rec["key"].split("/", 1)
        rec["shard_dir"] = ("D:/rd-staging/wmn1_feat/shards_v3/"
                            f"{batch}/{tail.replace('__', '/')}")
    (env["pack"] / "INDEX.json").write_text(json.dumps(index, indent=1),
                                            encoding="utf-8")
    manifest = json.loads(
        (env["pack"] / "manifest.json").read_text("utf-8"))
    # the metadata audit still runs BEFORE any label read and holds
    LI.audit_pack_identity(manifest, index)
    spy = parquet_spy.reset()
    plan = LI.plan_sidecar(env["pack"])          # NO shards_root (GPU box)
    assert spy == []                 # the recorded old D:/ source never opens
    assert [r["shard_rel"] for r in plan["train_runs"]] == [
        "chunk4/w1/base", "chunk4/w1/native", "chunk4/w2/native"]


def _foreign_source_prefix() -> str:
    """An absolute recorded prefix of the OTHER path flavor: never a real
    location on this host."""
    return "/rd-staging/wmn1_feat/shards_v3" if os.name == "nt" \
        else "D:/rd-staging/wmn1_feat/shards_v3"


def test_foreign_flavor_recorded_source_uses_local_shards_mirror_only(
        mixed_pack, parquet_spy, tmp_path):
    env = mixed_pack
    index = json.loads((env["pack"] / "INDEX.json").read_text("utf-8"))
    prefix = _foreign_source_prefix()
    for rec in index["runs"]:
        batch, tail = rec["key"].split("/", 1)
        rec["shard_dir"] = f"{prefix}/{batch}/{tail.replace('__', '/')}"
    (env["pack"] / "INDEX.json").write_text(json.dumps(index, indent=1),
                                            encoding="utf-8")
    spy = parquet_spy.reset()
    plan = LI.plan_sidecar(env["pack"], env["shards"])   # LOCAL mirror root
    assert spy == []
    assert [r["shard_rel"] for r in plan["train_runs"]] == [
        "chunk4/w1/base", "chunk4/w1/native", "chunk4/w2/native"]
    # a build reads ONLY the local mirror, never the recorded old source
    sidecar = tmp_path / "sidecar_foreign"
    receipt = LI.build_sidecar(env["pack"], env["shards"], sidecar)
    assert receipt["runs"][0]["shard_dir_recorded"].startswith(prefix)
    assert spy and all(str(env["shards"]) in p for p in spy)


# ------------------------------------------------ conservative spellings --
def test_plan_refuses_unknown_split_spelling(tmp_path):
    pack, shards = _pack_with_extra_run(tmp_path, _mixed_runs()[3])
    manifest = json.loads((pack / "manifest.json").read_text("utf-8"))
    manifest["items"][0]["split"] = "VALIDATION"     # unknown spelling
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    with pytest.raises(LI.LabelIsolationError,
                       match="unknown split spelling"):
        LI.plan_sidecar(pack, shards)


@pytest.mark.parametrize("bad_chunk", ["chunk-5", "chunk_5", "Chunk 5",
                                       "chunkX"])
def test_plan_refuses_chunk5_spelling_variants(tmp_path, bad_chunk,
                                               parquet_spy):
    pack, shards = _pack_with_extra_run(tmp_path, dict(
        chunk=bad_chunk, world="w9", arm="native", policy="native",
        split="TRAIN", t0_s=100.0, n_ticks=1, n_dec=1))
    spy = parquet_spy.reset()
    with pytest.raises(LI.LabelIsolationError):
        LI.plan_sidecar(pack, shards)
    assert spy == []


# ------------------------------------------------------ dataset refusal ---
def test_mixed_pack_train_without_sidecar_refuses_before_any_read(mixed_env):
    env = mixed_env
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    assert len(corpus) == 3
    with pytest.raises(CorpusRefusal, match="mixed-split corpus pack"):
        corpus[0]
    with pytest.raises(CorpusRefusal):
        corpus.outcome_target(0)
    assert env["spy"] == []


def test_missing_sidecar_dir_refuses_before_any_read(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV,
                       str(env["sidecar"].parent / "nope"))
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal):
        corpus[0]
    assert env["spy"] == []


def test_sidecar_mode_never_opens_mixed_or_cal(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    corpus = MemmapCorpus(env["pack"], split="TRAIN", pair_sampling=True)
    for i in range(len(corpus)):
        corpus[i]
        corpus.outcome_target(i)
    assert env["spy"] == [str(_sidecar_parquet(env))]
    assert str(_mixed_table(env)) not in env["spy"]
    assert str(_cal_shard(env)) not in env["spy"]


def test_synthetic_cal_keeps_authorized_mixed_read(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    corpus = MemmapCorpus(env["pack"], split="CALIBRATION")
    assert len(corpus) == 1 and corpus.worlds() == ["c1"]
    item = corpus[0]
    assert item["split"] == "CALIBRATION"
    assert env["spy"] == [str(_mixed_table(env))]


# ------------------------------------- mixed pack identity is mandatory ---
def _pack_variant_without_index(pack: Path, dst: Path) -> Path:
    """A fresh synthetic variant of ``pack`` that NEVER contains INDEX.json.

    Every file (packed table, manifest, shard tree) is copied
    byte-identically into ``dst`` EXCEPT ``INDEX.json``, which is simply
    never created there; the source pack and all of its files stay
    preserved untouched.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(pack.rglob("*")):
        rel = path.relative_to(pack)
        if path.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        elif rel != Path("INDEX.json"):
            (dst / rel).write_bytes(path.read_bytes())
    return dst


def test_mixed_pack_missing_index_fails_closed(mixed_pack, parquet_spy):
    env = mixed_pack
    pack = _pack_variant_without_index(
        env["pack"], env["pack"].parent / "pack_without_index")
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(pack, split="TRAIN")
    with pytest.raises(CorpusRefusal, match="mixed-split corpus pack"):
        corpus[0]
    with pytest.raises(CorpusRefusal):
        corpus.outcome_target(0)
    assert spy == []


@pytest.mark.parametrize("mutate", ["empty_runs", "missing_shard_dir",
                                    "blank_shard_dir", "missing_rows"])
def test_mixed_pack_incomplete_identity_fails_closed(mixed_pack, mutate,
                                                     monkeypatch,
                                                     parquet_spy):
    env = mixed_pack
    index = json.loads((env["pack"] / "INDEX.json").read_text("utf-8"))
    if mutate == "empty_runs":
        index["runs"] = []
    elif mutate == "missing_shard_dir":
        del index["runs"][0]["shard_dir"]
    elif mutate == "blank_shard_dir":
        index["runs"][0]["shard_dir"] = ""
    else:
        del index["runs"][0]["n_outcome_rows"]
    (env["pack"] / "INDEX.json").write_text(json.dumps(index, indent=1),
                                            encoding="utf-8")
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="mixed-split corpus pack"):
        corpus[0]
    with pytest.raises(CorpusRefusal):
        corpus.outcome_target(0)
    assert spy == []


def test_dataset_refuses_unknown_item_split_spelling(tmp_path):
    pack = write_idea_a_pack(tmp_path / "data", _train_runs(), k=8)
    manifest = json.loads((pack / "manifest.json").read_text("utf-8"))
    manifest["items"][0]["split"] = "VALIDATION"     # unknown spelling
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    with pytest.raises(CorpusRefusal, match="unknown split spelling"):
        MemmapCorpus(pack, split="TRAIN")


# ----------------------------------------------------- values and dtypes --
def _outcome_tensors(corpus):
    out = {}
    for i in range(len(corpus)):
        it = corpus[i]
        out[(it["world"], it["arm"])] = (
            it["outcome_quantiles"], it["outcome_mask"], it["collision"])
    return out


def test_sidecar_values_dtypes_and_loss_mapping_match_authorized_read(
        mixed_env, tmp_path, monkeypatch):
    env = mixed_env
    twin = write_idea_a_pack(tmp_path / "twin", _train_runs(), k=8)
    env["spy"].reset()
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)

    twin_corpus = MemmapCorpus(twin, split="TRAIN")
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    side_corpus = MemmapCorpus(env["pack"], split="TRAIN")
    side_tensors = _outcome_tensors(side_corpus)
    twin_tensors = _outcome_tensors(twin_corpus)
    assert set(side_tensors) == set(twin_tensors)
    for ident, tensors in side_tensors.items():
        for got, want in zip(tensors, twin_tensors[ident]):
            assert torch.equal(got, want), ident

    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    twin_rows = pd.read_parquet(twin / "outcomes.parquet")
    side_rows = LI.load_train_outcomes(env["pack"], env["sidecar"])
    shared = list(twin_rows.columns)
    assert (list(side_rows.reset_index(drop=True)[shared].dtypes)
            == list(twin_rows[shared].dtypes))
    pd.testing.assert_frame_equal(side_rows.reset_index(drop=True)[shared],
                                  twin_rows[shared])

    # The loss-target MAPPING itself is unchanged (delay = 10 + tau/30 and
    # w1/base has tau 570 unavailable in the synthetic shards), but owner
    # decision (b) removes outcome supervision for the excluded arms, so the
    # DELIVERED tensors for w1/base are all-zero.  The raw mapping is checked
    # directly from the packed rows (t0-selected) to prove the data is intact.
    assert "base" in OUTCOME_SUPERVISION_EXCLUDED_ARMS
    quantiles, mask, _collision = side_tensors[("w1", "base")]
    assert not mask.any()
    assert torch.count_nonzero(quantiles) == 0
    manifest = json.loads((twin / "manifest.json").read_text("utf-8"))
    base_item = next(it for it in manifest["items"]
                     if (it["world"], it["arm"]) == ("w1", "base"))
    raw_rows = twin_rows.iloc[base_item["outcome_rows"]]
    raw_q, raw_m, _ = map_outcome_targets(raw_rows, t0_s=base_item["t0_s"])
    assert raw_m.tolist() == [True, True, True, True, False]
    assert raw_q[:, 0].tolist() == pytest.approx(
        [10.0 + t / 30.0 for t in TAUS])
    # a non-excluded arm still delivers its (unchanged) mapping
    native_q, native_m, _ = side_tensors[("w1", "native")]
    assert bool(native_m.any())
    assert native_q[:, 0].tolist() == pytest.approx(
        [10.0 + t / 30.0 for t in TAUS])


def test_receipt_records_and_enforces_exact_dtypes(mixed_env, monkeypatch):
    env = mixed_env
    # SETUP read of the built sidecar table: like any read it lands in the
    # spy, so assert it explicitly and then clear the spy so it covers ONLY
    # the operation under test below.
    built = {c: str(d) for c, d in
             pd.read_parquet(_sidecar_parquet(env)).dtypes.items()}
    assert _receipt(env["sidecar"])["sidecar"]["columns"] == built
    assert env["spy"] == [str(_sidecar_parquet(env))]
    env["spy"].reset()
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    receipt = _receipt(env["sidecar"])
    receipt["sidecar"]["columns"]["hard_brake_s"] = "int64"   # lying receipt
    _write_receipt(env["sidecar"], receipt)
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="columns/dtypes"):
        corpus[0]
    # the sha-validated sidecar parquet itself was opened EXACTLY ONCE by
    # production (the dtype check runs right after the read); nothing else
    # was opened and no labels were returned
    assert env["spy"] == [str(_sidecar_parquet(env))]
    assert str(_mixed_table(env)) not in env["spy"]
    assert str(_cal_shard(env)) not in env["spy"]


def test_loc_uses_original_global_row_ids(mixed_env):
    env = mixed_env
    manifest = json.loads((env["pack"] / "manifest.json").read_text("utf-8"))
    item = next(it for it in manifest["items"]
                if (it["world"], it["arm"]) == ("w1", "base"))
    ids = [int(p) for p in item["outcome_rows"]]
    frame = LI.load_train_outcomes(env["pack"], env["sidecar"])
    assert ids == [5, 6, 7, 8, 9]
    assert frame.loc[ids]["tau"].tolist() == TAUS
    assert frame.loc[ids[::-1]]["tau"].tolist() == TAUS[::-1]


# ------------------------------------------------------------ fail-closed -
def test_tampered_sidecar_parquet_fails_closed(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    with open(_sidecar_parquet(env), "ab") as fh:
        fh.write(b"\x00")
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="sidecar parquet sha256"):
        corpus[0]
    assert env["spy"] == []


def test_tampered_receipt_coverage_fails_closed(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    receipt = _receipt(env["sidecar"])
    receipt["selection"]["n_rows"] += 1
    _write_receipt(env["sidecar"], receipt)
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="row coverage mismatch"):
        corpus[0]
    assert env["spy"] == []


def test_tampered_receipt_item_rows_fails_closed(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    receipt = _receipt(env["sidecar"])
    receipt["items"][0]["n_rows"] += 1
    _write_receipt(env["sidecar"], receipt)
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal):
        corpus[0]
    assert env["spy"] == []


def test_tampered_pack_manifest_fails_closed(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    manifest = json.loads((env["pack"] / "manifest.json").read_text("utf-8"))
    # shift the first (CAL) item's rows: different bytes -> different sha
    manifest["items"][0]["outcome_rows"] = list(
        range(1, len(manifest["items"][0]["outcome_rows"]) + 1))
    (env["pack"] / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="manifest sha256"):
        corpus[0]
    with pytest.raises(CorpusRefusal, match="manifest sha256"):
        corpus.outcome_target(0)
    assert env["spy"] == []


def test_tampered_pack_index_fails_closed(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    index = json.loads((env["pack"] / "INDEX.json").read_text("utf-8"))
    index["runs"][0]["n_outcome_rows"] += 1
    (env["pack"] / "INDEX.json").write_text(json.dumps(index, indent=1),
                                            encoding="utf-8")
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="INDEX sha256"):
        corpus[0]
    assert env["spy"] == []


def test_wrong_split_cal_sidecar_is_refused(mixed_env, monkeypatch):
    env = mixed_env
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(env["sidecar"]))
    receipt = _receipt(env["sidecar"])
    receipt["split"] = "CALIBRATION"
    _write_receipt(env["sidecar"], receipt)
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="CAL sidecars are refused"):
        corpus[0]
    assert env["spy"] == []


def test_missing_receipt_fails_closed(mixed_env, monkeypatch):
    env = mixed_env
    bare = env["sidecar"].parent / "sidecar_without_receipt"
    bare.mkdir(parents=True, exist_ok=True)
    # byte-identical COPY of the sidecar table; the receipt is never
    # created in this fresh variant directory (the built sidecar,
    # receipt included, stays preserved untouched)
    (bare / LI.SIDECAR_PARQUET_NAME).write_bytes(
        _sidecar_parquet(env).read_bytes())
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(bare))
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal, match="missing"):
        corpus[0]
    assert env["spy"] == []


def test_failed_receipt_write_preserves_temporary_receipt(tmp_path):
    target = tmp_path / "receipt.json"
    with pytest.raises(TypeError):
        LI._write_json_atomic(target, {"bad": object()})   # unserializable
    assert not target.exists()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers, "a failed receipt write must PRESERVE its temp file"


# ------------------------------------------------ pickling / DataLoader ---
def test_sidecar_corpus_is_picklable_and_reloads(tmp_path, monkeypatch):
    env_pkg = write_idea_a_pack(tmp_path / "data", _mixed_runs(), k=8)
    sidecar = tmp_path / "sidecar"
    LI.build_sidecar(env_pkg, tmp_path / "data" / "shards", sidecar)
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(sidecar))
    corpus = MemmapCorpus(env_pkg, split="TRAIN", pair_sampling=True)
    first = corpus[0]
    state = corpus.__getstate__()
    assert state["_mm"] == {} and state["_outcome_rows"] is None
    clone = pickle.loads(pickle.dumps(corpus, protocol=4))
    again = clone[0]
    assert sorted(first) == sorted(again)
    for key in sorted(k for k, v in first.items() if torch.is_tensor(v)):
        assert torch.equal(first[key], again[key]), key


def test_dataloader_batches_match_in_process_sidecar_batches(tmp_path,
                                                             monkeypatch):
    env_pkg = write_idea_a_pack(tmp_path / "data", _mixed_runs(), k=8)
    sidecar = tmp_path / "sidecar"
    LI.build_sidecar(env_pkg, tmp_path / "data" / "shards", sidecar)
    monkeypatch.setenv(LI.TRAIN_OUTCOMES_ENV, str(sidecar))
    corpus = MemmapCorpus(env_pkg, split="TRAIN", pair_sampling=True)
    sampler = PairAwareBatchSampler(corpus._items, batch_size=2, seed=3,
                                    epochs=2)
    loader = DataLoader(corpus, batch_sampler=sampler, collate_fn=collate)
    want_batches = [collate([corpus[i] for i in batch])
                    for epoch in (0, 1)
                    for batch in epoch_batches(corpus._items, 2, 3, epoch)]
    got_batches = list(loader)
    assert len(got_batches) == len(want_batches)
    for got, want in zip(got_batches, want_batches):
        assert got["meta"] == want["meta"]
        for key in sorted(k for k, v in want.items() if torch.is_tensor(v)):
            assert torch.equal(got[key], want[key]), key


# ------------------------------------------- single-split / legacy packs --
def test_wholly_train_real_pack_keeps_original_behavior(tmp_path,
                                                        monkeypatch,
                                                        parquet_spy):
    pack = write_idea_a_pack(tmp_path / "data", _train_runs(), k=8)
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(pack, split="TRAIN")
    assert len(corpus) == 3
    item = corpus[0]
    assert item["split"] == "TRAIN"
    assert spy == [str(pack / "outcomes.parquet")]


def test_manifest_only_train_only_pack_keeps_original_behavior(
        tmp_path, monkeypatch, parquet_spy):
    pack = tmp_path / "pack"
    write_pack(pack, make_runs()[:5])      # wholly TRAIN, manifest-only
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(pack, split="TRAIN")
    assert len(corpus) == 5
    corpus[0]
    assert spy == [str(pack / "outcomes.parquet")]


def test_manifest_only_mixed_pack_fails_closed_without_identity(
        tmp_path, monkeypatch, parquet_spy):
    pack = tmp_path / "pack"
    write_pack(pack, make_runs())          # mixed TRAIN+CAL, no INDEX.json
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(pack, split="TRAIN")
    with pytest.raises(CorpusRefusal, match="mixed-split corpus pack"):
        corpus[0]
    with pytest.raises(CorpusRefusal):
        corpus.outcome_target(0)
    assert spy == []


# --------------------- wholly-TRAIN manifest vs a PRESENT INDEX (audit) ----
def _train_pack_with_index(tmp_path):
    """A wholly TRAIN real-writer pack (INDEX.json present and agreeing)."""
    return write_idea_a_pack(tmp_path / "data", _train_runs(), k=8)


@pytest.mark.parametrize("mutate,match", [
    ("cal_run", "split identity drift"),
    # foreign_run REPLACES a valid INDEX run: the EXISTING manifest item
    # (chunk4/w1__base) becomes the unmatched side, and audit_pack_identity
    # walks MANIFEST ITEMS FIRST -- so the first refusal is the item-side
    # branch ("manifest item ... matches no INDEX run; the pack INDEX
    # contradicts the manifest identity").  The run-side branch ("INDEX
    # runs without a matching manifest item") runs only after every
    # manifest item matched; it is covered separately by
    # test_train_manifest_index_nontrain_foreign_run_fails_closed, which
    # APPENDS a foreign run instead of replacing one.
    ("foreign_run", "matches no INDEX run"),
    ("dropped_run", "matches no INDEX run"),
])
def test_train_manifest_contradicted_by_index_fails_closed(
        tmp_path, monkeypatch, parquet_spy, mutate, match):
    pack = _train_pack_with_index(tmp_path)
    shards = tmp_path / "data" / "shards"
    index = json.loads((pack / "INDEX.json").read_text(encoding="utf-8"))
    if mutate == "cal_run":
        # an INDEX run marked CALIBRATION inside a wholly-TRAIN manifest
        index["runs"][0]["split"] = "CALIBRATION"
    elif mutate == "foreign_run":
        # an INDEX run identity no manifest item names; shard_dir kept
        # consistent with the NEW key so run parsing/identity checks pass
        # and the FIRST refusal is exactly the item-side branch matched
        # above (validation order: manifest items are audited before the
        # run-side "INDEX runs without a matching manifest item" check)
        index["runs"][0]["key"] = "chunk4/w9__native"
        index["runs"][0]["shard_dir"] = str(
            shards / "chunk4" / "w9" / "native")
    else:
        # a manifest item with no INDEX run at all
        del index["runs"][0]
    (pack / "INDEX.json").write_text(json.dumps(index, indent=1),
                                     encoding="utf-8")
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(pack, split="TRAIN")
    with pytest.raises(CorpusRefusal, match="contradicts"):
        corpus[0]
    with pytest.raises(CorpusRefusal, match=match):
        corpus.outcome_target(0)
    assert spy == []                       # refused BEFORE any label read


def test_train_manifest_index_nontrain_foreign_run_fails_closed(
        tmp_path, monkeypatch, parquet_spy):
    pack = _train_pack_with_index(tmp_path)
    shards = tmp_path / "data" / "shards"
    index = json.loads((pack / "INDEX.json").read_text(encoding="utf-8"))
    index["runs"].append(dict(
        key="chunk4/w9__native", split="CALIBRATION",
        shard_dir=str(shards / "chunk4" / "w9" / "native"),
        n_outcome_rows=5))
    (pack / "INDEX.json").write_text(json.dumps(index, indent=1),
                                     encoding="utf-8")
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(pack, split="TRAIN")
    with pytest.raises(CorpusRefusal, match="without a matching manifest"):
        corpus[0]
    assert spy == []


def test_wholly_train_real_pack_without_index_keeps_legacy_read(
        tmp_path, monkeypatch, parquet_spy):
    full = _train_pack_with_index(tmp_path)     # preserved untouched
    pack = _pack_variant_without_index(full, tmp_path / "pack_without_index")
    monkeypatch.delenv(LI.TRAIN_OUTCOMES_ENV, raising=False)
    spy = parquet_spy.reset()
    corpus = MemmapCorpus(pack, split="TRAIN")
    assert len(corpus) == 3
    corpus[0]
    assert spy == [str(pack / "outcomes.parquet")]


def test_empty_manifest_pack_refuses_labels(tmp_path):
    pack = _train_pack_with_index(tmp_path)
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    manifest["items"] = []
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    # an empty manifest proves no training-only table: refused for TRAIN
    # and CALIBRATION alike, before any label read
    with pytest.raises(CorpusRefusal, match="no items"):
        MemmapCorpus(pack, split="TRAIN")
    with pytest.raises(CorpusRefusal, match="no items"):
        MemmapCorpus(pack, split="CALIBRATION")


def test_audit_pack_identity_agrees_on_real_pack_and_refuses_empty(
        mixed_pack):
    env = mixed_pack
    manifest = json.loads((env["pack"] / "manifest.json").read_text(
        encoding="utf-8"))
    index = json.loads((env["pack"] / "INDEX.json").read_text(
        encoding="utf-8"))
    LI.audit_pack_identity(manifest, index)    # 1:1 identity, splits agree
    with pytest.raises(LI.LabelIsolationError, match="no items"):
        LI.audit_pack_identity({"items": []}, index)


# ------------------------------------------------------------------ CLI ---
def test_cli_build_verify_roundtrip_and_source_rehash(tmp_path):
    pack = write_idea_a_pack(tmp_path / "data", _mixed_runs(), k=8)
    shards = tmp_path / "data" / "shards"
    out = tmp_path / "sidecar_cli"
    assert LI.main(["build", "--pack", str(pack), "--shards-root",
                    str(shards), "--out", str(out)]) == 0
    assert (out / LI.RECEIPT_NAME).is_file()
    assert (out / LI.SIDECAR_PARQUET_NAME).is_file()
    argv = ["verify", "--pack", str(pack), "--sidecar", str(out)]
    assert LI.main(argv) == 0
    assert LI.main(argv + ["--shards-root", str(shards),
                           "--rehash-sources"]) == 0
    assert LI.main(["verify", "--pack", str(pack),
                    "--sidecar", str(tmp_path / "missing")]) == 2

    # a tampered TRAIN source shard is caught only when sources are rehashed
    src = shards / "chunk4" / "w1" / "base" / "outcomes.parquet"
    frame = pd.read_parquet(src)
    frame.loc[0, "hard_brake_s"] = 123.0
    frame.to_parquet(src, index=False)
    assert LI.main(argv + ["--shards-root", str(shards),
                           "--rehash-sources"]) == 2
    assert LI.main(argv) == 0               # pack/sidecar binding still holds


def test_cli_build_refuses_existing_output(tmp_path, capsys):
    pack = write_idea_a_pack(tmp_path / "data", _mixed_runs(), k=8)
    shards = tmp_path / "data" / "shards"
    argv = ["build", "--pack", str(pack), "--shards-root", str(shards),
            "--out", str(tmp_path / "sidecar_cli")]
    assert LI.main(argv) == 0
    capsys.readouterr()
    assert LI.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False
