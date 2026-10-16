# ruff: noqa
# VENDORED from output/wd01/wm_n1/train/label_isolation.py (last changed in commit 855a992; copied at baseline bf7d4cb).
# Only change: lazy `features.outcomes` imports re-pointed to this vendor package.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""TRAIN label isolation: build/load the TRAIN-only outcome sidecar.

``train.dataset.MemmapCorpus`` on a MIXED-split corpus pack (TRAIN +
CALIBRATION, e.g. ``pack_v3``: 2153 TRAIN / 310 CALIBRATION items) used to
``pd.read_parquet`` the whole packed ``outcomes.parquet`` even for
``split="TRAIN"``, so every TRAIN run opened every CALIBRATION label row.
This module is the code-only isolation fix:

* :func:`plan_sidecar` derives, from the pack ``manifest.json`` and
  ``INDEX.json`` METADATA only (no parquet is opened), which global
  outcome-row ranges belong to which run: offsets accumulate each INDEX
  run's ``n_outcome_rows`` in INDEX order (``outcome_first`` is NOT assumed
  to be stored) and every manifest item's ``outcome_rows`` must reconcile
  exactly with its run's derived range.  Forbidden content (HOLDOUT/TEST/
  SEALED splits, chunk-5) is refused outright.
* :func:`build_sidecar` then opens ONLY positive-row per-run TRAIN shard
  ``outcomes.parquet`` files selected above (shard tree layout
  ``<shards-root>/<chunk>/<world>/<arm>``, e.g.
  ``data_cache/shards_v3``, fetched with ``python -m wms fetch get shards_v3``) and
  writes a NEW output directory
  holding the TRAIN label rows -- original shard columns, exact values and
  dtypes, each tagged with its original global row id -- plus an immutable
  receipt binding the raw manifest+INDEX shas, every TRAIN source-shard
  outcome sha, the sidecar parquet sha and the row-id coverage.  The mixed
  packed ``outcomes.parquet`` is never opened and CALIBRATION shard files
  are never touched.
* Every selected shard is addressed through its logical
  ``<chunk>/<world>/<arm>`` identity, so validation stays portable: the
  recorded ``shard_dir`` is parsed LEXICALLY and OS-independently
  (``PureWindowsPath``/``PurePosixPath``), so a Windows ``D:/...`` or UNC
  source path recorded by the staging host is recognized on a POSIX GPU
  box exactly like a ``/...`` POSIX path is on Windows -- the recorded
  source is never opened, created or resolved.  With ``shards_root=None``
  (dataset / GPU-box path) the recorded relative shard identity is still
  derived and enforced from metadata; with a ``shards_root`` every path
  component is additionally checked to be a real directory inside the
  root, and every ``outcomes.parquet`` FILE is checked to be a regular,
  non-symlink file resolving to its exact selected shard BEFORE it is read
  or hashed.
* :func:`build_sidecar` writes only into a FRESH output directory that is
  not inside (and travels through no symlink into) the pack or source-shard
  trees.  A failed receipt write PRESERVES its temporary file as the failure
  receipt (no deletion on failure).
* :func:`load_train_outcomes` re-validates the receipt against the pack
  (manifest/INDEX shas, run identities, recorded shard identities, row
  coverage, sidecar sha, exact sidecar columns/dtypes) BEFORE any label is
  returned, then returns the TRAIN frame indexed by the original global row
  ids so the dataset can ``.loc[item["outcome_rows"]]``.
* Split and chunk spellings are conservative: only ``TRAIN`` /
  ``CALIBRATION`` splits are recognized (unknown spellings are refused, not
  ignored) and every ``chunk5``-variant spelling (``chunk-5``, ``chunk_5``,
  ``Chunk 5``, ...) is treated as the sealed chunk; a ``chunk``-prefixed
  label without a digit is an unknown spelling and is refused.

The dataset picks the sidecar up through the ``WM_N1_TRAIN_OUTCOMES``
environment variable (a sidecar DIRECTORY), so every canonical dataset
consumer (``MemmapCorpus``, ``train.idea_a.IdeaACorpus``, eval/adapter
paths) isolates automatically and the same env path is portable to the GPU
box.  A mixed-split pack opened as TRAIN without a valid sidecar refuses
before any label read.  Wholly single-split packs and manifest-only
synthetic fixtures keep the original mixed-table behavior.  A receipt whose
split is not TRAIN (a CAL sidecar) is refused outright through this
utility.

CLI (code-only; nothing here is executed by adding it)::

    python -m train.label_isolation plan   --pack PACK --shards-root ROOT
    python -m train.label_isolation build  --pack PACK --shards-root ROOT --out NEW_DIR
    python -m train.label_isolation verify --pack PACK --sidecar DIR \
        [--shards-root ROOT --rehash-sources]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

SIDECAR_SCHEMA = "wm_n1_train_outcomes_sidecar/1"
SIDECAR_SCHEMA_V2 = "wm_n1_train_outcomes_sidecar/2"
SIDECAR_SCHEMA_V3 = "wm_n1_train_outcomes_sidecar/3"
SIDECAR_SCHEMAS = (SIDECAR_SCHEMA, SIDECAR_SCHEMA_V2, SIDECAR_SCHEMA_V3)
RECEIPT_NAME = "TRAIN_ISOLATION_RECEIPT.json"
SIDECAR_PARQUET_NAME = "train_outcomes.parquet"
# schema/3 overlay sidecar (train.label_overlay): the /2 layer ships as
# ``base_train_outcomes.parquet`` (a byte-copy of the pinned /2 parquet)
# and the zero-row-item rows live in ``overlay_outcomes.parquet`` keyed by
# full item identity instead of global row ids.
SIDECAR_PARQUET_NAME_V3_BASE = "base_train_outcomes.parquet"
SIDECAR_PARQUET_NAME_V3_OVERLAY = "overlay_outcomes.parquet"
OVERLAY_IDENTITY_COLUMNS = ("batch", "chunk_id", "world", "arm", "split",
                            "t0_s")
ROW_ID_COLUMN = "_global_row_id"
LABEL_SOURCE_SHAKEY = "label_source_sha256"
TRAIN_OUTCOMES_ENV = "WM_N1_TRAIN_OUTCOMES"

SIDE_SCHEMA = "wm_n1_pack/1"
CORPUS_INDEX_SCHEMA = "wm_n1_corpus_pack/2"
SPLIT = "TRAIN"
ALLOWED_SPLITS = ("TRAIN", "CALIBRATION")
# Mirrors of train.dataset guards (kept dependency-free on purpose).
FORBIDDEN_SPLITS = ("HOLDOUT", "TEST", "SEALED")
FORBIDDEN_CHUNKS = ("chunk5",)

_REQUIRED_INDEX_RUN_KEYS = ("key", "shard_dir", "split", "n_outcome_rows")
_ITEM_REQUIRED = ("world", "arm", "chunk", "split", "t0_s", "pair_key",
                  "tick", "plan_row", "outcome_rows")
# Host path flavor, used ONLY to decide whether a recorded ABSOLUTE path
# can be containment-resolved locally; recorded paths are always parsed
# lexically and flavor-independently (see ``_recorded_parts``).
_HOST_WINDOWS = os.name == "nt"


class LabelIsolationError(PermissionError):
    """Fail-closed: an isolation invariant was violated; nothing was read."""


# ------------------------------------------------------------- primitives --
def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise LabelIsolationError(f"required file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise LabelIsolationError(f"invalid JSON at {path}: {e}") from e


def _write_json_atomic(path, obj) -> Path:
    """Atomic JSON write that PRESERVES the temp file on failure.

    A failed write deliberately leaves its ``.tmp`` file in place: it is the
    failure receipt of the interrupted build and must never be deleted
    (review requirement: no ``os.unlink`` deletion-on-failure).
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return path


def _split_of(value) -> str:
    return str(value).strip().upper()


def _norm_chunk(chunk) -> str:
    """Separator/case-insensitive form of a chunk label (``chunk-5`` ==
    ``chunk5`` == ``Chunk 5``); used only for conservative refusal checks."""
    return re.sub(r"[\s\-_.]+", "", str(chunk)).lower()


def _refuse_content(kind: str, ident: str, split: str, chunk: str) -> None:
    s = _split_of(split)
    c = _norm_chunk(chunk)
    if s in FORBIDDEN_SPLITS:
        raise LabelIsolationError(
            f"{kind} {ident} carries forbidden split {s!r} "
            "(HOLDOUT/TEST are sealed)")
    if c.startswith("chunk5"):
        raise LabelIsolationError(
            f"{kind} {ident} is under {chunk!r} (chunk-5 TEST is sealed)")
    if s not in ALLOWED_SPLITS:
        raise LabelIsolationError(
            f"{kind} {ident} carries unknown split spelling {split!r}; "
            f"only {ALLOWED_SPLITS} are recognized")
    if c.startswith("chunk") and not c[5:6].isdigit():
        raise LabelIsolationError(
            f"{kind} {ident} carries unknown chunk spelling {chunk!r}")


def _guard_file(path, what: str) -> Path:
    """A file we are about to read or hash: regular, present, no symlink."""
    path = Path(path)
    if path.is_symlink():
        raise LabelIsolationError(
            f"{what} must be a regular file, not a symlink: {path}")
    if not path.is_file():
        raise LabelIsolationError(f"required file is missing: {path}")
    if path.resolve() != path.parent.resolve() / path.name:
        raise LabelIsolationError(
            f"{what} does not resolve inside its own directory: {path}")
    return path


def _guard_shard_dir(root, rel: str, what: str) -> Path:
    """Resolve ``root/rel`` with every component literal (no symlinks, no
    escapes); the shard directory must sit exactly at the selected relative
    location under the already-resolved shards root."""
    root = Path(root)
    rel_parts = Path(rel).parts
    cur = root
    for part in rel_parts:
        cur = cur / part
        if cur.is_symlink():
            raise LabelIsolationError(
                f"{what} component {cur} is a symlink; the selected shard "
                "path must stay literal inside the shards root")
    if cur.resolve() != root.joinpath(*rel_parts):
        raise LabelIsolationError(
            f"{what} {rel!r} escapes the shards root {str(root)!r}")
    if not cur.is_dir():
        raise LabelIsolationError(f"{what} directory is missing: {cur}")
    return cur


def _outcome_source(root: Path, run: dict) -> tuple[Path | None, str | None]:
    """Bind one TRAIN run to DONE before any outcome parquet is opened."""
    shard = _guard_shard_dir(root, run["shard_rel"], "TRAIN shard")
    done = _read_json(_guard_file(shard / "DONE.json", "TRAIN shard DONE"))
    if not isinstance(done, dict):
        raise LabelIsolationError("TRAIN shard DONE must be an object")
    if (done.get("schema") != "wm_n1_features_index/1"
            or done.get("batch", done.get("chunk")) != run["batch"]
            or any(done.get(k) != run[k] for k in
                   ("chunk", "world", "arm", "split", "t0_s",
                    "input_sha256"))
            or type(done.get("n_outcome_rows")) is not int
            or done.get("n_outcome_rows") != run["n"]
            or not isinstance(done.get("shard_dir"), str)
            or _shard_rel_logical(dict(done, key=run["key"], batch=run["batch"]))
            != run["shard_rel"]):
        raise LabelIsolationError("TRAIN shard DONE identity/count differs from pack plan")
    if run.get("tick_start") is not None and run.get("tick_end") is not None \
            and done.get("n_ticks") != run["tick_end"] - run["tick_start"]:
        raise LabelIsolationError("TRAIN shard DONE tick count differs from pack plan")
    outputs = done.get("outputs")
    if not isinstance(outputs, dict):
        raise LabelIsolationError("TRAIN shard DONE outputs missing")
    path = shard / "outcomes.parquet"
    if run["n"] == 0:
        if "outcomes.parquet" in outputs or os.path.lexists(path):
            raise LabelIsolationError("zero-row TRAIN shard declares or holds outcomes")
        return None, None
    claimed = outputs.get("outcomes.parquet")
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        raise LabelIsolationError("positive TRAIN shard has no DONE outcome SHA")
    source = _guard_file(path, "selected TRAIN shard outcomes")
    actual = _sha256_file(source)
    if actual != claimed:
        raise LabelIsolationError("TRAIN shard outcome SHA differs from DONE")
    return source, actual


def _refuse_unsafe_output(out_dir, pack_dir, shards_root) -> None:
    """Sidecar output must be FRESH and cannot live inside (or travel
    through a symlink into) the pack or source-shard trees."""
    out = Path(out_dir)
    if out.is_symlink():
        raise LabelIsolationError(
            f"output path {out} is a symlink; refusing (sidecar output may "
            "not symlink into pack/source/frozen trees)")
    if out.exists():
        raise LabelIsolationError(
            f"refusing to touch existing output directory {out}; pick a "
            "fresh one (never overwrite)")
    for ancestor in (out, *out.parents):
        if ancestor.is_symlink():
            raise LabelIsolationError(
                f"output path {out} travels through symlink {ancestor}; "
                "refusing (sidecar output may not symlink into pack/source/"
                "frozen trees)")
    resolved_parent = out.parent.resolve()
    for name, tree in (("pack", pack_dir), ("source shards", shards_root)):
        if tree is None:
            continue
        resolved_tree = Path(tree).resolve()
        if resolved_parent == resolved_tree \
                or resolved_tree in resolved_parent.parents:
            raise LabelIsolationError(
                f"output directory {out} lies inside the {name} tree "
                f"{resolved_tree}; pick a fresh location outside pack/source "
                "trees")


def _recorded_parts(value) -> tuple:
    """Lexically parse a RECORDED path string on ANY host OS.

    Returns ``(parts, absolute, windows)``: the literal components, whether
    the recorded path is absolute, and whether it carries a Windows drive
    or UNC prefix.  A Windows ``D:/...`` / ``D:\\...`` drive or
    ``//server/share/...`` UNC spelling is recognized via
    :class:`PureWindowsPath` even on a POSIX host, and a ``/...`` POSIX
    spelling is recognized even on Windows; the parse is purely lexical,
    so the recorded source is NEVER opened, created or resolved here.
    Empty spellings, drive-relative spellings (a Windows drive without a
    root) and any ``..`` traversal component are refused outright.
    """
    text = str(value)
    win = PureWindowsPath(text)
    if win.drive and not win.is_absolute():
        raise LabelIsolationError(
            f"recorded path {value!r} is drive-relative (a Windows drive "
            "without a root); this spelling is refused")
    if win.drive:
        parts, absolute, windows = win.parts, True, True
    else:
        posix = PurePosixPath(text)
        parts, absolute, windows = posix.parts, posix.is_absolute(), False
    if not parts:
        raise LabelIsolationError(f"recorded path {value!r} is empty")
    if ".." in parts:
        raise LabelIsolationError(
            f"recorded path {value!r} contains '..' traversal; refused")
    return parts, absolute, windows


def _shard_rel_logical(run: dict) -> str:
    """Logical ``<chunk>/<world>/<arm>`` identity of a run's recorded shard.

    Portable validation: derived purely from the recorded ``shard_dir`` and
    the run identity, it needs NO filesystem access, so receipt/plan
    reconciliation enforces the recorded relative shard identity even on a
    machine without the shard tree (``shards_root=None``).  Parsing is
    lexical and OS-independent (:func:`_recorded_parts`): a recorded
    Windows ``D:/...`` absolute path parses identically on a POSIX host and
    a ``/...`` POSIX path on Windows.  A relative ``shard_dir`` must have
    exactly three components; an absolute one must END in the three
    identity components; ``..`` traversal spellings are refused outright,
    so they can never match the identity (and the recorded source is never
    opened or resolved).
    """
    parts, absolute, _windows = _recorded_parts(run["shard_dir"])
    if absolute:
        if len(parts) < 3:
            raise LabelIsolationError(
                f"TRAIN shard {run['shard_dir']!r} is too short to end in "
                f"<chunk>/<world>/<arm> for run {run['key']!r}")
        rel_parts = parts[-3:]
    else:
        if len(parts) != 3:
            raise LabelIsolationError(
                f"relative TRAIN shard {run['shard_dir']!r} is not "
                f"<chunk>/<world>/<arm> for run {run['key']!r}")
        rel_parts = parts
    head, world, arm = rel_parts
    if (world, arm) != (run["world"], run["arm"]) \
            or head not in (run["batch"], run["chunk"]):
        raise LabelIsolationError(
            f"TRAIN shard {run['shard_dir']!r} does not match run identity "
            f"{run['key']!r}")
    return PurePosixPath(*rel_parts).as_posix()


def _shard_rel_of(run: dict, shards_root: Path) -> str:
    """Filesystem containment + identity check of a selected TRAIN shard.

    A recorded absolute path of the HOST's own flavor must still resolve
    inside the (already resolved) shards root exactly as recorded.  A
    relative recorded identity -- or an absolute one recorded on the OTHER
    OS flavor (e.g. a ``D:/...`` path recorded on Windows, validated
    against a local mirror on the POSIX GPU box) -- is addressed by its
    logical identity under the local root; the recorded source itself is
    never opened or resolved.
    """
    logical = _shard_rel_logical(run)
    parts, absolute, windows = _recorded_parts(run["shard_dir"])
    if absolute and windows == _HOST_WINDOWS:
        shard_dir = Path(run["shard_dir"])
    else:
        rel_parts = parts[-3:] if absolute else parts
        shard_dir = Path(shards_root).joinpath(*rel_parts)
    try:
        rel = shard_dir.resolve().relative_to(shards_root)
    except ValueError:
        raise LabelIsolationError(
            f"TRAIN shard {run['shard_dir']!r} escapes the shards root "
            f"{str(shards_root)!r}") from None
    parts = rel.parts
    if len(parts) != 3:
        raise LabelIsolationError(
            f"TRAIN shard {run['shard_dir']!r} is not "
            f"<chunk>/<world>/<arm> under {str(shards_root)!r}")
    head, world, arm = parts
    if (world, arm) != (run["world"], run["arm"]) \
            or head not in (run["batch"], run["chunk"]):
        raise LabelIsolationError(
            f"TRAIN shard {run['shard_dir']!r} does not match run identity "
            f"{run['key']!r}")
    if rel.as_posix() != logical:
        raise LabelIsolationError(
            f"TRAIN shard resolves to {rel.as_posix()!r}, not the recorded "
            f"logical identity {logical!r} for run {run['key']!r}")
    return rel.as_posix()


# ------------------------------------------------------- planning (no IO) --
def plan_sidecar(pack_dir, shards_root=None) -> dict:
    """Derive the TRAIN selection from pack metadata; opens NO parquet.

    Every run's logical ``<chunk>/<world>/<arm>`` shard identity is always
    derived and enforced from the recorded ``shard_dir`` (portable, no
    filesystem access).  ``shards_root`` (when given) additionally validates
    that every selected TRAIN shard lives under the root as
    ``<chunk>/<world>/<arm>`` (no path escape, literal components).  With
    ``shards_root=None`` the plan is pure metadata with the logical identity
    checks intact (the dataset-side validation path, which must work without
    the shard tree, e.g. on the GPU box).
    """
    pack_dir = Path(pack_dir)
    manifest = _read_json(pack_dir / "manifest.json")
    if manifest.get("schema") != SIDE_SCHEMA:
        raise LabelIsolationError(
            f"pack manifest schema {manifest.get('schema')!r} != "
            f"{SIDE_SCHEMA!r} at {pack_dir}")
    items = _manifest_items(manifest)
    index = _read_json(pack_dir / "INDEX.json")
    if index.get("schema") != CORPUS_INDEX_SCHEMA:
        raise LabelIsolationError(
            f"pack INDEX schema {index.get('schema')!r} != "
            f"{CORPUS_INDEX_SCHEMA!r} at {pack_dir}")
    runs_raw = index.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise LabelIsolationError(
            f"pack INDEX.json at {pack_dir} has no runs list")

    runs = []
    cursor = 0
    for rec in runs_raw:
        run = _derived_run(rec)
        run["first"] = cursor
        cursor += run["n"]
        # portable, metadata-only shard identity -- always validated
        run["shard_rel"] = _shard_rel_logical(run)
        runs.append(run)
    n_rows_total = cursor

    by_ident = {}
    for run in runs:
        ident = (run["batch"], run["world"], run["arm"])
        if ident in by_ident:
            raise LabelIsolationError(
                f"duplicate INDEX run identity {ident}")
        by_ident[ident] = run
    for item in items:
        ident = (str(item.get("batch") or item["chunk"]),
                 str(item["world"]), str(item["arm"]))
        run = by_ident.get(ident)
        if run is None:
            raise LabelIsolationError(
                f"manifest item {ident} matches no INDEX run")
        if item["split"] != run["split"]:
            raise LabelIsolationError(
                f"manifest item {ident} split {item['split']!r} != INDEX run "
                f"split {run['split']!r} (non-TRAIN selection drift)")
        t0 = run.get("t0_s")
        if t0 is not None and abs(float(item["t0_s"]) - float(t0)) > 1e-6:
            raise LabelIsolationError(
                f"manifest item {ident} t0_s {item['t0_s']!r} != run "
                f"t0_s {t0!r}")
        expected = list(range(run["first"], run["first"] + run["n"]))
        if [int(p) for p in item["outcome_rows"]] != expected:
            raise LabelIsolationError(
                f"manifest item {ident} outcome_rows do not equal its run's "
                f"derived global range {expected[0]}..{expected[-1]} "
                f"({run['n']} rows); overlapping/drifted row ranges are "
                "refused")
        tick_start, tick_end = run.get("tick_start"), run.get("tick_end")
        if tick_start is not None and tick_end is not None \
                and not tick_start <= int(item["tick"]) < int(tick_end):
            raise LabelIsolationError(
                f"manifest item {ident} tick {item['tick']} outside its run "
                f"tick range [{tick_start}, {tick_end})")
        run["item"] = item
    unmatched = [run["key"] for run in runs if "item" not in run]
    if unmatched:
        raise LabelIsolationError(
            f"INDEX runs without a matching manifest item: {unmatched}")

    train_runs = sorted((r for r in runs if r["split"] == SPLIT),
                        key=lambda r: r["first"])
    if shards_root is not None:
        root = Path(shards_root).resolve()
        if not root.is_dir():
            raise LabelIsolationError(
                f"shards root {str(shards_root)!r} is not a directory (the raw shard tree is "
                f"Drive-only: `python -m wms fetch get shards_v3`)")
        for run in train_runs:
            run["shard_rel"] = _shard_rel_of(run, root)
    previous_end = -1
    for run in train_runs:
        if run["first"] <= previous_end and run["n"]:
            raise LabelIsolationError(
                f"selected TRAIN row ranges overlap at run {run['key']!r}")
        previous_end = max(previous_end, run["first"] + run["n"] - 1)

    excluded: dict = {}
    for run in runs:
        if run["split"] != SPLIT:
            excluded[run["split"]] = excluded.get(run["split"], 0) + 1
    train_rows = sum(run["n"] for run in train_runs)
    row_ids = ([run["first"] for run in train_runs]
               + [run["first"] + run["n"] - 1 for run in train_runs
                  if run["n"]]) if train_runs else []
    plan = dict(
        pack_dir=str(pack_dir),
        k_max=index.get("k_max"), format=index.get("format"),
        manifest=manifest, index=index, runs=runs, items=items,
        train_runs=train_runs,
        n_rows_total=n_rows_total,
        selection=dict(
            n_runs=len(train_runs), n_rows=train_rows,
            row_id_min=min(row_ids) if row_ids else None,
            row_id_max=max(row_ids) if row_ids else None,
            row_id_column=ROW_ID_COLUMN,
            excluded_splits=excluded,
            n_items_train=len(train_runs),
            n_items_other=len(items) - len(train_runs)),
    )
    return plan


def _manifest_items(manifest: dict) -> list:
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise LabelIsolationError("pack manifest has no items")
    out = []
    for raw in items:
        missing = [k for k in _ITEM_REQUIRED if k not in raw]
        if missing:
            raise LabelIsolationError(
                f"manifest item {raw.get('world')}/{raw.get('arm')} misses "
                f"required key(s) {missing}")
        split = _split_of(raw["split"])
        _refuse_content("manifest item",
                        f"{raw.get('world')}/{raw.get('arm')}",
                        split, raw.get("chunk", ""))
        out.append(dict(raw, split=split))
    return out


def _derived_run(rec) -> dict:
    if not isinstance(rec, dict):
        raise LabelIsolationError(f"INDEX run record is not a dict: {rec!r}")
    missing = [k for k in _REQUIRED_INDEX_RUN_KEYS
               if k not in rec or rec[k] is None or rec[k] == ""]
    if missing:
        raise LabelIsolationError(
            f"INDEX run record misses required key(s) {missing}: "
            f"{rec.get('key')!r}")
    key = str(rec["key"])
    try:
        batch, tail = key.split("/", 1)
        world, arm = tail.rsplit("__", 1)
    except ValueError:
        raise LabelIsolationError(
            f"INDEX run key {key!r} is not <batch>/<world>__<arm>") from None
    split = _split_of(rec["split"])
    chunk = str(rec.get("chunk") or batch)
    _refuse_content("INDEX run", key, split, chunk)
    n = int(rec["n_outcome_rows"])
    if n < 0:
        raise LabelIsolationError(
            f"INDEX run {key!r} has negative n_outcome_rows {n}")
    return dict(
        key=key, batch=batch, world=world, arm=arm, chunk=chunk, split=split,
        t0_s=rec.get("t0_s"), shard_dir=str(rec["shard_dir"]),
        shard_rel=None, input_sha256=rec.get("input_sha256"), n=n,
        tick_start=rec.get("tick_start"), tick_end=rec.get("tick_end"),
        dec_start=rec.get("dec_start"))


# --------------------------------------------- pack identity audit (no IO) -
def audit_pack_identity(manifest: dict, index: dict) -> None:
    """Metadata-only reconciliation of a pack manifest against its INDEX.

    Fail-closed BEFORE any label read; opens NO parquet and NO shard tree.
    Every INDEX run must be a usable record (``key`` parseable as
    ``<batch>/<world>__<arm>``, recognized split, no chunk-5 spelling,
    ``shard_dir`` matching that identity, non-negative ``n_outcome_rows``);
    every manifest item must match exactly one INDEX run by
    ``(batch, world, arm)`` identity with an agreeing split and (when the
    run records one) ``t0_s``; and every INDEX run must be matched by a
    manifest item.

    The dataset calls this on a wholly single-split TRAIN manifest whenever
    a pack INDEX is PRESENT: an INDEX naming a non-TRAIN or unmatched run
    contradicts the manifest's training-only claim, so the packed outcomes
    table may hold excluded rows and the pack refuses.  A manifest-only
    pack with NO INDEX keeps its legacy single-split behavior (the caller
    decides; a MIXED manifest requires a usable INDEX regardless).  An
    empty manifest proves no training-only table and refuses here too.
    """
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise LabelIsolationError(
            "pack manifest has no items; an empty manifest proves no "
            "training-only outcome table and refuses every label read")
    runs_raw = index.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise LabelIsolationError("pack INDEX.json has no runs list")
    by_ident: dict = {}
    for rec in runs_raw:
        run = _derived_run(rec)
        _shard_rel_logical(run)        # portable, metadata-only, no FS
        ident = (run["batch"], run["world"], run["arm"])
        if ident in by_ident:
            raise LabelIsolationError(f"duplicate INDEX run identity {ident}")
        by_ident[ident] = run
    matched = set()
    for item in items:
        split = _split_of(item.get("split", ""))
        _refuse_content("manifest item",
                        f"{item.get('world')}/{item.get('arm')}",
                        split, item.get("chunk", ""))
        ident = (str(item.get("batch") or item["chunk"]),
                 str(item["world"]), str(item["arm"]))
        run = by_ident.get(ident)
        if run is None:
            raise LabelIsolationError(
                f"manifest item {ident} matches no INDEX run; the pack "
                "INDEX contradicts the manifest identity")
        if split != run["split"]:
            raise LabelIsolationError(
                f"manifest item {ident} split {split!r} != INDEX run split "
                f"{run['split']!r} (non-TRAIN split identity drift)")
        t0 = run.get("t0_s")
        if t0 is not None and abs(float(item["t0_s"]) - float(t0)) > 1e-6:
            raise LabelIsolationError(
                f"manifest item {ident} t0_s {item['t0_s']!r} != INDEX run "
                f"t0_s {t0!r} (INDEX contradicts the manifest identity)")
        matched.add(ident)
    unmatched = sorted(set(by_ident) - matched)
    if unmatched:
        raise LabelIsolationError(
            f"INDEX runs without a matching manifest item: {unmatched}; the "
            "pack INDEX contradicts the manifest identity")


# ------------------------------------------------------------- build ------
def build_sidecar(pack_dir, shards_root, out_dir) -> dict:
    """Build the TRAIN-only sidecar in a NEW directory; never overwrites.

    Selection happens BEFORE any open: only the planned TRAIN shard
    ``outcomes.parquet`` files are read, each guarded as a regular file at
    its exact selected shard (no symlink, no escape) right before the read
    and its hash.  The output directory must be fresh and outside the
    pack/source trees.  The packed mixed outcomes table and every non-TRAIN
    shard are never touched.
    """
    pack_dir = Path(pack_dir)
    out_dir = Path(out_dir)
    _refuse_unsafe_output(out_dir, pack_dir, shards_root)
    plan = plan_sidecar(pack_dir, shards_root)
    train_runs = plan["train_runs"]
    if not train_runs:
        raise LabelIsolationError(
            f"pack {pack_dir} has no TRAIN runs; nothing to isolate")
    root = Path(shards_root).resolve()

    import pandas as pd

    frames = []
    source_shas = []
    for run in train_runs:
        src, source_sha = _outcome_source(root, run)
        source_shas.append(source_sha)
        if src is None:
            continue
        frame = pd.read_parquet(src)
        if _sha256_file(src) != source_sha:
            raise LabelIsolationError("TRAIN shard outcome changed during read")
        if len(frame) != run["n"]:
            raise LabelIsolationError(
                f"shard outcomes {src} holds {len(frame)} rows but INDEX "
                f"claims {run['n']} for {run['key']!r}")
        frame = frame.assign(
            **{ROW_ID_COLUMN: range(run["first"], run["first"] + run["n"])})
        frames.append(frame)

    if not frames:
        raise LabelIsolationError("TRAIN selection has no outcome rows")
    side = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True)
    parquet_path = out_dir / SIDECAR_PARQUET_NAME
    side.to_parquet(parquet_path, index=False)

    receipt = dict(
        schema=SIDECAR_SCHEMA, created_utc=_utc_now(), split=SPLIT,
        tool="train.label_isolation",
        pack=dict(
            dir=str(pack_dir),
            manifest_name="manifest.json",
            manifest_sha256=_sha256_file(pack_dir / "manifest.json"),
            index_name="INDEX.json",
            index_sha256=_sha256_file(pack_dir / "INDEX.json"),
            k_max=plan["k_max"], format=plan["format"],
            n_items=len(plan["items"]),
            n_items_train=plan["selection"]["n_items_train"],
            n_items_other=plan["selection"]["n_items_other"],
            n_rows_total=plan["n_rows_total"]),
        selection=dict(plan["selection"],
                       shards_root=str(root),
                       policy="TRAIN-only outcomes from original per-run "
                              "shards; mixed pack outcomes and non-TRAIN "
                              "shards never opened"),
        runs=[dict(key=run["key"], split=run["split"],
                   shard_rel=run["shard_rel"],
                   shard_dir_recorded=run["shard_dir"],
                   input_sha256=run["input_sha256"],
                   outcomes_sha256=source_sha,
                   first_global_row=run["first"],
                   n_outcome_rows=run["n"])
             for run, source_sha in zip(train_runs, source_shas)],
        items=[dict(world=item["world"], arm=item["arm"],
                    chunk=item["chunk"], batch=item.get("batch"),
                    split=item["split"], t0_s=item["t0_s"],
                    pair_key=item["pair_key"], policy=item.get("policy"),
                    tick=int(item["tick"]), plan_row=int(item["plan_row"]),
                    first_global_row=run["first"], n_rows=run["n"])
              for run, item in ((r, r["item"]) for r in train_runs)],
        sidecar=dict(file=SIDECAR_PARQUET_NAME,
                     sha256=_sha256_file(parquet_path),
                     n_rows=int(len(side)), parquet_index=ROW_ID_COLUMN,
                     columns={col: str(dt)
                              for col, dt in side.dtypes.items()}),
    )
    _write_json_atomic(out_dir / RECEIPT_NAME, receipt)
    return receipt


# ------------------------------------------------------- load / verify -----
def _validate(pack_dir, sidecar_dir, *, shards_root=None,
              rehash_sources: bool = False) -> tuple:
    """Receipt-vs-pack validation; opens NO label bytes (hashing only)."""
    pack_dir = Path(pack_dir)
    sidecar_dir = Path(sidecar_dir)
    receipt = _read_json(_guard_file(sidecar_dir / RECEIPT_NAME,
                                     "sidecar receipt"))
    outer_schema = receipt.get("schema")
    if outer_schema not in SIDECAR_SCHEMAS:
        raise LabelIsolationError(
            f"sidecar receipt schema {outer_schema!r} not in "
            f"{SIDECAR_SCHEMAS!r} at {sidecar_dir}")
    if outer_schema == SIDECAR_SCHEMA_V2:
        _reconcile_rebind_block(receipt)
    if outer_schema == SIDECAR_SCHEMA_V3:
        receipt = _validate_v3_header(receipt, sidecar_dir)
        _reconcile_rebind_block(receipt)
    if _split_of(receipt.get("split")) != SPLIT:
        raise LabelIsolationError(
            f"sidecar receipt split {receipt.get('split')!r} is not "
            f"{SPLIT!r}; CAL sidecars are refused through this utility")
    manifest_sha = _sha256_file(pack_dir / "manifest.json")
    index_sha = _sha256_file(pack_dir / "INDEX.json")
    pack = receipt.get("pack") or {}
    if manifest_sha != pack.get("manifest_sha256"):
        raise LabelIsolationError(
            "pack manifest sha256 does not match the sidecar receipt "
            f"({manifest_sha} != {pack.get('manifest_sha256')})")
    if index_sha != pack.get("index_sha256"):
        raise LabelIsolationError(
            "pack INDEX sha256 does not match the sidecar receipt "
            f"({index_sha} != {pack.get('index_sha256')})")

    plan = plan_sidecar(pack_dir, shards_root)
    _reconcile_receipt(receipt, plan)

    if outer_schema == SIDECAR_SCHEMA_V3:
        # base + overlay parquet SHAs were verified in _validate_v3_header
        # against the OUTER receipt (the effective receipt below is the
        # embedded /2 receipt, which pins the /2 parquet the base copies)
        pass
    else:
        parquet_path = _guard_file(sidecar_dir / SIDECAR_PARQUET_NAME,
                                   "sidecar parquet")
        parquet_sha = _sha256_file(parquet_path)
        if parquet_sha != (receipt.get("sidecar") or {}).get("sha256"):
            raise LabelIsolationError(
                "sidecar parquet sha256 does not match the receipt "
                f"({parquet_sha} != "
                f"{(receipt.get('sidecar') or {}).get('sha256')})")

    if rehash_sources:
        if shards_root is None:
            raise LabelIsolationError(
                "--rehash-sources requires --shards-root")
        root = Path(shards_root).resolve()
        for rec, run in zip(receipt.get("runs") or [], plan["train_runs"]):
            _src, got = _outcome_source(root, run)
            if got != rec["outcomes_sha256"]:
                raise LabelIsolationError(
                    f"TRAIN source {run['key']} sha256 {got} != receipt "
                    f"{rec['outcomes_sha256']}")
    return receipt, plan


def _validate_v3_header(receipt: dict, sidecar_dir) -> dict:
    """Schema/3: validate the embedded input /2 receipt and the overlay
    pins; returns the EFFECTIVE base receipt (the embedded /2 receipt with
    the /3 pack echo, so the /2 reconciliation runs unchanged)."""
    embedded = receipt.get("input_sidecar_receipt")
    if not isinstance(embedded, dict):
        raise LabelIsolationError("schema/3 receipt lacks the embedded "
                                  "input /2 receipt")
    if embedded.get("schema") != SIDECAR_SCHEMA_V2 \
            or _split_of(embedded.get("split")) != SPLIT:
        raise LabelIsolationError(
            "schema/3 input receipt is not a TRAIN /2 receipt")
    pins = receipt.get("input_sidecar") or {}
    if not re.fullmatch(r"[0-9a-f]{64}", str(pins.get("receipt_sha256"))) \
            or not re.fullmatch(r"[0-9a-f]{64}",
                                str(pins.get("parquet_sha256"))):
        raise LabelIsolationError("schema/3 input pins are not SHA256s")
    sidecar_dir = Path(sidecar_dir)
    # The /3 directory is self-contained: the input RECEIPT pin was checked
    # by the builder against the input /2 dir; at load time the base layer
    # (a byte-copy of the pinned /2 parquet) is authenticated directly, and
    # the /3 receipt itself is bound by the sidecar SHA256SUMS.
    got = _sha256_file(_guard_file(sidecar_dir / SIDECAR_PARQUET_NAME_V3_BASE,
                                   "schema/3 base parquet"))
    if got != pins.get("parquet_sha256"):
        raise LabelIsolationError(
            f"schema/3 base parquet sha256 {got} != pinned "
            f"{pins.get('parquet_sha256')}")
    side = receipt.get("sidecar") or {}
    got_base = _sha256_file(
        _guard_file(sidecar_dir / SIDECAR_PARQUET_NAME_V3_BASE,
                    "sidecar base parquet"))
    if got_base != side.get("base_sha256"):
        raise LabelIsolationError(
            "sidecar base parquet sha256 does not match the receipt "
            f"({got_base} != {side.get('base_sha256')})")
    got_overlay = _sha256_file(
        _guard_file(sidecar_dir / SIDECAR_PARQUET_NAME_V3_OVERLAY,
                    "sidecar overlay parquet"))
    if got_overlay != side.get("sha256"):
        raise LabelIsolationError(
            "sidecar overlay parquet sha256 does not match the receipt "
            f"({got_overlay} != {side.get('sha256')})")
    sums_path = sidecar_dir / "SHA256SUMS"
    if sums_path.is_file():
        expect = {SIDECAR_PARQUET_NAME_V3_BASE: got_base,
                  SIDECAR_PARQUET_NAME_V3_OVERLAY: got_overlay,
                  RECEIPT_NAME: _sha256_file(_guard_file(
                      sidecar_dir / RECEIPT_NAME, "sidecar receipt"))}
        listed: dict = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                listed[parts[1].strip()] = parts[0].strip()
        if listed != expect:
            raise LabelIsolationError(
                "schema/3 SHA256SUMS does not bind the sidecar files: "
                f"{listed!r} != {expect!r}")
    # the /3 top-level runs/items/pack/selection must mirror the embedded
    # /2 receipt (the base layer it ships)
    for key in ("runs", "items", "pack", "selection"):
        if receipt.get(key) != embedded.get(key):
            raise LabelIsolationError(
                f"schema/3 top-level {key!r} differs from the embedded /2 "
                "receipt (the base layer must be the pinned /2 sidecar)")
    # overlay block presence
    overlay = receipt.get("overlay") or {}
    for key in ("n_accepted_groups", "n_accepted_rows", "n_excluded_groups",
                "alignment", "counts_by_batch_arm", "groups", "excluded"):
        if key not in overlay:
            raise LabelIsolationError(f"schema/3 overlay block lacks {key!r}")
    return embedded


def _reconcile_receipt(receipt: dict, plan: dict) -> None:
    """Receipt coverage/identities must equal the re-derived plan exactly."""
    sel_r = dict(receipt.get("selection") or {})
    sel_p = dict(plan["selection"])
    for key in ("n_runs", "n_rows", "row_id_min", "row_id_max",
                "n_items_train", "n_items_other", "excluded_splits"):
        if sel_r.get(key) != sel_p.get(key):
            raise LabelIsolationError(
                f"sidecar receipt selection.{key} "
                f"{sel_r.get(key)!r} != pack-derived {sel_p.get(key)!r} "
                "(row coverage mismatch)")
    runs_r = receipt.get("runs") or []
    runs_p = plan["train_runs"]
    is_v2 = receipt.get("schema") == SIDECAR_SCHEMA_V2
    if len(runs_r) != len(runs_p):
        raise LabelIsolationError(
            f"sidecar receipt lists {len(runs_r)} TRAIN runs, pack derives "
            f"{len(runs_p)}")
    for rec, run in zip(runs_r, runs_p):
        for key, want in (("key", run["key"]),
                          ("split", run["split"]),
                          ("first_global_row", run["first"]),
                          ("n_outcome_rows", run["n"]),
                          ("input_sha256", run["input_sha256"])):
            if rec.get(key) != want:
                raise LabelIsolationError(
                    f"sidecar receipt run field {key} {rec.get(key)!r} != "
                    f"pack-derived {want!r} for {run['key']!r}")
        # portable identity: the recorded relative shard and the recorded
        # shard_dir string must BOTH equal the pack-derived logical identity
        # (never skipped when shards_root is absent)
        if rec.get("shard_rel") != run["shard_rel"]:
            raise LabelIsolationError(
                f"sidecar receipt shard_rel {rec.get('shard_rel')!r} != "
                f"pack-derived {run['shard_rel']!r} for {run['key']!r}")
        if rec.get("shard_dir_recorded") != run["shard_dir"]:
            raise LabelIsolationError(
                "sidecar receipt shard_dir_recorded "
                f"{rec.get('shard_dir_recorded')!r} != pack-derived "
                f"{run['shard_dir']!r} for {run['key']!r}")
        if _split_of(rec.get("split")) != SPLIT:
            raise LabelIsolationError(
                f"sidecar receipt run {rec.get('key')!r} is not a TRAIN run")
        source_sha = rec.get("outcomes_sha256", object())
        rebound = rec.get("rebound") is True
        excluded = rec.get("excluded") is True
        if run["n"] == 0:
            if source_sha is not None or "outcomes_sha256" not in rec:
                raise LabelIsolationError("zero-row TRAIN receipt needs null outcome SHA")
        elif rebound or excluded:
            if not is_v2:
                raise LabelIsolationError(
                    "rebound/excluded run markers require schema/2")
            if source_sha is not None:
                raise LabelIsolationError(
                    "rebound/excluded run must not claim the shard outcome SHA")
            if excluded:
                if not isinstance(rec.get("reason"), str) or not rec["reason"]:
                    raise LabelIsolationError(
                        "excluded run needs a non-empty reason")
            elif not re.fullmatch(r"[0-9a-f]{64}",
                                  str(rec.get(LABEL_SOURCE_SHAKEY))):
                raise LabelIsolationError(
                    "rebound run needs a label_source_sha256")
        elif (not isinstance(source_sha, str)
              or not re.fullmatch(r"[0-9a-f]{64}", source_sha)):
            raise LabelIsolationError("positive TRAIN receipt needs outcome SHA")
    items_r = receipt.get("items") or []
    items_p = [(run["item"], run) for run in runs_p]
    if len(items_r) != len(items_p):
        raise LabelIsolationError(
            f"sidecar receipt lists {len(items_r)} items, pack derives "
            f"{len(items_p)}")
    for rec, (item, run) in zip(items_r, items_p):
        got = (rec.get("world"), rec.get("arm"), rec.get("split"),
               round(float(rec.get("t0_s")), 6),
               int(rec.get("tick")), int(rec.get("plan_row")),
               int(rec.get("first_global_row")), int(rec.get("n_rows")))
        want = (str(item["world"]), str(item["arm"]), item["split"],
                round(float(item["t0_s"]), 6), int(item["tick"]),
                int(item["plan_row"]), int(run["first"]), int(run["n"]))
        if got != want:
            raise LabelIsolationError(
                f"sidecar receipt item {got!r} != pack-derived {want!r}")


def _label_source_of(batch, chunk, arm, policy):
    from wms.legacy.vendor.features.outcomes import label_source
    return label_source(batch, chunk, arm, policy)


def _reconcile_rebind_block(receipt: dict) -> None:
    """Verify the schema/2 ``label_rebind`` block against run identities."""
    from wms.legacy.vendor.features.outcomes import EXPECTED_LABEL_IDENTITY
    block = receipt.get("label_rebind")
    if not isinstance(block, dict):
        raise LabelIsolationError("schema/2 receipt lacks label_rebind block")
    for key in ("input_receipt_sha256", "input_sidecar_sha256", "code_commit",
                "n_rebound_runs", "n_rebound_rows", "n_excluded_runs",
                "n_excluded_rows", "n_would_gain_label", "rebound", "excluded",
                "would_gain_label"):
        if key not in block:
            raise LabelIsolationError(f"label_rebind block lacks {key!r}")
    for key in ("input_receipt_sha256", "input_sidecar_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(block.get(key))):
            raise LabelIsolationError(f"label_rebind {key} is not a SHA256")
    for name, count_key in (("rebound", "n_rebound_runs"),
                            ("excluded", "n_excluded_runs"),
                            ("would_gain_label", "n_would_gain_label")):
        entries = block.get(name)
        if not isinstance(entries, list) \
                or len(entries) != block.get(count_key):
            raise LabelIsolationError(
                f"label_rebind {name} list disagrees with {count_key}")
    runs_r = receipt.get("runs") or []
    items_r = receipt.get("items") or []
    by_key = {str(r.get("key")): r for r in runs_r}
    item_by_key = {str(r.get("key")): it for r, it in zip(runs_r, items_r)}
    rebound_keys = set()
    for entry in block.get("rebound") or []:
        if not isinstance(entry, dict):
            raise LabelIsolationError("rebound entry must be an object")
        key = str(entry.get("key"))
        run = by_key.get(key)
        if run is None or run.get("rebound") is not True:
            raise LabelIsolationError(
                f"label_rebind rebound {key!r} is not a rebound run")
        item = item_by_key.get(key) or {}
        batch = item.get("batch", item.get("chunk"))
        resolved = _label_source_of(batch, item.get("chunk"),
                                    item.get("arm"), item.get("policy"))
        if resolved is None or entry.get("new_source") != resolved \
                or entry.get("new_policy") != EXPECTED_LABEL_IDENTITY.get(
                    resolved):
            raise LabelIsolationError(
                f"label_rebind rebound {key!r} source/policy correspondence "
                "is not table-supported")
        if not re.fullmatch(r"[0-9a-f]{64}",
                            str(entry.get("label_source_sha256"))):
            raise LabelIsolationError("rebound label source SHA missing")
        if int(entry.get("rows", -1)) != int(run.get("n_outcome_rows", -2)):
            raise LabelIsolationError("rebound row count disagrees with run")
        rebound_keys.add(key)
    excluded_keys = set()
    for entry in block.get("excluded") or []:
        if not isinstance(entry, dict):
            raise LabelIsolationError("excluded entry must be an object")
        key = str(entry.get("key"))
        run = by_key.get(key)
        if run is None or run.get("excluded") is not True:
            raise LabelIsolationError(
                f"label_rebind excluded {key!r} is not an excluded run")
        if not entry.get("reason"):
            raise LabelIsolationError("excluded entry needs a reason")
        excluded_keys.add(key)
    flagged_rb = {str(r.get("key")) for r in runs_r
                  if r.get("rebound") is True}
    flagged_ex = {str(r.get("key")) for r in runs_r
                  if r.get("excluded") is True}
    if flagged_rb != rebound_keys or flagged_ex != excluded_keys:
        raise LabelIsolationError(
            "label_rebind lists disagree with the run rebound/excluded flags")


def _open_validated_frame(pack_dir, sidecar_dir, *, shards_root=None,
                          rehash_sources: bool = False):
    receipt, plan = _validate(pack_dir, sidecar_dir, shards_root=shards_root,
                              rehash_sources=rehash_sources)
    import pandas as pd

    sidecar_dir = Path(sidecar_dir)
    base_p = sidecar_dir / SIDECAR_PARQUET_NAME_V3_BASE
    is_v3 = base_p.is_file()          # schema/3 layout: base + overlay
    frame = pd.read_parquet(_guard_file(
        base_p if is_v3 else sidecar_dir / SIDECAR_PARQUET_NAME,
        "sidecar parquet"))
    if ROW_ID_COLUMN not in frame.columns:
        raise LabelIsolationError(
            f"sidecar parquet lacks the {ROW_ID_COLUMN!r} column")
    want_cols = (receipt.get("sidecar") or {}).get("columns") or {}
    if want_cols:
        # schema/3: the effective receipt is the embedded /2 receipt, so
        # this pins the byte-copied base layer to the /2 columns exactly.
        got_cols = {col: str(dt) for col, dt in frame.dtypes.items()}
        if got_cols != want_cols:
            raise LabelIsolationError(
                "sidecar parquet columns/dtypes do not match the receipt "
                f"exactly: {got_cols!r} != {want_cols!r}")
    n_rows = int((receipt.get("sidecar") or {}).get("n_rows", -1))
    is_v2 = receipt.get("schema") == SIDECAR_SCHEMA_V2
    if is_v2:
        if len(frame) != n_rows:
            raise LabelIsolationError(
                f"sidecar parquet holds {len(frame)} rows; receipt claims "
                f"{n_rows}")
    elif len(frame) != n_rows or len(frame) != plan["selection"]["n_rows"]:
        raise LabelIsolationError(
            f"sidecar parquet holds {len(frame)} rows; receipt claims "
            f"{n_rows}; pack derives {plan['selection']['n_rows']}")
    ids = [int(v) for v in frame[ROW_ID_COLUMN].tolist()]
    if len(set(ids)) != len(ids):
        raise LabelIsolationError("sidecar parquet has duplicate row ids")
    expected: set = set()
    for rec in receipt.get("runs") or []:
        if is_v2 and rec.get("excluded") is True:
            continue
        first = int(rec["first_global_row"])
        expected.update(range(first, first + int(rec["n_outcome_rows"])))
    if set(ids) != expected:
        raise LabelIsolationError(
            "sidecar parquet row ids do not equal the receipt coverage")
    if is_v2:
        _verify_rebound_rows(receipt, frame)
    return receipt, frame, is_v3


def _verify_rebound_rows(receipt: dict, frame) -> None:
    """Every rebound run's sidecar rows carry its corrected source/policy."""
    from wms.legacy.vendor.features.outcomes import EXPECTED_LABEL_IDENTITY
    by_id = frame.set_index(ROW_ID_COLUMN)
    for rec in receipt.get("runs") or []:
        if rec.get("rebound") is not True:
            continue
        first = int(rec["first_global_row"])
        n = int(rec["n_outcome_rows"])
        want_source = rec.get("resolved_source")
        want_policy = EXPECTED_LABEL_IDENTITY.get(want_source)
        if want_source is None or want_policy is None:
            raise LabelIsolationError("rebound run lacks a table source")
        got = by_id.loc[list(range(first, first + n))]
        if not (got["source"].astype(str) == want_source).all() \
                or not (got["policy"].astype(str) == want_policy).all():
            raise LabelIsolationError(
                f"rebound run {rec.get('key')!r} rows do not carry the "
                "corrected source/policy")


def sidecar_schema_of(sidecar_dir) -> str:
    """The receipt schema string of a sidecar directory (receipt only)."""
    receipt = _read_json(_guard_file(Path(sidecar_dir) / RECEIPT_NAME,
                                     "sidecar receipt"))
    return str(receipt.get("schema") or "")


def load_train_outcomes(pack_dir, sidecar_dir):
    """Validated TRAIN-only outcome frame indexed by original global row ids.

    Every hash/coverage/identity check runs BEFORE the sidecar parquet is
    opened; the packed mixed outcomes table is never read here.  A schema/3
    overlay sidecar returns its BASE layer (byte-identical to the pinned /2
    sidecar); the zero-row-item rows are read separately through
    :func:`load_train_overlay`.
    """
    _receipt, frame, _is_v3 = _open_validated_frame(pack_dir, sidecar_dir)
    return frame.set_index(ROW_ID_COLUMN)


def load_train_overlay(pack_dir, sidecar_dir):
    """Validated OVERLAY groups of a schema/3 sidecar.

    Returns ``(groups, receipt)`` where ``groups`` maps the full item
    identity ``(batch, chunk, world, arm, split, t0_s)`` to the label rows
    of that zero-row TRAIN item.  Fail-closed BEFORE any row is returned:

    * the base layer is validated exactly like a /2 sidecar
      (:func:`load_train_outcomes`), the overlay parquet matches its
      receipt SHA/columns/row count;
    * every accepted group is a real zero-row TRAIN pack item (identity
      re-derived from the pack plan), its arm is not rule-(b) excluded,
      its resolved source is table-supported and re-checked against the
      rows (:func:`features.outcomes._check_identity`), and the per-group
      row / t0-row counts match the receipt;
    * COVERAGE: every zero-row TRAIN item is accepted, excluded with a
      reason, or rule-(b) excluded -- an uncovered item refuses.
    """
    import pandas as pd

    from wms.legacy.vendor.features.outcomes import _check_identity, label_source

    receipt, frame, is_v3 = _open_validated_frame(pack_dir, sidecar_dir)
    sidecar_dir = Path(sidecar_dir)
    if not is_v3:
        raise LabelIsolationError(
            f"sidecar {sidecar_dir} is not a schema/3 overlay sidecar")
    # the outer receipt (schema/3) with the overlay block; the effective
    # /2 receipt was already validated as the base layer
    outer = _read_json(_guard_file(sidecar_dir / RECEIPT_NAME,
                                   "sidecar receipt"))
    overlay_block = outer.get("overlay") or {}
    overlay_path = _guard_file(sidecar_dir / SIDECAR_PARQUET_NAME_V3_OVERLAY,
                               "sidecar overlay parquet")
    ov = pd.read_parquet(overlay_path)
    want_cols = (outer.get("sidecar") or {}).get("columns") or {}
    got_cols = {col: str(dt) for col, dt in ov.dtypes.items()}
    if want_cols and got_cols != want_cols:
        raise LabelIsolationError(
            "overlay parquet columns/dtypes do not match the receipt: "
            f"{got_cols!r} != {want_cols!r}")
    if len(ov) != int((outer.get("sidecar") or {}).get("n_rows", -2)):
        raise LabelIsolationError(
            f"overlay parquet holds {len(ov)} rows; receipt claims "
            f"{(outer.get('sidecar') or {}).get('n_rows')}")
    for col in OVERLAY_IDENTITY_COLUMNS:
        if col not in ov.columns:
            raise LabelIsolationError(
                f"overlay parquet lacks identity column {col!r}")

    by_key: dict = {}
    for key, g in ov.groupby(["batch", "chunk_id", "world", "arm", "split"],
                             sort=False):
        t0 = round(float(g["t0_s"].iloc[0]), 6)
        if not (g["t0_s"].round(6) == t0).all():
            raise LabelIsolationError(
                f"overlay group {key} mixes t0_s values")
        by_key[(str(key[0]), str(key[1]), str(key[2]), str(key[3]),
                str(key[4]), t0)] = g.reset_index(drop=True)

    plan = plan_sidecar(pack_dir, None)
    runs_by_key = {r["key"]: r for r in plan["train_runs"] if r["n"] == 0}
    accepted = overlay_block.get("groups") or []
    excluded = overlay_block.get("excluded") or []
    if len(accepted) != int(overlay_block.get("n_accepted_groups", -1)) \
            or len(excluded) != int(overlay_block.get("n_excluded_groups",
                                                      -1)):
        raise LabelIsolationError(
            "overlay receipt group counts disagree with its lists")
    seen: dict = {}
    for g in accepted:
        key = (str(g["batch"]), str(g["chunk"]), str(g["world"]),
               str(g["arm"]), str(g["split"]), round(float(g["t0_s"]), 6))
        run = runs_by_key.get(str(g.get("key")))
        if run is None:
            raise LabelIsolationError(
                f"overlay accepted group {g.get('key')!r} is not a "
                "zero-row TRAIN pack item")
        item = run["item"]
        if key != (str(item.get("batch") or item["chunk"]),
                   str(item["chunk"]), str(item["world"]),
                   str(item["arm"]), item["split"],
                   round(float(item["t0_s"]), 6)):
            raise LabelIsolationError(
                f"overlay accepted group {g.get('key')!r} identity differs "
                "from the pack item")
        if str(item["arm"]).strip().lower() in \
                ("base", "native_base"):
            raise LabelIsolationError(
                f"overlay accepted group {g.get('key')!r} is a rule-(b) "
                "excluded arm; refusing")
        source = label_source(key[0], key[1], key[3], item.get("policy"))
        if source is None or source != str(g.get("resolved_source")):
            raise LabelIsolationError(
                f"overlay accepted group {g.get('key')!r} resolved source "
                f"{g.get('resolved_source')!r} is not table-supported")
        rows = by_key.get(key)
        if rows is None:
            raise LabelIsolationError(
                f"overlay accepted group {key!r} has no parquet rows")
        if len(rows) != int(g.get("n_rows", -1)):
            raise LabelIsolationError(
                f"overlay group {key!r} row count differs from the receipt")
        n_t0 = int((rows["t"].round(6) == key[5]).sum())
        if n_t0 != int(g.get("n_rows_t0", -1)) or n_t0 == 0:
            raise LabelIsolationError(
                f"overlay group {key!r} t0-row count differs from the "
                "receipt or is zero")
        item_meta = dict(world=key[2], chunk=key[1], arm=key[3],
                         batch=key[0], policy=item.get("policy"),
                         split=key[4], t0_s=float(item["t0_s"]))
        _check_identity(rows, source, item_meta)
        seen[key] = rows
    if len(seen) != len(accepted):
        raise LabelIsolationError(
            "overlay parquet carries groups outside the receipt accepted "
            "list")

    # COVERAGE: every zero-row TRAIN item is accepted, excluded with a
    # reason, or rule-(b) excluded.
    excluded_keys = set()
    for g in excluded:
        if not g.get("reason"):
            raise LabelIsolationError(
                f"overlay excluded group {g.get('key')!r} lacks a reason")
        key = (str(g["batch"]), str(g["chunk"]), str(g["world"]),
               str(g["arm"]), str(g["split"]), round(float(g["t0_s"]), 6))
        run = runs_by_key.get(str(g.get("key")))
        if run is None:
            raise LabelIsolationError(
                f"overlay excluded group {g.get('key')!r} is not a zero-row "
                "TRAIN pack item")
        item = run["item"]
        if key != (str(item.get("batch") or item["chunk"]),
                   str(item["chunk"]), str(item["world"]),
                   str(item["arm"]), item["split"],
                   round(float(item["t0_s"]), 6)):
            raise LabelIsolationError(
                f"overlay excluded group {g.get('key')!r} identity differs "
                "from the pack item")
        excluded_keys.add(key)
    if len(excluded_keys) != len(excluded):
        raise LabelIsolationError("overlay excluded list has duplicates")
    missing = []
    for key, run in runs_by_key.items():
        item = run["item"]
        ikey = (str(item.get("batch") or item["chunk"]),
                str(item["chunk"]), str(item["world"]), str(item["arm"]),
                item["split"], round(float(item["t0_s"]), 6))
        if ikey in seen or ikey in excluded_keys:
            continue
        if str(item["arm"]).strip().lower() in ("base", "native_base"):
            continue                      # decision (b)
        missing.append(str(run["key"]))
    if missing:
        raise LabelIsolationError(
            f"overlay coverage gap: {len(missing)} zero-row TRAIN items "
            f"are neither accepted nor excluded: {missing[:5]}")
    if seen.keys() - {k for k in by_key}:
        raise LabelIsolationError("internal overlay key mismatch")
    return seen, outer


def verify_sidecar(pack_dir, sidecar_dir, *, shards_root=None,
                   rehash_sources: bool = False) -> dict:
    """Validate a sidecar against its pack; returns a JSON-safe report."""
    receipt, frame, is_v3 = _open_validated_frame(pack_dir, sidecar_dir,
                                                  shards_root=shards_root,
                                                  rehash_sources=rehash_sources)
    sidecar_dir = Path(sidecar_dir)
    schema = receipt.get("schema")
    overlay_info = None
    if is_v3:
        outer = _read_json(_guard_file(sidecar_dir / RECEIPT_NAME,
                                       "sidecar receipt"))
        schema = outer.get("schema")
        ob = outer.get("overlay") or {}
        overlay_info = dict(
            n_accepted_groups=ob.get("n_accepted_groups"),
            n_accepted_rows=ob.get("n_accepted_rows"),
            n_excluded_groups=ob.get("n_excluded_groups"),
            alignment=ob.get("alignment"))
    return dict(ok=True, schema=schema, split=SPLIT,
                sidecar_dir=str(sidecar_dir),
                pack_dir=str(Path(pack_dir)),
                n_runs=len(receipt.get("runs") or []),
                n_rows=int(len(frame)),
                overlay=overlay_info,
                created_utc=receipt.get("created_utc"))


# ----------------------------------------------------------------- CLI ----
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="train.label_isolation",
        description="TRAIN-only outcome sidecar (CAL labels are never "
                    "opened; the packed mixed outcomes table is never read "
                    "for TRAIN)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="metadata-only TRAIN selection")
    p_plan.add_argument("--pack", required=True)
    p_plan.add_argument("--shards-root", default=None)

    p_build = sub.add_parser("build", help="build a sidecar in a NEW dir")
    p_build.add_argument("--pack", required=True)
    p_build.add_argument("--shards-root", required=True)
    p_build.add_argument("--out", required=True,
                         help="fresh output directory (never overwritten)")

    p_verify = sub.add_parser("verify", help="validate sidecar vs pack")
    p_verify.add_argument("--pack", required=True)
    p_verify.add_argument("--sidecar", required=True)
    p_verify.add_argument("--shards-root", default=None)
    p_verify.add_argument("--rehash-sources", action="store_true",
                          help="also re-hash every TRAIN source shard "
                               "outcome against the receipt")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "plan":
            plan = plan_sidecar(args.pack, args.shards_root)
            print(json.dumps(dict(
                ok=True, pack=plan["pack_dir"],
                selection=plan["selection"],
                train_runs=[r["key"] for r in plan["train_runs"]],
                n_rows_total=plan["n_rows_total"]), indent=1))
        elif args.cmd == "build":
            receipt = build_sidecar(args.pack, args.shards_root, args.out)
            print(json.dumps(dict(
                ok=True, out=str(args.out), split=receipt["split"],
                n_runs=len(receipt["runs"]),
                n_rows=receipt["sidecar"]["n_rows"],
                receipt=str(Path(args.out) / RECEIPT_NAME)), indent=1))
        elif args.cmd == "verify":
            report = verify_sidecar(args.pack, args.sidecar,
                                    shards_root=args.shards_root,
                                    rehash_sources=args.rehash_sources)
            print(json.dumps(report, indent=1))
    except LabelIsolationError as e:
        print(json.dumps(dict(ok=False, error=str(e)), indent=1))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
