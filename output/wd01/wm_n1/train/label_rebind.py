"""WM-N1 label-lookup fix: rebuild the TRAIN outcome sidecar with rebinding.

The frozen WM-N1 TRAIN outcome sidecar (``train.label_isolation``, schema
``wm_n1_train_outcomes_sidecar/1``) was built by resolving every pack item's
v0.4a label source to the run-dir ``arm``.  That is wrong for the chunk-2 /
chunk-3 CAMPAIGN per-CAV switching runs: their pack items carry the run-dir
arm ``native`` / ``pi`` with policy ``switching``, while the v0.4a producer
writes those labels as ``<world>__swnative`` / ``<world>__swpi`` and keeps
``<world>__native`` / ``<world>__pi`` for the separate J3 TRUE fleet arms
(``partB_v0data/v04a_sources.py:168-182``; file naming
``partB_v0data/build_v04a.py:250,273``).  Consequently 196 TRAIN items
(49 each of chunk-2 native / pi and chunk-3 native / pi; 69 rows each;
13,524 rows) carry fleet labels whose shard ``outcomes`` SHA is identical to
the truearms shard.

:func:`features.outcomes.label_source` now resolves the correct source.  This
module rebuilds the TRAIN sidecar from the pinned /1 sidecar and the frozen
v0.4a label parquets:

* every TRAIN run whose resolved source differs from the source of its
  existing rows is REBOUND: the item's ``decision_t`` is read from its TRAIN
  shard ``meta.json``, the matching ``<world>__<swnative|swpi>_labels.parquet``
  is read from the frozen v0.4a root (SHA verified against that root's
  ``SHA256SUMS`` when listed), the rows are selected exactly as
  :func:`features.outcomes.load_outcome_rows` selects them, the identity is
  hard-checked and the selected count must equal the run's
  ``n_outcome_rows``;
* unchanged runs' rows are copied value-identical (asserted);
* a rebound run whose count / identity / columns fail is EXCLUDED (its rows
  are removed and the reason recorded), never guessed;
* TRAIN items whose fixed lookup WOULD find labels but whose pack has zero
  outcome rows are listed (not added) as an owner decision.

The caller supplies the pinned input receipt + sidecar SHAs; labels reach
training ONLY through the sidecar (``WM_N1_TRAIN_OUTCOMES`` ->
``train.label_isolation.load_train_outcomes`` ->
``train.dataset._outcomes`` / ``_select_outcome_rows`` ``.loc`` by original
``_global_row_id``), so the identity sidecar (``WM_N1_TRAIN_IDENTITIES``,
vehicle ids only) is unaffected.  The all-native reference of the CONTRAST
loss is decided by the pack POLICY (``train/dataset.py:107-114``), so
rebinding the label source does not change arm/pair semantics.

CLI (code-only here; the owner runs it once)::

    python -m train.label_rebind build \
        --pack PACK --shards-root SHARDS --input-sidecar DIR \
        --input-receipt-sha256 SHA --input-sidecar-sha256 SHA \
        --v04a-root V04A --out NEW_DIR --code-commit REV
    python -m train.label_rebind verify --pack PACK --sidecar DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

SIDECAR_SCHEMA = "wm_n1_train_outcomes_sidecar/1"
SIDECAR_SCHEMA_V2 = "wm_n1_train_outcomes_sidecar/2"
RECEIPT_NAME = "TRAIN_ISOLATION_RECEIPT.json"
SIDECAR_PARQUET_NAME = "train_outcomes.parquet"
ROW_ID_COLUMN = "_global_row_id"
REBIND_REPORT_NAME = "REBIND_REPORT.json"
LABEL_SOURCE_SHAKEY = "label_source_sha256"

#: Minimum free space (bytes) required on the output volume before a build.
MIN_FREE_BYTES = 512 * 1024 * 1024

_SHA_RE = "[0-9a-f]{64}"


class LabelRebindError(PermissionError):
    """Fail-closed: a rebind invariant was violated; nothing was guessed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    out = h.hexdigest()
    if re.fullmatch(_SHA_RE, out) is None:
        raise LabelRebindError("internal SHA computation failed")
    return out


def _read_json(path) -> dict:
    from .label_isolation import _read_json as read
    return read(path)


def _write_json(path, obj) -> None:
    from .label_isolation import _write_json_atomic
    _write_json_atomic(path, obj)


def _label_source(batch, chunk, arm, policy):
    from features.outcomes import label_source
    return label_source(batch, chunk, arm, policy)


def _expected_policy(source):
    from features.outcomes import EXPECTED_LABEL_IDENTITY
    return EXPECTED_LABEL_IDENTITY.get(source)


def _v04a_labels_path(v04a_root, chunk, world, source) -> Path:
    from features.outcomes import labels_path
    return labels_path(chunk, world, source, v04a_root)


def _check_identity(df, source, item) -> None:
    from features.outcomes import _check_identity
    _check_identity(df, source, item)


def _load_sums(root: Path) -> dict:
    """Parse a v04a ``SHA256SUMS`` (``<sha>  <relpath>``), else ``{}``."""
    path = Path(root) / "SHA256SUMS"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        out[parts[1].strip()] = parts[0].strip()
    return out


def _free_bytes(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(str(probe)).free


def _refuse_out(out_dir: Path, pack_dir: Path, shards_root: Path,
                v04a_root: Path) -> None:
    if out_dir.exists():
        raise LabelRebindError(
            f"refusing to touch existing output directory {out_dir}; "
            "pick a fresh one")
    for ancestor in (out_dir, *out_dir.parents):
        if ancestor.is_symlink():
            raise LabelRebindError(
                f"output path {out_dir} travels through symlink {ancestor}")
    resolved = out_dir.parent.resolve()
    for name, tree in (("pack", pack_dir), ("shards", shards_root),
                       ("v04a", v04a_root)):
        tree_r = Path(tree).resolve()
        if resolved == tree_r or tree_r in resolved.parents:
            raise LabelRebindError(
                f"output directory {out_dir} lies inside the {name} tree "
                f"{tree_r}")
    if out_dir.drive.lower() == "d:" and not out_dir.as_posix().lower() \
            .startswith("d:/rd-staging/"):
        raise LabelRebindError(
            "D: output is allowed only under D:/rd-staging/")
    if _free_bytes(out_dir.parent) < MIN_FREE_BYTES:
        raise LabelRebindError(
            f"insufficient free space under {out_dir.parent} "
            f"(< {MIN_FREE_BYTES} bytes)")


def _conform(frame, columns, dtypes):
    """Lossless cast of ``frame`` to the pinned column set/dtypes."""
    import numpy as np

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise LabelRebindError(f"rebound rows lack pinned columns {missing}")
    frame = frame[list(columns)].copy()
    for col in columns:
        want = dtypes[col]
        got = frame[col].dtype
        if str(got) == str(want):
            continue
        series = frame[col]
        if str(want) == "object":
            frame[col] = series.astype(object)
            continue
        if np.issubdtype(want, np.integer) and np.issubdtype(
                got, np.floating):
            values = series.to_numpy()
            if not np.all(np.isfinite(values)) \
                    or not np.all(values == np.round(values)):
                raise LabelRebindError(
                    f"lossy cast for column {col!r}: non-integral float")
            frame[col] = series.astype(want)
            continue
        if np.issubdtype(want, np.floating) and np.issubdtype(
                got, np.integer):
            frame[col] = series.astype(want)
            continue
        raise LabelRebindError(
            f"lossy/unsupported cast for column {col!r}: {got} -> {want}")
    return frame


def _existing_rows(pinned, first: int, n: int):
    return pinned.loc[first:first + n - 1]


def _rebound_rows(label_path, source, item, decision_t, n):
    import pandas as pd

    df = pd.read_parquet(label_path)
    want_t = {round(float(t), 6) for t in decision_t}
    want_tau = {30.0, 60.0, 120.0, 300.0, 570.0}
    keep = df["t"].round(6).isin(want_t) & df["tau"].round(6).isin(want_tau)
    out = df.loc[keep].copy().sort_values(["t", "tau"]).reset_index(drop=True)
    _check_identity(out, source, item)
    if len(out) != n:
        raise LabelRebindError(
            f"rebound selection holds {len(out)} rows, run claims {n}")
    return out


def build_rebound_sidecar(pack_dir, shards_root, input_sidecar_dir,
                          v04a_root, out_dir, code_commit,
                          input_receipt_sha256, input_sidecar_sha256) -> dict:
    """Build the schema-2 TRAIN sidecar from the pinned /1 sidecar."""
    import pandas as pd

    from .label_isolation import plan_sidecar

    pack_dir = Path(pack_dir)
    shards_root = Path(shards_root)
    input_sidecar_dir = Path(input_sidecar_dir)
    v04a_root = Path(v04a_root)
    out_dir = Path(out_dir)

    if re.fullmatch(_SHA_RE, str(input_receipt_sha256)) is None \
            or re.fullmatch(_SHA_RE, str(input_sidecar_sha256)) is None:
        raise LabelRebindError("input receipt/sidecar SHA256 pins required")

    receipt_path = input_sidecar_dir / RECEIPT_NAME
    parquet_path = input_sidecar_dir / SIDECAR_PARQUET_NAME
    if _sha256_file(receipt_path) != input_receipt_sha256:
        raise LabelRebindError("input receipt SHA256 does not match the pin")
    if _sha256_file(parquet_path) != input_sidecar_sha256:
        raise LabelRebindError("input sidecar parquet SHA256 does not match "
                               "the pin")
    input_receipt = _read_json(receipt_path)
    if input_receipt.get("schema") != SIDECAR_SCHEMA \
            or input_receipt.get("split") != "TRAIN":
        raise LabelRebindError("input is not a /1 TRAIN isolation receipt")
    plan = plan_sidecar(pack_dir, None)
    plan_runs = {r["key"]: r for r in plan["train_runs"]}
    runs_raw = input_receipt.get("runs") or []
    items_raw = input_receipt.get("items") or []
    if len(runs_raw) != len(plan_runs) or len(items_raw) != len(runs_raw):
        raise LabelRebindError("input receipt run/item count differs from pack")
    header = input_receipt.get("sidecar") or {}
    pin_dtypes = (header.get("columns") or {})
    if ROW_ID_COLUMN not in pin_dtypes:
        raise LabelRebindError("pinned sidecar columns lack the row-id column")
    pinned_raw = pd.read_parquet(parquet_path)
    pin_columns = list(pinned_raw.columns)
    if set(pin_columns) != set(pin_dtypes):
        raise LabelRebindError(
            "pinned parquet columns != receipt columns")
    for col, dtype in pinned_raw.dtypes.items():
        if str(dtype) != pin_dtypes.get(col):
            raise LabelRebindError(
                f"pinned dtype drift for {col!r}: {dtype} != {pin_dtypes[col]}")
    pinned = pinned_raw.set_index(ROW_ID_COLUMN)
    if pinned.index.tolist() != sorted(pinned.index.tolist()):
        raise LabelRebindError("pinned global row ids are not ascending")

    _refuse_out(out_dir, pack_dir, shards_root, v04a_root)
    sums = _load_sums(v04a_root)

    _data_cols = [c for c in pin_columns if c != ROW_ID_COLUMN]
    out_runs = []
    rebound_list = []
    excluded_list = []
    would_gain_list = []
    zero_list = []
    order = []
    total_in = 0
    for run, item in zip(runs_raw, items_raw):
        key = str(run["key"])
        if key not in plan_runs:
            raise LabelRebindError(f"input run {key!r} is not a pack TRAIN run")
        plan_run = plan_runs[key]
        for field, want in (("first_global_row", plan_run["first"]),
                            ("n_outcome_rows", plan_run["n"]),
                            ("input_sha256", plan_run["input_sha256"])):
            if run.get(field) != want:
                raise LabelRebindError(
                    f"input run {key!r} {field} conflicts with the pack plan")
        n = int(run["n_outcome_rows"])
        first = int(run["first_global_row"])
        total_in += n
        batch = item.get("batch", item.get("chunk"))
        chunk = item.get("chunk")
        arm = item.get("arm")
        policy = item.get("policy")
        source = _label_source(batch, chunk, arm, policy)
        rec = dict(run)
        rec["resolved_source"] = source
        rec["label_policy"] = _expected_policy(source) if source else None
        rec["rebound"] = False
        rec["excluded"] = False
        if n == 0:
            if source is not None:
                label_path = _v04a_labels_path(v04a_root, chunk,
                                               item["world"], source)
                if label_path.is_file():
                    would_gain_list.append(dict(
                        key=key, world=item["world"], chunk=chunk, arm=arm,
                        policy=policy, resolved_source=source,
                        label_source_path=str(label_path)))
                else:
                    zero_list.append(key)
            else:
                zero_list.append(key)
            order.append((first, n, None))
            out_runs.append(rec)
            continue
        existing = _existing_rows(pinned, first, n)
        existing_sources = sorted(set(existing["source"].astype(str)))
        if source is not None and existing_sources == [source]:
            rec["rebound"] = False
            order.append((first, n, ("copy", existing.reset_index())))
            out_runs.append(rec)
            continue
        if source is None:
            rec["excluded"] = True
            rec["reason"] = (
                "no v0.4a source for batch/arm/policy "
                f"{batch}/{arm}/{policy}")
            rec["outcomes_sha256"] = None
            excluded_list.append(dict(key=key, reason=rec["reason"],
                                      n_outcome_rows=n, world=item["world"],
                                      chunk=chunk, arm=arm, policy=policy))
            order.append((first, n, None))
            out_runs.append(rec)
            continue
        item_meta = dict(world=item["world"], chunk=chunk, arm=arm,
                         batch=batch, policy=policy,
                         split=item.get("split"), t0_s=item["t0_s"])
        label_path = _v04a_labels_path(v04a_root, chunk, item["world"], source)
        if not label_path.is_file():
            reason = f"missing v0.4a label file {label_path}"
            rec["excluded"] = True
            rec["reason"] = reason
            rec["outcomes_sha256"] = None
            excluded_list.append(dict(key=key, reason=reason,
                                      n_outcome_rows=n, world=item["world"],
                                      chunk=chunk, arm=arm, policy=policy))
            order.append((first, n, None))
            out_runs.append(rec)
            continue
        try:
            meta_path = shards_root / run["shard_rel"] / "meta.json"
            meta = _read_json(meta_path)
            if str(meta.get("split")).upper() != "TRAIN":
                raise LabelRebindError("shard meta split is not TRAIN")
            for mk, want in (("world", item["world"]), ("arm", arm),
                             ("chunk", chunk),
                             ("batch", batch if batch is not None
                              else chunk)):
                if meta.get(mk) is not None and meta.get(mk) != want:
                    raise LabelRebindError(
                        f"shard meta {mk} {meta.get(mk)!r} != item {want!r}")
            decision_t = [float(t) for t in (meta.get("decision_t") or [])]
            if not decision_t:
                raise LabelRebindError("shard meta has no decision_t")
            label_sha = _sha256_file(label_path)
            rel = label_path.relative_to(v04a_root).as_posix()
            if rel in sums and sums[rel] != label_sha:
                raise LabelRebindError(
                    f"v0.4a label file {rel} SHA differs from SHA256SUMS")
            rows = _rebound_rows(label_path, source, item_meta, decision_t, n)
            frame = _conform(rows, _data_cols, pin_dtypes)
            frame[ROW_ID_COLUMN] = range(first, first + n)
            frame = frame[pin_columns]
        except Exception as exc:  # noqa: BLE001 - exclude, never guess
            reason = f"{type(exc).__name__}: {exc}"
            rec["excluded"] = True
            rec["reason"] = reason
            rec["outcomes_sha256"] = None
            excluded_list.append(dict(key=key, reason=reason,
                                      n_outcome_rows=n, world=item["world"],
                                      chunk=chunk, arm=arm, policy=policy))
            order.append((first, n, None))
            out_runs.append(rec)
            continue
        old_policy = sorted(set(existing["policy"].astype(str)))
        rec["rebound"] = True
        rec["old_source"] = existing_sources
        rec["old_policy"] = old_policy
        rec["outcomes_sha256"] = None
        rec[LABEL_SOURCE_SHAKEY] = label_sha
        rec["label_source_path"] = str(label_path)
        rec["decision_t_count"] = len(decision_t)
        rec["rows"] = n
        rebound_list.append(dict(
            key=key, old_source=existing_sources, old_policy=old_policy,
            new_source=source, new_policy=_expected_policy(source),
            label_source_path=str(label_path),
            label_source_sha256=label_sha,
            decision_t_count=len(decision_t), rows=n))
        order.append((first, n, ("rebound", frame)))
        out_runs.append(rec)

    order.sort(key=lambda x: x[0])
    frames = []
    seen = set()
    for first, n, payload in order:
        if n == 0 or payload is None:
            continue
        kind, frame = payload
        ids = [int(v) for v in frame[ROW_ID_COLUMN].tolist()]
        if ids != list(range(first, first + n)):
            raise LabelRebindError(
                f"run at {first} did not keep its contiguous row ids")
        if seen.intersection(ids):
            raise LabelRebindError("rebound row-id overlap")
        seen.update(ids)
        frames.append(frame)
    if not frames:
        raise LabelRebindError("rebound sidecar has no rows")
    side = pd.concat(frames, ignore_index=True)
    side = _conform(side, pin_columns, pin_dtypes)
    side = side.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    ids = [int(v) for v in side[ROW_ID_COLUMN].tolist()]
    if len(set(ids)) != len(ids):
        raise LabelRebindError("rebound sidecar has duplicate row ids")
    if ids != sorted(ids):
        raise LabelRebindError("rebound sidecar row ids are not ascending")

    out_dir.mkdir(parents=True)
    out_parquet = out_dir / SIDECAR_PARQUET_NAME
    side.to_parquet(out_parquet, index=False)
    side_sha = _sha256_file(out_parquet)

    new_receipt = dict(input_receipt)
    new_receipt["schema"] = SIDECAR_SCHEMA_V2
    new_receipt["created_utc"] = _utc_now()
    new_receipt["runs"] = out_runs
    new_receipt["sidecar"] = dict(header)
    new_receipt["sidecar"]["sha256"] = side_sha
    new_receipt["sidecar"]["n_rows"] = int(len(side))
    new_receipt["label_rebind"] = dict(
        input_sidecar_dir=str(input_sidecar_dir),
        input_receipt_sha256=input_receipt_sha256,
        input_sidecar_sha256=input_sidecar_sha256,
        input_receipt_schema=SIDECAR_SCHEMA,
        code_commit=str(code_commit),
        v04a_root=str(v04a_root),
        shards_root=str(shards_root),
        n_rebound_runs=len(rebound_list),
        n_rebound_rows=sum(r["rows"] for r in rebound_list),
        n_excluded_runs=len(excluded_list),
        n_excluded_rows=sum(r["n_outcome_rows"] for r in excluded_list),
        n_would_gain_label=len(would_gain_list),
        rebound=rebound_list,
        excluded=excluded_list,
        would_gain_label=would_gain_list,
        zero_label_runs=len(zero_list),
    )
    _write_json(out_dir / RECEIPT_NAME, new_receipt)

    report = dict(
        schema="wm_n1_label_rebind_report/1", created_utc=_utc_now(),
        split="TRAIN", code_commit=str(code_commit),
        pack_dir=str(pack_dir), shards_root=str(shards_root),
        v04a_root=str(v04a_root), out_dir=str(out_dir),
        input_sidecar_dir=str(input_sidecar_dir),
        input_receipt_sha256=input_receipt_sha256,
        input_sidecar_sha256=input_sidecar_sha256,
        sidecar_sha256=side_sha,
        n_rows_in=int(total_in), n_rows_out=int(len(side)),
        n_rebound_runs=len(rebound_list),
        n_rebound_rows=sum(r["rows"] for r in rebound_list),
        n_excluded_runs=len(excluded_list),
        n_excluded_rows=sum(r["n_outcome_rows"] for r in excluded_list),
        n_would_gain_label=len(would_gain_list),
        n_zero_label_runs=len(zero_list),
        rebound=rebound_list,
        excluded=excluded_list,
        would_gain_label=would_gain_list,
    )
    _write_json(out_dir / REBIND_REPORT_NAME, report)
    return report


def verify_rebound_sidecar(pack_dir, sidecar_dir) -> dict:
    """Validate a /2 sidecar through the loader; returns a JSON-safe report."""
    from .label_isolation import load_train_outcomes
    frame = load_train_outcomes(pack_dir, sidecar_dir)
    receipt = _read_json(Path(sidecar_dir) / RECEIPT_NAME)
    block = receipt.get("label_rebind") or {}
    return dict(ok=True, schema=SIDECAR_SCHEMA_V2,
                sidecar_dir=str(Path(sidecar_dir)),
                n_rows=int(len(frame)),
                n_rebound_runs=block.get("n_rebound_runs"),
                n_excluded_runs=block.get("n_excluded_runs"),
                n_would_gain_label=block.get("n_would_gain_label"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="train.label_rebind")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--pack", required=True)
    p_build.add_argument("--shards-root", required=True)
    p_build.add_argument("--input-sidecar", required=True)
    p_build.add_argument("--input-receipt-sha256", required=True)
    p_build.add_argument("--input-sidecar-sha256", required=True)
    p_build.add_argument("--v04a-root", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--code-commit", required=True)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--pack", required=True)
    p_verify.add_argument("--sidecar", required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "build":
            report = build_rebound_sidecar(
                args.pack, args.shards_root, args.input_sidecar,
                args.v04a_root, args.out, args.code_commit,
                args.input_receipt_sha256, args.input_sidecar_sha256)
            print(json.dumps(dict(
                ok=True, out=str(args.out),
                n_rebound_runs=report["n_rebound_runs"],
                n_rebound_rows=report["n_rebound_rows"],
                n_excluded_runs=report["n_excluded_runs"],
                n_would_gain_label=report["n_would_gain_label"],
                sidecar_sha256=report["sidecar_sha256"]), indent=1))
        else:
            print(json.dumps(verify_rebound_sidecar(args.pack, args.sidecar),
                             indent=1))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(dict(ok=False, error_type=type(exc).__name__,
                              error=str(exc)), indent=1))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
