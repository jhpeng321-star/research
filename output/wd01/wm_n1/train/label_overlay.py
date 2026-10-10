"""WM-N1 TRAIN outcome OVERLAY sidecar (schema/3; owner-approved
2026-10-04, runs/active/WM-N1-LABELCOV-20261004).

``pack_v3`` holds 1,047 TRAIN items with no packed outcome rows (classes
R/M/N of T3's ``LABEL_COVERAGE_PLAN.md``); adding their rows to the pack
would change every downstream row offset (a ~49 GB repack that does not
fit on disk).  This module is the no-repack contract: a schema/3
TRAIN-only OVERLAY sidecar that carries label rows for zero-row TRAIN
items, keyed by FULL ITEM IDENTITY ``(batch, chunk, world, arm, split,
t0_s)`` instead of global row ids, and consumed behind the SAME
``WM_N1_TRAIN_OUTCOMES`` environment variable as the schema/2 sidecar
(:mod:`train.label_isolation` detects the schema and loads both layers).

Sidecar directory (FRESH output; never overwritten)::

    TRAIN_ISOLATION_RECEIPT.json   schema wm_n1_train_outcomes_sidecar/3
    base_train_outcomes.parquet    byte-copy of the pinned /2 parquet
    overlay_outcomes.parquet       the new rows (pin columns minus the
                                   row-id column, plus ``batch``/``arm``)
    SHA256SUMS                     covers all three files

Fail-closed invariants (all inherited from the schema/2 contract):

* the pinned input /2 receipt + parquet SHAs are supplied by the caller
  and re-verified; the base layer is validated against the pack exactly
  like a /2 sidecar (identity, row coverage, rebind block);
* only TRAIN runs with ``n_outcome_rows == 0`` are accepted; any non-TRAIN
  spelling is refused by the shared guards;
* owner decision (b): ``base`` / ``native_base`` items are NEVER overlaid
  (excluded with a receipt; the dataset zeroing stays the only writer of
  their supervision);
* per-item ALIGNMENT assertion before acceptance: the shard snapshot time
  must be ``t0 + 0.5 s`` like every other branch arm -- the shard
  ``meta.json`` must carry the branch origin (``t_origin == t0_s``) and
  the raw run's FIRST physics row must be stamped exactly ``t0 + 0.5``;
  any other observation EXCLUDES the item with a receipt reason (never
  guessed, never shifted);
* label identity: rows are selected at the shard ``decision_t`` times and
  the five taus, then hard-checked against the item
  (``features.outcomes._check_identity``: world/source/chunk/policy/
  t0_s/split) and against the source's ``SHA256SUMS`` entry;
* the receipt records counts per ``(batch, arm)``: labelled / excluded
  with reasons, plus the alignment census.

Loader: :func:`train.label_isolation.load_train_outcomes` returns the
BASE layer unchanged; :func:`load_train_overlay` returns the validated
overlay groups keyed by full item identity, after re-deriving the pack
plan and checking coverage: every zero-row TRAIN item is either in the
overlay, excluded with a reason, or owner-(b) excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: E402

SIDECAR_SCHEMA_V2 = "wm_n1_train_outcomes_sidecar/2"
SIDECAR_SCHEMA_V3 = "wm_n1_train_outcomes_sidecar/3"
RECEIPT_NAME = "TRAIN_ISOLATION_RECEIPT.json"
BASE_PARQUET_NAME = "base_train_outcomes.parquet"
OVERLAY_PARQUET_NAME = "overlay_outcomes.parquet"
SUMS_NAME = "SHA256SUMS"
IDENTITY_COLUMNS = ("batch", "chunk", "world", "arm", "split", "t0_s")
ROW_ID_COLUMN = "_global_row_id"
LABEL_SOURCE_SHAKEY = "label_source_sha256"
OWNER_EXCLUDED_ARMS = ("base", "native_base")
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
SNAPSHOT_OFFSET_S = 0.5            # the branch-arm shard snapshot time
SNAPSHOT_TOL = 1e-6

MIN_FREE_BYTES = 512 * 1024 * 1024
_SHA_RE = "[0-9a-f]{64}"


class LabelOverlayError(PermissionError):
    """Fail-closed: an overlay invariant was violated; nothing was read."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path) -> dict:
    from .label_isolation import _read_json as read
    return read(path)


def _write_json(path, obj) -> None:
    from .label_isolation import _write_json_atomic
    _write_json_atomic(path, obj)


def overlay_identity_key(rec: dict) -> tuple:
    """The full item identity key of a receipt group / pack item."""
    return (str(rec.get("batch") if rec.get("batch") is not None
                else rec["chunk"]),
            str(rec["chunk"]), str(rec["world"]), str(rec["arm"]),
            str(rec["split"]), round(float(rec["t0_s"]), 6))


def _load_sums(root: Path) -> dict:
    path = Path(root) / SUMS_NAME
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0].strip()
    return out


def _free_bytes(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(str(probe)).free


def _refuse_out(out_dir: Path, pack_dir: Path, shards_root: Path,
                input_sidecar: Path) -> None:
    if out_dir.exists():
        raise LabelOverlayError(
            f"refusing to touch existing output directory {out_dir}; "
            "pick a fresh one")
    for ancestor in (out_dir, *out_dir.parents):
        if ancestor.is_symlink():
            raise LabelOverlayError(
                f"output path {out_dir} travels through symlink {ancestor}")
    resolved = out_dir.parent.resolve()
    for name, tree in (("pack", pack_dir), ("shards", shards_root),
                       ("input sidecar", input_sidecar)):
        tree_r = Path(tree).resolve()
        if resolved == tree_r or tree_r in resolved.parents:
            raise LabelOverlayError(
                f"output directory {out_dir} lies inside the {name} tree "
                f"{tree_r}")
    if out_dir.drive.lower() == "d:" and not out_dir.as_posix().lower() \
            .startswith("d:/rd-staging/"):
        raise LabelOverlayError(
            "D: output is allowed only under D:/rd-staging/")
    if _free_bytes(out_dir.parent) < MIN_FREE_BYTES:
        raise LabelOverlayError(
            f"insufficient free space under {out_dir.parent}")


def _first_physics_time(physics: Path) -> float:
    """Time of the FIRST row of a physics_steps.jsonl (rows are written in
    time order; one line is read)."""
    import json as _json
    with open(physics, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            return float(_json.loads(line)["time_s"])
    raise LabelOverlayError(f"{physics}: no rows")


def _conform(frame, columns, dtypes):
    """Lossless cast to the pinned column set/dtypes (mirrors
    train.label_rebind._conform)."""
    import numpy as np

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise LabelOverlayError(f"overlay rows lack pinned columns {missing}")
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
                raise LabelOverlayError(
                    f"lossy cast for column {col!r}: non-integral float")
            frame[col] = series.astype(want)
            continue
        if np.issubdtype(want, np.floating) and np.issubdtype(
                got, np.integer):
            frame[col] = series.astype(want)
            continue
        raise LabelOverlayError(
            f"lossy/unsupported cast for column {col!r}: {got} -> {want}")
    return frame


def build_overlay_sidecar(pack_dir, shards_root, input_sidecar_dir,
                          input_receipt_sha256, input_sidecar_sha256,
                          cov_root, out_dir, code_commit,
                          frozen_root=None) -> dict:
    """Build the schema/3 overlay sidecar; returns the summary report."""
    import pandas as pd

    from features.outcomes import (_check_identity, label_source,
                                   resolve_labels_file)
    from .label_isolation import (_guard_file,
                                  _guard_shard_dir, plan_sidecar)

    pack_dir = Path(pack_dir)
    shards_root = Path(shards_root)
    input_sidecar_dir = Path(input_sidecar_dir)
    cov_root = Path(cov_root)
    out_dir = Path(out_dir)
    frozen_root = Path(frozen_root) if frozen_root else None

    if re.fullmatch(_SHA_RE, str(input_receipt_sha256)) is None \
            or re.fullmatch(_SHA_RE, str(input_sidecar_sha256)) is None:
        raise LabelOverlayError("input receipt/sidecar SHA256 pins required")
    receipt_path = _guard_file(input_sidecar_dir / RECEIPT_NAME,
                               "input sidecar receipt")
    parquet_path = _guard_file(input_sidecar_dir / "train_outcomes.parquet",
                               "input sidecar parquet")
    if _sha256_file(receipt_path) != input_receipt_sha256 \
            or _sha256_file(parquet_path) != input_sidecar_sha256:
        raise LabelOverlayError(
            "input /2 sidecar does not match its pinned SHAs")
    input_receipt = _read_json(receipt_path)
    if input_receipt.get("schema") != SIDECAR_SCHEMA_V2 \
            or input_receipt.get("split") != "TRAIN":
        raise LabelOverlayError("input is not a schema/2 TRAIN sidecar")
    pin_columns = list((input_receipt.get("sidecar") or {})
                       .get("columns") or {})
    if ROW_ID_COLUMN not in pin_columns:
        raise LabelOverlayError("pinned /2 columns lack the row-id column")
    pin_dtypes = dict((input_receipt.get("sidecar") or {}).get("columns")
                      or {})

    _refuse_out(out_dir, pack_dir, shards_root, input_sidecar_dir)
    plan = plan_sidecar(pack_dir, None)
    plan_runs = {r["key"]: r for r in plan["train_runs"]}
    if len(input_receipt.get("runs") or []) != len(plan_runs):
        raise LabelOverlayError(
            "input /2 receipt run count differs from the pack plan")

    sums_frozen = _load_sums(frozen_root) if frozen_root else {}
    sums_cov = _load_sums(cov_root)

    data_cols = [c for c in pin_columns if c != ROW_ID_COLUMN]
    frames = []
    groups = []
    excluded = []
    counts: dict = {}
    align = dict(n_checked=0, n_refused=0)
    zero_total = 0
    for run in plan["train_runs"]:
        if run["n"] != 0:
            continue
        zero_total += 1
        item = run["item"]
        batch = str(item.get("batch") or item["chunk"])
        chunk = str(item["chunk"])
        world = str(item["world"])
        arm = str(item["arm"])
        split = str(item["split"])
        t0_s = float(item["t0_s"])
        policy = item.get("policy")
        bucket = counts.setdefault(f"{batch}|{arm}",
                                   dict(labelled=0, excluded=0,
                                        excluded_reasons={}))
        crec = dict(batch=batch, chunk=chunk, world=world, arm=arm,
                    split=split, t0_s=t0_s, policy=policy, key=run["key"])
        if arm.lower() in OWNER_EXCLUDED_ARMS:
            reason = ("owner decision (b): arm is outcome-supervision-"
                      "excluded (base/native_base)")
            crec["reason"] = reason
            excluded.append(crec)
            bucket["excluded"] += 1
            bucket["excluded_reasons"][reason] = \
                bucket["excluded_reasons"].get(reason, 0) + 1
            continue
        source = label_source(batch, chunk, arm, policy)
        if source is None:
            reason = f"no v0.4a source for batch/arm/policy {batch}/{arm}" \
                     f"/{policy}"
            crec["reason"] = reason
            excluded.append(crec)
            bucket["excluded"] += 1
            bucket["excluded_reasons"].setdefault(reason, 0)
            bucket["excluded_reasons"][reason] += 1
            continue
        label_path = resolve_labels_file(chunk, world, source,
                                         frozen_root=frozen_root,
                                         cov_root=cov_root)
        if label_path is None:
            reason = (f"no label parquet for {world}__{source} in the "
                      "frozen or coverage roots")
            crec.update(reason=reason, resolved_source=source)
            excluded.append(crec)
            bucket["excluded"] += 1
            bucket["excluded_reasons"].setdefault(reason, 0)
            bucket["excluded_reasons"][reason] += 1
            continue
        root_of = None                       # (rel, sums, which)
        for root, sums, which in ((frozen_root, sums_frozen, "frozen"),
                                  (cov_root, sums_cov, "cov")):
            if root is not None:
                try:
                    rel = label_path.resolve().relative_to(
                        Path(root).resolve()).as_posix()
                except ValueError:
                    continue
                root_of = (rel, sums, which)
                break
        if root_of is None:
            reason = f"label parquet {label_path} is outside both roots"
            crec.update(reason=reason, resolved_source=source)
            excluded.append(crec)
            bucket["excluded"] += 1
            bucket["excluded_reasons"].setdefault(reason, 0)
            bucket["excluded_reasons"][reason] += 1
            continue
        rel, sums, which = root_of
        label_sha = _sha256_file(label_path)
        if rel in sums and sums[rel] != label_sha:
            reason = f"label parquet {rel} SHA differs from SHA256SUMS"
            crec.update(reason=reason, resolved_source=source)
            excluded.append(crec)
            bucket["excluded"] += 1
            bucket["excluded_reasons"].setdefault(reason, 0)
            bucket["excluded_reasons"][reason] += 1
            continue

        # ---- alignment: the shard snapshot must sit at t0 + 0.5 s ----
        align["n_checked"] += 1
        try:
            shard = _guard_shard_dir(shards_root, run["shard_rel"],
                                     "TRAIN shard")
            meta = _read_json(_guard_file(shard / "meta.json",
                                          "TRAIN shard meta"))
            if str(meta.get("split", "")).upper() != "TRAIN":
                raise LabelOverlayError("shard meta split is not TRAIN")
            for mk, want in (("world", world), ("arm", arm),
                             ("chunk", chunk),
                             ("batch", batch)):
                if meta.get(mk) is not None and str(meta.get(mk)) != want:
                    raise LabelOverlayError(
                        f"shard meta {mk} {meta.get(mk)!r} != item {want!r}")
            if abs(float(meta.get("t0_s", np.nan)) - t0_s) > 1e-6:
                raise LabelOverlayError(
                    f"shard meta t0_s {meta.get('t0_s')!r} != item {t0_s!r}")
            t_origin = float(meta.get("t_origin", np.nan))
            if abs(t_origin - t0_s) > 1e-6:
                raise LabelOverlayError(
                    f"shard t_origin {t_origin!r} is not the branch "
                    f"convention t0 {t0_s!r}")
            physics = Path(str(meta.get("input_physics")))
            if not physics.is_file():
                raise LabelOverlayError(
                    f"shard raw physics is missing: {physics}")
            t_first = _first_physics_time(physics)
            if abs(t_first - (t0_s + SNAPSHOT_OFFSET_S)) > SNAPSHOT_TOL:
                raise LabelOverlayError(
                    f"shard snapshot time {t_first!r} != t0 + "
                    f"{SNAPSHOT_OFFSET_S} s (misaligned; excluded, "
                    "never shifted)")
            decision_t = [float(t) for t in (meta.get("decision_t") or [])]
            if not decision_t:
                raise LabelOverlayError("shard meta has no decision_t")
        except Exception as exc:              # noqa: BLE001 - exclude, never
            reason = f"alignment: {type(exc).__name__}: {exc}"  # guess
            align["n_refused"] += 1
            crec.update(reason=reason, resolved_source=source,
                        label_source_path=str(label_path))
            excluded.append(crec)
            bucket["excluded"] += 1
            bucket["excluded_reasons"].setdefault(reason, 0)
            bucket["excluded_reasons"][reason] += 1
            continue

        try:
            df = pd.read_parquet(label_path)
            want_t = {round(t, 6) for t in decision_t}
            want_tau = {round(float(x), 6) for x in TAUS}
            keep = df["t"].round(6).isin(want_t) \
                & df["tau"].round(6).isin(want_tau)
            rows = df.loc[keep].copy().sort_values(
                ["t", "tau"]).reset_index(drop=True)
            item_meta = dict(world=world, chunk=chunk, arm=arm, batch=batch,
                             policy=policy, split=split, t0_s=t0_s)
            _check_identity(rows, source, item_meta)
            if len(rows) == 0:
                raise LabelOverlayError(
                    "no label rows at the shard decision times")
            n_t0 = int((rows["t"].round(6)
                        == round(t0_s, 6)).sum())
            if n_t0 == 0:
                raise LabelOverlayError("no label rows at t == t0")
        except Exception as exc:              # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            crec.update(reason=reason, resolved_source=source,
                        label_source_path=str(label_path))
            excluded.append(crec)
            bucket["excluded"] += 1
            bucket["excluded_reasons"].setdefault(reason, 0)
            bucket["excluded_reasons"][reason] += 1
            continue

        frame = _conform(rows, data_cols, pin_dtypes)
        frame["batch"] = batch
        frame["arm"] = arm
        frames.append(frame)
        bucket["labelled"] += 1
        crec.update(resolved_source=source,
                    label_policy=_expected_policy(source),
                    label_source_path=str(label_path),
                    label_source_sha256=label_sha,
                    label_source_root=which,
                    n_rows=int(len(rows)), n_rows_t0=n_t0,
                    n_decision_t=len(decision_t),
                    snapshot_t=t_first)
        groups.append(crec)

    if not frames:
        raise LabelOverlayError(
            "overlay build accepted no items; refusing to write an empty "
            "sidecar (check the exclusion receipt)")
    overlay = pd.concat(frames, ignore_index=True)
    overlay_cols = data_cols + ["batch", "arm"]

    out_dir.mkdir(parents=True)
    base_sha = _sha256_file(parquet_path)
    base_dst = out_dir / BASE_PARQUET_NAME
    shutil.copyfile(parquet_path, base_dst)
    if _sha256_file(base_dst) != base_sha:
        raise LabelOverlayError("base parquet copy failed its SHA check")
    overlay_path = out_dir / OVERLAY_PARQUET_NAME
    overlay = overlay[overlay_cols]
    overlay.to_parquet(overlay_path, index=False)
    overlay_sha = _sha256_file(overlay_path)

    receipt = dict(
        schema=SIDECAR_SCHEMA_V3, created_utc=_utc_now(), split="TRAIN",
        tool="train.label_overlay",
        pack=dict(input_receipt.get("pack") or {}),
        # top-level pack/selection/runs/items MUST mirror the embedded /2
        # receipt byte-for-byte (the loader enforces the equality; the
        # overlay policy is documented in the overlay block only)
        selection=dict(input_receipt.get("selection") or {}),
        input_sidecar=dict(dir=str(input_sidecar_dir),
                           receipt_sha256=input_receipt_sha256,
                           parquet_sha256=input_sidecar_sha256,
                           schema=SIDECAR_SCHEMA_V2),
        input_sidecar_receipt=input_receipt,
        runs=input_receipt.get("runs") or [],
        items=input_receipt.get("items") or [],
        sidecar=dict(base_file=BASE_PARQUET_NAME,
                     base_sha256=base_sha,
                     overlay_file=OVERLAY_PARQUET_NAME,
                     sha256=overlay_sha,
                     n_rows=int(len(overlay)),
                     columns={col: str(dt)
                              for col, dt in overlay.dtypes.items()}),
        overlay=dict(
            code_commit=str(code_commit),
            frozen_root=str(frozen_root) if frozen_root else None,
            cov_root=str(cov_root),
            shards_root=str(shards_root),
            overlay_policy=("zero-row TRAIN items only; owner-(b) arms and "
                            "misaligned shards excluded with reasons; the "
                            "shard snapshot must be t0+0.5 s"),
            n_zero_row_items=zero_total,
            n_accepted_groups=len(groups),
            n_accepted_rows=int(len(overlay)),
            n_accepted_rows_t0=int(sum(g["n_rows_t0"] for g in groups)),
            n_excluded_groups=len(excluded),
            alignment=align,
            counts_by_batch_arm=counts,
            groups=groups,
            excluded=excluded),
    )
    _write_json(out_dir / RECEIPT_NAME, receipt)
    lines = [f"{base_sha}  {BASE_PARQUET_NAME}",
             f"{overlay_sha}  {OVERLAY_PARQUET_NAME}",
             f"{_sha256_file(out_dir / RECEIPT_NAME)}  {RECEIPT_NAME}"]
    (out_dir / SUMS_NAME).write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
    return dict(
        schema=SIDECAR_SCHEMA_V3, out_dir=str(out_dir),
        base_sha256=base_sha, overlay_sha256=overlay_sha,
        n_zero_row_items=zero_total,
        n_accepted_groups=len(groups), n_accepted_rows=int(len(overlay)),
        n_accepted_rows_t0=int(sum(g["n_rows_t0"] for g in groups)),
        n_excluded_groups=len(excluded), alignment=dict(align),
        counts_by_batch_arm=counts)


def _expected_policy(source):
    from features.outcomes import EXPECTED_LABEL_IDENTITY
    return EXPECTED_LABEL_IDENTITY.get(source)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="train.label_overlay")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pack", required=True)
    p.add_argument("--shards-root", required=True)
    p.add_argument("--input-sidecar", required=True)
    p.add_argument("--input-receipt-sha256", required=True)
    p.add_argument("--input-sidecar-sha256", required=True)
    p.add_argument("--cov-root", required=True)
    p.add_argument("--frozen-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--code-commit", required=True)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--pack", required=True)
    p_verify.add_argument("--sidecar", required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "build":
            report = build_overlay_sidecar(
                args.pack, args.shards_root, args.input_sidecar,
                args.input_receipt_sha256, args.input_sidecar_sha256,
                args.cov_root, args.out, args.code_commit,
                frozen_root=args.frozen_root)
            print(json.dumps({k: v for k, v in report.items()
                              if k != "counts_by_batch_arm"}, indent=1))
        else:
            from .label_isolation import load_train_overlay
            overlay, receipt = load_train_overlay(args.pack, args.sidecar)
            print(json.dumps(dict(
                ok=True, schema=receipt.get("schema"),
                n_groups=len(overlay),
                n_rows=int(sum(len(f) for f in overlay.values())),
                n_accepted=(receipt.get("overlay") or {})
                .get("n_accepted_groups")), indent=1))
    except Exception as exc:                  # noqa: BLE001
        print(json.dumps(dict(ok=False, error_type=type(exc).__name__,
                              error=str(exc)), indent=1))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
