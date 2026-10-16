# ruff: noqa
# VENDORED from output/wd01/wm_n1/dspeed/overlay.py (last changed in commit 40a9149; copied at baseline bf7d4cb).
# Copied byte-for-byte apart from this header.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""A9 step 4a: schema/3-style TRAIN outcome overlay for the dspeed pack.

The ``pack_dspeed`` manifest deliberately carries ZERO packed outcome rows
(the frozen featurizer resolves no v0.4a source for the ``trajectory`` arm),
so its labels are supplied as a ``schema/3-style`` overlay sidecar keyed by
FULL ITEM IDENTITY ``(batch, chunk, world, arm, split, t0_s)`` -- the same
no-repack contract as :mod:`train.label_overlay`, but self-contained for the
dspeed label root.

Fail-closed invariants:

* only zero-row TRAIN items are overlaid; rule-(b) ``base``/``native_base``
  arms are excluded with a reason;
* per-item ALIGNMENT before acceptance: the shard ``meta.json`` must carry
  the branch origin (``t_origin == t0_s``), matching identity, and the raw
  run's FIRST physics row must be exactly ``t0 + 0.5 s`` (``ALIGN.json``);
* label identity: rows must carry ``source == arm``, ``policy ==
  'trajectory'``, ``chunk_id``, ``split == 'TRAIN'`` and ``t0_s``;
* the ``t == t0`` row pick is exact: a duplicate tau at t0 fails closed, and
  the receipt records ``label_t0_rows`` / ``label_t0_violations`` telemetry;
* every zero-row TRAIN item is accepted, excluded with a reason, or rule-(b)
  excluded (coverage).

Output (FRESH; never overwritten): ``overlay_outcomes.parquet``,
``TRAIN_ISOLATION_RECEIPT.json`` (schema
``wm_n1_dspeed_outcome_overlay/1``), ``SHA256SUMS``.

Usage::

    python -m dspeed.overlay
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import common as D  # noqa: E402

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
OWNER_EXCLUDED_ARMS = ("base", "native_base")
SNAP_OFFSET_S = 0.5
SCHEMA = "wm_n1_dspeed_outcome_overlay/1"
_T0W = time.time()


def _log(msg):
    print(f"[{time.time() - _T0W:7.1f}s] {msg}", flush=True)


def _identity_ok(df: pd.DataFrame, it: dict) -> None:
    want = {"source": str(it["arm"]), "policy": "trajectory",
            "chunk_id": str(it["chunk"]), "split": "TRAIN"}
    for col, val in want.items():
        if col not in df.columns:
            raise D.DspeedError(f"label parquet lacks column {col!r}")
        if len(df) and not (df[col].astype(str) == val).all():
            raise D.DspeedError(
                f"label identity mismatch {col!r}: expected {val!r}, got "
                f"{sorted(set(df[col].astype(str)))[:3]}")
    if len(df):
        t0s = df["t0_s"].astype(float).to_numpy()
        if not np.allclose(t0s, float(it["t0_s"]), atol=1e-6, rtol=0.0):
            raise D.DspeedError("label t0_s mismatch")


def _t0_pick(df: pd.DataFrame, t0: float):
    """Rows stamped exactly at t0, keyed by tau; duplicate tau fails closed."""
    sel = df.loc[df["t"].round(6) == round(t0, 6)]
    by_tau = {}
    for _, row in sel.iterrows():
        tau = round(float(row["tau"]), 6)
        if tau in by_tau:
            raise D.DspeedError(
                f"duplicate label row for tau {tau} at t0 {t0} (fail closed)")
        by_tau[tau] = row
    return by_tau


def build(limit_worlds: set | None = None) -> dict:
    man = json.loads((D.PACK_ROOT / "manifest.json")
                     .read_text(encoding="utf-8"))
    items = [it for it in man.get("items", [])
             if str(it.get("split", "")).upper() == "TRAIN"]
    if limit_worlds:
        items = [it for it in items if it["world"] in limit_worlds]

    frames, groups, excluded = [], [], []
    n_zero = 0
    for it in items:
        if list(it.get("outcome_rows") or []):
            continue
        n_zero += 1
        batch = str(it.get("batch") or it["chunk"])
        chunk, world, arm = str(it["chunk"]), str(it["world"]), str(it["arm"])
        t0 = float(it["t0_s"])
        key = (batch, chunk, world, arm, "TRAIN", round(t0, 6))
        rec = dict(batch=batch, chunk=chunk, world=world, arm=arm,
                   split="TRAIN", t0_s=t0, policy=it.get("policy"), key=str(key))
        if arm.lower() in OWNER_EXCLUDED_ARMS:
            rec["reason"] = ("decision (b): arm is outcome-supervision "
                             "excluded")
            excluded.append(rec)
            continue
        lab = D.LABELS_ROOT / "worlds" / chunk / f"{world}__{arm}_labels.parquet"
        if not lab.is_file():
            rec["reason"] = f"no label parquet {lab.name}"
            excluded.append(rec)
            continue
        try:
            meta = json.loads((D.SHARDS_ROOT / chunk / world / arm
                               / "meta.json").read_text(encoding="utf-8"))
            align = json.loads((D.SHARDS_ROOT / chunk / world / arm
                                / "ALIGN.json").read_text(encoding="utf-8"))
            if str(meta.get("world")) != world or str(meta.get("arm")) != arm \
                    or str(meta.get("chunk")) != chunk \
                    or str(meta.get("split")).upper() != "TRAIN":
                raise D.DspeedError("shard meta identity mismatch")
            if abs(float(meta.get("t_origin", np.nan)) - t0) > 1e-6:
                raise D.DspeedError("shard t_origin != t0")
            if abs(float(meta.get("t0_s", np.nan)) - t0) > 1e-6:
                raise D.DspeedError("shard t0_s != item t0")
            first = float(align.get("first_physics_time", np.nan))
            if abs(first - (t0 + SNAP_OFFSET_S)) > 1e-6:
                raise D.DspeedError(
                    f"first physics {first} != t0+0.5 (misaligned)")
            decision_t = [float(x) for x in (meta.get("decision_t") or [])]
            if not decision_t:
                raise D.DspeedError("shard meta has no decision_t")
            df = pd.read_parquet(lab)
            _identity_ok(df, it)
            want_t = {round(t, 6) for t in decision_t}
            keep = df["t"].round(6).isin(want_t) \
                & df["tau"].round(6).isin({round(x, 6) for x in TAUS})
            rows = df.loc[keep].copy().sort_values(
                ["t", "tau"]).reset_index(drop=True)
            if len(rows) == 0:
                raise D.DspeedError("no label rows at the shard decision_t")
            by_tau = _t0_pick(rows, t0)
            if not by_tau:
                raise D.DspeedError("no label rows at t == t0")
        except Exception as exc:                       # noqa: BLE001
            rec["reason"] = f"{type(exc).__name__}: {exc}"
            excluded.append(rec)
            continue
        rows["batch"] = batch
        rows["arm"] = arm
        frames.append(rows)
        rec.update(n_rows=int(len(rows)), n_rows_t0=int(len(by_tau)),
                   n_decision_t=len(decision_t), resolved_source=arm,
                   label_policy="trajectory", label_source_path=str(lab),
                   label_source_sha256=D.sha256_file(lab),
                   label_t0_rows=int(len(by_tau)),
                   label_t0_violations=0, snapshot_t=first)
        groups.append(rec)

    if not frames:
        raise D.DspeedError("overlay accepted no items; refusing empty sidecar")
    overlay = pd.concat(frames, ignore_index=True)
    out = D.OVERLAY_ROOT
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)
    op = out / "overlay_outcomes.parquet"
    overlay.to_parquet(op, index=False)
    ol_sha = D.sha256_file(op)
    receipt = dict(
        schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        split="TRAIN", tool="dspeed.overlay",
        pack=dict(dir=str(D.PACK_ROOT),
                  manifest_sha256=D.sha256_file(D.PACK_ROOT
                                                / "manifest.json")),
        identity_columns=["batch", "chunk", "world", "arm", "split", "t0_s"],
        labels_root=str(D.LABELS_ROOT), shards_root=str(D.SHARDS_ROOT),
        overlay_policy=("zero-row TRAIN items only; rule-(b) arms and "
                        "misaligned shards excluded with reasons; first "
                        "physics row must be t0+0.5 s"),
        n_zero_row_items=n_zero, n_accepted_groups=len(groups),
        n_accepted_rows=int(len(overlay)),
        n_accepted_rows_t0=int(sum(g["n_rows_t0"] for g in groups)),
        n_label_t0_rows=int(sum(g["label_t0_rows"] for g in groups)),
        n_label_t0_violations=int(sum(g["label_t0_violations"]
                                      for g in groups)),
        n_excluded_groups=len(excluded),
        columns={c: str(t) for c, t in overlay.dtypes.items()},
        groups=groups, excluded=excluded)
    D.atomic_write_json(out / "TRAIN_ISOLATION_RECEIPT.json", receipt)
    lines = [f"{ol_sha}  overlay_outcomes.parquet",
             f"{D.sha256_file(out / 'TRAIN_ISOLATION_RECEIPT.json')}  "
             "TRAIN_ISOLATION_RECEIPT.json"]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                   encoding="utf-8")
    return dict(root=str(out), n_accepted_groups=len(groups),
                n_accepted_rows=int(len(overlay)),
                n_excluded_groups=len(excluded),
                n_label_t0_rows=int(sum(g["label_t0_rows"] for g in groups)),
                overlay_sha256=ol_sha)


def load(overlay_root=None):
    """``(groups, receipt)``; groups keyed by full item identity."""
    root = Path(overlay_root) if overlay_root else D.OVERLAY_ROOT
    receipt = json.loads((root / "TRAIN_ISOLATION_RECEIPT.json")
                         .read_text(encoding="utf-8"))
    df = pd.read_parquet(root / "overlay_outcomes.parquet")
    groups = {}
    for _, frame in df.groupby(["batch", "chunk_id", "world", "arm", "split",
                                "t0_s"], sort=False):
        first = frame.iloc[0]
        key = (str(first["batch"]), str(first["chunk_id"]),
               str(first["world"]), str(first["arm"]), str(first["split"]),
               round(float(first["t0_s"]), 6))
        groups[key] = frame.reset_index(drop=True)
    excluded = {(str(g["batch"]), str(g["chunk"]), str(g["world"]),
                 str(g["arm"]), str(g["split"]), round(float(g["t0_s"]), 6))
                for g in receipt.get("excluded") or []}
    return groups, receipt, excluded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dspeed.overlay")
    ap.add_argument("--limit-worlds", default="")
    args = ap.parse_args(argv)
    limit = {w for w in args.limit_worlds.split(",") if w}
    rep = build(limit_worlds=limit or None)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
