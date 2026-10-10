"""OWNER-APPROVED damping-snap outcome overlay (schema/3-style sidecar).

Supplies the 263 re-aligned TRAIN ``base`` / ``native_base`` items' packed
labels from the existing v0.4a coverage rows (``damping.LABELS_COV`` base
layer) keyed by FULL ITEM IDENTITY, exactly like :mod:`dspeed.overlay` but
WITHOUT the owner-(b) arm exclusion (these items ARE the fix).

Fail-closed invariants per item:

* alignment: the re-aligned shard ``meta.json`` has ``t_origin == t0_s`` and
  the first physics row is ``t0 + 0.5 s`` (``ALIGN.json`` ``snapshot_at_t0``);
* the shard's first decision time is the item ``t0`` and the labels carry a
  row stamped exactly at ``t`` == ``t0`` (the whole point of the fix);
* label identity: ``source == arm``, ``chunk_id == chunk``, ``split ==
  TRAIN``, ``t0_s`` matches, and the label ``policy`` equals the frozen
  source registry class (``base -> damping_fleet``, ``native_base ->
  native_fleet``).

Output (FRESH; never overwritten): ``_out/labels_overlay_damping/``.

Usage::

    python -m dspeed.build_damping_overlay
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import common as D  # noqa: E402
from . import damping as DM  # noqa: E402

HERE = Path(__file__).resolve().parent
_WM_N1 = HERE.parent
if str(_WM_N1) not in sys.path:
    sys.path.insert(0, str(_WM_N1))

from features.outcomes import EXPECTED_LABEL_IDENTITY  # noqa: E402

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
SNAP_OFFSET_S = 0.5
SCHEMA = "wm_n1_damping_outcome_overlay/1"
_T0W = time.time()


def _log(msg):
    print(f"[{time.time() - _T0W:7.1f}s] {msg}", flush=True)


def build() -> dict:
    items = DM.load_base_items()
    labels = pd.read_parquet(DM.LABELS_COV / "base_train_outcomes.parquet")
    groups = {}
    for key, frame in labels.groupby(["world", "source", "chunk_id", "t0_s"],
                                     sort=False):
        groups[key] = frame

    frames, accepted, excluded = [], [], []
    for it in items:
        chunk, world, arm = str(it["chunk"]), str(it["world"]), str(it["arm"])
        t0 = float(it["t0_s"])
        rec = dict(batch=chunk, chunk=chunk, world=world, arm=arm,
                   split="TRAIN", t0_s=t0,
                   key=f"{chunk}/{world}__{arm}@{t0:g}")
        try:
            meta = json.loads((DM.SHARDS_ROOT / chunk / world / arm
                               / "meta.json").read_text(encoding="utf-8"))
            align = json.loads((DM.SHARDS_ROOT / chunk / world / arm
                                / "ALIGN.json").read_text(encoding="utf-8"))
            if abs(float(meta.get("t_origin", np.nan)) - t0) > 1e-6:
                raise DM.DampingError("shard t_origin != t0")
            if not align.get("snapshot_at_t0"):
                raise DM.DampingError("ALIGN snapshot_at_t0 not set")
            first = float(align.get("first_physics_time", np.nan))
            if abs(first - (t0 + SNAP_OFFSET_S)) > 1e-6:
                raise DM.DampingError("first physics != t0+0.5")
            decision_t = [float(x) for x in (meta.get("decision_t") or [])]
            if not decision_t or abs(decision_t[0] - t0) > 1e-6:
                raise DM.DampingError("first decision_t != t0")
            frame = groups.get((world, arm, chunk, t0))
            if frame is None:
                raise DM.DampingError(
                    f"no coverage label group {world}__{arm}@{t0:g}")
            want_policy = EXPECTED_LABEL_IDENTITY[arm]
            if len(frame) and not (frame["source"].astype(str) == arm).all():
                raise DM.DampingError("label source != arm")
            if len(frame) and not (frame["chunk_id"].astype(str) == chunk).all():
                raise DM.DampingError("label chunk_id != chunk")
            if len(frame) and not (frame["split"].astype(str) == "TRAIN").all():
                raise DM.DampingError("label split != TRAIN")
            if len(frame) and not (frame["policy"].astype(str)
                                   == want_policy).all():
                raise DM.DampingError("label policy != frozen registry")
            if len(frame) and not np.allclose(
                    frame["t0_s"].astype(float).to_numpy(), t0,
                    atol=1e-6, rtol=0.0):
                raise DM.DampingError("label t0_s mismatch")
            want_t = {round(t, 6) for t in decision_t}
            want_tau = {round(x, 6) for x in TAUS}
            rows = frame.loc[frame["t"].round(6).isin(want_t)
                             & frame["tau"].round(6).isin(want_tau)].copy()
            if len(rows) == 0:
                raise DM.DampingError("no label rows at the shard decision_t")
            t0_rows = rows.loc[rows["t"].round(6) == round(t0, 6)]
            if len(t0_rows) == 0:
                raise DM.DampingError("no label row stamped exactly at t0")
            if t0_rows["tau"].round(6).duplicated().any():
                raise DM.DampingError("duplicate tau at t0 (fail closed)")
        except Exception as exc:                       # noqa: BLE001
            rec["reason"] = f"{type(exc).__name__}: {exc}"
            excluded.append(rec)
            continue
        rows["batch"] = chunk
        rows["arm"] = arm
        frames.append(rows)
        rec.update(n_rows=int(len(rows)), n_rows_t0=int(len(t0_rows)),
                   n_decision_t=len(decision_t), resolved_source=arm,
                   label_source="labels_cov/base_train_outcomes.parquet",
                   snapshot_t=first)
        accepted.append(rec)

    if excluded:
        raise DM.DampingError(
            f"overlay refused {len(excluded)} item(s): {excluded[:3]}")
    if not frames:
        raise DM.DampingError("overlay accepted no items")
    overlay = pd.concat(frames, ignore_index=True)
    out = DM.OVERLAY_ROOT
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    op = out / "overlay_outcomes.parquet"
    overlay.to_parquet(op, index=False)
    ol_sha = D.sha256_file(op)
    receipt = dict(
        schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        split="TRAIN", tool="dspeed.build_damping_overlay",
        identity_columns=["batch", "chunk_id", "world", "arm", "split",
                          "t0_s"],
        pack=dict(dir=str(DM.PACK_ROOT),
                  manifest_sha256=D.sha256_file(DM.PACK_ROOT
                                                / "manifest.json")),
        labels_root=str(DM.LABELS_COV),
        overlay_policy=("damping re-aligned items only; t_origin==t0, first "
                        "physics row t0+0.5, label row at t==t0; identity and "
                        "frozen policy class checked"),
        n_accepted_groups=len(accepted), n_accepted_rows=int(len(overlay)),
        n_accepted_rows_t0=int(sum(g["n_rows_t0"] for g in accepted)),
        groups=accepted)
    D.atomic_write_json(out / "TRAIN_ISOLATION_RECEIPT.json", receipt)
    lines = [f"{ol_sha}  overlay_outcomes.parquet",
             f"{D.sha256_file(out / 'TRAIN_ISOLATION_RECEIPT.json')}  "
             "TRAIN_ISOLATION_RECEIPT.json"]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(root=str(out), n_accepted_groups=len(accepted),
                n_accepted_rows=int(len(overlay)),
                overlay_sha256=ol_sha)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dspeed.build_damping_overlay")
    ap.parse_args(argv)
    print(json.dumps(build(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
