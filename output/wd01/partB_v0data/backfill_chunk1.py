"""One-shot migration: backfill chunk/fix-flag columns onto the existing
chunk-1 V0 files and move them into the per-chunk layout.

For every source in the pre-chunk MANIFEST (flat worlds/, files keyed by
bare names):
  - moves the four files into worlds/chunk1/,
  - rewrites labels.parquet and td.parquet adding chunk_id / pre_fix /
    p40 (cav_speed_mode is already a column),
  - rewrites feat.npz adding the same three scalars,
  - rewrites the done marker (out-relative file keys + fresh sha256s +
    flags_backfilled: true),
  - rekeys MANIFEST worlds to "chunk1/<world>__<source>" with the new
    file paths + hashes and creates the chunks.chunk1 section.
Resumable: sources already flagged and hash-verified are skipped.

Usage:  python backfill_chunk1.py [--out DIR] [--chunk chunk1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_v0data import (DEFAULT_CHUNK, DEFAULT_PLAN, DEFAULT_RUNS,  # noqa: E402
                          SCHEMA, atomic_write_json, resolve_speed_mode,
                          sha_of)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def add_flag_columns(df: pd.DataFrame, chunk_id: str) -> pd.DataFrame:
    df = df.copy()
    df["chunk_id"] = chunk_id
    df["pre_fix"] = df["cav_speed_mode"].astype(int) == 31
    df["p40"] = df["penetration"].astype(float).sub(0.4).abs() < 1e-9
    cols = ["chunk_id", "pre_fix", "p40"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    return df


def backfill_source(out: Path, chunk: str, key: str, entry: dict) -> dict:
    old_dir = out / "worlds"
    new_dir = out / "worlds" / chunk
    new_dir.mkdir(parents=True, exist_ok=True)
    names = {suffix: f"{key}_{suffix}" for suffix in
             ("feat.npz", "labels.parquet", "td.parquet")}

    f_lab = new_dir / names["labels.parquet"]
    f_td = new_dir / names["td.parquet"]
    f_npz = new_dir / names["feat.npz"]

    lab = pd.read_parquet(old_dir / names["labels.parquet"])
    td = pd.read_parquet(old_dir / names["td.parquet"])
    mode = int(lab["cav_speed_mode"].iloc[0])
    lab = add_flag_columns(lab, chunk)
    td = add_flag_columns(td, chunk)
    tmp = f_lab.with_name(f_lab.name + ".tmp")
    lab.to_parquet(tmp, index=False)
    os.replace(tmp, f_lab)
    tmp = f_td.with_name(f_td.name + ".tmp")
    td.to_parquet(tmp, index=False)
    os.replace(tmp, f_td)

    z = dict(np.load(old_dir / names["feat.npz"], allow_pickle=False))
    z["chunk_id"] = np.str_(chunk)
    z["pre_fix"] = np.bool_(mode == 31)
    z["p40"] = np.bool_(bool((np.abs(np.asarray(lab["penetration"])
                                     - 0.4) < 1e-9).any()))
    tmp = f_npz.with_name(f_npz.name + ".tmp.npz")
    np.savez_compressed(tmp, **z)
    os.replace(tmp, f_npz)
    for fname in names.values():               # remove the old-location file
        old = old_dir / fname
        if old.exists():
            old.unlink()

    speed_mode_source = "observed" \
        if len(entry.get("trace_stats", {}).get("smode_values", [])) == 1 \
        else "config"
    files = {f"worlds/{chunk}/{n}": sha_of(p)
             for n, p in ((names["feat.npz"], f_npz),
                          (names["labels.parquet"], f_lab),
                          (names["td.parquet"], f_td))}
    new_entry = dict(entry)
    new_entry.update(chunk_id=chunk, cav_speed_mode=mode,
                     speed_mode_source=speed_mode_source, files=files,
                     pre_fix=bool(mode == 31),
                     p40=bool(abs(float(entry.get("penetration", 0.0))
                                  - 0.4) < 1e-9),
                     flags_backfilled=True)
    atomic_write_json(new_dir / f"{key}_done.json",
                      dict(new_entry, backfilled_utc=datetime.now(timezone.utc)
                           .isoformat()))
    return new_entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--chunk", default=DEFAULT_CHUNK)
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    args = ap.parse_args(argv)
    out = Path(args.out)
    chunk = str(args.chunk)
    man_path = out / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if "worlds" not in man:
        log("MANIFEST has no flat worlds section; nothing to backfill")
        return 0
    old_worlds = man.pop("worlds")
    prefix = f"{chunk}/"
    entries = man.setdefault("worlds", {})
    n = 0
    for key0, entry in sorted(old_worlds.items()):
        key = key0[len(prefix):] if key0.startswith(prefix) else key0
        done_new = out / "worlds" / chunk / f"{key}_done.json"
        already = False
        if done_new.exists():
            d = json.loads(done_new.read_text(encoding="utf-8"))
            already = bool(d.get("flags_backfilled")) and all(
                (out / f_).exists() and sha_of(out / f_) == s
                for f_, s in d.get("files", {}).items())
        if already:
            d = json.loads(done_new.read_text(encoding="utf-8"))
            e = dict(entry)
            e.update(chunk_id=chunk, files=d["files"],
                     cav_speed_mode=int(d.get("cav_speed_mode", 31)),
                     speed_mode_source=d.get("speed_mode_source", "resumed"),
                     pre_fix=bool(int(d.get("cav_speed_mode", 31)) == 31),
                     p40=bool(abs(float(entry.get("penetration", 0.0))
                                  - 0.4) < 1e-9),
                     flags_backfilled=True)
            log(f"skip (already backfilled): {key}")
        else:
            e = backfill_source(out, chunk, key, entry)
            n += 1
            log(f"backfilled {key}")
        entries[prefix + key] = e
    man["chunks"] = man.get("chunks", {})
    man["chunks"][chunk] = dict(
        chunk_id=chunk, plan=args.plan, runs_root=args.runs,
        default_speed_mode=31,
        pre_fix_rule="cav_speed_mode == 31", p40_rule="penetration == 0.4",
        world_keys=sorted(k for k in entries if k.startswith(prefix)),
        n_sources=len([k for k in entries if k.startswith(prefix)]),
        skipped_holdout=man.pop("skipped_holdout", []),
        skipped_incomplete=man.pop("skipped_incomplete", []),
        backfilled_utc=datetime.now(timezone.utc).isoformat())
    man["build"] = dict(man.get("build", {}),
                        updated_utc=datetime.now(timezone.utc).isoformat(),
                        n_sources_backfilled_this_run=n,
                        chunks=sorted(man["chunks"]))
    atomic_write_json(man_path, man)
    log(f"MANIFEST migrated: chunk {chunk} with "
        f"{man['chunks'][chunk]['n_sources']} sources ({n} rewritten)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
