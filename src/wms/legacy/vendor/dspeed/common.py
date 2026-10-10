# ruff: noqa
# VENDORED from output/wd01/wm_n1/dspeed/common.py (last changed in commit 40a9149; copied at baseline bf7d4cb).
# Copied byte-for-byte apart from this header.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""Shared discovery/IO helpers for the WM-N1-DSPEED (A9) pipeline.

Everything here is read-only over the campaign tree and writes only under
the workspace ``_out/`` directory.  The world set comes exclusively from the
campaign ``worlds.json`` (already holdout-guarded and TRAIN-filtered); a run
directory is admitted only with a complete ``run_receipt.json`` whose
``(world, t0_s, arm_index)`` matches its directory name and the world row.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# workspace root (contains this package's repo) and the single approved
# output tree
WORKSPACE = Path(__file__).resolve().parents[4]      # .../dspeed-data
OUT = WORKSPACE / "_out"
LABELS_ROOT = OUT / "labels_v04a_dspeed"
SHARDS_ROOT = OUT / "shards_dspeed"
PACK_ROOT = OUT / "pack_dspeed"
OVERLAY_ROOT = OUT / "labels_overlay_dspeed"

# read-only campaign data (outside the workspace; reads only)
RESEARCH_ROOT = Path("E:/research-dense")
DSD_ROOT = RESEARCH_ROOT / "runs/active/WM-N1-DSPEED-20261004"
WORLDS_JSON = DSD_ROOT / "worlds.json"
SHARDS = ("box", "laptop")

RECEIPT_SCHEMA = "wm_n1_dspeed_trajectory_receipt/1"
ARM_RE = re.compile(r"^(?P<world>.+)__arm(?P<i>\d+)$")
WORLD_RE = re.compile(r"^w(\d+)m(\d+)p(\d+)t(\d+)$")


class DspeedError(RuntimeError):
    """A WM-N1-DSPEED pipeline contract violation (fail closed)."""


def load_worlds() -> dict:
    """``world -> {chunk, t0_s, base_dir, native_dir}`` from worlds.json."""
    raw = json.loads(WORLDS_JSON.read_text(encoding="utf-8"))
    worlds: dict = {}
    for w in raw["worlds"]:
        base_rel = w.get("base_run_rel")
        native_rel = w.get("native_run_rel")
        worlds[str(w["world"])] = dict(
            chunk=str(w["chunk"]),
            t0_s=float(w["t0_s"]),
            base_dir=(RESEARCH_ROOT / base_rel) if base_rel else None,
            native_dir=(RESEARCH_ROOT / native_rel) if native_rel else None,
        )
    if not worlds:
        raise DspeedError(f"no worlds in {WORLDS_JSON}")
    return worlds


def world_metadata(world: str) -> dict:
    """(demand_veh_h, penetration) parsed from the world id (metadata only).

    ``w8481m2700p02t480`` -> demand 2700, penetration 0.02.  Missing pattern
    yields zeros (the label identity checks never consume these).
    """
    m = WORLD_RE.match(world)
    if not m:
        return dict(demand_veh_h=0, penetration=0.0)
    return dict(demand_veh_h=int(m.group(2)),
                penetration=int(m.group(3)) / 100.0)


def read_receipt(run_dir: Path):
    p = Path(run_dir) / "run_receipt.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return None


def enumerate_runs(worlds: dict | None = None) -> list:
    """Every complete trajectory run under ``{box,laptop}/runs``.

    A run is admitted only when its ``run_receipt.json`` is present, parses,
    has ``exit_status == 'complete'``, the receipt schema matches, and its
    recorded ``(world, t0_s, arm_index)`` agree with the directory name and
    the world row.  Returns records sorted by ``(chunk, world, arm_index,
    shard)``.  Re-runnable/idempotent: only the currently present complete
    receipts are returned; adding later runs is one more invocation.
    """
    worlds = load_worlds() if worlds is None else worlds
    runs: list = []
    for shard in SHARDS:
        rdir = DSD_ROOT / shard / "runs"
        if not rdir.is_dir():
            continue
        for run_dir in sorted(p for p in rdir.iterdir() if p.is_dir()):
            m = ARM_RE.match(run_dir.name)
            if not m:
                continue
            world = m.group("world")
            arm_index = int(m.group("i"))
            if world not in worlds:
                raise DspeedError(
                    f"run dir {run_dir} names world {world!r} absent from "
                    "worlds.json; refusing")
            rec = read_receipt(run_dir)
            if rec is None:
                continue
            if rec.get("schema") != RECEIPT_SCHEMA:
                raise DspeedError(
                    f"{run_dir}: receipt schema {rec.get('schema')!r} != "
                    f"{RECEIPT_SCHEMA!r}")
            if rec.get("exit_status") != "complete":
                continue
            row = worlds[world]
            if str(rec.get("world")) != world \
                    or abs(float(rec.get("t0_s", float("nan")))
                           - row["t0_s"]) > 1e-6 \
                    or int(rec.get("arm_index", -1)) != arm_index:
                raise DspeedError(
                    f"{run_dir}: receipt identity "
                    f"(world={rec.get('world')!r}, t0={rec.get('t0_s')!r}, "
                    f"arm={rec.get('arm_index')!r}) disagrees with the "
                    f"directory/world row ({world}, {row['t0_s']}, "
                    f"{arm_index}); refusing")
            gz = run_dir / "physics_steps.jsonl.gz"
            if not gz.is_file():
                raise DspeedError(f"{run_dir}: missing {gz.name}")
            runs.append(dict(
                shard=shard, world=world, chunk=row["chunk"],
                t0_s=row["t0_s"], arm=f"arm{arm_index}",
                arm_index=arm_index, run_dir=run_dir, physics=gz,
                base_dir=row["base_dir"], native_dir=row["native_dir"],
                receipt=rec))
    runs.sort(key=lambda r: (r["chunk"], r["world"], r["arm_index"],
                             r["shard"]))
    return runs


def sha256_file(path, block: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def first_physics_time(physics: Path) -> float:
    """Time of the FIRST physics row (``.gz`` transparent)."""
    import gzip
    opener = gzip.open if str(physics).endswith(".gz") else open
    with opener(physics, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return float(json.loads(line)["time_s"])
    raise DspeedError(f"{physics}: no rows")
