"""OWNER-APPROVED damping-snap: re-align the TRAIN ``base`` / ``native_base``
items to their decision time.

Owner decision (b) excluded every TRAIN ``base`` / ``native_base`` item from
outcome supervision because the frozen featurizer snapshots a full-run arm
from origin 0 (``features/featurizer.py`` ``base_like -> t_origin = 0.0``),
235-540 s before the item's ``t0`` label.  This module enumerates exactly
those items and resolves their read-only raw campaign runs so
:mod:`dspeed.build_damping` can re-featurize each with the snapshot origin at
``t0`` (first physics row ``t0 + 0.5 s``, like a branch arm).

TRAIN only.  The world set is the pack_v3 TRAIN item list; nothing here opens
CAL / HOLDOUT / chunk-5 / J4B / INC1 data, and every raw run is read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import common as D

#: The two full-run arms whose origin-0 snapshot is mis-aligned with t0.
DAMPING_ARMS = ("base", "native_base")

#: The frozen TRAIN pack the item list is drawn from (read-only).
BASE_PACK = Path("D:/rd-staging/wmn1_feat/pack_v3")

#: Existing v0.4a label coverage rows for the 263 items (read-only).
LABELS_COV = Path("D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z")

#: Read-only chunk-1 penetration-0.4 staging fallback named by the brief.
RAW_CHUNK1_P40 = Path("D:/rd-staging/wmn1_feat/raw_chunk1_p40")

# Workspace-local derived outputs (all under the single approved ``_out/``).
PACK_ROOT = D.OUT / "pack_damping"
SHARDS_ROOT = D.OUT / "shards_damping"
OVERLAY_ROOT = D.OUT / "labels_overlay_damping"
IDENTITIES_ROOT = D.OUT / "identities_damping"
EMPTY_OUTCOMES = D.OUT / "no_outcomes"

#: Owner constraint: stop if E: free would fall below this ceiling.
MIN_FREE_BYTES = 12 * (1 << 30)

CAMPAIGN_ROOT = D.RESEARCH_ROOT / "runs"
_CHUNK_DIR = {
    "chunk1": "campaign_stage1_chunk1",
    "chunk2": "campaign_stage1_chunk2",
    "chunk3": "campaign_stage1_chunk3",
    "chunk4": "campaign_stage1_chunk4",
}

#: arm -> the pack ``policy`` string (drives is_reference / treatment).
ARM_POLICY = {"base": "damping", "native_base": "native"}


class DampingError(D.DspeedError):
    """A damping-snap contract violation (fail closed)."""


def load_base_items(base_pack: Path | str | None = None) -> list:
    """Every TRAIN ``base`` / ``native_base`` item carrying packed outcome
    rows in the frozen pack (the owner-(b) excluded 263).  Identity-checked,
    sorted, de-duplicated."""
    base_pack = Path(BASE_PACK if base_pack is None else base_pack)
    man = json.loads((base_pack / "manifest.json")
                     .read_text(encoding="utf-8"))
    items = []
    seen = set()
    for raw in man.get("items") or []:
        if str(raw.get("split", "")).upper() != "TRAIN":
            continue
        arm = str(raw.get("arm"))
        if arm not in DAMPING_ARMS:
            continue
        if not (raw.get("outcome_rows") or []):
            continue
        key = (str(raw["chunk"]), str(raw["world"]), arm, float(raw["t0_s"]))
        if key in seen:
            raise DampingError(f"duplicate base item identity {key}")
        seen.add(key)
        items.append(dict(raw))
    if not items:
        raise DampingError("no TRAIN base/native_base items in the base pack")
    items.sort(key=lambda it: (str(it["chunk"]), str(it["world"]),
                               str(it["arm"]), float(it["t0_s"])))
    return items


def run_dir_for(item: dict) -> Path:
    """Resolve the read-only raw run directory for one item (fail closed)."""
    chunk = str(item["chunk"])
    if chunk not in _CHUNK_DIR:
        raise DampingError(f"unsupported damping chunk {chunk!r}")
    name = f"{item['world']}__{item['arm']}"
    cands = [CAMPAIGN_ROOT / _CHUNK_DIR[chunk] / name]
    if chunk == "chunk1":
        cands.append(RAW_CHUNK1_P40 / name)
    for cand in cands:
        if cand.is_dir():
            return cand
    raise DampingError(
        f"no raw run directory for {name} (tried "
        f"{', '.join(str(c) for c in cands)})")


def physics_for(run_dir: Path) -> Path:
    run = Path(run_dir) / "run"
    for name in ("physics_steps.jsonl", "physics_steps.jsonl.gz"):
        p = run / name
        if p.is_file():
            return p
    raise DampingError(f"{run_dir}: missing run/physics_steps.jsonl[.gz]")


def validate_run(item: dict, run_dir: Path) -> dict:
    """Hard-check the raw receipt against the pack_v3 identity (fail closed)."""
    p = Path(run_dir) / "RUN_RECEIPT.json"
    if not p.is_file():
        raise DampingError(f"{run_dir}: missing RUN_RECEIPT.json")
    rec = json.loads(p.read_text(encoding="utf-8"))
    want = dict(world=str(item["world"]), arm=str(item["arm"]),
                split="TRAIN", t0_s=float(item["t0_s"]))
    if str(rec.get("world")) != want["world"]:
        raise DampingError(f"{run_dir}: receipt world {rec.get('world')!r} != "
                           f"{want['world']!r}")
    if str(rec.get("arm")) != want["arm"]:
        raise DampingError(f"{run_dir}: receipt arm {rec.get('arm')!r} != "
                           f"{want['arm']!r}")
    if str(rec.get("split", "")).upper() != "TRAIN":
        raise DampingError(f"{run_dir}: receipt split {rec.get('split')!r} "
                           "!= TRAIN")
    if abs(float(rec.get("t0_s", float("nan"))) - want["t0_s"]) > 1e-6:
        raise DampingError(f"{run_dir}: receipt t0 {rec.get('t0_s')!r} != "
                           f"{want['t0_s']!r}")
    if rec.get("exit_status") != "complete":
        raise DampingError(f"{run_dir}: receipt exit_status "
                           f"{rec.get('exit_status')!r} != complete")
    phys = physics_for(run_dir)
    return dict(receipt=rec, physics=phys)


def enumerate_runs(base_pack: Path | str = BASE_PACK) -> list:
    """Item records + resolved raw runs for the damping pack build."""
    runs = []
    for it in load_base_items(base_pack):
        run_dir = run_dir_for(it)
        info = validate_run(it, run_dir)
        runs.append(dict(
            item=it, chunk=str(it["chunk"]), world=str(it["world"]),
            arm=str(it["arm"]), t0_s=float(it["t0_s"]),
            policy=ARM_POLICY[str(it["arm"])], run_dir=run_dir,
            physics=info["physics"], receipt=info["receipt"],
            input_sha256=D.sha256_file(info["physics"])))
    return runs
