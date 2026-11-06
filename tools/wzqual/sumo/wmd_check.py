"""Identity check of WMQUAL-B prefix tapes against the original labelled natives (no SUMO).

    python wmd_check.py JOBS.jsonl OUT.json

Per job: routes sha equal; per-second J_road / J_src / arrivals equal over the original native's
horizon (max abs diff reported, pass iff <= 1e-6); CAV snapshot rows (fields id, edge, lane, x, v only) and
inserted counts equal at every original save time; collisions/teleports 0.
A tape that fails is flagged ``identical=false`` and excluded from the action-response check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


FIELDS = ("id", "edge", "lane", "x", "v")


def _rows(cavs):
    """Compared CAV snapshot fields only (the COOPKINDS core adds read-only leader fields)."""
    return sorted(tuple(r.get(f) for f in FIELDS) for r in cavs)


def check(job: dict) -> dict:
    out = Path(job["out_dir"]) / "RESULT.json"
    row = dict(job_id=job["job_id"], seed=job["seed"], cell=job["cell"])
    if not out.exists():
        return dict(row, identical=False, reason="missing RESULT.json")
    mine = json.loads(out.read_text())
    if mine.get("status") != "ok":
        return dict(row, identical=False, reason=f"status {mine.get('status')}")
    ref = json.loads(Path(job["ref_native"]).read_text())
    reasons = []
    if mine["routes_sha256"] != ref["routes_sha256"]:
        reasons.append("routes sha differs")
    diffs = {}
    for key in ("J_road", "J_src", "arrivals"):
        a, b = mine[key], ref[key]
        n = len(b)
        if len(a) < n:
            reasons.append(f"{key} shorter than reference")
            continue
        diffs[key] = max((abs(float(x) - float(y)) for x, y in zip(a[:n], b)), default=0.0)
        if diffs[key] > 1e-6:
            reasons.append(f"{key} max diff {diffs[key]:.3g}")
    snap_bad = []
    for tag, s in ref.get("snapshots", {}).items():
        m = mine.get("snapshots", {}).get(tag)
        if m is None:
            snap_bad.append(f"{tag} missing")
            continue
        if _rows(m["cavs"]) != _rows(s["cavs"]) or m.get("inserted") != s.get("inserted"):
            snap_bad.append(tag)
    if snap_bad:
        reasons.append(f"snapshots differ at {snap_bad[:5]}")
    if mine.get("collisions") or mine.get("teleports"):
        reasons.append("collision/teleport in prefix")
    return dict(row, identical=not reasons, reasons=reasons, max_abs_diff=diffs,
                n_snapshots=len(ref.get("snapshots", {})), tape_sha256=mine.get("tape_sha256"),
                ref_native=job["ref_native"], label_root=job.get("label_root"))


def main():
    jobs = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    rows = [check(j) for j in jobs]
    doc = dict(schema="wzqual_prefix_identity/1", n=len(rows), n_identical=sum(r["identical"] for r in rows),
               rows=rows)
    Path(sys.argv[2]).write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(n=doc["n"], n_identical=doc["n_identical"])))


if __name__ == "__main__":
    main()
