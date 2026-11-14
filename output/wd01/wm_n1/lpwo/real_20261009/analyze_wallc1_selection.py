"""Selection-level counts for the LPWO wall test sets (NO pack arrays needed).

Reproduces materialize_wall's window selection (removed train-origin walls in
retained worlds, explicit engaged pilots, corresponding seen controls) using only
the staged metadata + wall timelines, so the test-set sizes are known even though
the pack arrays cannot be staged onto the workspace volume.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lpwo.guard import DataGuard
from lpwo.wall_timeline import read_wall_timeline
from lpwo.materialize_wall import ENGAGED, _rows

W = Path(r"E:/rd-wt/ws-wallc1")
INPUTS = W / "_out" / "wallc1_inputs"
S = W / "_out" / "lpwo_verified_20261009"


def dkey(row):
    return (row.get("pack"), row.get("world"), row.get("arm"), row.get("t0_s"))


def main() -> int:
    guard = DataGuard(str(INPUTS / "canonical_exclusions.json"),
                      str(INPUTS / "canonical_guard.py"), [str(W / "_out")])
    removed = _rows(json.loads((S / "wall_removed_windows.json").read_text(encoding="utf-8")))
    retained_train = _rows(json.loads(
        (S / "windows_train_noPI_noWall.json").read_text(encoding="utf-8")))
    retained_val = _rows(json.loads(
        (S / "windows_val_noPI_noWall.json").read_text(encoding="utf-8")))
    train_worlds = {str(r["world"]) for r in retained_train}

    # wall_same candidates: removed train-origin in retained training worlds
    same = [r for r in removed if r.get("origin_list") == "train"
            and str(r["world"]) in train_worlds]
    same_by_pack = Counter(r["pack"] for r in same)
    same_by_reason = Counter(r["reason"] for r in same)
    same_with_timeline = [r for r in same if r.get("timeline")]
    same_tl_by_reason = Counter(r["reason"] for r in same_with_timeline)

    # explicit engaged pilots from the staged bindings
    bindings = json.loads((W / "_out" / "wallc1_pilot_bindings.json").read_text(encoding="utf-8"))
    index_rows = {str(r["key"]): r for r in json.loads(
        (INPUTS / "pack_v3" / "INDEX.json").read_text(encoding="utf-8"))["runs"]}
    pilot_out = Counter()
    engaged_pilots = []
    for key in bindings:
        if key not in index_rows:
            raise KeyError(key)
        path = Path(bindings[key])
        guard.check_path(path)
        tl = read_wall_timeline(path, guard=guard)
        pilot_out[str(tl.get("status"))] += 1
        if tl.get("engaged") is True and tl.get("status") in ENGAGED:
            engaged_pilots.append(key)
    # pilots that overlap retained training worlds are additionally wall_same
    pilot_same = [k for k in engaged_pilots
                  if str(index_rows[k]["world"]) in train_worlds]

    # corresponding seen controls: retained windows matching any accepted target
    targets = {(str(r["world"]), float(r["t0_s"])) for r in same_with_timeline}
    targets |= {(str(index_rows[k]["world"]), float(index_rows[k]["t0_s"]))
                for k in engaged_pilots}
    controls = [w for w in retained_train + retained_val
                if (str(w["world"]), float(w["t0_s"])) in targets]

    report = {
        "retained_train_windows": len(retained_train),
        "retained_val_windows": len(retained_val),
        "retained_train_worlds": len(train_worlds),
        "wall_same_candidates": len(same),
        "wall_same_candidates_by_pack": dict(same_by_pack),
        "wall_same_candidates_by_reason": dict(same_by_reason),
        "wall_same_with_timeline": len(same_with_timeline),
        "wall_same_with_timeline_by_reason": dict(same_tl_by_reason),
        "wall_pilot_requested": len(bindings),
        "wall_pilot_status_counts": dict(pilot_out),
        "wall_pilot_engaged": len(engaged_pilots),
        "wall_pilot_engaged_overlapping_training_worlds": len(pilot_same),
        "seen_controls_candidates": len(controls),
        "seen_controls_by_pack": dict(Counter(w["pack"] for w in controls)),
        "needed_packs": sorted({r["pack"] for r in same} | {"pack_v3"} |
                               {r["pack"] for r in controls}),
        "guard_refusals": len(guard.refusals),
    }
    (W / "_out" / "WALL_C1_SELECTION.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
