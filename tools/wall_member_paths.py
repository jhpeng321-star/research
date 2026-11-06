"""Export recorded wall-member paths that the ``wall_same.json`` state export drops.

``traffic_shepherd.c4.wallregen_states`` records futures for the t0 roster only, so a
wall member that enters after t0 is absent from ``recorded`` even though the source
wall tape contains it.  This tool reads the original wallregen sim directory and
writes a small sidecar with those members' recorded states, keyed by state ``key``:

    python tools/wall_member_paths.py --set E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json \
        --sim E:/agent-work/c4-wallhist/_ref/sim --out _out/wall_members_h60.json

The sidecar is consumed by ``tools/v8wm3_check.py --wall-members insert
--wall-member-paths ...``; the original states file is never modified.  Worlds whose
sim directory is not present are skipped and counted in the receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from traffic_shepherd.c4.wallregen_states import load_world  # noqa: E402
from wms.eval.normalize import initial_frame, iter_windows  # noqa: E402


def member_rows(tl, j0: int, member: str, vid: int, steps: int):
    """Recorded rows of one member from the t0 index; no t0-roster pruning."""
    out = {"id": member, "time_s": [], "s": [], "lane": [], "v": [], "present": []}
    for j in range(j0, min(len(tl.t), j0 + int(steps) + 1)):
        pres = tl.present(j)
        out["time_s"].append(float(tl.t[j]))
        if vid not in pres:
            out["s"].append(0.0)
            out["lane"].append(-1)
            out["v"].append(0.0)
            out["present"].append(False)
            continue
        row = tl.row(j, vid, pres[vid])
        out["s"].append(float(row["s"]))
        out["lane"].append(int(row["lane"]))
        out["v"].append(float(row["v"]))
        out["present"].append(bool(row.get("present", True)))
    return out


def build(set_path: str, sim: str, horizon_s: float) -> dict:
    worlds: dict[str, dict] = {}
    windows: dict[str, dict] = {}
    n_available = n_members = 0
    for index, rec in enumerate(iter_windows(set_path)):
        timeline = rec.get("wall_timeline") or {}
        members = [str(x) for x in (timeline.get("members") or [])]
        if not members:
            continue
        world = str(rec.get("world", ""))
        wdir = Path(sim) / "worlds" / world
        if not wdir.is_dir():
            continue
        if world not in worlds:
            worlds[world] = load_world(wdir)
            n_available += 1
        _, _, timelines = worlds[world]
        arm = str(rec.get("source_arm", ""))
        if arm not in timelines:
            continue
        tl = timelines[arm]
        t0_s = float(initial_frame(rec)["time_s"])
        j0 = tl.index_of(t0_s)
        if j0 is None:
            continue
        present = tl.present(j0)
        t0_ids = {tl.registry[int(vid)] for vid in present}
        missing = [m for m in members if m not in t0_ids]
        if not missing:
            continue
        by_name = {name: i for i, name in enumerate(tl.registry)}
        missing = [m for m in missing if m in by_name]
        if not missing:
            continue
        steps = int(round(float(horizon_s) / 0.5))
        rows = [member_rows(tl, j0, m, by_name[m], steps) for m in missing]
        windows[str(rec.get("key", index))] = {"index": index, "world": world,
                                               "arm": arm, "members": rows}
        n_members += len(rows)
    return {"schema": "wall_member_paths/1", "set": str(set_path), "sim": str(sim),
            "worlds_available": n_available, "windows_with_members": len(windows),
            "member_paths": n_members, "windows": windows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default=(
        "E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json"))
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon-s", type=float, default=120.0)
    a = ap.parse_args()
    payload = build(a.set, a.sim, a.horizon_s)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "worlds_available", "windows_with_members", "member_paths")}, indent=1))


if __name__ == "__main__":
    main()
