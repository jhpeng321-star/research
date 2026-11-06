"""WMQUAL-B job planner: one native prefix per labelled wz3to2 world x cell (task WM-D).

    python wmd_plan.py OUT.jsonl [--t-end 1200] [--root R ...]

Scans the label roots' original natives (``<root>/runs/<cell>/s<seed>/native/RESULT.json``,
status ok) and writes one prefix job per unique (seed, cell): same seed / demand / penetration,
t_end = 1200 s (= latest labelled t0 900 s + the 300 s label window), the original native's save
times as snapshot times, and the original RESULT.json as the identity reference.
Never plans HELD-OUT 989600-989699 or 964000-964199 / 962xxx / 977xxx worlds (refused).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOTS = ["/home/ubuntu/newmap2_20261106", "/home/ubuntu/plangen_20261106",
         "/home/ubuntu/coopkinds20261106", "/home/ubuntu/lowcav_20261106"]
TRAIN = (989200, 989599)


def forbidden(seed: int) -> bool:
    return not (TRAIN[0] <= seed <= TRAIN[1])


def cell_key(d, p):
    return f"D{int(d)}_p{int(round(float(p) * 100)):02d}"


def plan(roots, t_end: float, base: Path):
    jobs, seen = [], {}
    for root in roots:
        for res in sorted(Path(root).glob("runs/*/s*/native/RESULT.json")):
            r = json.loads(res.read_text())
            if r.get("status") != "ok":
                continue
            j = r["job"]
            seed, d, p = int(j["seed"]), int(j["demand"]["main"]), float(j["penetration"])
            if forbidden(seed):
                raise SystemExit(f"refused: seed {seed} outside TRAIN {TRAIN}")
            cell = cell_key(d, p)
            if (seed, cell) in seen:
                continue
            seen[(seed, cell)] = str(res)
            jobs.append(dict(kind="prefix", job_id=f"wmq_{cell}_{seed}", map="wz3to2",
                             map_spec=str(base / "maps" / "wz3to2" / "MAP_SPEC.json"), seed=seed,
                             demand={"main": d}, penetration=p, cell=cell, t_end=float(t_end),
                             snap_times=list(j.get("save_times", [])), ref_native=str(res),
                             label_root=root, out_dir=str(base / "runs" / cell / f"s{seed}" / "prefix")))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--t-end", type=float, default=1200.0)
    ap.add_argument("--root", action="append", default=None)
    args = ap.parse_args()
    jobs = plan(args.root or ROOTS, args.t_end, BASE)
    Path(args.out).write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(len(jobs), "jobs", sum(j["t_end"] for j in jobs), "sim-s ->", args.out)


if __name__ == "__main__":
    main()
