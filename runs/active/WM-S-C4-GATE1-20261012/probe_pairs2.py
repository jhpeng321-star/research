"""Gate 1 probe 2: pairing statistics over ALL TRAIN wall groups.

For every Gate-1 arm we materialise the t0 frame (shift 0) and the +0.5 s frame
and compare against the native arm: exact, tolerant (<=1 m, <=1 m/s), and
1-tick-shifted.  Read-only.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"

from wms.eval.states import Identity, Pack, build_state

DATA = Path("E:/research-dense/data_cache")
GATE_ARMS = ("native", "base", "pi", "w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")
TOL_S, TOL_V = 1.0, 1.0


def fmap(frame):
    return {str(v["id"]): (float(v["s"]), int(v["lane"]), float(v["v"]))
            for v in frame["vehicles"] if v.get("present", True)}


def cmp(ref, other, ts=0, tv=0.0):
    if set(ref) != set(other):
        return "idset"
    ds = [abs(ref[i][0] - other[i][0]) for i in ref]
    dv = [abs(ref[i][2] - other[i][2]) for i in ref]
    lane = sum(1 for i in ref if ref[i][1] != other[i][1])
    exact = max(ds) == 0.0 and max(dv) == 0.0 and lane == 0
    tol = max(ds) <= TOL_S + ts and max(dv) <= TOL_V + tv and lane == 0
    return "exact" if exact else ("tol" if tol else "off")


def main():
    packs = {"pack_v3": Pack("pack_v3", DATA / "pack_v3"),
             "pack_damping": Pack("pack_damping", DATA / "pack_damping")}
    ids = {"pack_v3": Identity("pack_v3", DATA / "sidecars_packv3_v2" / "identities"),
           "pack_damping": Identity("pack_damping", DATA / "identities_damping")}
    manifest = json.loads((DATA / "pack_v3" / "manifest.json").read_text(encoding="utf-8"))
    groups = {}
    for item in manifest["items"]:
        if str(item.get("split")) != "TRAIN":
            continue
        groups.setdefault((str(item["world"]), round(float(item["t0_s"]), 3)), set()).add(str(item["arm"]))
    groups = {k: v for k, v in groups.items() if "w-lo-early" in v}
    stats = {arm: Counter() for arm in GATE_ARMS[1:]}
    arms_per_group = Counter()
    for (world, t0) in sorted(groups):
        recs = {}
        for arm in GATE_ARMS:
            pack = "pack_damping" if arm == "base" else "pack_v3"
            window = {"pack": pack, "world": world, "arm": arm, "t0_s": t0,
                      "chunk": None, "key": f"{world}__{arm}", "run": None,
                      "batch": None, "policy": None, "split": "TRAIN",
                      "_extra_baselines": []}
            try:
                recs[arm] = build_state(window, packs, ids, horizon_s=1.0,
                                        extra_baselines=(), history_s=60.0)
            except Exception as exc:
                print(f"  {world} {arm}: ERROR {type(exc).__name__}: {exc}")
        if "native" not in recs:
            continue
        ref0 = fmap(recs["native"]["initial_state"])
        ref1 = fmap(recs["native"]["recorded"]["native"][1])
        matched = ["native"]
        for arm in GATE_ARMS[1:]:
            if arm not in recs:
                stats[arm]["missing"] += 1
                continue
            own = recs[arm]["recorded"][arm]
            c0 = cmp(ref0, fmap(own[0]))
            c1 = cmp(ref0, fmap(own[1]))          # native t0 vs arm t0+0.5 (1-tick late)
            verdict = c0 if c0 != "off" else f"{c1}+1tick"
            stats[arm][verdict] += 1
            if c0 in ("exact", "tol"):
                matched.append(arm)
        arms_per_group[len(matched)] += 1
    print(json.dumps({arm: dict(cnt) for arm, cnt in stats.items()}, indent=1))
    print("matched arms per group (tol):", dict(sorted(arms_per_group.items())))
    print("n groups:", len(groups))


if __name__ == "__main__":
    main()
