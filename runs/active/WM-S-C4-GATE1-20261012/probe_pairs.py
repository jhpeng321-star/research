"""Gate 1 probe: do paired arms share the same physical state at the branch tick?

Materialises the t0 snapshot of every Gate-1 arm for a few TRAIN wall groups and
compares frame0 across arms (exact / tolerant / time-shifted).  Read-only on
E:/research-dense; writes nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"

import json

import numpy as np

from wms.eval.states import Identity, Pack, build_state

DATA = Path("E:/research-dense/data_cache")
PACK_V3 = DATA / "pack_v3"
PACK_DAMP = DATA / "pack_damping"
ID_V3 = DATA / "sidecars_packv3_v2" / "identities"
ID_DAMP = DATA / "identities_damping"
GATE_ARMS = ("native", "base", "pi", "w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")


def train_wall_groups() -> dict[tuple[str, float], set[str]]:
    manifest = json.loads((PACK_V3 / "manifest.json").read_text(encoding="utf-8"))
    groups: dict[tuple[str, float], set[str]] = {}
    for item in manifest["items"]:
        if str(item.get("split")) != "TRAIN":
            continue
        key = (str(item["world"]), round(float(item["t0_s"]), 3))
        groups.setdefault(key, set()).add(str(item["arm"]))
    return {k: v for k, v in groups.items() if "w-lo-early" in v}


def frame_map(frame: dict) -> dict[str, tuple[float, int, float]]:
    return {str(v["id"]): (float(v["s"]), int(v["lane"]), float(v["v"]))
            for v in frame["vehicles"] if v.get("present", True)}


def compare(a: dict, b: dict) -> dict:
    ids_a, ids_b = set(a), set(b)
    common = ids_a & ids_b
    out = {"only_a": len(ids_a - ids_b), "only_b": len(ids_b - ids_a),
           "n_common": len(common)}
    if common:
        ds = max(abs(a[i][0] - b[i][0]) for i in common)
        dv = max(abs(a[i][2] - b[i][2]) for i in common)
        lane = sum(1 for i in common if a[i][1] != b[i][1])
        out.update(max_ds_m=ds, max_dv_mps=dv, lane_diff=lane)
    return out


def main() -> int:
    packs = {"pack_v3": Pack("pack_v3", PACK_V3), "pack_damping": Pack("pack_damping", PACK_DAMP)}
    ids = {"pack_v3": Identity("pack_v3", ID_V3), "pack_damping": Identity("pack_damping", ID_DAMP)}
    groups = train_wall_groups()
    print(f"{len(groups)} TRAIN wall groups")
    for (world, t0) in list(sorted(groups))[:4]:
        print("=" * 72)
        print(world, t0)
        frames = {}
        for arm in GATE_ARMS:
            pack = "pack_damping" if arm == "base" else "pack_v3"
            window = {"pack": pack, "world": world, "arm": arm, "t0_s": t0,
                      "chunk": None, "key": f"{world}__{arm}", "run": None,
                      "batch": None, "policy": None, "split": "TRAIN",
                      "_extra_baselines": []}
            try:
                rec = build_state(window, packs, ids, horizon_s=10.0,
                                  extra_baselines=(), history_s=60.0)
                frames[arm] = frame_map(rec["initial_state"])
                print(f"  {arm:12s} n={len(frames[arm]):3d} "
                      f"hist_steps={rec['history']['history_steps']}")
            except Exception as exc:
                print(f"  {arm:12s} ERROR {type(exc).__name__}: {exc}")
        ref = frames.get("native")
        if ref is None:
            continue
        for arm, fm in frames.items():
            if arm == "native":
                continue
            c = compare(ref, fm)
            print(f"  native vs {arm:12s} -> {json.dumps(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
