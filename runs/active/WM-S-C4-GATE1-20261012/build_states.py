"""Gate 1 state builder: paired multi-arm states from the same saved branch state.

Groups pack_v3 TRAIN items by (world, t0) that carry the wall arms, materialise
the native branch snapshot plus the recorded 60 s future of every Gate-1 arm
(base sourced from the re-aligned pack_damping snap), and keep as *matched* the
arms whose t0 frame equals the native t0 frame within tolerance (same ids, same
lanes, max |ds| <= 1 m, max |dv| <= 1 m/s).  Arms that started from a different
physical state are recorded but excluded from pairing.

Outputs (this directory): ``states_gate1.json`` (large, regenerable, not
committed) and ``groups_gate1.json`` (small pairing summary).

Usage:  python build_states.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RUN))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"

from wms.eval.states import Identity, Pack, build_state

DATA = Path("E:/research-dense/data_cache")
PACKS = {"pack_v3": DATA / "pack_v3", "pack_damping": DATA / "pack_damping"}
IDS = {"pack_v3": DATA / "sidecars_packv3_v2" / "identities",
       "pack_damping": DATA / "identities_damping"}
GATE_ARMS = ("native", "base", "pi", "w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")
V3_ARMS = ("native", "pi", "w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")
TOL_DS_M, TOL_DV_MPS = 1.0, 1.0
HORIZON_S = 60.0
HISTORY_S = 60.0


def forbidden(path: str) -> bool:
    low = str(path).lower()
    return any(tok in low for tok in ("sealed", "holdout", "heldout", "cal_split",
                                      "chunk5", "j4b", "inc1"))


def train_wall_groups() -> dict[tuple[str, float], set[str]]:
    manifest = json.loads((PACKS["pack_v3"] / "manifest.json").read_text(encoding="utf-8"))
    assert not forbidden(str(PACKS["pack_v3"]))
    groups: dict[tuple[str, float], set[str]] = {}
    for item in manifest["items"]:
        if str(item.get("split")) != "TRAIN":
            continue
        if forbidden(str(item.get("chunk", ""))) or forbidden(str(item.get("world", ""))):
            continue
        key = (str(item["world"]), round(float(item["t0_s"]), 3))
        groups.setdefault(key, set()).add(str(item["arm"]))
    return {k: v for k, v in groups.items() if "w-lo-early" in v and "native" in v
            and "pi" in v}


def frame_map(frame: dict) -> dict[str, tuple[float, int, float]]:
    return {str(v["id"]): (float(v["s"]), int(v["lane"]), float(v["v"]))
            for v in frame["vehicles"] if v.get("present", True)}


def pair_status(ref: dict, other: dict) -> dict:
    if set(ref) != set(other):
        return {"matched": False, "reason": "id_set_differs"}
    ds = max((abs(ref[i][0] - other[i][0]) for i in ref), default=0.0)
    dv = max((abs(ref[i][2] - other[i][2]) for i in ref), default=0.0)
    lane = sum(1 for i in ref if ref[i][1] != other[i][1])
    ok = ds <= TOL_DS_M and dv <= TOL_DV_MPS and lane == 0
    return {"matched": ok, "reason": None if ok else "initial_physical_state_mismatch",
            "max_ds_m": round(ds, 4), "max_dv_mps": round(dv, 4), "lane_diff": lane}


def build_group(world: str, t0: float, packs, ids) -> tuple[dict, dict]:
    window = {"pack": "pack_v3", "world": world, "arm": "native", "t0_s": t0,
              "chunk": None, "key": f"{world}__native", "run": None, "batch": None,
              "policy": None, "split": "TRAIN",
              "_extra_baselines": tuple(a for a in V3_ARMS if a != "native")}
    rec = build_state(window, packs, ids, horizon_s=HORIZON_S, history_s=HISTORY_S,
                      extra_baselines=window["_extra_baselines"])
    damp = dict(window, pack="pack_damping", arm="base", key=f"{world}__base",
                _extra_baselines=())
    rec2 = build_state(damp, packs, ids, horizon_s=HORIZON_S, history_s=HISTORY_S)
    rec["recorded"]["base"] = rec2["recorded"]["base"]
    rec["cav_controllers"] = {}

    ref = frame_map(rec["initial_state"])
    arms_status = {}
    for arm in GATE_ARMS:
        frames = rec["recorded"].get(arm)
        if not frames:
            arms_status[arm] = {"matched": False, "reason": "no_recorded_future"}
            continue
        arms_status[arm] = pair_status(ref, frame_map(frames[0]))
        if not arms_status[arm]["matched"]:
            del rec["recorded"][arm]
    rec["paired"] = sum(1 for a in GATE_ARMS if arms_status[a]["matched"]) >= 2
    meta = {"world": world, "t0_s": t0, "pairing": arms_status,
            "matched_arms": [a for a in GATE_ARMS if arms_status[a]["matched"]],
            "n_vehicles_t0": len(ref)}
    return rec, meta


def main() -> int:
    packs = {name: Pack(name, root) for name, root in PACKS.items()}
    ids = {name: Identity(name, root) for name, root in IDS.items()}
    groups = train_wall_groups()
    print(f"{len(groups)} candidate TRAIN groups (world+t0 with native, pi, wall arms)")
    states_path = RUN / "states_gate1.json"
    summary = []
    n_written = 0
    t_start = time.time()
    with states_path.open("w", encoding="utf-8") as fh:
        fh.write('{"schema":"gate1_states/1","horizon_s":60.0,"history_s":60.0,'
                 '"pair_tolerance":{"max_ds_m":1.0,"max_dv_mps":1.0},'
                 '"gate_arms":%s,"windows":[' % json.dumps(list(GATE_ARMS)))
        first = True
        for world, t0 in sorted(groups):
            try:
                rec, meta = build_group(world, t0, packs, ids)
            except Exception as exc:
                meta = {"world": world, "t0_s": t0,
                        "error": f"{type(exc).__name__}: {exc}"}
                rec = None
            summary.append(meta)
            if rec is None or not meta.get("matched_arms"):
                print(f"  skip {world}@{t0}: {meta.get('error') or 'no matched arms'}")
                continue
            if not first:
                fh.write(",")
            fh.write(json.dumps(rec, separators=(",", ":")))
            first = False
            n_written += 1
            print(f"  built {world}@{t0}: matched={meta['matched_arms']} "
                  f"({time.time()-t_start:.0f}s)", flush=True)
        fh.write("]}")
    matched_counts = {}
    for meta in summary:
        for arm in meta.get("matched_arms", []):
            matched_counts[arm] = matched_counts.get(arm, 0) + 1
    payload = {"schema": "gate1_groups/1", "n_groups_requested": len(groups),
               "n_states": n_written, "matched_arm_groups": matched_counts,
               "tolerance": {"max_ds_m": TOL_DS_M, "max_dv_mps": TOL_DV_MPS},
               "groups": summary}
    (RUN / "groups_gate1.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {states_path.name} ({states_path.stat().st_size/1e6:.1f} MB), "
          f"{n_written} states; matched arm groups: {matched_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
