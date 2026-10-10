"""Build per-seed All-HDV reward refs for paired-seed training.

For each seed writes fairness_r_ref / flow_q_ref / flow_v_ref using the same
edges and steady window as config/defaults.py comments:
  - discharge: vehicles leaving MERGE_EDGE per step (t in [300, 900])
  - upstream speed: mean speed on PRE_MERGE_EDGE (t in [300, 900])
  - fairness: network mean_time_loss / mean_TT from all_hdv JSON
    (local FCD fairness is heavier; network ratio tracks seed hardness well)

Usage:
  python tools/build_paired_hdv_refs.py --density dense
  python tools/build_paired_hdv_refs.py --density medium
  python tools/build_paired_hdv_refs.py --density dense --seeds talk
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(r"D:\research-dense")
sys.path.insert(0, str(ROOT))

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE  # noqa: E402
from config.defaults import (  # noqa: E402
    FLOW_Q_REF_BY_DENSITY,
    FLOW_V_REF_BY_DENSITY,
)

SUMO = Path(r"D:\research\highway_merge_shepherd")
HDV = ROOT / "runs" / "eval"

# Talk / report seeds held out of the train pool.
TALK_SEEDS = list(range(42, 47))
SELECT_SEEDS = list(range(56, 61))
# 50 train seeds: 47-55 + 61-101 (talk 42-46, select 56-60 held out)
TRAIN_SEEDS = list(range(47, 56)) + list(range(61, 102))
T0, T1 = 300.0, 900.0


def _flow_refs_from_fcd(fcd_path: Path) -> tuple[float, float]:
    """Return (q_ref veh/step, v_ref m/s) over the steady window."""
    if not fcd_path.exists():
        raise FileNotFoundError(fcd_path)

    cur_t = None
    step_discharge = 0
    step_speed_sum = 0.0
    step_speed_n = 0
    prev_road: dict[str, str] = {}
    discharges: list[int] = []
    speed_sums: list[float] = []

    def flush_step(t: float, seen: set[str]) -> None:
        nonlocal step_discharge, step_speed_sum, step_speed_n, prev_road
        for vid, road in list(prev_road.items()):
            if vid not in seen and road == MERGE_EDGE:
                step_discharge += 1
                prev_road.pop(vid, None)
            elif vid not in seen:
                prev_road.pop(vid, None)
        if T0 <= t <= T1:
            discharges.append(step_discharge)
            if step_speed_n > 0:
                speed_sums.append(step_speed_sum / step_speed_n)
        step_discharge = 0
        step_speed_sum = 0.0
        step_speed_n = 0

    context = ET.iterparse(str(fcd_path), events=("start", "end"))
    seen: set[str] = set()
    for event, elem in context:
        if event == "start" and elem.tag == "timestep":
            t = float(elem.attrib["time"])
            if cur_t is not None and t != cur_t:
                flush_step(cur_t, seen)
            cur_t = t
            seen = set()
        elif event == "end" and elem.tag == "vehicle" and cur_t is not None:
            vid = elem.attrib["id"]
            lane = elem.attrib.get("lane", "")
            road = lane.rsplit("_", 1)[0] if lane else ""
            speed = float(elem.attrib.get("speed", 0.0))
            seen.add(vid)
            prev = prev_road.get(vid)
            if prev == MERGE_EDGE and road != MERGE_EDGE:
                step_discharge += 1
            if road:
                prev_road[vid] = road
            if road == PRE_MERGE_EDGE:
                step_speed_sum += speed
                step_speed_n += 1
            elem.clear()
        elif event == "end" and elem.tag == "timestep":
            elem.clear()

    if cur_t is not None:
        flush_step(cur_t, seen)

    if not discharges:
        raise RuntimeError(f"no steady-window samples in {fcd_path}")
    q_ref = sum(discharges) / len(discharges)
    v_ref = (sum(speed_sums) / len(speed_sums)) if speed_sums else 27.0
    return float(q_ref), float(v_ref)


def _resolve_seeds(spec: str) -> list[int]:
    if spec == "train":
        return list(TRAIN_SEEDS)
    if spec == "talk":
        return list(TALK_SEEDS)
    if spec == "all":
        # Prefer whatever All-HDV JSONs exist (1..100 typically).
        found = sorted(
            int(p.stem.split("seed")[-1])
            for p in HDV.glob("all_hdv_*.seed*.json")
            if ".seed" in p.stem
        )
        # Dedup across densities by filtering later; return unique ints.
        return sorted(set(found)) if found else list(range(1, 101))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--density",
        default="dense",
        choices=["light", "medium", "dense", "hyper"],
    )
    ap.add_argument(
        "--seeds",
        default="all",
        help="train | talk | all | comma list (e.g. 42,43,44)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: runs/eval/paired_hdv_refs_{density}.json",
    )
    args = ap.parse_args()

    density = args.density
    out = args.out or (HDV / f"paired_hdv_refs_{density}.json")
    seeds = _resolve_seeds(args.seeds)
    # When --seeds all, keep only seeds that have this density's HDV file.
    if args.seeds == "all":
        seeds = [s for s in range(1, 101) if (HDV / f"all_hdv_{density}.seed{s}.json").exists()]

    q_fallback = float(FLOW_Q_REF_BY_DENSITY.get(density, 0.9))
    v_fallback = float(FLOW_V_REF_BY_DENSITY.get(density, 27.0))

    refs: dict[str, dict] = {}
    for seed in seeds:
        hdv_path = HDV / f"all_hdv_{density}.seed{seed}.json"
        fcd_path = SUMO / f"fcd.{density}.seed{seed}.xml"
        if not hdv_path.exists():
            print(f"WARN skip seed{seed}: missing {hdv_path.name}", flush=True)
            continue
        h = json.loads(hdv_path.read_text(encoding="utf-8"))
        tt = float(h["travel_time"]["all"]["mean"])
        mtl = float(h["mean_time_loss"])
        fairness = mtl / max(tt, 1e-6)
        try:
            q_ref, v_ref = _flow_refs_from_fcd(fcd_path)
        except Exception as exc:
            print(
                f"WARN seed{seed} FCD flow failed ({exc}); "
                f"using density defaults q={q_fallback} v={v_fallback}",
                flush=True,
            )
            q_ref, v_ref = q_fallback, v_fallback
        refs[str(seed)] = {
            "fairness_r_ref": round(fairness, 4),
            "flow_q_ref": round(q_ref, 4),
            "flow_v_ref": round(v_ref, 4),
            "mean_tt": round(tt, 4),
            "mean_time_loss": round(mtl, 4),
        }
        print(
            f"seed{seed}: r_ref={fairness:.3f} q={q_ref:.3f} v={v_ref:.2f} tt={tt:.2f}",
            flush=True,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    all_seeds = sorted(int(k) for k in refs)
    payload = {
        "density": density,
        # Full seed pool for training (talk/select holdouts are optional
        # protocol, not required). Callers that want a holdout can filter.
        "train_seeds": all_seeds,
        "talk_seeds": [s for s in TALK_SEEDS if str(s) in refs],
        "select_seeds": [s for s in SELECT_SEEDS if str(s) in refs],
        "n": len(refs),
        "refs": refs,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} n={len(refs)}", flush=True)
    if args.seeds == "train" and len(refs) < len(TRAIN_SEEDS):
        missing = [s for s in TRAIN_SEEDS if str(s) not in refs]
        print(f"MISSING {len(missing)}: {missing[:20]}...", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
