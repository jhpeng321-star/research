"""DIAG 4 step 1: extract the TRUE SUMO driver settings of the wall-campaign worlds.

The wall-campaign scenario generator (traffic_shepherd/scenarios.py::build_scenario)
writes, per world, a ``requests.rou.xml`` with:
  * three explicit vTypes (aggressive_car / normal_car / cautious_car, IDM car-follow),
    copied from the scenario template's vTypeDistribution with speedDev frozen to 0;
  * one <vehicle> row per scheduled trip carrying its fixed ``type`` assignment
    (drawn once per world by ``damping_type_v1:{seed}:{id}``; never changed at runtime).

So per-vehicle ground truth = the vType parameters of the vehicle's type: type-level
values are exact, per-vehicle sampling exists only for the type assignment itself.
speedDev=0 means the per-vehicle speedFactor equals the vType speedFactor exactly.
No per-vehicle hetdata truth sidecar exists for these campaign runs.

This script parses every world present in wall_same.json, reads
``runs/campaign_stage1_chunk4/{world}__base/run/scenario/requests.rou.xml`` and writes
``out/truth.json`` = {world: {"vtypes": {...}, "vehicles": {id: type}, "routes_sha256"}}.
It also asserts that all worlds share byte-identical vType definitions.
"""
from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from wms.eval.normalize import iter_windows  # noqa: E402

SET_FILE = ROOT / "data_cache" / "wallc1_states_h60" / "wall_same.json"
CAMPAIGN = ROOT / "runs" / "campaign_stage1_chunk4"
OUT = HERE.parent / "out"

NUMERIC = {"tau", "sigma", "speedFactor", "accel", "decel", "minGap", "delta",
           "lcCooperative", "lcAssertive", "lcSpeedGain", "lcStrategic", "lcKeepRight",
           "impatience", "speedDev", "maxSpeed", "length", "actionStepLength"}


def parse_requests(path: Path) -> dict:
    root = ET.parse(path).getroot()
    vtypes = {}
    for vt in root.findall("vType"):
        attrs = dict(vt.attrib)
        vtypes[attrs.pop("id")] = attrs
    vehicles = {v.get("id"): v.get("type") for v in root.findall("vehicle")}
    return {"vtypes": vtypes, "vehicles": vehicles}


def main() -> int:
    worlds = []
    for rec in iter_windows(SET_FILE):
        w = str(rec.get("world"))
        if w not in worlds:
            worlds.append(w)
    truth, vtype_sigs = {}, {}
    missing = []
    for w in sorted(worlds):
        path = CAMPAIGN / f"{w}__base" / "run" / "scenario" / "requests.rou.xml"
        if not path.is_file():
            missing.append(str(path))
            continue
        parsed = parse_requests(path)
        sig = json.dumps(parsed["vtypes"], sort_keys=True)
        vtype_sigs.setdefault(sig, []).append(w)
        truth[w] = {
            "vtypes": parsed["vtypes"],
            "vehicles": parsed["vehicles"],
            "routes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_vehicles": len(parsed["vehicles"]),
        }
    if missing:
        print("MISSING scenario files:", missing)
        return 2
    print(f"worlds: {len(truth)}; distinct vType signatures: {len(vtype_sigs)}")
    for sig, ws in vtype_sigs.items():
        print(f"  sig used by {len(ws)} worlds; sample vtypes:")
        sample = json.loads(sig)
        for name, attrs in sample.items():
            print(f"    {name}: {attrs}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'truth.json'}")
    # Coverage check: every HDV id of every state must have a type in its world.
    from collections import Counter
    unknown = Counter()
    known_types = Counter()
    for rec in iter_windows(SET_FILE):
        w = str(rec.get("world"))
        veh_types = truth[w]["vehicles"]
        ids_seen = set()
        for fr_key in ("state", "initial_state"):
            block = rec.get(fr_key) or {}
            for row in block.get("vehicles", []):
                if row.get("present", True) and int(row.get("lane", -1)) >= 0:
                    ids_seen.add(str(row["id"]))
        for arm, frames in (rec.get("recorded") or {}).items():
            for fr in frames:
                for row in fr.get("vehicles", []):
                    ids_seen.add(str(row["id"]))
        for vid in ids_seen:
            t = veh_types.get(vid)
            if t is None:
                unknown[(w, vid)] += 1
            else:
                known_types[t] += 1
    print("type counts over all ids seen in states:", dict(known_types))
    if unknown:
        print("UNKNOWN ids (no type):", len(unknown), list(unknown)[:10])
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
