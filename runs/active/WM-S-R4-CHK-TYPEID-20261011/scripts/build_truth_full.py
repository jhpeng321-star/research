"""Extend DIAG-ORACLE's truth (29 wall_same worlds) to all worlds of the real-history
set (wallc1_states_h60_real, 34 worlds), same method: parse
runs/campaign_stage1_chunk4/{world}__base/run/scenario/requests.rou.xml.

Writes out/truth_full.json = {world: {"vtypes", "vehicles"}}; asserts the vType
signature of every world equals the DIAG-ORACLE one.  Evaluation-only ground truth.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from wms.eval.normalize import iter_windows  # noqa: E402

DIAG_TRUTH = ROOT / "runs/active/WM-S-R4-DIAG-ORACLE-20261011/out/truth.json"
SETS = [ROOT / "data_cache/wallc1_states_h60/wall_same.json",
        ROOT / "data_cache/wallc1_states_h60_real/wall_same.json"]
CAMPAIGN = ROOT / "runs/campaign_stage1_chunk4"
OUT = HERE.parent / "out/truth_full.json"


def parse_requests(path: Path) -> dict:
    root = ET.parse(path).getroot()
    vtypes = {}
    for vt in root.findall("vType"):
        attrs = dict(vt.attrib)
        vtypes[attrs.pop("id")] = attrs
    vehicles = {v.get("id"): v.get("type") for v in root.findall("vehicle")}
    return {"vtypes": vtypes, "vehicles": vehicles}


def main() -> int:
    diag = json.loads(DIAG_TRUTH.read_text(encoding="utf-8"))
    ref_sig = json.dumps(diag[next(iter(diag))]["vtypes"], sort_keys=True)
    worlds = []
    for setfile in SETS:
        for rec in iter_windows(setfile):
            w = str(rec.get("world"))
            if w not in worlds:
                worlds.append(w)
    truth = {}
    for w in sorted(worlds):
        if w in diag:
            parsed = {"vtypes": diag[w]["vtypes"], "vehicles": diag[w]["vehicles"]}
        else:
            parsed = parse_requests(CAMPAIGN / f"{w}__base/run/scenario/requests.rou.xml")
        sig = json.dumps(parsed["vtypes"], sort_keys=True)
        assert sig == ref_sig, f"{w}: vType signature differs from DIAG-ORACLE"
        truth[w] = parsed
    from collections import Counter
    counts = Counter(t for w in truth for t in truth[w]["vehicles"].values())
    OUT.write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {len(truth)} worlds; type counts {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
