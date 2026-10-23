"""R12 registered MID-RUN read at the stage-120 checkpoint (one-step, filled replay presence).  Always exits 0.

Input: --os-dir = output of tools/r10b_checks/run_onestep.py with arms r9 (ROUND 9) and r10b (round-12 stage-120
checkpoint), run with --replay-present filled --base-cols.  Row classes = tools/r12_checks/base_calibration.py
(free_road: lane-1, 5 < v < 20 m/s, leader gap > 60 m or none; recorded_braking: lane-1, rec < -1 m/s^2).

REGISTERED DECISION (signed mean one-step bias = prediction - recorded, NOT magnitudes):
  STOP      if round-12 free-road bias < -0.6 m/s^2 AND < round-9 free-road bias - 0.5 (both under the same filled
            replay)  OR  round-12 braking bias > round-9 braking bias + 0.3;
  The free-road branch is relative as well (S-R12 amendment 2026-10-24 07:45, before launch): the filled replay
  alone shifts ROUND 9's free-road one-step bias from +0.01 to -0.42 m/s^2 (12 TRAIN windows; after-merge -0.04 ->
  -0.66), because junction-transit cars keep their state.  An absolute -0.6 cut would sit next to round 9 itself.
  The stale-temporal signature it targets is about -1.0 relative to round 9 (offline: -0.93 vs +0.11).
  CONTINUE  otherwise;
  UNDECIDED if a class needed for an untriggered condition has < 5 rows (treated as CONTINUE; said so).
Usage: midrun_check.py --os-dir DIR --out-json JSON --out-md MD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import base_calibration as BC  # noqa: E402

TH_FREE = -0.6
TH_FREE_REL = 0.5
TH_BRAKE = 0.3


def decide(stats: dict) -> dict:
    """``stats`` = {arm: base_calibration.class_stats}.  -> {"decision", "reasons", "free_road_bias", ...}."""
    def bias(arm, cls):
        c = stats.get(arm, {}).get(cls, {})
        v = c.get("bias", float("nan"))
        return float(v) if np.isfinite(v) else float("nan")
    fr = bias(BC.R12, "free_road")
    b12, b9 = bias(BC.R12, "recorded_braking"), bias(BC.REF, "recorded_braking")
    reasons, missing = [], []
    fr9 = bias(BC.REF, "free_road")
    if np.isfinite(fr) and np.isfinite(fr9):
        if fr < TH_FREE and fr < fr9 - TH_FREE_REL:
            reasons.append(f"round-12 free-road bias {fr:+.3f} < {TH_FREE} and < round-9 {fr9:+.3f} - {TH_FREE_REL}")
    else:
        missing.append("free_road (round 12 and/or round 9)")
    if np.isfinite(b12) and np.isfinite(b9):
        if b12 > b9 + TH_BRAKE:
            reasons.append(f"round-12 braking bias {b12:+.3f} > round-9 {b9:+.3f} + {TH_BRAKE}")
    else:
        missing.append("recorded_braking (round 12 and/or round 9)")
    decision = "STOP" if reasons else ("UNDECIDED" if missing else "CONTINUE")
    return {"decision": decision, "reasons": reasons, "missing_classes": missing,
            "treated_as": "CONTINUE" if decision == "UNDECIDED" else decision,
            "free_road_bias_round12": fr, "braking_bias_round12": b12, "braking_bias_round9": b9,
            "free_road_bias_round9": bias(BC.REF, "free_road"),
            "n": {arm: {cls: stats.get(arm, {}).get(cls, {}).get("n", 0) for cls in ("free_road", "recorded_braking")}
                  for arm in (BC.REF, BC.R12)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--os-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args(argv)
    out_md = Path(a.out_md)
    try:
        stats = {}
        for arm in (BC.REF, BC.R12):
            d = BC.load_rows(a.os_dir, arm)
            if d is not None:
                stats[arm] = BC.class_stats(d, a.boot)
        res = decide(stats)
        res["classes"] = stats
    except Exception as exc:  # exit 0 always; an error is UNDECIDED (treated as CONTINUE), said so
        res = {"decision": "UNDECIDED", "treated_as": "CONTINUE", "reasons": [], "error": repr(exc)}
    # BASE table for context (base_calibration on the same rows)
    base_md = out_md.with_name(out_md.stem + "_base.md")
    base_json = out_md.with_name(out_md.stem + "_base.json")
    try:
        BC.main(["--os", a.os_dir, "--out", str(base_md), "--json", str(base_json), "--boot", str(a.boot)])
        base_text = base_md.read_text(encoding="utf-8")
    except Exception as exc:
        base_text = f"BASE table unavailable: {exc!r}\n"
    with open(a.out_json, "w", encoding="utf-8") as stream:
        json.dump(res, stream, indent=1, default=float)
    L = ["# R12 mid-run read (stage-120 checkpoint, one-step, filled replay presence)\n",
         f"**Decision: {res['decision']}**" + (" (treated as CONTINUE)" if res["decision"] == "UNDECIDED" else ""),
         "", f"Rule: STOP if round-12 free-road bias < {TH_FREE} AND < round-9 free-road bias - {TH_FREE_REL} OR round-12 braking bias > round-9 braking bias + "
         f"{TH_BRAKE} (signed means, prediction - recorded, m/s^2); UNDECIDED if a class has < 5 rows.", ""]
    if "error" in res:
        L.append(f"Error: {res['error']}")
    else:
        L += [f"- round-12 free-road bias {res['free_road_bias_round12']:+.3f} (round 9 {res['free_road_bias_round9']:+.3f})",
              f"- recorded-braking bias round 12 {res['braking_bias_round12']:+.3f}, round 9 {res['braking_bias_round9']:+.3f}",
              f"- rows: {json.dumps(res['n'])}",
              f"- reasons: {'; '.join(res['reasons']) or 'none'}; missing: {', '.join(res['missing_classes']) or 'none'}"]
    L += ["", "---", "", "Context (base_calibration on the same rows; its rules are for the final checkpoint):", "", base_text]
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(base_text)
    print(f"MIDRUN: {res['decision']}")   # last line (midrun_arm.sh keeps the tail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
