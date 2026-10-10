"""Summarise stored per-state scores: blow-up counts and worst states per model.

No rollouts, no model loads: reads only the stored states__*.json score sheets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diaglib as D  # noqa: E402

MODELS = ["c1_struct", "oldhist", "blackbox_id", "blackbox_traj",
          "pure_physics", "keep_speed"]
OUT = Path(__file__).resolve().parent / "stored_summary.json"


def main() -> int:
    report: dict = {}
    for model in MODELS:
        files = D.model_files(model)
        report[model] = {}
        for name, path in files.items():
            if not path.is_file():
                report[model][name] = {"missing": str(path)}
                continue
            states = D.load_states_file(path)
            # arm selection: PI for unseen, else prefer native then base then first
            arms = sorted({a for s in states for a in s.get("arms", {})})
            arm = "pi" if name in ("u_new", "u_same") and "pi" in arms else (
                "native" if "native" in arms else (arms[0] if arms else "native"))
            rows = []
            nblow = 0
            for s in states:
                b = D.blowup(s, arm)
                if b:
                    nblow += 1
                ds60, p90_60 = D.hdv_ds(s, arm, 60.0)
                ds120, p90_120 = D.hdv_ds(s, arm, 120.0)
                dv60, _ = D.hdv_dv(s, arm, 60.0)
                rows.append({"world": s.get("world"), "t0_s": s.get("t0_s"),
                             "arm": arm, "blow": b,
                             "ds60": ds60, "ds60_p90": p90_60,
                             "ds120": ds120, "ds120_p90": p90_120, "dv60": dv60})
            rows.sort(key=lambda r: -(r["ds120"] if D.finite(r["ds120"]) else -1))
            report[model][name] = {"path": str(path), "n": len(states), "arm": arm,
                                   "arms": arms, "n_blow": nblow,
                                   "blow_frac": (nblow / len(states) if states else None),
                                   "worst": rows[:20]}
            print(f"{model:14s} {name:7s} n={len(states):4d} arm={arm:6s} "
                  f"blow={nblow:3d} ({nblow/len(states)*100 if states else 0:.1f}%)",
                  flush=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
