"""Audit a saved purported native-speed indicator against active control rows.

Does not change any frozen result or launch SUMO. Equality on active rows is a
negative control: it prevents claiming this indicator alone proves handback.
"""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("runs/probe/native_selectable_damping_v2_check")
    rows = {}
    for name in ["native_release_switching", "learned_pulse", "learned_sharp_brake"]:
        path = root / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        active = {c["step"] for c in data["commands"] if c["operation"] == "slow_down"}
        recorded = [r for r in data["trace"] if r["step"] in active]
        differences = [abs(r["native_speed"] - r["vehicles"][1][1]) for r in recorded]
        rows[name] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "active_halfsteps": len(recorded),
            "max_actual_minus_reported_native_speed_mps": max(differences),
            "all_active_rows_equal": all(d == 0 for d in differences),
            "release_steps": [c["step"] for c in data["commands"] if c["operation"] == "release_to_native"],
        }
    result = {
        "schema": "native_speed_indicator_negative_control_v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "indicator_discriminates_native_from_active_in_saved_data": not all(
            row["all_active_rows_equal"] for row in rows.values()
        ),
        "interpretation": (
            "getSpeedWithoutTraCI equality is not independent evidence of release in these captures: "
            "it also holds during deliberate active slowDown commands. Source/setSpeed(-1), "
            "recorded release/re-engagement, mock API tests and exact always-native identity remain "
            "separate evidence. A controlled same-state release versus persistent-command branch "
            "would be required for a stronger physical handback attribution. No SUMO bug is asserted."
        ),
    }
    output = root / "handback_evidence_qualification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
