"""Check merged engineering records without opening historical outcome sources."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import experiment_history as history

ROOT = Path(__file__).resolve().parents[1]
IDS = {"WM-S-TRAINSETUP-20261009", "WALL-C1-PREP-20261009"}


def main() -> int:
    current = history.load()
    baseline = json.loads(subprocess.check_output(
        ["git", "show", "ec6f948:protocols/experiment_history.json"], cwd=ROOT))
    before = history.validate(baseline)
    after = history.validate(current)
    new_errors = sorted(set(after[0]) - set(before[0]))
    scoped = dict(current)
    allowed = [row for row in current.get("source_verification", {}).get("allowed_sources", [])
               if IDS.intersection(row.get("entries", []))]
    if allowed:
        # These engineering-only records should never introduce historical
        # outcome opens. An unexpected registration requires explicit review.
        raise ValueError("unexpected outcome source registered for engineering scope")
    scoped["source_verification"] = {"allowed_sources": []}
    report, messages, expected, performed = history.verify_sources(scoped)
    regenerated = history.build_summary_text(current, history.sha256_of(history.LEDGER))
    summary_current = (ROOT / "docs/RESULTS_SUMMARY.md").read_text(encoding="utf-8") == regenerated
    data = {"schema": "wms_main2_record_checks/1", "scope": sorted(IDS),
            "baseline": "ec6f948", "baseline_hard_errors": len(before[0]),
            "current_hard_errors": len(after[0]), "new_hard_errors": new_errors,
            "current_warnings": len(after[1]), "summary_current": summary_current,
            "scoped_verify_sources": {"status": "not_applicable", "registered_sources": 0,
                                      "expected_checks": expected, "performed_checks": performed,
                                      "messages": messages, "report": report},
            "historical_outcome_sources_opened": 0,
            "status": "passed" if not new_errors and summary_current else "failed"}
    out = ROOT / "_out/MAIN2_RECORD_CHECKS.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    print(json.dumps(data, indent=2))
    return 0 if data["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
