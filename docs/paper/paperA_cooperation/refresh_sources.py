"""Capture an explicit report allowlist, or detect deliveries since the last cut.

python -B refresh_sources.py --capture   # snapshot existing reports, no inference
python -B refresh_sources.py --check     # read-only changed/missing-file status
This does not poll in the background or automatically promote scientific claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from build_figures import ROOT, guarded

REPORTS = {
    "owner_read_first": "E:/research-dense/docs/OWNER_UPDATE_20261030_EVENING.md",
    "registration": "E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md",
    "wmvalue_design": "E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md",
    "wmvalue_report": "E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md",
    "sumocf_design": "E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md",
    "sumocf_run": "E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md",
    "sumocf_artifact": "E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md",
    "sumocf_root_run": "E:/agent-work/c4-sumocf/SUMOCF_RUN.md",
    "sumocf_canary": "E:/agent-work/c4-sumocf/tools/sumocf/evidence/CANARY_V2.json",
    "audit_design": "E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md",
    "audit_report": "E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md",
    "audit_canary": "E:/agent-work/c4-coopmeasure/tools/coopaudit/evidence/CANARY_VERIFIED.json",
    "audit_exposure_note": "E:/agent-work/c4-coopmeasure/tools/coopaudit/evidence/EXPOSURE_NOTE.md",
    "behavior_report": "E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md",
    "eval_report": "E:/agent-work/c4-coopeval/docs/paper_results/COOPEVAL.md",
    "eval_horizons": "E:/agent-work/c4-coopeval/tools/coopeval/__init__.py",
    "historical_cooperation": "E:/research-dense/docs/paper_results/COOPMEAS2.md",
    "sumoes_report": "E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md",
    "sumoes_log": "E:/agent-work/c4-sumoes/tools/sumoes/RUN.md",
    "pilotdiag_report": "E:/agent-work/c4-pilotdiag/docs/paper_results/PILOTDIAG.md",
    "wmrequal3_report": "E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md",
    "speedhead_report": "E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md",
    "wmfix_retarget": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md",
    "wmfix_v2data": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md",
    "wmcalib_report": "E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md",
    "wmqueue_report": "E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md",
    "c1pidiag_report": "E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md",
    "c1gap_report": "E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md",
    "c1floor_report": "E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR.md",
    "c1switch_report": "E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md",
    "c1closed_report": "E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md",
    "wm_design": "E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md",
    "prior_checklist": "E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md",
    "prior_draft": "E:/research-dense/docs/paper/IEEE_IV_2027_DRAFT.md",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "source_refresh.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    current = {}
    changes = []
    for key, filename in REPORTS.items():
        path = guarded(Path(filename))
        entry = {"path": path.as_posix(), "exists": path.is_file()}
        if entry["exists"]:
            data = path.read_bytes()
            entry.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                         modified_local=datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat())
            snapshot = f"source_snapshots/{key}_{entry['sha256'][:16]}{path.suffix}"
            entry["snapshot"] = snapshot
            if args.capture:
                dest = guarded(ROOT / snapshot, output=True)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    assert dest.read_bytes() == data, dest
                else:
                    with dest.open("xb") as handle:
                        handle.write(data)
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
        old = previous.get("reports", {}).get(key, {})
        if (old.get("exists"), old.get("sha256")) != (entry["exists"], entry.get("sha256")):
            changes.append(key)
        current[key] = entry
    result = {"captured_local": datetime.now().astimezone().isoformat(), "reports": current,
              "changed_since_prior_cut": changes,
              "scope": "explicit local report snapshots only; no simulator/model or restricted input access",
              "claim_review": "manual; file appearance or a canary is not an efficacy result"}
    if args.capture:
        if previous:
            old_bytes = manifest_path.read_bytes()
            hist = guarded(ROOT / ("source_snapshots/refresh_manifest_" +
                                    hashlib.sha256(old_bytes).hexdigest()[:16] + ".json"), output=True)
            if not hist.exists():
                hist.write_bytes(old_bytes)
        manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"captured_local": result["captured_local"], "changed": changes,
                      "missing": [key for key, row in current.items() if not row["exists"]],
                      "mode": "capture" if args.capture else "read-only check"}, indent=2))


if __name__ == "__main__":
    main()
