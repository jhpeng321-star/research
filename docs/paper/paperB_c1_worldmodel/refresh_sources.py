"""Inventory only the task-named evidence files; never starts agent/run processes.

Writes a timestamped source-status receipt. Newly present/changed reports need
semantic review before any claim promotion. Does not alter claims or artwork.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "C1 gap authoritative report": "E:/agent-work/c4-c1gap/C1_GAPS.md",
    "C1 gap report alternate documented delivery": "E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md",
    "C1 closed-loop authoritative report": "E:/agent-work/c4-c1closed/C1CLOSED.md",
    "C1 closed-loop report alternate delivery": "E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md",
    "Recovered wall result": "E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md",
    "Wall recovery provenance": "E:/agent-work/c4-c1gap/docs/paper_results/c1gap/recovery.json",
    "Main claim checklist": "E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md",
    "Main draft": "E:/research-dense/docs/paper/IEEE_IV_2027_DRAFT.md",
    "Model change log": "E:/research-dense/docs/WM_CHANGE_LOG.md",
    "Model state plan": "E:/research-dense/docs/WM_STATE_PLAN.md",
    "E1 selection Markdown": "E:/research-dense/docs/paper_results/E1_C1_fair.md",
    "E1 selection CSV": "E:/research-dense/docs/paper_results/E1_C1_fair.csv",
    "E1 direction": "E:/research-dense/docs/paper_results/E1_WM_direction.md",
    "E1 fidelity": "E:/research-dense/docs/paper_results/E1_WM_fidelity.md",
    "E2 direction Markdown": "E:/research-dense/docs/paper_results/E2_direction_horizon.md",
    "E2 direction CSV": "E:/research-dense/docs/paper_results/E2_direction_horizon.csv",
    "E2 delay": "E:/research-dense/docs/paper_results/E2_delay_horizon.md",
    "Results scope README": "E:/research-dense/docs/paper_results/README.md",
    "Corrected native-lane qualification": "E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md",
    "Prior-attempt audit": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PRIOR.md",
    "Entry retarget": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md",
    "Failed pilot": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_FAIL.md",
    "Failed smoothing feasibility": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_V2_FEAS.md",
    "Data-only successor status": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md",
    "Final calibration": "E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md",
    "Calibration handoff": "E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB_HANDOFF.md",
    "Calibration resume": "E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB_RESUME.md",
    "Speed-only headroom": "E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md",
}


def main() -> None:
    prior_path = ROOT / "SOURCE_REFRESH.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    previous = prior.get("sources", {})
    manifest = json.loads((ROOT / "data/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    snapshots = {r["source_file"]: r["sha256"] for r in manifest.values()}
    records = {}
    for label, filename in SOURCES.items():
        path = Path(filename).resolve()
        if any(s in str(path).lower() for s in ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")):
            raise ValueError(f"Excluded path: {path}")
        if not path.is_file():
            records[label] = {"path": filename, "status": "MISSING"}
            continue
        blob = path.read_bytes()
        text = blob.decode("utf-8-sig")
        sha = hashlib.sha256(blob).hexdigest()
        records[label] = {"path": filename, "status": "PRESENT", "sha256": sha,
                          "bytes":len(blob), "heading":next((s for s in text.splitlines() if s.startswith("#")), ""),
                          "changed_since_previous_refresh": previous.get(label,{}).get("sha256") not in (None,sha),
                          "matches_figure_snapshot": sha == snapshots[filename] if filename in snapshots else None}
    receipt = {"checked_at":datetime.now().astimezone().isoformat(timespec="seconds"),
               "scope":"Explicit permitted evidence paths only; no automatic scientific status promotion", "sources":records}
    prior_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"checked_at":receipt["checked_at"],
                      "missing":[k for k,v in records.items() if v["status"]=="MISSING"],
                      "changed":[k for k,v in records.items() if v.get("changed_since_previous_refresh")],
                      "figure_source_changes":[k for k,v in records.items() if v.get("matches_figure_snapshot") is False]},indent=2))


if __name__ == "__main__":
    main()
