"""Fit world-split CQR on loop TRAIN real pairs rescored with the accepted WM.

Input JSONL is the PROBE_POINTS contract (also accepts rescored pre-probe pairs).
Each row needs K=8 paired_benefit and full-ledger realized_benefit at 60/120/300.
"""
import argparse
import hashlib
import json
from pathlib import Path

from traffic_shepherd.c4.probe import fit_cqr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--wm-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for p in args.pairs for line in p.read_text().splitlines()]
    doc = fit_cqr(rows, args.wm_sha256)
    doc["sources_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.pairs}
    with args.output.open("x") as stream:
        stream.write(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"ready": doc["ready"], "calibration_worlds": len(doc["calibration_worlds"])}))
