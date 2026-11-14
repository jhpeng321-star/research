"""JSON reports under ``preflight/reports/`` + ``PREFLIGHT_SUMMARY.json``."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SUMMARY_FILENAME = "PREFLIGHT_SUMMARY.json"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json.dumps(payload, indent=1).encode("utf-8"))
    os.replace(tmp, path)
    return path


def write_gate_report(reports_dir, result) -> Path:
    payload = dict(result.to_json())
    payload["generated_utc"] = _utc()
    return _write(Path(reports_dir) / f"{result.gate}.json", payload)


def write_summary(reports_dir, results, *, dry: bool) -> Path:
    results = list(results)
    n_pass = sum(1 for r in results if r.passed)
    payload = {
        "schema": "wm_n1_preflight_summary/1",
        "generated_utc": _utc(),
        "dry": bool(dry),
        "n_gates": len(results),
        "n_pass": n_pass,
        "n_fail": len(results) - n_pass,
        "all_passed": n_pass == len(results),
        "gates": {r.gate: {"passed": bool(r.passed),
                           "verdict": "PASS" if r.passed else "FAIL",
                           "detail": r.detail,
                           "entry_points": list(r.entry_points)}
                  for r in results},
    }
    return _write(Path(reports_dir) / SUMMARY_FILENAME, payload)
