"""Workspace-only ledger consistency and entry-scoped source verification."""
import importlib.util
import json
import subprocess
from .guard import WORKSPACE, workspace_path

ENTRY = "WALL-C1-PREP-20261009"


def main():
    spec = importlib.util.spec_from_file_location("wallc1_ledger", WORKSPACE / "tools/experiment_history.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    data = tool.load()
    errors, warnings, _, _, _, _ = tool.validate(data)
    original = json.loads(subprocess.check_output(
        ["git", "show", "HEAD:protocols/experiment_history.json"], cwd=WORKSPACE))
    baseline_errors, _, _, _, _, _ = tool.validate(original)
    introduced = sorted(set(errors) - set(baseline_errors))
    sources = [s for s in data.get("source_verification", {}).get("allowed_sources", [])
               if ENTRY in s.get("entries", [])]
    for source in sources:
        workspace_path(tool.resolve(source["path"]))
    if sources:
        scoped = dict(data, source_verification={"allowed_sources": sources})
        _, mismatches, expected, performed = tool.verify_sources(scoped)
    else:
        mismatches, expected, performed = [], 0, 0
    report = {"schema": "wallc1_closeout_check/1", "entry": ENTRY,
              "hard_error_count": len(errors), "baseline_hard_error_count": len(baseline_errors),
              "introduced_hard_errors": introduced, "error_examples": errors[:3],
              "warning_count": len(warnings),
              "scoped_source_verification": {"n_sources": len(sources), "expected": expected,
                  "performed": performed, "mismatches": mismatches,
                  "status": "not_applicable_engineering_no_outcomes" if not sources else "verified"}}
    print(json.dumps(report))
    return int(bool(errors or mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
