"""Verdict C1 producer: unchanged C1 zero-shot regression (PI arm / unseen wall).

Fail-closed by design. The original C1 zero-shot cohorts (unseen-PI and
unseen-wall per-world units) and their scorer belong to the C1 lineage
runtime; this driver refuses unless that runtime, its cohort archive and a
registered two-model (WM2 base, WMFIX candidate) unit plan are staged and
hash-bound. It never fabricates rows and never writes a placeholder PASS;
a missing cohort/runtime is reported as the blocker it is (design: missing
cohort/runtime/safety result blocks adoption).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_COHORT_KEYS = ("schema", "canonical_cohort_sha256", "canonical_worlds",
                        "units", "scorer_files")
REQUIRED_UNIT = ("world", "cohort", "t0_s", "tape_sha256")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(blockers, args, out):
    doc = dict(schema="wmfix_c1_regression/1", reference_label="C1",
               canonical_cohort_sha256=None, canonical_worlds={}, rows=[],
               status="BLOCKED", blockers=blockers,
               candidate_sha256=digest(args.candidate) if Path(args.candidate).is_file() else None,
               reference_sha256=digest(args.reference) if Path(args.reference).is_file() else None)
    doc["binding"] = json.loads(Path(args.binding).read_text())
    doc["producer"] = json.loads(Path(args.producer).read_text())
    if Path(out).exists():
        raise SystemExit("refusing to overwrite existing C1 output")
    with Path(out).open("x") as stream:
        json.dump(doc, stream, allow_nan=False, indent=1)
    print(json.dumps(doc["blockers"], indent=1))
    raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cohort", required=True,
                        help="staged C1 zero-shot cohort receipt (unseen-PI + unseen-wall units)")
    parser.add_argument("--runner", required=True,
                        help="registered C1 lineage runner entry (base+candidate unit execution)")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--producer", required=True)
    args = parser.parse_args()
    blockers = []
    cohort_path, runner_path = Path(args.cohort), Path(args.runner)
    if not cohort_path.is_file():
        blockers.append("C1 cohort receipt missing: stage the canonical unseen-PI/unseen-wall "
                        "cohort (worlds, t0, tape hashes) and bind it here")
    else:
        cohort = json.loads(cohort_path.read_text())
        for key in REQUIRED_COHORT_KEYS:
            if key not in cohort:
                blockers.append("C1 cohort receipt lacks " + key)
        for cohort_name in ("PI", "unseen_wall"):
            worlds = cohort.get("canonical_worlds", {}).get(cohort_name)
            if not worlds:
                blockers.append("C1 cohort lacks worlds for " + cohort_name)
        for unit in cohort.get("units", []):
            if any(k not in unit for k in REQUIRED_UNIT):
                blockers.append("C1 cohort unit incomplete: " + json.dumps(unit)[:120])
        if blockers:
            pass  # detailed blockers already recorded
    if not runner_path.is_file():
        blockers.append("C1 lineage runner registration missing: the original zero-shot "
                        "scorer must be staged for WM2-base and WMFIX-candidate units with "
                        "the full deployed admission path (conserve+learned admission ON)")
    else:
        runner = json.loads(runner_path.read_text())
        for key in ("script", "files", "args"):
            if key not in runner:
                blockers.append("C1 runner registration lacks " + key)
        for path, sha in runner.get("files", {}).items():
            if not Path(path).is_file() or digest(path) != sha:
                blockers.append("C1 runner closure missing/changed: " + str(path))
    if blockers:
        fail(blockers, args, args.out)
    raise SystemExit("C1 execution path not yet implemented in this session; "
                     "the registered runner must be invoked here after review")


if __name__ == "__main__":
    main()
