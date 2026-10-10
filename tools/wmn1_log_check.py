"""Standalone stdlib integrity checker for raw IdeaATrainer ``train_log.jsonl``.

Checks ONLY the exact per-step record schema emitted by
``output/wd01/wm_n1/train/idea_a.py`` (``IdeaATrainer._log_step``):
``{"step", "total", "terms", "contrast_support", "s_per_step",
"cuda_mem_bytes", "cuda_peak_bytes"}`` with the five required loss terms
``residual_nll, rollout, arrival, correction_crps, response_contrast``.

The label-time fix (a) adds two optional, all-or-none telemetry fields,
``label_t0_asserted`` (summed per-item t0-aligned outcome label rows on the
batch) and ``label_t0_violations`` (per-item t0 pick violations).  A legacy
log without them stays readable; when present they are validated and the file
FAILS on any ``label_t0_violations > 0`` or a zero ``label_t0_asserted``
total.  The report carries a ``label_t0`` evidence block.

Every-step validation is performed on the raw records; no summary, floor or
repair is ever substituted.  Fail-closed: any malformed, missing, non-finite,
boolean, unsupported zero, duplicated, out-of-order, truncated, extra or
mixed-device record rejects the file.  Negative finite ``residual_nll`` is
legitimate.  A zero ``response_contrast`` is accepted only with complete,
finite, exactly-zero residual proof for every active supported sub-term.

The log carries no literal device string; the actual device is signaled by
``cuda_mem_bytes``/``cuda_peak_bytes`` (``null`` on cpu, non-negative ints on
an active cuda device).  ``--expected-device`` pins that signal for every
record.

The report contains the raw-file SHA-256, per-term first/last/min/max and a
simple least-squares trend slope.  Trends are evidence for the ROOT decision
only: they are NOT an automatic stability threshold and this report is NEVER
an approval.

Stdlib only; no project imports; no data execution.  The log path must be
passed explicitly; there is no default path and no search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REQUIRED_TERMS = (
    "residual_nll",
    "rollout",
    "arrival",
    "correction_crps",
    "response_contrast",
)

RECORD_KEYS = (
    "step",
    "total",
    "terms",
    "contrast_support",
    "s_per_step",
    "cuda_mem_bytes",
    "cuda_peak_bytes",
)

#: (a) label-time-fix telemetry added to the trainer's per-step record:
#: t0-aligned outcome label rows asserted on the batch and any per-item t0
#: pick violations.  Written on CPU and CUDA alike.  These are OPTIONAL as a
#: schema (a legacy log without them stays readable) but all-or-none across a
#: file, and when present every record is validated: any
#: ``label_t0_violations > 0`` or a zero ``label_t0_asserted`` total rejects
#: the file (fail closed).
LABEL_T0_KEYS = ("label_t0_asserted", "label_t0_violations")
ALLOWED_KEYS = RECORD_KEYS + LABEL_T0_KEYS

EXPECTED_STEPS_CHOICES = (40, 50)
EXPECTED_DEVICE_CHOICES = ("cpu", "cuda")
# The registered outcome head is [tau, 3 targets, 19 quantiles].  These are
# source-pinned schema dimensions, not values inferred from a claimed proof.
OUTCOME_TARGET_CHANNELS = 3
OUTCOME_QUANTILES = 19

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_zero_contrast_proof(support, where: str) -> list:
    """Require raw, supported zero residuals for every active contrast term."""
    fail = []
    if not isinstance(support, dict):
        return [f"{where}: zero response_contrast lacks support proof"]
    names = ("pairs", "outcome_pairs", "outcome_taus", "hdv_pairs",
             "hdv_slots")
    counts = {}
    for name in names:
        value = support.get(name)
        if not _is_int(value) or value < 0:
            fail.append(f"{where}: invalid contrast support count {name!r}")
        else:
            counts[name] = value
    for name in ("outcome_subterm", "hdv_subterm"):
        if type(support.get(name)) is not bool:
            fail.append(f"{where}: invalid contrast support flag {name!r}")
    if fail:
        return fail
    if counts["pairs"] < 1:
        fail.append(f"{where}: zero contrast has no sibling pair")
    if (counts["outcome_pairs"] > counts["pairs"]
            or counts["hdv_pairs"] > counts["pairs"]
            or counts["outcome_taus"] < counts["outcome_pairs"]
            or counts["hdv_slots"] < counts["hdv_pairs"]):
        fail.append(f"{where}: inconsistent contrast support counts")
    active = set()
    for term, pair_name, unit_name in (
            ("outcome", "outcome_pairs", "outcome_taus"),
            ("hdv", "hdv_pairs", "hdv_slots")):
        flag = support[f"{term}_subterm"]
        pairs, units = counts[pair_name], counts[unit_name]
        if flag:
            active.add(term)
            if pairs < 1 or units < 1:
                fail.append(f"{where}: active {term} has no support")
        elif pairs != 0 or units != 0:
            fail.append(f"{where}: inactive {term} claims support")
    if not active:
        fail.append(f"{where}: zero contrast has no active supported subterm")
    if "hdv" in active and support.get("hdv_id_aligned") is not True:
        fail.append(f"{where}: zero HDV contrast lacks verified ID alignment")

    checks = support.get("residual_checks")
    if not isinstance(checks, dict) or set(checks) != active:
        fail.append(f"{where}: residual proof does not cover exact active subterms")
        return fail
    expected = {
        "hdv": counts["hdv_slots"],
        "outcome": counts["outcome_taus"] * OUTCOME_TARGET_CHANNELS
                   * OUTCOME_QUANTILES,
    }
    for term in sorted(active):
        proof = checks[term]
        if not isinstance(proof, dict) or set(proof) != {
                "count", "nonzero_count", "max_abs", "finite"}:
            fail.append(f"{where}: malformed {term} residual proof")
            continue
        count, nonzero = proof["count"], proof["nonzero_count"]
        maximum, finite = proof["max_abs"], proof["finite"]
        if not _is_int(count) or count != expected[term] or count < 1:
            fail.append(f"{where}: inconsistent {term} residual count")
        if not _is_int(nonzero) or nonzero != 0:
            fail.append(f"{where}: {term} residuals are nonzero or malformed")
        if (not _is_number(maximum) or not math.isfinite(maximum)
                or maximum != 0.0):
            fail.append(f"{where}: {term} residual maximum is not finite zero")
        if finite is not True:
            fail.append(f"{where}: {term} residuals are not proven finite")
    return fail


def parse_log_text(text: str) -> tuple[list, list]:
    """Parse raw log text into (records, failures); fail-closed on bad lines.

    Only a single trailing newline after the last record is tolerated; a
    blank line anywhere else is a malformed record.
    """
    records: list = []
    failures: list = []
    lines = text.splitlines()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            failures.append(f"line {lineno}: blank/malformed record")
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(f"line {lineno}: not valid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            failures.append(f"line {lineno}: record is not a JSON object")
            continue
        records.append(record)
    return records, failures


def check_record(record: dict, position: int, expected_device: str) -> list:
    """Validate one raw record at 1-based ``position``; return failures."""
    failures: list = []
    where = f"step position {position}"
    for key in RECORD_KEYS:
        if key not in record:
            failures.append(f"{where}: missing key {key!r}")
    if failures:
        return failures
    extra = sorted(set(record) - set(ALLOWED_KEYS))
    if extra:
        failures.append(f"{where}: unexpected extra keys {extra}")
    step = record["step"]
    if not _is_int(step):
        failures.append(f"{where}: step {step!r} is not a plain integer")
    elif step != position:
        failures.append(
            f"{where}: step {step!r} breaks exact 1..N order (expected "
            f"{position})")
    if not isinstance(record["terms"], dict):
        failures.append(f"{where}: 'terms' is not an object")
        terms = {}
    else:
        terms = record["terms"]
    for term in REQUIRED_TERMS:
        if term not in terms:
            failures.append(f"{where}: missing loss term {term!r}")
            continue
        value = terms[term]
        if isinstance(value, bool) or not _is_number(value):
            failures.append(
                f"{where}: term {term!r} value {value!r} is not a number "
                "(booleans rejected)")
            continue
        if not math.isfinite(value):
            failures.append(f"{where}: term {term!r} is NaN/Infinity")
            continue
        if value == 0.0:
            if term == "response_contrast":
                failures.extend(_check_zero_contrast_proof(
                    record["contrast_support"], where))
            else:
                failures.append(f"{where}: term {term!r} is zero")
            continue
        if term == "response_contrast" and value < 0.0:
            failures.append(
                f"{where}: response_contrast {value!r} is not strictly "
                "positive")
    total = record["total"]
    if isinstance(total, bool) or not _is_number(total):
        failures.append(
            f"{where}: 'total' {total!r} is not a finite number")
    elif not math.isfinite(total):
        failures.append(f"{where}: 'total' is NaN/Infinity")
    s_per_step = record["s_per_step"]
    if isinstance(s_per_step, bool) or not _is_number(s_per_step):
        failures.append(
            f"{where}: 's_per_step' {s_per_step!r} is not a finite number")
    elif not math.isfinite(s_per_step):
        failures.append(f"{where}: 's_per_step' is NaN/Infinity")
    elif s_per_step < 0.0:
        failures.append(f"{where}: 's_per_step' is negative")
    support = record["contrast_support"]
    if support is not None and not isinstance(support, dict):
        failures.append(
            f"{where}: 'contrast_support' must be an object or null")
    mem = record["cuda_mem_bytes"]
    peak = record["cuda_peak_bytes"]
    if expected_device == "cpu":
        if mem is not None or peak is not None:
            failures.append(
                f"{where}: expected cpu but cuda memory fields are "
                f"mem={mem!r} peak={peak!r} (mixed/actual device)")
    else:
        if not _is_int(mem) or not _is_int(peak):
            failures.append(
                f"{where}: expected cuda but cuda memory fields are "
                f"mem={mem!r} peak={peak!r} (must be non-null integers)")
        elif mem < 0 or peak < 0:
            failures.append(f"{where}: negative cuda memory value")
    for key in LABEL_T0_KEYS:
        if key in record:
            value = record[key]
            if not _is_int(value) or value < 0:
                failures.append(
                    f"{where}: {key} {value!r} is not a non-negative plain "
                    "integer")
    violations = record.get("label_t0_violations")
    if _is_int(violations) and violations > 0:
        failures.append(
            f"{where}: label_t0_violations {violations} > 0 (a per-item t0 "
            "label assertion failed)")
    return failures


def check_log(
    records: list,
    parse_failures: list,
    expected_steps: int,
    expected_device: str,
) -> list:
    """Validate the raw record sequence; return the full failure list."""
    failures = list(parse_failures)
    if not records:
        failures.append("no records: log is empty")
    if len(records) != expected_steps:
        failures.append(
            f"expected exactly {expected_steps} records, found "
            f"{len(records)} (truncated/extra)")
    for position, record in enumerate(records, start=1):
        failures.extend(
            check_record(record, position, expected_device))
    if records:
        per_record = [all(key in record for key in LABEL_T0_KEYS)
                      for record in records]
        if any(per_record) and not all(per_record):
            failures.append(
                "label_t0 fields present in only some records (inconsistent "
                "schema; all-or-none required)")
        elif all(per_record):
            asserted = sum(int(record["label_t0_asserted"])
                           for record in records
                           if _is_int(record["label_t0_asserted"]))
            if asserted == 0:
                failures.append(
                    "label_t0_asserted totals 0 across all records (no "
                    "t0-aligned label was asserted)")
    return failures


def summarize_term(values: list) -> dict:
    """first/last/min/max/mean plus a simple least-squares trend slope."""
    steps = list(range(1, len(values) + 1))
    n = len(values)
    mean = sum(values) / n
    mean_x = (n + 1) / 2.0
    denom = sum((x - mean_x) ** 2 for x in steps)
    if denom == 0.0:
        slope = 0.0
    else:
        slope = sum(
            (x - mean_x) * (y - mean) for x, y in zip(steps, values)) / denom
    if slope > 0.0:
        trend = "increasing"
    elif slope < 0.0:
        trend = "decreasing"
    else:
        trend = "flat"
    return {
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "slope_per_step": slope,
        "trend": trend,
        "n_negative": sum(1 for v in values if v < 0.0),
    }


def summarize(records: list) -> dict:
    """Per-term evidence summary over finite numeric term values only."""
    summary: dict = {}
    for term in REQUIRED_TERMS:
        values = []
        for record in records:
            terms = record.get("terms")
            if not isinstance(terms, dict):
                continue
            value = terms.get(term)
            if _is_number(value) and math.isfinite(value):
                values.append(float(value))
        if values:
            summary[term] = summarize_term(values)
        else:
            summary[term] = None
    return summary


def summarize_label_t0(records: list) -> dict:
    """Report the (a) label_t0 telemetry.

    ``present`` is False for a legacy log without the fields (then the totals
    are 0 and no failure is raised on absence).  For the current trainer
    schema every record carries the fields, so ``asserted_total`` /
    ``violations_total`` are the raw evidence the checker validates.
    """
    asserted = [int(record["label_t0_asserted"]) for record in records
                if _is_int(record.get("label_t0_asserted"))]
    violations = [int(record["label_t0_violations"]) for record in records
                  if _is_int(record.get("label_t0_violations"))]
    present = bool(records) and len(asserted) == len(records)
    return {
        "present": present,
        "asserted_total": sum(asserted),
        "violations_total": sum(violations),
        "steps_with_assertions": sum(1 for value in asserted if value > 0),
    }


def sha256_file(path: Path) -> tuple[str, int]:
    """Stream the raw file bytes; return (hex digest, byte count)."""
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 16)
            if not chunk:
                break
            digest.update(chunk)
            count += len(chunk)
    return digest.hexdigest(), count


def check_file(path: Path, expected_steps: int, expected_device: str) -> dict:
    """Full raw-file check; returns the report dict (no approval semantics)."""
    failures: list = []
    sha256 = None
    nbytes = None
    records: list = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        failures.append(f"unreadable log: {exc}")
        text = None
    if text is not None:
        records, failures = parse_log_text(text)
        failures = check_log(
            records, failures, expected_steps, expected_device)
        try:
            sha256, nbytes = sha256_file(path)
        except OSError as exc:
            failures.append(f"unreadable for hashing: {exc}")
    integrity = "pass" if not failures else "fail"
    return {
        "tool": "wmn1_log_check",
        "integrity": integrity,
        "note": (
            "format integrity evidence only; trends are evidence for the "
            "ROOT decision, not an automatic stability threshold, and this "
            "report is not an approval"),
        "mode": {
            "expected_steps": expected_steps,
            "expected_device": expected_device,
        },
        "log": {
            "path": str(path),
            "sha256": sha256,
            "bytes": nbytes,
            "records": len(records),
        },
        "failures": failures,
        "terms": summarize(records),
        "label_t0": summarize_label_t0(records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone stdlib integrity checker for a raw IdeaATrainer "
            "train_log.jsonl (explicit file only; fail-closed; no repair; "
            "not an approval)"))
    parser.add_argument(
        "--log", required=True,
        help="explicit path to the raw train_log.jsonl (no defaults, no "
             "search)")
    parser.add_argument(
        "--expected-steps", required=True, type=int,
        choices=EXPECTED_STEPS_CHOICES,
        help="exact raw step count required (40 cpu laptop smoke or 50 cuda)")
    parser.add_argument(
        "--expected-device", required=True,
        choices=EXPECTED_DEVICE_CHOICES,
        help="actual device required in every record (via the cuda memory "
             "fields of the exact schema)")
    parser.add_argument(
        "--out", default=None,
        help="exclusive JSON report path; refuses to overwrite; report goes "
             "to stdout when omitted")
    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"wmn1_log_check: log not found: {log_path}", file=sys.stderr)
        return EXIT_USAGE
    if args.out is not None and Path(args.out).exists():
        print(
            f"wmn1_log_check: --out refuses to overwrite existing file: "
            f"{args.out}",
            file=sys.stderr)
        return EXIT_USAGE
    report = check_file(log_path, args.expected_steps, args.expected_device)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.out is None:
        print(payload)
    else:
        with Path(args.out).open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return EXIT_PASS if report["integrity"] == "pass" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
