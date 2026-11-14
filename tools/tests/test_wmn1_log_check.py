"""Draft synthetic tests for tools/wmn1_log_check.py (DRAFT — NOT RUN).

The authoring agent did not execute these tests (CODE ONLY constraint);
root will run them later under the accepted guard.  All fixtures are
synthetic records written in the exact ``IdeaATrainer._log_step`` schema
(``output/wd01/wm_n1/train/idea_a.py``): no real run log, pack or project
module is read.  Only the checker module and the stdlib are imported.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "wmn1_log_check.py"
_SPEC = importlib.util.spec_from_file_location("wmn1_log_check", _MODULE_PATH)
wmn1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wmn1)

CPU40 = dict(expected_steps=40, expected_device="cpu")
CUDA50 = dict(expected_steps=50, expected_device="cuda")


def _terms(step: int) -> dict:
    terms = {name: 1.0 + step / 10.0 for name in wmn1.REQUIRED_TERMS}
    terms["residual_nll"] = -0.5 - step / 100.0
    return terms


def _record(step: int, *, device: str = "cpu", terms: dict | None = None,
            total=None, s_per_step: float = 0.5, extra: dict | None = None):
    record = {
        "step": step,
        "total": float(sum((terms or _terms(step)).values()))
        if total is None else total,
        "terms": terms if terms is not None else _terms(step),
        "contrast_support": None,
        "s_per_step": s_per_step,
        "cuda_mem_bytes": 1024 if device == "cuda" else None,
        "cuda_peak_bytes": 2048 if device == "cuda" else None,
    }
    if extra:
        record.update(extra)
    return record


def _write_log(tmp_path: Path, lines, name: str = "train_log.jsonl") -> Path:
    path = tmp_path / name
    if isinstance(lines, str):
        path.write_text(lines, encoding="utf-8")
    else:
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in lines),
            encoding="utf-8")
    return path


def _steps(n: int, **kwargs) -> list:
    return [_record(i, **kwargs) for i in range(1, n + 1)]


def _zero_contrast_record(step=1, *, outcome=False, hdv=True):
    terms = _terms(step)
    terms["response_contrast"] = 0.0
    record = _record(step, terms=terms)
    outcome_taus = 2 if outcome else 0
    hdv_slots = 133 if hdv else 0
    checks = {}
    if outcome:
        checks["outcome"] = {
            "count": outcome_taus * 3 * 19, "nonzero_count": 0,
            "max_abs": 0.0, "finite": True}
    if hdv:
        checks["hdv"] = {
            "count": hdv_slots, "nonzero_count": 0,
            "max_abs": 0.0, "finite": True}
    record["contrast_support"] = {
        "pairs": 1, "outcome_pairs": int(outcome),
        "outcome_taus": outcome_taus, "hdv_pairs": int(hdv),
        "hdv_slots": hdv_slots, "outcome_subterm": outcome,
        "hdv_subterm": hdv, "hdv_id_aligned": hdv,
        "residual_checks": checks,
    }
    return record


# ----------------------------------------------------------- acceptance ----

def test_valid_cpu40_log_passes_and_accepts_negative_nll(tmp_path):
    path = _write_log(tmp_path, _steps(40))
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "pass"
    assert report["failures"] == []
    values = report["terms"]["residual_nll"]
    assert values["first"] < 0.0 and values["min"] < 0.0


def test_valid_cuda50_log_passes(tmp_path):
    path = _write_log(tmp_path, _steps(50, device="cuda"))
    report = wmn1.check_file(path, **CUDA50)
    assert report["integrity"] == "pass", report["failures"]


def test_main_exit_codes_and_report_sha(tmp_path):
    path = _write_log(tmp_path, _steps(40))
    assert wmn1.main(["--log", str(path), "--expected-steps", "40",
                      "--expected-device", "cpu"]) == wmn1.EXIT_PASS
    report = wmn1.check_file(path, **CPU40)
    assert report["log"]["sha256"] == hashlib.sha256(
        path.read_bytes()).hexdigest()
    assert report["log"]["records"] == 40
    assert "not an approval" in report["note"]


def test_per_term_summary_evidence_fields(tmp_path):
    path = _write_log(tmp_path, _steps(40))
    report = wmn1.check_file(path, **CPU40)
    summary = report["terms"]["rollout"]
    assert set(summary) == {"first", "last", "min", "max", "mean",
                            "slope_per_step", "trend", "n_negative"}
    assert summary["min"] <= summary["mean"] <= summary["max"]


def test_out_report_written_exclusively(tmp_path):
    path = _write_log(tmp_path, _steps(40))
    out = tmp_path / "report.json"
    code = wmn1.main(["--log", str(path), "--expected-steps", "40",
                      "--expected-device", "cpu", "--out", str(out)])
    assert code == wmn1.EXIT_PASS
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["integrity"] == "pass"


# ------------------------------------------------------------ rejection ----

def test_wrong_step_count_rejected(tmp_path):
    path = _write_log(tmp_path, _steps(39))
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"
    assert any("exactly 40" in f for f in report["failures"])


def test_extra_step_rejected(tmp_path):
    path = _write_log(tmp_path, _steps(41))
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"
    assert any("found 41" in f for f in report["failures"])


def test_duplicate_step_rejected(tmp_path):
    records = _steps(40)
    records[5] = dict(records[4])
    path = _write_log(tmp_path, records)
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"
    assert any("order" in f for f in report["failures"])


def test_out_of_order_steps_rejected(tmp_path):
    records = _steps(40)
    records[2], records[3] = records[3], records[2]
    path = _write_log(tmp_path, records)
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"


def test_step_starting_at_zero_rejected(tmp_path):
    records = [_record(i) for i in range(0, 40)]
    path = _write_log(tmp_path, records)
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"


def test_boolean_term_value_rejected(tmp_path):
    terms = _terms(1)
    terms["rollout"] = True
    path = _write_log(tmp_path, [_record(1, terms=terms)])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("booleans rejected" in f for f in report["failures"])


def test_boolean_step_rejected(tmp_path):
    record = _record(1)
    record["step"] = True
    path = _write_log(tmp_path, [record])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("not a plain integer" in f for f in report["failures"])


def test_nan_and_infinity_terms_rejected(tmp_path):
    terms = _terms(1)
    terms["arrival"] = float("nan")
    terms["correction_crps"] = float("inf")
    line = json.dumps(_record(1, terms=terms), sort_keys=True)
    path = _write_log(tmp_path, line)
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("NaN/Infinity" in f for f in report["failures"])
    assert len(report["failures"]) >= 2


def test_zero_term_rejected(tmp_path):
    terms = _terms(1)
    terms["rollout"] = 0.0
    path = _write_log(tmp_path, [_record(1, terms=terms)])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("term 'rollout' is zero" in f for f in report["failures"])


def test_response_contrast_must_be_strictly_positive(tmp_path):
    terms = _terms(1)
    terms["response_contrast"] = -0.25
    path = _write_log(tmp_path, [_record(1, terms=terms)])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("not strictly positive" in f for f in report["failures"])


def test_zero_response_contrast_rejected(tmp_path):
    terms = _terms(1)
    terms["response_contrast"] = 0.0
    path = _write_log(tmp_path, [_record(1, terms=terms)])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("term 'response_contrast' is zero" in f
               or "lacks support proof" in f for f in report["failures"])


def test_one_proven_zero_inside_full_cpu40_passes(tmp_path):
    records = _steps(40)
    records[2] = _zero_contrast_record(3)
    report = wmn1.check_file(_write_log(tmp_path, records), **CPU40)
    assert report["integrity"] == "pass", report["failures"]
    assert report["terms"]["response_contrast"]["min"] == 0.0


@pytest.mark.parametrize("outcome,hdv", [(True, False), (True, True)])
def test_proven_zero_covers_each_active_subterm(tmp_path, outcome, hdv):
    record = _zero_contrast_record(outcome=outcome, hdv=hdv)
    report = wmn1.check_file(_write_log(tmp_path, [record]), 1, "cpu")
    assert report["integrity"] == "pass", report["failures"]


@pytest.mark.parametrize("change", [
    lambda rec: rec["contrast_support"].pop("residual_checks"),
    lambda rec: rec["contrast_support"]["residual_checks"].clear(),
    lambda rec: rec["contrast_support"]["residual_checks"]["hdv"].update(
        count=132),
    lambda rec: rec["contrast_support"]["residual_checks"]["hdv"].update(
        count=True),
    lambda rec: rec["contrast_support"]["residual_checks"]["hdv"].update(
        nonzero_count=1),
    lambda rec: rec["contrast_support"]["residual_checks"]["hdv"].update(
        nonzero_count=-1),
    lambda rec: rec["contrast_support"]["residual_checks"]["hdv"].update(
        max_abs=float("inf")),
    lambda rec: rec["contrast_support"]["residual_checks"]["hdv"].update(
        max_abs=-1.0),
    lambda rec: rec["contrast_support"]["residual_checks"]["hdv"].update(
        finite=False),
    lambda rec: rec["contrast_support"].update(hdv_slots=0),
    lambda rec: rec["contrast_support"].update(hdv_subterm=False),
    lambda rec: rec["contrast_support"].update(pairs=0),
    lambda rec: rec["contrast_support"].update(hdv_id_aligned=False),
    lambda rec: rec["contrast_support"]["residual_checks"].update(
        outcome={"count": 57, "nonzero_count": 0,
                 "max_abs": 0.0, "finite": True}),
])
def test_zero_contrast_incomplete_or_inconsistent_proof_fails(tmp_path, change):
    record = _zero_contrast_record()
    change(record)
    report = wmn1.check_file(_write_log(tmp_path, [record]), 1, "cpu")
    assert report["integrity"] == "fail"
    assert report["failures"]


def test_zero_with_unproved_outcome_subterm_fails(tmp_path):
    record = _zero_contrast_record(outcome=True, hdv=True)
    del record["contrast_support"]["residual_checks"]["outcome"]
    report = wmn1.check_file(_write_log(tmp_path, [record]), 1, "cpu")
    assert report["integrity"] == "fail"
    assert any("exact active subterms" in failure
               for failure in report["failures"])


def test_zero_without_any_supported_subterm_fails(tmp_path):
    record = _zero_contrast_record(outcome=False, hdv=False)
    report = wmn1.check_file(_write_log(tmp_path, [record]), 1, "cpu")
    assert report["integrity"] == "fail"
    assert any("no active supported subterm" in failure
               for failure in report["failures"])


def test_zero_with_wrong_outcome_quantile_count_fails(tmp_path):
    record = _zero_contrast_record(outcome=True, hdv=False)
    record["contrast_support"]["residual_checks"]["outcome"]["count"] = 2 * 19
    report = wmn1.check_file(_write_log(tmp_path, [record]), 1, "cpu")
    assert report["integrity"] == "fail"
    assert any("inconsistent outcome residual count" in failure
               for failure in report["failures"])


@pytest.mark.parametrize("term", ["residual_nll", "rollout", "arrival",
                                  "correction_crps"])
def test_other_four_losses_still_refuse_zero(tmp_path, term):
    record = _zero_contrast_record()
    record["terms"][term] = 0.0
    report = wmn1.check_file(_write_log(tmp_path, [record]), 1, "cpu")
    assert report["integrity"] == "fail"
    assert any(f"term {term!r} is zero" in failure
               for failure in report["failures"])


def test_missing_term_rejected(tmp_path):
    terms = _terms(1)
    del terms["arrival"]
    path = _write_log(tmp_path, [_record(1, terms=terms)])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("missing loss term 'arrival'" in f
               for f in report["failures"])


def test_malformed_json_line_rejected(tmp_path):
    good = json.dumps(_record(1), sort_keys=True)
    path = _write_log(tmp_path, good + "\n{\"step\": 2, \"tot\n")
    report = wmn1.check_file(path, expected_steps=2, expected_device="cpu")
    assert any("not valid JSON" in f for f in report["failures"])


def test_truncated_log_last_record_cut(tmp_path):
    good = "".join(json.dumps(r, sort_keys=True) + "\n"
                   for r in _steps(39))
    cut = json.dumps(_record(40), sort_keys=True)[:20]
    path = _write_log(tmp_path, good + cut)
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"
    assert any("not valid JSON" in f for f in report["failures"])


def test_blank_line_midfile_rejected(tmp_path):
    good = "".join(json.dumps(r, sort_keys=True) + "\n"
                   for r in _steps(2))
    path = _write_log(tmp_path, good.splitlines()[0] + "\n\n"
                      + good.splitlines()[1] + "\n")
    report = wmn1.check_file(path, expected_steps=2, expected_device="cpu")
    assert any("blank/malformed" in f for f in report["failures"])


def test_empty_log_rejected(tmp_path):
    path = _write_log(tmp_path, "")
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"
    assert any("no records" in f for f in report["failures"])


def test_missing_numeric_total_null_rejected(tmp_path):
    record = _record(1)
    record["total"] = None
    path = _write_log(tmp_path, [record])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("'total'" in f and "not a finite number" in f
               for f in report["failures"])


def test_negative_s_per_step_rejected(tmp_path):
    path = _write_log(tmp_path, [_record(1, s_per_step=-1.0)])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("s_per_step" in f for f in report["failures"])


def test_extra_unknown_key_rejected(tmp_path):
    path = _write_log(tmp_path, [_record(1, extra={"epoch": 3})])
    report = wmn1.check_file(path, expected_steps=1, expected_device="cpu")
    assert any("unexpected extra keys" in f for f in report["failures"])


# ---------------------------------------------------------------- device ----

def test_mixed_device_cpu_log_rejected(tmp_path):
    records = _steps(40)
    records[17] = _record(18, device="cuda")
    path = _write_log(tmp_path, records)
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "fail"
    assert any("expected cpu" in f for f in report["failures"])


def test_cuda_expected_but_null_fields_rejected(tmp_path):
    records = _steps(50, device="cuda")
    records[9] = _record(10, device="cpu")
    path = _write_log(tmp_path, records)
    report = wmn1.check_file(path, **CUDA50)
    assert any("expected cuda" in f for f in report["failures"])


def test_cpu_log_checked_against_cuda_mode_rejected(tmp_path):
    path = _write_log(tmp_path, _steps(50))
    report = wmn1.check_file(path, **CUDA50)
    assert report["integrity"] == "fail"


# ------------------------------------------------------------- CLI misc ----

def test_out_refuses_to_overwrite(tmp_path):
    path = _write_log(tmp_path, _steps(40))
    out = tmp_path / "report.json"
    out.write_text("keep me", encoding="utf-8")
    code = wmn1.main(["--log", str(path), "--expected-steps", "40",
                      "--expected-device", "cpu", "--out", str(out)])
    assert code == wmn1.EXIT_USAGE
    assert out.read_text(encoding="utf-8") == "keep me"


def test_missing_log_file_is_usage_error(tmp_path):
    code = wmn1.main(["--log", str(tmp_path / "absent.jsonl"),
                      "--expected-steps", "40",
                      "--expected-device", "cpu"])
    assert code == wmn1.EXIT_USAGE


# ------------------------------------------------------- label_t0 (a) ----

def _label_steps(n: int, *, asserted: int = 1) -> list:
    return [_record(i, extra={"label_t0_asserted": asserted,
                              "label_t0_violations": 0})
            for i in range(1, n + 1)]


def test_label_t0_fields_accepted_and_reported(tmp_path):
    path = _write_log(tmp_path, _label_steps(40, asserted=3))
    report = wmn1.check_file(path, **CPU40)
    assert report["integrity"] == "pass", report["failures"]
    assert report["label_t0"] == {"present": True, "asserted_total": 120,
                                  "violations_total": 0,
                                  "steps_with_assertions": 40}


def test_label_t0_fields_accepted_on_cuda(tmp_path):
    records = [_record(i, device="cuda",
                       extra={"label_t0_asserted": 2,
                              "label_t0_violations": 0})
               for i in range(1, 51)]
    report = wmn1.check_file(_write_log(tmp_path, records), **CUDA50)
    assert report["integrity"] == "pass", report["failures"]
    assert report["label_t0"]["present"] is True
    assert report["label_t0"]["asserted_total"] == 100
    assert report["label_t0"]["violations_total"] == 0


def test_label_t0_violation_rejected(tmp_path):
    records = _label_steps(40)
    records[7] = _record(8, extra={"label_t0_asserted": 1,
                                   "label_t0_violations": 2})
    report = wmn1.check_file(_write_log(tmp_path, records), **CPU40)
    assert report["integrity"] == "fail"
    assert any("label_t0_violations" in f for f in report["failures"])


def test_label_t0_zero_asserted_total_rejected(tmp_path):
    report = wmn1.check_file(
        _write_log(tmp_path, _label_steps(40, asserted=0)), **CPU40)
    assert report["integrity"] == "fail"
    assert any("label_t0_asserted totals 0" in f
               for f in report["failures"])


def test_label_t0_fields_must_be_all_or_none(tmp_path):
    records = _label_steps(40)
    del records[3]["label_t0_violations"]
    report = wmn1.check_file(_write_log(tmp_path, records), **CPU40)
    assert report["integrity"] == "fail"
    assert any("only some records" in f for f in report["failures"])


def test_label_t0_non_integer_rejected(tmp_path):
    records = _label_steps(40)
    records[0]["label_t0_asserted"] = True
    report = wmn1.check_file(_write_log(tmp_path, records), **CPU40)
    assert report["integrity"] == "fail"
    assert any("non-negative plain integer" in f
               for f in report["failures"])


def test_legacy_log_without_label_t0_fields_still_passes(tmp_path):
    report = wmn1.check_file(_write_log(tmp_path, _steps(40)), **CPU40)
    assert report["integrity"] == "pass", report["failures"]
    assert report["label_t0"]["present"] is False


def test_explicit_arguments_are_required():
    with pytest.raises(SystemExit):
        wmn1.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        wmn1.build_parser().parse_args(
            ["--log", "x", "--expected-steps", "40"])
    with pytest.raises(SystemExit):
        wmn1.build_parser().parse_args(
            ["--log", "x", "--expected-steps", "41",
             "--expected-device", "cpu"])
