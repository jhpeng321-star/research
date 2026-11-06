"""Synthetic JSONL contract tests for the round-nine training gate."""

import json

import pytest


REQUIRED_FIELDS = (
    "step",
    "stage_s",
    "loss",
    "merge_pivot_on_objective_disabled",
    "merge_pivot_tf_positive_mass_events",
    "merge_pivot_tf_zero_mass_events",
    "merge_pivot_tf_scored_seconds",
    "merge_pivot_tf_scored_cars",
    "merge_pivot_tf_excluded_motion",
    "merge_pivot_tau",
    "merge_pivot_w",
    "merge_pivot_grad_norm",
    "cuda_peak_mb",
)
POSITIVE_TOTAL_FIELDS = (
    "merge_pivot_tf_positive_mass_events",
    "merge_pivot_tf_scored_cars",
    "merge_pivot_grad_norm",
)
ZERO_TOTAL_FIELDS = (
    "merge_pivot_tf_zero_mass_events",
)
NONNEGATIVE_FIELDS = (
    *POSITIVE_TOTAL_FIELDS,
    *ZERO_TOTAL_FIELDS,
    "merge_pivot_tf_scored_seconds",
    "merge_pivot_tf_excluded_motion",
)
POSITIVE_FIELDS = ("stage_s", "merge_pivot_tau", "merge_pivot_w", "cuda_peak_mb")
INVALID_NUMBERS = (
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param("1", id="numeric-string"),
    pytest.param("invalid", id="text"),
    pytest.param([], id="array"),
    pytest.param({}, id="object"),
)


@pytest.fixture(scope="session")
def check_log():
    from wms.train.round9_gate import check_log

    return check_log


@pytest.fixture
def rows():
    return [
        {
            "step": index + 1,
            "stage_s": stage,
            "loss": (1.0, 0.0, -0.25, 0.5)[index],
            "merge_pivot_on_objective_disabled": 1,
            "merge_pivot_tf_positive_mass_events": (0, 2, 0, 1)[index],
            "merge_pivot_tf_zero_mass_events": 0,
            "merge_pivot_tf_scored_seconds": (0.0, 2.0, 0.0, 3.0)[index],
            "merge_pivot_tf_scored_cars": (0, 2, 0, 1)[index],
            "merge_pivot_tf_excluded_motion": (0, 1, 0, 0)[index],
            "merge_pivot_tau": 0.5,
            "merge_pivot_w": 0.25,
            "merge_pivot_grad_norm": (0.0, 0.2, 0.0, 0.1)[index],
            "cuda_peak_mb": (128.0, 512.0, 256.0, 384.0)[index],
        }
        for index, stage in enumerate((120, 120, 300, 300))
    ]


def write_rows(path, rows):
    # allow_nan deliberately exercises rejection of Python's nonfinite JSON tokens.
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def logs(tmp_path, rows):
    train_log = tmp_path / "train_log.jsonl"
    smoke_log = tmp_path / "smoke.log"
    write_rows(train_log, rows)
    smoke_log.write_text("smoke completed successfully\n", encoding="utf-8")
    return train_log, smoke_log


def assert_verdict(result, *, ok):
    assert isinstance(result, dict)
    assert result["ok"] is ok
    assert isinstance(result["errors"], list)
    assert all(isinstance(error, str) and error.strip() for error in result["errors"])
    assert isinstance(result["metrics"], dict)
    if ok:
        assert result["errors"] == []
    else:
        assert result["errors"]


def reject_rows(check_log, logs, rows):
    train_log, smoke_log = logs
    write_rows(train_log, rows)
    assert_verdict(check_log(train_log, smoke_log), ok=False)


def mutation_row(rows, field):
    if field == "stage_s":
        # Preserve two valid rows at each exact stage, isolating stage validation.
        rows.append({**rows[-1], "step": 5})
        return rows[-1]
    # True must replace step 1: otherwise sequence validation alone rejects it.
    return rows[0] if field == "step" else rows[1]


def test_valid_logs_report_peak_cuda_memory(check_log, logs):
    result = check_log(*logs)
    assert_verdict(result, ok=True)
    # The maximum is neither the last row nor the average or sum.
    assert result["metrics"]["cuda_peak_mb"] == 512.0


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
@pytest.mark.parametrize("mode", ("missing", "null"))
def test_required_field_missing_or_null(check_log, logs, rows, field, mode):
    row = mutation_row(rows, field)
    if mode == "missing":
        del row[field]
    else:
        row[field] = None
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_required_field_rejects_invalid_number(check_log, logs, rows, field, value):
    mutation_row(rows, field)[field] = value
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("field", POSITIVE_FIELDS)
@pytest.mark.parametrize("value", (0, -1))
def test_strictly_positive_field_on_one_row(check_log, logs, rows, field, value):
    mutation_row(rows, field)[field] = value
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("field", NONNEGATIVE_FIELDS)
def test_negative_metric_on_one_row(check_log, logs, rows, field):
    # Positive-total fields still have a positive aggregate after this mutation.
    rows[0][field] = -0.01
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("field", POSITIVE_TOTAL_FIELDS)
def test_required_total_cannot_be_all_zero(check_log, logs, rows, field):
    for row in rows:
        row[field] = 0
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("field", ZERO_TOTAL_FIELDS)
def test_zero_mass_total_must_be_zero_independently(check_log, logs, rows, field):
    rows[1][field] = 1
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("field", ZERO_TOTAL_FIELDS)
def test_zero_mass_negative_counts_cannot_cancel(check_log, logs, rows, field):
    rows[0][field] = -1
    rows[1][field] = 1
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("threshold", (120, 300))
def test_scored_seconds_required_at_each_stage_threshold(check_log, logs, rows, threshold):
    for row in rows:
        if row["stage_s"] >= threshold:
            row["merge_pivot_tf_scored_seconds"] = 0
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("value", (0, 2))
def test_retired_onpolicy_objective_must_be_flagged_disabled_on_every_row(
        check_log, logs, rows, value):
    rows[2]["merge_pivot_on_objective_disabled"] = value
    reject_rows(check_log, logs, rows)


def test_excluded_motion_is_reported_with_its_rate(check_log, logs):
    result = check_log(*logs)
    assert_verdict(result, ok=True)
    assert result["metrics"]["merge_pivot_tf_excluded_motion"] == 1
    assert result["metrics"]["merge_pivot_tf_scored_cars"] == 3
    assert result["metrics"]["tf_excluded_motion_rate"] == 0.25


@pytest.mark.parametrize(
    "steps",
    (
        pytest.param((0, 1, 2, 3), id="zero-start"),
        pytest.param((-1, 0, 1, 2), id="negative-start"),
        pytest.param((2, 3, 4, 5), id="not-from-one"),
        pytest.param((1, 2, 2, 3), id="duplicate"),
        pytest.param((1, 3, 2, 4), id="out-of-order"),
        pytest.param((1, 2, 4, 5), id="gap"),
        pytest.param((1, 2.5, 3, 4), id="fractional"),
        pytest.param((1, 2.0, 3, 4), id="integer-valued-float"),
    ),
)
def test_steps_must_be_positive_consecutive_integers(check_log, logs, rows, steps):
    for row, step in zip(rows, steps):
        row["step"] = step
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("count", (1, 2, 3))
def test_at_least_four_rows_required(check_log, logs, rows, count):
    reject_rows(check_log, logs, rows[:count])


@pytest.mark.parametrize(
    "stages",
    (
        pytest.param((120, 180, 300, 300), id="only-one-exact-120"),
        pytest.param((120, 120, 300, 360), id="only-one-exact-300"),
        pytest.param((180, 180, 300, 300), id="missing-exact-120"),
        pytest.param((120, 120, 360, 360), id="missing-exact-300"),
        pytest.param((120, 120, 120, 120), id="missing-stage-300"),
    ),
)
def test_two_rows_required_for_each_exact_stage(check_log, logs, rows, stages):
    for row, stage in zip(rows, stages):
        row["stage_s"] = stage
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("content", ("", " \n\t\n", "{broken json}\n"))
def test_empty_or_malformed_train_log(check_log, logs, content):
    train_log, smoke_log = logs
    train_log.write_text(content, encoding="utf-8")
    assert_verdict(check_log(train_log, smoke_log), ok=False)


@pytest.mark.parametrize("position", (0, 2, 4))
def test_malformed_line_among_valid_rows(check_log, logs, rows, position):
    train_log, smoke_log = logs
    lines = [json.dumps(row) for row in rows]
    lines.insert(position, "{broken json}")
    train_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert_verdict(check_log(train_log, smoke_log), ok=False)


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(None, id="null"),
        pytest.param([], id="array"),
        pytest.param("text", id="string"),
        pytest.param(1, id="number"),
        pytest.param(True, id="boolean"),
    ),
)
@pytest.mark.parametrize("surrounded_by_valid_rows", (False, True))
def test_nonobject_train_row(check_log, logs, rows, value, surrounded_by_valid_rows):
    invalid_rows = rows[:2] + [value] + rows[2:] if surrounded_by_valid_rows else [value]
    reject_rows(check_log, logs, invalid_rows)


@pytest.mark.parametrize("log_index", (0, 1), ids=("train", "smoke"))
def test_missing_log(check_log, logs, log_index):
    logs[log_index].unlink()
    assert_verdict(check_log(*logs), ok=False)


@pytest.mark.parametrize("exit_code", (-9, 1, 2))
def test_nonzero_train_exit_code(check_log, logs, exit_code):
    assert_verdict(check_log(*logs, train_exit_code=exit_code), ok=False)


@pytest.mark.parametrize("position", ("start", "middle", "end"))
def test_traceback_anywhere_in_smoke_log(check_log, logs, position):
    train_log, smoke_log = logs
    contents = {
        "start": "Traceback (most recent call last):\nsmoke completed\n",
        "middle": "smoke started\nprefix Traceback suffix\nsmoke completed\n",
        "end": "smoke completed\nTraceback",
    }
    smoke_log.write_text(contents[position], encoding="utf-8")
    assert_verdict(check_log(train_log, smoke_log), ok=False)


def test_traceback_in_otherwise_valid_train_log(check_log, logs, rows):
    rows[1]["message"] = "prefix Traceback (most recent call last): suffix"
    reject_rows(check_log, logs, rows)


@pytest.mark.parametrize("failure", (None, "checkpoint", "train_exit", "stale_report"))
def test_cli_only_passes_with_reload_and_fresh_report(logs, monkeypatch, failure):
    from wms.train import round9_gate

    train_log, smoke_log = logs
    report = train_log.parent / "smoke_r9.gate.json"
    calls = []

    def reload(directory, rows):
        calls.append(directory)
        assert len(rows) == 4
        if failure == "checkpoint":
            raise ValueError("broken saved model")
        return [{"reloaded": True}]

    monkeypatch.setattr(round9_gate, "check_checkpoints", reload)
    if failure == "stale_report":
        report.write_text("prior evidence", encoding="utf-8")
    code = round9_gate.main([
        "--smoke-dir", str(train_log.parent), "--smoke-log", str(smoke_log),
        "--train-exit-code", "7" if failure == "train_exit" else "0",
        "--report", str(report),
    ])
    assert code == (0 if failure is None else 3)
    if failure == "stale_report":
        assert report.read_text(encoding="utf-8") == "prior evidence"
    else:
        result = json.loads(report.read_text(encoding="utf-8"))
        assert result["ok"] is (failure is None)
        assert result["metrics"]["cuda_peak_mb"] == 512.0
    assert bool(calls) is (failure != "train_exit")
