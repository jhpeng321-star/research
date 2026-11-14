import json
from pathlib import Path

from output.multi_wall_probe_v1.prefix import verify


def _write(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_verify_ignores_only_header_path_metadata_and_filters_inclusive_boundary(tmp_path):
    source = tmp_path / "source"
    new = tmp_path / "new"
    source.mkdir()
    new.mkdir()
    header_left = {"schema": "stream/1", "path": "old/run", "vtype_path": "old/type"}
    header_right = {"schema": "stream/1", "path": "new/run", "vtype_path": "new/type"}
    rows = [header_left, {"time_s": 600, "state": {"speed": 26}}, {"time_s": 600.5, "state": {"speed": 25}}]
    rows_new = [header_right, rows[1], rows[2], {"time_s": 601, "state": {"speed": 1}}]
    _write(source / "telemetry.jsonl", rows)
    _write(new / "telemetry.jsonl", rows_new)
    result = verify(source, new, 600, streams=("telemetry.jsonl",))
    assert result["passed"]


def test_verify_uses_interval_end_and_reports_first_mismatch(tmp_path):
    source = tmp_path / "source"
    new = tmp_path / "new"
    source.mkdir()
    new.mkdir()
    left = [
        {"schema": "interval/1", "path": "source"},
        {"start_s": 599, "end_s": 600, "value": 1},
        {"start_s": 600, "end_s": 601, "value": 2},
    ]
    right = [
        {"schema": "interval/1", "path": "new"},
        {"start_s": 599, "end_s": 600, "value": 1},
        {"start_s": 600, "end_s": 601, "value": 9},
    ]
    _write(source / "intervals.jsonl", left)
    _write(new / "intervals.jsonl", right)
    result = verify(source, new, 600, streams=("intervals.jsonl",))
    assert result["passed"]

    right[1]["value"] = 8
    _write(new / "intervals.jsonl", right)
    result = verify(source, new, 600, streams=("intervals.jsonl",))
    difference = result["first_difference"]
    assert not result["passed"]
    assert difference["prefix_position"] == 1
    assert difference["source"]["value"] == 1
    assert difference["new"]["value"] == 8


def test_verify_stops_at_first_stream_mismatch(tmp_path):
    source = tmp_path / "source"
    new = tmp_path / "new"
    source.mkdir()
    new.mkdir()
    _write(source / "policy.jsonl", [{"time_s": 1, "phase": "hold"}])
    _write(new / "policy.jsonl", [{"time_s": 1, "phase": "abort"}])
    _write(source / "commands.jsonl", [{"time_s": 1, "accel": 1}])
    _write(new / "commands.jsonl", [{"time_s": 1, "accel": 2}])
    result = verify(source, new, 600, streams=("policy.jsonl", "commands.jsonl"))
    assert result["first_difference"]["stream"] == "policy.jsonl"
