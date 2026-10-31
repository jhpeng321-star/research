import json

import pytest

from tools.wmfixmerge.verdict_panels import load_group_file, parse_groups, save_group_file


BINDING = dict(candidate_sha256="c" * 64, candidate_config_sha256="f" * 64,
               reference_sha256="r" * 64)


def test_parse_groups_ranges_lists_and_singletons():
    assert parse_groups("0-17", 36) == list(range(18))
    assert parse_groups("18,19", 36) == [18, 19]
    assert parse_groups("3", 36) == [3]
    assert parse_groups("5-3,1", 36) == [1, 3, 4, 5]


def test_parse_groups_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        parse_groups("", 36)
    with pytest.raises(ValueError):
        parse_groups("36", 36)
    with pytest.raises(ValueError):
        parse_groups("-1-2", 36)


def test_group_file_roundtrip_and_foreign_binding_rejected(tmp_path):
    path = tmp_path / "A_candidate_007.json"
    rows = [{"pair_key": "k", "arm": "native", "draw": 0, "values": {}}]
    save_group_file(path, rows, BINDING)
    assert load_group_file(path, BINDING) == rows
    foreign = dict(BINDING, candidate_sha256="0" * 64)
    with pytest.raises(ValueError):
        load_group_file(path, foreign)
    doc = json.loads(path.read_text())
    assert doc["binding"] == BINDING and doc["rows"] == rows


def test_group_file_rejects_empty_rows_and_missing_fields(tmp_path):
    path = tmp_path / "A_base_000.json"
    path.write_text(json.dumps(dict(binding=BINDING, rows=[])))
    with pytest.raises(ValueError):
        load_group_file(path, BINDING)
    path.write_text(json.dumps(dict(rows=[{"a": 1}])))
    with pytest.raises(ValueError):
        load_group_file(path, BINDING)


def test_shard_cli_requires_final_checkpoint_and_existing_dir(tmp_path, capsys):
    import sys
    from tools.wmfixmerge import verdict_shard
    (tmp_path / "out").mkdir()
    with pytest.raises((ValueError, OSError, json.JSONDecodeError)):
        verdict_shard.main(["--train-dir", str(tmp_path), "--out", str(tmp_path / "out"),
                            "--groups", "0-17"])
