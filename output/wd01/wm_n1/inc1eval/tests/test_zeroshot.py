"""INC1 zero-shot assertion: event-mask counts and scenario token scan."""
from __future__ import annotations

import json

import numpy as np

from inc1eval import zeroshot as Z


def _pack(tmp_path, nonzero=False):
    pack = tmp_path / "pack"
    pack.mkdir(parents=True)
    arr = np.zeros((4, 8), dtype=np.uint8)
    if nonzero:
        arr[2, 3] = 1
    np.save(pack / "event_mask.npy", arr)
    (pack / "manifest.json").write_text(json.dumps({
        "schema": "wm_n1_pack/1",
        "arrays": {"event_mask": {"file": "event_mask.npy", "dtype": "uint8",
                                  "shape": [4, 8]}},
    }), encoding="utf-8")
    return pack


def _source(tmp_path, dirty=False):
    root = tmp_path / "scenarios"
    root.mkdir(parents=True)
    text = ("scenario: merge\nramp: default\n"
            + ("stopped_obstacle: true\nsetStop: 42\n" if dirty else ""))
    (root / "merge.sumocfg").write_text(text, encoding="utf-8")
    return root


def test_event_mask_zero_and_blocked(tmp_path):
    clean = Z.count_event_mask(_pack(tmp_path / "a", nonzero=False))
    assert clean["nonempty_rows"] == 0 and clean["nonzero_entries"] == 0
    dirty = Z.count_event_mask(_pack(tmp_path / "b", nonzero=True))
    assert dirty["nonempty_rows"] == 1 and dirty["nonzero_entries"] == 1


def test_scenario_token_counts(tmp_path):
    clean = Z.count_scenario_tokens(_source(tmp_path / "a", dirty=False))
    assert sum(clean["counts"].values()) == 0
    dirty = Z.count_scenario_tokens(_source(tmp_path / "b", dirty=True))
    assert dirty["counts"]["setStop"] == 1
    assert dirty["counts"]["stopped_obstacle"] == 1


def test_assert_zero_shot_pass_and_block(tmp_path):
    clean = Z.assert_zero_shot(_pack(tmp_path / "p1", nonzero=False),
                               _source(tmp_path / "s1", dirty=False))
    assert clean["blocked"] is False
    blocked = Z.assert_zero_shot(_pack(tmp_path / "p2", nonzero=True),
                                 _source(tmp_path / "s2", dirty=False))
    assert blocked["blocked"] is True
    blocked2 = Z.assert_zero_shot(_pack(tmp_path / "p3", nonzero=False),
                                  _source(tmp_path / "s3", dirty=True))
    assert blocked2["blocked"] is True


def test_cli_exit_codes(tmp_path, capsys):
    clean_pack = _pack(tmp_path / "cp", nonzero=False)
    clean_src = _source(tmp_path / "cs", dirty=False)
    assert Z.main(["--pack", str(clean_pack),
                   "--source-root", str(clean_src)]) == 0
    dirty_pack = _pack(tmp_path / "dp", nonzero=True)
    assert Z.main(["--pack", str(dirty_pack),
                   "--source-root", str(clean_src)]) == 2


def test_missing_event_mask_raises(tmp_path):
    import pytest

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        Z.count_event_mask(empty)
