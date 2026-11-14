"""tests/test_domination_scoreboard.py"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_scoreboard():
    spec = importlib.util.spec_from_file_location(
        "domination_scoreboard", ROOT / "tools" / "domination_scoreboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = [
    ("fair_safe_v9_i40_dt05_holdout101_150", 27),
    ("fair_safe_i90_dt05_holdout101_150", 34),
    ("fair_safe_v8_i150_dt05_holdout101_150", 33),
]


@pytest.mark.parametrize("dirname,expected", CASES)
def test_reproduces_known_bars(dirname, expected):
    d = ROOT / "runs" / "eval" / dirname
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    result = _load_scoreboard().score_eval_dir(d)
    assert result["n"] == 50
    assert result["n_dominate"] == expected
    # Safety must never bind on these dirs: every fail is G- or loss-driven.
    for row in result["per_seed"]:
        assert row["d_collisions"] <= 0
        assert row["d_ttc"] <= 0
    # Structural seeds are reported separately and none is dominated.
    assert result["structural"]["n"] == 11
    assert result["n_winnable"] == 39
