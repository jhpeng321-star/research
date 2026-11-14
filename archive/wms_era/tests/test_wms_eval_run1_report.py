import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))
import pytest  # noqa: E402

from wms_eval.run1_report import rank_correlation, sign_agreement  # noqa: E402


def test_sign_agreement_counts_matching_and_zero_signs():
    agree, total = sign_agreement([1.0, -2.0, 0.0, 3.0], [2.0, -1.0, 0.0, -4.0])
    assert (agree, total) == (3, 4)


def test_rank_correlation_is_one_for_monotone_and_negative_for_reversed():
    assert rank_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
    assert rank_correlation([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_rank_correlation_handles_ties_and_degenerate_input():
    assert rank_correlation([1.0], [1.0]) is None
    assert rank_correlation([1.0, 1.0], [1.0, 2.0]) is None
