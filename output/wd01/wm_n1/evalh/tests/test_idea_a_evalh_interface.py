"""P8 Idea A eval-harness surface (implemented)."""
from __future__ import annotations

import numpy as np
import pytest

import evalh.idea_a as ia
from evalh.protocol import OutcomeRow


def test_variants_and_held_out_protocol():
    assert ia.VARIANT_A == "A"
    assert ia.IDEA_A_VARIANTS == ("T", "A", "C", "N")
    tests = ia.held_out_protocol()
    assert tuple(t.name for t in tests) == ("i", "ii", "iii")
    assert "J4B" in tests[0].eval_filter


def test_predictor_surface_and_not_loaded():
    p = ia.IdeaAVariantPredictor()
    assert p.name == "A"
    assert np.asarray(p.quantile_levels).size == 19
    assert p.predict_quantiles([]).shape == (0, 19)
    row = OutcomeRow(world="w", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                     policy="native", y_delay_s=1.0)
    with pytest.raises(ia.IdeaANotLoadedError):
        p.predict_quantiles([row])


def test_ood_config_defaults():
    cfg = ia.OodConfig()
    assert cfg.cal_quantile == 0.95
    assert cfg.min_summary_std == 1e-6
