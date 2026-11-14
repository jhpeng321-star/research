"""Read-only interface compatibility with the frozen Part B GBM."""
from __future__ import annotations

import numpy as np
import pytest

from combine.crps import crps_pinball
from combine.oof_tree import DEFAULT_QUANTILES


def test_crps_pinball_matches_partb_gbm():
    pytest.importorskip("torch")
    import partB_v0.gbm as gbm
    assert tuple(gbm.QUANTILES) == tuple(DEFAULT_QUANTILES)
    assert callable(getattr(gbm.GbmValueModel, "fit"))
    assert callable(getattr(gbm.GbmValueModel, "predict_quantiles_rows"))

    qs = sorted(gbm.QUANTILES)
    qv = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                   [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    y = np.array([3.2, 2.5])
    got = crps_pinball(qs, qv, y)
    ref = gbm.crps_from_quantiles(qs, {q: qv[:, i]
                                       for i, q in enumerate(qs)}, y)
    assert np.allclose(got, ref)
