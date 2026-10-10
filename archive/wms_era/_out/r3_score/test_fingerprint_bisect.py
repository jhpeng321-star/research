"""Regression: score_round fingerprints reproduce BISECT_RESIDUAL (old C1, 60 s).

CPU only (~2-4 min).  Pins the dedicated diagnostic
``E:/rd-wt/ws-bisect-residual/_out/BISECT_RESIDUAL.md`` for the old C1 struct on
the first 10 worlds of wall_same, 60 s horizon, first-60-s step window.  The
diagnostic scored sorted(recorded)[0] per state (native for 3 of the 10 states),
NOT the registered `wall` arm: that arm choice was the whole discrepancy with the
harness default (arm `wall`, 120 s rollout, 20 states):
mean |g*da| 1.765, p95 3.542, frac>0 0.441, mean signed 0.612, lane-change
rate pred 0.84 % (646 events) / recorded 1.23 % (950), speed bias -0.236,
wrong-lane share of |ds| @60 s 0.334, 77 847 HDV-steps per state set.

Run:  python -m pytest -q _out/r3_score/test_fingerprint_bisect.py \
          --basetemp _out/r3_score/_pytest_tmp -p no:cacheprovider
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_round as sr  # noqa: E402


@pytest.fixture(scope="module")
def fp():
    ref = sr.DEFAULT_REFERENCE
    staged = sr.stage_copy(ref / "wm_s.pt", "ref_last")
    spec = {"staged": str(staged), "physics": sr.receipt_physics(ref),
            "entries": False, "entries_v2": False}
    recs = sr.select_subset(sr.SETS["wall_same"])[:10]
    assert len(recs) == 10
    return sr.fingerprint_model("ref_last", spec, recs, "first_sorted",
                                 predict_horizon=60.0)


def test_bisect_values(fp):
    assert fp["window_s"] == 60.0
    assert fp["hdv_steps"] == pytest.approx(77847, rel=0.005)
    assert fp["g_delta_mean_abs"] == pytest.approx(1.765, rel=0.01)
    assert fp["g_delta_p95_abs"] == pytest.approx(3.542, rel=0.01)
    assert fp["g_delta_mean"] == pytest.approx(0.612, rel=0.01)
    assert fp["g_delta_frac_pos"] == pytest.approx(0.441, abs=0.002)
    assert fp["lane_changes_pred"] == pytest.approx(646, abs=3)
    assert fp["lane_changes_rec"] == 950
    assert fp["lane_rate_pred"] == pytest.approx(0.0084, abs=0.0002)
    assert fp["lane_rate_rec"] == pytest.approx(0.0123, abs=0.0002)
    assert fp["speed_bias_mean"] == pytest.approx(-0.236, abs=0.005)
    assert fp["wrong_lane_share"]["60"] == pytest.approx(0.334, abs=0.005)


def test_cut_history_masks_old_samples():
    """cut_history(…, 4.5) must hide everything older than 4.5 s (real long history)."""
    import numpy as np
    import torch

    sys.path.insert(0, str(sr.WM_N1))
    from wms.history import long_history_features

    t, k = 121, 3
    rng = np.random.default_rng(0)
    rec = {"history": {"kin": rng.normal(size=(t, k, 7)).tolist(),
                       "mask": np.ones((t, k), bool).tolist(),
                       "target_mask": np.ones((t - 1, k), bool).tolist()}}

    def valid(r):
        h = r["history"]
        _, m = long_history_features(
            torch.tensor(h["kin"], dtype=torch.float32), torch.tensor(h["mask"]),
            torch.tensor(h["target_mask"]), torch.tensor(h["mask"])[-1],
            history_s=60.0, history_stride_s=1.0, dt_s=0.5)
        return int(m.sum())

    assert valid(rec) == 61 * k
    assert valid(sr.cut_history(rec, 4.5)) <= 5 * k
