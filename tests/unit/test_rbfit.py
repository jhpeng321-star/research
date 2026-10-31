"""Inference/observable-contract regressions for physical fitting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "tools" / "rbfit" / "idm_fit.py"
spec = importlib.util.spec_from_file_location("rbfit_idm", MODULE)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def test_reject_non_fit_before_access(tmp_path):
    with pytest.raises(ValueError, match="Only registered FIT"):
        fit.native_episode(tmp_path, 969432)


def test_forward_acceleration_and_identity_reuse():
    # A reused packed slot is not the same driver; a cut-in leader is not the
    # leader that generated the previous interval's acceleration.
    kin = np.zeros((3, 3, 7))
    kin[:, :, 1] = 1
    kin[:, :, 6] = 5
    kin[:, :, 0] = [[0, 30, 60], [5, 36, 66], [11, 42, 72]]
    kin[:, :, 4] = [[10, 12, 12], [12, 12, 12], [14, 12, 12]]
    data = dict(veh_kin=kin, veh_mask=np.ones((3, 3), dtype=bool),
                is_cav=np.zeros((3, 3), dtype=bool),
                vid=np.array([[1, 2, 3], [1, 2, 3], [4, 2, 3]]),
                time_s=np.array([299.5, 300, 300.5]))
    rows, hist, _ = fit.extract_rows(data)
    ego = rows[rows[:, 1] == 1]
    assert len(ego) == 1
    assert fit.observed(ego)[0] == 4  # forward dv/dt, not stored channel 5
    assert ego[0, 5] == 25  # subtract LEADER length
    assert hist[1] == 0.5
    assert 4 not in hist  # post-300 identity cannot enter a causal posterior
    data["vid"][1, 1] = 99
    rows, _, _ = fit.extract_rows(data)
    assert not np.any(rows[:, 1] == 1)


def test_recovers_identifiable_synthetic_physical_parameters():
    rng = np.random.default_rng(33)
    physical = np.array([33.0, 1.3, 2.2, 3.8, 2.8])
    rows = np.zeros((600, 10))
    rows[:, 3] = rng.uniform(2, 34, len(rows))
    rows[:, 5] = rng.uniform(8, 120, len(rows))
    rows[:, 6] = rng.uniform(0, 35, len(rows))
    rows[:, 9] = 0.5
    rows[:, 4] = fit.next_speed(rows[:, 3], rows[:, 5], rows[:, 6], physical)
    _, z, cov, _, _, success = fit.posterior_job(
        ("synthetic", rows, np.log(fit.INITIAL), np.full(5, 4.0), 0.01))
    assert success
    np.testing.assert_allclose(np.exp(z), physical, rtol=0.001)
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.diag(cov).max() < 4


def test_receding_leader_uses_finite_free_speed():
    p = np.array([30, 1, 2.6, 4.5, 2.5])
    free = fit.next_speed(np.array([0, 20, 30]), np.full(3, np.inf), np.zeros(3), p)
    assert np.isfinite(free).all()
    assert 0 < free[0] <= 1.3
    assert 20 < free[1] <= 21.3
    assert abs(free[2] - 30) < 1e-6
