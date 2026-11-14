"""ScoreResult structure, quantile conventions and paired differences."""
from __future__ import annotations

import numpy as np
import pytest

from wm_api import API_VERSION
from wm_api.result import METRIC_NAMES, QUANTILE_LEVELS, level_key

from _tiny import default_plans, tiny_scorer, tiny_state, wall_plan


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scoring_ckpt")
    scorer = tiny_scorer(tmp, k=2)
    result = scorer.score(tiny_state(), default_plans(), horizon_s=60, allow_other_horizon=True)
    return scorer, result


def test_score_result_envelope(scored):
    scorer, result = scored
    data = result.to_json()
    assert data["api_version"] == API_VERSION == "1.0"
    assert data["horizon_s"] == pytest.approx(60.0)
    assert data["tau_s"] == pytest.approx(60.0)
    assert data["members"] == [1, 2]
    assert data["seed"] == 0 and data["n_samples"] == 1
    assert len(data["plans"]) == 2
    assert data["checkpoint"]["sha_verified"] is False
    assert set(data["checkpoint"]["sha256"]) == {"member_1.pt",
                                                 "member_2.pt"}
    assert all(len(h) == 64 for h in data["checkpoint"]["sha256"].values())


def test_metric_structure_and_quantile_monotonicity(scored):
    _, result = scored
    for plan in result.plans:
        assert set(plan.metrics) == set(METRIC_NAMES)
        for name, ms in plan.metrics.items():
            keys = list(ms.quantiles)
            assert keys == [level_key(lv) for lv in QUANTILE_LEVELS]
            assert keys[0] == "q05" and keys[-1] == "q95"
            vals = [ms.quantiles[k] for k in keys]
            assert all(np.diff(vals) >= -1e-12), name
            assert len(ms.member_draws) == 2
            assert ms.mean == pytest.approx(float(np.mean(
                ms.member_draws)))
            assert all(np.isfinite(v) for v in vals)


def test_median_quantile_is_a_member_draw(scored):
    _, result = scored
    plan = result.plans[0]
    ms = plan.metrics["delay_s_per_veh"]
    assert ms.quantiles["q50"] == pytest.approx(np.mean(ms.member_draws))


def test_plan0_is_reference_and_paired_sign_convention(scored):
    _, result = scored
    ref, treat = result.plans
    assert ref.is_reference and not treat.is_reference
    for name in METRIC_NAMES:
        block = ref.paired_diff_vs_plan0[name]
        assert block["mean"] == 0.0
        assert all(v == 0.0 for v in block["quantiles"].values())
        dv = treat.paired_diff_vs_plan0[name]
        # cost difference = treatment - reference; negative = improvement
        assert dv["mean"] == pytest.approx(
            treat.metrics[name].mean - ref.metrics[name].mean)
        for lv in ("q05", "q50", "q95"):
            assert dv["quantiles"][lv] == pytest.approx(
                treat.metrics[name].quantiles[lv]
                - ref.metrics[name].quantiles[lv])


def test_ood_score_present_and_finite(scored):
    _, result = scored
    for plan in result.plans:
        assert np.isfinite(plan.ood_score)
        assert plan.ood_score >= 0.0


def test_members_subset_and_n_samples(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("subset_ckpt")
    scorer = tiny_scorer(tmp, k=2)
    st = tiny_state()
    sub = scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True, members=[1])
    assert sub.members == [2]
    assert len(sub.plans[0].metrics["delay_s_per_veh"].member_draws) == 1
    multi = scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True, n_samples=3)
    assert multi.n_samples == 3
    with pytest.raises(Exception):
        scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True, members=[0, 5])


def test_wall_plan_scores_with_gates(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wall_ckpt")
    scorer = tiny_scorer(tmp, k=1)
    result = scorer.score(tiny_state(), [default_plans()[0], wall_plan()],
                          horizon_s=60, allow_other_horizon=True)
    wp = result.plans[1]
    assert "pair_gate" in wp.wall_gates
    assert wp.descriptor != result.plans[0].descriptor
    assert np.isfinite(wp.metrics["delay_s_per_veh"].mean)


def test_obstacle_state_scores(tmp_path_factory):
    from wm_api import StateSnapshot

    tmp = tmp_path_factory.mktemp("obs_ckpt")
    scorer = tiny_scorer(tmp, k=1)
    st = tiny_state(obstacles=1)
    assert st.n_model_vehicles == 7
    result = scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True)
    assert result.meta["obstacles"] == 1
    assert np.isfinite(result.plans[0].metrics["ttc3_exposure_s"].mean)
    data = StateSnapshot.from_json(st.to_json())
    assert len(data.obstacles) == 1
