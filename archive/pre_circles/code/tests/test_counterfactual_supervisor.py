import pytest

from counterfactual.supervisor import (
    LOCAL_SUPPORT_DENSITY_COHERENCE_V1,
    evaluate_predecision_supervisor,
)


def _features(**overrides):
    values = {
        "local_bin_100_200m_cav_count": 2,
        "traffic_lag_10_density_scaled": 0.8,
        "formation_soft_cav_speed_std_mps": 7.0,
    }
    values.update(overrides)
    return values


def test_v1_supervisor_admits_boundary_and_reports_observed_values():
    result = evaluate_predecision_supervisor(_features(), LOCAL_SUPPORT_DENSITY_COHERENCE_V1)
    assert result["admitted"] is True
    assert all(result["checks"].values())
    assert result["observed"]["traffic_lag_10_density_scaled"] == 0.8


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"local_bin_100_200m_cav_count": 1}, "local_cav_support"),
        ({"traffic_lag_10_density_scaled": 0.81}, "density_not_deeply_saturated"),
        ({"formation_soft_cav_speed_std_mps": 7.01}, "cav_speed_field_coherent"),
    ],
)
def test_v1_supervisor_refuses_each_failed_physical_condition(overrides, reason):
    result = evaluate_predecision_supervisor(
        _features(**overrides), LOCAL_SUPPORT_DENSITY_COHERENCE_V1
    )
    assert result["admitted"] is False
    assert reason in result["refusal_reasons"]
