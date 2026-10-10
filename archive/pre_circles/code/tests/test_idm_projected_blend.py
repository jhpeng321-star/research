"""Pure contracts for the development-only projected blend."""

import pytest

from idm.decentralized_damping import (
    ABSTAIN,
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
    decode_mode,
    nominal_mode_index,
)
from idm.projected_blend import (
    IDM_P30_PROJECTED_BLEND_V1,
    native_proxy_acceleration,
    projected_blend_acceleration,
)


def test_blend_contract_and_action_mapping_are_frozen():
    assert IDM_P30_PROJECTED_BLEND_V1.active_strengths == (0.25, 0.5, 0.75, 1.0)
    assert decode_mode(0, ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET).index == ABSTAIN
    assert decode_mode(1, ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET).value == 0.25
    assert decode_mode(2, ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET).value == 0.5
    assert decode_mode(3, ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET).value == 0.75
    assert decode_mode(4, ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET).value == 1.0
    assert decode_mode(99, ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET).index == ABSTAIN
    assert nominal_mode_index(ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET) == 4


def test_native_proxy_uses_allowed_acceleration_and_follow_speed_caps():
    target, acceleration = native_proxy_acceleration(
        current_speed_mps=20.0,
        allowed_speed_mps=30.0,
        maximum_acceleration_mps2=2.0,
        follow_speed_mps=21.0,
        dt_s=1.0,
    )
    assert target == 21.0
    assert acceleration == 1.0

    target, acceleration = native_proxy_acceleration(
        current_speed_mps=20.0,
        allowed_speed_mps=20.5,
        maximum_acceleration_mps2=2.0,
        follow_speed_mps=None,
        dt_s=1.0,
    )
    assert target == 20.5
    assert acceleration == 0.5


@pytest.mark.parametrize("strength", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_projected_blend_is_convex_and_has_exact_endpoints(strength):
    value = projected_blend_acceleration(
        projected_native_acceleration_mps2=0.5,
        projected_p30_acceleration_mps2=-1.0,
        strength=strength,
    )
    assert value == pytest.approx((1.0 - strength) * 0.5 + strength * -1.0)
    assert -1.0 <= value <= 0.5


def test_projected_blend_rejects_out_of_contract_strength():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        projected_blend_acceleration(
            projected_native_acceleration_mps2=0.5,
            projected_p30_acceleration_mps2=-1.0,
            strength=1.25,
        )
