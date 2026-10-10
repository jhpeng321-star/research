"""Tests for continuous-action magnitude-scaled intervention costs.

With a Gaussian policy the binary intervention cost is an undescendable
cliff (|a| > 1e-3 almost surely, so every step pays -1). When
``action_mag`` is provided (continuous path), the control cost and the
easy-intervention tax scale by |a|/a_full — gentler actions are cheaper.
The discrete path (action_mag=None) is unchanged.
"""

from env.rewards import compute_shepherd_reward


WEIGHTS = {
    "efficiency": 0.0,
    "safety": 0.0,
    "stability": 0.0,
    "fairness": 0.0,
    "control_cost": 1.0,
    "easy_intervention_tax": 1.0,
}


def _reward(mode, mag=None, gap=None, full=1.5):
    return compute_shepherd_reward(
        "cav0",
        mode,
        33.0,
        WEIGHTS,
        120.0,
        cav_ids={"cav0"},
        stream_delay_gap=gap,
        stream_delay_gap_ref=0.0,
        action_mag=mag,
        action_mag_full=full,
    )


def test_full_mag_pays_full_cost():
    # |a| = a_full -> scale 1.0 -> same as binary cost -1.
    assert _reward(1, mag=1.5) == -1.0
    assert _reward(1, mag=3.0) == -1.0  # clipped at full


def test_half_mag_pays_floored_share():
    # floor 0.3: scale = 0.3 + 0.7 * 0.5 = 0.65
    assert abs(_reward(1, mag=0.75) - (-0.65)) < 1e-9


def test_tiny_mag_pays_floor_plus_share():
    # mag 0.15/1.5 = 0.1 -> scale 0.3 + 0.7*0.1 = 0.37
    assert abs(_reward(1, mag=0.15) - (-0.37)) < 1e-9


def test_coast_mode_no_cost_even_with_mag():
    # The env maps |a| <= 1e-3 to mode 0 before reward; cost stays 0.
    assert _reward(0, mag=0.0) == 0.0
    assert _reward(0, mag=0.0005) == 0.0


def test_easy_tax_scales_too():
    # Easy corridor (|gap| <= ref=0.0): tax -1 scaled by the floored factor.
    assert abs(_reward(1, mag=0.75, gap=0.0) - (-0.65 - 0.65)) < 1e-9
    assert abs(_reward(1, mag=1.5, gap=0.0) - (-1.0 - 1.0)) < 1e-9


def test_discrete_path_unchanged():
    # action_mag=None: legacy binary behavior.
    assert _reward(1, mag=None) == -1.0
    assert _reward(0, mag=None) == 0.0
    assert _reward(1, mag=None, gap=0.0) == -2.0
