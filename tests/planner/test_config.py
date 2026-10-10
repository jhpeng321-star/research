"""PlannerConfig: PL01_PLAN defaults, frozen dataclass, validation."""
import dataclasses

import pytest

from traffic_shepherd.planner import PlannerConfig


def test_defaults_match_pl01_plan_table():
    cfg = PlannerConfig()
    assert cfg.horizon_s == 600.0
    assert cfg.trajectory_s == 60.0
    assert cfg.ttc_floor_s == 1.5
    assert cfg.abs_safety_quantile == 1.0
    assert cfg.rel_safety_tol_s == 0.0
    assert cfg.rel_safety_quantile == 0.90
    # PL-06 owner decision 2026-09-27 (option A): a native collision no
    # longer vetoes a plan whose own gates pass; it is logged instead.
    assert cfg.native_collision_vetoes_plan is False
    # PL-07 owner decision 2026-09-27: the v2 window-only relative rule.
    assert cfg.window_relative_nonincreasing is True
    # v2 wall assembly bar values owner-approved 2026-09-27.
    assert cfg.wall_min_p_assembly == 0.9
    assert cfg.wall_max_p_abort == 0.1
    assert cfg.abstain_safety_exception is True
    assert cfg.combined_plan_fallback is True
    assert cfg.decide_every_s == 1.0
    assert cfg.agreement == 0.80
    assert cfg.margin_floor_s == 60.0  # T-B #2 (owner): 60 veh*s floor
    assert cfg.margin_k == 1.0
    assert cfg.wall_agreement == 0.90
    assert cfg.wall_margin_k == 2.0
    assert cfg.group_cap == 5
    assert cfg.interaction_pos_m == 2.0
    assert cfg.interaction_speed_mps == 0.5
    assert cfg.abstain_min_observed == 0.8
    assert cfg.wall_speeds_mps == (12.0, 16.0, 20.0, 24.0)
    assert cfg.wall_release_options_s == (60.0, 120.0, 240.0, None)
    assert cfg.wall_open_hold_s == 3600.0
    assert cfg.wall_release_rule == "median_saving_gt0"
    assert cfg.wall_release_margin_s == 0.0
    assert cfg.wall_max_member_gap_m == 100.0
    assert cfg.max_walls is None
    assert cfg.wall_member_max_frame_age_s == 0.0


def test_frozen_and_replaceable():
    cfg = PlannerConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.group_cap = 3
    tuned = dataclasses.replace(cfg, group_cap=3, decide_every_s=30.0)
    assert tuned.group_cap == 3 and cfg.group_cap == 5


@pytest.mark.parametrize(
    "field, value",
    [
        ("horizon_s", 0.0),
        ("trajectory_s", -1.0),
        ("ttc_floor_s", -0.1),
        ("abs_safety_quantile", 1.5),
        ("rel_safety_tol_s", -0.1),
        ("rel_safety_quantile", -0.1),
        ("decide_every_s", 0.0),
        ("agreement", 1.1),
        ("wall_agreement", -0.1),
        ("margin_floor_s", -1.0),
        ("margin_k", -1.0),
        ("group_cap", 0),
        ("interaction_pos_m", 0.0),
        ("interaction_speed_mps", -1.0),
        ("abstain_min_observed", 1.5),
        ("wall_max_member_gap_m", -1.0),
        # PL-04e2: max_walls 0 is now LEGAL (walls off -- no proposals at
        # all); the first rejected value is -1
        ("max_walls", -1),
        ("wall_member_max_frame_age_s", -0.5),
        ("wall_release_margin_s", -0.1),
    ],
)
def test_validation_rejects_out_of_range(field, value):
    with pytest.raises(ValueError):
        PlannerConfig(**{field: value})


def test_wall_release_rule_vocabulary():
    assert PlannerConfig(
        wall_release_rule="switch_rule").wall_release_rule == "switch_rule"
    with pytest.raises(ValueError, match="wall_release_rule"):
        PlannerConfig(wall_release_rule="median_saving_ge0")


def test_valid_none_variants_accepted():
    cfg = PlannerConfig(trajectory_s=None, max_walls=None)
    assert cfg.trajectory_s is None and cfg.max_walls is None


def test_native_collision_vetoes_plan_bool_only():
    """PL-06: the native-collision veto switch accepts True/False only."""
    assert PlannerConfig(native_collision_vetoes_plan=True).native_collision_vetoes_plan is True
    assert PlannerConfig(native_collision_vetoes_plan=False).native_collision_vetoes_plan is False
    for bad in (1, 0, "True", None, 1.0):
        with pytest.raises(ValueError, match="native_collision_vetoes_plan"):
            PlannerConfig(native_collision_vetoes_plan=bad)


def test_window_relative_nonincreasing_bool_only():
    """PL-07 (owner 2026-09-27): the v2 window-only relative switch accepts
    True/False only; anything else is a config validation error."""
    assert (
        PlannerConfig(window_relative_nonincreasing=True).window_relative_nonincreasing is True
    )
    assert (
        PlannerConfig(window_relative_nonincreasing=False).window_relative_nonincreasing is False
    )
    for bad in (1, 0, "True", None, 1.0):
        with pytest.raises(ValueError, match="window_relative_nonincreasing"):
            PlannerConfig(window_relative_nonincreasing=bad)


def test_mode_typos_rejected_at_construction():
    """PL-09: mode is construction-validated vocabulary -- a case typo
    ('Live') and a missing-underscore typo ('closedloop') raise ValueError,
    so an unknown mode can only reach the G-1/G-2 gate via a config object
    doctored past this validation (see the gate's own refusal tests)."""
    for bad in ("Live", "closedloop"):
        with pytest.raises(ValueError, match="mode must be one of"):
            PlannerConfig(mode=bad)
    for good in ("offline", "controlled_experiment", "live", "closed_loop"):
        assert PlannerConfig(mode=good).mode == good
