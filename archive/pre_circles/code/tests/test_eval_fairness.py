"""Tests for the Task 6 fairness fixes.

- no_op baseline must run in shepherd control mode (action 0 ==
  non_intervention, handing control back to SUMO) instead of
  continuous_acceleration where action 0 locks the current speed.
- ``evaluate_sumo_baseline`` multi-seed aggregation (mean +/- std) and the
  driver-type mixture check (~10/70/20 from the route vTypeDistribution).
- ``tools/compare_to_hdv.judge_cell``: the "paper win" rule — efficiency (TT
  or throughput) improves AND safety (TTC + collisions + hard braking) does
  not regress AND the improvement exceeds the All-HDV seed std.
"""

from pathlib import Path

import pytest

from evaluate_policy import _env_config_for_policy
from evaluate_sumo_baseline import aggregate_seed_metrics, check_type_mixture
from policies import ConstantSpeedPolicy, NoOpPolicy, RandomPolicy
from tools.compare_to_hdv import group_summary_rows, judge_cell


def _cfg_for(policy_name, policy):
    return _env_config_for_policy(
        policy=policy,
        policy_name=policy_name,
        density="light",
        project_root=Path("."),
        sumo_binary="sumo",
        cav_penetration=0.0,
        max_episode_steps=None,
        hdv_mixture=None,
    )


# ---------------------------------------------------------------------------
# no_op trap fix
# ---------------------------------------------------------------------------
def test_no_op_runs_in_shepherd_control_mode():
    cfg = _env_config_for_policy(
        policy=NoOpPolicy(),
        policy_name="no_op",
        density="light",
        project_root=Path("."),
        sumo_binary="sumo",
        cav_penetration=0.0,
        max_episode_steps=None,
        hdv_mixture=None,
    )
    assert cfg["control_mode"] == "shepherd"


def test_no_op_action_is_non_intervention_mode():
    from config.constants import Mode

    actions = NoOpPolicy().act({"cav.1": None})
    assert actions == {"cav.1": int(Mode.COAST)}


# ---------------------------------------------------------------------------
# random / constant_speed baseline trap fixes (Task 12)
# ---------------------------------------------------------------------------
def test_random_runs_in_shepherd_control_mode():
    # RandomPolicy samples a Shepherd mode per CAV; under the continuous
    # default its reset() reads action_space.n off a Box -> AttributeError.
    cfg = _cfg_for("random", RandomPolicy(seed=42))
    assert cfg["control_mode"] == "shepherd"
    assert cfg["reward_mode"] == "shepherd"


def test_constant_speed_runs_in_shepherd_control_mode():
    # ConstantSpeedPolicy emits non_intervention (hands control back to SUMO);
    # under continuous mode that 0 would be misread as "0 m/s^2" speed lock.
    cfg = _cfg_for("constant_speed", ConstantSpeedPolicy())
    assert cfg["control_mode"] == "shepherd"
    assert cfg["reward_mode"] == "shepherd"


def test_constant_speed_action_is_non_intervention_mode():
    from config.constants import Mode

    actions = ConstantSpeedPolicy().act({"cav.1": None})
    assert actions == {"cav.1": int(Mode.COAST)}


# ---------------------------------------------------------------------------
# RL speedMode safe-speed clamp wiring (Task 12, Fix A)
# ---------------------------------------------------------------------------
def test_env_config_overrides_rl_speed_mode_with_safe_clamp():
    # DEFAULT_CONFIG still pins rl_speed_mode=6 (regard accel/decel only);
    # evaluation runs must override it with the clamped RL_SPEED_MODE so SUMO
    # caps RL setSpeed() commands at the safe car-following speed.
    from config.constants import RL_SPEED_MODE

    assert RL_SPEED_MODE & 1 == 1
    for name in ("selfish", "flat_marl", "rllib", "no_op"):
        cfg = _cfg_for(name, NoOpPolicy())
        assert cfg["rl_speed_mode"] == RL_SPEED_MODE


# ---------------------------------------------------------------------------
# Multi-seed aggregation for the All-HDV baseline
# ---------------------------------------------------------------------------
def _run(tt_all, throughput, ttc, hard_braking, collisions, teleports):
    return {
        "travel_time": {
            "all": {"mean": tt_all, "p90": tt_all + 10.0},
            "mainline": {"mean": tt_all - 1.0, "p90": tt_all + 9.0},
            "ramp": {"mean": tt_all + 2.0, "p90": tt_all + 12.0},
        },
        "ttc": {"total_events": ttc},
        "hard_braking": {"total_events": hard_braking},
        "throughput_veh_per_hour": throughput,
        "mean_speed": 28.0,
        "mean_time_loss": 3.0,
        "collisions": collisions,
        "teleports": teleports,
        "arrived": 400,
        "sim_seconds": 963.0,
    }


def test_aggregate_seed_metrics_mean_and_sample_std():
    agg = aggregate_seed_metrics([_run(50.0, 1500, 3, 1, 0, 0), _run(54.0, 1400, 5, 3, 1, 0)])
    assert agg["all_tt_mean"]["mean"] == pytest.approx(52.0)
    # sample std (ddof=1) of [50, 54] = sqrt(8)
    assert agg["all_tt_mean"]["std"] == pytest.approx(8.0**0.5)
    assert agg["all_tt_mean"]["values"] == [50.0, 54.0]
    assert agg["throughput_veh_per_hour"]["mean"] == pytest.approx(1450.0)
    assert agg["ttc_events"]["mean"] == pytest.approx(4.0)
    assert agg["hard_braking_events"]["mean"] == pytest.approx(2.0)
    assert agg["collisions"]["mean"] == pytest.approx(0.5)
    assert agg["teleports"]["mean"] == 0.0
    assert agg["mainline_tt_mean"]["mean"] == pytest.approx(51.0)
    assert agg["ramp_tt_mean"]["mean"] == pytest.approx(54.0)


def test_aggregate_seed_metrics_single_seed_has_zero_std():
    agg = aggregate_seed_metrics([_run(50.0, 1500, 3, 1, 0, 0)])
    assert agg["all_tt_mean"]["std"] == 0.0
    assert agg["all_tt_mean"]["mean"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Driver-type mixture check (~10/70/20 from the route vTypeDistribution)
# ---------------------------------------------------------------------------
def test_check_type_mixture_ok_within_tolerance():
    result = check_type_mixture({"aggressive_car": 0.12, "normal_car": 0.68, "cautious_car": 0.20})
    assert result["ok"] is True
    assert result["expected"] == {
        "aggressive_car": 0.10,
        "normal_car": 0.70,
        "cautious_car": 0.20,
    }


def test_check_type_mixture_flags_bad_mix():
    result = check_type_mixture({"aggressive_car": 0.50, "normal_car": 0.30, "cautious_car": 0.20})
    assert result["ok"] is False
    assert result["deviations"]["aggressive_car"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# "paper win" judging
# ---------------------------------------------------------------------------
def _hdv(tt=55.0, tt_std=2.0, thr=1500.0, thr_std=30.0, ttc=10.0, coll=0.0, hb=5.0):
    return {
        "tt_mean": tt,
        "tt_std": tt_std,
        "throughput_mean": thr,
        "throughput_std": thr_std,
        "ttc": ttc,
        "collisions": coll,
        "hard_braking": hb,
    }


def _pol(tt, thr, ttc=10.0, coll=0.0, hb=5.0):
    return {
        "tt_mean": tt,
        "throughput_mean": thr,
        "ttc": ttc,
        "collisions": coll,
        "hard_braking": hb,
    }


def test_judge_clear_paper_win():
    r = judge_cell(_pol(50.0, 1520.0), _hdv())
    assert r["win"] is True
    assert r["efficiency_improved"] is True
    assert r["safety_ok"] is True
    assert r["significant"] is True


def test_judge_ttc_regression_blocks_win():
    r = judge_cell(_pol(50.0, 1520.0, ttc=25.0), _hdv())
    assert r["win"] is False
    assert r["efficiency_improved"] is True
    assert r["safety_ok"] is False


def test_judge_collision_regression_blocks_win():
    r = judge_cell(_pol(50.0, 1520.0, coll=1.0), _hdv())
    assert r["win"] is False
    assert r["safety_ok"] is False


def test_judge_hard_braking_regression_blocks_win():
    r = judge_cell(_pol(50.0, 1520.0, hb=9.0), _hdv())
    assert r["win"] is False
    assert r["safety_ok"] is False


def test_judge_noise_level_gain_is_not_significant():
    # ΔTT = 1.0s < seed std 2.0s and Δthroughput = 5 < std 30.
    r = judge_cell(_pol(54.0, 1505.0), _hdv())
    assert r["win"] is False
    assert r["efficiency_improved"] is True
    assert r["significant"] is False


def test_judge_no_efficiency_gain():
    r = judge_cell(_pol(56.0, 1490.0), _hdv())
    assert r["win"] is False
    assert r["efficiency_improved"] is False


def test_judge_throughput_only_gain_can_win():
    # ΔTT = 0.5s < std 2.0s, but Δthroughput = 60 veh/h > std 30 veh/h.
    r = judge_cell(_pol(54.5, 1560.0), _hdv())
    assert r["win"] is True
    assert r["significant"] is True


def test_judge_zero_std_single_seed_baseline():
    # With a single-seed All-HDV baseline any positive Δ counts as significant.
    r = judge_cell(_pol(54.0, 1501.0), _hdv(tt_std=0.0, thr_std=0.0))
    assert r["win"] is True


def test_judge_zero_std_baseline_warns_unreliable_significance(capsys):
    # A zero seed std means a single-seed (or degenerate) baseline: the
    # significance gate then passes on any positive delta, so judge_cell must
    # warn on stderr that the verdict is unreliable.
    r = judge_cell(_pol(54.0, 1501.0), _hdv(tt_std=0.0, thr_std=0.0))
    assert r["win"] is True
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "significance" in err.lower() or "significant" in err.lower()


# ---------------------------------------------------------------------------
# Missing-data gates: absent safety/efficiency fields must fail closed
# (old summary rows lack the safety fields entirely; group_summary_rows
# yields NaN for them, and raw dicts may lack the keys).
# ---------------------------------------------------------------------------
def test_judge_nan_safety_values_block_win():
    pol = _pol(50.0, 1520.0, ttc=float("nan"), coll=float("nan"), hb=float("nan"))
    r = judge_cell(pol, _hdv())
    assert r["safety_ok"] is False
    assert r["win"] is False
    assert any("safety data: missing" in reason for reason in r["reasons"])


def test_judge_absent_safety_keys_block_win():
    pol = {"tt_mean": 50.0, "throughput_mean": 1520.0}  # no safety keys at all
    r = judge_cell(pol, _hdv())
    assert r["safety_ok"] is False
    assert r["win"] is False
    assert any("safety data: missing" in reason for reason in r["reasons"])


def test_judge_hdv_side_missing_safety_blocks_win():
    r = judge_cell(_pol(50.0, 1520.0), _hdv(ttc=None))
    assert r["safety_ok"] is False
    assert r["win"] is False
    assert any("safety data: missing" in reason for reason in r["reasons"])


def test_judge_missing_efficiency_values_block_win():
    pol = {
        "tt_mean": None,
        "throughput_mean": None,
        "ttc": 5.0,
        "collisions": 0.0,
        "hard_braking": 2.0,
    }
    r = judge_cell(pol, _hdv())
    assert r["efficiency_improved"] is False
    assert r["win"] is False
    assert any("efficiency data: missing" in reason for reason in r["reasons"])


def test_old_format_summary_row_does_not_win():
    # Old batch_eval rows carry travel-time means only — no safety/throughput
    # fields — and must never fabricate a win.
    rows = [
        {
            "ok": True,
            "baseline": "shepherd",
            "density": "light",
            "cav_penetration": 0.10,
            "all_mean": 50.0,
        }
    ]
    cells = group_summary_rows(rows)
    r = judge_cell(cells[("shepherd", "light", 0.10)], _hdv())
    assert r["win"] is False
    assert r["safety_ok"] is False
    assert any("safety data: missing" in reason for reason in r["reasons"])
