import json

from experiments.idm.run_idm_residual_checkpoint_eval import (
    _env_config,
    _residual_summary,
    build_summary,
)


def test_checkpoint_eval_env_preserves_idm_contract(tmp_path):
    cfg = _env_config(2200, 4107, tmp_path, "zero", 1800, None)
    assert cfg["control_mode"] == "idm_residual_smoothing"
    assert cfg["reward_mode"] == "idm_damping"
    assert cfg["idm_residual_reference_mode"] == "disturbance_p30"
    assert cfg["use_wave_signal"] is True
    assert cfg["use_idm_absorber_gate"] is True
    assert cfg["use_idm_controller_context"] is True
    assert cfg["idm_residual_filter_tau_s"] == 5.0
    assert "osm.dense.idm_m2200r400.sumocfg" in cfg["scenario_reward_refs"]
    assert cfg["centralized_critic"] is True
    assert cfg["local_obs_dim"] == 134
    assert cfg["max_cavs_critic"] == 48
    transition_cfg = _env_config(2700, 4107, tmp_path, "zero", 1800, None)
    assert transition_cfg["sumo_config"] == "osm.dense.idm_m2700r400.sumocfg"
    assert (
        transition_cfg["scenario_reward_refs"][transition_cfg["sumo_config"]]["demand_mainline_vph"]
        == 2700
    )


def test_residual_summary_reports_bounds_and_zero():
    empty = _residual_summary([])
    assert empty["count"] == 0
    assert empty["mean_mps2"] == 0.0
    history = [
        {
            "commands": [
                {"residual_mps2": -0.5},
                {"residual_mps2": 0.0},
                {"residual_mps2": 0.3},
            ]
        }
    ]
    summary = _residual_summary(history)
    assert summary["count"] == 3
    assert summary["minimum_mps2"] == -0.5
    assert summary["maximum_mps2"] == 0.3
    assert summary["near_lower_bound_fraction"] == 1 / 3
    assert summary["near_upper_bound_fraction"] == 1 / 3


def _row(arm, seed, wave, brakes, delay=10.0, arrived=100, collisions=0):
    return {
        "arm": arm,
        "demand_mainline_vph": 2200,
        "seed": seed,
        "trajectory": {
            "wave": {"oscillation_energy_m2ps3": wave},
            "safety": {"severe_braking_events": brakes},
        },
        "metrics": {
            "mean_time_loss": delay,
            "arrived": arrived,
            "collisions": collisions,
        },
        "residual": {
            "mean_mps2": 0.1 if arm != "zero" else 0.0,
            "mean_abs_mps2": 0.1 if arm != "zero" else 0.0,
            "near_lower_bound_fraction": 0.0,
            "near_upper_bound_fraction": 0.0,
        },
        "learning_reward": {
            "mean_per_selected_agent_step": 1.0 if arm != "zero" else 0.0,
            "component_means": {
                "local": 0.0,
                "efficiency": 0.0,
                "delay_budget": 0.0,
                "corridor_ttc": 0.0,
                "corridor_severe_braking": 0.0,
                "corridor_wave_energy": 0.0,
                "wave": 0.0,
                "residual_effort": 0.0,
            },
        },
    }


def test_summary_requires_four_of_six_and_service_safety(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    cases = []
    for seed in range(4107, 4113):
        for arm in ("zero", "candidate"):
            cases.append({"arm": arm, "demand": 2200, "seed": seed})
            row = _row(
                arm,
                seed,
                wave=100.0 if arm == "zero" else (80.0 if seed <= 4110 else 100.0),
                brakes=100.0,
            )
            (cases_dir / f"{arm}.m2200.seed{seed}.json").write_text(
                json.dumps(row), encoding="utf-8"
            )
    summary = build_summary(tmp_path, cases)
    gate = summary["candidate_gates"]["candidate"]
    assert gate["wave_wins_at_10pct"] == 4
    assert gate["development_pass"] is True
    assert summary["stage1_development_pass"] is True
