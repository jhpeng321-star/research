from pathlib import Path

import pytest

from tools.evaluate_idm_decentralized_clean_smoke import (
    _checkpoint_gate,
    _environment_config,
    _validate_inputs,
)
from tools.screen_idm_decentralized_activation_threshold import (
    _activation_probability,
    _probability_summary,
)


def test_analytical_arm_derives_observation_dimension_without_action_mask(tmp_path):
    config = _environment_config(
        seed=4107,
        output=tmp_path,
        label="analytical",
        decentralized=False,
        centralized=False,
    )

    assert config["action_masking"] is False
    assert config["idm_residual_decentralized_damping_enabled"] is False
    assert "local_obs_dim" not in config


def test_checkpoint_gate_requires_improvement_and_bounds_against_every_baseline():
    baseline = {
        "hard_braking": 10.0,
        "low_ttc": 4.0,
        "collisions": 0.0,
        "teleports": 0.0,
        "mean_time_loss": 20.0,
        "arrived": 100.0,
        "throughput": 400.0,
    }
    summary = {
        "candidate": {**baseline, "hard_braking": 8.0},
        "native": baseline,
        "forced": {**baseline, "hard_braking": 9.0},
        "analytical": {**baseline, "hard_braking": 11.0},
    }

    gate = _checkpoint_gate(
        summary, "candidate", ["native", "forced", "analytical"]
    )

    assert gate["pass"] is True
    assert all(row["pass"] for row in gate["comparators"].values())


def test_clean_smoke_rejects_nondevelopment_seed(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoints = {"checkpoint": run_dir / "checkpoints" / "iter_1"}
    checkpoints["checkpoint"].mkdir(parents=True)

    with pytest.raises(ValueError, match="development-only"):
        _validate_inputs(run_dir, checkpoints, [4201])


def test_binary_activation_probability_uses_softmax_logits():
    assert _activation_probability([0.0, 0.0]) == pytest.approx(0.5)
    assert _activation_probability([0.0, 1.0]) == pytest.approx(0.73105858)


def test_probability_summary_preserves_deployment_range():
    summary = _probability_summary([0.4, 0.5, 0.6])

    assert summary["count"] == 3
    assert summary["minimum"] == pytest.approx(0.4)
    assert summary["median"] == pytest.approx(0.5)
    assert summary["maximum"] == pytest.approx(0.6)
