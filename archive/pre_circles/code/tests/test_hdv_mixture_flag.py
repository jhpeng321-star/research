"""--hdv-mixture must re-enable TraCI-side HDV type assignment.

Driver types default to the route file's vTypeDistribution
(``assign_hdv_types`` is False in config/defaults.py, so env/core.py never
builds an HDVTypeAssigner). Passing an explicit ``--hdv-mixture`` means the
user wants TraCI-side assignment instead, so both entry points must set
``assign_hdv_types=True`` together with the proportions — otherwise the flag
is silently swallowed.
"""

import argparse
from pathlib import Path

from evaluate_policy import _env_config_for_policy
from policies import NoOpPolicy
from train import _build_env_config


# ---------------------------------------------------------------------------
# evaluate_policy._env_config_for_policy
# ---------------------------------------------------------------------------
def _eval_cfg(hdv_mixture):
    return _env_config_for_policy(
        policy=NoOpPolicy(),
        policy_name="no_op",
        density="light",
        project_root=Path("."),
        sumo_binary="sumo",
        cav_penetration=0.0,
        max_episode_steps=None,
        hdv_mixture=hdv_mixture,
    )


def test_eval_hdv_mixture_enables_traci_type_assignment():
    cfg = _eval_cfg([0.2, 0.6, 0.2])
    assert cfg["assign_hdv_types"] is True
    assert cfg["hdv_type_proportions"] == [0.2, 0.6, 0.2]


def test_eval_without_hdv_mixture_keeps_route_distribution():
    cfg = _eval_cfg(None)
    assert "assign_hdv_types" not in cfg
    assert "hdv_type_proportions" not in cfg


# ---------------------------------------------------------------------------
# train._build_env_config
# ---------------------------------------------------------------------------
def _train_args(hdv_mixture):
    return argparse.Namespace(
        sumo_binary="sumo",
        density="medium",
        cav_penetration=0.1,
        hdv_mixture=hdv_mixture,
        max_episode_steps=500,
        max_neighbors=20,
        max_cavs_critic=32,
        baseline="selfish",
        multi_density=False,
        curriculum_densities=None,
    )


def test_train_hdv_mixture_enables_traci_type_assignment():
    cfg = _build_env_config(_train_args([0.2, 0.6, 0.2]))
    assert cfg["assign_hdv_types"] is True
    assert cfg["hdv_type_proportions"] == [0.2, 0.6, 0.2]


def test_train_without_hdv_mixture_keeps_route_distribution():
    cfg = _build_env_config(_train_args(None))
    assert "assign_hdv_types" not in cfg
    assert "hdv_type_proportions" not in cfg
