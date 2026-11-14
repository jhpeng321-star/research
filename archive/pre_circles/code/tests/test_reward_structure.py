"""Task 9: bounded density-independent R_safe, baseline-referenced R_fair,
gamma 0.995, and per-density fairness_r_ref injection.

R_safe neighborhood part is ``-2.0 * share of violating neighbors`` (bounded
[-2, 0], density-independent); ego part is hard brake -1, low TTC -1,
collision ``-safety_collision_penalty`` (default 6.5).
"""

import argparse
import re
from pathlib import Path
from unittest.mock import patch

import train
from env.rewards import TimeLossTracker, _fairness, _safety
from evaluate_policy import _env_config_for_policy
from policies import NoOpPolicy
from train import _build_env_config


# ---------------------------------------------------------------------------
# R_safe: density-independent, bounded neighborhood term
# ---------------------------------------------------------------------------
def _run_safety(neighbor_accels, neighbor_ttcs, ego_accel=0.0, ego_ttc=99.0, colliding=()):
    """Run _safety with per-vehicle accel/TTC maps (defaults: no violation)."""
    neighbors = list(neighbor_accels)
    accels = dict(neighbor_accels)
    accels["cav1"] = ego_accel
    ttcs = dict(neighbor_ttcs)
    ttcs["cav1"] = ego_ttc
    with (
        patch("env.rewards.traci") as mock_traci,
        patch("env.rewards._local_vehicles", return_value=neighbors),
        patch("env.rewards.leader_ttc", side_effect=lambda vid: ttcs.get(vid, 99.0)),
    ):
        mock_traci.vehicle.getAcceleration.side_effect = lambda vid: accels[vid]
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = list(colliding)
        return _safety("cav1", 200.0)


def test_safety_density_independent_same_violation_share():
    """Same violating-neighbor share -> same penalty, regardless of count."""
    # 1 of 2 neighbors hard-braking.
    few = _run_safety({"h1": -5.0, "h2": 0.0}, {})
    # 5 of 10 neighbors hard-braking.
    many = _run_safety({f"h{i}": (-5.0 if i < 5 else 0.0) for i in range(10)}, {})
    assert few == many == -1.0  # -2.0 * 0.5


def test_safety_neighbor_term_counts_each_violator_once():
    """A neighbor both hard-braking and low-TTC is still one violator."""
    r = _run_safety({"h1": -5.0, "h2": 0.0}, {"h1": 0.5})
    assert r == -1.0  # share 1/2, not 2 events


def test_safety_bounded_worst_case_includes_collision():
    """30 violating neighbors + ego brake + ego TTC + ego collision."""
    neighbors = {f"h{i}": -5.0 for i in range(30)}
    ttcs = {f"h{i}": 0.5 for i in range(30)}
    r = _run_safety(neighbors, ttcs, ego_accel=-5.0, ego_ttc=0.5, colliding=["cav1"])
    # ego (-1 brake -1 TTC -6.5 collision) + neighbor share (-2) = -10.5
    assert r == -10.5


def test_safety_zero_when_clean():
    r = _run_safety({f"h{i}": 0.0 for i in range(20)}, {})
    assert r == 0.0


# ---------------------------------------------------------------------------
# R_fair: signed improvement vs the All-HDV baseline rate
# ---------------------------------------------------------------------------
def _run_fairness(rates, r_ref=0.5):
    tracker = TimeLossTracker()
    tracker._rates = dict(rates)
    with patch("env.rewards._local_vehicles", return_value=list(rates)):
        return _fairness(
            "cav1",
            200.0,
            max_speed=35.0,
            cav_ids={"cav1"},
            time_loss_tracker=tracker,
            r_ref=r_ref,
        )


def test_fairness_positive_when_mean_delay_below_baseline():
    # mean = 0.15 < r_ref = 0.5 -> (0.5 - 0.15)/0.5 = +0.7.
    r = _run_fairness({"h1": 0.1, "h2": 0.2})
    assert r > 0.0
    assert abs(r - 0.7) < 1e-6


def test_fairness_negative_when_mean_delay_above_baseline():
    # mean = 0.7 > r_ref = 0.5 -> (0.5 - 0.7)/0.5 = -0.4.
    r = _run_fairness({"h1": 0.9, "h2": 0.5})
    assert r < 0.0
    assert abs(r - (-0.4)) < 1e-6


def test_fairness_bounded_on_both_sides():
    # Far worse than baseline: saturates at -1.0.
    worst = _run_fairness({"h1": 100.0, "h2": 50.0})
    assert worst == -1.0
    # Far better than baseline: mean -> 0 -> +1.0.
    best = _run_fairness({"h1": 1e-9, "h2": 1e-9})
    assert 0.99 < best <= 1.0
    # Every scenario stays within [-1.0, +1.0].
    for rates in ({"h1": 0.3, "h2": 0.7, "h3": 1.5}, {"h1": 0.01}):
        r = _run_fairness(rates)
        assert -1.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# gamma 0.99 -> 0.995 (train.py CLI default)
# ---------------------------------------------------------------------------
def test_gamma_default_is_0995():
    src = Path(train.__file__).read_text(encoding="utf-8")
    m = re.search(r'add_argument\(\s*"--gamma"[^)]*default=([0-9.]+)', src)
    assert m, "--gamma argument not found in train.py"
    assert float(m.group(1)) == 0.995


# ---------------------------------------------------------------------------
# Per-density fairness_r_ref: table + injection at both entry points
# ---------------------------------------------------------------------------
def test_fairness_r_ref_table_covers_all_densities():
    from config.defaults import FAIRNESS_R_REF_BY_DENSITY

    assert set(FAIRNESS_R_REF_BY_DENSITY) == {"light", "medium", "dense", "hyper"}
    for value in FAIRNESS_R_REF_BY_DENSITY.values():
        assert 0.0 < value < 1.0


def _train_args(density):
    return argparse.Namespace(
        sumo_binary="sumo",
        density=density,
        cav_penetration=0.1,
        hdv_mixture=None,
        max_episode_steps=500,
        max_neighbors=20,
        max_cavs_critic=32,
        baseline="selfish",
    )


def test_train_env_config_injects_fairness_r_ref():
    from config.defaults import FAIRNESS_R_REF_BY_DENSITY

    for density, r_ref in FAIRNESS_R_REF_BY_DENSITY.items():
        cfg = _build_env_config(_train_args(density))
        assert cfg["fairness_r_ref"] == r_ref


def test_eval_env_config_injects_fairness_r_ref():
    from config.defaults import FAIRNESS_R_REF_BY_DENSITY

    for density, r_ref in FAIRNESS_R_REF_BY_DENSITY.items():
        cfg = _env_config_for_policy(
            policy=NoOpPolicy(),
            policy_name="no_op",
            density=density,
            project_root=Path("."),
            sumo_binary="sumo",
            cav_penetration=0.0,
            max_episode_steps=None,
            hdv_mixture=None,
        )
        assert cfg["fairness_r_ref"] == r_ref
