"""Regression tests: evaluation safety gates must fail closed, not silent-zero.

- Safety counters (collisions / teleports) come from SUMO's statistic-output
  XML. If that file is missing, the run's safety data must be marked missing
  instead of silently reporting 0 collisions.
- Fix 3: the evaluation seed must reach SUMO as ``--seed`` so TraCI runs pair
  with the seeded pure-SUMO All-HDV baseline runs.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import traci

import env.core as env_core
import evaluate_policy
from config.defaults import DEFAULT_CONFIG
from env.sumo_utils import start_sumo
from evaluate_policy import run_policy_evaluation
from policies import NoOpPolicy


class _FakeEnv:
    """Stand-in for TrafficShepherdMultiAgentEnv (no SUMO process)."""

    def __init__(self, env_config, workdir: Path, steps: int = 3, with_stats: bool = True):
        self.env_config = env_config
        self.action_space = gym.spaces.Discrete(5)
        self.eval_tripinfo_output = workdir / "tripinfos.eval.xml"
        self.eval_fcd_output = workdir / "fcd.eval.xml"
        self.eval_tripinfo_output.touch()
        self.eval_fcd_output.touch()
        self.eval_stats_output = workdir / "stats.eval.xml"
        if with_stats:
            self.eval_stats_output.touch()
        self._steps_left = steps
        self._total_steps = steps

    def reset(self, seed=None):
        return {}, {"episode_time": 0.0}

    def step(self, action_dict):
        self._steps_left -= 1
        done = self._steps_left <= 0
        sim_time = float(self._total_steps - self._steps_left)
        return {}, {}, {"__all__": done}, {"__all__": done}, {"episode_time": sim_time}

    def close(self):
        pass


def _run_eval_with_fake_env(monkeypatch, tmp_path, seed=42, with_stats=True):
    """Run run_policy_evaluation against _FakeEnv; capture evaluate_run kwargs."""
    captured = {}
    holder = {}

    def env_factory(env_config):
        env = _FakeEnv(env_config, tmp_path, with_stats=with_stats)
        holder["env"] = env
        return env

    def fake_evaluate_run(tripinfo_path, fcd_path, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(evaluate_policy, "TrafficShepherdMultiAgentEnv", env_factory)
    monkeypatch.setattr(evaluate_policy, "evaluate_run", fake_evaluate_run)

    metrics = run_policy_evaluation(
        policy=NoOpPolicy(),
        density="light",
        project_root=tmp_path,
        seed=seed,
        max_episode_steps=None,
        cav_penetration=0.1,
        policy_name="no_op",
    )
    return metrics, captured, holder["env"]


# ---------------------------------------------------------------------------
# Missing statistic-output must not silently report 0 collisions
# ---------------------------------------------------------------------------
def test_missing_stats_output_marks_safety_data_missing(monkeypatch, tmp_path, capsys):
    metrics, captured, _ = _run_eval_with_fake_env(monkeypatch, tmp_path, with_stats=False)

    # No silent zero: stats_path must be withheld, safety marked missing.
    assert captured["stats_path"] is None
    assert metrics["safety_data"] == "missing"
    # And the failure must be audible on stderr at least once.
    assert "WARNING" in capsys.readouterr().err


def test_stats_output_present_marks_safety_data_ok(monkeypatch, tmp_path, capsys):
    metrics, captured, _ = _run_eval_with_fake_env(monkeypatch, tmp_path, with_stats=True)

    assert captured["stats_path"] is not None
    assert metrics["safety_data"] == "ok"
    assert "WARNING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Fix 3: SUMO seed passthrough (RL eval pairs with All-HDV seeds 42/43/44)
# ---------------------------------------------------------------------------
def test_start_sumo_command_line_includes_seed(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(traci, "start", lambda cmd, port: started.append(list(cmd)))
    monkeypatch.setattr(traci, "isLoaded", lambda: False)

    start_sumo("sumo", "cfg.sumocfg", tmp_path, seed=42)

    cmd = started[0]
    assert cmd[cmd.index("--seed") + 1] == "42"


def test_start_sumo_command_line_without_seed_has_no_flag(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(traci, "start", lambda cmd, port: started.append(list(cmd)))
    monkeypatch.setattr(traci, "isLoaded", lambda: False)

    start_sumo("sumo", "cfg.sumocfg", tmp_path)

    assert "--seed" not in started[0]


def test_default_config_sumo_seed_is_none():
    assert DEFAULT_CONFIG["sumo_seed"] is None


def test_evaluation_seed_reaches_env_config_as_sumo_seed(monkeypatch, tmp_path):
    _, _, env = _run_eval_with_fake_env(monkeypatch, tmp_path, seed=42)

    assert env.env_config["sumo_seed"] == 42


def _patched_traci_idle(monkeypatch):
    """Make traci look alive-but-empty so env.reset() runs without SUMO."""
    monkeypatch.setattr(traci, "isLoaded", lambda: False)
    monkeypatch.setattr(traci, "simulationStep", lambda t: None)
    monkeypatch.setattr(traci.simulation, "getTime", lambda: 0.0)
    monkeypatch.setattr(traci.simulation, "getMinExpectedNumber", lambda: 0)
    monkeypatch.setattr(traci.simulation, "getDepartedIDList", lambda: [])
    monkeypatch.setattr(traci.simulation, "getArrivedIDList", lambda: [])
    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: [])


def _make_env_with_fake_cav(monkeypatch, env_config):
    """Construct the env with CAV warmup/obs stubbed to succeed without SUMO."""
    env = env_core.TrafficShepherdMultiAgentEnv(env_config)
    monkeypatch.setattr(env, "_update_cav_set", lambda: setattr(env, "_active_cavs", {"cav_1"}))
    monkeypatch.setattr(
        env,
        "_get_observations",
        lambda: {"cav_1": np.zeros(env.observation_space.shape, dtype=np.float32)},
    )
    return env


def test_env_reset_forwards_sumo_seed_to_start_sumo(monkeypatch, tmp_path):
    calls = {}

    def fake_start_sumo(**kwargs):
        calls.update(kwargs)
        return 12345

    monkeypatch.setattr(env_core, "start_sumo", fake_start_sumo)
    _patched_traci_idle(monkeypatch)

    env = _make_env_with_fake_cav(monkeypatch, {"project_root": tmp_path, "sumo_seed": 42})
    env.reset(seed=42)

    assert calls["seed"] == 42


def test_env_reset_without_sumo_seed_passes_none(monkeypatch, tmp_path):
    calls = {}

    def fake_start_sumo(**kwargs):
        calls.update(kwargs)
        return 12345

    monkeypatch.setattr(env_core, "start_sumo", fake_start_sumo)
    _patched_traci_idle(monkeypatch)

    env = _make_env_with_fake_cav(monkeypatch, {"project_root": tmp_path})
    env.reset(seed=42)

    assert calls["seed"] is None
