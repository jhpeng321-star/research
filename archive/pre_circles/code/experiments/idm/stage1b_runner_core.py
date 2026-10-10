"""Generic replay-branch core for Stage 1b-prep (new code in the new file).

Reuses the Stage 1a machinery by import (``env_overrides``,
``_json_safe``, ``_clean_reset_info``, ``_terminal_classification``,
``_event_counts``, ``_branch_summary``) but carries its own scenario file and
policy factory so Part B (band locator, any cell) and Part C (hold arms) can
run with whichever policy the caller needs. The replay loop body is identical
to ``run_sustained_wall_v1_stage1a._run_branch``: prefix digests, corridor
health, and step snapshots are produced the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from config.defaults import DEFAULT_CONFIG
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step
from env.wall_supervisor import corridor_state
from experiments.idm.run_sustained_wall_v1_stage1a import (
    _branch_summary,
    _clean_reset_info,
    _event_counts,
    _json_safe,
    _terminal_classification,
    env_overrides,
)

import traci


def replay_branch(
    *,
    seed: int,
    arm: str,
    decision_step: Optional[int],
    scenario_file: str,
    scenario_root: Path,
    sumo_binary: str,
    total_steps: int,
    policy_factory: Callable[[], Any],
    cav_penetration: float,
) -> ReplayTrace:
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": scenario_root,
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_file,
            "sumo_seed": int(seed),
            "cav_penetration": float(cav_penetration),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy = policy_factory()

    snapshots: list[StepSnapshot] = []
    audits: list[Dict[str, Any]] = []
    health_history: list[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
    engaged_steps = 0
    terminated_early = False
    termination_reason: Optional[str] = None
    terminal_expected: Optional[int] = None
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(
            env, None, 0, 0, 0, inflow_tracker, health_history
        )
        snapshots.append(
            capture_step(
                traci,
                relative_step=0,
                action=0,
                supervisor_observation=supervisor_observation,
                active_cav_ids=obs.keys(),
                digits=6,
            )
        )
        audits.append(
            {"relative_step": 0, "command": 0, "treatment_state": "DAMPING", "wall_active": False}
        )
        for relative_step in range(1, total_steps + 1):
            action_dict = policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit.setdefault("relative_step", relative_step)
            wall_active = bool(audit.get("wall_active"))
            if wall_active:
                engaged_steps += 1
            audits.append(audit)
            obs, _rewards, terminated, truncated, _infos = env.step(action_dict)
            supervisor_observation = corridor_state(
                env,
                previous_health,
                1 if wall_active else 0,
                engaged_steps,
                relative_step,
                inflow_tracker,
                health_history,
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(
                capture_step(
                    traci,
                    relative_step=relative_step,
                    action=1 if wall_active else 0,
                    supervisor_observation=supervisor_observation,
                    active_cav_ids=obs.keys(),
                    digits=6,
                )
            )
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                terminated_early = relative_step < total_steps
                termination_reason = str(_infos.get("end_reason") or "unknown")
                try:
                    terminal_expected = int(traci.simulation.getMinExpectedNumber())
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    terminal_expected = None
                break

        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_file)
        reset_info.setdefault("paired_seed", int(seed))
        metadata = {
            "runner": "sustained_wall_v1_stage1b_core",
            "arm": arm,
            "decision_step": int(decision_step) if decision_step is not None else None,
            "termination_reason": termination_reason or "horizon_reached",
            "terminal_expected_vehicle_count": terminal_expected,
            "controller_step_s": 1.0,
        }
        return ReplayTrace(
            seed=int(seed),
            decision_step=int(decision_step) if decision_step is not None else 0,
            horizon_steps=total_steps - (int(decision_step) if decision_step is not None else 0),
            prefix_action=0,
            post_action=0,
            treatment_steps=None,
            reset_info=reset_info,
            snapshots=tuple(snapshots),
            terminated_early=terminated_early,
            controller_metadata=_json_safe(metadata),
            controller_audits=tuple(audits),
        )
    finally:
        try:
            policy.close()
        finally:
            env.close()


__all__ = [
    "replay_branch",
    "env_overrides",
    "_branch_summary",
    "_clean_reset_info",
    "_json_safe",
]
