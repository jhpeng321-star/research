"""Deterministic replay checks for simulator-grounded counterfactual branches.

The first requirement for a counterfactual treatment dataset is that two
branches with identical pre-treatment histories and identical treatments
produce identical futures.  This module validates the safest baseline for the
current project: replay each branch from time zero with the same SUMO seed,
CAV-selection seed, and action prefix.

It intentionally does not use TraCI save/load.  The current demand files use
``<flow>`` definitions, and SUMO does not save every car-following and
lane-changing internal state.  Full replay is slower, but it is the reference
implementation against which any future snapshot-loading optimization must be
checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


Scalar = float | int | str
VehicleState = Tuple[Scalar, ...]


def _stable_float(value: float, digits: int) -> Scalar:
    """Round a float for stable comparison while preserving non-finite values."""
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return round(value, digits)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StepSnapshot:
    """Compact but diagnostic state captured after one supervisor step."""

    relative_step: int
    sim_time: Scalar
    action: int
    supervisor_observation: Tuple[Scalar, ...]
    vehicles: Tuple[VehicleState, ...]
    active_cavs: Tuple[str, ...]
    collision_vehicle_ids: Tuple[str, ...]
    starting_teleport_vehicle_ids: Tuple[str, ...]
    digest: str
    vehicle_count: int
    mean_speed: Scalar
    active_time_loss_sum: Scalar

    def vehicle_map(self) -> Dict[str, VehicleState]:
        return {str(row[0]): row for row in self.vehicles}


@dataclass(frozen=True)
class ReplayTrace:
    """One replayed branch and its deterministic state sequence."""

    seed: int
    decision_step: int
    horizon_steps: int
    prefix_action: int
    post_action: int
    treatment_steps: Optional[int]
    reset_info: Mapping[str, Any]
    snapshots: Tuple[StepSnapshot, ...]
    terminated_early: bool
    controller_metadata: Mapping[str, Any] = field(default_factory=dict)
    controller_audits: Tuple[Mapping[str, Any], ...] = ()

    @property
    def completed_steps(self) -> int:
        return max(0, len(self.snapshots) - 1)

    @property
    def trace_digest(self) -> str:
        return _digest({"steps": [step.digest for step in self.snapshots]})

    def summary(self) -> Dict[str, Any]:
        final = self.snapshots[-1]
        return {
            "seed": self.seed,
            "decision_step": self.decision_step,
            "horizon_steps": self.horizon_steps,
            "prefix_action": self.prefix_action,
            "post_action": self.post_action,
            "treatment_steps": self.treatment_steps,
            "completed_steps": self.completed_steps,
            "terminated_early": self.terminated_early,
            "reset_info": dict(self.reset_info),
            "trace_digest": self.trace_digest,
            "controller_metadata": dict(self.controller_metadata),
            "final": {
                "relative_step": final.relative_step,
                "sim_time": final.sim_time,
                "vehicle_count": final.vehicle_count,
                "mean_speed": final.mean_speed,
                "active_time_loss_sum": final.active_time_loss_sum,
                "health": final.supervisor_observation[0],
            },
        }


def capture_step(
    traci_module: Any,
    *,
    relative_step: int,
    action: int,
    supervisor_observation: Sequence[float],
    active_cav_ids: Iterable[str] = (),
    digits: int = 6,
) -> StepSnapshot:
    """Capture the network state used for exact replay comparison.

    Vehicle rows contain only pre-existing SUMO state: ID, road, lane, lane
    position, speed, acceleration, cumulative time loss, route index, type,
    and car-following tau. Rows are sorted by vehicle ID so TraCI iteration
    order cannot affect the fingerprint.
    """
    rows = []
    speeds = []
    time_losses = []
    for vehicle_id in sorted(traci_module.vehicle.getIDList()):
        speed = float(traci_module.vehicle.getSpeed(vehicle_id))
        time_loss = float(traci_module.vehicle.getTimeLoss(vehicle_id))
        try:
            tau = float(traci_module.vehicle.getTau(vehicle_id))
        except Exception:
            tau = math.nan
        row: VehicleState = (
            str(vehicle_id),
            str(traci_module.vehicle.getRoadID(vehicle_id)),
            str(traci_module.vehicle.getLaneID(vehicle_id)),
            _stable_float(traci_module.vehicle.getLanePosition(vehicle_id), digits),
            _stable_float(speed, digits),
            _stable_float(traci_module.vehicle.getAcceleration(vehicle_id), digits),
            _stable_float(time_loss, digits),
            int(traci_module.vehicle.getRouteIndex(vehicle_id)),
            str(traci_module.vehicle.getTypeID(vehicle_id)),
            _stable_float(tau, digits),
        )
        rows.append(row)
        speeds.append(speed)
        time_losses.append(time_loss)

    obs = tuple(_stable_float(value, digits) for value in supervisor_observation)
    active_cavs = tuple(sorted(str(vehicle_id) for vehicle_id in active_cav_ids))
    try:
        collision_vehicle_ids = tuple(
            sorted(str(value) for value in traci_module.simulation.getCollidingVehiclesIDList())
        )
    except (AttributeError, RuntimeError):
        collision_vehicle_ids = ()
    try:
        starting_teleport_vehicle_ids = tuple(
            sorted(str(value) for value in traci_module.simulation.getStartingTeleportIDList())
        )
    except (AttributeError, RuntimeError):
        starting_teleport_vehicle_ids = ()
    sim_time = _stable_float(traci_module.simulation.getTime(), digits)
    payload = {
        "relative_step": int(relative_step),
        "sim_time": sim_time,
        "action": int(action),
        "supervisor_observation": obs,
        "vehicles": rows,
        "active_cavs": active_cavs,
        "collision_vehicle_ids": collision_vehicle_ids,
        "starting_teleport_vehicle_ids": starting_teleport_vehicle_ids,
    }
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    return StepSnapshot(
        relative_step=int(relative_step),
        sim_time=sim_time,
        action=int(action),
        supervisor_observation=obs,
        vehicles=tuple(rows),
        active_cavs=active_cavs,
        collision_vehicle_ids=collision_vehicle_ids,
        starting_teleport_vehicle_ids=starting_teleport_vehicle_ids,
        digest=_digest(payload),
        vehicle_count=len(rows),
        mean_speed=_stable_float(mean_speed, digits),
        active_time_loss_sum=_stable_float(sum(time_losses), digits),
    )


def _clean_reset_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep deterministic reset metadata and remove process-local details."""
    cleaned: Dict[str, Any] = {}
    for key in (
        "episode_time",
        "warmup_steps",
        "sumo_config",
        "paired_seed",
        "reset_attempt",
    ):
        if key in info:
            value = info[key]
            cleaned[key] = float(value) if key == "episode_time" else value
    if "active_cavs" in info:
        cleaned["active_cavs"] = sorted(str(x) for x in info["active_cavs"])
    return cleaned


@dataclass(frozen=True)
class FrozenPulseSchedule:
    """One explicit post-decision wall pulse; no health-dependent treatment."""

    decision_step: int
    treatment_steps: int

    def __post_init__(self) -> None:
        if self.decision_step < 0:
            raise ValueError("decision_step must be non-negative")
        if self.treatment_steps <= 0:
            raise ValueError("treatment_steps must be positive")

    def __call__(self, relative_step: int) -> int:
        from policies.counterfactual_escalation import (
            DAMPING_ONLY,
            DAMPING_PLUS_WALL,
        )

        if self.decision_step < int(relative_step) <= (self.decision_step + self.treatment_steps):
            return int(DAMPING_PLUS_WALL)
        return int(DAMPING_ONLY)


def _json_safe(value: Any) -> Any:
    """Convert policy audit values to deterministic JSON-compatible objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    try:
        return _json_safe(value.item())
    except (AttributeError, TypeError, ValueError):
        return repr(value)


def _run_policy_replay(
    *,
    project_root: Path,
    sumo_binary: str,
    sumo_config: str,
    seed: int,
    decision_step: int,
    horizon_steps: int,
    treatment_steps: Optional[int],
    cav_penetration: float,
    sumo_step_length: float,
    supervisor_overrides: Optional[Mapping[str, Any]],
    digits: int,
    base_policy_factory: Callable[[], Any],
    intervention_schedule: Callable[[int], int],
    overlay_overrides: Optional[Mapping[str, Any]],
    environment_factory: Optional[Callable[[Mapping[str, Any]], Any]],
    controller_metadata: Optional[Mapping[str, Any]],
) -> ReplayTrace:
    """Replay the frozen damping policy with an explicit intervention schedule."""
    import traci

    from config.defaults import DEFAULT_CONFIG
    from env.wall_supervisor import corridor_state
    from policies.counterfactual_escalation import CounterfactualEscalationPolicy

    if environment_factory is None:
        from env.core import TrafficShepherdMultiAgentEnv

        environment_factory = TrafficShepherdMultiAgentEnv

    total_steps = decision_step + horizon_steps
    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": Path(project_root),
            "sumo_binary": str(sumo_binary),
            "sumo_config": str(sumo_config),
            "sumo_seed": int(seed),
            "cav_penetration": float(cav_penetration),
            "sumo_step_length": float(sumo_step_length),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
            # Frozen ``shepherd_pruned_v1_dense_mpen`` checkpoint contract.
            # The policy emits (mode, dial), so discrete shepherd silently
            # changes both the action and observation spaces.
            "control_mode": "shepherd_hybrid",
            "reward_mode": "shepherd",
            "centralized_critic": True,
            "use_v2_obs": True,
            "shepherd_modes": ["coast", "recover", "guard"],
            "shepherd_disabled_modes": [],
            "max_neighbors": 20,
            "cav_warmup_steps": 400,
            "max_cavs_critic": 32,
            "local_obs_dim": 124,
            "action_masking": True,
            "use_meter_load": False,
            "use_cav_count": False,
            "use_health_obs": False,
            "use_dv_obs": False,
            "use_margin_obs": False,
            "feasibility_mask": False,
            "feasibility_lmin_a": 0.0,
            "feasibility_lmin_b": 0.0,
            "fairness_r_ref": 0.45,
            "tail_r_ref": 0.53,
            "flow_q_ref": 0.95,
            "flow_v_ref": 17.5,
        }
    )
    env_config.update(dict(supervisor_overrides or {}))
    env = environment_factory(env_config)
    base_policy = base_policy_factory()
    policy = CounterfactualEscalationPolicy(
        base_policy=base_policy,
        **dict(overlay_overrides or {}),
    )

    snapshots = []
    audits = []
    terminated_early = False
    termination_reason: Optional[str] = None
    terminal_expected_vehicle_count: Optional[int] = None
    previous_health: Optional[float] = None
    health_history: list[float] = []
    inflow_tracker: Dict[str, Any] = {}
    engaged_steps = 0
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(
            env,
            None,
            0,
            0,
            0,
            inflow_tracker,
            health_history,
        )
        snapshots.append(
            capture_step(
                traci,
                relative_step=0,
                action=0,
                supervisor_observation=supervisor_observation,
                active_cav_ids=obs.keys(),
                digits=digits,
            )
        )
        audits.append(
            {
                "relative_step": 0,
                "command": 0,
                "command_name": "DAMPING_ONLY",
                "transition_reason": "reset",
            }
        )

        for relative_step in range(1, total_steps + 1):
            command = int(intervention_schedule(relative_step))
            policy.set_command(command)
            action_dict = policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit["relative_step"] = relative_step
            audits.append(audit)
            if bool(audit.get("wall_active")):
                engaged_steps += 1

            obs, _rewards, terminated, truncated, _infos = env.step(action_dict)
            supervisor_observation = corridor_state(
                env,
                previous_health,
                command,
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
                    action=command,
                    supervisor_observation=supervisor_observation,
                    active_cav_ids=obs.keys(),
                    digits=digits,
                )
            )
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                terminated_early = relative_step < total_steps
                termination_reason = str(_infos.get("end_reason") or "unknown")
                try:
                    terminal_expected_vehicle_count = int(
                        traci.simulation.getMinExpectedNumber()
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    terminal_expected_vehicle_count = None
                break

        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", str(sumo_config))
        reset_info.setdefault("paired_seed", int(seed))
        first_post_command = int(intervention_schedule(decision_step + 1))
        metadata = {
            "runner": "frozen_damping_with_explicit_overlay",
            "decision_step": int(decision_step),
            "treatment_steps": treatment_steps,
            "termination_reason": termination_reason or "horizon_reached",
            "terminal_expected_vehicle_count": terminal_expected_vehicle_count,
            **dict(controller_metadata or {}),
        }
        return ReplayTrace(
            seed=int(seed),
            decision_step=int(decision_step),
            horizon_steps=int(horizon_steps),
            prefix_action=0,
            post_action=first_post_command,
            treatment_steps=(int(treatment_steps) if treatment_steps is not None else None),
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


def run_replay(
    *,
    project_root: Path,
    sumo_binary: str,
    sumo_config: str,
    seed: int,
    decision_step: int,
    horizon_steps: int,
    prefix_action: int = 0,
    post_action: int = 0,
    treatment_steps: Optional[int] = None,
    cav_penetration: float = 0.2,
    sumo_step_length: float = 0.5,
    supervisor_overrides: Optional[Mapping[str, Any]] = None,
    digits: int = 6,
    base_policy_factory: Optional[Callable[[], Any]] = None,
    intervention_schedule: Optional[Callable[[int], int]] = None,
    overlay_overrides: Optional[Mapping[str, Any]] = None,
    environment_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    controller_metadata: Optional[Mapping[str, Any]] = None,
) -> ReplayTrace:
    """Replay one branch from time zero and return its state trace."""
    if decision_step < 0:
        raise ValueError("decision_step must be non-negative")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if treatment_steps is not None and treatment_steps <= 0:
        raise ValueError("treatment_steps must be positive when provided")
    if prefix_action not in (0, 1, 2) or post_action not in (0, 1, 2):
        raise ValueError("actions must be one of 0=off, 1=soft, 2=strong")

    using_policy_replay = base_policy_factory is not None or intervention_schedule is not None
    if using_policy_replay:
        if base_policy_factory is None or intervention_schedule is None:
            raise ValueError(
                "base_policy_factory and intervention_schedule must be provided together"
            )
        return _run_policy_replay(
            project_root=project_root,
            sumo_binary=sumo_binary,
            sumo_config=sumo_config,
            seed=seed,
            decision_step=decision_step,
            horizon_steps=horizon_steps,
            treatment_steps=treatment_steps,
            cav_penetration=cav_penetration,
            sumo_step_length=sumo_step_length,
            supervisor_overrides=supervisor_overrides,
            digits=digits,
            base_policy_factory=base_policy_factory,
            intervention_schedule=intervention_schedule,
            overlay_overrides=overlay_overrides,
            environment_factory=environment_factory,
            controller_metadata=controller_metadata,
        )

    # Imported lazily so comparison helpers remain unit-testable without SUMO.
    import traci
    from env.wall_supervisor import WallSupervisorEnv

    total_steps = decision_step + horizon_steps
    supervisor_config: Dict[str, Any] = {
        "num_actions": 3,
        "env_overrides": {
            "project_root": Path(project_root),
            "sumo_binary": str(sumo_binary),
            "sumo_config": str(sumo_config),
            "sumo_seed": int(seed),
            "cav_penetration": float(cav_penetration),
            "sumo_step_length": float(sumo_step_length),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
        },
    }
    supervisor_config.update(dict(supervisor_overrides or {}))
    env = WallSupervisorEnv(supervisor_config)

    snapshots = []
    terminated_early = False
    try:
        obs, info = env.reset(seed=seed)
        snapshots.append(
            capture_step(
                traci,
                relative_step=0,
                action=prefix_action,
                supervisor_observation=obs,
                active_cav_ids=env._last_obs.keys(),
                digits=digits,
            )
        )
        for relative_step in range(1, total_steps + 1):
            in_treatment_window = (
                treatment_steps is None or relative_step <= decision_step + treatment_steps
            )
            action = (
                prefix_action
                if relative_step <= decision_step
                else post_action
                if in_treatment_window
                else 0
            )
            obs, _reward, terminated, truncated, _step_info = env.step(action)
            snapshots.append(
                capture_step(
                    traci,
                    relative_step=relative_step,
                    action=action,
                    supervisor_observation=obs,
                    active_cav_ids=env._last_obs.keys(),
                    digits=digits,
                )
            )
            if terminated or truncated:
                terminated_early = relative_step < total_steps
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", str(sumo_config))
        reset_info.setdefault("paired_seed", int(seed))
        return ReplayTrace(
            seed=int(seed),
            decision_step=int(decision_step),
            horizon_steps=int(horizon_steps),
            prefix_action=int(prefix_action),
            post_action=int(post_action),
            treatment_steps=(int(treatment_steps) if treatment_steps is not None else None),
            reset_info=reset_info,
            snapshots=tuple(snapshots),
            terminated_early=terminated_early,
        )
    finally:
        env.close()


def _max_numeric_difference(left: Iterable[Scalar], right: Iterable[Scalar]) -> float:
    maximum = 0.0
    for a, b in zip(left, right):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            maximum = max(maximum, abs(float(a) - float(b)))
    return maximum


def compare_traces(left: ReplayTrace, right: ReplayTrace) -> Dict[str, Any]:
    """Compare two replay traces and diagnose the first divergence."""
    reset_match = dict(left.reset_info) == dict(right.reset_info)
    length_match = len(left.snapshots) == len(right.snapshots)
    first_mismatch: Optional[Dict[str, Any]] = None

    for step_left, step_right in zip(left.snapshots, right.snapshots):
        if step_left.digest == step_right.digest:
            continue
        left_vehicles = step_left.vehicle_map()
        right_vehicles = step_right.vehicle_map()
        all_ids = sorted(set(left_vehicles) | set(right_vehicles))
        changed_ids = [
            vehicle_id
            for vehicle_id in all_ids
            if left_vehicles.get(vehicle_id) != right_vehicles.get(vehicle_id)
        ]
        shared_ids = sorted(set(left_vehicles) & set(right_vehicles))
        vehicle_numeric_diff = 0.0
        for vehicle_id in shared_ids:
            vehicle_numeric_diff = max(
                vehicle_numeric_diff,
                _max_numeric_difference(left_vehicles[vehicle_id], right_vehicles[vehicle_id]),
            )
        first_mismatch = {
            "relative_step": step_left.relative_step,
            "left_sim_time": step_left.sim_time,
            "right_sim_time": step_right.sim_time,
            "left_action": step_left.action,
            "right_action": step_right.action,
            "left_vehicle_count": step_left.vehicle_count,
            "right_vehicle_count": step_right.vehicle_count,
            "changed_vehicle_count": len(changed_ids),
            "changed_vehicle_ids_sample": changed_ids[:20],
            "observation_max_abs_difference": _max_numeric_difference(
                step_left.supervisor_observation,
                step_right.supervisor_observation,
            ),
            "vehicle_numeric_max_abs_difference": vehicle_numeric_diff,
            "left_digest": step_left.digest,
            "right_digest": step_right.digest,
        }
        break

    if first_mismatch is None and not length_match:
        first_mismatch = {
            "relative_step": min(len(left.snapshots), len(right.snapshots)),
            "reason": "trace_length_mismatch",
            "left_snapshot_count": len(left.snapshots),
            "right_snapshot_count": len(right.snapshots),
        }

    passed = reset_match and length_match and first_mismatch is None
    return {
        "passed": passed,
        "reset_metadata_match": reset_match,
        "trace_length_match": length_match,
        "snapshot_count": min(len(left.snapshots), len(right.snapshots)),
        "left_trace_digest": left.trace_digest,
        "right_trace_digest": right.trace_digest,
        "first_mismatch": first_mismatch,
    }
