"""Executable local actor/history/actuator composition, separate from legacy PPO.

This controller is inference plumbing, not an accepted trained checkpoint.
The old multi-agent reward/termination path is deliberately not selected here.
"""

from pathlib import Path

import numpy as np
import torch

from env.native_damping import DampingContract, LocalHistory
from env.native_damping_sumo import SumoDampingActuator, SumoLocalSensor
from idm.damping_actor import SharedDampingActor


CHECKPOINT_SCHEMA = "sensor_local_damping_weights/1"


class SensorLocalDampingController:
    def __init__(
        self,
        api,
        *,
        contract: DampingContract = DampingContract(),
        actor: SharedDampingActor | None = None,
        hidden: int = 64,
        lane_end_distance_to_merge=None,
    ):
        self.api = api
        self.contract = contract
        self.actor = actor if actor is not None else SharedDampingActor(contract, hidden)
        if self.actor.contract.manifest() != contract.manifest():
            raise ValueError("actor/controller contract mismatch")
        self.actor.eval()
        self.sensor = SumoLocalSensor(api, contract, lane_end_distance_to_merge)
        self.history = LocalHistory(contract)
        self.actuator = SumoDampingActuator(api, contract)
        self.last_clock: float | None = None
        self.previous_cavs: set[str] = set()

    def step(self, cav_ids: set[str], *, now_s: float) -> dict:
        if self.last_clock is not None:
            if not np.isclose(
                now_s - self.last_clock, self.contract.control_dt_s, atol=1e-8, rtol=0
            ):
                raise ValueError("controller must run exactly once per control interval")
        execution = self.actuator.observe_execution(now_s=now_s, live_cav_ids=cav_ids)
        for cav in self.previous_cavs - cav_ids:
            self.history.drop(cav)
        frames, observations = {}, {}
        for cav in sorted(cav_ids):
            frames[cav] = self.sensor.capture(
                cav,
                now_s=now_s,
                previous_command_delta=self.actuator.previous_command_delta.get(cav, 0),
            )
            self.history.add(frames[cav], now_s=now_s)
            observations[cav] = self.history.vector(cav, now_s=now_s)
        actions = {}
        if observations:
            device = next(self.actor.parameters()).device
            batch = torch.as_tensor(np.stack(list(observations.values())), device=device)
            with torch.no_grad():
                values = self.actor(batch).cpu().numpy().reshape(-1)
            actions = {cav: float(value) for cav, value in zip(observations, values)}
        commands = self.actuator.apply(frames, actions, now_s=now_s, live_cav_ids=cav_ids)
        self.last_clock, self.previous_cavs = now_s, set(cav_ids)
        return {
            "observations": observations,
            "actions": actions,
            "commands": commands,
            "previous_execution": execution,
        }

    def save_weights(self, path: Path) -> None:
        """Save only inference weights; does not mark scientific acceptance."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("checkpoint weights must not overwrite existing evidence")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "controller": self.contract.manifest(),
            "hidden": self.actor.mean_head.in_features,
            "observation_map": self.sensor.lane_end_distance_to_merge,
            "accepted_checkpoint": False,
            "actor_state": self.actor.state_dict(),
        }
        # Exclusive creation avoids simultaneous writers replacing a checkpoint.
        with path.open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def from_weights(
        cls,
        api,
        path: Path,
        *,
        contract: DampingContract = DampingContract(),
        lane_end_distance_to_merge=None,
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported damping checkpoint schema")
        if payload.get("controller") != contract.manifest():
            raise ValueError("checkpoint observation/action contract mismatch")
        saved_map = payload["observation_map"]
        if lane_end_distance_to_merge is not None and dict(lane_end_distance_to_merge) != saved_map:
            raise ValueError("checkpoint observation map mismatch")
        actor = SharedDampingActor(contract, hidden=payload["hidden"])
        actor.load_state_dict(payload["actor_state"], strict=True)
        return cls(
            api,
            contract=contract,
            actor=actor,
            lane_end_distance_to_merge=saved_map,
        )
