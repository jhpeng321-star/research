"""Full-clock SUMO collector; actor lifetimes never terminate team returns."""

from bisect import bisect_right
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
import traci

from config.constants import DOWNSTREAM_EDGE, MERGE_EDGE, PRE_MERGE_EDGE, RAMP_EDGE
from env.native_damping import DampingContract, LocalHistory
from env.native_damping_sumo import SumoDampingActuator, SumoLocalSensor
from idm.damping_credit import TeamIntervalCost
from idm.damping_scenario import EPISODE_SECONDS, TripRequest


REGIONS = (PRE_MERGE_EDGE, MERGE_EDGE, DOWNSTREAM_EDGE, RAMP_EDGE)
CRITIC_DIM = 6 + 4 * len(REGIONS)


class RequestLedger:
    """Discrete SUMO-clock system-time accounting, including source waiting."""

    def __init__(self, requests: list[TripRequest]):
        self.requests = {r.vehicle_id: r for r in requests}
        if len(self.requests) != len(requests) or not requests:
            raise ValueError("requested population must be nonempty and unique")
        self.times = sorted(r.depart_s for r in requests)
        self.inserted: set[str] = set()
        self.arrived: set[str] = set()
        self.time = 0.0
        self.system_vehicle_seconds = 0.0
        self.source_vehicle_seconds = 0.0

    def due(self, now: float) -> int:
        return bisect_right(self.times, now)

    def advance(self, now: float, departed: list[str], arrived: list[str]) -> float:
        if now <= self.time:
            raise ValueError("ledger time must increase")
        for ids, seen in ((departed, self.inserted), (arrived, self.arrived)):
            if set(ids) - self.requests.keys() or set(ids) & seen or len(ids) != len(set(ids)):
                raise ValueError("unknown or duplicate lifecycle event")
        if set(arrived) - (self.inserted | set(departed)):
            raise ValueError("arrival before insertion")
        if any(self.requests[v].depart_s > now + 1e-6 for v in departed):
            raise ValueError("insertion before requested departure")
        dt = now - self.time
        # Requests inside this interval contribute only their remaining time.
        prior_due = self.due(self.time)
        new_times = self.times[prior_due : self.due(now)]
        requested_area = prior_due * dt + sum(now - value for value in new_times)
        # Arrival/insertion events are assigned to the reported tick boundary.
        # This convention is identical in every arm and has 0.5 s resolution.
        system = requested_area - len(self.arrived) * dt
        source = requested_area - len(self.inserted) * dt
        if min(system, source) < -1e-6:
            raise ValueError("negative service accounting")
        self.system_vehicle_seconds += max(0.0, system)
        self.source_vehicle_seconds += max(0.0, source)
        self.inserted.update(departed)
        self.arrived.update(arrived)
        self.time = now
        return max(0.0, system)


def critic_observation(api, ledger: RequestLedger, cav_ids: set[str], now_s: float) -> np.ndarray:
    population = len(ledger.requests)
    values = [
        now_s / EPISODE_SECONDS,
        ledger.due(now_s) / population,
        len(ledger.inserted) / population,
        len(ledger.arrived) / population,
        (ledger.due(now_s) - len(ledger.inserted)) / population,
        len(cav_ids) / 50,
    ]
    for edge in REGIONS:
        ids = api.edge.getLastStepVehicleIDs(edge)
        speeds = np.array([api.vehicle.getSpeed(v) for v in ids])
        values.extend(
            [
                len(ids) / 100,
                speeds.mean() / 30 if len(ids) else 0,
                speeds.std() / 10 if len(ids) else 0,
                float(np.mean(speeds < 5)) if len(ids) else 0,
            ]
        )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (CRITIC_DIM,) or not np.isfinite(result).all():
        raise ValueError("invalid privileged critic observation")
    return result


def observation_map(api) -> dict[str, float]:
    mapping = {}
    for edge in REGIONS:
        for index in range(api.edge.getLaneNumber(edge)):
            lane = f"{edge}_{index}"
            if edge == DOWNSTREAM_EDGE:
                merge_length = api.lane.getLength(f"{MERGE_EDGE}_0")
                mapping[lane] = -merge_length - api.lane.getLength(lane)
            elif edge == MERGE_EDGE:
                mapping[lane] = -api.lane.getLength(lane)
            else:
                mapping[lane] = 0.0
    return mapping


def collect_episode(
    scene: Path,
    requests: list[TripRequest],
    *,
    seed: int,
    sumo_binary: str,
    actor=None,
    critic=None,
    stochastic: bool = False,
    random_seed: int = 0,
    diagnostic_steps: int | None = None,
) -> dict:
    """One native or policy trajectory. Training callers require the full horizon.

    A diagnostic cap is explicit and never returns a team-terminal training
    sample. It exists solely for integration checks on already exposed seeds.
    """
    # Install launch protection before importing the launcher, as required by
    # the existing Windows interprocess SUMO startup contract.
    from experiments.idm import deterministic_sumo  # noqa: F401
    from env.sumo_utils import close_sumo, start_sumo

    cfg = DampingContract()
    horizon = EPISODE_SECONDS if diagnostic_steps is None else diagnostic_steps
    if not 1 <= horizon <= EPISODE_SECONDS:
        raise ValueError("invalid collection horizon")
    if actor is not None and actor.contract.manifest() != cfg.manifest():
        raise ValueError("actor contract mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(random_seed)
    if actor is not None:
        actor.eval()
    if critic is not None:
        critic.eval()
    started = time.monotonic()
    start_sumo(
        sumo_binary,
        str(scene),
        scene.parent,
        seed=seed,
        extra_args=[
            "--step-length",
            "0.5",
            "--end",
            str(EPISODE_SECONDS + 1),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ],
    )
    ledger = RequestLedger(requests)
    all_cavs = {r.vehicle_id for r in requests if r.is_cav}
    mapping = observation_map(traci)
    sensor = SumoLocalSensor(traci, cfg, mapping)
    actuator, history = SumoDampingActuator(traci, cfg), LocalHistory(cfg)
    actor_obs, actions, old_logp, actor_times, actor_ids = [], [], [], [], []
    critic_obs, values, intervals = [], [], []
    counts, reasons = Counter(), Counter()
    previous_cavs: set[str] = set()
    collision_previous: set[str] = set()
    tracking_abs_sum = 0.0
    ttc_below_3_vehicle_seconds = 0.0
    minimum_closing_ttc = None
    metric_min_gaps = {}
    try:
        for step in range(horizon):
            now = float(traci.simulation.getTime())
            if abs(now - step) > 1e-6:
                raise RuntimeError("collector/global-clock mismatch")
            cavs = all_cavs.intersection(traci.vehicle.getIDList())
            state = critic_observation(traci, ledger, cavs, now)
            critic_obs.append(state)
            with torch.no_grad():
                value = 0.0 if critic is None else float(critic(torch.from_numpy(state)).item())
            values.append(value)
            for cav in previous_cavs - cavs:
                history.drop(cav)
                counts["departed_actor_histories"] += 1
            previous_cavs = cavs
            frames, local_rows = {}, []
            if actor is not None:
                for cav in sorted(cavs):
                    frame = sensor.capture(
                        cav,
                        now_s=now,
                        previous_command_delta=actuator.previous_command_delta.get(cav, 0),
                    )
                    history.add(frame, now_s=now)
                    frames[cav] = frame
                    local_rows.append(history.vector(cav, now_s=now))
                action_dict = {}
                if local_rows:
                    batch = torch.as_tensor(np.asarray(local_rows), dtype=torch.float32)
                    with torch.no_grad():
                        if stochastic:
                            sample, logp = actor.sample(batch)
                        else:
                            sample = actor(batch)
                            logp = actor.log_prob(batch, sample)
                    action_dict = dict(zip(frames, sample.numpy().reshape(-1)))
                    actor_obs.extend(local_rows)
                    actions.extend(sample.numpy())
                    old_logp.extend(logp.numpy())
                    actor_times.extend([step] * len(frames))
                    actor_ids.extend(frames)
                commands = actuator.apply(frames, action_dict, now_s=now, live_cav_ids=cavs)
                for command in commands:
                    counts[command["operation"]] += 1
                    reasons.update(command.get("reasons", ()))
            braking = energy = service = 0.0
            collisions = teleports = 0
            for _ in range(2):
                traci.simulationStep()
                after = float(traci.simulation.getTime())
                live = traci.vehicle.getIDList()
                accelerations = np.array([traci.vehicle.getAcceleration(v) for v in live])
                # Privileged outcome instrumentation, never part of actor observations.
                # TraCI leader distance excludes ego minGap; restore bumper gap.
                speeds = {v: traci.vehicle.getSpeed(v) for v in live}
                for vehicle in live:
                    leader = traci.vehicle.getLeader(vehicle, 200.0)
                    if not leader or not leader[0] or leader[0] not in speeds:
                        continue
                    closing = speeds[vehicle] - speeds[leader[0]]
                    if closing <= 1e-6:
                        continue
                    if vehicle not in metric_min_gaps:
                        metric_min_gaps[vehicle] = traci.vehicle.getMinGap(vehicle)
                    gap = max(0.0, leader[1] + metric_min_gaps[vehicle])
                    ttc = gap / closing
                    minimum_closing_ttc = (
                        ttc if minimum_closing_ttc is None else min(minimum_closing_ttc, ttc)
                    )
                    ttc_below_3_vehicle_seconds += 0.5 * (ttc < 3.0)
                braking += 0.5 * float(np.sum(accelerations < -3.0))
                energy += 0.5 * float(np.dot(accelerations, accelerations))
                service += ledger.advance(
                    after,
                    list(traci.simulation.getDepartedIDList()),
                    list(traci.simulation.getArrivedIDList()),
                )
                colliding = set(traci.simulation.getCollidingVehiclesIDList())
                collisions += len(colliding - collision_previous)
                collision_previous = colliding
                teleports += traci.simulation.getStartingTeleportNumber()
            intervals.append(
                asdict(TeamIntervalCost(braking, energy, service, collisions, teleports))
            )
            after_cavs = all_cavs.intersection(traci.vehicle.getIDList())
            for execution in actuator.observe_execution(
                now_s=float(traci.simulation.getTime()), live_cav_ids=after_cavs
            ):
                if execution.get("execution_observed") and execution["tracking_error"] is not None:
                    tracking_abs_sum += abs(execution["tracking_error"])
                    counts["measured_control_intervals"] += 1
            if step % 300 == 299:
                print(
                    f"collector seed={seed} step={step + 1}/{horizon} arrived={len(ledger.arrived)}",
                    flush=True,
                )
        terminal = horizon == EPISODE_SECONDS
        if terminal:
            values.append(0.0)
        else:
            with torch.no_grad():
                state = critic_observation(
                    traci, ledger, all_cavs.intersection(traci.vehicle.getIDList()), horizon
                )
                values.append(
                    0.0 if critic is None else float(critic(torch.from_numpy(state)).item())
                )
        totals = {field: sum(row[field] for row in intervals) for field in intervals[0]}
        summary = {
            "scenario_identity": hashlib.sha256(
                json.dumps([asdict(request) for request in requests], sort_keys=True).encode()
                + scene.read_bytes()
                + (scene.parent / "requests.rou.xml").read_bytes()
            ).hexdigest(),
            "seed": seed,
            "global_control_steps": horizon,
            "team_terminal": terminal,
            "population": len(requests),
            "inserted": len(ledger.inserted),
            "arrived": len(ledger.arrived),
            "uninserted_due": ledger.due(horizon) - len(ledger.inserted),
            "unfinished_requested": ledger.due(horizon) - len(ledger.arrived),
            "system_seconds_per_requested_vehicle": ledger.system_vehicle_seconds / len(requests),
            "source_wait_vehicle_seconds": ledger.source_vehicle_seconds,
            "totals": totals,
            "ttc_below_3_vehicle_seconds": ttc_below_3_vehicle_seconds,
            "minimum_closing_ttc_s": minimum_closing_ttc,
            "commands": dict(counts),
            "reasons": dict(reasons),
            "mean_abs_command_tracking_error": tracking_abs_sum
            / max(1, counts["measured_control_intervals"]),
            "actor_samples": len(actor_times),
            "observation_map": mapping,
            "sumo_version": list(traci.getVersion()),
            "elapsed_seconds": time.monotonic() - started,
            "time_accounting": "requests_exact_with_lifecycle_at_0.5s_reported_tick_boundary",
        }
        return {
            "actor_observations": np.asarray(actor_obs, dtype=np.float32).reshape(
                -1, cfg.observation_dim
            ),
            "actions": np.asarray(actions, dtype=np.float32).reshape(-1, 1),
            "old_logp": np.asarray(old_logp, dtype=np.float32),
            "actor_times": np.asarray(actor_times, dtype=np.int64),
            "actor_ids": actor_ids,
            "critic_observations": np.asarray(critic_obs, dtype=np.float32),
            "old_values": np.asarray(values, dtype=np.float32),
            "intervals": intervals,
            "summary": summary,
        }
    finally:
        close_sumo()
