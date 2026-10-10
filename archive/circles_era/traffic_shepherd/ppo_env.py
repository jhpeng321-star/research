"""Interactive SUMO environment for the RSU PPO pilot.

One episode = one registered corridor scenario driven at the existing
1 s control / 0.5 s physics grid. The loop reuses the paired-experiment
machinery (build_scenario, CirclesController, CommandOwnership, sensor, RSU
report buffer, RequestLedger, braking/effort/TTC collectors) so the per-tick
interval counters feeding the reward are exactly the quantities the outcome
evaluator sums. The pilot actor is consulted only at decision ticks
(t % cadence == 0) while the phase is inactive or hold; everything else is
the autonomous wall_ppo/1 contract. There is no early termination: the
registered horizon is always completed, so no rollout or truncation boundary
can cut delayed credit, and no credit resets at wall release.
"""
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import ppo_contract as C
from . import ppo_features, ppo_reward
from .damping import CirclesController
from .config import load_config
from .escalation_observations import OfflineTrafficObserver
from .lifecycle_wall import policy_factory as lifecycle_factory
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .paired_experiments import (PolicyInput, PolicyOutput, digest, _member_measurements)
from .reporting import CAVReport, RSUReportBuffer
from .scenarios import build_scenario, observation_map, scenario_identity
from .sensor_batch import capture_batch
from .sensing import SumoLocalSensor
from .sim_clock import ClockViolationError, ExactClock
from .simulation import close_sumo, start_sumo
from .selector_wall_policy import SelectorWallPolicy
from .wall_ppo_policy import DampingPolicy, ScriptedReportRulePolicy, WallPpoPolicy
from .workflow import sha, write_json


def build_pilot_job(*, seed, loading, horizon, case_id, policy_spec,
                    decision_time_s=None):
    decision = C.DECISION_TIME_S if decision_time_s is None else decision_time_s
    job = dict(id=f"s{seed}_m{loading}_p{int(C.PENETRATION*100)}_t{decision}_{case_id}",
               case_id=case_id, seed=seed, loading=loading, penetration=C.PENETRATION,
               role="exposed", arm="wall", duration_s=horizon,
               decision_time_s=decision, policy_spec=policy_spec)
    from .paired_experiments import validate_job
    validate_job(job)
    return job


class PpoEpisode:
    """One bounded, chargeable episode; never resumed or retried."""

    def __init__(self, *, job, registration, member_ids, required_lanes,
                 policy_mode, actor=None, output, budget=None):
        if policy_mode not in {"ppo", "selector", "scripted_rule", "fixed_wall",
                               "damping"}:
            raise ValueError("unknown pilot policy mode")
        if policy_mode == "ppo" and actor is None:
            raise ValueError("ppo episodes require a pilot actor callback")
        if policy_mode == "selector" and actor is None:
            # named scripted comparator (inactive: ACTIVATE-if-offered else
            # WAIT; interior: MAINTAIN; base deadline: EXTEND once; extended
            # deadline: RELEASE) — phase-aware, never raw action indices.
            # A supplied actor is used unchanged through the same boundary.
            from .selector_wall_policy import scripted_selector_chain_actor
            actor = scripted_selector_chain_actor
        self.job, self.registration = job, registration
        self.member_ids, self.required_lanes = tuple(member_ids), tuple(required_lanes)
        self.policy_mode, self.actor = policy_mode, actor
        self.output = Path(output).resolve()
        self.budget = budget
        self.horizon = job["duration_s"]
        if self.output.exists():
            raise FileExistsError("preserving existing evidence; never resume or retry")

    # ---- policy construction --------------------------------------------

    def _make_policy(self):
        if self.policy_mode == "damping":
            return DampingPolicy(self.job, self.registration)
        if self.policy_mode == "fixed_wall":
            return lifecycle_factory(self.job, self.registration)
        if self.policy_mode == "scripted_rule":
            return ScriptedReportRulePolicy(self.job, self.registration,
                                            self.member_ids, self.required_lanes)
        if self.policy_mode == "selector":
            # wall_ppo/3: cohort bound at activation from current reports;
            # hold semantics identical to wall_ppo/2
            return SelectorWallPolicy(self.job, self.registration)
        return WallPpoPolicy(self.job, self.registration, self.member_ids, self.required_lanes)

    # ---- main loop -------------------------------------------------------

    def run(self):
        job, config = self.job, load_config()
        output = self.output
        output.mkdir(parents=True, exist_ok=False)
        import traci
        if traci.isLoaded():
            raise RuntimeError("refusing to reuse a pre-existing TraCI connection")
        binary = shutil.which("sumo")
        if binary is None:
            raise RuntimeError("SUMO must be on PATH")
        controller = CirclesController(config)
        policy = self._make_policy()
        manifest = dict(schema="paired_escalation_arm/1", status="preparing",
            **self.registration["identity"], seed=job["seed"], mainline_vph=job["loading"],
            penetration=job["penetration"], role=job["role"], controller=f"pilot_{self.policy_mode}",
            duration_s=self.horizon, mechanics_only=self.horizon != config.episode_seconds,
            decision_time_s=job["decision_time_s"], job_id=job["id"], case_id=job["case_id"],
            registration_sha256=self.registration["registration_sha256"],
            policy_spec=job["policy_spec"], policy_mode=self.policy_mode,
            policy_version=C.POLICY_VERSION, contract_sha256=C.contract_sha256(),
            assignment=self.registration["assignment"],
            allocated_simulated_seconds=self.horizon, scientifically_accepted=False,
            offline_traffic_observations=True, communication=self.registration["communication"])
        write_json(output / "manifest.json", manifest)
        api, owner, started = None, None, time.monotonic()
        try:
            scene, requests, scenario = build_scenario(output / "scenario", job["seed"],
                job["loading"], penetration=job["penetration"], role=job["role"], config=config)
            manifest.update(scenario=scenario, scenario_identity=scenario_identity(
                scene, requests, network_path=config.network_path), status="running")
            manifest["exogenous_future_sha256"] = digest(dict(
                requests=[asdict(r) for r in requests], sumo_seed=job["seed"],
                binary_sha256=manifest["sumo_binary_sha256"],
                scenario_identity=manifest["scenario_identity"]))
            write_json(output / "manifest.json", manifest)
            def _launch():
                start_sumo(binary, scene, seed=job["seed"], config=config, api=traci)
            if self.budget is None:
                _launch()
            else:
                from .budget_guard import reserve_and_launch
                reserve_and_launch(self.budget, target_dir=Path(self.budget["target_dir"]),
                                   launcher=_launch)
            api = traci
            cav_ids = frozenset(r.vehicle_id for r in requests if r.is_cav)
            owner = CommandOwnership(api, cav_ids, controller)
            lane_map = observation_map(api, config)
            sensor = SumoLocalSensor(api, config, lane_map)
            observer = OfflineTrafficObserver(api, config)
            comm = self.registration["communication"]
            buffer = RSUReportBuffer(max_age_s=comm["max_report_age_s"], history_s=comm["history_s"])
            ledger, totals, command_counts, invariant_counts = RequestLedger(requests), Counter(), Counter(), Counter()
            prior_commands, history, collisions_previous = {}, [], set()
            policy_counts = Counter()
            filenames = ("telemetry", "intervals", "traffic", "policy", "reward_log")
            streams = {name: (output / f"{name}.jsonl").open("x", encoding="utf-8")
                       for name in filenames}
            def emit(name, row):
                streams[name].write(json.dumps(row, allow_nan=False) + "\n")
            rollout = []       # per-decision records for the PPO worker
            previous_features = None
            pending_reward = 0.0
            wall_owned_steps = 0
            clock = ExactClock(step_fn=api.simulationStep,
                time_fn=lambda: float(api.simulation.getTime()), dt=.5, start=0.,
                context="corridor:" + job["id"])
            try:
                for tick in range(self.horizon):
                    now = clock.check_now(float(api.simulation.getTime()),
                                          context=f"control tick {tick}")
                    live = set(api.vehicle.getIDList())
                    emit("traffic", observer.sample(now_s=now))
                    frames, errors, cache_stats = capture_batch(api, live & cav_ids, now_s=now,
                        config=config, lane_end_distance_to_merge=lane_map)
                    invariant_counts.update(cache_stats)
                    for cav, error in errors.items():
                        emit("telemetry", dict(kind="observation_unavailable", vehicle_id=cav,
                                               timestamp_s=now, reason=error))
                    received_packets = []
                    for cav, frame in sorted(frames.items()):
                        previous = prior_commands.get(cav)
                        previous_row = previous.as_dict() if previous else {}
                        report = CAVReport(frame, now, previous.owner if previous else "native",
                            previous_row.get("network_output_mps2"),
                            previous.issued_acceleration_mps2 if previous else None,
                            previous.fallback_reason if previous else None)
                        buffer.receive(report, received_at_s=now)
                        history.append((now, report))
                        received_packets.append(dict(received_at_s=now, **report.payload()))
                        emit("telemetry", dict(kind="rsu_received_report", received_at_s=now,
                                               **report.payload()))
                    history = [(receipt, r) for receipt, r in history if receipt >= now - comm["history_s"]]
                    expected = {v: controller.decide(f, now_s=now) for v, f in frames.items()}
                    references = {v: d.command_acceleration_mps2 for v, d in expected.items()
                                  if d.fallback_reason is None}
                    native_references = frozenset(v for v, d in expected.items()
                                                  if d.fallback_reason is not None)
                    policy_input = PolicyInput(now, tuple(buffer.reports(now_s=now)),
                        tuple(history), references, native_references)
                    # ---- pilot decision boundary --------------------------------
                    # decision_context gates internally: inactive/interior on
                    # the cadence grid, the HOLD deadline at its own tick
                    if hasattr(policy, "decision_context"):
                        mask, kind = policy.decision_context(policy_input)
                        if mask is not None:
                            features, raw = ppo_features.build_features(policy, policy_input,
                                                                        previous_features)
                            previous_features = raw
                            if self.policy_mode == "scripted_rule":
                                action = policy.decide(features, mask, kind, now)
                            else:
                                # ppo and selector modes share one actor
                                # boundary: (mask, features, kind, now);
                                # the offer the features describe is the
                                # offer the decision consumes
                                action = self.actor(mask, features, kind, now)
                            if action is not None:
                                policy.set_decision(int(action))
                            rollout.append(dict(time_s=now, phase=kind, mask=list(mask),
                                                features=features, action=int(action),
                                                action_meaning=C.ACTION_MEANINGS[
                                                    "hold_deadline" if kind == "hold_deadline"
                                                    else "hold_interior" if kind == "hold_interior"
                                                    else "inactive"][int(action)] if action is not None
                                                else "none"))
                            if hasattr(policy, "offer_metadata"):
                                offer = policy.offer_metadata()
                                if offer is not None:
                                    rollout[-1]["offer"] = offer
                    result = policy(policy_input)
                    if not isinstance(result, PolicyOutput):
                        raise TypeError("policy must return the typed PolicyOutput boundary")
                    emit("policy", dict(time_s=now, assignment=self.registration["assignment"],
                                        **result.payload()))
                    policy_counts[result.phase] += 1
                    records = owner.step(frames, now_s=now, live_ids=live,
                                         wall_commands=result.wall_commands)
                    prior_commands = {v: r for v, r in prior_commands.items() if v in live}
                    for record in records:
                        invariant_counts["hdv_commands"] += int(record.vehicle_id not in cav_ids)
                        prior_commands[record.vehicle_id] = record
                        emit("telemetry", dict(kind="command", **record.as_dict()))
                    interval = Counter()
                    wait_before = ledger.source_vehicle_seconds
                    wall_owned = result.phase in {"assembly", "hold", "taper"}
                    wall_owned_steps += int(wall_owned)
                    log_formation = bool(result.member_ids and result.phase in
                                         {"assembly", "hold", "taper"})
                    for substep in range(2):
                        if log_formation:
                            emit("telemetry", dict(kind="formation_sample",
                                time_s=float(api.simulation.getTime()), phase=result.phase,
                                members=_member_measurements(sensor, result.member_ids, live,
                                    float(api.simulation.getTime()))))
                        after = clock.advance()
                        live = set(api.vehicle.getIDList())
                        interval.update(braking_effort(
                            [float(api.vehicle.getAcceleration(v)) for v in sorted(live)], .5))
                        exposure, _ = ttc_exposure(api, sorted(live), .5)
                        interval["ttc_below_3_vehicle_seconds"] += exposure
                        departed = list(api.simulation.getDepartedIDList())
                        arrived = list(api.simulation.getArrivedIDList())
                        interval["requested_not_arrived_vehicle_seconds"] += ledger.advance(
                            after, departed, arrived)
                        colliding = set(api.simulation.getCollidingVehiclesIDList())
                        interval["collision_involvements"] += len(colliding - collisions_previous)
                        collisions_previous = colliding
                        interval["starting_teleports"] += api.simulation.getStartingTeleportNumber()
                        for cav in sorted({r.vehicle_id for r in records} & live):
                            emit("telemetry", dict(kind="response", **owner.record_response(cav,
                                timestamp_s=after, speed_mps=float(api.vehicle.getSpeed(cav)),
                                acceleration_mps2=float(api.vehicle.getAcceleration(cav))).as_dict()))
                    wait_delta = ledger.source_vehicle_seconds - wait_before
                    row = ppo_reward.tick_reward(interval, wall_owned=wall_owned,
                                                 source_wait_delta=wait_delta)
                    row.update(start_s=now, end_s=after, phase=result.phase)
                    emit("reward_log", row)
                    pending_reward += row["reward"]
                    totals.update(interval)
                    emit("intervals", dict(start_s=now, end_s=after, **interval))
                    # Credit: the action taken at decision tick t owns the tick
                    # rewards of [t, t+cadence) - the boundary closes after the
                    # window's physics, so the previous decision receives the
                    # full window including its own first tick.
                    if (tick + 1) % int(C.CADENCE_S) == 0 or tick + 1 == self.horizon:
                        if rollout:
                            rollout[-1]["reward"] = pending_reward
                            rollout[-1]["done"] = tick + 1 == self.horizon
                        pending_reward = 0.0
                    for stream in streams.values():
                        stream.flush()
            finally:
                for stream in streams.values():
                    stream.close()
            summary = dict(schema="paired_escalation_outcomes/1", seed=job["seed"],
                controller=f"pilot_{self.policy_mode}", duration_s=self.horizon,
                mechanics_only=self.horizon != config.episode_seconds,
                complete_fixed_horizon=self.horizon == config.episode_seconds,
                **service_summary(ledger, self.horizon),
                requested_due_at_horizon=ledger.due(self.horizon), totals=dict(totals),
                command_counts=dict(command_counts), phase_seconds=dict(policy_counts),
                ownership_invariants=dict(invariant_counts),
                scenario_identity=manifest["scenario_identity"],
                elapsed_wall_seconds=time.monotonic() - started,
                sumo_version=list(api.getVersion()), assignment=self.registration["assignment"],
                scientifically_accepted=False, wall_owned_control_steps=wall_owned_steps,
                policy_version=C.POLICY_VERSION, contract_sha256=C.contract_sha256())
            summary["all_requested_service_completed"] = (
                summary["requested_due_at_horizon"] == summary["population"]
                and summary["arrived"] == summary["population"]
                and summary["unfinished_requested"] == 0)
            summary["censored_service"] = not summary["all_requested_service_completed"]
            summary["future_requests_at_horizon"] = summary["population"] - summary["requested_due_at_horizon"]
            owner.release_all(live_ids=set(api.vehicle.getIDList()))
            owner = None
            close_sumo(api)
            api = None
            write_json(output / "summary.json", summary)
            (output / "rollout.json").write_text(json.dumps(rollout, allow_nan=False), encoding="utf-8")
            if hasattr(policy, "decision_log"):
                (output / "decision_log.json").write_text(
                    json.dumps(policy.decision_log, allow_nan=False), encoding="utf-8")
            manifest.update(status="complete", summary_sha256=sha(output / "summary.json"))
            write_json(output / "manifest.json", manifest)
            return summary, rollout
        except ClockViolationError as exc:
            manifest.update(status="failed", error=repr(exc),
                            clock_receipt=exc.receipt)
            write_json(output / "manifest.json", manifest)
            raise
        except Exception as exc:
            manifest.update(status="failed", error=repr(exc))
            write_json(output / "manifest.json", manifest)
            raise
        finally:
            if api is not None:
                try:
                    if owner is not None:
                        owner.release_all(live_ids=set(api.vehicle.getIDList()))
                finally:
                    close_sumo(api)
