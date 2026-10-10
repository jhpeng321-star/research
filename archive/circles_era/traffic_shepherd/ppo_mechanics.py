"""Mechanics fixture for the RSU PPO pilot (wall_ppo/1 contract).

One registered 120 s allocation on the isolated small two-lane fixture (the
verified target-adjust/hold-extend fixture): masked ACTIVATE request before
the fixture window, then the full live chain WAIT -> ACTIVATE -> CONTINUE
once (H20 -> H30) -> RELEASE -> coast handback, under the shared
sim_clock.ExactClock contract (identical stepping/collection semantics to
the corridor ppo_env loop). Verifies decision logging, the admission gate,
reward accounting and the intervention-step bound. Mechanics success
demonstrates implementation behavior, never efficacy. A failed allocation
stops the campaign until reported (STOP receipt).
"""
import json
from pathlib import Path
from types import SimpleNamespace

from . import ppo_contract as C
from . import ppo_features, ppo_reward
from .damping import CirclesController
from .gentle_mechanics import _write_json
from .metrics import RequestLedger, braking_effort, ttc_exposure
from .ownership import CommandOwnership
from .paired_experiments import PolicyInput
from .reporting import CAVReport, RSUReportBuffer
from .sensing import SumoLocalSensor
from .sim_clock import ClockViolationError, ExactClock
from .simulation import close_sumo, start_sumo
from .target_adjust_mechanics import build_fixture
from .wall_ppo_policy import WallPpoPolicy

SCHEMA = "ppo_pilot_mechanics/1"
MEMBERS = ("member_a", "member_b")
FOLLOWERS = ("follower_a", "follower_b")
LANES = ("road_0", "road_1")
HORIZON_S = 120.
CASES = {
    "hold_ppo_chain": "masked_activate_request_then_activate_continue_release_"
                      "handback_chain_with_reward_and_intervention_accounting",
}


def plan():
    return dict(schema=SCHEMA, policy_version=C.POLICY_VERSION,
        contract_sha256=C.contract_sha256(),
        cases=[dict(case_id=cid, coverage=coverage) for cid, coverage in CASES.items()],
        horizon_s=HORIZON_S, cadence_s=C.CADENCE_S,
        maximum_allocations=2, maximum_simulated_seconds=240, automatic_retries=0,
        automatic_successors=False, simultaneous_runs=False,
        stop_rules=["A failed allocation stops the campaign until reported; never retry or overwrite."],
        mechanical_not_traffic_efficacy=True)


def _script(policy, now, mask, kind):
    """v2 chain decisions. kind comes from decision_context:
    'inactive' | 'hold_interior' | 'hold_deadline' (mask None elsewhere)."""
    if mask is None:
        return None
    if kind == "inactive":
        if now <= 5.0:
            return C.A_ACTIVATE  # deliberately masked before the fixture window
        return C.A_ACTIVATE if mask[C.A_ACTIVATE] else C.A_WAIT
    if kind == "hold_interior":
        return C.A_MAINTAIN               # keep the admitted commitment
    if kind == "hold_deadline":
        if policy.profile.hold_s == C.BASE_HOLD_S:
            return C.A_EXTEND             # H20 -> H30 at the first deadline
        return C.A_DEADLINE_RELEASE       # release at the H30 deadline
    return None


def run_case_chain(root, data, case_id, budget):
    import traci
    from .config import load_config
    config = load_config()
    output = Path(root) / "jobs" / case_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=SCHEMA, case_id=case_id, status="allocated",
                    completed_simulated_seconds=0., passed=False, mechanics_only=True)
    _write_json(output / "manifest.json", manifest)
    job = dict(arm="wall", decision_time_s=5, duration_s=int(HORIZON_S),
               policy_spec=dict(member_ids=list(MEMBERS), required_lanes=list(LANES)))
    policy = WallPpoPolicy(job, dict(registration_sha256=data["registration_sha256"]),
                           MEMBERS, LANES, activation_window=(10.0, 100.0))
    controller = CirclesController(config)
    reward_rows, rollout, configured = [], [], set()
    totals = dict(braking_vehicle_seconds=0., acceleration_energy=0.,
                  ttc_below_3_vehicle_seconds=0., requested_not_arrived_vehicle_seconds=0.,
                  collision_involvements=0, starting_teleports=0)
    checks = dict(activation_recorded=False, continue_accepted=False, release_recorded=False,
                  refusal_or_wait_recorded=False, handback_reached=False,
                  reward_rows_complete=False, rollout_contract=False,
                  clock_exact=False, intervention_bound=False, features_finite=True,
                  no_wall_commands_before_activation=True)
    instrumentation = dict(
        physics_step_length_s=config.physics_dt_s,
        default_action_step_length_s="SUMO default (1.0 s; not overridden) - recorded "
            "as diagnostic context only, never assumed to have caused the "
            "historical anomaly",
        command_line=None, traci_port=None, traci_module=None, sumo_version=None)
    manifest["instrumentation"] = instrumentation
    try:
        scene, requests = build_fixture(output / "scenario", data["sumo_binary"])
        ledger = RequestLedger(requests)
        owner = CommandOwnership(SimpleNamespace(vehicle=traci.vehicle, lane=traci.lane),
                                 MEMBERS, controller)
        sensor = SumoLocalSensor(SimpleNamespace(vehicle=traci.vehicle, lane=traci.lane),
                                 config, {lane: 0. for lane in LANES})
        buffer = RSUReportBuffer(max_age_s=1., history_s=20.)
        history, previous_records = [], {}
        from .budget_guard import reserve_and_launch
        # Instrumented launch: step log every substep into the message log
        # (options verified against the installed `sumo --help`); the exact
        # command line, port, TraCI module and server version are recorded
        # before any stepping begins.
        import traci as traci_module
        instrumentation["traci_module"] = getattr(traci_module, "__file__", None)
        message_log = output / "sumo_messages.log"
        command_record = []
        def _launcher():
            port = start_sumo(data["sumo_binary"], scene, seed=1, config=config,
                extra_cli=["--step-log.period", "1",
                           "--message-log", str(message_log)],
                no_step_log=False, command_record=command_record)
            instrumentation["command_line"] = " ".join(command_record[0])                 if command_record else None
            instrumentation["traci_port"] = port
            instrumentation["sumo_version"] = list(traci_module.getVersion())
            return port
        reservation, _port = reserve_and_launch(
            budget, target_dir=root, launcher=_launcher,
            campaign_stop_on_failure=bool(budget.get("campaign_stop_on_failure")))
        manifest["allocation_id"] = reservation["id"]
        # Exact-clock contract (sim_clock.ExactClock): one owner of simulator
        # advancement; exactly one bare simulationStep per 0.5 s physics
        # sample; control ticks at integer seconds; pilot decisions only at
        # the declared cadence; every sample collected exactly once. Any
        # repeated time, overshoot or out-of-grid read raises
        # ClockViolationError with an expected/observed receipt - no
        # tolerance-based skipping and no compensation for missed samples.
        clock = ExactClock(step_fn=traci.simulationStep,
            time_fn=lambda: float(traci.simulation.getTime()), dt=.5, start=0.,
            context="mechanics:" + case_id)
        for tick in range(int(HORIZON_S)):
            now = clock.check_now(float(traci.simulation.getTime()),
                                  context=f"control tick {tick}")
            live = set(traci.vehicle.getIDList())
            for vid in live - configured:
                traci.vehicle.setLaneChangeMode(vid, 512)
                configured.add(vid)
            if tick > 0:
                frames = {v: sensor.capture(v, now_s=now)
                          for v in sorted(live & set(MEMBERS))}
                reports = []
                for vid, frame in sorted(frames.items()):
                    previous = previous_records.get(vid)
                    packet = CAVReport(frame, now, previous.owner if previous else "native",
                        previous.network_output_mps2 if previous else None,
                        previous.issued_acceleration_mps2 if previous else None,
                        previous.fallback_reason if previous else None)
                    buffer.receive(packet, received_at_s=now)
                    history.append((now, packet))
                    reports.append(packet)
                history = [(t, r) for t, r in history if t >= now - 20.]
                # Current local damping references (or explicit native
                # fallback) for every captured member - the same contract as
                # evaluate_arm and the verified hold_extend mechanics; the
                # wall policy refuses to act without them.
                expected = {v: controller.decide(f, now_s=now) for v, f in frames.items()}
                current_reporters = {r.frame.vehicle_id for r in reports
                                     if r.frame.timestamp_s == now}
                references = {v: d.command_acceleration_mps2 for v, d in expected.items()
                              if d.fallback_reason is None and v in current_reporters}
                natives = frozenset(v for v, d in expected.items()
                                    if d.fallback_reason is not None
                                    and v in current_reporters)
                policy_input = PolicyInput(now, tuple(reports), tuple(history),
                                           references, natives)
                # decision_context gates internally: inactive/interior on the
                # cadence grid; the HOLD deadline decision fires AT the
                # scheduled end even when it falls off the cadence grid
                mask, kind = policy.decision_context(policy_input)
                if mask is not None:
                    features, _raw = ppo_features.build_features(policy, policy_input, None)
                    if not all(x == x and abs(x) != float("inf") for x in features):
                        checks["features_finite"] = False
                    action = _script(policy, now, mask, kind)
                    if action is not None:
                        policy.set_decision(int(action))
                    rollout.append(dict(time_s=now, phase=kind, mask=list(mask),
                                        features=features, action=int(action or 0)))
                result = policy(policy_input)
                if policy.activation_used == 0 and result.wall_commands:
                    checks["no_wall_commands_before_activation"] = False
                for record in owner.step(frames, now_s=now, live_ids=live,
                                         wall_commands=result.wall_commands):
                    previous_records[record.vehicle_id] = record
            wall_owned = result.phase in {"assembly", "hold", "taper"} if tick > 0 else False
            interval = dict(braking_vehicle_seconds=0., acceleration_energy=0.,
                            ttc_below_3_vehicle_seconds=0.,
                            requested_not_arrived_vehicle_seconds=0.,
                            collision_involvements=0, starting_teleports=0)
            for substep in (1, 2):
                after = clock.advance()
                live = set(traci.vehicle.getIDList())
                values = braking_effort(
                    [float(traci.vehicle.getAcceleration(v)) for v in sorted(live)], .5)
                interval["braking_vehicle_seconds"] += values["braking_vehicle_seconds"]
                interval["acceleration_energy"] += values["acceleration_energy"]
                exposure, _ = ttc_exposure(traci, live, .5)
                interval["ttc_below_3_vehicle_seconds"] += exposure
                interval["requested_not_arrived_vehicle_seconds"] += ledger.advance(
                    after, list(traci.simulation.getDepartedIDList()),
                    list(traci.simulation.getArrivedIDList()))
                interval["collision_involvements"] += int(
                    traci.simulation.getCollidingVehiclesNumber())
                interval["starting_teleports"] += int(traci.simulation.getStartingTeleportNumber())
            row = ppo_reward.tick_reward(interval, wall_owned=wall_owned, source_wait_delta=0.0)
            reward_rows.append(row)
            for key in totals:
                totals[key] += interval[key]
            # Completed time is the final OBSERVED simulator time of this
            # control interval (post-step), never the pre-step read: a 120 s
            # fixture must report 120.0 after its last interval, and a failed
            # run must retain the last clock-verified progress.
            manifest["completed_simulated_seconds"] = clock.last
        # ---- post-run checks ------------------------------------------------
        # Time contract (derived, not fitted to the implementation):
        #   physics: HORIZON_S / 0.5 = 240 samples, each collected exactly
        #     once, covering (0, 120] with the first interval (0, 0.5] and
        #     the final interval (119.5, 120] included;
        #   control: the controller is invoked once per integer second
        #     1..HORIZON_S-1 (post-step shape, matching the verified mechanics
        #     loops), i.e. 119 control ticks; tick 0 is initialization only;
        #   reward intervals: one row per 1 s control interval
        #     (t, t+1] for t in 0..HORIZON_S-1 -> exactly HORIZON_S rows,
        #     first (0, 1] and final (119, 120] included, none duplicated;
        #   rollout rows: EXACTLY the decision ticks (multiples of the 5 s
        #     cadence on control ticks 1..HORIZON_S-1) whose policy phase is
        #     inactive or hold - assembly/taper/terminal phases offer no
        #     decision mask and must not fabricate rows; the row count is
        #     therefore phase-dependent and is cross-checked 1:1 against the
        #     policy's decision log;
        #   completed time: the final observed simulator time (HORIZON_S on a
        #     complete run).
        expected_physics_samples = int(round(HORIZON_S / .5))
        expected_control_ticks = int(HORIZON_S) - 1
        expected_reward_rows = int(HORIZON_S)
        decision_ticks = [row["time_s"] for row in rollout]
        log = policy.decision_log
        release_entry = next((d for d in log if d["executed_action"] == "release"), None)
        summary_delivered_hold_s = (release_entry["time_s"] - policy.ready_at
                                    if release_entry is not None
                                    and policy.ready_at is not None else None)
        checks["activation_recorded"] = any(d["executed_action"] == "activate" for d in log)
        checks["continue_accepted"] = any(d["executed_action"] == "extend" for d in log)
        checks["release_recorded"] = any(d["executed_action"] == "release" for d in log)
        checks["refusal_or_wait_recorded"] = any(
            d["requested_action"] == C.A_ACTIVATE and d["executed_action"] is None
            and d["rejection_reason"] == "activate_masked_or_out_of_window"
            for d in log)
        checks["handback_reached"] = policy.terminal is not None and \
            policy.terminal[0] == "release"
        # the chain requires the full admitted H30 delivered via the actor's
        # scripted deadline RELEASE — a protective abort is not a pass
        checks["full_h30_delivered"] = (
            policy.terminal is not None and policy.terminal[0] == "release"
            and policy.early_exit_reason == "ppo_release_decision"
            and summary_delivered_hold_s is not None
            and abs(summary_delivered_hold_s - 30.0) <= 1e-9)
        finite_rows = all(r["reward"] == r["reward"]
                          and abs(r["reward"]) < float("inf") for r in reward_rows)
        checks["reward_rows_complete"] = (
            len(reward_rows) == expected_reward_rows
            and clock.calls == expected_physics_samples
            and abs(clock.last - HORIZON_S) <= 1e-9
            and finite_rows)
        checks["rollout_contract"] = (
            len(rollout) == len(log)
            and len(decision_ticks) == len(set(decision_ticks))
            and all(row["phase"] in {"inactive", "hold_interior",
                                     "hold_deadline"} for row in rollout)
            and all(abs(t % C.CADENCE_S) <= 1e-9 or row_is_deadline
                    for row, t, row_is_deadline in
                    ((r, r["time_s"],
                      r["phase"] == "hold_deadline") for r in rollout))
            and all(b > a for a, b in zip(decision_ticks, decision_ticks[1:]))
            and [d["time_s"] for d in log] == decision_ticks)
        checks["clock_exact"] = clock.calls == expected_physics_samples
        checks["intervention_bound"] = (
            all(r["intervention"] <= C.INTERVENTION_COST_PER_STEP + 1e-15 for r in reward_rows)
            and sum(r["intervention"] for r in reward_rows)
            <= C.INTERVENTION_MAX_CONTRIBUTION + 1e-9)
        summary = dict(manifest, status="complete", passed=all(checks.values()), checks=checks,
            decision_log=policy.decision_log, totals=totals,
            episode_return=ppo_reward.episode_return(reward_rows)["return_"],
            clock_calls=clock.calls,
            clock_history=list(clock.history),
            control_ticks=expected_control_ticks,
            completed_simulated_seconds=clock.last,
            terminal=policy.terminal, activation_used=policy.activation_used,
            phase_seconds=sorted({d["phase"] for d in log}),
            time_contract=dict(physics_samples=expected_physics_samples,
                physics_dt_s=.5, control_ticks=expected_control_ticks,
                control_dt_s=1., reward_rows=expected_reward_rows,
                decision_cadence_s=C.CADENCE_S,
                rollout_rows_phase_dependent=True),
            interpretation="Actual isolated SUMO execution; mechanics verification only, "
                           "never traffic efficacy.")
        summary["ownership"] = dict(
            wall_owned_control_steps=sum(1 for r in reward_rows if r["wall_owned"]),
            handback_terminal=policy.terminal,
            handback_verified=summary["checks"]["handback_reached"],
            scripted_release=(policy.early_exit_reason == "ppo_release_decision"),
            protective_abort=(policy.terminal is not None
                              and policy.terminal[0] == "abort"),
            collisions=totals["collision_involvements"],
            teleports=totals["starting_teleports"])
        _write_json(output / "summary.json", summary)
        (output / "rollout.json").write_text(
            json.dumps(rollout, allow_nan=False), encoding="utf-8")
        (output / "reward_rows.json").write_text(
            json.dumps(reward_rows, allow_nan=False), encoding="utf-8")
        manifest.update(status="complete", passed=summary["passed"])
        manifest["instrumentation"] = instrumentation
        _write_json(output / "manifest.json", manifest)
        return summary
    except ClockViolationError as exc:
        manifest.update(status="failed", error_type=type(exc).__name__,
                        error=str(exc), clock_receipt=exc.receipt,
                        instrumentation=instrumentation)
        _write_json(output / "manifest.json", manifest)
        raise
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        _write_json(output / "manifest.json", manifest)
        raise
    finally:
        close_sumo()


def run_case(root, data, case_id, budget):
    if case_id == "hold_ppo_chain":
        return run_case_chain(root, data, case_id, budget)
    raise ValueError("unknown mechanics case")
