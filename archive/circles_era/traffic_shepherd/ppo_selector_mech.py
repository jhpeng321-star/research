"""Selector mechanics fixture for the RSU PPO pilot (wall_ppo/3, PP-13).

ONE registered 120 s allocation on the isolated small two-lane fixture,
POPULATED with four reporting CAVs (two coherent candidates per registered
lane, so the causal search assesses four pairs) plus the two HDV followers
of the verified PP-10 geometry. The full live selector chain runs under the
shared sim_clock.ExactClock contract:

  masked ACTIVATE request before the fixture window (t=5) -> the frozen
  phase-aware scripted comparator: ACTIVATE the offered feasible proposal at
  the first in-window tick -> candidate-specific observations bound to the
  offer -> execution admits EXACTLY the approved proposal -> assembly ->
  MAINTAIN interior holds -> EXTEND H20->H30 at the deadline -> MAINTAIN ->
  RELEASE at the H30 deadline -> coast handback to identical damping.

Mechanics success demonstrates implementation behavior, never efficacy,
observation sufficiency, corridor-scale timing or learning. A failed
allocation stops the campaign until reported (STOP receipt).
"""
import json
import xml.etree.ElementTree as ET
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
from .selector_wall_policy import (SELECTOR_POLICY_VERSION, SELECTOR_VERSION,
                                   SelectorWallPolicy,
                                   scripted_selector_chain_actor)
from .sim_clock import ClockViolationError, ExactClock
from .simulation import close_sumo, start_sumo
from .target_speed_mechanics import CRUISE_MPS, ROAD_SPEED_MPS
from .scenarios import TripRequest

SCHEMA = "ppo_selector_mechanics/1"
MEMBERS = ("member_a", "member_b", "member_c", "member_d")
NEAR_MEMBERS = ("member_a", "member_b")     # positions 1000/990 (PP-10 geometry)
FAR_MEMBERS = ("member_c", "member_d")      # positions 400/390 (second cohort)
FOLLOWERS = ("follower_a", "follower_b")   # HDVs, well behind the near pair (610/600) so idle cruise traffic never intrudes on the wall rear-clearance box within the chain window
LANES = ("road_0", "road_1")
HORIZON_S = 120.
ACTIVATION_WINDOW_S = (10.0, 100.0)
CASES = {
    "selector_ppo_chain":
        "populated_selector_chain: reports_before_window_masked_probe_"
        "multi_pair_search_offer_binding_activate_same_proposal_assembly_"
        "maintain_extend_h30_release_handback_with_reward_and_intervention_"
        "accounting",
}


def plan():
    return dict(schema=SCHEMA, policy_version=SELECTOR_POLICY_VERSION,
                selector_version=SELECTOR_VERSION,
                contract_sha256=C.contract_sha256(),
                cases=[dict(case_id=cid, coverage=coverage)
                       for cid, coverage in CASES.items()],
                horizon_s=HORIZON_S, cadence_s=C.CADENCE_S,
                activation_window_s=list(ACTIVATION_WINDOW_S),
                fixture=dict(lanes=list(LANES), length_m=3500.,
                             road_speed_mps=ROAD_SPEED_MPS,
                             cruise_mps=CRUISE_MPS,
                             member_positions_m=[1000., 990., 400., 390.],
                             follower_positions_m=[610., 600.],
                             cav_count=4, assessed_pairs_expected=4),
                maximum_allocations=1, maximum_simulated_seconds=120,
                automatic_retries=0, automatic_successors=False,
                simultaneous_runs=False,
                stop_rules=["A failed allocation stops the campaign until "
                            "reported; never retry or overwrite."],
                mechanical_not_traffic_efficacy=True)


def build_selector_fixture(output, sumo_binary):
    """The verified PP-10 fixture network populated with FOUR reporting CAVs
    (two per lane: near pair at 1000/990, far pair at 400/390) and the two
    HDV followers. No cohort is preselected: the registered ranking chooses
    among all four assessed pairs at run time."""
    import subprocess
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "nodes.xml").write_text(
        '<nodes><node id="start" x="0" y="0"/><node id="end" x="3500" y="0"/>'
        '</nodes>')
    (output / "edges.xml").write_text(
        f'<edges><edge id="road" from="start" to="end" numLanes="2" '
        f'speed="{ROAD_SPEED_MPS:g}"/></edges>')
    binary = Path(sumo_binary)
    converter = binary.parent / ("netconvert.exe" if binary.suffix == ".exe"
                                 else "netconvert")
    result = subprocess.run(
        [str(converter), "--node-files", str(output / "nodes.xml"),
         "--edge-files", str(output / "edges.xml"),
         "--output-file", str(output / "network.net.xml")],
        capture_output=True, text=True)
    (output / "netconvert.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", dict(
        id="idm", carFollowModel="IDM", accel="2.6", decel="4.5",
        emergencyDecel="9", tau="1.1", minGap="2.5", sigma="0",
        speedFactor="1", speedDev="0", length="5",
        maxSpeed=f"{CRUISE_MPS:g}", lcStrategic="0", lcCooperative="0",
        lcSpeedGain="0", lcKeepRight="0"))
    ET.SubElement(routes, "route", dict(id="route", edges="road"))
    placements = [
        ("member_a", 1000., "0"), ("member_b", 990., "1"),
        ("member_c", 400., "0"), ("member_d", 390., "1"),
        ("follower_a", 610., "0"), ("follower_b", 600., "1"),
    ]
    for vehicle, position, lane in placements:
        ET.SubElement(routes, "vehicle", dict(
            id=vehicle, type="idm", route="route", depart="0",
            departLane=lane, departPos=str(position),
            departSpeed=f"{CRUISE_MPS:g}"))
    ET.indent(routes)
    ET.ElementTree(routes).write(output / "routes.rou.xml", encoding="utf-8",
                                 xml_declaration=True)
    (output / "scenario.sumocfg").write_text(
        '<configuration><input><net-file value="network.net.xml"/>'
        '<route-files value="routes.rou.xml"/></input><processing>'
        '<time-to-teleport value="-1"/><collision.action value="warn"/>'
        '</processing></configuration>')
    requests = [TripRequest(v, 0., "route", "idm", v in MEMBERS)
                for v in (*MEMBERS, *FOLLOWERS)]
    return output / "scenario.sumocfg", requests


def _selector_script(policy, now, mask, kind, features):
    """Frozen schedule: a deliberately masked ACTIVATE request at t=5
    (before the fixture window; exercises window enforcement exactly as in
    PP-10), then the PP-12 phase-aware comparator."""
    if mask is None:
        return None
    if kind == "inactive" and now <= 5.0:
        return C.A_ACTIVATE
    return scripted_selector_chain_actor(mask, features, kind, now)


def run_case(root, data, case_id, budget):
    import traci
    from .config import load_config
    config = load_config()
    output = Path(root) / "jobs" / case_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=SCHEMA, case_id=case_id, status="allocated",
                    completed_simulated_seconds=0., passed=False,
                    mechanics_only=True, policy_version=SELECTOR_POLICY_VERSION,
                    selector_version=SELECTOR_VERSION)
    _write_json(output / "manifest.json", manifest)
    job = dict(arm="wall", decision_time_s=5, duration_s=int(HORIZON_S),
               policy_spec=dict(member_ids=list(NEAR_MEMBERS),
                                required_lanes=list(LANES)))
    policy = SelectorWallPolicy(
        job, dict(registration_sha256=data["registration_sha256"],
                  communication=dict(max_report_age_s=1.0, history_s=20.0)),
        activation_window=ACTIVATION_WINDOW_S)
    controller = CirclesController(config)
    reward_rows, rollout, configured = [], [], set()
    totals = dict(braking_vehicle_seconds=0., acceleration_energy=0.,
                  ttc_below_3_vehicle_seconds=0.,
                  requested_not_arrived_vehicle_seconds=0.,
                  collision_involvements=0, starting_teleports=0)
    checks = dict(activation_recorded=False, extend_accepted=False,
                  release_recorded=False, refusal_or_wait_recorded=False,
                  handback_reached=False, full_h30_delivered=False,
                  reward_rows_complete=False, rollout_contract=False,
                  clock_exact=False, intervention_bound=False,
                  features_finite=True, no_wall_commands_before_activation=True,
                  nontrivial_selection=False, selection_deterministic=False,
                  binding_consistency=False, member_fixation=False)
    instrumentation = dict(
        physics_step_length_s=config.physics_dt_s,
        default_action_step_length_s="SUMO default (1.0 s; not overridden) - "
            "recorded as diagnostic context only, never assumed to have "
            "caused the historical anomaly",
        command_line=None, traci_port=None, traci_module=None,
        sumo_version=None)
    manifest["instrumentation"] = instrumentation
    try:
        scene, requests = build_selector_fixture(output / "scenario",
                                                 data["sumo_binary"])
        ledger = RequestLedger(requests)
        owner = CommandOwnership(
            SimpleNamespace(vehicle=traci.vehicle, lane=traci.lane),
            MEMBERS, controller)
        sensor = SumoLocalSensor(
            SimpleNamespace(vehicle=traci.vehicle, lane=traci.lane),
            config, {lane: 0. for lane in LANES})
        buffer = RSUReportBuffer(max_age_s=1., history_s=20.)
        history, previous_records = [], {}
        from .budget_guard import reserve_and_launch
        import traci as traci_module
        instrumentation["traci_module"] = getattr(traci_module, "__file__", None)
        message_log = output / "sumo_messages.log"
        command_record = []

        def _launcher():
            port = start_sumo(data["sumo_binary"], scene, seed=1, config=config,
                extra_cli=["--step-log.period", "1",
                           "--message-log", str(message_log)],
                no_step_log=False, command_record=command_record)
            instrumentation["command_line"] = " ".join(command_record[0]) \
                if command_record else None
            instrumentation["traci_port"] = port
            instrumentation["sumo_version"] = list(traci_module.getVersion())
            return port

        reservation, _port = reserve_and_launch(
            budget, target_dir=root, launcher=_launcher,
            campaign_stop_on_failure=bool(
                budget.get("campaign_stop_on_failure")))
        manifest["allocation_id"] = reservation["id"]
        clock = ExactClock(step_fn=traci.simulationStep,
            time_fn=lambda: float(traci.simulation.getTime()), dt=.5, start=0.,
            context="selector-mechanics:" + case_id)
        for tick in range(int(HORIZON_S)):
            now = clock.check_now(float(traci.simulation.getTime()),
                                  context=f"control tick {tick}")
            live = set(traci.vehicle.getIDList())
            for vid in live - configured:
                traci.vehicle.setLaneChangeMode(vid, 512)
                configured.add(vid)
            result = None
            if tick > 0:
                frames = {v: sensor.capture(v, now_s=now)
                          for v in sorted(live & set(MEMBERS))}
                reports = []
                for vid, frame in sorted(frames.items()):
                    previous = previous_records.get(vid)
                    packet = CAVReport(frame, now,
                        previous.owner if previous else "native",
                        previous.network_output_mps2 if previous else None,
                        previous.issued_acceleration_mps2 if previous else None,
                        previous.fallback_reason if previous else None)
                    buffer.receive(packet, received_at_s=now)
                    history.append((now, packet))
                    reports.append(packet)
                history = [(t, r) for t, r in history if t >= now - 20.]
                expected = {v: controller.decide(f, now_s=now)
                            for v, f in frames.items()}
                current_reporters = {r.frame.vehicle_id for r in reports
                                     if r.frame.timestamp_s == now}
                references = {v: d.command_acceleration_mps2
                              for v, d in expected.items()
                              if d.fallback_reason is None
                              and v in current_reporters}
                natives = frozenset(v for v, d in expected.items()
                                    if d.fallback_reason is not None
                                    and v in current_reporters)
                policy_input = PolicyInput(now, tuple(reports), tuple(history),
                                           references, natives)
                mask, kind = policy.decision_context(policy_input)
                if mask is not None:
                    features, _raw = ppo_features.build_features(
                        policy, policy_input, None)
                    if not all(x == x and abs(x) != float("inf")
                               for x in features):
                        checks["features_finite"] = False
                    action = _selector_script(policy, now, mask, kind, features)
                    if action is not None:
                        policy.set_decision(int(action))
                    row = dict(time_s=now, phase=kind, mask=list(mask),
                               features=features, action=int(action or 0))
                    offer = policy.offer_metadata()
                    if offer is not None:
                        row["offer"] = offer
                    rollout.append(row)
                result = policy(policy_input)
                if policy.activation_used == 0 and result.wall_commands:
                    checks["no_wall_commands_before_activation"] = False
                for record in owner.step(frames, now_s=now, live_ids=live,
                                         wall_commands=result.wall_commands):
                    previous_records[record.vehicle_id] = record
            wall_owned = (result.phase in {"assembly", "hold", "taper"}
                          if tick > 0 else False)
            interval = dict(braking_vehicle_seconds=0., acceleration_energy=0.,
                            ttc_below_3_vehicle_seconds=0.,
                            requested_not_arrived_vehicle_seconds=0.,
                            collision_involvements=0, starting_teleports=0)
            for substep in (1, 2):
                after = clock.advance()
                live = set(traci.vehicle.getIDList())
                values = braking_effort(
                    [float(traci.vehicle.getAcceleration(v))
                     for v in sorted(live)], .5)
                interval["braking_vehicle_seconds"] += values[
                    "braking_vehicle_seconds"]
                interval["acceleration_energy"] += values["acceleration_energy"]
                exposure, _ = ttc_exposure(traci, live, .5)
                interval["ttc_below_3_vehicle_seconds"] += exposure
                interval["requested_not_arrived_vehicle_seconds"] += ledger.advance(
                    after, list(traci.simulation.getDepartedIDList()),
                    list(traci.simulation.getArrivedIDList()))
                interval["collision_involvements"] += int(
                    traci.simulation.getCollidingVehiclesNumber())
                interval["starting_teleports"] += int(
                    traci.simulation.getStartingTeleportNumber())
            row = ppo_reward.tick_reward(interval, wall_owned=wall_owned,
                                         source_wait_delta=0.0)
            reward_rows.append(row)
            for key in totals:
                totals[key] += interval[key]
            manifest["completed_simulated_seconds"] = clock.last
        # ---- post-run checks (mechanics contract + selector binding) ------
        expected_physics_samples = int(round(HORIZON_S / .5))
        expected_control_ticks = int(HORIZON_S) - 1
        expected_reward_rows = int(HORIZON_S)
        decision_ticks = [row["time_s"] for row in rollout]
        log = policy.decision_log
        release_entry = next((d for d in log
                              if d["executed_action"] == "release"), None)
        summary_delivered_hold_s = (
            release_entry["time_s"] - policy.ready_at
            if release_entry is not None and policy.ready_at is not None
            else None)
        checks["activation_recorded"] = any(
            d["executed_action"] == "activate" for d in log)
        checks["extend_accepted"] = any(
            d["executed_action"] == "extend" for d in log)
        checks["release_recorded"] = any(
            d["executed_action"] == "release" for d in log)
        checks["refusal_or_wait_recorded"] = any(
            d["requested_action"] == C.A_ACTIVATE and d["executed_action"] is None
            and d["rejection_reason"] == "activate_masked_or_out_of_window"
            for d in log)
        checks["handback_reached"] = (policy.terminal is not None
                                      and policy.terminal[0] == "release")
        checks["full_h30_delivered"] = (
            policy.terminal is not None and policy.terminal[0] == "release"
            and policy.early_exit_reason == "ppo_release_decision"
            and summary_delivered_hold_s is not None
            and abs(summary_delivered_hold_s - 30.0) <= 1e-9)
        # selector binding evidence
        inactive_offers = [d.get("offer") for d in log
                           if d.get("phase") == "inactive"
                           and d.get("offer") is not None]
        checks["nontrivial_selection"] = bool(inactive_offers) and max(
            o["assessed_pairs"] for o in inactive_offers) >= 4
        offered_sets = {tuple(sorted(o["member_ids"]))
                        for o in inactive_offers}
        checks["selection_deterministic"] = len(offered_sets) == 1
        activation_entries = [d for d in log
                              if d["executed_action"] == "activate"]
        checks["binding_consistency"] = (
            len(activation_entries) == 1
            and activation_entries[0].get("offer", {}).get("committed") is True
            and activation_entries[0].get("committed_proposal_sha256")
            == activation_entries[0]["offer"]["proposal_fingerprint"]
            and (not offered_sets
                 or tuple(sorted(activation_entries[0]["offer"]["member_ids"]))
                 in offered_sets))
        committed_offers = [d["offer"] for d in log
                            if (d.get("offer") or {}).get("committed") is True]
        committed_sets = {tuple(sorted(o["member_ids"]))
                          for o in committed_offers}
        approved_set = (tuple(sorted(activation_entries[0]["offer"]["member_ids"]))
                        if activation_entries else None)
        checks["member_fixation"] = bool(
            policy.activation_used == 1
            and activation_entries
            and committed_sets == {approved_set})
        finite_rows = all(r["reward"] == r["reward"]
                          and abs(r["reward"]) < float("inf")
                          for r in reward_rows)
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
            all(r["intervention"] <= C.INTERVENTION_COST_PER_STEP + 1e-15
                for r in reward_rows)
            and sum(r["intervention"] for r in reward_rows)
            <= C.INTERVENTION_MAX_CONTRIBUTION + 1e-9)
        summary = dict(manifest, status="complete",
                       passed=all(checks.values()), checks=checks,
                       decision_log=policy.decision_log, totals=totals,
                       episode_return=ppo_reward.episode_return(reward_rows)["return_"],
                       clock_calls=clock.calls,
                       clock_history=list(clock.history),
                       control_ticks=expected_control_ticks,
                       completed_simulated_seconds=clock.last,
                       terminal=policy.terminal,
                       activation_used=policy.activation_used,
                       ready_at_s=policy.ready_at,
                       delivered_hold_s=summary_delivered_hold_s,
                       phase_seconds=sorted({d["phase"] for d in log}),
                       selector_timing=dict(
                           policy.search_stats,
                           note="fixture-scale wall-clock search cost only; "
                                "does not establish corridor-scale "
                                "computational performance"),
                       time_contract=dict(
                           physics_samples=expected_physics_samples,
                           physics_dt_s=.5,
                           control_ticks=expected_control_ticks,
                           control_dt_s=1., reward_rows=expected_reward_rows,
                           decision_cadence_s=C.CADENCE_S,
                           rollout_rows_phase_dependent=True),
                       interpretation="Actual isolated SUMO execution; "
                                      "selector mechanics verification only, "
                                      "never traffic efficacy, observation "
                                      "sufficiency or learning.")
        summary["ownership"] = dict(
            wall_owned_control_steps=sum(1 for r in reward_rows
                                         if r["wall_owned"]),
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
        manifest.update(status="failed", error_type=type(exc).__name__,
                        error=str(exc))
        _write_json(output / "manifest.json", manifest)
        raise
    finally:
        close_sumo()
