"""Supported fixed-baseline evaluation workflow; no learners or archived imports."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from .config import CONFIG_PATH, load_config
from .controller_factory import create_controller, create_pi_session
from .damping import CirclesController
from .escalation_observations import OfflineTrafficObserver
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .reporting import CAVReport
from .scenarios import build_scenario, observation_map, scenario_identity
from .seeds import exclusion_provenance, require_development_case
from .sensing import SumoLocalSensor
from .simulation import close_sumo, start_sumo
from .fleet_switch import FleetSwitchSession, SingleSwitchSpec
import os


# Transient Windows access/share denials on atomic replace (antivirus or
# indexer holding a freshly written file) are retried with the same payload,
# never regenerated.  Mirrors the reviewed budget_guard replacement retry.
_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_S = 0.05
_REPLACE_WINERRORS = frozenset((5, 32, 33))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    # Strictly an atomic replace retry: the prepared payload is never
    # regenerated between attempts (same reviewed pattern as budget_guard).
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)  # atomic: no half-write
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in _REPLACE_WINERRORS:
                raise
            if attempt + 1 == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_RETRY_DELAY_S)


def inspect_baseline():
    config = load_config()
    controller = CirclesController(config)
    return dict(
        status="working_baseline_available", scientifically_accepted=False,
        configuration_path=str(CONFIG_PATH), configuration_sha256=sha(CONFIG_PATH),
        model_path=str(config.model_path), model_sha256=sha(config.model_path),
        expected_model_sha256=config.model_sha256,
        singleton_probe_input_si=[20., 18., 30.],
        singleton_probe_output_mps2=controller.infer([20., 18., 30.]),
        network_path=str(config.network_path), network_available=config.network_path.is_file(),
        scenario_template_path=str(config.scenario_template_path),
        scenario_template_available=config.scenario_template_path.is_file(),
        control_dt_s=config.control_dt_s, physics_dt_s=config.physics_dt_s,
        sensor_range_m=config.sensor_range_m,
        source_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    )


def evaluate(*, seed, loading, penetration, controller_name, output, role="exposed", duration=None,
             collect_traffic=False, desired_speed_mps=None, progress_callback=None,
             fleet_switch=None, wall_suffix_path=None):
    """One explicitly selected previously exposed case; existing outputs never resume.

    Short runs retain the full exogenous requested-population denominator and
    report requests due at the short horizon separately. They verify mechanics,
    never full-trip efficacy, prospective acceptance, walls, or prediction.
    """
    config = load_config()
    if fleet_switch is not None and not isinstance(fleet_switch, SingleSwitchSpec):
        raise TypeError("fleet_switch must be SingleSwitchSpec or None")
    if fleet_switch is not None and "pi-external-u" in (fleet_switch.prefix, fleet_switch.suffix):
        if desired_speed_mps is None or float(desired_speed_mps) != 31.29:
            raise ValueError("fleet PI switch requires desired_speed_mps=31.29")
    if role not in {"exposed", "development"}:
        raise ValueError("active evaluation permits exposed/development cases only")
    require_development_case(seed, loading, role)
    if controller_name not in {"native", "damping", "pi-external-u"}:
        raise ValueError("choose native, damping, or pi-external-u")
    if fleet_switch is not None and controller_name != fleet_switch.prefix:
        raise ValueError("fleet switch prefix must match controller_name")
    fleet_has_pi = fleet_switch is not None and "pi-external-u" in (fleet_switch.prefix, fleet_switch.suffix)
    if fleet_has_pi:
        if desired_speed_mps is None or float(desired_speed_mps) != 31.29:
            raise ValueError("fleet PI switch requires desired_speed_mps=31.29")
    elif controller_name == "pi-external-u":
        if desired_speed_mps is None or float(desired_speed_mps) != 31.29:
            raise ValueError("pi-external-u requires desired_speed_mps=31.29")
    elif desired_speed_mps is not None:
        raise ValueError("desired_speed_mps is only valid for pi-external-u")
    if fleet_switch is not None:
        if abs(fleet_switch.switch_time_s / config.control_dt_s -
               round(fleet_switch.switch_time_s / config.control_dt_s)) > 1e-9:
            raise ValueError("fleet switch time must lie on the control grid")
    if not 0 <= penetration <= 1:
        raise ValueError("penetration must be in [0, 1]")
    horizon = config.episode_seconds if duration is None else duration
    if isinstance(horizon, bool) or int(horizon) != horizon or not 1 <= horizon <= config.episode_seconds:
        raise ValueError("duration must be an integer within the fixed episode horizon")
    if fleet_switch is not None and fleet_switch.switch_time_s >= horizon:
        raise ValueError("fleet switch must occur before the evaluation horizon")
    if abs(config.control_dt_s / config.physics_dt_s - round(config.control_dt_s / config.physics_dt_s)) > 1e-8:
        raise ValueError("control cadence must contain whole physics steps")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("evaluation output must be a new directory; never silently repeat a run")
    import traci
    if traci.isLoaded():
        raise RuntimeError("refusing to reuse or close a pre-existing TraCI connection")
    binary = shutil.which("sumo")
    if binary is None:
        raise RuntimeError("SUMO must be on PATH")
    controller = None
    controller_identity = None
    factory_speed = (desired_speed_mps if not fleet_has_pi or controller_name == "pi-external-u"
                     else None)
    if controller_name != "pi-external-u":
        controller, controller_identity = create_controller(
            controller_name, config, desired_speed_mps=factory_speed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema="traffic_shepherd_evaluation/1", status="preparing", seed=seed,
        mainline_vph=loading, penetration=penetration, role=role, controller=controller_name,
        duration_s=horizon, full_episode_seconds=config.episode_seconds,
        mechanics_only=horizon != config.episode_seconds, accepted_checkpoint=False,
        wall_generation_authorized=False, started_at_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(CONFIG_PATH), config={k: str(v) if isinstance(v, Path) else v for k,v in asdict(config).items()},
        model_sha256=sha(config.model_path), sumo_binary_sha256=sha(binary),
        active_source_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        seed_exclusion_provenance=exclusion_provenance(),
        allocated_simulated_seconds=horizon,
        offline_traffic_observations=bool(collect_traffic),
        controller_identity=controller_identity,
    )
    write_json(output / "manifest.json", manifest)
    started, api, ownership = time.monotonic(), None, None
    try:
        scene, requests, scenario = build_scenario(output / "scenario", seed, loading,
            penetration=penetration, role=role, config=config)
        cav_ids = {r.vehicle_id for r in requests if r.is_cav}
        if (controller_name == "pi-external-u" or fleet_has_pi) and not cav_ids:
            raise ValueError("pi-external-u requires a nonempty generated CAV roster")
        if controller_name == "pi-external-u" or fleet_has_pi:
            transitive = [
                Path(__file__).resolve().parents[1] / "output/controller_verification_v2/pi_saturation_v1.py",
                Path(__file__).resolve().parents[1] / "output/controller_verification_v4/pi_external_u.py",
                Path(__file__).resolve().parents[1] / "output/controller_verification_v7/pi_owner_adapter.py",
                Path(__file__).resolve().parents[1] / "traffic_shepherd/pi_session.py",
            ]
            manifest["pi_transitive_source_sha256"] = {str(p): sha(p) for p in transitive}
        fleet_has_wall = (fleet_switch is not None
                          and "wall-v2" in (fleet_switch.prefix, fleet_switch.suffix))
        if fleet_has_wall:
            if wall_suffix_path is None:
                wall_suffix_file = (Path(__file__).resolve().parents[1]
                                    / "output/wall_v2_integration_design/wall_v2_suffix.py")
            else:
                wall_suffix_file = Path(wall_suffix_path).resolve()
            fixed_supervisor = (Path(__file__).resolve().parents[1]
                                / "output/wall_v2_verification_v1/wall_v2_fixed.py")
            wall_transitive = [wall_suffix_file]
            if fixed_supervisor.is_file():
                wall_transitive.append(fixed_supervisor)
            if wall_suffix_path is None:
                wall_transitive.append(
                    Path(__file__).resolve().parents[1]
                    / "output/wall_v2_verification_v1/wall_v2.py")
            wall_transitive.append(
                Path(__file__).resolve().parents[1]
                / "output/wall_v2_verification_v1/REGISTRATION.json")
            manifest["wall_transitive_source_sha256"] = {str(p): sha(p) for p in wall_transitive}
        manifest.update(scenario=scenario, scenario_identity=scenario_identity(scene, requests,
            network_path=config.network_path), status="running")
        pi_session = None
        if controller_name == "pi-external-u" or (fleet_switch is not None and
                                                    "pi-external-u" in (fleet_switch.prefix, fleet_switch.suffix)):
            if desired_speed_mps is None or float(desired_speed_mps) != 31.29:
                raise ValueError("PI switch requires desired_speed_mps=31.29")
            pi_session = create_pi_session(config, cav_ids, desired_speed_mps=desired_speed_mps)
            if controller_name == "pi-external-u":
                controller = pi_session.adapter
                controller_identity = pi_session.identity
                manifest["controller_identity"] = controller_identity
        wall_session = None
        if fleet_has_wall:
            if wall_suffix_path is None:
                from output.wall_v2_integration_design.wall_v2_suffix import WallV2SuffixSession
                wall_suffix_file = Path(__file__).resolve().parents[1] / (
                    "output/wall_v2_integration_design/wall_v2_suffix.py")
            else:
                wall_suffix_file = Path(wall_suffix_path).resolve()
                import importlib.util
                _spec = importlib.util.spec_from_file_location(
                    "cv_wall_suffix_session", wall_suffix_file)
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules["cv_wall_suffix_session"] = _mod
                _spec.loader.exec_module(_mod)
                WallV2SuffixSession = _mod.WallV2SuffixSession
            # Registered wall constants (campaign registration): demand_ok=True
            # is the scenario-level E1 input for the m2700 corpus cells, pinned
            # at registration with the corpus-evidence rationale and never an
            # online observation; max_hold_s=300 is the registered active-
            # episode cap. The telemetry sink is bound after the run's
            # telemetry stream opens (wall_session.bind_emit below).
            wall_session = WallV2SuffixSession(cav_ids=cav_ids, emit=None, demand_ok=True,
                                               control_dt_s=config.control_dt_s,
                                               max_hold_s=300.0)
        circles_controller = None
        if fleet_switch is not None:
            circles_controller = (controller if controller_name != "pi-external-u"
                                  else CirclesController(config))
        fleet_session = (FleetSwitchSession(fleet_switch, circles_controller=circles_controller,
                                            pi_session=pi_session, cav_ids=cav_ids,
                                            wall_session=wall_session)
                         if fleet_switch is not None else None)
        if fleet_session is not None:
            manifest["fleet_switch"] = dict(prefix=fleet_switch.prefix,
                                             suffix=fleet_switch.suffix,
                                             switch_time_s=fleet_switch.switch_time_s)
            manifest["fleet_controller_identities"] = {
                "circles": {"model_sha256": sha(config.model_path),
                            "config_sha256": sha(CONFIG_PATH)},
                "pi": pi_session.identity if pi_session is not None else None,
                "native": "release_override_to_native_IDM",
                "wall": wall_session.identity if wall_session is not None else None,
            }
        write_json(output / "manifest.json", manifest)
        launch_command = []
        manifest["launch_command"] = launch_command
        write_json(output / "manifest.json", manifest)
        start_sumo(binary, scene, seed=seed, config=config, api=traci,
                   command_record=launch_command)
        api = traci  # Own cleanup only after our start succeeds.
        manifest.update(launch_command=launch_command, sumo_version=list(api.getVersion()))
        write_json(output / "manifest.json", manifest)
        ledger = RequestLedger(requests)
        ownership = CommandOwnership(api, cav_ids, controller)
        sensor = SumoLocalSensor(api, config, observation_map(api, config))
        traffic_observer = OfflineTrafficObserver(api, config) if collect_traffic else None
        totals = Counter()
        command_counts, fallback_counts = Counter(), Counter()
        collision_previous = set()
        minimum_ttc = None
        previous_live = set()
        retired_cavs = set()
        with (output / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry, \
                (output / "intervals.jsonl").open("x", encoding="utf-8") as intervals, \
                ((output / "traffic.jsonl").open("x", encoding="utf-8") if collect_traffic else nullcontext()) as traffic:
            def emit(kind, payload):
                telemetry.write(json.dumps(dict(kind=kind, **payload), allow_nan=False) + "\n")
            if wall_session is not None:
                # Wall events ride the owning tick's telemetry block: the sink
                # exists only inside the open telemetry context (design I8).
                wall_session.bind_emit(emit)

            for tick in range(int(horizon / config.control_dt_s)):
                now = float(api.simulation.getTime())
                if abs(now - tick * config.control_dt_s) > 1e-6:
                    raise RuntimeError("SUMO/control time grid mismatch")
                if traffic_observer is not None:
                    # This channel is not passed to sensor frames, reports or policies.
                    traffic.write(json.dumps(traffic_observer.sample(now_s=now), allow_nan=False) + "\n")
                live = set(api.vehicle.getIDList())
                disappeared = (previous_live - live) & cav_ids
                arrivals = set(api.simulation.getArrivedIDList()) & cav_ids
                retired_cavs.update(disappeared | arrivals)
                reappeared = retired_cavs & live
                retired_cavs.difference_update(reappeared)
                frames = {}
                for cav in sorted(live & cav_ids):
                    try:
                        frames[cav] = sensor.capture(cav, now_s=now)
                    except ValueError as exc:
                        # Invalid local measurements are explicit unavailability, not
                        # invented free-road observations or a privileged replacement.
                        emit("observation_unavailable", dict(vehicle_id=cav, timestamp_s=now, reason=str(exc)))
                try:
                    if fleet_session is not None:
                        emit("fleet_switch_attempt", dict(timestamp_s=now,
                            mode=fleet_session._mode_at(now)))
                    if fleet_session is not None:
                        records = fleet_session.step(ownership, frames, now_s=now, live_ids=live,
                            disappeared_ids=disappeared, reappeared_ids=reappeared)
                    elif controller_name == "pi-external-u":
                        records = pi_session.step(ownership, frames, now_s=now, live_ids=live,
                            native_ids=(), departed_ids=disappeared, id_reuse_ids=reappeared)
                    else:
                        records = ownership.step(frames, now_s=now, live_ids=live,
                            native_ids=live & cav_ids if controller_name == "native" else ())
                except BaseException:
                    if pi_session is not None:
                        write_json(output / "pi_evidence_rescue.json",
                                   {"status": "partial_batch", "rows": pi_session.pending_evidence()})
                    raise
                finally:
                    pi_active = (controller_name == "pi-external-u" or
                                 (fleet_session is not None and fleet_session.mode == "pi-external-u"))
                    if pi_active:
                        try:
                            pi_session.flush_evidence(
                                lambda payload: emit(payload.pop("kind"), payload),
                                finalize=False)
                            telemetry.flush()
                            pi_session.commit_evidence()
                        except BaseException:
                            write_json(output / "pi_evidence_rescue.json",
                                       {"status": "unflushed", "rows": pi_session.pending_evidence()})
                            raise
                commanded = {r.vehicle_id for r in records}
                if fleet_session is not None:
                    emit("fleet_switch_completed", dict(timestamp_s=now,
                        mode=fleet_session.mode, records=len(records)))
                for record in records:
                    row = record.as_dict()
                    emit("command", row)
                    command_counts[record.owner] += 1
                    if record.fallback_reason:
                        fallback_counts[record.fallback_reason] += 1
                    if record.vehicle_id in frames:
                        report = CAVReport(frames[record.vehicle_id], now, record.owner,
                            row["network_output_mps2"], row["issued_acceleration_mps2"], record.fallback_reason)
                        emit("cav_report", report.payload())
                interval = Counter()
                for substep in range(round(config.control_dt_s / config.physics_dt_s)):
                    api.simulationStep()
                    after = float(api.simulation.getTime())
                    # Record reached simulator time before any later integrity or
                    # metric check can fail; the caller owns durable accounting.
                    if progress_callback is not None:
                        progress_callback(after)
                    if abs(after - (now + (substep + 1) * config.physics_dt_s)) > 1e-6:
                        raise RuntimeError("SUMO physics time grid mismatch")
                    live_after = set(api.vehicle.getIDList())
                    if (controller_name == "pi-external-u" or
                            (fleet_session is not None and fleet_session.mode == "pi-external-u")):
                        disappeared_after = (live - live_after) & cav_ids
                        arrived_after = set(api.simulation.getArrivedIDList()) & cav_ids
                        retired_cavs.update(disappeared_after | arrived_after)
                        reappeared_after = retired_cavs & live_after
                        retired_cavs.difference_update(reappeared_after)
                        pi_session.observe_lifecycle(live_ids=live_after,
                            disappeared_ids=disappeared_after, id_reuse_ids=reappeared_after)
                    live = live_after
                    accelerations = [float(api.vehicle.getAcceleration(v)) for v in sorted(live)]
                    interval.update(braking_effort(accelerations, config.physics_dt_s))
                    exposure, closing_minimum = ttc_exposure(api, sorted(live), config.physics_dt_s)
                    interval["ttc_below_3_vehicle_seconds"] += exposure
                    if closing_minimum is not None:
                        minimum_ttc = closing_minimum if minimum_ttc is None else min(minimum_ttc, closing_minimum)
                    interval["requested_not_arrived_vehicle_seconds"] += ledger.advance(after,
                        list(api.simulation.getDepartedIDList()), list(api.simulation.getArrivedIDList()))
                    colliding = set(api.simulation.getCollidingVehiclesIDList())
                    interval["collision_involvements"] += len(colliding - collision_previous)
                    collision_previous = colliding
                    interval["starting_teleports"] += api.simulation.getStartingTeleportNumber()
                    for cav in sorted(commanded & live):
                        response = ownership.record_response(cav, timestamp_s=after,
                            speed_mps=float(api.vehicle.getSpeed(cav)),
                            acceleration_mps2=float(api.vehicle.getAcceleration(cav)))
                        emit("response", response.as_dict())
                totals.update(interval)
                intervals.write(json.dumps(dict(start_s=now, end_s=after, **interval), allow_nan=False) + "\n")
                telemetry.flush()
                intervals.flush()
                previous_live = live
        summary = dict(
            schema="traffic_shepherd_outcomes/1", seed=seed, controller=controller_name,
            duration_s=horizon, mechanics_only=horizon != config.episode_seconds,
            complete_fixed_horizon=horizon == config.episode_seconds, accepted_checkpoint=False,
            **service_summary(ledger, horizon), requested_due_at_horizon=ledger.due(horizon),
            service_denominator="entire_fixed_exogenous_requested_population_including_future_requests_in_short_checks",
            designated_cavs=len(cav_ids), realized_designation_fraction=len(cav_ids)/len(requests),
            totals=dict(totals), minimum_closing_ttc_s=minimum_ttc,
            command_counts=dict(command_counts), fallback_counts=dict(fallback_counts),
            scenario_identity=manifest["scenario_identity"], elapsed_wall_seconds=time.monotonic()-started,
            sumo_version=list(api.getVersion()),
        )
        if fleet_session is not None:
            summary["fleet_switch"] = manifest["fleet_switch"]
            summary["controller_identity"] = manifest.get("controller_identity")
            summary["fleet_controller_identities"] = manifest["fleet_controller_identities"]
        # Commit completion only after explicit handback and clean SUMO shutdown.
        ownership.release_all(live_ids=set(api.vehicle.getIDList()))
        ownership = None
        close_sumo(api)
        api = None
        write_json(output / "summary.json", summary)
        manifest.update(status="complete", telemetry_sha256=sha(output / "telemetry.jsonl"),
                        intervals_sha256=sha(output / "intervals.jsonl"), summary_sha256=sha(output / "summary.json"))
        if collect_traffic:
            manifest["traffic_sha256"] = sha(output / "traffic.jsonl")
        write_json(output / "manifest.json", manifest)
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if api is not None:
            try:
                if ownership is not None:
                    ownership.release_all(live_ids=set(api.vehicle.getIDList()))
            finally:
                close_sumo(api)
