"""INC1 gated runner: warm-up -> saveState -> per-arm incident episodes.

Registration sections 2-8.  One warm-up run per world warms the fresh
scenario native to ``t0 = 300 s`` and saves the state; every arm restores
that SAME state (one ``loadState`` per process), inserts the stopped obstacle
at t0, and runs its response for 300 s (1 s control / 0.5 s physics) through
``CommandOwnership`` as the sole writer.  Per-vehicle physics logs use the
chunk-4 ``physics_steps.jsonl`` row shape so the featurizer/WM can read the
t0 states, and every run writes a section-8 receipt.

ZERO SUMO in this build: the real traci/session factories are lazily imported
and are never called by the tests, which inject a fake api + fake ownership.
The launcher refuses without a valid authorization + model lock and a
sufficient budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
    from . import gates
    from . import obstacles
    from . import schedules
    from . import wall_v22_freeze
except ImportError:  # script execution
    import constants as C  # type: ignore
    import gates  # type: ignore
    import obstacles  # type: ignore
    import schedules  # type: ignore
    import wall_v22_freeze  # type: ignore


class RunnerError(RuntimeError):
    """An INC1 runner contract violation."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------- scenario (chunk-4 template)
def inc1_request_schedule(seed: int, mainline_vph: int, *,
                          penetration: float, role: str = "exposed"):
    """Exact mirror of ``traffic_shepherd.scenarios.request_schedule``.

    The frozen scheduler gates on ``require_development_case``; INC1's seeds
    are fresh and authorized by ``REGISTRATION_WM_N1_INC1.md`` + the committed
    seed audit, so the identical algorithmic body (same RNG key strings, same
    piecewise demand, same type/CAV draws) is reproduced here WITHOUT the
    development-case gate.  A parity test pins this against the frozen
    scheduler for a registered exposed case.
    """
    import random
    from traffic_shepherd.scenarios import TripRequest

    if not 1900 <= mainline_vph <= 2800:
        raise RunnerError("loading outside declared development family")
    if not 0 <= penetration <= 1:
        raise RunnerError("penetration must be in [0, 1]")
    requests = []
    for stream in ("mainline", "ramp"):
        rng = random.Random(f"damping_requests_v1:{seed}:{stream}")
        rates = (
            [(0, 1200, mainline_vph)]
            + [
                (1200 + 100 * i, 1300 + 100 * i,
                 round(mainline_vph + (1500 - mainline_vph) * (i + 1) / 3))
                for i in range(3)
            ]
            + [(1500, 1800, 1500)]
        )
        if stream == "ramp":
            rates = [(0, 1800, 400)]
        ordinal = 0
        for begin, end, rate in rates:
            t = float(begin)
            while True:
                t += rng.expovariate(rate / 3600)
                if t >= end:
                    break
                identity = f"{stream}.{ordinal}"
                draw = random.Random(f"damping_type_v1:{seed}:{identity}").random()
                driver = ("aggressive_car" if draw < 0.1
                          else "normal_car" if draw < 0.8 else "cautious_car")
                cav = random.Random(
                    f"damping_cav_v1:{seed}:{identity}").random() < penetration
                requests.append(TripRequest(identity, round(t, 6),
                                            stream + "_route", driver, cav))
                ordinal += 1
    return sorted(requests, key=lambda r: (r.depart_s, r.vehicle_id))


def build_scenario(root: Path, seed: int, loading: int, penetration: float):
    """The chunk-4 scenario template with INC1's fresh, gated schedule."""
    import xml.etree.ElementTree as ET
    from traffic_shepherd.config import load_config

    config = load_config()
    if (config.episode_seconds, config.demand_end_seconds,
            config.ramp_vph) != (2400, 1800, 400):
        raise RunnerError("configuration differs from the tested fixture")
    requests = inc1_request_schedule(seed, loading, penetration=penetration)
    template = config.scenario_template_path
    original = ET.parse(template).getroot()
    route = ET.Element("routes")
    for vtype in original.findall("./vTypeDistribution/vType"):
        attrs = {k: v for k, v in vtype.attrib.items() if k != "probability"}
        attrs["speedDev"] = "0"
        ET.SubElement(route, "vType", attrs)
    for element in original.findall("route"):
        ET.SubElement(route, "route", element.attrib)
    for request in requests:
        ET.SubElement(route, "vehicle", {
            "id": request.vehicle_id, "type": request.driver_type,
            "route": request.route, "depart": str(request.depart_s),
            "departLane": "best", "departSpeed": "max"})
    ET.indent(route)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    route_path = root / "requests.rou.xml"
    scene_path = root / "scenario.sumocfg"
    scene = ET.Element("configuration")
    inputs = ET.SubElement(scene, "input")
    ET.SubElement(inputs, "net-file", value=str(config.network_path))
    ET.SubElement(inputs, "route-files", value=route_path.name)
    processing = ET.SubElement(scene, "processing")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.indent(scene)
    route_path.write_text(ET.tostring(route, encoding="unicode"),
                          encoding="utf-8")
    scene_path.write_text(ET.tostring(scene, encoding="unicode"),
                          encoding="utf-8")
    return scene_path, requests


# --------------------------------------------------------------- physics rows
def physics_row(api, *, cav_ids) -> dict:
    """A chunk-4 ``physics_steps.jsonl`` row (FORMAT.md section 0)."""
    vehs = {}
    for vid in api.vehicle.getIDList():
        vehs[str(vid)] = {
            "lane_id": api.vehicle.getLaneID(vid),
            "lane_position_m": float(api.vehicle.getLanePosition(vid)),
            "speed_mps": float(api.vehicle.getSpeed(vid)),
            "acceleration_mps2": float(api.vehicle.getAcceleration(vid)),
            "road_id": api.vehicle.getRoadID(vid),
            "length_m": float(api.vehicle.getLength(vid)),
        }
    return {
        "time_s": float(api.simulation.getTime()),
        "vehicles": vehs,
        "departed_ids": [str(v) for v in getattr(api.simulation, "departed_ids", ())],
        "cav_ids": sorted(str(v) for v in cav_ids),
    }


def build_states(api, cav_ids) -> list:
    """t0/current CAV states for the schedules (report-frames-only view).

    Captures the native IDM acceleration and, when the api exposes
    ``getLeader``, the current same-lane leader gap/speed the R5/R6 shared
    generators consume.  The runner never issues a lateral command.
    """
    states = []
    accel_getter = getattr(api.vehicle, "getAcceleration", None)
    leader_getter = getattr(api.vehicle, "getLeader", None)
    for vid in cav_ids:
        lane_id = api.vehicle.getLaneID(vid)
        native_accel = (float(accel_getter(vid))
                        if accel_getter is not None else None)
        leader_gap = None
        leader_v = None
        if leader_getter is not None:
            lead = leader_getter(vid)
            if lead:
                leader_id, distance = lead[0], float(lead[1])
                leader_gap = distance
                try:
                    leader_v = float(api.vehicle.getSpeed(leader_id))
                except Exception:  # noqa: BLE001 - leader may have left
                    leader_v = None
        states.append(schedules.CavState(
            vehicle_id=str(vid),
            speed_mps=float(api.vehicle.getSpeed(vid)),
            s_down_m=float(api.vehicle.getLanePosition(vid)),
            lane_phys=int(_lane_phys(lane_id)),
            allowed_speed_mps=float(api.vehicle.getAllowedSpeed(vid)),
            native_accel_mps2=native_accel,
            accel_mps2=(0.0 if native_accel is None else native_accel),
            leader_gap_m=leader_gap,
            leader_v_mps=leader_v,
        ))
    return states


def _lane_phys(lane_id: str) -> int:
    """main_right/left lane_phys from the SUMO edge_lane (FORMAT.md 1)."""
    if lane_id.endswith("_0") and (lane_id.startswith("1191554636")
                                   or lane_id.startswith("11553048")):
        # entry right lane is lane_phys 1; ramp (accel) lane is lane_phys 2
        return 2 if lane_id.startswith("11553048") else 1
    if lane_id.endswith("_1"):
        return 0 if lane_id.startswith("1191554636") else 1
    if lane_id.endswith("_2"):
        return 1
    return -1


# ------------------------------------------------------------------- receipts
def build_receipt(*, world, arm, machine_id, platform, sumo_version,
                  saved_state_digest, obstacle, schedule_check,
                  collisions, teleports, executed_time_s, physics_rows,
                  complete=True, note=None) -> dict:
    """The section-8 receipt (schema-checked; never re-run once complete)."""
    return {
        "schema": C.RECEIPT_SCHEMA,
        "run": f"{world['world']}__{arm.lower()}",
        "world": world["world"],
        "arm": arm,
        "complete": bool(complete),
        "machine_id": str(machine_id),
        "platform": str(platform),
        "sumo_version": str(sumo_version),
        "registration_lf_sha256": C.REGISTRATION_LF_SHA,
        "saved_state_digest": saved_state_digest,
        "obstacle": obstacle,
        "arm_schedule_check": schedule_check,
        "collisions": list(collisions),
        "teleports": list(teleports),
        "executed_time_s": float(executed_time_s),
        "physics_rows": int(physics_rows),
        "control_dt_s": C.CONTROL_DT_S,
        "physics_dt_s": C.PHYSICS_DT_S,
        "finished_utc": _utc(),
        "note": note,
    }


RECEIPT_REQUIRED = (
    "schema", "run", "world", "arm", "complete", "machine_id", "platform",
    "sumo_version", "registration_lf_sha256", "saved_state_digest", "obstacle",
    "arm_schedule_check", "collisions", "teleports", "executed_time_s",
    "physics_rows", "finished_utc",
)


def validate_receipt(receipt: dict) -> None:
    for key in RECEIPT_REQUIRED:
        if key not in receipt:
            raise RunnerError(f"receipt missing {key!r}")
    if receipt["schema"] != C.RECEIPT_SCHEMA:
        raise RunnerError("receipt schema mismatch")
    if receipt["registration_lf_sha256"] != C.REGISTRATION_LF_SHA:
        raise RunnerError("receipt does not bind the registration")


def write_receipt(run_dir: Path, receipt: dict) -> Path:
    validate_receipt(receipt)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "RUN_RECEIPT.json"
    if path.exists():
        raise RunnerError(f"refusing to overwrite a receipt: {path}")
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------- runner
class IncidentRunner:
    """Gated orchestrator; inject ``api_factory``/``ownership_factory`` in tests."""

    def __init__(self, plan: dict, runs_root: Path | None = None, *,
                 machine_id: str, platform: str, gate_info: dict,
                 budget: gates.BudgetCounter | None = None,
                 api_factory=None, ownership_factory=None,
                 sumo_version: str = "SUMO 1.27.1"):
        self.plan = plan
        self.runs_root = Path(runs_root) if runs_root else C.RUNS_ROOT
        self.machine_id = machine_id
        self.platform = platform
        self.gate_info = gate_info
        self.budget = budget or gates.BudgetCounter()
        self.api_factory = api_factory
        self.ownership_factory = ownership_factory
        self.sumo_version = sumo_version
        # runner-start gates: the frozen wall-law v2.2 copies must match
        # PATCH_SHAS.json (registration e7fd3536) and the INC1 hook patch
        # provenance must verify (ADDENDUM_1 A1.6) before any arm can run.
        self.wall_v22_freeze = wall_v22_freeze.verify_frozen_v22_patches()
        self.inc1_hook_freeze = wall_v22_freeze.verify_inc1_hook_patch()

    # -- warm-up ---------------------------------------------------------
    def warm_up(self, world: dict, *, api=None) -> dict:
        run_dir = self.runs_root / world["warmup_run"]
        gates.refuse_retry_of_complete(run_dir)
        own_api = api is None
        if own_api:
            api = self._start_api(world, run_dir)
        try:
            cav_ids = self._cav_ids(api)
            rows = self._advance_native(api, world, run_dir, cav_ids)
            state_path = run_dir / "run" / "state.xml"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_state(api, state_path)
            digest = C.raw_sha256(state_path) if state_path.is_file() else None
            receipt = build_receipt(
                world=world, arm="warmup", machine_id=self.machine_id,
                platform=self.platform, sumo_version=self.sumo_version,
                saved_state_digest=digest,
                obstacle=None,
                schedule_check={"mode": "warmup_native", "t0_s": C.T0_S},
                collisions=self._collisions(api), teleports=self._teleports(api),
                executed_time_s=C.T0_S, physics_rows=rows)
            write_receipt(run_dir, receipt)
            return receipt
        finally:
            if own_api and api is not None:
                self._close(api)

    def _advance_native(self, api, world, run_dir, cav_ids) -> int:
        log = (run_dir / "run" / "physics_steps.jsonl")
        log.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        while float(api.simulation.getTime()) < C.T0_S - 1e-9:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(physics_row(api, cav_ids=cav_ids)) + "\n")
            rows += 1
            api.simulation.simulationStep()
        return rows

    # -- arms ------------------------------------------------------------
    def run_arm(self, world: dict, arm: str, *, api=None, ownership=None,
                schedule=None, restore_digest=None, states_at_t0=None,
                obstacle=None) -> dict:
        run_dir = self.runs_root / f"{world['world']}__{arm.lower()}"
        gates.refuse_retry_of_complete(run_dir)
        self.budget.check(C.ARM_HORIZON_S)
        own_api = api is None
        if own_api:
            api = self._start_api(world, run_dir)
        try:
            cav_ids = self._cav_ids(api)
            restored = self._restore(api, world)
            obstacle_record = obstacle
            if obstacle_record is None:
                obstacle_record = self._insert_obstacle(api, world)
            if states_at_t0 is None:
                states_at_t0 = build_states(api, cav_ids)
            if schedule is None:
                schedule = schedules.make_schedule(
                    arm, states_at_t0, world["incident_s_down_m"])
            if ownership is None:
                if self.ownership_factory is not None:
                    ownership = self.ownership_factory(api, cav_ids)
                else:
                    ownership = RealOwnershipAdapter(api, cav_ids)
            result = self._drive(api, world, arm, run_dir, cav_ids, ownership,
                                 schedule)
            check = schedule_check(schedule, arm)
            receipt = build_receipt(
                world=world, arm=arm, machine_id=self.machine_id,
                platform=self.platform, sumo_version=self.sumo_version,
                saved_state_digest=restore_digest or restored,
                obstacle=obstacle_record,
                schedule_check=check,
                collisions=self._collisions(api), teleports=self._teleports(api),
                executed_time_s=result["executed_time_s"],
                physics_rows=result["physics_rows"])
            receipt["arm_feasibility"] = {
                "feasible": bool(check["feasible"]),
                "reason": check["infeasible_reason"],
            }
            write_receipt(run_dir, receipt)
            self.budget.charge(item="arm", sim_s=C.ARM_HORIZON_S,
                               run=receipt["run"])
            return receipt
        finally:
            if own_api and api is not None:
                self._close(api)

    def _drive(self, api, world, arm, run_dir, cav_ids, ownership, schedule):
        log = run_dir / "run" / "physics_steps.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        obstacle_id = "incident.obstacle"
        physics_steps = int(round(C.ARM_HORIZON_S / C.PHYSICS_DT_S))
        control_stride = int(round(C.CONTROL_DT_S / C.PHYSICS_DT_S))
        start = float(api.simulation.getTime())
        rows = 0
        for step in range(physics_steps):
            if step % control_stride == 0:
                now = float(api.simulation.getTime())
                states = build_states(api, cav_ids)
                plan = schedule.step(now, states)
                if ownership is not None:
                    ownership.apply(plan, now_s=now, live_ids=cav_ids,
                                    states=states, api=api)
            if hasattr(api.vehicle, "setSpeed"):
                try:
                    api.vehicle.setSpeed(obstacle_id, 0.0)
                except Exception:  # noqa: BLE001 - obstacle may not be present
                    pass
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(physics_row(api, cav_ids=cav_ids)) + "\n")
            rows += 1
            api.simulation.simulationStep()
        return {"executed_time_s": float(api.simulation.getTime()) - start,
                "physics_rows": rows}

    # -- R4 (patched frozen v2.2 driver; ADDENDUM_1 A1.2/A1.3) ------------
    def run_r4_arm(self, world: dict, *, api=None, session=None, ownership=None,
                   sensor=None, warmup_receipt=None, obstacle_hook=None,
                   hold_s: float | None = None) -> dict:
        """One R4 branch through ``inc1.r4_driver`` (feasibility flagged).

        The wall request is made at t0 (addendum A1.3); a proposer refusal or
        an aborted assembly is logged by the driver, the run completes and the
        receipt is flagged infeasible (never forced).
        """
        from . import r4_driver

        run_dir = self.runs_root / f"{world['world']}__r4"
        gates.refuse_retry_of_complete(run_dir)
        self.budget.check(C.ARM_HORIZON_S)
        own_api = api is None
        if own_api:
            api = self._start_api(world, run_dir)
        try:
            cav_ids = self._cav_ids(api)
            restored = self._restore(api, world)
            if warmup_receipt is None:
                warmup_receipt = {"saved_state_digest": restored}
            if obstacle_hook is None:
                def obstacle_hook(_api, _t_s, _world=world):
                    return self._insert_obstacle(_api, _world)
            if session is None or ownership is None:
                session, ownership, _sensor = r4_driver.build_r4_stack(
                    api, cav_ids)
                if sensor is None:
                    sensor = _sensor
            result = r4_driver.run_r4_branch(
                api, world, run_dir, warmup_receipt, session=session,
                ownership=ownership, sensor=sensor, cav_ids=cav_ids,
                obstacle_hook=obstacle_hook,
                hold_s=(r4_driver.R4_HOLD_S if hold_s is None else hold_s))
            schedule_check = {
                "arm": "R4", "schedule": "r4_driver",
                "feasible": bool(result["feasible"]),
                "infeasible_reason": result["infeasible_reason"],
                "wall_outcome": result["wall_outcome"],
            }
            receipt = build_receipt(
                world=world, arm="R4", machine_id=self.machine_id,
                platform=self.platform, sumo_version=self.sumo_version,
                saved_state_digest=restored, obstacle=None,
                schedule_check=schedule_check,
                collisions=self._collisions(api), teleports=self._teleports(api),
                executed_time_s=C.ARM_HORIZON_S,
                physics_rows=int(round(C.ARM_HORIZON_S / C.PHYSICS_DT_S)),
                note=("R4 patched frozen v2.2 driver"
                      if result["feasible"]
                      else f"R4 infeasible: {result['infeasible_reason']}"))
            receipt["arm_feasibility"] = {
                "feasible": bool(result["feasible"]),
                "reason": result["infeasible_reason"],
            }
            receipt["r4_result"] = str(run_dir / "r4_result.json")
            write_receipt(run_dir, receipt)
            self.budget.charge(item="arm", sim_s=C.ARM_HORIZON_S,
                               run=receipt["run"])
            return receipt
        finally:
            if own_api and api is not None:
                self._close(api)
    # -- dispatch --------------------------------------------------------
    def run_all(self, *, worlds=None, arms=None, workers: int = 1) -> dict:
        """Warm-up + every arm for every world (the owner's gated dispatch).

        R4 is driven through :meth:`run_r4_arm` (the patched frozen v2.2
        driver); R0-R3 run the INC1 loop.
        """
        worlds = self.plan["worlds"] if worlds is None else worlds
        arms = list(C.ARM_NAMES) if arms is None else list(arms)
        started = _utc()
        receipts = []
        for world in worlds:
            if self.api_factory is None:
                raise RunnerError("no api_factory: cannot dispatch the real run")
            self.budget.charge(item="warmup", sim_s=C.T0_S,
                               run=world["warmup_run"], detail="native to t0")
            receipts.append(self.warm_up(world))
            for arm in arms:
                if arm == "R4":
                    receipts.append(self.run_r4_arm(world))
                else:
                    receipts.append(self.run_arm(world, arm))
        summary = {
            "schema": "wm_n1_inc1_launch_summary/1",
            "started_utc": started, "finished_utc": _utc(),
            "worlds": len(worlds), "arms": arms,
            "runs": len(receipts),
            "workers": int(workers),
            "registration_lf_sha256": C.REGISTRATION_LF_SHA,
            "complete": all(r.get("complete") for r in receipts),
        }
        out = self.runs_root / "LAUNCH_SUMMARY.json"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    # -- helpers ---------------------------------------------------------
    def _start_api(self, world, run_dir):
        if self.api_factory is None:
            self.api_factory = default_api_factory
        return self.api_factory(world, run_dir)

    def _cav_ids(self, api) -> list:
        ids = api.vehicle.getIDList()
        return [str(v) for v in ids if _is_cav(v, api)]

    def _restore(self, api, world):
        state = self.runs_root / world["warmup_run"] / "run" / "state.xml"
        if not state.is_file():
            raise RunnerError(f"warm-up state is missing: {state}")
        if hasattr(api.simulation, "loadState"):
            api.simulation.loadState(str(state))
        return C.raw_sha256(state)

    def _insert_obstacle(self, api, world):
        lane_id = obstacles.incident_lane_id(world["incident_s_down_m"])
        return obstacles.place_obstacle(
            api, lane_id=lane_id,
            desired_s_down_m=world["incident_s_down_m"],
            s_down_base_m=obstacles.incident_lane_base(world["incident_s_down_m"]),
            obstacle_id="incident.obstacle")

    def _collisions(self, api):
        getter = getattr(api.simulation, "getCollisions", None)
        if getter is None:
            return []
        collisions = getter()
        out = []
        for row in collisions or ():
            if isinstance(row, dict):
                out.append(row)
            else:
                out.append({"collider": str(getattr(row, "collider", row))})
        return out

    def _teleports(self, api):
        out = []
        for name in ("getStartingTeleportNumber", "getEndingTeleportNumber"):
            getter = getattr(api.simulation, name, None)
            if getter is not None:
                count = int(getter())
                if count:
                    out.append({"kind": name, "count": count})
        return out

    def _save_state(self, api, path: Path):
        if hasattr(api.simulation, "saveState"):
            api.simulation.saveState(str(path))

    def _close(self, api):
        close = getattr(api, "close", None)
        if callable(close):
            close()


def ensure_scenario(world: dict, runs_root: Path) -> Path:
    """Build the world's SUMO scenario once (in the warm-up dir) and reuse it.

    All arms start from the SAME scenario file (common random numbers) and
    differ only by the restored state, exactly as registration section 2
    requires.
    """
    warmup = Path(runs_root) / world["warmup_run"]
    scene = warmup / "scenario" / "scenario.sumocfg"
    if not scene.is_file():
        build_scenario(warmup / "scenario", world["seed"],
                       world["loading_veh_h"], world["penetration"])
    return scene


def default_api_factory(world: dict, run_dir: Path):
    """Start SU(M)O from the world's shared scenario and return the traci api.

    Lazily imports ``traci``/``sumolib``; ONLY the gated launcher reaches
    this path (the tests inject a fake api).
    """
    import os

    import traci

    scene = ensure_scenario(world, Path(run_dir).parent)
    binary = os.environ.get("SUMO_BINARY", "sumo")
    command = [
        binary, "-c", str(scene), "--seed", str(int(world["seed"])),
        "--step-length", str(C.PHYSICS_DT_S), "--no-step-log", "true",
        "--collision.action", "warn", "--time-to-teleport", "-1",
    ]
    traci.start(command)
    return traci


class RealOwnershipAdapter:
    """The frozen CommandOwnership + SumoLocalSensor, as an ``apply`` seam.

    Lazily imports the frozen control stack; never constructed in this build
    (the tests inject a fake ownership).
    """

    def __init__(self, api, cav_ids, *, cav_speed_mode: int = C.CAV_SPEED_MODE):
        from traffic_shepherd.controller_factory import create_controller
        from traffic_shepherd.ownership import CommandOwnership
        from traffic_shepherd.sensing import SumoLocalSensor

        config = _load_config()
        self.api = api
        self.sensor = SumoLocalSensor(api, config)
        self.ownership = CommandOwnership(
            api, cav_ids, create_controller("damping", config),
            cav_speed_mode=cav_speed_mode)
        self._previous_delta: dict = {}

    def apply(self, plan, *, now_s, live_ids, states, api):
        frames = {}
        for vid in live_ids:
            frames[vid] = self.sensor.capture(
                vid, now_s=now_s,
                previous_command_delta=self._previous_delta.get(vid, 0.0))
        records = self.ownership.step(
            frames, now_s=now_s, live_ids=live_ids,
            wall_commands=plan.wall, native_ids=plan.native)
        for row in records:
            self._previous_delta[row.vehicle_id] = (
                float(row.issued_acceleration_mps2 or 0.0))
        return records


def _load_config():
    from traffic_shepherd.config import load_config
    return load_config()


def _is_cav(vid, api) -> bool:
    getter = getattr(api.vehicle, "getTypeID", None)
    if getter is None:
        return False
    type_id = str(getter(vid))
    return "cav" in type_id.lower()


def schedule_check(schedule, arm: str) -> dict:
    """The per-arm schedule self-check recorded in the receipt."""
    kind = type(schedule).__name__
    check = {"arm": arm, "schedule": kind}
    selected = getattr(schedule, "selected", None)
    if selected is not None:
        check["selected_vehicle"] = selected
    members = getattr(schedule, "members", None)
    if members is not None:
        check["wall_members"] = list(members)
    if isinstance(schedule, schedules.R1NearestStop):
        check["fell_back_to_R0"] = schedule.selected is None
    if isinstance(schedule, schedules.R2EarlySlow):
        check["controlled_n"] = len(schedule.controlled)
    feasible, reason = schedules.arm_feasibility(arm, schedule)
    check["feasible"] = bool(feasible)
    check["infeasible_reason"] = reason
    return check


# ------------------------------------------------------------------------- CLI
def _authorized_runner(argv=None) -> IncidentRunner:
    parser = argparse.ArgumentParser(description="INC1 gated incident runner")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--platform", default="linux")
    parser.add_argument("--machine-id", default="box")
    parser.add_argument("--runs-root", type=Path, default=C.RUNS_ROOT)
    parser.add_argument("--plan", type=Path, default=C.PLAN_PATH)
    parser.add_argument("--authorization", type=Path, default=C.AUTHORIZATION_PATH)
    parser.add_argument("--model-lock", type=Path, default=C.MODEL_LOCK_PATH)
    parser.add_argument("--registration", type=Path, default=C.REGISTRATION_PATH)
    parser.add_argument("--dispatch", action="store_true",
                        help="actually start SUMO (owner-run only; gated)")
    parser.add_argument("--world-limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise gates.GateError("worker count must be >= 1")
    gate_info = gates.check_launcher(
        authorization_path=args.authorization,
        model_lock_path=args.model_lock,
        registration_path=args.registration)
    gate_info["release_gates"] = gates.check_inc1_release_gates()
    gates.check_disk_preflight(args.runs_root)
    plan = gates.load_json(args.plan)
    if plan.get("schema") != C.PLAN_SCHEMA:
        raise gates.GateError(f"plan schema mismatch: {plan.get('schema')!r}")
    expected = gates.expected_sim_seconds(plan)
    budget = gates.BudgetCounter()
    budget.check(expected["subtotal_sim_s"])
    runner = IncidentRunner(plan, args.runs_root, machine_id=args.machine_id,
                            platform=args.platform, gate_info=gate_info,
                            budget=budget)
    return runner, args


def main(argv=None) -> int:
    runner, args = _authorized_runner(argv)
    print(json.dumps({
        "authorized": True,
        "worlds": len(runner.plan["worlds"]),
        "runs": len(runner.plan["runs"]),
        "registration_lf_sha256": runner.gate_info["registration_lf_sha256"],
        "wall_v22_freeze_all_match": runner.wall_v22_freeze["all_match"],
        "wall_v22_freeze_files": [r["file"]
                                  for r in runner.wall_v22_freeze["files"]],
        "inc1_hook_sha256": runner.inc1_hook_freeze["actual_lf_sha256"],
        "generators_manifest": runner.gate_info.get(
            "generators_manifest", {}).get("generators_manifest"),
        "generator_module_sha256": runner.gate_info.get(
            "generators_manifest", {}).get("generator_module_sha256"),
        "r4_driver_path": "inc1.r4_driver",
        "budget_total_sim_s": C.BUDGET_TOTAL_S,
        "dispatch": bool(args.dispatch),
    }, indent=1))
    if not args.dispatch:
        print("GATED launcher ready; pass --dispatch (owner) to run SUMO.")
        return 0
    worlds = runner.plan["worlds"]
    if args.world_limit is not None:
        worlds = worlds[:int(args.world_limit)]
    summary = runner.run_all(worlds=worlds, workers=args.workers)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
