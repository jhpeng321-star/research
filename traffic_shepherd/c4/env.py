"""CavSumoEnv: ``wms.control.contract.CavEnv`` on the preserved merge network.

One environment = one TraCI connection in this process (run one per worker process).
Scenario: ``traffic_shepherd.scenarios.build_scenario`` (exogenous Poisson demand, driver
types, nested CAV identity draws at ``penetration``), 0.5 s physics, 1 s control,
CAV speedMode 27 (the campaign mode, ``CommandOwnership(cav_speed_mode=27)``), CAV
acceleration bounds [-3, 1.5] m/s^2 from the adapter config for damping / PI / wall; ``direct``
commands use the upper bound ``direct_accel_max_mps2`` (default 3.0 = the largest SUMO driver
accel; ACCELCAP_NOTES.md) so a policy's own stored cap (legacy 1.5, new 2.6) is what binds.
Every CAV uses the single "normal_car" vType (``cav_vtype``, default
CAV_VTYPE; also when released to native) -- a CAV has no driver personality.  Under speedMode 27
(bit 1 = respect the vType max acceleration) SUMO caps a CAV at that type's accel (2.6).
``cav_vtype=None`` restores the legacy behaviour (CAVs keep their drawn driver type).
``population="textbook"`` replaces the three discrete HDV types by a
continuous per-vehicle IDM/LC2013 population centred on the textbook highway values and puts every
CAV on the ``cav_typical`` vType (population medians, accel 1.2, decel 2.0; see
:mod:`traffic_shepherd.c4.population`); ``cav_vtype`` is then ignored.  Default "legacy3" keeps
every existing run reproducible.

Every CAV command goes through :class:`traffic_shepherd.ownership.CommandOwnership`
(``wall_commands={vid: accel}`` -> ``direct_target`` -> ``slowDown``), the sole
longitudinal writer; HDVs are never commanded.  Episode controllers:

* ``direct``  - per-CAV absolute accelerations from ``step(accel_by_slot)``
  (non-finite / missing entries release that CAV to native for the tick, or hand it to
  the frozen CIRCLES damping / a fresh frozen PI external-U session with
  ``step(..., fallback="damping" | "pi")``);
* ``native``  - every CAV released to SUMO IDM;
* ``damping`` - frozen CIRCLES ONNX controller;
* ``pi``      - frozen PI external-U (registered U = 31.29 m/s);
* ``wall``    - wall law v2.2 shape (``wms.physics.wall_exec``) on one CAV pair per
  mainline lane, other CAVs damping (see :mod:`traffic_shepherd.c4.baselines`).

The warmup (``warmup_s``) always runs native; native -> damping/pi uses the frozen
``FleetSwitchSession`` one-boundary switch at ``t = warmup_s``.

Lane commands (``direct`` only): ``step(accel, lanes=...)`` with ``lanes``
``[1,K]`` (-1 = stay / none, 0/1/2 = target WM lane) puts every CAV with a finite direct command
this tick under policy lane control: SUMO's own discretionary lane changing is disabled for it with
``laneChangeMode`` = ``LC_MODE_POLICY`` = 513, i.e. bits 0-1 = 01 (strategic changes required by
the route - lane ends, the on-ramp merge - stay ON unless they conflict with a TraCI request), bits
2-3 = 00 (no cooperative changes), bits 4-5 = 00 (no speed-gain changes), bits 6-7 = 00 (no
keep-right changes), bits 8-9 = 10 (TraCI-requested changes respect the speed/brake gaps of other
drivers, adapting speed), bits 10-11 = 00 (no sublane model).  The vehicle's previous mode is
restored when the CAV leaves policy lane control (released / no lanes passed).  A target lane is
issued as ``traci.vehicle.changeLane(vid, sumo_index, 1.0)`` with the SUMO index of the target WM
lane on the CAV's CURRENT edge (``geometry.LANE_TABLE``); requests that would straddle an edge end
within the 1 s request (SUMO indices differ across edges) are deferred, junction transits refused.
HDVs and non-policy controllers are never touched.  Outcomes are counted per tick in
``info["lane_commands"]`` (issued / completed / not_completed / deferred_edge_end / refused_*),
written to the action timeline (``lane_command``) and to ``episode.npz`` ``lane_command_target``;
the requested targets (before this safety mask) go to ``lane_request_target``.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from traffic_shepherd.config import load_config
from traffic_shepherd.metrics import RequestLedger, braking_effort, closing_ttc, service_summary
from traffic_shepherd.scenarios import build_scenario, observation_map

from . import ROOT
from .geometry import LANE_TABLE, is_junction_lane, lane_and_s
from .recorder import EpisodeRecorder
from .seeds import check_case
from .state import (FrameBuilder, HistoryBuffer, ReferenceFrame, frame_cell_counts,
                    frame_to_state, rsu_detections)
from .sensing import TableSensor
from .sumo_io import (VehicleReader, action_entry, boundary_edges, build_physics_row,
                      crossings_between)

CONTROLLERS = ("direct", "native", "damping", "pi", "wall")
#: laneChangeMode of CAVs under policy lane control (see the module docstring for the bits).
LC_MODE_POLICY = 0b0010_0000_0001
LANE_REQUEST_S = 1.0               # changeLane duration = one control interval
LANE_EDGE_GUARD_M = 10.0           # same margin as wms.control.lanes.LaneRules.edge_guard_m
CAV_SPEED_MODE = 27
SPEED_LIMIT_MPS = 31.29
PI_U_MPS = 31.29
DEFAULT_K = 192
#: Upper bound of DIRECT commands (wms.control.contract.DIRECT_ACCEL_MAX_MPS2; ACCELCAP_NOTES.md).
DIRECT_ACCEL_MAX_MPS2 = 3.0
#: a CAV has no driver personality -- every CAV uses the "normal_car" vType (also
#: when released to native).  None = legacy (CAVs keep their drawn aggressive/normal/cautious type).
CAV_VTYPE = "normal_car"


class CavEnvError(RuntimeError):
    pass


class CavSumoEnv:
    """SUMO episode environment with a WM-frame slot state (batch 1)."""

    def __init__(self, *, k_slots: int = DEFAULT_K, controller: str = "direct",
                 scenario_root=None, check_parity: bool = False,
                 history_ticks: int = 241, use_subscriptions: bool = True,
                 ttc: bool = True, sumo_binary=None, device="cpu",
                 sensor_check_every_s: int = 10, physics_log: bool = True,
                 extra_sumo_cli=None, entry_queue_in_state: bool = False,
                 direct_accel_max_mps2: float = DIRECT_ACCEL_MAX_MPS2,
                 cav_vtype: str | None = CAV_VTYPE, population: str = "legacy3"):
        if controller not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}")
        self.config = load_config()
        self.k = int(k_slots)
        self.controller = controller
        self.scenario_root = (Path(scenario_root) if scenario_root is not None
                              else ROOT / "runs" / "active" / "c4_tmp")
        self.check_parity = bool(check_parity)
        self.sensor_check_every_s = max(1, int(sensor_check_every_s))
        self.physics_log = bool(physics_log)
        self.ttc = bool(ttc)
        self.use_subscriptions = bool(use_subscriptions)
        self.device = device
        self.history = HistoryBuffer(history_ticks)
        # extra SUMO launch arguments, e.g. ``--save-state.precision 17 --save-state.rng true``
        # for saved-state branching (traffic_shepherd.c4.branching)
        self.extra_sumo_cli = [str(a) for a in (extra_sumo_cli or ())]
        # opt-in ``state["entry_queue"]`` [1] (see :meth:`state`); off by default because
        # deploy / rsu_check / coop_imagine repeat or gather every state key.
        self.entry_queue_in_state = bool(entry_queue_in_state)
        # Direct-output upper bound (the policy's stored cap binds first; damping/PI/wall keep 1.5).
        self.direct_accel_max_mps2 = float(direct_accel_max_mps2)
        from .population import TEXTBOOK_CAV_VTYPE, check_population
        self.population = check_population(population)
        if self.population == "textbook":
            cav_vtype = TEXTBOOK_CAV_VTYPE          # CAVs have no driver type
        elif cav_vtype not in (None, "normal_car", "aggressive_car", "cautious_car"):
            raise ValueError(f"cav_vtype must be None or a scenario vType, got {cav_vtype!r}")
        self.cav_vtype = cav_vtype
        if not 0.0 < self.direct_accel_max_mps2 <= DIRECT_ACCEL_MAX_MPS2:
            raise ValueError(f"direct_accel_max_mps2 must be in (0, {DIRECT_ACCEL_MAX_MPS2}]")
        self.last_entry_queue_steps: list = []
        self.last_source_wait_steps: list = []
        self.binary = sumo_binary or shutil.which("sumo")
        if self.binary is None:
            raise CavEnvError("SUMO binary not found on PATH")
        self._circles = None
        self.api = None
        self.recorder = None
        self.done = True

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed: int, demand_vph: int, penetration: float, warmup_s: float,
              horizon_s: float, controller: str | None = None, episode_dir=None,
              meta: dict | None = None, warmup_hook=None):
        """Start a fresh SUMO episode, run the native warmup, return the t = warmup state.

        ``warmup_hook(env)`` (optional) is called after every native warmup control tick
        (e.g. to feed an RSU belief tracker before the evaluation window starts)."""
        import traci

        from traffic_shepherd.damping import CirclesController
        from traffic_shepherd.ownership import CommandOwnership
        from traffic_shepherd.sensing import SumoLocalSensor
        from traffic_shepherd.simulation import start_sumo

        self.close()
        mode = self.controller if controller is None else controller
        if mode not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}")
        check_case(seed, demand_vph)
        warmup, horizon = float(warmup_s), float(horizon_s)
        if (warmup < 0 or horizon < 1 or warmup != int(warmup) or horizon != int(horizon)
                or warmup + horizon > self.config.episode_seconds):
            raise ValueError("warmup/horizon must be whole seconds within the 2400 s fixture")
        if not 0.0 <= float(penetration) <= 1.0:
            raise ValueError("penetration must be in [0, 1]")
        self.mode, self.seed, self.demand = mode, int(seed), int(demand_vph)
        self.penetration = float(penetration)
        self.warmup_s, self.horizon_s = warmup, horizon
        self.end_s = warmup + horizon
        self._wall_started = time.monotonic()
        out = episode_dir if episode_dir is not None else None
        if out is not None:
            self.recorder = EpisodeRecorder(out, k=self.k, physics_log=self.physics_log)
            scene_root = self.recorder.scenario_dir
        else:
            self.scenario_root.mkdir(parents=True, exist_ok=True)
            scene_root = Path(tempfile.mkdtemp(prefix=f"s{seed}_m{demand_vph}_",
                                               dir=self.scenario_root))
        scene, requests, scenario_manifest = build_scenario(
            scene_root, self.seed, self.demand, penetration=self.penetration, role="exposed",
            config=self.config, cav_vtype=self.cav_vtype, population=self.population)
        self.requests = requests
        self.cav_ids = frozenset(r.vehicle_id for r in requests if r.is_cav)
        self.scenario_manifest = scenario_manifest
        if traci.isLoaded():
            raise CavEnvError("a TraCI connection is already active in this process")
        command = []
        self.scene = scene              # relaunch target (saved-state branching)
        start_sumo(self.binary, scene, seed=self.seed, config=self.config, api=traci,
                   command_record=command, extra_cli=self.extra_sumo_cli or None)
        self.api = traci
        self.done = False
        if self._circles is None:
            self._circles = CirclesController(self.config)
        self.ownership = CommandOwnership(traci, self.cav_ids, self._circles,
                                          cav_speed_mode=CAV_SPEED_MODE)
        lane_map = observation_map(traci, self.config)
        self.sensor = TableSensor(traci, self.config, lane_map)
        self.slow_sensor = (SumoLocalSensor(traci, self.config, lane_map)
                            if self.check_parity else None)
        self._table: dict = {}
        self.sensor_mismatches = 0
        self.sensor_checks = 0
        self.reader = VehicleReader(traci, cav_ids=self.cav_ids,
                                    use_subscriptions=self.use_subscriptions)
        self.builder = FrameBuilder(self.k, self.cav_ids)
        self.reference = ReferenceFrame(traci, self.config) if self.check_parity else None
        self.edges = boundary_edges(self.config)
        self.ledger = RequestLedger(requests)
        self.history.clear()
        self.last_entry_queue_steps, self.last_source_wait_steps = [], []
        self._setup_controllers(mode)
        self._previous_row = None
        self._previous_live: set = set()
        self._previous_live_sub: set = set()
        self._retired_cavs: set = set()
        self._collision_previous: set = set()
        self.totals = Counter()
        self.window_totals = Counter()
        self.minimum_ttc = None
        self.command_counts = Counter()
        self.fallback_counts = Counter()
        self.parity_failures = 0
        self.dropped_cav_steps = 0
        self.max_present = 0
        self._last_actions: dict = {}
        self._lc_original: dict = {}       # vid -> laneChangeMode before policy lane control
        self._lane_lengths: dict = {}
        self.lane_command_counts = Counter()
        self._last_lane_counts: dict = {}
        if self.recorder is not None:
            self.recorder.start(dict(
                seed=self.seed, demand_vph=self.demand, penetration=self.penetration,
                warmup_s=warmup, horizon_s=horizon, controller=mode, k_slots=self.k,
                cav_speed_mode=CAV_SPEED_MODE, scenario=scenario_manifest,
                launch_command=[str(c) for c in command[0]] if command else None,
                sumo_version=list(traci.getVersion()), **(meta or {})))
        # t = 0 state (empty network) so state index i <-> time 0.5 i
        self.frame = self.builder.build(0.0, {})
        self.history.append(self.frame)
        if self.recorder is not None:
            self.recorder.on_frame(self.frame, None)
        while self.now < warmup - 1e-9:
            self._control_tick(None)
            if warmup_hook is not None:
                warmup_hook(self)
        self.window_totals = Counter()
        self._window_wall_started = time.monotonic()
        self._window_ledger0 = (self.ledger.system_vehicle_seconds,
                                self.ledger.source_vehicle_seconds, len(self.ledger.arrived))
        return self.state()

    def close(self) -> None:
        if self.api is not None:
            try:
                if self.api.isLoaded():
                    try:
                        self.ownership.release_all(live_ids=set(self.api.vehicle.getIDList()))
                    finally:
                        self.api.close()
            finally:
                self.api = None
        if self.recorder is not None and not self.done:
            self.recorder.fail("closed before the horizon")
        self.recorder = None
        self.done = True

    # -------------------------------------------------------------------- control
    def _setup_controllers(self, mode: str) -> None:
        from traffic_shepherd.controller_factory import create_pi_session
        from traffic_shepherd.fleet_switch import FleetSwitchSession, SingleSwitchSpec

        self.pi_session = None
        self._direct_pi = False         # a per-tick PI session inside a direct episode
        self.fleet = None
        self.wall_law = None
        if mode == "pi":
            if not self.cav_ids:
                raise CavEnvError("pi requires a nonempty CAV roster")
            self.pi_session = create_pi_session(self.config, self.cav_ids,
                                                desired_speed_mps=PI_U_MPS)
        if mode in ("damping", "pi"):
            suffix = "pi-external-u" if mode == "pi" else "damping"
            if self.warmup_s > 0:
                self.fleet = FleetSwitchSession(
                    SingleSwitchSpec("native", suffix, self.warmup_s),
                    circles_controller=self._circles, pi_session=self.pi_session,
                    cav_ids=self.cav_ids)
            elif mode == "pi":
                self.ownership.controller = self.pi_session.adapter
        if mode == "wall":
            from .baselines import WallLaw
            self.wall_law = WallLaw(self.k)

    @property
    def now(self) -> float:
        return float(self.api.simulation.getTime())

    def state(self) -> dict:
        state = frame_to_state(self.frame, device=self.device)
        if self.entry_queue_in_state:
            import torch
            queue = self.last_entry_queue_steps[-1] if self.last_entry_queue_steps else 0.0
            state["entry_queue"] = torch.tensor([float(queue)], dtype=torch.float32,
                                                device=self.device)
        return state

    def entry_queue_now(self) -> int:
        """Vehicles requested (``ledger.due(now)``) but not yet inserted, at the current time."""
        return int(self.ledger.due(self.now) - len(self.ledger.inserted))

    def slot_vehicle_ids(self) -> list:
        return self.builder.slot_vehicle_ids(self.frame)

    def _accel_by_vehicle(self, accel_by_slot) -> dict:
        if accel_by_slot is None:
            return {}
        try:
            import torch
            if torch.is_tensor(accel_by_slot):
                accel_by_slot = accel_by_slot.detach().cpu().numpy()
        except ImportError:  # pragma: no cover
            pass
        acc = np.asarray(accel_by_slot, dtype=np.float64).reshape(-1)
        if acc.shape[0] != self.k:
            raise ValueError(f"accel_by_slot must have K={self.k} entries (shape [1,K])")
        out = {}
        f = self.frame
        for slot in np.flatnonzero(f.present & f.is_cav).tolist():
            vid = self.builder.vehicle_ids[int(f.vid_index[slot])]
            out[vid] = float(acc[slot])
        return out

    def _lanes_by_vehicle(self, lanes) -> dict:
        """``[1,K]`` WM target lanes -> {vid: lane} for present CAV slots (-1 / non-finite dropped)."""
        try:
            import torch
            if torch.is_tensor(lanes):
                lanes = lanes.detach().cpu().numpy()
        except ImportError:  # pragma: no cover
            pass
        arr = np.asarray(lanes, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.k:
            raise ValueError(f"lanes must have K={self.k} entries (shape [1,K])")
        out = {}
        f = self.frame
        for slot in np.flatnonzero(f.present & f.is_cav).tolist():
            value = arr[slot]
            if math.isfinite(value) and value >= 0:
                out[self.builder.vehicle_ids[int(f.vid_index[slot])]] = int(value)
        return out

    def _lane_control(self, policy_cavs: set, live: set) -> None:
        """Policy lane control on/off (laneChangeMode); HDVs are never touched."""
        api = self.api
        for vid in sorted(policy_cavs - set(self._lc_original)):
            self._lc_original[vid] = int(api.vehicle.getLaneChangeMode(vid))
            api.vehicle.setLaneChangeMode(vid, LC_MODE_POLICY)
        for vid in sorted(set(self._lc_original) - policy_cavs):
            original = self._lc_original.pop(vid)
            if vid in live:
                api.vehicle.setLaneChangeMode(vid, original)

    def _issue_lane_commands(self, targets: dict, policy_cavs: set) -> tuple[dict, Counter]:
        """``changeLane`` for each policy CAV whose target differs from its current WM lane."""
        api = self.api
        issued, counts = {}, Counter()
        for vid, target in sorted(targets.items()):
            if vid not in policy_cavs:
                counts["refused_not_policy"] += 1
                continue
            row = self._table.get(vid)
            if row is None or is_junction_lane(row["lane_id"]):
                counts["refused_junction"] += 1
                continue
            current, _ = lane_and_s(row["lane_id"], row["lane_position_m"])
            if target == current:
                counts["noop"] += 1
                continue
            if target not in (0, 1, 2) or abs(target - current) != 1 or target == 2:
                counts["refused_not_allowed"] += 1     # never into the ramp lane, adjacent only
                continue
            edge = row["road_id"]
            index = next((i for (e, i), (lane, _) in LANE_TABLE.items()
                          if e == edge and lane == target), None)
            if index is None:
                counts["refused_no_lane"] += 1
                continue
            length = self._lane_lengths.get(row["lane_id"])
            if length is None:
                length = self._lane_lengths[row["lane_id"]] = float(api.lane.getLength(row["lane_id"]))
            if length - row["lane_position_m"] <= row["speed_mps"] * LANE_REQUEST_S + LANE_EDGE_GUARD_M:
                counts["deferred_edge_end"] += 1
                continue
            api.vehicle.changeLane(vid, int(index), LANE_REQUEST_S)
            issued[vid] = (target, int(index))
            counts["issued"] += 1
        return issued, counts

    def _control_tick(self, accel_by_slot, fallback: str = "native", lanes=None,
                      fallback_ids=None) -> dict:
        api = self.api
        now = self.now
        if abs(now - round(now)) > 1e-6:
            raise CavEnvError(f"SUMO time {now} is off the 1 s control grid")
        lane_issued, lane_counts = {}, Counter()
        lane_requested: dict = {}
        live = set(api.vehicle.getIDList())
        cav_ids = self.cav_ids
        disappeared = (self._previous_live - live) & cav_ids
        arrivals = set(api.simulation.getArrivedIDList()) & cav_ids
        self._retired_cavs.update(disappeared | arrivals)
        reappeared = self._retired_cavs & live
        self._retired_cavs.difference_update(reappeared)
        live_cavs = live & cav_ids
        frames = {}
        self.sensor.bind(self._table)
        for cav in sorted(live_cavs):
            try:
                frames[cav] = self.sensor.capture(cav, now_s=now)
            except (ValueError, KeyError):
                frames.pop(cav, None)
            if self.slow_sensor is not None and int(round(now)) % self.sensor_check_every_s == 0:
                self.sensor_checks += 1
                try:
                    reference = self.slow_sensor.capture(cav, now_s=now)
                except ValueError:
                    reference = None
                if reference != frames.get(cav):
                    self.sensor_mismatches += 1
        in_window = now >= self.warmup_s - 1e-9
        mode = self.mode if in_window else "native"
        # Direct-mode ticks raise the wall/direct request ceiling; every other tick (wall law,
        # damping, PI, native) keeps the adopted config bound.
        self.ownership.wall_accel_max_mps2 = None
        requested = {}
        if mode != "direct" and self._lc_original:
            self._lane_control(set(), live)
        if self.fleet is not None:
            records = self.fleet.step(self.ownership, frames, now_s=now, live_ids=live,
                                      disappeared_ids=disappeared, reappeared_ids=reappeared)
        elif mode == "pi":
            records = self.pi_session.step(self.ownership, frames, now_s=now, live_ids=live,
                                           native_ids=(), departed_ids=disappeared,
                                           id_reuse_ids=reappeared)
        elif mode == "native":
            records = self.ownership.step(frames, now_s=now, live_ids=live, native_ids=live_cavs)
        elif mode == "damping":
            records = self.ownership.step(frames, now_s=now, live_ids=live)
        elif mode == "direct":
            self.ownership.wall_accel_max_mps2 = self.direct_accel_max_mps2
            requested = self._accel_by_vehicle(accel_by_slot)
            wall = {v: a for v, a in requested.items()
                    if v in live_cavs and math.isfinite(a)}
            # CAVs without a finite direct command: SUMO native (default) or the frozen
            # CIRCLES damping instance of CommandOwnership (fallback="damping").
            # fallback="pi": a FRESH frozen PI session (registered U) owns them from this tick
            # while PI stays selected (the session requires every-tick continuity); leaving PI
            # hands the owner back to the identical CIRCLES instance.
            # fallback_ids (optional): only these CAVs may use the fallback controller; other
            # live CAVs without a finite direct command are released to native.
            restricted = (frozenset() if fallback_ids is None
                          else (live_cavs - set(wall)) - frozenset(fallback_ids))
            if fallback == "pi":
                if not self._direct_pi:
                    from traffic_shepherd.controller_factory import create_pi_session
                    self.pi_session = create_pi_session(self.config, self.cav_ids,
                                                        desired_speed_mps=PI_U_MPS)
                    self.ownership.controller = self.pi_session.adapter
                    self._direct_pi = True
                records = self.pi_session.step(self.ownership, frames, now_s=now, live_ids=live,
                                               native_ids=restricted, wall_commands=wall,
                                               departed_ids=disappeared, id_reuse_ids=reappeared)
            else:
                if self._direct_pi:
                    self.ownership.controller = self._circles
                    self.pi_session, self._direct_pi = None, False
                native = live_cavs - set(wall) if fallback == "native" else restricted
                records = self.ownership.step(frames, now_s=now, live_ids=live,
                                              wall_commands=wall, native_ids=native)
            # Policy lane control: only CAVs with a finite direct command AND a lanes tensor.
            policy_cavs = set(wall) if lanes is not None else set()
            self._lane_control(policy_cavs, live)
            if lanes is not None:
                lane_requested = self._lanes_by_vehicle(lanes)
                lane_issued, lane_counts = self._issue_lane_commands(lane_requested, policy_cavs)
        elif mode == "wall":
            wall = self.wall_law.commands(self.frame, self.builder, now_s=now,
                                          live_cavs=live_cavs)
            requested = dict(wall)
            records = self.ownership.step(frames, now_s=now, live_ids=live, wall_commands=wall)
        else:  # pragma: no cover
            raise CavEnvError(mode)
        if self.pi_session is not None:
            self.pi_session.commit_evidence()       # PI evidence is not persisted here
        actions, decisions = {}, {}
        for r in records:
            self.command_counts[r.owner] += 1
            if r.fallback_reason:
                self.fallback_counts[r.fallback_reason] += 1
            if r.owner == "native":
                name = "native"
            elif mode == "pi" or (self._direct_pi and r.owner != "wall"):
                name = "pi-external-u"
            elif r.owner == "wall":
                name = "direct"
            else:
                name = "damping"
            actions[r.vehicle_id] = action_entry(r.owner, name, r.issued_target_speed_mps)
            slot = self.builder.pool.slot_of(r.vehicle_id)
            if slot is not None and self.frame.present[slot]:
                decisions[slot] = (name, r.issued_acceleration_mps2, r.issued_target_speed_mps,
                                   requested.get(r.vehicle_id))
        if self.recorder is not None:
            lane_rows = {}
            for vid, (target, index) in lane_issued.items():
                slot = self.builder.pool.slot_of(vid)
                if slot is not None and self.frame.present[slot]:
                    lane_rows[slot] = {"target_lane": target, "sumo_lane_index": index,
                                       "duration_s": LANE_REQUEST_S}
            request_rows = {}
            for vid, target in lane_requested.items():
                slot = self.builder.pool.slot_of(vid)
                if slot is not None and self.frame.present[slot]:
                    request_rows[slot] = int(target)
            self.recorder.on_control(now, self.frame, decisions, self.builder.vehicle_ids,
                                     lane_commands=lane_rows, lane_requests=request_rows)
        self._last_actions = actions
        commanded = {r.vehicle_id for r in records}
        info_interval = Counter()
        source_wait_steps = []
        for substep in range(2):
            api.simulationStep()
            after = self.now
            if abs(after - (now + (substep + 1) * 0.5)) > 1e-6:
                raise CavEnvError("SUMO physics time grid mismatch")
            sub = self._after_physics(after, actions, now, commanded)
            source_wait_steps.append(float(sub.get("source_wait_vehicle_seconds", 0.0)))
            info_interval.update(sub)
        # per-physics-step source wait (vehicle-seconds) and its time-averaged queue (/ 0.5 s)
        self.last_source_wait_steps = source_wait_steps
        self.last_entry_queue_steps = [w / 0.5 for w in source_wait_steps]
        self._previous_live = set(api.vehicle.getIDList())
        for vid, (target, _) in lane_issued.items():
            slot = self.builder.pool.slot_of(vid)
            done = (slot is not None and bool(self.frame.present[slot])
                    and int(self.frame.lane[slot]) == target)
            lane_counts["completed" if done else "not_completed"] += 1
        self.lane_command_counts.update(lane_counts)
        self._last_lane_counts = dict(lane_counts)
        self.totals.update(info_interval)
        if in_window:
            self.window_totals.update(info_interval)
        return dict(info_interval)

    def _after_physics(self, after: float, actions: dict, command_time: float,
                       commanded: set) -> Counter:
        api = self.api
        sim = api.simulation
        departed = list(sim.getDepartedIDList())
        arrived = list(sim.getArrivedIDList())
        colliding = set(sim.getCollidingVehiclesIDList())
        starting = list(sim.getStartingTeleportIDList())
        ending = list(sim.getEndingTeleportIDList())
        table = self.reader.read()
        self._table = table
        if self.pi_session is not None and ((self.mode == "pi" and after > self.warmup_s)
                                            or self._direct_pi):
            live_after = set(table)
            prev = self._previous_live_sub
            disappeared = (prev - live_after) & self.cav_ids
            arrived_cav = set(arrived) & self.cav_ids
            self._retired_cavs.update(disappeared | arrived_cav)
            reappeared = self._retired_cavs & live_after
            self._retired_cavs.difference_update(reappeared)
            self.pi_session.observe_lifecycle(live_ids=live_after, disappeared_ids=disappeared,
                                              id_reuse_ids=reappeared)
        self._previous_live_sub = set(table)
        frame = self.builder.build(after, table)
        self.frame = frame
        self.history.append(frame)
        self.max_present = max(self.max_present, int(frame.present.sum()))
        if frame.dropped:
            self.dropped_cav_steps += sum(1 for v in frame.dropped if v in self.cav_ids)
        interval = Counter()
        interval["dropped_vehicle_steps"] += len(frame.dropped)
        speeds = {v: t["speed_mps"] for v, t in table.items()}
        accelerations = [t["acceleration_mps2"] for t in table.values()]
        interval.update(braking_effort(accelerations, 0.5))
        interval["speed_deficit_vehicle_seconds"] += 0.5 * sum(
            max(0.0, 1.0 - s / SPEED_LIMIT_MPS) for s in speeds.values())
        interval["vehicle_seconds_on_road"] += 0.5 * len(table)
        if self.ttc:
            exposure = 0.0
            for vid, t in table.items():
                leader = t["leader"]
                if leader is None or leader[0] not in speeds:
                    continue
                ttc = closing_ttc(leader[1] + t["min_gap_m"], speeds[vid], speeds[leader[0]])
                if ttc is not None:
                    self.minimum_ttc = ttc if self.minimum_ttc is None else min(
                        self.minimum_ttc, ttc)
                    exposure += 0.5 if ttc < 3.0 else 0.0
            interval["ttc_below_3_vehicle_seconds"] += exposure
        before = (self.ledger.system_vehicle_seconds, self.ledger.source_vehicle_seconds)
        interval["requested_not_arrived_vehicle_seconds"] += self.ledger.advance(
            after, departed, arrived)
        interval["source_wait_vehicle_seconds"] += self.ledger.source_vehicle_seconds - before[1]
        interval["arrivals"] += len(arrived)
        interval["departures"] += len(departed)
        interval["collision_involvements"] += len(colliding - self._collision_previous)
        self._collision_previous = colliding
        interval["starting_teleports"] += len(starting)
        row = None
        if self.recorder is not None and self.physics_log:
            row = build_physics_row(after, table, previous_row=self._previous_row,
                                    cav_ids=self.cav_ids, cav_actions=actions, edges=self.edges,
                                    departed=departed, arrived=arrived, colliding=colliding,
                                    starting_teleports=starting, ending_teleports=ending)
            crossings = row["boundary_crossings"]
        else:
            crossings = crossings_between((self._previous_row or {}).get("vehicles", {}),
                                          table, self.edges)
        for key, value in crossings.items():
            interval[f"{key}_crossings"] += value
        self._previous_row = {"vehicles": table}
        if self.check_parity:
            ref = self.reference.cell_counts(table)
            mine = frame_cell_counts(frame)
            if not np.array_equal(ref, mine):
                self.parity_failures += 1
                interval["rsu_parity_failures"] += 1
        if self.recorder is not None:
            self.recorder.on_frame(frame, row)
        return interval

    def step(self, accel_by_slot=None, *, fallback: str = "native", lanes=None, lane=None,
             fallback_ids=None):
        """Apply one 1 s control interval; returns ``(state, info)``.

        ``accel_by_slot`` ``[1, K]`` (torch or numpy) absolute CAV accelerations, used by the
        ``direct`` controller only (ignored otherwise; non-CAV slots ignored).  ``fallback``
        (``direct`` only): controller of CAVs without a finite command this tick, ``native``
        (release to SUMO IDM, default), ``damping`` (frozen CIRCLES) or ``pi`` (frozen PI
        external-U, a fresh session each time PI is (re)selected); per-tick mixing used by the
        C4 deployment check, ``wms.control.deploy``.  ``lanes`` (``direct`` only, optional) ``[1,K]``
        WM target lanes (-1 = stay / none) puts the commanded CAVs under policy lane control (module
        docstring); ``None`` keeps SUMO's own lane changing for every vehicle (accel-only callers).
        ``fallback_ids`` (``direct`` only, optional): vehicle ids allowed to use the fallback
        controller; every other live CAV without a finite command is released to native (subset
        arms of :mod:`traffic_shepherd.c4.branching`)."""
        if fallback not in ("native", "damping", "pi"):
            raise ValueError("fallback must be 'native', 'damping' or 'pi'")
        if lane is not None:                      # alias (``lane=`` [1,K], -1 = no change)
            if lanes is not None:
                raise ValueError("pass either lanes= or lane=, not both")
            lanes = lane
        if self.done:
            raise CavEnvError("episode is done; call reset()")
        t = self.now
        interval = self._control_tick(accel_by_slot, fallback, lanes, fallback_ids)
        done = self.now >= self.end_s - 1e-9
        info = {"time_s": self.now, "decision_time_s": t, "interval": interval,
                "actions": {v: a["action"] for v, a in self._last_actions.items()},
                "lane_commands": dict(getattr(self, "_last_lane_counts", {})),
                "rsu_detections": rsu_detections(self.frame, self.builder.vehicle_ids),
                "dropped_vehicles": list(self.frame.dropped), "done": done,
                "source_wait_steps": list(self.last_source_wait_steps),
                "entry_queue_steps": list(self.last_entry_queue_steps)}
        if done:
            info["summary"] = self.summary()
            if self.recorder is not None:
                self.recorder.finish(info["summary"], vehicle_ids=self.builder.vehicle_ids,
                                     cav_ids=self.cav_ids, meta=self._meta())
            self.done = True
            self.recorder = None
            self._close_sumo()
        return self.state(), info

    def _close_sumo(self) -> None:
        if self.api is not None and self.api.isLoaded():
            try:
                self.ownership.release_all(live_ids=set(self.api.vehicle.getIDList()))
            finally:
                self.api.close()
        self.api = None

    def _meta(self) -> dict:
        return dict(seed=self.seed, demand_vph=self.demand, penetration=self.penetration,
                    warmup_s=self.warmup_s, horizon_s=self.horizon_s, controller=self.mode,
                    k_slots=self.k, cav_speed_mode=CAV_SPEED_MODE, dt_s=0.5, control_dt_s=1.0)

    def summary(self) -> dict:
        now = self.now
        wall = time.monotonic() - self._wall_started
        window_wall = time.monotonic() - self._window_wall_started
        sys0, src0, arr0 = self._window_ledger0
        window = dict(self.window_totals)
        window.update(
            system_vehicle_seconds=self.ledger.system_vehicle_seconds - sys0,
            source_wait_vehicle_seconds=self.ledger.source_vehicle_seconds - src0,
            arrivals=len(self.ledger.arrived) - arr0)
        return dict(
            schema="c4_sumo_episode_summary/1", **self._meta(), sim_end_s=now,
            designated_cavs=len(self.cav_ids), population=len(self.requests),
            episode=dict(**service_summary(self.ledger, now), totals=dict(self.totals)),
            window=window, minimum_closing_ttc_s=self.minimum_ttc,
            command_counts=dict(self.command_counts),
            fallback_counts=dict(self.fallback_counts),
            lane_commands=dict(self.lane_command_counts), lane_change_mode_policy=LC_MODE_POLICY,
            collisions=int(self.totals.get("collision_involvements", 0)),
            teleports=int(self.totals.get("starting_teleports", 0)),
            max_present_vehicles=self.max_present,
            dropped_vehicle_steps=int(self.totals.get("dropped_vehicle_steps", 0)),
            dropped_cav_steps=self.dropped_cav_steps,
            rsu_parity_checked=self.check_parity, rsu_parity_failures=self.parity_failures,
            sensor_parity_checks=self.sensor_checks,
            sensor_parity_mismatches=self.sensor_mismatches,
            wall_events=(None if self.wall_law is None else dict(
                formed=self.wall_law.n_walls,
                counts=dict(Counter(e["event"] for e in self.wall_law.events)),
                abort_reasons=dict(Counter(e.get("reason") for e in self.wall_law.events
                                           if e.get("reason"))))),
            wall_seconds=wall, sim_seconds=now,
            sim_s_per_wall_s=(now / wall if wall > 0 else None),
            window_wall_seconds=window_wall,
            window_sim_s_per_wall_s=((now - self.warmup_s) / window_wall
                                     if window_wall > 0 else None))

    # --------------------------------------------------------------- WM inputs
    def history_state(self, history_s=None, history_stride_s: float = 1.0) -> dict:
        """``state["history"]`` (wms.eval slot-indexed schema) at the current time."""
        return self.history.history(history_s, history_stride_s)

    def driver_history(self, history_s=None, history_stride_s: float = 1.0,
                       history_len: int = 10):
        return self.history.driver_history(history_s, history_stride_s, history_len)
