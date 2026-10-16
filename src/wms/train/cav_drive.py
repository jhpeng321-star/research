"""Episode-persistent, timeline-driven CAV callbacks for optional closed-loop training.

Runs the published controller code on predicted states, without SUMO.  The model
plant is an approximation; only the default recorded teacher reproduces measured
generation trajectories exactly.  Controller state survives truncated-BPTT chunks.
The frozen controller classes are imported lazily (they live in the sim layer).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

from wms.data.action_timeline import CONTROLLERS

#: Timeline controller ids that need the sim-layer controllers: damping=1, pi-external-u=2.
_CONTROLLER_IDS_NEEDING_SIM = (1, 2)


def load_controller_dependencies(ids: np.ndarray) -> tuple:
    """The frozen sim-layer controller classes needed for ``ids``, via :mod:`wms.sim_bridge`.

    Returns ``(config, CirclesController, direct_target, NeighborMeasurement, SensorFrame,
    PIExternalU, UReport, PISaturationParameters)``; the three PI entries are ``None`` unless a
    PI controller (id 2) occurs in ``ids``.
    """
    from wms import sim_bridge

    try:
        deps = sim_bridge.load_controller_dependencies()
    except sim_bridge.SimBridgeError as exc:
        raise ValueError("closed_loop damping/PI requires the published controller "
                         f"sources: {exc}") from exc
    needs_pi = bool((ids == 2).any())
    pi = ((deps.pi_type, deps.u_report_type, deps.pi_params_type) if needs_pi
          else (None, None, None))
    return (deps.config, deps.circles_type, deps.direct_target, deps.neighbor_type,
            deps.frame_type, *pi)


class TimelineControllers:
    """``cav_controller`` callback: per-CAV native/damping/PI/direct commands on model states."""

    def __init__(self, model: Any, timeline: Optional[Mapping[str, Any]], *,
                 dependencies: Optional[tuple] = None) -> None:
        if timeline is None:
            raise ValueError("closed_loop training requires an action timeline")
        self.model, self.timeline = model, timeline
        self.ids = torch.as_tensor(timeline["controller"]).detach().cpu().numpy()
        self.acc = torch.as_tensor(timeline["acceleration_mps2"]).detach().cpu().numpy()
        self.target = torch.as_tensor(timeline["target_speed_mps"]).detach().cpu().numpy()
        self.step_index = 0
        self.active: dict[str, float] = {}
        self.damping: dict[str, Any] = {}
        self.pi_active: set[str] = set()
        self.dependencies = dependencies
        self.pi = None
        if np.isin(self.ids, _CONTROLLER_IDS_NEEDING_SIM).any():
            if dependencies is None:
                dependencies = load_controller_dependencies(self.ids)
            self.dependencies = dependencies
            _cfg, _c, _dt, _n, _f, pi_type, u_type, params_type = dependencies
            if (self.ids == 2).any():
                self.pi = pi_type(sample_period_s=1.0, parameters=params_type())
                self.u_report = u_type(31.29, "fixed_registered_setpoint", "cv07-registered-u31.29")

    def _invalidate(self, vid: str) -> None:
        if vid in self.pi_active:
            # PIExternalU.invalidate keeps only the state of ``vehicle_ids`` (the live set):
            # pass every active PI vehicle so dropping one does not erase the others.
            self.pi.invalidate(vehicle_ids=set(self.pi_active), native_ids=[vid])
            self.pi_active.discard(vid)

    def _new_command(self, action: str, vid: str, b: int, k: int, control: int,
                     state: Mapping[str, torch.Tensor]) -> Optional[float]:
        """Commanded acceleration of a damping / PI controller at a control boundary."""
        cfg, circles_type, direct_target, neighbor_type, frame_type, *_ = self.dependencies
        s, v, lane, present = state["s_m"], state["v_mps"], state["lane"], state["present"].bool()
        ahead = present[b] & (lane[b] == lane[b, k]) & (s[b] > s[b, k])
        ahead &= (s[b] - s[b, k]) <= cfg.sensor_range_m
        js = torch.nonzero(ahead).flatten()
        lead = int(js[torch.argmin(s[b, js])]) if js.numel() else None
        front = None
        coverage = cfg.sensor_range_m
        if lead is not None:
            gap = max(0.0, float(s[b, lead] - state["length_m"][b, lead] - s[b, k]))
            front = neighbor_type(gap, float(v[b, lead] - v[b, k]))
            coverage = max(gap, min(cfg.sensor_range_m, float(s[b, lead] - s[b, k])))
        frame = frame_type(
            vehicle_id=vid, timestamp_s=float(control), speed_mps=float(v[b, k]),
            acceleration_mps2=float(state["a_mps2"][b, k]),
            allowed_speed_mps=float(self.model.physics.speed_limit_mps),
            lane_index=int(lane[b, k]), front=front, front_coverage_m=coverage,
            rear_coverage_m=cfg.sensor_range_m, position_m=float(s[b, k]))
        if action == "damping":
            if vid not in self.damping:
                self.damping[vid] = circles_type(cfg)
            target = self.damping[vid].decide(frame, now_s=float(control)).target_speed_mps
        elif front is None:
            self._invalidate(vid)
            target = None
        else:
            if vid not in self.pi_active:
                self.pi.reset(vid, measured_ego_speed_mps=float(v[b, k]))
                self.pi_active.add(vid)
            proposal = self.pi.propose(
                vid, timestamp_s=float(control), ego_speed_mps=float(v[b, k]),
                leader_speed_mps=float(v[b, k] + front.relative_speed_mps), gap_m=front.gap_m,
                u_report=self.u_report)
            command, target = direct_target(frame, proposal.command_mps - float(v[b, k]), cfg)
            if command is None:
                self.pi.discard(proposal)
                self._invalidate(vid)
                target = None
            else:
                self.pi.commit(proposal, writer_ack=True)
        if target is None:
            return None
        return min(cfg.acceleration_max_mps2,
                   max(cfg.acceleration_min_mps2, float(target) - float(v[b, k])))

    @torch.no_grad()
    def __call__(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        model = self.model
        dt = float(model.physics.dt_s)
        s, v, lane = state["s_m"], state["v_mps"], state["lane"]
        cav = state["is_cav"].bool() & state["present"].bool()
        # Native handback remains native at each physics tick.
        accel = model.physics_accel(state).detach().clone()
        boundary = self.step_index % 2 == 0
        control = self.step_index // 2
        if control >= self.ids.shape[1]:
            raise ValueError("closed_loop action timeline exhausted")
        for b in range(s.shape[0]):
            for k in range(s.shape[1]):
                vid = f"{b}:{k}"
                if not cav[b, k] or lane[b, k] < 0:
                    self._invalidate(vid)
                    self.active.pop(vid, None)
                    continue
                action = CONTROLLERS[int(self.ids[b, control, k])]
                if boundary:
                    if action != "pi-external-u":
                        self._invalidate(vid)
                    self.active.pop(vid, None)
                    if action == "direct":
                        self.active[vid] = float(self.acc[b, control, k])
                    elif action != "native":
                        command = self._new_command(action, vid, b, k, control, state)
                        if command is not None:
                            self.active[vid] = command
                if vid in self.active:
                    request = self.active[vid]
                    if action == "direct":
                        request = min(request,
                                      (float(self.target[b, control, k]) - float(v[b, k])) / dt)
                    # speedMode31's safe plant can brake below the requested command.
                    accel[b, k] = min(float(accel[b, k]), request)
        self.step_index += 1
        speed = (v + accel * dt).clamp_min(0)
        position = s + 0.5 * (v + speed) * dt
        return torch.stack((position, lane.float(), speed, accel), dim=-1)
