"""Episode-persistent, timeline-driven CAV callbacks for optional training.

Runs the published controller code on predicted states, without SUMO. The model
plant is an approximation; only the default recorded teacher reproduces measured
generation trajectories exactly. Controller state survives truncated BPTT chunks.
"""
from __future__ import annotations

import numpy as np
import torch

from .action_timeline import CONTROLLERS


class TimelineControllers:
    def __init__(self, model, timeline, *, dependencies=None):
        if timeline is None:
            raise ValueError("closed_loop training requires an action timeline")
        self.model, self.timeline = model, timeline
        self.ids = torch.as_tensor(timeline["controller"]).detach().cpu().numpy()
        self.acc = torch.as_tensor(timeline["acceleration_mps2"]).detach().cpu().numpy()
        self.target = torch.as_tensor(timeline["target_speed_mps"]).detach().cpu().numpy()
        self.step_index = 0
        self.active, self.previous, self.damping = {}, {}, {}
        self.pi_active = set()
        self.dependencies = dependencies
        self.pi = None
        if np.isin(self.ids, [1, 2]).any():
            if dependencies is None:
                from traffic_shepherd.config import load_config
                from traffic_shepherd.damping import CirclesController, direct_target
                from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame

                pi_type = u_type = params_type = None
                if (self.ids == 2).any():
                    try:
                        from output.controller_verification_v4.pi_external_u import PIExternalU, UReport
                        from output.controller_verification_v2.pi_saturation_v1 import PISaturationParameters
                    except ImportError as exc:
                        raise ValueError("closed_loop PI requires merged published PI sources") from exc
                    pi_type, u_type, params_type = PIExternalU, UReport, PISaturationParameters
                dependencies = (load_config(), CirclesController, direct_target,
                                NeighborMeasurement, SensorFrame, pi_type, u_type, params_type)
            self.dependencies = dependencies
            cfg, _c, _dt, _n, _f, pi_type, u_type, params_type = dependencies
            if (self.ids == 2).any():
                self.pi = pi_type(sample_period_s=1.0, parameters=params_type())
                self.u_report = u_type(31.29, "fixed_registered_setpoint", "cv07-registered-u31.29")

    def _invalidate(self, vid):
        if vid in self.pi_active:
            self.pi.invalidate(vehicle_ids=[vid], native_ids=[vid])
            self.pi_active.discard(vid)

    @torch.no_grad()
    def __call__(self, state):
        model = self.model
        dt = float(model.physics.dt_s)
        s, v = state["s_m"], state["v_mps"]
        lane = state["lane"]
        present = state["present"].bool()
        cav = state["is_cav"].bool() & present
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
                        cfg, circles_type, direct_target, neighbor_type, frame_type, *_ = self.dependencies
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
                        frame = frame_type(vehicle_id=vid, timestamp_s=float(control),
                            speed_mps=float(v[b, k]), acceleration_mps2=float(state["a_mps2"][b, k]),
                            allowed_speed_mps=float(model.physics.speed_limit_mps),
                            lane_index=int(lane[b, k]), front=front, front_coverage_m=coverage,
                            rear_coverage_m=cfg.sensor_range_m, position_m=float(s[b, k]))
                        if action == "damping":
                            if vid not in self.damping:
                                self.damping[vid] = circles_type(cfg)
                            decision = self.damping[vid].decide(frame, now_s=float(control))
                            target = decision.target_speed_mps
                        elif front is None:
                            self._invalidate(vid)
                            target = None
                        else:
                            if vid not in self.pi_active:
                                self.pi.reset(vid, measured_ego_speed_mps=float(v[b, k]))
                                self.pi_active.add(vid)
                            proposal = self.pi.propose(vid, timestamp_s=float(control),
                                ego_speed_mps=float(v[b, k]), leader_speed_mps=float(v[b, k] + front.relative_speed_mps),
                                gap_m=front.gap_m, u_report=self.u_report)
                            command, target = direct_target(frame, proposal.command_mps - float(v[b, k]), cfg)
                            if command is None:
                                self.pi.discard(proposal)
                                self._invalidate(vid)
                                target = None
                            else:
                                self.pi.commit(proposal, writer_ack=True)
                        if target is not None:
                            self.active[vid] = min(cfg.acceleration_max_mps2,
                                max(cfg.acceleration_min_mps2, float(target) - float(v[b, k])))
                    self.previous[vid] = action
                if vid in self.active:
                    request = self.active[vid]
                    if action == "direct":
                        request = min(request, (float(self.target[b, control, k]) - float(v[b, k])) / dt)
                    # speedMode31's safe plant can brake below the requested command.
                    accel[b, k] = min(float(accel[b, k]), request)
        self.step_index += 1
        speed = (v + accel * dt).clamp_min(0)
        position = s + 0.5 * (v + speed) * dt
        return torch.stack((position, lane.float(), speed, accel), dim=-1)
