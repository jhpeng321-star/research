"""The only module of ``wms`` that touches the frozen sim layer (``traffic_shepherd``).

Controller code (CIRCLES damping, PI external-U) is hash-pinned evidence and is
never copied or edited; it is imported lazily, on first use, from the repository
root (or from a staged ``_in/pi_deps`` tree on boxes).  The helpers here build
sensor frames and per-vehicle controllers exactly as the legacy training and
evaluation code did, so closed-loop CAV driving in a model plant is unchanged.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from .config.presets import repo_root
from .core.errors import WmsError

#: Setpoint passed to the PI controller ("cv07-registered-u31.29" in the evidence).
PI_U_SETPOINT_MPS = 31.29
PI_U_SOURCE = "fixed_registered_setpoint"
PI_U_LABEL = "cv07-registered-u31.29"
#: Control period of the CAV controllers in seconds.
PI_SAMPLE_PERIOD_S = 1.0

_STAGED_DEPS = Path("_in") / "pi_deps"
_REQUIRED_SOURCES = (
    Path("traffic_shepherd") / "damping.py",
    Path("output") / "controller_verification_v4" / "pi_external_u.py",
    Path("output") / "controller_verification_v2" / "pi_saturation_v1.py",
)


class SimBridgeError(WmsError, RuntimeError):
    """The frozen controller sources are missing or cannot be imported."""


@dataclass(frozen=True)
class ControllerDependencies:
    """The classes and functions of the frozen controller sources."""

    config: Any
    circles_type: Any
    direct_target: Any
    neighbor_type: Any
    frame_type: Any
    pi_type: Any
    u_report_type: Any
    pi_params_type: Any


def _source_root(root: Optional[Path]) -> Path:
    base = Path(root) if root is not None else repo_root()
    staged = base / _STAGED_DEPS
    if staged.is_dir():
        return staged
    if not all((base / rel).is_file() for rel in _REQUIRED_SOURCES):
        raise SimBridgeError(f"closed-loop controller source tree is missing under {base}")
    return base


def load_controller_dependencies(root: Optional[Path] = None) -> ControllerDependencies:
    """Import the frozen controller sources (lazily, once per call site)."""
    source = _source_root(root)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from output.controller_verification_v2.pi_saturation_v1 import PISaturationParameters
        from output.controller_verification_v4.pi_external_u import PIExternalU, UReport
        from traffic_shepherd.config import load_config
        from traffic_shepherd.damping import CirclesController, direct_target
        from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
    except ImportError as exc:
        raise SimBridgeError(f"closed-loop controller source import failed: {exc}") from exc
    return ControllerDependencies(load_config(), CirclesController, direct_target,
                                  NeighborMeasurement, SensorFrame, PIExternalU, UReport,
                                  PISaturationParameters)


def make_sensor_frame(deps: ControllerDependencies, *, vehicle_id: str, timestamp_s: float,
                      speed_mps: float, acceleration_mps2: float, allowed_speed_mps: float,
                      lane_index: int, position_m: float, front_gap_m: Optional[float] = None,
                      front_relative_speed_mps: float = 0.0,
                      front_coverage_m: Optional[float] = None) -> Any:
    """Build the CAV's local sensor frame (same-lane leader only, 100 m range)."""
    cfg = deps.config
    front = (None if front_gap_m is None
             else deps.neighbor_type(front_gap_m, front_relative_speed_mps))
    coverage = cfg.sensor_range_m if front_coverage_m is None else front_coverage_m
    return deps.frame_type(
        vehicle_id=vehicle_id, timestamp_s=timestamp_s, speed_mps=speed_mps,
        acceleration_mps2=acceleration_mps2, allowed_speed_mps=allowed_speed_mps,
        lane_index=lane_index, front=front, front_coverage_m=coverage,
        rear_coverage_m=cfg.sensor_range_m, position_m=position_m)


def clamp_acceleration(deps: ControllerDependencies, target_speed_mps: float,
                       speed_mps: float) -> float:
    """Acceleration request toward ``target_speed_mps``, clipped to the controller limits."""
    cfg = deps.config
    return min(cfg.acceleration_max_mps2,
               max(cfg.acceleration_min_mps2, float(target_speed_mps) - float(speed_mps)))


class CavController(Protocol):
    """Episode-persistent longitudinal controller of a set of CAVs."""

    def target_speed(self, vehicle_id: str, frame: Any, *, now_s: float) -> Optional[float]:
        """Target speed for this control period, or ``None`` to hand back to native."""

    def invalidate(self, vehicle_id: str) -> None:
        """Forget per-vehicle state (the CAV left, or control was handed back)."""


class DampingController:
    """CIRCLES damping, one controller instance per vehicle, created on first use."""

    def __init__(self, deps: ControllerDependencies) -> None:
        self._deps = deps
        self._by_vehicle: dict[str, Any] = {}

    def target_speed(self, vehicle_id: str, frame: Any, *, now_s: float) -> Optional[float]:
        if vehicle_id not in self._by_vehicle:
            self._by_vehicle[vehicle_id] = self._deps.circles_type(self._deps.config)
        return self._by_vehicle[vehicle_id].decide(frame, now_s=now_s).target_speed_mps

    def invalidate(self, vehicle_id: str) -> None:
        self._by_vehicle.pop(vehicle_id, None)


class PiFleetController:
    """PI external-U fleet controller: one shared instance, per-vehicle activation."""

    def __init__(self, deps: ControllerDependencies) -> None:
        self._deps = deps
        self._pi = deps.pi_type(sample_period_s=PI_SAMPLE_PERIOD_S, parameters=deps.pi_params_type())
        self._u_report = deps.u_report_type(PI_U_SETPOINT_MPS, PI_U_SOURCE, PI_U_LABEL)
        self._active: set[str] = set()

    def target_speed(self, vehicle_id: str, frame: Any, *, now_s: float) -> Optional[float]:
        front = frame.front
        if front is None:
            self.invalidate(vehicle_id)
            return None
        ego = float(frame.speed_mps)
        if vehicle_id not in self._active:
            self._pi.reset(vehicle_id, measured_ego_speed_mps=ego)
            self._active.add(vehicle_id)
        proposal = self._pi.propose(
            vehicle_id, timestamp_s=now_s, ego_speed_mps=ego,
            leader_speed_mps=ego + front.relative_speed_mps, gap_m=front.gap_m,
            u_report=self._u_report)
        command, target = self._deps.direct_target(
            frame, proposal.command_mps - ego, self._deps.config)
        if command is None:
            self._pi.discard(proposal)
            self.invalidate(vehicle_id)
            return None
        self._pi.commit(proposal, writer_ack=True)
        return target

    def invalidate(self, vehicle_id: str) -> None:
        if vehicle_id in self._active:
            self._pi.invalidate(vehicle_ids=[vehicle_id], native_ids=[vehicle_id])
            self._active.discard(vehicle_id)


ControllerKind = Literal["damping", "pi"]


def make_cav_controller(kind: ControllerKind,
                        deps: Optional[ControllerDependencies] = None) -> CavController:
    """Create the controller for ``kind`` ("damping" or "pi"); loads sources when ``deps`` is None."""
    if kind not in ("damping", "pi"):
        raise SimBridgeError(f"unknown CAV controller kind {kind!r}; use 'damping' or 'pi'")
    deps = deps if deps is not None else load_controller_dependencies()
    return DampingController(deps) if kind == "damping" else PiFleetController(deps)
