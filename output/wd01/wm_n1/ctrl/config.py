"""P1 controller plug-in configuration (Idea A).

Freezes the CAV action alphabet and the controller constants reused from
``partA_ctrl.emulators`` (the verified CIRCLES damping ONNX port, the PI
external-U recurrence and the wall pacing law).  The wall-v2.2 constants are
NOT hard-coded here: they arrive through :mod:`ctrl.wall_hook` from
``zcode/wallmr`` when that branch lands.
"""
from __future__ import annotations

from dataclasses import dataclass

# sha256 pinned by baselines/circles/adapter.json; partA_ctrl.emulators refuses
# any other payload (emulators.py:57).
CIRCLES_MODEL_SHA256 = (
    "762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee")

# Canonical per-CAV actions.  "none" == native (no command); CAVs only.
CAV_ACTIONS = ("native", "damping", "pi", "wall")
ACTION_CODES = {"none": -1, "native": 0, "damping": 1, "pi": 2, "wall": 3}
NONE, NATIVE, DAMPING, PI, WALL = -1, 0, 1, 2, 3

# Wall-law version tags accepted by CtrlConfig.wall_law.
WALL_LAW_V21 = "wall-v2.1-closure"
WALL_LAW_V22 = "wall-v2.2"


@dataclass
class CtrlConfig:
    """Constants of the batched CAV mover.

    Physics and control cadence mirror the frozen Part A contract: a 0.5 s
    physics tick, a 1 s control tick, a 100 m same-lane sensing contract, the
    registered PI external target U = 31.29 m/s and the CIRCLES direct-target
    acceleration clip ``[-3.0, +1.5]`` m/s^2.
    """

    dt_s: float = 0.5
    control_dt_s: float = 1.0
    sensor_range_m: float = 100.0
    accel_min_mps2: float = -3.0
    accel_max_mps2: float = 1.5
    model_sha256: str = CIRCLES_MODEL_SHA256
    pi_u_mps: float = 31.29
    wall_law: str = WALL_LAW_V22
    wall_close_max_delta_mps: float = 2.0
    wall_close_gain_per_s: float = 0.1
    wall_assembly_timeout_s: float = 60.0
    wall_buffer_decel_mps2: float = -3.0
    wall_comfort_accel_mps2: float = 1.5
    nominal_length_m: float = 5.0

    def __post_init__(self) -> None:
        if self.wall_law not in (WALL_LAW_V21, WALL_LAW_V22):
            raise ValueError(
                f"wall_law must be {WALL_LAW_V21!r} or {WALL_LAW_V22!r}, "
                f"got {self.wall_law!r}")
        if float(self.pi_u_mps) != 31.29:
            raise ValueError("pi-u requires the registered U = 31.29 m/s")
        if self.dt_s <= 0 or self.control_dt_s <= 0:
            raise ValueError("dt_s and control_dt_s must be positive")

    @property
    def law_version(self) -> str:
        """Alias of :attr:`wall_law` (both wall laws are selectable via the
        law version: ``wall-v2.1-closure`` for the J4 training rollouts and
        gate G2 on J4, ``wall-v2.2`` merge-release otherwise)."""
        return self.wall_law
