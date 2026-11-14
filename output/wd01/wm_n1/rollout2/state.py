"""P3 rollout state and strategy actions (Idea A step contract).

All world tensors are batched ``[B, K]`` where ``K`` (the slot width) is a
FORMAT parameter -- 128 for the frozen v2 format, 192 for the v3 format of
amendment 5 -- and is carried by the tensors themselves
(:attr:`WorldState.slots`); no rollout code hardcodes it.  ``DT = 0.5 s``.
``lane`` is 0/1/2 (main_left/main_right/accel) or -1 for an empty
slot.  Positions are downstream metres.

Tensors (frozen):

=============  =========  ============  ====================================
name           dtype      shape         meaning
=============  =========  ============  ====================================
s_m            float32    [B,K]         downstream position
v_mps          float32    [B,K]         speed
a_mps2         float32    [B,K]         last-step acceleration
lane           int64      [B,K]         0/1/2 or -1
length_m       float32    [B,K]         bumper-to-bumper length
present        bool       [B,K]         slot occupied
is_cav         bool       [B,K]         CAV flag (never an HDV-net input)
wall_group     int64      [B,K]         wall id, -1 = not a member
slot_id        int64      [B,K]         vehicle identity (-1 empty)
=============  =========  ============  ====================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class WorldState:
    """One batched world snapshot at a rollout step boundary."""

    s_m: Tensor
    v_mps: Tensor
    a_mps2: Tensor
    lane: Tensor
    length_m: Tensor
    present: Tensor
    is_cav: Tensor
    wall_group: Tensor
    slot_id: Tensor
    # INC1 (WM-N1-INC1): a fixed-obstacle slot is pinned at constant ``s``,
    # ``v = 0``, ``a = 0`` by the step/validity layer and never moved by the
    # HDV net; neighbours still see it as a normal stopped leader.  ``None``
    # (default) means no obstacle in the world.
    obstacle: Optional[Tensor] = None

    @property
    def batch(self) -> int:
        return int(self.s_m.shape[0])

    @property
    def slots(self) -> int:
        return int(self.s_m.shape[1])


@dataclass
class StrategyActions:
    """The strategy assignment for this t0, held for the whole rollout.

    ``action_code``: ``int64 [B, K]`` per-slot CAV action
    (:data:`ctrl.config.ACTION_CODES`); ignored for non-CAV slots.
    ``wall_speed_mps``: ``float32 [B, K]`` wall target speed for wall slots.
    A strategy name is NEVER passed to the HDV network.
    """

    action_code: Tensor
    wall_speed_mps: Optional[Tensor] = None
    wall_closure: Optional[Tensor] = None


@dataclass
class Rollout2Config:
    """P3 engine configuration.

    ``params`` is the frozen ``partA_v2g/params_nominal.json`` mapping
    (read-only; loaded from the partA_v2g file when ``None``).
    ``integrator`` selects the position rule (INTERFACES_V2 Mismatch 5):
    ``"d1"`` trapezoidal (Idea A default) or ``"v2g"`` ballistic
    (``s + v_new*dt``), the like-for-like G1 mode in which the validity
    layer also skips the gap repair (partA_v2g never repairs).
    """

    params: Optional[dict] = None
    integrator: str = "d1"                 # "d1" | "v2g"
    residual_enabled: bool = False
    arrivals_enabled: bool = False
    lane_changes_enabled: bool = True
    enforce_lane_legality: bool = True     # d1 only; auto-off in v2g mode
    max_insert_per_entrance: int = 3
    t0_s: float = 0.0
    state: Optional["Rollout2State"] = None
    # INC1 ablation switch: the P6 event tokens are untrained on the incident
    # distribution, so the headline run keeps them OFF.  ``True`` feeds them
    # as an explicitly labelled ablation only.
    event_tokens: bool = False
    # Physical-coordinate road exits (default OFF = legacy behaviour).  The
    # rollout's ``s_m`` is the model ``s_down`` coordinate (``geometry.s_down_of``
    # = physical s - lane offset), while ``params.llen`` is the PHYSICAL
    # downstream lane end.  The legacy test ``s_down > llen`` therefore never
    # fires for the main lanes (offset ~2645 m), so no vehicle ever departs.
    # With this switch the test becomes ``s_down > llen - off`` exactly as the
    # featurizer/recorded ``cav_plan_ch`` defines "on the road" (a vehicle is
    # absent from the physics row once its physical s passes the lane end).
    # OFF keeps the emitted world byte-identical to the historical rollout.
    physical_exits: bool = False
    # SUMO-fidelity mode (default OFF = byte-identical historical rollout).
    # The P3 world ``s_m`` is the model ``s_down`` coordinate
    # (``features.geometry.s_down_of``) while the frozen v2g geometry is in
    # PHYSICAL lane s, so the nominal baseline never reaches the accel-lane
    # merge zone and its segment starts/lane ends are offset by ~2645 m.
    # ``sumo_fidelity=True`` translates the params into the ``s_down`` frame
    # and replaces the IDM block by the ``hdv_mix``-weighted SUMO vType means
    # (``rollout2.baseline.model_frame_params``).  ``sumo_speed_factor``
    # (default None = 1.0) optionally scales every segment limit by the
    # observed/free-speed multiplier.
    sumo_fidelity: bool = False
    sumo_speed_factor: Optional[float] = None
    # Replace the IDM block by the ``hdv_mix``-weighted SUMO vType means when
    # ``sumo_fidelity`` is on (default True = full SUMO match).  Set False for
    # the geometry-only variant (s_down frame, nominal gentle IDM).
    sumo_idm: bool = True
    # WM-S T3 per-driver desired speed: optional per-slot ``v0`` on the world
    # GLOBAL slot axis, overriding the segment limit in the own-lane IDM term.
    # ``None`` (default) = OFF and byte-identical to the frozen rollout.
    desired_speed_mps: Optional[np.ndarray] = None
    # Calibrated SUMO-like mandatory-merge physics (default OFF =
    # byte-identical historical rollout).  Implies the ``s_down``-frame
    # geometry of ``sumo_fidelity``; replaces the strict v2g merge rule with an
    # lc2013-style strategic merge plus cooperative mainline yielding
    # (``rollout2.merge_cal``, parameters fitted on chunk-4 TRAIN only).
    merge_cal: bool = False
    merge_cal_params: Optional[object] = None

    def __post_init__(self) -> None:
        if self.integrator not in ("d1", "v2g"):
            raise ValueError(
                f"integrator must be 'd1' or 'v2g', got {self.integrator!r}")

    @property
    def gap_projection(self) -> bool:
        return self.integrator == "d1"


@dataclass
class Rollout2State:
    """Persistent per-rollout state (one per sample).

    ``held_target_mps``/``held_reason`` carry the last control tick's
    command (a command holds for the two 0.5 s physics steps of its
    tick, partA_v2g/engine.py:568-572).  ``lc_last`` is the per-slot
    last lane-change time (v2g ``_lc_last``; -1e9 = never).  ``anomalies``
    mirrors the v2g counters (e.g. ``accel_end_clamped``).
    """

    batch: int
    slots: int
    ctrl_state: object = None
    held_target_mps: Optional[Tensor] = None   # float64 [B, K], nan = none
    held_reason: Optional[Tensor] = None       # int64 [B, K]
    lc_last: Optional[Tensor] = None           # float64 [B, K]
    next_slot_id: Optional[Tensor] = None      # int64 [B]
    anomalies: Dict[str, int] = field(default_factory=dict)
    collisions_by_step: Dict[int, int] = field(default_factory=dict)
    last_lane_margins: Optional[Dict] = None   # per-world telemetry view
    last_step_collisions: int = 0
    step_i: int = 0
    cmd_t_s: float = -math.inf

    def __post_init__(self) -> None:
        dev = torch.device("cpu")
        if self.held_target_mps is None:
            self.held_target_mps = torch.full(
                (self.batch, self.slots), float("nan"), dtype=torch.float64,
                device=dev)
        if self.held_reason is None:
            self.held_reason = torch.zeros(
                (self.batch, self.slots), dtype=torch.int64, device=dev)
        if self.lc_last is None:
            self.lc_last = torch.full(
                (self.batch, self.slots), -1e9, dtype=torch.float64,
                device=dev)
        if self.next_slot_id is None:
            self.next_slot_id = torch.zeros(
                (self.batch,), dtype=torch.int64, device=dev)


@dataclass
class Rollouts:
    """Rollout trajectories, shape ``[n_samples, n_steps+1, B, K]``."""

    positions: Tensor
    speeds: Tensor
    lanes: Tensor
    masks: Tensor
    accel: Tensor
    arrivals: Tensor                       # [n_samples, n_steps, B, 2]
    insertions: Tensor                     # [n_samples, n_steps, B, 2]
    validity: List = field(default_factory=list)
    cav_commands: Optional[Tensor] = None  # [n_samples, n_steps, B, K]
    collisions_by_step: Optional[List] = None  # per sample {step: pairs}
    anomalies: Optional[List] = None       # per sample {name: count}
    # A6 (contrastfix follow-up): the ORIGINAL t0 channels, captured from the
    # P3 input world BEFORE any controller step / teacher mutation.  The
    # evaluation state token must be built from these exact values -- the
    # trajectory tensors (``accel[0]``, lane/pos/speed snapshots) describe the
    # FUTURE after the first applied step and are not the t0 state.
    # ``None`` (default) keeps direct-construction call sites on the legacy
    # path; the real engine always sets all three.
    t0_kin: Optional[Tensor] = None        # [B,K,7] kinematics_view of t0
    t0_mask: Optional[Tensor] = None       # [B,K] t0 present mask
    t0_counter: Optional[Tensor] = None    # [B,2,30] t0 entrance counters
    # Original self-identified CAV membership and per-boundary identity.
    # Optional for legacy construction; correction scoring requires them.
    t0_is_cav: Optional[Tensor] = None     # bool [B,K]
    t0_slot_id: Optional[Tensor] = None    # int64 [B,K]
    slot_ids: Optional[Tensor] = None      # int64 [S,T+1,B,K]
