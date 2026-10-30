"""WM-S: a differentiable IDM/MOBIL residual next-state model.

The model owns no controller policy.  HDVs use differentiable physics plus a gated
residual; CAVs are copied from a real-controller teacher trajectory (or supplied by a
controller callback).  ``history`` is an explicit interface so driver-history features
can be merged without changing the step contract.

Parameter names of the registered submodules (``residual_net``, ``history_proj``,
``k_embedding``, ``controller_embed``, ``command_proj``, ``temporal``,
``gate_increments``, ``driver_encoder``, ``lane_change_head``) are the checkpoint
format and must not change; they are also created in this order.
"""
from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from wms.core import fastpath, speed_flags
from wms.physics.accel import apply_emergency_envelope, baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.lane_conflict import resolve_lane_conflicts
from wms.physics.mobil import (MASKED_LOGIT, merge_end_blocked, merge_impatience_level,
                               mobil_merge_logits, update_merge_wait)
from wms.physics.neighbours import lane_neighbour, lane_of, lead_neighbour
from wms.physics.projection import FeasibilityProjection
from wms.physics.state import KIN_F, state_kin, state_present

from . import cav as cav_motion
from .controllers import DIRECT_ID
from .driver_encoder import DRIVER_ENCODER_HIDDEN, PARAMETER_NAMES, DriverEncoder
from .driver_update import EmaDriverUpdater
from .entries import section_ids
from .gate import monotone_gate, saturating_gate
from .history import (LONG_HISTORY_F, history_spec, history_step_features,
                      initial_history, initial_lane_age, push_history)
from .lane_hazard import (HAZARD_F, HAZARD_F_V2, HazardLaneModel, blocked_indicator,
                          hazard_feasible, hazard_lane_logits, initial_lane_blocked,
                          sample_lane_choice, update_blocked)
from .lane_head import HistoryLaneHead, lane_context
from . import lane_v5 as lv5
from .merge_negotiation import (MergeNegotiationNet, pair_features,
                              scatter_follower_accel)
from .merge_core import (MergeCoreNet, core_features, load_init,
                         secure_gap_margin, secure_gap_sides, update_core_alpha)
from .follower_courtesy import (INTENT_SOURCES, ZONE_STATE_KEY, FollowerCourtesyHead,
                                follower_courtesy_pairs, pivot_clearance_intent,
                                initial_zone_time, update_zone_time)
from .merge_frailty import MergeFrailtyNet, update_frailty_state
from . import merge_pivot as pivot
from .merge_threshold import MergeThresholdNet, update_threshold_state
from .mainline_residual import (MainlineInteractionResidual,
                                mainline_interaction_features, mainline_lane_mask)
from .residual_bound import bounded_accel_correction, bounded_value
from .residual_net import HdvConfig, HdvResidualNet, courtesy_features
from .spec import ModelSpec
from .temporal_slow import inverse_softplus, slow_residual_step, softplus_positive

#: Output width of the temporal MLP: one acceleration delta plus three lane logits.
_TEMPORAL_OUT = 4
#: SECUREGAP per-step tensors carried into the rollout result and the merge-core loss.
_MERGE_CORE_LOG_KEYS = (
    "merge_core_gap_l", "merge_core_gap_f", "merge_core_v_l", "merge_core_v_f",
    "merge_core_a_l", "merge_core_a_f", "merge_core_has_l", "merge_core_has_f",
    "merge_core_intent", "merge_core_alpha_log_location", "merge_core_alpha_scale",
    "merge_core_margin", "merge_core_margin_l", "merge_core_margin_f",
    "merge_core_residual", "merge_core_probability", "merge_core_yield_accel",
    "merge_core_follower_index", "merge_core_valid_pair", "merge_core_s0",
    "merge_core_T", "merge_core_b", "merge_core_tau_a", "merge_core_temperature",
    "merge_core_require_both")
_MERGE_PIVOT_LOG_KEYS = tuple("merge_pivot_" + name for name in (
    "c", "eligible", "physical_eligible", "reservation_veto", "cross", "committed",
    "cdf", "event_mass", "st",
    "s_f", "T_f", "b_f", "s_l", "T_l", "b_l", "tau", "w")) + tuple(
        "merge_pivot_geometry_" + name for name in
        ("gap_f", "gap_l", "v_f", "v_l", "has_f", "has_l"))
_MERGE_PIVOT_STATE_KEYS = ("z", "maximum", "present", "identity", "terminal", "started",
                           "epsilon", "rng_seed", "rng_generation", "decision_step", "history", "logS")


@dataclass

class _GradScale(torch.autograd.Function):
    """Identity in the forward pass; multiplies the incoming gradient by ``factor``."""

    @staticmethod
    def forward(ctx, x: Tensor, factor: float) -> Tensor:  # type: ignore[override]
        ctx.factor = float(factor)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor):  # type: ignore[override]
        return grad * ctx.factor, None


def _scale_grad(x: Tensor, factor: float) -> Tensor:
    """Forward-exact gradient damping (round 5b ``grad_decay_per_step``)."""
    return _GradScale.apply(x, factor)

class _ShareAccumulator:
    """Physics vs learned-residual acceleration magnitude on learned slots (hybrid)."""

    physics: Optional[Tensor] = None
    residual: Optional[Tensor] = None
    count: int = 0

    def add(self, out: dict[str, Tensor]) -> None:
        mag = out["hdv_mask"].to(out["accel_mps2"].dtype)
        physics_step = (out["physics_accel_mps2"].abs() * mag).sum()
        residual_step = (out["accel_delta_mps2"].abs() * mag).sum()
        self.physics = physics_step if self.physics is None else self.physics + physics_step
        self.residual = (residual_step if self.residual is None
                         else self.residual + residual_step)
        self.count += int(mag.sum())


class WmSModel(nn.Module):
    """Physics-informed per-vehicle next-state model.

    ``state`` carries ``veh_kin [B,K,7]`` (or the scalar fields ``s_m``/``v_mps``/
    ``a_mps2``/``lane``), ``present [B,K]`` and optionally ``is_cav [B,K]``.
    ``history`` is ``[B,K,H,F_history]`` with an optional ``history_mask``.

    ``apply_direct_commands`` (class default ``True``) controls whether a ``direct``
    action-timeline entry drives a CAV closed loop from its issued acceleration and
    target speed; the trainer sets it ``False`` for teacher-driven CAVs.
    """

    apply_direct_commands: bool = True
    #: LANEFIX runtime switch (``loss.lane_hazard_decouple``; set by the trainer, never part
    #: of a checkpoint): the hazard's learned delta enters the rollout lane logits detached,
    #: so the rollout lane CE / flip-rate guard do not train the hazard (forward unchanged).
    lane_hazard_detach: bool = False
    #: V8WM runtime switch (``loss.merge_timing``; set by the unroll runner, never part of a
    #: checkpoint): with ``lane_model='hazard_v5'`` every step also returns the row's own head
    #: logit WITH gradient (``lane_v5_head_logit``, recomputed outside the decouple detach) and
    #: the acting-row mask (``lane_v5_acts``: learned, physics-feasible, outside the forced
    #: window unless ``physics.forced_window_mode='hazard'``).  Forward values and decisions
    #: are unchanged.
    emit_merge_hazard: bool = False
    #: R5MERGE runtime switch (``loss.post_merge_residual``; set by the unroll runner,
    #: never part of a checkpoint): also log the physics-core IDM acceleration per step
    #: so the one-sided post-merge residual penalty can condition on ``IDM > 0``.
    #: Default OFF: no extra tensor is recorded and the rollout is unchanged.
    emit_physics_accel: bool = False

    def __init__(self, spec: Optional[ModelSpec] = None, *,
                 physics: Optional[PhysicsConfig] = None, hdv: Optional[HdvConfig] = None,
                 physics_step: Optional[Callable] = None) -> None:
        super().__init__()
        spec = spec or ModelSpec()
        self.spec = spec
        self.physics = physics or PhysicsConfig()
        if (getattr(self.physics, "secure_gap_veto", False)
                and spec.merge_decision != "first_passage"):
            raise ValueError("secure_gap_veto requires merge_decision='first_passage'")
        if physics_step is not None and not callable(physics_step):
            raise TypeError("physics_step must be a callable batched Torch module")
        self.physics_step = physics_step
        self._init_bounds(spec)
        self.rollout_mode = spec.rollout_mode
        self.lane_head_mode = spec.lane_head
        self.lane_model = getattr(spec, "lane_model", "residual")
        self.lane_conflict_resolve = getattr(spec, "lane_conflict_resolve", False)
        # LANEFIX (hazard mode only; defaults = WMLANE A behaviour).  ``lane_decision`` and
        # ``lane_logit_offsets`` are plain attributes an evaluator may override.
        self.lane_decision = getattr(spec, "lane_decision", "argmax")
        self.lane_logit_offsets = tuple(getattr(spec, "lane_logit_offsets", (0.0, 0.0, 0.0)))
        self.lane_hazard_features = getattr(spec, "lane_hazard_features", "v1")
        self.merge_hazard_zone = getattr(spec, "merge_hazard_zone", "forced_late")
        # LANEV5 (lane_model='hazard_v5' only): gated-decision gates and the configured
        # per-head TRAIN-val AP (negative = use the AP stored in the lane network's buffer).
        self.lane_v5_gates = tuple(getattr(spec, "lane_v5_gates", lv5.DEFAULT_GATES))
        self.lane_v5_head_ap = tuple(getattr(spec, "lane_v5_head_ap", (lv5.UNKNOWN_AP,) * 3))
        # V8LANE (default 'default' = the lane_decision rule; plain attribute an evaluator may
        # override): mainline overtake / return decision rule (lv5.MAINLINE_MODES).
        self.lane_v5_mainline = getattr(spec, "lane_v5_mainline", "default")
        # MERGECAL (None = off): per-position merge-head offsets ``(edges_m, offsets)``, set by
        # ``load_checkpoint`` from the payload key ``lv5.MERGE_CALIB_KEY`` (plain attribute).
        self.lane_v5_merge_calib = None
        # LANERATECAL (None = off): closed-loop per-head rate calibration read by
        # ``load_checkpoint`` from ``lane_calib.json`` next to the checkpoint (plain attribute;
        # the additive offsets are already folded into ``lane_logit_offsets``).
        self.lane_rate_calib = None
        self.driver_update = getattr(spec, "driver_update", "off")
        self.history_features = int(spec.history_features)
        self.history_len = int(spec.history_len)
        self.history_s = spec.history_s
        self.history_stride_s = float(spec.history_stride_s)
        self.long_history = spec.long_history
        _, _, self.driver_history_len = history_spec(
            spec.history_s, spec.history_stride_s, self.physics.dt_s)
        # Fused DriverEncoder embedding width (pooled trunk + interaction projection
        # + last observed feature); also the residual-net conditioning width.
        self.driver_embed_dim = 2 * DRIVER_ENCODER_HIDDEN + self.history_features
        self.hdv = hdv or HdvConfig()
        if (spec.rollout_mode == "hybrid" or self.long_history) \
                and int(self.hdv.conditioning_dim) == 0:
            self.hdv = replace(self.hdv, conditioning_dim=self.driver_embed_dim)
        if bool(getattr(spec, "courtesy_inputs", False)) and not self.hdv.courtesy_inputs:
            # V10COURTESY: the checkpoint/config switch exists independently of the
            # explicit HdvConfig override so loading a config enables the channels.
            self.hdv = replace(self.hdv, courtesy_inputs=True)
        if bool(getattr(spec, "residual_head_rescale", False)) \
                and not self.hdv.residual_head_rescale:
            # R5MERGE: the output-divisor switch is config-driven too; it is exactly
            # identity at load and adds one scalar parameter.
            self.hdv = replace(self.hdv, residual_head_rescale=True)
        self.black_box = bool(spec.black_box)
        self.n_controllers = int(spec.n_controllers)
        self.controller_embed_dim = int(spec.controller_embed_dim)
        self.checkpoint_residual = bool(spec.checkpoint_residual)
        self.gate_kind = spec.gate_kind
        self.gate_g_max_init = float(spec.gate_g_max_init)
        self.gate_tau_steps_init = float(spec.gate_tau_steps_init)
        self.lane_gate_kind = spec.lane_gate_kind
        self.checkpoint_step = bool(spec.checkpoint_step)
        self.grad_decay_per_step = float(getattr(spec, "grad_decay_per_step", 1.0))
        # Inference-time diagnostic switches (round 5, default OFF; set by the scorer,
        # never part of a checkpoint): population-default driver settings.
        self.diag_encoder_population = False
        # Runtime memory switch (C4 imagination, never part of a checkpoint): with gradients
        # enabled, recompute the pure per-step ``forward`` (encoders, physics, residual net)
        # during backward instead of storing its activations.  Unlike ``checkpoint_step`` this
        # never replays the CAV controller callback (it runs outside ``forward``), so it is
        # safe with stateful controllers and leaves values and gradients unchanged.
        self.checkpoint_forward = False
        # Braking diagnostic switches (V10BRKPROBE, inference only, never part of a
        # checkpoint; all default OFF and bit-identical when off):
        #   diag_zero_temporal      zero the temporal residual output (history_proj +
        #                           k_embedding + temporal) while keeping the attention head
        #                           and driver adaptation.
        #   diag_residual_bound_mps2 hard bound on the COMBINED learned residual (attention +
        #                           temporal) after the configured bounds; None = off.
        #   diag_temporal_tau_s     causal first-order low-pass of the temporal acceleration
        #                           output with this time constant in seconds; None = off.
        self.diag_zero_temporal = False
        # R13F (spec, part of the checkpoint; default False = round 12): the temporal acceleration
        # emission is an exact 0 in training and rollout -- the same tensor diag_zero_temporal zeroes.
        self.temporal_accel_off = bool(getattr(spec, "temporal_accel_off", False))
        # R12C (spec, part of the checkpoint; default False = round 12): physics-only core
        # calibration.  residual_accel_off replaces the residual attention/accel emission
        # (base["accel_delta_mps2"]) by an exact 0 (the BRAKE_DIAG 'physics only' arm);
        # follower_courtesy_off makes the courtesy offset an exact 0 and skips its addition.
        # The lane read-outs of both nets are untouched; both are bit-identical when off.
        self.residual_accel_off = bool(getattr(spec, "residual_accel_off", False))
        self.follower_courtesy_off = bool(getattr(spec, "follower_courtesy_off", False))
        self.diag_residual_bound_mps2 = None
        self.diag_temporal_tau_s = None
        self._diag_temporal_lp = None
        # SLOWRES (default 'direct' = bit-identical): the learned low-pass temporal state.
        # Created as plain attributes here; the parameters themselves are appended after
        # the temporal MLP in ``_build_modules`` only in slow mode.
        self.temporal_residual_mode = str(getattr(spec, "temporal_residual_mode", "direct"))
        self.temporal_tau_init_s = float(getattr(spec, "temporal_tau_init_s", 0.05))
        self.temporal_tau_conditioned = bool(getattr(spec, "temporal_tau_conditioned", False))
        self.temporal_slew_init_mps2 = getattr(spec, "temporal_slew_init_mps2", None)
        # Runtime execution-cap switch (EXECCAP_NOTES.md, never part of a checkpoint): "off"
        # (default, bit-identical) or "idm" -- every COMMAND-driven CAV realises at most the
        # SUMO-consistent ceiling min(own physics-core IDM accel, (v0-v)/dt, accel param).
        self.cav_exec_cap = "off"
        self._build_modules(spec)
        # R13M: per-lane constant temporal emission (created last: default key set / RNG unchanged).
        self.temporal_accel_lane_bias = bool(getattr(spec, "temporal_accel_lane_bias", False))
        self.temporal_lane_bias = (nn.Parameter(torch.zeros(3)) if self.temporal_accel_lane_bias
                                   else None)
        # R13S: S-FACTORIAL state-conditional mean tables (parameters) + fitted row counts (buffers, backoff).
        self.temporal_accel_state_table = bool(getattr(spec, "temporal_accel_state_table", False))
        self.temporal_state_mean = None
        if self.temporal_accel_state_table:
            from . import temporal_state_table as tst
            self.temporal_state_mean = nn.ParameterList(
                [nn.Parameter(torch.zeros(tst.level_shape(lv))) for lv in range(len(tst.LEVELS))])
            for lv in range(len(tst.LEVELS)):
                self.register_buffer(f"temporal_state_n{lv}", torch.zeros(tst.level_shape(lv)))
        # Diagnostic (never part of a checkpoint, default off = no extra output): emit the pre-gate
        # temporal acceleration as ``temporal_accel_pregate_mps2`` (tools/r13m_lane_bias_init.py).
        self.diag_emit_temporal = False
        self.mainline_merge_yield_learnable = False
        self.merge_yield = None
        if spec.mainline_merge_yield_learnable:
            self.enable_mainline_merge_yield_learnable()
        # R16 (S-RAMPSPEED): learned ramp-HDV speed adaptation to the target lane (default off:
        # no module, tensors, RNG draws or forward operations).
        self.ramp_speed_adapt_learnable = False
        self.ramp_speed_adapt = None
        if getattr(spec, "ramp_speed_adapt_learnable", False):
            self.enable_ramp_speed_adapt_learnable()
        # S-RAMPFIX (default OFF): learned mainline-HDV interaction residual.  No module,
        # tensors, RNG draws or forward operations when disabled.
        self.mainline_residual = None
        if bool(getattr(spec, "mainline_interaction_residual", False)):
            self.enable_mainline_residual(int(getattr(spec, "mainline_residual_hidden", 64)))
        if self.physics.lane_rule == "learned_discretionary":
            self._zero_lane_outputs()
        self.merge_response = None
        if getattr(spec, "merge_response", False):
            self.enable_merge_response()

    def enable_merge_response(self) -> None:
        """Append the opt-in zero-output head after all legacy initialization."""
        if self.merge_response is not None:
            raise ValueError("merge response already enabled")
        if self.physics_step is not None:
            raise ValueError("merge response requires the in-module emergency envelope")
        from .merge_response import MergeResponseField
        self.merge_response = MergeResponseField().to(next(self.parameters()))
        self.spec = replace(self.spec, merge_response=True)

    def _init_bounds(self, spec: ModelSpec) -> None:
        """Residual and lane-logit bounds (plain attributes; the trainer may reset them)."""
        self.residual_cap_mps2 = (None if spec.residual_cap_mps2 is None
                                  else float(spec.residual_cap_mps2))
        self.residual_cap_neg_mps2 = (None if spec.residual_cap_neg_mps2 is None
                                      else float(spec.residual_cap_neg_mps2))
        self.lane_logit_cap = (None if spec.lane_logit_cap is None
                               else float(spec.lane_logit_cap))
        self.residual_soft_bound_mps2 = spec.residual_soft_bound_mps2
        self.residual_soft_bound_neg_mps2 = spec.residual_soft_bound_neg_mps2

    def _build_modules(self, spec: ModelSpec) -> None:
        history_hidden = spec.history_hidden
        self.residual_net = HdvResidualNet(self.hdv)
        self.history_proj = nn.Sequential(
            nn.Linear(max(1, self.history_features * self.history_len) + self.history_len,
                      history_hidden), nn.GELU(),
            nn.Linear(history_hidden, history_hidden))
        self.k_embedding = nn.Embedding(self.physics.max_gap_steps + 1, spec.k_hidden)
        temporal_in = history_hidden + spec.k_hidden
        self.controller_embed = None
        self.command_proj = None
        if self.black_box:
            self.controller_embed = nn.Embedding(self.n_controllers,
                                                 self.controller_embed_dim)
            self.command_proj = nn.Linear(2, self.controller_embed_dim, bias=False)
            temporal_in += self.controller_embed_dim
        self.temporal = nn.Sequential(
            nn.Linear(temporal_in, history_hidden), nn.GELU(),
            nn.Linear(history_hidden, _TEMPORAL_OUT))
        # SLOWRES: the learned positive time constant and optional slew-rate scale.  Created
        # only in slow mode, after the temporal MLP, so the default model keeps its exact
        # parameter set and RNG stream.
        self.temporal_tau_raw = None
        self.temporal_tau_head = None
        self.temporal_slew_raw = None
        if self.temporal_residual_mode == "slow":
            tau_raw = inverse_softplus(max(float(self.temporal_tau_init_s), 1.0e-4))
            self.temporal_tau_raw = nn.Parameter(torch.tensor(tau_raw, dtype=torch.float32))
            if self.temporal_tau_conditioned:
                hidden = max(8, min(32, int(history_hidden)))
                self.temporal_tau_head = nn.Sequential(
                    nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
                last = self.temporal_tau_head[-1]
                with torch.no_grad():
                    last.weight.zero_()
                    last.bias.zero_()
            if self.temporal_slew_init_mps2 is not None:
                slew_raw = inverse_softplus(float(self.temporal_slew_init_mps2))
                self.temporal_slew_raw = nn.Parameter(
                    torch.tensor(slew_raw, dtype=torch.float32))
        # Exact zero at k=0 by construction; positive increments make the gate
        # monotone and the small initial value keeps early steps physics-dominated.
        if spec.gate_kind == "cumsum":
            self.gate_increments = nn.Parameter(
                torch.full((self.physics.max_gap_steps,), -5.0))
        else:
            g_max_logit = torch.logit(torch.tensor(spec.gate_g_max_init))
            tau_raw = torch.log(torch.expm1(torch.tensor(spec.gate_tau_steps_init)))
            self.gate_g_max_logit = nn.Parameter(g_max_logit)
            self.gate_tau_raw = nn.Parameter(tau_raw)
        # Separate saturating lane gate (round 5): created after the acceleration
        # gate so checkpoint key order of every earlier model is unchanged.
        if spec.lane_gate_kind == "saturating":
            lane_g_max_logit = torch.logit(torch.tensor(spec.lane_gate_g_max_init))
            lane_tau_raw = torch.log(torch.expm1(torch.tensor(spec.lane_gate_tau_steps_init)))
            self.lane_gate_g_max_logit = nn.Parameter(lane_g_max_logit)
            self.lane_gate_tau_raw = nn.Parameter(lane_tau_raw)
        self.driver_encoder = DriverEncoder(self.history_features,
                                            long_history=self.long_history,
                                            driver_param_mode=spec.driver_param_mode,
                                            physics=self.physics)
        self.lane_change_head = (HistoryLaneHead(self.driver_embed_dim)
                                 if spec.lane_head == "history" else None)
        # WMLANE switches (default OFF): created last so the key order and the RNG
        # stream of every earlier model are unchanged.
        n_hazard = (HAZARD_F_V2 if getattr(spec, "lane_hazard_features", "v1") == "v2"
                    else HAZARD_F)
        self.lane_hazard = (HazardLaneModel(self.driver_embed_dim, spec.lane_hazard_hidden,
                                            spec.lane_hazard_cap, n_features=n_hazard)
                            if getattr(spec, "lane_model", "residual") == "hazard" else None)
        self.driver_updater = (EmaDriverUpdater(self.driver_embed_dim, spec.driver_update_rate,
                                               driver_ema_update=spec.driver_ema_update)
                               if getattr(spec, "driver_update", "off") == "ema" else None)
        # LANEV5 (default OFF): created last; absent (no parameters, no RNG draw) otherwise.
        if getattr(spec, "lane_model", "residual") == "hazard_v5":
            self.lane_v5 = lv5.LaneV5Net(
                spec.lane_hazard_hidden, spec.lane_hazard_cap,
                extra_features=bool(getattr(spec, "lane_v5_extra_features", False)))
        elif lv5.forced_window_hazard(self.physics):
            raise ValueError("physics.forced_window_mode='hazard' requires "
                             "model.lane_model='hazard_v5'")
        # V10NEGOT (default OFF): learned ramp/follower merge negotiation, created last so
        # old checkpoints keep their key order and load unchanged.  Zero-initialised.
        self.merge_negotiation = (MergeNegotiationNet()
                                  if bool(getattr(spec, "merge_negotiation", False)) else None)
        # R5THR (default 'hazard' = the per-step Bernoulli merge draw): a persistent learned
        # gap-acceptance threshold replaces the merge decision.  Created only when enabled, so
        # the default checkpoint layout is unchanged (the module comes after every existing one).
        self.merge_decision = getattr(spec, "merge_decision", "hazard")
        self.merge_threshold = None
        self.merge_frailty = None
        self.merge_core = None
        self.merge_pivot = None
        if self.merge_decision == "threshold":
            input_dim = (self.driver_embed_dim
                         if self.rollout_mode in ("driver_params", "hybrid")
                         or self.long_history else 0)
            self.merge_threshold = MergeThresholdNet(input_dim=input_dim)
        elif self.merge_decision == "frailty":
            # R5FR: ONE persistent per-driver latent (location + positive scale) on top of
            # the per-step merge hazard.  Created last for checkpoint-key compatibility.
            input_dim = (self.driver_embed_dim
                         if self.rollout_mode in ("driver_params", "hybrid")
                         or self.long_history else 0)
            self.merge_frailty = MergeFrailtyNet(input_dim=input_dim)
        elif self.merge_decision in ("secure_gap", "first_passage"):
            # SECUREGAP: learned LC2013-like secure-gap core.  Created last for key order;
            # parameters start at the values fitted on recorded TRAIN-val data.
            input_dim = (self.driver_embed_dim
                         if self.rollout_mode in ("driver_params", "hybrid")
                         or self.long_history else 0)
            self.merge_core = MergeCoreNet(input_dim=input_dim, init=load_init())
            if self.merge_decision == "first_passage":
                # Retain round-8 tensors for exact checkpoint transfer, but never execute
                # their anticipation/residual/yield/latent heads in the A-only pivot.
                self.merge_core.requires_grad_(False)
                self.merge_pivot = pivot.FirstPassageNet(
                    hybrid=spec.merge_passage_law == "hybrid", w_init=spec.merge_w_init)
        self.merge_st_width = float(spec.merge_st_width)
        # R10B (default OFF): learned pre-merge follower courtesy head.  Created after EVERY other
        # module so the default model keeps its exact key order, parameter set and RNG stream; the
        # zero last layer makes a freshly enabled head an exact no-op.
        self.follower_courtesy = None
        self.follower_courtesy_rollout_grad = bool(getattr(spec, "follower_courtesy_rollout_grad",
                                                           False))
        # R13A (defaults = round 12): ``state_grad`` keeps the pair features' continuous state
        # derivative (repairs the input-state adjoint promised by ``follower_courtesy_offset``);
        # ``intent`` selects the ramp car's intent input (lane_v5 = round 12, pivot, none).
        self.follower_courtesy_state_grad = bool(getattr(spec, "follower_courtesy_state_grad",
                                                         False))
        self.follower_courtesy_intent = str(getattr(spec, "follower_courtesy_intent", "lane_v5"))
        if self.follower_courtesy_intent not in INTENT_SOURCES:
            raise ValueError(f"follower_courtesy_intent must be one of {INTENT_SOURCES}")
        if bool(getattr(spec, "follower_courtesy", False)):
            if self.follower_courtesy_intent == "pivot" and self.merge_pivot is None:
                raise ValueError("follower_courtesy_intent='pivot' needs "
                                 "merge_decision='first_passage'")
            embed = (self.driver_embed_dim
                     if bool(getattr(spec, "follower_courtesy_embedding", False)) else 0)
            # R11: version 2 appends the ramp car's acceleration (t, t-1) and its merge-zone time.
            self.follower_courtesy = FollowerCourtesyHead(
                hidden=int(getattr(spec, "follower_courtesy_hidden", 32)), embed_dim=embed,
                version=int(getattr(spec, "follower_courtesy_version", 1)))

    def enable_mainline_merge_yield_learnable(self) -> None:
        """Enable R13Y on a loaded legacy core; call before building an optimizer.

        Native configurations use ModelSpec; this hook supports offline old-checkpoint
        comparisons. Creation is deterministic and leaves the legacy RNG stream alone.
        """
        if (self.black_box or not self.physics.lane_learnable
                or self.physics_step is not None
                or not (self.rollout_mode in ("driver_params", "hybrid") or self.long_history)):
            raise ValueError("learnable mainline yield needs a learned-driver lane_learnable core")
        if self.merge_yield is None:
            from .merge_yield import LearnedMergeYield
            like = next(self.parameters())
            self.merge_yield = LearnedMergeYield().to(device=like.device, dtype=like.dtype)
        self.mainline_merge_yield_learnable = True

    def enable_ramp_speed_adapt_learnable(self, init=None) -> None:
        """Enable R16 on a (loaded legacy) core; call before building an optimizer.

        ``init`` optionally overrides the TRAIN-fitted initialization (offline comparisons).
        Creation is deterministic and leaves the legacy RNG stream alone.
        """
        if (self.black_box or self.physics_step is not None
                or not (self.rollout_mode in ("driver_params", "hybrid") or self.long_history)):
            raise ValueError("ramp speed adaptation needs a learned-driver structured core")
        if self.ramp_speed_adapt is None or init is not None:
            from .ramp_speed_adapt import LearnedRampSpeedAdapt
            like = next(self.parameters())
            self.ramp_speed_adapt = LearnedRampSpeedAdapt(init).to(device=like.device, dtype=like.dtype)
        self.ramp_speed_adapt_learnable = True

    def enable_mainline_residual(self, hidden: int = 64) -> None:
        """Enable the S-RAMPFIX mainline interaction residual on a core; call before
        building an optimizer.

        Creation is deterministic (zero-initialised read-out, exact no-op until
        trained) and leaves the legacy RNG stream alone, so a loaded WM1 core can
        be extended offline without touching any existing weight.
        """
        if self.black_box or self.physics_step is not None:
            raise ValueError("mainline interaction residual needs the structured in-module "
                             "physics core")
        if self.mainline_residual is None:
            like = next(self.parameters())
            self.mainline_residual = MainlineInteractionResidual(
                hidden=hidden).to(device=like.device, dtype=like.dtype)


    @property
    def _tracks_courtesy_zone(self) -> bool:
        """R11: the rollout / replay carries the ramp cars' merge-zone time (courtesy v2 input)."""
        head = getattr(self, "follower_courtesy", None)
        return head is not None and int(getattr(head, "version", 1)) >= 2

    def initial_courtesy_zone(self, present: Tensor, kin: Tensor,
                              driver_history: Optional[Tensor],
                              driver_history_mask: Optional[Tensor]) -> Tensor:
        """R11: merge-zone time at t0 (incl. t0) from the observed pre-t0 long history."""
        return initial_zone_time(present, kin, self.physics,
                                 driver_history if self.long_history else None,
                                 driver_history_mask if self.long_history else None,
                                 self._history_stride_steps())

    # ------------------------------------------------------------ R10B courtesy
    def follower_courtesy_offset(self, inputs: Tensor, *, live: bool = True) -> Tensor:
        """Courtesy head output for prepared inputs ``[..., n_in]``.

        ``live=False`` evaluates the same function with DETACHED parameters: no rollout loss reaches
        the head's weights.  Gradients flow through the inputs only when they carry a graph, i.e. with
        ``follower_courtesy_state_grad`` (R13A); in round 10b-12 the pair features were detached, so
        the courtesy offset had NO state derivative despite this design intent.
        """
        head = self.follower_courtesy
        if head is None:
            raise RuntimeError("follower courtesy head is not enabled (model.follower_courtesy)")
        if live:
            return head(inputs)
        params = {name: value.detach() for name, value in head.named_parameters()}
        return functional_call(head, params, (inputs,))

    def follower_courtesy_total(self, pre_courtesy: Tensor, offset: Tensor) -> Tensor:
        """The final acceleration from the pre-courtesy acceleration and the courtesy offset
        (exactly the composition used inside :meth:`forward`, including the plant clamp)."""
        accel = pre_courtesy + offset
        if self.rollout_mode != "driver_params":
            accel = accel.clamp(self.physics.accel_min_mps2, self.physics.accel_max_mps2)
        return accel

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Round-8 -> pivot: duplicate loaded positive gap scalars, match latent variance.

        Native pivot checkpoints contain all seven keys and are loaded verbatim. Legacy
        tensors remain present and unchanged; the old anticipation/temperature are inactive.
        """
        if self.merge_pivot is not None and not any(
                key.startswith("merge_pivot.") for key in state_dict):
            state_dict = state_dict.copy()
            for side in ("f", "l"):
                for new, old in (("s", "s0"), ("T", "t"), ("b", "b")):
                    raw = state_dict.get(f"merge_core.{old}_raw")
                    if raw is not None:
                        value = torch.nn.functional.softplus(raw) + 1.0e-6
                        state_dict[f"merge_pivot.{new}_{side}_raw"] = (
                            value + torch.log(-torch.expm1(-value)))
            raw = state_dict.get("merge_core.alpha_log_sigma_raw")
            if raw is not None:
                value = raw.clamp(-4.0, 2.0).exp() * math.sqrt(3.0) / math.pi
                state_dict["merge_pivot.tau_raw"] = value + torch.log(-torch.expm1(-value))
        if (self.merge_pivot is not None and self.merge_pivot.hybrid
                and "merge_pivot.w_raw" not in state_dict):
            state_dict = state_dict.copy()
            state_dict["merge_pivot.w_raw"] = self.merge_pivot.w_raw.detach().clone()
        if (self.merge_pivot is not None and not self.merge_pivot.hybrid
                and "merge_pivot.w_raw" in state_dict):
            # Explicit strict comparison of a hybrid fit reuses its seven
            # boundary/heterogeneity tensors; w has no role in the strict law.
            state_dict = state_dict.copy()
            state_dict.pop("merge_pivot.w_raw")
        if self.merge_yield is not None and not any(
                key.startswith("merge_yield.") for key in state_dict):
            state_dict = state_dict.copy()
            state_dict.update({f"merge_yield.{key}": value.detach().clone()
                               for key, value in self.merge_yield.state_dict().items()})
        if self.ramp_speed_adapt is not None and not any(
                key.startswith("ramp_speed_adapt.") for key in state_dict):
            # R16: a pre-R16 core starts from the TRAIN-fitted initialization.
            state_dict = state_dict.copy()
            state_dict.update({f"ramp_speed_adapt.{key}": value.detach().clone()
                               for key, value in self.ramp_speed_adapt.state_dict().items()})
        if self.mainline_residual is not None and not any(
                key.startswith("mainline_residual.") for key in state_dict):
            # S-RAMPFIX: a pre-RAMPFIX core (WM1) starts from the zero-initialised head,
            # an exact no-op until trained.
            state_dict = state_dict.copy()
            state_dict.update({f"mainline_residual.{key}": value.detach().clone()
                               for key, value in self.mainline_residual.state_dict().items()})
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ------------------------------------------------------------------ physics
    def _zero_lane_outputs(self) -> None:
        """learned_discretionary: a freshly built model starts with ZERO learned lane deltas,
        so discretionary changes (prior below 'stay') begin at none and are learned from data
        (random init otherwise flips lanes ~60x the recorded rate).  Loading a checkpoint
        (init_from) overwrites these weights as usual."""
        with torch.no_grad():
            last = self.residual_net.lane[-1] if isinstance(self.residual_net.lane, nn.Sequential)                 else self.residual_net.lane
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()
            self.temporal[-1].weight[1:].zero_()
            self.temporal[-1].bias[1:].zero_()

    def load_a11_state_dict(self, state_dict: dict[str, Tensor], *, strict: bool = False):
        """Initialise the reusable residual core from an A11 state dict."""
        own = self.residual_net.state_dict()
        copied = {}
        for key, value in state_dict.items():
            name = key.removeprefix("hdv.").removeprefix("residual_net.")
            if name in own and own[name].shape == value.shape:
                copied[name] = value
        result = self.residual_net.load_state_dict(copied, strict=strict)
        return {"loaded": len(copied), "available": len(own),
                "missing": list(result.missing_keys)}

    def reset_diag_state(self) -> None:
        """Clear the causal low-pass state of the braking diagnostic (no-op when off)."""
        self._diag_temporal_lp = None

    def gate(self, k_steps: Tensor | int) -> Tensor:
        """Learned monotone gate with exactly ``g(0) == 0``."""
        if self.gate_kind == "saturating":
            return saturating_gate(self.gate_g_max_logit, self.gate_tau_raw, k_steps,
                                   self.physics.max_gap_steps)
        return monotone_gate(self.gate_increments, k_steps, self.physics.max_gap_steps)

    def lane_gate(self, k_steps: Tensor | int, shared: Optional[Tensor] = None) -> Tensor:
        """Gate on the learned lane-logit correction (own saturating gate when configured).

        With ``lane_gate_kind == "shared"`` (default) the lane correction uses the
        acceleration gate unchanged —the returned tensor is the provided ``shared``
        value (or a fresh ``self.gate`` evaluation), so default models are bit-identical.
        """
        if self.lane_gate_kind != "saturating":
            return shared if shared is not None else self.gate(k_steps)
        return saturating_gate(self.lane_gate_g_max_logit, self.lane_gate_tau_raw, k_steps,
                               self.physics.max_gap_steps)

    def physics_accel(self, state: dict[str, Tensor],
                      driver_params: Optional[dict[str, Tensor]] = None, *,
                      emergency_envelope: bool = True) -> Tensor:
        """Physics-baseline acceleration ``[B, K]``.

        R13E: the emergency envelope is applied last (after the learned yield cap);
        ``emergency_envelope=False`` returns the un-enveloped composition input.
        """
        kin = state_kin(state)
        present = state_present(state, kin)
        accel = baseline_accel(kin, present, self.physics, driver_params,
                               emergency_envelope=False)
        if self.mainline_merge_yield_learnable:
            accel, _ = self.merge_yield(kin, present, accel, self.physics, driver_params,
                                        is_cav=state.get("is_cav"),
                                        merge_wait_s=state.get("merge_wait_s"))
        if emergency_envelope and self.physics.emergency_braking:
            accel = torch.where(present, apply_emergency_envelope(
                kin, present, accel, self.physics, driver_params), torch.zeros_like(accel))
        return accel

    def mobil_merge_logits(self, state: dict[str, Tensor],
                           driver_params: Optional[dict[str, Tensor]] = None) -> Tensor:
        """MOBIL plus mandatory-merge baseline lane logits ``[B, K, 3]``."""
        kin = state_kin(state)
        if getattr(self.physics, "lane_change_secure_gap", False):
            # R14 D1: HDV mainline changes need learned-parameter gs secure gaps (CAVs exempt).
            return mobil_merge_logits(kin, state_present(state, kin), self.physics, driver_params,
                                      is_cav=state.get("is_cav"))
        return mobil_merge_logits(kin, state_present(state, kin), self.physics, driver_params)

    def _physics_step(self, state: dict[str, Tensor],
                      driver_params: Optional[dict[str, Tensor]] = None, *,
                      emergency_envelope: bool = True, lane_logits: bool = True
                      ) -> tuple[Tensor, Optional[Tensor]]:
        """Baseline ``(accel, lane logits)`` from the injected backend or in-module physics.

        ``emergency_envelope=False`` (R13E forward) omits the emergency envelope so the
        caller can apply it once to the final composed acceleration.
        """
        env = emergency_envelope
        if self.physics_step is None:
            if self.mainline_merge_yield_learnable:
                # Keep the unconstrained baseline for residual/courtesy composition.
                # The virtual-leader cap is applied ONCE after those terms below.
                kin = state_kin(state)
                # S-WMSPEED: ``lane_logits=False`` (caller overwrites them) skips the pure,
                # RNG-free MOBIL pass whose result would be discarded.
                return (baseline_accel(kin, state_present(state, kin), self.physics, driver_params,
                                       emergency_envelope=env),
                        self.mobil_merge_logits(state, driver_params) if lane_logits else None)
            # V8WM3 merge_impatience: the carried lane-end waiting time (None = legacy paths).
            impatience = merge_impatience_level(state.get("merge_wait_s"), self.physics)
            if (self.physics.merge_end_wait or self.physics.mainline_merge_yield) \
                    and not getattr(self.physics, "lane_learnable", False):
                # V6MERGE: one blocked-merger mask per step, shared by accel and lane logits.
                # V10UNBLOCK lane_learnable: the physical envelope replaces this coupling.
                kin = state_kin(state)
                present = state_present(state, kin)
                if impatience is None:
                    blocked = merge_end_blocked(kin, present, self.physics)
                    return (baseline_accel(kin, present, self.physics, driver_params,
                                           merge_blocked=blocked, emergency_envelope=env),
                            mobil_merge_logits(kin, present, self.physics, driver_params,
                                               merge_blocked=blocked))
                blocked = merge_end_blocked(kin, present, self.physics, impatience)
                return (baseline_accel(kin, present, self.physics, driver_params,
                                       merge_blocked=blocked, impatience=impatience,
                                       emergency_envelope=env),
                        mobil_merge_logits(kin, present, self.physics, driver_params,
                                           merge_blocked=blocked, impatience=impatience))
            if impatience is not None:
                kin = state_kin(state)
                present = state_present(state, kin)
                return (baseline_accel(kin, present, self.physics, driver_params,
                                       impatience=impatience, emergency_envelope=env),
                        mobil_merge_logits(kin, present, self.physics, driver_params,
                                           impatience=impatience))
            return (self.physics_accel(state, driver_params, emergency_envelope=env),
                    self.mobil_merge_logits(state, driver_params))
        result = self.physics_step(state)
        if not isinstance(result, Mapping) or "accel_mps2" not in result:
            raise ValueError("physics_step must return mapping with accel_mps2")
        accel = result["accel_mps2"]
        baseline_lane = result.get("lane_logits")
        kin = state_kin(state)
        expected = kin.shape[:2]
        if tuple(accel.shape) != tuple(expected):
            raise ValueError(f"physics accel shape {tuple(accel.shape)} != {expected}")
        if not torch.isfinite(accel).all():
            raise ValueError("physics_step returned non-finite acceleration")
        if baseline_lane is None:
            baseline_lane = self.mobil_merge_logits(state)
        elif tuple(baseline_lane.shape) != (*expected, 3):
            raise ValueError(
                f"physics lane logits shape {tuple(baseline_lane.shape)} != "
                f"{tuple(expected) + (3,)}")
        if not torch.isfinite(baseline_lane).all():
            raise ValueError("physics_step returned non-finite lane logits")
        return accel.to(kin.dtype), baseline_lane.to(kin.dtype)

    # ------------------------------------------------------------- learned parts
    def _temporal_delta(self, history: Optional[Tensor], history_mask: Optional[Tensor],
                        k_steps: Tensor, shape: tuple[int, int], dtype, device,
                        controller_emb: Optional[Tensor] = None) -> Tensor:
        """History/step-gap MLP output ``[B, K, 4]`` = accel delta + 3 lane logits."""
        b, k = shape
        h, f = self.history_len, self.history_features
        if history is None:
            flat = torch.zeros(b, k, max(1, f * h) + h, dtype=dtype, device=device)
        else:
            if history.ndim != 4 or history.shape[:2] != (b, k):
                raise ValueError("history must be [B,K,H,F_history]")
            if history.shape[-1] != f or history.shape[2] != h:
                raise ValueError("history window/feature width does not match model")
            m = torch.ones(history.shape[:-1], dtype=dtype, device=device) \
                if history_mask is None else history_mask.to(dtype)
            flat = torch.cat(((history * m.unsqueeze(-1)).reshape(b, k, h * f), m), -1)
        z = self.history_proj(flat)
        kx = self.k_embedding(k_steps.clamp(0, self.physics.max_gap_steps))
        parts = [z, kx]
        if self.black_box or controller_emb is not None:
            if controller_emb is None:
                controller_emb = torch.zeros(b, k, self.controller_embed_dim,
                                             dtype=dtype, device=device)
            parts.append(controller_emb.to(dtype))
        return self.temporal(torch.cat(parts, dim=-1))

    def _slow_temporal_accel(self, target: Tensor, state: dict[str, Tensor],
                             kin: Tensor, present: Tensor,
                             physics_accel: Tensor) -> Tensor:
        """SLOWRES: one step of the learned low-pass temporal residual.

        ``target`` is the existing temporal-path acceleration; the returned value is the
        carried state ``r_t``.  A slot that was absent at the previous step is a new car:
        its state starts at zero.  ``tau`` is the softplus time constant (optionally
        conditioned on the IDM acceleration, gap and closing speed) and the optional
        slew-rate cap is a learned positive scale.
        """
        zeros = torch.zeros_like(target)
        previous = state.get("temporal_residual_mps2")
        if previous is None or tuple(previous.shape) != tuple(target.shape):
            previous = zeros
        else:
            previous = previous.to(dtype=target.dtype, device=target.device)
        prev_present = state.get("temporal_residual_present")
        if prev_present is None or tuple(prev_present.shape) != tuple(present.shape):
            prev_present = present
        else:
            prev_present = prev_present.to(device=present.device).bool()
        reset = present.bool() & ~prev_present
        previous = torch.where(reset, zeros, previous)

        tau = softplus_positive(self.temporal_tau_raw).to(target.dtype)
        if self.temporal_tau_head is not None:
            gap, lead_v = lead_neighbour(kin, present, self.physics)
            closing = kin[..., 4] - lead_v
            features = torch.stack((
                (physics_accel / 5.0).clamp(-2.0, 2.0),
                (gap.clamp(0.0, 200.0) / 50.0),
                (closing.clamp(-20.0, 20.0) / 10.0)), dim=-1)
            modulation = torch.tanh(self.temporal_tau_head(features)).squeeze(-1)
            tau = tau * torch.exp(0.5 * modulation)
        slew = (None if self.temporal_slew_raw is None
                else softplus_positive(self.temporal_slew_raw).to(target.dtype))
        residual = slow_residual_step(target, previous, tau, self.physics.dt_s,
                                      slew_max_mps2=slew)
        return torch.where(present.bool(), residual, zeros)

    def _controller_embedding(self, controller_id: Optional[Tensor], is_cav: Tensor,
                              shape: tuple[int, int], dtype, device,
                              action_command: Optional[Tensor] = None) -> Optional[Tensor]:
        """Controller-id embedding on CAV slots only (``None`` for the structured model)."""
        if not self.black_box:
            return None
        b, k = shape
        if controller_id is None:
            cid = torch.zeros(b, k, dtype=torch.long, device=device)
            cav = torch.zeros(b, k, dtype=torch.bool, device=device)
        else:
            cid = torch.as_tensor(controller_id, device=device).long()
            if cid.ndim == 0:
                cid = cid.expand(b, k)
            elif cid.ndim == 1:
                cid = cid.reshape(b, 1).expand(b, k)
            elif tuple(cid.shape) != (b, k):
                raise ValueError("controller_id must be scalar, [B] or [B,K]")
            cid = cid.clamp(0, self.n_controllers - 1)
            cav = is_cav
        emb = self.controller_embed(cid)
        if action_command is not None:
            command = torch.as_tensor(action_command, dtype=dtype, device=device)
            if command.ndim == 2 and tuple(command.shape) == (b, 2):
                command = command.unsqueeze(1).expand(b, k, 2)
            if tuple(command.shape) != (b, k, 2):
                raise ValueError("action_command must have shape [B,K,2]")
            emb = emb + self.command_proj(command)
        return emb * cav.unsqueeze(-1).to(dtype)

    def _driver_interaction(self, kin: Tensor, present: Tensor) -> Tensor:
        """Leader/follower gap and relative speed ``[B, K, 4]`` for the driver encoder."""
        gap, lead_v = lead_neighbour(kin, present, self.physics)
        follow_gap, follow_v = lane_neighbour(kin, present, lane_of(kin), self.physics,
                                              ahead=False)
        return torch.stack((gap.clamp(0, 250) / 100, (lead_v - kin[..., 4]) / 30,
                            follow_gap.clamp(0, 250) / 100,
                            (follow_v - kin[..., 4]) / 30), -1)

    def _default_driver_parameters(self, kin: Tensor, b: int, slots: int) -> Tensor:
        """Default driver parameters ``[B, K, 9]`` in :data:`PARAMETER_NAMES` order."""
        p = self.physics
        a_default, b_default, t_default, s0_default, _ = p.idm()
        defaults = kin.new_tensor((p.v0_mps, t_default, s0_default, a_default, b_default,
                                   0.2, p.mobil_threshold_mps2, p.lane_safe_gap_m, 1.0))
        return defaults.view(1, 1, -1).expand(b, slots, -1)

    def encode_drivers(self, state: dict[str, Tensor], *,
                       driver_history: Optional[Tensor] = None,
                       driver_history_mask: Optional[Tensor] = None,
                       history: Optional[Tensor] = None,
                       history_mask: Optional[Tensor] = None,
                       driver_defaults: Optional[Tensor] = None):
        """Encode driver parameters/embedding once from a state (round-5 FIX 1).

        Performs exactly the encoding :meth:`rollout` performs at its first step
        with the same inputs, so training (encode once at t0, reuse across
        truncated-BPTT chunks) and inference (encode once at t0) agree.
        Returns ``(encoder outputs or None, embedding or None, params dict or None)``.
        """
        kin = state_kin(state).float()
        present = state_present(state, kin)
        return self._encode_drivers(kin, present, driver_history, driver_history_mask,
                                    history, history_mask, driver_defaults, None, None)

    def _encode_drivers(self, kin: Tensor, present: Tensor,
                        driver_history: Optional[Tensor],
                        driver_history_mask: Optional[Tensor],
                        history: Optional[Tensor], history_mask: Optional[Tensor],
                        driver_defaults: Optional[Tensor],
                        driver_params: Optional[dict[str, Tensor]],
                        driver_embedding: Optional[Tensor]):
        """Infer driver parameters/embedding from the pre-t0 history when needed.

        Hybrid and long-history models need the embedding on every step (it
        conditions the residual net), so it is re-encoded whenever missing;
        ``driver_params`` mode only encodes while the parameters are not fixed yet.
        Returns ``(encoder outputs or None, embedding, driver_params)``.
        """
        b, slots = kin.shape[:2]
        observed = driver_history if driver_history is not None else history
        observed_mask = (driver_history_mask if driver_history_mask is not None
                         else history_mask)
        embedding_needed = (self.rollout_mode == "hybrid" or self.long_history
                            or self._wmlane_embedding) and driver_embedding is None
        params_needed = self.rollout_mode != "residual" and driver_params is None
        if not (params_needed or embedding_needed):
            return None, driver_embedding, driver_params
        if observed is None or observed_mask is None:
            if self.long_history:
                raise ValueError("--history-s requires masked pre-t0 driver_history")
            observed = torch.zeros((b, slots, self.history_len, self.history_features),
                                   device=kin.device, dtype=kin.dtype)
            observed_mask = torch.zeros((b, slots, self.history_len),
                                        device=kin.device, dtype=torch.bool)
        if getattr(self, "diag_encoder_population", False):
            # Inference-time diagnostic (round 5): treat every driver as unobserved.
            # The encoder then returns exactly the population-default parameters
            # (zero shrinkage) and an interaction-only embedding.
            observed_mask = torch.zeros_like(observed_mask)
        if driver_defaults is None:
            driver_defaults = self._default_driver_parameters(kin, b, slots)
        if self.long_history and (observed.shape != (
                b, slots, self.driver_history_len, LONG_HISTORY_F)
                or observed_mask.shape != observed.shape[:-1]):
            raise ValueError("long driver history shape does not match configured duration")
        encoded = self.driver_encoder(observed, observed_mask,
                                      self._driver_interaction(kin, present),
                                      driver_defaults)
        driver_embedding = encoded["embedding"]
        if driver_params is None and self.rollout_mode != "residual":
            values = encoded["values"]
            driver_params = {name: values[..., i] for i, name in enumerate(PARAMETER_NAMES)}
        return encoded, driver_embedding, driver_params

    def _residual_prediction(self, kin: Tensor, present: Tensor,
                             conditioning: Optional[Tensor],
                             courtesy: Optional[Tensor] = None) -> dict[str, Tensor]:
        if self.checkpoint_residual and self.training and conditioning is None:
            return checkpoint(self.residual_net, kin, present, courtesy=courtesy,
                              use_reentrant=False, preserve_rng_state=False)
        return self.residual_net(kin, present, conditioning=conditioning,
                                 courtesy=courtesy)

    def set_courtesy_inputs(self, enabled: bool = True) -> None:
        """Enable/disable the V10COURTESY input adapter without losing weights.

        Old (7-channel kinematics) checkpoints have no courtesy tensors; the rebuilt
        adapter is zero-initialised, so enabling it is an exact no-op until trained
        and every existing weight is copied by name/shape.
        """
        enabled = bool(enabled)
        if bool(self.hdv.courtesy_inputs) == enabled:
            return
        old = self.residual_net.state_dict()
        self.hdv = replace(self.hdv, courtesy_inputs=enabled)
        self.residual_net = HdvResidualNet(self.hdv)
        self.residual_net.load_state_dict(old, strict=False)

    def set_merge_negotiation(self, enabled: bool = True) -> None:
        """Enable/disable the V10NEGOT merge-negotiation block on a loaded checkpoint.

        The block is created zero-initialised; enabling it on an older checkpoint is an
        exact no-op until fine-tuning and every existing weight is preserved.
        """
        enabled = bool(enabled)
        if (self.merge_negotiation is not None) == enabled:
            return
        if not enabled:
            self.merge_negotiation = None
            return
        module = MergeNegotiationNet().to(self.device_of_parameters())
        self.merge_negotiation = module

    def device_of_parameters(self):
        """Device of the first parameter (``cpu`` for a parameterless model)."""
        return next(self.parameters()).device if any(True for _ in self.parameters()) \
            else torch.device("cpu")

    # ---------------------------------------------------------------- forward
    def forward(self, state: dict[str, Tensor], *, k_steps: Tensor | int = 0,
                history: Optional[Tensor] = None,
                history_mask: Optional[Tensor] = None,
                driver_history: Optional[Tensor] = None,
                driver_history_mask: Optional[Tensor] = None,
                driver_defaults: Optional[Tensor] = None,
                driver_params: Optional[dict[str, Tensor]] = None,
                driver_embedding: Optional[Tensor] = None,
                lane_age: Optional[Tensor] = None,
                baseline_lane_logits: Optional[Tensor] = None,
                controller_id: Optional[Tensor] = None,
                action_command: Optional[Tensor] = None,
                accel_noise: Optional[Tensor] = None,
                accel_noise_sigma_max: Optional[float] = None,
                lane_blocked: Optional[Tensor] = None,
                lane_v5_acc: Optional[Tensor] = None) -> dict[str, Tensor]:
        """Predict acceleration and lane logits for every slot.

        ``k_steps`` is the number of steps since the last real observation (scalar or
        ``[B, K]``).  Returns the physics and final accelerations, the residual terms,
        lane logits, the gate and the learned-slot mask (``hdv_mask``), plus driver
        parameters/embedding and lane probabilities when those components are active.

        ``accel_noise`` (optional ``[B, K]`` standard-normal draws, default None = deterministic,
        bit-identical) adds aleatoric noise ``accel_noise * exp(residual log-sigma)`` (sigma capped
        at ``accel_noise_sigma_max`` when given) to the acceleration of learned (HDV) slots
        before the plant clamp; used by ``wms.control.futures`` for sampled futures.

        LANEFIX: ``lane_blocked`` (v2 hazard features) is the blocked-step count carried from
        the previous step (``None`` at a rollout start: from the long driver history); the
        updated count is returned as ``lane_blocked``.  With ``lane_decision='sample'`` the
        physics feasibility mask is returned as ``lane_feasible [B, K, 3]``.

        LANEV5: ``lane_v5_acc`` (``lane_model='hazard_v5'``) are the lane network's recurrent
        accumulators after the previous step (``None`` at a rollout start: from the long
        driver history); the updated ones are returned as ``lane_v5_acc``, and with sampled /
        gated decisions the slots that draw as ``lane_sample [B, K]``.
        """
        kin = state_kin(state).float()
        present = state_present(state, kin)
        if kin.shape[-1] != KIN_F:
            raise ValueError(f"veh_kin must have {KIN_F} features")
        b, slots = kin.shape[:2]
        is_cav = state.get("is_cav", torch.zeros_like(present)).bool()
        # Structured WM-S learns HDVs only and teacher-forces CAVs; the black-box
        # variant treats every present vehicle as a learned mover.
        learned = present if self.black_box else (present & ~is_cav)
        native_lane = state.get(cav_motion.WM_LANE_KEY)
        lane_learned = learned if native_lane is None else learned | (present & native_lane.bool())
        k = torch.as_tensor(k_steps, device=kin.device, dtype=torch.long)
        if k.ndim == 0:
            k = k.expand(b, slots)
        elif tuple(k.shape) != (b, slots):
            raise ValueError("k_steps must be scalar or [B,K]")

        encoded = None
        final_envelope = bool(getattr(self.physics, "emergency_braking", False))             and self.physics_step is None
        if self.rollout_mode in ("driver_params", "hybrid") or self.long_history:
            encoded, driver_embedding, driver_params = self._encode_drivers(
                kin, present, driver_history, driver_history_mask, history, history_mask,
                driver_defaults, driver_params, driver_embedding)
            # S-WMSPEED (exact): with a native-lane CAV mask and secure-gap lane changes the
            # baseline lane logits are recomputed just below; do not build them twice.
            replaced = (native_lane is not None
                        and getattr(self.physics, "lane_change_secure_gap", False)
                        and fastpath.dedupe_mobil())
            physics, lane_base = self._physics_step({**state, "veh_kin": kin}, driver_params,
                                                    emergency_envelope=not final_envelope,
                                                    lane_logits=not replaced)
        else:
            physics, lane_base = self._physics_step({**state, "veh_kin": kin},
                                                    emergency_envelope=not final_envelope)
        if native_lane is not None and getattr(self.physics, "lane_change_secure_gap", False):
            lane_base = self.mobil_merge_logits(
                {**state, "veh_kin": kin, "is_cav": is_cav & ~native_lane.bool()}, driver_params)
        # R13E review B2: the emergency envelope bounds the FINAL pre-clamp acceleration
        # (after residual, courtesy and yield). ``physics`` stays the un-enveloped
        # composition input; the reported physics baseline is enveloped as before.
        physics_reported = physics
        if final_envelope:
            physics_reported = torch.where(present, apply_emergency_envelope(
                kin, present, physics, self.physics, driver_params), torch.zeros_like(physics))

        yield_bound = yield_selected = yield_weight = None
        if self.mainline_merge_yield_learnable:
            yield_bound, yield_selected, yield_weight = self.merge_yield(
                kin, present, physics, self.physics, driver_params, is_cav=is_cav,
                merge_wait_s=state.get("merge_wait_s"), return_influence=True)

        conditioning = (driver_embedding
                        if self.rollout_mode == "hybrid" or self.long_history else None)
        lane_base = lane_base if baseline_lane_logits is None else baseline_lane_logits
        blocked_now = v5_acc = v5_sample = v5_hazard = None
        v5_precomputed = None
        merge_prob = merge_logit = None
        if getattr(self, "lane_v5", None) is not None and (
                bool(self.hdv.courtesy_inputs) or self.merge_negotiation is not None
                or self.follower_courtesy is not None):
            # V10COURTESY / V10NEGOT: the residual / negotiation consumes the merge
            # head's current probability, so the lane-v5 pass runs first (forward values
            # are unchanged; frozen dropout / deterministic features make the reorder exact).
            v5_precomputed = self._lane_v5_forward(
                kin, present, lane_learned, lane_base, lane_age, lane_v5_acc, driver_history,
                driver_history_mask, merge_wait=state.get("merge_wait_s"))
            with torch.no_grad():
                merge_target = int(getattr(self.physics, "merge_target_lane", 1))
                p_merge = torch.softmax(v5_precomputed[0].float(), dim=-1)[..., merge_target]
                merge_prob = torch.nan_to_num(p_merge, nan=0.0).to(kin.dtype)
                # R5YIELD: detached merge-head score (the intent trend is its difference).
                merge_logit = v5_precomputed[0][..., merge_target].detach().to(kin.dtype)
        negotiation = None
        if self.merge_negotiation is not None:
            pair = pair_features(
                kin, present, self.physics, merge_prob=merge_prob,
                merge_logit=merge_logit,
                prev_merge_logit=state.get("merge_neg_intent_logit"),
                wait_s=state.get("merge_wait_s"),
                accel_ema=state.get("merge_neg_accel_ema"))
            negotiation = self.merge_negotiation(
                pair["features"], ext_features=pair["ext_features"],
                follower_index=pair["follower_index"],
                valid_pair=pair["valid_pair"], ramp=pair["ramp"])
        core = None
        if self.merge_core is not None and self.merge_pivot is None:
            # SECUREGAP: anticipated two-sided secure-gap geometry of every ramp slot.
            core = self._merge_core_forward(
                kin, present, driver_embedding, (b, slots), merge_prob)
        courtesy = None
        if bool(self.hdv.courtesy_inputs):
            courtesy = courtesy_features(
                kin, present, self.physics, lane_age=lane_age,
                merge_wait_s=state.get("merge_wait_s"), merge_prob=merge_prob)
        residual_kin = kin
        if (self.merge_pivot is not None and self.spec.merge_pivot_state_st
                and "merge_pivot_lane_weights" in state):
            # Hard one-hot forward, ST backward into the next residual/dynamics
            # step. Discrete neighbor lookup remains exact. Observed/projected
            # lane corrections and new identities do not inherit a stale adjoint.
            weights = state["merge_pivot_lane_weights"]
            valid = present & (weights.detach().argmax(-1) == kin[..., 1:4].argmax(-1))
            if "merge_pivot_identity" in state:
                identity = state.get("vehicle_id", state.get("identity"))
                if identity is not None:
                    valid = valid & (identity == state["merge_pivot_identity"])
            residual_kin = kin.clone()
            residual_kin[..., 1:4] = kin[..., 1:4] + torch.where(
                valid.unsqueeze(-1), weights - weights.detach(), torch.zeros_like(weights))
        base = self._residual_prediction(residual_kin, present, conditioning, courtesy)
        controller_emb = self._controller_embedding(
            controller_id, is_cav, (b, slots), kin.dtype, kin.device,
            action_command=action_command)
        temporal = self._temporal_delta(history, history_mask, k, (b, slots),
                                        kin.dtype, kin.device, controller_emb=controller_emb)
        temporal_accel = temporal[..., 0]
        temporal_residual_state = None
        if self.diag_zero_temporal or self.temporal_accel_off:
            # V10BRKPROBE (a) / R13F: remove only the temporal acceleration emission; the
            # attention head, driver adaptation and the temporal lane path stay intact.
            temporal_accel = torch.zeros_like(temporal_accel)
        elif self.temporal_lane_bias is not None:
            # R13M: the emission is the learned per-lane constant of the step-start lane (0 off road).
            on_road = kin[..., 1:4].sum(-1) > 0
            lane_now = kin[..., 1:4].argmax(-1)
            temporal_accel = torch.where(on_road, self.temporal_lane_bias.to(kin.dtype)[lane_now],
                                         torch.zeros_like(temporal_accel))
        elif self.temporal_state_mean is not None:
            # R13S: deterministic state-conditional mean of the emission (current state + k only).
            from . import temporal_state_table as tst
            bins = tst.bin_index(*tst.state_features(kin.detach(), present, k))
            counts = [getattr(self, f"temporal_state_n{lv}") for lv in range(len(tst.LEVELS))]
            temporal_accel = tst.lookup(list(self.temporal_state_mean), counts, bins).to(kin.dtype)
        if self.temporal_residual_mode == "slow":
            # SLOWRES: replace the direct temporal emission by the carried low-pass state.
            temporal_accel = self._slow_temporal_accel(
                temporal_accel, state, kin, present, physics)
            temporal_residual_state = temporal_accel
        if self.diag_temporal_tau_s is not None:
            # V10BRKPROBE (c): causal first-order low-pass on the temporal output.  The
            # state is cleared by ``reset_diag_state`` at every new rollout.
            tau = float(self.diag_temporal_tau_s)
            alpha = float(self.physics.dt_s) / (tau + float(self.physics.dt_s))
            previous = self._diag_temporal_lp
            if previous is None or tuple(previous.shape) != tuple(temporal_accel.shape) \
                    or previous.device != temporal_accel.device:
                previous = torch.zeros_like(temporal_accel)
            filtered = previous + alpha * (temporal_accel - previous)
            self._diag_temporal_lp = filtered.detach()
            temporal_accel = filtered
        gate = self.gate(k).to(kin.dtype)

        # R12C physics-only calibration: the residual attention/accel emission is an exact 0
        # (gate * (0 + temporal) with temporal_accel_off as well -> correction exactly 0);
        # the residual lane read-out keeps its value (unused by the hazard_v5 lane model).
        residual_delta = base["accel_delta_mps2"]
        if self.residual_accel_off:
            residual_delta = torch.zeros_like(residual_delta)
        accel_delta = (torch.zeros_like(physics) if self.rollout_mode == "driver_params"
                       else residual_delta + temporal_accel)
        mainline_delta = None
        if (self.mainline_residual is not None and self.rollout_mode != "driver_params"
                and not self.residual_accel_off):
            # S-RAMPFIX: learned mainline interaction offset joins the combined learned
            # correction BEFORE the gate / soft bound / cap / plant clamp (gated and
            # bounded exactly like the existing residual; zero at init; the CAV ``learned``
            # mask downstream excludes CAV slots and the head masks to mainline geometry).
            # driver_params keeps its exact no-residual contract; the R12C physics-only arm
            # (residual_accel_off) zeroes this emission like every other residual one.
            mainline_mask = mainline_lane_mask(kin, present, self.physics)
            mainline_out = self.mainline_residual(
                mainline_interaction_features(kin, present, self.physics), mainline_mask)
            mainline_delta = mainline_out["accel_delta_mps2"]
            accel_delta = accel_delta + mainline_delta
        raw_residual = gate * accel_delta
        correction = bounded_accel_correction(
            gate * accel_delta, soft_bound_pos=self.residual_soft_bound_mps2,
            soft_bound_neg=self.residual_soft_bound_neg_mps2,
            cap_pos=self.residual_cap_mps2, cap_neg=self.residual_cap_neg_mps2)
        if self.diag_residual_bound_mps2 is not None:
            # V10BRKPROBE (b): hard inference bound on the COMBINED learned residual.
            limit = float(self.diag_residual_bound_mps2)
            correction = correction.clamp(-limit, limit)
        if accel_noise is None:
            accel = torch.where(learned, physics + correction, physics)
        else:
            sigma = base["accel_log_sigma"].exp()
            if accel_noise_sigma_max is not None:
                sigma = sigma.clamp_max(float(accel_noise_sigma_max))
            accel = torch.where(learned, physics + correction + accel_noise.to(sigma.dtype) * sigma,
                                physics)
        ema = None
        intent_logit = None
        if negotiation is not None:
            # (a) learned yield: the ramp car's paired lane-1 follower decelerates.
            yield_delta = scatter_follower_accel(
                negotiation["yield_accel"], negotiation["follower_index"],
                negotiation["valid_pair"], slots) * learned.to(kin.dtype)
            accel = accel + yield_delta
            # (b) gap-conditioned ramp speed adaptation (IDM core + small learned residual).
            accel = accel + negotiation["ramp_accel_delta"] * learned.to(kin.dtype)
            # Follower-acceleration history carried across steps (detached: a discrete EMA).
            prev = state.get("merge_neg_accel_ema")
            prev = (torch.zeros_like(kin[..., 5]) if prev is None else prev.detach().float())
            ema = 0.7 * prev + 0.3 * kin[..., 5].detach().float()
            intent_logit = (torch.zeros_like(kin[..., 5]) if merge_logit is None
                            else merge_logit.detach().float())
        if core is not None:
            # SECUREGAP: a learned follower yield only for a near-zero follower margin.
            core_yield = scatter_follower_accel(
                core["yield_accel"], core["follower_index"], core["valid_pair"], slots)
            accel = accel + core_yield * learned.to(kin.dtype)
        courtesy_out = None
        if self.follower_courtesy is not None:
            # R10B: learned courtesy offset on the target lane-1 follower of a ramp-lane car.
            # Exactly zero for a slot without a ramp partner and for every non-learned (CAV) slot.
            # R11 (version 2): + the ramp car's acceleration at t / t-1 (state + short motion
            # history) and its carried merge-zone time (missing -> the current state only).
            # R13A: the intent input is computed from the step-START state (no algebraic loop with
            # this step's acceleration); ``state_grad`` keeps the pair features' state derivative.
            v2 = self._tracks_courtesy_zone
            if self.follower_courtesy_intent == "pivot":
                courtesy_intent = pivot_clearance_intent(self.merge_pivot, kin, present,
                                                         self.physics, is_cav)
            elif self.follower_courtesy_intent == "none":
                courtesy_intent = None
            else:
                courtesy_intent = merge_prob
            pairs = follower_courtesy_pairs(
                kin, present, self.physics, merge_wait_s=state.get("merge_wait_s"),
                lane_age=lane_age, merge_prob=courtesy_intent,
                state_grad=self.follower_courtesy_state_grad,
                **({} if not v2 else {"version": 2, "zone_time_s": state.get(ZONE_STATE_KEY),
                                      "history": history, "history_mask": history_mask}))
            inputs = pairs["features"]
            if self.follower_courtesy.embed_dim:
                # Ablation only: the follower's driver embedding (detached; the encoder is frozen).
                emb = (driver_embedding.detach().to(kin.dtype) if driver_embedding is not None
                       else kin.new_zeros(b, slots, self.follower_courtesy.embed_dim))
                inputs = torch.cat((inputs, torch.where(pairs["has_partner"].unsqueeze(-1), emb,
                                                        torch.zeros_like(emb))), -1)
            applies = pairs["has_partner"] & learned
            raw = self.follower_courtesy_offset(inputs, live=self.follower_courtesy_rollout_grad)
            offset = torch.where(applies, raw, torch.zeros_like(raw))
            if self.follower_courtesy_off:
                # R12C: the courtesy offset is an exact 0 (emission off; the head may still be
                # trained by loss.follower_courtesy through its own internal replay).
                offset = torch.zeros_like(offset)
            courtesy_out = {"courtesy_accel_mps2": offset, "courtesy_applies": applies,
                            "courtesy_inputs": inputs.detach(),
                            "accel_pre_courtesy_mps2": accel.detach()}
            if not self.follower_courtesy_off:
                accel = accel + offset
        if yield_bound is not None:
            # Courtesy repair A and the physics yield are competing constraints:
            # the stronger deceleration wins; never sum two yielding offsets.
            # Retain the uninfluenced share of a positive correction: otherwise
            # even a zero-weight distant pair would cap it at the raw baseline.
            yield_bound = yield_bound + (1.0 - yield_weight) * (accel - physics).clamp_min(0.0)
            accel = torch.where(yield_selected, torch.minimum(accel, yield_bound), accel)
        ramp_adapt_weight = None
        if self.ramp_speed_adapt_learnable:
            # R16: ramp HDVs relax toward the target-lane leader along the acceleration lane
            # (deceleration-only virtual-leader bound; acceptance stays with the merge pivot).
            if driver_params is None:
                raise ValueError("ramp speed adaptation requires learned driver parameters")
            accel, _, ramp_adapt_weight = self.ramp_speed_adapt(
                kin, present, accel, self.physics, driver_params, learned=learned)
        if self.merge_response is not None:
            is_cav = state.get("is_cav", torch.zeros_like(present))
            accel = accel + self.merge_response(kin, present, is_cav)
        if final_envelope:
            # Once, on the final composed acceleration; d/d(accel) in [0, 1].
            accel = torch.where(present, apply_emergency_envelope(
                kin, present, accel, self.physics, driver_params), accel)
        if self.rollout_mode != "driver_params":
            accel = accel.clamp(self.physics.accel_min_mps2, self.physics.accel_max_mps2)

        if v5_precomputed is not None:
            lane_logits, v5_acc, v5_sample, v5_hazard = v5_precomputed
        elif getattr(self, "lane_v5", None) is not None:
            lane_logits, v5_acc, v5_sample, v5_hazard = self._lane_v5_forward(
                kin, present, lane_learned, lane_base, lane_age, lane_v5_acc, driver_history,
                driver_history_mask, merge_wait=state.get("merge_wait_s"))
        elif self.lane_hazard is not None:
            # WMLANE A: the dedicated hazard model replaces the residual lane delta;
            # physics-masked targets stay masked (hard feasibility).
            if self.lane_hazard_features == "v2":
                if lane_blocked is None:
                    lane_blocked = self.initial_lane_blocked(
                        present, kin, driver_history, driver_history_mask, driver_params)
                blocked_now = update_blocked(
                    lane_blocked, blocked_indicator(kin, present, self.physics, driver_params))
            lane_logits, _ = hazard_lane_logits(
                self.lane_hazard, kin, present, lane_learned, lane_base, self.physics,
                driver_params, driver_embedding, lane_age,
                **self._hazard_switches(blocked_now, rollout=True))
        else:
            lane_logits = self._residual_lane_logits(lane_base, base, temporal, k, gate)
        if negotiation is not None and self.lane_change_head is None:
            # (b) learned gap acceptance: additive correction on the ramp merge logit.
            target = int(getattr(self.physics, "merge_target_lane", 1))
            ramp = present & (torch.argmax(kin[..., 1:4], -1) == int(self.physics.merge_lane))
            value = lane_logits.gather(-1, torch.full_like(ramp, target, dtype=torch.long)
                                       .unsqueeze(-1)).squeeze(-1)
            value = torch.where(ramp, value + negotiation["merge_logit_delta"], value)
            lane_logits = lane_logits.scatter(
                -1, torch.full_like(ramp, target, dtype=torch.long).unsqueeze(-1),
                value.unsqueeze(-1))
        lane_probabilities = None
        if self.lane_change_head is not None:
            if driver_embedding is None:
                raise ValueError("history lane head requires its pre-t0 encoder embedding")
            context, feasible_actions = lane_context(
                kin, present, lane_base, self.physics, driver_params, lane_age)
            lane_logits = self.lane_change_head(driver_embedding, context, feasible_actions)
            lane_probabilities = torch.softmax(lane_logits, dim=-1)

        result = {"physics_accel_mps2": physics_reported, "accel_mps2": accel,
                  "raw_residual_mps2": raw_residual,
                  "accel_delta_mps2": accel - physics_reported, "lane_logits": lane_logits,
                  "gate": gate, "residual_log_sigma": base["accel_log_sigma"],
                  "hdv_mask": learned}
        if native_lane is not None:
            result["lane_decider_mask"] = lane_learned
        if ramp_adapt_weight is not None:
            result["ramp_speed_adapt_weight"] = ramp_adapt_weight
        if mainline_delta is not None:
            # The APPLIED contribution (gate included): exactly zero at k=0 and wherever
            # the head is masked, so the log never overstates the correction.
            result["mainline_residual_mps2"] = gate * mainline_delta
        if self.diag_emit_temporal:
            result["temporal_accel_pregate_mps2"] = temporal_accel
        if temporal_residual_state is not None:
            result["temporal_residual_mps2"] = temporal_residual_state
            result["temporal_residual_present"] = present
        if courtesy_out is not None:
            result.update(courtesy_out)
        if self.merge_threshold is not None:
            # R5THR: per-driver threshold location [B, K] (with gradient for the first-passage
            # loss; the rollout decision uses a detached sample).
            result["merge_threshold_location"] = self.merge_threshold(
                driver_embedding, (b, slots))
        if self.merge_frailty is not None:
            # R5FR: per-driver latent location and positive scale [B, K] (with gradient for
            # the frailty survival loss; the rollout decision uses a detached draw).
            location, sigma = self.merge_frailty(driver_embedding, (b, slots))
            result["merge_frailty_location"] = location
            result["merge_frailty_sigma"] = sigma
        if encoded is not None:
            result.update({"driver_parameters": encoded["values"],
                           "driver_uncertainty": encoded["uncertainty"],
                           "driver_shrinkage": encoded["shrinkage"]})
        if driver_embedding is not None and (self.lane_change_head is not None
                                             or self.rollout_mode == "hybrid"
                                             or self.long_history or self._wmlane_embedding):
            result["driver_embedding"] = driver_embedding
        if lane_probabilities is not None:
            result["lane_probabilities"] = lane_probabilities
        if blocked_now is not None:
            result["lane_blocked"] = blocked_now
        if v5_acc is not None:
            result["lane_v5_acc"] = v5_acc
        if v5_hazard is not None:
            result["lane_v5_head_logit"], result["lane_v5_acts"] = v5_hazard
        if negotiation is not None:
            result["merge_negotiation_yield_logit"] = negotiation["yield_logit"]
            result["merge_negotiation_yield_accel"] = negotiation["yield_accel"]
            result["merge_negotiation_ramp_delta"] = negotiation["ramp_accel_delta"]
            result["merge_negotiation_required_brake"] = negotiation["required_brake"]
            result["merge_negotiation_conflict_logit"] = negotiation["conflict_logit"]
            result["merge_negotiation_conflict_prob"] = negotiation["conflict_prob"]
            result["merge_negotiation_merge_delta"] = negotiation["merge_logit_delta"]
            result["merge_negotiation_follower"] = negotiation["follower_index"]
            result["merge_negotiation_valid"] = negotiation["valid_pair"]
            result["merge_neg_accel_ema"] = ema
            result["merge_neg_intent_logit"] = intent_logit
        if core is not None:
            # SECUREGAP: core parameters (with gradient), latent distribution, margins and the
            # per-step probability (detached only where the rollout needs a scalar decision).
            for key in ("gap_l", "gap_f", "v_l", "v_f", "a_l", "a_f", "has_l", "has_f",
                        "ramp", "intent", "alpha_log_location", "alpha_scale",
                        "margin", "margin_l", "margin_f", "residual", "probability",
                        "yield_accel", "follower_index", "valid_pair", "require_both"):
                result[f"merge_core_{key}"] = core[key]
            for key, value in core["params"].items():
                result[f"merge_core_{key}"] = value
        if self._samples_lanes:
            result["lane_feasible"] = hazard_feasible(lane_base, kin)
            if v5_sample is not None:
                result["lane_sample"] = v5_sample
        return result

    def _residual_lane_logits(self, lane_base: Tensor, base: dict, temporal: Tensor,
                              k: Tensor, gate: Tensor) -> Tensor:
        """Physics lane logits plus the gated, capped residual lane delta (default path)."""
        lane_delta = (torch.zeros_like(lane_base) if self.rollout_mode == "driver_params"
                      else base["lane_logit_delta"] + temporal[..., 1:])
        if self.long_history and self.rollout_mode == "driver_params":
            # Inferred style conditions the lane correction as well as IDM/MOBIL.
            lane_delta = base["lane_logit_delta"]
        lane_correction = bounded_value(
            self.lane_gate(k, shared=gate).to(lane_delta.dtype).unsqueeze(-1) * lane_delta,
            self.lane_logit_cap)
        lane_logits = lane_base + lane_correction
        if (self.physics.lane_rule in ("forced_merge_only", "learned_discretionary")
                or self.physics.merge_end_wait):
            # The learned residual lane delta can never unmask a forbidden change:
            # entries the physics masked stay masked after the correction and cap
            # (C4; in the default 'mobil' mode this branch is skipped, so outputs
            # are bit-identical to the pre-lane_rule model).
            lane_logits = torch.where(lane_base <= MASKED_LOGIT, lane_base, lane_logits)
        return lane_logits

    # ------------------------------------------------------------ WMLANE A / C
    @property
    def _wmlane_embedding(self) -> bool:
        """The hazard lane model / driver updater consume the driver embedding."""
        return (getattr(self, "lane_hazard", None) is not None
                or getattr(self, "driver_updater", None) is not None)

    @property
    def _tracks_lane_age(self) -> bool:
        return (self.lane_change_head is not None or getattr(self, "lane_hazard", None) is not None
                or getattr(self, "lane_v5", None) is not None)

    # ------------------------------------------------------------------ LANEFIX
    @property
    def _samples_lanes(self) -> bool:
        """Hazard lane decisions are sampled from the masked categorical (not argmax)."""
        if getattr(self, "lane_v5", None) is not None:
            # V8WM2 forced_window_mode='hazard': forced-window rows always draw.
            # V10UNBLOCK lane_learnable: every learned slot always draws its own head.
            return (getattr(self, "lane_decision", "argmax") in ("sample", "gated")
                    or lv5.forced_window_hazard(self.physics)
                    or bool(getattr(self.physics, "lane_learnable", False))
                    or self._mainline_mode != "default")
        return (getattr(self, "lane_hazard", None) is not None
                and getattr(self, "lane_decision", "argmax") == "sample")

    @property
    def _mainline_mode(self) -> str:
        """V8LANE mainline decision rule of the hazard_v5 network ('default' otherwise)."""
        if getattr(self, "lane_v5", None) is None:
            return "default"
        return getattr(self, "lane_v5_mainline", "default")

    @property
    def _tracks_lane_blocked(self) -> bool:
        return (getattr(self, "lane_hazard", None) is not None
                and getattr(self, "lane_hazard_features", "v1") == "v2")

    def _hazard_switches(self, lane_blocked: Optional[Tensor], *, rollout: bool) -> dict:
        """Keyword switches of :func:`hazard_lane_logits` (empty = WMLANE A defaults)."""
        kwargs: dict = {}
        if getattr(self, "lane_hazard_features", "v1") != "v1":
            kwargs.update(features=self.lane_hazard_features, lane_blocked=lane_blocked)
        if getattr(self, "merge_hazard_zone", "forced_late") != "forced_late":
            kwargs["merge_zone"] = self.merge_hazard_zone
        offsets = tuple(getattr(self, "lane_logit_offsets", (0.0, 0.0, 0.0)))
        if any(float(x) != 0.0 for x in offsets):
            kwargs["offsets"] = offsets
        if rollout and self.lane_hazard_detach:
            kwargs["detach_delta"] = True
        return kwargs

    # ------------------------------------------------------------------ LANEV5
    @property
    def _tracks_lane_v5(self) -> bool:
        return getattr(self, "lane_v5", None) is not None

    def _history_stride_steps(self) -> int:
        return round(self.history_stride_s / self.physics.dt_s) if self.long_history else 1

    def initial_lane_v5_acc(self, present: Tensor, kin: Tensor,
                            driver_history: Optional[Tensor],
                            driver_history_mask: Optional[Tensor]) -> Tensor:
        """Lane-v5 accumulators before t0 from the observed pre-t0 long driver history."""
        return lv5.initial_acc(present, kin, self.physics,
                               driver_history if self.long_history else None,
                               driver_history_mask if self.long_history else None,
                               self._history_stride_steps())

    @property
    def _tracks_merge_wait(self) -> bool:
        """V8WM3: the rollout carries the lane-end waiting time ``state['merge_wait_s']``."""
        return bool(getattr(self.physics, "merge_impatience", False))

    def initial_merge_wait(self, present: Tensor, kin: Tensor,
                           driver_history: Optional[Tensor],
                           driver_history_mask: Optional[Tensor]) -> Tensor:
        """V8WM3: lane-end waiting time at t0 (incl. t0) from the observed pre-t0 long history."""
        from wms.model.merge_wait import initial_merge_wait
        return initial_merge_wait(present, kin, self.physics,
                                  driver_history if self.long_history else None,
                                  driver_history_mask if self.long_history else None,
                                  self._history_stride_steps())

    def lane_v5_base(self, kin: Tensor, present: Tensor) -> Tensor:
        """Physics lane logits (feasibility mask) of the teacher-forced lane-v5 rows.

        Population-default driver settings: under ``lane_gap_rule='speed'`` the mask does not
        depend on the driver settings, so the offline pre-fit and the training loss agree.
        """
        state = {"veh_kin": kin.float(), "present": present.bool()}
        if self.physics_step is None:
            return self.mobil_merge_logits(state, None)
        return self._physics_step(state, None)[1]

    def lane_v5_mask_fn(self):
        """Feasibility source of the teacher-forced lane-v5 rows: ``None`` = read it from the
        lane features (in-module physics with the speed gap rule; identical, tested), else
        :meth:`lane_v5_base`."""
        if self.physics_step is None and lv5.fast_mask_applies(self.physics):
            return None
        return self.lane_v5_base

    def lane_v5_head_ap_effective(self) -> tuple[float, float, float]:
        """Per-head TRAIN-val AP used by ``lane_decision='gated'`` (config value where >= 0,
        else the AP stored in the lane network by the pre-fit / calibration tool)."""
        stored = self.lane_v5.head_ap.tolist()
        return tuple(float(c) if float(c) >= 0.0 else float(s)
                     for c, s in zip(self.lane_v5_head_ap, stored))  # type: ignore[return-value]

    def _lane_v5_forward(self, kin: Tensor, present: Tensor, learned: Tensor, lane_base: Tensor,
                         lane_age: Optional[Tensor], acc: Optional[Tensor],
                         driver_history: Optional[Tensor],
                         driver_history_mask: Optional[Tensor],
                         merge_wait: Optional[Tensor] = None):
        """Lane logits, updated accumulators, the drawing slots and (V8WM emit_merge_hazard) the
        own-head logit with gradient + acting rows of the lane-v5 model.  ``merge_wait``: V8WM3
        lane-end waiting time (per-car forced-window hazard floor; ``None`` = constant floor)."""
        if acc is None:
            acc = self.initial_lane_v5_acc(present, kin, driver_history, driver_history_mask)
        raw, acc, inst = lv5.step_features(acc, kin, present, self.physics, lane_age,
                                           extra=self.lane_v5.extra_features)
        offsets = tuple(getattr(self, "lane_logit_offsets", (0.0, 0.0, 0.0)))
        with torch.no_grad() if self.lane_hazard_detach else contextlib.nullcontext():
            logits, head, acts = lv5.lane_logits_v5(self.lane_v5, raw, inst, learned, lane_base,
                                                    self.physics, offsets, merge_wait,
                                                    self.lane_v5_merge_calib)
            if getattr(self.physics, "lane_learnable", False):
                # V10LANEOPT: in 'courtesy' the rollout decision of the mainline heads is frozen
                # outside the learned lane-1 -> lane-0 approach-zone overtake; the merge head and
                # the teacher-forced lane loss are untouched.
                logits = lv5.restrict_courtesy_logits(logits, inst["lane"], inst["s"], learned,
                                                      self.physics)
        hazard = None
        if (getattr(self, "emit_merge_hazard", False)
                or self.merge_decision in ("threshold", "frailty", "secure_gap")):
            # V8WM merge-timing loss / R5THR threshold decision: the own-head logit with
            # gradient on the rollout state.
            if self.lane_hazard_detach and torch.is_grad_enabled():
                _, head, _ = lv5.lane_logits_v5(self.lane_v5, raw, inst, learned, lane_base,
                                                self.physics, offsets, merge_wait,
                                                self.lane_v5_merge_calib)
            hazard = (head, acts)
        sample = None
        if self._samples_lanes:
            if getattr(self.physics, "lane_learnable", False):
                # V10UNBLOCK: all three heads sample their own calibrated probability,
                # independent of the lane_decision AP gate and of the mainline rule.
                # V10LANEOPT mainline_lane_mode='courtesy': only the merge head and the
                # lane-1 -> lane-0 approach-zone overtake draw (rest of the mainline frozen).
                sample = lv5.courtesy_sample_mask(learned.bool().clone(), inst["lane"], inst["s"],
                                                  learned, self.physics)
            elif self.lane_decision == "gated":
                sample = lv5.sample_mask(inst["lane"], learned, lv5.gate_open(
                    self.lane_v5_head_ap_effective(), self.lane_v5_gates))
            elif self.lane_decision == "sample":
                sample = learned
            else:
                sample = torch.zeros_like(learned, dtype=torch.bool)
            mainline = self._mainline_mode
            if getattr(self.physics, "lane_learnable", False):
                mainline = "default"
            if mainline != "default":
                # V8LANE: overtake / return rows draw (sample) or are decided by the
                # population budget in step() (budget*: they do not draw here).
                main = lv5.mainline_rows(inst["lane"], learned)
                sample = ((sample.bool() | main) if mainline == "sample"
                          else (sample.bool() & ~main))
            if lv5.forced_window_hazard(self.physics):
                # V8WM2: the forced window's merge is a hazard draw whatever the decision rule
                # (a closed gate / argmax would otherwise never merge at P < 0.5).
                sample = sample.bool() | (learned.bool()
                                          & lv5.forced_window(inst["lane"], inst["s"],
                                                              self.physics))
        return logits, acc, sample, hazard

    def initial_lane_blocked(self, present: Tensor, kin: Tensor,
                             driver_history: Optional[Tensor],
                             driver_history_mask: Optional[Tensor],
                             driver_params: Optional[Mapping[str, Tensor]]) -> Tensor:
        """Blocked-step count before t0 (v2 hazard features) from the long driver history."""
        v0 = (torch.full_like(kin[..., 4], float(self.physics.v0_mps)) if driver_params is None
              else driver_params["v0"]).clamp_min(1.0)
        stride = (round(self.history_stride_s / self.physics.dt_s) if self.long_history else 1)
        return initial_lane_blocked(present, kin.dtype, driver_history, driver_history_mask,
                                    v0, stride)

    def lane_event_logits(self, state: dict[str, Tensor], *,
                          driver_params: Optional[dict[str, Tensor]] = None,
                          driver_embedding: Optional[Tensor] = None,
                          lane_age: Optional[Tensor] = None,
                          lane_blocked: Optional[Tensor] = None) -> Tensor:
        """One-step absolute lane logits ``[B, K, 3]`` of the hazard model on ``state``.

        Teacher-forced (recorded-state) evaluation used by the lane-event loss and the
        one-step event diagnostics; requires ``lane_model='hazard'``.  ``lane_blocked`` (v2
        features) is the blocked-step count INCLUDING this state (``None``: 1 if blocked now).
        The rollout-only decouple detach never applies here.
        """
        if getattr(self, "lane_v5", None) is not None:
            kin = state_kin(state).float()
            present = state_present(state, kin)
            is_cav = state.get("is_cav", torch.zeros_like(present)).bool()
            acc = state.get("lane_v5_acc")
            if acc is None:
                acc = lv5.initial_acc(present, kin, self.physics, None, None, 1)
            raw, _, inst = lv5.step_features(acc, kin, present, self.physics, lane_age,
                                             extra=self.lane_v5.extra_features)
            with torch.no_grad():
                lane_base = self.lane_v5_base(kin, present)
            logits, _, _ = lv5.lane_logits_v5(self.lane_v5, raw, inst, present & ~is_cav,
                                              lane_base, self.physics,
                                              tuple(getattr(self, "lane_logit_offsets",
                                                            (0.0, 0.0, 0.0))),
                                              merge_calib=getattr(self, "lane_v5_merge_calib",
                                                                  None))
            return logits
        if self.lane_hazard is None:
            raise ValueError("lane_event_logits requires model.lane_model='hazard'")
        kin = state_kin(state).float()
        present = state_present(state, kin)
        is_cav = state.get("is_cav", torch.zeros_like(present)).bool()
        with torch.no_grad():
            # The physics prior is a fixed input of the event model here (memory: the
            # loss runs over every recorded step of the unroll at once).
            if self.physics_step is None:
                lane_base = self.mobil_merge_logits({**state, "veh_kin": kin}, driver_params)
            else:
                _, lane_base = self._physics_step({**state, "veh_kin": kin}, driver_params)
        if self._tracks_lane_blocked and lane_blocked is None:
            lane_blocked = update_blocked(torch.zeros_like(kin[..., 0]),
                                          blocked_indicator(kin, present, self.physics,
                                                            driver_params))
        logits, _ = hazard_lane_logits(self.lane_hazard, kin, present, present & ~is_cav,
                                       lane_base, self.physics, driver_params,
                                       driver_embedding, lane_age,
                                       **self._hazard_switches(lane_blocked, rollout=False))
        return logits

    def update_drivers(self, driver_params: dict[str, Tensor],
                       driver_embedding: Optional[Tensor], state: dict[str, Tensor],
                       observed: Tensor, *, predicted_v: Optional[Tensor] = None,
                       predicted_accel: Optional[Tensor] = None):
        """WMLANE C: refine driver settings of the ``observed`` slots of a TRUE state.

        Deployment entry point for belief/RSU updates (``rollout`` calls it through
        ``observe_states``/``observe_mask``).  Identity when ``driver_update='off'``.
        """
        if self.driver_updater is None:
            return driver_params, driver_embedding
        kin = state_kin(state).float()
        present = state_present(state, kin)
        return self.driver_updater(driver_params, driver_embedding, kin, present, observed,
                                   self.physics, predicted_v=predicted_v,
                                   predicted_accel=predicted_accel)

    def _observe_step(self, current: dict, out: dict, nxt_present: Tensor, obs_kin: Tensor,
                      obs_mask: Tensor, params, embedding):
        """Reset observed HDV slots to their true state and update their driver settings."""
        is_cav = current.get("is_cav")
        hdv = nxt_present.bool() if is_cav is None else nxt_present.bool() & ~is_cav.bool()
        obs = obs_mask.bool() & hdv & (obs_kin[..., 1:4].sum(-1) > 0)
        s = torch.where(obs, obs_kin[..., 0].to(out["next_s_m"].dtype), out["next_s_m"])
        v = torch.where(obs, obs_kin[..., 4].to(out["next_v_mps"].dtype), out["next_v_mps"])
        a = torch.where(obs, obs_kin[..., 5].to(out["accel_mps2"].dtype), out["accel_mps2"])
        lane = torch.where(obs, obs_kin[..., 1:4].argmax(-1).to(out["next_lane"].dtype),
                           out["next_lane"])
        if self.driver_updater is not None and params is not None:
            post = {key: value for key, value in current.items() if key != "veh_kin"}
            post.update({"s_m": s, "v_mps": v, "a_mps2": a, "lane": lane,
                         "present": nxt_present})
            params, embedding = self.update_drivers(
                params, embedding, post, obs, predicted_v=out["next_v_mps"],
                predicted_accel=out["accel_mps2"])
        return s, v, a, lane, params, embedding

    # ------------------------------------------------------------------- step
    def hold_teacher(self, state: dict[str, Tensor]) -> Tensor:
        """Native hold-speed CAV motion ``[B, K, 4] = s, lane, v, a`` (no controller)."""
        return cav_motion.hold_speed_teacher(state["s_m"], state["lane"], state["v_mps"],
                                             self.physics.dt_s)

    @property
    def _uses_driver_embedding(self) -> bool:
        return (self.rollout_mode == "hybrid" or self.lane_change_head is not None
                or self.long_history or self._wmlane_embedding)

    def step(self, state: dict[str, Tensor], *, k_steps: Tensor | int = 0,
             history: Optional[Tensor] = None, history_mask: Optional[Tensor] = None,
             driver_history: Optional[Tensor] = None,
             driver_history_mask: Optional[Tensor] = None,
             driver_defaults: Optional[Tensor] = None,
             driver_params: Optional[dict[str, Tensor]] = None,
             driver_embedding: Optional[Tensor] = None,
             lane_age: Optional[Tensor] = None,
             cav_teacher: Optional[Tensor] = None,
             cav_controller: Optional[Callable[[dict[str, Tensor]], Tensor]] = None,
             baseline_lane_logits: Optional[Tensor] = None,
             controller_id: Optional[Tensor] = None,
             action_id: Optional[Tensor] = None,
             action_command: Optional[Tensor] = None,
             projection: Optional[FeasibilityProjection] = None,
             accel_noise: Optional[Tensor] = None,
             accel_noise_sigma_max: Optional[float] = None,
             lane_blocked: Optional[Tensor] = None,
             lane_uniform: Optional[Tensor] = None,
             lane_v5_acc: Optional[Tensor] = None) -> dict[str, Tensor]:
        """Advance one 0.5 s step; CAVs are never learned movers in structured mode.

        With entries enabled the state carries an ``entrant`` mask: t0 CAVs keep the
        recorded teacher, while entrant CAVs (which have no recorded plan) run the
        supplied ``cav_controller`` or native hold-speed motion.  ``projection``
        (optional) projects the next state onto the feasible set.

        LANEFIX ``lane_decision='sample'``: learned slots draw their lane from the masked
        categorical with ``lane_uniform [B, K]`` (``None``: from the global torch RNG).
        """
        native_lane_mask = getattr(cav_controller, "native_lane_mask", None)
        if native_lane_mask is not None:
            if self.black_box or self.merge_pivot is None:
                raise ValueError("native_lane_change requires a structured first-passage WM")
            if self.rollout_mode not in ("driver_params", "hybrid") and not self.long_history:
                raise ValueError("native_lane_change requires observation-inferred driver settings")
            state = {**state, cav_motion.WM_LANE_KEY: native_lane_mask(
                {**state, "physics_step_index": k_steps})}
        forward_kwargs = dict(
            k_steps=k_steps, history=history, history_mask=history_mask,
            driver_history=driver_history, driver_history_mask=driver_history_mask,
            driver_defaults=driver_defaults, driver_params=driver_params,
            baseline_lane_logits=baseline_lane_logits,
            controller_id=(controller_id if action_id is None else action_id),
            action_command=action_command)
        if self.merge_pivot is not None and projection is not None:
            raise ValueError("first_passage uses exact geometry; padded projection must be off")
        if self._uses_driver_embedding:
            forward_kwargs["driver_embedding"] = driver_embedding
        if self._tracks_lane_age:
            forward_kwargs["lane_age"] = lane_age
        if accel_noise is not None:
            forward_kwargs.update(accel_noise=accel_noise, accel_noise_sigma_max=accel_noise_sigma_max)
        if self._tracks_lane_blocked:
            forward_kwargs["lane_blocked"] = lane_blocked
        if self._tracks_lane_v5:
            forward_kwargs["lane_v5_acc"] = lane_v5_acc
        if self.checkpoint_forward and torch.is_grad_enabled():
            out = checkpoint(partial(self.forward, **forward_kwargs), state,
                             use_reentrant=False, preserve_rng_state=self.training)
        else:
            out = self.forward(state, **forward_kwargs)
        lane_deciders = out.get("lane_decider_mask", out["hdv_mask"])

        s, v = state["s_m"], state["v_mps"]
        present = state_present(state, state_kin(state))
        cav = state.get("is_cav", torch.zeros_like(present)).bool() & present
        dt = self.physics.dt_s
        if self.merge_pivot is not None:
            moved = pivot.advance_longitudinal(state_kin(state), out["accel_mps2"], dt,
                                               self.physics.stop_time_integration)
            next_s, next_v = moved[..., 0], moved[..., 4]
        else:
            next_v = (v + out["accel_mps2"] * dt).clamp_min(0.0)
            next_s = s + 0.5 * (v + next_v) * dt
            if self.physics.stop_time_integration:
                accel = out["accel_mps2"]
                decel = torch.where(accel < 0.0, -accel, torch.ones_like(accel))
                stopping_distance = v.square() / (2.0 * decel)
                next_s = torch.where(v + accel * dt < 0.0, s + stopping_distance, next_s)
        if self._samples_lanes:
            if lane_uniform is None:
                lane_uniform = torch.rand(present.shape, device=present.device)
            choice = sample_lane_choice(out["lane_logits"], out["lane_feasible"],
                                        out.get("lane_sample", lane_deciders), lane_uniform)
            if self._mainline_mode in ("budget", "budget_prop"):
                # V8LANE population-matched overtake / return decisions (lv5.budget_choice).
                choice, out["lane_v5_budget"] = lv5.budget_choice(
                    choice, out["lane_logits"], out["lane_feasible"], state["lane"].long(),
                    lane_deciders, state.get("lane_v5_budget"), lane_uniform,
                    self._mainline_mode)
        else:
            choice = out["lane_logits"].argmax(-1)
        if self.lane_change_head is None:
            next_lane = choice
        else:
            lane = state["lane"].long()
            next_lane = lane + (choice == 2).long() - (choice == 1).long()
        if self.merge_threshold is not None:
            # R5THR: the merge head no longer draws a per-step Bernoulli; a persistent
            # threshold decides the first feasible crossing on its own logit.
            next_lane = self._apply_merge_threshold(state, state_kin(state), present, out,
                                                    next_lane)
        elif self.merge_frailty is not None:
            # R5FR: per-step Bernoulli opportunity with ONE persistent per-driver latent.
            next_lane = self._apply_merge_frailty(state, state_kin(state), present, out,
                                                  next_lane)
        elif self.merge_core is not None and self.merge_pivot is None:
            # SECUREGAP: smooth secure-gap probability with ONE persistent assertiveness latent.
            next_lane = self._apply_merge_core(state, state_kin(state), present, out,
                                               next_lane)

        direct = (torch.zeros_like(cav)
                  if action_id is None or not self.apply_direct_commands else
                  (action_id == DIRECT_ID) & cav)
        # The callback runs on empty-CAV ticks too (lifecycle/entry bookkeeping).
        # Its vectorized execution law still needs the real per-slot cap there;
        # align cap availability with that callback contract, not CAV presence.
        exec_cap = (self._exec_cap(out, state, driver_params)
                    if (cav.any() or cav_controller is not None)
                    and getattr(self, "cav_exec_cap", "off") != "off" else None)
        if direct.any():
            direct, next_s, next_v = self._apply_direct_commands(
                out, direct, s, v, next_s, next_v, action_command, exec_cap=exec_cap)
        if self.merge_pivot is not None and cav_controller is None and cav_teacher is None:
            # The no-command fallback (no controller AND no recorded teacher) is
            # longitudinal prediction with the current lane held; a supplied
            # ``cav_teacher`` (recorded-arm replay) is honoured below exactly as on the
            # non-pivot path (S-FIX9: it used to be silently dropped here).
            next_lane = torch.where(cav, state["lane"].long(), next_lane)
        # The controller callback owns episode-persistent lifecycle state and must run exactly
        # once per physics tick even after every CAV has exited (empty CAV mask).
        if (not self.black_box and (cav.any() or cav_controller is not None)
                and (self.merge_pivot is None or cav_controller is not None
                     or cav_teacher is not None)):
            next_s, next_v, next_lane = self._apply_cav_overrides(
                out, state, cav, direct, present, k_steps, action_id, action_command,
                cav_teacher,
                cav_controller, next_s, next_v, next_lane, exec_cap=exec_cap,
                driver_params=driver_params)

        if self.merge_pivot is not None:
            # P1: exactly one longitudinal update above; CAV current-action execution
            # precedes LC geometry, while EVERY neighbor keeps its pre-change lane.
            # The merge is evaluated once, then final conflicts are resolved once.
            pinned = None
            if getattr(self.physics, "ramp_lane_end_stop", False):
                # Decision geometry sees the car held at the real lane end; the hold is
                # kept only for cars that stay on the ramp lane (a merging car moves on).
                free_s, free_v = next_s, next_v
                next_s, next_v, pinned = pivot.ramp_lane_end_stop(
                    state_kin(state), next_s, next_v, present, self.physics)
            next_lane = self._apply_merge_pivot(state, out, next_s, next_v, next_lane,
                                               lane_uniform, driver_params,
                                               delegated=out.get(cav_motion.WM_MERGE_KEY))
            if pinned is not None:
                release = pinned & (next_lane != int(self.physics.merge_lane))
                next_s = torch.where(release, free_s, next_s)
                next_v = torch.where(release, free_v, next_v)

        if (self.merge_pivot is None and cav_motion.WM_MERGE_KEY in out
                and bool(out[cav_motion.WM_MERGE_KEY].any())):
            raise ValueError("a CAV callback delegated ramp merges to the WM merge law, which "
                             "needs merge_decision='first_passage'")
        if self.merge_pivot is None and (self.physics.merge_end_wait or lv5.forced_window_hazard(self.physics)
                or getattr(self.physics, "lane_learnable", False)):
            # V6MERGE: a vehicle staying on the accel lane never passes its end; it stops there
            # (V8WM2 hazard window: a feasible car may now wait in the window, so it must not
            # creep past the end, where no merge is ever allowed).  V10UNBLOCK lane_learnable:
            # lane-end stopping is kept even though every behavioural veto is removed.
            end = self.physics.merge_zone_s1_m
            overrun = (present & (next_lane == self.physics.merge_lane) & (s <= end)
                       & (next_s > end))
            next_s = torch.where(overrun, torch.full_like(next_s, end), next_s)
            next_v = torch.where(overrun, torch.zeros_like(next_v), next_v)
        next_present = present.clone()
        exit_s = state.get("exit_s_m")
        if exit_s is not None:
            next_present = next_present & (next_s <= exit_s)
        if self.lane_conflict_resolve and self.merge_pivot is None:
            next_lane = resolve_lane_conflicts(
                next_s, state_kin(state)[..., 6], state["lane"].long(), next_lane,
                next_present, out["hdv_mask"], self.physics.gap_overlap_margin_m)
        if projection is not None:
            length = state.get("length_m", torch.full_like(s, 5.0))
            proj = projection(next_s, next_v, next_lane, next_present, length,
                              is_cav=state.get("is_cav"))
            next_s, next_v, next_lane = proj["s_m"], proj["v_mps"], proj["lane"]
            out["projection_active"] = proj["active"]
        out.update({"next_s_m": next_s, "next_v_mps": next_v,
                    "next_lane": next_lane, "next_present": next_present})
        return out

    def pivot_decision_geometry(self, state, accel, *, next_s=None, next_v=None,
                                fixed_next_lane=None, route_shift=None, delegated=None):
        """P1 LC substep shared by deployment and recorded causal replay.

        Decision is after the single trapezoid/stop-time longitudinal update and
        causal CAV execution, before exits/projection; lanes are the t snapshot.
        This is the WM contract, not a claim of verified SUMO internal ordering.
        """
        kin = state_kin(state)
        if next_s is None or next_v is None:
            moved = pivot.advance_longitudinal(kin, accel, self.physics.dt_s,
                                               self.physics.stop_time_integration)
        else:
            moved = kin.clone()
            moved[..., 0], moved[..., 4], moved[..., 5] = next_s, next_v, accel
        present = state_present(state, kin)
        if getattr(self.physics, "ramp_lane_end_stop", False):
            # SUMO never moves a ramp-lane car past its lane end (idempotent when the
            # caller already applied it).  Opt-in physics flag; default off = unchanged.
            moved[..., 0], moved[..., 4], _ = pivot.ramp_lane_end_stop(
                kin, moved[..., 0], moved[..., 4], present, self.physics, route_shift)
        cav = state.get("is_cav", torch.zeros_like(present)).bool()
        # LOOP14C ``delegated``: CAV slots whose ramp merge follows the HDV law (decision makers,
        # never immutable target occupancy).  None keeps the legacy CAV exclusion exactly.
        decider = ~cav if delegated is None else ~cav | delegated.bool()
        fixed = cav if delegated is None else cav & ~delegated.bool()
        fixed_target = (None if fixed_next_lane is None else
                        present & fixed & (fixed_next_lane == int(self.physics.merge_target_lane)))
        return pivot.geometry(moved, present, self.physics, decision_mask=decider,
                              fixed_target=fixed_target, route_shift=route_shift)

    def _apply_merge_pivot(self, state, out, next_s, next_v, next_lane, uniform,
                            driver_params=None, delegated=None):
        kin = state_kin(state)
        present = state_present(state, kin)
        geometry_delegated = delegated
        native_lane = state.get(cav_motion.WM_LANE_KEY)
        if native_lane is not None:
            # A native mainline insertion competes like an HDV insertion, rather
            # than gaining the immutable occupancy/priority of a policy command.
            geometry_delegated = (native_lane.bool() if delegated is None else
                                  delegated.bool() | native_lane.bool())
        geo = self.pivot_decision_geometry(state, out["accel_mps2"],
                                           next_s=next_s, next_v=next_v,
                                           fixed_next_lane=next_lane, delegated=geometry_delegated)
        own = present & out.get("lane_decider_mask", out["hdv_mask"]).bool()
        if delegated is not None:
            # LOOP14C: delegated ramp CAVs merge by the same learned law (own driver settings).
            own = own | (present & delegated.bool())
        ramp_lane, target_lane = int(self.physics.merge_lane), int(self.physics.merge_target_lane)
        ramp = own & (state["lane"].long() == ramp_lane)
        geo["eligible"] = geo["eligible"] & own
        c = self.merge_pivot(geo)
        veto_margin = None
        willingness = c
        if getattr(self.physics, "secure_gap_veto", False):
            if driver_params is None:
                driver_params = {name: out["driver_parameters"][..., i]
                                 for i, name in enumerate(PARAMETER_NAMES)}
            # Explicit rollout opt-in: recorded likelihood and courtesy intent keep
            # calling the original ungated boundary, even when this flag is on.
            c = self.merge_pivot(geo, driver_params)
            veto_margin = self.merge_pivot.secure_gap_margin(geo, driver_params)
        params = self.merge_pivot.params()
        carry = {key: state["merge_pivot_" + key] for key in _MERGE_PIVOT_STATE_KEYS
                 if "merge_pivot_" + key in state}
        identity = state.get("vehicle_id", state.get("identity"))
        # Integrated rollout diagnostics are not the backward estimator. The
        # likelihood term rebuilds its own graph; hard lane credit uses ST below.
        hybrid = self.merge_pivot.hybrid
        updated = pivot.update_state(c.detach() if hybrid else c,
                                      params["tau"].detach() if hybrid else params["tau"],
                                      geo["ramp"] & own, present,
                                      carry=carry, identity=identity, uniform=uniform,
                                      retain=(geo["ramp_open"] & own if getattr(
                                          self.physics, "ramp_risk_retain", False) else None),
                                      w=params["w"].detach() if hybrid else None,
                                      quadrature_order=self.spec.merge_quadrature_order,
                                      # S-SLOWROLL plain attribute (no state_dict entry) or the
                                      # S-SPEED2 ``pivot_diag_off`` switch: skip the diagnostics.
                                      diagnostics=bool(getattr(
                                          self, "merge_pivot_rollout_diagnostics", True))
                                      and not speed_flags.enabled("pivot_diag_off"))
        crossing = updated["cross"] & ramp
        # Analytically scored, pre-willingness opportunities own their intervals.
        # Competing learned mainline insertions cannot make E depend on other z.
        rear = next_s - kin[..., 6]
        intersects = (torch.maximum(rear.unsqueeze(-1), rear.unsqueeze(-2))
                      < torch.minimum(next_s.unsqueeze(-1), next_s.unsqueeze(-2)))
        reserved_overlap = (intersects & geo["eligible"].unsqueeze(-2)).any(-1)
        other_incoming = (own & ~ramp & (state["lane"] != target_lane)
                          & (next_lane == target_lane) & reserved_overlap)
        next_lane = torch.where(other_incoming, state["lane"].long(), next_lane)
        proposed = torch.where(ramp, torch.where(crossing, target_lane, ramp_lane), next_lane)
        moved = kin.clone()
        moved[..., 0], moved[..., 4] = next_s, next_v
        final = pivot.resolve_conflicts(moved, present, state["lane"].long(), proposed, own)
        committed = ramp & (final == target_lane)
        if bool((crossing & ~committed).any()):
            raise RuntimeError("first_passage reservation/commit inconsistency")
        active = geo["eligible"] & ~updated["terminal"]
        st = pivot.hard_st(willingness, params["tau"], updated["z"], self.merge_st_width, active,
                           w=params.get("w"), epsilon=updated.get("epsilon"),
                           secure_gap_margin=veto_margin)
        updated["terminal"] = updated["terminal"] | committed
        hard_weights = torch.nn.functional.one_hot(final.clamp_min(0), 3).to(c.dtype)
        direction = c.new_zeros(3)
        direction[target_lane], direction[ramp_lane] = 1.0, -1.0
        weights = hard_weights + (st - st.detach()).unsqueeze(-1) * direction
        out["merge_pivot_lane_weights"] = weights
        probabilities = out.get("lane_probabilities", torch.softmax(out["lane_logits"], -1))
        out["lane_probabilities"] = torch.where(ramp.unsqueeze(-1), weights, probabilities)
        previous = carry.get("maximum", torch.full_like(c, -torch.inf))
        old_identity = carry.get("identity")
        reset = ~carry.get("present", torch.zeros_like(present)).bool() | ~present
        if identity is not None and old_identity is not None:
            reset = reset | (identity != old_identity)
        previous = torch.where(reset, torch.full_like(c, -torch.inf), previous)
        log_mass = pivot.log_event_mass(previous, updated["maximum"], params["tau"])
        out.update({"merge_pivot_c": c, "merge_pivot_eligible": geo["eligible"],
                    "merge_pivot_physical_eligible": geo["physical_eligible"],
                    "merge_pivot_reservation_veto": geo["reservation_veto"],
                    "merge_pivot_cross": crossing, "merge_pivot_committed": committed,
                    "merge_pivot_st": st, "merge_pivot_cdf": -torch.expm1(
                        pivot.log_survival(updated["maximum"], params["tau"])),
                    "merge_pivot_event_mass": log_mass.exp()})
        if self.merge_pivot.hybrid:
            out["merge_pivot_cdf"] = updated["cdf"]
            out["merge_pivot_event_mass"] = updated["event_mass"]
        for key in _MERGE_PIVOT_STATE_KEYS:
            if key in updated:
                out["merge_pivot_" + key] = updated[key]
        for key, value in params.items():
            out["merge_pivot_" + key] = value
        for key in ("gap_f", "gap_l", "v_f", "v_l", "has_f", "has_l"):
            out["merge_pivot_geometry_" + key] = geo[key]
        return final

    def _merge_core_forward(self, kin: Tensor, present: Tensor,
                            driver_embedding: Optional[Tensor], shape: tuple[int, int],
                            intent: Optional[Tensor]) -> dict:
        """SECUREGAP: anticipated margins, latent distribution, probability and follower yield."""
        feat = core_features(kin, present, self.physics, intent=intent)
        latent = self.merge_core(driver_embedding, shape)
        params = self.merge_core.params()
        require_both = torch.zeros_like(kin[..., 0], dtype=torch.bool)
        # Population mean of alpha for the logged margin; the rollout uses its own draw.
        alpha_mean = torch.exp(latent["alpha_log_location"].clamp(-8.0, 8.0))
        margin_l, margin_f = secure_gap_sides(
            feat["gap_l"], feat["gap_f"], kin[..., 4], feat["v_l"], feat["v_f"],
            feat["a"], feat["a_l"], feat["a_f"], feat["has_l"], feat["has_f"],
            params["s0"], params["T"], params["b"], params["tau_a"], alpha_mean,
            require_both)
        residual = self.merge_core.margin_residual(
            margin_l, margin_f, kin[..., 4], feat["v_l"], feat["v_f"],
            feat["gap_l"], feat["gap_f"], feat["intent"])
        margin = torch.minimum(margin_l, margin_f) + residual
        temperature = params["temperature"]
        probability = torch.sigmoid(margin / temperature.clamp_min(1.0e-3))
        yield_accel = self.merge_core.yield_accel(
            margin_f, feat["intent"], kin[..., 4], feat["v_l"], feat["v_f"],
            feat["gap_f"], temperature)
        return {**feat, "alpha_log_location": latent["alpha_log_location"],
                "alpha_scale": latent["alpha_scale"], "margin_l": margin_l,
                "margin_f": margin_f, "residual": residual, "margin": margin,
                "probability": probability, "yield_accel": yield_accel,
                "valid_pair": feat["ramp"] & feat["has_f"], "params": params,
                "require_both": require_both}

    def _apply_merge_threshold(self, state: dict, kin: Tensor, present: Tensor,
                               out: dict, next_lane: Tensor) -> Tensor:
        """R5THR: persistent gap-acceptance threshold instead of the Bernoulli merge draw.

        Uses the merge head's logit ``z_t`` (with the negotiation correction when active),
        the envelope-feasible in-zone mask (``lane_v5_acts``) and the carried threshold /
        running maximum / previous presence.  Returns the updated lane decision; the carried
        state entries are stored in ``out`` for the rollout loop.
        """
        lane = int(self.physics.merge_lane)
        target = int(getattr(self.physics, "merge_target_lane", 1))
        ramp_lane = (state.get("lane") if state.get("lane") is not None
                     else kin[..., 1:4].argmax(-1)).long()
        current_lane = torch.as_tensor(ramp_lane, device=kin.device).long()
        s = kin[..., 0]
        end = float(self.physics.merge_zone_s1_m)
        start = float(getattr(self.physics, "merge_zone_s0_m", 0.0))
        ramp = present.bool() & (current_lane == lane)
        in_zone = ramp & (s >= start) & (s <= end)
        head = out.get("lane_v5_head_logit")
        acts = out.get("lane_v5_acts")
        if not torch.is_tensor(head) or not torch.is_tensor(acts):
            return next_lane
        z = head.to(kin.dtype)
        delta = out.get("merge_negotiation_merge_delta")
        if torch.is_tensor(delta):
            z = z + delta.to(z.dtype)
        feasible = in_zone & acts.bool()
        location = out.get("merge_threshold_location")
        if not torch.is_tensor(location):
            location = torch.zeros_like(z)
        carry = update_threshold_state(
            z, ramp, feasible, present, location,
            u=state.get("merge_threshold_u"), has_u=state.get("merge_fp_has_u"),
            m=state.get("merge_fp_max"), prev_present=state.get("merge_fp_present"))
        out["merge_threshold_u"] = carry["u"]
        out["merge_fp_has_u"] = carry["has_u"]
        out["merge_fp_max"] = carry["m"]
        out["merge_fp_present"] = carry["prev_present"]
        out["merge_fp_cross"] = carry["cross"]
        # Never let the per-step head's own merge draw through: only the threshold crosses.
        attempted = (current_lane == lane) & (next_lane.long() == target)
        next_lane = torch.where(attempted, current_lane, next_lane.long())
        return torch.where(carry["cross"], torch.full_like(next_lane, target), next_lane)

    def _apply_merge_frailty(self, state: dict, kin: Tensor, present: Tensor,
                             out: dict, next_lane: Tensor) -> Tensor:
        """R5FR: per-step Bernoulli merge hazard with ONE persistent per-driver latent.

        Uses the merge head's logit ``z_t`` (with the negotiation correction when active),
        the envelope-feasible in-zone mask (``lane_v5_acts``) and a latent ``w`` drawn at the
        car's first ramp appearance from ``Normal(location, sigma)``.  The car merges when a
        per-step Bernoulli draw with prob ``sigmoid(z_t + w)`` succeeds.  The carried latent
        and presence are stored in ``out`` for the rollout loop (reset on slot reuse).
        """
        lane = int(self.physics.merge_lane)
        target = int(getattr(self.physics, "merge_target_lane", 1))
        ramp_lane = (state.get("lane") if state.get("lane") is not None
                     else kin[..., 1:4].argmax(-1)).long()
        current_lane = torch.as_tensor(ramp_lane, device=kin.device).long()
        s = kin[..., 0]
        end = float(self.physics.merge_zone_s1_m)
        start = float(getattr(self.physics, "merge_zone_s0_m", 0.0))
        ramp = present.bool() & (current_lane == lane)
        in_zone = ramp & (s >= start) & (s <= end)
        head = out.get("lane_v5_head_logit")
        acts = out.get("lane_v5_acts")
        if not torch.is_tensor(head) or not torch.is_tensor(acts):
            return next_lane
        z = head.to(kin.dtype)
        delta = out.get("merge_negotiation_merge_delta")
        if torch.is_tensor(delta):
            z = z + delta.to(z.dtype)
        feasible = in_zone & acts.bool()
        location = out.get("merge_frailty_location")
        if not torch.is_tensor(location):
            location = torch.zeros_like(z)
        sigma = out.get("merge_frailty_sigma")
        if not torch.is_tensor(sigma):
            sigma = torch.ones_like(z)
        carry = update_frailty_state(
            z, ramp, feasible, present, location, sigma,
            w=state.get("merge_frailty_w"), has_w=state.get("merge_frailty_has_w"),
            prev_present=state.get("merge_frailty_present"))
        out["merge_frailty_w"] = carry["w"]
        out["merge_frailty_has_w"] = carry["has_w"]
        out["merge_frailty_present"] = carry["prev_present"]
        out["merge_frailty_hazard"] = carry["hazard"]
        out["merge_frailty_cross"] = carry["cross"]
        # Never let the per-step head's own merge draw through: only the frailty draw crosses.
        attempted = (current_lane == lane) & (next_lane.long() == target)
        next_lane = torch.where(attempted, current_lane, next_lane.long())
        return torch.where(carry["cross"], torch.full_like(next_lane, target), next_lane)

    def _apply_merge_core(self, state: dict, kin: Tensor, present: Tensor,
                          out: dict, next_lane: Tensor) -> Tensor:
        """SECUREGAP: smooth secure-gap decision with ONE persistent assertiveness latent.

        The forward pass has already produced the anticipated gaps / required gaps and the
        mean-alpha margin.  This routine draws (once per car) the assertiveness ``alpha`` and
        evaluates the exact margin, then samples the learned sigmoid probability on
        envelope-feasible in-zone steps.  A recorded merge therefore always has positive
        likelihood; the hard physical envelope remains a veto.
        """
        lane = int(self.physics.merge_lane)
        target = int(getattr(self.physics, "merge_target_lane", 1))
        ramp_lane = (state.get("lane") if state.get("lane") is not None
                     else kin[..., 1:4].argmax(-1)).long()
        current_lane = torch.as_tensor(ramp_lane, device=kin.device).long()
        s = kin[..., 0]
        end = float(self.physics.merge_zone_s1_m)
        start = float(getattr(self.physics, "merge_zone_s0_m", 0.0))
        ramp = present.bool() & (current_lane == lane)
        in_zone = ramp & (s >= start) & (s <= end)
        acts = out.get("lane_v5_acts")
        location = out.get("merge_core_alpha_log_location")
        sigma = out.get("merge_core_alpha_scale")
        if not torch.is_tensor(acts) or not torch.is_tensor(location):
            return next_lane
        feasible = in_zone & acts.bool()
        carry = update_core_alpha(
            location, sigma if torch.is_tensor(sigma) else torch.ones_like(location),
            ramp, present, alpha=state.get("merge_core_alpha"),
            has_alpha=state.get("merge_core_has_alpha"),
            prev_present=state.get("merge_core_present"))
        out["merge_core_alpha"] = carry["alpha"]
        out["merge_core_has_alpha"] = carry["has_alpha"]
        out["merge_core_present"] = carry["prev_present"]
        params = {key: out.get(f"merge_core_{name}") for key, name in (
            ("s0", "s0"), ("T", "T"), ("b", "b"), ("tau_a", "tau_a"),
            ("temperature", "temperature"))}
        if not all(torch.is_tensor(value) for value in params.values()):
            return next_lane
        margin = secure_gap_margin(
            out["merge_core_gap_l"], out["merge_core_gap_f"], kin[..., 4],
            out["merge_core_v_l"], out["merge_core_v_f"], kin[..., 5],
            out["merge_core_a_l"], out["merge_core_a_f"],
            out["merge_core_has_l"], out["merge_core_has_f"],
            params["s0"], params["T"], params["b"], params["tau_a"], carry["alpha"],
            out.get("merge_core_residual"), out.get("merge_core_require_both"))
        probability = torch.sigmoid(margin / params["temperature"].clamp_min(1.0e-3))
        uniform = torch.rand(probability.shape, dtype=probability.dtype,
                             device=probability.device)
        cross = feasible & carry["has_alpha"].bool() & (uniform < probability)
        out["merge_core_margin"] = margin
        out["merge_core_probability"] = probability
        out["merge_core_cross"] = cross
        # Never let the per-step head's own merge draw through: only the core crosses.
        attempted = (current_lane == lane) & (next_lane.long() == target)
        next_lane = torch.where(attempted, current_lane, next_lane.long())
        return torch.where(cross, torch.full_like(next_lane, target), next_lane)

    def _exec_cap(self, out: dict, state: dict,
                  driver_params: Optional[dict[str, Tensor]]) -> Tensor:
        """SUMO-consistent ceiling ``[B, K]`` of commanded CAVs (EXECCAP_NOTES.md).

        ``out["physics_accel_mps2"]`` is the physics-core IDM acceleration of every present
        slot (CAV slots included) with the driver settings of this step; the settings are
        the supplied ``driver_params``, else those encoded in this forward, else defaults.
        """
        mode = cav_motion.check_exec_cap_mode(self.cav_exec_cap)
        from wms.control.execution import enabled as execution_parity_enabled
        if execution_parity_enabled():
            if mode != "idm":
                raise ValueError("execution parity requires learned-driver cav_exec_cap='idm'")
            params = driver_params
            if params is None and "driver_parameters" in out:
                params = {name: out["driver_parameters"][..., i]
                          for i, name in enumerate(PARAMETER_NAMES)}
            if params is None:
                raise ValueError("execution parity requires learned driver parameters")
            from wms.control.execution import inferred_follow_cap
            return inferred_follow_cap(state, params, dt=self.physics.dt_s)
        if mode.startswith("idm_normal"):
            return self._exec_cap_normal(state, cav_motion.exec_cap_settings(mode))
        if mode != "idm":
            raise ValueError(f"unsupported cav_exec_cap {mode!r}")
        params = driver_params
        if params is None and "driver_parameters" in out:
            params = {name: out["driver_parameters"][..., i]
                      for i, name in enumerate(PARAMETER_NAMES)}
        physics = out["physics_accel_mps2"]
        if params is None:
            v0, accel_limit = self.physics.v0_mps, self.physics.idm()[0]
        else:
            v0, accel_limit = params["v0"], params["accel"]
        return cav_motion.exec_accel_cap(physics, state["v_mps"].to(physics.dtype),
                                         self.physics.dt_s, v0, accel_limit)

    def _exec_cap_normal(self, state: dict, normal: Mapping[str, float]) -> Tensor:
        """INFRAV5 ``cav_exec_cap="idm_normal[:<table>]"``: the EXECCAP ceiling with fixed,
        operator-known CAV settings ``normal`` (``cav.CAV_SETTINGS_TABLES``: textbook
        ``cav_typical`` a 1.2 / b 2.0 / tau 1.2 / s0 2.0, or legacy ``normal_car`` a 2.6 / b 4.5 /
        tau 1.1 / s0 2.5; v0 = speedFactor x speed limit) on every slot instead of the
        WM-inferred ones (only commanded CAV slots ever use it).  Same physics-core IDM
        (leader search, accel-lane-end obstacle, plant bounds)."""
        if self.physics_step is not None:
            raise ValueError("cav_exec_cap='idm_normal' needs the in-module physics core")
        kin = state_kin(state).float()
        b, slots = kin.shape[:2]
        values = self._default_driver_parameters(kin, b, slots)
        params = {name: values[..., i] for i, name in enumerate(PARAMETER_NAMES)}
        v0 = float(self.physics.speed_limit_mps) * float(normal["speed_factor"])
        for name, value in (("v0", v0), ("time_headway", normal["time_headway"]),
                            ("min_gap", normal["min_gap"]), ("accel", normal["accel"]),
                            ("decel", normal["decel"])):
            params[name] = torch.full_like(params[name], float(value))
        physics = self.physics_accel({**state, "veh_kin": kin}, params)
        return cav_motion.exec_accel_cap(physics, state["v_mps"].to(physics.dtype),
                                         self.physics.dt_s, v0, float(normal["accel"]))

    def _apply_direct_commands(self, out: dict, direct: Tensor, s: Tensor, v: Tensor,
                               next_s: Tensor, next_v: Tensor,
                               action_command: Optional[Tensor],
                               exec_cap: Optional[Tensor] = None
                               ) -> tuple[Tensor, Tensor, Tensor]:
        """Drive ``direct`` CAVs from their issued command.

        Updates ``out["accel_mps2"]`` and returns ``(applied mask, next_s, next_v)``.
        ``exec_cap`` (``cav_exec_cap="idm"``) bounds the issued acceleration from above.
        """
        if action_command is None:
            raise ValueError("direct timeline action requires acceleration and target speed")
        command = torch.as_tensor(action_command, dtype=v.dtype, device=v.device)
        if exec_cap is not None:
            command = torch.stack((torch.minimum(command[..., 0], exec_cap.to(command.dtype)),
                                   command[..., 1]), -1)
        direct_s, direct_v, accel, feasible = cav_motion.direct_command_motion(
            s, v, command, float(self.physics.dt_s), self.physics.accel_min_mps2,
            self.physics.accel_max_mps2)
        applied = direct & feasible
        out["accel_mps2"] = torch.where(applied, accel, out["accel_mps2"])
        return applied, torch.where(applied, direct_s, next_s), torch.where(applied, direct_v, next_v)

    def _apply_cav_overrides(self, out: dict, state: dict, cav: Tensor, direct: Tensor,
                             present: Tensor, k_steps, action_id, action_command,
                             cav_teacher, cav_controller, next_s, next_v, next_lane,
                             exec_cap: Optional[Tensor] = None,
                             driver_params: Optional[dict[str, Tensor]] = None):
        """Replace the learned motion of CAV slots by teacher/controller motion.

        The controller callback runs once per physics tick (also on direct-only ticks and
        ticks with no present CAV) because the dispatcher owns episode-persistent controller
        lifecycle state; an all-false ``cav`` mask applies no override.
        With ``exec_cap`` the callback state carries it under ``cav.EXEC_CAP_KEY``; command
        executors apply it to their commanded slots (``cav.apply_exec_cap``), teachers ignore it.
        Policy14: the callback also receives the rollout's learned per-slot driver settings
        (``inferred_driver_parameters``, plus the 13E gs inputs T and b when the WM veto is on),
        the same ones the WM step uses, so the policy's lane mask and the WM agree.
        """
        callback_teacher = None
        if cav_controller is not None:
            callback_state = {**state, "physics_step_index": k_steps}
            if cav_motion.WM_LANE_KEY in state:
                callback_state[cav_motion.WM_LANE_TARGET_KEY] = next_lane
            if self.merge_pivot is not None:
                callback_state["merge_pivot_native_accel"] = out["accel_mps2"]
            # The controller's consequence guard needs learned settings even when
            # the WM's own optional 13E veto is off. Never infer from driver labels.
            if driver_params is not None:
                inferred = ({name: out["driver_parameters"][..., i]
                             for i, name in enumerate(PARAMETER_NAMES)}
                            if "driver_parameters" in out else {})
                inferred.update(driver_params)
                if all(name in inferred for name in PARAMETER_NAMES):
                    callback_state["inferred_driver_parameters"] = torch.stack(
                        [inferred[name] for name in PARAMETER_NAMES], -1)
            elif "driver_parameters" in out:
                callback_state["inferred_driver_parameters"] = out["driver_parameters"]
            callback_state["inferred_idm_delta"] = self.physics.idm()[4]
            settings = driver_params
            if settings is None and "driver_parameters" in out:
                settings = {name: out["driver_parameters"][..., i]
                            for i, name in enumerate(PARAMETER_NAMES)}
            # LOOP14 fix: a step output without learned settings (e.g. the merge-pivot
            # recorded-geometry pass of a timeline-command window) exports no gs inputs;
            # a consumer that needs them fails closed on the missing keys.
            if self.physics.secure_gap_veto and settings is not None:
                # R13E gs (review B3) has no standstill term: T and b only.
                for key, name in (("secure_gap_T_s", "time_headway"),
                                  ("secure_gap_b_mps2", "decel")):
                    callback_state[key] = settings[name]
            if exec_cap is not None:
                callback_state[cav_motion.EXEC_CAP_KEY] = exec_cap
            if action_id is not None:
                callback_state = {**callback_state, "action_id": action_id,
                                  "action_command": action_command}
            callback_teacher = cav_controller(callback_state)
            # LOOP14C: ramp CAVs whose merge the callback hands to the WM's HDV merge law.
            delegated = callback_state.get(cav_motion.WM_MERGE_KEY)
            if delegated is not None:
                out[cav_motion.WM_MERGE_KEY] = delegated.bool() & cav
        entrant = state.get("entrant")
        if entrant is None:
            cav_base, cav_new = cav, torch.zeros_like(cav)
        else:
            entrant = entrant.bool() & present
            cav_base, cav_new = cav & ~entrant, cav & entrant
        cav_base = cav_base & ~direct
        cav_new = cav_new & ~direct
        if cav_base.any():
            if cav_teacher is None and cav_controller is None:
                raise ValueError("CAVs require a real-controller teacher or callback")
            teacher = cav_teacher if cav_teacher is not None else callback_teacher
            cav_motion.check_teacher(teacher, state["s_m"])
            next_s, next_lane, next_v = self._take_teacher(out, teacher, cav_base,
                                                           next_s, next_lane, next_v)
        if cav_new.any():
            teacher = (callback_teacher if cav_controller is not None
                       else self.hold_teacher(state))
            next_s, next_lane, next_v = self._take_teacher(out, teacher, cav_new,
                                                           next_s, next_lane, next_v)
        return next_s, next_v, next_lane

    @staticmethod
    def _take_teacher(out: dict, teacher: Tensor, mask: Tensor, next_s: Tensor,
                      next_lane: Tensor, next_v: Tensor):
        next_s = torch.where(mask, teacher[..., 0], next_s)
        next_lane = torch.where(mask, teacher[..., 1].long(), next_lane)
        next_v = torch.where(mask, teacher[..., 2], next_v)
        out["accel_mps2"] = torch.where(mask, teacher[..., 3], out["accel_mps2"])
        return next_s, next_lane, next_v

    # ---------------------------------------------------------------- rollout
    def rollout(self, state: dict[str, Tensor], *, n_steps: int, k0: int = 0,
                cav_teacher: Optional[Tensor] = None,
                present_seq: Optional[Tensor] = None,
                junction_seq: Optional[Tensor] = None,
                is_cav_seq: Optional[Tensor] = None,
                history: Optional[Tensor] = None,
                history_mask: Optional[Tensor] = None,
                driver_history: Optional[Tensor] = None,
                driver_history_mask: Optional[Tensor] = None,
                driver_defaults: Optional[Tensor] = None,
                driver_params: Optional[dict[str, Tensor]] = None,
                driver_embedding: Optional[Tensor] = None,
                cav_controller: Optional[Callable[[dict[str, Tensor]], Tensor]] = None,
                truncation_steps: Optional[int] = None,
                lane_teacher: Optional[Tensor] = None,
                merge_lane_teacher: Optional[Tensor] = None,
                merge_teacher_frac: float = 0.0,
                controller_id: Optional[Tensor] = None,
                action_timeline: Optional[dict[str, Tensor]] = None,
                entries: Optional[object] = None,
                state_noise_std_m: float = 0.0,
                state_noise_std_mps: float = 0.0,
                projection: Optional[FeasibilityProjection] = None,
                observe_states: Optional[Tensor] = None,
                observe_mask: Optional[Tensor] = None,
                accel_noise: Optional[Tensor] = None,
                accel_noise_sigma_max: Optional[float] = None,
                lane_uniform: Optional[Tensor] = None,
                lane_seed: Optional[int] = None,
                lane_crn: bool = False,
                state_noise: Optional[Tensor] = None,
                entrant_driver_sampler: Optional[Callable] = None) -> dict[str, Tensor]:
        """Multi-step rollout; ``k`` (steps since the last observation) is ``k0 + i``.

        ``cav_teacher[:, i+1]`` is the CAV state after step ``i`` and ``present_seq
        [:, i+1]`` (optional) the recorded presence after it (exits are exogenous).
        ``junction_seq[:, i+1]`` marks a junction internal-lane transit during which a
        slot stays present.  The motion history is a rolling buffer fed with the
        model's own predicted states; the returned ``state``/``history`` allow a
        chunked (truncated-BPTT) continuation.

        ``entries`` (an entry process) inserts observed-counter arrivals into free
        padding slots before each step; the returned ``entrant`` marks slots that ever
        held a post-t0 newcomer.  With entries active the per-step ``is_cav_seq``
        override is skipped.  ``state_noise_std_m``/``state_noise_std_mps`` add
        zero-mean Gaussian noise to the predicted HDV position/speed (regulariser);
        zero draws nothing. ``state_noise [B, n_steps, K, 2]`` optionally supplies
        standard-normal draws (last axis: position, speed), scaled by those stds.
        It is chunk-local, just like ``lane_uniform``; callers key its generation
        by absolute steps. None preserves the global torch RNG path exactly.

        ``accel_noise`` (``[B, n_steps, K]`` standard-normal draws, default None) samples the
        HDV acceleration aleatorically with the residual head's predicted sigma (see
        :meth:`forward`); the draws are an input so callers control common random numbers.

        ``lane_teacher`` (``[B, T+1, K]`` recorded lanes; diagnostic, round 5) forces
        every step's lane decision to the recorded lane where the entry is ``>= 0``
        (post-step, so CAV teacher overrides stay in charge of CAV slots).

        Beyond the per-step ``positions``/``speeds``/``lanes``/``masks``/``accel``/
        ``lane_logits`` the result carries ``entrant_seq [B, T, K]`` (per-step entrant
        membership) and ``sections [B, T, K]`` (model-frame section id, -1 if absent).

        WMLANE C: ``observe_states [B, T+1, K, 7]`` + ``observe_mask [B, T+1, K]`` (index
        ``i+1`` = after step ``i``) reveal TRUE HDV states during the rollout (deployment:
        belief/RSU updates; training: a teacher-observed prefix).  The logged prediction
        of an observed step is the model's own; the carried state is reset to the
        observation and, with ``driver_update='ema'``, the car's driver settings and
        embedding are refined (returned as ``driver_parameters_updated`` /
        ``driver_embedding_updated``).  ``None`` (default) leaves the rollout unchanged.

        LANEFIX (``lane_decision='sample'`` only; ignored otherwise): ``lane_uniform``
        (``[B or 1, T, K]`` uniforms in [0, 1), the caller's draws: expand them to
        counterfactual rows like ``accel_noise`` for common random numbers), else
        ``lane_seed`` (a private generator: deterministic given the seed), else the global
        torch RNG; ``lane_crn`` shares one draw per (step, slot) across all batch rows.
        """
        current = dict(state)
        if controller_id is None:
            controller_id = current.get("controller_id")
        current.pop("controller_id", None)
        kin0 = state_kin(current)
        pres0 = state_present(current, kin0)
        if state_noise is not None:
            state_noise = torch.as_tensor(state_noise, device=kin0.device, dtype=kin0.dtype)
            if state_noise.shape != (*pres0.shape[:1], int(n_steps), pres0.shape[1], 2):
                raise ValueError("state_noise must be [B, n_steps, K, 2] (position, speed)")
        # Entrant membership is carried across truncated-BPTT chunks.
        entrant = current.get("entrant")
        entrant = (entrant.clone().bool() if torch.is_tensor(entrant)
                   else torch.zeros(pres0.shape, dtype=torch.bool, device=pres0.device))
        if history is None:
            gap0, _ = lead_neighbour(kin0, pres0, self.physics)
            history, history_mask = initial_history(
                history_step_features(kin0, gap0), pres0, self.history_len)
        lane_age = self._initial_lane_age(current, pres0, kin0, driver_history,
                                          driver_history_mask)
        lane_blocked = current.get("lane_blocked") if self._tracks_lane_blocked else None
        lane_v5_acc = current.get("lane_v5_acc") if self._tracks_lane_v5 else None
        if self._tracks_merge_wait and current.get("merge_wait_s") is None:
            # V8WM3: carried across truncated-BPTT chunks in the returned state.
            current["merge_wait_s"] = self.initial_merge_wait(pres0, kin0, driver_history,
                                                              driver_history_mask)
        if self._tracks_courtesy_zone and current.get(ZONE_STATE_KEY) is None:
            # R11: merge-zone time of the courtesy v2 input, carried across chunks.
            current[ZONE_STATE_KEY] = self.initial_courtesy_zone(pres0, kin0, driver_history,
                                                                 driver_history_mask)
        if self.merge_threshold is not None and current.get("merge_fp_present") is None:
            # R5THR: presence at the rollout start (first-step new-car detection).
            current["merge_fp_present"] = pres0.clone()
        if self.merge_frailty is not None and current.get("merge_frailty_present") is None:
            # R5FR: presence at the rollout start (first-step new-car detection).
            current["merge_frailty_present"] = pres0.clone()
        if (self.merge_core is not None and self.merge_pivot is None
                and current.get("merge_core_present") is None):
            # SECUREGAP: presence at the rollout start (first-step new-car detection).
            current["merge_core_present"] = pres0.clone()
        lane_draw = (self._lane_draws(lane_uniform, lane_seed, lane_crn, pres0)
                     if self._samples_lanes else None)
        timeline_dt = cav_motion.timeline_period(action_timeline)

        log = {name: [] for name in (
            "positions", "speeds", "lanes", "masks", "accel", "lane_logits",
            "raw_residuals", "change_probabilities", "lane_probabilities", "entrant",
            "physics_accels")}
        projection_totals: dict[str, float] = {}
        share = _ShareAccumulator()
        # Per-step activation checkpointing (round 5): only in training with gradients
        # and never with a stateful CAV controller, whose callback must not be replayed
        # a second time during recomputation.
        step_checkpoint = (self.checkpoint_step and self.training
                           and torch.is_grad_enabled() and cav_controller is None)
        if self.checkpoint_forward and os.environ.get('WMS_LOOPLONG_SPEED') == '1':
            # Keep the learned forward activation checkpoint, avoiding a second
            # nested checkpoint over the integrator/pivot. Besides redundant
            # recomputation, the outer checkpoint would replay random pivot
            # stream allocation inside batched gradient-parity VJPs. Retain
            # these physical-state operations instead; objective/RNG unchanged.
            step_checkpoint = False
        fixed_driver_params = driver_params
        fixed_driver_embedding = driver_embedding
        initial_driver_embedding = driver_embedding
        encoded_parameters = None
        noise_on = float(state_noise_std_m) > 0.0 or float(state_noise_std_mps) > 0.0
        if entrant_driver_sampler is not None and fixed_driver_params is None:
            _, fixed_driver_embedding, fixed_driver_params = self.encode_drivers(
                current, driver_history=driver_history, driver_history_mask=driver_history_mask,
                history=history, history_mask=history_mask, driver_defaults=driver_defaults)
            if fixed_driver_params is None:
                raise ValueError("entrant driver sampling requires persistent driver parameters")

        for step_i in range(int(n_steps)):
            if entries is not None:
                current = {key: (value.clone() if torch.is_tensor(value) else value)
                           for key, value in current.items()}
                newborn = entries.insert(current)
                entrant = entrant | newborn
                current["entrant"] = entrant
                if entrant_driver_sampler is not None:
                    fixed_driver_params = entrant_driver_sampler(current, newborn, fixed_driver_params)
            elif is_cav_seq is not None:
                current = {**current, "is_cav": is_cav_seq[:, step_i].bool()}
            log["entrant"].append(entrant)
            teacher = None if cav_teacher is None else cav_teacher[:, step_i + 1]
            action_id = action_command = None
            if action_timeline is not None:
                action_id, action_command = cav_motion.timeline_controls(
                    action_timeline, k0 + step_i, self.physics.dt_s, timeline_dt, pres0,
                    current["v_mps"].dtype)
            step_fn = partial(
                self.step, k_steps=k0 + step_i, history=history,
                history_mask=history_mask, cav_teacher=teacher,
                driver_history=driver_history if step_i == 0 else None,
                driver_history_mask=driver_history_mask if step_i == 0 else None,
                driver_defaults=driver_defaults if step_i == 0 else None,
                driver_params=fixed_driver_params, driver_embedding=fixed_driver_embedding,
                lane_age=lane_age, cav_controller=cav_controller,
                controller_id=controller_id, action_id=action_id,
                action_command=action_command, projection=projection,
                **({} if accel_noise is None else {
                    "accel_noise": accel_noise[:, step_i],
                    "accel_noise_sigma_max": accel_noise_sigma_max}),
                **({} if not self._tracks_lane_blocked else {"lane_blocked": lane_blocked}),
                **({} if not self._tracks_lane_v5 else {"lane_v5_acc": lane_v5_acc}),
                **({} if lane_draw is None else {"lane_uniform": lane_draw(step_i)}))
            out = checkpoint(step_fn, current, use_reentrant=False) if step_checkpoint \
                else step_fn(current)
            for key, value in (out.get("projection_active") or {}).items():
                projection_totals[key] = projection_totals.get(key, 0.0) + float(value)
            if lane_teacher is not None and not getattr(self.physics, "lane_learnable", False):
                # V10UNBLOCK: never let a lane teacher overwrite a free (learned) rollout.
                recorded_lane = lane_teacher[:, step_i + 1].long()
                out["next_lane"] = torch.where(recorded_lane >= 0, recorded_lane,
                                               out["next_lane"])
            if (merge_lane_teacher is not None and float(merge_teacher_frac) > 0.0
                    and step_i < int(round(float(merge_teacher_frac) * int(n_steps)))):
                # V10NEGOT curriculum: only the RAMP cars follow the recorded lane for the
                # first fraction of each window (the merge decision is teacher-forced while
                # the free-running state still comes from the model).
                rec_lane = merge_lane_teacher[:, step_i + 1].long()
                ramp_now = ((current["lane"].long() == int(self.physics.merge_lane))
                            & current["present"].bool())
                use = ramp_now & (rec_lane >= 0)
                out["next_lane"] = torch.where(use, rec_lane, out["next_lane"])
            if self.rollout_mode in ("driver_params", "hybrid") \
                    and fixed_driver_params is None:
                # The same driver identity/parameters persist through the rollout.
                fixed_driver_params = {name: out["driver_parameters"][..., i]
                                       for i, name in enumerate(PARAMETER_NAMES)}
                encoded_parameters = out["driver_parameters"]
            if fixed_driver_embedding is None and "driver_embedding" in out:
                fixed_driver_embedding = out["driver_embedding"]
                initial_driver_embedding = fixed_driver_embedding
            if self.rollout_mode == "hybrid":
                share.add(out)
            nxt_present = out["next_present"]
            if present_seq is not None:
                nxt_present = nxt_present & present_seq[:, step_i + 1]
            if junction_seq is not None:
                # A junction transit is not a departure: carry the slot through it.
                nxt_present = nxt_present | junction_seq[:, step_i + 1].bool()
            if noise_on:
                self._add_state_noise(
                    out, current, state_noise_std_m, state_noise_std_mps,
                    **({} if state_noise is None else {"state_noise": state_noise[:, step_i]}))
            self._record_step(log, out, current)
            log["masks"].append(nxt_present)
            nxt_s, nxt_v, nxt_a, nxt_lane = (out["next_s_m"], out["next_v_mps"],
                                             out["accel_mps2"], out["next_lane"])
            if observe_mask is not None:
                nxt_s, nxt_v, nxt_a, nxt_lane, fixed_driver_params, fixed_driver_embedding = \
                    self._observe_step(current, out, nxt_present, observe_states[:, step_i + 1],
                                       observe_mask[:, step_i + 1], fixed_driver_params,
                                       fixed_driver_embedding)
            if self._tracks_lane_age:
                changed = nxt_lane != current["lane"]
                lane_age = torch.where(changed, torch.zeros_like(lane_age), lane_age + 1)
            current = {**current, "s_m": nxt_s, "v_mps": nxt_v,
                       "a_mps2": nxt_a, "lane": nxt_lane,
                       "present": nxt_present}
            if self._tracks_lane_age:
                current["lane_age"] = lane_age
            if self._tracks_lane_blocked:
                lane_blocked = out["lane_blocked"]
                current["lane_blocked"] = lane_blocked
            if self._tracks_lane_v5:
                lane_v5_acc = out["lane_v5_acc"]
                current["lane_v5_acc"] = lane_v5_acc
            if "merge_neg_accel_ema" in out:
                current["merge_neg_accel_ema"] = out["merge_neg_accel_ema"].detach()
            if "merge_neg_intent_logit" in out:
                current["merge_neg_intent_logit"] = out["merge_neg_intent_logit"].detach()
            if "temporal_residual_mps2" in out:
                # SLOWRES: carried across steps (and detached at truncated-BPTT chunks).
                current["temporal_residual_mps2"] = out["temporal_residual_mps2"]
                current["temporal_residual_present"] = out["temporal_residual_present"]
            if "lane_v5_budget" in out:
                current["lane_v5_budget"] = out["lane_v5_budget"]
            current.pop("veh_kin", None)
            kin = state_kin(current)
            if self._tracks_merge_wait:
                current["merge_wait_s"] = update_merge_wait(current.get("merge_wait_s"), kin,
                                                            nxt_present, self.physics)
            if self._tracks_courtesy_zone:
                current[ZONE_STATE_KEY] = update_zone_time(current.get(ZONE_STATE_KEY), kin,
                                                           nxt_present, self.physics)
            if "merge_threshold_u" in out:
                # R5THR: threshold / running maximum / presence persist across chunks and
                # slot moves; a reused slot is reset by the next step's new-car check.
                current["merge_threshold_u"] = out["merge_threshold_u"].detach()
                current["merge_fp_has_u"] = out["merge_fp_has_u"].detach()
                current["merge_fp_max"] = out["merge_fp_max"].detach()
                current["merge_fp_present"] = out["merge_fp_present"].detach()
            if "merge_frailty_w" in out:
                # R5FR: ONE latent + presence persist across chunks and slot moves; a reused
                # slot is reset (redrawn) by the next step's new-car check.
                current["merge_frailty_w"] = out["merge_frailty_w"].detach()
                current["merge_frailty_has_w"] = out["merge_frailty_has_w"].detach()
                current["merge_frailty_present"] = out["merge_frailty_present"].detach()
            if "merge_core_alpha" in out:
                # SECUREGAP: ONE assertiveness latent + presence persist across chunks and
                # slot moves; a reused slot is reset (redrawn) by the new-car check.
                current["merge_core_alpha"] = out["merge_core_alpha"].detach()
                current["merge_core_has_alpha"] = out["merge_core_has_alpha"].detach()
                current["merge_core_present"] = out["merge_core_present"].detach()
            if "merge_pivot_z" in out:
                for key in _MERGE_PIVOT_STATE_KEYS:
                    if "merge_pivot_" + key in out:
                        current["merge_pivot_" + key] = out["merge_pivot_" + key].detach()
                current["merge_pivot_lane_weights"] = (
                    out["merge_pivot_lane_weights"] if self.spec.merge_pivot_state_st
                    else out["merge_pivot_lane_weights"].detach())
            gap, _ = lead_neighbour(kin, nxt_present, self.physics)
            history, history_mask = push_history(
                history, history_mask, history_step_features(kin, gap), nxt_present)
            if self.grad_decay_per_step < 1.0 and torch.is_grad_enabled():
                # Round 5b: damp gradients flowing back through the carried state by a
                # constant factor per step (forward values unchanged), so credit from far
                # in the future fades instead of exploding through chaotic dynamics.
                g = self.grad_decay_per_step
                current = {key: (_scale_grad(value, g)
                                 if torch.is_tensor(value) and value.requires_grad else value)
                           for key, value in current.items()}
                if history.requires_grad:
                    history = _scale_grad(history, g)
            if truncation_steps and (step_i + 1) % int(truncation_steps) == 0:
                current = {key: (value.detach() if torch.is_tensor(value) else value)
                           for key, value in current.items()}
                history = history.detach()

        result = self._rollout_result(
            log, current, history, history_mask, entrant, pres0, kin0, projection,
            projection_totals, encoded_parameters, initial_driver_embedding, share)
        if entrant_driver_sampler is not None:
            result['driver_parameters'] = torch.stack(
                [fixed_driver_params[name] for name in PARAMETER_NAMES], -1)
        if observe_mask is not None and fixed_driver_params is not None:
            result["driver_parameters_updated"] = torch.stack(
                [fixed_driver_params[name] for name in PARAMETER_NAMES], -1)
            if fixed_driver_embedding is not None:
                result["driver_embedding_updated"] = fixed_driver_embedding
        return result

    @staticmethod
    def _lane_draws(lane_uniform: Optional[Tensor], lane_seed: Optional[int], crn: bool,
                    present: Tensor) -> Callable[[int], Tensor]:
        """Per-step lane uniforms ``[B, K]`` for sampled lane decisions (see :meth:`rollout`)."""
        b, k = present.shape
        device = present.device
        if lane_uniform is not None:
            draws = torch.as_tensor(lane_uniform, device=device)
            if draws.ndim != 3 or draws.shape[0] not in (1, b) or draws.shape[2] != k:
                raise ValueError("lane_uniform must be [B or 1, T, K]")
            return lambda i: draws[:, i].expand(b, k)
        rows = 1 if crn else b
        if lane_seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(lane_seed))
            return lambda i: torch.rand((rows, k), generator=gen).to(device).expand(b, k)
        return lambda i: torch.rand((rows, k), device=device).expand(b, k)

    def _initial_lane_age(self, current: dict, present: Tensor, kin: Tensor,
                          driver_history: Optional[Tensor],
                          driver_history_mask: Optional[Tensor]) -> Optional[Tensor]:
        """Steps since each slot's last lane change (lane-head / hazard models only)."""
        if not self._tracks_lane_age:
            return None
        lane_age = current.get("lane_age")
        if lane_age is not None:
            return lane_age
        stride_steps = (round(self.history_stride_s / self.physics.dt_s)
                        if self.long_history else None)
        return initial_lane_age(present, kin.dtype, driver_history, driver_history_mask,
                                long_history=self.long_history, stride_steps=stride_steps)

    def _add_state_noise(self, out: dict, current: dict, std_m: float,
                         std_mps: float, state_noise: Optional[Tensor] = None) -> None:
        """Perturb the predicted position/speed of learned (non-CAV) movers."""
        present = current.get("present")
        if present is None:
            present = state_present(current, state_kin(current))
        cav = current.get("is_cav")
        learned = present.bool() if cav is None else (present.bool() & ~cav.bool())
        learned = learned.to(out["next_s_m"].dtype)
        if float(std_m) > 0.0:
            noise = (torch.randn_like(out["next_s_m"]) if state_noise is None
                     else state_noise[..., 0])
            out["next_s_m"] = out["next_s_m"] + noise * float(std_m) * learned
        if float(std_mps) > 0.0:
            noise = (torch.randn_like(out["next_v_mps"]) if state_noise is None
                     else state_noise[..., 1])
            out["next_v_mps"] = (out["next_v_mps"]
                                  + noise * float(std_mps) * learned).clamp_min(0.0)

    def _record_step(self, log: dict, out: dict, current: dict) -> None:
        log["positions"].append(out["next_s_m"])
        log["speeds"].append(out["next_v_mps"])
        log["lanes"].append(out["next_lane"])
        log["accel"].append(out["accel_mps2"])
        log["lane_logits"].append(out["lane_logits"])
        if "lane_v5_head_logit" in out:
            # V8WM (emit_merge_hazard): per-step own-head logit (with gradient) + acting rows.
            log.setdefault("lane_v5_head_logit", []).append(out["lane_v5_head_logit"])
            log.setdefault("lane_v5_acts", []).append(out["lane_v5_acts"])
        if "merge_threshold_location" in out:
            # R5THR: the first-passage loss integrates the threshold out at this location.
            log.setdefault("merge_threshold_location", []).append(
                out["merge_threshold_location"])
        if "merge_frailty_location" in out:
            # R5FR: the frailty survival loss integrates the latent at this location / scale.
            log.setdefault("merge_frailty_location", []).append(
                out["merge_frailty_location"])
            log.setdefault("merge_frailty_sigma", []).append(
                out["merge_frailty_sigma"])
            log.setdefault("merge_frailty_hazard", []).append(
                out["merge_frailty_hazard"])
            log.setdefault("merge_frailty_cross", []).append(
                out["merge_frailty_cross"])
        # Monitoring uses the raw gate * delta_a before any residual/plant bound.
        log["raw_residuals"].append(
            out.get("raw_residual_mps2", torch.zeros_like(out["accel_mps2"])))
        if getattr(self, "emit_physics_accel", False):
            log["physics_accels"].append(out["physics_accel_mps2"])
        probs = out.get("lane_probabilities")
        if probs is None:
            probs = torch.softmax(out["lane_logits"], dim=-1)
        stay = (probs[..., 0] if self.lane_change_head is not None else
                probs.gather(-1, current["lane"].clamp(0, 2).unsqueeze(-1)).squeeze(-1))
        log["change_probabilities"].append(1.0 - stay)
        if self.lane_change_head is not None:
            log["lane_probabilities"].append(out["lane_probabilities"])
        for key in ("merge_negotiation_yield_accel", "merge_negotiation_ramp_delta",
                    "merge_negotiation_required_brake", "merge_negotiation_conflict_logit",
                    "merge_negotiation_conflict_prob"):
            if key in out:
                log.setdefault(key, []).append(out[key])
        for key in _MERGE_CORE_LOG_KEYS:
            if key in out:
                log.setdefault(key, []).append(out[key])
        for key in _MERGE_PIVOT_LOG_KEYS:
            if key in out:
                log.setdefault(key, []).append(out[key])

    def _rollout_result(self, log: dict, current: dict, history: Tensor,
                        history_mask: Tensor, entrant: Tensor, pres0: Tensor, kin0: Tensor,
                        projection, projection_totals: dict, encoded_parameters,
                        driver_embedding, share: _ShareAccumulator) -> dict:
        batch, slots = pres0.shape
        positions, masks = log["positions"], log["masks"]
        entrant_seq = (torch.stack(log["entrant"], 1) if log["entrant"]
                       else torch.zeros(batch, 0, slots, dtype=torch.bool,
                                        device=pres0.device))
        sections = (torch.stack([section_ids(pos, msk) for pos, msk in zip(positions, masks)], 1)
                    if positions else torch.zeros(batch, 0, slots, dtype=torch.long,
                                                  device=pres0.device))
        result = {"positions": torch.stack(positions, 1),
                  "raw_residual_mps2": torch.stack(log["raw_residuals"], 1),
                  "lane_change_probabilities": torch.stack(log["change_probabilities"], 1),
                  **({"physics_accel_mps2": torch.stack(log["physics_accels"], 1)}
                     if log["physics_accels"] else {}),
                  "speeds": torch.stack(log["speeds"], 1),
                  "lanes": torch.stack(log["lanes"], 1),
                  "masks": torch.stack(masks, 1), "accel": torch.stack(log["accel"], 1),
                  "lane_logits": torch.stack(log["lane_logits"], 1),
                  "entrant": entrant, "entrant_seq": entrant_seq, "sections": sections,
                  "state": current, "history": history, "history_mask": history_mask}
        for key in ("merge_negotiation_yield_accel", "merge_negotiation_ramp_delta",
                    "merge_negotiation_required_brake", "merge_negotiation_conflict_logit",
                    "merge_negotiation_conflict_prob"):
            if log.get(key):
                result[key] = torch.stack(log[key], 1)
        for key in _MERGE_CORE_LOG_KEYS:
            if log.get(key):
                values = log[key]
                result[key] = torch.stack(values, 0 if values[0].ndim == 0 else 1)
        for key in _MERGE_PIVOT_LOG_KEYS:
            if log.get(key):
                values = log[key]
                result[key] = torch.stack(values, 0 if values[0].ndim == 0 else 1)
        if projection is not None:
            result["projection_stats"] = projection_totals
        if encoded_parameters is not None:
            result["driver_parameters"] = encoded_parameters
        if driver_embedding is not None:
            result["driver_embedding"] = driver_embedding
        if self.rollout_mode == "hybrid":
            zero = torch.zeros((), dtype=kin0.dtype, device=kin0.device)
            result["physics_accel_magnitude"] = (share.physics if share.physics is not None
                                                 else zero)
            result["residual_magnitude"] = (share.residual if share.residual is not None
                                            else zero)
            result["residual_count"] = int(share.count)
        if log["lane_probabilities"]:
            result["lane_probabilities"] = torch.stack(log["lane_probabilities"], 1)
        if log.get("lane_v5_head_logit"):
            result["lane_v5_head_logit"] = torch.stack(log["lane_v5_head_logit"], 1)
            result["lane_v5_acts"] = torch.stack(log["lane_v5_acts"], 1)
        if log.get("merge_threshold_location"):
            result["merge_threshold_location"] = torch.stack(
                log["merge_threshold_location"], 1)
        if log.get("merge_frailty_location"):
            result["merge_frailty_location"] = torch.stack(
                log["merge_frailty_location"], 1)
            result["merge_frailty_sigma"] = torch.stack(
                log["merge_frailty_sigma"], 1)
            result["merge_frailty_hazard"] = torch.stack(
                log["merge_frailty_hazard"], 1)
            result["merge_frailty_cross"] = torch.stack(
                log["merge_frailty_cross"], 1)
        return result
