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
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.lane_conflict import resolve_lane_conflicts
from wms.physics.mobil import MASKED_LOGIT, merge_end_blocked, mobil_merge_logits
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
from .residual_bound import bounded_accel_correction, bounded_value
from .residual_net import HdvConfig, HdvResidualNet
from .spec import ModelSpec

#: Output width of the temporal MLP: one acceleration delta plus three lane logits.
_TEMPORAL_OUT = 4


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

    def __init__(self, spec: Optional[ModelSpec] = None, *,
                 physics: Optional[PhysicsConfig] = None, hdv: Optional[HdvConfig] = None,
                 physics_step: Optional[Callable] = None) -> None:
        super().__init__()
        spec = spec or ModelSpec()
        self.spec = spec
        self.physics = physics or PhysicsConfig()
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
        # Runtime execution-cap switch (EXECCAP_NOTES.md, never part of a checkpoint): "off"
        # (default, bit-identical) or "idm" -- every COMMAND-driven CAV realises at most the
        # SUMO-consistent ceiling min(own physics-core IDM accel, (v0-v)/dt, accel param).
        self.cav_exec_cap = "off"
        self._build_modules(spec)
        if self.physics.lane_rule == "learned_discretionary":
            self._zero_lane_outputs()

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
            self.lane_v5 = lv5.LaneV5Net(spec.lane_hazard_hidden, spec.lane_hazard_cap)

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

    def gate(self, k_steps: Tensor | int) -> Tensor:
        """Learned monotone gate with exactly ``g(0) == 0``."""
        if self.gate_kind == "saturating":
            return saturating_gate(self.gate_g_max_logit, self.gate_tau_raw, k_steps,
                                   self.physics.max_gap_steps)
        return monotone_gate(self.gate_increments, k_steps, self.physics.max_gap_steps)

    def lane_gate(self, k_steps: Tensor | int, shared: Optional[Tensor] = None) -> Tensor:
        """Gate on the learned lane-logit correction (own saturating gate when configured).

        With ``lane_gate_kind == "shared"`` (default) the lane correction uses the
        acceleration gate unchanged — the returned tensor is the provided ``shared``
        value (or a fresh ``self.gate`` evaluation), so default models are bit-identical.
        """
        if self.lane_gate_kind != "saturating":
            return shared if shared is not None else self.gate(k_steps)
        return saturating_gate(self.lane_gate_g_max_logit, self.lane_gate_tau_raw, k_steps,
                               self.physics.max_gap_steps)

    def physics_accel(self, state: dict[str, Tensor],
                      driver_params: Optional[dict[str, Tensor]] = None) -> Tensor:
        """Physics-baseline acceleration ``[B, K]``."""
        kin = state_kin(state)
        return baseline_accel(kin, state_present(state, kin), self.physics, driver_params)

    def mobil_merge_logits(self, state: dict[str, Tensor],
                           driver_params: Optional[dict[str, Tensor]] = None) -> Tensor:
        """MOBIL plus mandatory-merge baseline lane logits ``[B, K, 3]``."""
        kin = state_kin(state)
        return mobil_merge_logits(kin, state_present(state, kin), self.physics, driver_params)

    def _physics_step(self, state: dict[str, Tensor],
                      driver_params: Optional[dict[str, Tensor]] = None
                      ) -> tuple[Tensor, Tensor]:
        """Baseline ``(accel, lane logits)`` from the injected backend or in-module physics."""
        if self.physics_step is None:
            if self.physics.merge_end_wait or self.physics.mainline_merge_yield:
                # V6MERGE: one blocked-merger mask per step, shared by accel and lane logits.
                kin = state_kin(state)
                present = state_present(state, kin)
                blocked = merge_end_blocked(kin, present, self.physics)
                return (baseline_accel(kin, present, self.physics, driver_params,
                                       merge_blocked=blocked),
                        mobil_merge_logits(kin, present, self.physics, driver_params,
                                           merge_blocked=blocked))
            return (self.physics_accel(state, driver_params),
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
                             conditioning: Optional[Tensor]) -> dict[str, Tensor]:
        if self.checkpoint_residual and self.training and conditioning is None:
            return checkpoint(self.residual_net, kin, present,
                              use_reentrant=False, preserve_rng_state=False)
        return self.residual_net(kin, present, conditioning=conditioning)

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
        k = torch.as_tensor(k_steps, device=kin.device, dtype=torch.long)
        if k.ndim == 0:
            k = k.expand(b, slots)
        elif tuple(k.shape) != (b, slots):
            raise ValueError("k_steps must be scalar or [B,K]")

        encoded = None
        if self.rollout_mode in ("driver_params", "hybrid") or self.long_history:
            encoded, driver_embedding, driver_params = self._encode_drivers(
                kin, present, driver_history, driver_history_mask, history, history_mask,
                driver_defaults, driver_params, driver_embedding)
            physics, lane_base = self._physics_step({**state, "veh_kin": kin}, driver_params)
        else:
            physics, lane_base = self._physics_step({**state, "veh_kin": kin})

        conditioning = (driver_embedding
                        if self.rollout_mode == "hybrid" or self.long_history else None)
        base = self._residual_prediction(kin, present, conditioning)
        controller_emb = self._controller_embedding(
            controller_id, is_cav, (b, slots), kin.dtype, kin.device,
            action_command=action_command)
        temporal = self._temporal_delta(history, history_mask, k, (b, slots),
                                        kin.dtype, kin.device, controller_emb=controller_emb)
        gate = self.gate(k).to(kin.dtype)

        accel_delta = (torch.zeros_like(physics) if self.rollout_mode == "driver_params"
                       else base["accel_delta_mps2"] + temporal[..., 0])
        raw_residual = gate * accel_delta
        correction = bounded_accel_correction(
            gate * accel_delta, soft_bound_pos=self.residual_soft_bound_mps2,
            soft_bound_neg=self.residual_soft_bound_neg_mps2,
            cap_pos=self.residual_cap_mps2, cap_neg=self.residual_cap_neg_mps2)
        if accel_noise is None:
            accel = torch.where(learned, physics + correction, physics)
        else:
            sigma = base["accel_log_sigma"].exp()
            if accel_noise_sigma_max is not None:
                sigma = sigma.clamp_max(float(accel_noise_sigma_max))
            accel = torch.where(learned, physics + correction + accel_noise.to(sigma.dtype) * sigma,
                                physics)
        if self.rollout_mode != "driver_params":
            accel = accel.clamp(self.physics.accel_min_mps2, self.physics.accel_max_mps2)

        lane_base = lane_base if baseline_lane_logits is None else baseline_lane_logits
        blocked_now = v5_acc = v5_sample = None
        if getattr(self, "lane_v5", None) is not None:
            lane_logits, v5_acc, v5_sample = self._lane_v5_forward(
                kin, present, learned, lane_base, lane_age, lane_v5_acc, driver_history,
                driver_history_mask)
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
                self.lane_hazard, kin, present, learned, lane_base, self.physics,
                driver_params, driver_embedding, lane_age,
                **self._hazard_switches(blocked_now, rollout=True))
        else:
            lane_logits = self._residual_lane_logits(lane_base, base, temporal, k, gate)
        lane_probabilities = None
        if self.lane_change_head is not None:
            if driver_embedding is None:
                raise ValueError("history lane head requires its pre-t0 encoder embedding")
            context, feasible_actions = lane_context(
                kin, present, lane_base, self.physics, driver_params, lane_age)
            lane_logits = self.lane_change_head(driver_embedding, context, feasible_actions)
            lane_probabilities = torch.softmax(lane_logits, dim=-1)

        result = {"physics_accel_mps2": physics, "accel_mps2": accel,
                  "raw_residual_mps2": raw_residual,
                  "accel_delta_mps2": accel - physics, "lane_logits": lane_logits,
                  "gate": gate, "residual_log_sigma": base["accel_log_sigma"],
                  "hdv_mask": learned}
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
            return getattr(self, "lane_decision", "argmax") in ("sample", "gated")
        return (getattr(self, "lane_hazard", None) is not None
                and getattr(self, "lane_decision", "argmax") == "sample")

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
                         driver_history_mask: Optional[Tensor]):
        """Lane logits, updated accumulators and the drawing slots of the lane-v5 model."""
        if acc is None:
            acc = self.initial_lane_v5_acc(present, kin, driver_history, driver_history_mask)
        raw, acc, inst = lv5.step_features(acc, kin, present, self.physics, lane_age)
        offsets = tuple(getattr(self, "lane_logit_offsets", (0.0, 0.0, 0.0)))
        with torch.no_grad() if self.lane_hazard_detach else contextlib.nullcontext():
            logits, _, _ = lv5.lane_logits_v5(self.lane_v5, raw, inst, learned, lane_base,
                                              self.physics, offsets)
        sample = None
        if self._samples_lanes:
            if self.lane_decision == "gated":
                sample = lv5.sample_mask(inst["lane"], learned, lv5.gate_open(
                    self.lane_v5_head_ap_effective(), self.lane_v5_gates))
            else:
                sample = learned
        return logits, acc, sample

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
            raw, _, inst = lv5.step_features(acc, kin, present, self.physics, lane_age)
            with torch.no_grad():
                lane_base = self.lane_v5_base(kin, present)
            logits, _, _ = lv5.lane_logits_v5(self.lane_v5, raw, inst, present & ~is_cav,
                                              lane_base, self.physics,
                                              tuple(getattr(self, "lane_logit_offsets",
                                                            (0.0, 0.0, 0.0))))
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
        forward_kwargs = dict(
            k_steps=k_steps, history=history, history_mask=history_mask,
            driver_history=driver_history, driver_history_mask=driver_history_mask,
            driver_defaults=driver_defaults, driver_params=driver_params,
            baseline_lane_logits=baseline_lane_logits,
            controller_id=(controller_id if action_id is None else action_id),
            action_command=action_command)
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

        s, v = state["s_m"], state["v_mps"]
        present = state_present(state, state_kin(state))
        cav = state.get("is_cav", torch.zeros_like(present)).bool() & present
        dt = self.physics.dt_s
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
                                        out.get("lane_sample", out["hdv_mask"]), lane_uniform)
        else:
            choice = out["lane_logits"].argmax(-1)
        if self.lane_change_head is None:
            next_lane = choice
        else:
            lane = state["lane"].long()
            next_lane = lane + (choice == 2).long() - (choice == 1).long()

        direct = (torch.zeros_like(cav)
                  if action_id is None or not self.apply_direct_commands else
                  (action_id == DIRECT_ID) & cav)
        exec_cap = (self._exec_cap(out, state, driver_params)
                    if cav.any() and getattr(self, "cav_exec_cap", "off") != "off" else None)
        if direct.any():
            direct, next_s, next_v = self._apply_direct_commands(
                out, direct, s, v, next_s, next_v, action_command, exec_cap=exec_cap)
        if cav.any() and not self.black_box:
            next_s, next_v, next_lane = self._apply_cav_overrides(
                out, state, cav, direct, present, k_steps, action_id, action_command,
                cav_teacher, cav_controller, next_s, next_v, next_lane, exec_cap=exec_cap)

        if self.physics.merge_end_wait:
            # V6MERGE: a vehicle staying on the accel lane never passes its end; it stops there.
            end = self.physics.merge_zone_s1_m
            overrun = (present & (next_lane == self.physics.merge_lane) & (s <= end)
                       & (next_s > end))
            next_s = torch.where(overrun, torch.full_like(next_s, end), next_s)
            next_v = torch.where(overrun, torch.zeros_like(next_v), next_v)
        next_present = present.clone()
        exit_s = state.get("exit_s_m")
        if exit_s is not None:
            next_present = next_present & (next_s <= exit_s)
        if self.lane_conflict_resolve:
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

    def _exec_cap(self, out: dict, state: dict,
                  driver_params: Optional[dict[str, Tensor]]) -> Tensor:
        """SUMO-consistent ceiling ``[B, K]`` of commanded CAVs (EXECCAP_NOTES.md).

        ``out["physics_accel_mps2"]`` is the physics-core IDM acceleration of every present
        slot (CAV slots included) with the driver settings of this step; the settings are
        the supplied ``driver_params``, else those encoded in this forward, else defaults.
        """
        mode = cav_motion.check_exec_cap_mode(self.cav_exec_cap)
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
                             exec_cap: Optional[Tensor] = None):
        """Replace the learned motion of CAV slots by teacher/controller motion.

        The controller callback runs once per physics tick (also on direct-only ticks)
        because the dispatcher owns episode-persistent controller lifecycle state.
        With ``exec_cap`` the callback state carries it under ``cav.EXEC_CAP_KEY``; command
        executors apply it to their commanded slots (``cav.apply_exec_cap``), teachers ignore it.
        """
        callback_teacher = None
        if cav_controller is not None:
            callback_state = {**state, "physics_step_index": k_steps}
            if exec_cap is not None:
                callback_state[cav_motion.EXEC_CAP_KEY] = exec_cap
            if action_id is not None:
                callback_state = {**callback_state, "action_id": action_id,
                                  "action_command": action_command}
            callback_teacher = cav_controller(callback_state)
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
                state_noise: Optional[Tensor] = None) -> dict[str, Tensor]:
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
        lane_draw = (self._lane_draws(lane_uniform, lane_seed, lane_crn, pres0)
                     if self._samples_lanes else None)
        timeline_dt = cav_motion.timeline_period(action_timeline)

        log = {name: [] for name in (
            "positions", "speeds", "lanes", "masks", "accel", "lane_logits",
            "raw_residuals", "change_probabilities", "lane_probabilities", "entrant")}
        projection_totals: dict[str, float] = {}
        share = _ShareAccumulator()
        # Per-step activation checkpointing (round 5): only in training with gradients
        # and never with a stateful CAV controller, whose callback must not be replayed
        # a second time during recomputation.
        step_checkpoint = (self.checkpoint_step and self.training
                           and torch.is_grad_enabled() and cav_controller is None)
        fixed_driver_params = driver_params
        fixed_driver_embedding = driver_embedding
        initial_driver_embedding = driver_embedding
        encoded_parameters = None
        noise_on = float(state_noise_std_m) > 0.0 or float(state_noise_std_mps) > 0.0

        for step_i in range(int(n_steps)):
            if entries is not None:
                current = {key: (value.clone() if torch.is_tensor(value) else value)
                           for key, value in current.items()}
                entrant = entrant | entries.insert(current)
                current["entrant"] = entrant
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
            if lane_teacher is not None:
                recorded_lane = lane_teacher[:, step_i + 1].long()
                out["next_lane"] = torch.where(recorded_lane >= 0, recorded_lane,
                                               out["next_lane"])
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
            current.pop("veh_kin", None)
            kin = state_kin(current)
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
        # Monitoring uses the raw gate * delta_a before any residual/plant bound.
        log["raw_residuals"].append(
            out.get("raw_residual_mps2", torch.zeros_like(out["accel_mps2"])))
        probs = out.get("lane_probabilities")
        if probs is None:
            probs = torch.softmax(out["lane_logits"], dim=-1)
        stay = (probs[..., 0] if self.lane_change_head is not None else
                probs.gather(-1, current["lane"].clamp(0, 2).unsqueeze(-1)).squeeze(-1))
        log["change_probabilities"].append(1.0 - stay)
        if self.lane_change_head is not None:
            log["lane_probabilities"].append(out["lane_probabilities"])

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
                  "speeds": torch.stack(log["speeds"], 1),
                  "lanes": torch.stack(log["lanes"], 1),
                  "masks": torch.stack(masks, 1), "accel": torch.stack(log["accel"], 1),
                  "lane_logits": torch.stack(log["lane_logits"], 1),
                  "entrant": entrant, "entrant_seq": entrant_seq, "sections": sections,
                  "state": current, "history": history, "history_mask": history_mask}
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
        return result

