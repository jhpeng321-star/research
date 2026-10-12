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

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import MASKED_LOGIT, mobil_merge_logits
from wms.physics.neighbours import lane_neighbour, lane_of, lead_neighbour
from wms.physics.projection import FeasibilityProjection
from wms.physics.state import KIN_F, state_kin, state_present

from . import cav as cav_motion
from .controllers import DIRECT_ID
from .driver_encoder import DRIVER_ENCODER_HIDDEN, PARAMETER_NAMES, DriverEncoder
from .entries import section_ids
from .gate import monotone_gate, saturating_gate
from .history import (LONG_HISTORY_F, history_spec, history_step_features,
                      initial_history, initial_lane_age, push_history)
from .lane_head import HistoryLaneHead, lane_context
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
                                            long_history=self.long_history)
        self.lane_change_head = (HistoryLaneHead(self.driver_embed_dim)
                                 if spec.lane_head == "history" else None)

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
        embedding_needed = (self.rollout_mode == "hybrid" or self.long_history) \
            and driver_embedding is None
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
                action_command: Optional[Tensor] = None) -> dict[str, Tensor]:
        """Predict acceleration and lane logits for every slot.

        ``k_steps`` is the number of steps since the last real observation (scalar or
        ``[B, K]``).  Returns the physics and final accelerations, the residual terms,
        lane logits, the gate and the learned-slot mask (``hdv_mask``), plus driver
        parameters/embedding and lane probabilities when those components are active.
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
        accel = torch.where(learned, physics + correction, physics)
        if self.rollout_mode != "driver_params":
            accel = accel.clamp(self.physics.accel_min_mps2, self.physics.accel_max_mps2)

        lane_base = lane_base if baseline_lane_logits is None else baseline_lane_logits
        lane_delta = (torch.zeros_like(lane_base) if self.rollout_mode == "driver_params"
                      else base["lane_logit_delta"] + temporal[..., 1:])
        if self.long_history and self.rollout_mode == "driver_params":
            # Inferred style conditions the lane correction as well as IDM/MOBIL.
            lane_delta = base["lane_logit_delta"]
        lane_correction = bounded_value(
            self.lane_gate(k, shared=gate).to(lane_delta.dtype).unsqueeze(-1) * lane_delta,
            self.lane_logit_cap)
        lane_logits = lane_base + lane_correction
        if self.physics.lane_rule in ("forced_merge_only", "learned_discretionary"):
            # The learned residual lane delta can never unmask a forbidden change:
            # entries the physics masked stay masked after the correction and cap
            # (C4; in the default 'mobil' mode this branch is skipped, so outputs
            # are bit-identical to the pre-lane_rule model).
            lane_logits = torch.where(lane_base <= MASKED_LOGIT, lane_base, lane_logits)
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
                                             or self.long_history):
            result["driver_embedding"] = driver_embedding
        if lane_probabilities is not None:
            result["lane_probabilities"] = lane_probabilities
        return result

    # ------------------------------------------------------------------- step
    def hold_teacher(self, state: dict[str, Tensor]) -> Tensor:
        """Native hold-speed CAV motion ``[B, K, 4] = s, lane, v, a`` (no controller)."""
        return cav_motion.hold_speed_teacher(state["s_m"], state["lane"], state["v_mps"],
                                             self.physics.dt_s)

    @property
    def _uses_driver_embedding(self) -> bool:
        return (self.rollout_mode == "hybrid" or self.lane_change_head is not None
                or self.long_history)

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
             projection: Optional[FeasibilityProjection] = None) -> dict[str, Tensor]:
        """Advance one 0.5 s step; CAVs are never learned movers in structured mode.

        With entries enabled the state carries an ``entrant`` mask: t0 CAVs keep the
        recorded teacher, while entrant CAVs (which have no recorded plan) run the
        supplied ``cav_controller`` or native hold-speed motion.  ``projection``
        (optional) projects the next state onto the feasible set.
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
        if self.lane_change_head is not None:
            forward_kwargs["lane_age"] = lane_age
        out = self.forward(state, **forward_kwargs)

        s, v = state["s_m"], state["v_mps"]
        present = state_present(state, state_kin(state))
        cav = state.get("is_cav", torch.zeros_like(present)).bool() & present
        dt = self.physics.dt_s
        next_v = (v + out["accel_mps2"] * dt).clamp_min(0.0)
        next_s = s + 0.5 * (v + next_v) * dt
        choice = out["lane_logits"].argmax(-1)
        if self.lane_change_head is None:
            next_lane = choice
        else:
            lane = state["lane"].long()
            next_lane = lane + (choice == 2).long() - (choice == 1).long()

        direct = (torch.zeros_like(cav)
                  if action_id is None or not self.apply_direct_commands else
                  (action_id == DIRECT_ID) & cav)
        if direct.any():
            direct, next_s, next_v = self._apply_direct_commands(
                out, direct, s, v, next_s, next_v, action_command)
        if cav.any() and not self.black_box:
            next_s, next_v, next_lane = self._apply_cav_overrides(
                out, state, cav, direct, present, k_steps, action_id, action_command,
                cav_teacher, cav_controller, next_s, next_v, next_lane)

        next_present = present.clone()
        exit_s = state.get("exit_s_m")
        if exit_s is not None:
            next_present = next_present & (next_s <= exit_s)
        if projection is not None:
            length = state.get("length_m", torch.full_like(s, 5.0))
            proj = projection(next_s, next_v, next_lane, next_present, length,
                              is_cav=state.get("is_cav"))
            next_s, next_v, next_lane = proj["s_m"], proj["v_mps"], proj["lane"]
            out["projection_active"] = proj["active"]
        out.update({"next_s_m": next_s, "next_v_mps": next_v,
                    "next_lane": next_lane, "next_present": next_present})
        return out

    def _apply_direct_commands(self, out: dict, direct: Tensor, s: Tensor, v: Tensor,
                               next_s: Tensor, next_v: Tensor,
                               action_command: Optional[Tensor]
                               ) -> tuple[Tensor, Tensor, Tensor]:
        """Drive ``direct`` CAVs from their issued command.

        Updates ``out["accel_mps2"]`` and returns ``(applied mask, next_s, next_v)``.
        """
        if action_command is None:
            raise ValueError("direct timeline action requires acceleration and target speed")
        command = torch.as_tensor(action_command, dtype=v.dtype, device=v.device)
        direct_s, direct_v, accel, feasible = cav_motion.direct_command_motion(
            s, v, command, float(self.physics.dt_s), self.physics.accel_min_mps2,
            self.physics.accel_max_mps2)
        applied = direct & feasible
        out["accel_mps2"] = torch.where(applied, accel, out["accel_mps2"])
        return applied, torch.where(applied, direct_s, next_s), torch.where(applied, direct_v, next_v)

    def _apply_cav_overrides(self, out: dict, state: dict, cav: Tensor, direct: Tensor,
                             present: Tensor, k_steps, action_id, action_command,
                             cav_teacher, cav_controller, next_s, next_v, next_lane):
        """Replace the learned motion of CAV slots by teacher/controller motion.

        The controller callback runs once per physics tick (also on direct-only ticks)
        because the dispatcher owns episode-persistent controller lifecycle state.
        """
        callback_teacher = None
        if cav_controller is not None:
            callback_state = {**state, "physics_step_index": k_steps}
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
                projection: Optional[FeasibilityProjection] = None) -> dict[str, Tensor]:
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
        zero draws nothing.

        ``lane_teacher`` (``[B, T+1, K]`` recorded lanes; diagnostic, round 5) forces
        every step's lane decision to the recorded lane where the entry is ``>= 0``
        (post-step, so CAV teacher overrides stay in charge of CAV slots).

        Beyond the per-step ``positions``/``speeds``/``lanes``/``masks``/``accel``/
        ``lane_logits`` the result carries ``entrant_seq [B, T, K]`` (per-step entrant
        membership) and ``sections [B, T, K]`` (model-frame section id, -1 if absent).
        """
        current = dict(state)
        if controller_id is None:
            controller_id = current.get("controller_id")
        current.pop("controller_id", None)
        kin0 = state_kin(current)
        pres0 = state_present(current, kin0)
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
                action_command=action_command, projection=projection)
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
            if self.rollout_mode == "hybrid":
                share.add(out)
            nxt_present = out["next_present"]
            if present_seq is not None:
                nxt_present = nxt_present & present_seq[:, step_i + 1]
            if junction_seq is not None:
                # A junction transit is not a departure: carry the slot through it.
                nxt_present = nxt_present | junction_seq[:, step_i + 1].bool()
            if noise_on:
                self._add_state_noise(out, current, state_noise_std_m, state_noise_std_mps)
            self._record_step(log, out, current)
            log["masks"].append(nxt_present)
            if self.lane_change_head is not None:
                changed = out["next_lane"] != current["lane"]
                lane_age = torch.where(changed, torch.zeros_like(lane_age), lane_age + 1)
            current = {**current, "s_m": out["next_s_m"], "v_mps": out["next_v_mps"],
                       "a_mps2": out["accel_mps2"], "lane": out["next_lane"],
                       "present": nxt_present}
            if self.lane_change_head is not None:
                current["lane_age"] = lane_age
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

        return self._rollout_result(
            log, current, history, history_mask, entrant, pres0, kin0, projection,
            projection_totals, encoded_parameters, fixed_driver_embedding, share)

    def _initial_lane_age(self, current: dict, present: Tensor, kin: Tensor,
                          driver_history: Optional[Tensor],
                          driver_history_mask: Optional[Tensor]) -> Optional[Tensor]:
        """Steps since each slot's last lane change (lane-head models only)."""
        if self.lane_change_head is None:
            return None
        lane_age = current.get("lane_age")
        if lane_age is not None:
            return lane_age
        stride_steps = (round(self.history_stride_s / self.physics.dt_s)
                        if self.long_history else None)
        return initial_lane_age(present, kin.dtype, driver_history, driver_history_mask,
                                long_history=self.long_history, stride_steps=stride_steps)

    def _add_state_noise(self, out: dict, current: dict, std_m: float,
                         std_mps: float) -> None:
        """Perturb the predicted position/speed of learned (non-CAV) movers."""
        present = current.get("present")
        if present is None:
            present = state_present(current, state_kin(current))
        cav = current.get("is_cav")
        learned = present.bool() if cav is None else (present.bool() & ~cav.bool())
        learned = learned.to(out["next_s_m"].dtype)
        if float(std_m) > 0.0:
            out["next_s_m"] = out["next_s_m"] + torch.randn_like(
                out["next_s_m"]) * float(std_m) * learned
        if float(std_mps) > 0.0:
            out["next_v_mps"] = (out["next_v_mps"] + torch.randn_like(
                out["next_v_mps"]) * float(std_mps) * learned).clamp_min(0.0)

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

