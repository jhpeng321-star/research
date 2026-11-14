"""Strong black-box trajectory baseline (leave-controller-out), legacy checkpoints only.

A purely learned next-state model with no physics base: HDV acceleration and lane
logits come entirely from a neural network over the same neighbour kinematics and
rolling motion history as WM-S.  CAVs are teacher-forced exactly as in the structured
model.  The WM-S ``forward``/``step``/``rollout`` interface is inherited unchanged;
only the IDM+MOBIL base is replaced by a learned absolute predictor.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wms.model.history import HISTORY_F, HISTORY_H
from wms.model.residual_net import HdvConfig, KinBlock, slot_geometry
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.state import KIN_F, state_kin, state_present


class LearnedDynamics(nn.Module):
    """Neighbour-attention trunk with absolute accel/lane heads (no physics).

    Same input surface as the residual net (kinematics only; no CAV flag, action or
    label channel); the heads emit an absolute acceleration and absolute lane logits.
    """

    def __init__(self, cfg: Optional[HdvConfig] = None) -> None:
        super().__init__()
        c = cfg or HdvConfig()
        if c.use_cav_flag:
            raise ValueError(
                "the black-box trajectory baseline has no CAV-flag input "
                "(leakage-free kinematics only); use_cav_flag must be False")
        self.cfg = c
        self.norm = nn.LayerNorm(c.in_features)
        self.embed = nn.Sequential(
            nn.Linear(c.in_features, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, c.d_model))
        self.blocks = nn.ModuleList([KinBlock(c) for _ in range(c.n_layers)])
        self.accel = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 2))          # absolute mean accel + log-sigma
        self.lane = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 3))          # absolute 3-way lane logits

    def forward(self, kin: Tensor, kin_mask: Tensor) -> dict:
        if kin.dim() != 3 or kin.shape[-1] != KIN_F:
            raise ValueError(
                "black-box dynamics accepts kinematics only [B,K,%d]; got %s"
                % (KIN_F, tuple(kin.shape)))
        mask = kin_mask.bool()
        geo = slot_geometry(kin)
        x = self.embed(self.norm(kin)) * mask.unsqueeze(-1).to(kin.dtype)
        # The attention blocks are checkpointed while training: the K x K pairwise
        # bias MLP otherwise dominates activation memory (math unchanged, dropout 0).
        use_ckpt = self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                x, _ = checkpoint(block, x, mask, geo, use_reentrant=False)
            else:
                x, _ = block(x, mask, geo)
        acc = self.accel(x)
        return {"accel_mean_mps2": acc[..., 0],
                "accel_log_sigma": acc[..., 1].clamp(-6.0, 2.0),
                "lane_logits": self.lane(x) * mask.unsqueeze(-1).to(kin.dtype)}


class BlackBoxTrajModel(WmSModel):
    """WM-S interface with a learned HDV base instead of IDM/MOBIL.

    ``physics`` is retained only for the integration step, the acceleration clamp
    and the gap-history feature geometry.  ``action_conditioned=False`` rebuilds the
    pre-switch layout without controller/command input so historical checkpoints
    load and score identically.
    """

    def __init__(self, *, physics: Optional[PhysicsConfig] = None,
                 hdv: Optional[HdvConfig] = None,
                 history_features: int = HISTORY_F, history_len: int = HISTORY_H,
                 history_hidden: int = 64, k_hidden: int = 32,
                 action_conditioned: bool = True) -> None:
        spec = ModelSpec(history_features=history_features, history_len=history_len,
                         history_hidden=history_hidden, k_hidden=k_hidden,
                         black_box=bool(action_conditioned))
        super().__init__(spec, physics=physics, hdv=hdv)
        self.action_conditioned = bool(action_conditioned)
        # Keep the CAV teacher-forcing path while still using the controller/command
        # conditioned temporal input.
        self.black_box = False
        self.residual_net = LearnedDynamics(self.hdv)
        if self.action_conditioned:
            self.controller_embed = nn.Embedding(self.n_controllers, self.controller_embed_dim)
            self.command_proj = nn.Linear(2, self.controller_embed_dim, bias=False)

    def physics_accel(self, state: dict, driver_params=None) -> Tensor:
        raise NotImplementedError("black-box trajectory model has no physics base (no IDM)")

    def mobil_merge_logits(self, state: dict, driver_params=None) -> Tensor:
        raise NotImplementedError("black-box trajectory model has no physics base (no MOBIL)")

    def _command_embedding(self, controller_id, action_command, state, present, b, slots,
                           kin) -> Tensor:
        """Controller-id (+command) embedding on CAV slots; zeros without an id."""
        emb = torch.zeros((b, slots, self.controller_embed_dim), dtype=kin.dtype,
                          device=kin.device)
        if controller_id is None:
            return emb
        cid = torch.as_tensor(controller_id, device=kin.device).long()
        if cid.ndim == 0:
            cid = cid.expand(b, slots)
        elif cid.ndim == 1:
            cid = cid.reshape(b, 1).expand(b, slots)
        elif tuple(cid.shape) != (b, slots):
            raise ValueError("controller_id must be scalar, [B] or [B,K]")
        emb = self.controller_embed(cid.clamp(0, self.n_controllers - 1))
        if action_command is not None:
            command = torch.as_tensor(action_command, device=kin.device, dtype=kin.dtype)
            if tuple(command.shape) != (b, slots, 2):
                raise ValueError("action_command must be [B,K,2]")
            emb = emb + self.command_proj(command)
        is_cav = state.get("is_cav", torch.zeros_like(present)).bool()
        return emb * is_cav.unsqueeze(-1).to(kin.dtype)

    def forward(self, state: dict, *, k_steps: Tensor | int = 0,
                history: Optional[Tensor] = None,
                history_mask: Optional[Tensor] = None,
                driver_history: Optional[Tensor] = None,
                driver_history_mask: Optional[Tensor] = None,
                driver_defaults: Optional[Tensor] = None,
                driver_params: Optional[dict] = None,
                baseline_lane_logits: Optional[Tensor] = None,
                controller_id: Optional[Tensor] = None,
                action_command: Optional[Tensor] = None) -> dict:
        # No driver-parameter mode: the driver_* arguments keep the shared rollout
        # interface uniform and are deliberately ignored.
        kin = state_kin(state).float()
        present = state_present(state, kin)
        if kin.shape[-1] != KIN_F:
            raise ValueError(f"veh_kin must have {KIN_F} features")
        b, slots = kin.shape[:2]
        k = torch.as_tensor(k_steps, device=kin.device, dtype=torch.long)
        if k.ndim == 0:
            k = k.expand(b, slots)
        elif tuple(k.shape) != (b, slots):
            raise ValueError("k_steps must be scalar or [B,K]")
        base = self.residual_net(kin, present)
        controller_emb = (self._command_embedding(controller_id, action_command, state,
                                                  present, b, slots, kin)
                          if self.action_conditioned else None)
        temporal = self._temporal_delta(history, history_mask, k, (b, slots),
                                        kin.dtype, kin.device, controller_emb=controller_emb)
        gate = self.gate(k).to(kin.dtype)
        # At k=0 the whole prediction is the learned base (g(0)=0); the history/k MLP
        # correction grows with the gap since the last real observation.
        accel_mean = base["accel_mean_mps2"]
        accel = (accel_mean + gate * temporal[..., 0]).clamp(
            self.physics.accel_min_mps2, self.physics.accel_max_mps2)
        lane_base = (base["lane_logits"] if baseline_lane_logits is None
                     else baseline_lane_logits)
        lane_logits = lane_base + gate.unsqueeze(-1) * temporal[..., 1:]
        hdv = present & ~state.get("is_cav", torch.zeros_like(present)).bool()
        return {"learned_accel_mps2": accel_mean, "accel_mps2": accel,
                "accel_delta_mps2": accel - accel_mean, "lane_logits": lane_logits,
                "gate": gate, "residual_log_sigma": base["accel_log_sigma"],
                "hdv_mask": hdv}
