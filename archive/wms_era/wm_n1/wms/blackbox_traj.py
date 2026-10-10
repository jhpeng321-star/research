"""WM-N1 strong black-box trajectory baseline (C1 leave-controller-out).

A purely learned next-state model with **no physics base**: HDV acceleration
and lane logits come entirely from a neural network over the same neighbour
kinematics and rolling motion history as WM-S.  CAVs are teacher-forced from
the recorded controller trajectory, exactly as in the structured model, so the
controller's own motion is known; only the HDV mechanism is learned.  This
isolates the value of WM-S's physics-structured HDV base on controllers the
model has never seen.

The model subcontracts the WM-S interface (``forward`` / ``step`` / ``rollout``
/ ``gate``) but replaces the IDM+MOBIL base with a learned absolute predictor.
The rollout contract -- CAV teacher forcing, exogenous presence/exit masks,
rolling history buffer, chunked-BPTT compatibility -- is inherited unchanged so
the same trainer, losses, curriculum and ``check400`` gate apply verbatim.

Selected by ``python -m wms.cli ... --blackbox-traj`` (default off); WM-S
default behaviour is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

_WM_ROOT = Path(__file__).resolve().parents[1]
if str(_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(_WM_ROOT))

from hdv.config import KIN_F, HdvConfig  # noqa: E402
from hdv.net import _KinBlock  # noqa: E402

from .model import (  # noqa: E402
    HISTORY_F, HISTORY_H, PhysicsConfig, WmSModel, _present, _state_kin)


class _LearnedDynamics(nn.Module):
    """Neighbour-attention trunk with ABSOLUTE accel/lane heads (no physics).

    Same input surface and attention mechanism as :class:`hdv.net.HdvResidualNet`
    (kinematics only, no CAV flag / action / label channel); the difference is
    that the heads emit an absolute acceleration and absolute lane logits
    instead of a correction on top of an IDM/MOBIL base.
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
        self.blocks = nn.ModuleList([_KinBlock(c) for _ in range(c.n_layers)])
        self.accel = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 2))          # absolute mean accel + log-sigma
        self.lane = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 3))          # absolute 3-way lane logits

    @staticmethod
    def _geometry(kin: Tensor) -> dict:
        onehot = kin[..., 1:4]
        lane = onehot.argmax(dim=-1)
        empty = onehot.sum(dim=-1) <= 0
        lane = torch.where(empty, torch.full_like(lane, -1), lane)
        return dict(s=kin[..., 0], lane=lane, v=kin[..., 4])

    def forward(self, kin: Tensor, kin_mask: Tensor) -> dict:
        if kin.dim() != 3 or kin.shape[-1] != KIN_F:
            raise ValueError(
                "black-box dynamics accepts kinematics only [B,K,%d]; got %s"
                % (KIN_F, tuple(kin.shape)))
        mask = kin_mask.bool()
        geo = self._geometry(kin)
        x = self.embed(self.norm(kin)) * mask.unsqueeze(-1).to(kin.dtype)
        # Checkpoint the attention blocks: the K x K pairwise-geometry bias MLP
        # ([B,K,K,ffn]) otherwise dominates activation memory across the
        # truncated-BPTT window.  Recomputing it in backward keeps the process
        # comfortably under the 6 GB rule; the math is untouched (dropout=0),
        # so this is numerically identical to the uncheckpointed forward.
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

    ``physics`` is retained only for the integration step (``dt_s``), the
    acceleration clamp and the neighbour-gap history feature geometry; no IDM or
    MOBIL computation ever occurs.  ``physics_accel`` and ``mobil_merge_logits``
    raise :class:`NotImplementedError` to make that explicit.
    """

    def __init__(self, *, physics: Optional[PhysicsConfig] = None,
                 hdv: Optional[HdvConfig] = None,
                 history_features: int = HISTORY_F, history_len: int = HISTORY_H,
                 history_hidden: int = 64, k_hidden: int = 32,
                 physics_step=None, action_conditioned: bool = True) -> None:
        if physics_step is not None:
            raise ValueError(
                "the black-box trajectory baseline has no pluggable physics "
                "base; physics_step must be None")
        # ``action_conditioned=False`` rebuilds the pre-switch (C1) layout with
        # no controller/command input, so historical checkpoints load and score
        # byte-identically.
        self.action_conditioned = bool(action_conditioned)
        super().__init__(physics=physics, hdv=hdv,
                         history_features=history_features,
                         history_len=history_len,
                         history_hidden=history_hidden, k_hidden=k_hidden,
                         black_box=self.action_conditioned)
        # Retain the historical CAV teacher-forcing path while using the
        # controller/command-conditioned temporal input for fair action labels.
        self.black_box = False
        # Replace the residual head with the learned absolute predictor under the
        # same attribute name; forward() below reinterprets its outputs.
        self.residual_net = _LearnedDynamics(self.hdv)
        if self.action_conditioned:
            self.controller_embed = nn.Embedding(self.n_controllers,
                                                 self.controller_embed_dim)
            self.command_proj = nn.Linear(2, self.controller_embed_dim, bias=False)

    def physics_accel(self, state: dict) -> Tensor:  # pragma: no cover - guard
        raise NotImplementedError(
            "black-box trajectory model has no physics base (no IDM)")

    def mobil_merge_logits(self, state: dict) -> Tensor:  # pragma: no cover
        raise NotImplementedError(
            "black-box trajectory model has no physics base (no MOBIL)")

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
        # The black-box trajectory baseline has no driver-parameter mode; the
        # driver_* arguments are accepted so the shared rollout/step interface
        # is uniform and are deliberately ignored.
        kin = _state_kin(state).float()
        present = _present(state, kin)
        if kin.shape[-1] != KIN_F:
            raise ValueError(f"veh_kin must have {KIN_F} features")
        b, slots = kin.shape[:2]
        k = torch.as_tensor(k_steps, device=kin.device, dtype=torch.long)
        if k.ndim == 0:
            k = k.expand(b, slots)
        elif tuple(k.shape) != (b, slots):
            raise ValueError("k_steps must be scalar or [B,K]")
        base = self.residual_net(kin, present)
        controller_emb = (torch.zeros((b, slots, self.controller_embed_dim),
                                      dtype=kin.dtype, device=kin.device)
                          if self.action_conditioned else None)
        if controller_id is not None and self.action_conditioned:
            cid = torch.as_tensor(controller_id, device=kin.device).long()
            if cid.ndim == 0:
                cid = cid.expand(b, slots)
            elif cid.ndim == 1:
                cid = cid.reshape(b, 1).expand(b, slots)
            elif tuple(cid.shape) != (b, slots):
                raise ValueError("controller_id must be scalar, [B] or [B,K]")
            controller_emb = self.controller_embed(cid.clamp(0, self.n_controllers - 1))
            if action_command is not None:
                command = torch.as_tensor(action_command, device=kin.device, dtype=kin.dtype)
                if tuple(command.shape) != (b, slots, 2):
                    raise ValueError("action_command must be [B,K,2]")
                controller_emb = controller_emb + self.command_proj(command)
            controller_emb = controller_emb * state.get(
                "is_cav", torch.zeros_like(present)).bool().unsqueeze(-1).to(kin.dtype)
        temporal = self._temporal_delta(history, history_mask, k, (b, slots),
                                        kin.dtype, kin.device,
                                        controller_emb=controller_emb)
        gate = self.gate(k).to(kin.dtype)
        # At k=0 the whole prediction is the learned neighbour base (g(0)=0);
        # the history/k MLP correction grows with the gap since the last real
        # observation, exactly as the WM-S gate does.  No physics term.
        accel_mean = base["accel_mean_mps2"]
        accel = accel_mean + gate * temporal[..., 0]
        accel = accel.clamp(self.physics.accel_min_mps2,
                            self.physics.accel_max_mps2)
        lane_base = (base["lane_logits"] if baseline_lane_logits is None
                     else baseline_lane_logits)
        lane_logits = lane_base + gate.unsqueeze(-1) * temporal[..., 1:]
        hdv = present & ~state.get("is_cav", torch.zeros_like(present)).bool()
        return {"learned_accel_mps2": accel_mean, "accel_mps2": accel,
                "accel_delta_mps2": accel - accel_mean, "lane_logits": lane_logits,
                "gate": gate, "residual_log_sigma": base["accel_log_sigma"],
                "hdv_mask": hdv}


def build(checkpoint: str, flags=None, physics=None):
    """Return the C1 evaluation adapter for a black-box-traj checkpoint.

    The adapter reuses :class:`wms_eval.wms_plugin.WmsCheckpointModel` in
    ``blackbox_traj`` mode: CAVs are teacher-forced from the recorded arm exactly
    as in the structured model, so the only difference under test is the learned
    (physics-free) HDV base.
    """
    from wms_eval.wms_plugin import WmsCheckpointModel

    return WmsCheckpointModel(checkpoint, flags, physics=physics,
                              model_kind="blackbox_traj")


__all__ = ["BlackBoxTrajModel", "build"]
