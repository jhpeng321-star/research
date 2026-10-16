"""Neighbour-attention residual network for HDV acceleration and lane logits.

The network consumes neighbour *kinematics only* (``kin [B, K, 7]`` and the slot
mask).  The headline model never receives a CAV flag, action one-hot, strategy
name or outcome label; :func:`input_schema` declares that contract.  The slot
count ``K`` is carried by the input tensors, the network has no K-dependent
weights.

Switches:

* ``HdvConfig.zero_residual`` returns an identically zero residual;
* ``HdvConfig.use_cav_flag`` (ablation only) appends one extra input channel;
* ``HdvConfig.conditioning_dim > 0`` adds a zero-initialised adapter for a
  per-slot driver embedding, built last so the baseline layers keep their
  initialisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from wms.physics.state import KIN_F

#: Kinematics-only input fields, in column order of ``veh_kin``.
KIN_FIELDS = (
    "s_down_m", "lane_main_left", "lane_main_right", "lane_accel",
    "v_mps", "a_mps2", "length_m")

#: Inputs the headline network may consume.
_ALLOWED_INPUTS = ("kin", "kin_mask")
_ABLATION_INPUTS = ("cav_flag",)
#: Channels that must never reach the headline network.
_FORBIDDEN_INPUTS = (
    "action", "action_code", "action_onehot", "actions",
    "strategy", "strategy_name", "is_cav", "cav_flag",
    "hdv_base", "cav_plan_ch", "outcome", "outcomes", "label", "labels",
)


@dataclass
class HdvConfig:
    """Residual attention network shape and output scales."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ffn: int = 512
    radius_m: float = 150.0
    dropout: float = 0.0
    use_cav_flag: bool = False          # ablation only
    accel_delta_scale_mps2: float = 1.5
    lane_logit_scale: float = 1.0
    zero_residual: bool = False         # no-op control: reproduce the baseline exactly
    conditioning_dim: int = 0           # width of the optional driver embedding

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if int(self.conditioning_dim) < 0:
            raise ValueError("conditioning_dim must be non-negative")

    @property
    def in_features(self) -> int:
        return KIN_F + (1 if self.use_cav_flag else 0)


def input_schema(cfg: Optional[HdvConfig] = None) -> dict:
    """Exact input surface of the residual network (label-leak contract)."""
    cfg = cfg or HdvConfig()
    optional = tuple(_ABLATION_INPUTS) if cfg.use_cav_flag else ()
    forbidden = tuple(
        f for f in _FORBIDDEN_INPUTS if not (cfg.use_cav_flag and f == "cav_flag"))
    return {
        "module": "hdv.net.HdvResidualNet",
        "kin_fields": tuple(KIN_FIELDS),
        "kin_f": int(KIN_F),
        "in_features": int(cfg.in_features),
        "allowed_inputs": _ALLOWED_INPUTS,
        "optional_inputs": optional,
        "forbidden_inputs": forbidden,
        "use_cav_flag": bool(cfg.use_cav_flag),
        "zero_residual": bool(cfg.zero_residual),
        "conditioning_dim": int(cfg.conditioning_dim),
        "identity_outputs": ("accel_delta_mps2", "lane_logit_delta"),
    }


def slot_geometry(kin: Tensor) -> dict[str, Tensor]:
    """Position, lane index (-1 for empty slots) and speed of every slot."""
    onehot = kin[..., 1:4]
    lane = onehot.argmax(dim=-1)
    empty = onehot.sum(dim=-1) <= 0
    lane = torch.where(empty, torch.full_like(lane, -1), lane)
    return dict(s=kin[..., 0], lane=lane, v=kin[..., 4])


class _KinAttention(nn.Module):
    """Relative-geometry attention over kinematics-only vehicle slots.

    Keys are masked by slot padding and the neighbourhood rule (same or adjacent
    lane within ``radius_m``; self always allowed).  Relative geometry (ds, dlane,
    dv) is a per-head attention bias, so the operator is permutation-equivariant
    in the slot axis.
    """

    def __init__(self, cfg: HdvConfig) -> None:
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.radius = float(cfg.radius_m)
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        self.bias = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(), nn.Linear(64, cfg.n_heads))

    def _pair_bias(self, geo: dict) -> Tensor:
        s, lane, v = geo["s"], geo["lane"], geo["v"]
        ds = s[:, :, None] - s[:, None, :]
        dlane = lane[:, :, None] - lane[:, None, :]
        dv = v[:, :, None] - v[:, None, :]
        same_lane = (lane[:, :, None] == lane[:, None, :]) & (lane[:, :, None] >= 0)
        eye = torch.eye(s.shape[1], dtype=torch.bool, device=s.device)[None]
        feats = torch.stack([
            ds / self.radius,
            dlane.to(s.dtype) / 2.0,
            dv / 20.0,
            ds.abs() / self.radius,
            same_lane.to(s.dtype),
            eye.expand_as(ds).to(s.dtype),
        ], dim=-1)
        return self.bias(feats).permute(0, 3, 1, 2)

    def _neighbourhood(self, geo: dict, mask: Tensor) -> Tensor:
        s, lane = geo["s"], geo["lane"]
        ds = (s[:, :, None] - s[:, None, :]).abs()
        dl = (lane[:, :, None] - lane[:, None, :]).abs()
        valid = mask[:, :, None] & mask[:, None, :]
        adjacent = ((dl <= 1) & (lane[:, :, None] >= 0)
                    & (lane[:, None, :] >= 0))
        near = ds <= self.radius
        eye = torch.eye(s.shape[1], dtype=torch.bool, device=s.device)[None]
        return valid & (eye | (adjacent & near))

    def forward(self, x: Tensor, mask: Tensor, geo: dict) -> tuple[Tensor, Tensor]:
        b, k, _ = x.shape
        q = self.q(x).view(b, k, self.h, self.dh).transpose(1, 2)
        kk = self.k(x).view(b, k, self.h, self.dh).transpose(1, 2)
        vv = self.v(x).view(b, k, self.h, self.dh).transpose(1, 2)
        logits = (q @ kk.transpose(-1, -2)) / math.sqrt(self.dh)
        logits = logits + self._pair_bias(geo)
        neigh = self._neighbourhood(geo, mask)[:, None, :, :]
        logits = logits.masked_fill(~neigh, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ vv).transpose(1, 2).reshape(b, k, -1)
        return self.o(out) * mask.unsqueeze(-1).to(out.dtype), attn


class KinBlock(nn.Module):
    """Pre-norm transformer block over vehicle slots."""

    def __init__(self, cfg: HdvConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = _KinAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn, cfg.d_model))

    def forward(self, x: Tensor, mask: Tensor, geo: dict) -> tuple[Tensor, Tensor]:
        a, attn = self.attn(self.ln1(x), mask, geo)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x * mask.unsqueeze(-1).to(x.dtype), attn


class HdvResidualNet(nn.Module):
    """Additive acceleration and lane-logit correction on top of the physics."""

    def __init__(self, cfg: Optional[HdvConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or HdvConfig()
        c = self.cfg
        self.norm = nn.LayerNorm(c.in_features)
        self.embed = nn.Sequential(
            nn.Linear(c.in_features, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, c.d_model))
        self.blocks = nn.ModuleList([KinBlock(c) for _ in range(c.n_layers)])
        self.accel = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 2))                      # delta + log-sigma
        self.lane = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 3))                      # 3-way lane logit delta
        # Built last so the baseline layers keep their initialisation; the final
        # layer is zero so the adapter is a no-op at init.
        self.cond_proj = None
        if int(c.conditioning_dim) > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(int(c.conditioning_dim), c.d_model), nn.GELU(),
                nn.Linear(c.d_model, c.d_model))
            nn.init.zeros_(self.cond_proj[-1].weight)
            nn.init.zeros_(self.cond_proj[-1].bias)

    def input_schema(self) -> dict:
        """:func:`input_schema` of this network's configuration."""
        return input_schema(self.cfg)

    def _input_features(self, kin: Tensor, cav_flag: Optional[Tensor]) -> Tensor:
        """Network input columns: kinematics, plus the ablation-only CAV flag."""
        if not self.cfg.use_cav_flag:
            if kin.shape[-1] > KIN_F:
                raise ValueError(
                    "headline HDV net accepts kinematics only (KIN_F=%d); got "
                    "last dim %d. Set HdvConfig.use_cav_flag=True for the "
                    "ablation." % (KIN_F, kin.shape[-1]))
            return kin
        if kin.shape[-1] == KIN_F:
            if cav_flag is None:
                raise ValueError("use_cav_flag=True needs cav_flag [B, K] (ablation)")
            return torch.cat([kin, cav_flag.to(kin.dtype).unsqueeze(-1)], dim=-1)
        if kin.shape[-1] == KIN_F + 1:
            return kin
        raise ValueError("kin last dim must be %d (+1 ablation)" % KIN_F)

    def forward(self, kin: Tensor, kin_mask: Tensor, *,
                cav_flag: Optional[Tensor] = None,
                conditioning: Optional[Tensor] = None) -> dict:
        """One residual prediction.

        ``kin`` is ``[B, K, 7]``, ``kin_mask`` the ``[B, K]`` occupied slots;
        ``conditioning`` an optional ``[B, K, conditioning_dim]`` driver embedding
        (ignored unless ``conditioning_dim > 0``).  Returns
        ``accel_delta_mps2 [B, K]``, ``accel_log_sigma [B, K]``,
        ``lane_logit_delta [B, K, 3]`` and ``attn [B, H, K, K]`` (layer mean).
        """
        c = self.cfg
        if kin.dim() != 3:
            raise ValueError("kin must be [B, K, F]; got shape %s" % (tuple(kin.shape),))
        mask = kin_mask.bool()
        x_in = self._input_features(kin, cav_flag)

        geo = slot_geometry(kin[..., :KIN_F])
        x = self.embed(self.norm(x_in)) * mask.unsqueeze(-1).to(x_in.dtype)
        if self.cond_proj is not None and conditioning is not None:
            if conditioning.shape[:2] != mask.shape \
                    or conditioning.shape[-1] != self.cfg.conditioning_dim:
                raise ValueError(
                    "conditioning must be [B,K,conditioning_dim]=%s"
                    % ((int(mask.shape[0]), int(mask.shape[1]),
                        int(self.cfg.conditioning_dim)),))
            x = x + self.cond_proj(conditioning.to(x.dtype)) \
                * mask.unsqueeze(-1).to(x.dtype)

        attns = []
        for blk in self.blocks:
            x, a = blk(x, mask, geo)
            attns.append(a)
        attn = torch.stack(attns, dim=0).mean(dim=0) if attns else None

        if c.zero_residual:
            zeros = torch.zeros_like(kin[..., 0])
            return {
                "accel_delta_mps2": zeros,
                "accel_log_sigma": torch.zeros_like(zeros),
                "lane_logit_delta": torch.zeros(
                    *kin.shape[:-1], 3, dtype=kin.dtype, device=kin.device),
                "attn": attn,
            }

        acc = self.accel(x)
        delta = c.accel_delta_scale_mps2 * torch.tanh(acc[..., 0])
        log_sigma = acc[..., 1].clamp(-6.0, 2.0)
        lane_delta = c.lane_logit_scale * self.lane(x)
        delta = delta * mask.to(delta.dtype)
        lane_delta = lane_delta * mask.unsqueeze(-1).to(lane_delta.dtype)
        return {"accel_delta_mps2": delta, "accel_log_sigma": log_sigma,
                "lane_logit_delta": lane_delta, "attn": attn}
