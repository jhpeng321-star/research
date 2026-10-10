"""P2 HDV residual network: attention over neighbour kinematics.

Returns an additive correction on top of the Part A v2g baseline (IDM +
MOBIL/merge): an acceleration correction and a lane-change logit correction.
The headline model has NO CAV flag; ``use_cav_flag`` is the registered
ablation only.

The only inputs are ``kin [B, K, KIN_F]`` (the P9 ``veh_kin`` view: position,
lane one-hot, speed, acceleration, length) and ``kin_mask [B, K]``.  The slot
count ``K`` is a FORMAT parameter (amendment 5: 128 for the frozen v2 format,
192 for v3) and is carried by the input tensors; the net has no K-dependent
weights.  Action one-hots, strategy names, CAV flags and outcome labels are
never accepted: :func:`input_schema` declares that contract and the tests
perturb a full feature dict to prove the output is invariant to them.

Gate switches:

* ``HdvConfig.zero_residual=True`` (G1): return an identically zero residual;
* ``HdvConfig.use_cav_flag=True`` (ablation): append exactly one extra input
  channel; the headline config keeps ``use_cav_flag=False``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import KIN_F, KIN_FIELDS, HdvConfig

# ------------------------------------------------------------------ schema --
# Inputs the headline net is allowed to consume.  Deliberately excludes every
# action / strategy / CAV-flag / label channel.
_ALLOWED_INPUTS = ("kin", "kin_mask")
_ABLATION_INPUTS = ("cav_flag",)
_FORBIDDEN_INPUTS = (
    "action", "action_code", "action_onehot", "actions",
    "strategy", "strategy_name", "is_cav", "cav_flag",
    "hdv_base", "cav_plan_ch", "outcome", "outcomes", "label", "labels",
)


def input_schema(cfg: Optional[HdvConfig] = None) -> dict:
    """G3 label-leak contract for the HDV residual net.

    Returns the exact input surface.  In the headline config the optional
    input tuple is empty and ``cav_flag`` is forbidden; only the registered
    ablation moves ``cav_flag`` into ``optional_inputs``.
    """
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
        # Hybrid-only per-slot conditioning adapter (0 = absent, baseline).
        "conditioning_dim": int(cfg.conditioning_dim),
        "identity_outputs": ("accel_delta_mps2", "lane_logit_delta"),
    }


# --------------------------------------------------------------- attention --
class _KinAttention(nn.Module):
    """Relative-geometry attention over kinematics-only vehicle slots.

    Keys are masked by slot padding and by the neighbourhood rule (same or
    adjacent lane within ``radius_m``; self always allowed).  Relative
    geometry (ds, dlane, dv) is a per-head attention bias, so the operator is
    permutation-equivariant in the slot axis.
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

    def _pair_bias(self, geo: dict, mask: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, mask: Tensor, geo: dict):
        b, k, _ = x.shape
        q = self.q(x).view(b, k, self.h, self.dh).transpose(1, 2)
        kk = self.k(x).view(b, k, self.h, self.dh).transpose(1, 2)
        vv = self.v(x).view(b, k, self.h, self.dh).transpose(1, 2)
        logits = (q @ kk.transpose(-1, -2)) / math.sqrt(self.dh)
        logits = logits + self._pair_bias(geo, mask)
        neigh = self._neighbourhood(geo, mask)[:, None, :, :]
        logits = logits.masked_fill(~neigh, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ vv).transpose(1, 2).reshape(b, k, -1)
        return self.o(out) * mask.unsqueeze(-1).to(out.dtype), attn


class _KinBlock(nn.Module):
    def __init__(self, cfg: HdvConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = _KinAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn, cfg.d_model))

    def forward(self, x: Tensor, mask: Tensor, geo: dict):
        a, attn = self.attn(self.ln1(x), mask, geo)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x * mask.unsqueeze(-1).to(x.dtype), attn


# ------------------------------------------------------------------ network --
class HdvResidualNet(nn.Module):
    """One HDV residual member (design A3.1(b))."""

    def __init__(self, cfg: Optional[HdvConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or HdvConfig()
        c = self.cfg
        self.norm = nn.LayerNorm(c.in_features)
        self.embed = nn.Sequential(
            nn.Linear(c.in_features, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, c.d_model))
        self.blocks = nn.ModuleList(
            [_KinBlock(c) for _ in range(c.n_layers)])
        self.accel = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 2))                      # delta + log-sigma
        self.lane = nn.Sequential(
            nn.Linear(c.d_model, c.ffn), nn.GELU(),
            nn.Linear(c.ffn, 3))                      # 3-way lane logit delta
        # Hybrid rollout: an optional per-driver embedding adapter, built LAST so
        # the baseline layers keep the exact baseline initialisation and a
        # ``conditioning_dim=0`` config stays byte-identical.  The final layer is
        # zero-initialised, so at init the adapter is a no-op and a hybrid model
        # with default driver parameters reproduces the residual model exactly.
        self.cond_proj = None
        if int(c.conditioning_dim) > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(int(c.conditioning_dim), c.d_model), nn.GELU(),
                nn.Linear(c.d_model, c.d_model))
            nn.init.zeros_(self.cond_proj[-1].weight)
            nn.init.zeros_(self.cond_proj[-1].bias)

    def input_schema(self) -> dict:
        """Instance-level :func:`input_schema` (uses this net's config)."""
        return input_schema(self.cfg)

    @torch.no_grad()
    def assert_no_cav_flag(self, kin: Tensor) -> None:
        """Raise if a CAV flag would leak into the headline model."""
        if not self.cfg.use_cav_flag and kin.shape[-1] > KIN_F:
            raise ValueError(
                "headline HDV net accepts kinematics only (KIN_F=%d); got "
                "last dim %d. Set HdvConfig.use_cav_flag=True for the "
                "ablation." % (KIN_F, kin.shape[-1]))

    def _geometry(self, kin: Tensor) -> dict:
        onehot = kin[..., 1:4]
        lane = onehot.argmax(dim=-1)
        empty = onehot.sum(dim=-1) <= 0
        lane = torch.where(empty, torch.full_like(lane, -1), lane)
        return dict(s=kin[..., 0], lane=lane, v=kin[..., 4])

    def forward(self, kin: Tensor, kin_mask: Tensor, *,
                cav_flag: Optional[Tensor] = None,
                conditioning: Optional[Tensor] = None) -> dict:
        """One t0 (or rollout-step) residual prediction.

        Parameters
        ----------
        kin:
            ``float [B, K, KIN_F]`` kinematics-only slot features; the order
            is :data:`hdv.config.KIN_FIELDS`.  With ``use_cav_flag=True`` the
            caller may pass an extra trailing column.
        kin_mask:
            ``bool [B, K]`` occupied slots.
        cav_flag:
            ``bool [B, K]`` ONLY when the ablation is enabled, else ``None``.
            The headline model never passes it (and ignores it if given).
        conditioning:
            optional ``float [B, K, conditioning_dim]`` per-slot embedding
            (e.g. the inferred driver parameters) added to the embedded
            kinematics.  Only honoured when ``conditioning_dim > 0``; otherwise
            ``None`` keeps the output byte-identical to the baseline.

        Returns
        -------
        dict with

        * ``accel_delta_mps2``: ``float [B, K]`` additive IDM correction;
        * ``accel_log_sigma``: ``float [B, K]`` residual Gaussian log-scale;
        * ``lane_logit_delta``: ``float [B, K, 3]`` additive MOBIL/merge logit
          correction over the baseline lane-change scores;
        * ``attn``: ``float [B, H, K, K]`` attention diagnostics (mean over
          layers).
        """
        c = self.cfg
        if kin.dim() != 3:
            raise ValueError("kin must be [B, K, F]; got shape %s"
                             % (tuple(kin.shape),))
        mask = kin_mask.bool()

        if c.use_cav_flag:
            if kin.shape[-1] == KIN_F:
                if cav_flag is None:
                    raise ValueError(
                        "use_cav_flag=True needs cav_flag [B, K] (ablation)")
                x_in = torch.cat(
                    [kin, cav_flag.to(kin.dtype).unsqueeze(-1)], dim=-1)
            elif kin.shape[-1] == KIN_F + 1:
                x_in = kin
            else:
                raise ValueError("kin last dim must be %d (+1 ablation)"
                                 % KIN_F)
        else:
            self.assert_no_cav_flag(kin)
            x_in = kin

        # geometry always comes from the frozen kinematics columns only.
        geo = self._geometry(kin[..., :KIN_F])
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


def lane_logit_delta_to_logits(base_logits: Tensor, delta: Tensor) -> Tensor:
    """Baseline lane logits + residual correction (shape-preserving sum)."""
    return base_logits + delta
