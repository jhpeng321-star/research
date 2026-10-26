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
  initialisation;
* ``HdvConfig.courtesy_inputs`` (default off) adds per-slot courtesy channels
  (ramp distance-to-end, speed/slow/waiting time, merge intent) and a per-pair
  neighbour courtesy bias, both zero-initialised so a freshly enabled net is a
  no-op and old checkpoints load unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from wms.core import speed_flags
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

#: V10COURTESY per-slot inputs (ramp car and its predicted merge intent).  All
#: channels are zero for a vehicle that is not on the acceleration lane.
COURTESY_FIELDS = (
    "ramp",          # 1 on the acceleration lane (lane 2)
    "dend",          # distance to the acceleration-lane end / 50 m
    "near_end",      # dend < 40 m
    "v",             # speed / 20 m/s
    "slow",          # clamp(1 - v / 8, 0, 1): slow or queued ramp car
    "wait",          # time spent waiting/slow in lane 2 / 20 s
    "intent",        # merge head's P(merge) for that ramp car (detached)
)
COURTESY_F = len(COURTESY_FIELDS)
#: Per-pair courtesy features of the neighbour (attention bias).
COURTESY_PAIR_FIELDS = ("nb_ramp", "nb_dend", "nb_near", "nb_slow", "nb_intent")
COURTESY_PAIR_F = len(COURTESY_PAIR_FIELDS)


def courtesy_features(kin: Tensor, present: Tensor, physics,
                      *, lane_age: Optional[Tensor] = None,
                      merge_wait_s: Optional[Tensor] = None,
                      merge_prob: Optional[Tensor] = None,
                      near_end_m: float = 40.0) -> Tensor:
    """Per-slot courtesy inputs ``[B, K, COURTESY_F]`` from recorded/rollout state.

    ``merge_prob`` is the merge head's current probability for each slot (the
    public :func:`wms.model.lane_v5` decision probability for a lane-2 car); it is
    DETACHED on purpose.  The lane head is trained by its own all-risk BCE on the
    same data, while the residual must not be able to move the intent target it
    consumes (otherwise the follower-prediction error would train the lane head
    through the input, double-counting the lane objective and making the
    "uses-intent" ablation ambiguous).  The intent still updates through that
    objective, so the residual sees a causal, moving prediction.

    ``merge_wait_s`` (V8WM3 carried lane-end waiting time) is used when present;
    otherwise ``lane_age`` (steps since the last lane change, i.e. time spent in
    lane 2) is the waiting-time proxy.  Neither is future information.
    """
    present = present.bool()
    s = kin[..., 0]
    v = kin[..., 4]
    onehot = kin[..., 1:4]
    lane = onehot.argmax(dim=-1)
    lane = torch.where(onehot.sum(dim=-1) <= 0, torch.full_like(lane, -1), lane)
    ramp = present & (lane == int(getattr(physics, "merge_lane", 2)))
    zero = torch.zeros_like(s)
    dend_m = float(getattr(physics, "merge_zone_s1_m", 75.44)) - s
    dend = torch.where(ramp, (dend_m / 50.0).clamp(-2.0, 6.0), zero)
    near = torch.where(ramp, (dend_m < float(near_end_m)).to(s.dtype), zero)
    speed = torch.where(ramp, (v / 20.0).clamp(0.0, 2.0), zero)
    slow = torch.where(ramp, (1.0 - v / 8.0).clamp(0.0, 1.0), zero)
    if merge_wait_s is not None:
        wait_s = torch.as_tensor(merge_wait_s, device=s.device, dtype=s.dtype).clamp_min(0.0)
    elif lane_age is not None:
        wait_s = (torch.as_tensor(lane_age, device=s.device, dtype=s.dtype).clamp_min(0.0)
                  * float(getattr(physics, "dt_s", 0.5)))
    else:
        wait_s = zero
    wait = torch.where(ramp, (wait_s / 20.0).clamp(0.0, 3.0), zero)
    if merge_prob is None:
        intent = zero
    else:
        intent = torch.where(ramp, torch.as_tensor(
            merge_prob, device=s.device, dtype=s.dtype).clamp(0.0, 1.0).detach(), zero)
    feats = torch.stack((ramp.to(s.dtype), dend, near, speed, slow, wait, intent), dim=-1)
    return torch.where(present.unsqueeze(-1), feats, torch.zeros_like(feats))


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
    courtesy_inputs: bool = False       # V10COURTESY ramp-car / merge-intent channels
    residual_head_rescale: bool = False  # R5MERGE learned pre-tanh divisor (identity at load)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if int(self.conditioning_dim) < 0:
            raise ValueError("conditioning_dim must be non-negative")
        if not isinstance(self.courtesy_inputs, bool):
            raise ValueError("courtesy_inputs must be a boolean")
        if not isinstance(self.residual_head_rescale, bool):
            raise ValueError("residual_head_rescale must be a boolean")

    @property
    def in_features(self) -> int:
        return KIN_F + (1 if self.use_cav_flag else 0)


def input_schema(cfg: Optional[HdvConfig] = None) -> dict:
    """Exact input surface of the residual network (label-leak contract)."""
    cfg = cfg or HdvConfig()
    optional = tuple(_ABLATION_INPUTS) if cfg.use_cav_flag else ()
    allowed = _ALLOWED_INPUTS + (("courtesy",) if cfg.courtesy_inputs else ())
    forbidden = tuple(
        f for f in _FORBIDDEN_INPUTS if not (cfg.use_cav_flag and f == "cav_flag"))
    return {
        "module": "hdv.net.HdvResidualNet",
        "kin_fields": tuple(KIN_FIELDS),
        "kin_f": int(KIN_F),
        "in_features": int(cfg.in_features),
        "allowed_inputs": allowed,
        "optional_inputs": optional,
        "courtesy_inputs": bool(cfg.courtesy_inputs),
        "courtesy_fields": tuple(COURTESY_FIELDS) if cfg.courtesy_inputs else (),
        "courtesy_f": int(COURTESY_F) if cfg.courtesy_inputs else 0,
        "forbidden_inputs": forbidden,
        "use_cav_flag": bool(cfg.use_cav_flag),
        "zero_residual": bool(cfg.zero_residual),
        "residual_head_rescale": bool(cfg.residual_head_rescale),
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
        # V10COURTESY: built after the baseline bias; zero so the initial operator is
        # unchanged and old checkpoints need no new tensor.
        self.courtesy_bias = None
        if cfg.courtesy_inputs:
            self.courtesy_bias = nn.Linear(COURTESY_PAIR_F, cfg.n_heads)
            nn.init.zeros_(self.courtesy_bias.weight)
            nn.init.zeros_(self.courtesy_bias.bias)

    def _pair_bias(self, geo: dict, courtesy: Optional[Tensor] = None) -> Tensor:
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
        bias = self.bias(feats)
        if self.courtesy_bias is not None and courtesy is not None:
            # [B, i, j, f] = courtesy of the key/neighbour j.
            nb = courtesy.unsqueeze(1).expand(-1, s.shape[1], -1, -1)
            cfeats = torch.stack((
                nb[..., 0], nb[..., 1], nb[..., 2], nb[..., 4], nb[..., 6]), dim=-1)
            bias = bias + self.courtesy_bias(cfeats)
        return bias.permute(0, 3, 1, 2)

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

    def forward(self, x: Tensor, mask: Tensor, geo: dict,
                courtesy: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        b, k, _ = x.shape
        q = self.q(x).view(b, k, self.h, self.dh).transpose(1, 2)
        kk = self.k(x).view(b, k, self.h, self.dh).transpose(1, 2)
        vv = self.v(x).view(b, k, self.h, self.dh).transpose(1, 2)
        logits = (q @ kk.transpose(-1, -2)) / math.sqrt(self.dh)
        logits = logits + self._pair_bias(geo, courtesy)
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

    def forward(self, x: Tensor, mask: Tensor, geo: dict,
                courtesy: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        a, attn = self.attn(self.ln1(x), mask, geo, courtesy)
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
        # V10COURTESY: built last (after every baseline/conditioning module) so the
        # existing initialisation and state-dict keys are unchanged.  The final layer
        # is zero, so the adapter is a no-op until trained.
        self.courtesy_proj = None
        if c.courtesy_inputs:
            self.courtesy_proj = nn.Sequential(
                nn.Linear(COURTESY_F, c.d_model), nn.GELU(),
                nn.Linear(c.d_model, c.d_model))
            nn.init.zeros_(self.courtesy_proj[-1].weight)
            nn.init.zeros_(self.courtesy_proj[-1].bias)
        # R5MERGE: the head saturates at tanh's flat tail (WAVE.md).  Dividing the
        # pre-tanh activation by exp(log_scale) is exactly the identity at load
        # (exp(0) = 1) while log_scale can learn to shrink the activation and restore
        # the head gradient.  No tensor is created when the key is off.
        self.residual_head_log_scale = None
        if c.residual_head_rescale:
            self.residual_head_log_scale = nn.Parameter(torch.zeros(()))

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
                conditioning: Optional[Tensor] = None,
                courtesy: Optional[Tensor] = None) -> dict:
        """One residual prediction (S-SPEED2 ``compact_slots``: present slots only)."""
        # Changing the slot shape changes dropout's RNG consumption and mask assignment.
        # Keep the original stochastic operator whenever training dropout is active.
        if (speed_flags.enabled("compact_slots") and kin.dim() == 3
                and not any(block.ffn[2].training and block.ffn[2].p > 0 for block in self.blocks)):
            return self._forward_compact(kin, kin_mask, cav_flag=cav_flag,
                                         conditioning=conditioning, courtesy=courtesy)
        return self._forward_dense(kin, kin_mask, cav_flag=cav_flag,
                                   conditioning=conditioning, courtesy=courtesy)

    def _forward_compact(self, kin: Tensor, kin_mask: Tensor, *,
                         cav_flag: Optional[Tensor] = None,
                         conditioning: Optional[Tensor] = None,
                         courtesy: Optional[Tensor] = None) -> dict:
        """S-SPEED2: :meth:`_forward_dense` on the present slots, scattered back.

        Every row keeps its present slots in slot order, padded to ``P + 1`` columns with
        absent slots (at least one per row).  A present slot only attends to present
        neighbours, so its outputs and gradients are the dense ones up to float
        reassociation.  Absent slots get exactly what the dense network gives them: zero
        deltas / attention and the network's absent-row ``accel_log_sigma``, which is taken
        from the row's own padding column (same layers, same weights).
        """
        mask = kin_mask.bool()
        b, k = mask.shape
        # Preserve the dense API's shape errors and its ignored optional inputs.
        if ((self.courtesy_proj is not None and courtesy is not None
             and tuple(courtesy.shape) != (b, k, COURTESY_F))
                or (self.cond_proj is not None and conditioning is not None
                    and (conditioning.shape[:2] != mask.shape
                         or conditioning.shape[-1] != self.cfg.conditioning_dim))):
            return self._forward_dense(kin, kin_mask, cav_flag=cav_flag,
                                       conditioning=conditioning, courtesy=courtesy)
        width = int(mask.sum(1).max()) + 1 if b else k + 1
        if width >= k:           # no width reduction: avoid sorting/scattering a full grid
            return self._forward_dense(kin, kin_mask, cav_flag=cav_flag,
                                       conditioning=conditioning, courtesy=courtesy)
        # Stable sort: present slots first, each group in slot order.
        idx = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)[:, :width]

        def take(value: Optional[Tensor]) -> Optional[Tensor]:
            if value is None or not torch.is_tensor(value):
                return value
            if value.dim() == 2:
                return value.gather(1, idx)
            return value.gather(1, idx.unsqueeze(-1).expand(-1, -1, value.shape[-1]))

        out = self._forward_dense(
            take(kin), take(kin_mask),
            cav_flag=take(cav_flag) if self.cfg.use_cav_flag and kin.shape[-1] == KIN_F else None,
            conditioning=take(conditioning) if self.cond_proj is not None else None,
            courtesy=take(courtesy) if self.courtesy_proj is not None else None)

        def put(value: Tensor, fill: Optional[Tensor] = None) -> Tensor:
            shape = (b, k, *value.shape[2:])
            full = (value.new_zeros(shape) if fill is None
                    else fill.reshape(b, 1, *value.shape[2:]).expand(shape).clone())
            index = idx.reshape(b, width, *([1] * (value.dim() - 2))).expand_as(value)
            return full.scatter(1, index, value)

        sigma = out["accel_log_sigma"]
        result = {"accel_delta_mps2": put(out["accel_delta_mps2"]),
                  "accel_log_sigma": put(sigma, sigma[:, -1]),
                  "lane_logit_delta": put(out["lane_logit_delta"])}
        attn = out["attn"]
        if attn is not None:
            h = attn.shape[1]
            cols = attn.new_zeros(b, h, width, k).scatter(
                3, idx[:, None, None, :].expand(b, h, width, width), attn)
            attn = attn.new_zeros(b, h, k, k).scatter(
                2, idx[:, None, :, None].expand(b, h, width, k), cols)
        result["attn"] = attn
        return result

    def _forward_dense(self, kin: Tensor, kin_mask: Tensor, *,
                       cav_flag: Optional[Tensor] = None,
                       conditioning: Optional[Tensor] = None,
                       courtesy: Optional[Tensor] = None) -> dict:
        """One residual prediction.

        ``kin`` is ``[B, K, 7]``, ``kin_mask`` the ``[B, K]`` occupied slots;
        ``conditioning`` an optional ``[B, K, conditioning_dim]`` driver embedding
        (ignored unless ``conditioning_dim > 0``); ``courtesy`` the optional
        ``[B, K, COURTESY_F]`` ramp-car / merge-intent inputs (used only with
        ``courtesy_inputs=True``).  Returns
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
        if self.courtesy_proj is not None:
            if courtesy is None:
                courtesy = kin.new_zeros(*kin.shape[:2], COURTESY_F)
            if tuple(courtesy.shape) != (*kin.shape[:2], COURTESY_F):
                raise ValueError(
                    "courtesy must be [B,K,COURTESY_F]=%s; got %s"
                    % ((int(kin.shape[0]), int(kin.shape[1]), COURTESY_F),
                       tuple(courtesy.shape)))
            x = x + self.courtesy_proj(courtesy.to(x.dtype)) \
                * mask.unsqueeze(-1).to(x.dtype)
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
            x, a = blk(x, mask, geo, courtesy)
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
        activation = acc[..., 0]
        if self.residual_head_log_scale is not None:
            activation = activation / torch.exp(self.residual_head_log_scale)
        delta = c.accel_delta_scale_mps2 * torch.tanh(activation)
        log_sigma = acc[..., 1].clamp(-6.0, 2.0)
        lane_delta = c.lane_logit_scale * self.lane(x)
        delta = delta * mask.to(delta.dtype)
        lane_delta = lane_delta * mask.unsqueeze(-1).to(lane_delta.dtype)
        return {"accel_delta_mps2": delta, "accel_log_sigma": log_sigma,
                "lane_logit_delta": lane_delta, "attn": attn}
