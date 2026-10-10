"""WM-N1 network: lane-aware attention encoder + stochastic step decoder,
arrival head and per-tau outcome heads.

Implements ``output/wd01/wm_n1/WM_N1_DESIGN.md`` sections 2-3 and the frozen
``REGISTRATION_WM_N1.md``.  Inputs are EXACTLY the frozen featurizer tensors
(:mod:`model.constants`, ``features/FORMAT.md``):

* ``veh_step``  float [B, K, 16]
* ``veh_mask``  bool/int [B, K]
* ``plan``      float [B, K, 19, 8]
* ``wall``      float [B, W, 11]
* ``wall_mask`` bool/int [B, W]
* ``counter``   float [B, 2, 30]

No forbidden input (``type_id``, ``speed_mode``, true penetration, ``pre_fix``,
world strings, seed, ``t0``) is a parameter of :meth:`WmN1Net.forward`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C


# ------------------------------------------------------------------ config --
@dataclass
class WmN1Config:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn: int = 768
    n_mix: int = C.N_MIX
    radius_m: float = 150.0
    bias_hidden: int = 64
    counter_channels: int = 128
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


# ------------------------------------------------------------- utilities ----
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (x * m).sum(dim=1) / denom


def lane_index_from_step(veh: torch.Tensor) -> torch.Tensor:
    """[B, K, 16] -> [B, K] lane index 0/1/2, or -1 for a junction row."""
    onehot = veh[..., C.LANE_LEFT:C.LANE_ACCEL + 1]
    lane = onehot.argmax(dim=-1)
    empty = onehot.sum(dim=-1) <= 0.0
    return torch.where(empty, torch.full_like(lane, -1), lane)


# --------------------------------------------------------------- attention --
class LaneAwareAttention(nn.Module):
    """Relative-geometry multi-head attention over vehicle slots.

    Keys are masked by slot padding and by the neighbourhood rule: same or
    adjacent lane within ``radius_m`` (self is always allowed).  Relative
    geometry (ds, dlane, dv) is fed to a small MLP as a per-head attention
    bias, like a GNN over a radius graph (design section 2.1).
    """

    def __init__(self, cfg: WmN1Config):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.radius = cfg.radius_m
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        self.bias = nn.Sequential(
            nn.Linear(6, cfg.bias_hidden), nn.GELU(),
            nn.Linear(cfg.bias_hidden, cfg.n_heads))

    def _pair_bias(self, geo, mask):
        s, lane, v = geo["s"], geo["lane"], geo["v"]
        ds = s[:, :, None] - s[:, None, :]              # i - j
        dlane = lane[:, :, None] - lane[:, None, :]
        dv = v[:, :, None] - v[:, None, :]
        r = self.radius
        same_lane = (lane[:, :, None] == lane[:, None, :]) & (lane[:, :, None] >= 0)
        eye = torch.eye(s.shape[1], dtype=torch.bool, device=s.device)[None]
        feats = torch.stack([
            ds / r,
            dlane.to(s.dtype) / 2.0,
            dv / 20.0,
            ds.abs() / r,
            same_lane.to(s.dtype),
            eye.expand_as(ds).to(s.dtype),
        ], dim=-1)
        b = self.bias(feats)                            # [B, K, K, H]
        return b.permute(0, 3, 1, 2)

    def _neighbourhood(self, geo, mask):
        s, lane = geo["s"], geo["lane"]
        ds = (s[:, :, None] - s[:, None, :]).abs()
        dl = (lane[:, :, None] - lane[:, None, :]).abs()
        valid = mask[:, :, None] & mask[:, None, :]
        adjacent = (dl <= 1) & (lane[:, :, None] >= 0) & (lane[:, None, :] >= 0)
        near = ds <= self.radius
        eye = torch.eye(s.shape[1], dtype=torch.bool, device=s.device)[None]
        return valid & (eye | (adjacent & near))

    def forward(self, x, mask, geo):
        B, K, _ = x.shape
        q = self.q(x).view(B, K, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, K, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, K, self.h, self.dh).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        logits = logits + self._pair_bias(geo, mask)
        neigh = self._neighbourhood(geo, mask)[:, None, :, :]
        logits = logits.masked_fill(~neigh, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ v).transpose(1, 2).reshape(B, K, -1)
        return self.o(out) * mask.unsqueeze(-1).to(out.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: WmN1Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = LaneAwareAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn, cfg.d_model))

    def forward(self, x, mask, geo):
        x = x + self.attn(self.ln1(x), mask, geo)
        x = x + self.ffn(self.ln2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


# --------------------------------------------------------------- encoders ---
class CounterEncoder(nn.Module):
    """1-D conv over 30 bins x 2 entrances -> a global token (design 2.1)."""

    def __init__(self, cfg: WmN1Config):
        super().__init__()
        ch = cfg.counter_channels
        self.conv = nn.Sequential(
            nn.Conv1d(C.N_ENTRANCES, ch, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=5, padding=2), nn.GELU())
        self.proj = nn.Linear(ch, cfg.d_model)

    def forward(self, counter):
        h = self.conv(counter.float())
        return self.proj(h.mean(dim=-1))


class PlanUpdater(nn.Module):
    """Plan tokens attend into their own CAV token (design 2.1)."""

    def __init__(self, cfg: WmN1Config):
        super().__init__()
        self.d = cfg.d_model
        self.embed = nn.Linear(C.PLAN_F, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(C.N_SEGMENTS, cfg.d_model))
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, x, plan, apply_mask):
        B, K, S, _ = plan.shape
        pe = self.embed(plan.float()) + self.pos[None, None]
        q = self.q(x).unsqueeze(2)                       # [B, K, 1, d]
        k = self.k(pe)
        v = self.v(pe)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        att = torch.softmax(att, dim=-1)
        upd = (att @ v).squeeze(2)
        upd = self.ln(x + upd) - x
        return x + upd * apply_mask.unsqueeze(-1).to(x.dtype)


class WallUpdater(nn.Module):
    """Wall tokens attend to their member vehicles and vice versa (2.1)."""

    def __init__(self, cfg: WmN1Config):
        super().__init__()
        self.d = cfg.d_model
        self.embed = nn.Linear(C.WALL_F, cfg.d_model)
        self.wq = nn.Linear(cfg.d_model, cfg.d_model)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model)
        self.vq = nn.Linear(cfg.d_model, cfg.d_model)
        self.vk = nn.Linear(cfg.d_model, cfg.d_model)
        self.vv = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, x, wall, wall_mask, member):
        B, W, _ = wall.shape
        we = self.embed(wall.float()) * wall_mask.unsqueeze(-1).to(x.dtype)
        # wall <- member vehicles
        q = self.wq(we)
        k = self.wk(x)
        v = self.wv(x)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        att = att.masked_fill(~member[:, None, :], float("-inf"))
        att = torch.nan_to_num(torch.softmax(att, dim=-1), nan=0.0)
        we = we + att @ v
        # vehicle <- wall (only member slots)
        qv = self.vq(x)
        kw = self.vk(we)
        vw = self.vv(we)
        att2 = (qv @ kw.transpose(-1, -2)) / math.sqrt(self.d)
        att2 = att2.masked_fill(~wall_mask[:, None, :], float("-inf"))
        att2 = torch.nan_to_num(torch.softmax(att2, dim=-1), nan=0.0)
        upd = att2 @ vw
        x = x + upd * member.unsqueeze(-1).to(x.dtype)
        wall_global = masked_mean(we, wall_mask)
        return x, wall_global


# ------------------------------------------------------------------ heads ---
class StepHead(nn.Module):
    """Stochastic 0.5 s step decoder: K-component GMM accel + 3-way lane."""

    def __init__(self, cfg: WmN1Config):
        super().__init__()
        d = cfg.d_model
        self.n_mix = cfg.n_mix
        self.accel = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, cfg.n_mix * 3))
        self.lane = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, C.N_LANE_CLASSES))

    def forward(self, x):
        a = self.accel(x).view(*x.shape[:-1], self.n_mix, 3)
        mu = a[..., 0]
        log_sigma = a[..., 1].clamp(-6.0, 2.0)
        mix_logits = a[..., 2]
        lane_logits = self.lane(x)
        return dict(accel_mu=mu, accel_log_sigma=log_sigma,
                    accel_mix_logits=mix_logits, lane_logits=lane_logits)


class ArrivalHead(nn.Module):
    """Per-entrance negative-binomial count + insertion speed (design 2.3)."""

    def __init__(self, cfg: WmN1Config):
        super().__init__()
        d = cfg.d_model
        self.count = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, C.N_ENTRANCES * 2))
        self.speed = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, C.N_ENTRANCES * 2))

    def forward(self, g):
        B = g.shape[0]
        c = self.count(g).view(B, C.N_ENTRANCES, 2)
        count_mu = F.softplus(c[..., 0])
        count_log_alpha = c[..., 1]
        s = self.speed(g).view(B, C.N_ENTRANCES, 2)
        speed_mu = F.softplus(s[..., 0]) + 1.0
        speed_log_sigma = s[..., 1].clamp(-6.0, 2.0)
        return dict(count_mu=count_mu, count_log_alpha=count_log_alpha,
                    speed_mu=speed_mu, speed_log_sigma=speed_log_sigma)


class OutcomeHead(nn.Module):
    """Per-tau quantile heads for the outcome targets + collision logits."""

    def __init__(self, cfg: WmN1Config):
        super().__init__()
        d = cfg.d_model
        self.trunk = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.q = nn.ModuleList([
            nn.Linear(d, C.N_OUTCOME_TARGETS * C.N_QUANTILES)
            for _ in range(C.N_TAU)])
        self.collision = nn.ModuleList([nn.Linear(d, 1)
                                        for _ in range(C.N_TAU)])

    def forward(self, pooled):
        h = self.trunk(pooled)
        qs = []
        cs = []
        for tau_i in range(C.N_TAU):
            q = self.q[tau_i](h).view(
                h.shape[0], C.N_OUTCOME_TARGETS, C.N_QUANTILES)
            qs.append(q)
            cs.append(self.collision[tau_i](h).squeeze(-1))
        return dict(outcome_quantiles=torch.stack(qs, dim=1),
                    collision_logits=torch.stack(cs, dim=1))

    @staticmethod
    def paired_contrast(q_a, q_b):
        """Comonotone quantile difference (design section 3, line 114-115)."""
        return q_a - q_b


# -------------------------------------------------------------------- net ---
class WmN1Net(nn.Module):
    """One WM-N1 member (design sections 2-3)."""

    def __init__(self, cfg: WmN1Config | None = None):
        super().__init__()
        self.cfg = cfg or WmN1Config()
        c = self.cfg
        self.veh_norm = nn.LayerNorm(C.VEH_F)
        self.veh_embed = nn.Sequential(
            nn.Linear(C.VEH_F, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, c.d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(c) for _ in range(c.n_layers)])
        self.plan = PlanUpdater(c)
        self.wall = WallUpdater(c)
        self.counter = CounterEncoder(c)
        self.global_norm = nn.LayerNorm(c.d_model)
        self.step = StepHead(c)
        self.arrival = ArrivalHead(c)
        self.outcome = OutcomeHead(c)

    # -- geometry helpers ---------------------------------------------------
    def _geometry(self, veh):
        return dict(s=veh[..., C.S_DOWN], lane=lane_index_from_step(veh),
                    v=veh[..., C.V_IDX])

    # -- forward ------------------------------------------------------------
    def forward(self, veh_step, veh_mask, plan, wall, wall_mask, counter):
        """One t0 snapshot -> encoder state + step/arrival/outcome heads.

        All six arguments are frozen featurizer tensors.  No forbidden input
        is accepted.
        """
        veh = veh_step.float()
        mask = veh_mask > 0
        x = self.veh_embed(self.veh_norm(veh)) * mask.unsqueeze(-1).to(veh.dtype)
        geo = self._geometry(veh)
        for blk in self.blocks:
            x = blk(x, mask, geo)

        is_cav = (veh[..., C.IS_CAV_IDX] > 0) & mask
        x = self.plan(x, plan, is_cav)
        member = (veh[..., C.WALL_MEMBER_IDX] > 0) & mask
        x, wall_global = self.wall(x, wall, wall_mask > 0, member)

        veh_global = masked_mean(x, mask)
        counter_global = self.counter(counter)
        g = self.global_norm(veh_global + counter_global + wall_global)

        step = self.step(x)
        arrival = self.arrival(g)
        outcome = self.outcome(torch.cat([veh_global, g], dim=-1))
        return dict(slot_hidden=x, global_token=g, step=step,
                    arrival=arrival, **outcome)


# ---------------------------------------------------------------- helpers ---
def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_member(seed: int | None = None, cfg: WmN1Config | None = None):
    """Build one member, optionally with a per-member torch seed."""
    if seed is not None:
        torch.manual_seed(int(seed))
    return WmN1Net(cfg)
