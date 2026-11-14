"""Part B delay estimator (DESIGN_PARTS_A_B.md §3.3).

Per-lane 1D conv encoder over the 10 m bin field -> lane tokens, plus
CAV tokens, wall tokens, a demand token and a context token, fused by a
small transformer; a masked mean pooling feeds two heteroscedastic heads
(mean + log sigma) for R_native and dR = R_plan - R_native.

~1-3 M parameters at the default width; batched over plans x M x K;
device-agnostic (all tensors created with device=, no .cuda() calls);
checkpoints are plain state_dicts (torch.load(..., weights_only=True)).
No training on real branch data exists yet -- synthetic tests only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .featurize import (CAV_RAW, CTX_RAW, DEMAND_RAW, LMAX_BINS, N_LANES,
                        WALL_RAW)


@dataclass
class DelayEstimatorConfig:
    d_model: int = 192
    nhead: int = 4
    n_layers: int = 3
    dim_ff: int = 512
    dropout: float = 0.0
    conv_channels: int = 32
    conv_kernel: int = 5
    lmax_bins: int = LMAX_BINS
    cav_raw: int = CAV_RAW
    wall_raw: int = WALL_RAW
    demand_raw: int = DEMAND_RAW
    ctx_raw: int = CTX_RAW
    max_cavs: int = 64
    max_walls: int = 4


class LaneFieldEncoder(nn.Module):
    """Shared-weight Conv1d encoder over each lane's bin sequence."""

    def __init__(self, cfg: DelayEstimatorConfig):
        super().__init__()
        ch = cfg.conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(3, ch, cfg.conv_kernel, padding=cfg.conv_kernel // 2),
            nn.GELU(),
            nn.Conv1d(ch, ch, cfg.conv_kernel, padding=cfg.conv_kernel // 2),
            nn.GELU(),
        )
        self.proj = nn.Linear(ch, cfg.d_model)

    def forward(self, lane_bins, lane_mask):
        """lane_bins [B, 3, L, 3], lane_mask [B, 3, L] -> [B, 3, d]."""
        b, nl, l, _ = lane_bins.shape
        x = lane_bins.permute(0, 1, 3, 2).reshape(b * nl, 3, l)
        m = lane_mask.reshape(b * nl, 1, l)
        h = self.conv(x * m) * m               # zero out padded bins
        denom = m.sum(dim=2).clamp(min=1.0)
        pooled = (h * m).sum(dim=2) / denom    # masked mean over bins
        return self.proj(pooled).reshape(b, nl, -1)


class DelayEstimator(nn.Module):
    """Field + tokens -> heteroscedastic (R_native, dR) estimates.

    forward returns a [B, 4] tensor:
        (R_mean, R_logvar, dR_mean, dR_logvar)
    """

    def __init__(self, cfg: DelayEstimatorConfig | None = None):
        super().__init__()
        cfg = self.cfg = cfg or DelayEstimatorConfig()
        d = cfg.d_model
        self.field_encoder = LaneFieldEncoder(cfg)
        self.cav_proj = nn.Linear(cfg.cav_raw, d)
        self.wall_proj = nn.Linear(cfg.wall_raw, d)
        self.demand_proj = nn.Linear(cfg.demand_raw, d)
        self.ctx_proj = nn.Linear(cfg.ctx_raw, d)
        # token-type embedding: 0..2 lanes, 3 cavs, 4 walls, 5 demand, 6 ctx
        self.type_emb = nn.Embedding(7, d)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(block, cfg.n_layers,
                                            enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(self, batch: dict) -> torch.Tensor:
        dev = self.type_emb.weight.device
        tens = {k: torch.as_tensor(v, dtype=torch.float32, device=dev)
                for k, v in batch.items()}     # numpy or torch in
        lane_bins = tens["lane_bins"]                      # [B,3,L,3]
        lane_mask = tens["lane_mask"]                      # [B,3,L]
        cav = tens["cav_tokens"]                           # [B,C,raw]
        cav_mask = tens["cav_mask"]                        # [B,C]
        wall = tens["wall_tokens"]                         # [B,W,raw]
        wall_mask = tens["wall_mask"]                      # [B,W]
        demand = tens["demand"]                            # [B,raw]
        ctx = tens["context"]                              # [B,raw]
        b = lane_bins.shape[0]
        d = self.cfg.d_model

        lane_tok = self.field_encoder(lane_bins, lane_mask)      # [B,3,d]
        cav_tok = self.cav_proj(cav)
        wall_tok = self.wall_proj(wall)
        dem_tok = self.demand_proj(demand).unsqueeze(1)          # [B,1,d]
        ctx_tok = self.ctx_proj(ctx).unsqueeze(1)                # [B,1,d]

        tokens = torch.cat(
            [lane_tok, cav_tok, wall_tok, dem_tok, ctx_tok], dim=1)
        n_lanes, n_cav, n_wall = lane_tok.shape[1], cav_tok.shape[1], \
            wall_tok.shape[1]
        type_ids = torch.cat([
            torch.arange(0, n_lanes),
            torch.full((n_cav,), 3), torch.full((n_wall,), 4),
            torch.full((1,), 5), torch.full((1,), 6)]).to(dev)
        tokens = tokens + self.type_emb(type_ids)

        pad = torch.cat(
            [torch.zeros(b, n_lanes + n_wall + 2, dtype=torch.bool,
                         device=dev), cav_mask <= 0], dim=1)  # True = pad
        h = self.encoder(tokens, src_key_padding_mask=pad)
        pooled = h.masked_fill(pad.unsqueeze(-1), 0.0).sum(dim=1) \
            / (~pad).sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
