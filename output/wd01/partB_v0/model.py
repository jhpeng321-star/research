"""Part B V0 policy- and tau-conditioned value network (PLAN_PARTB_V0_TRAINING.md).

Encoder base reused from part4/partB (DelayEstimatorConfig layout,
LaneFieldEncoder conv, token structure): lane-bin field through a shared
1-D conv per lane, CAV tokens, wall tokens, a demand token and a context
token, fused by a small transformer with masked mean pooling.

V0 extensions per the plan:
  * tau conditioning: sinusoidal tau embedding (own token + added to the
    pooled representation before the heads);
  * policy conditioning: learned embedding over POLICIES
    {switching_mix, damping_fleet}, table sized n_policies=4 so future
    policies extend without a schema change;
  * pre_fix input flag appended to the context scalars (chunk fleet fix,
    DATA_CARD_V0 caveat 3);
  * lane_geom static validity mask multiplied into the lane mask;
  * heads conditioned on (tau, policy): V_delay / V_exp / V_brake as
    symlog twohot distributions (255 bins, DreamerV3 style) and P_coll
    Bernoulli.  P_closing_pair and future_min_ttc_q05 have NO labels in
    the V0 dataset, so they exist as optional heads, DISABLED by default;
    enabling them without labels would invent supervision (they are here
    only so the schema is forward-compatible).

~1.7 M parameters at the default width (target 1-3 M per member);
batched, device-agnostic (all tensors created with device=, no .cuda()
calls); checkpoints are plain state_dicts loadable with
torch.load(..., weights_only=True).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn

from partB.featurize import CAV_RAW, CTX_RAW, DEMAND_RAW, LMAX_BINS, \
    WALL_RAW  # part4/partB reuse contract (sys.path set in __init__)

# canonical policy ids (V0).  v0.3.2 fix #3: the campaign '__native'/
# '__pi' arms of chunks 1-3 + chunk1_fix27 are random per-CAV switching
# schedules (per-CAV actions from switch_schedule.json /
# suffix_commands.jsonl), NOT a fleet-wide native/PI policy -- the policy
# id is "switching_mix"; there is no 'native' policy in this dataset.
POLICIES = ("switching_mix", "damping_fleet")
N_POLICIES_MAX = 4                       # reserved slots for future policies

N_BINS = 255                             # twohot bins (DreamerV3)
SYM_MAX = 20.0                           # symlog space half-width: |x| <= e^20-1
TAU_MAX_S = 600.0                        # conditioning range [0, TAU_MAX_S]
CTX_V0_RAW = CTX_RAW + 1                 # + pre_fix flag


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def check_token_widths(state_dict: dict, cfg, where: str = "checkpoint") \
        -> None:
    """Loud token-width guard (BRIEF_PARTB_WALL_ACTION_INPUTS_DRAFT.md:
    config-driven widths; old checkpoints fail LOUDLY on a mismatch).

    The v0.4 featurizer bumped the token widths (CAV_RAW 17 -> 20,
    WALL_RAW 6 -> 11).  Every checkpoint carries its cfg_model JSON, so a
    faithful reload reuses the RECORDED widths; this guard catches
    sidecar-less/edited checkpoints and any data/config mismatch BEFORE
    torch's generic shape error, naming both widths.
    """
    got = {}
    for proj, key in (("cav", "cav_proj.weight"), ("wall", "wall_proj.weight")):
        w = state_dict.get(key)
        if w is not None:
            got[proj] = int(w.shape[1])
    want = {"cav": int(getattr(cfg, "cav_raw")),
            "wall": int(getattr(cfg, "wall_raw"))}
    bad = {k: (want[k], got[k]) for k in got if got[k] != want[k]}
    if bad:
        detail = ", ".join(f"{k}_raw: checkpoint {g} vs config {w}"
                           for k, (w, g) in bad.items())
        raise ValueError(
            f"{where}: token-width mismatch ({detail}).  Token widths are "
            "config-driven; v0.4 changed the layout (CAV_RAW 20, "
            "WALL_RAW 11; pre-v0.4 checkpoints used CAV_RAW 17, "
            "WALL_RAW 6).  Load old checkpoints with their recorded "
            "cfg_model (member_*.json) -- never with new defaults.")


def _check_input_widths(cav: torch.Tensor, wall: torch.Tensor, cfg) -> None:
    """Loud data-vs-config width check at forward time."""
    if int(cav.shape[-1]) != int(cfg.cav_raw) or \
            int(wall.shape[-1]) != int(cfg.wall_raw):
        raise ValueError(
            f"input token widths cav={cav.shape[-1]}, "
            f"wall={wall.shape[-1]} do not match the model config "
            f"(cav_raw={cfg.cav_raw}, wall_raw={cfg.wall_raw}); the "
            "dataset and the net must use the same v0.4 token layout")


def inv_symlog(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * torch.expm1(y.abs())


def bin_centers() -> torch.Tensor:
    """Real-space twohot bin centers (symlog-uniform grid)."""
    return inv_symlog(torch.linspace(-SYM_MAX, SYM_MAX, N_BINS,
                                     dtype=torch.float32))


def twohot_encode(x: torch.Tensor) -> torch.Tensor:
    """Real values -> twohot probabilities over the symlog bin grid.

    Splits the mass of each row between the two neighbouring bins exactly
    as DreamerV3; out-of-range values clamp to the end bins.  [B] -> [B, K].
    """
    x = x.reshape(-1)
    grid = torch.linspace(-SYM_MAX, SYM_MAX, N_BINS, dtype=x.dtype,
                          device=x.device)     # already symlog coordinates
    w = grid[1] - grid[0]
    y = symlog(x.to(torch.float32)).clamp(-SYM_MAX, SYM_MAX)
    u = (y - grid[0]) / w
    lo = torch.clamp(torch.floor(u).long(), 0, N_BINS - 2)
    frac = (u - lo.to(torch.float32)).clamp(0.0, 1.0)
    out = torch.zeros(x.shape[0], N_BINS, dtype=x.dtype, device=x.device)
    out.scatter_add_(-1, lo.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    out.scatter_add_(-1, (lo + 1).clamp(max=N_BINS - 1).unsqueeze(-1),
                     frac.unsqueeze(-1))
    return out


def tau_sinusoidal(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of tau in [0, TAU_MAX_S]; [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32,
                                   device=tau.device)
                      * (-math.log(TAU_MAX_S) / max(half - 1, 1)))
    ang = (tau.to(torch.float32) / TAU_MAX_S).unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


@dataclass
class V0ValueConfig:
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
    ctx_raw: int = CTX_V0_RAW          # featurize context + pre_fix flag
    max_cavs: int = 64
    max_walls: int = 4
    n_policies: int = N_POLICIES_MAX
    tau_emb_dim: int = 64
    head_hidden: int = 256
    n_bins: int = N_BINS
    sym_max: float = SYM_MAX
    enable_p_closing_pair: bool = False   # no V0 labels -- see module docstring
    enable_future_min_ttc_q05: bool = False

    def to_json(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, d: dict) -> "V0ValueConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class _Head(nn.Module):
    """Conditioned MLP head: concat(pooled, tau, policy) -> out."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class V0ValueNet(nn.Module):
    """State s + (tau, policy, pre_fix) -> value-head distributions.

    forward(batch) returns dict of logits:
        v_delay [B, K], v_exp [B, K], v_brake [B, K], p_coll [B]
    (+ p_closing_pair [B], future_min_ttc_q05 [B, n_q] when enabled).
    Distribution heads are symlog twohot logits; decode helpers live in
    distributions.py.
    """

    def __init__(self, cfg: V0ValueConfig | None = None):
        super().__init__()
        cfg = self.cfg = cfg or V0ValueConfig()
        d = cfg.d_model
        self.field_encoder = _LaneField(cfg)
        self.cav_proj = nn.Linear(cfg.cav_raw, d)
        self.wall_proj = nn.Linear(cfg.wall_raw, d)
        self.demand_proj = nn.Linear(cfg.demand_raw, d)
        self.ctx_proj = nn.Linear(cfg.ctx_raw, d)
        # token types: 0..2 lanes, 3 cavs, 4 walls, 5 demand, 6 ctx,
        # 7 tau, 8 policy
        self.type_emb = nn.Embedding(9, d)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(block, cfg.n_layers,
                                             enable_nested_tensor=False)
        self.tau_proj = nn.Linear(cfg.tau_emb_dim, d)
        self.policy_emb = nn.Embedding(cfg.n_policies, d)

        cond = 3 * d                          # pooled + tau + policy
        self.head_delay = _Head(cond, cfg.head_hidden, cfg.n_bins)
        self.head_exp = _Head(cond, cfg.head_hidden, cfg.n_bins)
        self.head_brake = _Head(cond, cfg.head_hidden, cfg.n_bins)
        self.head_coll = _Head(cond, cfg.head_hidden, 1)
        self.head_closing = (_Head(cond, cfg.head_hidden, 1)
                             if cfg.enable_p_closing_pair else None)
        self.head_ttc_q05 = (_Head(cond, cfg.head_hidden, 1)
                             if cfg.enable_future_min_ttc_q05 else None)

    def forward(self, batch: dict) -> dict:
        dev = self.type_emb.weight.device
        tens = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray) and not v.flags.writeable:
                v = v.copy()      # np.load/broadcast_to views: torch warns
            if k in ("policy",):
                tens[k] = torch.as_tensor(v, dtype=torch.long, device=dev)
            elif k in ("tau",):
                tens[k] = torch.as_tensor(v, dtype=torch.float32, device=dev)
            else:
                tens[k] = torch.as_tensor(v, dtype=torch.float32, device=dev)
        lane_bins = tens["lane_bins"]                      # [B,3,L,3]
        lane_mask = tens["lane_mask"] * tens["lane_geom"]  # valid & physical
        cav = tens["cav_tokens"]
        cav_mask = tens["cav_mask"]
        wall = tens["wall_tokens"]
        wall_mask = tens["wall_mask"]
        _check_input_widths(cav, wall, self.cfg)
        demand = tens["demand"]
        ctx = tens["context"]                              # [B, ctx_raw]
        b = lane_bins.shape[0]
        d = self.cfg.d_model

        lane_tok = self.field_encoder(lane_bins, lane_mask)   # [B,3,d]
        cav_tok = self.cav_proj(cav)
        wall_tok = self.wall_proj(wall)
        dem_tok = self.demand_proj(demand).unsqueeze(1)
        ctx_tok = self.ctx_proj(ctx).unsqueeze(1)
        tau_tok = self.tau_proj(tau_sinusoidal(
            tens["tau"], self.cfg.tau_emb_dim)).unsqueeze(1)
        pol_tok = self.policy_emb(
            tens["policy"].clamp(0, self.cfg.n_policies - 1)).unsqueeze(1)

        tokens = torch.cat([lane_tok, cav_tok, wall_tok, dem_tok, ctx_tok,
                            tau_tok, pol_tok], dim=1)
        n_lane, n_cav = lane_tok.shape[1], cav_tok.shape[1]
        n_wall = wall_tok.shape[1]
        type_ids = torch.cat([
            torch.arange(n_lane),
            torch.full((n_cav,), 3), torch.full((n_wall,), 4),
            torch.full((1,), 5), torch.full((1,), 6),
            torch.full((1,), 7), torch.full((1,), 8)]).to(dev)
        tokens = tokens + self.type_emb(type_ids)

        n_cond = 4                          # demand, ctx, tau, policy tokens
        pad = torch.cat([torch.zeros(b, n_lane + n_wall + n_cond,
                                     dtype=torch.bool, device=dev),
                         cav_mask <= 0], dim=1)          # True = pad
        h = self.encoder(tokens, src_key_padding_mask=pad)
        pooled = h.masked_fill(pad.unsqueeze(-1), 0.0).sum(dim=1) \
            / (~pad).sum(dim=1, keepdim=True).clamp(min=1)

        z = torch.cat([pooled,
                       tau_tok.squeeze(1), pol_tok.squeeze(1)], dim=-1)
        out = dict(
            v_delay=self.head_delay(z),
            v_exp=self.head_exp(z),
            v_brake=self.head_brake(z),
            p_coll=self.head_coll(z).squeeze(-1))
        if self.head_closing is not None:
            out["p_closing_pair"] = self.head_closing(z).squeeze(-1)
        if self.head_ttc_q05 is not None:
            out["future_min_ttc_q05"] = self.head_ttc_q05(z).squeeze(-1)
        return out

    # --------------------------------------------------------- inference --
    def point_estimates(self, out: dict) -> dict:
        """Twohot logits -> real-space posterior-mean point estimates."""
        return {k: decode_mean(out[k]) for k in ("v_delay", "v_exp",
                                                 "v_brake")}


class _LaneField(nn.Module):
    """Shared-weight Conv1d encoder over each lane's bin sequence
    (part4/partB LaneFieldEncoder conventions)."""

    def __init__(self, cfg: V0ValueConfig):
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
        h = self.conv(x * m) * m
        denom = m.sum(dim=2).clamp(min=1.0)
        pooled = (h * m).sum(dim=2) / denom
        return self.proj(pooled).reshape(b, nl, -1)


# v0.4a policy classes (WD-01-PB-V04A): the two legacy names keep their
# ids 0/1 (v0.3.x checkpoints and pickles stay valid); the true-policy
# classes are appended.  The GBM value model one-hot encodes over this
# tuple; the legacy nets never see the extra ids (N_POLICIES_MAX is
# unchanged).
POLICIES_V04A = POLICIES + ("native_fleet", "pi_fleet", "fixed_mix")


def policy_id(name: str) -> int:
    try:
        return POLICIES.index(name)
    except ValueError:
        pass
    try:
        return POLICIES_V04A.index(name)
    except ValueError:
        raise ValueError(
            f"unknown policy {name!r}; known: {POLICIES_V04A} "
            "(extend POLICIES + n_policies together)") from None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------- twohot decode (torch) --

def decode_mean(logits: torch.Tensor) -> torch.Tensor:
    """Twohot logits -> real-space expectation (training-time metric)."""
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    return probs @ bin_centers().to(probs.device)


def nll_twohot(logits: torch.Tensor, target_probs: torch.Tensor,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy of twohot targets; weighted mean over the batch."""
    nll = -(target_probs.to(torch.float32)
            * torch.log_softmax(logits.to(torch.float32), dim=-1)).sum(-1)
    if weight is None:
        return nll.mean()
    w = weight.to(torch.float32)
    return (nll * w).sum() / w.sum().clamp(min=1e-8)


# ===========================================================================
# v0.3 net (BRIEF_B3_V03.md A1-A5, C13; B3b).  The v0.2 surface above is
# kept verbatim: train.py (B3a owns its persistence edits), evaluate.py's
# shipped functions and the v0.2 tests keep working against it.  The v0.3
# net trains on PER-TAU-NORMALISED targets (A1) over a grid covering only
# the realistic normalised range (A2), conditions every head layer on tau
# with FiLM plus a learned per-tau output scale/bias (A3), consumes
# TRAIN-standardised features and a summary-feature token (A4, built in
# data.py), and carries an optional paired Delta head (C13).
# ===========================================================================

SYM_MIN_V03 = -2.0        # symlog grid over the REALISTIC range only (A2):
SYM_MAX_V03 = 12.0        # normalised y in [e^-2-1, e^12-1] ~ [-6.4, 1.6e5]
N_BINS_V03 = 255
SUMMARY_DIM = 40          # diag_tau30.py summary features (data.py)


def twohot_encode_grid(x: torch.Tensor, sym_min: float, sym_max: float,
                       n_bins: int) -> torch.Tensor:
    """Twohot encoding on an arbitrary symlog grid (shared v0.2/v0.3).

    Out-of-place (one_hot arithmetic, not scatter_add_) so the same code
    runs under torch.func.vmap in the concurrent v0.3 trainer (B10)."""
    x = x.reshape(-1)
    grid = torch.linspace(sym_min, sym_max, n_bins, dtype=torch.float32,
                          device=x.device)
    w = grid[1] - grid[0]
    y = symlog(x.to(torch.float32)).clamp(sym_min, sym_max)
    u = (y - grid[0]) / w
    lo = torch.clamp(torch.floor(u).long(), 0, n_bins - 2)
    frac = (u - lo.to(torch.float32)).clamp(0.0, 1.0)
    oh = torch.nn.functional.one_hot
    return (oh(lo, n_bins).to(torch.float32) * (1.0 - frac).unsqueeze(-1)
            + oh((lo + 1).clamp(max=n_bins - 1), n_bins).to(torch.float32)
            * frac.unsqueeze(-1))


def twohot_encode_v03(x: torch.Tensor) -> torch.Tensor:
    """A2: twohot over the realistic normalised range [SYM_MIN, SYM_MAX]."""
    return twohot_encode_grid(x, SYM_MIN_V03, SYM_MAX_V03, N_BINS_V03)


class PmfGrid:
    """Bin geometry + pmf operations on a symlog grid (numpy, evaluation).

    The v0.3 grid is in NORMALISED units; every consumer that reports real
    units passes the per-row scale (the TRAIN per-tau median) explicitly.
    CRPS is LINEAR under joint scaling of distribution and observation:
    CRPS(F*c, y*c) = c*E|X-y| - 0.5*c*E|X-X'| = c * CRPS(F, y).
    """

    def __init__(self, sym_min: float = SYM_MIN_V03,
                 sym_max: float = SYM_MAX_V03, n_bins: int = N_BINS_V03):
        grid = torch.linspace(sym_min, sym_max, n_bins, dtype=torch.float64)
        step = float(grid[1] - grid[0])
        edges_sym = torch.cat(
            [grid - step / 2, torch.tensor([float(grid[-1]) + step / 2])])
        self.sym_min, self.sym_max, self.n_bins = sym_min, sym_max, n_bins
        self.centers = inv_symlog(grid).numpy().astype(np.float64)
        self.edges = inv_symlog(edges_sym).numpy().astype(np.float64)
        self.pair = np.abs(self.centers[:, None] - self.centers[None, :])

    # -- point forecasts (A5: MEDIAN of the predictive distribution) ------
    def quantile(self, probs: np.ndarray, q: float) -> np.ndarray:
        """Quantile via linear interpolation of the bin CDF; [N,K]->[N]."""
        cdf = np.cumsum(probs.astype(np.float64), axis=1)
        out = np.empty(cdf.shape[0])
        k = self.n_bins
        for i in range(cdf.shape[0]):
            j = int(np.searchsorted(cdf[i], q, side="left"))
            j = min(max(j, 1), k - 1)
            f_lo, f_hi = cdf[i, j - 1], cdf[i, j]
            frac = 0.0 if f_hi <= f_lo else (q - f_lo) / (f_hi - f_lo)
            out[i] = self.centers[j - 1] + frac * (self.centers[j]
                                                   - self.centers[j - 1])
        return out

    def median(self, probs: np.ndarray) -> np.ndarray:
        return self.quantile(probs, 0.5)

    def quantile_torch(self, probs: torch.Tensor, q: float) -> torch.Tensor:
        """Torch quantile (A5 median decode); pmf [B,K] -> [B]."""
        centers = torch.as_tensor(self.centers, dtype=probs.dtype,
                                  device=probs.device)
        cdf = torch.cumsum(torch.softmax(probs.to(torch.float32), dim=-1),
                           -1)
        idx = torch.clamp(torch.searchsorted(
            cdf, torch.full((cdf.shape[0], 1), q, device=cdf.device)
        ).squeeze(-1), 1, cdf.shape[-1] - 1)
        rows = torch.arange(cdf.shape[0], device=cdf.device)
        f_hi = cdf[rows, idx]
        f_lo = cdf[rows, idx - 1]
        frac = torch.where(f_hi > f_lo,
                           (q - f_lo) / (f_hi - f_lo).clamp(min=1e-12),
                           torch.zeros_like(f_hi))
        return (centers[idx - 1] * (1 - frac)
                + centers[idx] * frac).to(probs.dtype)

    # -- scoring -----------------------------------------------------------
    def crps_norm(self, probs: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
        """Exact CRPS in NORMALISED units; [N,K],[N] -> [N]."""
        p = probs.astype(np.float64)
        e_abs = np.abs(self.centers[None, :] - np.asarray(y_norm)[:, None])
        term1 = (e_abs * p).sum(1)
        term2 = 0.5 * np.einsum("nk,kl,nl->n", p, self.pair, p)
        return term1 - term2

    def crps_real(self, probs: np.ndarray, y_norm: np.ndarray,
                  scale: np.ndarray) -> np.ndarray:
        """CRPS after de-normalisation by the per-row scale (real units).

        Both terms scale LINEARLY in the scale c (E|cX-cy| = c E|X-y| and
        E|cX-cX'| = c E|X-X'|), so crps_real = c * crps_norm -- never
        c^2 on the pairwise term (a c^2 factor dominates at per-tau
        scales ~1e3 and drives CRPS negative, which corrupts OOB
        selection and every report table)."""
        t1, t2 = self._crps_terms(probs, y_norm)
        s = np.asarray(scale, dtype=np.float64)
        return s * (t1 - t2)

    def _crps_terms(self, probs, y_norm):
        p = probs.astype(np.float64)
        e_abs = np.abs(self.centers[None, :] - np.asarray(y_norm)[:, None])
        term1 = (e_abs * p).sum(1)
        term2 = 0.5 * np.einsum("nk,kl,nl->n", p, self.pair, p)
        return term1, term2

    def pit(self, probs: np.ndarray, y_norm: np.ndarray,
            seed: int = 20260928) -> np.ndarray:
        """Randomized PIT in (0, 1) for discrete predictives."""
        p = probs.astype(np.float64)
        cdf = np.cumsum(p, axis=1)
        rows = np.arange(p.shape[0])
        k = np.clip(np.searchsorted(self.centers, y_norm, side="right"), 0,
                    self.n_bins - 1)
        lo = np.where(k > 0, cdf[rows, np.clip(k - 1, 0, self.n_bins - 1)],
                      0.0)
        hi = cdf[rows, k]
        u = np.random.default_rng(seed).uniform(0.0, 1.0, size=p.shape[0])
        return np.clip(lo + u * (hi - lo), 1e-6, 1.0 - 1e-6)


@dataclass
class V03ValueConfig:
    """v0.3 net configuration (A2-A4, C13)."""
    d_model: int = 192
    nhead: int = 4
    n_layers: int = 3
    dim_ff: int = 512
    dropout: float = 0.0                 # ablation variant: 0.1
    conv_channels: int = 32
    conv_kernel: int = 5
    lmax_bins: int = LMAX_BINS
    cav_raw: int = CAV_RAW
    wall_raw: int = WALL_RAW
    demand_raw: int = DEMAND_RAW
    ctx_raw: int = CTX_RAW               # featurize context (no pre_fix:
    summary_dim: int = SUMMARY_DIM       #   pre_fix lives in the summary)
    max_cavs: int = 64
    max_walls: int = 4
    n_policies: int = N_POLICIES_MAX
    tau_emb_dim: int = 64
    head_hidden: int = 256
    head_layers: int = 2
    n_bins: int = N_BINS_V03
    sym_min: float = SYM_MIN_V03
    sym_max: float = SYM_MAX_V03
    enable_delta: bool = True            # C13 paired Delta head
    delta_hidden: int = 128
    delta_layers: int = 2

    def to_json(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, d: dict) -> "V03ValueConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class FiLMHead(nn.Module):
    """A3: every hidden layer is tau-conditioned (FiLM) and the output
    carries a learned per-tau scale and bias.

    logits = softplus(out_scale(tau)) * W h_L + out_bias(tau);  the bias is
    zero-initialised and the scale starts at softplus(0) ~ 0.69, so at init
    the head is a plain linear map.

    prior_center/prior_sigma (distribution heads only): the output bias is
    initialised to a Gaussian bump in SYMLOG space centred on the typical
    normalised target (~1, A1), so the initial pmf is already concentrated
    near the realistic range instead of uniform over a grid that reaches
    e^12 -- without this the decoded mean at init is ~1e3-1e4 and the
    consistency/TD penalties swamp the MC gradient for hundreds of steps.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 tau_dim: int, n_layers: int = 2,
                 prior_center: float | None = None,
                 prior_sigma: float = 2.0,
                 sym_min: float = -2.0, sym_max: float = 12.0):
        super().__init__()
        self.hidden = hidden
        nlin = max(1, n_layers)
        self.fcs = nn.ModuleList(
            [nn.Linear(in_dim, hidden)] +
            [nn.Linear(hidden, hidden) for _ in range(nlin - 1)])
        self.films = nn.ModuleList(
            [nn.Linear(tau_dim, 2 * hidden) for _ in range(nlin)])
        self.out = nn.Linear(hidden, out_dim)
        self.out_scale = nn.Linear(tau_dim, 1)
        self.out_bias = nn.Linear(tau_dim, out_dim)
        nn.init.zeros_(self.out_bias.weight)
        nn.init.zeros_(self.out_bias.bias)
        if prior_center is not None:
            with torch.no_grad():
                grid = torch.linspace(sym_min, sym_max, out_dim)
                self.out.bias.add_(
                    -0.5 * ((grid - prior_center) / prior_sigma) ** 2)

    def forward(self, z: torch.Tensor, tau_emb: torch.Tensor) -> torch.Tensor:
        h = z
        for fc, film in zip(self.fcs, self.films):
            h = torch.nn.functional.gelu(fc(h))
            gb = film(tau_emb)
            h = gb[..., :self.hidden] * h + gb[..., self.hidden:]
        scale = torch.nn.functional.softplus(self.out_scale(tau_emb))
        return scale * self.out(h) + self.out_bias(tau_emb)


class V03ValueNet(nn.Module):
    """v0.3 value net: standardised inputs + summary token (A4), shared
    transformer encoder (B9: damping + switching-mix rows together), FiLM
    heads with per-tau output scale/bias (A3), twohot over the realistic
    normalised range (A2), median point decode (A5), and an optional
    paired Delta head (C13) predicting the damping-fleet minus
    switching-mix delay contrast from the state (v0.3.2 fix #13: the
    '__native' campaign arms are random per-CAV switching schedules, not
    a fleet-native policy -- the head carries the switching-mix
    contrast, no fleet-native claim).

    forward(batch) -> v_delay/v_exp/v_brake [B,K] twohot logits (targets
    are per-tau-normalised, A1, by the trainer), p_coll [B] logits and
    delta [B] (when enabled): symlog-space forecast of the scaled policy
    contrast (scaled by the TRAIN per-tau delta stats, data.py).
    """

    def __init__(self, cfg: V03ValueConfig | None = None):
        super().__init__()
        cfg = self.cfg = cfg or V03ValueConfig()
        d = cfg.d_model
        self.field_encoder = _LaneField(cfg)
        self.cav_proj = nn.Linear(cfg.cav_raw, d)
        self.wall_proj = nn.Linear(cfg.wall_raw, d)
        self.demand_proj = nn.Linear(cfg.demand_raw, d)
        self.ctx_proj = nn.Linear(cfg.ctx_raw, d)
        self.summary_proj = nn.Linear(cfg.summary_dim, d)
        # token types: 0..2 lanes, 3 cavs, 4 walls, 5 demand, 6 ctx,
        # 7 tau, 8 policy, 9 summary
        self.type_emb = nn.Embedding(10, d)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(block, cfg.n_layers,
                                             enable_nested_tensor=False)
        self.tau_proj = nn.Linear(cfg.tau_emb_dim, d)
        self.policy_emb = nn.Embedding(cfg.n_policies, d)

        cond = 3 * d                          # pooled + tau + policy
        te = cfg.tau_emb_dim
        # symlog-space init prior at the typical NORMALISED target (~1,
        # A1: y / per-tau median): initial pmfs concentrate near the
        # realistic range instead of uniform over the full grid
        prior_c = float(torch.log1p(torch.tensor(1.0)))   # symlog(1.0)
        self.head_delay = FiLMHead(cond, cfg.head_hidden, cfg.n_bins, te,
                                   cfg.head_layers, prior_center=prior_c,
                                   sym_min=cfg.sym_min,
                                   sym_max=cfg.sym_max)
        self.head_exp = FiLMHead(cond, cfg.head_hidden, cfg.n_bins, te,
                                 cfg.head_layers, prior_center=prior_c,
                                 sym_min=cfg.sym_min, sym_max=cfg.sym_max)
        self.head_brake = FiLMHead(cond, cfg.head_hidden, cfg.n_bins, te,
                                   cfg.head_layers, prior_center=prior_c,
                                   sym_min=cfg.sym_min,
                                   sym_max=cfg.sym_max)
        self.head_coll = FiLMHead(cond, cfg.head_hidden, 1, te,
                                  cfg.head_layers)
        self.head_delta = None
        if cfg.enable_delta:
            layers = [nn.Linear(d + te, cfg.delta_hidden),
                      nn.GELU()]
            for _ in range(cfg.delta_layers - 1):
                layers += [nn.Linear(cfg.delta_hidden, cfg.delta_hidden),
                           nn.GELU()]
            layers += [nn.Linear(cfg.delta_hidden, 1)]
            self.head_delta = nn.Sequential(*layers)

    def _tensors(self, batch: dict) -> dict:
        dev = self.type_emb.weight.device
        tens = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray) and not v.flags.writeable:
                v = v.copy()
            if k == "policy":
                tens[k] = torch.as_tensor(v, dtype=torch.long, device=dev)
            elif k == "tau":
                tens[k] = torch.as_tensor(v, dtype=torch.float32, device=dev)
            else:
                tens[k] = torch.as_tensor(v, dtype=torch.float32, device=dev)
        return tens

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encoder -> (pooled [B,d], tau_emb [B,te])."""
        tens = self._tensors(batch)
        lane_bins = tens["lane_bins"]
        lane_mask = tens["lane_mask"] * tens["lane_geom"]
        _check_input_widths(tens["cav_tokens"], tens["wall_tokens"],
                            self.cfg)
        b = lane_bins.shape[0]

        lane_tok = self.field_encoder(lane_bins, lane_mask)
        cav_tok = self.cav_proj(tens["cav_tokens"])
        wall_tok = self.wall_proj(tens["wall_tokens"])
        dem_tok = self.demand_proj(tens["demand"]).unsqueeze(1)
        ctx_tok = self.ctx_proj(tens["context"]).unsqueeze(1)
        sum_tok = self.summary_proj(tens["summary"]).unsqueeze(1)
        tau_emb = tau_sinusoidal(tens["tau"], self.cfg.tau_emb_dim)
        tau_tok = self.tau_proj(tau_emb).unsqueeze(1)
        pol_tok = self.policy_emb(
            tens["policy"].clamp(0, self.cfg.n_policies - 1)).unsqueeze(1)

        tokens = torch.cat([lane_tok, cav_tok, wall_tok, dem_tok, ctx_tok,
                            tau_tok, pol_tok, sum_tok], dim=1)
        n_lane, n_cav = lane_tok.shape[1], cav_tok.shape[1]
        n_wall = wall_tok.shape[1]
        type_ids = torch.cat([
            torch.arange(n_lane),
            torch.full((n_cav,), 3), torch.full((n_wall,), 4),
            torch.full((1,), 5), torch.full((1,), 6),
            torch.full((1,), 7), torch.full((1,), 8),
            torch.full((1,), 9)]).to(tokens.device)
        tokens = tokens + self.type_emb(type_ids)

        n_cond = 5                     # demand, ctx, tau, policy, summary
        pad = torch.cat([torch.zeros(b, n_lane + n_wall + n_cond,
                                     dtype=torch.bool, device=tokens.device),
                         tens["cav_mask"] <= 0], dim=1)   # True = pad
        h = self.encoder(tokens, src_key_padding_mask=pad)
        pooled = h.masked_fill(pad.unsqueeze(-1), 0.0).sum(dim=1) \
            / (~pad).sum(dim=1, keepdim=True).clamp(min=1)
        return pooled, tau_emb

    def forward(self, batch: dict) -> dict:
        pooled, tau_emb = self.encode(batch)
        pol_tok = self.policy_emb(
            self._tensors(batch)["policy"].clamp(
                0, self.cfg.n_policies - 1))
        # cond = pooled + PROJECTED tau + policy (3*d, as v0.2); the raw
        # tau embedding feeds the FiLM/scale/bias maps separately
        z = torch.cat([pooled, self.tau_proj(tau_emb), pol_tok], dim=-1)
        out = dict(
            v_delay=self.head_delay(z, tau_emb),
            v_exp=self.head_exp(z, tau_emb),
            v_brake=self.head_brake(z, tau_emb),
            p_coll=self.head_coll(z, tau_emb).squeeze(-1))
        if self.head_delta is not None:
            out["delta"] = self.head_delta(
                torch.cat([pooled, tau_emb], dim=-1)).squeeze(-1)
        return out

    # --------------------------------------------------------- inference --
    def point_estimates(self, out: dict) -> dict:
        """A5: median of the predictive distribution (normalised units)."""
        grid = PmfGrid(self.cfg.sym_min, self.cfg.sym_max, self.cfg.n_bins)
        return {k: grid.quantile_torch(out[k], 0.5)
                for k in ("v_delay", "v_exp", "v_brake")}
