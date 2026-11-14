"""Exact fast path for the A11 HDV residual net during closed-loop rollouts.

Two mathematically exact rewrites of ``hdv.net.HdvResidualNet.forward``
(same parameters, same float64 dtype), used ONLY for inference speed:

1. **slot compaction** -- the net is permutation-equivariant over slots and
   masked slots neither attend nor are attended (``_neighbourhood`` masks
   them, outputs are multiplied by the mask), so running it on the union of
   occupied slots across the batch and scattering back gives the same
   per-slot outputs (unoccupied slots are 0 either way);
2. **sparse pair bias** -- the per-pair attention bias MLP is evaluated only
   on pairs inside the neighbourhood mask; every other logit is set to -inf
   by the original code before the softmax, so its bias value never matters.

Parity with the original forward is unit-tested (max abs diff ~1e-12).
"""
from __future__ import annotations

import math

import numpy as np
import torch


class FastHdv(torch.nn.Module):
    def __init__(self, net, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        if self.device.type != "cpu":
            import copy

            net = copy.deepcopy(net).to(self.device)
        self.net = net
        self.cfg = net.cfg

    @torch.no_grad()
    def forward(self, kin, kin_mask, *, cav_flag=None):
        net = self.net
        c = net.cfg
        if c.use_cav_flag or c.zero_residual:
            return net(kin, kin_mask, cav_flag=cav_flag)
        mask_full = kin_mask.bool()
        b_n, k_n = mask_full.shape
        idx = torch.nonzero(mask_full.any(dim=0), as_tuple=False).reshape(-1)
        out_delta = torch.zeros(b_n, k_n, dtype=kin.dtype)
        out_sigma = torch.zeros(b_n, k_n, dtype=kin.dtype)
        out_lane = torch.zeros(b_n, k_n, 3, dtype=kin.dtype)
        if idx.numel() == 0:
            return {"accel_delta_mps2": out_delta, "accel_log_sigma": out_sigma,
                    "lane_logit_delta": out_lane, "attn": None}
        kin_c = kin[:, idx].to(self.device)
        mask = mask_full[:, idx].to(self.device)
        net.assert_no_cav_flag(kin_c)
        geo = net._geometry(kin_c[..., :kin_c.shape[-1]])
        x = net.embed(net.norm(kin_c)) * mask.unsqueeze(-1).to(kin_c.dtype)
        for blk in net.blocks:
            a = _sparse_attention(blk.attn, blk.ln1(x), mask, geo)
            x = x + a
            x = x + blk.ffn(blk.ln2(x))
            x = x * mask.unsqueeze(-1).to(x.dtype)
        acc = net.accel(x)
        delta = c.accel_delta_scale_mps2 * torch.tanh(acc[..., 0])
        log_sigma = acc[..., 1].clamp(-6.0, 2.0)
        lane_delta = c.lane_logit_scale * net.lane(x)
        delta = delta * mask.to(delta.dtype)
        lane_delta = lane_delta * mask.unsqueeze(-1).to(lane_delta.dtype)
        out_delta[:, idx] = delta.cpu().to(kin.dtype)
        out_sigma[:, idx] = log_sigma.cpu().to(kin.dtype)
        out_lane[:, idx] = lane_delta.cpu().to(kin.dtype)
        return {"accel_delta_mps2": out_delta, "accel_log_sigma": out_sigma,
                "lane_logit_delta": out_lane, "attn": None}


def _sparse_attention(att, x, mask, geo):
    b, k, _ = x.shape
    q = att.q(x).view(b, k, att.h, att.dh).transpose(1, 2)
    kk = att.k(x).view(b, k, att.h, att.dh).transpose(1, 2)
    vv = att.v(x).view(b, k, att.h, att.dh).transpose(1, 2)
    logits = (q @ kk.transpose(-1, -2)) / math.sqrt(att.dh)
    neigh = att._neighbourhood(geo, mask)                     # [b, k, k]
    s, lane, v = geo["s"], geo["lane"], geo["v"]
    bi, ii, jj = torch.nonzero(neigh, as_tuple=True)
    ds = s[bi, ii] - s[bi, jj]
    dl = lane[bi, ii] - lane[bi, jj]
    dv = v[bi, ii] - v[bi, jj]
    same = (lane[bi, ii] == lane[bi, jj]) & (lane[bi, ii] >= 0)
    eye = ii == jj
    feats = torch.stack([ds / att.radius, dl.to(s.dtype) / 2.0, dv / 20.0,
                         ds.abs() / att.radius, same.to(s.dtype), eye.to(s.dtype)], dim=-1)
    bias = att.bias(feats)                                    # [n, h]
    full = torch.full((b, k, k, att.h), float("-inf"), dtype=logits.dtype,
                      device=logits.device)
    full[bi, ii, jj] = bias.to(logits.dtype)
    logits = logits + full.permute(0, 3, 1, 2)
    attn = torch.softmax(logits, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = (attn @ vv).transpose(1, 2).reshape(b, k, -1)
    return att.o(out) * mask.unsqueeze(-1).to(out.dtype)


def fast_bundle(bundle, device: str = "cpu"):
    """A shallow copy of ``bundle`` whose ``hdv`` uses :class:`FastHdv`."""
    from dataclasses import replace

    hdv = getattr(bundle, "hdv", None)
    if hdv is None or isinstance(hdv, FastHdv):
        return bundle
    return replace(bundle, hdv=FastHdv(hdv, device=device))


# ---------------------------------------------------------------- gap repair
@torch.no_grad()
def fast_project_same_lane_gaps(s_m, lane, v_mps, length_m, present, *, dt_s=0.5,
                                n_lanes=3, counts=None):
    """Exact numpy twin of ``rollout2.validity.project_same_lane_gaps``.

    Same per-lane front-to-back order (stable descending sort on the
    pre-projection positions, ties in ascending slot order -- Python's
    ``sort(reverse=True)`` is stable), same float64 arithmetic in the same
    sequence; only the tensor element indexing is replaced by numpy arrays.
    """
    from model.rollout import ValidityCounts

    counts = counts if counts is not None else ValidityCounts()
    s_out = s_m.clone()
    v_out = v_mps.clone()
    s_np = s_out.numpy()
    v_np = v_out.numpy()
    lane_np = lane.detach().to(torch.int64).numpy()
    pres_np = present.detach().to(torch.bool).numpy()
    len_np = length_m.detach().to(torch.float64).numpy()
    b_n, _k = s_np.shape
    n_proj = 0
    for b in range(b_n):
        for ln in range(n_lanes):
            idx = np.flatnonzero(pres_np[b] & (lane_np[b] == ln))
            if idx.size < 2:
                continue
            order = idx[np.argsort(-s_np[b, idx], kind="stable")]
            for front, back in zip(order[:-1].tolist(), order[1:].tolist()):
                min_gap = 0.5 * (float(len_np[b, front]) + float(len_np[b, back]))
                limit = float(s_np[b, front]) - min_gap
                old_s = float(s_np[b, back])
                if old_s > limit:
                    s_np[b, back] = limit
                    vmax = max(0.0, (limit - old_s) / dt_s + float(v_np[b, back]))
                    v_np[b, back] = min(float(v_np[b, back]), vmax)
                    n_proj += 1
    counts.gap_projection += n_proj
    return s_out, v_out, counts




def install_fast_gap_projection() -> None:
    """Point ``rollout2.step``'s gap repair at the exact numpy twin."""
    import rollout2.step as step_mod

    step_mod.project_same_lane_gaps = fast_project_same_lane_gaps
