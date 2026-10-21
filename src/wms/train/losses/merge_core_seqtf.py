"""Round-8 recorded-state sequence likelihood for the secure-gap merge core.

The old teacher-forced term scored every recorded ramp row independently.  This module
replaces it with one discrete-time survival likelihood per recorded ramp car on the
*recorded* states:

    S_q(t) = prod_{u < t} (1 - h_uq),      h_uq = sigmoid(margin_uq / temperature)
    event at e:    -logsumexp_q(log w_q + log S_q(e) + log h_eq)
    censored at c: -logsumexp_q(log w_q + sum_{u <= c} log(1 - h_uq))

with the persistent assertiveness ``alpha_q = exp(clamp(sigma * z_q, -8, 8))``
(population location mu = 0) integrated by Gauss-Hermite quadrature.  The normalised
posterior over the nodes is carried across truncated-BPTT chunks, so the chunk
contributions telescope to one mixture likelihood.

Scene geometry uses every present recorded neighbour in the target lane, CAVs
included; only HDV, non-entrant ramp cars are supervised.  The margin, missing-side
semantics and quadrature are shared with the runtime core (``_core_margin``,
``core_features``, ``gauss_hermite``) rather than re-implemented.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.model.merge_core import core_features
from wms.model.merge_frailty import gauss_hermite

from .merge_core import _core_margin
from .merge_first_passage import _first_index, _last_index

__all__ = ["build_recorded_scene", "seq_tf_chunk"]


def _scalar(x: Tensor) -> Tensor:
    """Collapse a per-step parameter view to the scalar shared by the recorded row."""
    if not torch.is_tensor(x):
        return torch.as_tensor(x)
    return x.mean() if x.numel() > 1 else x.reshape(())


def _car_sigma(sigma: Tensor, bsz: int, k: int) -> Tensor:
    """Per-car ``[B, 1, K]`` (or scalar) sigma for the ``[B, Q, K]`` quadrature."""
    sigma = torch.as_tensor(sigma)
    if sigma.dim() == 0:
        return sigma
    if sigma.dim() > 2:
        sigma = sigma.mean(dim=1)
    if sigma.shape != (bsz, k):
        sigma = sigma.reshape(bsz, k)
    return sigma.unsqueeze(1)


def _gather_step(mask: Tensor, index: Tensor) -> Tensor:
    """``mask[b, index[b, k], k]`` for an integer ``[B, K]`` index (clamped)."""
    n = mask.shape[1]
    safe = index.clamp(0, max(n - 1, 0)).unsqueeze(1)
    return mask.gather(1, safe).squeeze(1)


def build_recorded_scene(targets, *, merge_lane: int, zone_s0_m: float, zone_s1_m: float,
                         k: int, target_lane: int = 1) -> dict:
    """Detached recorded per-step tensors of the whole unroll (computed once)."""
    n = int(targets.n_steps)
    with torch.no_grad():
        rec = targets.states[:, : n + 1].float()
        kk = min(int(k), int(rec.shape[2]))
        rec = rec[:, :, :kk]
        onehot = rec[..., 1:4]
        lane = onehot.argmax(-1)
        lane = torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))
        cav = targets.is_cav[:, : n + 1, :kk].bool()
        loss_valid = targets.loss_valid[:, : n + 1, :kk].bool()
        present = loss_valid[:, :n]
        next_present = loss_valid[:, 1: n + 1]
        next_lane = lane[:, 1: n + 1]
        kin = rec[:, :n]
        own = present & ~cav[:, :n]
        ramp_mask = own & (lane[:, :n] == int(merge_lane))
        supervised = ramp_mask
        geometry = SimpleNamespace(merge_lane=int(merge_lane),
                                   merge_target_lane=int(target_lane))
        feat = core_features(kin, present, geometry)
        v = kin[..., 4]
        a = kin[..., 5]
        has_l = feat["has_l"].bool()
        has_f = feat["has_f"].bool()
        in_zone = (kin[..., 0] >= float(zone_s0_m)) & (kin[..., 0] <= float(zone_s1_m))
        feasible = present & (lane[:, :n] == int(merge_lane)) & in_zone

        event = supervised & next_present & (next_lane == int(target_lane))
        rec_event = _first_index(event)
        last_obs = _last_index(supervised & (lane[:, :n] == int(merge_lane)) & ~event)

        t_idx = torch.arange(n, device=rec.device).view(1, n, 1).expand_as(own)
        prev_own = torch.cat((torch.zeros_like(own[:, :1]), own[:, :-1]), 1)
        new_run = own & ~prev_own
        run_start = torch.cummax(torch.where(new_run, t_idx, torch.full_like(t_idx, -1)),
                                 dim=1).values
        run_start = torch.where(own, run_start, torch.full_like(run_start, -1))

        safe_event = rec_event.clamp(0, max(n - 1, 0)).unsqueeze(1)
        event_run = run_start.gather(1, safe_event).squeeze(1)
        safe_last = last_obs.clamp(0, max(n - 1, 0)).unsqueeze(1)
        last_run = run_start.gather(1, safe_last).squeeze(1)
        identity = torch.where(rec_event >= 0, event_run, last_run)
        same_car = run_start == identity.unsqueeze(1)
        entry = _first_index(supervised & same_car)
        identity = torch.where(entry >= 0, identity, torch.full_like(identity, -1))

        event_ok = (rec_event >= 0) & _gather_step(feasible & supervised & same_car,
                                                   rec_event)
        life = (supervised & feasible & same_car
                & (t_idx <= last_obs.unsqueeze(1))
                & ((rec_event < 0).unsqueeze(1) | (t_idx < rec_event.unsqueeze(1))))
        event_row = same_car & (t_idx == rec_event.unsqueeze(1)) & event_ok.unsqueeze(1)
        has_exposure = (life | event_row).any(1)

        scene = {
            "gap_l": feat["gap_l"].float(), "gap_f": feat["gap_f"].float(),
            "v": v, "v_l": feat["v_l"].float(), "v_f": feat["v_f"].float(),
            "a": a, "a_l": feat["a_l"].float(), "a_f": feat["a_f"].float(),
            "has_l": has_l, "has_f": has_f,
            "feasible": feasible, "supervised": supervised,
            "rec_event": rec_event, "last_obs": last_obs, "identity": identity,
            "entry": entry, "run_start": run_start, "has_exposure": has_exposure,
            "n_cars": float((identity >= 0).sum()),
            "k": kk, "n_steps": n, "dt_s": float(targets.dt_s),
        }
    return scene


def seq_tf_chunk(scene: dict, c0: int, c1: int, *, s0, T, b, tau_a, temperature, sigma,
                 n_quad: int, log_post: Optional[Tensor],
                 require_both: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor, dict]:
    """One chunk of the per-car recorded-state sequence NLL (persistent quadrature)."""
    c0, c1 = int(c0), int(c1)
    k = int(scene["k"])
    gap_l = scene["gap_l"]
    bsz = int(gap_l.shape[0])
    dtype, device = gap_l.dtype, gap_l.device
    n_quad = int(n_quad)
    nodes, weights = gauss_hermite(n_quad, dtype=dtype, device=device)
    logw = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
    identity = scene["identity"]
    prior = logw.view(1, -1, 1).expand(bsz, n_quad, k)
    if c1 <= c0:
        new_log_post = prior if log_post is None else log_post.detach()
        return (gap_l.new_zeros(()), gap_l.new_zeros(()), new_log_post,
                {"events": 0.0, "censored": 0.0, "scored_rows": 0.0,
                 "zero_exposure": 0.0, "unsupported_events": 0.0, "n_cars": 0.0})

    def cut(x: Tensor) -> Tensor:
        return x[:, c0:c1]

    feasible = cut(scene["feasible"])
    supervised = cut(scene["supervised"])
    same_car = cut(scene["run_start"]) == identity.unsqueeze(1)
    rec_event = scene["rec_event"]
    last_obs = scene["last_obs"]
    entry = scene["entry"]
    step = torch.arange(c0, c1, device=device).view(1, c1 - c0, 1)
    event_ok = (rec_event >= 0) & _gather_step(scene["feasible"] & scene["supervised"]
                                               & (scene["run_start"] == identity.unsqueeze(1)),
                                               rec_event)
    supported = feasible & supervised & same_car
    before_event = (rec_event < 0).unsqueeze(1) | (step < rec_event.unsqueeze(1))
    survival = supported & (step <= last_obs.unsqueeze(1)) & before_event
    at_event = supported & (step == rec_event.unsqueeze(1)) & event_ok.unsqueeze(1)

    sig = _car_sigma(sigma, bsz, k)
    if sig.dim() == 0:
        alpha = torch.exp((sig * nodes).clamp(-8.0, 8.0)).view(1, n_quad, 1, 1)
    else:
        alpha = torch.exp((sig * nodes.view(1, n_quad, 1)).clamp(-8.0, 8.0)).unsqueeze(2)

    if require_both is not None:
        require_both = cut(require_both.bool()).unsqueeze(1)
    s0c, Tc, bc, tauc = (_scalar(x).to(dtype=dtype, device=device)
                         for x in (s0, T, b, tau_a))
    temp = _scalar(temperature).to(dtype=dtype, device=device).clamp_min(1.0e-3)
    margin_q = _core_margin(
        alpha, cut(scene["gap_l"]).unsqueeze(1), cut(scene["gap_f"]).unsqueeze(1),
        cut(scene["v"]).unsqueeze(1), cut(scene["v_l"]).unsqueeze(1),
        cut(scene["v_f"]).unsqueeze(1), cut(scene["a"]).unsqueeze(1),
        cut(scene["a_l"]).unsqueeze(1), cut(scene["a_f"]).unsqueeze(1),
        cut(scene["has_l"]).unsqueeze(1), cut(scene["has_f"]).unsqueeze(1),
        s0c, Tc, bc, tauc, None, require_both)
    logits = margin_q / temp
    log_h = F.logsigmoid(logits)
    log_1mh = F.logsigmoid(-logits)

    g_chunk = ((survival.unsqueeze(1) * log_1mh).sum(dim=2)
               + (at_event.unsqueeze(1) * log_h).sum(dim=2))
    terminal = torch.where(rec_event >= 0, rec_event, last_obs)
    eligible = (identity >= 0) & (entry < c1) & (terminal >= c0)
    g_chunk = torch.where(eligible.unsqueeze(1), g_chunk, torch.zeros_like(g_chunk))

    if log_post is None:
        log_post_eff = prior
    else:
        log_post_eff = log_post.detach().to(dtype=dtype, device=device)
        previous = scene.get("_post_identity")
        if torch.is_tensor(previous) and previous.shape == identity.shape:
            changed = previous.to(device) != identity
            log_post_eff = torch.where(changed.unsqueeze(1), prior, log_post_eff)
    scene["_post_identity"] = identity.detach().clone()

    a_q = log_post_eff + g_chunk
    d = torch.logsumexp(a_q, dim=1)
    nll = -(d * eligible.to(d.dtype)).sum()
    new_log_post = (a_q - d.unsqueeze(1)).detach()

    with torch.no_grad():
        event_any = at_event.any(1)
        in_chunk_event = (rec_event >= c0) & (rec_event < c1) & (identity >= 0)
        censored = ((rec_event < 0) & (last_obs >= c0) & (last_obs < c1)
                    & (identity >= 0))
        rows = (survival | at_event).sum((1, 2))
        zero = (eligible & ~scene["has_exposure"].to(device)
                & (entry >= c0) & (entry < c1))
        metrics = {
            "events": float(event_any.sum()),
            "censored": float(censored.sum()),
            "scored_rows": float(rows.sum()),
            "zero_exposure": float(zero.sum()),
            "unsupported_events": float((in_chunk_event & ~event_ok).sum()),
            "n_cars": float(eligible.sum()),
        }
    n_cars = eligible.to(dtype).sum()
    return nll, n_cars, new_log_post, metrics
