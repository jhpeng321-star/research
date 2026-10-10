"""WM-N1 autoregressive K-sample rollout with the physical-validity layer.

Design ``WM_N1_DESIGN.md`` section 2.2 (stochastic 0.5 s step), 2.3
(arrivals/insertions) and 2.5 (validity layer).  The rollout re-encodes every
step, samples the step decoder, integrates position with the clipped
acceleration (trapezoidal on speed, ``v_new = max(0, v + a*dt)``,
``s_new = s + 0.5*(v + v_new)*dt``; design section 2.2), projects each step
through the validity layer and inserts arrivals; it is deterministic under a
fixed seed.

The validity layer guarantees:

* acceleration clipped to ``[-9, +3]`` m/s^2 (counted);
* no negative speed (counted);
* a lane change is accepted only at in-bounds lanes and with target gap >= 0
  (rejected otherwise, counted);
* a same-lane speed projection keeps the rear bumper gap >= 0 (counted).

Collisions remain possible and are not silently fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import constants as C


# ----------------------------------------------------------- validity state -
@dataclass
class ValidityCounts:
    accel_clip: int = 0
    speed_clip: int = 0
    lane_reject: int = 0
    gap_projection: int = 0

    def add(self, other: "ValidityCounts") -> None:
        self.accel_clip += other.accel_clip
        self.speed_clip += other.speed_clip
        self.lane_reject += other.lane_reject
        self.gap_projection += other.gap_projection

    def as_dict(self) -> dict:
        return dict(accel_clip=self.accel_clip, speed_clip=self.speed_clip,
                    lane_reject=self.lane_reject,
                    gap_projection=self.gap_projection)


def _lane_targets(lane: torch.Tensor, cls: torch.Tensor):
    """Return (target lane, valid) for each vehicle, honouring lane bounds."""
    tgt = lane.clone()
    valid = torch.zeros_like(lane, dtype=torch.bool)
    left = cls == C.LANE_LEFT_CLS
    right = cls == C.LANE_RIGHT_CLS
    ok_left = left & (lane == 1)                 # main_right -> main_left
    tgt = torch.where(ok_left, torch.zeros_like(tgt), tgt)
    valid |= ok_left
    r0 = right & (lane == 0)                     # main_left -> main_right
    r1 = right & (lane == 1)                     # main_right -> accel
    r2 = right & (lane == 2)                     # accel -> main_right (merge)
    tgt = torch.where(r0, torch.ones_like(tgt), tgt)
    tgt = torch.where(r1, torch.full_like(tgt, 2), tgt)
    tgt = torch.where(r2, torch.ones_like(tgt), tgt)
    valid |= r0 | r1 | r2
    return tgt, valid


def project_step(s, lane, v, length, mask, accel, lane_class, dt=C.DT_S):
    """One kinematic step plus the physical-validity-layer projection.

    Kinematics (design ``WM_N1_DESIGN.md`` section 2.2, "stochastic step
    decoder (dt = 0.5 s)") use the *clipped* acceleration ``a`` and a
    trapezoidal rule on speed (semi-implicit)::

        v_new = max(0, v + a * dt)
        s_new = s + 0.5 * (v + v_new) * dt

    which is exact for constant ``a`` when the speed does not clamp
    (``s + v*dt + 0.5*a*dt**2``), is unconditionally stable at the 0.5 s tick,
    and never moves a vehicle backwards once ``v == v_new == 0``.  Documented
    integrator choice: trapezoidal-on-speed over explicit or semi-implicit
    because it is second-order accurate and the executed ``a_used`` equals the
    closed-form acceleration whenever the speed does not clip.

    The validity layer is then layered *on top of* the integrated positions:
    lane changes are accepted only at in-bounds lanes with target bumper
    gap >= 0, and a same-lane projection keeps the rear bumper gap >= 0.

    Returns ``(s, lane, v, accel_used, counts)`` with all guarantees applied.
    Inputs are ``[B, K]`` tensors; lane changes and gap projection are exact
    per batch item.
    """
    counts = ValidityCounts()
    a = accel.clamp(C.A_MIN, C.A_MAX)
    counts.accel_clip = int((accel != a).sum().item())
    v_new = v + a * dt
    neg = v_new < 0
    counts.speed_clip = int(neg.sum().item())
    v_new = v_new.clamp(min=0.0)
    a_used = torch.where(a >= 0, a, (v_new - v) / dt)

    # trapezoidal integration on speed (semi-implicit, exact for constant a)
    s_new = s + 0.5 * (v + v_new) * dt
    lane_new = lane.clone()
    B, K = mask.shape
    for b in range(B):
        work = lane_new[b].clone()
        s_w = s_new[b].clone()
        mb = mask[b]
        tgt, valid = _lane_targets(work, lane_class[b])
        for i in range(K):
            if not bool(mb[i]) or int(lane_class[b, i]) == 0:
                continue
            if not bool(valid[i]):
                counts.lane_reject += 1
                continue
            L = int(tgt[i])
            if L < 0 or L >= C.N_LANES:
                counts.lane_reject += 1
                continue
            blocked = False
            for j in range(K):
                if j == i or not bool(mb[j]) or int(work[j]) != L:
                    continue
                min_gap = 0.5 * (float(length[b, i]) + float(length[b, j]))
                if abs(float(s_w[i]) - float(s_w[j])) < min_gap:
                    blocked = True
                    break
            if blocked:
                counts.lane_reject += 1
            else:
                work[i] = L
        lane_new[b] = work
        s_new[b] = s_w
    # same-lane gap projection (rear bumper gap >= 0)
    for b in range(B):
        for L in range(C.N_LANES):
            idx = [i for i in range(K)
                   if bool(mask[b, i]) and int(lane_new[b, i]) == L]
            idx.sort(key=lambda i: float(s_new[b, i]), reverse=True)
            for front, back in zip(idx[:-1], idx[1:]):
                min_gap = 0.5 * (float(length[b, front])
                                 + float(length[b, back]))
                limit = float(s_new[b, front]) - min_gap
                if float(s_new[b, back]) > limit:
                    old_s = float(s_new[b, back])
                    s_new[b, back] = limit
                    vmax = max(0.0, (limit - old_s) / dt + float(v_new[b, back]))
                    v_new[b, back] = min(float(v_new[b, back]), vmax)
                    counts.gap_projection += 1
    return s_new, lane_new, v_new, a_used, counts


# ------------------------------------------------------------- sampling -----
def _uniform(shape, generator, device):
    """Uniform ``[0, 1)`` on ``device`` (the generator may live on CPU)."""
    if generator is None or str(generator.device) == str(device):
        return torch.rand(shape, generator=generator, device=device)
    return torch.rand(shape, generator=generator).to(device)


def _sample_categorical(probs, generator):
    c = probs.clamp_min(0.0).cumsum(-1)
    c = c / c[..., -1:].clamp_min(1e-9)
    u = _uniform(probs.shape[:-1], generator, probs.device)
    return (u.unsqueeze(-1) > c).sum(-1).clamp(
        max=probs.shape[-1] - 1).long()


def _sample_gmm(step, generator):
    mu = step["accel_mu"]
    log_sigma = step["accel_log_sigma"]
    pi = F.softmax(step["accel_mix_logits"], dim=-1)
    comp = _sample_categorical(pi, generator)
    mu_sel = torch.gather(mu, -1, comp.unsqueeze(-1)).squeeze(-1)
    sig_sel = torch.exp(
        torch.gather(log_sigma, -1, comp.unsqueeze(-1)).squeeze(-1))
    if generator is None or str(generator.device) == str(mu_sel.device):
        eps = torch.randn(mu_sel.shape, generator=generator,
                          device=mu_sel.device)
    else:
        eps = torch.randn(mu_sel.shape, generator=generator).to(mu_sel.device)
    return mu_sel + sig_sel * eps


# ------------------------------------------------------------ feature glue --
def _assemble(veh, s, lane, v, a):
    veh = veh.clone()
    veh[..., C.S_DOWN] = s
    veh[..., C.LANE_LEFT] = (lane == 0).to(veh.dtype)
    veh[..., C.LANE_RIGHT] = (lane == 1).to(veh.dtype)
    veh[..., C.LANE_ACCEL] = (lane == 2).to(veh.dtype)
    veh[..., C.V_IDX] = v
    veh[..., C.A_IDX] = a
    return veh


def _insert_arrivals(veh, mask, s, lane, v, counts, speed_mu, speed_sigma, rng):
    """Insert sampled arrivals as new generic (is_cav=0) tokens.

    Only inserts into a free slot; the entry position is behind the current
    lane occupancy so the rear-bumper gap stays >= 0.  Returns the number
    actually inserted per entrance.
    """
    B = mask.shape[0]
    inserted = np.zeros((B, C.N_ENTRANCES), dtype=np.int64)
    for b in range(B):
        for e in range(C.N_ENTRANCES):
            n = int(counts[b, e])
            entry_lane = 1 if e == 0 else 2
            mu, sig = float(speed_mu[b, e]), float(speed_sigma[b, e])
            for _ in range(min(n, 3)):
                free = torch.nonzero(~mask[b], as_tuple=False)
                if free.numel() == 0:
                    break
                slot = int(free[0, 0])
                occ = [float(s[b, j]) for j in range(mask.shape[1])
                       if bool(mask[b, j]) and int(lane[b, j]) == entry_lane]
                pos = (min(occ) - C.NOMINAL_LENGTH_M
                       if occ else 0.0 - C.NOMINAL_LENGTH_M)
                spd = max(0.0, mu + sig * float(rng.standard_normal()))
                veh[b, slot] = 0.0
                veh[b, slot, C.LANE_LEFT + entry_lane] = 1.0
                veh[b, slot, C.V_IDX] = spd
                veh[b, slot, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
                veh[b, slot, C.ACT_UNKNOWN] = 1.0
                veh[b, slot, C.PRESENT_IDX] = 1.0
                mask[b, slot] = True
                s[b, slot] = pos
                lane[b, slot] = entry_lane
                v[b, slot] = spd
                inserted[b, e] += 1
    return inserted


# ----------------------------------------------------------------- rollout --
def _rollout_once(net, batch, n_steps, gen, rng):
    veh = batch["veh_step"].clone().float()
    mask = (batch["veh_mask"] > 0)
    plan = batch["plan"].float()
    wall = batch["wall"].float()
    wall_mask = batch["wall_mask"] > 0
    counter = batch["counter"].float()
    B, K, _ = veh.shape

    s = veh[..., C.S_DOWN].clone()
    lane = torch.where(
        veh[..., C.LANE_LEFT:C.LANE_ACCEL + 1].sum(-1) > 0,
        veh[..., C.LANE_LEFT:C.LANE_ACCEL + 1].argmax(-1).long(),
        torch.full((B, K), -1, dtype=torch.long, device=veh.device))
    v = veh[..., C.V_IDX].clone()
    length = veh[..., C.LENGTH_IDX].clone()
    a = torch.zeros_like(v)

    pos_hist = [s.clone()]
    vel_hist = [v.clone()]
    lane_hist = [lane.clone()]
    mask_hist = [mask.clone()]
    acc_hist = []
    arr_hist = []
    ins_hist = []
    totals = ValidityCounts()

    for _ in range(n_steps):
        vf = _assemble(veh, s, lane, v, a)
        out = net(veh_step=vf, veh_mask=mask, plan=plan, wall=wall,
                  wall_mask=wall_mask, counter=counter)
        accel = _sample_gmm(out["step"], gen)
        lane_cls = _sample_categorical(
            F.softmax(out["step"]["lane_logits"], dim=-1), gen)
        s, lane, v, a_used, counts = project_step(
            s, lane, v, length, mask, accel, lane_cls)
        totals.add(counts)
        # sample arrivals (negative binomial) and insert
        mu = out["arrival"]["count_mu"].detach()
        r = (F.softplus(out["arrival"]["count_log_alpha"].detach())
             + 1e-4)
        p = (r / (r + mu + 1e-6)).clamp(1e-6, 1 - 1e-6).cpu().numpy()
        counts_np = rng.negative_binomial(
            np.maximum(r.cpu().numpy(), 1e-3), p, size=(B, C.N_ENTRANCES))
        counts_np = np.clip(counts_np, 0, 3).astype(np.int64)
        sp_mu = out["arrival"]["speed_mu"].detach().cpu().numpy()
        sp_sigma = torch.exp(
            out["arrival"]["speed_log_sigma"].detach()).cpu().numpy()
        ins = _insert_arrivals(veh, mask, s, lane, v, counts_np, sp_mu,
                               sp_sigma, rng)
        length = veh[..., C.LENGTH_IDX]
        a = a_used
        pos_hist.append(s.clone())
        vel_hist.append(v.clone())
        lane_hist.append(lane.clone())
        mask_hist.append(mask.clone())
        acc_hist.append(a_used.clone())
        arr_hist.append(counts_np)
        ins_hist.append(ins)

    return dict(
        positions=torch.stack(pos_hist),
        speeds=torch.stack(vel_hist),
        lanes=torch.stack(lane_hist),
        masks=torch.stack(mask_hist),
        accel=torch.stack(acc_hist),
        arrivals=torch.tensor(np.stack(arr_hist), device=s.device),
        insertions=torch.tensor(np.stack(ins_hist), device=s.device),
        validity=totals)


def rollout(net, batch, *, n_steps, n_samples=1, seed=None, generator=None):
    """Autoregressive K-sample rollout (design 2.2/2.3/2.5).

    ``batch`` holds the six featurizer tensors.  Deterministic under a fixed
    ``seed`` (or a caller-supplied ``generator``).  Returns trajectories with
    shape ``[n_samples, n_steps+1, B, K]`` plus the per-sample validity counts.
    """
    was_training = net.training
    net.eval()
    if generator is None:
        try:
            device = next(net.parameters()).device
        except StopIteration:                            # pragma: no cover
            device = torch.device("cpu")
        if str(device) == "meta":                        # meta has no generator
            device = torch.device("cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(0 if seed is None else int(seed))
    rng = np.random.default_rng(0 if seed is None else int(seed))
    samples = []
    try:
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(_rollout_once(net, batch, n_steps,
                                             generator, rng))
    finally:
        if was_training:
            net.train()
    out = {}
    for key in ("positions", "speeds", "lanes", "masks", "accel",
                "arrivals", "insertions"):
        out[key] = torch.stack([sm[key] for sm in samples])
    out["validity"] = [sm["validity"] for sm in samples]
    return out
