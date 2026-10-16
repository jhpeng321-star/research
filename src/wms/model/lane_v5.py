"""LANEV5 (2026-10-14): one shared lane-change network, one logit per ROAD
transition.

``model.lane_model = "hazard_v5"`` replaces the v4 hazard (one shared left/right output,
raw inputs, the driver embedding as input) by:

* **Road manoeuvres, not driver classes.**  A car's lane fixes its single learned
  transition: lane 1 -> 0 *overtake*, lane 0 -> 1 *return*, ramp lane 2 -> 1 *merge*
  (the only moves the physics ever allows besides the forced merge; 1 -> 2 is never
  allowed).  One shared trunk reads situation features + road context and three heads
  give one logit (vs stay = 0) per transition; the car's lane picks the head.  NO driver
  embedding and no driver settings enter the network: 'v0' below is the car's own maximum
  observed speed (an own-history situation statistic), so the network is a pure function of
  the rollout / recorded state and the car's own past, and can be pre-fitted offline from data
  alone (``tools/lanev5_pretrain.py``).
* **Scaled inputs.**  Every raw physical feature (m, m/s, s, counts) goes through a signed
  log ``sign(x) log(1 + |x|)`` and then a fixed affine standardisation ``(x - mu) / sd``
  whose ``mu``/``sd`` are buffers (identity by default; the pre-fit tool stores the TRAIN
  statistics, and they travel inside every WM checkpoint).
* **Features** (:data:`FEATURE_NAMES`, from the evidence in tmp/laneranking): target-lane
  leader/follower gaps, speeds and accelerations, the speed-gap-rule margins of the target
  gaps (the same rule as the feasibility mask), the IDM incentive and the deceleration
  imposed on the new follower, safe-speed gain, mean speed / density ahead in the target and
  the current lane, blocked time, LC2013-style speed-gain and keep-right accumulators,
  5 s / 15 s averaged safe-speed gains, deficit EMAs, neighbours' recent lane changes, plus
  road context (current lane one-hot, ramp-merge flag, distance to the merge end, time in the
  merge zone, position).  Accumulators are a recurrent state ``[B, K, ACC_F]`` carried by
  the rollout (``state["lane_v5_acc"]``, across truncated-BPTT chunks) and initialised at t0
  from the observed pre-t0 long driver history; the teacher-forced pass recomputes them on
  the recorded states from the same initial value.  No future information.
* **Merges.**  The physics mask decides feasibility everywhere (never unmasked).  Inside the
  merge zone the merge head is a learned gap acceptance on feasible rows; in the last
  :data:`FORCE_WINDOW_M` a feasible merge is forced (logit ``max(physics, 8)``).
* **Decisions** (``lane_decision``): ``argmax``; ``sample`` (seeded masked categorical, as
  v4); ``gated``: a head samples only if its teacher-forced feasible-row AP on TRAIN-val
  (buffer :attr:`LaneV5Net.head_ap`, written by the pre-fit / calibration tool, or the config
  ``model.lane_v5_head_ap``) reaches its gate (overtake 0.25, return 0.10, merge 0.5), else
  argmax.  ``lane_logit_offsets`` = per-head logit offsets (calibration).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn

from wms.physics.config import PhysicsConfig
from wms.physics.mobil import (MASKED_LOGIT, anticipated_kin, merge_end_waiting, secure_gaps,
                               target_overlap)
from wms.physics.neighbours import MIN_GAP_M, lane_neighbour, lane_of

#: Road transitions (head index): overtake 1->0, return 0->1, merge 2->1.
TRANSITIONS = ("overtake", "return", "merge")
#: Head of a car in lane 0, 1, 2 and the target lane of that head.
HEAD_OF_LANE = (1, 0, 2)
TARGET_OF_LANE = (1, 0, 1)
#: Initial head biases ~ logit of the TRAIN per-step change rate on feasible rows
#: (tmp/laneranking val: overtake 0.34 %, return 0.14 %, merge 8.8 %).
HEAD_BIAS_INIT = (-5.7, -6.6, -2.3)
#: ``lane_decision='gated'``: minimum teacher-forced feasible-row AP for a head to sample.
DEFAULT_GATES = (0.25, 0.10, 0.5)
#: Unknown AP (never samples under 'gated').
UNKNOWN_AP = -1.0
LANE_DECISIONS_V5 = ("argmax", "sample", "gated")
#: Forced-merge window before the acceleration-lane end; minimum forced logit.
FORCE_WINDOW_M = 5.0
FORCED_MERGE_LOGIT = 8.0
#: Recurrent accumulator channels.
ACC_NAMES = ("v0h", "blk", "blk30", "def10", "def30", "sg", "kr", "krd", "tzone",
             "vs5_0", "vs5_1", "vs5_2", "vs15_0", "vs15_1", "vs15_2", "seen")
ACC_F = len(ACC_NAMES)
_A = {name: i for i, name in enumerate(ACC_NAMES)}
FEATURE_NAMES = (
    "v", "a", "v0h", "defic", "vratio", "age_s", "never",
    "clg", "cldv", "cla",
    "tlg", "tldv", "tla", "tfg", "tfdv", "adv",
    "ml", "mf", "aov", "idm_inc", "foll_dec", "vsgain",
    "tms", "cms", "tc100", "tc200", "cc100", "cc200",
    "blk_s", "blkn", "blk30", "def10", "def30", "sg", "kr", "krd", "vs5gain", "vs15gain",
    "rec_t", "rec_c",
    "lane0", "lane1", "lane2", "is_merge", "dend", "in_zone", "tzone", "s_km")
N_FEATURES = len(FEATURE_NAMES)
#: Feature constants (round values / the SUMO vType of the data, as in tmp/laneranking).
MIN_V0_MPS = 5.0
BLOCKED_GAP_M = 50.0
BLOCKED_MARGIN_MPS = 2.0
LOOKAHEAD_M = 100.0
DENSITY_FAR_M = 200.0
RECENT_CHANGE_S = 5.0
RECENT_RANGE_M = 100.0
AGE_SENTINEL = 999.0
_IDM = dict(a=2.6, b=4.5, s0=2.5, T=1.1)
_VSAFE_B, _VSAFE_TAU = 4.5, 1.0
_PAIR_BUDGET = 1 << 24


def slog(x: Tensor) -> Tensor:
    """Signed log ``sign(x) log(1 + |x|)``."""
    return torch.sign(x) * torch.log1p(x.abs())


def _idm(v: Tensor, vl: Tensor, gap: Tensor, v0) -> Tensor:
    a, b, s0, t = _IDM["a"], _IDM["b"], _IDM["s0"], _IDM["T"]
    ss = s0 + (v * t + v * (v - vl) / (2.0 * math.sqrt(a * b))).clamp_min(0.0)
    v0 = torch.as_tensor(v0, dtype=v.dtype, device=v.device).clamp_min(1.0)
    return a * (1.0 - (v / v0) ** 4 - (ss / gap.clamp_min(0.3)) ** 2)


def _vsafe_raw(lead_v: Tensor, gap: Tensor, free: Tensor) -> Tensor:
    """Krauss safe speed behind a leader (``inf`` on a free lane; capped at v0 later)."""
    bt = _VSAFE_B * _VSAFE_TAU
    x = -bt + torch.sqrt(bt * bt + lead_v * lead_v + 2.0 * _VSAFE_B * (gap - 2.5).clamp_min(0.0))
    return torch.where(free, torch.full_like(x, float("inf")), x)


def _ema(prev: Tensor, x: Tensor, tau_s: float, dt: float) -> Tensor:
    return prev + (1.0 - math.exp(-dt / tau_s)) * (x - prev)


def _lookup(table: Sequence[float], index: Tensor, dtype=None) -> Tensor:
    t = torch.tensor(table, device=index.device,
                     dtype=dtype if dtype is not None else torch.long)
    return t[index.clamp(0, len(table) - 1)]


def _free_gap(physics: PhysicsConfig) -> float:
    return float(physics.free_road_gap_m) if physics.open_road else 1.0e3


# --------------------------------------------------------------------- stage A
def instant_features(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                     lane_age: Optional[Tensor]) -> dict[str, Tensor]:
    """Instantaneous (accumulator-free, v0-free) quantities of every slot ``[N, K]``.

    Per lane leader/follower, mean speed and densities ahead, the speed-gap-rule margins of
    the target lane (anticipated configuration, exactly :func:`wms.physics.mobil.speed_gap_safe`),
    the IDM incentive, the imposed follower deceleration and neighbours' recent changes.
    """
    kin = kin.float()
    present = present.bool()
    s, v, a, length = kin[..., 0], kin[..., 4], kin[..., 5], kin[..., 6]
    lane = lane_of(kin)
    n, k = s.shape
    free = _free_gap(physics)
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)                     # [N,i,j] = s_j - s_i
    eye = torch.eye(k, dtype=torch.bool, device=s.device)
    other = present.unsqueeze(-2) & ~eye
    ahead, behind = ds > 0, ds < 0
    bump_a = ds - length.unsqueeze(-2)
    bump_b = -ds - length.unsqueeze(-1)
    inf = torch.full_like(ds, float("inf"))
    lg, lv, la, fg, fv = [], [], [], [], []
    lane_j = lane.unsqueeze(-2)
    age = (torch.full_like(s, AGE_SENTINEL) if lane_age is None else lane_age.float())
    key_a = torch.where(other & ahead, bump_a, inf)
    key_b = torch.where(other & behind, bump_b, inf)
    for q in range(3):
        g, gi = torch.where(lane_j == q, key_a, inf).min(-1)
        has = torch.isfinite(g)
        lg.append(torch.where(has, g.clamp_min(MIN_GAP_M), torch.full_like(g, free)))
        lv.append(torch.where(has, v.gather(-1, gi), v))
        la.append(torch.where(has, a.gather(-1, gi), torch.zeros_like(a)))
        g, gi = torch.where(lane_j == q, key_b, inf).min(-1)
        has = torch.isfinite(g)
        fg.append(torch.where(has, g.clamp_min(MIN_GAP_M), torch.full_like(g, free)))
        fv.append(torch.where(has, v.gather(-1, gi), v))
    # Per-lane counts / speed sums as batched matmuls with the lane one-hot [N, K, 3].
    onehot = torch.nn.functional.one_hot(lane, 3).to(v.dtype)
    w_ahead = other & ahead
    w100 = (w_ahead & (ds <= LOOKAHEAD_M)).to(v.dtype)
    c100_3 = torch.matmul(w100, onehot)
    c200_3 = torch.matmul((w_ahead & (ds <= DENSITY_FAR_M)).to(v.dtype), onehot)
    total3 = torch.matmul(w100, onehot * v.unsqueeze(-1))
    ms3 = torch.where(c100_3 > 0, total3 / c100_3.clamp_min(1.0),
                      torch.full_like(total3, float("inf")))
    recent_j = (present & (age * physics.dt_s < RECENT_CHANGE_S)).unsqueeze(-2) & ~eye         & (ds.abs() < RECENT_RANGE_M)
    rec3 = torch.matmul(recent_j.to(v.dtype), onehot)
    lg3, lv3, la3 = torch.stack(lg, -1), torch.stack(lv, -1), torch.stack(la, -1)
    fg3, fv3 = torch.stack(fg, -1), torch.stack(fv, -1)
    tq = _lookup(TARGET_OF_LANE, lane)
    g_t = lambda x: x.gather(-1, tq.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    g_c = lambda x: x.gather(-1, lane.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    # Current-lane leader; the acceleration-lane end is a standing obstacle (lane 2).
    clg, clv, cla = g_c(lg3), g_c(lv3), g_c(la3)
    end_gap = (physics.merge_zone_s1_m - s).clamp_min(MIN_GAP_M)
    obstacle = (lane == physics.merge_lane) & (end_gap < clg)
    clg = torch.where(obstacle, end_gap, clg)
    clv = torch.where(obstacle, torch.zeros_like(clv), clv)
    cla = torch.where(obstacle, torch.zeros_like(cla), cla)
    tlg, tlv, tfg, tfv = g_t(lg3), g_t(lv3), g_t(fg3), g_t(fv3)
    # Speed-gap-rule margins of the target lane (same rule and anticipation as the mask).
    p = physics
    kin_a = anticipated_kin(kin, p.gap_anticipation_s)
    gap_l, lv_l = lane_neighbour(kin_a, present, tq, p, ahead=True)
    gap_f, fv_f = lane_neighbour(kin_a, present, tq, p, ahead=False)
    va = kin_a[..., 4]
    req_l = p.gap_s0_lead_m + p.gap_t_lead_s * va \
        + (va * va - lv_l * lv_l).clamp_min(0.0) / (2.0 * p.gap_b_lead_mps2)
    req_f = p.gap_s0_follow_m + p.gap_t_follow_s * fv_f \
        + (fv_f * fv_f - va * va).clamp_min(0.0) / (2.0 * p.gap_b_follow_mps2)
    aov = target_overlap(kin_a, present, tq, p.gap_overlap_margin_m)
    fr = free - 1.0e-3
    target_feasible = None
    if fast_mask_applies(p):
        # The physics feasibility of the car's own transition, from the quantities above
        # (identical to wms.physics.mobil.mobil_merge_logits > MASKED_LOGIT for this target;
        # tested): ramp-zone rule, speed gap rule, stopped-merger relaxation.
        safe = (gap_l >= req_l) & (gap_f >= req_f) & ~aov
        is_ramp = lane == p.merge_lane
        if p.gap_stopped_relax:
            stopped = is_ramp & (v < p.gap_stopped_speed_mps)
            req_fs = p.gap_stopped_min_m + (tfv * tfv - v * v).clamp_min(0.0) / (
                2.0 * p.gap_b_follow_mps2)
            relax = (stopped & (tlg >= p.gap_stopped_min_m) & (tfg >= req_fs)
                     & ~target_overlap(kin, present, tq, p.gap_stopped_min_m))
            safe = safe | relax
        if p.merge_end_wait:
            # V6MERGE: the same final veto as mobil_merge_logits (merge_end_blocked), from the
            # target-lane neighbours above (current: tlg/tlv/tfg/tfv; anticipated: gap_*/aov).
            with torch.no_grad():
                secure = (secure_gaps(v, tlg, tlv, tfg, tfv, p)
                          & ~target_overlap(kin, present, tq, p.gap_overlap_margin_m)
                          & secure_gaps(va, gap_l, lv_l, gap_f, fv_f, p) & ~aov)
                safe = safe & ~(merge_end_waiting(kin, present, p) & ~secure)
        in_zone = (s >= p.merge_zone_s0_m) & (s <= p.merge_zone_s1_m)
        allowed = (~is_ramp | in_zone) & safe
        if p.lane_rule == "forced_merge_only":
            allowed = allowed & is_ramp
        target_feasible = allowed & present
    x3 = _vsafe_raw(lv3, lg3, lg3 >= fr)
    x_cur = _vsafe_raw(clv, clg, clg >= fr)
    return {
        "lane": lane, "tq": tq, "s": s, "v": v, "a": a, "present": present,
        "clg": clg, "clv": clv, "cla": cla, "cl_free": clg >= fr,
        "tlg": tlg, "tlv": tlv, "tla": g_t(la3), "tfg": tfg, "tfv": tfv, "tl_free": tlg >= fr,
        "ml": gap_l - req_l, "mf": gap_f - req_f, "aov": aov,
        "idm_inc": (_idm(v, tlv, tlg, 1.0e3) - _idm(v, clv, clg, 1.0e3)),
        "foll_dec": _idm(tfv, v, tfg, float(p.v0_mps)),
        "f_has": tfg < fr,
        "ms3": ms3, "c100_3": c100_3, "c200_3": c200_3, "x3": x3, "x_cur": x_cur,
        "rl_g": lg3[..., 1], "rl_v": lv3[..., 1], "rec3": rec3, "age": age,
        **({} if target_feasible is None else {"target_feasible": target_feasible}),
    }


def fast_mask_applies(physics: PhysicsConfig) -> bool:
    """The own-transition feasibility can be read from :func:`instant_features` (speed gap
    rule, no merge calibration, merge target lane 1); otherwise the physics module is used."""
    return (physics.lane_gap_rule == "speed" and not physics.merge_cal
            and int(physics.merge_target_lane) == 1 and int(physics.merge_lane) == 2)


def instant_features_chunked(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                             lane_age: Optional[Tensor]) -> dict[str, Tensor]:
    """:func:`instant_features` over many rows ``[N, K]`` in memory-bounded chunks."""
    n, k = present.shape
    step = max(1, _PAIR_BUDGET // max(1, k * k))
    if n <= step:
        return instant_features(kin, present, physics, lane_age)
    parts = [instant_features(kin[i:i + step], present[i:i + step], physics,
                              None if lane_age is None else lane_age[i:i + step])
             for i in range(0, n, step)]
    return {key: torch.cat([part[key] for part in parts], 0) for key in parts[0]}


# --------------------------------------------------------------------- stage B
def initial_acc(present: Tensor, kin: Tensor, physics: PhysicsConfig,
                driver_history: Optional[Tensor], driver_history_mask: Optional[Tensor],
                stride_steps: int) -> Tensor:
    """Accumulators ``[B, K, ACC_F]`` before t0 from the observed pre-t0 long history.

    Uses the history samples strictly before t0 (channels v/30, lane one-hot, gap/100,
    displacement/250, leader dv/30, leader-seen): maximum speed, blocked run, blocked and
    deficit EMAs and time in the merge zone (sample period ``stride_steps * dt``).  The
    speed-gain / keep-right accumulators start at 0 and the safe-speed EMAs at the t0 value
    (they need neighbour lanes).  Without a long history every car starts 'unseen' and is
    initialised by its first update.
    """
    acc = torch.zeros((*present.shape, ACC_F), dtype=torch.float32, device=present.device)
    acc[..., _A["vs5_0"]:_A["vs15_2"] + 1] = -1.0
    if driver_history is None or driver_history_mask is None \
            or driver_history.shape[-1] < 9 or driver_history.shape[-2] <= 1:
        return acc
    dt = float(physics.dt_s)
    period = dt * float(stride_steps)
    with torch.no_grad():
        hist = driver_history[..., :-1, :].float()
        seen_h = driver_history_mask[..., :-1].bool()
        s0 = kin[..., 0].float()
        v0h = torch.zeros_like(s0)
        blk = torch.zeros_like(s0)
        be = torch.zeros_like(s0)
        d10 = torch.zeros_like(s0)
        d30 = torch.zeros_like(s0)
        tz = torch.zeros_like(s0)
        zero = torch.zeros_like(s0)
        for h in range(hist.shape[-2]):
            seen = seen_h[..., h]
            v = hist[..., h, 0] * 30.0
            lane = hist[..., h, 2:5].argmax(-1)
            gap = hist[..., h, 5] * 100.0
            s = s0 + hist[..., h, 6] * 250.0
            lead_v = v + hist[..., h, 7] * 30.0
            has = hist[..., h, 8] > 0.5
            v0h = torch.where(seen, torch.maximum(v0h, v), zero)
            v0e = v0h.clamp_min(MIN_V0_MPS)
            blocked = seen & has & (gap <= BLOCKED_GAP_M) & (lead_v < v0e - BLOCKED_MARGIN_MPS)
            blk = torch.where(blocked, blk + float(stride_steps), zero)
            be = torch.where(seen, _ema(be, blocked.float(), 30.0, period), zero)
            deficit = (v0e - v).clamp_min(0.0)
            d10 = torch.where(seen, _ema(d10, deficit, 10.0, period), zero)
            d30 = torch.where(seen, _ema(d30, deficit, 30.0, period), zero)
            in_zone = seen & (lane == physics.merge_lane) & (s >= physics.merge_zone_s0_m)
            tz = torch.where(in_zone, tz + period, zero)
        last_seen = seen_h[..., -1] & present.bool()
        for name, value in (("v0h", v0h), ("blk", blk), ("blk30", be), ("def10", d10),
                            ("def30", d30), ("tzone", tz)):
            acc[..., _A[name]] = torch.where(last_seen, value, zero)
        acc[..., _A["seen"]] = last_seen.float()
    return acc


def update_acc(acc: Tensor, inst: dict[str, Tensor], physics: PhysicsConfig) -> Tensor:
    """One causal accumulator update with the current step's instantaneous quantities."""
    dt = float(physics.dt_s)
    acc = acc.detach().float()
    present, v, s, lane = inst["present"], inst["v"], inst["s"], inst["lane"]
    zero = torch.zeros_like(v)
    new = present & (acc[..., _A["seen"]] < 0.5)
    reset = new | (inst["age"] < 0.5)
    get = lambda name: acc[..., _A[name]]  # noqa: E731
    v0h = torch.maximum(torch.where(new, zero, get("v0h")), v)
    v0e = v0h.clamp_min(MIN_V0_MPS)
    blocked = present & (inst["clg"] < BLOCKED_GAP_M) & (inst["clv"] < v0e - BLOCKED_MARGIN_MPS)
    blk = torch.where(blocked, torch.where(new, zero, get("blk")) + 1.0, zero)
    be = _ema(torch.where(new, zero, get("blk30")), blocked.float(), 30.0, dt)
    deficit = (v0e - v).clamp_min(0.0)
    d10 = _ema(torch.where(new, zero, get("def10")), deficit, 10.0, dt)
    d30 = _ema(torch.where(new, zero, get("def30")), deficit, 30.0, dt)
    vs3 = torch.minimum(inst["x3"], v0e.unsqueeze(-1))
    vs_cur = torch.minimum(inst["x_cur"], v0e)
    vs_t = vs3.gather(-1, inst["tq"].unsqueeze(-1)).squeeze(-1)
    gain = (vs_t - vs_cur) / vs_t.clamp_min(10.0)
    sg = torch.where(reset, zero, get("sg"))
    sg = torch.where(gain > 0, sg + dt * gain, 0.5 * sg)
    # LC2013-like keep-right accumulator (lane 0 only; its right lane is 1).
    on0 = lane == 0
    vmax = v0e
    acc_t = 7.0 * (vmax / float(physics.v0_mps)) * v.clamp_min(1.0)
    rl_g, rl_v = inst["rl_g"], inst["rl_v"]
    brake = vmax * vmax / (2.0 * 4.5)
    fsg = ((rl_g + 50.0).clamp_max(300.0) - brake).clamp_min(0.0)
    fsds = torch.minimum(acc_t, fsg / vmax.clamp_min(1.0))
    slow = (rl_g < _free_gap(physics) - 1.0e-3) & (rl_v < vmax)
    need = 2.5 + vmax + (vmax * vmax - rl_v * rl_v).clamp_min(0.0) / 9.0
    fsg2 = torch.minimum(fsg, rl_g - need).clamp_min(0.0)
    fsds = torch.where(slow, torch.minimum(fsds, fsg2 / (vmax - rl_v).clamp_min(0.1)), fsds)
    dprob = 2.0 * (fsds / acc_t.clamp_min(1.0e-3)) / 5.0
    kr = torch.where(on0 & ~reset, get("kr") + dt * dprob, torch.where(on0, dt * dprob, zero))
    krd = torch.where(on0, _ema(torch.where(reset, zero, get("krd")), dprob, 10.0, dt), zero)
    in_zone = (lane == physics.merge_lane) & (s >= physics.merge_zone_s0_m)
    tz = torch.where(in_zone, torch.where(new, zero, get("tzone")) + dt, zero)
    vs5_old = acc[..., _A["vs5_0"]:_A["vs5_2"] + 1]
    vs15_old = acc[..., _A["vs15_0"]:_A["vs15_2"] + 1]
    fresh = (new.unsqueeze(-1) | (vs5_old < 0))
    vs5 = torch.where(fresh, vs3, _ema(vs5_old, vs3, 5.0, dt))
    vs15 = torch.where(fresh, vs3, _ema(vs15_old, vs3, 15.0, dt))
    out = torch.cat((torch.stack((v0h, blk, be, d10, d30, sg, kr, krd, tz), -1), vs5, vs15,
                     present.float().unsqueeze(-1)), -1)
    return torch.where(present.unsqueeze(-1), out, torch.zeros_like(out).index_fill_(
        -1, torch.arange(_A["vs5_0"], _A["vs15_2"] + 1, device=out.device), -1.0))


# --------------------------------------------------------------------- stage C
def assemble_features(inst: dict[str, Tensor], acc: Tensor, physics: PhysicsConfig) -> Tensor:
    """Raw feature vector ``[N, K, N_FEATURES]`` (physical units) from stage A + the updated
    accumulators (zero on absent slots)."""
    v, a, s, lane, tq = inst["v"], inst["a"], inst["s"], inst["lane"], inst["tq"]
    get = lambda name: acc[..., _A[name]]  # noqa: E731
    v0e = get("v0h").clamp_min(MIN_V0_MPS)
    vs3 = torch.minimum(inst["x3"], v0e.unsqueeze(-1))
    vs_cur = torch.minimum(inst["x_cur"], v0e)
    g_t = lambda x: x.gather(-1, tq.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    g_c = lambda x: x.gather(-1, lane.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    ms3 = torch.minimum(inst["ms3"], v0e.unsqueeze(-1))
    vs5 = acc[..., _A["vs5_0"]:_A["vs5_2"] + 1]
    vs15 = acc[..., _A["vs15_0"]:_A["vs15_2"] + 1]
    lead_t = torch.where(inst["tl_free"], v0e, inst["tlv"])
    lead_c = torch.where(inst["cl_free"], v0e, inst["clv"])
    age = inst["age"]
    dt = float(physics.dt_s)
    in_zone = ((lane == physics.merge_lane) & (s >= physics.merge_zone_s0_m)
               & (s <= physics.merge_zone_s1_m))
    one = lambda b: b.to(v.dtype)  # noqa: E731
    cols = [
        v, a, v0e, v0e - v, v / v0e, (age * dt).clamp(0.0, 300.0), one(age >= AGE_SENTINEL),
        inst["clg"].clamp_max(250.0), inst["clv"] - v, inst["cla"],
        inst["tlg"].clamp_max(250.0), inst["tlv"] - v, inst["tla"],
        inst["tfg"].clamp_max(250.0), inst["tfv"] - v,
        torch.minimum(lead_t, v0e) - torch.minimum(lead_c, v0e),
        inst["ml"].clamp(-50.0, 100.0), inst["mf"].clamp(-50.0, 100.0), one(inst["aov"]),
        inst["idm_inc"].clamp(-10.0, 10.0),
        torch.where(inst["f_has"], inst["foll_dec"].clamp(-10.0, 3.0), torch.full_like(v, 3.0)),
        g_t(vs3) - vs_cur,
        g_t(ms3) - v0e, g_c(ms3) - v0e,
        g_t(inst["c100_3"]), g_t(inst["c200_3"]), g_c(inst["c100_3"]), g_c(inst["c200_3"]),
        (get("blk") * dt).clamp_max(60.0), one(get("blk") > 0), get("blk30"),
        get("def10"), get("def30"), get("sg"), get("kr"), get("krd"),
        g_t(vs5) - g_c(vs5), g_t(vs15) - g_c(vs15),
        g_t(inst["rec3"]), g_c(inst["rec3"]),
        one(lane == 0), one(lane == 1), one(lane == 2), one(lane == physics.merge_lane),
        (physics.merge_zone_s1_m - s).clamp(-100.0, 500.0), one(in_zone), get("tzone"),
        s / 1000.0,
    ]
    feats = torch.stack(cols, -1)
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(inst["present"].unsqueeze(-1), feats, torch.zeros_like(feats))


def step_features(acc_prev: Tensor, kin: Tensor, present: Tensor, physics: PhysicsConfig,
                  lane_age: Optional[Tensor]) -> tuple[Tensor, Tensor, dict]:
    """Rollout step: ``(raw features [B,K,F], updated accumulators, instantaneous dict)``."""
    with torch.no_grad():
        inst = instant_features(kin.detach(), present, physics,
                                None if lane_age is None else lane_age.detach())
        acc = update_acc(acc_prev, inst, physics)
        return assemble_features(inst, acc, physics), acc, inst


def sequence_features(states: Tensor, present: Tensor, physics: PhysicsConfig,
                      lane_age: Tensor, acc0: Tensor, *, return_instant: bool = False):
    """Teacher-forced: raw features ``[B, T, K, F]`` and accumulators ``[B, T, K, ACC_F]`` on
    recorded states ``[B, T, K, 7]`` (accumulators run causally from ``acc0``)."""
    b, t, k = states.shape[:3]
    with torch.no_grad():
        inst = instant_features_chunked(states.reshape(b * t, k, -1).float(),
                                        present.reshape(b * t, k).bool(), physics,
                                        lane_age.reshape(b * t, k))
        inst = {key: value.reshape(b, t, *value.shape[1:]) for key, value in inst.items()}
        accs = []
        acc = acc0
        for i in range(t):
            acc = update_acc(acc, {key: value[:, i] for key, value in inst.items()}, physics)
            accs.append(acc)
        acc_seq = torch.stack(accs, 1)
        flat_inst = {key: value.reshape(b * t, *value.shape[2:]) for key, value in inst.items()}
        feats = assemble_features(flat_inst, acc_seq.reshape(b * t, k, -1), physics)
        if return_instant:
            return feats.reshape(b, t, k, -1), acc_seq, inst
        return feats.reshape(b, t, k, -1), acc_seq


# --------------------------------------------------------------------- network
class LaneV5Net(nn.Module):
    """Shared trunk over the scaled situation features + one head per road transition."""

    def __init__(self, hidden: int = 64, cap: float = 10.0) -> None:
        super().__init__()
        self.cap = float(cap)
        #: Trunk dropout (plain attribute, never saved; 0 = off). Only the offline pre-fit
        #: (tools/lanev5_pretrain.py --dropout) sets it, and resets it before saving.
        self.dropout = 0.0
        self.trunk = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.heads = nn.Linear(hidden, len(TRANSITIONS))
        nn.init.zeros_(self.heads.weight)
        with torch.no_grad():
            self.heads.bias.copy_(torch.tensor(HEAD_BIAS_INIT))
        self.register_buffer("feature_mu", torch.zeros(N_FEATURES))
        self.register_buffer("feature_sd", torch.ones(N_FEATURES))
        self.register_buffer("head_ap", torch.full((len(TRANSITIONS),), UNKNOWN_AP))

    def scale(self, raw: Tensor) -> Tensor:
        return (slog(raw) - self.feature_mu) / self.feature_sd.clamp_min(1.0e-6)

    def forward(self, raw: Tensor) -> Tensor:
        """Head logits ``[..., 3]`` (overtake, return, merge) from raw features ``[..., F]``."""
        if raw.shape[-1] != N_FEATURES:
            raise ValueError(f"lane v5 features must end in {N_FEATURES}")
        x = self.scale(raw.to(self.feature_mu.dtype))
        p = float(getattr(self, "dropout", 0.0))
        if p > 0.0 and self.training:
            for layer in self.trunk:
                x = layer(x)
                if isinstance(layer, nn.GELU):
                    x = nn.functional.dropout(x, p, training=True)
        else:
            x = self.trunk(x)
        out = self.heads(x)
        return self.cap * torch.tanh(out / self.cap)

    def head_logit(self, raw: Tensor, lane: Tensor) -> Tensor:
        """The logit of each row's own transition (picked by its lane)."""
        return self(raw).gather(-1, _lookup(HEAD_OF_LANE, lane).unsqueeze(-1)).squeeze(-1)


def lane_logits_v5(net: LaneV5Net, raw: Tensor, inst: dict[str, Tensor], learned: Tensor,
                   lane_base: Tensor, physics: PhysicsConfig,
                   offsets: Optional[Sequence[float]] = None) -> tuple[Tensor, Tensor, Tensor]:
    """Absolute lane logits ``[B, K, 3]``, the per-row head logit and the 'acts' mask.

    The learned transition's entry is the head logit (+ offset) on learned slots whose target
    the physics allows (``lane_base > MASKED_LOGIT``) outside the forced window; a feasible
    merge in the last :data:`FORCE_WINDOW_M` gets ``max(physics, 8)``; everything else keeps
    the physics logit (masked targets stay masked; the current lane is 'stay' = 0).
    """
    lane, tq, s = inst["lane"], inst["tq"], inst["s"]
    head = _lookup(HEAD_OF_LANE, lane)
    logit = net(raw).gather(-1, head.unsqueeze(-1)).squeeze(-1)
    if offsets is not None and any(float(x) != 0.0 for x in offsets):
        logit = logit + _lookup(tuple(float(x) for x in offsets), head, dtype=logit.dtype)
    prior = lane_base.gather(-1, tq.unsqueeze(-1)).squeeze(-1)
    feasible = learned.bool() & (prior > MASKED_LOGIT)
    is_merge = lane == physics.merge_lane
    forced = is_merge & (s >= physics.merge_zone_s1_m - FORCE_WINDOW_M) \
        & (s <= physics.merge_zone_s1_m)
    acts = feasible & ~forced
    value = torch.where(acts, logit.to(lane_base.dtype), prior)
    value = torch.where(feasible & forced, prior.clamp_min(FORCED_MERGE_LOGIT), value)
    logits = lane_base.scatter(-1, tq.unsqueeze(-1), value.unsqueeze(-1))
    return logits, logit, acts


def gate_open(head_ap: Sequence[float], gates: Sequence[float]) -> tuple[bool, bool, bool]:
    """Per head: sample under ``lane_decision='gated'`` (AP known and >= gate)."""
    return tuple(float(ap) >= 0.0 and float(ap) >= float(g)
                 for ap, g in zip(head_ap, gates))  # type: ignore[return-value]


def sample_mask(lane: Tensor, learned: Tensor, open_heads: Sequence[bool]) -> Tensor:
    """``[B, K]`` slots that draw their lane (learned and their head's gate open)."""
    ok = _lookup(tuple(1 if x else 0 for x in open_heads), _lookup(HEAD_OF_LANE, lane))
    return learned.bool() & ok.bool()


# --------------------------------------------------------------------- teacher-forced rows
def recorded_rows(states: Tensor, valid: Tensor, is_cav: Tensor, physics: PhysicsConfig,
                  lane_base_fn: Optional[Callable[[Tensor, Tensor], Tensor]], lane_age: Tensor,
                  acc0: Tensor) -> dict[str, Tensor]:
    """Teacher-forced rows of every recorded step of the windows.

    ``states [B, T+1, K, 7]`` / ``valid [B, T+1, K]`` (on-road, supervised) / ``is_cav``;
    ``lane_age [B, T, K]`` recorded steps since the last change; ``acc0`` pre-t0
    accumulators.  A row = an HDV-step t whose transition t -> t+1 is valid (both ends valid,
    at most one lane), whose own transition target the physics allows at t and that lies
    outside the forced window.  ``lane_base_fn(kin, present) -> [N, K, 3]`` gives the physics
    lane logits; ``None`` reads the feasibility from :func:`instant_features` (speed gap rule,
    :func:`fast_mask_applies`; identical, tested).  Returns raw features ``[R, F]``, head ``[R]``, label ``[R]``
    (1 = the car made its transition) plus counts (events, infeasible events).
    """
    b, t1, k = states.shape[:3]
    t = t1 - 1
    lanes = states[..., 1:4].argmax(-1)
    present = valid[:, :t].bool()
    feats, _, inst = sequence_features(states[:, :t], present, physics, lane_age, acc0,
                                       return_instant=True)
    with torch.no_grad():
        prev, nxt = lanes[:, :t], lanes[:, 1:]
        hdv = present & ~is_cav[:, :t].bool()
        ok = hdv & valid[:, 1:].bool() & ((nxt - prev).abs() <= 1)
        tq = _lookup(TARGET_OF_LANE, prev)
        if lane_base_fn is None and "target_feasible" in inst:
            feasible = inst["target_feasible"]
        else:
            if lane_base_fn is None:
                raise ValueError("this physics needs lane_base_fn for the feasibility mask")
            flat_kin = states[:, :t].reshape(b * t, k, -1).float()
            flat_present = present.reshape(b * t, k)
            lane_base = torch.cat([lane_base_fn(flat_kin[i:i + 256], flat_present[i:i + 256])
                                   for i in range(0, b * t, 256)], 0).reshape(b, t, k, 3)
            prior = lane_base.gather(-1, tq.unsqueeze(-1)).squeeze(-1)
            feasible = prior > MASKED_LOGIT
        s = states[:, :t, :, 0]
        forced = (prev == physics.merge_lane) & (s >= physics.merge_zone_s1_m - FORCE_WINDOW_M) \
            & (s <= physics.merge_zone_s1_m)
        label = nxt == tq
        own = (nxt == prev) | label
        rows = ok & own & feasible & ~forced
        head = _lookup(HEAD_OF_LANE, prev)
        events = ok & label
        stats = {"rows": int(rows.sum()), "events": int((rows & label).sum()),
                 "infeasible_events": int((events & ~feasible).sum()),
                 "forced_window_events": int((events & feasible & forced).sum())}
        for h, name in enumerate(TRANSITIONS):
            sel = rows & (head == h)
            stats[f"{name}_rows"] = int(sel.sum())
            stats[f"{name}_events"] = int((sel & label).sum())
    return {"features": feats[rows], "head": head[rows], "label": label[rows].float(),
            "stats": stats}


def head_bce(logits: Tensor, head: Tensor, label: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Sum over heads of the per-head mean binary CE (plain, unweighted) on the rows."""
    total = logits.sum() * 0.0
    stats: dict[str, float] = {}
    for h, name in enumerate(TRANSITIONS):
        sel = head == h
        if bool(sel.any()):
            loss = nn.functional.binary_cross_entropy_with_logits(logits[sel], label[sel])
            total = total + loss
            stats[f"{name}_bce"] = float(loss.detach())
    return total, stats


# --------------------------------------------------------------------- pre-fit files
PRETRAIN_FORMAT = "lanev5_pretrain/1"


def save_pretrain(net: LaneV5Net, path, meta: dict) -> None:
    """Write a pre-fit file: the lane network's state dict (incl. scaling + head AP) + meta."""
    torch.save({"format": PRETRAIN_FORMAT, "feature_names": list(FEATURE_NAMES),
                "hidden": int(net.heads.in_features), "cap": float(net.cap),
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "meta": dict(meta)}, path)


def load_pretrain(net: LaneV5Net, path) -> dict:
    """Load a :func:`save_pretrain` file into ``net`` (strict); returns a small report."""
    import hashlib
    from pathlib import Path

    p = Path(path)
    payload = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != PRETRAIN_FORMAT:
        raise ValueError(f"{p} is not a {PRETRAIN_FORMAT} file")
    if list(payload.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError(f"{p}: lane v5 feature set differs from this code")
    device = net.feature_mu.device
    net.load_state_dict({k: v.to(device) for k, v in payload["state_dict"].items()}, strict=True)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "head_ap": [float(x) for x in net.head_ap.tolist()],
            "meta": {k: payload.get("meta", {}).get(k) for k in ("rows", "windows", "epochs")}}
