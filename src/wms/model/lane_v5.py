"""LANEV5 (2026-10-14, owner-approved): one shared lane-change network, one logit per ROAD
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
* **R5INPUTS** (``model.lane_v5_extra_features``, default OFF = bit-identical): the base
  vector is unchanged; :data:`EXTRA_FEATURE_NAMES` is APPENDED when the switch is on:
  observed target-follower acceleration and its short EMA (the merge-negotiation
  ``0.7 prev + 0.3 a`` definition), target leader/follower gap closing rates and ~2 s
  gap trends, the next-behind (follower's follower) and next-ahead (leader's leader)
  clear gaps / relative speeds / accelerations / validity masks, the predicted
  time-to-align with the next-behind gap under the ego's current and maximum feasible
  acceleration, and the lane-end travel time.  Their accumulators are appended at the
  end of :data:`ACC_NAMES`; the new scaling buffers are identity (mu 0, sd 1) and an old
  checkpoint's first trunk layer is ZERO-expanded on load (exact identity at init).
* **Merges.**  The physics mask decides feasibility everywhere (never unmasked).  Inside the
  merge zone the merge head is a learned gap acceptance on feasible rows; in the last
  :data:`FORCE_WINDOW_M` a feasible merge is forced (logit ``max(physics, 8)``).  V8WM2
  ``physics.forced_window_mode = 'hazard'``: the merge head acts in that window too, its logit
  floored at ``logit(physics.forced_window_min_hazard)``; the window's rows are then ordinary
  learnable rows (``recorded_rows``, the merge-timing loss) and always draw their lane.
* **Decisions** (``lane_decision``): ``argmax``; ``sample`` (seeded masked categorical, as
  v4); ``gated``: a head samples only if its teacher-forced feasible-row AP on TRAIN-val
  (buffer :attr:`LaneV5Net.head_ap`, written by the pre-fit / calibration tool, or the config
  ``model.lane_v5_head_ap``) reaches its gate (overtake 0.25, return 0.10, merge 0.5), else
  argmax.  ``lane_logit_offsets`` = per-head logit offsets (calibration).  V8LANE
  ``lane_v5_mainline`` overrides the rule for the overtake / return heads only
  (:data:`MAINLINE_MODES`; merges keep ``lane_decision``).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn

from wms.physics.config import PhysicsConfig
from wms.physics.mobil import (MASKED_LOGIT, anticipated_kin, closing_open, impatient_margins,
                               merge_end_waiting, merge_impatience_floor, merge_impatience_level,
                               physical_merge_envelope, secure_gaps, target_overlap)
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
#:
#: R5INPUTS (2026-10-20, default OFF): the last three channels are the extra-neighbour
#: history.  ``tfa_ema`` is the target follower's acceleration EMA with the SAME definition
#: as the merge-negotiation follower-acceleration EMA (``merge_neg_accel_ema`` in
#: ``wms.model.wms`` / the negotiation path): ``0.7 * prev + 0.3 * a``.  ``tfg_e2`` /
#: ``tlg_e2`` are the target follower / leader gap EMAs over ~2 s (``dt``-aware); ``-1``
#: marks 'no EMA yet' so a new or reappearing neighbour starts its trend at zero.
#: They are appended so every pre-existing channel index is unchanged.
ACC_NAMES = ("v0h", "blk", "blk30", "def10", "def30", "sg", "kr", "krd", "tzone",
             "vs5_0", "vs5_1", "vs5_2", "vs15_0", "vs15_1", "vs15_2", "seen",
             "tfa_ema", "tfg_e2", "tlg_e2")
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
#: R5INPUTS (``model.lane_v5_extra_features``, default OFF): extra merge-input columns,
#: APPENDED after :data:`FEATURE_NAMES` only when the switch is on.  Keeping them out of
#: ``FEATURE_NAMES`` keeps the off-path network (and every old checkpoint) bit-identical.
EXTRA_FEATURE_NAMES = (
    "tfa", "tfa_ema",
    "tfg_rate", "tfg_trend", "tlg_rate", "tlg_trend",
    "nbg", "nbv", "nba", "nbh",
    "nag", "nav", "naa", "nah",
    "tta_cur", "tta_max",
    "tle_s",
)
FEATURE_NAMES_EXTRA = FEATURE_NAMES + EXTRA_FEATURE_NAMES
N_FEATURES_EXTRA = len(FEATURE_NAMES_EXTRA)
#: Merge-negotiation follower-acceleration EMA (``0.7 prev + 0.3 a``), shared definition.
NEG_ACCEL_EMA_DECAY = 0.7
#: Target-gap EMA time constant of the closing-rate trend features (s).
GAP_TREND_TAU_S = 2.0
#: Horizon of the predicted time-to-align features (s); unreachable = the horizon.
ALIGN_HORIZON_S = 60.0
#: Horizon of the lane-end travel time feature (s).
LANE_END_HORIZON_S = 120.0
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


def _second_nearest(key: Tensor, first: Tensor, inf: Tensor) -> tuple[Tensor, Tensor]:
    """Second-smallest finite ``key`` after dropping the first-nearest index.

    ``key [N, i, j]`` holds ``inf`` for invalid candidates; ``first [N, i]`` is the index
    returned by ``key.min(-1)``.  Dropping exactly that index (not every tie) reproduces
    'the next car after the nearest one' and leaves the first-nearest choice untouched.
    """
    k = key.shape[-1]
    ar = torch.arange(k, device=key.device).view(1, 1, k)
    dropped = torch.where(ar == first.unsqueeze(-1), inf, key)
    return dropped.min(-1)


def _gap_trend(gap: Tensor, ema: Tensor, dt: float, tau_s: float) -> Tensor:
    """Rate of change of ``gap`` implied by its EMA: ``(gap - ema) / (tau * (1 - alpha))``.

    ``ema`` is the EMA AFTER the current sample and ``alpha = 1 - exp(-dt / tau)``; the
    inverse ``1 - alpha`` recovers the pre-update EMA, so a newly initialised gap has a
    zero trend and a genuine change keeps its physical sign.
    """
    alpha = 1.0 - math.exp(-dt / tau_s)
    return (gap - ema) / (tau_s * max(1.0 - alpha, 1.0e-6))


def _time_to_align(d: Tensor, dv: Tensor, a: Tensor, horizon: float) -> Tensor:
    """Smallest non-negative time to close a relative distance ``d`` at ``dv`` / ``a``.

    Solves ``0.5 a t^2 + dv t - d = 0``; an unreachable / negative solution is the
    ``horizon`` (a finite sentinel, never ``inf``).
    """
    big = torch.full_like(d, float(horizon))
    eps = 1.0e-6
    disc = dv * dv + 2.0 * a * d
    ok = disc >= 0.0
    sq = torch.sqrt(disc.clamp_min(0.0))
    a_safe = torch.where(a.abs() > eps, a, torch.ones_like(a))
    t1 = (-dv + sq) / a_safe
    t2 = (-dv - sq) / a_safe
    v1 = ok & (t1 >= 0.0) & (t1 <= horizon)
    v2 = ok & (t2 >= 0.0) & (t2 <= horizon)
    quad = torch.where(v1 & v2, torch.minimum(t1, t2),
                       torch.where(v1, t1, torch.where(v2, t2, big)))
    lin = torch.where(dv.abs() > eps, d / dv,
                      torch.where(d.abs() < eps, torch.zeros_like(d), big))
    lin = torch.where(lin >= 0.0, lin, big).clamp_max(horizon)
    return torch.where(a.abs() > eps, quad, lin)


def expand_input_state(state: dict, n_old: int, n_new: int) -> dict:
    """Expand a pre-R5INPUTS lane state dict to ``n_new`` inputs with ZERO new weights.

    The first trunk layer gains zero columns and the scaling buffers gain identity
    entries (``mu = 0``, ``sd = 1``), so the logits of an old checkpoint are EXACTLY
    unchanged.  Unknown / differently sized entries are left to ``load_state_dict``.
    """
    if n_new < n_old:
        raise ValueError("lane v5 input expansion cannot shrink the feature vector")
    out = dict(state)
    weight = out.get("trunk.0.weight")
    if weight is not None and int(weight.shape[1]) == n_old and n_old != n_new:
        pad = weight.new_zeros((int(weight.shape[0]), n_new - n_old))
        out["trunk.0.weight"] = torch.cat((weight, pad), 1)
    for key, fill in (("feature_mu", 0.0), ("feature_sd", 1.0)):
        buf = out.get(key)
        if buf is not None and int(buf.numel()) == n_old and n_old != n_new:
            pad = buf.new_full((n_new - n_old,), fill)
            out[key] = torch.cat((buf.reshape(-1), pad))
    return out


# --------------------------------------------------------------------- stage A
def instant_features(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                     lane_age: Optional[Tensor],
                     impatience: Optional[Tensor] = None,
                     extra: bool = False) -> dict[str, Tensor]:
    """Instantaneous (accumulator-free, v0-free) quantities of every slot ``[N, K]``.

    Per lane leader/follower, mean speed and densities ahead, the speed-gap-rule margins of
    the target lane (anticipated configuration, exactly :func:`wms.physics.mobil.speed_gap_safe`),
    the IDM incentive, the imposed follower deceleration and neighbours' recent changes.

    V8WM3 ``impatience`` (``merge_impatience`` level ``[N, K]``, ``None`` = legacy) relaxes
    only the ramp rows of the ``target_feasible`` mask (as the physics mask); the margin
    FEATURES ``ml`` / ``mf`` keep the unrelaxed rule.
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
    lg, lv, la, fg, fv, fa = [], [], [], [], [], []
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
        fa.append(torch.where(has, a.gather(-1, gi), torch.zeros_like(a)))
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
    extra_inst: dict[str, Tensor] = {}
    if extra:
        # R5INPUTS: second-nearest target-lane neighbours (ego -> follower's follower /
        # leader's leader).  The first-nearest index is dropped rather than every tie, so
        # the existing tlg/tfg columns are untouched.
        tq3 = tq.unsqueeze(-1)                 # [N, i, 1] target lane of each row
        in_target = lane_j == tq3              # [N, i, j]
        key_a_t = torch.where(in_target, key_a, inf)
        key_b_t = torch.where(in_target, key_b, inf)
        g_a1, i_a1 = key_a_t.min(-1)
        g_a2, i_a2 = _second_nearest(key_a_t, i_a1, inf)
        g_b1, i_b1 = key_b_t.min(-1)
        g_b2, i_b2 = _second_nearest(key_b_t, i_b1, inf)
        safe_a2 = i_a2.clamp(0, max(k - 1, 0))
        safe_b2 = i_b2.clamp(0, max(k - 1, 0))
        nb_has = torch.isfinite(g_b1) & torch.isfinite(g_b2) & present
        na_has = torch.isfinite(g_a1) & torch.isfinite(g_a2) & present
        # Alignment point: midway between the next-behind car's front and the immediate
        # follower's rear; the ego front aims there.  Its motion is approximated by the
        # next-behind car's speed (the follower may still yield).
        s_f1 = s.gather(-1, i_b1.clamp(0, max(k - 1, 0)))
        l_f1 = length.gather(-1, i_b1.clamp(0, max(k - 1, 0)))
        s_nb = s.gather(-1, safe_b2)
        nb_aim_d = 0.5 * (s_nb + (s_f1 - l_f1)) - s
        zero = torch.zeros_like(s)
        nb_gap = torch.where(nb_has, g_b2.clamp_min(MIN_GAP_M), zero)
        na_gap = torch.where(na_has, g_a2.clamp_min(MIN_GAP_M), zero)
        extra_inst = {
            "tfa": g_t(torch.stack(fa, -1)),
            "nb_gap": nb_gap,
            "nb_v": torch.where(nb_has, v.gather(-1, safe_b2) - v, zero),
            "nb_a": torch.where(nb_has, a.gather(-1, safe_b2), zero),
            "nb_has": nb_has,
            "nb_aim_d": torch.where(nb_has, nb_aim_d, zero),
            "nb_speed": torch.where(nb_has, v.gather(-1, safe_b2), zero),
            "na_gap": na_gap,
            "na_v": torch.where(na_has, v.gather(-1, safe_a2) - v, zero),
            "na_a": torch.where(na_has, a.gather(-1, safe_a2), zero),
            "na_has": na_has,
        }
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
    if getattr(p, "lane_learnable", False):
        # V10UNBLOCK: the strictly physical envelope replaces every behavioural veto.
        target_feasible = physical_merge_envelope(kin, present, p)["admissible"]
    elif fast_mask_applies(p):
        # The physics feasibility of the car's own transition, from the quantities above
        # (identical to wms.physics.mobil.mobil_merge_logits > MASKED_LOGIT for this target;
        # tested): ramp-zone rule, speed gap rule, stopped-merger relaxation.
        is_ramp = lane == p.merge_lane
        if impatience is None:
            safe = (gap_l >= req_l) & (gap_f >= req_f) & ~aov
        else:
            # V8WM3: the ramp merge's relaxed margins (wms.physics.mobil.speed_gap_safe).
            s0l, tl, s0f, tf = impatient_margins(p, impatience)
            req_li = s0l + tl * va + (va * va - lv_l * lv_l).clamp_min(0.0) / (
                2.0 * p.gap_b_lead_mps2)
            req_fi = s0f + tf * fv_f + (fv_f * fv_f - va * va).clamp_min(0.0) / (
                2.0 * p.gap_b_follow_mps2)
            safe = torch.where(is_ramp, (gap_l >= req_li) & (gap_f >= req_fi),
                               (gap_l >= req_l) & (gap_f >= req_f)) & ~aov
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
                secure = (secure_gaps(v, tlg, tlv, tfg, tfv, p, impatience)
                          & ~target_overlap(kin, present, tq, p.gap_overlap_margin_m)
                          & secure_gaps(va, gap_l, lv_l, gap_f, fv_f, p, impatience) & ~aov)
                safe = safe & ~(merge_end_waiting(kin, present, p) & ~secure)
        in_zone = (s >= p.merge_zone_s0_m) & (s <= p.merge_zone_s1_m)
        allowed = (~is_ramp | closing_open(s, p, in_zone)) & safe
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
        **extra_inst,
    }


def fast_mask_applies(physics: PhysicsConfig) -> bool:
    """The own-transition feasibility can be read from :func:`instant_features` (speed gap
    rule, no merge calibration, merge target lane 1); otherwise the physics module is used."""
    return (physics.lane_gap_rule == "speed" and not physics.merge_cal
            and int(physics.merge_target_lane) == 1 and int(physics.merge_lane) == 2)


def instant_features_chunked(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                             lane_age: Optional[Tensor],
                             impatience: Optional[Tensor] = None,
                             extra: bool = False) -> dict[str, Tensor]:
    """:func:`instant_features` over many rows ``[N, K]`` in memory-bounded chunks."""
    n, k = present.shape
    step = max(1, _PAIR_BUDGET // max(1, k * k))
    if n <= step:
        return instant_features(kin, present, physics, lane_age, impatience, extra)
    parts = [instant_features(kin[i:i + step], present[i:i + step], physics,
                              None if lane_age is None else lane_age[i:i + step],
                              None if impatience is None else impatience[i:i + step],
                              extra)
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
    acc[..., _A["tfg_e2"]] = -1.0
    acc[..., _A["tlg_e2"]] = -1.0
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


def update_acc(acc: Tensor, inst: dict[str, Tensor], physics: PhysicsConfig,
               extra: bool = False) -> Tensor:
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
    if extra:
        # R5INPUTS: target follower acceleration EMA (same 0.7/0.3 definition as the merge
        # negotiation's follower EMA) and the ~2 s leader / follower gap EMAs.  ``-1`` marks
        # 'no EMA yet' for a missing target neighbour so a reappearing one starts at zero
        # trend instead of a spurious jump.
        f_has, l_has = inst["f_has"], ~inst["tl_free"]
        tfa = torch.where(f_has, inst["tfa"], zero)
        prev_a = torch.where(new, zero, get("tfa_ema"))
        tfa_ema = NEG_ACCEL_EMA_DECAY * prev_a + (1.0 - NEG_ACCEL_EMA_DECAY) * tfa
        alpha = 1.0 - math.exp(-dt / GAP_TREND_TAU_S)
        missing = torch.full_like(v, -1.0)
        prev_f = torch.where(new, missing, get("tfg_e2"))
        cur_f = torch.where(f_has, inst["tfg"], missing)
        tfg_e2 = torch.where(cur_f >= 0.0,
                             torch.where(prev_f >= 0.0, prev_f + alpha * (cur_f - prev_f), cur_f),
                             missing)
        prev_l = torch.where(new, missing, get("tlg_e2"))
        cur_l = torch.where(l_has, inst["tlg"], missing)
        tlg_e2 = torch.where(cur_l >= 0.0,
                             torch.where(prev_l >= 0.0, prev_l + alpha * (cur_l - prev_l), cur_l),
                             missing)
        extra_cols = torch.stack((tfa_ema, tfg_e2, tlg_e2), -1)
    else:
        extra_cols = torch.zeros((*present.shape, 3), dtype=torch.float32,
                                 device=present.device)
    out = torch.cat((torch.stack((v0h, blk, be, d10, d30, sg, kr, krd, tz), -1), vs5, vs15,
                     present.float().unsqueeze(-1), extra_cols), -1)
    return torch.where(present.unsqueeze(-1), out, torch.zeros_like(out).index_fill_(
        -1, torch.arange(_A["vs5_0"], _A["vs15_2"] + 1, device=out.device), -1.0))


# --------------------------------------------------------------------- stage C
def assemble_features(inst: dict[str, Tensor], acc: Tensor, physics: PhysicsConfig,
                      extra: bool = False) -> Tensor:
    """Raw feature vector ``[N, K, N_FEATURES]`` (physical units) from stage A + the updated
    accumulators (zero on absent slots).

    R5INPUTS ``extra=True`` appends :data:`EXTRA_FEATURE_NAMES` after the base columns
    (``N_FEATURES_EXTRA`` total); ``extra=False`` is bit-identical to the legacy vector.
    """
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
    if extra:
        # R5INPUTS (``model.lane_v5_extra_features``): observed target-follower motion, the
        # leader / follower closing rates and ~2 s gap trends, the next-behind / next-ahead
        # neighbours, predicted alignment times and the lane-end travel time.  Missing
        # neighbours are neutral zeros (validity masks are separate columns).
        zeros = torch.zeros_like(v)
        f_has, l_has = inst["f_has"], ~inst["tl_free"]
        tfa = torch.where(f_has, inst["tfa"], zeros)
        tfa_ema = torch.where(f_has, get("tfa_ema"), zeros)
        tfg_rate = torch.where(f_has, v - inst["tfv"], zeros)
        tlg_rate = torch.where(l_has, inst["tlv"] - v, zeros)
        e2f, e2l = get("tfg_e2"), get("tlg_e2")
        tfg_trend = torch.where(f_has & (e2f >= 0.0),
                                _gap_trend(inst["tfg"], e2f, dt, GAP_TREND_TAU_S), zeros)
        tlg_trend = torch.where(l_has & (e2l >= 0.0),
                                _gap_trend(inst["tlg"], e2l, dt, GAP_TREND_TAU_S), zeros)
        nbh, nah = inst["nb_has"], inst["na_has"]
        dv_nb = v - inst["nb_speed"]
        a_max = torch.full_like(a, float(physics.accel_max_mps2))
        tta_cur = torch.where(nbh, _time_to_align(inst["nb_aim_d"], dv_nb, a,
                                                  ALIGN_HORIZON_S), zeros)
        tta_max = torch.where(nbh, _time_to_align(inst["nb_aim_d"], dv_nb, a_max,
                                                  ALIGN_HORIZON_S), zeros)
        tle = ((physics.merge_zone_s1_m - s).clamp(0.0, 500.0) / v.clamp_min(1.0)
               ).clamp(0.0, LANE_END_HORIZON_S)
        cols.extend([
            tfa, tfa_ema, tfg_rate, tfg_trend, tlg_rate, tlg_trend,
            inst["nb_gap"], inst["nb_v"], inst["nb_a"], one(nbh),
            inst["na_gap"], inst["na_v"], inst["na_a"], one(nah),
            tta_cur, tta_max, tle,
        ])
    feats = torch.stack(cols, -1)
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(inst["present"].unsqueeze(-1), feats, torch.zeros_like(feats))


def step_features(acc_prev: Tensor, kin: Tensor, present: Tensor, physics: PhysicsConfig,
                  lane_age: Optional[Tensor], extra: bool = False) -> tuple[Tensor, Tensor, dict]:
    """Rollout step: ``(raw features [B,K,F], updated accumulators, instantaneous dict)``."""
    with torch.no_grad():
        inst = instant_features(kin.detach(), present, physics,
                                None if lane_age is None else lane_age.detach(), extra=extra)
        acc = update_acc(acc_prev, inst, physics, extra=extra)
        return assemble_features(inst, acc, physics, extra=extra), acc, inst


def sequence_features(states: Tensor, present: Tensor, physics: PhysicsConfig,
                      lane_age: Tensor, acc0: Tensor, *, return_instant: bool = False,
                      impatience: Optional[Tensor] = None, extra: bool = False):
    """Teacher-forced: raw features ``[B, T, K, F]`` and accumulators ``[B, T, K, ACC_F]`` on
    recorded states ``[B, T, K, 7]`` (accumulators run causally from ``acc0``)."""
    b, t, k = states.shape[:3]
    with torch.no_grad():
        inst = instant_features_chunked(states.reshape(b * t, k, -1).float(),
                                        present.reshape(b * t, k).bool(), physics,
                                        lane_age.reshape(b * t, k),
                                        None if impatience is None
                                        else impatience.reshape(b * t, k), extra)
        inst = {key: value.reshape(b, t, *value.shape[1:]) for key, value in inst.items()}
        accs = []
        acc = acc0
        for i in range(t):
            acc = update_acc(acc, {key: value[:, i] for key, value in inst.items()}, physics,
                             extra=extra)
            accs.append(acc)
        acc_seq = torch.stack(accs, 1)
        flat_inst = {key: value.reshape(b * t, *value.shape[2:]) for key, value in inst.items()}
        feats = assemble_features(flat_inst, acc_seq.reshape(b * t, k, -1), physics,
                                  extra=extra)
        if return_instant:
            return feats.reshape(b, t, k, -1), acc_seq, inst
        return feats.reshape(b, t, k, -1), acc_seq


# --------------------------------------------------------------------- network
class LaneV5Net(nn.Module):
    """Shared trunk over the scaled situation features + one head per road transition.

    ``extra_features`` (R5INPUTS, ``model.lane_v5_extra_features``, default OFF) widens the
    input layer by :data:`EXTRA_FEATURE_NAMES`; the pre-existing features / keys are
    unchanged, and an old checkpoint is expanded with ZERO new columns in
    :func:`wms.model.checkpoint.load_checkpoint` (exact identity at init).
    """

    def __init__(self, hidden: int = 64, cap: float = 10.0,
                 extra_features: bool = False) -> None:
        super().__init__()
        self.cap = float(cap)
        self.extra_features = bool(extra_features)
        n_in = self.n_features
        #: Trunk dropout (plain attribute, never saved; 0 = off). Only the offline pre-fit
        #: (tools/lanev5_pretrain.py --dropout) sets it, and resets it before saving.
        self.dropout = 0.0
        self.trunk = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.heads = nn.Linear(hidden, len(TRANSITIONS))
        nn.init.zeros_(self.heads.weight)
        with torch.no_grad():
            self.heads.bias.copy_(torch.tensor(HEAD_BIAS_INIT))
        self.register_buffer("feature_mu", torch.zeros(n_in))
        self.register_buffer("feature_sd", torch.ones(n_in))
        self.register_buffer("head_ap", torch.full((len(TRANSITIONS),), UNKNOWN_AP))

    @property
    def n_features(self) -> int:
        """Input width of this network (base only, or base + extra columns)."""
        return N_FEATURES_EXTRA if self.extra_features else N_FEATURES

    def feature_names(self) -> tuple[str, ...]:
        """Feature names in network-input order (``FEATURE_NAMES`` + extra when enabled)."""
        return FEATURE_NAMES_EXTRA if self.extra_features else FEATURE_NAMES

    def scale(self, raw: Tensor) -> Tensor:
        return (slog(raw) - self.feature_mu) / self.feature_sd.clamp_min(1.0e-6)

    def forward(self, raw: Tensor) -> Tensor:
        """Head logits ``[..., 3]`` (overtake, return, merge) from raw features ``[..., F]``."""
        if raw.shape[-1] != self.n_features:
            raise ValueError(f"lane v5 features must end in {self.n_features}")
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


#: MERGECAL (2026-10-18): per-position merge-head calibration ``(edges_m, offsets)``: ``edges_m``
#: ascending distances to the acceleration-lane end (``merge_zone_s1_m - s``), ``offsets`` one
#: logit offset per bin ``[edges_m[i], edges_m[i+1])``.  Written by ``tools/merge_calib.py`` into a
#: checkpoint copy (payload key :data:`MERGE_CALIB_KEY`, applied by ``load_checkpoint``); ``None``
#: (default) = off, bit-identical.
MERGE_CALIB_KEY = "lane_v5_merge_calib"


def check_merge_calib(calib) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate and normalise a merge-position calibration ``(edges_m, offsets)``."""
    edges = tuple(float(x) for x in calib[0])
    offs = tuple(float(x) for x in calib[1])
    if len(edges) < 2 or len(offs) != len(edges) - 1:
        raise ValueError("merge calibration needs n+1 bin edges and n offsets")
    if any(math.isnan(x) for x in edges) or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("merge calibration edges must be strictly ascending")
    if not all(math.isfinite(x) for x in offs):
        raise ValueError("merge calibration offsets must be finite")
    return edges, offs


def merge_position_offset(lane: Tensor, s: Tensor, physics: PhysicsConfig, calib) -> Tensor:
    """``[B, K]`` per-row merge-head logit offset of a :data:`MERGE_CALIB_KEY` calibration
    (0 off the ramp lane and outside the calibrated distance range)."""
    edges, offs = check_merge_calib(calib)
    d = (float(physics.merge_zone_s1_m) - s).to(torch.float32)
    e = torch.tensor(edges, dtype=torch.float32, device=s.device)
    idx = torch.bucketize(d, e, right=True) - 1          # bin i: edges[i] <= d < edges[i+1]
    inside = (idx >= 0) & (idx < len(offs)) & (lane == physics.merge_lane)
    table = torch.tensor(offs, dtype=torch.float32, device=s.device)
    return torch.where(inside, table[idx.clamp(0, len(offs) - 1)], torch.zeros_like(d))


def lane_logits_v5(net: LaneV5Net, raw: Tensor, inst: dict[str, Tensor], learned: Tensor,
                   lane_base: Tensor, physics: PhysicsConfig,
                   offsets: Optional[Sequence[float]] = None,
                   merge_wait: Optional[Tensor] = None,
                   merge_calib=None) -> tuple[Tensor, Tensor, Tensor]:
    """Absolute lane logits ``[B, K, 3]``, the per-row head logit and the 'acts' mask.

    The learned transition's entry is the head logit (+ offset) on learned slots whose target
    the physics allows (``lane_base > MASKED_LOGIT``) outside the forced window; a feasible
    merge in the last :data:`FORCE_WINDOW_M` gets ``max(physics, 8)``; everything else keeps
    the physics logit (masked targets stay masked; the current lane is 'stay' = 0).

    V8WM2 ``forced_window_mode='hazard'``: feasible forced-window rows act too (in ``acts``),
    with value and returned head logit ``max(head, logit(forced_window_min_hazard))``.
    V8WM3 ``physics.merge_impatience`` with a carried ``merge_wait`` ``[B, K]`` (s): the floor is
    per car, ``logit(min(1, h0 + (1 - h0) w / tau))`` capped at :data:`FORCED_MERGE_LOGIT`
    (:func:`wms.physics.mobil.merge_impatience_floor`).
    MERGECAL ``merge_calib = (edges_m, offsets)``: per-position merge-head logit offset
    (:func:`merge_position_offset`), added before the forced-window floor.
    """
    lane, tq, s = inst["lane"], inst["tq"], inst["s"]
    head = _lookup(HEAD_OF_LANE, lane)
    logit = net(raw).gather(-1, head.unsqueeze(-1)).squeeze(-1)
    if offsets is not None and any(float(x) != 0.0 for x in offsets):
        logit = logit + _lookup(tuple(float(x) for x in offsets), head, dtype=logit.dtype)
    if merge_calib is not None:
        logit = logit + merge_position_offset(lane, s, physics, merge_calib).to(logit.dtype)
    prior = lane_base.gather(-1, tq.unsqueeze(-1)).squeeze(-1)
    feasible = learned.bool() & (prior > MASKED_LOGIT)
    if getattr(physics, "lane_learnable", False):
        # V10UNBLOCK: every envelope-feasible row acts with the head's own logit; no forced
        # last-5 m replacement, no impatience / hazard floor, no AP gate.
        acts = feasible
        value = torch.where(acts, logit.to(lane_base.dtype), prior)
        logits = lane_base.scatter(-1, tq.unsqueeze(-1), value.unsqueeze(-1))
        return logits, logit, acts
    forced = forced_window(lane, s, physics)
    if forced_window_hazard(physics):
        # V8WM2: the learned hazard (floored) decides inside the window as well.
        if merge_wait is not None and getattr(physics, "merge_impatience", False):
            h = merge_impatience_floor(merge_wait, physics).clamp_max(1.0 - 1.0e-6)
            floor = torch.log(h / (1.0 - h)).clamp_max(FORCED_MERGE_LOGIT).to(logit.dtype)
            logit = torch.where(forced, torch.maximum(logit, floor), logit)
        else:
            logit = torch.where(forced, logit.clamp_min(forced_window_floor_logit(physics)),
                                logit)
        acts = feasible
        value = torch.where(acts, logit.to(lane_base.dtype), prior)
    else:
        acts = feasible & ~forced
        value = torch.where(acts, logit.to(lane_base.dtype), prior)
        value = torch.where(feasible & forced, prior.clamp_min(FORCED_MERGE_LOGIT), value)
    logits = lane_base.scatter(-1, tq.unsqueeze(-1), value.unsqueeze(-1))
    return logits, logit, acts


def forced_window(lane: Tensor, s: Tensor, physics: PhysicsConfig) -> Tensor:
    """``[B, K]``: ramp-lane rows inside the last :data:`FORCE_WINDOW_M` of the accel lane."""
    return (lane == physics.merge_lane) & (s >= physics.merge_zone_s1_m - FORCE_WINDOW_M) \
        & (s <= physics.merge_zone_s1_m)


def forced_window_hazard(physics: PhysicsConfig) -> bool:
    """V8WM2: the forced window keeps the learned merge hazard (``forced_window_mode``)."""
    return getattr(physics, "forced_window_mode", "deterministic") == "hazard"


def forced_window_floor_logit(physics: PhysicsConfig) -> float:
    """V8WM2 ``hazard`` mode: logit of ``forced_window_min_hazard`` (vs stay = 0)."""
    h = float(physics.forced_window_min_hazard)
    return math.log(h / (1.0 - h))


#: V8LANE (2026-10-17) mainline (overtake / return) decision rule, ``model.lane_v5_mainline``:
#: ``default`` = the ``lane_decision`` rule (gated: argmax unless the head's AP passes its gate);
#: ``sample`` = lane-0/1 cars always draw their own transition (Bernoulli of the calibrated
#: hazard); ``budget`` / ``budget_prop`` = population-matched ("rank-matched") decisions: per
#: batch row and head an accumulator adds the expected count ``sum P`` of the step and, each time
#: it passes 1, fires the highest-hazard feasible car (``budget``) or a car drawn in proportion
#: to its hazard (``budget_prop``, Gumbel top-k on the lane draws).  The expected count of a
#: rollout equals that of independent sampling, with (nearly) no count noise.
MAINLINE_MODES = ("default", "sample", "budget", "budget_prop")
#: Mainline heads (overtake: lane 1 cars, return: lane 0 cars) and their source lanes.
MAINLINE_SOURCE_LANES = (1, 0)


def mainline_rows(lane: Tensor, learned: Tensor) -> Tensor:
    """``[B, K]`` learned slots on the mainline (lanes 0 / 1), whose head is overtake / return."""
    return learned.bool() & (lane <= 1)


#: V10LANEOPT (2026-10-19) ``physics.mainline_lane_mode`` (``lane_learnable`` only):
#: ``free`` = every learned head draws its own calibrated probability (current behaviour);
#: ``courtesy`` = merges stay fully learned while the mainline heads only decide the learned
#: lane-1 -> lane-0 overtake of HDVs inside the courtesy approach zone upstream of the merge
#: (returns 0 -> 1 and overtakes elsewhere are frozen at the physics prior).  The lane-head LOSS
#: (``recorded_rows``) is identical in both modes, so the free stage starts calibrated.
MAINLINE_LANE_MODES = ("free", "courtesy")


def courtesy_zone_rows(lane: Tensor, s: Tensor, learned: Tensor,
                       physics: PhysicsConfig) -> Tensor:
    """``[B, K]`` mainline rows allowed to draw under ``mainline_lane_mode='courtesy'``.

    Only lane-1 learned (HDV) slots at ``s`` inside the approach zone
    ``[merge_zone_s1_m - mainline_courtesy_zone_m, merge_zone_s1_m]`` (model frame, acceleration
    lane end = ``merge_zone_s1_m``) qualify; lane-0 returns never do.  Ramp merges are not
    mainline rows and are unaffected (see :func:`courtesy_sample_mask`).
    """
    end = physics.merge_zone_s1_m
    zone = float(getattr(physics, "mainline_courtesy_zone_m", 0.0))
    return (mainline_rows(lane, learned) & (lane == 1)
            & (s >= end - zone) & (s <= end))


def courtesy_sample_mask(sample: Tensor, lane: Tensor, s: Tensor, learned: Tensor,
                         physics: PhysicsConfig) -> Tensor:
    """Restrict a lane-v5 draw mask to the ``courtesy`` mainline moves (rollout only).

    The merge head keeps every learned draw; the mainline heads keep only the courtesy overtakes
    of :func:`courtesy_zone_rows`.  Bit-identical to ``sample`` when
    ``physics.mainline_lane_mode`` is ``'free'``.
    """
    if getattr(physics, "mainline_lane_mode", "free") != "courtesy":
        return sample
    allowed = ~mainline_rows(lane, learned) | courtesy_zone_rows(lane, s, learned, physics)
    return sample.bool() & allowed


def restrict_courtesy_logits(logits: Tensor, lane: Tensor, s: Tensor, learned: Tensor,
                             physics: PhysicsConfig) -> Tensor:
    """Mask the target-lane logits of the mainline moves ``courtesy`` disables.

    Applies on the rollout decision only (the teacher-forced lane-head loss is unaffected): a
    blocked mainline row's transition entry becomes :data:`MASKED_LOGIT`, so both the sampled
    draw and the argmax fallback stay in the current lane.  Bit-identical to ``logits`` when
    ``physics.mainline_lane_mode`` is ``'free'``.
    """
    if getattr(physics, "mainline_lane_mode", "free") != "courtesy":
        return logits
    main = mainline_rows(lane, learned)
    block = main & ~courtesy_zone_rows(lane, s, learned, physics)
    target = _lookup(TARGET_OF_LANE, lane)
    value = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    value = torch.where(block, torch.full_like(value, MASKED_LOGIT), value)
    return logits.scatter(-1, target.unsqueeze(-1), value.unsqueeze(-1))

def budget_choice(choice: Tensor, logits: Tensor, feasible: Tensor, lane: Tensor,
                  learned: Tensor, budget: Optional[Tensor], uniform: Tensor,
                  mode: str) -> tuple[Tensor, Tensor]:
    """V8LANE population-matched mainline decisions (see :data:`MAINLINE_MODES`).

    ``choice [B, K]`` = the lane choice of the ordinary rule (kept for every non-mainline slot);
    ``logits [B, K, 3]`` / ``feasible [B, K, 3]`` as in
    :func:`wms.model.lane_hazard.sample_lane_choice`; ``budget [B, 2]`` the carried
    accumulators (``None`` at a rollout start: initialised from the row's first lane draw, a
    random systematic-sampling offset in [0, 1), so the expected count is unbiased); ``uniform
    [B, K]`` the step's lane draws.  Returns the new choice and the updated accumulators.
    """
    masked = torch.where(feasible, logits, torch.full_like(logits, float("-inf")))
    probs = torch.softmax(masked.detach().float(), -1)
    target = _lookup(TARGET_OF_LANE, lane)
    p = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    p = torch.nan_to_num(p, nan=0.0)
    main = mainline_rows(lane, learned)
    u = uniform.float().clamp(1.0e-12, 1.0 - 1.0e-7)
    if budget is None:
        budget = u[:, :1].expand(-1, len(MAINLINE_SOURCE_LANES)).clone()
    budget = budget.float()
    fire = torch.zeros_like(main)
    new_budget = []
    for h, src in enumerate(MAINLINE_SOURCE_LANES):
        rows = main & (lane == src) & (p > 0.0)
        ph = torch.where(rows, p, torch.zeros_like(p))
        acc = budget[:, h] + ph.sum(-1)
        n = torch.minimum(acc.floor(), rows.sum(-1).to(acc.dtype))
        if mode == "budget_prop":
            key = torch.log(ph.clamp_min(1.0e-30)) - torch.log(-torch.log(u))
        else:
            key = ph
        key = torch.where(rows, key, torch.full_like(key, float("-inf")))
        rank = key.argsort(-1, descending=True).argsort(-1)
        fh = rows & (rank < n.unsqueeze(-1).long())
        fire = fire | fh
        new_budget.append((acc - fh.sum(-1).to(acc.dtype)).clamp(0.0, 1.0))
    stay_or_go = torch.where(fire, target, lane)
    return torch.where(main, stay_or_go, choice), torch.stack(new_budget, -1)


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
                  acc0: Tensor, merge_wait0: Optional[Tensor] = None,
                  extra: bool = False) -> dict[str, Tensor]:
    """Teacher-forced rows of every recorded step of the windows.

    ``states [B, T+1, K, 7]`` / ``valid [B, T+1, K]`` (on-road, supervised) / ``is_cav``;
    ``lane_age [B, T, K]`` recorded steps since the last change; ``acc0`` pre-t0
    accumulators.  A row = an HDV-step t whose transition t -> t+1 is valid (both ends valid,
    at most one lane), whose own transition target the physics allows at t and that lies
    outside the forced window (V8WM2 ``forced_window_mode='hazard'``: inside it too).
    ``lane_base_fn(kin, present) -> [N, K, 3]`` gives the physics lane logits; ``None`` reads the feasibility from :func:`instant_features` (speed gap rule,
    :func:`fast_mask_applies`; identical, tested).  Returns raw features ``[R, F]``, head ``[R]``, label ``[R]``
    (1 = the car made its transition) plus counts (events, infeasible events).

    V8WM3 ``physics.merge_impatience`` with ``merge_wait0`` (``[B, K]``,
    :func:`wms.model.merge_wait.initial_merge_wait`): the fast feasibility mask uses the
    recorded waiting time (:func:`wms.model.merge_wait.recorded_merge_wait`), as the rollout.
    """
    b, t1, k = states.shape[:3]
    t = t1 - 1
    lanes = states[..., 1:4].argmax(-1)
    present = valid[:, :t].bool()
    impatience = None
    if merge_wait0 is not None and getattr(physics, "merge_impatience", False):
        from wms.model.merge_wait import recorded_merge_wait
        impatience = merge_impatience_level(
            recorded_merge_wait(states[:, :t], present, physics, merge_wait0), physics)
    feats, _, inst = sequence_features(states[:, :t], present, physics, lane_age, acc0,
                                       return_instant=True, impatience=impatience,
                                       extra=extra)
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
        forced = forced_window(prev, s, physics)
        head = _lookup(HEAD_OF_LANE, prev)
        if getattr(physics, "lane_learnable", False):
            # V10UNBLOCK all-risk rows: EVERY HDV step present at t is at risk (stay is a
            # survival negative; a recorded non-own transition is a negative too).  Rows whose
            # endpoint at t+1 is missing are censored (counted, excluded from the loss).  The
            # envelope-infeasible rows are excluded and counted.  Forced-window rows are
            # included (the window is no longer special).
            endpoint = valid[:, 1:].bool()
            label = (nxt == tq) & endpoint
            rows = hdv & feasible & endpoint   # astra bug 5: censored (missing t+1) rows carry no loss
            stats = {"rows": int(rows.sum()), "events": int((rows & label).sum()),
                     "infeasible_rows": int((hdv & ~feasible).sum()),
                     "infeasible_events": int(((nxt == tq) & endpoint & ~feasible).sum()),
                     "censored": int((hdv & ~endpoint).sum()),
                     "forced_window_rows": int((rows & forced).sum()),
                     "forced_window_events": int((rows & label & forced).sum())}
            for h, name in enumerate(TRANSITIONS):
                sel = rows & (head == h)
                stats[f"{name}_rows"] = int(sel.sum())
                stats[f"{name}_events"] = int((sel & label).sum())
            return {"features": feats[rows], "head": head[rows],
                    "label": label[rows].float(), "stats": stats}
        label = nxt == tq
        own = (nxt == prev) | label
        rows = ok & own & feasible
        if not forced_window_hazard(physics):
            rows = rows & ~forced       # V8WM2 hazard mode: the window's rows are learnable
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
    torch.save({"format": PRETRAIN_FORMAT, "feature_names": list(net.feature_names()),
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
    names = list(payload.get("feature_names", []))
    state = payload["state_dict"]
    if names != list(net.feature_names()):
        if net.extra_features and names == list(FEATURE_NAMES):
            # R5INPUTS: a pre-switch pre-fit loads with ZERO new columns (exact identity).
            state = expand_input_state(state, N_FEATURES, N_FEATURES_EXTRA)
        else:
            raise ValueError(f"{p}: lane v5 feature set differs from this code")
    device = net.feature_mu.device
    net.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "head_ap": [float(x) for x in net.head_ap.tolist()],
            "meta": {k: payload.get("meta", {}).get(k) for k in ("rows", "windows", "epochs")}}
