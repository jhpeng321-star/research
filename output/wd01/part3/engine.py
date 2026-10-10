"""WD-01 part 3 engine: rollouts (model + baselines + B4), phase-1 samples,
window metrics.  Zero simulation; device-agnostic torch.

Rollout conventions (documented in REGISTRATION_P3.md):
- integrator: v' = clip(v + a*0.5, 0, 1.2*lane_limit); s' = s + v'*0.5
  (the MEASURED SUMO form);
- lane changes (model only): greedy argmax of the topology-masked head,
  instantaneous at the step boundary (s unchanged);
- births: departed-only, inserted at the logged tick with logged
  (lane, s, v) carrying the TRUTH vid for id-matched metrics; z from the
  learned prior;
- exits: s beyond the physical lane length;
- collisions: per-lane order inversions after the step, counted, never
  repaired; hard cap a in [-9, 3] m/s^2; v cap 1.2 x lane limit.

kind 'idm_merge' (part 3k) = kind 'idm_sampled' (unchanged) plus SUMO-like
merge dynamics M1-M4 -- see the merge block near the module top; the
'idm_sampled' branch is untouched.
"""
from __future__ import annotations

from collections import defaultdict

import os
import sys

import numpy as np
import torch

from pathlib import Path as _P

HERE = _P(__file__).resolve().parent
P2C = HERE.parent / "partA_data"
PART4D = HERE.parent / "part4" / "delay"
sys.path.insert(0, str(P2C))
sys.path.insert(0, str(PART4D))
import pdl as PD
from geometry import wm30_geometry  # noqa: E402  (part 4, read-only)
import p3lib as P
from p3lib import (CKPT, DT, GAP_MAX, HORIZONS, LANE_LIMIT, N_HIST, OFFSETS, ROLL_TICKS,
                   VTYPES, PartAModel, device, ego_features, history_at,
                   idm_acc, lane_lengths, neighbour_slots, root_arrays)

SLOT_OWN_LEAD = 0
GEO = None
MERGE_ZONE_S0 = 378.66            # accel mandatory-merge zone start
NOISE_MU = {(0, 1): -0.75, (0, 0): -0.40,
            (1, 1): -0.24, (1, 0): -0.21,
            (2, 1): -0.22, (2, 0): -0.08}      # (vtype, free/following)
SIGMA_ACCEL = {0: 0.6 * 3.0, 1: 0.4 * 2.6, 2: 0.2 * 2.0}
NOISE_SD = {(0, 1): 0.90, (0, 0): 0.39,
            (1, 1): 0.57, (1, 0): 0.26,
            (2, 1): 0.46, (2, 0): 0.41}

# ---- SUMO-like merge dynamics (kind 'idm_merge', part 3k) -----------------
# p3j measured the missing delay as merge behaviour: in SUMO, ramp vehicles
# QUEUE in the mandatory merge zone waiting for gaps and main_right slows
# around the merge; idm_sampled's stochastic LC hazard lets them slip in
# immediately.  idm_merge = idm_sampled with:
#   M1 accel-lane end as a virtual standing obstacle at s = ACCEL_END_S
#      (IDM gap to the lane end) so unmerged vehicles decelerate and queue;
#   M2 mandatory gap acceptance accel->main_right inside the merge zone:
#      merge iff gap-to-target-leader >= minGap + max(0, v*tau_accept) AND
#      the target-lane follower's required deceleration <= b_accept
#      (fires as soon as acceptable; otherwise the vehicle keeps waiting;
#      the idm_sampled stochastic accel->main_right merge is removed);
#   M3 cooperative slow-down: a main_right follower in the merge d-window
#      whose leader-to-be is a waiting accel vehicle ahead within COOP_LOOK
#      adapts its IDM leader to that vehicle with per-vType probability
#      LC_COOP_PRIOR * p_coop_scale (SUMO lcCooperative approximation);
#   M4 discretionary 0<->1 changes stay on the existing LC hazard.
ACCEL_END_S = 454.10              # accel lane end (== p3lib.ACCEL_LEN)
MERGE_ZONE_MAIN_D = (-100.0, 100.0)   # main_right merge d-window (M3)
COOP_LOOK = 100.0                 # max d-gap follower->waiting leader (M3)
LC_COOP_PRIOR = (0.3, 0.6, 1.0)   # lcCooperative prior per vType 0/1/2
# part 3m TASK B knobs (default-neutral: 0.0 keeps the p3k/p3l behaviour
# exactly; the p3m fit selects them on TRAIN):
#   end_force_s   > 0 relaxes M2 acceptance for accel vehicles within
#                 this distance of the lane end (SUMO strategic
#                 urgency): tau_accept capped at MERGE_FORCE_TAU and
#                 b_accept floored at MERGE_FORCE_B, so the ramp
#                 discharges with ROLLING merges instead of stopping at
#                 the lane-end obstacle (part 3m TASK A: the excess
#                 TTC<3 exposure concentrates in the stop-and-go queue
#                 DOWNSTREAM of the merge point);
#   v_merge_blend in [0, 1]: a just-merged vehicle's speed is blended
#                 toward the mean speed of target-lane vehicles within
#                 +/- MERGE_BLEND_WINDOW m of its d (post-merge
#                 relaxation candidate; 0 keeps the merge speed).
MERGE_FORCE_TAU = 0.5
MERGE_FORCE_B = 6.0
MERGE_BLEND_WINDOW = 50.0
MERGE_PARAMS_DEFAULT = dict(tau_accept=1.1, b_accept=3.0, p_coop_scale=1.0,
                            end_force_s=0.0, v_merge_blend=0.0)


def end_force_params(s, tau_accept, b_accept, end_force_s,
                     force_tau=MERGE_FORCE_TAU, force_b=MERGE_FORCE_B):
    """M2 near-lane-end acceptance relaxation (pure): an accel vehicle
    within end_force_s m of the lane end uses (min(tau, force_tau),
    max(b, force_b)); end_force_s <= 0 (default) is the exact p3k/p3l
    behaviour."""
    if end_force_s <= 0.0:
        return tau_accept, b_accept
    if (ACCEL_END_S - s) <= float(end_force_s):
        return min(tau_accept, force_tau), max(b_accept, force_b)
    return tau_accept, b_accept


def local_target_speed(d_all, v_all, on_target, d_query,
                       window=MERGE_BLEND_WINDOW):
    """Mean speed of target-lane vehicles within +/- window in d of each
    query d (pure numpy); -1.0 where the window is empty."""
    d_all = np.asarray(d_all, dtype=np.float64)
    v_all = np.asarray(v_all, dtype=np.float64)
    on_target = np.asarray(on_target, dtype=bool)
    out = np.full(len(d_query), -1.0, dtype=np.float64)
    for k, dq in enumerate(d_query):
        m = on_target & (np.abs(d_all - dq) <= window)
        if m.any():
            out[k] = float(v_all[m].mean())
    return out


def merge_acceptance(gap_lead, v_ego, s0_ego, gap_foll, v_foll, s0_foll,
                     tau_accept, b_accept, return_parts=False):
    """M2 gap acceptance (pure, numpy, vectorized): merge allowed iff the
    d-gap to the target-lane leader >= minGap + max(0, v*tau_accept) AND
    the target-lane follower's required deceleration
    b_req = (v_foll^2 - v_ego^2) / (2 * max(gap_foll - minGap_foll, 0.3))
    to keep its safe gap is <= b_accept.  gap_* == GAP_MAX means the slot
    is empty (no leader / no follower).

    return_parts=True (part 3l TASK A instrumentation, default off)
    returns (ok, ok_lead, ok_foll) so callers can attribute rejections to
    the lead-gap vs the follower-deceleration term."""
    gap_lead = np.asarray(gap_lead, dtype=np.float64)
    gap_foll = np.asarray(gap_foll, dtype=np.float64)
    v_ego = np.asarray(v_ego, dtype=np.float64)
    v_foll = np.asarray(v_foll, dtype=np.float64)
    s0_ego = np.asarray(s0_ego, dtype=np.float64)
    s0_foll = np.asarray(s0_foll, dtype=np.float64)
    ok_lead = gap_lead >= s0_ego + np.maximum(0.0, v_ego * tau_accept)
    has_f = gap_foll < GAP_MAX
    with np.errstate(divide="ignore", invalid="ignore"):
        b_req = (v_foll ** 2 - v_ego ** 2) / \
            (2.0 * np.maximum(gap_foll - s0_foll, 0.3))
    ok_foll = (~has_f) | (b_req <= b_accept)
    if return_parts:
        return ok_lead & ok_foll, ok_lead, ok_foll
    return ok_lead & ok_foll


def coop_leader_map(lane, s, v, vtype, u, p_coop):
    """M3 (pure, numpy): main_right vehicles inside MERGE_ZONE_MAIN_D whose
    nearest accel vehicle ahead in d is within COOP_LOOK adapt their IDM
    leader to it where u < p_coop (u, p_coop per-vehicle arrays).  Returns
    (follower_idx, d_gap, leader_v) numpy arrays (possibly empty)."""
    lane = np.asarray(lane)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = np.asarray(OFFSETS)[lane] - s
    l1 = np.where((lane == 1)
                  & (d >= MERGE_ZONE_MAIN_D[0])
                  & (d <= MERGE_ZONE_MAIN_D[1]))[0]
    w2 = np.where((lane == 2)
                  & (d >= MERGE_ZONE_MAIN_D[0])
                  & (d <= MERGE_ZONE_MAIN_D[1]))[0]
    fi, fg, fv = [], [], []
    if len(l1) and len(w2):
        for i in l1:
            # d decreases downstream: the leader-to-be is the nearest accel
            # vehicle with SMALLER d
            cand = w2[d[w2] < d[i]]
            if not len(cand):
                continue
            m = cand[np.argmax(d[cand])]
            gap = d[i] - d[m]
            if gap > COOP_LOOK or u[i] >= p_coop[i]:
                continue
            fi.append(int(i))
            fg.append(gap)
            fv.append(float(v[m]))
    return (np.asarray(fi, dtype=np.int64),
            np.asarray(fg, dtype=np.float64),
            np.asarray(fv, dtype=np.float64))


def lane_end_obstacle_acc(v, s, on_accel, a_max, b, tau, s0, v0):
    """M1 (pure, torch): IDM acceleration against the virtual standing
    obstacle at the accel-lane end; vehicles off the accel lane get +inf
    (unconstrained; take the min with the ordinary IDM acceleration)."""
    gap_end = (ACCEL_END_S - s).clamp(min=0.3)
    acc = idm_acc(v, gap_end, v.clone(), a_max, b, tau, s0, v0)
    return torch.where(on_accel, acc, torch.full_like(acc, 1e9))
_P3D_MODELS = None


def _p3d_weights_dir():
    """Untracked part3d weights dir; WD01_DATA_ROOT (part 3k) points at a
    populated copy of output/wd01, default keeps the in-repo layout."""
    dr = os.environ.get("WD01_DATA_ROOT", "").strip()
    base = _P(dr) if dr else HERE.parent
    return base / "part3d" / "weights" / "run3d"


def _p3d_module():
    p3d_dir = HERE.parent / "part3d"
    if str(p3d_dir) not in sys.path:
        sys.path.insert(0, str(p3d_dir))
    import models as M
    return M


def _p3d_models():
    global _P3D_MODELS
    if _P3D_MODELS is None:
        M = _p3d_module()
        wdir = _p3d_weights_dir()
        post = M.Posterior()
        sd = torch.load(wdir / "posterior.pt",
                        map_location="cpu", weights_only=True)
        post.load_state_dict(sd["model_state"])
        post.eval()
        haz = M.Hazard()
        sd = torch.load(wdir / "hazard.pt",
                        map_location="cpu", weights_only=True)
        haz.load_state_dict(sd["model_state"])
        haz.eval()
        _P3D_MODELS = (post, haz)
    return _P3D_MODELS


def _p3d_posterior():
    return _p3d_models()[0]


def _p3d_hazard():
    return _p3d_models()[1]


def geometry():
    global GEO
    if GEO is None:
        GEO = wm30_geometry()
    return GEO


_SEG_TABLE = None


def _seg_table():
    """{lane: (segment_starts, segment_limits)} for vectorized lookup."""
    global _SEG_TABLE
    if _SEG_TABLE is None:
        geo = geometry()
        _SEG_TABLE = {}
        for li in (0, 1, 2):
            segs = geo.segments[li]
            starts = np.array([sg.s0 for sg in segs], dtype=np.float64)
            lims = np.array([sg.speed_limit for sg in segs],
                            dtype=np.float64)
            _SEG_TABLE[li] = (starts, lims)
    return _SEG_TABLE


def lane_speed_limits(lane, s):
    """Network speed limit per vehicle (numpy, vectorized): the
    containing segment's SUMO lane limit (main 31.29; accel ramp 22.22,
    merge 31.29)."""
    tbl = _seg_table()
    out = np.full(len(lane), np.nan, dtype=np.float64)
    for li in (0, 1, 2):
        m = lane == li
        if not m.any():
            continue
        starts, lims = tbl[li]
        idx = np.searchsorted(starts, s[m], side="right") - 1
        idx = np.clip(idx, 0, len(lims) - 1)
        out[m] = lims[idx]
    return out
ADJ = {0: (None, 1), 1: (0, 2), 2: (1, None)}
LARR = np.array(LANE_LIMIT)
_LIM_T = None


def _lim_tensor(dev):
    global _LIM_T
    if _LIM_T is None or _LIM_T.device != dev:
        _LIM_T = torch.tensor(LANE_LIMIT, device=dev, dtype=torch.float64)
    return _LIM_T


class Rollout:
    """One window rollout at the 0.5 s step for kind in
    {model, persistence (B1), idm_normal (B2), idm_oracle (B3)}."""

    def __init__(self, root, t0, kind="model", weights=None, dev=None,
                 member_seed=0, lc_scale=1.0, merge_params=None,
                 dawdle_scale=1.0, lc_scale_disc=None,
                 merge_accept_prob=1.0, merge_cooldown_ticks=0,
                 merge_early_d=0.0, merge_early_prob=None):
        assert kind in ("model", "persistence", "idm_normal", "idm_oracle",
                        "idm_sampled", "idm_merge")
        self.ra = ra = root_arrays(root)
        self.dev = dev or device()
        self.kind = kind
        self.root, self.t0 = root, float(t0)
        self.k0 = k0 = int(round((t0 - PD.WINDOW_T0) / DT))
        assert 0 <= k0 and k0 + ROLL_TICKS < PD.N_TICKS
        idx = ra.tick_rows(k0)
        idx = idx[ra.lane_m[idx] >= 0]
        self.vid = ra.vid[idx].copy()
        self.lane = torch.as_tensor(ra.lane_m[idx], dtype=torch.long,
                                    device=self.dev).clone()
        self.s = torch.as_tensor(ra.s[idx], dtype=torch.float64,
                                 device=self.dev).clone()
        self.v = torch.as_tensor(ra.v[idx], dtype=torch.float64,
                                 device=self.dev).clone()
        self.a_prev = torch.as_tensor(ra.a[idx], dtype=torch.float64,
                                      device=self.dev).clone()
        self.cav = torch.as_tensor(ra.cav[idx], dtype=torch.float32,
                                   device=self.dev).clone()
        self.typ = ra.typ[idx].copy()
        lim = torch.as_tensor(lane_speed_limits(
            self.lane.cpu().numpy(), self.s.detach().cpu().numpy()),
            device=self.dev, dtype=torch.float64)
        self.lim = lim.clone()
        self.vcap = 1.2 * lim
        self.lc_threshold = 0.5
        self.member_seed = int(member_seed)
        self.lc_scale = float(lc_scale)
        # part 3o: LC-by-type hazard scales (gate-v2 (b) over-count fix).
        # lc_scale_disc scales ONLY the idm_merge discretionary 0<->1
        # hazard (None falls back to lc_scale, the shipped behaviour);
        # merge_accept_prob thins accepted M2 merges (1.0 = the shipped
        # deterministic acceptance; < 1.0 accepts a gap-accepted candidate
        # with that probability from a dedicated RNG stream, so the
        # shared streams and the flag==1.0 path are untouched).
        self.lc_scale_disc = (float(lc_scale_disc)
                              if lc_scale_disc is not None
                              else self.lc_scale)
        self.merge_accept_prob = float(merge_accept_prob)
        self._merge_rng = None
        # part 3q: count-effective merge levers, ALL default no-ops.
        # merge_cooldown_ticks adds hysteresis to the thinning path: a
        # candidate rejected BY THINNING cannot re-attempt M2 for this
        # many ticks (p3o evidence: per-tick retries make plain thinning
        # delay-only, ~1/p ticks).  merge_early_d / merge_early_prob make
        # the thinning probability position-dependent: a candidate still
        # EARLY in the zone (d = OFFSETS[2] - s > merge_early_d; d falls
        # from 0 at zone entry to -75.4 at the lane end) draws against
        # merge_early_prob instead of merge_accept_prob.  Defaults
        # (ticks 0; early_d 0.0 -> empty early set; early_prob None)
        # never draw on a new path and never create cooldown state, so
        # the merge decision stays byte-identical to shipped.
        self.merge_cooldown_ticks = int(merge_cooldown_ticks)
        self.merge_early_d = float(merge_early_d)
        self.merge_early_prob = (None if merge_early_prob is None
                                 else float(merge_early_prob))
        self._merge_cd = None
        # bookkeeping only (never read by the dynamics): one record per
        # executed M2 merge -- the part 3q merge-position/timing
        # diagnosis basis.
        self.merge_events = []
        self.dawdle_scale = float(dawdle_scale)
        self.mp = dict(MERGE_PARAMS_DEFAULT)
        if merge_params:
            self.mp.update(merge_params)
        self.rng = np.random.default_rng((int(member_seed), int(t0)))
        self.vtype = ra.typ[idx].copy()
        # part 3m TASK A instrumentation: vid -> SAMPLED vType for every
        # vehicle ever in this rollout (the live self.vtype is re-indexed
        # by exits/inserts, so a per-tick snapshot cannot recover it).
        # Pure bookkeeping; no behaviour change.
        self.vtype_map = {int(v): int(t)
                          for v, t in zip(self.vid, self.vtype)}
        self.merge_d = []
        self.p3d_post = _p3d_posterior() if kind in ("idm_sampled",
                                                     "idm_merge") else None
        self.p3d_haz = _p3d_hazard() if kind in ("idm_sampled",
                                                 "idm_merge") else None
        if kind in ("idm_sampled", "idm_merge"):
            self.member_rng = np.random.default_rng(
                (self.member_seed, int(t0)))
            hist = P.history_at(ra, k0)
            H = np.zeros((len(self.vid), 20, 3), dtype=np.float32)
            has = np.zeros(len(self.vid), dtype=bool)
            for r_, vi in enumerate(self.vid):
                if vi in hist:
                    H[r_] = hist[vi]
                    has[r_] = True
            with torch.no_grad():
                probs = torch.softmax(self.p3d_post(torch.as_tensor(
                    H, device=self.dev)), dim=1).cpu().numpy()
            prior_cdf = np.cumsum([0.1, 0.7, 0.2])
            u = self.member_rng.random(len(self.vid))
            self.vtype = np.empty(len(self.vid), dtype=np.int64)
            for r_ in range(len(self.vid)):
                cdf = np.cumsum(probs[r_]) if has[r_] else prior_cdf
                self.vtype[r_] = int(np.searchsorted(cdf, u[r_], "left"))
            self.probs = probs.copy()
        ll = lane_lengths()
        self.llen = torch.tensor([ll[0], ll[1], ll[2]], device=self.dev,
                                 dtype=torch.float64)[self.lane].clone()
        self.model = None
        self.z = None
        if kind == "model":
            assert weights is not None
            self.model = PartAModel().to(self.dev)
            sd = torch.load(weights, map_location=self.dev,
                            weights_only=True)
            self.model.load_state_dict(sd["model_state"])
            self.model.eval()
            self._init_z()
        self.births = defaultdict(list)
        bm = ra.b_dep & (ra.b_tick > k0) & (ra.b_tick <= k0 + ROLL_TICKS)
        bm &= ra.b_lane >= 0
        for i in np.where(bm)[0]:
            self.births[int(ra.b_tick[i]) - k0].append(
                (int(ra.b_vid[i]), int(ra.b_lane[i]), float(ra.b_s[i]),
                 float(ra.b_v[i]), float(ra.b_cav[i])))
        self.collisions = 0
        self.capviol = 0
        self.lc_count = 0
        self.n_birthed = 0
        self.n_exited = 0
        self.snapshots = {}

    # -- z at t0 from the truth history (frozen for the window; newborns
    # get the learned prior at insertion)
    def _init_z(self):
        hist = history_at(self.ra, self.k0)
        H = np.zeros((len(self.vid), N_HIST, 3), dtype=np.float32)
        for r, vi in enumerate(self.vid):
            if vi in hist:
                H[r] = hist[vi]
        Ht = torch.as_tensor(H, device=self.dev)
        slots = self._slots_t()
        prior_in = PartAModel.prior_input(
            self.lane, self.v.float(), slots[:, 0, 1], slots[:, 0, 2],
            self.cav)
        with torch.no_grad():
            self.z = self.model.z_of(Ht, prior_in)

    def _slots_t(self):
        sl = neighbour_slots(self.lane.cpu().numpy(),
                             self.s.detach().cpu().numpy(),
                             self.v.detach().cpu().numpy(),
                             self.cav.cpu().numpy())
        return torch.as_tensor(sl, device=self.dev)

    def snapshot(self):
        return dict(
            vid=self.vid.copy(),
            lane=self.lane.detach().cpu().numpy().copy(),
            s=self.s.detach().cpu().numpy().copy(),
            v=self.v.detach().cpu().numpy().copy(),
            a=self.a_prev.detach().cpu().numpy().copy(),
            collisions=self.collisions, capviol=self.capviol,
            lc=self.lc_count, n_birthed=self.n_birthed)

    @torch.no_grad()
    def step(self, j):
        for (tvid, l, ss, vv, cc) in self.births.get(j, ()):
            self._insert(tvid, l, ss, vv, cc)
        if len(self.vid) == 0:
            return
        # refresh per-vehicle segment speed limits every step (vehicles
        # cross segment boundaries; defect 4b)
        self.lim = torch.as_tensor(lane_speed_limits(
            self.lane.detach().cpu().numpy(),
            self.s.detach().cpu().numpy()), device=self.dev,
            dtype=torch.float64)
        self.vcap = 1.2 * self.lim
        pre_lane = self.lane.clone()
        pre_s = self.s.clone()
        self.vid_of = torch.as_tensor(self.vid, device=self.dev)
        pre_leader = same_lane_leader_idx(pre_lane, pre_s)
        if self.kind == "model":
            slots = self._slots_t()
            ego = ego_features(self.lane, self.s.float(), self.v.float(),
                               self.a_prev.float(), self.cav)
            da, lc_logits, _ = self.model(ego, slots, self.z)
            th = self.model.theta_of(self.z)
            gap = slots[:, 0, 1].double()
            rel = slots[:, 0, 2].double()
            free = slots[:, 0, 0] < 0.5
            gap = torch.where(free, torch.full_like(gap, GAP_MAX), gap)
            dv = torch.where(free, torch.zeros_like(rel), -rel)
            v0 = th["sf"] * self.lim
            aidm = idm_acc(self.v, gap, dv, th["a_max"], th["b"], th["tau"],
                           th["s0"], v0)
            a = (aidm + torch.tanh(da.double()) * P.RES_SCALE).clamp(-9.0, 3.0)
            v_new = (self.v + a * DT).clamp_min(0.0).clamp_max(self.vcap)
            s_new = self.s + v_new * DT          # measured SUMO form
            mask = torch.as_tensor([[1., 1., 0.], [1., 1., 0.],
                                    [0., 1., 1.]],
                                   device=self.dev)[self.lane]
            # accel -> main_right only inside the mandatory merge zone
            acc_far = (self.lane == 2) & (self.s < MERGE_ZONE_S0)
            mask[acc_far, 1] = 0.0
            probs = torch.softmax(lc_logits.float(), dim=1) * mask
            tgt = probs.argmax(1)
            conf = probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
            changed = (tgt != self.lane) & (conf >= self.lc_threshold)
            self.lc_count += int(changed.sum())
            # keep d constant across the change (d = OFFSET[lane] - s)
            off = torch.as_tensor(OFFSETS, device=self.dev,
                                  dtype=torch.float64)
            self.s = torch.where(changed,
                                 self.s + off[tgt] - off[self.lane],
                                 self.s)
            self.lane = torch.where(changed, tgt, self.lane)
            # refresh per-vehicle lane attributes (defect 3)
            if changed.any():
                lim2 = torch.as_tensor(lane_speed_limits(
                    self.lane.cpu().numpy(), self.s.detach().cpu().numpy()),
                    device=self.dev, dtype=torch.float64)
                self.lim = torch.where(changed, lim2, self.lim)
                self.vcap = 1.2 * self.lim
                ll = lane_lengths()
                llen2 = torch.as_tensor([ll[0], ll[1], ll[2]], device=self.dev,
                                        dtype=torch.float64)[self.lane]
                self.llen = torch.where(changed, llen2, self.llen)
            self.capviol += int((v_new > self.vcap + 1e-6).sum())
            self.a_prev = a
            self.v, self.s = v_new, s_new
        elif self.kind == "persistence":
            self.a_prev = torch.zeros_like(self.v)
            self.s = self.s + self.v * DT
        elif self.kind == "idm_sampled":
            # structured member: stochastic hazard lane change first
            # (d kept constant, attributes refreshed), then IDM + noise
            n = len(self.vid)
            if n:
                sl = self._slots_t()
                feats = torch.zeros((n, 19), device=self.dev)
                feats[:, 0] = self.v.float() / 40.0
                feats[:, 1] = torch.clamp(
                    (454.10 - self.s.float()) / 500.0, 0.0, 1.0)
                feats[:, 2:14] = torch.clamp(
                    sl[:, :, 1:3].reshape(n, 12).float() / 250.0,
                    -1.0, 1.0)
                feats[:, 14] = (
                    (self.lane == 2) & (self.s >= MERGE_ZONE_S0)
                ).double()
                feats[:, 15:19] = 0.0   # trained with zero tail
                logits = self.p3d_haz(feats)
                pl = torch.sigmoid(logits[:, 0]).double()
                pr_ = torch.sigmoid(logits[:, 1]).double()
                ul = torch.as_tensor(self.rng.random(n), device=self.dev)
                ur = torch.as_tensor(self.rng.random(n), device=self.dev)
                fire_l = (self.lane >= 1) & (ul < pl * self.lc_scale)
                fire_r = (self.lane <= 0) & (ur < pr_ * self.lc_scale)
                off = torch.as_tensor(OFFSETS, device=self.dev,
                                      dtype=torch.float64)
                new_lane = self.lane.clone()
                new_lane = torch.where(fire_l, self.lane - 1, new_lane)
                new_lane = torch.where(fire_r, self.lane + 1, new_lane)
                changed = new_lane != self.lane
                self.lc_count += int(changed.sum())
                off_new = off[new_lane]
                off_old = off[self.lane]
                self.s = torch.where(changed, self.s + off_new - off_old,
                                     self.s)
                self.lane = new_lane
                lim = torch.as_tensor(lane_speed_limits(
                    self.lane.detach().cpu().numpy(),
                    self.s.detach().cpu().numpy()), device=self.dev,
                    dtype=torch.float64)
            else:
                lim = torch.as_tensor(lane_speed_limits(
                    self.lane.detach().cpu().numpy(),
                    self.s.detach().cpu().numpy()), device=self.dev,
                    dtype=torch.float64)
            lim = torch.as_tensor(lane_speed_limits(
                self.lane.detach().cpu().numpy(),
                self.s.detach().cpu().numpy()), device=self.dev,
                dtype=torch.float64)
            slots = self._slots_t()
            gap = slots[:, 0, 1].double()
            rel = slots[:, 0, 2].double()
            free = slots[:, 0, 0] < 0.5
            gap = torch.where(free, torch.full_like(gap, GAP_MAX), gap)
            dv = torch.where(free, torch.zeros_like(rel), -rel)
            a_max = torch.as_tensor(self._per_type("a_max"), device=self.dev)
            b = torch.as_tensor(self._per_type("b"), device=self.dev)
            tau = torch.as_tensor(self._per_type("tau"), device=self.dev)
            s0v = torch.as_tensor(self._per_type("s0"), device=self.dev)
            sf = torch.as_tensor(self._per_type("sf"), device=self.dev)
            v0 = sf * lim
            acc = idm_acc(self.v, gap, dv, a_max, b, tau, s0v, v0)
            # measured residual noise per (vtype, regime)
            reg_sd = torch.as_tensor([
                (NOISE_SD[(int(t), 1 if fr else 0)])
                for t, fr in zip(self.vtype, free)], device=self.dev,
                dtype=torch.float64)
            reg_mu = torch.as_tensor([
                (NOISE_MU[(int(t), 1 if fr else 0)])
                for t, fr in zip(self.vtype, free)], device=self.dev,
                dtype=torch.float64)
            acc = acc.clamp(-9.0, 3.0)
            # SUMO dawdle: v' = max(0, v_idm - U(0,1) * sigma * accel * dt)
            dawdle_max = np.array([
                SIGMA_ACCEL[(int(t))] * DT for t in self.vtype])
            dawd = self.rng.random(len(acc)) * dawdle_max
            v_dawdled = np.maximum(
                self.v.detach().cpu().numpy() + acc.detach().cpu().numpy()
                * DT - dawd, 0.0)
            acc = torch.as_tensor(
                (v_dawdled - self.v.detach().cpu().numpy()) / DT,
                device=self.dev, dtype=torch.float64)
            v_new = (self.v + acc * DT).clamp_min(0.0).clamp_max(1.2 * lim)
            s_new = self.s + v_new * DT
            self.capviol += int((v_new > self.vcap + 1e-6).sum())
            self.a_prev = acc
            self.v, self.s = v_new, s_new
        elif self.kind == "idm_merge":
            # structured member + SUMO-like merge dynamics (part 3k):
            # M2 mandatory gap-accepted merge accel(2)->main_right(1) in
            # the merge zone, M4 hazard only for 0<->1, M3 cooperative
            # leaders on main_right, M1 lane-end standing obstacle, then
            # the idm_sampled dynamics (per-type theta + noise + dawdle).
            n = len(self.vid)
            self._m2_tick = j          # tick for _gap_accept_merge
            merged = self._gap_accept_merge()
            sl = self._slots_t()
            feats = torch.zeros((n, 19), device=self.dev)
            feats[:, 0] = self.v.float() / 40.0
            feats[:, 1] = torch.clamp(
                (454.10 - self.s.float()) / 500.0, 0.0, 1.0)
            feats[:, 2:14] = torch.clamp(
                sl[:, :, 1:3].reshape(n, 12).float() / 250.0,
                -1.0, 1.0)
            feats[:, 14] = (
                (self.lane == 2) & (self.s >= MERGE_ZONE_S0)
            ).double()
            feats[:, 15:19] = 0.0   # trained with zero tail
            logits = self.p3d_haz(feats)
            pl = torch.sigmoid(logits[:, 0]).double()
            pr_ = torch.sigmoid(logits[:, 1]).double()
            ul = torch.as_tensor(self.rng.random(n), device=self.dev)
            ur = torch.as_tensor(self.rng.random(n), device=self.dev)
            # M4: discretionary hazard restricted to 0<->1 (accel->main
            # right is governed by M2 gap acceptance only); part 3o:
            # scaled by lc_scale_disc (per-type lever), not lc_scale
            # (getattr default keeps hand-built __new__ stubs working)
            disc = getattr(self, "lc_scale_disc", self.lc_scale)
            fire_l = (self.lane == 1) & (ul < pl * disc)
            fire_r = (self.lane == 0) & (ur < pr_ * disc)
            off = torch.as_tensor(OFFSETS, device=self.dev,
                                  dtype=torch.float64)
            new_lane = self.lane.clone()
            new_lane = torch.where(merged, torch.ones_like(new_lane),
                                   new_lane)
            new_lane = torch.where(fire_l, self.lane - 1, new_lane)
            new_lane = torch.where(fire_r, self.lane + 1, new_lane)
            changed = new_lane != self.lane
            self.lc_count += int(changed.sum())
            mg = merged.detach().cpu().numpy()
            if mg.any():
                d_m = (OFFSETS[2] - self.s.detach().cpu().numpy())[mg]
                self.merge_d.extend(float(x) for x in d_m)
                # part 3q bookkeeping: per-event position/timing record
                # (s on the accel lane BEFORE the change, global frame
                # shared with the truth lc_s rows)
                s_mg = self.s.detach().cpu().numpy()[mg]
                v_mg = self.v.detach().cpu().numpy()[mg]
                me = getattr(self, "merge_events", None)
                if me is None:       # hand-built stubs skip __init__
                    me = self.merge_events = []
                for vid_i, s_i, d_i, v_i in zip(self.vid[mg], s_mg, d_m,
                                                v_mg):
                    me.append(dict(
                        tick=int(j), vid=int(vid_i), s=float(s_i),
                        d=float(d_i), v=float(v_i)))
                # part 3m knob: post-merge relaxation -- blend the
                # just-merged vehicles' speed toward the mean speed of
                # target-lane vehicles within +/- MERGE_BLEND_WINDOW m
                # of their d (v_merge_blend 0.0, the shipped default,
                # keeps the merge speed exactly as before)
                blend = float(self.mp.get("v_merge_blend", 0.0))
                if blend > 0.0:
                    lane_np = self.lane.detach().cpu().numpy()
                    s_np = self.s.detach().cpu().numpy()
                    v_np = self.v.detach().cpu().numpy()
                    d_all = np.array([OFFSETS[l] - s for l, s in
                                      zip(lane_np, s_np)])
                    tgt = local_target_speed(d_all, v_np,
                                             lane_np == 1, d_m)
                    has = tgt >= 0.0
                    if has.any():
                        v_mg = v_np[mg]
                        v_new_mg = np.where(
                            has, (1.0 - blend) * v_mg
                            + blend * np.maximum(tgt, 0.0), v_mg)
                        sel = np.where(mg)[0][has]
                        self.v[torch.as_tensor(
                            sel, device=self.dev,
                            dtype=torch.long)] = torch.as_tensor(
                            v_new_mg[has], device=self.dev,
                            dtype=torch.float64)
            lim = torch.as_tensor(lane_speed_limits(
                self.lane.detach().cpu().numpy(),
                self.s.detach().cpu().numpy()), device=self.dev,
                dtype=torch.float64)
            if changed.any():
                self.s = torch.where(changed, self.s + off[new_lane]
                                     - off[self.lane], self.s)
                self.lane = new_lane
                lim = torch.as_tensor(lane_speed_limits(
                    self.lane.detach().cpu().numpy(),
                    self.s.detach().cpu().numpy()), device=self.dev,
                    dtype=torch.float64)
                # refresh ALL per-vehicle lane attributes (unlike
                # idm_sampled, which leaves stale llen/vcap after a
                # merge: a merged vehicle must not be dropped by
                # _drop_exits through its old accel-lane length)
                self.lim = lim
                self.vcap = 1.2 * lim
                ll = lane_lengths()
                self.llen = torch.where(
                    changed,
                    torch.as_tensor([ll[0], ll[1], ll[2]], device=self.dev,
                                    dtype=torch.float64)[self.lane],
                    self.llen)
            # M3: cooperative slow-down on the post-change state
            fi, fg, fv = self._coop_leaders()
            slots = self._slots_t()
            gap = slots[:, 0, 1].double()
            rel = slots[:, 0, 2].double()
            free = slots[:, 0, 0] < 0.5
            gap = torch.where(free, torch.full_like(gap, GAP_MAX), gap)
            dv = torch.where(free, torch.zeros_like(rel), -rel)
            if len(fi):
                ft = torch.as_tensor(fi, device=self.dev, dtype=torch.long)
                gap = gap.clone()
                dv = dv.clone()
                gap[ft] = torch.as_tensor(
                    fg, device=self.dev,
                    dtype=torch.float64).clamp(min=0.3)
                dv[ft] = self.v[ft] - torch.as_tensor(
                    fv, device=self.dev, dtype=torch.float64)
            a_max = torch.as_tensor(self._per_type("a_max"), device=self.dev)
            b = torch.as_tensor(self._per_type("b"), device=self.dev)
            tau = torch.as_tensor(self._per_type("tau"), device=self.dev)
            s0v = torch.as_tensor(self._per_type("s0"), device=self.dev)
            sf = torch.as_tensor(self._per_type("sf"), device=self.dev)
            v0 = sf * lim
            acc = idm_acc(self.v, gap, dv, a_max, b, tau, s0v, v0)
            # M1: standing obstacle at the accel-lane end (queues the
            # unmerged; a no-op once merged off the accel lane)
            on2 = self.lane == 2
            if bool(on2.any()):
                acc = torch.where(
                    on2,
                    torch.minimum(acc, lane_end_obstacle_acc(
                        self.v, self.s, on2, a_max, b, tau, s0v, v0)),
                    acc)
            # measured residual noise per (vtype, regime), as idm_sampled
            reg_sd = torch.as_tensor([
                (NOISE_SD[(int(t), 1 if fr else 0)])
                for t, fr in zip(self.vtype, free)], device=self.dev,
                dtype=torch.float64)
            reg_mu = torch.as_tensor([
                (NOISE_MU[(int(t), 1 if fr else 0)])
                for t, fr in zip(self.vtype, free)], device=self.dev,
                dtype=torch.float64)
            acc = acc.clamp(-9.0, 3.0)
            # SUMO dawdle: v' = max(0, v_idm - U(0,1) * sigma * accel * dt)
            dawdle_max = np.array([
                SIGMA_ACCEL[(int(t))] * DT for t in self.vtype])
            dawd = self.rng.random(len(acc)) * dawdle_max * getattr(self, "dawdle_scale", 1.0)
            v_dawdled = np.maximum(
                self.v.detach().cpu().numpy() + acc.detach().cpu().numpy()
                * DT - dawd, 0.0)
            acc = torch.as_tensor(
                (v_dawdled - self.v.detach().cpu().numpy()) / DT,
                device=self.dev, dtype=torch.float64)
            v_new = (self.v + acc * DT).clamp_min(0.0).clamp_max(1.2 * lim)
            s_new = self.s + v_new * DT
            self.capviol += int((v_new > self.vcap + 1e-6).sum())
            self.a_prev = acc
            self.v, self.s = v_new, s_new
        elif self.kind in ("idm_normal", "idm_oracle"):

            slots = self._slots_t()
            gap = slots[:, 0, 1].double()
            rel = slots[:, 0, 2].double()
            free = slots[:, 0, 0] < 0.5
            gap = torch.where(free, torch.full_like(gap, GAP_MAX), gap)
            dv = torch.where(free, torch.zeros_like(rel), -rel)
            if self.kind == "idm_normal":
                pr = VTYPES[1]
                a_max = torch.full_like(self.v, pr["a_max"])
                b = torch.full_like(self.v, pr["b"])
                tau = torch.full_like(self.v, pr["tau"])
                s0 = torch.full_like(self.v, pr["s0"])
                sf = torch.full_like(self.v, pr["sf"])
            else:                                # B3 ORACLE (uses vType)
                def per(k):
                    return torch.tensor([VTYPES[int(t)][k] for t in
                                         self.typ], device=self.dev,
                                        dtype=torch.float64)
                a_max, b, tau, s0, sf = (per("a_max"), per("b"), per("tau"),
                                         per("s0"), per("sf"))
            v0 = sf * self.lim
            a = idm_acc(self.v, gap, dv, a_max, b, tau, s0, v0).clamp(
                -9.0, 3.0)
            v_new = (self.v + a * DT).clamp_min(0.0).clamp_max(self.vcap)
            self.capviol += int((v_new > self.vcap + 1e-6).sum())
            s_new = self.s + v_new * DT
            self.a_prev = a
            self.v, self.s = v_new, s_new
        self._count_collisions(pre_lane, pre_s, pre_leader)
        self._drop_exits()

    def _insert(self, tvid, lane, s, v, cav):
        dev = self.dev
        lane_t = torch.tensor([lane], device=dev, dtype=torch.long)
        s_t = torch.tensor([float(s)], device=dev, dtype=torch.float64)
        lim = torch.as_tensor(lane_speed_limits(
            lane_t.cpu().numpy(), s_t.detach().cpu().numpy()), device=dev,
            dtype=torch.float64)
        v_t = torch.tensor([float(v)], device=dev,
                           dtype=torch.float64).clamp(0.0, float(1.2 * lim))
        ll = lane_lengths()
        self.vid = np.append(self.vid, tvid)
        self.lane = torch.cat([self.lane, lane_t])
        self.s = torch.cat([self.s, s_t])
        self.v = torch.cat([self.v, v_t])
        self.a_prev = torch.cat([self.a_prev, torch.zeros(
            1, device=dev, dtype=torch.float64)])
        self.cav = torch.cat([self.cav, torch.tensor([cav], device=dev,
                                                     dtype=torch.float32)])
        self.typ = np.append(self.typ, 1)
        u = self.rng.random()
        vt = int(np.searchsorted(np.cumsum([0.1, 0.7, 0.2]), u, 'left'))
        self.vtype = np.append(self.vtype, vt)
        self.vtype_map[int(tvid)] = vt   # part 3m instrumentation
        self.lim = torch.cat([self.lim, lim])
        self.vcap = torch.cat([self.vcap, 1.2 * lim])
        self.llen = torch.cat([self.llen, torch.tensor(
            [ll[lane]], device=dev, dtype=torch.float64)])
        if self.kind == "model":
            prior_in = PartAModel.prior_input(
                lane_t, v_t.float(), torch.full((1,), GAP_MAX, device=dev),
                torch.zeros(1, device=dev),
                torch.tensor([cav], device=dev))
            with torch.no_grad():
                self.z = torch.cat([self.z, self.model.z_prior(prior_in)],
                                   dim=0)
        self.n_birthed += 1

    def _per_type(self, key):
        return np.array([VTYPES[int(t)][key] for t in self.vtype],
                        dtype=np.float64)

    def _gap_accept_merge(self):
        """M2: mandatory gap-accepted merge accel(2)->main_right(1) inside
        the merge zone (s >= MERGE_ZONE_S0).  Front-most candidate first;
        a just-accepted vehicle occupies the target lane for later
        candidates in the same step.  Returns a bool tensor over the
        current vehicle slots (the caller applies the change).  The step
        tick arrives via self._m2_tick (set by step()); direct calls
        without it keep the pre-3q signature and skip the cooldown
        consult (cooldown state is only consulted with a tick)."""
        lane_np = self.lane.detach().cpu().numpy()
        s_np = self.s.detach().cpu().numpy()
        v_np = self.v.detach().cpu().numpy()
        out = np.zeros(len(lane_np), dtype=bool)
        cand = np.where((lane_np == 2) & (s_np >= MERGE_ZONE_S0))[0]
        if not len(cand):
            return torch.as_tensor(out, device=self.dev)
        cand = cand[np.argsort(-s_np[cand])]          # front first
        on1 = np.where(lane_np == 1)[0]
        occ_d = np.array([OFFSETS[1] - x for x in s_np[on1]],
                         dtype=np.float64) if len(on1) else np.zeros(0)
        occ_v = v_np[on1].astype(np.float64).copy()
        occ_s0 = (np.array([VTYPES[int(t)]["s0"] for t in
                            self.vtype[on1]], dtype=np.float64)
                  if len(on1) else np.zeros(0))
        tau_a = float(self.mp["tau_accept"])
        b_a = float(self.mp["b_accept"])
        force_s = float(self.mp.get("end_force_s", 0.0))
        # part 3l TASK A: rejection-cause diagnostics on the two M2 terms
        # (lead-gap acceptance vs follower required deceleration).  Pure
        # bookkeeping: the merge decision itself is unchanged.
        st = getattr(self, "merge_stats", None)
        if st is None:
            st = self.merge_stats = dict(
                checks=0, accepted=0, rej_lead=0, rej_foll=0, rej_both=0,
                with_foll=0, breq_hist=[0] * 6, breq_foll_rej=[])
        # part 3q levers (getattr defaults keep hand-built __new__ stubs
        # working): cooldown state is created only when the flag is > 0
        # and lives per-vid as "earliest tick allowed to re-attempt".
        cd_ticks = int(getattr(self, "merge_cooldown_ticks", 0))
        cd = getattr(self, "_merge_cd", None)
        if cd_ticks > 0 and cd is None:
            cd = self._merge_cd = {}
        early_d = float(getattr(self, "merge_early_d", 0.0))
        early_prob = getattr(self, "merge_early_prob", None)
        j = getattr(self, "_m2_tick", None)
        for i in cand:
            d_i = OFFSETS[2] - s_np[i]
            if cd is not None and j is not None:
                until = cd.get(int(self.vid[i]))
                if until is not None and j < until:
                    st["cd_blocked"] = st.get("cd_blocked", 0) + 1
                    continue
            # part 3m knob: near-lane-end acceptance relaxation (a no-op
            # while end_force_s == 0.0, the shipped default)
            tau_i, b_i = end_force_params(s_np[i], tau_a, b_a, force_s)
            ahead = occ_d < d_i                    # smaller d = ahead
            gap_lead = (GAP_MAX if not ahead.any()
                        else float(d_i - occ_d[ahead].max()))
            behind_idx = np.where(occ_d > d_i)[0]  # larger d = behind
            if len(behind_idx):
                # part 3l TASK A fix: the M2 follower is the NEAREST
                # main_right vehicle behind (SMALLEST d above d_i).
                # p3k shipped argmax here, which selects the FARTHEST
                # vehicle behind; its inflated gap made b_req ~ always
                # tiny, so the follower term never bound and the fit
                # was flat in b_accept.  Caught by the part 3l unit
                # tests (test_follower_slot_is_nearest_behind).
                j = behind_idx[np.argmin(occ_d[behind_idx])]
                gap_foll = float(occ_d[j] - d_i)
                v_f = float(occ_v[j])
                s0_f = float(occ_s0[j])
            else:
                gap_foll, v_f, s0_f = GAP_MAX, 0.0, 0.0
            s0_i = VTYPES[int(self.vtype[i])]["s0"]
            ok_a, ok_lead_a, ok_foll_a = merge_acceptance(
                np.array([gap_lead]), np.array([v_np[i]]),
                np.array([s0_i]), np.array([gap_foll]),
                np.array([v_f]), np.array([s0_f]), tau_i, b_i,
                return_parts=True)
            ok = bool(ok_a[0])
            st["checks"] += 1
            # part 3o: merge-hazard thinning -- a gap-accepted candidate
            # merges with probability merge_accept_prob (1.0 = the shipped
            # deterministic acceptance; the dedicated stream is drawn ONLY
            # on this path so flag==1.0 behaviour is byte-identical).
            # part 3q: the probability is POSITION-DEPENDENT -- a
            # candidate early in the zone (d_i > merge_early_d) draws
            # against merge_early_prob when set; with early_prob None the
            # p_i below equals mprob for every candidate and the p3o
            # behaviour (including the RNG consumption order) is
            # unchanged.  A thinning rejection under an active cooldown
            # silences this candidate for cd_ticks ticks.
            prob_rej = False
            mprob = getattr(self, "merge_accept_prob", 1.0)
            early_path = early_prob is not None and d_i > early_d
            p_i = early_prob if early_path else mprob
            if ok and p_i < 1.0:
                if getattr(self, "_merge_rng", None) is None:
                    self._merge_rng = np.random.default_rng(
                        (self.member_seed, int(self.t0), 9107))
                if self._merge_rng.random() >= p_i:
                    prob_rej = True
                    ok = False
                    st["rej_prob_early" if early_path else
                       "rej_prob"] = st.get("rej_prob_early"
                                            if early_path
                                            else "rej_prob", 0) + 1
                    if cd_ticks > 0 and j is not None:
                        cd[int(self.vid[i])] = j + cd_ticks
                        st["cd_set"] = st.get("cd_set", 0) + 1
            b_req_v = None
            if gap_foll < GAP_MAX:
                st["with_foll"] += 1
                b_req_v = ((v_f ** 2 - float(v_np[i]) ** 2)
                           / (2.0 * max(gap_foll - s0_f, 0.3)))
                st["breq_hist"][0 if b_req_v <= 0.0 else
                                1 if b_req_v <= 1.0 else
                                2 if b_req_v <= 1.5 else
                                3 if b_req_v <= 2.0 else
                                4 if b_req_v <= 3.0 else 5] += 1
            if ok:
                st["accepted"] += 1
                out[i] = True
                occ_d = np.append(occ_d, d_i)
                occ_v = np.append(occ_v, float(v_np[i]))
                occ_s0 = np.append(occ_s0, s0_i)
            elif prob_rej:
                pass                     # gap criteria passed; thinned only
            elif not ok_lead_a[0] and not ok_foll_a[0]:
                st["rej_both"] += 1
            elif not ok_lead_a[0]:
                st["rej_lead"] += 1
            else:
                st["rej_foll"] += 1
                if len(st["breq_foll_rej"]) < 4096:
                    st["breq_foll_rej"].append(float(b_req_v))
        return torch.as_tensor(out, device=self.dev)

    def _coop_leaders(self):
        """M3: cooperative IDM-leader adaptation for main_right followers
        whose leader-to-be is a waiting accel vehicle ahead (numpy bridge
        into coop_leader_map with fresh per-step uniforms)."""
        u = self.rng.random(len(self.vid))
        p = np.array([LC_COOP_PRIOR[int(t)] for t in self.vtype],
                     dtype=np.float64) * float(self.mp["p_coop_scale"])
        return coop_leader_map(self.lane.detach().cpu().numpy(),
                               self.s.detach().cpu().numpy(),
                               self.v.detach().cpu().numpy(),
                               self.vtype, u, p)

    def _count_collisions(self, pre_lane, pre_s, pre_leader, vid=None):
        """Adjacent-in-lane pairs (pre-step) whose order inverted or
        touched after the step are PREDICTED COLLISIONS: counted, never
        repaired.  s increases forward; inversion = follower new s >= its
        pre-step leader's new s.  Pairs that changed lane are skipped (the
        lane change is that step's event)."""
        post = {}
        for v, ss in zip(self.vid, self.s.detach().cpu().numpy()):
            post[int(v)] = float(ss)
        post_lane = {}
        for v, l in zip(self.vid, self.lane.detach().cpu().numpy()):
            post_lane[int(v)] = int(l)
        n = len(pre_lane)
        pl = pre_lane.detach().cpu().numpy()
        ps = pre_s.detach().cpu().numpy()
        for i in range(n):
            j = int(pre_leader[i])
            if j < 0:
                continue
            vi, vj = int((self.vid_of if vid is None else vid)[i]),                 int((self.vid_of if vid is None else vid)[j])
            if vi not in post or vj not in post:
                continue
            if post_lane[vi] != pl[i] or post_lane[vj] != pl[j]:
                continue
            if post[vi] >= post[vj] - 1e-9:
                self.collisions += 1

    def _drop_exits(self):
        gone = self.s > self.llen
        if gone.any():
            keep = ~gone
            self.vid = self.vid[keep]
            self.lane = self.lane[keep]
            self.s, self.v = self.s[keep], self.v[keep]
            self.a_prev, self.cav = self.a_prev[keep], self.cav[keep]
            self.typ = self.typ[keep]
            self.vtype = self.vtype[keep]
            self.lim, self.vcap, self.llen = (self.lim[keep],
                                              self.vcap[keep],
                                              self.llen[keep])
            if self.kind == "model":
                self.z = self.z[keep]
            self.n_exited += int(gone.sum())

    def run(self, n_ticks=None):
        """Roll the window; n_ticks < ROLL_TICKS stops early.  Metrics at
        horizons <= n_ticks * DT are IDENTICAL to the full roll (each step
        depends only on the past; part 3m uses this to halve the H<=60 s
        runtime); default None keeps the full 120 s rollout, so every
        existing caller and result is unchanged."""
        if n_ticks is None:
            n_ticks = ROLL_TICKS
        self.snapshots = {0: self.snapshot()}
        for j in range(1, n_ticks + 1):
            self.step(j)
            self.snapshots[j] = self.snapshot()
        return self


# --------------------------------------------------------- B4: v3 at 1 s ----

class V3Baseline:
    """The part-1 v3 surviving checkpoint (action-blind; the action one-hot
    is fixed to the native arm) at its native 1 s step with the RAW
    integrator on the model world.  Edgewise group from the frozen
    topology segments; leader within group by s (gap<=250, else free);
    no lane changes.  Births are inserted on the 1 s grid with logged
    (lane, s, v)."""

    IN_DIM = 12

    def __init__(self, root, t0, dev=None):
        self.dev = dev or device()
        ra = root_arrays(root)
        self.ra = ra
        self.root, self.t0 = root, float(t0)
        self.k0 = k0 = int(round((t0 - PD.WINDOW_T0) / 1.0))
        assert 0 <= k0 and k0 + 120 < PD.N_TICKS
        self.K = 120
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        import importlib.util
        v3_src = PD.SRC / "output" / "worldmodel_v3_vehicle" /             "train_vehicle_dynamics.py"
        spec = importlib.util.spec_from_file_location(
            "wd01_p3_v3_train_mod", str(v3_src))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.net = mod.DynamicsMLP(in_dim=ck["arch"]["in_dim"],
                                   hidden=tuple(ck["arch"]["hidden"]))
        self.net.load_state_dict(ck["model_state"])
        self.net = self.net.to(self.dev).eval()
        self.x_mean = np.asarray(ck["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(ck["x_std"], dtype=np.float32)
        idx = ra.tick_rows(2 * k0)
        idx = idx[ra.lane_m[idx] >= 0]
        self.vid = ra.vid[idx].copy()
        self.lane = ra.lane_m[idx].copy()
        self.s = ra.s[idx].copy()
        self.v = ra.v[idx].copy()
        ll = lane_lengths()
        self.llen = np.array([ll[0], ll[1], ll[2]])[self.lane]
        self.segb = _seg_bounds()
        self.births = defaultdict(list)
        bm = ra.b_dep & (ra.b_tick >= 2 * k0) & (ra.b_tick < 2 * (k0 + 120))
        bm &= ra.b_lane >= 0
        for i in np.where(bm)[0]:
            t1 = int(-(-int(ra.b_tick[i]) // 2) - k0)
            self.births[t1].append((int(ra.b_vid[i]), int(ra.b_lane[i]),
                                    float(ra.b_s[i]), float(ra.b_v[i])))
        self.collisions = self.capviol = self.lc_count = 0
        self.n_birthed = 0
        self.snapshots = {}

    def snapshot(self):
        return dict(vid=self.vid.copy(), lane=self.lane.copy(),
                    s=self.s.copy(), v=self.v.copy(),
                    a=np.zeros_like(self.v), collisions=self.collisions,
                    capviol=self.capviol, lc=self.lc_count,
                    n_birthed=self.n_birthed)

    def _snap2(self):
        d = self.snapshot()
        d["dt_scale"] = 2
        return d

    @torch.no_grad()
    def step(self, j):
        for (tvid, l, ss, vv) in self.births.get(j, ()):
            self.vid = np.append(self.vid, tvid)
            self.lane = np.append(self.lane, l)
            self.s = np.append(self.s, ss)
            self.v = np.append(self.v, min(vv, 1.2 * LARR[l]))
            self.llen = np.append(self.llen, lane_lengths()[l])
            self.n_birthed += 1
        if len(self.vid) == 0:
            return
        n = len(self.vid)
        grp = np.zeros(n, dtype=np.int64)
        for i in range(n):
            li = int(self.lane[i])
            for lo, hi, gi in self.segb[li]:
                if lo <= self.s[i] < hi:
                    grp[i] = gi
                    break
        v_lead = self.v.copy()
        gap = np.full(n, GAP_MAX)
        free = np.ones(n, dtype=np.float64)
        for g in np.unique(grp):
            ids = np.where(grp == g)[0]
            if len(ids) < 2:
                continue
            o = ids[np.argsort(-self.s[ids], kind="stable")]
            for jj in range(len(o) - 1):
                a, b2 = int(o[jj]), int(o[jj + 1])
                gp = self.s[a] - self.s[b2]
                if 0 < gp <= GAP_MAX:
                    v_lead[a] = self.v[b2]
                    gap[a] = gp
                    free[a] = 0.0
        act = 1                                   # native arm index
        x = np.zeros((n, self.IN_DIM), dtype=np.float32)
        x[:, 0] = self.v
        x[:, 1] = v_lead
        x[:, 2] = gap
        x[np.arange(n), 3 + grp] = 1.0
        x[:, 7] = free
        x[np.arange(n), 8 + act] = 1.0
        xn = (x - self.x_mean) / self.x_std
        dv = self.net(torch.as_tensor(xn, device=self.dev)).cpu().numpy()
        v_new = np.maximum(self.v + dv, 0.0)      # RAW 1 s integrator
        s_new = self.s + v_new * 1.0
        self.capviol += int((v_new > 1.2 * LARR[self.lane]).sum())
        self.v, self.s = v_new, s_new
        for li in (0, 1, 2):
            ids = np.where(self.lane == li)[0]
            if len(ids) < 2:
                continue
            o = ids[np.argsort(-self.s[ids], kind="stable")]
            self.collisions += int((np.diff(self.s[o]) <= 1e-9).sum())
        alive = self.s <= self.llen
        if (~alive).any():
            self.vid, self.lane = self.vid[alive], self.lane[alive]
            self.s, self.v = self.s[alive], self.v[alive]
            self.llen = self.llen[alive]

    def run(self):
        self.snapshots = {0: self._snap2()}
        for j in range(1, self.K + 1):
            self.step(j)
            self.snapshots[j] = self._snap2()
        return self


def sys_path():
    import sys
    return sys.path


_SEGB = None


def _seg_bounds():
    global _SEGB
    if _SEGB is None:
        topo, _, _ = PD.topology()
        _SEGB = {}
        for name, li in PD.LANE_IDS.items():
            _SEGB[li] = [(seg.s0, seg.s1, seg.group)
                         for seg in topo.segments_of(name)]
    return _SEGB


def ens_snapshot(member_snaps, j):
    """Member-averaged snapshot at tick j: s/v averaged over members
    where present; lane = member median; scalar counters averaged."""
    ids = member_snaps[0][j]["vid"]
    M = len(member_snaps)
    S = np.zeros((len(ids), M))
    V = np.zeros((len(ids), M))
    L = np.zeros((len(ids), M), dtype=np.int64)
    ok = np.zeros((len(ids), M), dtype=bool)
    for mi, per_tick in enumerate(member_snaps):
        sn = per_tick[j]
        pos = {int(v): i for i, v in enumerate(sn["vid"])}
        for r, v_ in enumerate(ids):
            i = pos.get(int(v_))
            if i is None:
                continue
            ok[r, mi] = True
            S[r, mi] = sn["s"][i]
            V[r, mi] = sn["v"][i]
            L[r, mi] = sn["lane"][i]
    keep = ok.any(1)
    return dict(vid=ids[keep].copy(),
                lane=np.array([np.median(L[r][ok[r]]) for r in
                               range(len(ids)) if keep[r]]),
                s=np.array([S[r][ok[r]].mean() for r in range(len(ids))
                            if keep[r]]),
                v=np.array([V[r][ok[r]].mean() for r in range(len(ids))
                            if keep[r]]),
                a=np.zeros(int(keep.sum())),
                collisions=float(np.mean([sn[j]["collisions"]
                                          for sn in member_snaps])),
                capviol=float(np.mean([sn[j]["capviol"]
                                       for sn in member_snaps])),
                lc=float(np.mean([sn[j]["lc"] for sn in member_snaps])),
                n_birthed=member_snaps[0][j]["n_birthed"])


def ttc3_flag_map(ra):
    """(vid, tick) -> 1 if the vehicle closes on its same-lane leader with
    TTC < 3 s at that tick (follower->leader direction)."""
    out = {}
    by_tick = defaultdict(list)
    for i in range(ra.n):
        by_tick[int(ra.tick[i])].append(i)
    for k, ids in by_tick.items():
        ids = np.asarray(ids, dtype=np.int64)
        lane = ra.lane_m[ids]
        s = ra.s[ids]
        v = ra.v[ids]
        for li in (0, 1, 2):
            sel = np.where(lane == li)[0]
            if len(sel) < 2:
                continue
            o = sel[np.argsort(-s[sel], kind="stable")]
            for jj in range(len(o) - 1):
                l_, f = o[jj], o[jj + 1]
                gap = s[l_] - s[f]
                closing = v[f] - v[l_]
                if gap > 0 and closing > 0 and gap / closing < 3.0:
                    out[(int(ra.vid[f]), int(ra.tick[f]))] = 1
    return out


def phase1_samples(roots, max_per_root=20000, seed=0):
    """Teacher-forced one-step samples (numpy) with full 20-tick truth
    histories.  Targets: v', s' through the measured integrator form;
    lane target 0=stay/1=left/2=right; aux y3 = same-vehicle TTC<3 s
    within 2 s after the target tick, yb = a < -3 m/s^2 within 2 s."""
    X_h, X_e, X_s, X_p = [], [], [], []
    Y_v, Y_s, Y_d, Y_l, Y_a = [], [], [], [], []
    for root in roots:
        ra = root_arrays(root)
        flag = ttc3_flag_map(ra)
        rng = np.random.default_rng(seed)
        cand = []
        for vi, idxs in ra._rows.items():
            for p in range(N_HIST - 1, len(idxs) - 1):
                if ra.lane_m[idxs[p]] < 0 or ra.lane_m[idxs[p + 1]] < 0:
                    continue
                ok = True
                for q in range(p - N_HIST + 2, p + 1):
                    if ra.tick[idxs[q]] - ra.tick[idxs[q - 1]] != 1:
                        ok = False
                        break
                if ok:
                    cand.append((vi, p))
        if len(cand) > max_per_root:
            keep = rng.choice(len(cand), max_per_root, replace=False)
            cand = [cand[i] for i in sorted(keep)]
        slot_cache = {}
        for vi, p in cand:
            idxs = ra._rows[vi]
            i_b, i_a = idxs[p], idxs[p - 1]
            base = p - (N_HIST - 1)
            h = np.zeros((N_HIST, 3))
            for r in range(N_HIST - 1):
                i, nxt = idxs[base + r], idxs[base + r + 1]
                h[r] = (ra.v[i], ra.a[i], max(ra.s[nxt] - ra.s[i], 0.0))
            h[-1] = (ra.v[i_a], ra.a[i_a], 0.0)
            lane_a = int(ra.lane_m[i_a])
            tick_a = int(ra.tick[i_a])
            if tick_a not in slot_cache:
                ids_w = ra.tick_rows(tick_a)
                if len(ids_w) == 0:
                    continue
                full = neighbour_slots(ra.lane_m[ids_w], ra.s[ids_w],
                                       ra.v[ids_w], ra.cav[ids_w])
                slot_cache[tick_a] = (ids_w, full)
            ids_w, full = slot_cache[tick_a]
            where = np.where(ids_w == i_a)[0]
            if len(where) == 0:
                continue
            sl = full[where[0]]
            X_h.append(h)
            X_e.append((lane_a, ra.s[i_a], ra.v[i_a], ra.a[i_a],
                        ra.cav[i_a]))
            X_s.append(sl)
            X_p.append((lane_a, ra.v[i_a], GAP_MAX, 0.0, ra.cav[i_a]))
            Y_v.append(ra.v[i_b])
            Y_s.append(ra.s[i_b])
            # d-coordinate target (comparable across lane changes)
            Y_d.append(OFFSETS[int(ra.lane_m[i_b])] - ra.s[i_b])
            Y_l.append(int(ra.lane_m[i_b]) - lane_a + 1)
            tb = int(ra.tick[i_b])
            y3 = int(any(flag.get((vi, tb + q), 0) for q in range(4)))
            a_win = ra.a[idxs[p + 1:min(p + 5, len(idxs))]]
            yb = float(a_win.min() < -3.0) if len(a_win) else 0.0
            Y_a.append((y3, yb))
    return dict(hist=np.asarray(X_h, dtype=np.float32),
                ego=np.asarray(X_e, dtype=np.float32),
                slots=np.asarray(X_s, dtype=np.float32),
                prin=np.asarray(X_p, dtype=np.float32),
                y_v=np.asarray(Y_v, dtype=np.float32),
                y_s=np.asarray(Y_s, dtype=np.float32),
                y_lane=np.asarray(Y_l, dtype=np.int64),
                y_d=np.asarray(Y_d, dtype=np.float32),
                y_aux=np.asarray(Y_a, dtype=np.float32))



def same_lane_leader_idx(lane, s):
    """Pre-step same-lane leader index per vehicle (-1 if free).  s
    increases FORWARD: the leader is the next larger s in the lane."""
    out = np.full(len(lane), -1, dtype=np.int64)
    sv = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s
    lv = lane.detach().cpu().numpy() if isinstance(lane, torch.Tensor) else lane
    for li in (0, 1, 2):
        ids = np.where(lv == li)[0]
        if len(ids) < 2:
            continue
        o = ids[np.argsort(sv[ids], kind="stable")]   # rear (small s) first
        for j in range(len(o) - 1):
            out[o[j]] = o[j + 1]                      # leader = next larger s
    return out


def ttc_stats(lane, s, v):
    """Same-physical-lane follower->leader TTC at one 0.5 s tick:
    (n_lt15, n_lt3, min_ttc or inf).  s increases FORWARD: the leader is
    the next larger s; closing = v_follower - v_leader > 0."""
    lt15 = lt3 = 0
    m = float("inf")
    for li in (0, 1, 2):
        ids = np.where(lane == li)[0]
        if len(ids) < 2:
            continue
        o = ids[np.argsort(-s[ids], kind="stable")]  # front (large s) first
        for j in range(len(o) - 1):
            l_, f = o[j], o[j + 1]     # l_ ahead, f behind
            gap = s[l_] - s[f]
            closing = v[f] - v[l_]
            if gap > 0 and closing > 0:
                t = gap / closing
                lt3 += t < 3.0
                lt15 += t < 1.5
                m = min(m, t)
    return lt15, lt3, m


def corr(a, b, kind):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    if kind == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    ra_ = np.argsort(np.argsort(a)).astype(np.float64)
    rb_ = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra_, rb_)[0, 1])


def agg_window_rows(rows):
    """Aggregate per-window metric dicts (mean of scalars, pooled RMSE,
    window-level flag recall/precision + base rates, exposure/bias/corr)."""
    n = len(rows)
    out = dict(n_windows=n)
    for k in ("pred_vehicle_steps", "matched_vehicle_steps",
              "pred_collisions", "pred_lc", "true_lc",
              "pred_ttc3_exposure", "true_ttc3_exposure",
              "pred_ttc15_exposure", "true_ttc15_exposure",
              "pred_hard_brake_s", "true_hard_brake_s",
              "delay_pred", "delay_true"):
        out[k] = float(np.mean([r[k] for r in rows]))
    out["collisions_per_1000"] = float(np.mean(
        [r["collisions_per_1000"] for r in rows]))
    out["speed_rmse"] = float(np.sqrt(np.mean(
        [r["speed_rmse"] ** 2 for r in rows])))
    out["pos_rmse"] = float(np.sqrt(np.mean(
        [r["pos_rmse"] ** 2 for r in rows])))
    out["lane_frac"] = float(np.mean([r["lane_frac"] for r in rows]))
    out["capviol_mean"] = float(np.mean([r["capviol"] for r in rows]))
    for f in ("ttc15", "ttc3", "hb"):
        t = [r[f"any_{f}_true"] for r in rows]
        pr = [r[f"any_{f}_pred"] for r in rows]
        tp = sum(1 for x, y in zip(t, pr) if x and y)
        out[f"any_{f}_recall"] = tp / sum(t) if sum(t) else None
        out[f"any_{f}_precision"] = tp / sum(pr) if sum(pr) else None
        out[f"any_{f}_base_rate"] = sum(t) / n
    for k in ("ttc3_exposure", "ttc15_exposure", "hard_brake_s"):
        t = [r[f"true_{k}"] for r in rows]
        pr = [r[f"pred_{k}"] for r in rows]
        out[f"{k}_bias"] = float(np.mean(np.asarray(pr) - np.asarray(t)))
        out[f"{k}_pearson"] = corr(t, pr, "pearson")
        out[f"{k}_spearman"] = corr(t, pr, "spearman")
    pair = [(r["true_min_ttc"], r["pred_min_ttc"]) for r in rows
            if r["true_min_ttc"] is not None
            and r["pred_min_ttc"] is not None]
    if pair:
        tm = [x for x, _ in pair]
        pm = [y for _, y in pair]
        out["min_ttc_bias"] = float(np.mean(np.asarray(pm) - np.asarray(tm)))
        out["min_ttc_pearson"] = corr(tm, pm, "pearson")
        out["min_ttc_spearman"] = corr(tm, pm, "spearman")
    else:
        out["min_ttc_bias"] = None
        out["min_ttc_pearson"] = None
        out["min_ttc_spearman"] = None
    out["delay_bias"] = out["delay_pred"] - out["delay_true"]
    out["delay_pearson"] = corr([r["delay_true"] for r in rows],
                                [r["delay_pred"] for r in rows], "pearson")
    out["delay_spearman"] = corr([r["delay_true"] for r in rows],
                                 [r["delay_pred"] for r in rows], "spearman")
    return out


def window_metrics(snaps, ra, k0, H, dt=0.5, stride=1):
    """Metrics for one (window, model) at horizon H (s), from per-tick
    snapshots vs truth.  Junction truth rows are excluded from TTC (their
    s is boundary-paused) and lane-fraction uses the mapped lane_m."""
    K = int(round(H / dt))
    pred_steps = 0
    v2 = s2 = n_m = 0
    lane_ok = lane_n = 0
    t3p = t3t = t15p = t15t = 0
    bp = bt = 0.0
    minp, mint = [], []
    d_pred = d_true = 0.0
    for j in range(1, K + 1):
        sn = snaps[j * stride]
        # truth at this tick
        ti = ra.tick_rows(k0 + j * stride)
        tv = {int(v): (int(l), float(ss), float(vv), bool(ij), float(aa))
              for v, l, ss, vv, ij, aa in zip(ra.vid[ti], ra.lane_m[ti],
                                              ra.s[ti], ra.v[ti],
                                              ra.inj[ti], ra.a[ti])}
        pv = {int(v): (int(l), float(ss), float(vv))
              for v, l, ss, vv in zip(sn["vid"], sn["lane"], sn["s"],
                                      sn["v"])}
        pred_steps += len(pv)
        common = set(pv) & set(tv)
        for vid in common:
            l, ss, vv, ij, aa = tv[vid]
            pl, ps_, pv_ = pv[vid]
            v2 += (pv_ - vv) ** 2
            # position RMSE in the d coordinate (comparable across lanes)
            s2 += ((OFFSETS[pl] - ps_) - (OFFSETS[l] - ss)) ** 2
            n_m += 1
            lane_ok += (pl == l)
            lane_n += 1
        d_pred += len(pv) * dt
        d_true += len(tv) * dt
        l15p, l3p, mp = ttc_stats(sn["lane"], sn["s"], sn["v"])
        tmask = np.array([not tv[int(v)][3] for v in sn["vid"]
                          if int(v) in tv]) if len(sn["vid"]) else \
            np.zeros(0, bool)
        if len(sn["vid"]):
            tl = np.array([tv[int(v)][0] if int(v) in tv else -1
                           for v in sn["vid"]])
            ts = np.array([tv[int(v)][1] if int(v) in tv else np.nan
                           for v in sn["vid"]])
            tvv = np.array([tv[int(v)][2] if int(v) in tv else np.nan
                            for v in sn["vid"]])
            keep = tl >= 0
            if keep.any():
                l15t, l3t, mt = ttc_stats(tl[keep], ts[keep], tvv[keep])
                t15t += l15t
                t3t += l3t
                mint.append(mt)
        t15p += l15p
        t3p += l3p
        minp.append(mp)
        if len(sn["vid"]):
            bp += float((sn["a"] < -3.0).sum())
        if len(common):
            bt += float(sum(1 for vid in common if tv[vid][4] < -3.0))
    out = dict(
        H=H, pred_vehicle_steps=pred_steps,
        matched_vehicle_steps=n_m,
        speed_rmse=float((v2 / max(n_m, 1))) ** 0.5,
        pos_rmse=float((s2 / max(n_m, 1))) ** 0.5,
        lane_frac=lane_ok / max(lane_n, 1),
        pred_collisions=(snaps[K]["collisions"] - snaps[0]["collisions"]),
        collisions_per_1000=(snaps[K]["collisions"]
                             - snaps[0]["collisions"])
        / max(pred_steps, 1) * 1000.0,
        capviol=(snaps[K]["capviol"] - snaps[0]["capviol"]),
        pred_lc=snaps[K]["lc"] - snaps[0]["lc"],
        true_lc=int(((ra.lc_tick > k0)
                      & (ra.lc_tick <= k0 + K * stride)).sum()),
        pred_ttc3_exposure=t3p * dt, true_ttc3_exposure=t3t * dt,
        pred_ttc15_exposure=t15p * dt, true_ttc15_exposure=t15t * dt,
        pred_hard_brake_s=bp * dt, true_hard_brake_s=bt * dt,
        pred_min_ttc=(None if not minp or not np.isfinite(minp).any()
                      else float(np.nanmin(minp))),
        true_min_ttc=(None if not mint or not np.isfinite(mint).any()
                      else float(np.nanmin(mint))),
        any_ttc3_pred=t3p > 0, any_ttc3_true=t3t > 0,
        any_ttc15_pred=t15p > 0, any_ttc15_true=t15t > 0,
        any_hb_pred=bp > 0, any_hb_true=bt > 0,
        delay_pred=d_pred, delay_true=d_true)
    return out
