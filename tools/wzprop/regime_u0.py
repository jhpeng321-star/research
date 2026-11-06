"""WMEV (claim 4a): WM-predicted expected benefit of the S-REGIMEU0 / S-LOWSHARE-U0 rule strategies ON THE MAPPO u0
BASE, per demand x CAV-share cell, for comparison with the SUMO measurements (zero SUMO; WM rollouts only).

Extends WM-C's ``regime_wm`` (one activation per state, WM-native base, H 300) to the SUMO protocol of S-REGIMEU0:

* start state = the world at t0 = 300 s (end of the native warm-up, identical in every SUMO arm), built from WM-D's
  native prefix tapes of TRAIN-type worlds under the RSU-observable rule of ``tapes.observable`` (CAV self-states +
  CAV-detected neighbours + RSU entrance counters).  No world label reaches the WM.
* base = MAPPO u0 (speed-only MAPPO169 d9042cc5, deterministic, the C4 ``PolicyExecutor`` networks) driving EVERY CAV
  slot of the WM rollout (observed CAVs and CAV entrants) longitudinally, with the plant safety layer
  (``wms.control.safety.safe_accel``) and the WM's own (native, HDV-law) lane decisions, as in SUMO u0 (native lanes);
* strategy = the S-REGIMEU0 rule (vendored verbatim below from c4/imitreplay c08ff784 tools/coopmwm/regimeu0.py):
  every whole second release active members (hold end / s >= 75.44 / gone), then if t >= next_ok run ``pick_wall`` on
  the non-member CAVs (one CAV per lane in s in [-1724.56, -324.56], spread <= 200 m, key (spread, -max s)); if a wall
  exists its members (all, or only the SUMO-lane-L one for ``*_soloL``) start the WM-A planapi 'hold' law
  a = clip(v_w - v, -dmax, amax) at once (lane held; lane-end merge delegated to the WM first-passage law) and
  next_ok = t + hold_s (cooldown = hold).  The member command passes the same safety layer as the u0 command
  (the plant's single writer: pre-safety request, MAPPO safety downstream).  HDVs are never commanded.
* entries: Poisson arrivals at the rate of the observed RSU counter (``TorchEntryProcess`` statistics), one arrival
  stream per CRN draw k shared by every arm of that draw (common random numbers), CAV entrants with the RSU-observable
  CAV entry share (CAV self-reports among counted entries over the counter window), full FIFO backlog (source waiting).
* draws: lane-sampling uniforms and arrivals of draw k = f(seed, k, state digest), shared by all arms (CRN); the
  u0 and strategy rows of one draw are identical until the first activation (tested).

Output per (state, arm): K draws x per-100-s-window road delay (veh-s, sum dt * max(0, 1 - v/31.29) over predicted
road vehicles), source waiting (dt * backlog), mean vehicle count, activations, first activation, first divergence
from the u0 row of the same draw.  ``regime_u0_compare`` turns these into gains vs u0 and compares them with SUMO.

CLI (``python -m tools.wzprop.regime_u0``): states | score | cost.  See WMEV_READY.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import geometry as G
from . import tapes as TP

SCHEMA = 'wmev.regime_u0.v1'
RULE_SOURCE = 'c4/imitreplay c08ff784 tools/coopmwm/regimeu0.py (S-REGIMEU0 / S-LOWSHARE-U0 rule, verbatim parameters)'
# ------------------------------------------------------------------------------ vendored rule (regimeu0.py)
X_B = 2696.0
S_OFF = 2620.56                       # s = x - S_OFF (N1 descriptor: drop x 2696 <-> s 75.44)
ZONE_S = (X_B - 1800.0 - S_OFF, X_B - 400.0 - S_OFF)
WALL_SPREAD = 200.0
RELEASE_S = X_B - S_OFF
WALL_H = dict(v_w=15.0, hold_s=90, dmax=1.5, amax=1.0)
WALL_G = dict(v_w=22.0, hold_s=120, dmax=1.0, amax=1.0)          # pg2_plan VARIANTS['wall_g'] (S-LOWSHARE-U0)
VARIANT_PARAMS = dict(wall_h=WALL_H, wall_g=WALL_G)
STRATEGIES = {f'{w}{suffix}': (w, solo) for w in ('wall_h', 'wall_g')
              for suffix, solo in (('', None), ('_solo0', 0), ('_solo1', 1), ('_solo2', 2))}
# ------------------------------------------------------------------------------ WMEV protocol
BASE = 'u0'
NATIVE = 'native'                      # diagnostic arm: every CAV is a WM native driver (no policy, no rule)
DEFAULT_ARMS = ('auto',)               # per-cell SUMO arms (CELL_ARMS); add 'native' for the horizon check
T0_S = 300.0                           # SUMO: native warm-up 0-300 s in every arm, rules act on 300-1200 s
H_MAX_S = 900.0
WINDOW_S = 100.0
DT_S = 0.5
SPEED_LIMIT_MPS = 31.29
STEPS_PER_CONTROL = 2
CAV_SHARE_WINDOW_S = 120.0             # = EntryConfig.window_bins (12) x 10 s: the counter-rate window
DEFAULT_PAD = 256
DEFAULT_SLOTS = 'auto'                  # fixed slot count per demand level (CELL_SLOTS), see evaluate_states
# WM cost grows ~n^2 in the slot count (pairwise attention / lane neighbours over ALL slots).  SUMO native tapes
# (WM-D, 300-1200 s) peak at 146 / 256 / 356 vehicles at D2400 / D2800 / D3200; a shortage is counted (no_slot)
# and fails the horizon check.
CELL_SLOTS = {2400: 256, 2800: 352, 3200: 448}
DEFAULT_SEED = 20261107
CELLS = ('D2400_p40', 'D2400_p60', 'D2800_p40', 'D2800_p60', 'D3200_p20', 'D3200_p30', 'D3200_p40', 'D3200_p60')
# 'auto' arms = exactly the SUMO-measured arms of each cell (S-REGIMEU0 / S-LOWSHARE-U0), u0 first
_CORE = ('u0', 'wall_h_solo2', 'wall_h')
CELL_ARMS = {c: _CORE for c in CELLS}
CELL_ARMS.update({c: _CORE + ('wall_h_solo0',) for c in ('D3200_p60', 'D3200_p40')})
CELL_ARMS.update({c: _CORE + ('wall_g',) for c in ('D3200_p30', 'D3200_p20')})


def arms_for(cell, arms):
    """Arms of one cell: 'auto' (the SUMO arms of the cell) plus any extra arms, or an explicit list."""
    arms = list(arms)
    if 'auto' in arms:
        extra = [a for a in arms if a != 'auto']
        base = list(CELL_ARMS.get(cell, _CORE))
        return base + [a for a in extra if a not in base]
    return arms


def sumo_lane(wm_lane):
    return 2 - int(wm_lane)


def pick_wall(cavs):
    """VERBATIM regimeu0.pick_wall (reference for the vectorised version; rows have s_m and WM lane)."""
    z = [c for c in cavs if ZONE_S[0] <= c['s_m'] <= ZONE_S[1] and int(c['lane']) in (0, 1, 2)]
    lanes = [[c for c in z if sumo_lane(c['lane']) == ln] for ln in (0, 1, 2)]
    if any(not x for x in lanes):
        return None
    best = None
    import itertools
    for combo in itertools.product(*lanes):
        xs = [c['s_m'] for c in combo]
        sp = max(xs) - min(xs)
        if sp <= WALL_SPREAD:
            key = (sp, -max(xs))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else best[1]


def pick_wall_slots(s, lane, slots):
    """Vectorised pick_wall over candidate slots (same product order and key): [slot of SUMO lane 0, 1, 2] or None."""
    s, lane, slots = np.asarray(s, float), np.asarray(lane), np.asarray(slots)
    inz = (s >= ZONE_S[0]) & (s <= ZONE_S[1]) & np.isin(lane, (0, 1, 2))
    per = [np.flatnonzero(inz & (lane == 2 - ln)) for ln in (0, 1, 2)]        # SUMO lane ln = WM lane 2 - ln
    if any(len(p) == 0 for p in per):
        return None
    a, b, c = np.meshgrid(s[per[0]], s[per[1]], s[per[2]], indexing='ij')     # C order = itertools.product order
    mx = np.maximum(np.maximum(a, b), c).ravel()
    sp = mx - np.minimum(np.minimum(a, b), c).ravel()
    ok = np.flatnonzero(sp <= WALL_SPREAD)
    if not len(ok):
        return None
    sp_ok = sp[ok]
    best = ok[sp_ok == sp_ok.min()]
    best = best[mx[best] == mx[best].max()]
    i, j, k = np.unravel_index(int(best[0]), (len(per[0]), len(per[1]), len(per[2])))
    return [int(slots[per[0][i]]), int(slots[per[1][j]]), int(slots[per[2][k]])]


def arm_rule(arm):
    """(params, solo SUMO lane or None) of an arm, None for the base / native arms."""
    if arm in (BASE, NATIVE):
        return None
    if arm not in STRATEGIES:
        raise ValueError(f'unknown arm {arm!r}')
    var, solo = STRATEGIES[arm]
    return dict(VARIANT_PARAMS[var]), solo


# ================================================================================ start states (zero WM)
def cav_entry_share(tape, t0, entries=None, window_s=CAV_SHARE_WINDOW_S):
    """RSU-observable CAV share of entries over [t0 - window, t0): CAV self-reports among the counted insertions."""
    t, _ent = entries if entries is not None else TP.first_appearance(tape)
    is_cav = np.asarray(tape['is_cav'], bool)
    sel = np.isfinite(t) & (t < t0) & (t >= t0 - window_s)
    n = int(sel.sum())
    return (float(is_cav[sel].sum()) / n if n else None), n


def build_states(tapes_root, out_path, *, cells=CELLS, t0=T0_S, ref_out=None, max_worlds=None):
    """One observable start state per world tape at t0 (default 300 s) + optional native reference windows (validation only).

    The state rows follow WM-C's MOMENTS format (``wm_backend.moment_input``); ``ref`` (the native tape's per-100-s
    road delay after t0) goes to a SEPARATE file and is never passed to the WM."""
    root = Path(tapes_root) / 'runs'
    rows, ref, report = [], {}, dict(states=0, by_cell={}, t0=t0, source=str(tapes_root))
    for cell in cells:
        d = root / cell
        if not d.exists():
            continue
        tps = [p / 'prefix' / 'TAPE.npz' for p in sorted(d.glob('s*'))]
        tps = [p for p in tps if p.exists()][:max_worlds]
        for tp in tps:
            tape = TP.load_tape(tp)                                    # schema + TRAIN seed guard
            meta = tape['meta']
            if meta['cell'] != cell:
                raise ValueError(f'{tp}: tape cell {meta["cell"]} != directory {cell}')
            seed = int(meta['seed'])
            entries = TP.first_appearance(tape)
            frame = TP.frame_at(tape, float(t0))
            cavs, det, hidden = TP.observable(frame)
            share, n_ent = cav_entry_share(tape, float(t0), entries)
            cav_rows = [dict(id=c['id'], edge=c['edge'], lane=c['lane'], x=round(c['x'], 3), v=round(c['v'], 4))
                        for c in cavs]
            dem, pen = cell.split('_')
            sid = f'rgu0_{cell}_s{seed}_t{int(t0)}'
            rows.append(dict(cell=cell, demand=int(dem[1:]), penetration=int(pen[1:]) / 100.0, seed=seed,
                             window=None, t0=int(t0), valid=True, sid=sid, all_cavs_t0=cav_rows, variants={},
                             feasible={},
                             observed=dict(vehicles=det, counter=TP.counter_at(tape, float(t0), entries),
                                           cav_entry_share=share, cav_entry_n=n_ent,
                                           coverage=dict(cavs=len(cavs), detected=len(det), undetected=hidden),
                                           tape=dict(path=str(tp), seed=seed, cell=cell))))
            if ref_out is not None:
                ref[sid] = native_ref_windows(tape, float(t0))
            report['states'] += 1
            report['by_cell'][cell] = report['by_cell'].get(cell, 0) + 1
    out_path = G.check_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x') as fh:
        fh.write(json.dumps(rows))
    if ref_out is not None:
        with G.check_path(ref_out).open('x') as fh:
            fh.write(json.dumps(dict(schema=SCHEMA + '.native_ref', t0=t0, window_s=WINDOW_S, ref=ref)))
    report['sha256'] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return report


def native_ref_windows(tape, t0, h=H_MAX_S, window_s=WINDOW_S):
    """SUMO native per-window road delay (veh-s) after t0 from the full tape (VALIDATION TARGET, never a WM input)."""
    ts = tape['time_s']
    out = []
    for w in range(int(round(h / window_s))):
        a, b = t0 + w * window_s, t0 + (w + 1) * window_s
        steps = np.flatnonzero((ts >= a - 1e-9) & (ts < b - 1e-9))
        if not len(steps):
            out.append(None)
            continue
        rows = np.isin(tape['step'], steps)
        v = np.asarray(tape['v'], float)[rows]
        out.append(float(DT_S * np.clip(1.0 - v / SPEED_LIMIT_MPS, 0.0, None).sum()))
    return out


def load_states(path, sids=None, shard=None, max_per_cell=None):
    """Valid states (TRAIN seed guard); ``max_per_cell`` keeps the first N per cell in sid order (outcome-blind)."""
    rows = [m for m in json.loads(G.check_path(path).read_text()) if m.get('valid')]
    for m in rows:
        G.check_seed(m['seed'], where=f"(state {m['sid']})")
    if max_per_cell:
        keep, seen = [], {}
        for m in sorted(rows, key=lambda r: r['sid']):
            seen[m['cell']] = seen.get(m['cell'], 0) + 1
            if seen[m['cell']] <= int(max_per_cell):
                keep.append(m)
        rows = keep
    if sids:
        want = set(sids)
        rows = [m for m in rows if m['sid'] in want]
    if shard:
        i, n = shard
        rows = rows[i::n]
    return rows


# ================================================================================ WM side
def _torch():
    import torch
    return torch


class GroupedEntryProcess:
    """Arrival process of the WM rollout: ``TorchEntryProcess`` placement statistics with one Poisson stream per CRN
    group g = (start state, draw k) shared by every row (arm) of the group; per-group observed-counter rates and
    RSU-observable CAV entry share; per-row lane split from the row's own t0 state; a full FIFO backlog per row and
    entrance (CAV identity kept while queued); CAV entrants may use any free slot (never-used slots first).
    ``specs[g] = dict(counter=[2, 30], cav_share=float, seed=int)``.  Built lazily (no WM import at module load)."""

    def __new__(cls, groups, specs, *, dt_s=DT_S):
        from wms.model.entries import EntryConfig, N_ENTRANCES, TorchEntryProcess, observed_rates

        class _Grouped(TorchEntryProcess):
            def __init__(self):
                g0 = specs[int(groups[0])]
                super().__init__(np.asarray(g0['counter'], np.float32), dt_s=dt_s, seed=0, batch=len(groups),
                                 cfg=EntryConfig(cav_share=float(g0['cav_share']), max_pending=10 ** 9))
                self.groups = np.asarray(groups, np.int64)
                self.group_rows = {int(g): np.flatnonzero(self.groups == g) for g in np.unique(self.groups)}
                self.group_rates = {g: observed_rates(np.asarray(specs[g]['counter'], np.float32),
                                                      self.cfg.window_bins) * float(self.cfg.rate_scale)
                                    for g in self.group_rows}
                self.group_rng = {g: np.random.default_rng(int(specs[g]['seed'])) for g in self.group_rows}
                self.row_share = np.array([float(specs[int(g)]['cav_share']) for g in self.groups])
                self.row_p_lane1 = None
                self.queues = [[[] for _ in range(N_ENTRANCES)] for _ in range(len(self.groups))]
                self.backlog_log = []
                self.n_entrances = N_ENTRANCES

            def _observe(self, state):
                if self.row_p_lane1 is not None:
                    return
                present = state['present'].detach().bool().cpu().numpy()
                lane = state['lane'].detach().cpu().numpy()
                n0 = (present & (lane == 0)).sum(1)
                n1 = (present & (lane == 1)).sum(1)
                self.row_p_lane1 = np.where(n0 + n1 > 0, n1 / np.maximum(n0 + n1, 1), 0.5)
                self.p_lane1 = float(self.row_p_lane1[0])
                self.virgin = ~present & self._pool_row(0, present.shape[1])[None, :]
                self._cav_slots = np.zeros_like(present)

            def insert(self, state):
                torch = _torch()
                from wms.model.entries import _Working
                self._observe(state)
                new_mask = torch.zeros(state['present'].shape, dtype=torch.bool, device=state['present'].device)
                work = _Working(state)
                for g, rows in self.group_rows.items():
                    rng = self.group_rng[g]
                    n = rng.poisson(self.group_rates[g] * self.dt)
                    arrivals = [[] for _ in range(self.n_entrances)]
                    for e in range(self.n_entrances):
                        for _ in range(int(n[e])):
                            arrivals[e].append((float(rng.random()), float(rng.random())))
                    for b in rows:
                        for e in range(self.n_entrances):
                            self.queues[b][e].extend(arrivals[e])
                            self._serve(work, int(b), e)
                out = self._commit(state, work, new_mask)
                self.backlog_log.append(self.pending.sum(-1).astype(np.float64).copy())
                return out

            def _serve(self, work, b, e):
                q = self.queues[b][e]
                if not q:
                    self.pending[b, e] = 0
                    return
                self.cav_share = float(self.row_share[b])
                p1 = float(self.row_p_lane1[b])
                done = 0
                for u_lane, u_cav in q:
                    lanes_try = ([1, 0] if u_lane < p1 else [0, 1]) if e == 0 else [2]
                    if not any(self._try_lane(work, b, e, ln, u_cav) for ln in lanes_try):
                        self.stats[b].deferred_steps += 1
                        break
                    done += 1
                del q[:done]
                self.pending[b, e] = len(q)

            def _place(self, work, b, e, ln, cav, v_new, length_new):
                stt = self.stats[b]
                free = np.flatnonzero(~work.present[b] & self._pool_row(b, work.present.shape[1]))
                if not free.size:
                    stt.no_slot += 1
                    return False
                fresh = free[self.virgin[b, free] & ~self._cav_slots[b, free]]
                k = int(fresh[0] if fresh.size else free[0])
                work.s[b, k], work.v[b, k] = self.entry_s[ln], v_new
                work.length[b, k], work.lane[b, k] = length_new, ln
                work.present[b, k], work.is_cav[b, k] = True, bool(cav)
                if cav:
                    self._cav_slots[b, k] = True
                    stt.inserted_cav += 1
                stt.inserted += 1
                stt.per_entrance[e] += 1
                work.inserted.append((b, k))
                return True

            def summary(self, b=0):
                st = self.stats[b]
                g = int(self.groups[b])
                return dict(rates_veh_s=[float(x) for x in self.group_rates[g]],
                            p_lane1=None if self.row_p_lane1 is None else float(self.row_p_lane1[b]),
                            cav_share=float(self.row_share[b]), inserted=st.inserted, inserted_cav=st.inserted_cav,
                            per_entrance=list(st.per_entrance), deferred_steps=st.deferred_steps,
                            no_slot=st.no_slot, backlog_end=int(self.pending[b].sum()))

        return _Grouped()


class U0RuleController:
    """WM CAV callback [B, n, 4] for rows (arm, draw): MAPPO u0 on every CAV slot (native WM lanes) + the S-REGIMEU0 rule
    (hold members) on strategy rows; ``native`` rows drive every CAV with the WM's native law (diagnostic)."""

    def __init__(self, arms, n_slots, device, policy, group_encoder, *, t0_s=T0_S, accel_max=None, heartbeat=0,
                 label=''):
        torch = _torch()
        self.heartbeat, self.label, self.t_wall = int(heartbeat or 0), str(label), time.time()
        from wms.control.contract import ACCEL_MIN_MPS2
        B = len(arms)
        self.arms = list(arms)
        self.dev = device
        self.t0 = float(t0_s)
        self.policy, self.group_encoder = policy, group_encoder
        self.accel_max = float(accel_max if accel_max is not None else getattr(policy, 'accel_max', 1.5))
        self.accel_min = float(ACCEL_MIN_MPS2)
        self.rules = [arm_rule(a) for a in arms]
        self.native_row = torch.tensor([a == NATIVE for a in arms], dtype=torch.bool, device=device)[:, None]
        shape = (B, int(n_slots))
        self.member = torch.zeros(shape, dtype=torch.bool, device=device)
        self.start = torch.zeros(shape, device=device)
        self.v_w, self.dmax, self.amax, self.hold = (torch.zeros(shape, device=device) for _ in range(4))
        self.next_ok = [-math.inf] * B
        self.acts = [[] for _ in range(B)]               # [t_rel, [slots], span]
        self.held = None
        self.prev_mask = None
        self.last_cmd = torch.zeros(shape, device=device)
        self.updated_step = -1
        self.steps = 0

    # -------------------------------------------------------------------- rule (whole seconds)
    def _mask(self, state):
        return state['present'].bool() & state['is_cav'].bool() & (state['lane'] >= 0)

    def _update(self, state, index):
        if index == self.updated_step:
            return
        self.updated_step = index
        if index % STEPS_PER_CONTROL:
            return
        t = index * DT_S                                   # time since t0 (rule semantics are shift-invariant)
        mask = self._mask(state)
        s = state['s_m']
        gone = self.member & ~mask
        ended = self.member & ((t - self.start) >= self.hold)
        at_drop = self.member & (s >= RELEASE_S)
        self.member = self.member & ~(gone | ended | at_drop)
        due = [b for b, r in enumerate(self.rules) if r is not None and t >= self.next_ok[b]]
        if not due:
            return
        s_h, lane_h = s.detach().cpu().numpy(), state['lane'].detach().cpu().numpy()
        free_h = (mask & ~self.member).detach().cpu().numpy()
        for b in due:
            slots = np.flatnonzero(free_h[b])
            wall = pick_wall_slots(s_h[b, slots], lane_h[b, slots], slots)
            if wall is None:
                continue
            params, solo = self.rules[b]
            members = [j for ln, j in enumerate(wall) if solo is None or ln == solo]
            self.next_ok[b] = t + params['hold_s']
            for j in members:
                self.member[b, j] = True
                self.start[b, j] = t
                self.v_w[b, j], self.dmax[b, j] = params['v_w'], params['dmax']
                self.amax[b, j], self.hold[b, j] = params['amax'], params['hold_s']
            xs = s_h[b, wall]
            self.acts[b].append([t, members, round(float(xs.max() - xs.min()), 2)])

    def native_lane_mask(self, state):
        """CAV slots whose lane follows the WM's native (HDV) lane path: every CAV except active members."""
        index = int(_step_index(state))
        self._update(state, index)
        return self._mask(state) & ~self.member

    # -------------------------------------------------------------------- motion
    def __call__(self, state):
        torch = _torch()
        from wms.control.obs import build_observations
        from wms.control.rsu_field import build_rsu_field
        from wms.control.safety import safe_accel
        from wms.model import cav as cav_motion
        index = int(_step_index(state))
        self._update(state, index)
        self.steps += 1
        if self.heartbeat and index and index % self.heartbeat == 0:
            print(json.dumps(dict(heartbeat=self.label, step=index, sim_s=index * DT_S,
                                  wall_s=round(time.time() - self.t_wall, 1),
                                  acts=[len(a) for a in self.acts])), flush=True)
        mask = self._mask(state)
        s, v, lane = state['s_m'], state['v_mps'], state['lane'].long()
        decision = index % STEPS_PER_CONTROL == 0
        if decision:
            known = mask if self.prev_mask is None else mask & self.prev_mask
            zeros = torch.zeros_like(s)
            obs = build_observations({**state, 'cmd_accel_mps2': torch.where(known, self.last_cmd, zeros),
                                      'lane_intent': zeros})
            signal = self.group_encoder(obs, s, build_rsu_field(state))
            out = self.policy(obs, signal, deterministic=True)
            pol = out[0] if isinstance(out, tuple) else out.accel
            law = torch.maximum(torch.minimum(self.v_w - v, self.amax), -self.dmax)
            self.held = torch.where(self.member, law, pol.to(v.dtype))
        elif self.held is None:
            self.held = torch.zeros_like(s)
        if self.prev_mask is not None and not decision:
            self.held = torch.where(mask & self.prev_mask, self.held, torch.zeros_like(self.held))
        self.held = torch.where(mask, self.held, torch.zeros_like(self.held))
        self.prev_mask = mask
        # lanes: native (WM HDV path) for non-members; members hold their lane; lane-end zone delegated
        native_lanes = mask & ~self.member
        targets = state.get(cav_motion.WM_LANE_TARGET_KEY)
        lane_out = lane
        if targets is not None:
            upstream_or_through = (lane != 2) | (s < 0.0)            # wz3to2: lane-2 changes upstream of the zone
            lane_out = torch.where(native_lanes & upstream_or_through, targets.long(), lane)
        state[cav_motion.WM_MERGE_KEY] = mask & (lane_out == 2)
        # longitudinal: safety in the current lane (deployment parity), executed with SUMO's ballistic update
        executed = safe_accel({**state, 'lane': lane}, self.held, accel_max=self.accel_max)
        executed = cav_motion.apply_exec_cap(state, executed, mask)
        if decision:
            self.last_cmd = torch.where(mask, executed, torch.zeros_like(executed))
        present = state['present'].bool()
        ss = torch.where(present, s, torch.zeros_like(s))
        vv = torch.where(present, v, torch.zeros_like(v))
        command = torch.stack((executed, vv + self.accel_max * DT_S), -1)
        next_s, next_v, applied, _ = cav_motion.direct_command_motion(ss, vv, command, DT_S, self.accel_min,
                                                                      self.accel_max)
        # native rows: the WM's own acceleration for every CAV (planapi native motion), lanes native
        native = state.get('merge_pivot_native_accel')
        if native is not None and bool(self.native_row.any()):
            a_n = native.to(v.dtype)
            v_n = (v + a_n * DT_S).clamp_min(0.0)
            s_n = s + 0.5 * (v + v_n) * DT_S
            use = self.native_row.expand_as(mask)
            next_s = torch.where(use, s_n, next_s)
            next_v = torch.where(use, v_n, next_v)
            applied = torch.where(use, (v_n - v) / DT_S, applied)
        return torch.stack((next_s, lane_out.to(next_s.dtype), next_v, applied), -1)


def _step_index(state):
    idx = state.get('physics_step_index', 0)
    try:
        import torch
        if torch.is_tensor(idx):
            return int(idx.reshape(-1)[0].item())
    except ImportError:
        pass
    return int(idx)


def wm_moment(m):
    """planapi moment of a start state: observable rows only (WM-C moment_input; no world labels)."""
    from . import wm_backend as W
    mom = W.moment_input(m)
    return dict(map=mom['map'], t0=mom['t0'], cavs=mom['cavs'], vehicles=mom.get('vehicles', []),
                counter=mom.get('counter'))


def group_seed(seed, k, digest):
    return int(hashlib.sha256(f'wmev:entries:{seed}:{k}:{digest}'.encode()).hexdigest()[:15], 16)


def _prep_state(handle, m, arms):
    from wms import planapi as P
    arms_m = arms_for(m['cell'], arms)
    if BASE not in arms_m:
        raise ValueError('the u0 base arm is required (gains are vs u0)')
    for a in arms_m:
        arm_rule(a)
    st = P.observe(handle, wm_moment(m))
    veh = st['vehicles']
    share = m['observed'].get('cav_entry_share')
    if share is None:                                   # no counted entry in the window: snapshot share
        share = sum(1 for r in veh if r['is_cav']) / max(1, len(veh))
    return dict(m=m, arms=arms_m, state=st, digest=P.state_digest(st), vehicles=veh, share=float(share),
                counter=st.get('counter'))


def evaluate_states(handle, policy_exe, states, *, arms=DEFAULT_ARMS, K=4, H=H_MAX_S, seed=DEFAULT_SEED,
                    slots=DEFAULT_SLOTS, pad=DEFAULT_PAD, max_rows=None, trace=None, heartbeat=0):
    """Roll the WM from several start states at once (rows = state x draw x arm, chunked by whole (state, draw)
    groups so every arm of a draw shares one call); returns one result dict per state (input order).

    The slot count n is FIXED (``slots``, default max n_obs + ``pad``) so a state's draws do not depend on which
    other states share its call (lane uniforms are [T, n] per (seed, k, state digest))."""
    torch = _torch()
    from wms import planapi as P
    preps = [_prep_state(handle, m, arms) for m in states]
    model, dev = handle.model, torch.device(handle.device)
    n_steps = int(round(float(H) / DT_S))
    steps_w = int(round(WINDOW_S / DT_S))
    n_win = int(math.ceil(n_steps / steps_w))
    n_max_obs = max(len(p['vehicles']) for p in preps)
    n = resolve_slots(slots, [p['m'] for p in preps], n_max_obs, pad)
    if n < n_max_obs + 32:
        raise ValueError(f'slots {n} leave < 32 entry slots for a state with {n_max_obs} observed vehicles')
    exit_s = handle.road.road_end_m if getattr(model.physics, 'exit_at_road_end', True) else None
    max_rows = int(max_rows or handle.max_rows)
    outs = [{a: dict(road_w=[None] * K, src_w=[None] * K, n_w=[None] * K, activations=[None] * K,
                     first_act_s=[None] * K, first_div_s=[None] * K, no_slot=[None] * K, backlog_end=[None] * K)
             for a in p['arms']} for p in preps]
    groups_all = [(si, k) for si in range(len(preps)) for k in range(int(K))]
    chunks, cur, cur_rows = [], [], 0
    for si, k in groups_all:
        need = len(preps[si]['arms'])
        if cur and cur_rows + need > max_rows:
            chunks.append(cur)
            cur, cur_rows = [], 0
        cur.append((si, k))
        cur_rows += need
    if cur:
        chunks.append(cur)
    t_start = time.time()
    for chunk in chunks:
        rows = [(si, a, k) for si, k in chunk for a in preps[si]['arms']]
        b = len(rows)
        S, V, L, LEN = (np.zeros((b, n), np.float32) for _ in range(4))
        LEN[:] = P.NOMINAL_LENGTH_M
        PRES, CAV = np.zeros((b, n), bool), np.zeros((b, n), bool)
        for i, (si, _a, _k) in enumerate(rows):
            veh = preps[si]['vehicles']
            no = len(veh)
            S[i, :no] = [r['s'] for r in veh]
            V[i, :no] = [r['v'] for r in veh]
            L[i, :no] = [r['lane'] for r in veh]
            LEN[i, :no] = [r.get('length', P.NOMINAL_LENGTH_M) for r in veh]
            PRES[i, :no] = True
            CAV[i, :no] = [bool(r['is_cav']) for r in veh]
        present = torch.tensor(PRES, device=dev)
        initial = {'s_m': torch.tensor(S, device=dev), 'v_mps': torch.tensor(V, device=dev),
                   'a_mps2': torch.zeros(b, n, device=dev), 'lane': torch.tensor(L, device=dev).long(),
                   'length_m': torch.tensor(LEN, device=dev), 'present': present,
                   'is_cav': torch.tensor(CAV, device=dev)}
        if exit_s is not None:
            initial['exit_s_m'] = torch.full((b, n), float(exit_s), device=dev)
        controller = U0RuleController([a for _si, a, _k in rows], n, dev, policy_exe.policy,
                                      policy_exe.group_encoder, accel_max=policy_exe.accel_max, heartbeat=heartbeat,
                                      label=','.join(sorted({preps[si]['m']['sid'] for si, _k in chunk})))
        uniform = torch.stack([P._draw_uniform(seed, k, preps[si]['digest'], n_steps, n)
                               for si, _a, k in rows]).to(dev)
        entries = None
        if all(p['counter'] is not None for p in preps):
            gid = {g: j for j, g in enumerate(chunk)}
            specs = {gid[(si, k)]: dict(counter=preps[si]['counter'], cav_share=preps[si]['share'],
                                        seed=group_seed(seed, k, preps[si]['digest'])) for si, k in chunk}
            entries = GroupedEntryProcess([gid[(si, k)] for si, _a, k in rows], specs)
        hist = {}
        if getattr(model, 'long_history', False):
            from wms.model.history import LONG_HISTORY_F
            hist = {'driver_history': torch.zeros(b, n, model.driver_history_len, LONG_HISTORY_F, device=dev),
                    'driver_history_mask': torch.zeros(b, n, model.driver_history_len, dtype=torch.bool,
                                                       device=dev)}
        with torch.no_grad():
            roll = model.rollout(initial, n_steps=n_steps, k0=0, cav_controller=controller, lane_uniform=uniform,
                                 entries=entries, **hist)
        speeds = torch.cat((initial['v_mps'][:, None], roll['speeds'][:, :-1]), 1)
        masks = torch.cat((present[:, None], roll['masks'][:, :-1].bool()), 1)
        deficit = ((1.0 - speeds / SPEED_LIMIT_MPS).clamp_min(0.0) * masks).sum(2).double()     # [b, T]
        count = masks.sum(2).double()
        del roll
        pad_t = n_win * steps_w - n_steps
        if pad_t:
            deficit = torch.cat((deficit, deficit.new_zeros(b, pad_t)), 1)
            count = torch.cat((count, count.new_zeros(b, pad_t)), 1)
        road_w = (DT_S * deficit.reshape(b, n_win, steps_w).sum(2)).cpu().numpy()
        n_w = (count.reshape(b, n_win, steps_w).sum(2) / steps_w).cpu().numpy()
        if entries is not None:
            backlog = np.stack(entries.backlog_log, 1)                                          # [b, T]
            backlog = np.concatenate((backlog, np.zeros((b, pad_t))), 1) if pad_t else backlog
            src_w = DT_S * backlog.reshape(b, n_win, steps_w).sum(2)
        else:
            src_w = np.zeros_like(road_w)
        per_step = deficit[:, :n_steps].cpu().numpy()
        base_row = {(si, k): i for i, (si, a, k) in enumerate(rows) if a == BASE}
        for i, (si, a, k) in enumerate(rows):
            if not (np.all(np.isfinite(road_w[i])) and np.all(np.isfinite(src_w[i]))):
                raise FloatingPointError(f'non-finite WM delay ({preps[si]["m"]["sid"]}, {a}, k={k})')
            acts = controller.acts[i]
            diff = np.flatnonzero(np.abs(per_step[i] - per_step[base_row[(si, k)]]) > 1e-6)
            o = outs[si][a]
            o['road_w'][k] = [round(float(x), 4) for x in road_w[i]]
            o['src_w'][k] = [round(float(x), 4) for x in src_w[i]]
            o['n_w'][k] = [round(float(x), 2) for x in n_w[i]]
            o['activations'][k] = len(acts)
            o['first_act_s'][k] = acts[0][0] if acts else None
            o['first_div_s'][k] = float(diff[0] * DT_S) if len(diff) else None
            o['no_slot'][k] = int(entries.stats[i].no_slot) if entries is not None else 0
            o['backlog_end'][k] = int(entries.pending[i].sum()) if entries is not None else 0
        if trace is not None:
            trace.setdefault('rows', []).extend([(a, k) if len(preps) == 1 else (si, a, k) for si, a, k in rows])
            trace.setdefault('acts', []).extend(controller.acts)
            trace.setdefault('per_step', []).append(per_step)
            trace.setdefault('speeds', []).append(speeds.cpu())
            trace['n_slots'] = n
            trace.setdefault('entry_stats', []).extend(
                [entries.summary(i) for i in range(b)] if entries is not None else [])
    wall = round(time.time() - t_start, 2)
    return [dict(sid=p['m']['sid'], cell=p['m']['cell'], seed=int(p['m']['seed']), t0=int(p['m']['t0']),
                 H=float(H), K=int(K), window_s=WINDOW_S, arms=o, n_obs=len(p['vehicles']), n_slots=n,
                 cav_entry_share=p['share'], digest=p['digest'][:16], wall_s=wall, batch=len(preps))
            for p, o in zip(preps, outs)]


def resolve_slots(slots, states, n_max_obs=0, pad=DEFAULT_PAD):
    """Slot count of one call: an int, 'auto' (CELL_SLOTS of the demand level; all states of a call must share
    it), or None (largest observed count + pad)."""
    if slots in (None, '', 'none'):
        return int(n_max_obs) + int(pad)
    if str(slots) == 'auto':
        vals = {CELL_SLOTS.get(int(str(m['cell']).split('_')[0][1:]), 448) for m in states}
        if len(vals) != 1:
            raise ValueError(f'states of one call need one slot count, got {sorted(vals)}')
        return vals.pop()
    return int(slots)


def evaluate_state(handle, policy_exe, m, **kw):
    """One start state (see :func:`evaluate_states`)."""
    return evaluate_states(handle, policy_exe, [m], **kw)[0]


def load_policy(path, device='cpu'):
    from wms.control.mappo.eval_bridge import load_policy_fn
    exe = load_policy_fn(path, deterministic=True, device=device)
    if exe.has_lane_head:
        raise ValueError('u0 is speed-only MAPPO (native lanes); this checkpoint has a lane head')
    return exe


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _chunk_count(n_arms_list, K, max_rows):
    """Number of rollout calls of evaluate_states for states with these arm counts (same packing rule)."""
    calls, cur = 0, 0
    for na in n_arms_list:
        for _k in range(int(K)):
            if cur and cur + na > max_rows:
                calls, cur = calls + 1, 0
            cur += na
    return calls + (1 if cur else 0)


def _device_setup(device, threads=None, mem_fraction=None):
    torch = _torch()
    if threads:
        torch.set_num_threads(int(threads))
    if mem_fraction and str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction))


def _batches(states, states_per_call, slots):
    """Consecutive batches of <= states_per_call states sharing one slot count (one demand level when 'auto')."""
    spc = max(1, int(states_per_call))
    out, cur, key = [], [], None
    for m in states:
        k = resolve_slots(slots, [m], len(m['all_cavs_t0']) + len(m['observed']['vehicles'])) \
            if str(slots) == 'auto' else None
        if cur and (len(cur) >= spc or k != key):
            out.append(cur)
            cur = []
        cur.append(m)
        key = k
    if cur:
        out.append(cur)
    return out


def score(states_path, out_path, *, checkpoint, u0, map_spec=None, device='cpu', arms=DEFAULT_ARMS, K=8, H=H_MAX_S,
          seed=DEFAULT_SEED, slots=DEFAULT_SLOTS, max_rows=32, states_per_call=1, sids=None, shard=None,
          threads=None, allow_fresh=False, limit=None, max_per_cell=None, mem_fraction=None, heartbeat=400):
    """Append one JSON line per start state to ``out_path`` (resumable: states already in the file are skipped).

    ``states_per_call`` start states share each batched WM call (rows <= ``max_rows``); results do not depend on the
    grouping because the slot count is fixed (``slots``) and every draw is keyed by (seed, k, state digest)."""
    _device_setup(device, threads, mem_fraction)
    from wms import planapi as P
    if not checkpoint and not allow_fresh:
        raise ValueError('a trained WM checkpoint is required (allow_fresh only for dry runs)')
    handle = P.load(checkpoint or None, map_spec=map_spec, device=device, max_rows=max_rows)
    exe = load_policy(u0, device=device)
    out_path = G.check_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(schema=SCHEMA + '.scores_meta', checkpoint=str(checkpoint), checkpoint_sha256=handle.meta.get(
        'checkpoint_sha256'), trained=bool(handle.meta.get('trained')), u0=str(u0), u0_sha256=sha256_file(u0),
        road_digest=handle.road.digest, arms=list(arms), K=int(K), H=float(H), seed=int(seed), slots=str(slots),
        max_rows=int(max_rows), states_per_call=int(states_per_call), device=device, states=str(states_path),
        states_sha256=sha256_file(G.check_path(states_path)), rule_source=RULE_SOURCE)
    meta_path = Path(str(out_path) + '.meta.json')
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        keys = ('checkpoint_sha256', 'u0_sha256', 'arms', 'K', 'H', 'seed', 'slots', 'states_sha256')
        if any(old.get(k) != meta.get(k) for k in keys):
            raise ValueError(f'{meta_path}: different run settings; use a new --out')
    else:
        meta_path.write_text(json.dumps(meta, indent=1))
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)['sid'])
    states = load_states(states_path, sids=sids, shard=shard, max_per_cell=max_per_cell)
    todo = [m for m in states if m['sid'] not in done][:limit]
    report = dict(todo=len(todo), skipped=len(states) - len(todo), wall_s=0.0)
    t0 = time.time()
    for batch in _batches(todo, states_per_call, slots):
        try:
            results = evaluate_states(handle, exe, batch, arms=arms, K=K, H=H, seed=seed, slots=slots,
                                      max_rows=max_rows, heartbeat=heartbeat)
        except Exception as exc:                                       # recorded, never imputed
            results = [dict(sid=m['sid'], cell=m['cell'], seed=int(m['seed']), error=f'{type(exc).__name__}: {exc}')
                       for m in batch]
        with out_path.open('a') as fh:
            for res in results:
                fh.write(json.dumps(res) + '\n')
            fh.flush()
        for res in results:
            print(json.dumps(dict(sid=res['sid'], wall_s=res.get('wall_s'), error=res.get('error'))), flush=True)
    report['wall_s'] = round(time.time() - t0, 1)
    return report


# ================================================================================ cost (dry run + calibration)
def cost_estimate(states_path, *, arms=DEFAULT_ARMS, K=8, H=H_MAX_S, max_rows=32, states_per_call=1,
                  s_per_step=None, workers=1, parallel_efficiency=0.8, sids=None, max_per_cell=None,
                  slots=DEFAULT_SLOTS):
    """Rollout calls / steps of a full run and the wall minutes for a measured (or assumed) seconds per rollout step
    of one call (calibrate at the planned max_rows / states_per_call)."""
    states = load_states(states_path, sids=sids, max_per_cell=max_per_cell)
    n_steps = int(round(H / DT_S))
    by_cell, row_steps = {}, 0
    for m in states:
        by_cell[m['cell']] = by_cell.get(m['cell'], 0) + 1
        row_steps += len(arms_for(m['cell'], arms)) * K * n_steps
    calls, calls_by_slots = 0, {}
    for batch in _batches(states, states_per_call, slots):
        c = _chunk_count([len(arms_for(m['cell'], arms)) for m in batch], K, max_rows)
        n = resolve_slots(slots, batch, max(len(m['all_cavs_t0']) + len(m['observed']['vehicles']) for m in batch))
        calls += c
        calls_by_slots[n] = calls_by_slots.get(n, 0) + c
    out = dict(states=len(states), by_cell=by_cell, arms=list(arms), K=K, H=H, max_rows=max_rows,
               states_per_call=int(states_per_call), slots=str(slots), calls=calls, calls_by_slots=calls_by_slots,
               steps_per_call=n_steps, call_steps=calls * n_steps, row_steps=row_steps)
    if s_per_step:
        # s_per_step is calibrated at the LARGEST slot count; smaller calls scale ~ (n / n_max)^2 (conservative)
        n_max = max(calls_by_slots) if calls_by_slots else 1
        eff_calls = sum(c * (n / n_max) ** 2 for n, c in calls_by_slots.items()) if str(slots) == 'auto' else calls
        out['effective_calls_at_max_slots'] = round(eff_calls, 1)
        serial_min = eff_calls * n_steps * float(s_per_step) / 60.0
        wall = serial_min / max(1.0, workers * parallel_efficiency)
        out.update(s_per_step=float(s_per_step), serial_minutes=round(serial_min, 1), workers=int(workers),
                   parallel_efficiency=parallel_efficiency, wall_minutes=round(wall, 1))
    return out


def calibrate(states_path, *, checkpoint, u0, map_spec=None, device='cpu', arms=DEFAULT_ARMS, K=8, steps=40,
              max_rows=32, states_per_call=1, slots=DEFAULT_SLOTS, threads=None, allow_fresh=False, sid=None,
              mem_fraction=None, H=H_MAX_S):
    """Seconds per rollout step of ONE call at the planned packing (states_per_call heaviest states, rows <= max_rows)
    on this device (excludes model loading; includes the CPU-side entry / rule work).  On CUDA also the peak memory
    of the call and its linear extrapolation to the full horizon H (the rollout keeps per-step logs)."""
    torch = _torch()
    _device_setup(device, threads, mem_fraction)
    from wms import planapi as P
    if not checkpoint and not allow_fresh:
        raise ValueError('checkpoint required')
    handle = P.load(checkpoint or None, map_spec=map_spec, device=device, max_rows=max_rows)
    exe = load_policy(u0, device=device)
    states = load_states(states_path, sids=[sid] if sid else None)
    states = sorted(states, key=lambda r: -(len(r['all_cavs_t0']) + len(r['observed']['vehicles'])))
    batch = _batches(states, states_per_call, slots)[0]
    per_state = len(arms_for(batch[0]['cell'], arms))
    kk = max(1, min(int(K), max_rows // max(1, per_state * len(batch))))
    evaluate_states(handle, exe, batch, arms=arms, K=kk, H=2 * DT_S, slots=slots, max_rows=max_rows)   # warm-up
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    mem = {}
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t = time.time()
    res = evaluate_states(handle, exe, batch, arms=arms, K=kk, H=steps * DT_S, slots=slots, max_rows=max_rows)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    dt = (time.time() - t) / steps
    if device.startswith('cuda'):
        peak = torch.cuda.max_memory_allocated() - base
        mem = dict(peak_gb=round(peak / 2 ** 30, 3), peak_gb_full_H=round(peak / 2 ** 30 * float(H) / (steps * DT_S), 2),
                   note='linear in steps (per-step rollout logs); plan max_rows so peak_gb_full_H fits mem_fraction')
    rows = sum(len(r['arms']) for r in res) * kk
    return dict(sids=[m['sid'] for m in batch], device=device, rows=rows, draws_per_state=kk, steps=steps,
                s_per_step=round(dt, 4), slots=res[0]['n_slots'], threads=threads, memory=mem)


# ================================================================================ CLI
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.regime_u0')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('states', help='observable start states at t0 from WM-D native tapes (zero WM)')
    b.add_argument('--tapes', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--ref-out')
    b.add_argument('--cells', default=','.join(CELLS))
    b.add_argument('--t0', type=float, default=T0_S)
    b.add_argument('--max-worlds', type=int)

    def common(p):
        p.add_argument('--states', required=True)
        p.add_argument('--checkpoint')
        p.add_argument('--u0', help='speed-only MAPPO169 policy_00169.pt (sha d9042cc5)')
        p.add_argument('--map-spec')
        p.add_argument('--device', default='cpu')
        p.add_argument('--arms', default=','.join(DEFAULT_ARMS), help="'auto' = the SUMO arms of each cell; "
                                                                     "'auto,native' adds the horizon-check arm")
        p.add_argument('--K', type=int, default=8)
        p.add_argument('--H', type=float, default=H_MAX_S)
        p.add_argument('--max-rows', type=int, default=32)
        p.add_argument('--states-per-call', type=int, default=1)
        p.add_argument('--slots', default=DEFAULT_SLOTS, help="'auto' (CELL_SLOTS per demand) or an int")
        p.add_argument('--max-per-cell', type=int)
        p.add_argument('--threads', type=int)
        p.add_argument('--mem-fraction', type=float, help='CUDA per-process memory cap (shared GPU)')
        p.add_argument('--allow-fresh', action='store_true', help='dry run with an untrained WM (never a result)')
    s = sub.add_parser('score', help='WM rollouts (u0 base + rule strategies), JSONL per start state, resumable')
    common(s)
    s.add_argument('--out', required=True)
    s.add_argument('--seed', type=int, default=DEFAULT_SEED)
    s.add_argument('--sids')
    s.add_argument('--shard', help='i/n: this process scores states i, i+n, ...')
    s.add_argument('--limit', type=int)
    s.add_argument('--heartbeat', type=int, default=400, help='progress line every N physics steps (0 = off)')
    c = sub.add_parser('cost', help='dry-run cost; --calibrate measures s/step of one call on --device')
    common(c)
    c.add_argument('--s-per-step', type=float)
    c.add_argument('--workers', type=int, default=1)
    c.add_argument('--parallel-efficiency', type=float, default=0.8)
    c.add_argument('--calibrate', action='store_true')
    c.add_argument('--steps', type=int, default=40)
    a = ap.parse_args(argv)
    if a.cmd == 'states':
        print(json.dumps(build_states(a.tapes, a.out, cells=a.cells.split(','), t0=a.t0, ref_out=a.ref_out,
                                      max_worlds=a.max_worlds)))
        return
    arms = tuple(x for x in a.arms.split(',') if x)
    if a.cmd == 'score':
        if not a.u0:
            ap.error('--u0 is required')
        shard = tuple(int(x) for x in a.shard.split('/')) if a.shard else None
        print(json.dumps(score(a.states, a.out, checkpoint=a.checkpoint, u0=a.u0, map_spec=a.map_spec,
                               device=a.device, arms=arms, K=a.K, H=a.H, seed=a.seed, slots=a.slots,
                               max_rows=a.max_rows, states_per_call=a.states_per_call,
                               sids=a.sids.split(',') if a.sids else None, shard=shard, threads=a.threads,
                               allow_fresh=a.allow_fresh, limit=a.limit, max_per_cell=a.max_per_cell,
                               mem_fraction=a.mem_fraction, heartbeat=a.heartbeat)))
        return
    sps = a.s_per_step
    cal = None
    if a.calibrate:
        if not a.u0:
            ap.error('--u0 is required for --calibrate')
        cal = calibrate(a.states, checkpoint=a.checkpoint, u0=a.u0, map_spec=a.map_spec, device=a.device, arms=arms,
                        K=a.K, steps=a.steps, max_rows=a.max_rows, states_per_call=a.states_per_call,
                        slots=a.slots, threads=a.threads, allow_fresh=a.allow_fresh, mem_fraction=a.mem_fraction,
                        H=a.H)
        sps = cal['s_per_step']
    est = cost_estimate(a.states, arms=arms, K=a.K, H=a.H, max_rows=a.max_rows, states_per_call=a.states_per_call,
                        s_per_step=sps, workers=a.workers, parallel_efficiency=a.parallel_efficiency,
                        max_per_cell=a.max_per_cell, slots=a.slots)
    print(json.dumps(dict(calibration=cal, estimate=est), indent=1))


if __name__ == '__main__':
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    main()
