"""Cooperation-move perturbations for WM data (WM-B): abreast walls, closing-lane pairs, random groups.

Pure numpy over the env frame (no SUMO import). Move semantics are the NEWMAP2 (N3) rule-based
library on wz3to2, projected into the WM frame (closing lane = WM lane 2, adjacent = WM lane 1,
drop at s = drop_s; NEWMAP2 x = s + anchor):

* member command each 1 s tick: a = clip(v_w - v, -dmax, amax) (direct acceleration, recorded
  as controller ``direct``), released at hold end, at the drop (s >= drop_s), when the vehicle
  leaves, or (pair_t) when the closing-lane anchor reaches drop_s - 150 m / leaves;
* selection zone s in [drop_s - 1800, drop_s - 400] (NEWMAP2 ZONE), non-junction CAVs;
  pair = closing-lane CAV + adjacent-lane CAV within 100 m (min gap, then most downstream);
  wall = one CAV per WM lane within 200 m (min spread, then most downstream);
  random group R = 2-3 zone CAVs within 300 m with (v_w, hold_s, dmax) drawn from R_GRID.

A schedule draws a move family every 30-90 s after the previous move ends (first attempt 20-60 s
after the control start); one move is active at a time. All draws come from a per-episode
``random.Random(key)``. HDVs are never selected (members are present CAV slots only).
"""
from __future__ import annotations

import itertools
import random

import numpy as np

AMAX = 1.0
VARIANTS = {
    'pair_g': ('closing_pair', dict(v_w=22.0, hold_s=120, dmax=1.0)),
    'pair_h': ('closing_pair', dict(v_w=15.0, hold_s=90, dmax=1.5)),
    'pair_t': ('closing_pair', dict(v_w=22.0, hold_s=150, dmax=1.0, anchor_release_ds=-150.0)),
    'wall_g': ('abreast_wall', dict(v_w=22.0, hold_s=120, dmax=1.0)),
    'wall_h': ('abreast_wall', dict(v_w=15.0, hold_s=90, dmax=1.5)),
    'wall_l': ('abreast_wall', dict(v_w=18.0, hold_s=180, dmax=1.0)),
}
R_GRID = dict(v_w=(15.0, 18.0, 22.0), hold_s=(90, 120, 180), dmax=(1.0, 1.5))
FAMILY_P = (('abreast_wall', 0.35), ('closing_pair', 0.35), ('random_group', 0.30))
ZONE_DS = (-1800.0, -400.0)
PAIR_DX, WALL_SPREAD, R_SPREAD = 100.0, 200.0, 300.0
FIRST_S, GAP_S, RETRY_S = (20.0, 60.0), (30.0, 90.0), 10.0


def zone_cavs(frame, vehicle_ids, drop_s, exclude=()):
    """[(vid, slot, wm_lane, s, v)] of present, non-junction CAVs in the selection zone."""
    present = np.asarray(frame.present, bool) & np.asarray(frame.is_cav, bool)
    present &= ~np.asarray(frame.junction, bool)
    s = np.asarray(frame.s, np.float64)
    lo, hi = drop_s + ZONE_DS[0], drop_s + ZONE_DS[1]
    out = []
    for slot in np.flatnonzero(present & (s >= lo) & (s <= hi)).tolist():
        vid = vehicle_ids[int(frame.vid_index[slot])]
        if vid in exclude:
            continue
        lane = int(frame.lane[slot])
        if lane < 0:
            continue
        out.append((vid, slot, lane, float(s[slot]), float(frame.v[slot])))
    return sorted(out)


def pick_pair(cands, closing_lane, adjacent_lane):
    best = None
    for c0 in (c for c in cands if c[2] == closing_lane):
        for c1 in (c for c in cands if c[2] == adjacent_lane):
            dx = abs(c0[3] - c1[3])
            if dx <= PAIR_DX:
                key = (dx, -c0[3], c0[0], c1[0])
                if best is None or key < best[0]:
                    best = (key, [c0, c1])
    return None if best is None else best[1]


def pick_wall(cands, lanes):
    per = [[c for c in cands if c[2] == ln] for ln in lanes]
    if any(not x for x in per):
        return None
    best = None
    for combo in itertools.product(*per):
        xs = [c[3] for c in combo]
        sp = max(xs) - min(xs)
        if sp <= WALL_SPREAD:
            key = (sp, -max(xs), tuple(c[0] for c in combo))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else best[1]


def pick_random(cands, rng):
    k = rng.choice([2, 3])
    for kk in (k, 2):
        if len(cands) < kk:
            continue
        for _ in range(200):
            mem = rng.sample(cands, kk)
            xs = [c[3] for c in mem]
            if max(xs) - min(xs) <= R_SPREAD:
                return sorted(mem)
    return None


class CoopSchedule:
    """Per-episode perturbation schedule; ``commands(now, frame, vehicle_ids)`` each control tick."""

    def __init__(self, key, *, drop_s, closing_lane=2, adjacent_lane=1, lanes=(0, 1, 2),
                 start_s=300.0, end_s=1200.0):
        self.key = str(key)
        self.rng = random.Random(self.key)
        self.drop_s = float(drop_s)
        self.closing_lane, self.adjacent_lane = int(closing_lane), int(adjacent_lane)
        self.lanes = tuple(int(x) for x in lanes)
        self.start_s, self.end_s = float(start_s), float(end_s)
        self.next_attempt = self.start_s + self.rng.uniform(*FIRST_S)
        self.active = None            # dict(move record, members state)
        self.moves = []               # completed + active move records
        self.attempts = []            # (t, family, outcome)

    # ------------------------------------------------------------------ selection
    def _draw_family(self):
        u, acc = self.rng.random(), 0.0
        for fam, p in FAMILY_P:
            acc += p
            if u < acc:
                return fam
        return FAMILY_P[-1][0]

    def _try_family(self, fam, cands):
        if fam == 'closing_pair':
            mem = pick_pair(cands, self.closing_lane, self.adjacent_lane)
            if mem is None:
                return None
            variant = self.rng.choice(['pair_g', 'pair_h', 'pair_t'])
            params = dict(VARIANTS[variant][1])
            anchor = mem[0][0]                                  # the closing-lane member
            return variant, params, mem, anchor
        if fam == 'abreast_wall':
            mem = pick_wall(cands, self.lanes)
            if mem is None:
                return None
            variant = self.rng.choice(['wall_g', 'wall_h', 'wall_l'])
            return variant, dict(VARIANTS[variant][1]), mem, None
        mem = pick_random(cands, self.rng)
        if mem is None:
            return None
        params = dict(v_w=self.rng.choice(R_GRID['v_w']), hold_s=self.rng.choice(R_GRID['hold_s']),
                      dmax=self.rng.choice(R_GRID['dmax']))
        return 'R_random', params, mem, None

    def _start(self, now, frame, vehicle_ids):
        cands = zone_cavs(frame, vehicle_ids, self.drop_s)
        first = self._draw_family()
        order = [first] + [f for f, _ in FAMILY_P if f != first]
        for fam in order:
            got = self._try_family(fam, cands)
            if got is None:
                continue
            variant, params, mem, anchor = got
            rec = dict(t_start=float(now), family=fam if variant != 'R_random' else 'random_group',
                       drawn_family=first, variant=variant, params=params, anchor=anchor,
                       members=[dict(id=c[0], lane=c[2], s=round(c[3], 3), v=round(c[4], 4)) for c in mem],
                       releases={}, issued_ticks=0)
            self.active = dict(rec=rec, live={c[0] for c in mem})
            self.moves.append(rec)
            self.attempts.append((float(now), first, variant))
            return True
        self.attempts.append((float(now), first, None))
        return False

    # ------------------------------------------------------------------ per tick
    def commands(self, now, frame, vehicle_ids):
        """(accel {slot: a}, hold_lane {slot: wm lane or -1}) for the active move's members."""
        now = float(now)
        if now < self.start_s - 1e-9 or now >= self.end_s - 1e-9:
            return {}, {}
        if self.active is None and now >= self.next_attempt - 1e-9:
            if not self._start(now, frame, vehicle_ids):
                self.next_attempt = now + RETRY_S
        if self.active is None:
            return {}, {}
        rec, live = self.active['rec'], self.active['live']
        p = rec['params']
        slot_of = {}
        present = np.asarray(frame.present, bool)
        for slot in np.flatnonzero(present).tolist():
            slot_of[vehicle_ids[int(frame.vid_index[slot])]] = slot
        anchor_ok = True
        if rec['anchor'] is not None:
            a_slot = slot_of.get(rec['anchor'])
            if a_slot is None or float(frame.s[a_slot]) >= self.drop_s + p['anchor_release_ds']:
                anchor_ok = False
        accel, hold = {}, {}
        for vid in sorted(live):
            slot = slot_of.get(vid)
            why = None
            if slot is None:
                why = 'left'
            elif now >= rec['t_start'] + float(p['hold_s']) - 1e-9:
                why = 'hold_end'
            elif float(frame.s[slot]) >= self.drop_s:
                why = 'at_drop'
            elif not anchor_ok:
                why = 'anchor'
            if why is not None:
                rec['releases'][vid] = dict(t=now, why=why)
                continue
            v = float(frame.v[slot])
            accel[slot] = max(-float(p['dmax']), min(AMAX, float(p['v_w']) - v))
            hold[slot] = -1 if bool(frame.junction[slot]) else int(frame.lane[slot])
        live.difference_update(rec['releases'])
        if accel:
            rec['issued_ticks'] += 1
        if not live:
            rec['t_end'] = now
            self.active = None
            self.next_attempt = now + self.rng.uniform(*GAP_S)
        return accel, hold

    def record(self):
        return dict(key=self.key, moves=self.moves, attempts=self.attempts,
                    n_moves=len(self.moves),
                    by_family={f: sum(m['family'] == f for m in self.moves) for f, _ in FAMILY_P})
