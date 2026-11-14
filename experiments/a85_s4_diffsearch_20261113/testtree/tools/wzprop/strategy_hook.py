"""SUMO executor of a strategy-level rule on the clean4 plant (S-REGIMEU0 harness, MAPPO u0 base).

Drop-in for tools.coopmwm.regimeu0.RuleHook (same controller_hook protocol: hook(controller, job, prefixes) wraps
controller.apply(rows, commands, now) -> desired; per-second totals series via the harness Recorder base): every
whole second t >= warmup:
  1. active members: released at the end of their program (onset + hold), at the drop (s >= 75.44), or when gone;
     otherwise the hold law a = clip(v_w - v, -dmax, amax) is requested as the absolute pre-safety acceleration
     (single writer, MAPPO safety downstream) once t >= start + onset, and the lane target (own lane, or the role's
     adjacent lc_target after its onset) goes through the G5 single-writer lane path (controller.g5_lane_targets);
  2. if t >= next_ok and an activation slot is free: strategy_search.select_members on the free CAV rows (WM rows ->
     SUMO frame: x = s + 2620.56, SUMO lane = 2 - WM lane) with the observable rows for the field / start gates and
     phase = number of earlier activations (alternating holders); members start at once; next_ok = t + cooldown.
CAV rows only (is_cav); HDVs are never commanded. The member selection and programs are the SAME functions the WM
search uses (strategy_search.select_members / roles), so imagination and SUMO execute one rule.
"""
from __future__ import annotations

from . import geometry as G
from . import strategy_search as SS

S_OFF = G.X_B - G.DROP_S
RELEASE_S = G.DROP_S


def sumo_lane(wm_lane):
    return 2 - int(wm_lane)


def wm_lane_of(sumo):
    return 2 - int(sumo)


def to_sumo_row(r):
    return dict(id=r['vid'], edge='up' if float(r['s_m']) < RELEASE_S else 'wz', lane=sumo_lane(r['lane']),
                x=float(r['s_m']) + S_OFF, v=float(r['v_mps']))


def observed_rows(rows, report_range=100.0, lane_reach=1):
    """CAV rows + non-CAV rows a CAV detects (N1 cav_reports: |ds| <= 100 m, |dWM lane| <= 1), SUMO frame."""
    cav = [r for r in rows if r.get('is_cav')]
    out = [to_sumo_row(r) for r in cav]
    for r in rows:
        if r.get('is_cav'):
            continue
        if any(abs(float(r['s_m']) - float(c['s_m'])) <= report_range and abs(int(r['lane']) - int(c['lane'])) <= lane_reach
               for c in cav):
            out.append(to_sumo_row(r))
    return out


class StrategyCore:
    """Plant-independent rule state machine (tested with fake rows); the harness hook wraps it."""

    def __init__(self, strategy, warmup_s=300):
        self.s = SS.validate(dict(strategy), allow_banned=True)     # executing a banned comparator is allowed
        self.warmup = int(warmup_s)
        self.active = {}             # vid -> dict(start, role, act)
        self.acts, self.issued = [], []
        self.next_ok = -1
        self.lane_held = 0
        self.lane_requests = 0

    def law(self, role, v):
        return max(-float(role['dmax']), min(1.0, float(role['v_w']) - float(v)))

    def step(self, rows, commands, t):
        """rows: WM-frame plant rows (vid, s_m, v_mps, lane, is_cav); commands: base accel per CAV vid.
        Returns ({vid: requested accel}, {vid: target WM lane})."""
        by = {r['vid']: r for r in rows if r.get('is_cav')}
        accel, lanes = {}, {}
        if t < self.warmup:
            return accel, lanes
        for vid, m in list(self.active.items()):
            el = t - m['start']
            role = m['role']
            row = by.get(vid)
            end = int(role['onset_s']) + int(role['hold_s'])
            why = ('left' if row is None or vid not in commands else 'hold_end' if el >= end
                   else 'at_drop' if float(row['s_m']) >= RELEASE_S else None)
            if why is not None:
                m['act']['release'][vid] = [el, why]
                self.active.pop(vid)
                continue
            if el >= int(role['onset_s']):
                accel[vid] = self.law(role, row['v_mps'])
                self.issued.append([t, vid, round(accel[vid], 4)])
                if role['lc_target'] is not None:
                    lanes[vid] = wm_lane_of(role['lc_target'])
                    self.lane_requests += 1
                else:
                    lanes[vid] = int(row['lane'])
            else:
                lanes[vid] = int(row['lane'])
        busy = len({id(m['act']) for m in self.active.values()})
        if t >= self.next_ok and busy < int(self.s.get('slots', 1)):
            free = [to_sumo_row(r) for vid, r in by.items() if vid not in self.active and vid in commands]
            phase = len(self.acts)
            mem = SS.select_members(self.s, free, observed_rows(rows), phase=phase)
            if mem:
                act = dict(t=t, strategy=self.s['name'], members=[c['id'] for c in mem], phase=phase,
                           sumo_lanes=[c['lane'] for c in mem], x=[round(c['x'], 2) for c in mem], release={})
                self.acts.append(act)
                self.next_ok = t + int(self.s['cooldown_s'])
                for role, c in zip(SS.active_roles(self.s, phase), mem):
                    self.active[c['id']] = dict(start=t, role=role, act=act)
                    row = by[c['id']]
                    lanes[c['id']] = int(row['lane'])
                    if int(role['onset_s']) == 0:
                        accel[c['id']] = self.law(role, row['v_mps'])
                        self.issued.append([t, c['id'], round(accel[c['id']], 4)])
                        if role['lc_target'] is not None:
                            lanes[c['id']] = wm_lane_of(role['lc_target'])
                            self.lane_requests += 1
        self.lane_held += len(lanes)
        return accel, lanes


def make_hook(strategy, warmup_s=300, per_vehicle=True):
    """Harness hook class instance (imports the S-REGIMEU0 Recorder lazily: tools.coopmwm.imitreplay); with
    per_vehicle the eval-only LSSEARCH / MULTICRIT per-vehicle logger (pervehicle.PerVehicle) is attached to every
    arm, u0 included (hook.pv.finish() -> PERVEH record)."""
    from tools.coopmwm import imitreplay as ir
    from . import pervehicle as PV

    class StrategyHook(ir.Recorder):
        def __init__(self):
            super().__init__()
            self.core = StrategyCore(strategy, warmup_s) if strategy is not None else None
            self.strategy = None if strategy is None else strategy['name']
            self.pv = PV.PerVehicle() if per_vehicle else None

        @property
        def acts(self):
            return [] if self.core is None else self.core.acts

        @property
        def issued(self):
            return [] if self.core is None else self.core.issued

        @property
        def lane_held(self):
            return 0 if self.core is None else self.core.lane_held

        def __call__(self, controller, job, prefixes):
            self.controller = controller
            if self.pv is not None:
                PV.install_on_observe(self.pv)
            if self.core is None:
                return
            previous = controller.apply

            def apply(rows, commands, now):
                desired = previous(rows, commands, now)
                accel, lanes = self.core.step(rows, commands, int(now))
                for vid, a in accel.items():
                    desired[vid] = dict(requested_raw=a, acceleration_residual=a - commands[vid],
                                        wmsearch_rule=self.strategy)
                if lanes:
                    targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
                    targets.update(lanes)
                    controller.g5_lane_targets = targets
                return desired

            controller.apply = apply

    return StrategyHook()
