"""A75 hold forms inside the frozen WM-S rollout (plan-to-action translation wrapper). Zero SUMO, no training.

``evaluate`` replaces ``wms.planapi:evaluate`` through the WMA backend ``api`` override (the WM tree is NOT edited):
for one call it binds
  - ``planapi._member_specs``      -> the original + ``form`` (the member's new A75 role fields, params['form'])
  - ``planapi.HoldSpeedController`` -> ``HoldFormController`` (original law for members without a form)
and then calls the E1-FIX common-random-number evaluate (``wm_fix_crn.evaluate``: arrivals, entrant CAV share and
merge-noise keys depend on (seed, state digest, draw index) only - never on the action list or chunking).

HoldFormController mirrors the executor ``a75_holdforms.HoldFormCore._member`` timeline (SPEC_HOLDFORMS.md) for one
activation at t = 0 of the rollout (the WM search scores one activation per start state):
  decisions at whole seconds (el = t), law a = clip(target - v, -dmax, amax) recomputed every 0.5 s physics step as
  the original HoldSpeedController does;
  onset roles: event [onset_s, onset_s + hold_s);
  gap roles: armed [onset_s, onset_s + gap_arm_s); an event fires at the first armed second at which a vehicle on
  WM lane 2 (closing lane), upstream of the drop, present, within the member's own 100 m / +-1 lane sensing has
  dx = s_veh - s_cav in [gap_dx_lo, gap_dx_hi] (and drop - s_veh <= gap_drop_m when set); event ends after hold_s
  ('timer') or at the first second >= 1 s after firing with no such vehicle ('clear'); up to gap_events events;
  dv_rel / taper_in_s / taper_out_s / pulse_on_s+pulse_off_s exactly as executor hold_target / pulse_on;
  release at the drop (x >= release_x) as the original controller.
Off-pulse seconds and pre-event seconds use the WM native motion (as pre-onset seconds already do); members keep
their lane for the whole rollout (original WM member semantics).
"""
from __future__ import annotations

import math

import torch

import wms.planapi as P

X_B = 2696.0
SENSE_M = 100.0
SENSE_LANES = 1
CLOSING_WM_LANE = 2
FORM_KEYS = ('start', 'gap_dx_lo', 'gap_dx_hi', 'gap_drop_m', 'gap_arm_s', 'gap_release', 'gap_events', 'dv_rel',
             'taper_in_s', 'taper_out_s', 'pulse_on_s', 'pulse_off_s')
_REAL = dict(member_specs=P._member_specs, controller=P.HoldSpeedController)


def member_specs(actions, index):
    """Original planapi member specs + ``form`` (dict of the new role fields) for 'hold' members that carry one."""
    out = _REAL['member_specs'](actions, index)
    for a, mem in zip(actions, out):
        src = list(a.get('members', ()) or ())
        if len(src) != len(mem):
            raise P.WMContractError(f"action {a.get('id')}: member spec count mismatch")
        for m, spec in zip(src, mem):
            form = (m.get('params') or {}).get('form')
            if form:
                if str(m.get('role', 'hold')) != 'hold':
                    raise P.WMContractError('hold forms are only defined for hold members')
                bad = set(form) - set(FORM_KEYS)
                if bad:
                    raise P.WMContractError(f'unknown form keys {sorted(bad)}')
                spec['form'] = dict(form)
    return out


class HoldFormController(P.HoldSpeedController):
    """HoldSpeedController + the A75 hold-form timeline for members whose spec carries ``form``."""

    def __init__(self, members_by_row, n_slots, anchor_x, dt_s, device):
        super().__init__(members_by_row, n_slots, anchor_x, dt_s, device)
        self.any_form = any(m.get('form') for mem in members_by_row for m in mem)
        if not self.any_form:
            return
        rows = len(members_by_row)
        shape = (rows, n_slots)
        self.drop_s = X_B - float(anchor_x)

        def full(value, dtype=torch.float32):
            return torch.full(shape, value, dtype=dtype, device=device)
        self.fm, self.gap, self.clear = full(False, torch.bool), full(False, torch.bool), full(False, torch.bool)
        self.dxlo, self.dxhi, self.dropm, self.arm = full(0.0), full(0.0), full(math.inf), full(0.0)
        self.nmax, self.dv = full(1.0), full(math.nan)
        self.tin, self.tout, self.pon, self.poff = full(0.0), full(0.0), full(0.0), full(0.0)
        self.ev, self.done = full(False, torch.bool), full(False, torch.bool)
        self.fire, self.v0, self.nev = full(0.0), full(0.0), full(0.0)
        gslots = []
        for b, members in enumerate(members_by_row):
            gs = []
            for m in members:
                f = m.get('form')
                if not f:
                    continue
                j = m['slot']
                self.fm[b, j] = True
                if f.get('start', 'onset') == 'gap':
                    gs.append(j)
                    self.gap[b, j] = True
                    self.dxlo[b, j], self.dxhi[b, j] = float(f['gap_dx_lo']), float(f['gap_dx_hi'])
                    if f.get('gap_drop_m') is not None:
                        self.dropm[b, j] = float(f['gap_drop_m'])
                    self.arm[b, j] = float(f['gap_arm_s'])
                    self.clear[b, j] = f.get('gap_release', 'timer') == 'clear'
                    self.nmax[b, j] = float(f.get('gap_events', 1))
                if f.get('dv_rel') is not None:
                    self.dv[b, j] = float(f['dv_rel'])
                self.tin[b, j] = float(f.get('taper_in_s', 0) or 0)
                self.tout[b, j] = float(f.get('taper_out_s', 0) or 0)
                if f.get('pulse_on_s') is not None:
                    self.pon[b, j], self.poff[b, j] = float(f['pulse_on_s']), float(f['pulse_off_s'])
            gslots.append(gs)
        gm = max([len(g) for g in gslots] + [1])
        self.gslot = torch.zeros((rows, gm), dtype=torch.long, device=device)
        self.gmask = torch.zeros((rows, gm), dtype=torch.bool, device=device)
        for b, gs in enumerate(gslots):
            for i, j in enumerate(gs):
                self.gslot[b, i], self.gmask[b, i] = j, True
        self.has_gap = bool(self.gmask.any())
        self.hold = self.end - self.start

    # -------------------------------------------------------------------------------- pure pieces
    def candidate(self, s, lane, present):
        """[B, n] bool: gap member slot j sees a qualifying closing-lane vehicle (executor gap_candidate)."""
        if not self.has_gap:
            return torch.zeros_like(self.fm)
        gs = self.gslot
        sj = torch.gather(s, 1, gs)[:, :, None]
        lj = torch.gather(lane, 1, gs)[:, :, None]
        si, li = s[:, None, :], lane[:, None, :]
        dx = si - sj
        lo = torch.gather(self.dxlo, 1, gs)[:, :, None]
        hi = torch.gather(self.dxhi, 1, gs)[:, :, None]
        dm = torch.gather(self.dropm, 1, gs)[:, :, None]
        slot = torch.arange(s.shape[1], device=s.device)[None, None, :]
        ok = (present.bool()[:, None, :] & (li == CLOSING_WM_LANE) & (si < self.drop_s) & (slot != gs[:, :, None])
              & (dx.abs() <= SENSE_M) & ((li - lj).abs() <= SENSE_LANES) & (dx >= lo) & (dx <= hi)
              & ((self.drop_s - si) <= dm))
        hit = (ok.any(-1) & self.gmask).long()
        out = torch.zeros(s.shape, dtype=torch.long, device=s.device)
        out.scatter_add_(1, gs, hit)
        return out > 0

    def decide(self, el, s, v, lane, present):
        """One decision second el (int) of the executor timeline, in place."""
        live = self.fm & ~self.done & ~self.released
        onset, hold = self.start, self.hold
        og = live & ~self.gap
        end_o = og & self.ev & (el >= onset + hold)
        self.ev &= ~end_o
        self.done |= end_o
        start_o = og & ~self.ev & ~self.done & (self.nev < 1) & (el >= onset) & (el < onset + hold)
        self._fire(start_o, el, v)
        g = live & self.gap
        if bool(g.any()):
            cand = self.candidate(s, lane, present)
            run = g & self.ev
            hh = el - self.fire
            end_t = run & (hh >= hold)
            end_c = run & ~end_t & self.clear & (hh >= 1) & ~cand
            self.ev &= ~(end_t | end_c)
            idle = g & ~self.ev
            self.done |= idle & ((self.nev >= self.nmax) | (el >= onset + self.arm))
            fire = g & ~self.ev & ~self.done & (el >= onset) & cand
            self._fire(fire, el, v)

    def _fire(self, mask, el, v):
        self.ev |= mask
        self.fire = torch.where(mask, torch.full_like(self.fire, float(el)), self.fire)
        self.v0 = torch.where(mask, v.to(self.v0.dtype), self.v0)
        self.nev = self.nev + mask.to(self.nev.dtype)

    def target(self, h):
        """Executor hold_target per slot at hold second h (tensor)."""
        tgt = torch.where(torch.isnan(self.dv), self.v_w, torch.maximum(self.v_w, self.v0 - self.dv))
        f = torch.ones_like(tgt)
        f = torch.where(self.tin > 0, torch.minimum(f, (h + 1.0) / self.tin.clamp_min(1.0)), f)
        f = torch.where(self.tout > 0, torch.minimum(f, (self.hold - 1.0 - h).clamp_min(0.0)
                                                     / self.tout.clamp_min(1.0)), f)
        return torch.where(f >= 1.0, tgt, self.v0 + (tgt - self.v0) * f)

    def pulse_on(self, h):
        period = (self.pon + self.poff).clamp_min(1.0)
        return (self.pon <= 0) | (torch.remainder(h, period) < self.pon)

    # -------------------------------------------------------------------------------- callback
    def __call__(self, state):
        if not self.any_form:
            return super().__call__(state)
        from wms.model import cav as cav_motion

        s, v, lane = state["s_m"], state["v_mps"], state["lane"].long()
        present = state.get("present")
        if present is None:
            present = torch.ones_like(self.member)
        t = float(state.get("physics_step_index", 0)) * self.dt
        native = state.get("merge_pivot_native_accel")
        if native is None:
            native = torch.zeros_like(v)
        x = s + self.anchor_x
        ax = torch.gather(x, 1, self.anchor.clamp_min(0))
        self.released = self.released | (self.member & (
            (x >= self.release) | ((self.anchor >= 0) & (ax >= self.anchor_release))))
        if abs(t - round(t)) < 1e-6:
            self.decide(int(round(t)), s, v, lane, present)
        plain = self.member & ~self.fm & (t >= self.start) & (t < self.end) & ~self.released
        h = torch.floor(t - self.fire + 1e-6)
        form = self.fm & self.ev & ~self.done & ~self.released & self.pulse_on(h)
        tgt = torch.where(form, self.target(h), self.v_w)
        hold = torch.maximum(torch.minimum((tgt - v) / 1.0, self.amax), -self.dmax)
        a = torch.where(plain | form, hold, native.to(v.dtype))
        v_next = (v + a * self.dt).clamp_min(0.0)
        a = (v_next - v) / self.dt
        s_next = s + 0.5 * (v + v_next) * self.dt
        state[cav_motion.WM_MERGE_KEY] = self.member & (lane == 2)
        return torch.stack((s_next, lane.to(v.dtype), v_next, a), dim=-1)


def evaluate(handle, state, actions, *, K=8, H=300.0, seed=0, trace=None):
    """``{action_id: [J_1..J_K]}`` as wms.planapi.evaluate, with hold forms and E1-FIX common random numbers."""
    import wm_fix_crn as CRN
    real = (P._member_specs, P.HoldSpeedController)
    P._member_specs, P.HoldSpeedController = member_specs, HoldFormController
    try:
        return CRN.evaluate(handle, state, actions, K=K, H=H, seed=seed, trace=trace)
    finally:
        P._member_specs, P.HoldSpeedController = real
