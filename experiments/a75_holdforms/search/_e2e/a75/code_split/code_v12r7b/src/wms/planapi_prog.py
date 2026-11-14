"""LCMOVES 'prog' member executor for :mod:`wms.planapi` (WM-A, 2026-11-06; MAIN go).

Semantics follow ``lcm_core.Arm._prog`` (c4/wz-wm-propose lcm_pipeline, lcm_core 48aaae38) in the
WM frame.  'hold' members are untouched: :class:`ProgramController` subclasses
:class:`wms.planapi.HoldSpeedController` and only overrides 'prog' slots, and a call without any
'prog' member never builds it (``tests/wz/test_planapi_hold_golden.py`` pins hold bit-identity).

Per member, decided at whole seconds (the SUMO control tick) and executed for 1 s:

* releases (sticky): slot absent; ``t >= start_s + hold_s``; ``x >= release_x``;
  ``release_on_partner_merge`` partner not on the closing lane before the drop (an unobserved
  partner releases at once, as SUMO's missing position); ``merger``: left the closing lane after
  being on it, or ``x >= fallback_x`` (default drop - 15 m).  Nothing happens before ``start_s``.
* lane program: from ``lc_from`` s and ``x >= lc_from_x``, while on the approach (x < drop) and
  ``x < lc_giveup_x`` (default drop - 300 m), a persistent request toward ``lc_target`` (SUMO 'up'
  lane index -> WM lane).  It executes one lane per 0.5 s step when the move is legal on the road
  (closing lane enterable only on the approach of a lane-drop road) and passes the WM's
  SUMO-LC2013-like secure-gap rule (``wms.physics.mobil.speed_gap_safe``).  Otherwise the member
  keeps its lane (lcMode 513: no own changes).  ``lc_done`` once on the target lane at a tick.
* speed (from ``sp_from``; no ``speed`` = native motion): ``hold`` -> clip((v_w - v) / 1 s);
  ``track`` -> clip(kp (mean ref x + track_off - x) + kv (mean ref v - v)) until settled
  (|err| <= settle_gap after the lane change) or refs gone, then toward v_w (native when v_w is
  absent); ``match`` -> clip(mean speed of the target lane within 100 m - v), only until the lane
  change; ``accel`` -> amax; clip to [-dmax, amax]; commanded speed capped at the speed limit
  (SUMO ``slowDown(min(v + a, allowed), 1 s)``).

STATED LIMITATIONS (approximations):
* ``tau`` (SUMO setTau): only the member's NATIVE (uncommanded) acceleration changes, to IDM with
  time headway ``tau`` and the WM default IDM settings otherwise; SUMO's safe-speed cap of
  commanded speeds (speedMode 27, which depends on tau) is NOT modelled.
* ``lc_mode`` is recorded but not interpreted (513 assumed: only programmed changes).
* After release (and before start) a member moves with the WM native acceleration and keeps its
  lane, except the WM first-passage merge in the lane-end zone; SUMO restores native lane changes.
* Lane changes are instantaneous within one 0.5 s step (as every WM lane change).
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

from wms.planapi import HoldSpeedController, WMContractError

SPEED_KINDS = {None: 0, "hold": 1, "track": 2, "match": 3, "accel": 4}
PROG_KEYS = frozenset({
    "start_s", "hold_s", "release_x", "release_on_partner_merge", "merger", "fallback_x",
    "lc_target", "lc_from", "lc_from_x", "lc_giveup_x", "lc_mode", "tau", "tau_from", "speed",
    "sp_from", "amax", "dmax", "v_w", "track_ref", "track_off", "kp", "kv", "settle_gap",
    "anchor", "anchor_release_x", "partner"})
MATCH_RANGE_M = 100.0


def _num(p: Mapping, key: str, default: float) -> float:
    value = p.get(key)
    value = default if value is None else float(value)
    if math.isnan(value):
        raise WMContractError(f"prog param {key} is NaN")
    return value


def prog_spec(action: Mapping, m: Mapping, index: dict) -> dict:
    aid = action.get("id")
    if str(m["id"]) not in index:
        raise WMContractError(f"action {aid}: member {m['id']} not in the state")
    p = dict(m.get("params", {}))
    unknown = set(p) - PROG_KEYS
    if unknown:
        raise WMContractError(f"action {aid}: unknown prog params {sorted(unknown)}")
    kind = p.get("speed")
    if kind not in SPEED_KINDS:
        raise WMContractError(f"action {aid}: unsupported prog speed kind {kind!r}")
    if kind is not None and p.get("dmax") is None:
        raise WMContractError(f"action {aid}: prog speed needs dmax")
    if kind == "hold" and p.get("v_w") is None:
        raise WMContractError(f"action {aid}: prog hold needs v_w")
    if kind == "track" and not p.get("track_ref"):
        raise WMContractError(f"action {aid}: prog track needs track_ref")
    pm = p.get("release_on_partner_merge")
    return {
        "role": "prog", "slot": index[str(m["id"])], "id": str(m["id"]),
        "start_s": _num(p, "start_s", 0.0), "hold_s": _num(p, "hold_s", math.inf),
        "release_x": _num(p, "release_x", math.inf),
        "partner": None if pm is None else index.get(str(pm), -2),
        "merger": bool(p.get("merger", False)), "fallback_x": p.get("fallback_x"),
        "lc_target": None if p.get("lc_target") is None else int(p["lc_target"]),
        "lc_from": _num(p, "lc_from", 0.0), "lc_from_x": _num(p, "lc_from_x", -1.0e9),
        "lc_giveup_x": p.get("lc_giveup_x"), "tau": p.get("tau"),
        "tau_from": _num(p, "tau_from", 0.0), "speed": SPEED_KINDS[kind],
        "sp_from": _num(p, "sp_from", 0.0), "amax": _num(p, "amax", 1.0),
        "dmax": _num(p, "dmax", 0.0),
        "v_w": math.nan if p.get("v_w") is None else float(p["v_w"]),
        "track_ref": [index[str(r)] for r in (p.get("track_ref") or []) if str(r) in index],
        "track_off": _num(p, "track_off", 0.0), "kp": _num(p, "kp", 0.15),
        "kv": _num(p, "kv", 0.6),
        "settle_gap": math.nan if p.get("settle_gap") is None else float(p["settle_gap"])}


class ProgramController(HoldSpeedController):
    """Hold members exactly as :class:`HoldSpeedController`; 'prog' members per the module doc."""

    FIELDS = ("start", "end", "release_x", "fallback_x", "lc_from", "lc_from_x", "giveup_x",
              "tau", "tau_from", "sp_from", "amax", "dmax", "v_w", "track_off", "kp", "kv",
              "settle_gap")

    def __init__(self, hold_rows, prog_rows, n_slots: int, anchor_x: float, dt_s: float, device,
                 *, physics, road):
        super().__init__(hold_rows, n_slots, anchor_x, dt_s, device)
        shape = (len(prog_rows), n_slots)
        self.physics, self.road = physics, road
        self.drop_x = float(anchor_x) + float(road.merge_zone_s_m[1])
        self.speed_limit = float(road.speed_limit_mps)
        closing_edges = {e for e, _i, wm, _x in road.frame_lanes if wm == road.closing_lane}
        up = {i: wm for e, i, wm, _x in road.frame_lanes if e in closing_edges}

        def full(value, dtype=torch.float32):
            return torch.full(shape, value, dtype=dtype, device=device)
        f = {key: full(math.nan) for key in self.FIELDS}
        self.f = f
        self.prog = full(False, torch.bool)
        self.partner = full(-1, torch.long)
        self.merger = full(False, torch.bool)
        self.tl = full(-1, torch.long)
        self.kind = full(0, torch.long)
        n_ref = max([len(m["track_ref"]) for ms in prog_rows for m in ms] + [1])
        self.refs = torch.full(shape + (n_ref,), -1, dtype=torch.long, device=device)
        for b, members in enumerate(prog_rows):
            for m in members:
                j = m["slot"]
                self.prog[b, j] = True
                f["start"][b, j], f["end"][b, j] = m["start_s"], m["start_s"] + m["hold_s"]
                f["release_x"][b, j] = m["release_x"]
                f["fallback_x"][b, j] = (self.drop_x - 15.0 if m["fallback_x"] is None
                                         else float(m["fallback_x"]))
                f["lc_from"][b, j], f["lc_from_x"][b, j] = m["lc_from"], m["lc_from_x"]
                f["giveup_x"][b, j] = (self.drop_x - 300.0 if m["lc_giveup_x"] is None
                                       else float(m["lc_giveup_x"]))
                f["tau"][b, j] = math.nan if m["tau"] is None else float(m["tau"])
                for key in ("tau_from", "sp_from", "amax", "dmax", "v_w", "track_off", "kp",
                            "kv", "settle_gap"):
                    f[key][b, j] = m[key]
                if m["partner"] is not None:
                    self.partner[b, j] = m["partner"]
                self.merger[b, j] = m["merger"]
                if m["lc_target"] is not None:
                    if m["lc_target"] not in up:
                        raise WMContractError(f"lc_target {m['lc_target']} is not an approach lane")
                    self.tl[b, j] = up[m["lc_target"]]
                self.kind[b, j] = m["speed"]
                for r, slot in enumerate(m["track_ref"]):
                    self.refs[b, j, r] = slot
        z = full(False, torch.bool)
        self.p_released, self.was_lane0, self.lc_done = z.clone(), z.clone(), z.clone()
        self.settled, self.req, self.tau_on = z.clone(), z.clone(), z.clone()
        self.cmd = full(math.nan)
        self.events: list = []

    # ------------------------------------------------------------------ decisions (1 s tick)
    def _clip(self, a):
        return torch.maximum(torch.minimum(a, self.f["amax"]), -self.f["dmax"])

    def _decide(self, t: float, x, v, lane, present) -> None:
        f = self.f
        live = self.prog & ~self.p_released
        started = live & (t >= f["start"])
        on_up = x < self.drop_x
        rel = (live & ~present) | (started & ((t >= f["end"]) | (x >= f["release_x"])))
        pidx = self.partner.clamp_min(0)
        p_on0 = (torch.gather(present, 1, pidx) & (torch.gather(lane, 1, pidx) == 2)
                 & (torch.gather(x, 1, pidx) < self.drop_x))
        rel = rel | (started & (self.partner != -1) & ((self.partner == -2) | ~p_on0))
        act = started & ~rel
        on0 = on_up & (lane == 2)
        self.was_lane0 = self.was_lane0 | (act & self.merger & on0)
        rel = rel | (act & self.merger & ((~on0 & self.was_lane0) | (x >= f["fallback_x"])))
        self.p_released = self.p_released | rel
        act = act & ~rel
        self.tau_on = act & ~torch.isnan(f["tau"]) & (t >= f["tau_from"])
        lc_on = act & (self.tl >= 0) & (t >= f["lc_from"]) & (x >= f["lc_from_x"])
        self.lc_done = self.lc_done | (lc_on & on_up & (lane == self.tl))
        self.req = lc_on & on_up & (x < f["giveup_x"])
        lc_ok = (self.tl < 0) | self.lc_done
        cmd = torch.full_like(v, math.nan)
        speed_on = act & (self.kind > 0) & (t >= f["sp_from"])
        still = torch.zeros_like(speed_on)
        tracking = speed_on & (self.kind == 2) & ~self.settled
        if bool(tracking.any()):
            rr = self.refs.clamp_min(0)
            b, k, r = rr.shape
            flat = rr.reshape(b, -1)
            ok = (self.refs >= 0) & torch.gather(present, 1, flat).reshape(b, k, r)
            n_ok = ok.sum(-1)
            xr = (torch.gather(x, 1, flat).reshape(b, k, r) * ok).sum(-1) / n_ok.clamp_min(1)
            vr = (torch.gather(v, 1, flat).reshape(b, k, r) * ok).sum(-1) / n_ok.clamp_min(1)
            err = xr + f["track_off"] - x
            refs_gone = tracking & (n_ok == 0)
            settle = (tracking & ~refs_gone & ~torch.isnan(f["settle_gap"]) & lc_ok
                      & (err.abs() <= torch.nan_to_num(f["settle_gap"], nan=0.0)))
            self.settled = self.settled | refs_gone | settle
            still = tracking & ~refs_gone & ~settle
            cmd = torch.where(still, self._clip(f["kp"] * err + f["kv"] * (vr - v)), cmd)
        match = speed_on & (self.kind == 3) & ~lc_ok & on_up
        if bool(match.any()):
            same = (lane.unsqueeze(1) == self.tl.unsqueeze(2)) & present.unsqueeze(1)
            near = (((x.unsqueeze(1) - x.unsqueeze(2)).abs() <= MATCH_RANGE_M)
                    & (x.unsqueeze(1) < self.drop_x))
            eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device).unsqueeze(0)
            nb = same & near & ~eye
            n = nb.sum(-1)
            vt = (v.unsqueeze(1) * nb).sum(-1) / n.clamp_min(1)
            cmd = torch.where(match & (n > 0), self._clip((vt - v) / 1.0), cmd)
        cmd = torch.where(speed_on & (self.kind == 4), f["amax"].expand_as(v), cmd)
        to_vw = speed_on & ~still & ((self.kind == 1) | ((self.kind == 2) & self.settled))
        cmd = torch.where(to_vw & ~torch.isnan(f["v_w"]), self._clip((f["v_w"] - v) / 1.0), cmd)
        self.cmd = torch.where(act, cmd, torch.full_like(cmd, math.nan))

    def _tau_native(self, kin, present, native):
        from wms.physics.neighbours import lead_neighbour

        a_max, b_comf, _t, s0, delta = self.physics.idm()
        v0 = float(self.physics.v0_mps)
        gap, lead_v = lead_neighbour(kin, present, self.physics)
        v = kin[..., 4]
        tau = torch.nan_to_num(self.f["tau"], nan=1.0)
        s_star = s0 + (v * tau + v * (v - lead_v) / (2.0 * math.sqrt(a_max * b_comf))).clamp_min(0.0)
        idm = a_max * (1.0 - (v / v0).clamp_min(0.0) ** delta - (s_star / gap.clamp_min(0.1)) ** 2)
        return torch.where(self.tau_on, idm.to(native.dtype), native)

    # ------------------------------------------------------------------ per physics step
    def __call__(self, state: dict) -> torch.Tensor:
        from wms.model import cav as cav_motion
        from wms.physics.mobil import speed_gap_safe
        from wms.physics.state import state_kin

        teacher = super().__call__(state)
        hold_merge = state.get(cav_motion.WM_MERGE_KEY)
        s, v, lane = state["s_m"], state["v_mps"], state["lane"].long()
        present = state["present"].bool()
        step_index = int(state.get("physics_step_index", 0))
        x = s + self.anchor_x
        if step_index % 2 == 0:                       # SUMO control tick (whole seconds)
            self._decide(float(step_index) * self.dt, x, v, lane, present)
        kin = state_kin(state)
        native = state.get("merge_pivot_native_accel")
        native = torch.zeros_like(v) if native is None else native.to(v.dtype)
        if bool(self.tau_on.any()):
            native = self._tau_native(kin, present, native)
        commanded = self.prog & ~torch.isnan(self.cmd)
        v_cmd = (v + torch.nan_to_num(self.cmd) * self.dt).clamp(0.0, self.speed_limit)
        v_nat = (v + native * self.dt).clamp_min(0.0)
        v_next = torch.where(commanded, v_cmd, v_nat)
        accel = (v_next - v) / self.dt
        s_next = s + 0.5 * (v + v_next) * self.dt
        want = self.prog & self.req & (self.tl >= 0) & (lane != self.tl)
        new_lane = lane
        if bool(want.any()):
            target = lane + torch.sign(self.tl - lane)
            closing = int(self.road.closing_lane)
            legal = (target >= 0) & (target <= 2) & ((target != closing) | (
                (x < self.drop_x) & bool(self.road.into_closing_allowed)))
            safe = speed_gap_safe(kin, present, target.clamp(0, 2), self.physics)
            go = want & legal & safe
            new_lane = torch.where(go, target, lane)
            if bool(go.any()):
                self.events.append((step_index, int(go.sum())))
        prog_t = torch.stack((s_next, new_lane.to(v.dtype), v_next, accel), dim=-1)
        out = torch.where(self.prog.unsqueeze(-1), prog_t, teacher)
        delegate = self.prog & (lane == 2) & ~want
        state[cav_motion.WM_MERGE_KEY] = delegate if hold_merge is None else hold_merge | delegate
        return out
