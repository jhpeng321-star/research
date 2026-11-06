"""Backend-agnostic executor of the labelled move rules (nm2_core.Arm / ck_core.Arm semantics).

The SUMO probes command CAV members once per whole control second ``el = t - t0``; this module
re-implements their state machine on a backend-neutral vehicle view so the same move can be run
inside a world model (closed loop) or the keep-speed baseline.  It emits commands only; the
backend executes them:

* ``("accel", a)``      SUMO slowDown(clip(v + a*1 s, 0, allowed), 1 s): constant a for 1 s
* ``("lane", wm_lane)`` SUMO changeLane request (2 s) towards a WM lane (executed when safe)
* ``("tau", tau)``      native car following with time headway tau (CACC-like roles)
* ``("lcmode", m)``     513 = no voluntary lane changes except strategic, 512 = TraCI-only
* ``("release", why)``  back to native (restore tau / lane-change mode, setSpeed(-1))

Views are in the WM frame ``{id: (s, wm_lane, v)}``; rules that test SUMO geometry (x, edge 'up',
SUMO lane 0 = closing lane, lane index order) are evaluated on the inverse projection.
HDVs are never commanded: only the listed member ids (CAVs) receive commands.
"""
from __future__ import annotations

from typing import Mapping, Optional

from . import frame as F

ROLES = ("hold", "pulse", "tau", "merge_early", "zip_merge", "zip_yield", "lane_to")
LC_MODE_POLICY = 513
LC_MODE_TRACI_ONLY = 512
_WM_OF_UP = {0: 2, 1: 1, 2: 0}
_UP_OF_WM = {v: k for k, v in _WM_OF_UP.items()}
_WZ_OF_WM = {1: 0, 0: 1}


def sumo_view(s: float, wm_lane: int):
    """(edge, sumo_lane, x) of a WM-frame position (junction folded into 'wz')."""
    x = F.s_to_x(s)
    if x < F.X_B:
        return "up", _UP_OF_WM.get(int(wm_lane), 1), x
    return "wz", _WZ_OF_WM.get(int(wm_lane), 0), x


class MoveExecutor:
    """One arm (group, solo or dither set) of a labelled move."""

    def __init__(self, members: list, *, t0: float = 0.0):
        self.members = []
        for m in members:
            role = m.get("role") or "hold"
            if role not in ROLES:
                raise ValueError(f"unknown move role {role!r}")
            self.members.append(dict(id=str(m["id"]), role=role, params=dict(m.get("params") or {}),
                                     released=False, release_el=None, lc_on=False, tau_on=False))
        self.t0 = float(t0)
        self.events: list = []

    @property
    def ids(self) -> list:
        return [m["id"] for m in self.members]

    def roles(self) -> set:
        return {m["role"] for m in self.members}

    def _release(self, m, el, why, out):
        if m["released"]:
            return
        m["released"] = True
        m["release_el"] = el
        out.setdefault(m["id"], []).append(("release", why))
        self.events.append((el, m["id"], "release", why))

    def control(self, el: int, view: Mapping[str, tuple]) -> dict:
        """Commands for control second ``el`` given ``view = {id: (s, wm_lane, v)}`` of live vehicles."""
        out: dict = {}
        for m in self.members:
            vid = m["id"]
            if m["released"]:
                continue
            if vid not in view:
                m["released"] = True
                m["release_el"] = el
                self.events.append((el, vid, "left", ""))
                continue
            p = m["params"]
            st = int(p.get("start_s", 0))
            if el < st:
                continue
            s, wl, v = view[vid]
            edge, lane, x = sumo_view(s, wl)
            role = m["role"]
            if role == "pulse":
                if el >= st + int(p["dur_s"]):
                    self._release(m, el, "pulse_end", out)
                    continue
                out.setdefault(vid, []).append(("accel", float(p["a"])))
                continue
            if el >= st + int(p.get("hold_s", 10 ** 6)):
                self._release(m, el, "hold_end", out)
                continue
            if x >= float(p.get("release_x", 10 ** 9)):
                self._release(m, el, "release_x", out)
                continue
            if role == "hold":
                anc = p.get("anchor")
                if anc is not None:
                    a = view.get(anc)
                    if a is None or sumo_view(*a[:2])[0] != "up" or \
                            sumo_view(*a[:2])[2] >= float(p["anchor_release_x"]):
                        self._release(m, el, "anchor", out)
                        continue
                if not m["lc_on"]:
                    out.setdefault(vid, []).append(("lcmode", LC_MODE_POLICY))
                    m["lc_on"] = True
                a = max(-float(p["dmax"]), min(float(p["amax"]), (float(p["v_w"]) - v) / 1.0))
                out.setdefault(vid, []).append(("accel", a))
            elif role == "tau":
                zone = p.get("zone")
                if zone is not None:
                    if x < zone[0]:
                        continue
                    if x > zone[1]:
                        self._release(m, el, "zone_end", out)
                        continue
                if not m["tau_on"]:
                    out.setdefault(vid, []).append(("lcmode", int(p.get("lc_mode", LC_MODE_POLICY))))
                    out[vid].append(("tau", float(p["tau"])))
                    m["tau_on"] = True
            elif role == "merge_early":
                if not m["lc_on"]:
                    out.setdefault(vid, []).append(("lcmode", LC_MODE_POLICY))
                    m["lc_on"] = True
                if edge == "up" and lane == 0 and x >= float(p["trigger_x"]):
                    out.setdefault(vid, []).append(("lane", _WM_OF_UP[int(p.get("target_lane", 1))]))
            elif role == "zip_merge":
                if edge != "up" or lane != 0:
                    self._release(m, el, "merged", out)
                    continue
                if x >= float(p["fallback_x"]):
                    self._release(m, el, "fallback", out)
                    continue
                if not m["lc_on"]:
                    out.setdefault(vid, []).append(("lcmode", LC_MODE_TRACI_ONLY))
                    m["lc_on"] = True
                if x >= float(p["merge_x"]):
                    out.setdefault(vid, []).append(("lane", _WM_OF_UP[1]))
            elif role == "zip_yield":
                partner = view.get(p["partner"])
                pe = sumo_view(*partner[:2]) if partner is not None else None
                if partner is None or pe[0] != "up" or pe[1] != 0:
                    self._release(m, el, "partner_merged", out)
                    continue
                if not m["lc_on"]:
                    out.setdefault(vid, []).append(("lcmode", LC_MODE_POLICY))
                    m["lc_on"] = True
                err = (pe[2] - float(p["gap"])) - x
                a = float(p["kp"]) * err + float(p["kv"]) * (partner[2] - v)
                a = max(-float(p["dmax"]), min(float(p["amax"]), a))
                out.setdefault(vid, []).append(("accel", a))
            elif role == "lane_to":
                if not m["lc_on"]:
                    out.setdefault(vid, []).append(("lcmode", LC_MODE_POLICY))
                    m["lc_on"] = True
                tl = int(p["target_lane"])
                if edge == "up" and lane < tl and x < float(p["giveup_x"]):
                    out.setdefault(vid, []).append(("lane", _WM_OF_UP[tl]))
        return out


def arms_for(members: list, *, with_solos: bool = True, dither: Optional[list] = None) -> dict:
    """Arm member lists: ``base`` (none), ``group`` (all), ``solo:<id>`` (each member alone,
    same params - nm2_plan.solo_jobs).  ``dither`` members (role pulse) are added to EVERY arm."""
    d = list(dither or [])
    arms = {"base": list(d), "group": list(members) + d}
    if with_solos and len(members) > 1:
        for m in members:
            arms[f"solo:{m['id']}"] = [m] + d
    return arms
