"""Lifecycle adapter for the existing C4 wall baseline; never writes to TraCI."""
from __future__ import annotations

from copy import copy
from math import isfinite

import numpy as np

from .baselines import WallLaw


class SwitchableWallController:
    """An episode-local library controller with ``command(env)`` interface.

    Formation, pacing, cooldown and nonmember damping are exactly the existing
    C4 ``WallLaw`` baseline (v2.2). Start needs no warmup or historical roster.
    A selection-boundary stop is an external lifecycle interruption, not a new
    safety abort or a change to the frozen session's HOLD-only release rule.
    """

    name = "wall"

    def __init__(self):
        self.law = None
        self.running = True
        self.started_at_s = None
        self.last_time_s = None
        self.events = []

    def start(self, *, now_s: float) -> None:
        self._check_time(now_s)
        if self.started_at_s is not None and self.running:
            return
        self.law = None
        self.running = True
        self.started_at_s = float(now_s)
        self.last_time_s = None
        self.events.append(dict(event="controller_start", timestamp_s=float(now_s)))

    def _check_time(self, now_s):
        if not isfinite(now_s) or (self.last_time_s is not None and now_s < self.last_time_s):
            raise ValueError("wall clock must be finite and nondecreasing")

    def stop(self, *, now_s: float) -> None:
        """Discard all wall ownership/memory; the owner performs the handback."""
        self._check_time(now_s)
        if not self.running:
            return
        phase = None if self.law is None or self.law.executor is None else int(
            self.law.executor.state.phase[0])
        self.events.append(dict(event="controller_stop", timestamp_s=float(now_s),
                                interrupted_phase=phase))
        self.law = None
        self.running = False
        self.last_time_s = float(now_s)

    def commands(self, frame, builder, *, now_s: float, live_cavs) -> dict:
        """Pure requests from current telemetry; no command or safety-layer writes."""
        self._check_time(now_s)
        if not self.running:
            return {}
        if self.started_at_s is None:
            self.start(now_s=now_s)
        if self.last_time_s is not None and now_s == self.last_time_s:
            raise ValueError("wall already evaluated at this decision")
        self.last_time_s = float(now_s)
        if self.law is None:
            self.law = WallLaw(len(frame.s))
        elif self.law.k != len(frame.s):
            raise ValueError("wall slot layout changed; start a fresh controller")
        # Do not let a changed CAV role, missing live id, or recycled slot keep
        # an old wall alive. Nonmembers remain available for rear surveillance.
        sensed = copy(frame)
        sensed.present = np.asarray(frame.present, bool).copy()
        sensed.is_cav = np.asarray(frame.is_cav, bool).copy()
        for slot in np.flatnonzero(sensed.present):
            vid = builder.vehicle_ids[int(frame.vid_index[slot])]
            sensed.is_cav[slot] &= vid in live_cavs
        for slot in self.law.members:
            sensed.present[slot] &= sensed.is_cav[slot]
        before = len(self.law.events)
        commands = self.law.commands(sensed, builder, now_s=now_s, live_cavs=live_cavs)
        self.events.extend(dict(e) for e in self.law.events[before:])
        return commands

    def command(self, env):
        """Return direct slot requests and the frozen nonmember damping fallback.

        ``env.step`` owns sensor validation, direct_target, safety mode and all
        writes through CommandOwnership. No lane request is ever returned.
        """
        if not self.running:
            return None, dict(fallback="native")
        live = set(env.api.vehicle.getIDList()) & set(env.cav_ids)
        commands = self.commands(env.frame, env.builder, now_s=env.now, live_cavs=live)
        accel = np.full((1, len(env.frame.s)), np.nan, dtype=np.float64)
        for slot in np.flatnonzero(env.frame.present & env.frame.is_cav):
            vid = env.builder.vehicle_ids[int(env.frame.vid_index[slot])]
            if vid in commands:
                accel[0, slot] = commands[vid]
        return accel, dict(fallback="damping")
