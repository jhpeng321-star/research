"""One-boundary fleet controller switch for the supported evaluator."""
from dataclasses import dataclass
from math import isfinite


MODES = frozenset(("native", "damping", "pi-external-u", "wall-v2"))


@dataclass(frozen=True)
class SingleSwitchSpec:
    prefix: str
    suffix: str
    switch_time_s: float

    def __post_init__(self):
        if self.prefix not in MODES or self.suffix not in MODES:
            raise ValueError("fleet switch modes must be native, damping, pi-external-u, or wall-v2")
        if (isinstance(self.switch_time_s, bool)
                or not isfinite(float(self.switch_time_s))
                or float(self.switch_time_s) <= 0
                or abs(float(self.switch_time_s) - round(float(self.switch_time_s))) > 1e-9):
            raise ValueError("fleet switch time must be a positive one-second-grid value")


class FleetSwitchSession:
    """Retain one owner and perform exactly one roster-wide mode transition."""

    def __init__(self, spec, *, circles_controller, pi_session, cav_ids,
                 wall_session=None):
        if not isinstance(spec, SingleSwitchSpec):
            raise TypeError("spec must be SingleSwitchSpec")
        self.spec = spec
        self.circles_controller = circles_controller
        self.pi_session = pi_session
        self.wall_session = wall_session
        self.cav_ids = frozenset(cav_ids)
        if spec.prefix == "pi-external-u" or spec.suffix == "pi-external-u":
            if pi_session is None:
                raise ValueError("PI switch mode requires a roster-bound PI session")
        if spec.prefix == "wall-v2" or spec.suffix == "wall-v2":
            if wall_session is None:
                raise ValueError("wall-v2 switch mode requires a roster-bound wall session")
        self._mode = None
        self._switched = False
        self._pi_exited = False
        self._owner = None
        self._stopped = False
        self._last_tick = None

    @property
    def mode(self):
        return self._mode

    def _mode_at(self, now_s):
        return self.spec.suffix if now_s >= self.spec.switch_time_s else self.spec.prefix

    def _enter(self, owner, mode, *, live_ids):
        if mode == "pi-external-u":
            if self._pi_exited:
                raise RuntimeError("PI mode cannot be re-entered after the switch boundary")
            owner.controller = self.pi_session.adapter
        else:
            if self._mode == "pi-external-u":
                self.pi_session.observe_lifecycle(live_ids=set(),
                                                   disappeared_ids=self.cav_ids,
                                                   id_reuse_ids=self.cav_ids)
                self._pi_exited = True
            owner.controller = self.circles_controller
        self._mode = mode

    def step(self, owner, frames, *, now_s, live_ids, disappeared_ids=(),
             reappeared_ids=()):
        if self._stopped:
            raise RuntimeError("fleet switch session is stopped")
        try:
            now_s = float(now_s)
            if not isfinite(now_s) or now_s < 0:
                raise ValueError("fleet switch time must be finite and nonnegative")
            if abs(now_s - round(now_s)) > 1e-9:
                raise ValueError("fleet switch step must lie on the one-second grid")
            if self._last_tick is not None and abs(now_s - self._last_tick - 1.) > 1e-9:
                raise ValueError("fleet switch requires every one-second control tick")
            if self._owner is None:
                if owner.cav_ids != self.cav_ids:
                    raise ValueError("fleet switch owner roster differs from registered CAVs")
                self._owner = owner
            elif owner is not self._owner:
                raise ValueError("fleet switch cannot replace its CommandOwnership instance")
            mode = self._mode_at(now_s)
            if self._mode is None:
                self._enter(owner, mode, live_ids=live_ids)
            elif mode != self._mode:
                if self._switched:
                    raise RuntimeError("fleet switch attempted more than once")
                self._switched = True
                self._enter(owner, mode, live_ids=live_ids)
            if mode == "pi-external-u":
                records = self.pi_session.step(owner, frames, now_s=now_s, live_ids=live_ids,
                                            native_ids=(), departed_ids=disappeared_ids,
                                            id_reuse_ids=reappeared_ids)
            elif mode == "wall-v2":
                # Wall ticks route through the roster-bound wall session; the
                # ownership base controller stays the single CIRCLES instance
                # (no controller swap in _enter), so wall release resumes the
                # identical damping instance by construction.
                records = self.wall_session.step(owner, frames, now_s=now_s,
                                                 live_ids=live_ids,
                                                 disappeared_ids=disappeared_ids,
                                                 reappeared_ids=reappeared_ids)
            else:
                native = set(live_ids) & self.cav_ids if mode == "native" else set()
                records = owner.step(frames, now_s=now_s, live_ids=live_ids, native_ids=native)
            self._last_tick = now_s
            return records
        except BaseException:
            self._stopped = True
            raise
