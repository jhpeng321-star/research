"""Shared exact-clock stepping contract for pilot mechanics and training.

One clear owner of simulator advancement: every physics sample is produced
by exactly one ``ExactClock.advance()`` call, which issues exactly one bare
``traci.simulationStep()`` (documented server contract: omitted/zero argument
performs exactly one step of the configured --step-length) and then requires
the observed time to equal the expected next grid time within a tight
floating-point tolerance (1e-9 s). There is no overshoot tolerance, no
retry, and no skipped-sample compensation: any repeated timestamp, overshoot,
underreach or out-of-grid read raises ``ClockViolationError`` carrying a
receipt with expected/observed times, the call index and caller context.

The 0.5 s physics grid and the 1 s control grid are both expressed on the
same clock: control ticks are the integer grid points, pilot decisions only
at their declared cadence. Metric collection is the caller's responsibility
but every collected sample must correspond to exactly one advance() call.
"""
from dataclasses import dataclass, field

TOLERANCE_S = 1e-9


class ClockViolationError(Exception):
    """Raised on any clock-contract violation; carries a diagnostic receipt."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("clock violation: " + json_receipt(receipt))


def json_receipt(receipt):
    import json
    return json.dumps(receipt, sort_keys=True)


@dataclass
class ExactClock:
    """Single owner of simulator advancement on a fixed grid."""

    step_fn: object                 # no-arg callable performing exactly one step
    time_fn: object                 # no-arg callable returning the sim time
    dt: float = 0.5
    start: float = 0.0
    context: str = "simulation"
    last: float = float("nan")
    calls: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.last = self.start
        self.calls = 0

    @property
    def expected(self):
        return self.last + self.dt

    def _receipt(self, observed, kind):
        return dict(kind=kind, context=self.context, call_index=self.calls,
                    expected=self.expected, observed=observed, last=self.last,
                    dt=self.dt, delta=observed - self.expected)

    def check_now(self, observed, context=None):
        """Validate an externally read time without advancing (control ticks)."""
        if abs(observed - self.last) > TOLERANCE_S:
            raise ClockViolationError(self._receipt(observed,
                "control_time_mismatch:" + str(context or "")))
        return observed

    def advance(self):
        """Perform exactly one step; require the expected next grid time."""
        self.step_fn()
        observed = float(self.time_fn())
        self.calls += 1
        if abs(observed - self.expected) > TOLERANCE_S:
            raise ClockViolationError(self._receipt(observed, "physics_step_mismatch"))
        self.last = observed
        self.history.append(observed)
        return observed

    def at_control_tick(self):
        """True when the current time is an integer-second control tick."""
        return abs(self.last - round(self.last)) <= TOLERANCE_S
