"""S-EXECPARITY 00:20 always-commanded adapter for the frozen E3-final runtime.

External wrapper only: no scientific runtime file is edited. The policy executor,
world metadata, seeds, deploy CLI and measurement observer are the frozen
E3-final ones. The sole change is the scoped command composer: every designated
CAV is commanded every tick (residual exactly zero included, i.e. the S-CTRLARM
CTRL0 write path) with the residual kept instead of forced to zero.

``law_commands`` is a verbatim, self-contained copy of this branch's unit-tested
``wms.control.alwayscmd.commands`` (the frozen runtime cannot import our module);
tests/unit/test_execparity_alwayscmd.py asserts the copy is bit-identical.
"""
from contextlib import contextmanager

import numpy as np

from wms.control.nativeinit import NativeResidualCommand, sumo_native_accel


def law_commands(api, residuals, lo, hi, *, dt=0.5):
    """Verbatim copy of wms.control.alwayscmd.commands (do not diverge)."""
    composed = {}
    for vid, residual in residuals.items():
        if not np.isfinite(residual):
            raise ValueError("nonfinite residual for a designated CAV")
        baseline = float(sumo_native_accel(api, vid, dt))
        if residual == 0.0:
            composed[vid] = baseline
        else:
            upper = min(float(hi), float(api.vehicle.getAccel(vid)))
            composed[vid] = float(np.clip(baseline + float(residual),
                                          min(float(lo), baseline),
                                          max(upper, baseline)))
    return composed


class AlwaysCommanded:
    """Run the original C1 executor; keep its typed residual (or force exact zero)."""

    def __init__(self, executor, *, force_zero):
        if not executor.residual_on_native:
            raise ValueError("always-commanded requires the native-residual C1 executor")
        self.executor = executor
        self.force_zero = force_zero

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def __call__(self, state, *, with_lanes=False, dwell_age=None):
        output = self.executor(state, with_lanes=with_lanes, dwell_age=dwell_age)
        command, lanes = output if with_lanes else (output, None)
        if not isinstance(command, NativeResidualCommand) or lanes is not None:
            raise ValueError("always-commanded requires C1 typed residuals and native lanes")
        residual = np.zeros_like(command.residual) if self.force_zero else command.residual
        typed = NativeResidualCommand(residual, command.accel_min, command.accel_max)
        return (typed, lanes) if with_lanes else typed


_LOG = None


def open_log(path):
    """Per-tick execution ledger: residual, base law, issued command, pre-step speed."""
    import gzip
    import os
    global _LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _LOG = gzip.open(path, "xt", encoding="utf-8", compresslevel=3)
    return _LOG


def close_log():
    global _LOG
    if _LOG is not None:
        _LOG.close()
        _LOG = None


def always_commanded_commands(api, residuals, lo, hi, *, dt=0.5):
    """Scoped composer replacing nativeinit.sumo_residual_commands for both writers."""
    import json
    composed = law_commands(api, residuals, lo, hi, dt=dt)
    if _LOG is not None:
        t = float(api.simulation.getTime())
        rows = []
        for vid, residual in residuals.items():
            leader = api.vehicle.getLeader(vid)
            rows.append({"t_s": t, "vehicle_id": vid,
                         "residual_mps2": float(residual),
                         "base_law_mps2": float(sumo_native_accel(api, vid, dt)),
                         "issued_mps2": composed[vid],
                         "speed_before_mps": float(api.vehicle.getSpeed(vid)),
                         "leader": None if leader is None else {
                             "gap_m": float(leader[1]),
                             "speed_mps": float(api.vehicle.getSpeed(leader[0]))}})
        _LOG.write("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        _LOG.flush()
    return composed


@contextmanager
def arm_loader(arm, checkpoint):
    """Scope the loader override to the always-commanded arms; other arms unchanged."""
    from wms.control import policy
    from wms.control import nativeinit
    if arm not in ("always0", "alwaysx"):
        yield
        return
    original = policy.load_policy_fn
    original_commands = nativeinit.sumo_residual_commands

    def load(path, **kwargs):
        if str(path) != str(checkpoint):
            raise ValueError("always-commanded loader received a different checkpoint")
        return AlwaysCommanded(original(path, **kwargs), force_zero=(arm == "always0"))

    policy.load_policy_fn = load
    nativeinit.sumo_residual_commands = always_commanded_commands
    try:
        yield
    finally:
        policy.load_policy_fn = original
        nativeinit.sumo_residual_commands = original_commands


def run_e3(job, inputs, out, *, measurement, log_path=None):
    """Call the unchanged E3-final runner, observer, CLI and executor path."""
    from traffic_shepherd.c4.evalprep import run_e3 as original

    if log_path is not None:
        open_log(str(log_path))
    try:
        with arm_loader(job["arm"], inputs["policy"]):
            return original(job, inputs, out, measurement=measurement)
    finally:
        close_log()
