"""S-SUMOES-RUN 'ours' arm: candidate-B cap-relative residual on the frozen E3 C1 executor.

Only two runtime attributes are scoped-overridden, exactly like the CTRL0 adapter
(tools/ctrlarm/zero.py 24095a10): ``policy.load_policy_fn`` (loads the unchanged C1
checkpoint, gate verified, then sets the 137 residual-head scalars to theta) and
``nativeinit.sumo_residual_commands`` (cap-relative composition, never released).

Composition per designated CAV per physics tick (no zero-release branch):
    native = sumo_native_accel(api, vid, dt)                      # E3 CTRL0 command
    r == 0 -> native (bitwise the CTRL0 forced-native command)
    else   -> clip(CTRL0_executed + r, min(lo, CTRL0_executed),
                   max(min(hi, getAccel), CTRL0_executed)), then mode-27 cap
MAIN1440 correction: the old helper added to native base despite its caprel name.
SUMO speedMode27 remains the final safety executor (unchanged runtime).
"""
from contextlib import contextmanager

import numpy as np

HEAD_IN = 136


def caprel_commands(api, residuals, lo, hi, *, dt=0.5):
    from wms.control.nativeinit import sumo_native_accel
    from tools.sumoes.caprel import scalar_executed, live_cap

    composed = {}
    for vid, residual in residuals.items():
        residual = float(residual)
        if not np.isfinite(residual):
            raise ValueError("nonfinite residual for a designated CAV")
        native = float(sumo_native_accel(api, vid, dt))
        if residual == 0.0:
            composed[vid] = native
            continue
        cap = live_cap(api, vid, dt)
        speed = float(api.vehicle.getSpeed(vid))
        value = scalar_executed(native, cap, speed, residual, lo, hi, dt,
                                float(api.vehicle.getAccel(vid)))
        ctrl0 = max(-speed / dt, min(native, cap))
        # Preserve the original CTRL0 write if its realised command is unchanged.
        composed[vid] = native if value == ctrl0 else value
    return composed


class HeadSensitivity:
    """Forward hook on residual_head: accumulates the ESLONG head sensitivity
    radius*sech^2(z)*sqrt(||f||^2+1) over ALL evaluated rows (padded slots included;
    descriptive, used only to calibrate sigma) and writes head_stats.json."""

    def __init__(self, radius=1.2):
        self.radius, self.sum_sq, self.rows, self.calls = float(radius), 0.0, 0, 0

    def __call__(self, module, args, output):
        import torch

        with torch.no_grad():
            f = args[0].detach().double().reshape(-1, args[0].shape[-1])
            z = output.detach().double().reshape(-1)
            sens = self.radius * (1 - torch.tanh(z).square()) * (f.square().sum(-1) + 1).sqrt()
            self.sum_sq += float(sens.square().sum())
            self.rows += int(sens.numel())
            self.calls += 1

    def summary(self):
        rms = (self.sum_sq / self.rows) ** 0.5 if self.rows else None
        return dict(rows=self.rows, calls=self.calls, sensitivity_rms=rms,
                    definition="radius*sech^2(z)*sqrt(||f||^2+1), all rows")


def set_head(executor, theta):
    import torch

    if not executor.residual_on_native:
        raise ValueError("ours requires the native-residual C1 executor")
    head = executor.policy.residual_head
    if tuple(head.weight.shape) != (1, HEAD_IN) or tuple(head.bias.shape) != (1,):
        raise ValueError("residual head must be Linear(136, 1)")
    theta = torch.as_tensor(theta, dtype=torch.float64).flatten()
    if theta.numel() != HEAD_IN + 1 or not bool(torch.isfinite(theta).all()):
        raise ValueError("theta must be 137 finite scalars")
    with torch.no_grad():
        head.weight.copy_(theta[:HEAD_IN].reshape(1, HEAD_IN).to(head.weight))
        head.bias.copy_(theta[HEAD_IN:].to(head.bias))
    return executor


class TypedResidualGuard:
    """Pass-through executor that only asserts the C1 typed-residual / native-lane contract."""

    def __init__(self, executor):
        self.executor = executor

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def __call__(self, state, *, with_lanes=False, dwell_age=None):
        from wms.control.nativeinit import NativeResidualCommand

        output = self.executor(state, with_lanes=with_lanes, dwell_age=dwell_age)
        command, lanes = output if with_lanes else (output, None)
        if not isinstance(command, NativeResidualCommand) or lanes is not None:
            raise ValueError("ours requires C1 typed residuals and native lanes")
        return output


@contextmanager
def ours_loader(checkpoint, theta, stats=None):
    from wms.control import nativeinit, policy

    original = policy.load_policy_fn
    original_commands = nativeinit.sumo_residual_commands

    def load(path, **kwargs):
        if str(path) != str(checkpoint):
            raise ValueError("ours loader received a different checkpoint")
        executor = set_head(original(path, **kwargs), theta)
        if stats is not None:
            executor.policy.residual_head.register_forward_hook(stats)
        return TypedResidualGuard(executor)

    policy.load_policy_fn = load
    nativeinit.sumo_residual_commands = caprel_commands
    try:
        yield
    finally:
        policy.load_policy_fn = original
        nativeinit.sumo_residual_commands = original_commands


def run_ours(job, inputs, out, *, measurement, theta):
    """Unchanged E3-final runner/observer/CLI path with the 'policy' controller."""
    from traffic_shepherd.c4.evalprep import run_e3 as original

    import json

    if job["arm"] != "ours":
        raise ValueError("run_ours only runs the ours arm")
    stats = HeadSensitivity()
    with ours_loader(inputs["policy"], theta, stats):
        result = original(job, inputs, out, measurement=measurement)
    result["head_stats"] = stats.summary()
    (out / "head_stats.json").write_text(json.dumps(result["head_stats"], indent=1) + "\n")
    return result
