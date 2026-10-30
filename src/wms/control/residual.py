"""Shared residual-to-executed acceleration algebra (no simulator dependency)."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch


class _ExactZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centre, executed, residual):
        return torch.where(residual == 0., centre, executed)

    @staticmethod
    def backward(ctx, gradient):
        # Retain the physical law's derivative, including its state derivatives.
        return None, gradient, None


def executed_from_residual(state, r, law):
    """Return executed acceleration, differentiable for tensor inputs.

    ``law`` is ``"base"`` (legacy/default), ``"ctrl0"``, or a mapping with
    ``residual_reference``, ``lo`` (-3), ``hi`` (1.2), ``dt`` (.5), and
    ``accel_max`` (1.2). ``state`` contains ``v_mps``, ``base_mps2`` and
    ``cap_mps2`` (scalars or tensors). A WM frame may instead omit base/cap;
    they are then inferred with the existing exact always-commanded geometry.
    Scalar inputs return a Python float evaluated in float64.

    ctrl0 = max(-v/dt, min(base, cap)); L'=min(lo, ctrl0),
    U'=max(min(hi, accel_max), ctrl0). The ctrl0 law clips ctrl0+r to
    [L', U'], then applies the unchanged cap and nonnegative-speed floor.
    Zero is bitwise CTRL0, without a zero-release branch that kills gradients.
    Nonfinite residual/base/cap/speed inputs are rejected in the opt-in law;
    infinite *leader gaps* remain supported by the nanfix geometry routines.
    """
    options = dict(law) if isinstance(law, Mapping) else {"residual_reference": law}
    reference = options.get("residual_reference", "base")
    if reference not in ("base", "ctrl0"):
        raise ValueError("residual_reference must be 'base' or 'ctrl0'")
    lo, hi, dt, accel_max = (float(options.get(k, default)) for k, default in
                            (("lo", -3.), ("hi", 1.2), ("dt", .5), ("accel_max", 1.2)))
    if not all(math.isfinite(x) for x in (lo, hi, dt, accel_max)) or dt <= 0 or lo > hi:
        raise ValueError("finite ordered residual bounds and positive dt required")
    speed = state["v_mps"]
    if "base_mps2" in state and "cap_mps2" in state:
        base, cap = state["base_mps2"], state["cap_mps2"]
    else:
        from .alwayscmd import inferred_base, inferred_cap
        base, cap = inferred_base(state, dt=dt), inferred_cap(state, dt=dt)
    tensor = next((x for x in (base, r, speed, cap) if torch.is_tensor(x)), None)
    scalar = tensor is None
    convert = (lambda x: torch.as_tensor(x, dtype=torch.float64)) if scalar else (
        lambda x: torch.as_tensor(x, dtype=tensor.dtype, device=tensor.device))
    base, cap, speed, r = (convert(x) for x in (base, cap, speed, r))
    if reference == "ctrl0":
        if not all(bool(torch.isfinite(x).all()) for x in (base, cap, speed, r)):
            raise ValueError("finite cap-relative execution inputs required")
        centre = torch.maximum(torch.minimum(base, cap), -speed / dt)
    else:
        centre = base
    lower = torch.minimum(centre, torch.full_like(centre, lo))
    upper = torch.maximum(centre, torch.full_like(centre, min(hi, accel_max)))
    issued = torch.minimum(torch.maximum(centre + r, lower), upper)
    out = torch.maximum(torch.minimum(issued, cap), -speed / dt)
    if reference == "ctrl0":
        out = _ExactZero.apply(centre, out, r)
    return float(out) if scalar else out
