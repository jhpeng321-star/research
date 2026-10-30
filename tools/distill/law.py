"""Execution adapters; the incumbent is always the existing shared physics law."""
from __future__ import annotations

import torch

from wms.control.alwayscmd import executed, inferred_cap

RADIUS = 1.2


def caprelative_available():
    try:
        from wms.control.caprel import executed_from_residual
    except ImportError:
        return False
    return callable(executed_from_residual)


def execute(state, residual, law="current"):
    if law == "current":
        return executed(state, residual, -3., RADIUS)
    if law == "cap_relative":
        # S-CAPREL integration point. No local approximation of its execution law.
        try:
            from wms.control.caprel import executed_from_residual
        except ImportError as exc:
            raise RuntimeError("S-CAPREL executed_from_residual is not integrated") from exc
        return executed_from_residual(state, residual, -3., RADIUS)
    raise ValueError(f"unknown execution law: {law}")


def absolute_executed(state, issued):
    """MAPPO issues an absolute command; speedMode27 is the final physics cap."""
    return torch.maximum(torch.minimum(issued, inferred_cap(state)), -state["v_mps"] / .5)


def representation(state, teacher, law="current"):
    zero = torch.zeros_like(teacher)
    base = execute(state, zero, law)
    low = execute(state, zero - RADIUS, law)
    high = execute(state, zero + RADIUS, law)
    projection = torch.minimum(torch.maximum(teacher, low), high)
    deviation = teacher - base
    # Continuous monotone law: its endpoint image is its entire reachable interval.
    expressible = (teacher >= low) & (teacher <= high)
    magnitude = deviation.abs()
    reproduced = (magnitude - (teacher - projection).abs()).clamp_min(0.)
    return dict(ctrl0=base, lower=low, upper=high, projected=projection,
                deviation=deviation, expressible=expressible,
                deviation_magnitude=magnitude, reproduced_magnitude=reproduced)
