"""Execution adapters; the incumbent is always the existing shared physics law."""
from __future__ import annotations

import torch

from wms.control.alwayscmd import inferred_cap
from wms.control.residual import executed_from_residual

RADIUS = 1.2


def caprelative_available():
    return callable(executed_from_residual)


def execute(state, residual, law="current"):
    if law == "current":
        return executed_from_residual(state, residual, "base")
    if law == "cap_relative":
        return executed_from_residual(state, residual, "ctrl0")
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


@torch.no_grad()
def inverse(state, teacher, law="current"):
    """Numerical inverse for DATA-fitted initialisation, not a behavioural rule."""
    fit = representation(state, teacher, law)
    lower, upper = torch.full_like(teacher, -RADIUS), torch.full_like(teacher, RADIUS)
    for _ in range(torch.finfo(teacher.dtype).bits):
        middle = (lower + upper) / 2
        below = execute(state, middle, law) < fit["projected"]
        lower, upper = torch.where(below, middle, lower), torch.where(below, upper, middle)
    return torch.where(teacher == fit["ctrl0"], torch.zeros_like(teacher), (lower + upper) / 2)
