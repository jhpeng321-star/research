"""Per-step diagnostics of a rollout, recorded by wrapping ``model.step``.

The wrapper observes inputs and outputs only; it does not change the rollout.  One record
is kept per physics step (batch element 0): which slots are live learned HDVs, the physics
acceleration, the gated residual correction actually applied, and the final acceleration.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

STEP_KEYS = ("gate", "physics_accel_mps2", "accel_delta_mps2", "accel_mps2")


def _first(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()[0]


@contextmanager
def trace_steps(model: Any) -> Iterator[list[dict[str, np.ndarray]]]:
    """Record ``model.step`` calls inside the ``with`` block into the yielded list.

    Each record holds ``present`` and ``is_cav`` of the step's input state plus the arrays
    ``gate``, ``physics``, ``correction`` and ``accel`` (per slot) from the step output.
    """
    records: list[dict[str, np.ndarray]] = []
    original = model.step

    def step(self, state, **kwargs):
        out = original(state, **kwargs)
        present = state.get("present")
        is_cav = state.get("is_cav")
        slots = state["s_m"].shape[-1]
        records.append({
            "present": (_first(present) if present is not None
                        else np.ones(slots, dtype=bool)),
            "is_cav": (_first(is_cav) if is_cav is not None
                       else np.zeros(slots, dtype=bool)),
            "gate": _first(out["gate"]).astype(float).reshape(-1),
            "physics": _first(out["physics_accel_mps2"]).astype(float),
            "correction": _first(out["accel_delta_mps2"]).astype(float),
            "accel": _first(out["accel_mps2"]).astype(float),
        })
        return out

    model.step = types.MethodType(step, model)
    try:
        yield records
    finally:
        del model.step
