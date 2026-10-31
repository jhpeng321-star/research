"""Independent smoke gates for a closed-cohort, physics-only rollout.

No arrivals or recorded future masks are permitted by this gate. An exit is
accounted for only when the predicted front bumper passes the supplied road end.
An entrant-enabled panel needs the evaluator's separate admission/source ledger.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def forbid_learned_corrections(model):
    """Reject execution of parameterized modules except the motion encoder.

    The guard is temporary and does not edit weights, flags, or global RNG state.
    This catches forgotten learned lane/merge/noise branches even when their
    contribution is numerically small or zero at initialization.
    """
    handles = []
    calls = []

    def reject(name):
        def hook(module, inputs):
            raise AssertionError(f"WM-PHYS executed learned correction module: {name}")
        return hook

    def count_encoder(module, inputs):
        calls.append("driver_encoder")

    try:
        for name, child in model.named_children():
            if name == "driver_encoder":
                handles.append(child.register_forward_pre_hook(count_encoder))
            elif any(True for _ in child.parameters()):
                handles.append(child.register_forward_pre_hook(reject(name)))
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def audit_closed_cohort(initial_present, rollout, *, exit_s_m=None):
    """Assert finite dynamics and exact slot conservation, return a small receipt."""
    masks = rollout["masks"].bool()
    if masks.ndim != 3 or masks.shape[1] == 0:
        raise AssertionError("smoke must contain [batch, nonempty time, slots] masks")
    if tuple(initial_present.shape) != (masks.shape[0], masks.shape[2]):
        raise AssertionError("initial and rollout slot shapes differ")
    for name in ("positions", "speeds", "accel", "lanes"):
        value = rollout[name]
        if value.shape != masks.shape or not torch.isfinite(value).all():
            raise AssertionError(f"invalid shape or nonfinite rollout {name}")
    if (rollout["speeds"][masks] < 0).any():
        raise AssertionError("negative speed for present vehicle")
    lanes = rollout["lanes"][masks]
    if ((lanes != 0) & (lanes != 1) & (lanes != 2)).any():
        raise AssertionError("invalid present vehicle lane")
    previous = torch.cat((initial_present.bool().unsqueeze(1), masks[:, :-1]), dim=1)
    arrivals = masks & ~previous
    if arrivals.any():
        raise AssertionError("closed-cohort rollout created or resurrected a vehicle")
    departed = previous & ~masks
    if exit_s_m is None:
        if departed.any():
            raise AssertionError("vehicle vanished without an exit boundary")
    else:
        boundary = torch.as_tensor(exit_s_m, device=masks.device,
                                   dtype=rollout["positions"].dtype)
        if boundary.ndim == 2:
            boundary = boundary.unsqueeze(1)
        if (departed & ~(rollout["positions"] > boundary)).any():
            raise AssertionError("vehicle vanished before the road exit")
    initial_n = initial_present.sum(dim=-1)
    exited_n = departed.sum(dim=(1, 2))
    final_n = masks[:, -1].sum(dim=-1)
    if not torch.equal(initial_n, final_n + exited_n):
        raise AssertionError("closed-cohort population not conserved")
    residual = rollout.get("raw_residual_mps2")
    if residual is not None and (not torch.isfinite(residual).all() or residual.count_nonzero()):
        raise AssertionError("WM-PHYS emitted a learned acceleration residual")
    return dict(finite=True, conserved=True, steps=int(masks.shape[1]),
                initial=initial_n.tolist(), exited=exited_n.tolist(), final=final_n.tolist())
