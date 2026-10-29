"""S-ZEROSTART: zero the native-residual head's OUTPUT layer after the C1 actor is loaded.

Registered 2026-10-30 00:35 (owner approved). The stopped candidate-A phase 1
(candA_prod_184542ab_p1) started from C1, whose residual is positive everywhere
(+0.01..+0.07 m/s^2); under the always-commanded mode-2 law the SUMO safe-speed cap
removes every positive residual (``min(issued, cap)``), so the executed action and the
gradient through it were exactly unchanged (independently rescored J bitwise constant).
The zero start initialises the residual output layer (weight AND bias) to 0 so the initial
policy issues residual 0 = the CTRL0 execution exactly (``alwayscmd.commands`` returns the
unclipped base law at residual == 0); trunk/encoders/group encoder are kept as loaded.

The head's Adam moments (retained from C1's training by the legacy migration) are cleared:
stale momentum along C1's direction would push the zeroed weights straight back to positive
residuals (the dead zone).  Every other optimiser state (fresh trunk moments) is unchanged.
"""
from __future__ import annotations

import torch

SCHEMA = "zero_residual_output/1"


def zero_residual_output(trainer) -> dict:
    """Zero ``policy.residual_head`` (weight and bias) and reset its Adam state."""
    policy = trainer.policy
    if not getattr(policy, "residual_on_native", False) or not hasattr(policy, "residual_head"):
        raise ValueError("zero_residual_output requires a native-residual actor")
    head = policy.residual_head
    receipt = {"schema": SCHEMA,
               "before": {"weight_abs_max": float(head.weight.detach().abs().max()),
                          "bias": [float(x) for x in head.bias.detach().flatten()]}}
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    cleared = 0
    optimizer = getattr(trainer, "actor_optimizer", None)
    if optimizer is not None:
        for parameter in head.parameters():
            if parameter in optimizer.state:
                del optimizer.state[parameter]
                cleared += 1
    receipt["adam_state_cleared_tensors"] = cleared
    receipt["after"] = {"weight_abs_max": float(head.weight.detach().abs().max()),
                        "bias": [float(x) for x in head.bias.detach().flatten()]}
    return receipt
