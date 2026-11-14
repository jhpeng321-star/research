"""Training diagnostics that are not losses: gradient-norm shares, driver-parameter stats.

``GradientShares`` offers the two legacy measurement modes:

* *probe* (forward-only, one probe chunk): each term's gradient-norm share is
  ``||g_i|| / sum_j ||g_j||`` where ``g_i`` is the term's gradient on that chunk;
* *accumulate* (during a real backward step): gradients of each term are summed
  over all chunks before the norms are taken (used for stateful closed-loop
  controllers that cannot be replayed).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from wms.model.driver_encoder import PARAMETER_NAMES


class GradientShares:
    """Per-term gradient-norm shares over a fixed parameter list."""

    def __init__(self, parameters: Sequence[Tensor]) -> None:
        self.parameters = list(parameters)
        self._summed: dict[str, list[Optional[Tensor]]] = {}
        self._probe_sq: dict[str, float] = {}

    def _grads(self, tensor: Tensor) -> list[Optional[Tensor]]:
        if not tensor.requires_grad:
            return [None] * len(self.parameters)
        return list(torch.autograd.grad(tensor, self.parameters, retain_graph=True,
                                        allow_unused=True))

    def capture(self, name: str, tensor: Tensor) -> None:
        """Add ``tensor``'s gradient to the running per-term sum (accumulate mode)."""
        summed = self._summed.setdefault(name, [None] * len(self.parameters))
        for j, grad in enumerate(self._grads(tensor)):
            if grad is not None:
                summed[j] = grad.detach().clone() if summed[j] is None else summed[j] + grad.detach()

    def accumulated(self) -> tuple[dict[str, float], dict[str, float]]:
        """``(shares, gradient norms)`` of the accumulate mode."""
        norms = {name: sum(float(g.square().sum()) for g in grads if g is not None) ** .5
                 for name, grads in self._summed.items()}
        total = max(sum(norms.values()), 1.e-12)
        return {name: value / total for name, value in norms.items()}, norms

    def probe(self, name: str, tensor: Tensor, *, keep_zero: bool = False) -> None:
        """Add the squared gradient norm of ``tensor`` (probe mode).

        Tensors without a graph are skipped unless ``keep_zero`` (then share 0).
        """
        if not tensor.requires_grad:
            if keep_zero:
                self._probe_sq.setdefault(name, 0.0)
            return
        self._probe_sq[name] = self._probe_sq.get(name, 0.0) + sum(
            float((g.detach() ** 2).sum()) for g in self._grads(tensor) if g is not None)

    def probed(self) -> dict[str, float]:
        """Probe-mode shares (all zero when every gradient vanishes)."""
        total = sum(value ** 0.5 for value in self._probe_sq.values())
        return {name: (value ** 0.5 / total if total > 0.0 else 0.0)
                for name, value in self._probe_sq.items()}


def driver_param_stats(values: Optional[Tensor], valid: Optional[Tensor]) -> Optional[dict]:
    """Per-driver-parameter min/median/max over valid HDV slots at t0 (one host transfer)."""
    if values is None or valid is None:
        return None
    valid = valid.to(values.device).bool()
    gathered = values[valid].reshape(-1, len(PARAMETER_NAMES))
    if gathered.numel() == 0:
        return None
    stats = torch.stack((gathered.min(dim=0).values, gathered.median(dim=0).values,
                         gathered.max(dim=0).values), dim=1)
    rows = stats.detach().cpu().tolist()
    return {name: {"min": row[0], "median": row[1], "max": row[2]}
            for name, row in zip(PARAMETER_NAMES, rows)}
