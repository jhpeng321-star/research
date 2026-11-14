"""WM-N1 GPU-box bundle/queue package.

Piece ``BUNDLE``.  Owns the dedicated GPU box's sequential queue defined by
REGISTRATION_WM_N1_AMENDMENT_1.md A3:

    verify-upload -> convert + verify -> 1-h timing gate STOP ->
    APPROVAL file check -> member 1 -> members 2-5.

:mod:`bundle.timing` implements INTERFACES.md Mismatch 2: it measures the REAL
``model.net.WmN1Net`` member (not the evalh ``Toy``) while reusing the evalh
timing-gate thresholds/outputs.
"""
from __future__ import annotations

__all__ = ["approval", "timing", "queue", "cli"]

from . import approval, timing, queue, cli
