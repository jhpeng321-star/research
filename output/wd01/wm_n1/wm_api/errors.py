"""Refusal and error types of the scoring API (all message-bearing)."""
from __future__ import annotations


class WmApiError(Exception):
    """Base class of every wm_api refusal."""


class UnknownActionError(WmApiError, ValueError):
    """A plan named an action outside {native, damping, pi, wall}."""


class CheckpointError(WmApiError):
    """A checkpoint directory could not be loaded."""


class CheckpointNotFoundError(CheckpointError):
    """The checkpoint directory or a member file does not exist."""


class CheckpointShaMismatch(CheckpointError):
    """A member file hash does not match the recorded SHA256SUMS."""


class HorizonError(WmApiError, ValueError):
    """``horizon_s != 300`` without ``allow_other_horizon=True``."""


class KOverflowError(WmApiError):
    """More concurrent vehicles than the frozen K = 128 slot pool.

    Reported, never silently dropped: the caller sees the vehicle count,
    the obstacle count and the slot budget.
    """


class StateSchemaError(WmApiError, ValueError):
    """The t0 snapshot violates the documented state schema."""


class PlanSchemaError(WmApiError, ValueError):
    """A candidate plan violates the documented plan schema."""
