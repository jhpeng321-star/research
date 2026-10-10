"""Exception types of the evaluation package."""

from __future__ import annotations


class EvalError(ValueError):
    """Base class for invalid evaluation inputs."""


class DataError(EvalError):
    """A state record or frame list does not satisfy the evaluation schema."""


class MaterializeError(EvalError):
    """A pack, identity sidecar or window descriptor is incomplete or inconsistent."""


class ComparisonError(EvalError):
    """A model comparison request is invalid (for example more than two models)."""


class PredictorError(RuntimeError):
    """A checkpoint or its rollout inputs cannot be scored."""
