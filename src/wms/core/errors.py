"""Exception hierarchy shared by every ``wms`` sub-package."""
from __future__ import annotations


class WmsError(Exception):
    """Base class of all errors raised deliberately by ``wms``."""


class TrainDataRefusal(WmsError, PermissionError):
    """A data source is not demonstrably TRAIN-only (restricted-token path)."""


class WmsCheckpointError(WmsError, RuntimeError):
    """A checkpoint cannot be loaded or does not match the expected format."""


class MaterializeError(WmsError, ValueError):
    """An evaluation state cannot be materialised from its source pack."""


class C1v2Error(WmsError, ValueError):
    """A C1v2 scoring input is inconsistent."""


class ConfigError(WmsError, ValueError):
    """A configuration is malformed, unknown-keyed or fails validation."""


class NotYetImplemented(WmsError, NotImplementedError):
    """A public signature whose implementing work package has not landed yet."""


def not_yet(package: str, qualname: str) -> NotYetImplemented:
    """Build the error raised by skeleton stubs, naming the owning work package."""
    return NotYetImplemented(
        f"{qualname} is a skeleton stub: implemented by {package} "
        "(see docs/ARCHITECTURE_V4.md section 5)")
