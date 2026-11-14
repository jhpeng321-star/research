"""Dynamic entry-point resolution with explicit "piece missing" errors.

The Idea A pieces (P1, P2, P3, P7, bundle e2e) land in parallel with P11, so a
gate must not crash when one is absent: :func:`resolve` raises
:class:`PieceMissing` with the exact dotted name, and each gate turns that into
a FAIL report.  Tests inject a fake resolver to exercise both paths.
"""
from __future__ import annotations

import importlib


class PieceMissing(RuntimeError):
    """A piece module or entry point required by a gate is absent."""


def resolve(dotted: str):
    """Return the callable/object at ``"pkg.mod.attr"`` or raise.

    ``dotted`` must be a module path plus one attribute name.  A missing
    module, an import error inside the module, or a missing attribute all
    raise :class:`PieceMissing` with the dotted name in the message.
    """
    if not isinstance(dotted, str) or "." not in dotted:
        raise PieceMissing(f"bad entry point {dotted!r}; expected 'pkg.mod.attr'")
    module_name, attr = dotted.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        missing = e.name or module_name
        raise PieceMissing(
            f"piece missing: {dotted} (module {missing!r} not importable)") \
            from e
    except Exception as e:                                # noqa: BLE001
        raise PieceMissing(
            f"piece missing: {dotted} (importing {module_name} failed: "
            f"{type(e).__name__}: {e})") from e
    obj = getattr(mod, attr, None)
    if obj is None:
        raise PieceMissing(
            f"entry point missing: {dotted} (module {module_name} has no "
            f"attribute {attr!r})")
    return obj
