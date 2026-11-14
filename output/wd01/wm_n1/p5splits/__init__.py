"""WM-S P5 held-out-strategy splits (T8 of WM-S-PREP-20261005).

Public surface::

    from p5splits import splits
    splits.materialize(records, cal_tune_worlds=...)
    splits.load_index(path)
"""
from __future__ import annotations

from . import splits

__all__ = ["splits"]
