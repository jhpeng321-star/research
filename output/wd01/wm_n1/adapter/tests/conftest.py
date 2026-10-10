"""Test bootstrap for the adapter suite.

Puts the wm_n1 package root on ``sys.path`` (so ``import adapter``, ``model``,
``evalh`` and ``combine`` resolve), keeps this tests dir importable for
``synth``, and honours ``WD_FORBID_D=1`` (mirrors the model/evalh suites).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]        # output/wd01/wm_n1
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

try:
    import torch
    torch.set_num_threads(1)
except Exception:                                    # pragma: no cover
    pass

if os.environ.get("WD_FORBID_D", "").strip() == "1":
    _WD01 = str(_ROOT.parent)                       # output/wd01
    if _WD01 not in sys.path:
        sys.path.insert(0, _WD01)
    import forbid_d_hook
    forbid_d_hook.install()
