"""WD-01 Part B V0: training + evaluation of V(s, tau, policy).

Platform-independent (PLAN_PARTB_V0_TRAINING.md + addenda 2/3/3a; the
Colab addendum is superseded).  One CLI:

    python -m partB_v0 train --data <dataset_root> --out <dir>
    python -m partB_v0 eval   --data <dataset_root> --out <dir>

CPU smoke tests only here -- real training runs on a rented CUDA GPU
(later wrapped in Docker by a separate task; no Dockerfile lives here).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WD01 = _HERE.parent            # output/wd01  (wdpaths)
_PART4 = _WD01 / "part4"        # part4 packages (partB reuse, delay)

for _p in (_PART4, _WD01):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from partB_v0.data import V0Dataset, load_manifest          # noqa: E402
from partB_v0.model import (V0ValueConfig, V0ValueNet,       # noqa: E402
                            count_parameters, policy_id, twohot_encode)

__all__ = ["V0Dataset", "V0ValueConfig", "V0ValueNet", "load_manifest",
           "count_parameters", "policy_id", "twohot_encode"]
