"""Process-wide seeding."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed ``random``, numpy and torch (when importable).

    With ``deterministic`` torch is asked for deterministic algorithms and the
    cuBLAS workspace is pinned, as needed for bit-exact golden comparisons.
    """
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
