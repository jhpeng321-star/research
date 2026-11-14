"""Seeded synthetic WM-S inputs (2 worlds x 64 slots, CAVs, junction, history)."""
from __future__ import annotations

import torch

B, K, N_STEPS = 2, 64, 12


def make_state(seed: int = 0, n_present: int = 40):
    g = torch.Generator().manual_seed(seed)
    lane = torch.randint(0, 3, (B, K), generator=g)
    s = torch.rand(B, K, generator=g) * 400.0 - 250.0
    s = torch.where(lane == 2, torch.rand(B, K, generator=g) * 90.0 - 40.0, s)
    v = torch.rand(B, K, generator=g) * 25.0 + 2.0
    a = torch.randn(B, K, generator=g) * 0.5
    present = torch.zeros(B, K, dtype=torch.bool)
    present[:, :n_present] = True
    is_cav = torch.zeros(B, K, dtype=torch.bool)
    is_cav[:, :5] = True
    return {"s_m": s, "v_mps": v, "a_mps2": a, "lane": lane,
            "length_m": torch.full((B, K), 5.0), "present": present, "is_cav": is_cav}


def make_history(seed: int = 1, length: int = 10, features: int = 6):
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn(B, K, length, features, generator=g) * 0.3
    mask = torch.rand(B, K, length, generator=g) > 0.2
    return hist, mask


def make_long_history(seed: int, length: int):
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn(B, K, length, 9, generator=g) * 0.3
    hist[..., 2:5] = torch.nn.functional.one_hot(
        torch.randint(0, 3, (B, K, length), generator=g), 3).float()
    mask = torch.rand(B, K, length, generator=g) > 0.1
    return hist, mask


def make_teacher(state, n_steps: int):
    """Hold-speed CAV teacher ``[B, n+1, K, 4]`` = s, lane, v, a."""
    s, v, lane = state["s_m"], state["v_mps"], state["lane"].float()
    rows = []
    for i in range(n_steps + 1):
        rows.append(torch.stack((s + v * 0.5 * i, lane, v, torch.zeros_like(v)), -1))
    return torch.stack(rows, 1)
