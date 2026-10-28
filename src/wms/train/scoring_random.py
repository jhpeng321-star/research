"""Fixed K-sample CRNs shared by gate/E3 scorers (FOLLOWUPS 41)."""
from contextlib import contextmanager

import numpy as np
import torch


def sample_seed(seed, draw=0):
    if int(draw) < 0:
        raise ValueError("draw must be nonnegative")
    return int(seed) + 1000003 * int(draw)


@contextmanager
def scoring_rng(seed, device):
    """Replay independent of earlier loads, actors, variants and the caller's RNG."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


def bind_merge_crn(inputs):
    """Persistent identity keys, never arbitrary batch-row RNG assignment."""
    if inputs.paired_random is None:
        raise ValueError("deterministic paired scoring requires PairedRandom")
    keys = inputs.paired_random._keys & np.uint64((1 << 52) - 1)
    inputs.initial["merge_pivot_rng_seed"] = torch.as_tensor(
        keys.astype(np.int64), device=inputs.initial["s_m"].device)
