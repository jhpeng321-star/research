"""Flag-gated, observable-only entrant-settings sampling; never reads driver truth."""
import hashlib

import torch


class LearnedMix:
    """Sample a persistent settings vector per birth, from pre-t0 inferred settings.

    The pool is frozen before rollout and contains only slots with actual observed
    pre-t0 history. A private identity/entrance/ordinal keyed draw never advances
    arrival, lane, merge or global RNG streams. Embeddings remain as in the prior.
    """

    def __init__(self, params, observed_mask, seed, row_keys, pools=None):
        self.names = tuple(params)
        observed = observed_mask.bool().any(-1)
        self.pools = ([p.detach().clone() for p in pools] if pools is not None else
            [torch.stack([params[n][b, observed[b]] for n in self.names], -1).detach()
             for b in range(observed.shape[0])])
        if any(len(p) == 0 for p in self.pools):
            raise ValueError("learned-mix requires observed pre-t0 inferred drivers")
        self.seed = int(seed)
        self.row_keys = list(row_keys)
        self.counts = [[0, 0] for _ in self.pools]
        self.assigned = [0] * len(self.pools)

    def __call__(self, state, newborn, params):
        result = {n: v.clone() for n, v in params.items()}
        for b, slot in newborn.nonzero().tolist():
            entrance = int(state['lane'][b, slot] == 2)
            ordinal = self.counts[b][entrance]
            key = f"errsrc-mix:{self.seed}:{self.row_keys[b]}:{entrance}:{ordinal}"
            index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % len(self.pools[b])
            values = self.pools[b][index]
            for j, name in enumerate(self.names):
                result[name][b, slot] = values[j]
            self.counts[b][entrance] += 1
            self.assigned[b] += 1
        return result


def install(model, scale):
    """Scale active persistent/decision merge noise, leave other streams intact."""
    if model.merge_pivot is None:
        raise ValueError("registered noise calibration requires merge-pivot model")
    model.merge_pivot.inference_noise_scale = float(scale)
    return {k: float(v.detach()) for k, v in model.merge_pivot.params().items()}
