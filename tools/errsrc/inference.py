"""Flag-gated, observable-only entrant-settings sampling; never reads driver truth."""
import hashlib

import torch


class LearnedMix:
    """Sample a persistent settings vector per birth, from pre-t0 inferred settings.

    The pool is frozen before rollout and contains only slots with actual observed
    pre-t0 history. A private identity/entrance/ordinal keyed draw never advances
    arrival, lane, merge or global RNG streams. Embeddings remain as in the prior.

    ``birth_ordinals`` (WMFIX): when the learned admission records the candidate
    ordinal of each successful placement, ``__call__`` consumes exactly that
    ``(b, slot) -> ordinal`` mapping instead of counting newborns in slot order,
    so the settings the admission tested are the settings the entrant moves with
    even when slot order differs from admission order (CAV virgin slots vs HDV
    recycled slots). In that recorded path a CAV newborn draws from its own
    ``cav_pools`` row (the t0 CAV inferred settings); the legacy slot-order path
    keeps the registered HDV-pool behaviour bit-for-bit.
    """

    def __init__(self, params, observed_mask, seed, row_keys, pools=None, cav_pools=None):
        self.names = tuple(params)
        observed = observed_mask.bool().any(-1)
        self.pools = ([p.detach().clone() for p in pools] if pools is not None else
            [torch.stack([params[n][b, observed[b]] for n in self.names], -1).detach()
             for b in range(observed.shape[0])])
        if any(len(p) == 0 for p in self.pools):
            raise ValueError("learned-mix requires observed pre-t0 inferred drivers")
        self.cav_pools = (None if cav_pools is None
                          else [p.detach().clone() for p in cav_pools])
        self.seed = int(seed)
        self.row_keys = list(row_keys)
        self.counts = [[0, 0] for _ in self.pools]
        self.assigned = [0] * len(self.pools)

    def pool_for(self, b, cav):
        if cav and self.cav_pools is not None:
            if len(self.cav_pools[b]) == 0:
                raise ValueError("CAV birth without an observed CAV settings pool")
            return self.cav_pools[b]
        return self.pools[b]

    def __call__(self, state, newborn, params, birth_ordinals=None):
        result = {n: v.clone() for n, v in params.items()}
        recorded = bool(birth_ordinals)
        cav_state = state.get("is_cav")
        for b, slot in newborn.nonzero().tolist():
            entrance = int(state['lane'][b, slot] == 2)
            cav = bool(cav_state[b, slot]) if cav_state is not None else False
            if recorded:
                ordinal = birth_ordinals.get((b, slot))
                if ordinal is None:
                    raise ValueError("newborn slot missing from the recorded birth ordinals")
                pool = self.pool_for(b, cav)
            else:
                ordinal = self.counts[b][entrance]
                pool = self.pools[b]
            key = f"errsrc-mix:{self.seed}:{self.row_keys[b]}:{entrance}:{ordinal}"
            index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % len(pool)
            values = pool[index]
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
