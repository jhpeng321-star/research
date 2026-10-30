"""Exact-zero replay planning; no gate heuristics or objective changes."""
import math

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .contract import PHYSICS_DT_S, STEPS_PER_CONTROL


PREFIX_OUTPUTS = frozenset(('rewards', 'costs', 'returns', 'values', 'entropy', 'actions',
    'action_mask', 'lanes', 'entrance_zone', 'components', 'action_audit',
    'antiexploit_commands'))


def dense_replay_horizon(trainer, batch, cfg):
    dense = trainer._imagination_config(batch).horizon
    quantum = math.lcm(STEPS_PER_CONTROL, cfg.imagination_checkpoint_segment or 1)
    steps = math.ceil((dense * STEPS_PER_CONTROL + 1) / quantum) * quantum
    return min(cfg.horizon, steps // STEPS_PER_CONTROL)


def replay_horizon(trainer, batch, cfg, cache, slots, vectors, spec, future, rows):
    """Conservative per-microbatch plan, in control steps (zero means no VJP).

    Unknown nonzero output paths and shared disagreement retain the full horizon.
    NaN/Inf count as nonzero. Inspect the actual total output cotangent, not ON rows.
    """
    if not (cfg.imagination_skip_zero_vjp or cfg.imagination_replay_dense_prefix):
        return cfg.horizon
    leaves = [None] * spec.num_leaves
    for slot, vector in zip(slots, vectors):
        leaves[slot] = vector
    outputs = tree_unflatten(leaves, spec)
    count = len(cache['items'])
    live = long_live = unknown = False
    for name, value in outputs.items():
        if name == 'states':
            tensors, _ = tree_flatten(value[future])
            selected = [v.index_select(0, rows.to(v.device)) for v in tensors
                        if torch.is_tensor(v)]
        elif name in PREFIX_OUTPUTS or name == 'gated300_long_difference':
            tensors, _ = tree_flatten(value)
            selected = []
            for v in tensors:
                if not torch.is_tensor(v):
                    continue
                if v.ndim < 3 or v.shape[0] != count:
                    unknown |= bool(torch.count_nonzero(v))
                else:
                    selected.append(v[future].index_select(1, rows.to(v.device)))
        else:
            tensors, _ = tree_flatten(value)
            # Detached metrics normally have None cotangents. Any new consumer
            # must explicitly establish its dependency geometry before optimizing.
            selected = []
            unknown |= any(bool(torch.count_nonzero(v)) for v in tensors if torch.is_tensor(v))
        nonzero = any(bool(torch.count_nonzero(v)) for v in selected)
        live |= nonzero
        long_live |= nonzero and name == 'gated300_long_difference'
    if not live and not unknown and cfg.imagination_skip_zero_vjp:
        return 0
    from . import gated300
    if (cfg.imagination_replay_dense_prefix and not long_live and not unknown
            and gated300.requested_rollout(trainer)):
        # One physics step beyond the prefix preserves next-state semantics:
        # entry rewards use the NEXT callback's post-insertion input state.
        # It also avoids treating the dense boundary as a WM terminal state.
        return dense_replay_horizon(trainer, batch, cfg)
    return cfg.horizon


def record_replay(trainer, start, future, horizon, full, elapsed):
    records = getattr(trainer, '_longskip_records', None)
    if records is None:
        return
    records.append(dict(phase=getattr(trainer, '_longgrad_phase', None) or 'combined',
        start=start, future=future, replay_horizon_s=horizon * STEPS_PER_CONTROL * PHYSICS_DT_S,
        full_horizon_s=full * STEPS_PER_CONTROL * PHYSICS_DT_S, wall_time_s=elapsed))


def metrics(trainer):
    records = getattr(trainer, '_longskip_records', [])
    live = [r for r in records if r['replay_horizon_s']]
    # A same-update linear-work estimate, not a GPU benchmark or a critical-path
    # estimate for parallel workers. Null if every replay was skipped.
    work = sum(r['replay_horizon_s'] for r in live)
    rate = sum(r['wall_time_s'] for r in live) / work if work else None
    saved = sum(r['full_horizon_s'] - r['replay_horizon_s'] for r in records)
    return dict(futures_replayed=len(live), replay_horizon_s=records,
                replay_seconds_saved_estimate=None if rate is None else saved * rate)
