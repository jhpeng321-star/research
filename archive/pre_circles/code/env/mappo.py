"""Helpers for Flat-MARL / Shepherd reward and MAPPO observation packing.

v3: Packed obs uses trust_weights (float) instead of binary mask.
Layout: local | peers(max_cavs * local) | trust_weights(max_cavs).
"""

from __future__ import annotations

import sys
from typing import Mapping

import numpy as np


_WARNED_OVERFLOW_MAX_CAVS: set[int] = set()


def acceleration_to_shepherd_mode(action: object, coast_tolerance: float = 1e-3) -> int:
    """Map continuous acceleration to the control-cost convention.

    The action remains a physical continuous acceleration. This mapping is used
    only by Flat-MARL's Shepherd reward: near-zero acceleration is coast
    (control cost 0); a nonzero intervention has control cost -1.
    """
    acceleration = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    return 0 if abs(acceleration) <= coast_tolerance else 1


def mappo_obs_dim(local_dim: int, max_cavs: int) -> int:
    """local | peers(max_cavs * local) | trust_weights(max_cavs)."""
    return local_dim + max_cavs * local_dim + max_cavs


def pack_mappo_observation(
    local_obs: np.ndarray,
    all_obs: Mapping[str, np.ndarray],
    max_cavs: int,
    trust_weights: Mapping[str, float] = None,
    ego_id: str = None,
) -> np.ndarray:
    """Pack local obs + padded peer obs stack + trust weights for CTDE critic.

    Args:
        local_obs: ego CAV's local observation vector.
        all_obs: mapping from CAV ID → local obs for all active CAVs.
        max_cavs: max number of peer CAVs to pack.
        trust_weights: mapping from CAV ID → trust weight [0, 1].
            If None, defaults to 1.0 for all active peers (uniform).
        ego_id: if given, excluded from the peer stack (ego is already
            encoded in ``local_obs``; including it would double-count).

    Peers are packed in sorted-ID order for determinism. If there are more
    active CAVs than ``max_cavs``, the overflow peers are truncated (the obs
    vector must stay fixed-size) but a warning is printed once per process —
    silently dropping peers biases the critic, so ``max_cavs_critic`` should
    be raised above the maximum concurrent active CAV count instead.
    """
    local = np.asarray(local_obs, dtype=np.float32).ravel()
    local_dim = local.shape[0]
    peers = np.zeros((max_cavs, local_dim), dtype=np.float32)
    weights = np.zeros(max_cavs, dtype=np.float32)

    sorted_ids = sorted(all_obs.keys())
    if ego_id is not None:
        sorted_ids = [aid for aid in sorted_ids if aid != ego_id]
    if len(sorted_ids) > max_cavs and max_cavs not in _WARNED_OVERFLOW_MAX_CAVS:
        print(
            f"[pack_mappo_observation] WARNING: {len(sorted_ids)} active CAVs "
            f"exceed max_cavs={max_cavs}; truncating to the first {max_cavs} "
            "in sorted-ID order. The critic will NOT see the overflow peers "
            "— raise max_cavs_critic.",
            file=sys.stderr,
        )
        _WARNED_OVERFLOW_MAX_CAVS.add(max_cavs)
    if len(sorted_ids) > max_cavs:
        sorted_ids = sorted_ids[:max_cavs]
    for i, aid in enumerate(sorted_ids):
        peers[i] = np.asarray(all_obs[aid], dtype=np.float32).ravel()
        if trust_weights is not None and aid in trust_weights:
            weights[i] = float(trust_weights[aid])
        else:
            weights[i] = 1.0

    return np.concatenate([local, peers.ravel(), weights], axis=0)


def unpack_mappo_observation(
    packed: np.ndarray, local_dim: int, max_cavs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (local, peers[max,local], trust_weights[max])."""
    x = np.asarray(packed, dtype=np.float32).ravel()
    local = x[:local_dim]
    peer_end = local_dim + max_cavs * local_dim
    peers = x[local_dim:peer_end].reshape(max_cavs, local_dim)
    trust = x[peer_end : peer_end + max_cavs]
    return local, peers, trust
