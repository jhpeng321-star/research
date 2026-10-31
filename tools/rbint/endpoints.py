"""Small predicted60s endpoint receipt; factual future motion is NEVER read here."""
import numpy as np

from tools.rbint.observations import load_observations


def attach_endpoints60(episodes, arrays, seed, initial_slots):
    start, numeric_ids, hashes = load_observations(seed)
    selected = start['state']['present'][0].bool()
    if len(numeric_ids) != initial_slots:
        raise ValueError('endpoint cohort differs from registered initial identities')
    names = [str(start['vehicle_ids'][int(vid)]) for vid in numeric_ids]
    hdv = (~start['state']['is_cav'][0, selected].bool()).tolist()
    positions, masks = arrays['positions'], arrays['masks'].astype(bool)
    if positions.shape != masks.shape or positions.shape[1] < 120 or len(episodes) != positions.shape[0]:
        raise ValueError('need complete half-second predicted motion through60s')
    # A reused slot must never be scored as the old t0 vehicle. Once it exits,
    # its60s endpoint is censored rather than silently replaced by a newborn.
    survived = masks[:, :120, :initial_slots].all(1)
    endpoint = positions[:, 119, :initial_slots]
    if not np.isfinite(endpoint).all():
        raise ValueError('nonfinite predicted endpoint')
    for i, episode in enumerate(episodes):
        if episode['seed'] != seed or episode['initial_sha256'] != start['initial_state_sha256']:
            raise ValueError('endpoint start identity differs')
        episode['endpoint60'] = dict(schema='wmphys-predicted-endpoints/1', horizon_s=60,
            vehicle_ids=names, is_hdv=hdv, survived=survived[i].tolist(),
            s_m=[float(value) if alive else None for value, alive in zip(endpoint[i], survived[i])],
            start_sha256=hashes['start.pt'], history_sha256=hashes['start_history.npz'],
            censoring='predicted exit/recycled slot; no fabricated continuation',
            factual_future_used=False)
