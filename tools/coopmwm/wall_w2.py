"""W2 (c4/wall-tune) option registry: one key set for B and R, everything default OFF.

Keys (all absent == False == the byte-identical pre-W2 path):

* ``wall_pressure_features`` -- observable merge-pressure inputs for the trigger head
  (``wall_pressure``): additive, zero-initialised head; the frozen MAPPO observation and
  the cooperation-layer residual features are untouched.
* ``wall_release`` -- wall proposals may carry the registered gradual-release macro shape
  (``wall_release``); executor commitments last onset+hold+release.
* ``wall_negative_labels`` -- ingest per-seed losing wall situations as explicit
  "do not wall" trigger labels (``wall_negatives``); optional ``wall_negatives`` path
  (default ``<run_root>/WALL_NEGATIVES.jsonl``; a configured path must exist).

B and R read the SAME keys (the track2 configuration is common to both arms). An optional
``w2_by_arm`` mapping is accepted only when B == R == the top-level values, so no
arm-specific W2 behaviour can be registered.
"""

KEYS = ('wall_pressure_features', 'wall_release', 'wall_negative_labels')
PATH_KEYS = ('wall_negatives',)
PROVENANCE = ('W2 build (c4/wall-tune, owner 2026-11-06 after COOP_PROBE_REPORT): merge-pressure '
              'trigger inputs, gradual-release wall macro, per-situation wall tuning, negative labels')


def settings(cfg):
    """Resolved W2 options (all False when absent); refuses malformed / arm-specific values."""
    cfg = cfg or {}
    out = {}
    for key in KEYS:
        value = cfg.get(key, False)
        if type(value) is not bool:
            raise ValueError(f'{key} must be bool')
        out[key] = value
    path = cfg.get('wall_negatives')
    if path is not None:
        if not isinstance(path, str) or not path:
            raise ValueError('wall_negatives must be a path string')
        if not out['wall_negative_labels']:
            raise ValueError('wall_negatives requires wall_negative_labels=True')
    out['wall_negatives'] = path
    by_arm = cfg.get('w2_by_arm')
    if by_arm is not None:
        if not isinstance(by_arm, dict) or set(by_arm) != {'B', 'R'}:
            raise ValueError('w2_by_arm must name exactly B and R')
        for arm, values in by_arm.items():
            if not isinstance(values, dict) or set(values)-set(KEYS)-set(PATH_KEYS):
                raise ValueError('w2_by_arm values must be W2 keys')
            resolved = {k: values.get(k, False if k in KEYS else None) for k in KEYS+PATH_KEYS}
            if resolved != {k: out[k] for k in KEYS+PATH_KEYS}:
                raise ValueError('W2 options must be identical for B and R (and equal the common keys)')
    return out


def any_on(cfg):
    resolved = settings(cfg)
    return any(resolved[k] for k in KEYS)
