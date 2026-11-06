"""Seed and path refusals for the W3 wall-fit data path (checked before any record is used).

Seeds: only registered TRAIN worlds (tools.coopmwm.protocol.worlds('train', i) for every
TRAIN_STARTS block: 986000-986071, 987000-987071, 988000-988071). Evaluation seeds
962000-962999, 964000-964199 and 977000-977999 are refused explicitly (and anything else that is
not a registered TRAIN world, e.g. final_test 9896xx or check/dev worlds).

Paths: never opened (and directories never listed) when the path contains a forbidden token:
holdout / heldout / cal_split / chunk5 / j4b / inc1 / sealed, or a driver-truth token
(hetdata, truth, vehicle_params, driver_params, vtype).
"""
import os
from pathlib import Path

FORBIDDEN_SEED_RANGES = ((962000, 962999), (964000, 964199), (977000, 977999))
# protocol.TRAIN_STARTS x (12 cells x 6 repeats) -- cross-checked by the CPU tests.
TRAIN_SEED_BLOCKS = ((986000, 986071), (987000, 987071), (988000, 988071))
FORBIDDEN_PATH_TOKENS = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'sealed',
                         'hetdata', 'truth', 'vehicle_params', 'driver_params', 'vtype')
# Heavy, irrelevant subtrees (SUMO branch tapes); pruned for speed, never needed for labels.
PRUNE_DIRS = ('branches', '__pycache__', '.git')


class ForbiddenSeedError(ValueError):
    """A record references an evaluation / non-TRAIN world seed: the whole build is refused."""


class ForbiddenPathError(ValueError):
    """An explicitly requested path contains a forbidden token."""


def forbidden_seed(seed):
    s = int(seed)
    return any(lo <= s <= hi for lo, hi in FORBIDDEN_SEED_RANGES)


def train_seed(seed):
    s = int(seed)
    return any(lo <= s <= hi for lo, hi in TRAIN_SEED_BLOCKS)


def check_seed(seed, where=''):
    """Return int(seed) for a registered TRAIN world; raise ForbiddenSeedError otherwise."""
    try:
        s = int(seed)
    except (TypeError, ValueError):
        raise ForbiddenSeedError(f'unparseable world seed {seed!r} {where}'.strip())
    if forbidden_seed(s):
        raise ForbiddenSeedError(f'evaluation seed {s} refused (962xxx / 964000-964199 / 977xxx) {where}'.strip())
    if not train_seed(s):
        raise ForbiddenSeedError(f'world seed {s} is not a registered TRAIN world {where}'.strip())
    return s


def path_allowed(path):
    low = str(path).replace('\\', '/').lower()
    return not any(tok in low for tok in FORBIDDEN_PATH_TOKENS)


def check_path(path):
    """Explicit inputs/outputs: refuse loudly (never open) a forbidden path."""
    if not path_allowed(path):
        raise ForbiddenPathError(f'forbidden path refused (never opened): {path}')
    return Path(path)


def walk_files(root, match, skipped=None, skip_dirs=()):
    """Yield files under `root` for which match(name) is true. Forbidden directories are pruned
    BEFORE they are listed and forbidden files are never yielded (counted in `skipped`). `skip_dirs`
    are extra directory names to prune (e.g. an environment clone such as gpuA 'cpuv_home')."""
    root = check_path(root)
    if root.is_file():
        if match(root.name):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in sorted(dirnames):
            if d in PRUNE_DIRS or d in skip_dirs:
                continue
            if not path_allowed(os.path.join(dirpath, d)):
                if skipped is not None:
                    skipped.append(os.path.join(dirpath, d))
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in sorted(filenames):
            if not match(f):
                continue
            p = os.path.join(dirpath, f)
            if not path_allowed(p):
                if skipped is not None:
                    skipped.append(p)
                continue
            yield Path(p)
