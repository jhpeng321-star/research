"""Amendment 77 (S-MAPPO20C) map adapter = tools/nm_mappo/n1_adapter.py with ONE registry delta.

The wz3to2 TRAIN seed range of THIS process is the fresh audited block 987200-987599 (ops/a77/SEED_AUDIT.json)
instead of 989200-989599, so the unchanged trainer / sanity_eval / mapbind / map_scenarios code accepts exactly the
new block and refuses the old one.  Nothing else changes: same N1 descriptor, spec digest 6fc6be65, net b6277898,
k_slots 512, cells, HELD-OUT range 989600-989699, forbidden blocks and env class (n1_adapter.install / make_env).

Config: ``[map] api = "tools.nm_mappo.n1_adapter_a77:descriptor"``; the descriptor returns this module's ``make_env``
as the rollout env so every spawned worker (spawn context) imports this module and applies the same range before
its env is built.  Applied at import, in ``descriptor`` and in ``make_env`` (idempotent; refuses any other value).
"""
from __future__ import annotations

from tools.coopmwm import map_scenarios as _ms
from tools.nm_mappo import n1_adapter as _base

A77_TRAIN_RANGE = (987200, 987599)
ORIGINAL_TRAIN_RANGE = (989200, 989599)
ENV_NAME = "tools.nm_mappo.n1_adapter_a77.make_env"


def _overlaps(a, b) -> bool:
    return not (a[1] < b[0] or a[0] > b[1])


def apply_range() -> tuple:
    """Set map_scenarios.TRAIN_RANGE to the A77 block (idempotent); refuse an unexpected prior value or overlap."""
    cur = tuple(_ms.TRAIN_RANGE)
    if cur not in (ORIGINAL_TRAIN_RANGE, A77_TRAIN_RANGE):
        raise RuntimeError(f"map_scenarios.TRAIN_RANGE is {cur}, expected {ORIGINAL_TRAIN_RANGE} or {A77_TRAIN_RANGE}")
    blocked = [tuple(_ms.HELDOUT_RANGE)] + [tuple(r) for r in _ms.FORBIDDEN_RANGES]
    blocked += [tuple(r) for r in getattr(_ms, "EXTRA_TRAIN_RANGES", ())]
    if any(_overlaps(A77_TRAIN_RANGE, r) for r in blocked):
        raise RuntimeError(f"A77 TRAIN block {A77_TRAIN_RANGE} overlaps a registered/forbidden range {blocked}")
    _ms.TRAIN_RANGE = A77_TRAIN_RANGE
    return A77_TRAIN_RANGE


apply_range()


def descriptor(name: str) -> dict:
    apply_range()
    desc = dict(_base.descriptor(name))
    if tuple(desc["train_seeds"]) != A77_TRAIN_RANGE:
        raise RuntimeError(f"descriptor TRAIN range {desc['train_seeds']} is not the A77 block")
    desc["env_name"] = ENV_NAME
    return desc


def make_env(map_name: str = "wz3to2", **kwargs):
    apply_range()
    return _base.make_env(map_name, **kwargs)
