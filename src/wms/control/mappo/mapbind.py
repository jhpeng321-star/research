"""Map binding for MAPPO runs on a non-default map (N7: speed-only MAPPO on wz3to2).

The MAPPO trainer is map-agnostic: it samples ``(seed, demand, penetration)`` per episode and
calls ``env.reset``.  A ``[map]`` table in the MAPPO config binds the run to a map descriptor
(the N1 map-descriptor API on ``c4/nm-abstract``; a refusing stub until it lands)::

    [map]
    name = "wz3to2"
    api = ""            # "package.module:function" -> N1 descriptor; "" = built-in stub

and the config uses sentinels that are resolved from the descriptor:

    [env] name = "map"                         -> descriptor env class + map-selecting kwargs
    [rollout] demands_vph = "map"              -> registered TRAIN demand cells
    [rollout] penetrations = "map"             -> registered TRAIN penetration cells

Seeds stay explicit in the config (``"989200-989599"``) and must lie inside the descriptor's
TRAIN range, outside its HELD-OUT range and outside the globally forbidden blocks
(964000-964199, 962xxx, 977xxx).  The stub can be used ONLY with the CPU fake env: binding a
real (SUMO) env to a stub descriptor raises, so no SUMO episode can run on a placeholder map.
Without a ``[map]`` table nothing changes (merge-map runs are byte-identical).

Expected N1 descriptor (attribute or mapping access; first matching alias wins):
``name``; ``env_name`` (dotted CavEnv class, default ``traffic_shepherd.c4.env.CavSumoEnv``);
``env_kwargs`` (map-selecting env kwargs, default ``{"map_name": name}``);
``train_seeds`` | ``seeds_train`` | ``train_seed_range``; ``heldout_seeds`` | ``eval_seeds`` |
``heldout_seed_range``; ``demands_vph`` | ``train_demands_vph`` | ``demand_cells_vph``;
``penetrations`` | ``train_penetrations`` | ``penetration_cells``; optional ``spec_sha256``.
Seed fields may be an iterable of ints, a ``(lo, hi)`` pair (inclusive) or a ``"lo-hi"`` string.
"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping

MAP_SENTINEL = "map"
DEFAULT_ENV_NAME = "traffic_shepherd.c4.env.CavSumoEnv"
FAKE_ENV_NAMES = ("fake",)

#: Never used for any C4 training (task rules): V5 EVAL, WALLREGEN eval block, E3/E3-final eval.
FORBIDDEN_SEED_RANGES = ((964000, 964199), (962000, 962999), (977000, 977999))

#: Registered wz3to2 blocks (newmap_spec.md N1): TRAIN and HELD-OUT final evaluation.
WZ3TO2_TRAIN = (989200, 989599)
WZ3TO2_HELDOUT = (989600, 989699)

#: Stub placeholders (CPU tests only).  N1's registered cells are authoritative; the stub
#: demand cells bracket the probe's chosen 2800 veh/h (DEMAND_CHOSEN.json), the penetration
#: cells are the original MAPPO's training mix.
_STUBS = {
    "wz3to2": {
        "name": "wz3to2",
        "env_name": DEFAULT_ENV_NAME,
        "env_kwargs": {"map_name": "wz3to2"},
        "train_seeds": WZ3TO2_TRAIN,
        "heldout_seeds": WZ3TO2_HELDOUT,
        "demands_vph": (2400, 2600, 2800),
        "penetrations": (0.05, 0.1, 0.2, 0.4),
        "spec_sha256": None,
    },
}

_ALIASES = {
    "name": ("name",),
    "env_name": ("env_name", "env_class", "env"),
    "env_kwargs": ("env_kwargs",),
    "train_seeds": ("train_seeds", "seeds_train", "train_seed_range"),
    "heldout_seeds": ("heldout_seeds", "eval_seeds", "heldout_seed_range"),
    "demands_vph": ("demands_vph", "train_demands_vph", "demand_cells_vph"),
    "penetrations": ("penetrations", "train_penetrations", "penetration_cells"),
    "spec_sha256": ("spec_sha256", "sha256"),
}


class MapBindingError(ValueError):
    """The map binding is inconsistent or would run SUMO on a placeholder descriptor."""


def _get(desc: Any, key: str, default: Any = None) -> Any:
    for alias in _ALIASES[key]:
        if isinstance(desc, Mapping) and alias in desc:
            return desc[alias]
        if not isinstance(desc, Mapping) and hasattr(desc, alias):
            value = getattr(desc, alias)
            return value() if callable(value) and not isinstance(value, type) else value
    return default


def seed_tuple(spec: Any) -> tuple[int, ...]:
    """Seeds from an iterable, an inclusive ``(lo, hi)`` pair, a ``range`` or ``"lo-hi,..."``."""
    if spec is None:
        return ()
    if isinstance(spec, range):
        return tuple(spec)
    if isinstance(spec, str):
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(part))
        return tuple(out)
    values = tuple(spec)
    if (len(values) == 2 and all(type(v) is int for v in values) and values[1] > values[0] + 1):
        return tuple(range(values[0], values[1] + 1))     # inclusive (lo, hi) pair
    return tuple(int(v) for v in values)


def compress(seeds) -> str:
    """``(1, 2, 3, 7)`` -> ``"1-3,7"`` (records only)."""
    seeds = sorted(set(int(s) for s in seeds))
    parts, i = [], 0
    while i < len(seeds):
        j = i
        while j + 1 < len(seeds) and seeds[j + 1] == seeds[j] + 1:
            j += 1
        parts.append(str(seeds[i]) if i == j else f"{seeds[i]}-{seeds[j]}")
        i = j + 1
    return ",".join(parts)


def forbidden(seed: int) -> bool:
    return any(lo <= int(seed) <= hi for lo, hi in FORBIDDEN_SEED_RANGES)


@dataclass(frozen=True)
class MapBinding:
    name: str
    env_name: str
    env_kwargs: dict = field(default_factory=dict)
    train_seeds: tuple[int, ...] = ()
    heldout_seeds: tuple[int, ...] = ()
    demands_vph: tuple[int, ...] = ()
    penetrations: tuple[float, ...] = ()
    is_stub: bool = True
    api: str = ""
    spec_sha256: str | None = None

    def record(self) -> dict:
        return {"name": self.name, "env_name": self.env_name, "env_kwargs": dict(self.env_kwargs),
                "train_seeds": compress(self.train_seeds),
                "heldout_seeds": compress(self.heldout_seeds),
                "demands_vph": list(self.demands_vph), "penetrations": list(self.penetrations),
                "is_stub": self.is_stub, "api": self.api or "stub", "spec_sha256": self.spec_sha256}

    def check_seeds(self, seeds) -> tuple[int, ...]:
        seeds = seed_tuple(seeds)
        if not seeds:
            raise MapBindingError("no training seeds")
        train, held = set(self.train_seeds), set(self.heldout_seeds)
        bad = sorted(s for s in seeds if forbidden(s))
        if bad:
            raise MapBindingError(f"forbidden seeds (964000-964199/962xxx/977xxx): {compress(bad)}")
        leak = sorted(set(seeds) & held)
        if leak:
            raise MapBindingError(f"{self.name}: HELD-OUT seeds in the training set: {compress(leak)}")
        outside = sorted(set(seeds) - train)
        if outside:
            raise MapBindingError(f"{self.name}: seeds outside the registered TRAIN range "
                                  f"{compress(self.train_seeds)}: {compress(outside)}")
        return seeds


def stub_descriptor(name: str) -> dict:
    if name not in _STUBS:
        raise MapBindingError(f"no stub descriptor for map {name!r}; set [map] api to the N1 loader")
    return copy.deepcopy(_STUBS[name])


def load_descriptor(name: str, api: str = ""):
    """The N1 descriptor via ``api = "module:function"`` (called with the map name), else the stub."""
    if not api:
        return stub_descriptor(name), True
    module_name, _, attr = api.partition(":")
    if not module_name or not attr:
        raise MapBindingError(f"[map] api must be 'module:function', got {api!r}")
    loader = getattr(importlib.import_module(module_name), attr)
    return loader(name), False


def binding_from_descriptor(desc, *, name: str, api: str = "", is_stub: bool = False) -> MapBinding:
    got = _get(desc, "name", name)
    if str(got) != name:
        raise MapBindingError(f"descriptor name {got!r} != configured map {name!r}")
    env_kwargs = _get(desc, "env_kwargs", None)
    binding = MapBinding(
        name=name,
        env_name=str(_get(desc, "env_name", DEFAULT_ENV_NAME) or DEFAULT_ENV_NAME),
        env_kwargs=dict(env_kwargs) if env_kwargs is not None else {"map_name": name},
        train_seeds=seed_tuple(_get(desc, "train_seeds")),
        heldout_seeds=seed_tuple(_get(desc, "heldout_seeds")),
        demands_vph=tuple(int(d) for d in (_get(desc, "demands_vph") or ())),
        penetrations=tuple(float(p) for p in (_get(desc, "penetrations") or ())),
        is_stub=bool(is_stub), api=api, spec_sha256=_get(desc, "spec_sha256"))
    if not binding.train_seeds:
        raise MapBindingError(f"{name}: descriptor has no TRAIN seed range")
    if set(binding.train_seeds) & set(binding.heldout_seeds):
        raise MapBindingError(f"{name}: descriptor TRAIN and HELD-OUT ranges overlap")
    if any(forbidden(s) for s in binding.train_seeds):
        raise MapBindingError(f"{name}: descriptor TRAIN range touches a forbidden block")
    if not binding.demands_vph or not binding.penetrations:
        raise MapBindingError(f"{name}: descriptor has no demand/penetration cells")
    if not all(0.0 < p <= 1.0 for p in binding.penetrations):
        raise MapBindingError(f"{name}: penetration cells must be in (0, 1]")
    return binding


def resolve_binding(map_section: Mapping) -> MapBinding:
    section = dict(map_section)
    unknown = set(section) - {"name", "api"}
    if unknown:
        raise MapBindingError(f"unknown [map] keys: {sorted(unknown)}")
    name = str(section.get("name", ""))
    if not name:
        raise MapBindingError("[map] name is required")
    api = str(section.get("api", "") or "")
    desc, is_stub = load_descriptor(name, api)
    return binding_from_descriptor(desc, name=name, api=api, is_stub=is_stub)


def apply_map(cfg: Mapping) -> tuple[dict, MapBinding]:
    """Resolve the ``[map]`` sentinels of a parsed MAPPO config (a deep copy is returned)."""
    cfg = copy.deepcopy(dict(cfg))
    binding = resolve_binding(cfg["map"])
    env = cfg.setdefault("env", {})
    env_name = env.get("name", MAP_SENTINEL)
    if env_name == MAP_SENTINEL:
        if binding.is_stub:
            raise MapBindingError(
                f"map {binding.name!r} is bound to the STUB descriptor (N1 map API not landed / "
                "[map] api unset): refusing to build a SUMO env; use --smoke (fake env) or set [map] api")
        env["name"] = binding.env_name
        env["kwargs"] = {**dict(env.get("kwargs", {})), **binding.env_kwargs}
    elif env_name not in FAKE_ENV_NAMES:
        raise MapBindingError(f"[env] name must be {MAP_SENTINEL!r} (descriptor env) or 'fake' "
                              f"in a [map] run, got {env_name!r}")
    rollout = cfg.setdefault("rollout", {})
    if rollout.get("demands_vph", MAP_SENTINEL) == MAP_SENTINEL:
        rollout["demands_vph"] = list(binding.demands_vph)
    if rollout.get("penetrations", MAP_SENTINEL) == MAP_SENTINEL:
        rollout["penetrations"] = list(binding.penetrations)
    if "seeds" not in rollout:
        raise MapBindingError("[rollout] seeds must be explicit in a [map] run")
    binding.check_seeds(rollout["seeds"])
    return cfg, binding
