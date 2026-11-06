"""N1 map API (c4/nm-abstract, tools/coopmwm/map_*) -> MAPPO trainer / sanity-eval binding (N7).

Config: ``[map] name = "wz3to2"``, ``api = "tools.nm_mappo.n1_adapter:descriptor"``.

* ``descriptor(name)`` (called by ``wms.control.mappo.mapbind`` in the TRAINER/EVAL MAIN process)
  installs the N1 runtime binding in that process (``map_runtime.install``: lane table, scenario
  builder, seed guard, map env class with the 512-slot pool, CAV-report-masked RSU field), so the
  PPO update rebuilds observations and the RSU field exactly like the rollout workers; it returns
  the registered TRAIN/HELD-OUT seed ranges and cells (``map_scenarios``) and the env factory.
* ``make_env(map_name, **kwargs)`` (the rollout ``env_name``) installs the same binding in each
  spawned worker before constructing the (rebound) ``traffic_shepherd.c4.env.CavSumoEnv``.

Process map selection: ``COOPMWM_MAP_SPEC`` (N1 convention). If unset, it is set to this code
root's ``assets/research/newmap_candidates/wz3to2/MAP_SPEC.json`` (the descriptor load verifies the
registered spec digest and the raw net-file sha256 b6277898...). One map per process.
"""
from __future__ import annotations

import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
SPECS = {"wz3to2": CODE / "assets" / "research" / "newmap_candidates" / "wz3to2" / "MAP_SPEC.json"}
ENV_NAME = "tools.nm_mappo.n1_adapter.make_env"


def install(map_name: str = "wz3to2"):
    """Bind ``map_name`` in this process (idempotent); returns (descriptor, install record)."""
    from tools.coopmwm import map_descriptor as md
    from tools.coopmwm import map_runtime
    if map_name not in SPECS:
        raise ValueError(f"no N1 map {map_name!r}")
    if not os.environ.get(md.ENV):
        os.environ[md.ENV] = str(SPECS[map_name])
    desc = md.current()
    if desc.name != map_name:
        raise ValueError(f"process map is {desc.name!r}, not {map_name!r}")
    map_runtime.install({"map": map_name}, leverprobe=False)
    return desc, map_runtime.installed_record()


def descriptor(name: str) -> dict:
    desc, record = install(name)
    from tools.coopmwm import map_scenarios as ms
    return {"name": name, "env_name": ENV_NAME, "env_kwargs": {"map_name": name},
            "train_seeds": tuple(ms.TRAIN_RANGE), "heldout_seeds": tuple(ms.HELDOUT_RANGE),
            "demands_vph": tuple(ms.DEMANDS), "penetrations": tuple(ms.SHARES),
            "spec_sha256": desc.digest, "network_file_sha256": desc.network_file_sha256,
            "k_slots": desc.k_slots, "install_record": record}


def make_env(map_name: str = "wz3to2", **kwargs):
    install(map_name)
    from traffic_shepherd.c4 import env as E
    if getattr(E.CavSumoEnv, "MAP_NAME", None) != map_name:
        raise RuntimeError("N1 map env class not bound")
    return E.CavSumoEnv(**kwargs)
