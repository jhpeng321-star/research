"""Synthetic packed-corpus fixture matching ``train/dataset.py`` (no SUMO).

The CORPUS piece owns the real packed writer; it is a stub in this checkout, so
the TRAIN tests build a minimal pack with the exact layout documented in
``train/dataset.py`` (interface note D3).  Shapes and feature indices mirror
``features/FORMAT.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wms.legacy.vendor.model import constants as C

TAUS = tuple(float(t) for t in C.TAUS)


def make_runs():
    """A small TRAIN+CAL run set with sibling arms (world, t0)."""
    return [
        dict(chunk="chunk4", world="w1", arm="base", policy="damping",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w1", arm="native", policy="native",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w1", arm="pi", policy="pi",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w2", arm="native", policy="native",
             split="TRAIN", t0_s=600.0),
        dict(chunk="chunk4", world="w2", arm="base", policy="damping",
             split="TRAIN", t0_s=600.0),
        dict(chunk="chunk2", world="c1", arm="native", policy="native",
             split="CALIBRATION", t0_s=300.0),
    ]


def _fill_veh(rng, veh, mask, base, t, k):
    for s in range(k):
        veh[base + t, s, C.S_DOWN] = -150.0 + 20.0 * s
        lane = s % C.N_LANES
        veh[base + t, s, C.LANE_LEFT + lane] = 1.0
        veh[base + t, s, C.V_IDX] = 8.0 + 2.0 * s
        veh[base + t, s, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
        veh[base + t, s, C.IS_CAV_IDX] = 1.0 if s < 3 else 0.0
        veh[base + t, s, C.ACT_NATIVE] = 1.0 if s < 3 else 0.0
        veh[base + t, s, C.ACT_UNKNOWN] = 0.0 if s < 3 else 1.0
        veh[base + t, s, C.PRESENT_IDX] = 1.0
        mask[base + t, s] = 1


def write_pack(root, runs, *, k: int = 16, n_ticks: int = 2, seed: int = 0):
    """Write a minimal packed corpus; returns the manifest dict."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_runs = len(runs)
    n_tick = n_runs * n_ticks

    veh = np.zeros((n_tick, k, C.VEH_F), dtype=np.float16)
    mask = np.zeros((n_tick, k), dtype=np.uint8)
    plan = np.zeros((n_runs, k, C.N_SEGMENTS, C.PLAN_F), dtype=np.float16)
    wall = np.zeros((n_tick, C.MAX_WALLS, C.WALL_F), dtype=np.float16)
    wall_mask = np.zeros((n_tick, C.MAX_WALLS), dtype=np.uint8)
    counter = rng.random((n_tick, C.N_ENTRANCES, C.COUNTER_BINS)).astype(
        np.float16)
    accel = rng.standard_normal((n_tick, k)).astype(np.float16)
    target_mask = np.zeros((n_tick, k), dtype=np.uint8)
    lane_next = np.zeros((n_tick, k), dtype=np.int8)
    arrivals = np.zeros((n_tick, C.N_ENTRANCES), dtype=np.int16)

    items = []
    outcome_rows = []
    for r, run in enumerate(runs):
        base = r * n_ticks
        for t in range(n_ticks):
            _fill_veh(rng, veh, mask, base, t, k)
            target_mask[base + t] = mask[base + t]
            if t == 1:
                lane_next[base + t, :k] = 0
        for s in range(min(3, k)):
            plan[r, s, :, 0] = 1.0                  # planned native

        rows = []
        for tau in TAUS:
            available = not (run["arm"] == "base" and tau == 570.0)
            rows.append({
                "world": run["world"], "source": run["arm"],
                "chunk": run["chunk"], "split": run["split"],
                "t0_s": run["t0_s"], "t": run["t0_s"], "tau": tau,
                "total_delay_ref_s": 10.0 + 3.0 * r + tau / 30.0,
                "ttc_lt3_veh_s": 1.0 + 0.1 * tau / 30.0,
                "hard_brake_s": 0.25 * r,
                "collision_any": 1.0 if run["arm"] == "pi" else 0.0,
                "tau_available": available,
            })
        first = len(outcome_rows)
        outcome_rows.extend(rows)
        items.append({
            "world": run["world"], "arm": run["arm"], "chunk": run["chunk"],
            "split": run["split"], "t0_s": float(run["t0_s"]),
            "pair_key": f"{run['world']}@t0={run['t0_s']:g}",
            "policy": run.get("policy", run["arm"]),
            "tick": base, "plan_row": r,
            "outcome_rows": list(range(first, first + len(TAUS))),
        })
        if "batch" in run:
            items[-1]["batch"] = run["batch"]

    arrays = {
        "veh_step": veh, "veh_mask": mask, "plan": plan, "wall": wall,
        "wall_mask": wall_mask, "counter": counter, "accel_next": accel,
        "target_mask": target_mask, "lane_next": lane_next,
        "arrivals": arrivals,
    }
    for name, arr in arrays.items():
        np.save(root / f"{name}.npy", arr)

    pd.DataFrame(outcome_rows).to_parquet(root / "outcomes.parquet")

    manifest = {
        "schema": "wm_n1_pack/1",
        "k_max": int(k),
        "arrays": {name: {"file": f"{name}.npy", "dtype": str(arr.dtype),
                          "shape": list(arr.shape)}
                   for name, arr in arrays.items()},
        "outcomes": {"file": "outcomes.parquet"},
        "items": items,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def write_default_pack(root, **kwargs):
    return write_pack(root, make_runs(), **kwargs)


#: the v1 arrays ``corpus.pack`` copies into a pack (the patchable surfaces).
V1_PACK_ARRAYS = ("veh_step", "veh_mask", "plan", "wall", "wall_mask",
                  "counter", "accel_next", "target_mask", "lane_next",
                  "arrivals")


def _persist_v1(shard, shard_dir) -> None:
    for name in V1_PACK_ARRAYS:
        np.save(Path(shard_dir) / f"{name}.npy", np.asarray(shard[name]))


def write_idea_a_pack(root, runs=None, *, k: int = 16, seed: int = 0,
                      patch=None):
    """Build a real-format Idea A pack through the corpus writer path.

    The runs go through ``corpus.tests.synth`` shards, the P9 Idea A channels
    (``corpus.idea_a.write_idea_a_channels``) and ``corpus.pack.pack_shards``,
    i.e. the exact path a GPU-box pack uses.  ``patch(shard, shard_dir)`` may
    mutate the loaded shard dict (``veh_step``/``veh_mask``/``target_mask``/
    ``accel_next`` ...) before the Idea A channels are derived and the shards
    are packed, so a test can craft slot reuse and run boundaries.  Returns
    the pack directory.
    """
    from corpus import idea_a as CIA
    from corpus import pack as PK
    from corpus.tests import synth as CS

    root = Path(root)
    shards = root / "shards"
    CS.write_shards(shards, runs=runs, k=k, seed=seed)
    index = json.loads((shards / "INDEX.json").read_text(encoding="utf-8"))
    for key, rec in index["runs"].items():
        shard_dir = Path(rec["shard_dir"])
        arrs = {name: np.load(shard_dir / f"{name}.npy")
                for name in CS.NPY_FILES}
        meta = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))
        meta["n_ticks"] = int(arrs["veh_mask"].shape[0])
        meta.setdefault("split", "TRAIN")
        shard = dict(arrs, slot_vid=[], outcomes=None, meta=meta)
        if patch is not None:
            patch(shard, shard_dir)
            _persist_v1(shard, shard_dir)
        out = CIA.write_idea_a_channels(shard, shard_dir)
        assert out["status"] == "written", key
    pack_dir = root / "pack"
    PK.pack_shards(shards, pack_dir, workers=1)
    return pack_dir
