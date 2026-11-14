"""Synthetic shard-tree fixtures for the corpus tests (no SUMO, no D:).

Builds a minimal but shape-correct ``features`` data root: the 12 per-run npy
arrays, ``outcomes.parquet``, ``DONE.json``, the ``INDEX.json`` and the
``PLAN_MAP.json`` that :mod:`corpus.pack` consumes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]        # output/wd01/wm_n1
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import constants as C  # noqa: E402

TAUS = tuple(float(t) for t in C.TAUS)
V1_NPY_FILES = ("veh_step", "veh_mask", "slot_index", "plan", "decision_t",
                "wall", "wall_mask", "counter", "accel_next", "target_mask",
                "lane_next", "arrivals")
IDEA_A_FILES = ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                "event_mask")
# every npy the shard tree carries (DONE.json records the v1 writer files
# ONLY; the Idea A channels carry their own IDEA_A_DONE.json receipt)
NPY_FILES = V1_NPY_FILES + IDEA_A_FILES
IDEA_A_SHAPE_F = {"veh_kin": 7, "hdv_base": 6, "cav_plan_ch": 8}
NON_NPY = ("slot_vid.json", "meta.json", "outcomes.parquet")
INDEX_SCHEMA = "wm_n1_features_index/1"


def default_runs():
    return [
        dict(chunk="chunk4", world="w1", arm="base", policy="damping",
             split="TRAIN", t0_s=540.0, n_ticks=3, n_dec=2),
        dict(chunk="chunk4", world="w1", arm="native", policy="native",
             split="TRAIN", t0_s=540.0, n_ticks=3, n_dec=2),
        dict(chunk="chunk4", world="w1", arm="pi", policy="pi",
             split="TRAIN", t0_s=540.0, n_ticks=3, n_dec=2),
        dict(chunk="chunk4", world="w2", arm="native", policy="native",
             split="TRAIN", t0_s=600.0, n_ticks=2, n_dec=2),
        dict(chunk="chunk2", world="c1", arm="native", policy="native",
             split="CALIBRATION", t0_s=300.0, n_ticks=2, n_dec=2),
    ]


def _arrays(k: int, n_ticks: int, n_dec: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    veh = np.zeros((n_ticks, k, C.VEH_F), np.float16)
    mask = np.zeros((n_ticks, k), np.uint8)
    for t in range(n_ticks):
        for s in range(k):
            veh[t, s, C.S_DOWN] = -150.0 + 20.0 * s
            veh[t, s, C.LANE_LEFT + (s % C.N_LANES)] = 1.0
            veh[t, s, C.V_IDX] = 8.0 + 2.0 * s
            veh[t, s, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
            veh[t, s, C.IS_CAV_IDX] = 1.0 if s < 3 else 0.0
            veh[t, s, C.ACT_NATIVE] = 1.0 if s < 3 else 0.0
            veh[t, s, C.ACT_UNKNOWN] = 0.0 if s < 3 else 1.0
            veh[t, s, C.PRESENT_IDX] = 1.0
            mask[t, s] = 1
    return dict(
        veh_step=veh,
        veh_mask=mask,
        slot_index=np.tile(np.arange(k, dtype=np.int32), (n_ticks, 1)),
        plan=np.zeros((n_dec, k, C.N_SEGMENTS, C.PLAN_F), np.float16),
        decision_t=np.arange(n_dec, dtype=np.float64) * 30.0,
        wall=np.zeros((n_ticks, C.MAX_WALLS, C.WALL_F), np.float16),
        wall_mask=np.zeros((n_ticks, C.MAX_WALLS), np.uint8),
        counter=rng.random((n_ticks, C.N_ENTRANCES, C.COUNTER_BINS)).astype(
            np.float16),
        accel_next=rng.standard_normal((n_ticks, k)).astype(np.float16),
        target_mask=mask.copy(),
        lane_next=np.zeros((n_ticks, k), np.int8),
        arrivals=np.zeros((n_ticks, C.N_ENTRANCES), np.int16),
        veh_kin=np.zeros((n_ticks, k, IDEA_A_SHAPE_F["veh_kin"]), np.float16),
        hdv_base=np.zeros((n_ticks, k, IDEA_A_SHAPE_F["hdv_base"]), np.float16),
        cav_plan_ch=np.zeros(
            (n_ticks, k, IDEA_A_SHAPE_F["cav_plan_ch"]), np.float16),
        event_tok=np.zeros((n_ticks, 8, 8), np.float16),
        event_mask=np.zeros((n_ticks, 8), np.uint8),
    )


def _outcomes(run: dict, first: int) -> pd.DataFrame:
    rows = []
    for tau in TAUS:
        available = not (run["arm"] == "base" and tau == 570.0)
        rows.append({
            "world": run["world"], "source": run["arm"],
            "chunk": run["chunk"], "split": run["split"],
            "t0_s": run["t0_s"], "t": run["t0_s"], "tau": tau,
            "total_delay_ref_s": 10.0 + tau / 30.0,
            "ttc_lt3_veh_s": 1.0 + 0.1 * tau / 30.0,
            "hard_brake_s": 0.25,
            "collision_any": 1.0 if run["arm"] == "pi" else 0.0,
            "tau_available": available,
            "n_vehicles_present": 30.0,
        })
    return pd.DataFrame(rows)


def write_shards(root: Path, runs=None, *, k: int = 16, seed: int = 0
                 ) -> dict:
    """Write the shard tree, ``INDEX.json`` and ``PLAN_MAP.json``."""
    root = Path(root)
    runs = list(default_runs() if runs is None else runs)
    index = dict(schema=INDEX_SCHEMA, runs={}, pairs={}, totals={})
    plan_map = {}
    for i, run in enumerate(runs):
        key = f"{run['chunk']}/{run['world']}__{run['arm']}"
        shard = root / run["chunk"] / run["world"] / run["arm"]
        shard.mkdir(parents=True, exist_ok=True)
        arrs = _arrays(k, int(run["n_ticks"]), int(run["n_dec"]), seed + i)
        for name, a in arrs.items():
            np.save(shard / f"{name}.npy", a)
        (shard / "slot_vid.json").write_text(json.dumps(
            [f"v{j}" for j in range(k)]), encoding="utf-8")
        (shard / "meta.json").write_text(json.dumps(
            dict(chunk=run["chunk"], world=run["world"], arm=run["arm"])),
            encoding="utf-8")
        _outcomes(run, i).to_parquet(shard / "outcomes.parquet", index=False)

        outputs = {name: _sha(shard / f"{name}.npy") for name in V1_NPY_FILES}
        outputs["slot_vid.json"] = _sha(shard / "slot_vid.json")
        outputs["meta.json"] = _sha(shard / "meta.json")
        outputs["outcomes.parquet"] = _sha(shard / "outcomes.parquet")
        total = sum((shard / f"{n}.npy").stat().st_size for n in NPY_FILES)
        done = dict(
            schema=INDEX_SCHEMA, chunk=run["chunk"], world=run["world"],
            arm=run["arm"], split=run["split"], t0_s=run["t0_s"],
            n_ticks=int(run["n_ticks"]), n_slots=k,
            n_dropped_slots=0, n_dropped_slot_steps=0,
            input_sha256=f"in-{key}", shard_dir=str(shard), outputs=outputs,
            total_bytes=int(total), n_outcome_rows=len(TAUS))
        (shard / "DONE.json").write_text(json.dumps(done, indent=1),
                                         encoding="utf-8")
        index["runs"][key] = dict(
            chunk=run["chunk"], world=run["world"], arm=run["arm"],
            split=run["split"], t0_s=run["t0_s"], n_ticks=int(run["n_ticks"]),
            total_bytes=int(total), shard_dir=str(shard),
            input_sha256=f"in-{key}")
        plan_map[key] = dict(chunk=run["chunk"], world=run["world"],
                             arm=run["arm"], split=run["split"],
                             t0_s=run["t0_s"], plan_path=run.get("plan_path"),
                             root_rel=run.get("root_rel", "runs/x"),
                             policy=run.get("policy", run["arm"]))
    (root / "INDEX.json").write_text(json.dumps(index, indent=1),
                                     encoding="utf-8")
    (root / "PLAN_MAP.json").write_text(json.dumps({"runs": plan_map}, indent=1),
                                        encoding="utf-8")
    return index


def write_world_plan(path: Path, worlds) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"worlds": list(worlds)}, indent=1),
                    encoding="utf-8")
    return path


def make_inventory(path: Path, runs) -> Path:
    rows = []
    for i, run in enumerate(runs):
        rows.append(dict(
            batch=run.get("batch", run["chunk"]), chunk=run["chunk"],
            world=run["world"], split=run["split"],
            root_rel=run.get("root_rel", "runs/x"), t0_s=run["t0_s"],
            arm=run["arm"], policy=run.get("policy", run["arm"]),
            policy_source="arm", is_fleet=False,
            schedule_arm=None, duration_s=600.0, present=True,
            receipt_status="complete", is_complete=True,
            physics_rows_written=10, physics_file_bytes=100,
            saved_state_id=None, is_smoke=False, mode="full_instrumented",
            run=run.get("run", f"{run['world']}__{run['arm']}")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
