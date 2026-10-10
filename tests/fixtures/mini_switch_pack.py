"""Workspace-only synthetic kinematics, optionally bound to real TRAIN sidecars.

No physical outcome/trajectory source is read. Copied small TRAIN identities and
timeline index can seed this fixture; issued timelines and kinematics are synthetic.
Every destination must be new; no deletion, external paths, or existing-pack edits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wms.data.mixdata import direct_family

WORKSPACE = Path(__file__).resolve().parents[2]
WORLD = "w8993m2400p20"
CONTROL_STEPS, K = 120, 192


def available():
    return True


def _write(path, doc):
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def build(dst: Path, *, ticks=(240, 250, 260, 270, 280, 290), arm="direct-mix",
          held_family=None, copied_meta: Path | None = None, original=False, k=192):
    from packsynth import write_pack

    dst = Path(dst).resolve()
    dst.relative_to(WORKSPACE)
    if dst.exists():
        raise ValueError(f"fixture destination exists: {dst}")
    dst.mkdir(parents=True)
    K = int(k)
    world = WORLD
    ids = np.arange(K, dtype=np.int64)
    vmap = [["hetdata", world, f"mainline.{s}"] for s in range(K)]
    if copied_meta is not None:
        copied_meta = Path(copied_meta).resolve()
        copied_meta.relative_to(WORKSPACE)
        index = json.loads((copied_meta / "ACTION_TIMELINE_INDEX_train.json").read_text())
        first = index["items"][0]
        world, arm = first["world"], first["arm"]
        ident = copied_meta / "identities_switching_train"
        rows = np.load(ident / "identities.npy", mmap_mode="r", allow_pickle=False)
        ids = np.asarray(rows[0]).copy()
        vmap = json.loads((ident / "vehicle_map.json").read_text())
    ids = ids[:K]
    family = 3 if held_family is None and copied_meta is not None else held_family
    seed = int(world.split("m")[0][1:])
    candidates = [s for s in range(K) if ids[s] >= 0]
    if family is None:
        family = 3
    cav = [s for s in candidates if direct_family(seed, vmap[int(ids[s])][2]) != family][:2]
    if not cav:
        raise ValueError("no non-held-family CAV identity for synthetic fixture")
    if original:
        arm = "native"
    pack = dst / ("pack_original_train" if original else "pack_switching_train")
    identity = dst / "identities_switching_train"
    manifest = write_pack(pack, [{"chunk": "hetdata", "batch": "hetdata", "world": world,
        "arm": arm, "policy": arm, "split": "TRAIN", "t0_s": 0.0}], k=K, n_ticks=600)
    veh = np.lib.format.open_memmap(pack / "veh_step.npy", mode="r+")
    veh[:, :, 7:14] = 0
    veh[:, :, 12] = 1
    veh[:, cav, 7] = 1
    veh[:, cav, 8] = 1
    veh[:, :, 0] += np.arange(600)[:, None] * 4.0
    veh.flush()
    for name in ("veh_mask", "target_mask"):
        arr = np.lib.format.open_memmap(pack / f"{name}.npy", mode="r+")
        arr[:, ids < 0] = 0
        arr.flush()
    for name, shape in (("veh_kin", (600, K, 7)), ("hdv_base", (600, K, 6)),
                        ("cav_plan_ch", (600, K, 8)), ("event_tok", (600, 8, 8)),
                        ("event_mask", (600, 8))):
        arr = np.lib.format.open_memmap(pack / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
        arr[:] = 0
        if name == "veh_kin":
            for t in range(600):
                arr[t] = veh[t, :, :7]
        arr.flush()
        manifest["arrays"][name] = {"file": f"{name}.npy", "shape": list(shape), "dtype": "float32"}
    base = manifest["items"][0]
    manifest["items"] = [{**base, "tick": int(t)} for t in ticks]
    _write(pack / "manifest.json", manifest)
    _write(pack / "INDEX.json", {"schema": "wm_n1_corpus_pack/2", "runs": [{
        "batch": "hetdata", "chunk": "hetdata", "world": world, "arm": arm, "policy": arm,
        "split": "TRAIN", "t0_s": 0.0, "tick_start": 0, "tick_end": 600,
        "n_ticks": 600, "key": f"hetdata/{world}__{arm}", "n_outcome_rows": 6,
        "shard_dir": f"hetdata/{world}/{arm}", "outcome_first": 0}]})
    identity.mkdir()
    np.save(identity / "identities.npy", np.broadcast_to(ids, (len(ticks), K)))
    np.save(identity / "global_positions.npy", np.arange(len(ticks), dtype=np.int64))
    _write(identity / "vehicle_map.json", vmap)
    _write(identity / "TRAIN_IDENTITY_RECEIPT.json", {
        "schema": "wm_n1_train_identity_sidecar/1", "split": "TRAIN", "n_items": len(ticks),
        "files": {n: hashlib.sha256((identity / n).read_bytes()).hexdigest() for n in
                  ("identities.npy", "global_positions.npy", "vehicle_map.json")}})
    ctrl = np.zeros((len(ticks), CONTROL_STEPS, K), np.int8)
    acc, tgt = np.zeros(ctrl.shape, np.float32), np.zeros(ctrl.shape, np.float32)
    facts = []
    for pos, tick in enumerate(ticks):
        if pos == 0:
            ctrl[pos, :, cav] = 1
        elif pos >= 2:
            for slot in cav:
                ctrl[pos, :10, slot] = 1
                ctrl[pos, 10:25, slot] = 3
                acc[pos, 10:25, slot] = .4
                tgt[pos, 10:25, slot] = 18
        facts.append({"position": pos, "tick": tick, "cav_slots": cav})
    for name, arr in (("controller_timeline", ctrl), ("issued_acceleration_mps2", acc),
                      ("issued_target_speed_mps", tgt)):
        np.save(pack / f"{name}.npy", arr)
    _write(dst / "ACTION_TIMELINE_INDEX_train.json", {"schema": "hetpack_switching_actions/1",
        "split": "TRAIN", "control_dt_s": 1.0, "control_steps": CONTROL_STEPS, "k_max": K,
        "controllers": ["native", "damping", "pi-external-u", "direct"], "n_items": len(ticks),
        "items": [{"position": p, "world": world, "arm": arm, "tick": int(t),
                   "valid_steps": CONTROL_STEPS} for p, t in enumerate(ticks)]})
    _write(dst / "windows_train_hetdata.json", {"schema": "wm_s_lpo_windows/1", "split": "TRAIN",
        "n_windows": 1, "windows": [{"batch": "hetdata", "world": world, "arm": arm,
            "key": f"hetdata/{world}__{arm}", "pack": pack.name, "t0_s": 0.0}]})
    if copied_meta is not None:
        held = json.loads((copied_meta / "ZERO_SHOT_HELDOUT.json").read_text())
    else:
        w = "w8859m2700p10"
        held = {"schema": "hetpack_switching_heldout/1", "held_family": family,
            "held_worlds": [w], "n_held_worlds": 1, "worlds": {w: {"n_families": 4}},
            "overlap_guard": {"overlap_guard_ok": True, "held_subset_of_test": True,
                              "disjoint": True, "problems": []}}
    _write(dst / "ZERO_SHOT_HELDOUT.json", held)
    return {"root": str(dst), "pack": str(pack), "identities": str(identity),
            "index": str(dst / "ACTION_TIMELINE_INDEX_train.json"),
            "windows": str(dst / "windows_train_hetdata.json"),
            "heldout": str(dst / "ZERO_SHOT_HELDOUT.json"), "facts": facts,
            "held_family": family, "world": world}
