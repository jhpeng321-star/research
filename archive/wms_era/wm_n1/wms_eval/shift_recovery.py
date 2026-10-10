"""OLDHIST driver-parameter recovery: original (~1-tick) vs shifted (10-step) history.

Runs the WM-S2 ``driver_params`` encoder on two versions of the same 318 C1 v2
states and compares the recovered per-vehicle parameters against the offline
vType labels:

* ``original`` -- ``E:/rd-wt/ws-evalh/_out/c1_states_v2hist`` (materialised at the
  stored t0, where the pack window begins ~1 tick before t0, so the encoder sees
  ~1 valid history step);
* ``shifted``  -- ``_out/c1_states_v2shift`` (t0 moved 5 s forward inside the same
  stored window, so the 10-step history is real recorded motion).

The offline labels are the deterministic scenario schedule replay from
``ws-errdecomp/_out/error_decomp.py`` (``recover_types``): vehicle id -> one of
``aggressive_car``/``normal_car``/``cautious_car``.  ``nearest-vType`` maps the
recovered ``(time_headway, min_gap, accel, decel)`` to the closest vType centre.
One state is processed at a time; no pack is opened.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from statistics import median

import numpy as np

VType_CENTRES = {
    "aggressive_car": (0.95, 1.5, 3.0, 4.5),
    "normal_car": (1.10, 2.5, 2.6, 4.5),
    "cautious_car": (1.50, 3.5, 2.0, 3.0),
}
CLASSES = tuple(VType_CENTRES)
#: encoder parameter indices used for the nearest-vType decision
IDX = {"time_headway": 1, "min_gap": 2, "accel": 3, "decel": 4}
WORLD_RE = re.compile(r"^w(\d+)m(\d+)p(\d+)t(\d+)$")


# --------------------------------------------------------------- offline labels
def _schedule_main(seed: int, mainline_vph: int, stream: str) -> dict:
    """Exact replay of the LPO scenario schedule (see error_decomp.recover_types)."""
    rng = random.Random(f"damping_requests_v1:{seed}:{stream}")
    rates = ([(0, 1200, mainline_vph)]
             + [(1200 + 100 * i, 1300 + 100 * i,
                 round(mainline_vph + (1500 - mainline_vph) * (i + 1) / 3))
                for i in range(3)]
             + [(1500, 1800, 1500)])
    if stream == "ramp":
        rates = [(0, 1800, 400)]
    ordinal, out = 0, {}
    for begin, end, rate in rates:
        t = float(begin)
        while True:
            t += rng.expovariate(rate / 3600)
            if t >= end:
                break
            identity = f"{stream}.{ordinal}"
            draw = random.Random(f"damping_type_v1:{seed}:{identity}").random()
            out[identity] = ("aggressive_car" if draw < 0.1
                             else "normal_car" if draw < 0.8 else "cautious_car")
            ordinal += 1
    return out


def recover_types(world: str) -> dict:
    match = WORLD_RE.match(str(world))
    if not match:
        return {}
    seed, demand = int(match.group(1)), int(match.group(2))
    out: dict = {}
    for stream in ("mainline", "ramp"):
        out.update(_schedule_main(seed, demand, stream))
    return out


def nearest_vtype(values: np.ndarray) -> str:
    point = np.asarray([values[IDX[k]] for k in ("time_headway", "min_gap",
                                                 "accel", "decel")], float)
    return min(CLASSES, key=lambda c: float(np.linalg.norm(
        point - np.asarray(VType_CENTRES[c], float))))


# ------------------------------------------------------------- encoder on states
def _valid_steps(history: dict) -> float:
    kin = np.asarray(history.get("kin", []))
    mask = np.asarray(history.get("mask", []))
    target = np.asarray(history.get("target_mask", []))
    if kin.ndim != 3 or kin.shape[0] == 0:
        return 0.0
    t, k = kin.shape[:2]
    same = np.ones((t, k), bool)
    if target.size:
        for j in range(t - 2, -1, -1):
            same[j] = same[j + 1] & target[j]
    return float(np.median((same & mask.astype(bool)).sum(0)))


def run_set(paths, checkpoint, physics_path, *, arm_of_set):
    from .wms_plugin import WmsCheckpointModel
    from .c1v2 import iter_windows

    physics = json.loads(Path(physics_path).read_text(encoding="utf-8")) \
        if physics_path else None
    holder = WmsCheckpointModel(checkpoint, physics=physics)
    model = holder.model or holder._load()
    import torch

    values_all, labels_all, valid_steps, keys = [], [], [], []
    for set_name, path in paths:
        arm = arm_of_set[set_name]
        for record in iter_windows(path):
            history = record.get("history") or {}
            valid_steps.append(_valid_steps(history))
            t0_s, ids, kin, is_cav = holder._inputs(record, arm)
            dh, dhm = holder._driver_history(record, ids)
            if dh is None:
                continue
            initial = {
                "veh_kin": torch.as_tensor(kin[None]),
                "present": torch.ones((1, len(ids)), dtype=torch.bool),
                "is_cav": torch.as_tensor(is_cav[None]),
            }
            with torch.no_grad():
                res = model(initial, k_steps=0, history=dh, history_mask=dhm,
                            driver_history=dh, driver_history_mask=dhm)
            params = res["driver_parameters"][0].numpy()
            types = recover_types(record.get("world"))
            for slot, vid in enumerate(ids):
                label = types.get(str(vid))
                if label is None:
                    continue
                values_all.append(params[slot])
                labels_all.append(label)
                keys.append((str(record.get("world")), str(vid)))
    return {"values": np.asarray(values_all, float),
            "labels": labels_all,
            "keys": keys,
            "valid_steps_median": (median(valid_steps) if valid_steps else 0.0)}


def summarise(result: dict) -> dict:
    values, labels = result["values"], result["labels"]
    confusion = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    correct = 0
    for value, label in zip(values, labels):
        pred = nearest_vtype(value)
        confusion[label][pred] += 1
        correct += int(pred == label)
    n = len(labels)
    means = {}
    for name, idx in IDX.items():
        means[name] = float(np.mean(values[:, idx])) if n else None
    base = {c: sum(1 for x in labels if x == c) / n for c in CLASSES} if n else {}
    return {
        "n_typed": n,
        "n_correct": correct,
        "accuracy": (correct / n) if n else None,
        "normal_base_rate": base.get("normal_car"),
        "confusion": confusion,
        "recovered_means_tau_min_gap_accel_decel": means,
        "valid_history_steps_median": result["valid_steps_median"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="shift_recovery")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--physics", default=None)
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--shifted-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    names = [("u_same", "unseen_u_same_world.json"), ("u_new", "unseen_u_new_world.json"),
             ("seen", "seen.json")]
    arm_of = {"u_same": "pi", "u_new": "pi", "seen": "native"}
    original = [(n, Path(args.original_dir) / f) for n, f in names]
    shifted = [(n, Path(args.shifted_dir) / f"{n}.json") for n, _ in names]
    orig_run = run_set(original, args.checkpoint, args.physics, arm_of_set=arm_of)
    shift_run = run_set(shifted, args.checkpoint, args.physics, arm_of_set=arm_of)
    # paired subset: the same (world, vehicle) identity recovered in both runs
    orig_pred = {k: nearest_vtype(v) for k, v in zip(orig_run["keys"], orig_run["values"])}
    shift_pred = {k: nearest_vtype(v) for k, v in zip(shift_run["keys"], shift_run["values"])}
    label_of = dict(zip(shift_run["keys"], shift_run["labels"]))
    label_of.update(dict(zip(orig_run["keys"], orig_run["labels"])))
    paired = sorted(set(orig_run["keys"]) & set(shift_run["keys"]))
    paired_stats = {
        "n": len(paired),
        "original_accuracy": (sum(orig_pred[k] == label_of[k] for k in paired) / len(paired)
                              if paired else None),
        "shifted_accuracy": (sum(shift_pred[k] == label_of[k] for k in paired) / len(paired)
                             if paired else None),
        "original_predicted_class_counts": {c: sum(1 for k in paired if orig_pred[k] == c)
                                            for c in CLASSES},
        "shifted_predicted_class_counts": {c: sum(1 for k in paired if shift_pred[k] == c)
                                           for c in CLASSES},
    }
    report = {
        "schema": "wms_eval_shift_recovery/1",
        "checkpoint": args.checkpoint,
        "parameter_names": ["v0", "time_headway", "min_gap", "accel", "decel",
                            "politeness", "mobil_threshold", "merge_gap", "merge_urgency"],
        "nearest_vtype_axes": ["time_headway", "min_gap", "accel", "decel"],
        "vtype_centres": VType_CENTRES,
        "original": summarise(orig_run),
        "shifted": summarise(shift_run),
        "paired_same_vehicle": paired_stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: {"n": report[k]["n_typed"], "acc": report[k]["accuracy"],
                          "valid": report[k]["valid_history_steps_median"]}
                      for k in ("original", "shifted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
