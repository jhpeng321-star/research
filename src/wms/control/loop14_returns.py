"""LOOP14: real critic returns from collected policy-pair branches; never runs SUMO.

    python -m wms.control.loop14_returns --source <converted c4_episodes> \\
        --output <new .npz> --policy-checkpoint <policy ckpt used for collection> [--kind POLICY]

Opens only the converted numeric arrays and each branch's ``summary.json`` (written by
``traffic_shepherd.c4.policy_pairs``).  The return of one branch is minus its authoritative J
ledger (on-road speed-deficit vehicle-seconds + source-waiting vehicle-seconds over the branch,
summed by the env per physics step): no reward is reconstructed.  ``--kind POLICY`` (default)
gives ``source_kind = current_policy`` targets of the policy that was collected; ``--kind B``
gives native targets of the same states.  Only TRAIN-role worlds are exported.  The NPZ follows
the strict contract of :mod:`wms.control.real_returns14` (and is validated by loading it).
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from .start_states import checked_path

KINDS = {"POLICY": "current_policy", "B": "native"}
SUMMARY_SCHEMA = "c4_loop14_pair_summary/1"


def export_pairs(source, output, *, policy_checkpoint: str, kind: str = "POLICY") -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    root, output = checked_path(source), checked_path(output)
    if output.suffix != ".npz":
        raise ValueError("output must be a new .npz")
    manifest = json.loads(checked_path(root / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
    states, targets, populations, horizons, seeds, receipts = [], [], [], [], [], []
    for entry in manifest["episodes"]:
        branch = entry.get("branch") or {}
        if entry.get("split_role") != "train" or branch.get("kind") != kind:
            continue
        seed = int(entry["seed"])
        if 964000 <= seed <= 964199:
            raise ValueError("excluded traffic seed")
        summary_path = checked_path(checked_path(entry["source"]) / "summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != SUMMARY_SCHEMA or int(summary["seed"]) != seed:
            raise ValueError(f"{summary_path}: not a LOOP14 pair summary of seed {seed}")
        if summary.get("dropped_vehicle_steps", 0) or summary.get("collisions", 0) or summary.get("teleports", 0):
            # An incomplete fleet or a SUMO incident makes the ledger non-authoritative.
            receipts.append({"summary": str(summary_path), "skipped": "dropped/collision/teleport"})
            continue
        episode = checked_path(root / entry["dir"])
        times = np.load(checked_path(episode / "time_s.npy"), allow_pickle=False)
        t = int(np.searchsorted(times, float(summary["warmup_s"]) - 1e-6))
        if t >= len(times) or abs(times[t] - float(summary["warmup_s"])) > 1e-6:
            raise ValueError("no exact branch-start tick")
        kin = np.load(checked_path(episode / "veh_kin.npy"), mmap_mode="r", allow_pickle=False)[t].copy()
        present = np.load(checked_path(episode / "veh_mask.npy"), mmap_mode="r", allow_pickle=False)[t].astype(bool)
        cav = np.load(checked_path(episode / "is_cav.npy"), mmap_mode="r", allow_pickle=False)[t].astype(bool)
        backlog = np.asarray(summary["start_entry_backlog"], np.float32)
        if backlog.shape != (2,) or (backlog < 0).any():
            raise ValueError("start_entry_backlog must be [mainline, ramp] counts")
        lane = np.where(present & (kin[:, 1:4].sum(-1) > 0), kin[:, 1:4].argmax(-1), -1)
        states.append(dict(s_m=kin[:, 0], v_mps=kin[:, 4], a_mps2=kin[:, 5], length_m=kin[:, 6],
                           lane=lane, present=present, is_cav=cav, entry_backlog=backlog))
        ledger = summary["ledger"]
        j = float(ledger["speed_deficit_vehicle_seconds"]) + float(ledger["source_wait_vehicle_seconds"])
        if not np.isfinite(j) or j <= 0:
            raise ValueError(f"{summary_path}: J must be positive finite vehicle-seconds")
        targets.append(-j)
        populations.append(int(summary["requested_population"]))
        horizons.append(float(summary["horizon_s"]))
        seeds.append(seed)
        receipts.append({"summary": str(summary_path),
                         "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                         "pair_key": entry.get("pair_key"), "J": j})
    if not states:
        raise ValueError(f"no eligible TRAIN {kind} branches")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state__" + key: np.stack([s[key] for s in states]) for key in states[0]}
    payload.update(returns=np.array(targets, dtype=np.float64), gamma=np.array(1.),
                   units=np.array("negative_total_vehicle_seconds"), split=np.array("TRAIN"),
                   source_kind=np.array(KINDS[kind]), requested_population=np.array(populations),
                   horizon_s=np.array(horizons), seeds=np.array(seeds, dtype=np.int64),
                   accounting=np.array("source_integral_plus_road_delay"),
                   terminal_semantics=np.array("finite_horizon_charged_to_end"))
    if kind == "POLICY":
        if not str(policy_checkpoint).strip():
            raise ValueError("current_policy targets need the collected policy checkpoint")
        payload["policy_checkpoint"] = np.array(str(policy_checkpoint))
    with output.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    from .real_returns14 import NPZRealReturnProvider
    NPZRealReturnProvider(output, seed=0)          # strict contract check of what was written
    receipt = {"source": str(root), "kind": kind, "source_kind": KINDS[kind],
               "targets": len(states), "policy_checkpoint": policy_checkpoint, "entries": receipts}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-checkpoint", default="")
    parser.add_argument("--kind", choices=sorted(KINDS), default="POLICY")
    args = parser.parse_args(argv)
    receipt = export_pairs(args.source, args.output, policy_checkpoint=args.policy_checkpoint,
                           kind=args.kind)
    print(json.dumps({k: receipt[k] for k in ("kind", "source_kind", "targets")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
