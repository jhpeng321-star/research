"""Export critic targets from existing TRAIN native summaries; never runs SUMO.

Only explicit numeric converted arrays and summary.json are opened. The summary
ledger supplies the exact window road/source integrals (not reconstructed endpoint
queues). One 600-second target per TRAIN world is a warm-start, not current-policy
value evidence. This small diagnostic source must not become the production corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from .start_states import checked_path


def export(source, output):
    root, output = checked_path(source), checked_path(output)
    manifest_path = checked_path(root / "C4_EPISODES_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    states, targets, populations, horizons, seeds, receipts = [], [], [], [], [], []
    for entry in manifest["episodes"]:
        if entry.get("split_role") != "train" or entry.get("controller") != "native":
            continue
        seed = int(entry["seed"])
        if 964000 <= seed <= 964199:
            raise ValueError("excluded traffic seed")
        episode = checked_path(root / entry["dir"])
        summary_path = checked_path(checked_path(entry["source"]) / "summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["seed"] != seed or summary["controller"] != "native" or summary["dropped_vehicle_steps"]:
            raise ValueError("native summary provenance/complete fleet mismatch")
        times = np.load(checked_path(episode / "time_s.npy"), allow_pickle=False)
        t = int(np.searchsorted(times, summary["warmup_s"]))
        if times[t] != summary["warmup_s"]:
            raise ValueError("no exact window-start tick")
        kin = np.load(checked_path(episode / "veh_kin.npy"), mmap_mode="r", allow_pickle=False)[t].copy()
        present = np.load(checked_path(episode / "veh_mask.npy"), mmap_mode="r", allow_pickle=False)[t].astype(bool)
        cav = np.load(checked_path(episode / "is_cav.npy"), mmap_mode="r", allow_pickle=False)[t].astype(bool)
        backlog = np.load(checked_path(episode / "entry_backlog.npy"), mmap_mode="r", allow_pickle=False)[t].copy()
        lane = np.where(present & (kin[:, 1:4].sum(-1) > 0), kin[:, 1:4].argmax(-1), -1)
        states.append(dict(s_m=kin[:, 0], v_mps=kin[:, 4], a_mps2=kin[:, 5],
                           length_m=kin[:, 6], lane=lane, present=present, is_cav=cav,
                           entry_backlog=backlog))
        window = summary["window"]
        targets.append(-(window["speed_deficit_vehicle_seconds"] + window["source_wait_vehicle_seconds"]))
        populations.append(summary["population"])
        horizons.append(summary["horizon_s"])
        seeds.append(seed)
        receipts.append({"summary": str(summary_path), "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                         "road_delay": window["speed_deficit_vehicle_seconds"],
                         "source_wait": window["source_wait_vehicle_seconds"],
                         "remaining_uninserted_due": summary["episode"]["uninserted_due"]})
    if not states:
        raise ValueError("no eligible TRAIN native summaries")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state__" + key: np.stack([s[key] for s in states]) for key in states[0]}
    payload.update(returns=np.array(targets, dtype=np.float64), gamma=np.array(1.),
                   units=np.array("negative_total_vehicle_seconds"), split=np.array("TRAIN"),
                   source_kind=np.array("native"), requested_population=np.array(populations),
                   horizon_s=np.array(horizons), seeds=np.array(seeds),
                   accounting=np.array("source_integral_plus_road_delay"),
                   terminal_semantics=np.array("finite_horizon_charged_to_end"))
    with output.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    receipt = {"source": str(root), "targets": len(states), "source_kind": "native",
               "off_policy_warm_start_only": True, "entries": receipts}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output)))
