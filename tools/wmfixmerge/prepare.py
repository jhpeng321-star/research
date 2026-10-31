"""Explicit TRAIN-only preparation; no model, GPU or simulator required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.wmfixmerge.data import (COMPONENTS, LEDGER_ARRAYS, digest, episode_arrays,
                                    install_guard, safe, select)


def prepare(root, out):
    import numpy as np
    from tools.wmrequal.labels import departures, recorded

    root, out = safe(root), safe(out)
    entries = select(root)  # Check roles and paths BEFORE opening any tape.
    tape_sets = {tuple(episode_arrays(root, row)) for row in entries}
    if len(tape_sets) != 1:
        raise ValueError("ledger array presence must be corpus-uniform")
    ledger = set(tape_sets.pop()) >= set(LEDGER_ARRAYS)
    files, labels = {}, []
    for row in entries:
        p = safe(root / row["dir"])
        # Hash exactly the observable arrays present (labels / production loader).
        for name in episode_arrays(root, row):
            path = safe(p / (name + ".npy"))
            files[str(path)] = digest(path)
        label = recorded(root, row, 300)
        if not label["source_available"] or label["queue0"] is None:
            raise ValueError("exact per-source Q0 is unavailable")
        labels.append(dict(pair_key=row["pair_key"], world=row["world"], arm=row["arm"],
                           native=row.get("branch", {}).get("kind") == "B",
                           seed=int(row["seed"]), demand=row["demand_vph"],
                           penetration=row["penetration"], **label))
        load = lambda name: np.load(safe(p / (name + ".npy")), mmap_mode="r", allow_pickle=False)
        times = load("time_s")
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), .5, atol=1e-6, rtol=0):
            raise ValueError("nonuniform 0.5s tape clock")
        tick = int(np.searchsorted(times, 300))
        if len(times) < tick + 601:
            raise ValueError("tape lacks the full 600-tick TRAIN window")
        # Identity-counted insertions per tick, exactly as the recorded labels.
        ids, mask, kin = load("vid"), load("veh_mask").astype(bool), load("veh_kin")
        latest = int(ids[:tick + 1][mask[:tick + 1]].max())
        increments, born_rows = [], []
        for t in range(tick + 1, tick + 601):
            born = np.flatnonzero(mask[t] & (ids[t] > latest))
            inserted = np.zeros(2, dtype=np.int64)
            for j in born:
                inserted[int(kin[t, j, 1:4].argmax() == 2)] += 1
            if len(born):
                latest = int(ids[t, born].max())
            born_rows.append(inserted)
        if ledger:
            q0 = load("entry_backlog")[tick]
            if not np.array_equal(q0, label["queue0"]):
                raise ValueError("recorded request ledger disagrees with independent Q0 reconstruction")
            q = load("entry_backlog")[tick:tick + 600]
            if not np.isfinite(q).all() or (q < 0).any() or not np.array_equal(q, np.rint(q)):
                raise ValueError("invalid recorded entrance backlog")
            for h in (60, 90, 120, 180, 300):
                if float(.5 * q[:h * 2].sum()) != label["values"][str(h)]["source"]:
                    raise ValueError("request ledger/source rectangle mismatch")
            # Observable due increments = backlog change + identity-counted insertions.
            queue = load("entry_backlog")
            for t, inserted in zip(range(tick + 1, tick + 601), born_rows):
                due = queue[t] - queue[t - 1] + inserted
                if (due < 0).any() or not np.array_equal(due, np.rint(due)):
                    raise ValueError("invalid recorded request increment")
                increments.append(due.astype(int).tolist())
        else:
            # S-E3DATA FIT tapes carry no ledger: rebuild the exact same quantities
            # from the deterministic request schedule and the counter Q0, i.e. the
            # identical arithmetic tools/wmrequal/labels.py recorded() supervises on.
            counter0 = load("counter")[tick].sum(-1).astype(np.int64)
            if int(counter0.sum()) != int(ids[:tick + 1][mask[:tick + 1]].max()) + 1:
                raise ValueError("counter/registry disagreement at t300")
            dep = [departures(int(row["seed"]), row["demand_vph"], "mainline"),
                   departures(int(row["seed"]), 400, "ramp")]
            window = np.asarray(times[tick:tick + 601])
            due_cum = np.stack([np.searchsorted(d, window, side="right") for d in dep], -1)
            inserted_cum = np.zeros((601, 2), np.int64)
            inserted_cum[0] = counter0
            inserted_cum[1:] = counter0 + np.cumsum(np.stack(born_rows), 0)
            queue_series = due_cum - inserted_cum
            if queue_series[0].tolist() != list(label["queue0"]):
                raise ValueError("schedule Q0 disagrees with recorded label Q0")
            if (queue_series < 0).any() or not np.array_equal(queue_series, np.rint(queue_series)):
                raise ValueError("negative/non-integer reconstructed TRAIN queue")
            for h in (60, 90, 120, 180, 300):
                if float(.5 * queue_series[:h * 2].sum()) != label["values"][str(h)]["source"]:
                    raise ValueError("schedule/label source rectangle mismatch")
            increments = (due_cum[1:] - due_cum[:-1]).astype(int).tolist()
        labels[-1]["due_increments"] = increments
    for key in {row["pair_key"] for row in labels}:
        group = [r for r in labels if r["pair_key"] == key]
        ref = next(r for r in group if r["native"])
        if any(r["queue0"] != ref["queue0"] or r["demand_rate"] != ref["demand_rate"]
               for r in group):
            raise ValueError("paired starts do not share queues/demand")
        if any(r["due_increments"] != ref["due_increments"] for r in group):
            raise ValueError("paired recorded request schedules differ")
        paired = [e for e in entries if e["pair_key"] == key]
        # Bind the complete observed prefix, not just position at a single tick.
        prefix_names = ("time_s", "veh_kin", "vid", "veh_mask", "is_cav", "counter") \
            + (("entry_backlog", "request_counter", "entry_cav_counter") if ledger else ())
        for name in prefix_names:
            prefixes = []
            for entry in paired:
                p = safe(root / entry["dir"])
                t = np.load(safe(p / "time_s.npy"), allow_pickle=False)
                a = np.load(safe(p / (name + ".npy")), mmap_mode="r", allow_pickle=False)
                prefixes.append(a[:int(np.searchsorted(t, 300)) + 1])
            if any(not np.array_equal(prefixes[0], a) for a in prefixes[1:]):
                raise ValueError("paired observable histories differ: " + name)
    contrasts = []
    widths = []
    for key in sorted({r["pair_key"] for r in labels}):
        group = [r for r in labels if r["pair_key"] == key]
        ref = next(r for r in group if r["native"])
        errors = [[r["values"][str(h)][c] - ref["values"][str(h)][c]
                   for c in COMPONENTS] for r in group if not r["native"]
                  for h in (60, 120, 300)]
        contrasts.append(np.square(errors).mean(0))
    for row in entries:
        p = safe(root / row["dir"])
        kin = np.load(safe(p / "veh_kin.npy"), mmap_mode="r", allow_pickle=False)
        mask = np.load(safe(p / "veh_mask.npy"), mmap_mode="r", allow_pickle=False).astype(bool)
        ids = np.load(safe(p / "vid.npy"), mmap_mode="r", allow_pickle=False)
        d = np.diff(kin[..., 0], axis=0)
        ok = mask[1:] & mask[:-1] & (ids[1:] == ids[:-1]) & (d > 0)
        if ok.any():
            widths.append(float(np.median(d[ok])))
    if not widths:
        raise ValueError("no identity-continuous positive TRAIN displacement for crossing width")
    scales = np.maximum(np.sqrt(np.mean(contrasts, axis=0)), np.finfo(np.float32).eps)
    # WMFIX_REVIEW item 7: anchors are scaled by their own TRAIN absolute-value
    # RMS, never by the paired-contrast RMS (which would inflate the anchor term
    # by the contrast/level ratio and let common-mode level fits dominate).
    levels = {3: [], 4: []}
    for row in labels:
        for h, cell in row["values"].items():
            levels[3].append(float(cell["discharge"]))
            levels[4].append(float(cell["road"]))
    anchor_scales = np.ones_like(scales)
    for column, values in levels.items():
        if not values:
            raise ValueError("TRAIN anchor column has no rows")
        anchor_scales[column] = max(float(np.sqrt(np.mean(np.square(values)))),
                                    np.finfo(np.float32).eps)
    doc = dict(schema="wmfix_train/1", root=str(root),
               seeds=sorted({int(row["seed"]) for row in entries}), episodes=entries,
               manifest_sha256=digest(root / "C4_EPISODES_MANIFEST.json"),
               split_sha256=digest(root / "SPLIT.json"), files=files, labels=labels,
               selection="TRAIN only", source_gradient="hard service; zero off events",
               ledger_checked=ledger)
    doc.update(components=list(COMPONENTS), scales=scales.tolist(),
               anchor_scales=anchor_scales.tolist(),
               crossing_width=float(np.median(widths)), horizon_seconds=[60, 120, 300])
    if not out.parent.is_dir():
        raise ValueError("output parent must already exist")
    with out.open("x") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    install_guard()
    prepare(args.root, args.out)
