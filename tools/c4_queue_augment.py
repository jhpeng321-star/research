"""V9ENTRY: add the entrance-queue arrays to an ALREADY-converted C4 source (new directory).

Use when the collected episode directories (``episode.npz`` + ``scenario/requests.rou.xml``) are
no longer on the host, so ``python -m wms.data.c4_episodes`` cannot re-convert them.  The C4 request
file is a deterministic function of (seed, demand, penetration):
``traffic_shepherd.scenarios.request_schedule(seed, demand, penetration=p, role="exposed")``
(``traffic_shepherd.c4.env`` builds every C4 scenario that way; departures, routes and CAV
identities do not depend on ``population`` / ``cav_vtype``).  The tool

1. regenerates the requests of every manifest entry;
2. recovers each converted vehicle's first occupied tick and entrance from the converted arrays
   (lane at that tick from the ``veh_kin`` lane one-hot; ramp = lane 2), and REFUSES the episode
   unless ``episode_counter`` with that lane reproduces the stored ``counter`` bit-exactly;
3. maps vehicles to request names per entrance in first-seen order (FIFO; within one tick the
   order is irrelevant for the arrays) and REFUSES unless every mapped request departed no later
   than its vehicle's first tick, no entrance inserts more vehicles than it has requests, and the
   CAV count of every (entrance, tick) insertion group equals that of its mapped requests;
4. calls the branch's own ``episode_queue_arrays`` / ``cav_entry_counter`` (the converter's
   functions) and writes ``entry_backlog``, ``request_counter``, ``entry_cav_counter``.

The output directory is NEW (refused if it exists).  Unchanged arrays are symlinked to the
source episode files (never copied, never written); ``episode.json`` and the source manifest are
rewritten with a ``queue_arrays`` record.  ``--compare DIR``: compare the arrays against a
source converted by the real converter (raw episodes with request files) and exit.
Evaluation seeds 964000-964199 are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from wms.data.c4_episodes import (COUNTER_BIN_S, MANIFEST_NAME, QUEUE_ARRAYS, RAMP_LANE,
                                  cav_entry_counter, episode_counter, episode_queue_arrays,
                                  read_source_manifest)

EVAL_SEEDS = range(964000, 964200)
METHOD = "c4_queue_augment/1: regenerated request_schedule(role=exposed) + FIFO name map"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def regenerated_requests(seed: int, demand: int, penetration: float):
    from traffic_shepherd.scenarios import request_schedule

    reqs = request_schedule(int(seed), int(demand), penetration=float(penetration), role="exposed")
    names = [r.vehicle_id for r in reqs]
    depart = np.asarray([r.depart_s for r in reqs], np.float64)
    entrance = np.asarray([1 if "ramp" in r.route.lower() else 0 for r in reqs], np.int64)
    is_cav = np.asarray([bool(r.is_cav) for r in reqs], bool)
    return names, depart, entrance, is_cav


def lane_proxy(kin: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Lane index from the ``veh_kin`` one-hot (columns 1..3); -1 absent / junction."""
    onehot = np.asarray(kin[..., 1:4]) > 0.5
    lane = np.where(onehot.any(-1), onehot.argmax(-1), -1).astype(np.int64)
    return np.where(present, lane, -1)


def first_entries(time_s, vid, present, lane):
    """{vid: (first time, entrance)} exactly as ``episode_counter`` detects them."""
    first = {}
    for t in range(vid.shape[0]):
        for v, ln in zip(vid[t][present[t]].tolist(), lane[t][present[t]].tolist()):
            if v >= 0 and v not in first:
                first[v] = (float(time_s[t]), 1 if ln == RAMP_LANE else 0)
    return first


def augment_episode(ep_dir: Path, entry: dict):
    """(arrays, stats) for one converted episode; raises ValueError on any failed check."""
    seed = int(entry["seed"])
    if seed in EVAL_SEEDS:
        raise ValueError(f"{ep_dir.name}: evaluation seed {seed} refused")
    if float(entry.get("t_start_s", 0.0)) != 0.0 or entry.get("branch"):
        raise ValueError(f"{ep_dir.name}: branch / late-start episodes are not supported")
    load = lambda name: np.load(ep_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)  # noqa: E731
    time_s = np.asarray(load("time_s"), np.float64)
    vid = np.asarray(load("vid"))
    present = np.asarray(load("veh_mask")).astype(bool)
    is_cav = np.asarray(load("is_cav")).astype(bool)
    lane = lane_proxy(load("veh_kin"), present)
    counter = np.asarray(load("counter"))
    # (2) the lane proxy must reproduce the converter's insertion counter bit-exactly
    if not np.array_equal(episode_counter(time_s, vid, present, lane), counter):
        raise ValueError(f"{ep_dir.name}: counter not reproduced from the lane one-hot")
    names, depart, entrance, req_cav = regenerated_requests(seed, entry["demand_vph"],
                                                            entry["penetration"])
    first = first_entries(time_s, vid, present, lane)
    cav_of = {}
    for v in np.unique(vid[present & is_cav]).tolist():
        cav_of[int(v)] = True
    n_vid = int(vid[present].max()) + 1 if present.any() else 0
    vehicle_ids = [f"__unmapped.{v}" for v in range(n_vid)]
    bad_depart = bad_cav = 0
    for e in (0, 1):
        req_idx = np.flatnonzero(entrance == e)            # depart order (schedule is sorted)
        vids = sorted((ts, v) for v, (ts, ent) in first.items() if ent == e)
        if len(vids) > len(req_idx):
            raise ValueError(f"{ep_dir.name}: entrance {e} inserts {len(vids)} > "
                             f"{len(req_idx)} requests")
        groups: dict[float, list[int]] = {}
        for k, (ts, v) in enumerate(vids):
            j = int(req_idx[k])
            vehicle_ids[v] = names[j]
            if depart[j] > ts + 1e-9:
                bad_depart += 1
            groups.setdefault(ts, []).append(k)
        for ts, ks in groups.items():
            n_cav_veh = sum(cav_of.get(vids[k][1], False) for k in ks)
            n_cav_req = int(req_cav[req_idx[ks]].sum())
            bad_cav += int(n_cav_veh != n_cav_req)
    if bad_depart or bad_cav:
        raise ValueError(f"{ep_dir.name}: FIFO map check failed (depart {bad_depart}, "
                         f"CAV groups {bad_cav})")
    queue, stats = episode_queue_arrays(time_s, vid, present, vehicle_ids, names, depart, entrance)
    queue["entry_cav_counter"] = cav_entry_counter(time_s, vid, present, is_cav, lane)
    if (queue["entry_backlog"] < 0).any():
        raise ValueError(f"{ep_dir.name}: negative backlog")
    stats.update({"n_vehicles": len(first), "n_cav_vehicles": len(cav_of),
                  "n_cav_requests": int(req_cav.sum()),
                  "backlog_max": [float(x) for x in queue["entry_backlog"].max(0)],
                  "backlog_t300": [float(x) for x in queue["entry_backlog"][
                      int(np.searchsorted(time_s, 300.0))]] if time_s[-1] >= 300.0 else None})
    return queue, stats


def run(src: Path, out: Path, limit: int | None = None) -> dict:
    manifest = read_source_manifest(src)
    if out.exists():
        raise FileExistsError(f"refusing to write into existing {out}")
    entries = manifest["episodes"][:limit] if limit else manifest["episodes"]
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"{tmp} exists (a previous attempt); remove it yourself first")
    (tmp / "episodes").mkdir(parents=True)
    new_entries, totals = [], {"n_requests": 0, "not_inserted_by_end": 0}
    for i, entry in enumerate(entries):
        ep_src = (src / entry["dir"]).resolve()
        queue, stats = augment_episode(ep_src, entry)
        ep_out = tmp / entry["dir"]
        ep_out.mkdir(parents=True)
        for f in sorted(ep_src.iterdir()):
            if f.name == "episode.json" or f.stem in QUEUE_ARRAYS:
                continue
            os.symlink(f, ep_out / f.name)
        for key, value in queue.items():
            np.save(ep_out / f"{key}.npy", value, allow_pickle=False)
        record = {"source": "regenerated", "method": METHOD, **stats}
        new_entry = {**entry, "queue_arrays": record}
        (ep_out / "episode.json").write_text(json.dumps(new_entry, indent=1, sort_keys=True),
                                             encoding="utf-8")
        new_entries.append(new_entry)
        totals["n_requests"] += stats["n_requests"]
        totals["not_inserted_by_end"] += stats["not_inserted_by_end"]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(entries)}", flush=True)
    doc = {**manifest, "episodes": new_entries, "n_episodes": len(new_entries),
           "n_train": sum(e.get("split_role") == "train" for e in new_entries),
           "n_val": sum(e.get("split_role") == "val" for e in new_entries),
           "queue_augment": {"method": METHOD, "source_dir": str(src),
                             "source_manifest_sha256": _sha256(src / MANIFEST_NAME),
                             "counter_bin_s": COUNTER_BIN_S, **totals}}
    (tmp / MANIFEST_NAME).write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    os.rename(tmp, out)
    return doc


def compare(src: Path, reference: Path) -> int:
    """Queue arrays of ``src`` episodes recomputed by this tool vs ``reference`` (real converter)."""
    manifest = read_source_manifest(reference)
    worst = 0
    for entry in manifest["episodes"]:
        ref_dir = reference / entry["dir"]
        queue, stats = augment_episode((src / entry["dir"]).resolve(), entry)
        same = {k: bool(np.array_equal(queue[k], np.load(ref_dir / f"{k}.npy"))) for k in queue}
        # the request file of the raw episode vs the regenerated schedule
        from wms.data.c4_episodes import read_requests
        raw = Path(entry["source"]) / "scenario" / "requests.rou.xml"
        req_same = None
        if raw.is_file():
            n1, d1, e1 = read_requests(raw)
            n2, d2, e2, _ = regenerated_requests(entry["seed"], entry["demand_vph"],
                                                 entry["penetration"])
            req_same = (n1 == n2 and np.array_equal(d1, d2) and np.array_equal(e1, e2))
        ok = all(same.values()) and req_same is not False
        worst |= int(not ok)
        print(entry["episode"], "arrays", same, "requests_equal", req_same,
              "ref_stats", entry.get("queue_arrays"), "backlog_max", stats["backlog_max"])
    print("ALL_EQUAL" if worst == 0 else "MISMATCH")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="converted C4 source directory")
    ap.add_argument("--out", type=Path, help="NEW output directory")
    ap.add_argument("--limit", type=int, help="first N episodes only (testing)")
    ap.add_argument("--compare", type=Path, help="real-converter source to compare against")
    args = ap.parse_args(argv)
    if args.compare is not None:
        return compare(args.src, args.compare)
    if args.out is None:
        ap.error("--out is required")
    doc = run(args.src, args.out, args.limit)
    print(json.dumps({"out": str(args.out), "n_episodes": doc["n_episodes"],
                      **doc["queue_augment"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
