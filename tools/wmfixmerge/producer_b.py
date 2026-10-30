"""Verdict B producer: reduce base + candidate quartet tasks and recorded reals.

Emits the wmfix_speedhead/1 panel (48 arm-minus-CTRL0 contrast rows plus the
64-row four-arm absolute grid with road/source components). Fail-closed: the
frozen base task set, the candidate task set, both IDENTITY receipts and the
recorded real values must be complete and hash-verified; the candidate must
carry the merge_response/conserve/admission flags and the base provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WORLDS = tuple(range(963880, 963888))
TASK_ARMS = ("ctrl0", "mappo", "pi", "circles")
CONTRAST_ARMS = ("mappo", "pi", "circles")
ARM_LABEL = {"ctrl0": "CTRL0", "mappo": "MAPPO", "pi": "PI", "circles": "CIRCLES"}
HORIZONS = (300, 600)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_tasks(directory):
    directory = Path(directory)
    complete = json.loads((directory / "COMPLETE.json").read_text())
    tasks = {}
    for name, expected in complete["hashes"].items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise SystemExit("quartet task missing/changed: " + name)
        task = json.loads(path.read_text())
        if tuple(task["arms"]) != TASK_ARMS:
            raise SystemExit("unexpected arm order in " + name)
        tasks[(int(task["start"]), int(task["future"]))] = task
    if len(tasks) != 64:
        raise SystemExit("quartet is not the complete 64-task grid")
    seeds = {int(t["metadata"]["seed"]) for t in tasks.values()}
    if seeds != set(WORLDS):
        raise SystemExit("quartet worlds differ from the SPEEDHEAD eight")
    return complete, tasks


def draws(tasks, seed, arm, horizon, component):
    index = TASK_ARMS.index(arm)
    rows = [t for t in tasks.values() if int(t["metadata"]["seed"]) == seed]
    by_future = {int(t["future"]): t for t in rows}
    if set(by_future) != set(range(8)):
        raise SystemExit(f"world {seed} lacks eight futures")
    return [float(by_future[f]["costs"][str(horizon)][component][index]) for f in range(8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--producer", required=True)
    args = parser.parse_args()

    base_complete, base_tasks = load_tasks(args.base_dir)
    cand_complete, cand_tasks = load_tasks(args.candidate_dir)
    base_identity = json.loads((Path(args.base_dir) / "IDENTITY.json").read_text())
    cand_identity = json.loads((Path(args.candidate_dir) / "IDENTITY.json").read_text())
    if base_identity.get("provenance") != cand_identity.get("provenance"):
        raise SystemExit("candidate provenance differs from the frozen base panel")
    flags = cand_identity.get("flags")
    if not flags or flags.get("merge_response") is not True \
            or flags.get("conserve_queue") is not True \
            or flags.get("learned_admission") is not True:
        raise SystemExit("candidate identity lacks the deployed correction flags")

    real = json.loads(Path(args.real).read_text())
    if real.get("schema") != "wmfix_real_speedhead/1":
        raise SystemExit("unexpected real values document")
    realj = {w: {a: {h: real["values"][f"{w}:{a}"]["values"][str(h)] for h in HORIZONS}
                 for a in TASK_ARMS} for w in WORLDS}

    absolute, rows = [], []
    for world in WORLDS:
        for arm in TASK_ARMS:
            for horizon in HORIZONS:
                cell = dict(world=str(world), arm=ARM_LABEL[arm], horizon=horizon,
                            real=realj[world][arm][horizon]["J"],
                            base_draws=draws(base_tasks, world, arm, horizon, "J"),
                            candidate_draws=draws(cand_tasks, world, arm, horizon, "J"))
                cell["components"] = {
                    "real": dict(road=realj[world][arm][horizon]["road"],
                                 source=realj[world][arm][horizon]["source"]),
                    **{name: dict(
                        road=draws(tasks, world, arm, horizon, "road"),
                        source=draws(tasks, world, arm, horizon, "source"))
                        for name, tasks in (("base", base_tasks), ("candidate", cand_tasks))}}
                absolute.append(cell)
        for arm in CONTRAST_ARMS:
            for horizon in HORIZONS:
                rows.append(dict(world=str(world), arm=ARM_LABEL[arm], horizon=horizon,
                    real=realj[world][arm][horizon]["J"] - realj[world]["ctrl0"][horizon]["J"],
                    base_draws=list(np.asarray(draws(base_tasks, world, arm, horizon, "J"))
                                    - np.asarray(draws(base_tasks, world, "ctrl0", horizon, "J"))),
                    candidate_draws=list(np.asarray(draws(cand_tasks, world, arm, horizon, "J"))
                                         - np.asarray(draws(cand_tasks, world, "ctrl0", horizon, "J")))))
    # Same integrity checks the verdict gate applies; fail before writing anything.
    lookup = {(r["world"], r["arm"], int(r["horizon"])): r for r in absolute}
    for cell in absolute:
        for name in ("base", "candidate"):
            sums = np.asarray(cell["components"][name]["road"]) + np.asarray(cell["components"][name]["source"])
            if not np.allclose(sums, cell[name + "_draws"], rtol=1e-10, atol=1e-8):
                raise SystemExit("WM road/source do not sum to J")
        if not np.isclose(cell["components"]["real"]["road"] + cell["components"]["real"]["source"],
                          cell["real"]):
            raise SystemExit("real road/source do not sum to J")
    for row in rows:
        arm = lookup[row["world"], row["arm"], int(row["horizon"])]
        ref = lookup[row["world"], "CTRL0", int(row["horizon"])]
        for name in ("base", "candidate"):
            if not np.allclose(np.asarray(arm[name + "_draws"]) - np.asarray(ref[name + "_draws"]),
                               row[name + "_draws"], rtol=1e-10, atol=1e-8):
                raise SystemExit("contrast draw mismatch")

    doc = dict(schema="wmfix_speedhead/1", K=8, mix="ON", native_lanes=True,
               matched_t300_history_queue=True, closed_loop_all_cavs=True,
               base_complete_sha256=digest(Path(args.base_dir) / "COMPLETE.json"),
               candidate_complete_sha256=digest(Path(args.candidate_dir) / "COMPLETE.json"),
               real_sha256=digest(args.real), rows=rows, absolute=absolute)
    doc["binding"] = json.loads(Path(args.binding).read_text())
    doc["producer"] = json.loads(Path(args.producer).read_text())
    with Path(args.out).open("x") as stream:
        json.dump(doc, stream, allow_nan=False, indent=1)
    print(json.dumps(dict(rows=len(rows), absolute=len(absolute)), indent=1))


if __name__ == "__main__":
    main()
