"""Assemble the recorded real SPEEDHEAD per-arm J components for verdict B.

--gather runs on one GPU host and records ``training_objective`` road/source/J
at 300/600 s for every per-arm job directory it finds, with per-file SHA256.
--merge combines the host partials (gpuA: ctrl0(+_r1 for 963880)/mappo;
gpuR: ctrl0/pi/circles), refuses duplicates, completes the 8-world x 4-arm
grid and cross-checks every arm-minus-CTRL0 contrast against the frozen
SPEEDHEAD.md table (values inlined below; report SHA256 embedded in output).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WORLDS = tuple(range(963880, 963888))
ARMS = ("ctrl0", "mappo", "pi", "circles")
HORIZONS = ("300", "600")
# SPEEDHEAD.md per-world paired arm-minus-CTRL0 (veh-s, J column), for cross-check.
TABLE = {
    (963880, "mappo", 300): 2830.1, (963880, "mappo", 600): 7770.3,
    (963880, "pi", 300): 578.8, (963880, "pi", 600): 946.8,
    (963880, "circles", 300): 981.5, (963880, "circles", 600): 2030.6,
    (963881, "mappo", 300): -1711.0, (963881, "mappo", 600): -114.5,
    (963881, "pi", 300): 1401.2, (963881, "pi", 600): 2153.8,
    (963881, "circles", 300): -153.2, (963881, "circles", 600): -796.7,
    (963882, "mappo", 300): 1150.6, (963882, "mappo", 600): 1341.1,
    (963882, "pi", 300): 96.3, (963882, "pi", 600): 1125.6,
    (963882, "circles", 300): 1644.1, (963882, "circles", 600): 4311.9,
    (963883, "mappo", 300): -2066.6, (963883, "mappo", 600): -5169.6,
    (963883, "pi", 300): 1991.0, (963883, "pi", 600): 122.7,
    (963883, "circles", 300): 63.9, (963883, "circles", 600): 699.2,
    (963884, "mappo", 300): 1525.6, (963884, "mappo", 600): 8023.3,
    (963884, "pi", 300): 1255.3, (963884, "pi", 600): 1127.2,
    (963884, "circles", 300): 3307.7, (963884, "circles", 600): 11325.0,
    (963885, "mappo", 300): -2916.6, (963885, "mappo", 600): -3040.1,
    (963885, "pi", 300): -2587.6, (963885, "pi", 600): -3575.1,
    (963885, "circles", 300): -1504.0, (963885, "circles", 600): -2016.9,
    (963886, "mappo", 300): 59.9, (963886, "mappo", 600): 1317.4,
    (963886, "pi", 300): 468.6, (963886, "pi", 600): 3161.1,
    (963886, "circles", 300): 5824.3, (963886, "circles", 600): 20147.2,
    (963887, "mappo", 300): -3215.7, (963887, "mappo", 600): -9865.5,
    (963887, "pi", 300): -892.9, (963887, "pi", 600): -5173.5,
    (963887, "circles", 300): 798.8, (963887, "circles", 600): 3858.1,
}
REPORT_SHA256 = "1f58ab94686e946821deb877d1037344294e34a5"  # SPEEDHEAD.md execution/evidence tag


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gather(root, out):
    root = Path(root)
    arms = {}
    for job in sorted((root / "jobs").iterdir()):
        if not job.is_dir() or not job.name.startswith("s9638"):
            continue
        seed, arm = job.name[1:7], job.name[8:]
        result = job / "result.json"
        metrics = json.loads(result.read_text())["metrics"]
        objective = metrics.get("training_objective")
        if not objective or not all(h in objective for h in HORIZONS):
            continue
        cell = {h: {k: float(objective[h][k]) for k in ("road", "source", "J")} for h in HORIZONS}
        key = (int(seed), arm)
        if key in arms:
            raise SystemExit(f"duplicate job {job.name}")
        arms[key] = dict(source=str(result), sha256=digest(result), values=cell,
                         t0_s=float(metrics["t0_s"]))
    with open(out, "x") as stream:
        json.dump(dict(schema="wmfix_real_speedhead_partial/1", root=str(root), arms=[
            dict(world=w, arm=a, **v) for (w, a), v in sorted(arms.items())]), stream, indent=1)
    print(f"gathered {len(arms)} arm-worlds from {root}")


def merge(partials, out):
    combined = {}
    sources = []
    for partial in partials:
        doc = json.loads(Path(partial).read_text())
        if doc.get("schema") != "wmfix_real_speedhead_partial/1":
            raise SystemExit("unexpected partial: " + partial)
        sources.append(dict(path=str(partial), sha256=digest(partial), root=doc["root"]))
        for row in doc["arms"]:
            key = (int(row["world"]), row["arm"])
            if key in combined:
                raise SystemExit(f"arm-world supplied twice: {key}")
            combined[key] = row
    # The frozen starts use the rerun ctrl0 tape for 963880 (make_starts _r1).
    for world in WORLDS:
        for arm in ARMS:
            wanted = "ctrl0_r1" if (world == 963880 and arm == "ctrl0") else arm
            if (world, wanted) not in combined:
                raise SystemExit(f"missing real arm-world {world} {wanted}")
    values = {}
    for world in WORLDS:
        for arm in ARMS:
            wanted = "ctrl0_r1" if (world == 963880 and arm == "ctrl0") else arm
            row = combined[(world, wanted)]
            for h in HORIZONS:
                cell = row["values"][h]
                if abs(cell["road"] + cell["source"] - cell["J"]) > 1e-6:
                    raise SystemExit(f"real components do not sum: {world} {arm} {h}")
            values[f"{world}:{arm}"] = row
            # ctrl0_r1 is the canonical CTRL0 column; plain ctrl0 must agree.
            if world == 963880 and arm == "ctrl0" and (963880, "ctrl0") in combined:
                plain = combined[(963880, "ctrl0")]["values"]
                rerun = row["values"]
                if plain != rerun:
                    raise SystemExit("963880 ctrl0 and ctrl0_r1 reals differ")
    for (world, arm, horizon), table in TABLE.items():
        cell = values[f"{world}:{arm}"]["values"][str(horizon)]["J"]
        ref = values[f"{world}:ctrl0"]["values"][str(horizon)]["J"]
        if abs((cell - ref) - table) > 1.0:
            raise SystemExit(f"SPEEDHEAD.md table mismatch {world} {arm} {horizon}: "
                             f"{cell - ref:.1f} vs {table}")
    doc = dict(schema="wmfix_real_speedhead/1", worlds=list(WORLDS), arms=list(ARMS),
               horizons=[int(h) for h in HORIZONS], sources=sources,
               report_sha256=REPORT_SHA256, values=values)
    with open(out, "x") as stream:
        json.dump(doc, stream, indent=1)
    print(f"merged {len(values)} arm-worlds; table cross-check PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gather")
    parser.add_argument("--merge", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.gather) == bool(args.merge):
        parser.error("exactly one of --gather / --merge")
    if args.gather:
        gather(args.gather, args.out)
    else:
        merge(args.merge, args.out)


if __name__ == "__main__":
    main()
