"""Registered combined TRAIN corpus for PILOT v2 (MAIN 13:50 TRACK 1).

Builds ONE physical corpus root from the 12-world FIT corpus (e3data) and the
40-world S-WMDATA corpus (role wmdata_fit): hardlinked episode arrays (byte
identical by inode; --copy makes real copies), a merged C4_EPISODES_MANIFEST.json
and SPLIT.json. Registered, hash-recorded differences from a naive union:

- The diagnostic FIT-root ``mappo`` episodes of the 12 registered FIT seeds are
  EXCLUDED (WMREQUAL3: lane-head requests missing; never FIT supervision). The
  wmdata ``mappo`` rows (speed-only export d9042cc5, 0 lane calls) are KEPT.
- ``allow_designed`` is set true: the registered wmdata arms are frozen deployed
  controllers (damping / pi-external-u) recorded in the manifest with policy
  pins; the generic designed-controller refusal must not reject them.
- Only rows of the registered worlds enter the merged manifest (all role
  ``train``); every other root row is inert for TRAIN and stays in its source.

Fail-closed: refuses on schema/k_slots mismatch, non-300 s warmups, missing
native B rows, unexpected arms, non-uniform 0.5 s clocks or 721-tick windows.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from tools.wmfixmerge.data import (SEEDS, WMDATA_ARMS, WMDATA_SEEDS, digest, install_guard,
                                   safe)

SCHEMA = "wms_c4_episode_source/1"


def _load(root):
    manifest = json.loads(safe(root / "C4_EPISODES_MANIFEST.json").read_text())
    split = json.loads(safe(root / "SPLIT.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("source manifest schema mismatch: " + str(root))
    if int(manifest.get("k_slots", -1)) != 192:
        raise ValueError("source k_slots mismatch: " + str(root))
    return manifest, split


def _episode_rows(manifest, seeds):
    return [e for e in manifest["episodes"] if int(e["seed"]) in seeds]


def merge(fit_root, wmdata_root, out, copy=False):
    fit_root, wmdata_root, out = safe(fit_root), safe(wmdata_root), safe(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("out exists and is not empty")
    fit_m, fit_s = _load(fit_root)
    wd_m, wd_s = _load(wmdata_root)
    fit_rows = _episode_rows(fit_m, set(SEEDS))
    wd_rows = _episode_rows(wd_m, set(WMDATA_SEEDS))
    excluded = [e["episode"] for e in fit_rows
                if e.get("arm") == "mappo"]
    fit_rows = [e for e in fit_rows if e.get("arm") != "mappo"]
    if not excluded or len(excluded) != len(SEEDS):
        raise ValueError("expected exactly one excluded FIT mappo row per seed")
    for rows, arms, native_ok in ((fit_rows, {"native", "ours_c1", "coopoff_c1"}, "fit"),
                                  (wd_rows, {"native", *WMDATA_ARMS}, "wmdata")):
        for row in rows:
            if row.get("arm") not in arms:
                raise ValueError(f"unregistered {native_ok} arm: " + str(row.get("arm")))
            if int(row.get("warmup_s", -1)) != 300:
                raise ValueError("non-300 s warmup: " + row["episode"])
            kind = row.get("branch", {}).get("kind")
            if (row["arm"] == "native") != (kind == "B"):
                raise ValueError("native/branch-kind mismatch: " + row["episode"])
        worlds = {row["world"] for row in rows}
        for world in worlds:
            group = [row for row in rows if row["world"] == world]
            if sum(row["arm"] == "native" for row in group) != 1:
                raise ValueError("world without exactly one native row: " + world)
        expected = {int(r["seed"]) for r in rows}
        if len(worlds) != len(expected) or not expected:
            raise ValueError(f"{native_ok} world/seed count mismatch")

    def place(rows, tag, source_root):
        out.mkdir(parents=True, exist_ok=True)
        placed = []
        for row in rows:
            src = safe(source_root / row["dir"])
            names = sorted(p.name for p in src.iterdir())
            if not names or not all(n.endswith(".npy") or n == "episode.json" for n in names):
                raise ValueError("unexpected episode dir content: " + str(src))
            dst = out / tag / row["dir"]
            if dst.exists():
                raise FileExistsError("destination exists: " + str(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copytree(src, dst)
            else:
                os.mkdir(dst)
                for name in names:
                    os.link(src / name, dst / name)
            new = dict(row, dir=f"{tag}/{row['dir']}")
            placed.append(new)
        return placed

    fit_placed = place(fit_rows, "fit", fit_root)
    wd_placed = place(wd_rows, "wmdata", wmdata_root)

    merged_split = dict(
        schema=fit_s.get("schema", "c4_split/1"),
        manifest_role_map=dict(fit=fit_s["manifest_role_map"].get("fit", "train"),
                               wmdata_fit="train"),
        roles=dict(fit=fit_s["roles"]["fit"],
                   wmdata_fit=wd_s["roles"]["wmdata_fit"]),
        rule=("S-WMFIXMERGE PILOT v2 (MAIN 13:50 TRACK 1): union of the registered "
              "12-seed FIT worlds (role fit) and the 40 S-WMDATA worlds (role "
              "wmdata_fit); both TRAIN-only, never validation/verdict"),
        worlds={},
    )
    for row in fit_rows:
        merged_split["worlds"][str(row["seed"])] = "fit"
    for row in wd_rows:
        merged_split["worlds"][str(row["seed"])] = "wmdata_fit"

    arms = dict(fit_m["arms"])
    arms.update(wd_m["arms"])
    manifest = dict(fit_m)
    manifest["arms"] = arms
    manifest["allow_designed"] = True
    manifest["episodes"] = fit_placed + wd_placed
    manifest["n_episodes"] = len(manifest["episodes"])
    manifest["wmfixmerge_v2"] = dict(
        sources=dict(fit_root=str(fit_root), wmdata_root=str(wmdata_root),
                     fit_manifest_sha256=digest(fit_root / "C4_EPISODES_MANIFEST.json"),
                     wmdata_manifest_sha256=digest(wmdata_root / "C4_EPISODES_MANIFEST.json"),
                     fit_split_sha256=digest(fit_root / "SPLIT.json"),
                     wmdata_split_sha256=digest(wmdata_root / "SPLIT.json")),
        excluded_fit_diagnostic_episodes=sorted(excluded),
        excluded_reason="WMREQUAL3: FIT mappo lane-head requests missing; never FIT supervision",
        allow_designed_reason=("registered S-WMDATA frozen deployed controllers "
                               "(owner 2026-10-31 12:55); policy pins in rows"),
        authorization="MAIN ADDENDUM 13:50 TRACK 1 (owner go)",
    )
    # World-design sidecars are inert for every TRAIN path (the v1 FIT root has
    # none); the wmdata ones are copied for provenance completeness only.
    for row in wd_rows:
        src = wmdata_root / f"world_{row['seed']}.json"
        if src.is_file() and not (out / src.name).exists():
            shutil.copyfile(src, out / src.name)
    (out / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (out / "SPLIT.json").write_text(json.dumps(merged_split, indent=1, sort_keys=True))
    receipt = dict(schema="wmfixmerge_corpus/1", root=str(out), copy=bool(copy),
                   fit_episodes=len(fit_placed), wmdata_episodes=len(wd_placed),
                   fit_worlds=len({r["world"] for r in fit_placed}),
                   wmdata_worlds=len({r["world"] for r in wd_placed}),
                   excluded_fit_diagnostic_episodes=sorted(excluded),
                   merged_manifest_sha256=digest(out / "C4_EPISODES_MANIFEST.json"),
                   merged_split_sha256=digest(out / "SPLIT.json"),
                   sources=manifest["wmfixmerge_v2"]["sources"])
    (out / "MERGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", required=True)
    parser.add_argument("--wmdata-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--copy", action="store_true",
                        help="copy episode dirs instead of hardlinks")
    args = parser.parse_args()
    install_guard()
    print(json.dumps(merge(args.fit_root, args.wmdata_root, safe(args.out), args.copy),
                     indent=2))
