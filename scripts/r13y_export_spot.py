"""Extend the twelve fixed BLOWUP TRAIN spot states to 300s, without simulation.

Reads only the two pinned TRAIN manifests, twelve exact episode directories,
and episode.json/time_s/veh_kin/veh_mask/vid/is_cav. Copies the original records
and changes ONLY recorded.native, horizon_s and recorded_horizon_steps.native.
The complete original 120s future must match the converted source exactly.
Pre-t0 history, t0 vehicles/CAVs, counters and identities are never rebuilt.

Default: validate and print the prospective export checksum/receipt. --write
creates a fresh deterministic .json.gz and .json.gz.receipt.json. One CPU process;
no torch, model, rollouts, original simulator files, or dataset-loader imports.

  python -B scripts/r13y_export_spot.py --out /existing/dir/r13y_spot300.json.gz --write

TRAIN here is the source manifest's outer split, as in the original BLOWUP
192-state export. Its source-internal train/val partition is retained in the
receipt (the fixed twelve episodes are source-internal val).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zlib

from diag_r13y_offline import (DT, check_metadata, guard_path, identity,
                               install_access_guard, np, read_states,
                               recording_problem, safe_text, sha256)

STATES = "/root/autodl-tmp/c4/v5/chk_r10b/states192.json.gz"
STATES_SHA256 = "6b7d15f6081ad703339c9e149b217cf382e8cf0b8aa83fb3ab13b786df3a37ec"
TRAIN_ROOT = "/root/autodl-tmp/c4/v5/data"
MANIFEST_SHA256 = {
    "v5_native": "31ead693ba76373fe15112d1d8dc8fb365e4cfae80e7237e8e0f0ede7081484f",
    "genrand_plain": "addb8ac6d5350b03cad0c935b6c769dbc5fd73157e3a5e1d2fe847335ced5cb2",
}
ARRAYS = ("time_s", "veh_kin", "veh_mask", "vid", "is_cav")


def read_train_manifest(root, source):
    path = guard_path(root / source / "C4_EPISODES_MANIFEST.json")
    digest = sha256(path)
    if digest != MANIFEST_SHA256[source]:
        raise ValueError(f"{source}: approved TRAIN manifest checksum mismatch")
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    if doc.get("schema") != "wms_c4_episode_source/1" or doc.get("split") != "TRAIN":
        raise ValueError(f"{source}: not a converted TRAIN source")
    if doc.get("dt_s") != DT:
        raise ValueError(f"{source}: expected native 0.5s data")
    return path, digest, doc


def episode_directory(root, rec, manifest):
    name = rec["episode"]
    if not re.fullmatch(r"w\d+m\d+p\d+t\d+__[A-Za-z0-9_-]+", name):
        raise ValueError("invalid exact TRAIN episode name")
    matches = [e for e in manifest["episodes"] if e.get("episode") == name]
    if len(matches) != 1:
        raise ValueError(f"{name}: expected unique TRAIN manifest membership")
    entry = matches[0]
    if int(entry["seed"]) != int(rec["seed"]):
        raise ValueError(f"{name}: manifest world seed mismatch")
    if entry.get("split_role") not in ("train", "val"):
        raise ValueError(f"{name}: unexpected source-internal partition")
    # Validate explicit paths; never enumerate episodes or follow metadata.source.
    expected = Path("episodes") / name
    if str(entry.get("dir", "")).replace("\\", "/") != expected.as_posix():
        raise ValueError(f"{name}: unexpected converted episode directory")
    source_root = guard_path(root / rec["source"])
    episode = guard_path(source_root / expected)
    if not episode.is_relative_to(source_root):
        raise ValueError("converted episode resolves outside its approved source")
    meta = json.loads(guard_path(episode / "episode.json").read_text(encoding="utf-8-sig"))
    check_metadata(meta)
    for key in ("episode", "seed", "demand_vph", "penetration"):
        if meta.get(key) != rec.get(key):
            raise ValueError(f"{name}: converted {key} does not match BLOWUP")
    if meta.get("world") != rec["world"].split("@", 1)[0]:
        raise ValueError(f"{name}: converted world identity mismatch")
    return episode, entry


def future_from_arrays(rec, arrays):
    """Inspected Timeline.future semantics: fixed roster, no revival after exit."""
    time = arrays["time_s"]
    kin, mask, vid, cav = (arrays[k] for k in ("veh_kin", "veh_mask", "vid", "is_cav"))
    if time.ndim != 1 or kin.ndim != 3 or kin.shape[2] != 7:
        raise ValueError("unexpected converted native array schema")
    if any(a.shape != kin.shape[:2] for a in (mask, vid, cav)) or len(time) != len(kin):
        raise ValueError("converted native array shapes disagree")
    matches = np.flatnonzero(np.isclose(time, float(rec["t0_s"]), rtol=0, atol=1e-6))
    if len(matches) != 1:
        raise ValueError("t0 missing or duplicated on the native time axis")
    j0 = int(matches[0])
    target = float(rec["t0_s"]) + DT * np.arange(601)
    if len(time[j0:j0 + 601]) != 601 or not np.allclose(time[j0:j0 + 601], target, rtol=0, atol=1e-6):
        raise ValueError("source lacks continuous native 300s coverage")
    initial = rec["initial_state"]
    if initial["time_s"] != rec["t0_s"]:
        raise ValueError("initial-state time differs from t0")
    roster, cav_by_id = [], {}
    for row in initial["vehicles"]:
        if not re.fullmatch(r"v\d+", row["id"]) or not row.get("present", True):
            raise ValueError("expected the original present vN roster")
        v = int(row["id"][1:])
        if v in cav_by_id:
            raise ValueError("duplicate t0 vehicle identity")
        roster.append(v)
        cav_by_id[v] = bool(row["is_cav"])
    alive, future = set(roster), []
    for j in range(j0, j0 + 601):
        slots = np.flatnonzero(mask[j] & (vid[j] >= 0))
        present = {int(vid[j, k]): int(k) for k in slots}
        if len(present) != len(slots):
            raise ValueError("native recording duplicates an occupied vehicle id")
        if j == j0 and set(present) != alive:
            raise ValueError("native t0 roster differs from the original BLOWUP roster")
        alive.intersection_update(present)
        rows = []
        for v in roster:
            if v not in alive:
                continue
            slot = present[v]
            if bool(cav[j, slot]) != cav_by_id[v]:
                raise ValueError("recorded CAV status differs from the original t0 flag")
            k = kin[j, slot]
            if not np.isfinite(k[:5]).all():
                raise ValueError("nonfinite recorded kinematics")
            lane = int(k[1:4].argmax()) if k[1:4].sum() > 0 else -1
            rows.append({"id": f"v{v}", "s": float(k[0]), "lane": lane,
                         "v": float(k[4]), "present": True, "is_cav": cav_by_id[v]})
        future.append({"time_s": float(time[j]), "vehicles": rows})
    if future[0] != initial:
        raise ValueError("native t0 kinematics differ from the original initial_state")
    if future[:241] != rec["recorded"]["native"]:
        raise ValueError("native 120s prefix differs from the pinned BLOWUP recording")
    return future


def build_export(states, states_sha256, train_root):
    root = guard_path(train_root)
    records = read_states(states, states_sha256)
    if len(records) != 192:
        raise ValueError("expected the original 192-state BLOWUP TRAIN export")
    manifests = {s: read_train_manifest(root, s) for s in MANIFEST_SHA256}
    result, provenance = [], []
    for idx in range(0, 48, 4):
        original = records[idx * 4]
        if original.get("horizon_s") != 120 or set(original["recorded"]) != {"native"}:
            raise ValueError("expected unchanged 120s native BLOWUP state records")
        manifest_path, manifest_hash, manifest = manifests[original["source"]]
        episode, entry = episode_directory(root, original, manifest)
        files = [guard_path(episode / (name + ".npy")) for name in ARRAYS]
        arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False)
                  for name, path in zip(ARRAYS, files)}
        future = future_from_arrays(original, arrays)
        extended = copy.deepcopy(original)
        extended["recorded"]["native"] = future
        extended["horizon_s"] = 300.0
        extended["recorded_horizon_steps"]["native"] = 600
        if recording_problem(extended, 300):
            raise ValueError("extended reference failed native coverage validation")
        # Whole-record equality after reverting the three intended changes.
        restored = copy.deepcopy(extended)
        for key in ("recorded", "horizon_s", "recorded_horizon_steps"):
            restored[key] = original[key]
        if restored != original:
            raise ValueError("extension changed original pre-t0 inputs or metadata")
        result.append(extended)
        provenance.append({"state": idx, "source_index": idx * 4, "identity": identity(original),
            "source_manifest": str(manifest_path), "source_manifest_sha256": manifest_hash,
            "source_outer_split": "TRAIN", "source_internal_partition": entry["split_role"],
            "episode": str(episode), "files_sha256": {
                str(path): sha256(path) for path in [episode / "episode.json", *files]},
            "native_frames": 601, "original_prefix_frames_exact": 241,
            "initial_vehicle_count": len(original["initial_state"]["vehicles"])})
        del arrays
    receipt = {"schema": "r13y_spot_export/1", "states": str(guard_path(states)),
        "states_sha256": states_sha256.lower(), "train_root": str(root),
        "selection": "original[::4][::4], fixed before outcomes", "n_states": 12,
        "horizon_s": 300, "dt_s": DT, "entries": "off; t0 roster only",
        "modified_record_fields": ["recorded.native", "horizon_s", "recorded_horizon_steps.native"],
        "pre_t0_inputs_exact": True, "t0_vehicles_and_cavs_exact": True,
        "original_120s_prefix_exact": True, "provenance": provenance}
    return result, receipt


def export_bytes(records):
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    stream = io.BytesIO()
    # Empty filename, fixed timestamp, and GzipFile's portable OS=255 header.
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=6, mtime=0) as compressor:
        compressor.write(encoded)
    return stream.getvalue()


def write_export(path, data, receipt):
    path = guard_path(path)
    receipt_path = guard_path(str(path) + ".receipt.json")
    if path.exists() or receipt_path.exists():
        raise ValueError("refusing to replace an existing export or receipt")
    if not path.parent.is_dir():
        raise ValueError("output parent must already exist")
    with path.open("xb") as stream:
        stream.write(data)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if sha256(path) != receipt["export_sha256"]:
        raise ValueError("written export checksum mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", default=STATES)
    parser.add_argument("--train-states-sha256", default=STATES_SHA256)
    parser.add_argument("--train-root", default=TRAIN_ROOT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--write", action="store_true", help="write the validated export and provenance receipt")
    args = parser.parse_args(argv)
    out = guard_path(args.out)
    if not str(out).endswith(".json.gz") or not out.parent.is_dir():
        parser.error("--out must be a .json.gz under an existing directory")
    safe_text(args.train_root)
    install_access_guard(out.parent if args.write else None)
    records, receipt = build_export(args.states, args.train_states_sha256, args.train_root)
    data = export_bytes(records)
    receipt.update(export=str(out), export_sha256=hashlib.sha256(data).hexdigest(),
        export_bytes=len(data), exporter_sha256=sha256(Path(__file__)),
        guard_sha256=sha256(Path(__file__).with_name("diag_r13y_offline.py")),
        python=sys.version, numpy=np.__version__, zlib=zlib.ZLIB_VERSION)
    if args.write:
        write_export(out, data, receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
