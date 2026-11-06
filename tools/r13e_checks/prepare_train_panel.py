"""Build the strictly train-role R13E panel from converted arrays on gpuR.

Run from the workstation (the source is piped to the CPU-only remote Python)::

    python tools/r13e_checks/remote_ops.py script tools/r13e_checks/prepare_train_panel.py

Only the two named outputs in diag_emerg are written. Existing identical outputs
are accepted; different outputs are refused. No raw episode assets are followed.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import zlib

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path("/root/autodl-tmp/c4/v5")
REPO = ROOT / "repo_r13e_8c7ff0a1"
WORK = ROOT / "diag_emerg"
SOURCES = ("v5_native", "genrand_plain")
PIN = "8c7ff0a19cfae59a50eebcd3146f7bcce51c5b37"
OFFLINE_BLOB = "af62f438a5420abbf1c0cf85cbf0c135c7f416c7"
T0, HISTORY, FUTURE, DT = 450.0, 60.0, 120.0, 0.5
MAX_OUTPUT = 50_000_000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def setup():
    require(sys.platform.startswith("linux"), "run this script on gpuR Linux")
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    # Pin to one available logical CPU as well as limiting numerical libraries.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for path in (ROOT, REPO, WORK):
        require(path.is_dir() and path.resolve() == path, f"unexpected directory: {path}")
    offline = REPO / "tools/r13e_checks/offline.py"
    require(offline.resolve() == offline, "offline.py must not be redirected")
    source = offline.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    normalized = source.replace(b"\r\n", b"\n")
    normalized_blob = hashlib.sha1(
        b"blob " + str(len(normalized)).encode() + b"\0" + normalized).hexdigest()
    require(OFFLINE_BLOB in (blob, normalized_blob),
            f"offline.py differs from pinned commit: blob={blob}, LF_blob={normalized_blob}")
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("r13e_pinned_offline", offline)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    study.cpu_limits()
    import torch

    torch.set_num_interop_threads(1)
    require(os.getpriority(os.PRIO_PROCESS, 0) == 19, "nice must equal 19")
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "torch must use one thread")
    return study


def original_audit(study, worlds):
    path = study.checked_path(ROOT / "chk_r10b/states192.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        doc = json.load(stream)
    records = doc if isinstance(doc, list) else next(
        doc[k] for k in ("states", "records", "windows") if k in doc)
    require(len(records) == 192, "original input must contain exactly 192 states")
    selected = records[::4]
    require(len(selected) == 48, "original stride must select exactly 48 states")
    counts = {"train": 0, "val": 0, "unknown": 0}
    rows = []
    for index, record in enumerate(selected):
        world = record.get("world", (record.get("meta") or {}).get("world"))
        study.guard_text(world)
        base_world = str(world).split("@", 1)[0]
        role = worlds.get(base_world, "unknown")
        counts[role] += 1
        rows.append({"original_index": index * 4, "world": world,
                     "manifest_world": base_world, "split_role": role})
    return {"path": str(path), "sha256": study.file_digest(path), "selection": "[::4]",
            "states": 48, "role_counts": counts, "all_roles_checked": True,
            "usable_as_strict_train_panel": counts["train"] == 48, "records": rows}


def array_digest(array):
    """Describe/hash only the consumed converted slice, not an entire large file."""
    import numpy as np

    array = np.ascontiguousarray(array)
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def make_record(study, root, entry):
    import numpy as np

    require(entry["split_role"] == "train", "refusing a non-train episode")
    directory = study.checked_path(root / entry["dir"])
    require(directory.is_relative_to(root), "episode escapes the converted source")
    consumed = {}

    def read(name, start=None, stop=None):
        path = study.checked_path(directory / f"{name}.npy")
        require(path.is_relative_to(directory), "array escapes its episode directory")
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        result = np.array(mapped[slice(start, stop)], copy=True)
        consumed[name] = {"path": str(path), "slice": [start, stop], **array_digest(result)}
        return result

    times = read("time_s")
    expected = np.arange(T0 - HISTORY, T0 + FUTURE + DT / 2, DT)
    first = int(np.searchsorted(times, T0 - HISTORY))
    stop = first + len(expected)
    require(np.array_equal(times[first:stop], expected),
            f"{entry['episode']}: need complete native grid 390..570 s")
    # Record only the native time slice actually used in the state.
    consumed["time_s"].update(slice=[first, stop], **array_digest(times[first:stop]))
    kin = read("veh_kin", first, stop)
    mask_raw = read("veh_mask", first, stop)
    vid = read("vid", first, stop)
    cav_raw = read("is_cav", first, stop)
    target_raw = read("target_mask", first, stop - 1)
    counter = read("counter", first + 120, first + 121)[0]
    mask, cav, target = mask_raw.astype(bool), cav_raw.astype(bool), target_raw.astype(bool)
    require(kin.ndim == 3 and kin.shape[0] == 361 and kin.shape[-1] == 7,
            "converted kin must be [361,K,7]")
    require(mask.shape == vid.shape == cav.shape == kin.shape[:2]
            and target.shape == (360, kin.shape[1]), "converted array shapes differ")
    require(np.isin(mask_raw, (0, 1)).all() and np.isin(cav_raw, (0, 1)).all()
            and np.isin(target_raw, (0, 1)).all(), "mask/CAV arrays must be binary")
    require(np.isfinite(kin[mask]).all() and (vid[mask] >= 0).all(),
            "occupied converted rows need finite kinematics and valid ids")
    require(np.isfinite(counter).all() and counter.shape == (2, 30), "invalid t0 counter")
    same = mask[:-1] & mask[1:] & (vid[:-1] == vid[1:]) & (vid[:-1] >= 0)
    require(np.array_equal(target, same), "target_mask disagrees with converted identities")
    hot = kin[..., 1:4]
    require(np.isin(hot[mask], (0, 1)).all() and (hot.sum(-1)[mask] <= 1).all(),
            "converted lanes must be one-hot or junction zeros")
    lanes = np.where(hot.sum(-1) > 0, hot.argmax(-1), -1)
    for t in range(361):
        require(len(np.unique(vid[t, mask[t]])) == int(mask[t].sum()),
                "duplicate occupied vehicle identity in a converted frame")
    j0 = 120
    road0 = mask[j0] & (lanes[j0] >= 0)
    require((road0 & ~cav[j0]).any(), "snapshot must contain an on-road HDV")
    require((road0 & cav[j0]).any(), "snapshot must retain an on-road CAV teacher")

    def vehicle(t, slot):
        return {"id": f"veh{int(vid[t, slot])}", "s": float(kin[t, slot, 0]),
                "lane": int(lanes[t, slot]), "v": float(kin[t, slot, 4]),
                "a": float(kin[t, slot, 5]), "length_m": float(kin[t, slot, 6]),
                "present": True, "is_cav": bool(cav[t, slot])}

    # Occupied junction rows remain present with lane=-1. A real exit or reused
    # slot permanently ends this t0 identity; later entrants never enter its future.
    alive = np.cumprod(np.concatenate((mask[j0:j0 + 1], target[j0:]), axis=0),
                       axis=0).astype(bool)
    require(alive.shape == (241, kin.shape[1]), "future must include t0 plus 240 steps")
    require(np.all(~alive | (vid[j0:] == vid[j0]))
            and np.all(~alive | (cav[j0:] == cav[j0])), "t0 roster identity/owner changed")
    future = [{"time_s": float(expected[t]),
               "vehicles": [vehicle(t, int(slot)) for slot in np.flatnonzero(alive[t - j0])]}
              for t in range(j0, 361)]

    # Id-key ALL observed pre-t0 vehicles, including leaders that departed before
    # t0, so slot reuse cannot contaminate a driver's native 60 s history.
    hist_ids = np.unique(vid[:j0 + 1][mask[:j0 + 1]])
    col_of = {int(value): col for col, value in enumerate(hist_ids)}
    hist_kin = np.zeros((121, len(hist_ids), 7), dtype=kin.dtype)
    hist_mask = np.zeros((121, len(hist_ids)), dtype=bool)
    hist_target = np.zeros((120, len(hist_ids)), dtype=bool)
    for t in range(121):
        slots = np.flatnonzero(mask[t])
        cols = [col_of[int(vid[t, slot])] for slot in slots]
        hist_kin[t, cols] = kin[t, slots]
        hist_mask[t, cols] = True
        if t < 120:
            hist_target[t, cols] = target[t, slots]
    require(np.array_equal(hist_target, hist_mask[:-1] & hist_mask[1:]),
            "id-keyed history continuity does not match the source")
    history = {"schema": "wms_eval_history/1", "ids": [f"veh{int(i)}" for i in hist_ids],
               "kin": hist_kin.tolist(), "mask": hist_mask.tolist(),
               "target_mask": hist_target.tolist(), "time_s": expected[:121].tolist(),
               "dt_s": DT, "history_s": HISTORY, "history_stride_s": 1.0,
               "available_span_s": HISTORY, "run_start_limited": False}
    arm, world = entry["arm"], entry["world"]
    record = {"key": f"r13e_train/{root.name}/{entry['episode']}@450",
              "world": world, "arm": arm, "split": "TRAIN", "split_role": "train",
              "source": root.name, "episode": entry["episode"], "t0_s": T0,
              "dt_s": DT, "horizon_s": FUTURE, "history": history,
              "initial_state": future[0], "recorded": {arm: future},
              "recorded_horizon_steps": {arm: 240}, "counter": counter.tolist(),
              "cav_controllers": {"cav_mode": "teacher"}, "arms": [arm]}
    # Verify the emitted CAV teacher against every original occupied CAV row in
    # the t0 cohort, including junction transit and genuine recorded exits.
    teacher_rows = 0
    for offset, frame in enumerate(future):
        t = j0 + offset
        actual = {row["id"]: row for row in frame["vehicles"] if row["is_cav"]}
        wanted = {f"veh{int(vid[t, s])}": vehicle(t, int(s))
                  for s in np.flatnonzero(alive[offset] & cav[j0])}
        require(actual == wanted, "emitted CAV teacher changed from the converted source")
        teacher_rows += len(actual)
    summary = {"source": root.name, "episode": entry["episode"], "world": world,
               "arm": arm, "split_role": entry["split_role"], "t0_s": T0,
               "source_tick": first + j0, "history_rows": 121,
               "history_id_columns": len(hist_ids), "future_frames": 241,
               "t0_hdv_on_road": int((road0 & ~cav[j0]).sum()),
               "t0_cav_on_road": int((road0 & cav[j0]).sum()),
               "cav_teacher_rows_verified": teacher_rows,
               "converted_slices": consumed}
    return record, summary


def encode(doc):
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def write_identical_or_new(study, path, payload):
    path = study.checked_path(path, exists=False)
    require(path.parent == WORK, "output must be directly inside diag_emerg")
    require(not path.is_symlink(), "output must not be a symlink")
    if path.exists():
        require(path.read_bytes() == payload, f"refusing to overwrite different output: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def main():
    study = setup()
    docs, worlds = study.train_manifests([ROOT / "data" / name for name in SOURCES])
    audit = original_audit(study, worlds)
    print("Original [::4] roles: " + json.dumps(audit["role_counts"], sort_keys=True), flush=True)
    records, summaries, sources = [], [], []
    seen = set()
    for root, doc in docs:
        entries = sorted((e for e in doc["episodes"] if e["split_role"] == "train"),
                         key=lambda e: (e["world"], e["arm"], e["episode"]))
        require(len(entries) >= 24, "each source needs at least 24 train episodes")
        indices = [(2 * i + 1) * len(entries) // 48 for i in range(24)]
        require(len(set(indices)) == 24, "episode selection must be unique")
        sources.append({"source": root.name, "path": str(root),
                        "manifest_sha256": study.file_digest(root / study.MANIFEST),
                        "role_counts": {role: sum(e["split_role"] == role for e in doc["episodes"])
                                        for role in ("train", "val")},
                        "selected_count": 24, "selected_sorted_train_indices": indices})
        for index in indices:
            entry = entries[index]
            require(worlds.get(entry["world"]) == "train", "world is not manifest train-role")
            require(entry["world"] not in seen, "replacement world occurs more than once")
            seen.add(entry["world"])
            record, summary = make_record(study, root, entry)
            summary["sorted_train_index"] = index
            records.append(record)
            summaries.append(summary)
        print(f"Prepared {root.name}: 24 genuine train-role snapshots", flush=True)
    require(len(records) == len(seen) == 48, "need exactly 48 unique train worlds")
    selection = {"method": "uniform midpoints over lexicographically sorted train episodes",
                 "sort_keys": ["world", "arm", "episode"],
                 "index_formula": "floor((2*i+1)*N_train/48), i=0..23, independently per source",
                 "incomplete_selected_episode": "fail; no substitutions or role overrides"}
    panel = {"schema": "wms_eval_states/1", "split": "TRAIN", "split_role": "train",
             "provenance": "converted authorized arrays; manifest train roles without overrides",
             "selection": selection, "states": records}
    study.audit_metadata(panel)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=6) as zipped:
        # Streaming JSON encoding avoids allocating another full uncompressed panel.
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
            json.dump(panel, text, sort_keys=True, separators=(",", ":"), allow_nan=False)
            text.write("\n")
    payload = buffer.getvalue()
    require(len(payload) < MAX_OUTPUT - 1_000_000, "compressed panel exceeds output budget")
    disk = os.statvfs(WORK)
    require(disk.f_bavail * disk.f_frsize > len(payload) + 10_000_000,
            "insufficient disk headroom for this small panel")
    output = WORK / "train_states48.json.gz"
    receipt_path = WORK / "panel_receipt.json"
    # Inspect both existing output paths before writing either.
    for path in (output, receipt_path):
        require(not path.is_symlink() and study.checked_path(path, exists=False) == path,
                "unexpected output target")
    write_identical_or_new(study, output, payload)
    accepted = study.load_train_states(output, worlds)
    require(len(accepted) == 48 and all(worlds.get(r["world"]) == "train" for r in accepted),
            "pinned load_train_states did not accept all 48 train worlds")
    require(accepted == records, "serialized states changed during pinned loader validation")
    digest = study.file_digest(output)
    require(digest == hashlib.sha256(payload).hexdigest(), "written panel digest mismatch")
    receipt = {"schema": "r13e_train_panel_receipt/1", "status": "complete",
               "output": {"path": str(output), "sha256": digest, "bytes": len(payload)},
               "states": 48, "unique_worlds": 48, "role_counts": {"train": 48, "val": 0},
               "t0_s": T0, "history_s": HISTORY, "history_rows": 121,
               "history_stride_s": 1.0, "native_dt_s": DT, "future_s": FUTURE,
               "future_frames_including_t0": 241,
               "cav_teacher": "exact converted t0 roster; junctions retained; first exit ends identity",
               "history": "id-keyed full observed scene; pre-t0 leaders retained; no slot-reuse mixing",
               "original_panel_audit": audit, "selection": selection,
               "sources": sources, "selected_episodes": summaries,
               "validation": {"loader": str(REPO / "tools/r13e_checks/offline.py"),
                              "loader_sha256": study.file_digest(REPO / "tools/r13e_checks/offline.py"),
                              "pinned_commit": PIN, "pinned_LF_git_blob": OFFLINE_BLOB,
                              "load_train_states": "passed", "accepted_states": len(accepted),
                              "cav_teacher_source_comparison": "passed all 48 states"},
               "execution": {"device": "cpu", "nice": 19, "threads": 1,
                             "cpu_affinity_count": len(os.sched_getaffinity(0)),
                             "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                             "python": sys.version.split()[0], "zlib": zlib.ZLIB_RUNTIME_VERSION,
                             "gzip_mtime": 0, "gzip_filename": "", "gzip_compresslevel": 6},
               "command": "python tools/r13e_checks/remote_ops.py script tools/r13e_checks/prepare_train_panel.py"}
    receipt_bytes = encode(receipt)
    require(len(payload) + len(receipt_bytes) < MAX_OUTPUT, "total output exceeds 50 MB")
    write_identical_or_new(study, receipt_path, receipt_bytes)
    print(json.dumps({"status": "complete", "panel": receipt["output"],
                      "receipt": {"path": str(receipt_path), "bytes": len(receipt_bytes),
                                  "sha256": hashlib.sha256(receipt_bytes).hexdigest()},
                      "original_role_counts": audit["role_counts"], "states": 48,
                      "per_source": {name: 24 for name in SOURCES},
                      "load_train_states": "passed", "nice": 19, "threads": 1},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
