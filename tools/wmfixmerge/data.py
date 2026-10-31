"""Fail-closed WMFIX TRAIN source binding, usable without torch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEEDS = (973002, 973053, 973060, 973083, 973093, 973123,
         973126, 973133, 973151, 973160, 973190, 973224)
# S-WMDATA (owner-approved 2026-10-31 12:55): registered FIT corpus for the WM
# merge/queue fix. Fresh C4_LOOP_TRAIN sub-block; role 'wmdata_fit' admits exactly
# these seeds with the native reference + the four S-WMDATA arms. The 12-seed FIT
# allowlist above is unchanged and remains the only admission for role 'fit'.
WMDATA_ROLE = "wmdata_fit"
WMDATA_SEEDS = tuple(range(969400, 969440))
WMDATA_ARMS = ("ctrl0", "mappo", "pi", "circles")
BAD = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
       "hetdata_truth", "vtype")
WM2_SHA = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
ARRAYS = ("time_s", "vid", "veh_mask", "veh_kin", "is_cav", "counter",
           "target_mask", "control_time_s", "controller", "issued_acceleration_mps2",
           "issued_target_speed_mps", "entry_backlog", "request_counter", "entry_cav_counter")
# Recorded request ledgers exist only on converter corpora (v5/pairs300 style).
# S-E3DATA FIT tapes predate them; their Q0/due come from the exact counter/
# deterministic-schedule reconstruction in tools/wmrequal/labels.py instead.
LEDGER_ARRAYS = ("entry_backlog", "request_counter", "entry_cav_counter")
CORE_ARRAYS = tuple(a for a in ARRAYS if a not in LEDGER_ARRAYS)


def episode_arrays(root, row):
    """Array names hash-bound for one episode; all CORE arrays are mandatory."""
    p = safe(Path(root) / row["dir"])
    present = sorted(name for name in ARRAYS if safe(p / (name + ".npy")).is_file())
    missing = [name for name in CORE_ARRAYS if name not in present]
    if missing:
        raise ValueError("missing core production array: " + missing[0])
    return present


COMPONENTS = ("J", "upstream", "source", "discharge", "road")
IMPLEMENTATION_FILES = ("src/wms/model/wms.py", "src/wms/model/entries.py",
    "src/wms/model/merge_response.py", "src/wms/model/source_ledger.py",
    "src/wms/model/learned_admission.py", "src/wms/control/entrant_mix.py",
    "src/wms/train/merge_response.py", "src/wms/train/inputs.py",
    "tools/errsrc/inference.py", "tools/wmrequal/labels.py",
    "tools/wmfixmerge/runtime.py", "tools/wmfixmerge/entrant_binding.py", "tools/wmfixmerge/train_config.json",
    "tools/wmrequal/score.py", "tools/wmrequal/native_lanes.py",
    "src/wms/model/spec.py", "src/wms/config/train.py", "src/wms/config/legacy_flat.py",
    "tools/wmfixmerge/train.py", "tools/wmfixmerge/feasibility.py",
    "tools/wmfixmerge/data.py", "tools/wmfixmerge/prepare.py", "src/wms/data/c4_pairs.py",
    "tools/wmfixmerge/verdict.py", "tools/wmfixmerge/verdict_metrics.py",
    "tools/wmfixmerge/verdict_panels.py", "tools/wmfixmerge/producer_runner.py",
    "tools/wmfixmerge/harness_contract.py", "tools/wmfixmerge/zero_head_panel.py",
    "tools/wmfixmerge/zero_head_compare.py", "tools/wmfixmerge/verdict_plan.json")


def safe(path):
    path = Path(path)
    for text in (str(path).lower(), str(path.resolve()).lower()):
        if any(word in text for word in BAD) or text.endswith((".xml", ".rou")):
            raise ValueError("restricted input path")
    return path


def digest(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: digest(root / name) for name in IMPLEMENTATION_FILES}


def select(root):
    root = safe(root)
    manifest = json.loads(safe(root / "C4_EPISODES_MANIFEST.json").read_text())
    split = json.loads(safe(root / "SPLIT.json").read_text())
    entries = []
    fit_seen, wmdata_seen = set(), set()
    for row in manifest["episodes"]:
        seed = int(row["seed"])
        role = split["worlds"].get(str(seed))
        if seed in SEEDS:
            # Original 12-seed FIT allowlist path (behavior unchanged).
            if role != "fit":
                raise ValueError(f"registered TRAIN seed {seed} is not FIT")
            if row.get("arm") == "mappo":
                continue  # WMREQUAL3: lane-head requests missing; diagnostic only, never FIT supervision.
            if row.get("branch", {}).get("kind") != "B" and row.get("arm") not in (
                    "ours_c1", "coopoff_c1"):
                raise ValueError("unregistered TRAIN arm")
            fit_seen.add(seed)
        elif role == WMDATA_ROLE:
            # S-WMDATA registered role: native reference + the four WMDATA arms.
            if seed not in WMDATA_SEEDS:
                raise ValueError(f"wmdata_fit seed {seed} outside the registered block")
            native = row.get("arm") == "native" or row.get("branch", {}).get("kind") == "B"
            if not native and row.get("arm") not in WMDATA_ARMS:
                raise ValueError("unregistered wmdata_fit arm")
            wmdata_seen.add(seed)
        else:
            continue
        path = safe(root / row["dir"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("episode escapes source root")
        if row.get("warmup_s") != 300:
            raise ValueError("exact counter-derived per-entrance Q0 requires t300")
        entries.append(row)
    if fit_seen and fit_seen != set(SEEDS):
        raise ValueError("registered TRAIN seeds missing")
    if not fit_seen and not wmdata_seen:
        raise ValueError("no registered TRAIN rows selected")
    groups = {}
    for row in entries:
        groups.setdefault(row["pair_key"], []).append(row)
    for key, rows in groups.items():
        if len({(r["seed"], r["world"], r["warmup_s"]) for r in rows}) != 1:
            raise ValueError("pair crosses worlds or start clocks")
        if sum(r.get("branch", {}).get("kind") == "B" for r in rows) != 1:
            raise ValueError(f"pair {key} must have exactly one native reference")
        if len(rows) < 2 or len({r["arm"] for r in rows}) != len(rows):
            raise ValueError("missing/duplicate paired arm")
    return entries


def verify(prepared):
    doc = json.loads(safe(prepared).read_text())
    if doc.get("schema") != "wmfix_train/1":
        raise ValueError("unexpected prepared corpus")
    selected = select(doc["root"])
    if set(int(s) for s in doc["seeds"]) != {int(row["seed"]) for row in selected}:
        raise ValueError("prepared corpus seeds differ from the registered selection")
    if doc["manifest_sha256"] != digest(Path(doc["root"]) / "C4_EPISODES_MANIFEST.json"):
        raise ValueError("source manifest changed")
    if doc["split_sha256"] != digest(Path(doc["root"]) / "SPLIT.json"):
        raise ValueError("source roles changed")
    if select(doc["root"]) != doc["episodes"]:
        raise ValueError("selected source rows changed")
    expected_paths = {str(safe(Path(doc["root"]) / row["dir"] / (name + ".npy")))
                      for row in doc["episodes"] for name in episode_arrays(doc["root"], row)}
    if len({tuple(episode_arrays(doc["root"], row)) for row in doc["episodes"]}) != 1:
        raise ValueError("ledger array presence must be corpus-uniform")
    if set(doc["files"]) != expected_paths:
        raise ValueError("incomplete or extraneous input hash binding")
    for path, expected in doc["files"].items():
        if digest(path) != expected:
            raise ValueError("source bytes changed: " + path)
    return doc


def install_guard():
    """Also guard inputs opened indirectly by incumbent production loaders."""
    import os
    import sys
    def guard(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            safe(os.fsdecode(values[0]))
    sys.addaudithook(guard)
