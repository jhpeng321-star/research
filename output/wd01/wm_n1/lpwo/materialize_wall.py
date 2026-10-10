"""Guarded, streaming LPWO wall materialisation (CPU / frozen packs only).

Run from ``output/wd01/wm_n1`` with ``python -m lpwo.materialize_wall``.
The output directory must be new. ``wall_same`` contains train-origin removed
walls and explicit TRAIN pilots in retained training worlds; ``wall_pilot``
contains engaged walls from an explicit pilot INDEX and explicit KEY=RUN_DIR
bindings, never discovered run paths. ``seen_controls`` uses the retained split
for successfully materialised world/t0 pairs. The pilot set overlaps wall_same
when a pilot world was retained in training; these are not disjoint cohorts.
All three sets expose world familiarity and genuine physical pairing separately.

Metadata lists are small; pack arrays are mmap-backed and states and rejection
receipts are written one at a time. No raw physics, model, or simulator is read.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from wms_eval.materialize import (
    DT_S, HISTORY_STEPS, Identity, MaterializeError, Pack, build_state,
)

WORKSPACE = Path(__file__).resolve().parents[4]
PILOT_CAMPAIGN = "campaign_walls_pilot_chunk4"
HORIZON_S = 120.0
SHIFT_STEPS = 9
STEPS = 240
DENIED = re.compile(
    r"(?:^|[^a-z0-9])cal(?:ibration)?(?:$|[^a-z0-9])|"
    r"holdout|sealed|chunk[-_ ]?5|j4b|inc1", re.IGNORECASE,
)
ENGAGED = {"assembled_released", "aborted_in_hold"}


class WallMaterializeError(MaterializeError):
    """Fail-closed source or state contract violation."""


def _deny(value) -> None:
    if DENIED.search(str(value)):
        raise WallMaterializeError(f"forbidden source identity: {value}")


def local_path(value: str | Path) -> Path:
    # Check spelling before resolving any symlink / junction.
    _deny(value)
    path = Path(value).resolve()
    _deny(path)
    try:
        path.relative_to(WORKSPACE)
    except ValueError as exc:
        raise WallMaterializeError(f"path outside workspace: {path}") from exc
    return path


def checked_path(value, guard) -> Path:
    path = local_path(value)
    if guard.check_path(path) is False:
        raise WallMaterializeError(f"guard denied path: {path}")
    return path


def checked_record(record: Mapping, guard) -> None:
    if not isinstance(record, Mapping):
        raise WallMaterializeError("source record must be an object")
    # Scan source identities recursively, including nested timeline paths. Never
    # infer TRAIN from a missing split (the generic materialiser does).
    def scan(value):
        if isinstance(value, Mapping):
            for key, child in value.items():
                _deny(key)
                scan(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                scan(child)
        elif isinstance(value, str):
            _deny(value)
    scan(record)
    if str(record.get("split", "")).upper() != "TRAIN":
        raise WallMaterializeError("source must explicitly declare split TRAIN")
    if str(record.get("chunk", "")).strip() == "5":
        raise WallMaterializeError("chunk 5 is forbidden")
    if guard.check_record(record) is False:
        raise WallMaterializeError(f"guard denied record: {record.get('key')}")


def read_json(path, guard):
    return guard.read_json(checked_path(path, guard))


def _rows(payload, name="windows") -> list[dict]:
    rows = payload.get(name) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise WallMaterializeError(f"metadata needs a {name} object list")
    return rows


def _pairs(entries: Sequence[str]) -> dict[str, Path]:
    result = {}
    for entry in entries:
        name, sep, path = entry.partition("=")
        if not sep or not name or not path or name in result:
            raise WallMaterializeError(f"expected unique NAME=PATH: {entry!r}")
        _deny(name)
        result[name] = local_path(path)
    return result


class GuardedPack(Pack):
    """Reuse Pack, with all actual opens guarded (including manifest filenames)."""

    def __init__(self, name, root, guard):
        self.guard = guard
        root = checked_path(root, guard)
        # Pack reads these itself; both guard checks must precede its constructor.
        manifest = read_json(root / "manifest.json", guard)
        read_json(root / "INDEX.json", guard)
        for spec in manifest.get("arrays", {}).values():
            file = Path(spec["file"])
            if file.is_absolute() or ".." in file.parts:
                raise WallMaterializeError("pack array filename must be relative to pack")
            checked_path(root / file, guard)
        super().__init__(name, root)

    def array(self, name):
        spec = self.arrays.get(name)
        if spec is not None:
            checked_path(self.root / spec["file"], self.guard)
        return super().array(name)


def guarded_identity(name, root, guard):
    root = checked_path(root, guard)
    for name_on_disk in ("identities.npy", "global_positions.npy", "vehicle_map.json"):
        checked_path(root / name_on_disk, guard)
    return Identity(name, root)


def _run(window, pack, guard):
    checked_record(window, guard)
    matches = [r for r in pack.runs.values()
               if str(r.get("world")) == str(window["world"])
               and str(r.get("arm")) == str(window["arm"])
               and float(r.get("t0_s", -1)) == float(window["t0_s"])]
    if len(matches) != 1:
        raise WallMaterializeError("run world/arm/t0 must resolve uniquely")
    run = matches[0]
    checked_record(run, guard)
    if str(run.get("chunk")) != str(window.get("chunk")):
        raise WallMaterializeError("window / pack run chunk mismatch")
    if window.get("run") and run.get("run") and window["run"] != run["run"]:
        raise WallMaterializeError("window / pack run identity mismatch")
    return run


def _geometry(window, pack, identity, guard):
    """Require full history/horizon, canonical occupied identities and counters.

    Return the FULL shifted physical state, including a/length, not just the
    surviving vehicles in build_state's reduced trajectory representation.
    Births or slot reuse during the shift are refused: their ten-step histories
    and canonical identities cannot be reconstructed by the reused builder.
    """
    run = _run(window, pack, guard)
    position, item = pack.item_for(run)
    checked_record({**run, **item}, guard)
    start = int(item["tick"])
    tick = start + SHIFT_STEPS
    end = int(run["tick_end"])
    if not (0 <= int(run["tick_start"]) <= start) or tick + STEPS >= end:
        raise WallMaterializeError("missing complete 120s horizon / run-local history")
    required = ("veh_kin", "veh_mask", "veh_step", "target_mask", "counter")
    arrays = {name: pack.array(name) for name in required}
    if any(a.shape[0] < end for a in arrays.values()):
        raise WallMaterializeError("array shorter than declared run")
    kin = np.asarray(arrays["veh_kin"][start:tick + 1])
    masks = np.asarray(arrays["veh_mask"][start:tick + 1]).astype(bool)
    tm = np.asarray(arrays["target_mask"][start:tick]).astype(bool)
    if kin.shape[0] != HISTORY_STEPS or tm.shape != (SHIFT_STEPS, kin.shape[1]):
        raise WallMaterializeError("missing real 10-step history / continuity")
    occupied = masks[-1]
    ids = identity.row_for_position(position)
    if len(ids) < kin.shape[1]:
        ids = np.pad(ids, (0, kin.shape[1] - len(ids)), constant_values=-1)
    if not masks[:, occupied].all() or not tm[:, occupied].all() or (ids[occupied] < 0).any():
        raise WallMaterializeError("missing continuous history / identity for occupied vehicle")
    if not occupied.any() or not np.isfinite(kin[:, occupied]).all():
        raise WallMaterializeError("empty or nonfinite physical history")
    counter = np.asarray(arrays["counter"][tick])
    if counter.shape != (2, 30) or not np.isfinite(counter).all() or (counter < 0).any():
        raise WallMaterializeError("entrance counter required: finite nonnegative [2,30]")
    physical = {}
    step_history = np.asarray(arrays["veh_step"][start:tick + 1])
    cav_history = step_history[:, occupied, 7]
    if (not np.isfinite(cav_history).all()
            or not ((cav_history > 0.5) == (cav_history[0] > 0.5)).all()):
        raise WallMaterializeError("nonfinite or changing CAV identity in history")
    steps = step_history[-1]
    for slot in np.flatnonzero(occupied):
        label = identity.label(int(ids[slot]))
        if label in physical:
            raise WallMaterializeError("duplicate physical vehicle identity")
        physical[label] = (*kin[-1, slot].tolist(), bool(steps[slot, 7] > 0.5))
    return physical, counter.tolist()


def _validate_state(state):
    if state["horizon_s"] != HORIZON_S or state["dt_s"] != DT_S:
        raise WallMaterializeError("materialised horizon must be exactly 120s / dt 0.5")
    hist = state["history"]
    n = len(state["initial_state"]["vehicles"])
    if (hist["history_steps"] != HISTORY_STEPS
            or np.asarray(hist["kin"]).shape != (10, n, 7)
            or np.asarray(hist["mask"]).shape != (10, n)
            or np.asarray(hist["target_mask"]).shape != (9, n)
            or not np.asarray(hist["mask"]).all()
            or not np.asarray(hist["target_mask"]).all()):
        raise WallMaterializeError("materialised state lacks complete real history")
    for frames in state["recorded"].values():
        if len(frames) != STEPS + 1:
            raise WallMaterializeError("materialised state lacks 241 horizon frames")
        for j, frame in enumerate(frames):
            if float(frame["time_s"]) != float(state["t0_s"]) + j * DT_S:
                raise WallMaterializeError("invalid future timestamp")
            for vehicle in frame["vehicles"]:
                if not all(math.isfinite(float(vehicle[k])) for k in ("s", "v", "lane")):
                    raise WallMaterializeError("nonfinite future physical state")


def validate_timeline(timeline, window, guard, *, require_engaged=False):
    checked_record({"split": window["split"], **timeline}, guard)
    status = timeline.get("status")
    if require_engaged and (timeline.get("engaged") is not True or status not in ENGAGED):
        raise WallMaterializeError("wall is refused / unassembled / unverifiable, not engaged")
    if status not in ENGAGED | {"aborted_in_assembly", "not_feasible"}:
        raise WallMaterializeError("unverifiable wall timeline")
    if timeline.get("engaged") is not (status in ENGAGED):
        raise WallMaterializeError("contradictory timeline engagement status")
    if timeline.get("owned") is not (status != "not_feasible"):
        raise WallMaterializeError("contradictory wall ownership status")
    if str(timeline.get("world")) != str(window["world"]):
        raise WallMaterializeError("timeline / window world mismatch")
    if str(timeline.get("arm")) != str(window["arm"]):
        raise WallMaterializeError("timeline / window arm mismatch")
    for key in ("t0_s",):
        if timeline.get(key) is None or not math.isfinite(float(timeline[key])):
            raise WallMaterializeError(f"missing wall timeline {key}")
    if float(timeline["t0_s"]) != float(window["t0_s"]):
        raise WallMaterializeError("wall timeline origin differs from pack window")
    members = timeline.get("members")
    if status == "not_feasible":
        if members or any(timeline.get(k) is not None for k in
                          ("request_s", "assembled_at_s", "owner_end_s")):
            raise WallMaterializeError("refused wall has ownership / assembly evidence")
        return
    if not isinstance(members, list) or not members:
        raise WallMaterializeError("wall timeline needs unique members")
    if not all(isinstance(m, str) and m for m in members) or len(set(members)) != len(members):
        raise WallMaterializeError("wall members must be unique nonempty vehicle identities")
    for key in ("request_s", "owner_end_s"):
        if timeline.get(key) is None or not math.isfinite(float(timeline[key])):
            raise WallMaterializeError(f"missing wall timeline {key}")
    if not (float(timeline["t0_s"]) <= float(timeline["request_s"])
            < float(timeline["owner_end_s"])):
        raise WallMaterializeError("inconsistent wall ownership / assembly times")
    assembled = timeline.get("assembled_at_s")
    if status in ENGAGED:
        if assembled is None or not math.isfinite(float(assembled)):
            raise WallMaterializeError("missing wall timeline assembled_at_s")
        if not (float(timeline["request_s"]) <= float(assembled)
                < float(timeline["owner_end_s"])):
            raise WallMaterializeError("inconsistent wall ownership / assembly times")
    elif assembled is not None:
        raise WallMaterializeError("unassembled wall has assembly evidence")


def materialize_state(window, packs, identities, guard, *, timeline=None,
                      source="wall_same", world_seen=True, min_vehicles=5):
    checked_record(window, guard)
    pack_name = str(window.get("pack"))
    if pack_name not in packs or pack_name not in identities:
        raise WallMaterializeError("window has no registered pack / identity")
    pack, identity = packs[pack_name], identities[pack_name]
    if source != "seen_controls":
        if timeline is None:
            raise WallMaterializeError("wall timeline required")
        validate_timeline(timeline, window, guard, require_engaged=source == "wall_pilot")
    physical = _geometry(window, pack, identity, guard)
    descriptor = {**window, "_extra_baselines": []}
    state = build_state(descriptor, packs, identities, horizon_s=HORIZON_S,
                        extra_baselines=(), shift_steps=SHIFT_STEPS,
                        min_vehicles=min_vehicles)
    _validate_state(state)
    state.update(source_set=source, world_seen_in_training=bool(world_seen),
                 controller_seen_in_training=source == "seen_controls",
                 unseen_controller=source != "seen_controls", paired=False,
                 native_pair_status="not_applicable_seen_control", wall_members=[],
                 wall_timeline=None, wall_engaged=False, wall_owned=False)
    if source == "seen_controls":
        return state

    members = timeline.get("members") or []
    initial = {v["id"]: v for v in state["initial_state"]["vehicles"]}
    if any(m not in initial or not initial[m]["is_cav"] for m in members):
        raise WallMaterializeError("wall member missing from initial CAV roster")
    original_arm = state["arm"]
    state["source_arm"] = original_arm
    state["recorded"] = {"wall": state["recorded"][original_arm]}
    state["recorded_horizon_steps"] = {"wall": STEPS}
    state["arm"] = "wall"
    state["arms"] = ["wall"]
    state["wall_members"] = list(members)
    state["wall_timeline"] = dict(timeline)
    state["wall_engaged"] = timeline["engaged"]
    state["wall_owned"] = timeline["owned"]
    state["wall_exposure_status"] = timeline["status"]
    t = state["t0_s"]
    state["wall_phase_at_t0"] = (
        "none" if not timeline["owned"] or t < float(timeline["request_s"]) else
        "handed_back" if t >= float(timeline["owner_end_s"]) else
        "assembly" if timeline.get("assembled_at_s") is None
        or t < float(timeline["assembled_at_s"]) else "hold")
    candidates = [r for r in pack.runs.values()
                  if str(r.get("world")) == str(window["world"])
                  and r.get("arm") == "native"
                  and float(r.get("t0_s", -1)) == float(window["t0_s"])]
    state["native_pair_status"] = "missing_native" if not candidates else "ambiguous_native"
    if len(candidates) == 1:
        native_window = {**candidates[0], "pack": pack_name, "_extra_baselines": []}
        # Access policy violations are fatal, never swallowed as pairing errors.
        checked_record(native_window, guard)
        try:
            native_physical = _geometry(native_window, pack, identity, guard)
            if native_physical != physical:
                state["native_pair_status"] = "initial_physical_state_mismatch"
            else:
                native = build_state(native_window, packs, identities,
                                     horizon_s=HORIZON_S, extra_baselines=(),
                                     shift_steps=SHIFT_STEPS, min_vehicles=min_vehicles)
                _validate_state(native)
                state["recorded"]["native"] = native["recorded"]["native"]
                state["recorded_horizon_steps"]["native"] = STEPS
                state["arms"] = ["native", "wall"]
                state["paired"] = True
                state["native_pair_status"] = "matched_initial_physical_state"
        except MaterializeError as exc:
            state["native_pair_status"] = "invalid_native_history_counter_or_horizon"
            state["native_pair_error"] = str(exc)
    return state


def _pilot_path(row, bindings, guard):
    # This must finish BEFORE calling the timeline reader or touching a run path.
    checked_record(row, guard)
    if str(row.get("policy", "")).lower() != "wall":
        raise WallMaterializeError("pilot INDEX request must be wall-labelled")
    root_rel = str(row.get("root_rel", "")).replace("\\", "/").rstrip("/")
    if root_rel.split("/")[-1] != PILOT_CAMPAIGN:
        raise WallMaterializeError("pilot INDEX must name campaign_walls_pilot_chunk4")
    key = str(row.get("key"))
    if key not in bindings:
        raise WallMaterializeError(f"pilot run needs explicit KEY=RUN_DIR binding: {key}")
    path = bindings[key]
    if PILOT_CAMPAIGN not in path.parts or path.parent.name != PILOT_CAMPAIGN:
        raise WallMaterializeError("pilot run dir must be directly under the permitted campaign")
    run_name = row.get("run") or key.rsplit("/", 1)[-1]
    if path.name != str(run_name):
        raise WallMaterializeError("pilot run binding name differs from INDEX")
    return checked_path(path, guard)


def write_stream(path, name, windows, packs, identities, guard, errors, *,
                 train_worlds, min_vehicles=5, accepted_targets=None):
    counts = Counter()
    with checked_path(path, guard).open("x", encoding="utf-8") as fh:
        fh.write('{"schema":"wms_eval_states/1","set":' + json.dumps(name)
                 + ',"horizon_s":120.0,"windows":[')
        for window, timeline in windows:
            counts["n_requested"] += 1
            try:
                record = materialize_state(
                    window, packs, identities, guard, timeline=timeline,
                    source=name, world_seen=str(window["world"]) in train_worlds,
                    min_vehicles=min_vehicles)
            except MaterializeError as exc:
                counts["n_rejected"] += 1
                errors.write(json.dumps({"set": name, "key": window.get("key"),
                                         "error": str(exc)}, allow_nan=False) + "\n")
                errors.flush()
                continue
            if counts["n_states"]:
                fh.write(",")
            json.dump(record, fh, separators=(",", ":"), allow_nan=False)
            counts["n_states"] += 1
            counts["paired" if record["paired"] else "unpaired"] += 1
            counts["pair_status:" + record["native_pair_status"]] += 1
            counts["seen_world" if record["world_seen_in_training"] else "unseen_world"] += 1
            if name != "seen_controls":
                counts["wall_status:" + record["wall_exposure_status"]] += 1
            if accepted_targets is not None:
                accepted_targets.add((str(window["world"]), float(window["t0_s"])))
        # Count AFTER streaming, so rejections never inflate n_states.
        fh.write('],"n_states":' + str(counts["n_states"]) + "}\n")
    return {"out": str(path), **{k: counts[k] for k in
            ("n_requested", "n_states", "n_rejected", "paired", "unpaired")},
            "counts": dict(counts)}


def run(args, *, guard=None, timeline_reader=None):
    if guard is None:
        from .guard import DataGuard
        guard = DataGuard(local_path(args.guard_spec), local_path(args.guard_tool),
                          [local_path(p) for p in args.allowed_root])
    if args.min_vehicles < 1:
        raise WallMaterializeError("min_vehicles must be positive")
    supplied_bindings = getattr(args, "pilot_bindings", None)
    if bool(args.pilot_index) != bool(args.pilot_run_dir or supplied_bindings):
        raise WallMaterializeError("pilot INDEX and explicit run bindings are required together")
    removed = _rows(read_json(args.wall_removed_windows, guard))
    retained_train = _rows(read_json(args.retained_train, guard))
    split_paths = args.retained_split or []
    if isinstance(split_paths, str):
        split_paths = [split_paths]
    retained = list(retained_train)
    for path in split_paths:
        retained.extend(_rows(read_json(path, guard)))
    for row in retained_train + retained:
        checked_record(row, guard)
        if "wall" in str(row.get("policy", "")).lower() or "wall" in str(row.get("arm", "")).lower():
            raise WallMaterializeError("retained split contains a wall controller")
        if str(row.get("origin_list", "train")) != "train" and row in retained_train:
            raise WallMaterializeError("retained training list has a non-train origin")
    train_worlds = {str(r["world"]) for r in retained_train}
    same = []
    selection_counts = Counter()
    for row in removed:
        # LPWO removal file also contains val-origin TRAIN metadata. That is not
        # an evaluation request for wall_same and must not open any run path.
        if row.get("origin_list") != "train":
            selection_counts["removed_non_train_origin"] += 1
            continue
        checked_record(row, guard)
        if str(row["world"]) not in train_worlds:
            selection_counts["removed_world_not_retained_in_training"] += 1
            continue
        same.append((row, row.get("timeline")))

    pilot = []
    bindings = _pairs(args.pilot_run_dir)
    if supplied_bindings:
        raw_bindings = read_json(supplied_bindings, guard)
        for key, value in raw_bindings.items():
            if key in bindings:
                raise WallMaterializeError("duplicate pilot binding")
            bindings.update(_pairs([f"{key}={value}"]))
    if args.pilot_index:
        if timeline_reader is None:
            from .wall_timeline import read_wall_timeline
            timeline_reader = read_wall_timeline
        index_rows = _rows(read_json(args.pilot_index, guard), "runs")
        indexed = {str(r.get("key")): r for r in index_rows}
        if len(indexed) != len(index_rows):
            raise WallMaterializeError("duplicate pilot INDEX keys")
        if set(bindings) - set(indexed):
            raise WallMaterializeError("explicit pilot binding absent from INDEX")
        # Preflight ALL requested source records before reading even the first run.
        requested = [indexed[k] for k in bindings]
        for row in requested:
            _pilot_path(row, bindings, guard)
        for row in requested:
            path = _pilot_path(row, bindings, guard)
            for child in ("branch_record.json", "wall_events.jsonl"):
                checked_path(path / "run" / child, guard)
            timeline = timeline_reader(path, guard=guard)
            checked_record({"split": row["split"], **timeline}, guard)
            if timeline.get("engaged") is not True or timeline.get("status") not in ENGAGED:
                selection_counts["pilot_excluded:" + str(timeline.get("status"))] += 1
                continue
            window = dict(row)
            if not window.get("pack"):
                if not args.pilot_pack:
                    raise WallMaterializeError("pilot INDEX needs pack or explicit --pilot-pack")
                window["pack"] = args.pilot_pack
            pilot.append((window, timeline))

    # Explicit, freshly guarded pilot timelines can complete partial LPWO
    # timeline metadata. Pilots also supplement wall_same only in TRAIN worlds
    # retained in training (never promote a val-origin removal to wall_same).
    def descriptor_key(row):
        return (row.get("pack"), row.get("world"), row.get("arm"), row.get("t0_s"))
    pilot_timelines = {descriptor_key(w): tl for w, tl in pilot}
    same = [(w, pilot_timelines.get(descriptor_key(w), tl)) for w, tl in same]
    same.extend((w, tl) for w, tl in pilot if str(w["world"]) in train_worlds
                and w.get("origin_list", "train") == "train")

    # Small window descriptors only; deduplicate before building large states.
    def unique(rows):
        seen = set()
        for row, timeline in rows:
            ident = descriptor_key(row)
            if ident not in seen:
                seen.add(ident)
                yield row, timeline
    same, pilot = list(unique(same)), list(unique(pilot))
    targets = {(str(w["world"]), float(w["t0_s"])) for w, _ in same + pilot}
    controls = list(unique((w, None) for w in retained
                          if (str(w["world"]), float(w["t0_s"])) in targets))
    pack_paths, id_paths = _pairs(args.pack), _pairs(args.identities)
    needed = {str(w.get("pack")) for w, _ in same + pilot + controls}
    if not needed <= pack_paths.keys() or not needed <= id_paths.keys():
        raise WallMaterializeError("every selected window needs explicit pack and identity roots")
    # Precheck every selected metadata record before opening packs.
    for row, _ in same + pilot + controls:
        checked_record(row, guard)
    out_dir = checked_path(args.out_dir, guard)
    out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "lpwo_wall_materialize_receipt/1", "status": "in_progress",
               "horizon_s": HORIZON_S, "shift_steps": SHIFT_STEPS,
               "history_steps": HISTORY_STEPS, "selection_counts": dict(selection_counts),
               "guard": {"spec": getattr(args, "guard_spec", None),
                         "tool": getattr(args, "guard_tool", None),
                         "allowed_roots": getattr(args, "allowed_root", [])},
               "sources": {"wall_removed_windows": str(args.wall_removed_windows),
                           "retained_train": str(args.retained_train),
                           "retained_split": str(args.retained_split or args.retained_train),
                           "pilot_index": args.pilot_index,
                           "pilot_bindings": {k: str(v) for k, v in bindings.items()},
                           "packs": {k: str(v) for k, v in pack_paths.items()},
                           "identities": {k: str(v) for k, v in id_paths.items()}},
                "sets": {}}
    with checked_path(out_dir / "MATERIALIZE_STARTED.json", guard).open("x", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2)
    try:
        packs = {n: GuardedPack(n, pack_paths[n], guard) for n in sorted(needed)}
        identities = {n: guarded_identity(n, id_paths[n], guard) for n in sorted(needed)}
        accepted_targets = set()
        with checked_path(out_dir / "REJECTIONS.jsonl", guard).open("x", encoding="utf-8") as errors:
            for name, windows in (("wall_same", same), ("wall_pilot", pilot),
                                  ("seen_controls", controls)):
                if name == "seen_controls":
                    windows = ((w, tl) for w, tl in windows
                               if (str(w["world"]), float(w["t0_s"])) in accepted_targets)
                receipt["sets"][name] = write_stream(
                    out_dir / (name + ".json"), name, windows, packs, identities,
                    guard, errors, train_worlds=train_worlds, min_vehicles=args.min_vehicles,
                    accepted_targets=accepted_targets if name != "seen_controls" else None)
        receipt["status"] = "complete_with_rejections" if any(
            s["n_rejected"] for s in receipt["sets"].values()) else "complete"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if hasattr(guard, "audit"):
            receipt["guard_audit"] = guard.audit()
        with checked_path(out_dir / "BUILD_RECEIPT.json", guard).open("x", encoding="utf-8") as fh:
            json.dump(receipt, fh, indent=2, allow_nan=False)
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wall-removed-windows", required=True)
    p.add_argument("--retained-train", required=True)
    p.add_argument("--retained-split", action="append", help="additional retained seen-control list (repeatable)")
    p.add_argument("--pack", action="append", required=True, help="NAME=PATH (repeatable)")
    p.add_argument("--identities", action="append", required=True, help="NAME=PATH (repeatable)")
    p.add_argument("--pilot-index")
    p.add_argument("--pilot-pack", help="pack name when absent from pilot INDEX rows")
    p.add_argument("--pilot-run-dir", action="append", default=[], help="INDEX_KEY=RUN_DIR (repeatable)")
    p.add_argument("--pilot-bindings", help="explicit JSON INDEX_KEY:RUN_DIR object (no directory discovery)")
    p.add_argument("--guard-spec", required=True)
    p.add_argument("--guard-tool", required=True)
    p.add_argument("--allowed-root", action="append", required=True)
    p.add_argument("--out-dir", required=True, help="new output directory (never overwrite)")
    p.add_argument("--min-vehicles", type=int, default=5)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        receipt = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
