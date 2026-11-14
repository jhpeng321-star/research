"""Metadata-first, selected-shard CAL source for a separately released eval.

This module does not grant CAL access.  The caller supplies the SHA-256 of a
separately registered release specification containing an allocation ID and a
data-release ID.  Only the selected CAL half's per-run shard files are opened;
the mixed packed ``outcomes.parquet`` is never opened.  The source deliberately
does not construct a predictor or claim that the frozen M1 checkpoint has a
compatible temporal/action contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .cal_split import CAL_CAL, CAL_TUNE, CAL_SPLIT_SCHEMA, LF_SHA_CAL_SPLIT_A4

SOURCE_SPEC_SCHEMA = "wm_n1_released_cal_source/1"
PACK_SCHEMA = "wm_n1_pack/1"
INDEX_SCHEMA = "wm_n1_corpus_pack/2"
SHARD_SCHEMA = "wm_n1_features_index/1"
IDEA_A_SCHEMA = "wm_n1_features_idea_a/1"
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_FORBIDDEN = re.compile(r"(?:holdout|chunk[\s_.-]*5|j4b|inc1|final[\s_.-]*test)", re.I)
_DT_S = 0.5
_HORIZONS = (10.0, 30.0, 60.0, 120.0)
_TRUTH_FILES = ("veh_kin.npy", "veh_mask.npy", "slot_index.npy",
                "target_mask.npy")


class RealSourceRefusal(PermissionError):
    """The requested source is unregistered, ambiguous, or sealed."""


def _sha(value, label):
    value = str(value)
    if not _SHA_RE.fullmatch(value):
        raise RealSourceRefusal(f"{label} needs an exact lowercase SHA-256")
    return value


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_pinned_json(path, expected_sha, label):
    path = Path(path)
    pin = _sha(expected_sha, label)
    if not path.is_file():
        raise RealSourceRefusal(f"{label} is absent or differs from its pin")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise RealSourceRefusal(f"{label} is absent or differs from its pin")
    result = json.loads(raw.decode("utf-8"))
    if not isinstance(result, dict):
        raise RealSourceRefusal(f"{label} must be a JSON object")
    return result


def _component(value, label):
    value = str(value)
    if not _COMPONENT_RE.fullmatch(value) or value in (".", ".."):
        raise RealSourceRefusal(f"unsafe {label} component {value!r}")
    if _FORBIDDEN.search(value):
        raise RealSourceRefusal(f"protected {label} component {value!r}")
    return value


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RealSourceRefusal(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise RealSourceRefusal(f"{label} must be finite")
    return result


def _item_key(item):
    batch = _component(item.get("batch"), "batch")
    world = _component(item.get("world"), "world")
    arm = _component(item.get("arm"), "arm")
    return f"{batch}/{world}__{arm}"


def select_cal_items(manifest: dict, index: dict, half_worlds) -> tuple:
    """Pure metadata validation/selection, before any outcome or array read.

    A selected item must map 1:1 to its pack INDEX run, with its snapshot,
    decision and outcome offsets inside that run.  A malformed item anywhere
    in the pack is refused, including an unrecognized split or sealed tag.
    """
    if manifest.get("schema") != PACK_SCHEMA or index.get("schema") != INDEX_SCHEMA:
        raise RealSourceRefusal("pack manifest/INDEX schema mismatch")
    items, runs = manifest.get("items"), index.get("runs")
    if not isinstance(items, list) or not items or not isinstance(runs, list):
        raise RealSourceRefusal("pack requires item and run metadata lists")
    expected_worlds = set(half_worlds)
    if not expected_worlds:
        raise RealSourceRefusal("released CAL half is empty")
    run_map = {}
    outcome_cursor = 0
    for run in runs:
        if not isinstance(run, dict):
            raise RealSourceRefusal("invalid pack run metadata")
        key = str(run.get("key", ""))
        if key in run_map or key != _item_key(run):
            raise RealSourceRefusal("duplicate or inconsistent pack run key")
        count = int(run["n_outcome_rows"])
        if count < 0:
            raise RealSourceRefusal("negative pack outcome count")
        if ("outcome_first" in run
                and int(run["outcome_first"]) != outcome_cursor):
            raise RealSourceRefusal("pack run outcome offset disagrees with order")
        run_map[key] = dict(run, _outcome_first=outcome_cursor)
        outcome_cursor += count
    selected = []
    seen = set()
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            raise RealSourceRefusal("invalid pack item metadata")
        key = _item_key(item)
        split = str(item.get("split", "")).upper()
        chunk = _component(item.get("chunk"), "chunk")
        if split not in ("TRAIN", "CALIBRATION"):
            raise RealSourceRefusal(f"unrecognized/protected pack split {split!r}")
        if key in seen:
            raise RealSourceRefusal(f"duplicate pack item {key!r}")
        seen.add(key)
        run = run_map.get(key)
        if run is None:
            raise RealSourceRefusal(f"pack item {key!r} lacks an INDEX run")
        if (str(run.get("split", "")).upper() != split
                or str(run.get("chunk")) != chunk
                or _finite(run.get("t0_s"), "run t0")
                != _finite(item.get("t0_s"), "item t0")):
            raise RealSourceRefusal(f"pack item/run metadata disagree for {key}")
        tick, plan_row = int(item["tick"]), int(item["plan_row"])
        lo, hi = int(run["tick_start"]), int(run["tick_end"])
        dec_lo, dec_hi = int(run["dec_start"]), int(run["dec_end"])
        if not (lo <= tick < hi and dec_lo <= plan_row < dec_hi):
            raise RealSourceRefusal(f"pack item offsets escape run {key}")
        first = int(run["_outcome_first"])
        end = first + int(run["n_outcome_rows"])
        positions = item.get("outcome_rows")
        if not isinstance(positions, list) or positions != list(range(first, end)):
            raise RealSourceRefusal(f"pack outcome offsets disagree for {key}")
        if split == "CALIBRATION" and str(item["world"]) in expected_worlds:
            selected.append((pos, item, run))
    if set(run_map) != seen:
        raise RealSourceRefusal("pack INDEX has unmatched runs")
    if not selected:
        raise RealSourceRefusal("released CAL half has no pack items")
    if {str(item["world"]) for _, item, _ in selected} != expected_worlds:
        raise RealSourceRefusal("released CAL worlds and selected pack worlds differ")
    return tuple(selected)


class ReleasedCalSource:
    """EvalSource whose data reads are limited to pinned selected shards.

    The release specification is externally registered and SHA-pinned.  It has
    fields ``allocation_id``, ``data_release_id``, ``half``, ``pack_dir``,
    ``manifest_sha256``, ``index_sha256``, ``cal_split_path``,
    ``shards_root`` and ``selected_shards``.  The last mapping has one entry
    per selected pack run: ``{key: {done_sha256, files: {name: sha256}}}``.
    A mere on-disk specification is not authorization; the calling allocation
    must independently supply its registered raw SHA-256.
    """

    def __init__(self, release_spec_path, *, expected_release_sha256):
        spec = _read_pinned_json(release_spec_path, expected_release_sha256,
                                 "CAL data release specification")
        if spec.get("schema") != SOURCE_SPEC_SCHEMA:
            raise RealSourceRefusal("unknown CAL data release schema")
        for field in ("allocation_id", "data_release_id"):
            if not str(spec.get(field, "")).strip():
                raise RealSourceRefusal(f"release lacks {field}")
        for field in ("source_head", "checkpoint_member_sha256"):
            if not str(spec.get(field, "")).strip():
                raise RealSourceRefusal(f"release lacks {field}")
        _sha(spec["checkpoint_member_sha256"], "checkpoint member")
        half = spec.get("half")
        if half not in (CAL_TUNE, CAL_CAL):
            raise RealSourceRefusal("release must name one CAL half")
        split_path = Path(spec["cal_split_path"])
        if not split_path.is_file():
            raise RealSourceRefusal("registered CAL split is absent")
        raw = split_path.read_bytes()
        lf_sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")
                            .replace(b"\r", b"\n")).hexdigest()
        if lf_sha != LF_SHA_CAL_SPLIT_A4:
            raise RealSourceRefusal("CAL split differs from frozen A4 identity")
        split = json.loads(raw.decode("utf-8"))
        if split.get("schema") != CAL_SPLIT_SCHEMA:
            raise RealSourceRefusal("CAL split schema mismatch")
        tune, cal = split.get(CAL_TUNE), split.get(CAL_CAL)
        if (not isinstance(tune, list) or not isinstance(cal, list)
                or not tune or not cal or set(tune) & set(cal)
                or len(tune) != len(set(tune)) or len(cal) != len(set(cal))):
            raise RealSourceRefusal("CAL halves are invalid or overlapping")
        pack_dir = Path(spec["pack_dir"])
        manifest = _read_pinned_json(pack_dir / "manifest.json",
                                     spec["manifest_sha256"], "pack manifest")
        index = _read_pinned_json(pack_dir / "INDEX.json",
                                  spec["index_sha256"], "pack INDEX")
        pack_cal_worlds = {str(item.get("world")) for item in
                           manifest.get("items", []) if isinstance(item, dict)
                           and str(item.get("split", "")).upper() == "CALIBRATION"}
        # Owner decision 2026-10-04 ("Approve the scorer fix"): A4 split the
        # 23 chunk-2..4 CAL worlds; the pack's extra chunk1/chunk1_fix27 CAL
        # worlds belong to NEITHER half and stay sealed (never selected/opened).
        a4_worlds = set(tune) | set(cal)
        if not a4_worlds <= pack_cal_worlds:
            raise RealSourceRefusal("frozen A4 halves name worlds absent from the pack")
        extra = pack_cal_worlds - a4_worlds
        extra_batches = {str(item.get("batch")) for item in manifest.get("items", [])
                         if isinstance(item, dict)
                         and str(item.get("split", "")).upper() == "CALIBRATION"
                         and str(item.get("world")) in extra}
        if not extra_batches <= {"chunk1", "chunk1_fix27"}:
            raise RealSourceRefusal(
                "pack CAL worlds outside both A4 halves come from an unexpected batch")
        self.sealed_unassigned_cal_worlds = sorted(extra)
        selected = select_cal_items(manifest, index, split[half])
        pins = spec.get("selected_shards")
        if not isinstance(pins, dict) or set(pins) != {
                _item_key(item) for _, item, _ in selected}:
            raise RealSourceRefusal("release must pin exactly selected CAL shards")
        self.spec = spec
        self.manifest = manifest
        self.index = index
        self.selected = selected
        self.pins = pins
        self.half = half
        self.shards_root = Path(spec["shards_root"])
        self.sign_threshold = 0.0

    def _shard(self, item):
        batch, world, arm = (_component(item[name], name)
                             for name in ("batch", "world", "arm"))
        path = self.shards_root / batch / world / arm
        root = self.shards_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RealSourceRefusal("selected shard escapes released root")
        return path

    def _checked_file(self, item, name):
        key = _item_key(item)
        rec = self.pins[key]
        files = rec.get("files")
        if not isinstance(files, dict) or name not in files:
            raise RealSourceRefusal(f"selected shard {key} lacks {name} pin")
        shard = self._shard(item)
        idea_a = name in ("veh_kin.npy", "hdv_base.npy", "cav_plan_ch.npy",
                          "event_tok.npy", "event_mask.npy")
        receipt_name = "IDEA_A_DONE.json" if idea_a else "DONE.json"
        receipt_pin = (rec.get("idea_a_done_sha256") if idea_a
                       else rec.get("done_sha256"))
        done = _read_pinned_json(shard / receipt_name, receipt_pin,
                                 f"{key} {receipt_name}")
        expected_schema = IDEA_A_SCHEMA if idea_a else SHARD_SCHEMA
        if done.get("schema") != expected_schema:
            raise RealSourceRefusal(f"selected shard {key} receipt schema mismatch")
        recorded = (done.get("arrays", {}).get(name[:-4], {}).get("sha256")
                    if idea_a else done.get("outputs", {}).get(
                        name[:-4] if name.endswith(".npy") else name))
        if (str(done.get("world")) != str(item["world"])
                or str(done.get("arm")) != str(item["arm"])
                or str(done.get("batch")) != str(item["batch"])
                or str(done.get("chunk")) != str(item["chunk"])
                or str(done.get("split", "")).upper() != "CALIBRATION"
                or _finite(done.get("t0_s"), "DONE t0")
                != _finite(item.get("t0_s"), "item t0")
                or recorded != files[name]):
            raise RealSourceRefusal(f"selected shard {key} DONE conflicts with pack/release")
        file_path = shard / name
        if not file_path.is_file() or _hash_file(file_path) != _sha(files[name], name):
            raise RealSourceRefusal(f"selected shard {key}/{name} differs from pin")
        return file_path

    def receipt_for(self, item):
        """Read and verify the pinned ``DONE.json`` receipt only.

        This is metadata: it opens the receipt (SHA-pinned) but never the
        shard's payload.  It exists so zero-label detection can be decided
        before any ``outcomes.parquet`` or ``.npy`` is opened.
        """
        key = _item_key(item)
        rec = self.pins[key]
        shard = self._shard(item)
        done = _read_pinned_json(shard / "DONE.json", rec.get("done_sha256"),
                                 f"{key} DONE.json")
        if done.get("schema") != SHARD_SCHEMA:
            raise RealSourceRefusal(f"selected shard {key} receipt schema mismatch")
        if (str(done.get("world")) != str(item["world"])
                or str(done.get("arm")) != str(item["arm"])
                or str(done.get("batch")) != str(item["batch"])
                or str(done.get("chunk")) != str(item["chunk"])
                or str(done.get("split", "")).upper() != "CALIBRATION"
                or _finite(done.get("t0_s"), "DONE t0")
                != _finite(item.get("t0_s"), "item t0")):
            raise RealSourceRefusal(
                f"selected shard {key} DONE conflicts with pack")
        return done

    def outcome_rows_for(self, item, run, position=0, *, columns=None):
        """Read one selected shard's declared label rows (payload open).

        The caller must already have cleared the shard as non-zero-label.
        Contract (1) is preserved: a decision time that differs from the t0
        snapshot and a repeated tau fail closed, and the label count must
        equal the pack INDEX.  ``columns`` restricts the parquet projection.
        """
        import pandas as pd
        from .protocol import OutcomeRow

        if str(item.get("arm")) in ("base", "native_base"):
            raise RealSourceRefusal(
                "excluded (b) base/native_base item must not be read; "
                "decision time differs from its snapshot")
        path = self._checked_file(item, "outcomes.parquet")
        frame = pd.read_parquet(path, columns=None if columns is None
                                else list(columns))
        if len(frame) != int(run["n_outcome_rows"]):
            raise RealSourceRefusal(
                "selected shard label count differs from pack INDEX")
        rows = []
        seen_tau = set()
        for record in frame.to_dict("records"):
            if (str(record.get("world")) != str(item["world"])
                    or str(record.get("source")) != str(item["arm"])
                    or str(record.get("split", "")).upper() != "CALIBRATION"):
                raise RealSourceRefusal("selected shard label identity mismatch")
            t0 = _finite(record.get("t0_s"), "label t0")
            decision = _finite(record.get("t"), "label decision time")
            tau = _finite(record.get("tau"), "label tau")
            if t0 != _finite(item["t0_s"], "item t0"):
                raise RealSourceRefusal(
                    "selected label t0 differs from the item t0")
            if decision != t0:
                # Owner-approved fix (a), 2026-10-04: a run holds labels for
                # several decision times; the item's targets are the rows at
                # its own t0 (same rule as training's t0 mapper).
                continue
            if tau in seen_tau:
                raise RealSourceRefusal(
                    "multiple decision labels share one tau; M1 training "
                    "kept only the last and its time binding is ambiguous")
            seen_tau.add(tau)
            available = _finite(record.get("tau_available"), "tau availability")
            if available <= 0:
                continue
            rows.append(OutcomeRow(
                world=str(item["world"]), t0_s=t0, decision_t_s=decision,
                tau_s=tau, policy=str(item["policy"]),
                y_delay_s=_finite(record.get("total_delay_ref_s"), "delay"),
                collision=_finite(record.get("collision_any"), "collision"),
                ttc_lt3_s=_finite(record.get("ttc_lt3_veh_s"), "TTC"),
                hard_brake_s=_finite(record.get("hard_brake_s"), "braking"),
                n_vehicles=_finite(record.get("n_vehicles_present"), "vehicles"),
                features={"pack_item_position": position,
                          "run_key": _item_key(item),
                          "arm": item["arm"], "chunk": item["chunk"]}))
        if not seen_tau:
            raise RealSourceRefusal(
                f"selected shard {_item_key(item)} has no label rows at its t0")
        return rows

    def outcome_rows(self, split="cal"):
        if split != "cal":
            raise RealSourceRefusal("released source allows only CAL")
        rows = []
        for pos, item, run in self.selected:
            rows.extend(self.outcome_rows_for(item, run, pos))
        if not rows:
            raise RealSourceRefusal("selected CAL half has no available label rows")
        return rows

    def rollout_rows(self, split="cal", *, horizons_s=_HORIZONS):
        """Truth at exact horizons for surviving members of the t0 roster.

        No global packed-array indexing or clamping into a following run is
        permitted.  Slot identity and target masks must prove survival through
        the largest horizon; the official predictor/metric bridge still needs
        its own matching-roster contract before this is scored.
        """
        if split != "cal":
            raise RealSourceRefusal("released source allows only CAL")
        import numpy as np
        from .protocol import RolloutRow

        ratios = tuple(_finite(h, "horizon") / _DT_S for h in horizons_s)
        if any(not ratio.is_integer() for ratio in ratios):
            raise RealSourceRefusal("rollout horizons must lie on the 0.5 s grid")
        steps = tuple(int(ratio) for ratio in ratios)
        if not steps or any(step <= 0 for step in steps):
            raise RealSourceRefusal("rollout horizons must be positive")
        result = []
        for pos, item, run in self.selected:
            arrays = {name: np.load(self._checked_file(item, name), mmap_mode="r")
                      for name in _TRUTH_FILES}
            local_tick = int(item["tick"]) - int(run["tick_start"])
            if (local_tick < 0 or local_tick + max(steps) >= int(run["n_ticks"])
                    or any(len(a) != int(run["n_ticks"]) for a in arrays.values())):
                raise RealSourceRefusal("truth horizon crosses the selected run boundary")
            kin, mask, slot, target = (arrays[name] for name in _TRUTH_FILES)
            if (kin.ndim != 3 or kin.shape[-1] != 7
                    or mask.shape != kin.shape[:2]
                    or slot.shape != mask.shape
                    or target.shape != mask.shape):
                raise RealSourceRefusal("selected shard truth channel shapes disagree")
            initial = np.flatnonzero(mask[local_tick].astype(bool))
            if not len(initial):
                raise RealSourceRefusal("selected t0 truth roster is empty")
            valid = initial.copy()
            for off in range(1, max(steps) + 1):
                live = (target[local_tick + off - 1, valid].astype(bool)
                        & mask[local_tick + off, valid].astype(bool))
                same_id = slot[local_tick + off, valid] == slot[local_tick, valid]
                valid = valid[live & same_id]
                if not len(valid):
                    raise RealSourceRefusal("no t0 vehicles survive the full truth horizon")
            positions = np.stack([np.asarray(kin[local_tick + s, valid, 0], float)
                                  for s in steps])
            speeds = np.stack([np.asarray(kin[local_tick + s, valid, 4], float)
                               for s in steps])
            if not np.isfinite(positions).all() or not np.isfinite(speeds).all():
                raise RealSourceRefusal("selected truth contains nonfinite states")
            result.append(RolloutRow(
                world=str(item["world"]), t0_s=_finite(item["t0_s"], "t0"),
                policy=str(item["policy"]), true_pos=positions, true_speed=speeds,
                horizons_s=tuple(horizons_s),
                true_dist=float(np.mean(np.abs(positions[-1] - kin[local_tick, valid, 0]))),
                true_speed_scale=float(np.mean(kin[local_tick, valid, 4])),
                features={"pack_item_position": pos, "run_key": _item_key(item),
                          "t0_slots": valid.tolist(),
                          "t0_slot_ids": slot[local_tick, valid].tolist()}))
        return result
