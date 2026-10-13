"""TRAIN trajectory datasets: in-memory arrays, trajectory files and lazy A11 pack windows.

Every record is a dict of numpy arrays in the ``states [T+1,K,7]`` contract (``s``, lane
one-hot(3), ``v``, ``a``, ``length``) with presence, loss-validity and identity sidecars;
:func:`wms.data.collate.collate` stacks records into batches.  The pack-backed dataset is
lazy: windows are decoded one at a time from the memory-mapped packs.

``from_real_packs`` draws from the *global* torch RNG for start-shift windows
(``torch.rand`` then ``torch.randint``); draw order is part of the contract (golden batches
seed the RNG per item).  Pack-backed datasets are single-process (``NUM_WORKERS == 0``).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from wms.core.constants import CONTROLLER_VOCAB, controller_index
from wms.core.errors import TrainDataRefusal
from wms.core.io import run_key
from wms.data.action_timeline import CONTROLLERS, SCHEMA, ActionTimeline
from wms.data.controller_hygiene import HYGIENE_MODES, DspeedRun
from wms.data.corpus import PackCorpus
from wms.data.history import history_spec, long_history_features
from wms.data.junction_fix import (JUNCTION_FIX_MODES, fill_junction_rows, pre_t0_rows,
                                   run_prefix)
from wms.data.pack_history import entry_features, prior_history
from wms.data.presence import junction_carry_presence, junction_mask_from_lane
from wms.data.sidecars import read_identity_rows
from wms.data.sources import validate_train_source
from wms.data.windows import select_lpo_positions

#: Pack ``policy``/``arm`` strings that name a fixed controller, mapped to timeline controllers.
_FIXED_ACTION_ALIASES = {"pi": "pi-external-u", "fleet-pi": "pi-external-u",
                         "dspeed": "direct", "direct-speed": "direct",
                         "direct_speed": "direct", "trajectory": "direct"}
#: ``veh_step`` ownership column -> timeline controller id (native, damping, pi, native).
_OWNERSHIP_TO_ACTION = (0, 1, 2, 0)


def timeline_controller_ids(ids: np.ndarray, arrays: dict) -> np.ndarray:
    """Normalize documented compact/legacy ids without guessing overlapping labels.

    ``wms_action_timeline/1`` declares the compact action vocabulary. An explicit
    ``controller_vocab`` array can also declare either vocabulary (including the six-label
    legacy vocabulary before ``direct`` was added). Unmarked legacy arrays with ids >= 4
    retain the historical adapter; ids 0 and 2 have identical meanings after normalization.
    Unmarked ids 1/3 are ambiguous and are refused.
    """
    schema = arrays.get("controller_timeline_schema")
    if schema is None and "schema" in arrays:
        candidate = np.asarray(arrays["schema"])
        if candidate.size == 1 and str(candidate.item()).startswith("wms_action_timeline/"):
            schema = str(candidate.item())
    if schema is not None:
        schema_array = np.asarray(schema)
        if schema_array.size != 1 or str(schema_array.item()) != SCHEMA:
            raise ValueError("unsupported controller_timeline_schema")
    vocab = arrays.get("controller_vocab")
    legacy = None
    if vocab is not None:
        vocabulary = tuple(np.asarray(vocab).reshape(-1).tolist())
        if vocabulary == CONTROLLERS:
            legacy = False
        elif vocabulary in (CONTROLLER_VOCAB, CONTROLLER_VOCAB[:-1]):
            legacy = True
        else:
            raise ValueError("unknown controller_vocab for controller_timeline")
        if schema is not None and legacy:
            raise ValueError("controller_vocab conflicts with action timeline schema")
    if ids.dtype.kind in "US":
        if legacy:
            raise ValueError("legacy controller_timeline must use numeric ids")
        return np.asarray([CONTROLLERS.index(label) for label in ids.reshape(-1)],
                          dtype=np.int64).reshape(ids.shape)
    if ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("controller_timeline must contain non-negative integer ids")
    if legacy is None:
        if schema is not None:
            legacy = False
        elif np.any(ids >= len(CONTROLLERS)):
            legacy = True
        elif np.any((ids == 1) | (ids == 3)):
            raise ValueError("ambiguous controller_timeline ids; declare "
                             "wms_action_timeline/1 schema or controller_vocab")
        else:
            legacy = False
    if legacy:
        legacy_to_action = np.asarray((0, 0, 2, 1, 0, 0, 3), dtype=np.int64)
        limit = len(CONTROLLER_VOCAB) if vocab is None else len(vocabulary)
        if np.any(ids >= limit):
            raise ValueError("controller_timeline contains an unknown legacy controller id")
        return legacy_to_action[ids]
    if np.any(ids >= len(CONTROLLERS)):
        raise ValueError("controller_timeline contains an unknown action controller id")
    return ids.astype(np.int64)


def _timeline_dict(timeline: ActionTimeline) -> dict:
    return {"controller": timeline.controller, "acceleration_mps2": timeline.acceleration_mps2,
            "target_speed_mps": timeline.target_speed_mps,
            "control_dt_s": timeline.control_dt_s}


def _n_control_steps(horizon: int) -> int:
    """Number of 1 s control steps covering a window of ``horizon`` 0.5 s ticks."""
    return max(1, int(np.ceil((horizon - 1) * 0.5)))


def _states_from_teacher(teacher: np.ndarray, length0: np.ndarray) -> np.ndarray:
    """``states [T+1,K,7]`` from the ``(s, lane, v, a)`` teacher and per-slot lengths."""
    lane = np.clip(teacher[..., 1].astype(np.int64), 0, 2)
    one_hot = np.eye(3, dtype=np.float32)[lane]
    return np.concatenate(
        (teacher[..., 0:1], one_hot, teacher[..., 2:4],
         np.broadcast_to(length0[:, None], (teacher.shape[0], teacher.shape[1], 1))), axis=-1)


class TrajectoryDataset(Dataset):
    """Sequence dataset with explicit identity/presence sidecars (eager or lazy)."""

    def __init__(self, records: list[dict], *, loader=None, receipt: Optional[dict] = None):
        if not records and loader is None:
            raise ValueError("TRAIN trajectory dataset is empty")
        self.records = records
        self._loader = loader
        self.receipt = receipt or {"mode": "arrays", "n_records": len(records)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self._loader(index) if self._loader is not None else self.records[index]

    # ------------------------------------------------------------- in-memory ----
    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], *, split: str = "TRAIN",
                    history_s=None, history_stride_s=1.0):
        if str(split).upper() != "TRAIN":
            raise TrainDataRefusal("TrajectoryDataset only accepts TRAIN")
        states = np.asarray(arrays["states" if "states" in arrays else "veh_kin"],
                            dtype=np.float32)
        if states.ndim == 3:
            states = states[None]
        if states.ndim != 4 or states.shape[-1] != 7 or states.shape[1] < 2:
            raise ValueError("states must be [N,T+1,K,7], with T >= 1")
        n, t1, k, _ = states.shape
        history_spec(history_s, history_stride_s)
        present = np.asarray(arrays.get("present", np.ones((n, t1, k), bool)))
        if present.ndim == 2:
            present = present[None]
        slot = np.asarray(arrays.get("slot_index", np.broadcast_to(
            np.arange(k, dtype=np.int64), (n, t1, k))))
        if slot.ndim == 2:
            slot = slot[None]
        identities = arrays.get("identities")
        if identities is not None:
            identities = np.asarray(identities)
            if identities.ndim == 2:
                identities = identities[None]
        is_cav = arrays.get("is_cav")
        if is_cav is not None:
            is_cav = np.asarray(is_cav, dtype=bool)
            if is_cav.ndim == 2:
                is_cav = is_cav[None]
        cav = arrays.get("cav_teacher", arrays.get("cav_states"))
        # Explicit junction transits may be supplied (``junction``); otherwise a slot that
        # still carries a vehicle but has no modelled lane one-hot is a junction
        # internal-lane transit.  Both feed ``junction_carry_presence`` so one junction step
        # never permanently drops the slot.
        junction = arrays.get("junction")
        if junction is not None:
            junction = np.asarray(junction, dtype=bool)
            if junction.ndim == 2:
                junction = junction[None]
        timeline_ids, timeline_accel, timeline_target = cls._array_timelines(arrays, n, k)
        records = []
        for i in range(n):
            actual_present = present[i].astype(bool)
            on_road = states[i, :, :, 1:4].sum(-1) > 0
            if junction is not None:
                jn = junction[i] if junction.ndim == 3 else junction
            else:
                jn = junction_mask_from_lane(states[i, :, :, 1:4], actual_present)
            if identities is None:
                present_eff, jn_eff = actual_present, jn
            else:
                first_id = identities[i, 0]
                same = (identities[i] == first_id) & (first_id >= 0)
                present_eff, jn_eff = actual_present & same, jn & same
            alive, _ = junction_carry_presence(present_eff, jn_eff)
            rec = {"states": states[i], "present": alive,
                   "loss_valid": alive & on_road,
                   "slot_index": slot[i].astype(np.int64)}
            if history_s is not None:
                rec["driver_hist"], rec["driver_hist_mask"] = cls._array_history(
                    arrays, i, n, k, states, actual_present, alive,
                    history_s, history_stride_s)
            if identities is not None:
                rec["identities"] = identities[i]
            if is_cav is not None:
                cav0 = is_cav[i, 0] if is_cav.ndim == 3 else is_cav[i]
                rec["is_cav"] = np.broadcast_to(cav0, alive.shape).copy() & alive
            if cav is not None:
                rec["cav_teacher"] = np.asarray(cav[i], dtype=np.float32)
            elif is_cav is not None:
                # Structured rollouts teacher-force CAVs; the recorded CAV slot of ``states``
                # IS the true next state, so no controller code is needed when no explicit
                # teacher is supplied.  [T,K,4]=s,lane,v,a.
                rec["cav_teacher"] = np.stack(
                    (states[i, ..., 0],
                     states[i, ..., 1:4].argmax(-1).astype(np.float32),
                     states[i, ..., 4], states[i, ..., 5]), axis=-1).astype(np.float32)
            if timeline_ids is not None:
                ActionTimeline(timeline_ids[i], timeline_accel[i], timeline_target[i])
                rec["action_timeline"] = {
                    "controller": timeline_ids[i].astype(np.int64),
                    "acceleration_mps2": timeline_accel[i],
                    "target_speed_mps": timeline_target[i], "control_dt_s": 1.0}
            records.append(rec)
        return cls(records)

    @staticmethod
    def _array_timelines(arrays: dict, n: int, k: int):
        """Validated ``(controller ids, issued accel, issued target)`` or ``(None,)*3``."""
        timeline_ids = arrays.get("controller_timeline")
        if timeline_ids is None:
            return None, None, None
        timeline_accel = arrays.get("issued_acceleration_mps2")
        timeline_target = arrays.get("issued_target_speed_mps")
        timeline_ids = np.asarray(timeline_ids)
        if timeline_ids.ndim == 2:
            timeline_ids = timeline_ids[None]
        if timeline_ids.ndim != 3 or timeline_ids.shape[0] != n \
                or timeline_ids.shape[1] < 1 or timeline_ids.shape[2] != k:
            raise ValueError("controller_timeline must be [N,control_steps,K]")
        timeline_ids = timeline_controller_ids(timeline_ids, arrays)
        if np.any(timeline_ids == CONTROLLERS.index("direct")) \
                and (timeline_accel is None or timeline_target is None):
            raise ValueError("direct actions require both issued command fields")
        commands = []
        for command in (timeline_accel, timeline_target):
            command = (np.zeros_like(timeline_ids, dtype=np.float32) if command is None
                       else np.asarray(command, dtype=np.float32))
            if command.ndim == 2:
                command = command[None]
            if command.shape != timeline_ids.shape:
                raise ValueError("issued command timeline shape differs from controller_timeline")
            commands.append(command)
        return timeline_ids, commands[0], commands[1]

    @staticmethod
    def _array_history(arrays, i, n, k, states, actual_present, alive,
                       history_s, history_stride_s):
        """Optional explicit pre-t0 native observations of window ``i`` -> long history."""
        if "history_kin" in arrays:
            hk = np.asarray(arrays["history_kin"], np.float32)
            hm = np.asarray(arrays["history_mask"], bool)
            ht = np.asarray(arrays["history_target_mask"], bool)
            if hk.ndim == 3:
                hk, hm, ht = hk[None], hm[None], ht[None]
            if hk.shape[0] != n or hk.shape[2:] != (k, 7):
                raise ValueError("history_kin must be [N,H,K,7]")
            if not np.array_equal(hk[i, -1], states[i, 0]):
                raise ValueError("history_kin must end at the window t0 state")
            raw, observed, continuous = hk[i], hm[i], ht[i]
        else:
            # Without explicit history an array window starts at t0 and has one observed tick.
            raw, observed = states[i, :1], actual_present[:1]
            continuous = np.zeros((0, k), bool)
        features, mask = long_history_features(
            torch.as_tensor(raw), torch.as_tensor(observed), torch.as_tensor(continuous),
            torch.as_tensor(alive[0]), history_s=history_s,
            history_stride_s=history_stride_s)
        return features.numpy(), mask.numpy()

    # ------------------------------------------------------------ trajectories ---
    @classmethod
    def from_pack(cls, path: str | Path, *, history_s=None,
                  history_stride_s=1.0) -> "TrajectoryDataset":
        """Load a ``.npz`` trajectory file or a directory of ``states.npy`` sidecars."""
        p = validate_train_source(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_file():
            if p.suffix.lower() != ".npz":
                raise ValueError("trajectory source file must be .npz")
            with np.load(p, allow_pickle=False) as data:
                return cls.from_arrays({k: data[k] for k in data.files},
                                       history_s=history_s, history_stride_s=history_stride_s)
        manifest_path = p / "manifest.json"
        if manifest_path.is_file():
            validate_train_source(p, json.loads(manifest_path.read_text(encoding="utf-8")))
        for name in ("trajectory.npz", "states.npz"):
            candidate = p / name
            if candidate.is_file():
                return cls.from_pack(candidate, history_s=history_s,
                                     history_stride_s=history_stride_s)
        for name in ("states.npy", "veh_kin.npy"):
            candidate = p / name
            if candidate.is_file():
                return cls._from_state_files(p, candidate, history_s, history_stride_s)
        raise TrainDataRefusal(
            f"{p} has no trajectory sequence; snapshot-only A11 packs are refused")

    @classmethod
    def _from_state_files(cls, root: Path, states_file: Path, history_s, history_stride_s):
        states = np.load(states_file, mmap_mode="r", allow_pickle=False)
        if states.ndim < 4:
            raise TrainDataRefusal(f"{states_file} has snapshots only; no multi-step sequence")
        arrays = {"states": states}
        optional = ["present.npy", "slot_index.npy", "cav_teacher.npy",
                    "controller_timeline.npy", "issued_acceleration_mps2.npy",
                    "issued_target_speed_mps.npy", "controller_vocab.npy",
                    "controller_timeline_schema.npy"]
        if history_s is not None:
            optional += [f"{name}.npy" for name in
                         ("history_kin", "history_mask", "history_target_mask")]
        for sidecar in optional:
            f = root / sidecar
            if f.is_file():
                arrays[sidecar.removesuffix(".npy")] = np.load(
                    f, mmap_mode="r", allow_pickle=False)
        return cls.from_arrays(arrays, history_s=history_s, history_stride_s=history_stride_s)

    # --------------------------------------------------------------- A11 packs ---
    @classmethod
    def from_real_packs(cls, paths: list[str | Path], *, identity_dirs: list[str | Path],
                        overlays: Optional[list[str | Path]] = None,
                        horizon_steps: int = 60,
                        start_shift_prob: float = 0.0,
                        exclude_keys: Optional[set[str]] = None,
                        select_windows: Optional[set[tuple[str, str]]] = None,
                        window_list: Optional[str | Path] = None,
                        exclude_window_list: Optional[str | Path | list[str]] = None,
                        pack_name_list: Optional[list[str]] = None,
                        entries_v2: bool = False,
                        timelines: Optional[dict] = None,
                        item_filter: Optional[Callable] = None,
                        identity_env_dir: Optional[str | Path] = None,
                        history_s=None, history_stride_s=1.0,
                        junction_fix: str = "off",
                        controller_hygiene: str = "off") -> "TrajectoryDataset":
        """Build a lazy TRAIN view over A11 packs and verified identity windows.

        ``PackCorpus`` performs the authoritative split and INDEX range checks.  The
        sidecars are read through the literal workspace paths given; no resolved external
        target is followed and no source array is copied.

        ``exclude_keys`` drops windows whose pack run key is listed (leave-PI-out); every
        listed key must match a corpus window or the load refuses.  ``select_windows``
        (``(pack, key)`` pairs) keeps only a validation subset; every requested pair must
        match or the load refuses.  Both filters run after the corpus split checks and
        before identity/position validation.

        Switching/direct packs (:mod:`wms.data.mixdata`): ``timelines`` maps a pack index to
        a provider whose ``timeline(position, n_control)`` returns that item's
        ``wms_action_timeline/1`` arrays bound to the pack slots (it replaces the legacy
        constant timeline); ``item_filter(corpus, kept_orig)`` returns the kept subset and a
        receipt (metadata/mmap only; used for the zero-shot held-out guard);
        ``identity_env_dir`` overrides the process-wide ``WM_N1_TRAIN_IDENTITIES`` for this
        corpus so two corpora with different identity sidecars can coexist.  Start-shift
        windows are refused with a timeline because their slot columns would no longer match
        the timeline.

        ``window_list`` keeps only listed ``(batch, world, arm, t0_s)`` windows (the
        leave-one-controller-out TRAIN set); ``exclude_window_list`` drops a bare key list.
        Both are metadata-only and fail closed on an unmatched keep key.

        ``junction_fix`` (``data.junction_fix``, :mod:`wms.data.junction_fix`): ``"off"``
        (default, unchanged windows) or ``"ffill"`` (junction-transit rows keep the vehicle's
        previous lane and an interpolated ``s``, the evaluation convention).

        ``controller_hygiene`` (``data.controller_hygiene``, :mod:`wms.data.controller_hygiene`):
        ``"off"`` (default, unchanged) or ``"strict"``: dspeed (trajectory-arm) windows may not
        overlap a ``merge_aimed`` / ``gap_opening`` segment of a directed CAV (items without an
        allowed start are dropped; a refused draw moves to an allowed start), their timelines
        mark only the directed CAVs ``direct`` (commands unknown: ``command_known = False``) and
        every pack timeline carries ``command_known``.
        """
        if junction_fix not in JUNCTION_FIX_MODES:
            raise ValueError(f"junction_fix must be one of {JUNCTION_FIX_MODES}")
        if controller_hygiene not in HYGIENE_MODES:
            raise ValueError(f"controller_hygiene must be one of {HYGIENE_MODES}")
        if not paths:
            raise ValueError("at least one trajectory pack is required")
        history_spec(history_s, history_stride_s)
        if len(identity_dirs) != len(paths):
            raise TrainDataRefusal(
                f"expected one identity sidecar per pack ({len(paths)}), got "
                f"{len(identity_dirs)}")
        if overlays is not None and len(overlays) != max(0, len(paths) - 1):
            raise TrainDataRefusal(
                f"expected one overlay per extra pack ({len(paths) - 1}), got {len(overlays)}")
        for path in paths:
            validate_train_source(path)
        if window_list is not None:
            validate_train_source(window_list)
        if timelines and float(start_shift_prob) > 0.0:
            raise TrainDataRefusal(
                "start-shift windows are refused with action timelines: the shifted "
                "window's slot columns do not match the t0-bound timeline columns")
        corpus = PackCorpus.open(
            paths[0], extra_pack_dirs=paths[1:], extra_overlay_dirs=overlays,
            extra_identity_dirs=identity_dirs[1:], horizon_steps=int(horizon_steps))
        if len(corpus) == 0:
            raise TrainDataRefusal("A11 TRAIN selection is empty")
        if identity_env_dir is not None:
            corpus.use_identity_dir(str(identity_env_dir))

        names = list(pack_name_list) if pack_name_list else [Path(p).name for p in paths]
        if len(names) != len(paths):
            raise TrainDataRefusal(
                f"pack_name_list has {len(names)} entries for {len(paths)} packs")
        identity_rows = [read_identity_rows(sidecar, corpus_k=corpus.k_slots)
                         for sidecar in identity_dirs]
        # Optional structured (batch, world, arm, t0_s) leave-controller-out keep/exclude
        # filter.  Metadata-only and fail-closed.
        filter_receipt = None
        lpo_keep: Optional[set[int]] = None
        if window_list is not None or exclude_window_list is not None:
            lpo_selected, filter_receipt = select_lpo_positions(
                corpus, names, window_list=window_list,
                exclude_window_list=exclude_window_list)
            lpo_keep = set(lpo_selected)
        # Filter by pack run key AFTER the corpus split checks and BEFORE any
        # identity/position validation (leave-PI-out exclusion + val selection).
        kept = _filter_run_keys(corpus, [Path(p).name for p in paths], lpo_keep,
                                exclude_keys, select_windows)
        kept_orig = kept.positions
        item_filter_receipt = None
        if item_filter is not None:
            kept_orig, item_filter_receipt = item_filter(corpus, list(kept_orig))
            kept_orig = [int(i) for i in kept_orig]
            if not kept_orig:
                raise TrainDataRefusal("item filter left no TRAIN windows")
        # Use the corpus-resolved ORIGINAL per-pack manifest positions (never the
        # TRAIN-filtered/composite-offset dataset position).
        positions = []
        for i in kept_orig:
            pack, position = corpus.pack_index(i), corpus.manifest_position(i)
            if position not in identity_rows[pack]:
                raise TrainDataRefusal(
                    f"identity sidecar {identity_dirs[pack]} lacks TRAIN manifest "
                    f"position {position}")
            positions.append((pack, position))
        hygiene_runs = hygiene_receipt = None
        if controller_hygiene == "strict":
            kept_orig, positions, hygiene_runs, hygiene_receipt = _dspeed_hygiene(
                corpus, kept_orig, positions, identity_rows, identity_dirs,
                int(horizon_steps), timelines)

        loader = _PackWindowLoader(
            corpus, kept_orig, positions, identity_rows, horizon_steps=int(horizon_steps),
            start_shift_prob=start_shift_prob, timelines=timelines, entries_v2=entries_v2,
            history_s=history_s, history_stride_s=history_stride_s, junction_fix=junction_fix,
            hygiene_runs=hygiene_runs)
        receipt = {
            "mode": "a11-index-window",
            "split": "TRAIN",
            "packs": [str(p) for p in paths],
            "identity_sidecars": [str(p) for p in identity_dirs],
            "overlays": [] if overlays is None else [str(p) for p in overlays],
            "n_records": len(kept_orig),
            "n_corpus_windows": len(corpus),
            "horizon_steps": int(horizon_steps),
            "start_shift_prob": float(start_shift_prob),
            "identity_rows": len(identity_rows),
            "kept_by_pack": dict(sorted(kept.kept_by_pack.items())),
            "total_by_pack": dict(sorted(kept.total_by_pack.items())),
            "exclude_keys_requested": (0 if exclude_keys is None else len(exclude_keys)),
            "excluded_records": int(kept.n_excluded),
            "select_windows_requested": (None if select_windows is None
                                         else len(select_windows)),
            "window_filter": filter_receipt,
            "item_filter": item_filter_receipt,
            "timeline_packs": (None if not timelines else sorted(int(k) for k in timelines)),
        }
        if history_s is not None:
            receipt["driver_history"] = {
                "history_s": history_s, "history_stride_s": history_stride_s,
                "short_history_policy": "front-pad and mask; no windows dropped"}
        if junction_fix != "off":
            receipt["junction_fix"] = junction_fix
        if hygiene_receipt is not None:
            receipt["controller_hygiene"] = hygiene_receipt
        dataset = cls([{} for _ in range(len(kept_orig))], loader=loader, receipt=receipt)
        dataset.positions = list(positions)      # [(pack, original manifest position)]
        return dataset


def _fixed_action(meta: dict) -> str:
    """Timeline controller name of a fixed-controller pack item (policy/arm aliases)."""
    name = str(meta.get("policy") or meta.get("arm") or "").lower()
    return _FIXED_ACTION_ALIASES.get(name, name)


def _dspeed_hygiene(corpus: PackCorpus, kept_orig: list[int], positions: list[tuple],
                    identity_rows: list[dict], identity_dirs: list, horizon: int,
                    timelines: Optional[dict]):
    """DATAHYG strict for the dspeed (trajectory-arm) items: per-item :class:`DspeedRun`.

    Returns the kept ``(kept_orig, positions)``, ``{kept index: DspeedRun}`` and a receipt.
    An item is dropped only when NO start (unshifted or start-shifted) avoids every designed
    segment; the loader moves a refused draw to an allowed start.
    """
    vehicle_maps: dict[int, list] = {}
    runs: dict[int, DspeedRun] = {}
    out_kept, out_pos = [], []
    n_dspeed = n_drop = n_unshifted_refused = n_selected = n_designed = 0
    allowed_frac = []
    for i, (pack, position) in zip(kept_orig, positions):
        meta = corpus.meta(i)
        if _fixed_action(meta) != "direct" or (timelines and pack in timelines):
            out_kept.append(i)
            out_pos.append((pack, position))
            continue
        n_dspeed += 1
        if pack not in vehicle_maps:
            vehicle_maps[pack] = json.loads(
                (Path(identity_dirs[pack]) / "vehicle_map.json").read_text(encoding="utf-8"))
        vmap = vehicle_maps[pack]
        tick = int(meta["tick"])
        end = int(corpus.run_end(tick))
        vm = np.asarray(corpus.array("veh_mask")[tick:end]).astype(bool)
        tm = np.asarray(corpus.array("target_mask")[tick:end]).astype(bool)
        cav0 = (np.asarray(corpus.array("veh_step")[tick, :, 7], np.float32) > 0.5) & vm[0]
        ids = identity_rows[pack][position]
        slot_vid = {int(k): str(vmap[int(v)][2]) for k, v in enumerate(ids) if v >= 0}
        run = DspeedRun(str(meta["world"]), str(meta["arm"]), slot_vid, vm, tm, cav0)
        room = len(vm) - 1 - int(horizon)
        allowed = run.allowed_starts(int(horizon))[:max(1, room + 1)]
        n_selected += len(run.selected_slots)
        n_designed += run.n_designed_segments
        allowed_frac.append(float(allowed.mean()))
        n_unshifted_refused += int(not allowed[0])
        if not allowed.any():
            n_drop += 1
            continue
        run.allowed_rows = np.flatnonzero(allowed)
        runs[len(out_kept)] = run
        out_kept.append(i)
        out_pos.append((pack, position))
    if not out_kept:
        raise TrainDataRefusal("controller hygiene left no TRAIN windows")
    receipt = {"mode": "strict", "dspeed_items": n_dspeed,
               "dspeed_items_dropped_no_allowed_start": n_drop,
               "dspeed_unshifted_window_refused": n_unshifted_refused,
               "dspeed_mean_allowed_start_fraction": (round(float(np.mean(allowed_frac)), 4)
                                                      if allowed_frac else None),
               "dspeed_directed_cavs": n_selected,
               "dspeed_designed_segments": n_designed,
               "horizon_steps": int(horizon)}
    return out_kept, out_pos, runs, receipt


class _KeptWindows:
    """Result of the run-key filter: kept dataset positions and per-pack tallies."""

    def __init__(self) -> None:
        self.positions: list[int] = []
        self.total_by_pack: Counter = Counter()
        self.kept_by_pack: Counter = Counter()
        self.n_excluded = 0


def _filter_run_keys(corpus: PackCorpus, pack_names: list[str], lpo_keep: Optional[set[int]],
                     exclude_keys: Optional[set[str]],
                     select_windows: Optional[set[tuple[str, str]]]) -> _KeptWindows:
    """Apply the run-key exclusion / ``(pack, key)`` selection; refuse unmatched requests."""
    kept = _KeptWindows()
    keys_present: set[str] = set()
    matched_select: set[tuple[str, str]] = set()
    for i in range(len(corpus)):
        if lpo_keep is not None and i not in lpo_keep:
            continue
        key = run_key(corpus.meta(i))
        pack = corpus.pack_index(i)
        pname = pack_names[pack] if 0 <= pack < len(pack_names) else str(pack)
        keys_present.add(key)
        kept.total_by_pack[pname] += 1
        pair = (pname, key)
        excluded = exclude_keys is not None and key in exclude_keys
        if excluded:
            kept.n_excluded += 1
        if select_windows is not None and pair in select_windows:
            matched_select.add(pair)
            if excluded:
                raise TrainDataRefusal(
                    f"window-list entry is also in the exclusion list: {pname}/{key}")
        if excluded or (select_windows is not None and pair not in select_windows):
            continue
        kept.positions.append(i)
        kept.kept_by_pack[pname] += 1
    if exclude_keys is not None:
        absent = sorted(k for k in exclude_keys if k not in keys_present)
        if absent:
            raise TrainDataRefusal(
                f"exclusion list has {len(absent)} key(s) absent from the TRAIN corpus, "
                f"e.g. {absent[:3]}")
    if select_windows is not None:
        missing = sorted(select_windows - matched_select)
        if missing:
            raise TrainDataRefusal(
                f"window-list has {len(missing)} entry(ies) absent from the TRAIN corpus, "
                f"e.g. {missing[:3]}")
    if not kept.positions:
        raise TrainDataRefusal("TRAIN window selection is empty after filtering")
    return kept


def road_presence(source, tick: int, horizon_max: int) -> dict:
    """Recorded on-road presence of an entries-v2 window (C4 aggregate loss targets).

    ``road_present [H+1, K]`` = every vehicle on the road at ``tick+t`` (``veh_mask``,
    INCLUDING vehicles that arrive after t0), ``road_continue [H, K]`` = the same vehicle
    is still in the slot at the next tick (``target_mask``) and ``road_valid [H+1]`` marks
    the ticks inside the run (window padding False).  The slot kinematics at every tick
    are already in ``states`` (read from ``veh_kin``); these TRAIN-window labels are only
    consumed by the aggregate loss, never as model inputs.
    """
    n_ticks = int(source.array("veh_kin").shape[0])
    end = min(int(source.run_end(tick)), n_ticks) if hasattr(source, "run_end") else n_ticks
    h = max(0, min(int(horizon_max), end - tick - 1))
    k = int(source.array("veh_mask").shape[1])
    present = np.zeros((horizon_max + 1, k), dtype=bool)
    cont = np.zeros((horizon_max, k), dtype=bool)
    valid = np.zeros(horizon_max + 1, dtype=bool)
    present[:h + 1] = np.asarray(source.array("veh_mask")[tick:tick + h + 1]).astype(bool)
    cont[:h] = np.asarray(source.array("target_mask")[tick:tick + h]).astype(bool)
    valid[:h + 1] = True
    return {"road_present": present, "road_continue": cont, "road_valid": valid}


class _PackWindowLoader:
    """Decode one pack window per call (the lazy loader of the pack-backed dataset)."""

    def __init__(self, corpus: PackCorpus, kept_orig: list[int], positions: list[tuple],
                 identity_rows: list[dict], *, horizon_steps: int, start_shift_prob: float,
                 timelines: Optional[dict], entries_v2: bool, history_s, history_stride_s,
                 junction_fix: str = "off", hygiene_runs: Optional[dict] = None):
        self.corpus = corpus
        self.kept_orig = kept_orig
        self.positions = positions
        self.identity_rows = identity_rows
        self.horizon_steps = horizon_steps
        self.start_shift_prob = start_shift_prob
        self.timelines = timelines
        self.entries_v2 = entries_v2
        self.history_s = history_s
        self.history_stride_s = history_stride_s
        self.junction_fix = junction_fix
        #: DATAHYG strict: ``{index: DspeedRun}`` of the dspeed items (None = hygiene off)
        self.hygiene_runs = hygiene_runs

    def __call__(self, index: int) -> dict:
        corpus = self.corpus
        item = corpus.item(self.kept_orig[index])
        shift = self._draw_start_shift(item)
        run = None if self.hygiene_runs is None else self.hygiene_runs.get(index)
        if run is not None:
            shift = self._hygiene_shift(run, shift)
        if shift:
            item = corpus.shifted_item(item, int(item["tick"]) + shift)
            window = corpus.window(item, identities=False)
        else:
            window = corpus.window(item)
        teacher = window.get("teacher")
        if teacher is None or teacher.shape[0] < 2:
            raise TrainDataRefusal(f"TRAIN item {index} has no multi-step target")
        teacher = teacher[:, 0].numpy().astype(np.float32)
        horizon = teacher.shape[0]
        on_road = teacher[..., 1].astype(np.int64) >= 0
        present = np.concatenate(
            (window["present0"][0].numpy().astype(bool)[None],
             window["rollout_mask"][:, 0].numpy().astype(bool)), axis=0)
        road = None
        if self.junction_fix != "off":
            teacher, present, road = self._fix_junctions(int(item["tick"]), teacher, present)
        states = _states_from_teacher(teacher, window["length0"][0].numpy())
        # A junction internal-lane step still carries the vehicle (the pack ``target_mask``
        # already keeps it), but its lane target is undefined, so exclude only that step
        # from the loss.
        loss_valid = present & on_road
        pack, position = self.positions[index]
        ids = self.identity_rows[pack][position]
        policy = (window.get("meta") or [{}])[0].get("policy")
        present0 = (window["present0"][0].numpy().astype(bool) if road is None
                    else present[0].copy())
        hist, hist_mask = prior_history(corpus, int(item["tick"]), present0)
        result = {
            "hist": hist, "hist_mask": hist_mask,
            "states": states,
            "present": present,
            "loss_valid": loss_valid,
            "slot_index": np.broadcast_to(
                np.arange(states.shape[1], dtype=np.int64), present.shape).copy(),
            "vehicle_ids": ids.copy() if not shift else np.full_like(ids, -1),
            "start_shift_ticks": int(shift),
            "controller_id": int(controller_index(policy or window.get("arm"))),
            "is_cav": (np.broadcast_to(
                window["is_cav"][0].numpy().astype(bool), present.shape).copy() & present),
            "cav_teacher": teacher,
            "meta": window["meta"][0],
        }
        action_name = str(policy or window.get("arm") or "").lower()
        action_name = _FIXED_ACTION_ALIASES.get(action_name, action_name)
        provider = None if self.timelines is None else self.timelines.get(pack)
        if run is not None:
            result["action_timeline"] = self._dspeed_timeline(run, shift, horizon,
                                                              states.shape[1])
        elif action_name in ("native", "damping", "direct", "pi-external-u"):
            result["action_timeline"] = self._fixed_timeline(action_name, item, horizon,
                                                             states.shape[1])
        if provider is not None:
            result["action_timeline"] = provider.timeline(position, _n_control_steps(horizon))
        elif "action_timeline" not in result:
            result["action_timeline"] = self._snapshot_timeline(item, horizon, states.shape[1])
        if self.hygiene_runs is not None and "command_known" not in result["action_timeline"]:
            # strict: a recorded issued command exists only on direct steps of a timeline
            # that recorded one (switching direct-mix); fixed/snapshot timelines carry none.
            timeline = result["action_timeline"]
            known = (np.asarray(timeline["controller"]) == CONTROLLERS.index("direct"))
            if provider is None:
                known = np.zeros_like(known)
            result["action_timeline"] = {**timeline, "command_known": known}
        if self.entries_v2:
            result["entry_feat"] = entry_features(corpus, item)
            result.update(road if road is not None
                          else road_presence(corpus, int(item["tick"]), horizon - 1))
        if self.history_s is not None:
            dh, dm = prior_history(corpus, int(item["tick"]), present0,
                                   history_s=self.history_s,
                                   history_stride_s=self.history_stride_s)
            result["driver_hist"], result["driver_hist_mask"] = dh, dm
        return result

    def _fix_junctions(self, tick: int, teacher: np.ndarray, present: np.ndarray):
        """``junction_fix = "ffill"``: fill junction rows; returns (teacher, present, road).

        ``road`` is the window's recorded road presence (:func:`road_presence`) with a dropped
        t0 vehicle removed for as long as it stays in its slot.
        """
        road = road_presence(self.corpus, tick, teacher.shape[0] - 1)
        lane = np.rint(teacher[..., 1]).astype(np.int64)
        need = road["road_present"][0] & (lane[0] < 0)
        pre_lane, pre_s, pre_age = pre_t0_rows(self.corpus, tick, need)
        teacher, _filled, unfilled = fill_junction_rows(
            teacher, road["road_present"], road["road_continue"], pre_lane, pre_s, pre_age)
        drop = present[0] & unfilled[0]
        if drop.any():
            present = present & ~drop[None]
            gone = run_prefix(road["road_present"], road["road_continue"], drop)
            road["road_present"] = road["road_present"] & ~gone
            road["road_continue"] = road["road_continue"] & ~gone[:-1]
        return teacher, present, road

    def _hygiene_shift(self, run: DspeedRun, shift: int) -> int:
        """Move a refused dspeed start to an allowed one (DATAHYG strict).

        The usual draw is kept when its window avoids every designed segment; otherwise the
        start is drawn uniformly among the allowed starts (global torch RNG), or the earliest
        allowed start when the dataset does not start-shift (validation).
        """
        rows = run.allowed_rows
        if int(shift) in set(rows.tolist()):
            return int(shift)
        if self.start_shift_prob > 0.0:
            return int(rows[int(torch.randint(0, len(rows), ()))])
        return int(rows[0])

    def _dspeed_timeline(self, run: DspeedRun, shift: int, horizon: int, k: int) -> dict:
        """Strict dspeed timeline: directed CAVs ``direct`` (command unknown), rest native."""
        n_control = _n_control_steps(horizon)
        rows = int(shift) + 2 * np.arange(n_control)
        ok = rows < run.directed.shape[0]
        directed = np.zeros((n_control, k), bool)
        directed[ok] = run.directed[rows[ok], :k]
        controller = np.where(directed, CONTROLLERS.index("direct"),
                              CONTROLLERS.index("native")).astype(np.int64)
        zeros = np.zeros((n_control, k), np.float32)
        return {"controller": controller, "acceleration_mps2": zeros,
                "target_speed_mps": zeros.copy(), "control_dt_s": 1.0,
                "command_known": np.zeros((n_control, k), bool)}

    def _draw_start_shift(self, item: dict) -> int:
        """Random later start tick (global torch RNG: ``rand`` then ``randint``)."""
        if not (self.start_shift_prob > 0.0 and float(torch.rand(())) < self.start_shift_prob):
            return 0
        tick0 = int(item["tick"])
        bounds = self.corpus.run_bounds(tick0)
        if bounds is None:
            return 0
        room = bounds[1] - tick0 - 1 - int(self.horizon_steps)
        return int(torch.randint(1, room + 1, ())) if room > 0 else 0

    def _fixed_timeline(self, action_name: str, item: dict, horizon: int, k: int) -> dict:
        """Legacy fixed-controller runs become constant action timelines."""
        n_control = _n_control_steps(horizon)
        fixed = ActionTimeline.constant(action_name, n_control, k)
        if action_name == "direct":
            # Original direct-speed packs: fixed controller identity, recorded issued
            # command channels, never invented zeros.
            tick = int(item["tick"])
            stop = min(tick + horizon, self.corpus.run_end(tick))
            commands = np.asarray(self.corpus.array("cav_plan_ch")[tick:stop:2], np.float32)
            count = min(n_control, commands.shape[0])
            fixed.acceleration_mps2[:count] = commands[:count, :, 1] * 3.0
            fixed.target_speed_mps[:count] = commands[:count, :, 0] * 40.0
        return _timeline_dict(fixed)

    def _snapshot_timeline(self, item: dict, horizon: int, k: int) -> dict:
        """Heterogeneous legacy snapshots: bind the recorded t0 controller per slot."""
        source = np.asarray(self.corpus.array("veh_step")[int(item["tick"])])
        cav0 = source[:, 7] > .5
        ids0 = np.zeros(k, np.int64)
        actions = source[:, 8:12]
        if np.any(cav0 & (actions.sum(-1) < .5)):
            raise TrainDataRefusal("original CAV has no supported controller label")
        if np.any(cav0 & (actions[:, 3] > .5)):
            raise TrainDataRefusal("wall timeline unsupported in mixed training")
        ids0[cav0] = np.asarray(_OWNERSHIP_TO_ACTION)[actions[cav0].argmax(-1)]
        shape = (_n_control_steps(horizon), k)
        return {"controller": np.broadcast_to(ids0, shape).copy(),
                "acceleration_mps2": np.zeros(shape, np.float32),
                "target_speed_mps": np.zeros(shape, np.float32), "control_dt_s": 1.0}
