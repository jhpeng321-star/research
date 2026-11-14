"""Pack -> state materialiser for the C1 evaluation.

``windows_val.json`` and ``pi_eval_states.json`` are metadata only (world / run /
arm / t0).  C1 needs, for every window, an *initial state* and the *recorded
future* (positions / speeds / lanes per vehicle to ``+horizon_s``).  This module
turns one window descriptor into one self-contained state record by reading the
frozen memmap packs exactly the way the WM-N1 reader does:

* ``veh_kin``  ``[N_ticks, K, 7]``  = ``(s_down, lane one-hot x3, v, a, length)``
* ``veh_mask`` ``[N_ticks, K]``     = slot occupied
* ``veh_step`` ``[N_ticks, K, 16]`` = ``col 7`` is ``is_cav``
* ``target_mask`` ``[N_ticks, K]``  = 1 iff the SAME vehicle is in the slot next
  tick (the continuity channel used by ``train.dataset`` / ``train.idea_a``).

The t0 (snapshot) vehicle identities come from the verified identity sidecar
(``identities.npy`` / ``global_positions.npy`` / ``vehicle_map.json``).  Slots
are followed with the SAME cumulative ``target_mask`` product the trainer uses,
so a slot reused mid-window never re-labels the departed t0 vehicle.

Memory discipline (owner rule): the packs are opened ``mmap_mode='r'`` and only
the one window's tick slice is ever touched; states are written to disk one at a
time, so RSS stays far below the 6 GB ceiling and the builders never hold a whole
pack or all runs in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DT_S = 0.5
DEFAULT_FREE_SPEED_MPS = 27.78
# Pre-t0 observed-history window in 0.5 s ticks (matches wms.model.HISTORY_H).
HISTORY_STEPS = 10
# veh_kin column order (features.idea_a.KIN_SOURCE_INDICES)
S_DOWN, LANE_LEFT, LANE_RIGHT, LANE_ACCEL, V_IDX, A_IDX, LENGTH_IDX = range(7)
IS_CAV_STEP_IDX = 7                      # veh_step column (FORMAT.md section 2)
DEFAULT_BASELINE_ARMS = ("native",)
DEFAULT_MIN_VEHICLES = 5


class MaterializeError(ValueError):
    """A pack, sidecar or window descriptor is incomplete or inconsistent."""


# --------------------------------------------------------------------------- #
# pack / sidecar metadata
# --------------------------------------------------------------------------- #
class Pack:
    """Read-only memmap view of one packed corpus."""

    def __init__(self, name: str, root: Path):
        self.name = name
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        index_path = self.root / "INDEX.json"
        if not manifest_path.is_file() or not index_path.is_file():
            raise MaterializeError(f"pack {name!r} lacks manifest.json/INDEX.json at {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.k_max = int(self.manifest.get("k_max", 0)) or None
        self.arrays = self.manifest.get("arrays", {})
        self.runs = {str(r.get("key")): r for r in self.index.get("runs", [])}
        self._mm: dict[str, np.ndarray] = {}
        self._items_by_run: dict[tuple, list[tuple[int, dict]]] = {}
        for pos, item in enumerate(self.manifest.get("items", [])):
            key = (str(item.get("chunk")), str(item.get("world")),
                   str(item.get("arm")), round(float(item.get("t0_s", 0.0)), 6))
            self._items_by_run.setdefault(key, []).append((pos, item))

    def array(self, name: str) -> np.ndarray:
        if name not in self._mm:
            spec = self.arrays.get(name)
            if spec is None:
                raise MaterializeError(f"pack {self.name!r} lacks array {name!r}")
            self._mm[name] = np.load(self.root / spec["file"], mmap_mode="r",
                                     allow_pickle=False)
        return self._mm[name]

    def run_for(self, chunk: str, world: str, arm: str) -> dict:
        key = f"{chunk}/{world}__{arm}"
        run = self.runs.get(key)
        if run is None:
            raise MaterializeError(f"pack {self.name!r} has no run {key!r}")
        return run

    def item_for(self, run: Mapping) -> tuple[int, dict]:
        key = (str(run.get("chunk")), str(run.get("world")), str(run.get("arm")),
               round(float(run.get("t0_s", 0.0)), 6))
        candidates = self._items_by_run.get(key)
        if not candidates:
            raise MaterializeError(
                f"pack {self.name!r} has no manifest item for run "
                f"{run.get('key')!r} at the run t0")
        return min(candidates, key=lambda pos_item: int(pos_item[1].get("tick", 0)))


class Identity:
    """Canonical slot->vehicle-id map for one pack (metadata only)."""

    def __init__(self, name: str, root: Path):
        self.name = name
        self.root = Path(root)
        ids_path = self.root / "identities.npy"
        pos_path = self.root / "global_positions.npy"
        map_path = self.root / "vehicle_map.json"
        if not ids_path.is_file() or not pos_path.is_file() or not map_path.is_file():
            raise MaterializeError(f"identity sidecar {name!r} incomplete at {self.root}")
        self.ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
        self.positions = np.load(pos_path, mmap_mode="r", allow_pickle=False)
        self.vehicle_map = json.loads(map_path.read_text(encoding="utf-8"))
        self.row_of = {int(p): i for i, p in enumerate(self.positions.tolist())}

    def label(self, canonical: int) -> str:
        entry = self.vehicle_map[int(canonical)]
        return str(entry[2])

    def row_for_position(self, position: int) -> np.ndarray:
        if position not in self.row_of:
            raise MaterializeError(
                f"identity sidecar {self.name!r} has no row for manifest position {position}")
        return np.asarray(self.ids[self.row_of[position]], dtype=np.int64)


# --------------------------------------------------------------------------- #
# window -> state record
# --------------------------------------------------------------------------- #
def _seed_of(world: str) -> int:
    return int(hashlib.sha256(str(world).encode("utf-8")).hexdigest()[:8], 16) % 100000


_WORLD_RE = None


def world_meta(world: str) -> dict:
    """Parse ``w<id>m<demand>p<pen>t<t0>`` into a stratification key.

    E1 stratifies the TRAIN evaluation set by world / demand / penetration /
    controller arm, so the encoded demand (``m``), CAV penetration percent
    (``p``) and t0 seconds (``t``) travel with every materialised state.  A
    world string that does not match is still usable: only the raw ``world``
    id is recorded and the numeric strata are ``None``.
    """
    import re

    global _WORLD_RE
    if _WORLD_RE is None:
        _WORLD_RE = re.compile(r"^w(\d+)m(\d+)p(\d+)t(\d+)$")
    match = _WORLD_RE.match(str(world))
    if not match:
        return {"world": str(world), "world_id": None, "demand": None,
                "penetration_pct": None, "t0_s": None}
    wid, demand, pen, t0 = match.groups()
    return {"world": str(world), "world_id": int(wid), "demand": int(demand),
            "penetration_pct": int(pen), "t0_s": float(t0)}


def _lane_of(kin_row: np.ndarray) -> int:
    one_hot = kin_row[LANE_LEFT:LANE_ACCEL + 1]
    return int(one_hot.argmax()) if one_hot.sum() > 0 else -1


def _snapshot(pack: Pack, identity: Identity, run: Mapping, item: Mapping):
    """Return ``(frame, tick, end, ids, is_cav)`` for one arm's t0 snapshot."""
    tick = int(item["tick"])
    end = int(run["tick_end"])
    kin = np.asarray(pack.array("veh_kin")[tick], np.float32)      # [K,7]
    mask = np.asarray(pack.array("veh_mask")[tick]).astype(bool)   # [K]
    step = pack.array("veh_step")[tick]                            # [K,16]
    is_cav = np.asarray(step[:, IS_CAV_STEP_IDX] > 0.5)
    ids = identity.row_for_position(int(item["_position"]))
    if ids.shape[0] < kin.shape[0]:
        ids = np.concatenate([ids, np.full(kin.shape[0] - ids.shape[0], -1, np.int64)])
    return kin, mask, is_cav, ids.astype(np.int64), tick, end


def _frame(time_s: float, kin_rows: np.ndarray, mask_rows: np.ndarray,
           is_cav: np.ndarray, ids: np.ndarray) -> dict:
    vehicles = []
    for k in range(kin_rows.shape[0]):
        if not bool(mask_rows[k]):
            continue
        canonical = int(ids[k])
        if canonical < 0:
            continue
        vehicles.append({
            "id": None, "_canonical": canonical,
            "s": float(kin_rows[k, S_DOWN]),
            "lane": _lane_of(kin_rows[k]),
            "v": float(kin_rows[k, V_IDX]),
            "present": True,
            "is_cav": bool(is_cav[k]),
        })
    return {"time_s": float(time_s), "vehicles": vehicles}


def _materialise_arm(pack: Pack, identity: Identity, run: Mapping, horizon_steps: int,
                     t0_s: float, label: str, shift_steps: int = 0) -> dict:
    """Recorded future frames for one arm, identities bound to the t0 snapshot.

    ``shift_steps`` moves the decision time forward inside the *same* stored pack
    window (the WM-S2 training ``start_shift`` geometry): the new snapshot is
    ``item.tick + shift_steps`` and the slot->vehicle identities from the original
    t0 are carried forward only while the pack ``target_mask`` continuity holds,
    so a slot reused by a new vehicle is dropped rather than mislabelled.  With
    ``shift_steps == HISTORY_STEPS - 1`` the ticks before the new t0 are the real
    recorded history the encoder was trained on.
    """
    position, item = pack.item_for(run)
    item = dict(item)
    item["_position"] = position
    kin0, mask0, is_cav, ids, tick, end = _snapshot(pack, identity, run, item)
    shift = max(0, int(shift_steps))
    new_tick = tick + shift
    if shift > 0 and "target_mask" in getattr(pack, "arrays", {}):
        chain = np.asarray(pack.array("target_mask")[tick:new_tick]).astype(bool)
        same = chain.all(axis=0) if chain.size else np.ones(kin0.shape[0], bool)
    else:
        same = np.ones(kin0.shape[0], bool)
    base_alive = mask0 & same
    horizon = max(0, min(int(horizon_steps), end - new_tick - 1))
    stop = new_tick + horizon + 1
    kin_seq = np.asarray(pack.array("veh_kin")[new_tick:stop], np.float32)   # [T,K,7]
    tm = np.asarray(pack.array("target_mask")[new_tick:new_tick + horizon]).astype(bool) \
        if horizon > 0 else np.zeros((0, kin_seq.shape[1]), bool)
    alive = np.zeros((horizon + 1, kin_seq.shape[1]), bool)
    alive[0] = base_alive
    for j in range(1, horizon + 1):
        alive[j] = alive[j - 1] & tm[j - 1]
    t0_new = t0_s + shift * DT_S
    frames = []
    for j in range(horizon + 1):
        frame = _frame(t0_new + j * DT_S, kin_seq[j], alive[j], is_cav, ids)
        frames.append(frame)
    return {"arm": label, "tick": new_tick, "horizon_steps": horizon,
            "frames": frames, "base_alive": base_alive, "ids": ids,
            "is_cav": is_cav}


def _resolve_arm(pack: Pack, world: str, t0_s: float, arm: str) -> dict | None:
    """Find the run for ``world/arm/t0`` by identity, ignoring the chunk prefix."""
    for run in pack.runs.values():
        if str(run.get("world")) == world and str(run.get("arm")) == arm \
                and abs(float(run.get("t0_s", -1)) - float(t0_s)) < 1e-6:
            return run
    return None


def _history_window(pack: Pack, run: Mapping, tick: int, slots: np.ndarray,
                    history_steps: int = HISTORY_STEPS, *,
                    history_s: float | None = None,
                    history_stride_s: float = 1.0) -> dict:
    """Raw pre-t0 motion window ``[T,S,7]``/``[T,S]``/``[T-1,S]`` for t0 slots.

    Reads only ticks ``<= t0`` and never crosses the run start (the WM-S2
    ``train._prior_history`` rule).  ``slots`` are the pack columns of the t0
    vehicles, in the same order as the initial frame, so the derived per-vehicle
    features line up with the model's sorted rollout slots.

    ``history_s`` opts into a duration-based window. All native ticks are kept;
    ``history_stride_s`` records the encoder's later sampling stride.
    """
    if history_s is not None:
        from wms.history import history_spec

        history_steps, _, _ = history_spec(history_s, history_stride_s, DT_S)
    start = int(run.get("tick_start", 0))
    lo = max(start, int(tick) - int(history_steps) + 1)
    kin = np.asarray(pack.array("veh_kin")[lo:tick + 1][:, slots, :], np.float32)
    mask = np.asarray(pack.array("veh_mask")[lo:tick + 1][:, slots]).astype(bool)
    if tick > lo and "target_mask" in getattr(pack, "arrays", {}):
        target = np.asarray(pack.array("target_mask")[lo:tick][:, slots]).astype(bool)
    else:
        target = np.zeros((0, len(slots)), dtype=bool)
    history = {"kin": kin, "mask": mask, "target_mask": target}
    if history_s is not None:
        history.update(history_s=float(history_s),
                       history_stride_s=float(history_stride_s), dt_s=DT_S,
                       available_span_s=(kin.shape[0] - 1) * DT_S,
                       run_start_limited=(int(tick) - int(history_steps) + 1 < start))
    return history


def build_state(window: Mapping, packs: Mapping[str, Pack],
                identities: Mapping[str, Identity], *, horizon_s: float,
                extra_baselines: Sequence[str] = DEFAULT_BASELINE_ARMS,
                min_vehicles: int = DEFAULT_MIN_VEHICLES,
                shift_steps: int = 0, history_s: float | None = None,
                history_stride_s: float = 1.0) -> dict:
    """One window descriptor -> one self-contained C1 state record.

    ``min_vehicles`` fails closed on a degenerate snapshot: the pack_v3 ``base``
    / ``native_base`` arms were featurised from origin 0 and carry the known
    t0-label misalignment (owner decision (b)), so their snapshot holds almost
    no vehicles.  Such a window is dropped rather than silently scored at the
    wrong time.

    ``shift_steps`` moves t0 forward inside the same stored pack window (see
    :func:`_materialise_arm`); the record's ``t0_s`` is the shifted time and its
    ``history`` is the real recorded H-step pre-t0 window.
    """
    pack_name = str(window.get("pack") or "")
    pack = packs.get(pack_name)
    if pack is None:
        raise MaterializeError(f"window {window.get('key')!r} names unknown pack {pack_name!r}")
    identity = identities.get(pack_name)
    if identity is None:
        raise MaterializeError(f"no identity sidecar registered for pack {pack_name!r}")
    chunk = str(window.get("chunk"))
    world = str(window.get("world"))
    arm = str(window.get("arm"))
    t0_s = float(window.get("t0_s", 0.0))
    horizon_steps = int(round(float(horizon_s) / DT_S))
    shift_steps = max(0, int(shift_steps))
    own_extra = window.get("_extra_baselines")
    baselines = tuple(own_extra) if own_extra is not None else tuple(extra_baselines)

    own_run = _resolve_arm(pack, world, t0_s, arm)
    if own_run is None:
        raise MaterializeError(f"pack {pack_name!r} has no run {world}__{arm} at t0={t0_s}")
    own = _materialise_arm(pack, identity, own_run, horizon_steps, t0_s, arm,
                           shift_steps=shift_steps)

    recorded: dict[str, list] = {arm: own["frames"]}
    arm_labels: dict[str, int] = {arm: own["horizon_steps"]}
    for baseline in baselines:
        if baseline == arm:
            continue
        base_run = _resolve_arm(pack, world, t0_s, baseline)
        if base_run is None:
            continue
        material = _materialise_arm(pack, identity, base_run, horizon_steps, t0_s,
                                    baseline, shift_steps=shift_steps)
        recorded[baseline] = material["frames"]
        arm_labels[baseline] = material["horizon_steps"]

    # initial state is the window arm's own snapshot at the (shifted) t0
    initial = own["frames"][0]
    init_mask = np.asarray(own["base_alive"]).astype(bool)
    init_ids = np.asarray(own["ids"], np.int64)
    init_tick = int(own["tick"])
    counter = None
    if "counter" in getattr(pack, "arrays", {}):
        counter = np.asarray(pack.array("counter")[init_tick], np.float32)
    if len(initial["vehicles"]) < int(min_vehicles):
        raise MaterializeError(
            f"window {window.get('key')!r} has a degenerate snapshot "
            f"({len(initial['vehicles'])} vehicles); the origin-0 base arm is "
            "not a valid t0 state")
    # Pre-t0 observed history for the WM-S2 driver-parameter encoder: the
    # t0-aligned window, in the initial frame's vehicle order, so a scoring
    # loader can rebuild per-vehicle features without touching the pack.
    hist_slots = np.flatnonzero(init_mask & (init_ids >= 0))
    if history_s is not None:
        # Keep observed historical neighbours which exited before t0 as context.
        # Anonymous empty t0 columns are not rollout identities.
        hist_slots = np.arange(len(init_mask))
    hist = _history_window(pack, own_run, init_tick, hist_slots,
                           history_s=history_s, history_stride_s=history_stride_s)
    history = {
        "ids": [identity.label(int(init_ids[k])) if init_mask[k] and init_ids[k] >= 0
                else f"__history_context_slot{k}" for k in hist_slots],
        "history_steps": int(hist["kin"].shape[0]),
        "kin": hist["kin"].tolist(),
        "mask": hist["mask"].tolist(),
        "target_mask": hist["target_mask"].tolist(),
    }
    if history_s is not None:
        for key in ("history_s", "history_stride_s", "dt_s", "available_span_s",
                    "run_start_limited"):
            history[key] = hist[key]
    for row in initial["vehicles"]:
        row["id"] = identity.label(row.pop("_canonical"))
    for frames in recorded.values():
        for frame in frames:
            for row in frame["vehicles"]:
                if "_canonical" in row:
                    row["id"] = identity.label(row.pop("_canonical"))

    return {
        "key": window.get("key"),
        "pack": pack_name,
        "world": world,
        "run": window.get("run"),
        "chunk": chunk,
        "batch": window.get("batch"),
        "arm": arm,
        "policy": window.get("policy"),
        "split": window.get("split", "TRAIN"),
        "t0_s": t0_s + shift_steps * DT_S,
        "t0_shift_steps": shift_steps,
        "t0_shift_s": shift_steps * DT_S,
        "horizon_s": own["horizon_steps"] * DT_S,
        "dt_s": DT_S,
        "seed": _seed_of(world),
        "v0_mps": DEFAULT_FREE_SPEED_MPS,
        "cav_controllers": {},
        "world_meta": world_meta(world),
        "counter": (counter.tolist() if counter is not None else None),
        "arms": sorted(recorded),
        "paired": ("native" in recorded and "pi" in recorded),
        "initial_state": initial,
        "history": history,
        "recorded": recorded,
        "recorded_horizon_steps": arm_labels,
    }


# --------------------------------------------------------------------------- #
# window selection
# --------------------------------------------------------------------------- #
def _load_windows(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("windows", "states", "records", "list"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list) or not raw:
        raise MaterializeError(f"window file {path} has no non-empty list")
    return [dict(w) for w in raw if isinstance(w, dict)]


def _round_robin(strata: Mapping[Any, list], order_key) -> list:
    groups = {k: list(v) for k, v in strata.items()}
    keys = sorted(groups, key=order_key)
    for k in keys:
        groups[k].sort(key=lambda w: (float(w.get("t0_s", 0.0)), str(w.get("key"))))
    out, index = [], {k: 0 for k in keys}
    while any(index[k] < len(groups[k]) for k in keys):
        for k in keys:
            if index[k] < len(groups[k]):
                out.append(groups[k][index[k]])
                index[k] += 1
    return out


def select_fleet_pi(windows: Iterable[dict]) -> list[dict]:
    return [w for w in windows if str(w.get("policy")) == "pi"]


def select_mixed_sample(windows: Iterable[dict], *, size: int = 200, seed: int = 0):
    remaining = [w for w in windows if str(w.get("policy")) != "pi"]
    strata: dict[tuple, list] = {}
    for w in remaining:
        strata.setdefault((str(w.get("policy")), str(w.get("arm"))), []).append(w)
    ordered = _round_robin(strata, lambda k: hashlib.sha256(
        f"{seed}:{k}".encode("utf-8")).hexdigest())
    return ordered[:size]


def select_seen_val(windows: Iterable[dict], *, size: int = 200) -> list[dict]:
    return list(windows)[:size]


def subsample(windows: Sequence[dict], *, limit: int | None, stride: int) -> list[dict]:
    """Optional deterministic ``windows[::stride][:limit]`` subset (E2E proofs)."""
    out = list(windows)
    if stride and int(stride) > 1:
        out = out[::int(stride)]
    if limit:
        out = out[:int(limit)]
    return out


# --------------------------------------------------------------------------- #
# streaming output
# --------------------------------------------------------------------------- #
def _write_stream(out_path: Path, set_name: str, windows,
                  packs: Mapping[str, Pack], identities: Mapping[str, Identity],
                  *, horizon_s: float, on_error: str = "skip",
                  min_vehicles: int = DEFAULT_MIN_VEHICLES,
                  shift_steps: int = 0, n_requested: int | None = None,
                  history_s: float | None = None,
                  history_stride_s: float = 1.0) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if n_requested is None:
        windows = list(windows)
        n_requested = len(windows)
    n_written = 0
    errors = []
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('{"schema":"wms_eval_states/1","set":%s,"horizon_s":%s,'
                 '"n_states":%d,"windows":[' % (
                     json.dumps(set_name), json.dumps(float(horizon_s)), n_requested))
        for i, window in enumerate(windows):
            try:
                record = build_state(window, packs, identities, horizon_s=horizon_s,
                                     min_vehicles=min_vehicles, shift_steps=shift_steps,
                                     history_s=history_s, history_stride_s=history_stride_s)
            except MaterializeError as exc:
                if on_error == "raise":
                    raise
                errors.append({"key": window.get("key"), "error": str(exc)})
                continue
            if n_written:
                fh.write(",")
            fh.write(json.dumps(record, separators=(",", ":")))
            n_written += 1
        fh.write("]}")
    return {"out": str(out_path), "set": set_name, "n_states": n_written,
            "n_requested": n_requested, "errors": errors}


def _parse_pairs(entries: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for entry in entries:
        name, sep, path = entry.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise MaterializeError(f"expected NAME=PATH, got {entry!r}")
        out[name.strip()] = Path(path.strip())
    return out


def _write_clean_sets(args, packs, identities, summary) -> None:
    """Materialise the clean C1 v2 sets (U-same / U-new / SEEN)."""
    from . import c1sets

    pi_windows = _load_windows(Path(args.pi_states))
    val_windows = _load_windows(Path(args.windows_val))
    train_windows = _load_windows(Path(args.train_windows)) if args.train_windows else []
    split = c1sets.unseen_split(pi_windows, train_windows, val_windows)
    shared = dict(horizon_s=args.horizon_s, min_vehicles=args.min_vehicles,
                  shift_steps=int(getattr(args, "shift_steps", 0) or 0),
                  history_s=getattr(args, "history_s", None),
                  history_stride_s=getattr(args, "history_stride_s", 1.0))
    for name, label, out_path, windows in (
            ("unseen_u_same", "unseen_u_same", args.unseen_u_same, split["u_same"]),
            ("unseen_u_new", "unseen_u_new", args.unseen_u_new, split["u_new"])):
        if out_path:
            summary[name] = _write_stream(Path(out_path), label, windows, packs, identities,
                                          **shared)
    if args.seen_v2:
        selection = c1sets.seen_selection(val_windows, split, train_windows,
                                          target=args.seen_size, seed=args.seed)
        summary["seen"] = _write_stream(Path(args.seen_v2), "seen", selection["seen"],
                                        packs, identities, **shared)
        summary["seen_strict"] = len(selection["strict"])
        summary["seen_paired"] = len(selection["paired"])
    summary["dropped_unseen"] = len(split["dropped"])


def _windows_from_set(path: Path, limit: int | None = None):
    """Stream an existing state set into the tiny window descriptors it encodes.

    The large materialised records are decoded one at a time (never all in RAM)
    and reduced to ``(pack, world, arm, t0_s)`` plus the arms that were recorded,
    so the same selection can be rebuilt at a shifted t0 without re-deriving it.
    """
    from .c1v2 import iter_windows

    for record in iter_windows(path, limit=limit):
        recorded = record.get("recorded")
        extra = ([arm for arm in recorded if arm != record.get("arm")]
                 if isinstance(recorded, dict) else [])
        yield {
            "pack": record.get("pack"), "world": record.get("world"),
            "arm": record.get("arm"), "t0_s": record.get("t0_s"),
            "chunk": record.get("chunk"), "key": record.get("key"),
            "run": record.get("run"), "batch": record.get("batch"),
            "policy": record.get("policy"), "split": record.get("split"),
            "_extra_baselines": extra,
        }


def _write_shifted_sets(args, packs, identities, summary) -> None:
    """Rebuild existing C1 v2 sets at a t0 shifted inside the stored pack window."""
    from . import c1sets

    out_dir = Path(args.out_dir)
    for entry in args.from_set:
        name, sep, path = entry.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise MaterializeError(f"expected NAME=PATH for --from-set, got {entry!r}")
        name, path = name.strip(), Path(path.strip())
        n_requested = c1sets.scan_states(path)["n_states"]
        if args.limit:
            n_requested = min(n_requested, int(args.limit))
        summary[name] = _write_stream(
            out_dir / f"{name}.json", name,
            _windows_from_set(path, limit=args.limit), packs, identities,
            horizon_s=args.horizon_s, min_vehicles=args.min_vehicles,
            shift_steps=int(args.shift_steps or 0), n_requested=n_requested,
            history_s=getattr(args, "history_s", None),
            history_stride_s=getattr(args, "history_stride_s", 1.0))


def run(args) -> int:
    history_flags = dict(history_s=getattr(args, "history_s", None),
                         history_stride_s=getattr(args, "history_stride_s", 1.0))
    if history_flags["history_s"] is not None:
        from wms.history import history_spec

        history_spec(history_flags["history_s"], history_flags["history_stride_s"], DT_S)
    if args.shift_steps is None:
        args.shift_steps = int(round(float(args.shift_seconds) / DT_S))
    args.shift_steps = max(0, int(args.shift_steps))
    packs = {name: Pack(name, root) for name, root in _parse_pairs(args.pack).items()}
    identities = {name: Identity(name, root)
                  for name, root in _parse_pairs(args.identities).items()}
    def subsample_all(windows):
        return subsample(windows, limit=args.limit, stride=args.stride)

    summary = {}
    if args.from_set:
        _write_shifted_sets(args, packs, identities, summary)
        print(json.dumps(summary))
        return 0
    if getattr(args, "unseen_u_same", None) or getattr(args, "unseen_u_new", None) \
            or getattr(args, "seen_v2", None):
        _write_clean_sets(args, packs, identities, summary)
    if args.unseen_pi:
        windows = subsample_all(select_fleet_pi(_load_windows(Path(args.pi_states))))
        summary["unseen_pi_fleet161"] = _write_stream(
            Path(args.unseen_pi), "unseen_pi_fleet161", windows, packs, identities,
            horizon_s=args.horizon_s, min_vehicles=args.min_vehicles, **history_flags)
    if args.unseen_mixed:
        windows = subsample_all(select_mixed_sample(_load_windows(Path(args.pi_states)),
                                                    size=args.mixed_size, seed=args.seed))
        summary["unseen_pi_mixed_sample"] = _write_stream(
            Path(args.unseen_mixed), "unseen_pi_mixed_sample", windows, packs, identities,
            horizon_s=args.horizon_s, min_vehicles=args.min_vehicles, **history_flags)
    if args.seen_val:
        windows = subsample_all(select_seen_val(_load_windows(Path(args.windows_val)),
                                                size=args.seen_size))
        summary["seen_val"] = _write_stream(
            Path(args.seen_val), "seen_val", windows, packs, identities,
            horizon_s=args.horizon_s, min_vehicles=args.min_vehicles, **history_flags)
    print(json.dumps(summary))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms_eval materialize")
    p.add_argument("--pack", action="append", required=True, help="NAME=PATH (repeatable)")
    p.add_argument("--identities", action="append", required=True,
                   help="PACK_NAME=PATH (repeatable)")
    p.add_argument("--pi-states", help="pi_eval_states.json (metadata)")
    p.add_argument("--windows-val", help="windows_val.json (metadata)")
    p.add_argument("--train-windows", help="windows_train_noPI.json (C1 v2 drop/exclude)")
    p.add_argument("--unseen-pi", help="output for the 161 fleet-PI states")
    p.add_argument("--unseen-mixed", help="output for the stratified mixed sample")
    p.add_argument("--seen-val", help="output for the seen non-PI states")
    p.add_argument("--unseen-u-same", help="output for the clean U-same-world PI states")
    p.add_argument("--unseen-u-new", help="output for the clean U-new-world PI states")
    p.add_argument("--seen-v2", help="output for the clean held-out SEEN states")
    p.add_argument("--mixed-size", type=int, default=200)
    p.add_argument("--seen-size", type=int, default=200)
    p.add_argument("--horizon-s", type=float, default=300.0)
    p.add_argument("--history-s", type=float, default=None,
                   help="opt-in pre-t0 history duration in seconds (keeps native ticks)")
    p.add_argument("--history-stride-s", type=float, default=1.0,
                   help="encoder sampling stride in seconds for --history-s")
    p.add_argument("--shift-seconds", type=float, default=0.0,
                   help="move t0 forward by this many seconds inside the same "
                        "stored pack window (real pre-t0 history for history models)")
    p.add_argument("--shift-steps", type=int, default=None,
                   help="t0 shift in 0.5 s ticks (overrides --shift-seconds)")
    p.add_argument("--from-set", action="append", default=[],
                   help="NAME=PATH existing state set to rebuild at the shifted t0 "
                        "(repeatable); writes NAME.json under --out-dir")
    p.add_argument("--out-dir", help="output directory for --from-set rebuilds")
    p.add_argument("--min-vehicles", type=int, default=DEFAULT_MIN_VEHICLES)
    p.add_argument("--limit", type=int, help="keep at most N windows per set (E2E proof)")
    p.add_argument("--stride", type=int, default=1, help="stride the selection (E2E proof)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (MaterializeError, OSError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
