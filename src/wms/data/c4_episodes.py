"""C4 Dreamer data loop: collected SUMO episodes -> a WM-S TRAIN source.

``python -m traffic_shepherd.c4.collect`` writes write-once episode directories
(``episode.npz``, schema ``c4_sumo_episode/1``; see ``SUMO_NOTES.md`` section 3).  This
module turns them into a training source the trainer mixes with the original packs
(``[data.c4_episodes]`` in the training config, wired by :mod:`wms.cli.assemble`).

Converter (no SUMO, no wms model)::

    python -m wms.data.c4_episodes --episodes <collect out dir|episode dir> [...] --out <dir>
        [--val-frac 0.2] [--split-seed 0] [--allow-designed]

writes ``<out>/C4_EPISODES_MANIFEST.json`` (schema ``wms_c4_episode_source/1``, split TRAIN)
and one memory-mappable directory per episode ``<out>/episodes/<world>__<arm>/`` holding the
pack-format arrays the loader reads one window slice at a time:

    veh_kin       f32 [T,K,7]   s, lane one-hot (junction / absent = zeros), v, a, 5 m length
    veh_mask      u1  [T,K]     slot occupied (junction transits included, as the packs)
    target_mask   u1  [T,K]     the SAME vehicle is still in the slot at the next tick
    counter       f32 [T,2,30]  10 s entry counts over the last 300 s (bin 0 newest;
                                entrance 0 mainline, 1 ramp) - the pack ``counter`` of
                                ``output/wd01/wm_n1/features/counters.py``
    vid           i32 [T,K]     episode-local vehicle index (-1 empty)
    is_cav        u1  [T,K]
    time_s        f64 [T];  control_time_s f64 [C]
    controller    i8  [C,K]     wms CONTROLLERS ids (native 0, damping 1, pi 2, direct 3)
    issued_acceleration_mps2, issued_target_speed_mps  f32 [C,K]

World ids follow the pack pattern ``w<seed>m<demand>p<penetration %, 2 digits>t<warmup s>``
(``wms.eval.states.world_meta`` parses them); the arm is the collector policy tag.  The
TRAIN/validation split is by episode, grouped by world (every episode of a world lands on
the same side, so the same demand/driver draw never straddles the split).

Windows (:func:`build_c4_datasets`) are cut at load time with the trainer's horizon and
reproduce the pack window loader (:class:`wms.data.dataset._PackWindowLoader`) key for key:
alive-since-t0 presence (``target_mask`` cumulative product: mid-window entrants are never
added and a reused slot never revives), the junction-step loss exclusion, CAV teacher,
short ``hist`` and long ``driver_hist`` from ticks <= t0 (``prior_history``), optional
``entry_feat`` (``entry_features`` on the converted counter) and the per-CAV
``action_timeline`` bound to the t0 slot columns (a slot whose t0 vehicle has left is
``native``).  Designed controllers (damping / PI) are refused unless explicitly allowed
(WM training sees native + direct actions only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from wms.core.constants import controller_index
from wms.core.errors import TrainDataRefusal
from wms.core.guard import refuse_restricted
from wms.data.action_timeline import CONTROLLERS, ActionTimeline
from wms.data.dataset import (TrajectoryDataset, _n_control_steps, _states_from_teacher,
                              road_presence)
from wms.data.sources import validate_train_source

SOURCE_SCHEMA = "wms_c4_episode_source/1"
EPISODE_SCHEMA = "c4_sumo_episode/1"
MANIFEST_NAME = "C4_EPISODES_MANIFEST.json"
DT_S = 0.5
NOMINAL_LENGTH_M = 5.0
COUNTER_BINS, COUNTER_BIN_S, N_ENTRANCES = 30, 10.0, 2
RAMP_LANE = 2
NATIVE = CONTROLLERS.index("native")
DIRECT = CONTROLLERS.index("direct")
DESIGNED = tuple(CONTROLLERS.index(n) for n in ("damping", "pi-external-u"))
ARRAYS = ("veh_kin", "veh_mask", "target_mask", "counter", "vid", "is_cav", "time_s",
          "control_time_s", "controller", "issued_acceleration_mps2", "issued_target_speed_mps")
#: Optional per-control-tick channels copied when the episode has them (stored for later use;
#: the window loader does not read them): the policy's REQUESTED acceleration (before the env's
#: speed-bound clip; ``issued_acceleration_mps2`` is the clipped command, the REALIZED
#: acceleration is ``veh_kin[..., 5]``), and the issued / requested WM lane targets (-1 none).
EXTRA_ARRAYS = ("requested_acceleration_mps2", "lane_command_target", "lane_request_target")


class C4EpisodeRefusal(TrainDataRefusal):
    """A collected episode cannot be used as WM TRAIN data (fail closed)."""


# ------------------------------------------------------------------ naming ---
def world_id(seed: int, demand_vph: float, penetration: float, t0_s: float) -> str:
    """Pack world id ``w<seed>m<demand>p<pct:02d>t<t0>`` (e.g. ``w4107m2700p20t300``)."""
    pct = float(penetration) * 100.0
    if abs(pct - round(pct)) > 1e-6:
        raise C4EpisodeRefusal(f"penetration {penetration} is not a whole percent")
    return f"w{int(seed)}m{int(round(float(demand_vph)))}p{int(round(pct)):02d}t{int(round(t0_s))}"


def _arm_tag(policy: str) -> str:
    text = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(policy or "policy"))
    return text.strip("._")[:60] or "policy"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- conversion ---
def find_episodes(roots: Sequence[str | Path]) -> list[Path]:
    """Complete episode directories under ``roots`` (a root may itself be an episode)."""
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        refuse_restricted(str(root), what="episode root")
        if not root.exists():
            raise FileNotFoundError(root)
        candidates = [root] if (root / "episode.npz").is_file() else sorted(
            p.parent for p in root.rglob("episode.npz"))
        for ep in candidates:
            if ".incomplete-" in ep.name:
                continue
            manifest = ep / "manifest.json"
            status = (json.loads(manifest.read_text(encoding="utf-8")).get("status")
                      if manifest.is_file() else None)
            if status == "complete":
                found.append(ep)
    unique = sorted({p.resolve() for p in found})
    if not unique:
        raise C4EpisodeRefusal(f"no complete episode.npz under {list(map(str, roots))}")
    return unique


def episode_counter(time_s: np.ndarray, vid: np.ndarray, present: np.ndarray,
                    lane_carry: np.ndarray, first_seen_s: Optional[np.ndarray] = None,
                    first_entrance: Optional[np.ndarray] = None) -> np.ndarray:
    """Pack ``counter [T,2,30]``: entries per entrance in 10 s bins over the last 300 s.

    An entry is a vehicle's first occupied tick (every vehicle on the network is converted,
    so this is its insertion); entrance 1 = ramp lane at that tick, else mainline.  Bin
    ``b`` holds entries with age ``t - t_entry`` in ``[10 b, 10 (b+1))`` (0 = newest),
    exactly ``DepartureLog.counter_at`` of the WM-N1 featurizer.  ``first_seen_s`` /
    ``first_entrance`` ``[V]`` (optional; NaN / -1 = unknown) give entry times recorded
    over a longer run than these arrays (a branch episode that starts with a truncated
    pre-history); vehicles they cover are not re-detected from the arrays.
    """
    t_count, _k = vid.shape
    first: dict[int, tuple[float, int]] = {}
    if first_seen_s is not None:
        for v, (ts, ent) in enumerate(zip(np.asarray(first_seen_s, np.float64).tolist(),
                                          np.asarray(first_entrance).tolist())):
            if math.isfinite(ts) and ent >= 0:
                first[v] = (float(ts), int(ent))
    for t in range(t_count):
        ids = vid[t][present[t]]
        lanes = lane_carry[t][present[t]]
        for v, ln in zip(ids.tolist(), lanes.tolist()):
            if v >= 0 and v not in first:
                first[v] = (float(time_s[t]), 1 if ln == RAMP_LANE else 0)
    entries = sorted(first.values())
    times = np.asarray([e[0] for e in entries], np.float64)
    ent = np.asarray([e[1] for e in entries], np.int64)
    out = np.zeros((t_count, N_ENTRANCES, COUNTER_BINS), np.float32)
    for t in range(t_count):
        age = float(time_s[t]) - times
        keep = (age >= 0.0) & (age < COUNTER_BINS * COUNTER_BIN_S)
        if keep.any():
            np.add.at(out[t], (ent[keep], np.floor(age[keep] / COUNTER_BIN_S).astype(np.int64)),
                      1.0)
    return out


def episode_arrays(ep: dict) -> dict[str, np.ndarray]:
    """Pack-format arrays of one loaded ``episode.npz`` (see the module docstring)."""
    present = np.asarray(ep["present"], bool)
    lane = np.asarray(ep["lane"], np.int64)
    vid = np.asarray(ep["vid_index"], np.int32)
    kin = np.zeros(present.shape + (7,), np.float32)
    kin[..., 0] = np.where(present, ep["s_m"], 0.0)
    for j in range(3):
        kin[..., 1 + j] = present & (lane == j)
    kin[..., 4] = np.where(present, ep["v_mps"], 0.0)
    kin[..., 5] = np.where(present, ep["a_mps2"], 0.0)
    kin[..., 6] = np.where(present, NOMINAL_LENGTH_M, 0.0)
    target = np.zeros_like(present)
    target[:-1] = present[:-1] & present[1:] & (vid[:-1] == vid[1:]) & (vid[:-1] >= 0)
    lane_carry = np.asarray(ep.get("lane_carry", lane), np.int64)
    extra = {name: np.asarray(ep[name]) for name in EXTRA_ARRAYS if name in ep}
    if "requested_acceleration_mps2" in extra:
        extra["requested_acceleration_mps2"] = extra["requested_acceleration_mps2"].astype(np.float32)
    for name in ("lane_command_target", "lane_request_target"):
        if name in extra:
            extra[name] = extra[name].astype(np.int8)
    return {**extra,
        "veh_kin": kin, "veh_mask": present.astype(np.uint8),
        "target_mask": target.astype(np.uint8),
        "counter": episode_counter(np.asarray(ep["time_s"]), vid, present, lane_carry,
                                   ep.get("entry_first_seen_s"), ep.get("entry_first_entrance")),
        "vid": vid, "is_cav": np.asarray(ep["is_cav"], bool).astype(np.uint8),
        "time_s": np.asarray(ep["time_s"], np.float64),
        "control_time_s": np.asarray(ep["control_time_s"], np.float64),
        "controller": np.asarray(ep["controller"], np.int8),
        "issued_acceleration_mps2": np.asarray(ep["issued_acceleration_mps2"], np.float32),
        "issued_target_speed_mps": np.asarray(ep["issued_target_speed_mps"], np.float32),
    }


def _load_episode_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    meta = json.loads(str(out.pop("meta_json")))
    if meta.get("schema") != EPISODE_SCHEMA:
        raise C4EpisodeRefusal(f"{path}: not a {EPISODE_SCHEMA} episode")
    vocab = tuple(str(x) for x in out.get("controller_vocab", CONTROLLERS))
    if vocab != tuple(CONTROLLERS[:len(vocab)]):
        raise C4EpisodeRefusal(f"{path}: controller vocabulary {vocab} != {CONTROLLERS}")
    out["meta"] = meta
    return out


def convert_episode(src: Path, dst_root: Path, *, allow_designed: bool = False) -> dict:
    """Convert one complete episode directory; returns its manifest entry."""
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    ep = _load_episode_npz(src / "episode.npz")
    meta = {**ep["meta"], **{k: manifest[k] for k in ("policy", "collector_job", "genrand",
                                                         "arm", "branch", "world_t0_s")
                             if k in manifest}}
    for key in ("seed", "demand_vph", "penetration", "warmup_s"):
        if key not in meta:
            raise C4EpisodeRefusal(f"{src}: episode meta lacks {key!r}")
    ctrl = np.asarray(ep["controller"])
    used = sorted({CONTROLLERS[int(i)] for i in np.unique(ctrl)})
    if not allow_designed and np.isin(ctrl, DESIGNED).any():
        raise C4EpisodeRefusal(
            f"{src}: designed controllers {used} in the action timeline; WM training takes "
            "native + direct actions only (pass --allow-designed to override)")
    # Branch episodes (traffic_shepherd.c4.genrand) start their windows at the branch time
    # (``warmup_s``) but keep the base world (``world_t0_s``) and carry a unique ``arm``.
    world = world_id(meta["seed"], meta["demand_vph"], meta["penetration"],
                     meta.get("world_t0_s", meta["warmup_s"]))
    arm = _arm_tag(meta.get("arm") or meta.get("policy") or meta.get("controller"))
    name = f"{world}__{arm}"
    out = dst_root / "episodes" / name
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out} (duplicate world/arm?)")
    out.mkdir(parents=True)
    arrays = episode_arrays(ep)
    for key, value in arrays.items():
        np.save(out / f"{key}.npy", value, allow_pickle=False)
    present = arrays["veh_mask"].astype(bool)
    cav = present & arrays["is_cav"].astype(bool)
    entry = {
        "episode": name, "world": world, "arm": arm, "dir": f"episodes/{name}",
        "policy": meta.get("policy"), "controller": meta.get("controller"),
        "seed": int(meta["seed"]), "demand_vph": int(round(float(meta["demand_vph"]))),
        "penetration": float(meta["penetration"]), "warmup_s": float(meta["warmup_s"]),
        "horizon_s": meta.get("horizon_s"), "n_states": int(present.shape[0]),
        "k_slots": int(present.shape[1]), "n_control": int(ctrl.shape[0]),
        "controllers_used": used,
        "direct_cav_steps": int((ctrl == DIRECT).sum()),
        "max_present": int(present.sum(1).max()), "cav_rows": int(cav.sum()),
        "source": str(src), "source_sha256": _sha256(src / "episode.npz"),
        "t_start_s": float(arrays["time_s"][0]),
        "extra_arrays": sorted(n for n in EXTRA_ARRAYS if n in arrays),
    }
    if isinstance(meta.get("branch"), dict):        # paired-branch group (genrand)
        entry["branch"] = meta["branch"]
        entry["pair_key"] = meta["branch"].get("pair_key")
    if isinstance(meta.get("genrand"), dict):
        entry["genrand"] = meta["genrand"]
    (out / "episode.json").write_text(json.dumps(entry, indent=1, sort_keys=True),
                                      encoding="utf-8")
    return entry


def split_worlds(worlds: Sequence[str], *, val_frac: float, seed: int) -> set[str]:
    """Validation worlds: the ``ceil(frac * n)`` smallest ``sha256(seed:world)`` (n >= 2)."""
    unique = sorted(set(worlds))
    if val_frac <= 0.0 or len(unique) < 2:
        return set()
    n_val = min(len(unique) - 1, max(1, math.ceil(val_frac * len(unique))))
    ranked = sorted(unique, key=lambda w: hashlib.sha256(f"{seed}:{w}".encode()).hexdigest())
    return set(ranked[:n_val])


def convert(roots: Sequence[str | Path], out: str | Path, *, val_frac: float = 0.2,
            split_seed: int = 0, allow_designed: bool = False) -> dict:
    """Convert every complete episode under ``roots`` into the source directory ``out``."""
    out = validate_train_source(Path(out))
    refuse_restricted(str(out), what="output directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory {out} is not empty (write-once)")
    if not 0.0 <= float(val_frac) < 1.0:
        raise ValueError("--val-frac must be in [0, 1)")
    episodes = find_episodes(roots)
    out.mkdir(parents=True, exist_ok=True)
    entries = [convert_episode(ep, out, allow_designed=allow_designed) for ep in episodes]
    k = {e["k_slots"] for e in entries}
    if len(k) != 1:
        raise C4EpisodeRefusal(f"episodes disagree on the slot count: {sorted(k)}")
    val_worlds = split_worlds([e["world"] for e in entries], val_frac=float(val_frac),
                              seed=int(split_seed))
    for e in entries:
        e["split_role"] = "val" if e["world"] in val_worlds else "train"
    manifest = {
        "schema": SOURCE_SCHEMA, "split": "TRAIN", "k_slots": k.pop(),
        "controller_vocab": list(CONTROLLERS), "dt_s": DT_S,
        "val_frac": float(val_frac), "split_seed": int(split_seed),
        "split_rule": "world-grouped: ceil(frac * n_worlds) smallest sha256(seed:world) "
                      "are validation (none when < 2 worlds)",
        "allow_designed": bool(allow_designed),
        "n_episodes": len(entries),
        "n_train": sum(e["split_role"] == "train" for e in entries),
        "n_val": sum(e["split_role"] == "val" for e in entries),
        "roots": [str(Path(r)) for r in roots], "episodes": entries,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True),
                                     encoding="utf-8")
    return manifest


# ------------------------------------------------------------------ loader ---
def read_source_manifest(root: str | Path) -> dict:
    root = validate_train_source(Path(root))
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise C4EpisodeRefusal(f"{root} has no {MANIFEST_NAME}; run python -m wms.data.c4_episodes")
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_train_source(root, doc)
    if doc.get("schema") != SOURCE_SCHEMA:
        raise C4EpisodeRefusal(f"{path}: schema {doc.get('schema')!r} != {SOURCE_SCHEMA}")
    return doc


class EpisodeArrays:
    """Memory-mapped arrays of one converted episode, with the pack-corpus accessors
    ``array(name)`` / ``run_start(tick)`` that ``prior_history`` / ``entry_features`` use."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._arrays = {name: np.load(self.root / f"{name}.npy", mmap_mode="r",
                                      allow_pickle=False) for name in ARRAYS}

    def array(self, name: str) -> np.ndarray:
        return self._arrays[name]

    @staticmethod
    def run_start(_tick: int) -> int:
        return 0


@lru_cache(maxsize=64)
def _open_episode(path: str) -> EpisodeArrays:
    return EpisodeArrays(Path(path))


def episode_window_ticks(ep: EpisodeArrays, *, horizon_steps: int, stride_s: float,
                         start_s: Optional[float], warmup_s: float) -> list[int]:
    """t0 ticks on the 1 s control grid with a full horizon and its action timeline."""
    stride = int(round(float(stride_s) / DT_S))
    if stride < 2 or stride % 2:
        raise ValueError("c4_episodes window stride must be a whole number of seconds >= 1")
    times = np.asarray(ep.array("time_s"))
    ctrl_t = np.asarray(ep.array("control_time_s"))
    present = ep.array("veh_mask")
    n_control = _n_control_steps(horizon_steps + 1)
    start = float(warmup_s if start_s is None else start_s)
    i0 = int(np.searchsorted(times, start - 1e-6))   # == start / DT_S when times[0] == 0
    i0 += int(round(float(times[min(i0, len(times) - 1)]) / DT_S)) % 2
    ticks = []
    while i0 + horizon_steps < len(times):
        t0 = float(times[i0])
        c0 = int(np.searchsorted(ctrl_t, t0 - 1e-6))
        if (c0 + n_control <= len(ctrl_t) and abs(ctrl_t[c0] - t0) < 1e-6
                and int(np.asarray(present[i0]).sum()) > 0):
            ticks.append(i0)
        i0 += stride
    return ticks


class C4WindowLoader:
    """Decode one C4 episode window into the pack window-record contract."""

    def __init__(self, root: Path, entries: list[dict], index: list[tuple[int, int]], *,
                 horizon_steps: int, history_s, history_stride_s: float, entries_v2: bool,
                 junction_fix: str = "off", command_known: bool = False):
        self.root = Path(root)
        self.entries = entries
        self.index = index
        self.horizon = int(horizon_steps)
        self.history_s = history_s
        self.history_stride_s = float(history_stride_s)
        self.entries_v2 = bool(entries_v2)
        # PAIRLOSS: JUNCTIONFIX convention on these windows ("ffill") and DATAHYG strict
        # ``command_known`` (every issued command of a C4 episode is recorded).
        self.junction_fix = str(junction_fix)
        self.command_known = bool(command_known)

    def episode(self, e: int) -> EpisodeArrays:
        return _open_episode(str(self.root / self.entries[e]["dir"]))

    def alive(self, ep: EpisodeArrays, tick: int) -> np.ndarray:
        h = self.horizon
        present0 = np.asarray(ep.array("veh_mask")[tick], bool)
        tm = np.asarray(ep.array("target_mask")[tick:tick + h], bool)
        return np.cumprod(np.concatenate((present0[None], tm), 0), axis=0).astype(bool)

    def timeline(self, ep: EpisodeArrays, tick: int, alive: np.ndarray) -> dict:
        n_control = _n_control_steps(self.horizon + 1)
        ctrl_t = np.asarray(ep.array("control_time_s"))
        c0 = int(np.searchsorted(ctrl_t, float(ep.array("time_s")[tick]) - 1e-6))
        sl = slice(c0, c0 + n_control)
        # Control step c is decided on state 2c of the window: a slot whose t0 vehicle has
        # left (or was never there) is native with zero commands, as the switching packs.
        held = alive[0:2 * n_control:2]
        ctrl = np.where(held, np.asarray(ep.array("controller")[sl], np.int64), NATIVE)
        acc = np.where(held, np.asarray(ep.array("issued_acceleration_mps2")[sl]), 0.0)
        tgt = np.where(held, np.asarray(ep.array("issued_target_speed_mps")[sl]), 0.0)
        timeline = ActionTimeline(ctrl, acc.astype(np.float32), tgt.astype(np.float32))
        out = {"controller": timeline.controller, "acceleration_mps2": timeline.acceleration_mps2,
               "target_speed_mps": timeline.target_speed_mps,
               "control_dt_s": timeline.control_dt_s}
        if self.command_known:
            out["command_known"] = np.ones(np.shape(timeline.controller), bool)
        return out

    def _fix_junctions(self, ep: EpisodeArrays, tick: int, teacher: np.ndarray,
                       alive: np.ndarray):
        """JUNCTIONFIX "ffill" on a C4 window (same convention as the pack loader)."""
        from wms.data.junction_fix import fill_junction_rows, pre_t0_rows, run_prefix

        road = road_presence(ep, tick, teacher.shape[0] - 1)
        lane = np.rint(teacher[..., 1]).astype(np.int64)
        need = road["road_present"][0] & (lane[0] < 0)
        pre_lane, pre_s, pre_age = pre_t0_rows(ep, tick, need)
        teacher, _filled, unfilled = fill_junction_rows(
            teacher, road["road_present"], road["road_continue"], pre_lane, pre_s, pre_age)
        drop = alive[0] & unfilled[0]
        if drop.any():
            alive = alive & ~drop[None]
            gone = run_prefix(road["road_present"], road["road_continue"], drop)
            road["road_present"] = road["road_present"] & ~gone
            road["road_continue"] = road["road_continue"] & ~gone[:-1]
        return teacher, alive, road

    def events(self, e: int, tick: int) -> tuple[int, int]:
        """``(first_switch_step, first_direct_step)`` of the window's timeline."""
        from wms.data.mixdata import event_steps

        ep = self.episode(e)
        alive = self.alive(ep, tick)
        ctrl = self.timeline(ep, tick, alive)["controller"]
        held = alive[0:2 * ctrl.shape[0]:2]
        return event_steps(ctrl, held[1:] & held[:-1])

    def jam_view(self, index: int) -> dict:
        """Cheap ``{states, present, is_cav}`` for the jam scan (``wms.data.jam``): the recorded
        speeds ``veh_kin[..., 4]`` at state index 4 and the alive mask, without the history,
        timeline or junction rows of :meth:`__call__` (junction-fix rows may differ slightly)."""
        e, tick = self.index[index]
        ep = self.episode(e)
        h = self.horizon
        kin = np.asarray(ep.array("veh_kin")[tick:tick + h + 1], np.float32)
        alive = self.alive(ep, tick)
        states = np.zeros(kin.shape[:2] + (5,), np.float32)
        states[..., 4] = kin[..., 4]
        is_cav = np.broadcast_to(np.asarray(ep.array("is_cav")[tick], bool), alive.shape) & alive
        return {"states": states, "present": alive, "is_cav": is_cav}

    def __call__(self, index: int) -> dict:
        from wms.data.pack_history import entry_features, prior_history

        e, tick = self.index[index]
        ep = self.episode(e)
        h = self.horizon
        kin = np.asarray(ep.array("veh_kin")[tick:tick + h + 1], np.float32)
        if kin.shape[0] != h + 1:
            raise C4EpisodeRefusal(f"window {index} has no full {h}-step horizon")
        lane_hot = kin[..., 1:4]
        lane = np.where(lane_hot.sum(-1) > 0, lane_hot.argmax(-1), -1).astype(np.float32)
        teacher = np.stack((kin[..., 0], lane, kin[..., 4], kin[..., 5]), -1).astype(np.float32)
        alive = self.alive(ep, tick)
        road = None
        on_road = lane >= 0
        if self.junction_fix != "off":
            teacher, alive, road = self._fix_junctions(ep, int(tick), teacher, alive)
        states = _states_from_teacher(teacher, kin[0, :, 6])
        present0 = alive[0]
        vid0 = np.asarray(ep.array("vid"), np.int64)[tick]
        entry = self.entries[e]
        hist, hist_mask = prior_history(ep, tick, present0)
        record = {
            "hist": hist, "hist_mask": hist_mask,
            "states": states, "present": alive, "loss_valid": alive & on_road,
            "slot_index": np.broadcast_to(np.arange(states.shape[1], dtype=np.int64),
                                          alive.shape).copy(),
            # episode-local ids made unique across the source (never pack sidecar ids)
            "vehicle_ids": np.where(vid0 >= 0, (e + 1) * 1_000_000 + vid0, -1),
            "start_shift_ticks": 0,
            "controller_id": int(controller_index(entry.get("controller") or "direct")),
            "is_cav": (np.broadcast_to(np.asarray(ep.array("is_cav")[tick], bool),
                                       alive.shape).copy() & alive),
            "cav_teacher": teacher,
            "meta": {"source": "c4_episodes", "world": entry["world"], "arm": entry["arm"],
                     "policy": entry.get("policy"), "episode": entry["episode"],
                     "tick": int(tick), "t0_s": float(ep.array("time_s")[tick]),
                     "pair_key": entry.get("pair_key")},
            "action_timeline": self.timeline(ep, tick, alive),
        }
        if self.entries_v2:
            record["entry_feat"] = entry_features(ep, {"tick": int(tick)})
            record.update(road if road is not None else road_presence(ep, int(tick), h))
        if self.history_s is not None:
            record["driver_hist"], record["driver_hist_mask"] = prior_history(
                ep, tick, present0, history_s=self.history_s,
                history_stride_s=self.history_stride_s)
        return record


def _one_split(root: Path, doc: dict, role: str, *, horizon_steps: int, history_s,
               history_stride_s: float, entries_v2: bool, stride_s: float,
               start_s: Optional[float], refuse_worlds=None, role_of=None,
               junction_fix: str = "off", command_known: bool = False,
               allow_empty: bool = False):
    roles = role_of or {}
    entries = [e for e in doc["episodes"]
               if roles.get(e["world"], e.get("split_role")) == role]
    if not entries:
        return None, None
    if refuse_worlds is not None:
        refuse_worlds([e["world"] for e in entries], where=f"c4 episodes ({role})")
    index: list[tuple[int, int]] = []
    per_episode = {}
    for i, entry in enumerate(entries):
        ep = _open_episode(str(root / entry["dir"]))
        ticks = episode_window_ticks(ep, horizon_steps=horizon_steps, stride_s=stride_s,
                                     start_s=start_s, warmup_s=float(entry["warmup_s"]))
        index.extend((i, t) for t in ticks)
        per_episode[entry["episode"]] = len(ticks)
    if not index and allow_empty:
        return None, None
    if not index:
        raise C4EpisodeRefusal(
            f"c4 episodes ({role}): no window with a full {horizon_steps * DT_S:.0f} s horizon "
            "after the warmup; collect longer episodes or shorten the curriculum")
    loader = C4WindowLoader(root, entries, index, horizon_steps=horizon_steps,
                            history_s=history_s, history_stride_s=history_stride_s,
                            entries_v2=entries_v2, junction_fix=junction_fix,
                            command_known=command_known)
    events = np.asarray([loader.events(e, t) for e, t in index], np.int64)
    receipt = {
        "mode": "c4-episodes", "split": "TRAIN", "role": role, "source": str(root),
        "schema": SOURCE_SCHEMA, "n_records": len(index), "n_episodes": len(entries),
        "windows_by_episode": per_episode, "horizon_steps": int(horizon_steps),
        "stride_s": float(stride_s), "start_s": start_s, "start_shift_prob": 0.0,
        "worlds": sorted({e["world"] for e in entries}),
        "n_direct_windows": int((events[:, 1] < _n_control_steps(horizon_steps + 1)).sum()),
        "kept_by_pack": {"c4_episodes": len(index)},
        "total_by_pack": {"c4_episodes": len(index)},
    }
    if history_s is not None:
        receipt["driver_history"] = {"history_s": history_s,
                                     "history_stride_s": history_stride_s}
    dataset = TrajectoryDataset([{} for _ in index], loader=loader, receipt=receipt)
    dataset.positions = list(index)
    return dataset, events


def build_c4_datasets(root: str | Path, *, horizon_steps: int, history_s=None,
                      history_stride_s: float = 1.0, entries_v2: bool = False,
                      stride_s: float = 10.0, start_s: Optional[float] = None,
                      k_slots: Optional[int] = None, heldout=None, role_of=None,
                      junction_fix: str = "off", command_known: bool = False,
                      allow_empty: bool = False) -> dict:
    """``{"train", "train_events", "val", "receipt"}`` of a converted episode source.

    ``heldout`` (a :class:`wms.data.mixdata.HeldOutGuard`) refuses held-out worlds;
    ``k_slots`` must match the packs it is mixed with (collation stacks slots).
    PAIRLOSS options (defaults = unchanged): ``role_of`` ``{world: "train"|"val"}`` overrides
    the manifest split (one side per world across sources), ``junction_fix`` / ``command_known``
    as :class:`C4WindowLoader`, ``allow_empty`` returns ``train = None`` instead of refusing
    when no TRAIN window has the full horizon.
    """
    root = Path(root)
    doc = read_source_manifest(root)
    if k_slots is not None and int(doc["k_slots"]) != int(k_slots):
        raise C4EpisodeRefusal(f"c4 episodes have K={doc['k_slots']} slots, packs K={k_slots}")
    if not doc.get("allow_designed", False):
        bad = [e["episode"] for e in doc["episodes"]
               if set(e.get("controllers_used", ())) - {"native", "direct"}]
        if bad:
            raise C4EpisodeRefusal(f"designed controllers in {bad[:3]}")
    common = dict(horizon_steps=int(horizon_steps), history_s=history_s,
                  history_stride_s=float(history_stride_s), entries_v2=bool(entries_v2),
                  stride_s=float(stride_s), start_s=start_s,
                  refuse_worlds=None if heldout is None else heldout.refuse_worlds)
    if role_of is not None or junction_fix != "off" or command_known or allow_empty:
        common.update(role_of=role_of, junction_fix=junction_fix, command_known=command_known,
                      allow_empty=allow_empty)
    train, events = _one_split(root, doc, "train", **common)
    if train is None and allow_empty:
        val, _ = _one_split(root, doc, "val", **common)
        return {"train": None, "train_events": None, "val": val,
                "receipt": {"source": str(root), "n_episodes": doc["n_episodes"],
                            "n_train_windows": 0,
                            "n_val_windows": 0 if val is None else len(val),
                            "train": None, "val": None if val is None else val.receipt}}
    if train is None:
        raise C4EpisodeRefusal(f"{root}: no TRAIN episodes")
    val, _ = _one_split(root, doc, "val", **common)
    receipt = {"source": str(root), "manifest_sha256": _sha256(root / MANIFEST_NAME),
               "n_episodes": doc["n_episodes"], "n_train_windows": len(train),
               "n_val_windows": 0 if val is None else len(val),
               "train": train.receipt, "val": None if val is None else val.receipt}
    return {"train": train, "train_events": events, "val": val, "receipt": receipt}


def add_c4_source(train, val, c4: dict, *, mix_ratio: float, jam_speed_mps: float = 5.0,
                  jam_min_fraction: float = 0.0):
    """Mix the C4 source into the assembled ``(train, val)`` datasets.

    The C4 source gets the sampling share ``mix_ratio``; the existing sources keep their
    relative shares, scaled by ``1 - mix_ratio`` (a plain or jam-oversampled original
    dataset becomes one source).  Validation windows are appended as their own source.
    """
    from wms.data import jam as jamdata
    from wms.data.mixdata import MixDataRefusal, MixedTrajectoryDataset, MixSource

    ratio = float(mix_ratio)
    if not 0.0 < ratio < 1.0:
        raise MixDataRefusal("data.c4_episodes.mix_ratio must be in (0, 1)")
    receipt_extra: dict = {}
    if isinstance(train, MixedTrajectoryDataset):
        base = [MixSource(s.name, s.dataset, share=share * (1.0 - ratio), events=s.events,
                          jam=s.jam) for s, share in zip(train.sources, train.shares)]
        event_oversample, jam_oversample = train.event_oversample, train.jam_oversample
        receipt_extra = {k: v for k, v in train.receipt.items()
                         if k not in ("mode", "n_records", "sources", "event_oversample",
                                      "jam_oversample", "stage_sampling")}
    elif isinstance(train, jamdata.JamOversampledDataset):
        base = [MixSource("original", train.dataset, share=1.0 - ratio,
                          jam=np.asarray(train.flags, bool))]
        event_oversample, jam_oversample = 1.0, train.oversample
        receipt_extra = {"jam": train.receipt.get("jam")}
    else:
        base = [MixSource("original", train, share=1.0 - ratio)]
        event_oversample, jam_oversample = 1.0, 1.0
    c4_jam = None
    if jam_oversample > 1.0:
        c4_jam = jamdata.compute_jam_flags(c4["train"], speed_mps=float(jam_speed_mps),
                                           min_fraction=float(jam_min_fraction))["flags"]
    mixed = MixedTrajectoryDataset(
        base + [MixSource("c4_episodes", c4["train"], share=ratio, events=c4["train_events"],
                          jam=c4_jam)],
        event_oversample=event_oversample, jam_oversample=jam_oversample)
    mixed.receipt.update(receipt_extra)
    mixed.receipt["c4_episodes"] = c4["receipt"]
    if c4["val"] is not None:
        if isinstance(val, MixedTrajectoryDataset):
            sources = [MixSource(s.name, s.dataset, share=1.0) for s in val.sources]
        elif val is not None:
            sources = [MixSource("original_val", val, share=1.0)]
        else:
            sources = []
        val = MixedTrajectoryDataset(sources + [MixSource("c4_episodes_val", c4["val"],
                                                          share=1.0)])
    return mixed, val


# --------------------------------------------------------------------- CLI ---
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wms.data.c4_episodes",
        description="Convert collected C4 SUMO episodes into a WM-S TRAIN source.")
    parser.add_argument("--episodes", nargs="+", required=True,
                        help="collector output dirs (or single episode dirs)")
    parser.add_argument("--out", required=True, help="new (empty) source directory")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="fraction of worlds held out for validation (default 0.2)")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--allow-designed", action="store_true",
                        help="accept damping / PI action timelines (off: native + direct only)")
    args = parser.parse_args(argv)
    doc = convert(args.episodes, args.out, val_frac=args.val_frac, split_seed=args.split_seed,
                  allow_designed=args.allow_designed)
    print(json.dumps({k: doc[k] for k in ("schema", "n_episodes", "n_train", "n_val",
                                          "k_slots")}, sort_keys=True))
    for e in doc["episodes"]:
        print(f"{e['split_role']:5s} {e['episode']}  states={e['n_states']} "
              f"direct_cav_steps={e['direct_cav_steps']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
