"""Canonical trajectory table (WM frame) and loaders.

A :class:`Traj` is a long table of vehicle rows at 0.5 s physics steps:
``time_s [T]``; per row ``step, vid, s, lane, v`` (rows sorted by step then vid); per vehicle
``ids [V]`` and ``is_cav [V]``; optional per-step ``backlog [T]`` (vehicles due but not yet
inserted, the SUMO source-waiting term of J) - ``None`` when the source is not modelled.

Loaders: WM-D prefix tapes (``wzqual_native_tape/1``, SUMO frame -> WM frame), WM-B episodes
(``c4_sumo_episode/1``, already in the WM frame), and WM predictor frames
(``[{"time_s", "vehicles": [{"id", "s", "lane", "v", "is_cav"}]}]``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from . import frame as F


@dataclass
class Traj:
    time_s: np.ndarray
    step: np.ndarray
    vid: np.ndarray
    s: np.ndarray
    lane: np.ndarray
    v: np.ndarray
    ids: list
    is_cav: np.ndarray
    backlog: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, np.float64)
        order = np.lexsort((np.asarray(self.vid), np.asarray(self.step)))
        self.step = np.asarray(self.step, np.int64)[order]
        self.vid = np.asarray(self.vid, np.int64)[order]
        self.s = np.asarray(self.s, np.float64)[order]
        self.lane = np.asarray(self.lane, np.int64)[order]
        self.v = np.asarray(self.v, np.float64)[order]
        self.ids = [str(i) for i in self.ids]
        self.is_cav = np.asarray(self.is_cav, bool)
        if self.backlog is not None:
            self.backlog = np.asarray(self.backlog, np.float64)
        self._starts = np.searchsorted(self.step, np.arange(len(self.time_s) + 1))
        self._index = {name: i for i, name in enumerate(self.ids)}

    # ------------------------------------------------------------------ access
    @property
    def n_steps(self) -> int:
        return len(self.time_s)

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time_s))) if self.n_steps > 1 else F.DT

    def step_at(self, t: float, tol: float = 1e-6) -> int:
        k = int(np.searchsorted(self.time_s, float(t) - tol))
        if k >= self.n_steps or abs(self.time_s[k] - float(t)) > tol:
            raise KeyError(f"no step at t={t}")
        return k

    def rows(self, k: int) -> slice:
        return slice(int(self._starts[k]), int(self._starts[k + 1]))

    def frame(self, k: int) -> dict:
        r = self.rows(k)
        vid = self.vid[r]
        return dict(time_s=float(self.time_s[k]), vid=vid, ids=[self.ids[i] for i in vid],
                    s=self.s[r], lane=self.lane[r], v=self.v[r], is_cav=self.is_cav[vid])

    def index_of(self, name: str) -> int:
        return self._index[str(name)]

    def window(self, t_a: float, t_b: float) -> "Traj":
        """Steps with t_a <= time <= t_b (same vehicle index)."""
        keep = (self.time_s >= t_a - 1e-9) & (self.time_s <= t_b + 1e-9)
        ks = np.flatnonzero(keep)
        if ks.size == 0:
            raise ValueError(f"empty window [{t_a}, {t_b}]")
        k0, k1 = int(ks[0]), int(ks[-1])
        r = slice(int(self._starts[k0]), int(self._starts[k1 + 1]))
        return Traj(self.time_s[k0:k1 + 1], self.step[r] - k0, self.vid[r], self.s[r],
                    self.lane[r], self.v[r], self.ids, self.is_cav,
                    None if self.backlog is None else self.backlog[k0:k1 + 1], dict(self.meta))

    def first_seen(self) -> np.ndarray:
        """Per vehicle: time of its first row (NaN if never present)."""
        return self.first_rows()[0]

    def first_rows(self) -> tuple:
        """Per vehicle: (time, WM lane) of its first row (NaN / -1 if never present)."""
        t = np.full(len(self.ids), np.nan)
        ln = np.full(len(self.ids), -1, np.int64)
        rev = np.arange(len(self.step))[::-1]
        t[self.vid[rev]] = self.time_s[self.step[rev]]
        ln[self.vid[rev]] = self.lane[rev]
        return t, ln


# ---------------------------------------------------------------------- loaders
def load_wzqual_tape(path: str | Path) -> Traj:
    """WM-D prefix tape (SUMO frame) -> Traj in the WM frame."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["meta_json"]))
    if meta.get("schema") != "wzqual_native_tape/1":
        raise ValueError(f"{path}: not a wzqual_native_tape/1")
    edges = [str(e) for e in d["edges"]]
    edge_code = np.asarray(d["edge"], np.int64)
    sumo_lane = np.asarray(d["lane"], np.int64)
    lane = np.empty_like(sumo_lane)
    for code, name in enumerate(edges):
        m = edge_code == code
        for idx in np.unique(sumo_lane[m]):
            lane[m & (sumo_lane == idx)] = F.wm_lane(name, int(idx))
    s = np.asarray(d["x"], np.float64) - F.ANCHOR_X
    meta = dict(meta, source=str(path), frame="wm")
    return Traj(d["time_s"], d["step"], d["vid"], s, lane, d["v"], list(d["ids"]), d["is_cav"],
                d["backlog"].astype(np.float64), meta)


def load_c4_episode(path: str | Path) -> Traj:
    """WM-B ``c4_sumo_episode/1`` episode.npz (WM frame, slot arrays) -> Traj."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d.pop("meta_json"))) if "meta_json" in d else {}
    present = np.asarray(d["present"], bool)
    vid_index = np.asarray(d["vid_index"], np.int64)
    t_idx, k_idx = np.nonzero(present & (vid_index >= 0))
    v_ids = vid_index[t_idx, k_idx]
    uniq = np.unique(v_ids)
    remap = {int(u): i for i, u in enumerate(uniq)}
    vid = np.asarray([remap[int(x)] for x in v_ids], np.int64)
    is_cav = np.zeros(len(uniq), bool)
    cav_rows = np.asarray(d["is_cav"], bool)[t_idx, k_idx]
    is_cav[vid[cav_rows]] = True
    names = d.get("vehicle_ids")
    ids = ([str(names[int(u)]) for u in uniq] if names is not None and len(names) > int(uniq.max(initial=0))
           else [f"v{int(u)}" for u in uniq])
    backlog = None
    for key in ("source_backlog", "backlog"):
        if key in d:
            backlog = np.asarray(d[key], np.float64)
    return Traj(d["time_s"], t_idx, vid, np.asarray(d["s_m"])[t_idx, k_idx],
                np.asarray(d["lane"])[t_idx, k_idx], np.asarray(d["v_mps"])[t_idx, k_idx],
                ids, is_cav, backlog, dict(meta, source=str(path), frame="wm"))


def from_frames(frames: Sequence[Mapping], *, backlog: Optional[Iterable[float]] = None,
                meta: Optional[dict] = None) -> Traj:
    """WM predictor frames -> Traj (rows with ``present`` False are dropped)."""
    index: dict[str, int] = {}
    cav: dict[int, bool] = {}
    cols = {k: [] for k in ("step", "vid", "s", "lane", "v")}
    times = []
    for k, fr in enumerate(frames):
        times.append(float(fr["time_s"]))
        for row in fr["vehicles"]:
            if not row.get("present", True):
                continue
            i = index.setdefault(str(row["id"]), len(index))
            cav[i] = cav.get(i, False) or bool(row.get("is_cav", False))
            cols["step"].append(k)
            cols["vid"].append(i)
            cols["s"].append(float(row["s"]))
            cols["lane"].append(int(row["lane"]))
            cols["v"].append(float(row["v"]))
    ids = sorted(index, key=index.get)
    return Traj(times, cols["step"], cols["vid"], cols["s"], cols["lane"], cols["v"], ids,
                np.asarray([cav[i] for i in range(len(ids))], bool),
                None if backlog is None else np.asarray(list(backlog), np.float64), dict(meta or {}))


def concat_steps(parts: Sequence[Traj]) -> Traj:
    """Concatenate trajectories in time over a common id space (later parts follow earlier)."""
    ids: dict[str, int] = {}
    cav: dict[int, bool] = {}
    cols = {k: [] for k in ("step", "vid", "s", "lane", "v")}
    times, back = [], []
    off = 0
    for p in parts:
        remap = np.asarray([ids.setdefault(n, len(ids)) for n in p.ids], np.int64)
        for j, n in enumerate(p.ids):
            cav[remap[j]] = cav.get(remap[j], False) or bool(p.is_cav[j])
        cols["step"].append(p.step + off)
        cols["vid"].append(remap[p.vid] if len(p.vid) else p.vid)
        for k in ("s", "lane", "v"):
            cols[k].append(getattr(p, k))
        times.append(p.time_s)
        back.append(p.backlog if p.backlog is not None else np.full(p.n_steps, np.nan))
        off += p.n_steps
    names = sorted(ids, key=ids.get)
    return Traj(np.concatenate(times), *(np.concatenate(cols[k]) for k in ("step", "vid", "s", "lane", "v")),
                names, np.asarray([cav[i] for i in range(len(names))], bool), np.concatenate(back),
                dict(parts[0].meta) if parts else {})


# ------------------------------------------------------------------ start states
@dataclass
class StartState:
    """Everything a backend may observe at t0: the t0 frame, pre-t0 history, entry log, meta.

    ``history`` is the truth trajectory restricted to [t0 - history_s, t0] (observable past);
    ``entry_times_s`` are first-appearance times of vehicles seen up to t0 (entrance counters);
    ``backlog_t0`` the source backlog at t0.  No post-t0 information is included.
    """

    world: str
    seed: int
    cell: str
    t0: float
    frame: dict
    history: Traj
    entry_times_s: np.ndarray
    backlog_t0: float
    meta: dict
    entry_lanes: np.ndarray = None

    def counter(self, *, bins: int = 30, bin_s: float = 10.0) -> np.ndarray:
        """Entrance counter [2, bins] at t0 (bin 0 newest; entrance 1 = closing-lane insertions),
        the WM pack ``counter`` definition (wms.data.c4_episodes.episode_counter)."""
        out = np.zeros((2, bins), np.float32)
        lanes = self.entry_lanes if self.entry_lanes is not None else np.zeros(len(self.entry_times_s), int)
        for t, ln in zip(self.entry_times_s, lanes):
            age = self.t0 - float(t)
            if 0.0 <= age < bins * bin_s:
                out[1 if int(ln) == F.CLOSING_LANE else 0, int(age // bin_s)] += 1.0
        return out

    @property
    def key(self) -> str:
        return f"{self.world}@{int(round(self.t0))}"


def start_state(traj: Traj, t0: float, *, history_s: float = 120.0) -> StartState:
    k = traj.step_at(t0)
    hist = traj.window(max(float(traj.time_s[0]), t0 - history_s), t0)
    upto = traj.window(float(traj.time_s[0]), t0)
    entries, entry_lane = upto.first_rows()
    seed = int(traj.meta.get("seed", -1))
    demand = traj.meta.get("demand")
    if isinstance(demand, Mapping):
        demand = demand.get("main")
    cell = traj.meta.get("cell") or (F.cell_key(demand, traj.meta.get("penetration"))
                                     if demand is not None else "unknown")
    backlog = float(traj.backlog[k]) if traj.backlog is not None else float("nan")
    order = np.argsort(entries[np.isfinite(entries)], kind="mergesort")
    return StartState(world=f"{seed}:{cell}", seed=seed, cell=str(cell), t0=float(t0),
                      frame=traj.frame(k), history=hist,
                      entry_times_s=entries[np.isfinite(entries)][order],
                      backlog_t0=backlog, meta=dict(traj.meta),
                      entry_lanes=entry_lane[np.isfinite(entries)][order])
