"""Collected C4 SUMO episodes (``episode.npz``, ``traffic_shepherd.c4.collect``) -> WM-S state records.

A record has the ``wms_eval_states/1`` layout the unseen-wall sets use
(:mod:`traffic_shepherd.c4.wallregen_states`) and that ``wms.eval.predictor.WmsPredictor`` reads:

* ``initial_state``: the t0 frame (every present vehicle; junction transits with lane -1, which
  the predictor drops from the roster);
* ``history``: id-keyed native pre-t0 history (``history_s`` 60 s = 121 rows at 0.5 s, stride
  1 s, t0 slots pinned with ``Timeline._align_history``) for long-history checkpoints;
* ``recorded[arm]``: the SUMO future of the t0 roster (a vehicle drops out at its first missing
  tick; entrants after t0 are not included) - the CAV teacher and the SUMO reference;
* ``counter``: the RSU entrance counter [2,30] at t0 (only used with entries ON).

The episode's own slot/time arrays are wrapped in a ``wallregen_states.Timeline`` (no base/branch
concatenation), so frame, future and history construction is the wallregen code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from traffic_shepherd.c4.wallregen_states import Timeline

DT_S = 0.5
HISTORY_ROWS = 121
HISTORY_S = 60.0
FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)
JOB_RE = re.compile(r"^s(\d+)_m(\d+)_p(\d{4})$")


class EpisodeTimeline(Timeline):
    """A ``Timeline`` over one collected episode (vid = ``vid_index`` into ``vehicle_ids``)."""

    def __init__(self, ep: dict):  # noqa: D401 - intentionally bypasses Timeline.__init__
        self.cols = {"time_s": np.asarray(ep["time_s"], np.float64),
                     "s": np.asarray(ep["s_m"], np.float32),
                     "lane": np.asarray(ep["lane"]).astype(np.int64),
                     "v": np.asarray(ep["v_mps"], np.float32),
                     "a": np.asarray(ep["a_mps2"], np.float32),
                     "mask": np.asarray(ep["present"], bool),
                     "vid": np.asarray(ep["vid_index"]).astype(np.int64)}
        self.n_base = 0
        ids = np.asarray(ep["vehicle_ids"]).astype(str)
        self.registry = [str(x) for x in ids]
        self.is_cav_reg = np.isin(ids, np.asarray(ep["cav_vehicle_ids"]).astype(str))
        self.first_seen_s = np.zeros(0)
        self.first_entrance = np.zeros(0, np.int64)
        self.k = self.cols["s"].shape[1]


def load_episode(path: str | Path) -> dict:
    path = Path(path)
    if FORBIDDEN.search(str(path)):
        raise SystemExit(f"refusing restricted path {path}")
    npz = path / "episode.npz" if path.is_dir() else path
    with np.load(npz, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = json.loads(str(ep.pop("meta_json")))
    return ep


def counter_at(ep: dict, j0: int) -> list:
    """RSU entrance counter [2,30] at tick ``j0`` (ticks <= j0 only)."""
    from wms.data.c4_episodes import episode_counter
    sl = slice(0, j0 + 1)
    c = episode_counter(np.asarray(ep["time_s"])[sl], np.asarray(ep["vid_index"])[sl],
                        np.asarray(ep["present"], bool)[sl],
                        np.asarray(ep.get("lane_carry", ep["lane"]))[sl])
    return c[-1].tolist()


def episode_record(ep: dict, *, t0_s: float, arm: str, horizon_steps: int = 240,
                   history_rows: int = HISTORY_ROWS, world: str | None = None) -> dict:
    """One state record at ``t0_s`` whose recorded future is stored under ``arm``."""
    tl = EpisodeTimeline(ep)
    j0 = tl.index_of(t0_s)
    if j0 is None:
        raise ValueError(f"t0 {t0_s} not on the episode time axis")
    if j0 + horizon_steps >= len(tl.t):
        raise ValueError(f"episode too short for t0 {t0_s} + {horizon_steps} steps")
    roster = sorted(tl.present(j0))
    meta = ep.get("meta", {})
    world = world or (f"w{meta.get('seed')}m{meta.get('demand_vph')}"
                      f"p{int(round(float(meta.get('penetration', 0)) * 100)):02d}t{int(t0_s)}")
    future = tl.future(j0, roster, horizon_steps)
    return {
        "key": f"c4ep/{world}__{arm}", "world": world, "arm": arm, "policy": arm,
        "t0_s": float(t0_s), "dt_s": DT_S, "horizon_s": (len(future) - 1) * DT_S,
        "cav_controllers": {}, "arms": [arm],
        "initial_state": tl.frame(j0, roster),
        "history": tl.history(j0, history_rows, HISTORY_S, False, align_ids=True),
        "recorded": {arm: future},
        "recorded_horizon_steps": {arm: len(future) - 1},
        "counter": counter_at(ep, j0),
        "seed": meta.get("seed"), "demand_vph": meta.get("demand_vph"),
        "penetration": meta.get("penetration"),
    }


def frames_identical(a: dict, b: dict, tol: float = 1e-4) -> bool:
    """Same vehicles with the same lane, position and speed (t0 identity across controllers)."""
    ra = {r["id"]: r for r in a["vehicles"]}
    rb = {r["id"]: r for r in b["vehicles"]}
    if set(ra) != set(rb):
        return False
    return all(ra[i]["lane"] == rb[i]["lane"] and abs(ra[i]["s"] - rb[i]["s"]) <= tol
               and abs(ra[i]["v"] - rb[i]["v"]) <= tol and ra[i]["is_cav"] == rb[i]["is_cav"]
               for i in ra)


def find_jobs(root: str | Path) -> dict[str, Path]:
    """``{job_id: episode_dir}`` of complete episodes under a collect output dir."""
    root = Path(root)
    if FORBIDDEN.search(str(root)):
        raise SystemExit(f"refusing restricted path {root}")
    out = {}
    for npz in sorted(root.glob("episodes/*/*/episode.npz")):
        d = npz.parent
        if not JOB_RE.match(d.name):
            continue
        try:
            ok = json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            ok = None
        if ok == "complete" and (d / "summary.json").is_file():
            out[d.name] = d
    return out


def sumo_full_metrics(ep: dict, t0_s: float, horizon_s: float, end_m: float = 75.44,
                      v_ref: float = 31.29) -> dict:
    """SUMO HDV metrics over ALL HDVs (incl. post-t0 entrants), for context only:
    discharge through ``end_m`` (vehicles crossing upward within (t0, t0+H]) and the delay proxy."""
    t = np.asarray(ep["time_s"])
    j0 = int(round((t0_s - t[0]) / DT_S))
    j1 = j0 + int(round(horizon_s / DT_S))
    vid = np.asarray(ep["vid_index"])[j0:j1 + 1]
    pres = np.asarray(ep["present"], bool)[j0:j1 + 1]
    s = np.asarray(ep["s_m"])[j0:j1 + 1]
    v = np.asarray(ep["v_mps"])[j0:j1 + 1]
    cav = np.asarray(ep["is_cav"], bool)[j0:j1 + 1]
    hdv = pres & ~cav
    delay = float(DT_S * np.clip(1.0 - v[1:] / v_ref, 0, None)[hdv[1:]].sum())
    first_pos: dict[int, float] = {}
    crossed = set()
    for i in range(s.shape[0]):
        for k in np.flatnonzero(hdv[i]):
            vk = int(vid[i, k])
            if vk < 0:
                continue
            if vk not in first_pos:
                first_pos[vk] = float(s[i, k])
            elif first_pos[vk] < end_m and s[i, k] >= end_m:
                crossed.add(vk)
    return {"discharge": len(crossed), "delay": delay}
