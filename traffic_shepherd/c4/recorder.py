"""Episode recorder and the converter into WM-S TRAIN trajectory windows.

Per episode directory (write-once)::

    manifest.json                   status (running/complete/failed), identity, config
    summary.json                    SUMO outcomes + wall-clock speed (complete only)
    episode.npz                     full-episode slot arrays (schema c4_sumo_episode/1)
    run/physics_steps.jsonl.gz      0.5 s wm41_logging/1 rows (WM-N1 featurizer input)
    run/action_timeline.jsonl.gz    1 s per-CAV issued commands
                                    (traffic_shepherd_action_timeline/1 rows)
    scenario/                       requests.rou.xml + scenario.sumocfg

``episode.npz`` (T = number of 0.5 s states incl. t = 0, C = control ticks, K slots):
``time_s [T]``, ``s_m/v_mps/a_mps2/length_true_m [T,K] f32``, ``lane [T,K] i8``
(-1 absent/junction), ``lane_carry [T,K] i8``, ``present/is_cav/junction [T,K] bool``,
``vid_index [T,K] i32`` -> ``vehicle_ids [V] str``, ``cav_vehicle_ids [C_v] str``,
``control_time_s [C]``, ``controller [C,K] i8`` (``wms.data.action_timeline.CONTROLLERS``
ids: native 0, damping 1, pi-external-u 2, direct 3), ``issued_acceleration_mps2 [C,K]``,
``issued_target_speed_mps [C,K]`` (0 where unused), ``requested_acceleration_mps2 [C,K]``
(policy output before SUMO-side clipping; NaN where none), ``lane_command_target`` /
``lane_request_target [C,K] i8`` (issued / requested WM target lane, -1 none), ``meta_json``.
Control row c is decided on state index ``2c`` (time c s) and holds for states 2c+1, 2c+2.

:func:`episode_windows` cuts an episode into the ``TrajectoryDataset.from_arrays``
contract (``states [N,T+1,K,7]`` = s, lane one-hot, v, a, nominal length 5 m;
``present``, ``identities``, ``is_cav``, ``controller_timeline``/``issued_*`` and the
pre-t0 ``history_kin``/``history_mask``/``history_target_mask``) and
:func:`write_trajectory_npz` saves it so ``TrajectoryDataset.from_pack(path)`` loads it.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path

import numpy as np

from .geometry import NOMINAL_LENGTH_M
from .sumo_io import dumps_row

EPISODE_SCHEMA = "c4_sumo_episode/1"
ACTION_LOG_SCHEMA = "traffic_shepherd_action_timeline/1"
CONTROLLERS = ("native", "damping", "pi-external-u", "direct")   # wms.data.action_timeline
DT_S = 0.5


def write_json(path, value) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


class EpisodeRecorder:
    """Collects every 0.5 s frame and every 1 s control decision of one episode."""

    def __init__(self, out_dir, *, k: int, physics_log: bool = True):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "run").mkdir()
        self.k = int(k)
        self.frames: list = []
        self.controls: list = []
        self.lane_controls: list = []
        self.lane_requests: list = []
        self._physics = (gzip.open(self.out / "run" / "physics_steps.jsonl.gz", "xt",
                                   encoding="utf-8", compresslevel=3) if physics_log else None)
        self._actions = gzip.open(self.out / "run" / "action_timeline.jsonl.gz", "xt",
                                  encoding="utf-8", compresslevel=3)
        self.manifest: dict = {"schema": EPISODE_SCHEMA, "status": "running"}

    @property
    def scenario_dir(self) -> Path:
        return self.out / "scenario"

    def start(self, manifest: dict) -> None:
        self.manifest.update(manifest)
        write_json(self.out / "manifest.json", self.manifest)

    def on_frame(self, frame, row=None) -> None:
        self.frames.append(frame)
        if row is not None and self._physics is not None:
            self._physics.write(dumps_row(row))

    def on_control(self, time_s: float, frame, decisions: dict, vehicle_ids: list,
                   lane_commands: dict | None = None, lane_requests: dict | None = None) -> None:
        """``decisions``: slot -> (controller name, issued accel, issued target, requested);
        ``lane_commands``: slot -> {target_lane, sumo_lane_index, duration_s} (issued changeLane);
        ``lane_requests``: slot -> requested WM target lane (the policy's lane output, before
        the env's safety mask; only slots with a request)."""
        lane_commands = lane_commands or {}
        ctrl, acc, tgt, req, lane_target, lane_req = control_row(
            self.k, decisions, lane_commands, lane_requests)
        for slot, (name, a, target, requested) in decisions.items():
            vid = vehicle_ids[int(frame.vid_index[slot])]
            extra = ({"lane_command": dict(lane_commands[slot])} if slot in lane_commands else {})
            if lane_requests and slot in lane_requests:
                extra["lane_request"] = int(lane_requests[slot])
            self._actions.write(json.dumps({
                "schema": ACTION_LOG_SCHEMA, "timestamp_s": int(round(time_s)),
                "vehicle_id": vid, "slot": int(slot), "owner": name, **extra,
                "issued_command": None if a is None else float(a),
                "issued_target_speed_mps": None if target is None else float(target),
                "requested_acceleration_mps2": (None if requested is None
                                                or not math.isfinite(requested)
                                                else float(requested)),
                "measured_speed_mps": float(frame.v[slot])}, allow_nan=False) + "\n")
        self.controls.append((float(time_s), ctrl, acc, tgt, req))
        self.lane_controls.append(lane_target)
        self.lane_requests.append(lane_req)

    def _close_streams(self) -> None:
        for stream in (self._physics, self._actions):
            if stream is not None and not stream.closed:
                stream.close()

    def finish(self, summary: dict, *, vehicle_ids: list, cav_ids, meta: dict) -> dict:
        self._close_streams()
        arrays = build_episode_arrays(self.frames, self.controls, self.lane_controls,
                                      self.lane_requests, k=self.k, vehicle_ids=vehicle_ids,
                                      cav_ids=cav_ids, meta=meta)
        np.savez_compressed(self.out / "episode.npz", **arrays)
        write_json(self.out / "summary.json", summary)
        self.manifest.update(status="complete")
        write_json(self.out / "manifest.json", self.manifest)
        return arrays

    def fail(self, error: str) -> None:
        self._close_streams()
        self.manifest.update(status="failed", error=error)
        write_json(self.out / "manifest.json", self.manifest)


def control_row(k: int, decisions: dict, lane_commands: dict | None = None,
                lane_requests: dict | None = None) -> tuple:
    """One control tick -> ``(controller, issued accel, issued target speed, requested accel,
    issued lane target, requested lane target)`` rows of length ``k``."""
    lane_target = np.full(k, -1, np.int8)
    for slot, command in (lane_commands or {}).items():
        lane_target[slot] = int(command["target_lane"])
    lane_req = np.full(k, -1, np.int8)
    for slot, target in (lane_requests or {}).items():
        lane_req[slot] = int(target)
    ctrl = np.zeros(k, np.int8)
    acc = np.zeros(k, np.float32)
    tgt = np.zeros(k, np.float32)
    req = np.full(k, np.nan, np.float32)
    for slot, (name, a, target, requested) in decisions.items():
        ctrl[slot] = CONTROLLERS.index(name)
        acc[slot] = 0.0 if a is None else a
        tgt[slot] = 0.0 if target is None else target
        if requested is not None and math.isfinite(requested):
            req[slot] = requested
    return ctrl, acc, tgt, req, lane_target, lane_req


def build_episode_arrays(frames: list, controls: list, lane_controls: list, lane_requests: list,
                         *, k: int, vehicle_ids: list, cav_ids, meta: dict) -> dict:
    """``episode.npz`` arrays (schema ``c4_sumo_episode/1``) from recorded frames / controls."""
    f = frames

    def stack(rows, dtype):
        return np.stack(rows) if rows else np.zeros((0, k), dtype)
    return {
        "time_s": np.asarray([x.time_s for x in f], np.float64),
        "s_m": np.stack([x.s for x in f]), "lane": np.stack([x.lane for x in f]),
        "lane_carry": np.stack([x.lane_carry for x in f]),
        "v_mps": np.stack([x.v for x in f]), "a_mps2": np.stack([x.a for x in f]),
        "length_true_m": np.stack([x.length_true for x in f]),
        "present": np.stack([x.present for x in f]),
        "is_cav": np.stack([x.is_cav for x in f]),
        "junction": np.stack([x.junction for x in f]),
        "vid_index": np.stack([x.vid_index for x in f]),
        "vehicle_ids": np.asarray(vehicle_ids if vehicle_ids else [""], dtype=str),
        "cav_vehicle_ids": np.asarray(sorted(cav_ids) or [""], dtype=str),
        "control_time_s": np.asarray([c[0] for c in controls], np.float64),
        "controller": stack([c[1] for c in controls], np.int8),
        "issued_acceleration_mps2": stack([c[2] for c in controls], np.float32),
        "issued_target_speed_mps": stack([c[3] for c in controls], np.float32),
        "requested_acceleration_mps2": stack([c[4] for c in controls], np.float32),
        # WM target lane of an issued lane command per control tick, -1 = none.
        "lane_command_target": stack(lane_controls, np.int8),
        # WM target lane REQUESTED by the policy (before the env's safety mask), -1 = none.
        "lane_request_target": stack(lane_requests, np.int8),
        "controller_vocab": np.asarray(CONTROLLERS, dtype=str),
        "meta_json": np.asarray(json.dumps({"schema": EPISODE_SCHEMA, **meta},
                                           allow_nan=False)),
    }


# ---------------------------------------------------------------- WM-S windows
def load_episode(path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "episode.npz"
    with np.load(p, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    meta = json.loads(str(out.pop("meta_json")))
    if meta.get("schema") != EPISODE_SCHEMA:
        raise ValueError(f"{p}: not a {EPISODE_SCHEMA} episode")
    out["meta"] = meta
    return out


def episode_kin(ep: dict) -> np.ndarray:
    """``[T,K,7]`` pack kinematics of every recorded state."""
    present = ep["present"]
    lane = ep["lane"]
    kin = np.zeros(present.shape + (7,), np.float32)
    kin[..., 0] = np.where(present, ep["s_m"], 0.0)
    for j in range(3):
        kin[..., 1 + j] = lane == j
    kin[..., 4] = np.where(present, ep["v_mps"], 0.0)
    kin[..., 5] = np.where(present, ep["a_mps2"], 0.0)
    kin[..., 6] = np.where(present, NOMINAL_LENGTH_M, 0.0)
    return kin


def episode_windows(ep: dict, *, window_s: float = 30.0, stride_s: float = 10.0,
                    start_s: float | None = None, end_s: float | None = None,
                    history_s: float | None = 60.0, min_vehicles: int = 1) -> dict:
    """Cut one episode into ``TrajectoryDataset.from_arrays`` TRAIN windows.

    Windows start on the 1 s control grid in ``[start_s, end_s - window_s]``
    (default start = the episode's warmup end).  ``history_s`` (None = 10 native
    ticks) of pre-t0 native states are attached as ``history_kin``.
    """
    if isinstance(ep, (str, Path)):
        ep = load_episode(ep)
    times = np.asarray(ep["time_s"])
    meta = ep.get("meta", {})
    t_steps = int(round(window_s / DT_S))
    stride = int(round(stride_s / DT_S))
    if t_steps < 2 or stride < 2 or t_steps % 2 or stride % 2:
        raise ValueError("window_s and stride_s must be whole control seconds >= 1")
    start = float(meta.get("warmup_s", 0.0) if start_s is None else start_s)
    end = float(times[-1] if end_s is None else end_s)
    hist_ticks = 10 if history_s is None else int(math.floor(history_s / DT_S + 1e-8)) + 1
    kin = episode_kin(ep)
    present, vid, is_cav = ep["present"], ep["vid_index"], ep["is_cav"]
    junction = ep["junction"]
    ctrl_t = np.asarray(ep["control_time_s"])
    n_ctrl = t_steps // 2
    k = present.shape[1]
    out = {name: [] for name in ("states", "present", "identities", "is_cav", "junction",
                                 "controller_timeline", "issued_acceleration_mps2",
                                 "issued_target_speed_mps", "history_kin", "history_mask",
                                 "history_target_mask", "t0_s")}
    i0 = int(round(start / DT_S))
    while times[min(i0, len(times) - 1)] + window_s <= end + 1e-9 and i0 + t_steps < len(times):
        t0 = float(times[i0])
        c0 = int(np.searchsorted(ctrl_t, t0 - 1e-6))
        if (c0 + n_ctrl <= len(ctrl_t) and abs(ctrl_t[c0] - t0) < 1e-6
                and int(present[i0].sum()) >= min_vehicles):
            sl = slice(i0, i0 + t_steps + 1)
            out["states"].append(kin[sl])
            out["present"].append(present[sl])
            out["identities"].append(vid[sl])
            out["is_cav"].append(is_cav[sl])
            out["junction"].append(junction[sl])
            out["controller_timeline"].append(ep["controller"][c0:c0 + n_ctrl].astype(np.int64))
            out["issued_acceleration_mps2"].append(
                ep["issued_acceleration_mps2"][c0:c0 + n_ctrl])
            out["issued_target_speed_mps"].append(ep["issued_target_speed_mps"][c0:c0 + n_ctrl])
            lo = i0 - hist_ticks + 1
            hk = np.zeros((hist_ticks, k, 7), np.float32)
            hm = np.zeros((hist_ticks, k), bool)
            hv = np.full((hist_ticks, k), -1, np.int64)
            src = slice(max(lo, 0), i0 + 1)
            pad = max(0, -lo)
            hk[pad:], hm[pad:], hv[pad:] = kin[src], present[src], vid[src]
            out["history_kin"].append(hk)
            out["history_mask"].append(hm)
            out["history_target_mask"].append((hv[:-1] == hv[1:]) & (hv[:-1] >= 0))
            out["t0_s"].append(t0)
        i0 += stride
    if not out["states"]:
        raise ValueError("episode yields no complete window in the requested range")
    arrays = {name: np.stack(values) for name, values in out.items()}
    arrays["controller_vocab"] = np.asarray(CONTROLLERS, dtype=str)
    return arrays


def write_trajectory_npz(path, arrays: dict) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
