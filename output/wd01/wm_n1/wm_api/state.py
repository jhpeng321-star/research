"""The per-vehicle t0 ``StateSnapshot`` schema + the physics-tick converter.

State schema (api_version 1.0)
==============================

===============================  ==========  ==============================
field                            type        meaning
===============================  ==========  ==============================
ids                              list[str]   vehicle ids (slot order)
lane_phys                        list[int]   0 main_left / 1 main_right /
                                             2 accel; -1 = not in the model
                                             world (junction rows are
                                             EXCLUDED, never guessed)
s_down_m                         list[float] merge-relative downstream
                                             coordinate (features.geometry
                                             convention; 0 at the merge
                                             start, negative upstream)
v_mps                            list[float] speed
a_mps2                           list[float] last-step acceleration
length_m                         list[float] bumper-to-bumper length
                                             (default 5.0 m nominal)
is_cav                           list[bool]  CAV flag (CAVs self-identify)
t0_s                             float       snapshot time (s)
entrance_counters                float[2,30] optional; entrance 0 mainline,
                                             1 ramp; 10 s departure bins over
                                             the last 300 s, bin 0 most
                                             recent (features.counters
                                             semantics).  Present -> the P4
                                             arrival head runs in the loop.
obstacles                        list        optional stopped-vehicle
                                             records ``{id, lane_phys,
                                             s_down_m, length_m}`` (INC1
                                             convention: held at s with
                                             v = 0, a = 0 for the window)
===============================  ==========  ==============================

The converter :func:`state_from_physics_tick` accepts ONE parsed
``physics_steps.jsonl`` row (the chunk-4 physics-log tick format::

    {"time_s": t,
     "vehicles": {"<id>": {"lane_id": ..., "lane_position_m": ...,
                           "speed_mps": ..., "acceleration_mps2": ...,
                           "road_id": ...}},
     "departed_ids": [...], "cav_ids": [...]}

) and maps it through the frozen ``features.geometry`` lane table.
:func:`counters_from_ticks` rebuilds the entrance counters from a tick
history (the last 300 s), mirroring ``features.counters``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .errors import StateSchemaError

#: The frozen slot budget (model.constants.K_MAX; never redefined here).
K_MAX = 128

_LANES = (-1, 0, 1, 2)


@dataclass(frozen=True)
class ObstacleRecord:
    """One held stopped vehicle (INC1 convention; v = 0, a = 0)."""

    id: str
    lane_phys: int
    s_down_m: float
    length_m: float = 5.0


@dataclass
class StateSnapshot:
    """Per-vehicle t0 snapshot (schema in the module docstring)."""

    ids: list
    lane_phys: np.ndarray
    s_down_m: np.ndarray
    v_mps: np.ndarray
    a_mps2: np.ndarray
    length_m: np.ndarray
    is_cav: np.ndarray
    t0_s: float = 0.0
    entrance_counters: Optional[np.ndarray] = None
    obstacles: tuple = field(default_factory=tuple)
    excluded: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.ids)
        arrays = (self.lane_phys, self.s_down_m, self.v_mps, self.a_mps2,
                  self.length_m, self.is_cav)
        for a in arrays:
            a = np.asarray(a)
            if a.shape != (n,):
                raise StateSchemaError(
                    f"state arrays must all have length {n} (ids); got "
                    f"{a.shape}")
        self.lane_phys = np.asarray(self.lane_phys, dtype=np.int64)
        bad = ~np.isin(self.lane_phys, _LANES)
        if bad.any():
            raise StateSchemaError(
                f"lane_phys must be one of {-1, 0, 1, 2}; offending ids "
                f"{[self.ids[i] for i in np.flatnonzero(bad)]}")
        self.s_down_m = np.asarray(self.s_down_m, dtype=np.float64)
        self.v_mps = np.asarray(self.v_mps, dtype=np.float64)
        self.a_mps2 = np.asarray(self.a_mps2, dtype=np.float64)
        self.length_m = np.asarray(self.length_m, dtype=np.float64)
        self.is_cav = np.asarray(self.is_cav, dtype=bool)
        if self.t0_s < 0:
            raise StateSchemaError("t0_s must be >= 0")
        if self.entrance_counters is not None:
            c = np.asarray(self.entrance_counters, dtype=np.float64)
            if c.shape != (2, 30):
                raise StateSchemaError(
                    f"entrance_counters must be [2, 30]; got {c.shape}")
            if not np.all(np.isfinite(c)) or (c < 0).any():
                raise StateSchemaError("entrance_counters must be finite >= 0")
            self.entrance_counters = c
        for ob in self.obstacles:
            if int(ob.lane_phys) not in (0, 1, 2):
                raise StateSchemaError(
                    f"obstacle {ob.id!r} needs a physical lane (0/1/2)")
            if not float(ob.length_m) > 0:
                raise StateSchemaError(
                    f"obstacle {ob.id!r} length_m must be > 0")

    @property
    def n_model_vehicles(self) -> int:
        """Vehicles in the model world (lane >= 0) plus obstacles."""
        return int((self.lane_phys >= 0).sum()) + len(self.obstacles)

    def model_rows(self) -> list:
        """``(index, lane_phys, s_down, v, a, length, is_cav)`` rows."""
        return [(i, int(self.lane_phys[i]), float(self.s_down_m[i]),
                 float(self.v_mps[i]), float(self.a_mps2[i]),
                 float(self.length_m[i]), bool(self.is_cav[i]))
                for i in range(len(self.ids)) if int(self.lane_phys[i]) >= 0]

    def to_json(self) -> dict:
        return dict(
            ids=[str(x) for x in self.ids],
            lane_phys=[int(x) for x in self.lane_phys],
            s_down_m=[float(x) for x in self.s_down_m],
            v_mps=[float(x) for x in self.v_mps],
            a_mps2=[float(x) for x in self.a_mps2],
            length_m=[float(x) for x in self.length_m],
            is_cav=[bool(x) for x in self.is_cav],
            t0_s=float(self.t0_s),
            entrance_counters=(None if self.entrance_counters is None
                               else [[float(x) for x in row] for row in
                                     self.entrance_counters]),
            obstacles=[dict(id=str(o.id), lane_phys=int(o.lane_phys),
                            s_down_m=float(o.s_down_m),
                            length_m=float(o.length_m))
                       for o in self.obstacles],
            excluded=dict(self.excluded))

    @classmethod
    def from_json(cls, data: Mapping) -> "StateSnapshot":
        if not isinstance(data, Mapping):
            raise StateSchemaError("state must be a JSON object")
        required = ("ids", "lane_phys", "s_down_m", "v_mps", "a_mps2",
                    "length_m", "is_cav")
        missing = [k for k in required if k not in data]
        if missing:
            raise StateSchemaError(f"state is missing fields {missing}")
        obstacles = tuple(
            ObstacleRecord(id=str(o["id"]), lane_phys=int(o["lane_phys"]),
                           s_down_m=float(o["s_down_m"]),
                           length_m=float(o.get("length_m", 5.0)))
            for o in data.get("obstacles") or ())
        counters = data.get("entrance_counters")
        return cls(
            ids=[str(x) for x in data["ids"]],
            lane_phys=np.asarray(data["lane_phys"], dtype=np.int64),
            s_down_m=np.asarray(data["s_down_m"], dtype=np.float64),
            v_mps=np.asarray(data["v_mps"], dtype=np.float64),
            a_mps2=np.asarray(data["a_mps2"], dtype=np.float64),
            length_m=np.asarray(data["length_m"], dtype=np.float64),
            is_cav=np.asarray(data["is_cav"], dtype=bool),
            t0_s=float(data.get("t0_s", 0.0)),
            entrance_counters=(None if counters is None
                               else np.asarray(counters, dtype=np.float64)),
            obstacles=obstacles,
            excluded=dict(data.get("excluded") or {}))


def state_from_physics_tick(tick: Mapping, *, t0_s: Optional[float] = None,
                            lengths: Optional[Mapping] = None,
                            nominal_length_m: float = 5.0) -> StateSnapshot:
    """Convert one chunk-4 physics-log tick (parsed JSON row) to a snapshot.

    Only the frozen fields are read (``lane_id``, ``lane_position_m``,
    ``speed_mps``, ``acceleration_mps2``, ``road_id``, ``departed_ids``,
    ``cav_ids``).  Junction-internal rows (``:``-prefixed lanes) and lane
    ids outside the frozen WM-30 table are EXCLUDED and reported in
    ``excluded`` (never silently guessed).  The CAV flag comes from the
    tick's ``cav_ids`` list (CAVs self-identify to the RSU).
    """
    from features.geometry import GeometryError, is_junction_lane, s_down_of

    vehs = tick.get("vehicles") or {}
    cav_ids = {str(x) for x in tick.get("cav_ids", ())}
    lengths = lengths or {}
    ids, lanes, sdown, v, a, length, cav = [], [], [], [], [], [], []
    excluded: dict = {}
    for vid, row in vehs.items():
        vid = str(vid)
        lane_id = (row or {}).get("lane_id")
        if lane_id is None or is_junction_lane(lane_id):
            excluded[vid] = "junction_or_missing_lane"
            continue
        try:
            lane_phys, s = s_down_of(lane_id, float(row["lane_position_m"]))
        except (GeometryError, KeyError, TypeError, ValueError) as exc:
            excluded[vid] = f"unmapped_lane ({exc})"
            continue
        ids.append(vid)
        lanes.append(lane_phys)
        sdown.append(s)
        v.append(float(row.get("speed_mps", 0.0) or 0.0))
        a.append(float(row.get("acceleration_mps2", 0.0) or 0.0))
        length.append(float(lengths.get(vid, nominal_length_m)))
        cav.append(vid in cav_ids)
    return StateSnapshot(
        ids=ids, lane_phys=np.asarray(lanes, dtype=np.int64),
        s_down_m=np.asarray(sdown, dtype=np.float64),
        v_mps=np.asarray(v, dtype=np.float64),
        a_mps2=np.asarray(a, dtype=np.float64),
        length_m=np.asarray(length, dtype=np.float64),
        is_cav=np.asarray(cav, dtype=bool),
        t0_s=float(t0_s if t0_s is not None else tick.get("time_s", 0.0)),
        excluded=excluded)


def counters_from_ticks(ticks: Sequence[Mapping], t_now: float) -> np.ndarray:
    """Entrance counters ``[2, 30]`` from a tick history (<= t_now rows).

    Mirrors the frozen featurizer: departures come ONLY from the tick lists
    ``departed_ids``; the entrance is the departing vehicle's road edge
    (mainline entry 1191554636 -> 0, ramp entry 11553048#1 -> 1; unknown
    roads fall back on the ``ramp.`` id prefix).  Bin b covers
    ``[t - 300 + 10 b, t - 300 + 10 (b+1))``; bin 0 is the most recent.
    """
    from features.counters import DepartureLog, entrance_of_edge
    from features.geometry import ENTRANCE_MAINLINE, ENTRANCE_RAMP

    dep = DepartureLog(float(t_now))
    for tick in ticks:
        t = float(tick.get("time_s", 0.0))
        if t > float(t_now) + 1e-9:
            continue
        vehs = tick.get("vehicles") or {}
        for vid in (str(x) for x in tick.get("departed_ids", ())):
            vv = vehs.get(vid)
            road = vv.get("road_id") if vv else None
            try:
                entrance = entrance_of_edge(road) if road else None
            except KeyError:
                entrance = None
            if entrance is None:
                entrance = (ENTRANCE_RAMP if vid.startswith("ramp.")
                            else ENTRANCE_MAINLINE)
            dep.add(t, entrance)
    return dep.counter_at(float(t_now))
