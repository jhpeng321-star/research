"""Cell Transmission Model (CTM) for the WM-S merge road, exact + smooth.

This is a purely macroscopic traffic model: it advances cell vehicle COUNTS
with a triangular fundamental diagram (FD), a merge cell that arbitrates two
competing demands (mainline + on-ramp) and a capacity drop once a queue exists.
It is deliberately *not* a per-car model: no vehicle identity, no controller.

Frame
-----
The road is expressed in the frozen WM-S model frame ``s_down`` (see
``features.geometry.s_down_of`` and ``wms.physics_torch``):

* mainline lanes 0/1 upstream occupy ``s in [-2645.67, 0]``;
* on-ramp lane 2 upstream occupies ``s in [-378.66, 0]``;
* the merge zone is ``s in [0, 75.44]`` with 3 lanes;
* the downstream mainline occupies ``s in [75.44, 553.33]`` with 2 lanes.

Cells are ~100 m (configurable).  Boundary inflows come only from the observed
t0 entrance counters, using exactly the ``wms.entries.observed_rates`` rule
(most recent ``window_bins`` 10 s bins, no future).  The downstream boundary is
a free outflow (all demand leaves).

Two implementations are provided and kept in lock step:

* :func:`run_numpy` - exact ``min``/``clamp`` reference used for evaluation;
* :class:`TorchCTM` - batched torch with a soft-min/soft-max temperature that
  tends to the exact model as ``tau -> 0`` (for future gradient coupling).

Both conserve vehicles exactly (up to float rounding) and are covered by
``tests/test_wms_eval_ctm.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# frozen geometry (model frame, metres)
# --------------------------------------------------------------------------- #
S_MAIN_START_M = -2645.67
S_RAMP_START_M = -378.66
S_MERGE_START_M = 0.0
S_MERGE_END_M = 75.44
S_DOWN_END_M = 553.33

LANES_MAIN = 2
LANES_RAMP = 1
LANES_MERGE = 3
LANES_DOWN = 2

SECTION_UPSTREAM, SECTION_MERGE, SECTION_DOWNSTREAM = 0, 1, 2
SECTION_NAMES = ("upstream", "merge", "downstream")

#: Cell role labels (main/ramp upstream chains, the merge cell, downstream).
ROLE_MAIN, ROLE_RAMP, ROLE_MERGE, ROLE_DOWN = 0, 1, 2, 3

#: Default section lengths used by ``wms_eval.metrics`` for density (km scale).
DEFAULT_SECTION_LENGTHS_M = {"upstream": 1000.0, "merge": 75.44, "downstream": 1000.0}

NOMINAL_V_FREE_MPS = 27.78          # 100 km/h free speed
NOMINAL_JAM_DENSITY_PER_LANE = 130.0  # veh/km/lane
NOMINAL_CAPACITY_PER_LANE = 2000.0    # veh/h/lane
NOMINAL_BACKWAVE_MPS = 5.0            # |w| backward wave speed


# --------------------------------------------------------------------------- #
# fundamental diagram
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TriangularFD:
    """Per-lane triangular fundamental diagram (SI + FD-friendly units)."""

    v_free_mps: float = NOMINAL_V_FREE_MPS
    capacity_vehph_per_lane: float = NOMINAL_CAPACITY_PER_LANE
    jam_density_vehpkmlane: float = NOMINAL_JAM_DENSITY_PER_LANE
    w_mps: float = NOMINAL_BACKWAVE_MPS

    @property
    def capacity_mps_per_lane(self) -> float:
        return self.capacity_vehph_per_lane / 3600.0

    @property
    def rho_crit_vehpkmlane(self) -> float:
        return self.capacity_vehph_per_lane / max(self.v_free_mps * 3.6, 1e-9)

    def to_dict(self) -> dict:
        return {
            "v_free_mps": float(self.v_free_mps),
            "capacity_vehph_per_lane": float(self.capacity_vehph_per_lane),
            "jam_density_vehpkmlane": float(self.jam_density_vehpkmlane),
            "w_mps": float(self.w_mps),
            "rho_crit_vehpkmlane": float(self.rho_crit_vehpkmlane),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TriangularFD":
        return cls(
            v_free_mps=float(payload["v_free_mps"]),
            capacity_vehph_per_lane=float(payload["capacity_vehph_per_lane"]),
            jam_density_vehpkmlane=float(payload["jam_density_vehpkmlane"]),
            w_mps=float(payload["w_mps"]),
        )


@dataclass
class CTMParams:
    """All CTM parameters (FDs per section + merge arbitration)."""

    up: TriangularFD = field(default_factory=TriangularFD)
    ramp: TriangularFD = field(default_factory=TriangularFD)
    merge: TriangularFD = field(default_factory=TriangularFD)
    down: TriangularFD = field(default_factory=TriangularFD)
    merge_capacity_drop: float = 0.9
    merge_priority_main: float = 2.0
    merge_priority_ramp: float = 1.0
    cell_length_m: float = 100.0
    dt_s: float = 0.5
    queue_speed_mps: float = 0.5
    counter_window_bins: int = 12

    def fd(self, role: int) -> TriangularFD:
        return {ROLE_MAIN: self.up, ROLE_RAMP: self.ramp,
                ROLE_MERGE: self.merge, ROLE_DOWN: self.down}[role]

    def to_dict(self) -> dict:
        return {
            "up": self.up.to_dict(), "ramp": self.ramp.to_dict(),
            "merge": self.merge.to_dict(), "down": self.down.to_dict(),
            "merge_capacity_drop": float(self.merge_capacity_drop),
            "merge_priority_main": float(self.merge_priority_main),
            "merge_priority_ramp": float(self.merge_priority_ramp),
            "cell_length_m": float(self.cell_length_m),
            "dt_s": float(self.dt_s),
            "queue_speed_mps": float(self.queue_speed_mps),
            "counter_window_bins": int(self.counter_window_bins),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CTMParams":
        return cls(
            up=TriangularFD.from_dict(payload["up"]),
            ramp=TriangularFD.from_dict(payload["ramp"]),
            merge=TriangularFD.from_dict(payload["merge"]),
            down=TriangularFD.from_dict(payload["down"]),
            merge_capacity_drop=float(payload.get("merge_capacity_drop", 0.9)),
            merge_priority_main=float(payload.get("merge_priority_main", 2.0)),
            merge_priority_ramp=float(payload.get("merge_priority_ramp", 1.0)),
            cell_length_m=float(payload.get("cell_length_m", 100.0)),
            dt_s=float(payload.get("dt_s", 0.5)),
            queue_speed_mps=float(payload.get("queue_speed_mps", 0.5)),
            counter_window_bins=int(payload.get("counter_window_bins", 12)),
        )


def default_params(**overrides) -> CTMParams:
    params = CTMParams()
    for key, value in overrides.items():
        if not hasattr(params, key):
            raise ValueError(f"unknown CTM parameter {key!r}")
        setattr(params, key, value)
    return params


# --------------------------------------------------------------------------- #
# cell layout
# --------------------------------------------------------------------------- #
def _uniform_cells(s_start: float, s_end: float, cell_length_m: float):
    span = max(0.0, float(s_end) - float(s_start))
    n = max(1, int(np.ceil(span / max(1e-6, float(cell_length_m)))))
    length = span / n
    starts = s_start + np.arange(n, dtype=np.float64) * length
    ends = starts + length
    return starts, ends, length


@dataclass
class CellLayout:
    """Ordered cells for the whole road, with chain index arrays."""

    s_start: np.ndarray
    s_end: np.ndarray
    length_m: np.ndarray
    lanes: np.ndarray
    role: np.ndarray
    section: np.ndarray
    main_idx: np.ndarray
    ramp_idx: np.ndarray
    down_idx: np.ndarray
    merge_idx: int

    @property
    def n_cells(self) -> int:
        return int(self.s_start.shape[0])

    def section_length_m(self, section: int) -> float:
        if section == SECTION_UPSTREAM:
            return float((self.s_end[self.section == SECTION_UPSTREAM]
                          - self.s_start[self.section == SECTION_UPSTREAM]).sum())
        if section == SECTION_MERGE:
            return S_MERGE_END_M - S_MERGE_START_M
        return float((self.s_end[self.section == SECTION_DOWNSTREAM]
                      - self.s_start[self.section == SECTION_DOWNSTREAM]).sum())


def build_layout(cell_length_m: float = 100.0) -> CellLayout:
    """Build the main/ramp/merge/downstream cell chains (~100 m)."""
    starts, ends, _ = _uniform_cells(S_MAIN_START_M, S_MERGE_START_M, cell_length_m)
    main_idx = np.arange(starts.shape[0], dtype=np.int64)
    all_starts = [starts]
    all_ends = [ends]
    all_lanes = [np.full(starts.shape[0], LANES_MAIN, dtype=np.float64)]
    all_role = [np.full(starts.shape[0], ROLE_MAIN, dtype=np.int64)]
    all_section = [np.full(starts.shape[0], SECTION_UPSTREAM, dtype=np.int64)]

    r_starts, r_ends, _ = _uniform_cells(S_RAMP_START_M, S_MERGE_START_M, cell_length_m)
    ramp_idx = main_idx.shape[0] + np.arange(r_starts.shape[0], dtype=np.int64)
    all_starts.append(r_starts)
    all_ends.append(r_ends)
    all_lanes.append(np.full(r_starts.shape[0], LANES_RAMP, dtype=np.float64))
    all_role.append(np.full(r_starts.shape[0], ROLE_RAMP, dtype=np.int64))
    all_section.append(np.full(r_starts.shape[0], SECTION_UPSTREAM, dtype=np.int64))

    merge_idx = int(main_idx.shape[0] + r_starts.shape[0])
    all_starts.append(np.array([S_MERGE_START_M]))
    all_ends.append(np.array([S_MERGE_END_M]))
    all_lanes.append(np.array([LANES_MERGE], dtype=np.float64))
    all_role.append(np.array([ROLE_MERGE], dtype=np.int64))
    all_section.append(np.array([SECTION_MERGE], dtype=np.int64))

    d_starts, d_ends, _ = _uniform_cells(S_MERGE_END_M, S_DOWN_END_M, cell_length_m)
    down_idx = merge_idx + 1 + np.arange(d_starts.shape[0], dtype=np.int64)
    all_starts.append(d_starts)
    all_ends.append(d_ends)
    all_lanes.append(np.full(d_starts.shape[0], LANES_DOWN, dtype=np.float64))
    all_role.append(np.full(d_starts.shape[0], ROLE_DOWN, dtype=np.int64))
    all_section.append(np.full(d_starts.shape[0], SECTION_DOWNSTREAM, dtype=np.int64))

    return CellLayout(
        s_start=np.concatenate(all_starts), s_end=np.concatenate(all_ends),
        length_m=np.concatenate([e - s for s, e in zip(all_starts, all_ends)]),
        lanes=np.concatenate(all_lanes), role=np.concatenate(all_role),
        section=np.concatenate(all_section), main_idx=main_idx, ramp_idx=ramp_idx,
        down_idx=down_idx, merge_idx=merge_idx)


def cell_index_of(layout: CellLayout, s_m, lane) -> np.ndarray:
    """Hard cell assignment for vehicle positions/lanes (``-1`` = off-road)."""
    s = np.asarray(s_m, dtype=np.float64)
    lane = np.asarray(lane)
    out = np.full(s.shape[0], -1, dtype=np.int64)
    is_ramp = lane == 2
    up = s < S_MERGE_START_M
    merge = (s >= S_MERGE_START_M) & (s <= S_MERGE_END_M)
    down = s > S_MERGE_END_M
    # upstream mainline
    sel = up & ~is_ramp
    if sel.any():
        idx = np.clip(np.searchsorted(layout.s_end[layout.main_idx], s[sel],
                                      side="right"), 0, layout.main_idx.shape[0] - 1)
        out[sel] = layout.main_idx[idx]
    sel = up & is_ramp
    if sel.any():
        idx = np.clip(np.searchsorted(layout.s_end[layout.ramp_idx], s[sel],
                                      side="right"), 0, layout.ramp_idx.shape[0] - 1)
        out[sel] = layout.ramp_idx[idx]
    out[merge] = layout.merge_idx
    sel = down & ~is_ramp
    if sel.any():
        idx = np.clip(np.searchsorted(layout.s_end[layout.down_idx], s[sel],
                                      side="right"), 0, layout.down_idx.shape[0] - 1)
        out[sel] = layout.down_idx[idx]
    return out


def hard_bin_counts(layout: CellLayout, s_m, lane, present=None, weight=None) -> np.ndarray:
    """Exact vehicle count per cell from t0 positions/lanes (evaluation path)."""
    s = np.asarray(s_m, dtype=np.float64).reshape(-1)
    lane = np.asarray(lane).reshape(-1)
    if present is not None:
        keep = np.asarray(present, dtype=bool).reshape(-1)
        s, lane = s[keep], lane[keep]
        if weight is not None:
            weight = np.asarray(weight, dtype=np.float64).reshape(-1)[keep]
    counts = np.zeros(layout.n_cells, dtype=np.float64)
    idx = cell_index_of(layout, s, lane)
    valid = idx >= 0
    if weight is None:
        np.add.at(counts, idx[valid], 1.0)
    else:
        np.add.at(counts, idx[valid], np.asarray(weight, dtype=np.float64)[valid])
    return counts


def soft_bin_counts(layout: CellLayout, s_m, lane, *, sigma_m: float = 50.0,
                    present=None) -> np.ndarray:
    """Soft kernel binning of t0 vehicles for future gradient training.

    Each vehicle's mass is shared between the nearest downstream cell of its
    chain and its two neighbours with a triangular kernel of half-width
    ``sigma_m``; the total mass is exactly the vehicle count.  ``sigma_m -> 0``
    recovers :func:`hard_bin_counts`.
    """
    s = np.asarray(s_m, dtype=np.float64).reshape(-1)
    lane = np.asarray(lane).reshape(-1)
    if present is not None:
        keep = np.asarray(present, dtype=bool).reshape(-1)
        s, lane = s[keep], lane[keep]
    counts = np.zeros(layout.n_cells, dtype=np.float64)
    for role, chain in ((ROLE_MAIN, layout.main_idx), (ROLE_RAMP, layout.ramp_idx),
                        (ROLE_DOWN, layout.down_idx)):
        is_ramp = lane == 2
        if role == ROLE_MAIN:
            sel = (s < S_MERGE_START_M) & ~is_ramp
        elif role == ROLE_RAMP:
            sel = (s < S_MERGE_START_M) & is_ramp
        else:
            sel = (s > S_MERGE_END_M) & ~is_ramp
        if not sel.any():
            continue
        centers = 0.5 * (layout.s_start[chain] + layout.s_end[chain])
        pos = s[sel][:, None]
        w = np.clip(1.0 - np.abs(pos - centers[None, :]) / max(1e-6, sigma_m), 0.0, None)
        row = w.sum(axis=1)
        empty = row <= 0
        if empty.any():
            nearest = np.abs(pos[empty] - centers[None, :]).argmin(axis=1)
            w[empty] = 0.0
            w[np.flatnonzero(empty), nearest] = 1.0
            row = w.sum(axis=1)
        w = w / row[:, None]
        np.add.at(counts, chain, w.sum(axis=0))
    merge_sel = (s >= S_MERGE_START_M) & (s <= S_MERGE_END_M)
    counts[layout.merge_idx] += float(merge_sel.sum())
    return counts


# --------------------------------------------------------------------------- #
# boundary inflows from the t0 entrance counters
# --------------------------------------------------------------------------- #
def counter_rates(counter, window_bins: int = 12) -> np.ndarray:
    """Per-entrance arrival rate [2] veh/s from the t0 counter (no future).

    Byte-for-byte the ``wms.entries.observed_rates`` rule: mean count over the
    most recent ``window_bins`` (x 10 s) bins.  Reimplemented here so this
    module has no import dependency on the torch world-model package.
    """
    c = np.asarray(counter, dtype=np.float64).reshape(2, 30)
    n = max(1, min(int(window_bins), 30))
    return c[:, :n].sum(axis=1) / (n * 10.0)


# --------------------------------------------------------------------------- #
# smooth primitives (shared by the numpy soft path and torch)
# --------------------------------------------------------------------------- #
def softmin(a, b, tau: float):
    tau = max(float(tau), 1e-12)
    return -tau * np.logaddexp(-np.asarray(a) / tau, -np.asarray(b) / tau)


def softmax(a, b, tau: float):
    tau = max(float(tau), 1e-12)
    return tau * np.logaddexp(np.asarray(a) / tau, np.asarray(b) / tau)


def merge_allocation(demand_main, demand_ramp, supply, *, p_main: float = 2.0,
                     p_ramp: float = 1.0, smooth: bool = False, tau: float = 1e-3):
    """Arbitrate two competing demands into one merge supply.

    Mainline has priority ``p_main``; when both approaches are metered the
    mainline receives at least ``p_main/(p_main+p_ramp)`` of the supply.  The
    exact rule is Daganzo's merge; the smooth path replaces ``min``/``max`` with
    soft-min/soft-max so ``tau -> 0`` recovers the exact rule.
    """
    d_main = np.asarray(demand_main, dtype=np.float64)
    d_ramp = np.asarray(demand_ramp, dtype=np.float64)
    s = np.asarray(supply, dtype=np.float64)
    frac = float(p_main) / max(float(p_main) + float(p_ramp), 1e-9)
    if not smooth:
        y_main = np.minimum(d_main, np.maximum(s - d_ramp, frac * s))
        y_ramp = np.minimum(d_ramp, s - y_main)
    else:
        floor = softmax(s - d_ramp, frac * s, tau)
        y_main = softmin(d_main, floor, tau)
        y_ramp = softmin(d_ramp, softmax(s - y_main, 0.0, tau), tau)
    y_main = np.minimum(np.maximum(y_main, 0.0), d_main)
    y_ramp = np.minimum(np.maximum(y_ramp, 0.0), d_ramp)
    return y_main, y_ramp


# --------------------------------------------------------------------------- #
# exact numpy simulation (evaluation reference)
# --------------------------------------------------------------------------- #
@dataclass
class CTMRun:
    """Record of a CTM rollout: per-step cell counts, speeds and boundary flow."""

    counts: np.ndarray            # [T, C] vehicles per cell at each sample
    speeds: np.ndarray            # [T, C] cell mean speed (m/s)
    merge_outflow: np.ndarray     # [T] vehicles crossing s=75.44 during the step
    main_inflow: np.ndarray       # [T]
    ramp_inflow: np.ndarray       # [T]
    outflow: np.ndarray           # [T] vehicles leaving the downstream boundary
    dt_s: float
    t0_s: float
    layout: CellLayout
    params: CTMParams

    @property
    def merge_queue(self) -> np.ndarray:
        return self.counts[:, self.layout.merge_idx]


def _demand_supply(counts: np.ndarray, layout: CellLayout, params: CTMParams, dt: float):
    """Demand (veh/step) and supply (veh/step) per cell for the whole batch."""
    n = np.asarray(counts, dtype=np.float64)
    rho = n / (layout.length_m[None, :] * layout.lanes[None, :])  # veh/m/lane
    rho_km = rho * 1000.0
    v_free = np.array([params.fd(r).v_free_mps for r in layout.role], dtype=np.float64)
    cap = np.array([params.fd(r).capacity_mps_per_lane for r in layout.role],
                   dtype=np.float64)
    jam = np.array([params.fd(r).jam_density_vehpkmlane for r in layout.role],
                   dtype=np.float64)
    w = np.array([params.fd(r).w_mps for r in layout.role], dtype=np.float64)
    lanes = layout.lanes[None, :]
    demand = np.minimum(v_free[None, :] * rho, cap[None, :]) * lanes * dt
    supply = np.minimum(cap[None, :], w[None, :] * np.maximum(jam[None, :] - rho_km, 0.0)
                        ) * lanes * dt
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(rho > 1e-12,
                         np.minimum(v_free[None, :],
                                    w[None, :] * np.maximum(jam[None, :] - rho_km, 0.0)
                                    / np.maximum(rho_km, 1e-9)),
                         v_free[None, :])
    return demand, supply, speed


def _step_numpy(n: np.ndarray, layout: CellLayout, params: CTMParams,
                inflow_main: float, inflow_ramp: float, *, smooth: bool, tau: float):
    dt = params.dt_s
    demand, supply, _ = _demand_supply(n[None, :], layout, params, dt)
    demand, supply = demand[0], supply[0]

    def link(d_from, s_to):
        return softmin(demand[d_from], supply[s_to], tau) if smooth \
            else np.minimum(demand[d_from], supply[s_to])

    y_main = link(layout.main_idx[:-1], layout.main_idx[1:])
    y_ramp = link(layout.ramp_idx[:-1], layout.ramp_idx[1:])
    y_down = link(layout.down_idx[:-1], layout.down_idx[1:])

    rho_main_last = n[layout.main_idx[-1]] / (layout.length_m[layout.main_idx[-1]]
                                              * layout.lanes[layout.main_idx[-1]])
    rho_ramp_last = n[layout.ramp_idx[-1]] / (layout.length_m[layout.ramp_idx[-1]]
                                              * layout.lanes[layout.ramp_idx[-1]])
    fd_main = params.fd(ROLE_MAIN)
    fd_ramp = params.fd(ROLE_RAMP)
    queue = bool(rho_main_last * 1000.0 > fd_main.rho_crit_vehpkmlane
                 or rho_ramp_last * 1000.0 > fd_ramp.rho_crit_vehpkmlane)
    drop = params.merge_capacity_drop if queue else 1.0
    merge_cap = params.fd(ROLE_MERGE).capacity_mps_per_lane * LANES_MERGE * dt * drop
    s_merge = softmin(supply[layout.merge_idx], merge_cap, tau) if smooth \
        else min(supply[layout.merge_idx], merge_cap)
    y_m2, y_r2 = merge_allocation(demand[layout.main_idx[-1]], demand[layout.ramp_idx[-1]],
                                  s_merge, p_main=params.merge_priority_main,
                                  p_ramp=params.merge_priority_ramp, smooth=smooth, tau=tau)
    y_merge_out = softmin(demand[layout.merge_idx], supply[layout.down_idx[0]], tau) \
        if smooth else np.minimum(demand[layout.merge_idx], supply[layout.down_idx[0]])

    in_main = softmin(inflow_main, supply[layout.main_idx[0]], tau) if smooth \
        else min(float(inflow_main), supply[layout.main_idx[0]])
    in_ramp = softmin(inflow_ramp, supply[layout.ramp_idx[0]], tau) if smooth \
        else min(float(inflow_ramp), supply[layout.ramp_idx[0]])
    out = demand[layout.down_idx[-1]]

    n = n.copy()
    n[layout.main_idx[0]] += in_main - y_main[0]
    if layout.main_idx.shape[0] > 1:
        n[layout.main_idx[1:-1]] += y_main[:-1] - y_main[1:]
    n[layout.main_idx[-1]] += y_main[-1] - y_m2
    n[layout.ramp_idx[0]] += in_ramp - y_ramp[0]
    if layout.ramp_idx.shape[0] > 1:
        n[layout.ramp_idx[1:-1]] += y_ramp[:-1] - y_ramp[1:]
    n[layout.ramp_idx[-1]] += y_ramp[-1] - y_r2
    n[layout.merge_idx] += y_m2 + y_r2 - y_merge_out
    n[layout.down_idx[0]] += y_merge_out - y_down[0]
    if layout.down_idx.shape[0] > 1:
        n[layout.down_idx[1:-1]] += y_down[:-1] - y_down[1:]
    n[layout.down_idx[-1]] += y_down[-1] - out
    np.maximum(n, 0.0, out=n)
    return n, float(y_merge_out), float(in_main), float(in_ramp), float(out)


def run_numpy(initial_counts: np.ndarray, layout: CellLayout, params: CTMParams,
              counter, *, steps: int, smooth: bool = False, tau: float = 1e-3,
              t0_s: float = 0.0) -> CTMRun:
    """Roll the CTM forward from t0 counts with counter-driven boundary inflow."""
    counts0 = np.asarray(initial_counts, dtype=np.float64).reshape(-1)
    if counts0.shape[0] != layout.n_cells:
        raise ValueError(f"initial_counts must have {layout.n_cells} cells")
    rates = counter_rates(counter, params.counter_window_bins)
    n = counts0.copy()
    nsteps = int(steps)
    hist = np.zeros((nsteps + 1, layout.n_cells), dtype=np.float64)
    speeds = np.zeros((nsteps + 1, layout.n_cells), dtype=np.float64)
    merge_out = np.zeros(nsteps, dtype=np.float64)
    main_in = np.zeros(nsteps, dtype=np.float64)
    ramp_in = np.zeros(nsteps, dtype=np.float64)
    out_flow = np.zeros(nsteps, dtype=np.float64)

    def record(t):
        hist[t] = n
        _, _, sp = _demand_supply(n[None, :], layout, params, params.dt_s)
        speeds[t] = sp[0]

    record(0)
    for t in range(nsteps):
        n, mo, mi, ri, out = _step_numpy(n, layout, params, rates[0] * params.dt_s,
                                         rates[1] * params.dt_s, smooth=smooth, tau=tau)
        merge_out[t], main_in[t], ramp_in[t], out_flow[t] = mo, mi, ri, out
        record(t + 1)
    return CTMRun(counts=hist, speeds=speeds, merge_outflow=merge_out,
                  main_inflow=main_in, ramp_inflow=ramp_in, outflow=out_flow,
                  dt_s=params.dt_s, t0_s=t0_s, layout=layout, params=params)


# --------------------------------------------------------------------------- #
# batched torch simulation (smooth, for future gradient coupling)
# --------------------------------------------------------------------------- #
class TorchCTM:
    """Batched torch CTM; ``tau -> 0`` matches :func:`run_numpy`."""

    def __init__(self, params: Optional[CTMParams] = None, layout: Optional[CellLayout] = None):
        import torch

        self.params = params or CTMParams()
        self.layout = layout or build_layout(self.params.cell_length_m)
        self.torch = torch
        self._register()

    def _register(self) -> None:
        torch = self.torch
        lay, p = self.layout, self.params
        self.dt = float(p.dt_s)
        self.v_free = torch.tensor([p.fd(r).v_free_mps for r in lay.role], dtype=torch.float64)
        self.cap = torch.tensor([p.fd(r).capacity_mps_per_lane for r in lay.role],
                                dtype=torch.float64)
        self.jam = torch.tensor([p.fd(r).jam_density_vehpkmlane for r in lay.role],
                                dtype=torch.float64)
        self.w = torch.tensor([p.fd(r).w_mps for r in lay.role], dtype=torch.float64)
        self.lanes = torch.tensor(lay.lanes, dtype=torch.float64)
        self.length = torch.tensor(lay.length_m, dtype=torch.float64)
        self.main_idx = torch.tensor(lay.main_idx, dtype=torch.long)
        self.ramp_idx = torch.tensor(lay.ramp_idx, dtype=torch.long)
        self.down_idx = torch.tensor(lay.down_idx, dtype=torch.long)
        self.merge_idx = int(lay.merge_idx)
        self.merge_cap = (p.fd(ROLE_MERGE).capacity_mps_per_lane * LANES_MERGE * self.dt)
        self.frac = float(p.merge_priority_main) / max(
            1e-9, float(p.merge_priority_main) + float(p.merge_priority_ramp))
        self.drop = float(p.merge_capacity_drop)

    # soft primitives
    def _softmin(self, a, b, tau):
        torch = self.torch
        t = max(float(tau), 1e-9)
        b = b if torch.is_tensor(b) else torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return -t * torch.logaddexp(-a / t, -b / t)

    def _softmax(self, a, b, tau):
        torch = self.torch
        t = max(float(tau), 1e-9)
        b = b if torch.is_tensor(b) else torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return t * torch.logaddexp(a / t, b / t)

    def _demand_supply(self, n):
        torch = self.torch
        rho = n / (self.length[None, :] * self.lanes[None, :])
        rho_km = rho * 1000.0
        demand = torch.minimum(self.v_free[None, :] * rho, self.cap[None, :]) \
            * self.lanes[None, :] * self.dt
        supply = torch.minimum(self.cap[None, :],
                               self.w[None, :] * torch.clamp(self.jam[None, :] - rho_km, min=0.0)
                               ) * self.lanes[None, :] * self.dt
        return demand, supply

    def run(self, n0, counter, *, steps: int, tau: float = 1e-3, t0_s: float = 0.0):
        """``n0``: ``[B, C]`` -> counts ``[B, T+1, C]``, speeds, boundary flows."""
        torch = self.torch
        n = torch.as_tensor(n0, dtype=torch.float64)
        if n.dim() == 1:
            n = n[None, :]
        rates = counter_rates(np.asarray(counter, dtype=np.float64),
                              self.params.counter_window_bins)
        inflow = torch.tensor(rates * self.dt, dtype=torch.float64)  # [2]
        batch = n.shape[0]
        hist = [n]
        out = []
        dom_out = []
        mi = []
        ri = []
        for _ in range(int(steps)):
            demand, supply = self._demand_supply(n)
            y_main = self._softmin(demand[:, self.main_idx[:-1]],
                                   supply[:, self.main_idx[1:]], tau)
            y_ramp = self._softmin(demand[:, self.ramp_idx[:-1]],
                                   supply[:, self.ramp_idx[1:]], tau)
            y_down = self._softmin(demand[:, self.down_idx[:-1]],
                                   supply[:, self.down_idx[1:]], tau)
            rho_ml = n[:, self.main_idx[-1]] / (self.length[self.main_idx[-1]]
                                                * self.lanes[self.main_idx[-1]])
            rho_rl = n[:, self.ramp_idx[-1]] / (self.length[self.ramp_idx[-1]]
                                                * self.lanes[self.ramp_idx[-1]])
            q_main = rho_ml * 1000.0 > self.params.up.rho_crit_vehpkmlane
            q_ramp = rho_rl * 1000.0 > self.params.ramp.rho_crit_vehpkmlane
            drop = torch.where(q_main | q_ramp,
                               torch.full_like(q_main.to(torch.float64), self.drop),
                               torch.ones_like(q_main.to(torch.float64)))
            s_merge = self._softmin(supply[:, self.merge_idx], self.merge_cap * drop, tau)
            d_main = demand[:, self.main_idx[-1]]
            d_ramp = demand[:, self.ramp_idx[-1]]
            floor = self._softmax(s_merge - d_ramp, self.frac * s_merge, tau)
            y_m2 = self._softmin(d_main, floor, tau)
            y_r2 = self._softmin(d_ramp, self._softmax(s_merge - y_m2, 0.0, tau), tau)
            y_merge_out = self._softmin(demand[:, self.merge_idx],
                                        supply[:, self.down_idx[0]], tau)
            in_main = self._softmin(inflow[0].expand(batch), supply[:, self.main_idx[0]], tau)
            in_ramp = self._softmin(inflow[1].expand(batch), supply[:, self.ramp_idx[0]], tau)
            free_out = demand[:, self.down_idx[-1]]

            new = n.clone()
            new[:, self.main_idx[0]] += in_main - y_main[:, 0]
            if self.main_idx.shape[0] > 1:
                new[:, self.main_idx[1:-1]] += y_main[:, :-1] - y_main[:, 1:]
            new[:, self.main_idx[-1]] += y_main[:, -1] - y_m2
            new[:, self.ramp_idx[0]] += in_ramp - y_ramp[:, 0]
            if self.ramp_idx.shape[0] > 1:
                new[:, self.ramp_idx[1:-1]] += y_ramp[:, :-1] - y_ramp[:, 1:]
            new[:, self.ramp_idx[-1]] += y_ramp[:, -1] - y_r2
            new[:, self.merge_idx] += y_m2 + y_r2 - y_merge_out
            new[:, self.down_idx[0]] += y_merge_out - y_down[:, 0]
            if self.down_idx.shape[0] > 1:
                new[:, self.down_idx[1:-1]] += y_down[:, :-1] - y_down[:, 1:]
            new[:, self.down_idx[-1]] += y_down[:, -1] - free_out
            n = torch.clamp(new, min=0.0)
            hist.append(n)
            out.append(y_merge_out)
            dom_out.append(free_out)
            mi.append(in_main)
            ri.append(in_ramp)
        counts = torch.stack(hist, dim=1)  # [B, T+1, C]
        speed = self._speeds(counts)
        return {
            "counts": counts,
            "speeds": speed,
            "merge_outflow": torch.stack(out, dim=1) if out else torch.zeros(batch, 0),
            "outflow": torch.stack(dom_out, dim=1) if dom_out else torch.zeros(batch, 0),
            "main_inflow": torch.stack(mi, dim=1) if mi else torch.zeros(batch, 0),
            "ramp_inflow": torch.stack(ri, dim=1) if ri else torch.zeros(batch, 0),
            "t0_s": t0_s,
        }

    def _speeds(self, counts):
        torch = self.torch
        rho = counts / (self.length[None, None, :] * self.lanes[None, None, :])
        rho_km = rho * 1000.0
        speed = torch.minimum(self.v_free[None, None, :],
                              self.w[None, None, :] * torch.clamp(self.jam[None, None, :] - rho_km,
                                                                  min=0.0)
                              / torch.clamp(rho_km, min=1e-9))
        speed = torch.where(rho > 1e-12, speed,
                            self.v_free[None, None, :].expand_as(rho))
        return speed


# --------------------------------------------------------------------------- #
# CTM step -> the same per-section, per-30 s bin traffic schema as wms_eval
# --------------------------------------------------------------------------- #
def ctm_traffic_metrics(run: CTMRun, *, bin_s: float = 30.0, free_speed_mps: float = 27.78,
                        section_lengths_m: Optional[dict] = None,
                        queue_speed_mps: float = 0.5) -> dict:
    """Traffic metrics from a :class:`CTMRun` in ``wms_eval.metrics`` schema.

    Mirrors ``wms_eval.metrics.traffic_metrics`` (30 s bins, section block with
    ``mean_speed``/``density``/``stopped_vehicles``/``merge_queue_length``/
    ``merge_flow``/``total_delay``) but reads cell counts/speeds instead of
    vehicles.  ``merge_queue_length`` and ``stopped_vehicles`` are documented
    cell-level proxies: a stopped vehicle occupies ``1/rho_jam`` of lane.
    """
    import math

    lengths = dict(DEFAULT_SECTION_LENGTHS_M)
    if section_lengths_m:
        lengths.update({k: float(v) for k, v in section_lengths_m.items()})
    lay = run.layout
    section_len_km = {"upstream": lengths["upstream"] / 1000.0,
                      "merge": lengths["merge"] / 1000.0,
                      "downstream": lengths["downstream"] / 1000.0}
    role_to_section = {ROLE_MAIN: SECTION_UPSTREAM, ROLE_RAMP: SECTION_UPSTREAM,
                       ROLE_MERGE: SECTION_MERGE, ROLE_DOWN: SECTION_DOWNSTREAM}
    cell_section = np.array([role_to_section[int(r)] for r in lay.role], dtype=np.int64)
    section_ids = {"upstream": SECTION_UPSTREAM, "merge": SECTION_MERGE,
                   "downstream": SECTION_DOWNSTREAM}
    # metres of lane occupied by one stopped vehicle on each chain
    spacing_of_role = {role: 1000.0 / max(1e-9, run.params.fd(role).jam_density_vehpkmlane)
                       for role in (ROLE_MAIN, ROLE_RAMP, ROLE_MERGE, ROLE_DOWN)}

    n_steps = run.counts.shape[0]
    dt = run.dt_s
    end = run.t0_s + (n_steps - 1) * dt
    n_bins = max(1, int(math.ceil(max(0.0, end - run.t0_s) / bin_s)))
    result = {}
    for idx in range(n_bins):
        lo, hi = run.t0_s + idx * bin_s, run.t0_s + (idx + 1) * bin_s
        steps = [t for t in range(n_steps) if lo <= run.t0_s + t * dt < hi]
        block = {}
        for name, sid in section_ids.items():
            mask = cell_section == sid
            counts = run.counts[:, mask]         # [T, Cm]
            speeds = run.speeds[:, mask]
            roles = lay.role[mask]
            spacing = np.array([spacing_of_role[int(r)] for r in roles])
            if steps:
                cnt = counts[steps]              # [S, Cm]
                spd = speeds[steps]
                tot = cnt.sum(axis=1)            # [S]
                mean_speed = (float(np.mean(
                    (cnt * spd).sum(axis=1) / np.maximum(tot, 1e-9)))
                    if float(tot.sum()) > 0 else None)
                density = float(np.mean(tot / max(section_len_km[name], 1e-9)))
                stopped = float(np.mean([cnt[s][spd[s] < queue_speed_mps].sum()
                                         for s in range(cnt.shape[0])]))
                delay = float(np.sum([
                    (cnt[s] * np.clip(free_speed_mps - spd[s], 0.0, None)
                     / max(free_speed_mps, 1e-9) * dt).sum()
                    for s in range(cnt.shape[0])]))
                queue = 0.0
                if name == "merge":
                    queue = float(np.mean([
                        (cnt[s][spd[s] < queue_speed_mps] *
                         spacing[spd[s] < queue_speed_mps]).sum()
                        for s in range(cnt.shape[0])]))
            else:
                mean_speed = density = stopped = delay = None
                queue = 0.0
            block[name] = {"mean_speed": mean_speed, "density": density,
                           "stopped_vehicles": stopped,
                           "merge_queue_length": queue if name == "merge" else 0.0,
                           "merge_flow": 0.0, "total_delay": delay}
        step_idx = [t for t in range(run.merge_outflow.shape[0])
                    if lo <= run.t0_s + t * dt < hi]
        flow = float(run.merge_outflow[step_idx].sum()) if step_idx else 0.0
        block["merge"]["merge_flow"] = flow / (bin_s / 60.0)
        result[str(idx)] = {"start_s": lo, "end_s": hi, "sections": block}
    return result


def traffic_error(ctm_traffic: dict, recorded_traffic: dict, *,
                  section: str, metric: str, bin_idx: int) -> Optional[float]:
    pred = ctm_traffic.get(str(bin_idx), {}).get("sections", {}).get(section, {}).get(metric)
    rec = recorded_traffic.get(str(bin_idx), {}).get("sections", {}).get(section, {}).get(metric)
    if pred is None or rec is None:
        return None
    return abs(float(pred) - float(rec))


__all__ = [
    "S_MAIN_START_M", "S_RAMP_START_M", "S_MERGE_START_M", "S_MERGE_END_M", "S_DOWN_END_M",
    "LANES_MAIN", "LANES_RAMP", "LANES_MERGE", "LANES_DOWN",
    "SECTION_UPSTREAM", "SECTION_MERGE", "SECTION_DOWNSTREAM", "SECTION_NAMES",
    "ROLE_MAIN", "ROLE_RAMP", "ROLE_MERGE", "ROLE_DOWN",
    "TriangularFD", "CTMParams", "default_params", "CellLayout", "build_layout",
    "cell_index_of", "hard_bin_counts", "soft_bin_counts", "counter_rates",
    "softmin", "softmax", "merge_allocation", "CTMRun", "run_numpy", "TorchCTM",
    "ctm_traffic_metrics", "traffic_error",
]
