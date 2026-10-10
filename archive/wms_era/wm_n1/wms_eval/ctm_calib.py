"""Calibrate the CTM triangular FDs on TRAIN windows only.

Data: the PI-free training windows in ``windows_train_noPI.json`` (never the
unseen PI states), read from the frozen memmap packs via
``wms_eval.materialize.Pack`` (``mmap_mode='r'``).  For each sampled window the
recorded trajectory is binned into the four CTM sections and reduced to
per-section ``(density, flow)`` points (per lane); a triangular FD is then
least-squares fitted with a single knot, and the merge capacity drop is
estimated from observed congested-vs-free merge discharge.  Parameters carry a
block bootstrap uncertainty over the sampled windows.

Nothing here touches SUMO, a GPU, a world model, or the PI test states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import ctm

DT_S = 0.5
LANE_LEFT, LANE_RIGHT, LANE_ACCEL = 0, 1, 2
S_IDX, V_IDX = 0, 4
DEFAULT_SECTION_LANES = {"up": ctm.LANES_MAIN, "ramp": ctm.LANES_RAMP,
                         "merge": ctm.LANES_MERGE, "down": ctm.LANES_DOWN}
FALLBACK = {
    "v_free_mps": (10.0, 40.0), "capacity_vehph_per_lane": (900.0, 2600.0),
    "jam_density_vehpkmlane": (80.0, 220.0), "w_mps": (1.0, 15.0),
}


class CalibrationError(ValueError):
    """Calibration input or fit is invalid."""


def load_train_windows(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("windows", "states", "records", "list"):
            if isinstance(raw.get(key), list):
                return {"schema": raw.get("schema"), "corpus": raw.get("corpus", {}),
                        "windows": [dict(w) for w in raw[key]]}
    if isinstance(raw, list):
        return {"windows": [dict(w) for w in raw]}
    raise CalibrationError(f"{path} has no window list")


def subsample_windows(windows: Sequence[dict], target: int) -> list[dict]:
    """Deterministic, policy-stratified subsample of TRAIN windows."""
    groups: dict[str, list[dict]] = {}
    for window in windows:
        groups.setdefault(str(window.get("policy")), []).append(dict(window))
    for group in groups.values():
        group.sort(key=lambda w: (str(w.get("pack")), str(w.get("world")),
                                  float(w.get("t0_s", 0.0)), str(w.get("key"))))
    names = sorted(groups)
    out: list[dict] = []
    idx = 0
    while len(out) < target:
        progressed = False
        for name in names:
            group = groups[name]
            if idx < len(group):
                out.append(group[idx])
                progressed = True
                if len(out) >= target:
                    break
        if not progressed:
            break
        idx += 1
    return out


def _section_arrays(kin: np.ndarray, sel: np.ndarray, full_mask: np.ndarray,
                    cont: np.ndarray, s_bound: float):
    """(count, crossing) for one section over the sampled ticks.

    ``kin`` is ``[T, K, 7]``; ``sel`` is the section membership ``[T, K]``;
    ``full_mask`` is occupancy on any lane ``[T, K]``; ``cont`` is the
    ``target_mask`` continuity ``[T-1, K]`` (same vehicle on the next tick, so a
    reused slot is not tracked).  Crossings of the downstream boundary must not
    require the vehicle to be inside the section on both ticks (it leaves the
    section as it crosses), so the detector uses occupancy, not ``sel``.
    ``s_bound`` is the section's downstream boundary.
    """
    s = kin[:, :, S_IDX].astype(np.float64)
    present = sel.astype(bool) & full_mask.astype(bool)
    count = present.sum(axis=1)
    cont = cont[:-1]
    prev_s, now_s = s[:-1], s[1:]
    prev_p, now_p = full_mask[:-1], full_mask[1:]
    crossing = (prev_p & now_p & cont & (prev_s <= s_bound)
                & (now_s > s_bound)).sum(axis=1)
    return count, crossing


def cell_successors(layout: ctm.CellLayout) -> np.ndarray:
    """Downstream successor cell for every cell (``-1`` = free outflow)."""
    succ = np.full(layout.n_cells, -1, dtype=np.int64)
    for chain, tail_to in ((layout.main_idx, layout.merge_idx),
                           (layout.ramp_idx, layout.merge_idx),
                           (layout.down_idx, -1)):
        for j in range(chain.shape[0] - 1):
            succ[chain[j]] = chain[j + 1]
        succ[chain[-1]] = tail_to
    succ[layout.merge_idx] = int(layout.down_idx[0])
    return succ


def _window_cells(pack, run: dict, layout: Optional[ctm.CellLayout] = None, *,
                  sample_stride: int = 4,
                  bin_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin, per-cell (rho veh/km/lane, q veh/h/lane) for one window."""
    layout = layout or ctm.build_layout()
    succ = cell_successors(layout)
    kin_all = pack.array("veh_kin")
    mask_all = pack.array("veh_mask")
    cont_all = pack.array("target_mask")
    lo = int(run.get("tick_start", 0))
    hi = int(run.get("tick_end", lo + 1))
    lo = max(lo, 0)
    kin = np.asarray(kin_all[lo:hi], dtype=np.float32)[::sample_stride]
    mask = np.asarray(mask_all[lo:hi]).astype(bool)[::sample_stride]
    cont = np.asarray(cont_all[lo:hi]).astype(bool)[::sample_stride]
    dt = DT_S * sample_stride
    n_ticks = kin.shape[0]
    n_cells = layout.n_cells
    cell_idx = np.full((n_ticks, kin.shape[1]), -1, dtype=np.int64)
    for t in range(n_ticks):
        present = mask[t]
        if present.any():
            lane = kin[t, :, 1:4].argmax(axis=1)
            cell_idx[t, present] = ctm.cell_index_of(layout, kin[t, present, S_IDX],
                                                     lane[present])
    per_bin = max(2, int(round(bin_s / dt)))
    rho_bins, q_bins = [], []
    for t0 in range(0, max(1, n_ticks - 1), per_bin):
        t1 = min(t0 + per_bin, n_ticks)
        if t1 - t0 < 2:
            break
        span = t1 - t0
        cnt = np.zeros(n_cells, dtype=np.float64)
        for t in range(t0, t1):
            valid = cell_idx[t] >= 0
            cnt += np.bincount(cell_idx[t][valid], minlength=n_cells)
        cnt /= span
        trans = np.zeros(n_cells, dtype=np.float64)
        for t in range(t0, min(t1, n_ticks - 1)):
            prev = cell_idx[t]
            now = cell_idx[t + 1]
            valid = (prev >= 0) & mask[t] & cont[t]
            if not valid.any():
                continue
            p = prev[valid]
            is_succ = now[valid] == succ[p]
            np.add.at(trans, p[is_succ], 1.0)
        seconds = (t1 - t0) * dt
        rho_bins.append(cnt / (layout.length_m / 1000.0) / layout.lanes)
        q_bins.append(trans / seconds * 3600.0 / layout.lanes)
    if not rho_bins:
        return np.zeros((0, n_cells)), np.zeros((0, n_cells))
    return np.stack(rho_bins, axis=0), np.stack(q_bins, axis=0)


_SECTION_OF_ROLE = {ctm.ROLE_MAIN: "up", ctm.ROLE_RAMP: "ramp",
                    ctm.ROLE_MERGE: "merge", ctm.ROLE_DOWN: "down"}


def _window_points(pack, run: dict, *,
                   layout: Optional[ctm.CellLayout] = None, sample_stride: int = 4,
                   bin_s: float = 30.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-section cell-level (rho veh/km/lane, q veh/h/lane) for one window."""
    layout = layout or ctm.build_layout()
    rho_bins, q_bins = _window_cells(pack, run, layout, sample_stride=sample_stride,
                                     bin_s=bin_s)
    section_of_cell = np.array([_SECTION_OF_ROLE[int(r)] for r in layout.role])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("up", "ramp", "merge", "down"):
        mask = section_of_cell == name
        rho = rho_bins[:, mask].reshape(-1) if rho_bins.size else np.zeros(0)
        q = q_bins[:, mask].reshape(-1) if q_bins.size else np.zeros(0)
        out[name] = (rho, q)
    return out


_SECTION_LEN_M = {"up": ctm.S_MERGE_START_M - ctm.S_MAIN_START_M,
                  "ramp": ctm.S_MERGE_START_M - ctm.S_RAMP_START_M,
                  "merge": ctm.S_MERGE_END_M - ctm.S_MERGE_START_M,
                  "down": ctm.S_DOWN_END_M - ctm.S_MERGE_END_M}


def _section_len_km(name: str) -> float:
    return _SECTION_LEN_M[name] / 1000.0


@dataclass
class FDFit:
    v_free_mps: float
    capacity_vehph_per_lane: float
    jam_density_vehpkmlane: float
    w_mps: float
    rho_crit: float
    n_points: int
    sse: float
    source: str = "fitted"

    def to_fd(self) -> ctm.TriangularFD:
        return ctm.TriangularFD(v_free_mps=self.v_free_mps,
                                capacity_vehph_per_lane=self.capacity_vehph_per_lane,
                                jam_density_vehpkmlane=self.jam_density_vehpkmlane,
                                w_mps=self.w_mps)

    def to_dict(self) -> dict:
        return {"v_free_mps": self.v_free_mps,
                "capacity_vehph_per_lane": self.capacity_vehph_per_lane,
                "jam_density_vehpkmlane": self.jam_density_vehpkmlane,
                "w_mps": self.w_mps, "rho_crit_vehpkmlane": self.rho_crit,
                "n_points": int(self.n_points), "sse": float(self.sse),
                "source": self.source}


def fit_triangular_fd(rho, q, *, min_points: int = 30) -> Optional[FDFit]:
    """Least-squares triangular FD with a single free/congested knot.

    Free branch ``q = v*rho`` (through the origin); congested branch
    ``q = a - w*rho`` (so ``rho_jam = a/w``, ``rho_crit = a/(v+w)``).  The knot
    is chosen over candidate sample indices to minimise the total SSE.
    """
    rho = np.asarray(rho, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    keep = np.isfinite(rho) & np.isfinite(q) & (rho >= 0.0) & (q >= 0.0)
    rho, q = rho[keep], q[keep]
    if rho.size < min_points:
        return None
    order = np.argsort(rho)
    rho, q = rho[order], q[order]
    n = rho.size
    candidates = range(max(5, n // 20), n - 5, max(1, n // 60))
    best = None
    for k in candidates:
        rl, ql = rho[:k], q[:k]
        rr, qr = rho[k:], q[k:]
        if rl.size < 3 or rr.size < 3:
            continue
        if rl.std() < 1e-6 or rr.std() < 1e-6:
            continue
        v = float((rl * ql).sum() / max((rl * rl).sum(), 1e-9))
        if not (FALLBACK["v_free_mps"][0] <= v <= FALLBACK["v_free_mps"][1]):
            continue
        A = np.stack([np.ones_like(rr), rr], axis=1)
        coef, *_ = np.linalg.lstsq(A, qr, rcond=None)
        a, slope = float(coef[0]), float(coef[1])
        w = -slope
        if not (FALLBACK["w_mps"][0] <= w <= FALLBACK["w_mps"][1]):
            continue
        rho_jam = a / w
        if not (FALLBACK["jam_density_vehpkmlane"][0] <= rho_jam
                <= FALLBACK["jam_density_vehpkmlane"][1]):
            continue
        rho_crit = a / (v + w)
        sse = float(((ql - v * rl) ** 2).sum() + ((qr - (a - w * rr)) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"v_free_mps": v, "w_mps": w, "rho_jam": rho_jam,
                    "rho_crit": rho_crit, "sse": sse, "k": k}
    if best is None:
        return None
    capacity = best["v_free_mps"] * best["rho_crit"]
    return FDFit(v_free_mps=best["v_free_mps"], capacity_vehph_per_lane=capacity,
                 jam_density_vehpkmlane=best["rho_jam"], w_mps=best["w_mps"],
                 rho_crit=best["rho_crit"], n_points=int(rho.size), sse=best["sse"])


def fit_envelope_fd(rho, q) -> Optional[FDFit]:
    """Robust triangular FD from the empirical envelope (fallback estimator).

    Free speed from the fastest ``q/rho`` points, capacity from the upper flow
    percentile, jam density from the upper density percentile, and the backward
    wave ``w = capacity / (rho_jam - rho_crit)``.  All values are clipped to the
    physical ranges in :data:`FALLBACK`; this is used when the scatter has no
    resolvable congested branch for the least-squares knot fit.
    """
    rho = np.asarray(rho, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    keep = np.isfinite(rho) & np.isfinite(q) & (rho >= 0.0) & (q >= 0.0)
    rho, q = rho[keep], q[keep]
    if rho.size < 10 or q.max() <= 0:
        return None
    speed_kmh = np.where(rho > 1.0, q / np.maximum(rho, 1e-9), 0.0)
    v_kmh = float(np.percentile(speed_kmh[speed_kmh > 0], 90)) if (speed_kmh > 0).any() \
        else ctm.NOMINAL_V_FREE_MPS * 3.6
    v = float(np.clip(v_kmh / 3.6, *FALLBACK["v_free_mps"]))
    cap = float(np.clip(np.percentile(q, 98), *FALLBACK["capacity_vehph_per_lane"]))
    rho_crit = cap / max(v * 3.6, 1e-9)
    # a 100 m cell is rarely observed fully jammed, so the density upper percentile
    # under-represents the true jam; anchor it to the physical range around 130
    # veh/km/lane and derive the backward wave from capacity.
    rho_jam = float(np.clip(np.percentile(rho, 99.5),
                            0.90 * ctm.NOMINAL_JAM_DENSITY_PER_LANE,
                            1.30 * ctm.NOMINAL_JAM_DENSITY_PER_LANE))
    if rho_jam <= 1.2 * rho_crit:
        rho_jam = min(1.3 * ctm.NOMINAL_JAM_DENSITY_PER_LANE, 3.0 * rho_crit)
    if rho_jam <= rho_crit:
        rho_jam = rho_crit * 1.5
    w_mps = float(np.clip((cap / max(rho_jam - rho_crit, 1e-9)) / 3.6, *FALLBACK["w_mps"]))
    return FDFit(v_free_mps=v, capacity_vehph_per_lane=cap,
                 jam_density_vehpkmlane=rho_jam, w_mps=w_mps, rho_crit=rho_crit,
                 n_points=int(rho.size), sse=float("nan"), source="envelope")


def fit_section_fd(rho, q, *, min_points: int = 30) -> Optional[FDFit]:
    """Least-squares knot fit if resolvable, else the robust envelope fit."""
    fit = fit_triangular_fd(rho, q, min_points=min_points)
    if fit is not None:
        return fit
    fit = fit_envelope_fd(rho, q)
    if fit is not None and fit.n_points < min_points:
        return None
    return fit


def _merge_via_windows(pack_by_name, windows, *, sample_stride: int = 4,
                       sample_bin_s: float = 30.0) -> dict:
    """Estimate merge discharge drop from observed congested vs free discharge."""
    layout = ctm.build_layout()
    section_of_cell = np.array([_SECTION_OF_ROLE[int(r)] for r in layout.role])
    merge_mask = section_of_cell == "merge"
    # queue indicator: density of the ~300 m of mainline immediately upstream of
    # the merge (a whole-section mean dilutes a local queue)
    near_mask = (section_of_cell == "up") & (layout.s_end >= -300.0)
    congested, free = [], []
    for window in windows:
        pack = pack_by_name.get(str(window.get("pack")))
        if pack is None:
            continue
        run = pack.runs.get(str(window.get("key")))
        if run is None:
            continue
        rho_bins, q_bins = _window_cells(pack, run, layout, sample_stride=sample_stride,
                                         bin_s=sample_bin_s)
        if rho_bins.size == 0:
            continue
        up_rho = rho_bins[:, near_mask].mean(axis=1)     # near-merge density per bin
        merge_q = q_bins[:, merge_mask].mean(axis=1)     # merge discharge per bin
        jam = ctm.NOMINAL_JAM_DENSITY_PER_LANE
        hi = up_rho >= 0.50 * jam
        lo = up_rho <= 0.15 * jam
        congested.extend(merge_q[hi].tolist())
        free.extend(merge_q[lo].tolist())
    free = np.asarray(free, np.float64)
    congested = np.asarray(congested, np.float64)
    if free.size < 20 or congested.size < 10:
        return {"merge_capacity_drop": 0.9, "source": "default",
                "n_free": int(free.size), "n_congested": int(congested.size)}
    cap = float(np.percentile(free, 90))
    discharge = float(np.median(congested))
    if cap <= 0:
        return {"merge_capacity_drop": 0.9, "source": "default",
                "n_free": int(free.size), "n_congested": int(congested.size)}
    drop = float(np.clip(discharge / cap, 0.4, 1.0))
    return {"merge_capacity_drop": drop, "source": "observed_discharge",
            "n_free": int(free.size), "n_congested": int(congested.size),
            "free_q_p90_vehph": cap, "congested_q_median_vehph": discharge}


def calibrate(train_windows_path, pack_roots: dict, *, target_windows: int = 180,
              n_boot: int = 200, seed: int = 0, cell_length_m: float = 100.0) -> dict:
    """Fit the four section FDs + merge drop from TRAIN windows.

    ``pack_roots`` maps pack name -> root directory.  Returns
    ``{"params": CTMParams, "report": {...}}``.
    """
    from .materialize import Pack

    meta = load_train_windows(train_windows_path)
    windows = subsample_windows(meta["windows"], target_windows)
    packs = {name: Pack(name, Path(root)) for name, root in pack_roots.items()}
    layout = ctm.build_layout(cell_length_m)

    section_points: dict[str, list] = {k: [] for k in ("up", "ramp", "merge", "down")}
    window_ids: dict[str, list] = {k: [] for k in section_points}
    used = 0
    for wid, window in enumerate(windows):
        pack = packs.get(str(window.get("pack")))
        if pack is None:
            continue
        run = pack.runs.get(str(window.get("key")))
        if run is None:
            continue
        used += 1
        pts = _window_points(pack, run, layout=layout)
        for name, (rho, q) in pts.items():
            section_points[name].append(np.stack([rho, q], axis=1))
            window_ids[name].append(np.full(rho.shape[0], wid, dtype=np.int64))

    fits: dict[str, FDFit] = {}
    uncertainties: dict[str, dict] = {}
    for name in ("up", "ramp", "down"):
        if not section_points[name]:
            continue
        points = np.concatenate(section_points[name], axis=0)
        fit = fit_section_fd(points[:, 0], points[:, 1])
        if fit is None:
            continue
        fits[name] = fit
        uncertainties[name] = _bootstrap(section_points[name], window_ids[name],
                                         n_boot=n_boot, seed=seed)

    drop = _merge_via_windows(packs, windows)

    up = fits.get("up")
    if up is None:
        up = FDFit(ctm.NOMINAL_V_FREE_MPS, ctm.NOMINAL_CAPACITY_PER_LANE,
                   ctm.NOMINAL_JAM_DENSITY_PER_LANE, ctm.NOMINAL_BACKWAVE_MPS,
                   ctm.NOMINAL_CAPACITY_PER_LANE / (ctm.NOMINAL_V_FREE_MPS * 3.6),
                   0, float("nan"), source="nominal_fallback")
        fits["up"] = up
    ramp = fits.get("ramp", up)
    # the merge cell is short and chronically queued, so its own scatter cannot
    # resolve a free branch; use the mainline per-lane FD (same lane behaviour)
    # with the merge lane count and the calibrated capacity drop instead.
    merge = FDFit(up.v_free_mps, up.capacity_vehph_per_lane,
                  up.jam_density_vehpkmlane, up.w_mps, up.rho_crit,
                  up.n_points, up.sse, source="inherited_from_up")
    fits["merge"] = merge
    down = fits.get("down", up)

    params = ctm.CTMParams(up=up.to_fd(), ramp=ramp.to_fd() if ramp else up.to_fd(),
                           merge=merge.to_fd() if merge else up.to_fd(),
                           down=down.to_fd() if down else up.to_fd(),
                           merge_capacity_drop=float(drop["merge_capacity_drop"]),
                           cell_length_m=float(cell_length_m))
    report = {
        "n_train_windows_available": len(meta["windows"]),
        "n_windows_sampled": len(windows),
        "n_windows_used": used,
        "n_bootstrap": int(n_boot),
        "sections": {name: fits[name].to_dict() for name in fits},
        "section_uncertainty": uncertainties,
        "merge_drop": drop,
        "params": params.to_dict(),
    }
    return {"params": params, "report": report}


def _bootstrap(points_list, ids_list, *, n_boot: int, seed: int) -> dict:
    """Block bootstrap over windows: std of each fitted FD parameter."""
    rng = np.random.default_rng(seed)
    groups: dict[int, np.ndarray] = {}
    for pts, ids in zip(points_list, ids_list):
        for wid in np.unique(ids):
            groups.setdefault(int(wid), []).append(pts[ids == wid])
    keys = sorted(groups)
    if len(keys) < 3:
        return {"n_windows": len(keys), "v_free_mps_sd": None,
                "capacity_sd": None, "jam_sd": None, "w_sd": None}
    v, cap, jam, w = [], [], [], []
    for _ in range(int(n_boot)):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sample = np.concatenate([np.concatenate(groups[k]) for k in pick], axis=0)
        fit = fit_section_fd(sample[:, 0], sample[:, 1], min_points=20)
        if fit is None:
            continue
        v.append(fit.v_free_mps)
        cap.append(fit.capacity_vehph_per_lane)
        jam.append(fit.jam_density_vehpkmlane)
        w.append(fit.w_mps)
    def sd(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else None
    return {"n_windows": len(keys), "n_boot_ok": len(v),
            "v_free_mps_sd": sd(v), "capacity_sd": sd(cap),
            "jam_sd": sd(jam), "w_sd": sd(w)}


__all__ = ["CalibrationError", "load_train_windows", "subsample_windows",
           "fit_triangular_fd", "calibrate", "FDFit"]
