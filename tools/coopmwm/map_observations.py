"""CAV-report observations and RSU aggregation for maps whose RSU source is 'cav_reports' (N1).

Project RSU rule (AGENTS.md; MAIN 2026-11-06 for wz3to2): the RSU aggregates only CAV-reported
observations -- timestamped CAV self-states plus neighbours a CAV detects locally -- with receipt
age, missingness and coverage; never true penetration or global ground truth.

* ``cav_report_rows(env, desc)``: the per-tick observation rows (same schema as
  ``tools.leverprobe.backend.observations``: vid, s_m, v_mps, lane, is_cav; sorted by vid) =
  every CAV's self state + every vehicle within ``report_range_m`` longitudinally and
  ``report_lane_reach`` lanes of a reporting CAV. Replaces the RSU-detector rows on such maps;
  these rows feed coopnative.encode (29 per-CAV features AND the 12-feature RSU block).
* ``masked_field_builder``: wraps the frozen ``wms.control.rsu_field.build_rsu_field`` so the frozen
  MAPPO RSU field (unchanged cell layout) only counts CAV-detected vehicles.
* ``report_summary``: receipt age, per-lane coverage of the RSU block zones and missingness, for
  receipts/diagnostics (the 12-feature layout has no slot for them; layout is frozen for MAPPO u0).

The original map never calls this module (its RSU source is 'rsu_detector').
"""
from __future__ import annotations

import numpy as np


def detected_mask(s, lane, present, is_cav, report_range_m, lane_reach):
    """[K] bool: present vehicles reported by some CAV (self report or local detection).

    Detectors are present CAVs with a modelled lane (>= 0); a vehicle is detected when
    |ds| <= range and |dlane| <= reach from a detector. CAVs always report themselves."""
    s = np.asarray(s, np.float64)
    lane = np.asarray(lane, np.int64)
    present = np.asarray(present, bool)
    is_cav = np.asarray(is_cav, bool) & present
    detectors = np.flatnonzero(is_cav & (lane >= 0))
    out = is_cav.copy()
    if len(detectors):
        ds = np.abs(s[None, :]-s[detectors, None])
        dl = np.abs(lane[None, :]-lane[detectors, None])
        seen = ((ds <= report_range_m) & (dl <= lane_reach) & (lane[None, :] >= 0)).any(0)
        out |= seen & present
    return out


def _frame_arrays(frame):
    lane_eff = np.where(frame.junction, frame.lane_carry, frame.lane).astype(np.int64)
    return np.asarray(frame.s, np.float64), lane_eff, np.asarray(frame.present, bool), np.asarray(frame.is_cav, bool)


def cav_report_rows(env, desc):
    """CAV self states + locally detected neighbours (rows schema of leverprobe observations)."""
    f, ids = env.frame, env.builder.vehicle_ids
    s, lane, present, is_cav = _frame_arrays(f)
    keep = detected_mask(s, lane, present, is_cav, desc.rsu.report_range_m, desc.rsu.report_lane_reach)
    rows = [dict(vid=ids[int(f.vid_index[i])], s_m=float(f.s[i]), v_mps=float(f.v[i]),
                 lane=int(lane[i]), is_cav=bool(is_cav[i] and present[i]))
            for i in np.flatnonzero(keep)]
    return sorted(rows, key=lambda r: r['vid'])


def cav_report_rows_with_length(env, desc):
    """``tools.coopnative.plant.observations`` schema (adds the SUMO vType length)."""
    return [dict(r, length_m=float(env.api.vehicle.getLength(r['vid']))) for r in cav_report_rows(env, desc)]


def rsu_block(rows, zones):
    """The 12-feature RSU block of coopnative.encode for explicit per-lane zones (same arithmetic)."""
    out = []
    for lane, (lo, hi) in enumerate(zones):
        local = [r for r in rows if r['lane'] == lane and lo <= r['s_m'] <= hi]
        out.extend([len(local)/10, sum(r['v_mps'] for r in local)/max(1, len(local))/30,
                    sum(r['s_m'] for r in local)/max(1, len(local))/100,
                    sum(r['is_cav'] for r in local)/10])
    return out


def report_summary(env_or_frame, desc, now_s=None):
    """Receipt age, per-lane block-zone coverage and missingness of the CAV reports."""
    frame = getattr(env_or_frame, 'frame', env_or_frame)
    s, lane, present, is_cav = _frame_arrays(frame)
    R, reach = desc.rsu.report_range_m, desc.rsu.report_lane_reach
    reporters = np.flatnonzero(is_cav & present & (lane >= 0))
    lanes = []
    for wm_lane, (lo, hi) in enumerate(desc.rsu.block_zone_s_m):
        spans = sorted((max(lo, s[i]-R), min(hi, s[i]+R)) for i in reporters
                       if abs(int(lane[i])-wm_lane) <= reach and s[i]+R >= lo and s[i]-R <= hi)
        covered, end = 0., lo
        for a, b in spans:
            a = max(a, end)
            if b > a:
                covered += b-a
                end = b
        lanes.append(dict(lane=wm_lane, zone_s_m=[lo, hi], coverage=covered/(hi-lo), missing=covered == 0.,
                          reporters=sum(1 for i in reporters if abs(int(lane[i])-wm_lane) <= reach
                                        and s[i]+R >= lo and s[i]-R <= hi)))
    return dict(time_s=float(frame.time_s if now_s is None else now_s), source=desc.rsu.source,
                receipt_age_s=desc.rsu.receipt_age_s, max_receipt_age_s=desc.rsu.max_receipt_age_s,
                reporting_cavs=int(len(reporters)), lanes=lanes)


def masked_field_builder(original, desc):
    """Frozen MAPPO RSU field over CAV-detected vehicles only (cell layout unchanged)."""
    R, reach = float(desc.rsu.report_range_m), int(desc.rsu.report_lane_reach)

    def build_rsu_field(state):
        import torch
        s = state['s_m']
        lane = state['lane'].long()
        present = state.get('present', torch.ones_like(s, dtype=torch.bool)).bool()
        is_cav = state.get('is_cav', torch.zeros_like(present)).bool() & present
        detectors = is_cav & (lane >= 0)
        ds = (s[:, None, :]-s[:, :, None]).abs()                    # [B, detector, target]
        dl = (lane[:, None, :]-lane[:, :, None]).abs()
        seen = (detectors[:, :, None] & (ds <= R) & (dl <= reach) & (lane[:, None, :] >= 0)).any(1)
        return original({**state, 'present': present & (seen | is_cav)})

    build_rsu_field.map_name = desc.name
    build_rsu_field.original = original
    return build_rsu_field
