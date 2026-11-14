"""WD-01 partA_v2g adapter: ONE reconstructor sample -> the engine table.

Reads ONLY the coming Part-A view fields (PARTA_V2_GENERAL_DESIGN.md
section 8):

- header: t_s, k, K, sample_weight;
- per-lane zone extent [s_min, s_max] (s_down convention);
- per vehicle: id, s_down_m (increasing downstream, = -d), lane_phys
  (0 main_left, 1 main_right, 2 accel), speed_mps, and the optional
  is_cav / length_m / acceleration_mps2;
- per-CAV controller block (action / wall fields), passed through.

A thin r0 mapping layer (wm62 SCHEMA_V1.md) accepts raw r0 records
(group g0..g3 + d, positive upstream) and derives s_down_m = -d and
lane_phys through a TESTED table; records still lacking s_down_m /
lane_phys after mapping are REJECTED.  StateSample.arrivals is IGNORED
(never refused).  No vType / driver type is read anywhere.

Conventions (design section 8, read carefully): the model-world s of a
vehicle is  s = OFFSETS[lane_phys] + s_down_m  with OFFSETS =
(2645.67, 2644.22, 378.66) and d = OFFSETS[lane] - s upstream-positive.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .engine import load_params

LANE_MAIN_LEFT = 0
LANE_MAIN_RIGHT = 1
LANE_ACCEL = 2

# r0 group -> lane_phys (SCHEMA_V1.md: g0/g1 = mainline by SUMO index,
# SUMO _0 is the rightmost lane -> main_right; g2 = the merge third lane;
# g3 = the ramp; both g2 and g3 are the accel chain, lane_phys 2).
R0_GROUP_TO_LANE_PHYS = {
    "g0": LANE_MAIN_RIGHT,
    "g1": LANE_MAIN_LEFT,
    "g2": LANE_ACCEL,
    "g3": LANE_ACCEL,
}
R0_RAMP_D_MAX = 378.66       # d of the ramp/merge boundary on the accel chain


class AdapterError(ValueError):
    """A recon record violates the Part-A input view."""


def r0_record_to_view(rec: Mapping) -> Mapping:
    """One raw r0 vehicle record -> Part-A view fields (s_down_m,
    lane_phys), or raise AdapterError.  NEVER uses r0's d downstream."""
    if "s_down_m" in rec and "lane_phys" in rec:
        return rec
    if "d" in rec and "group" in rec:
        g = str(rec["group"])
        if g not in R0_GROUP_TO_LANE_PHYS:
            raise AdapterError(
                f"unknown r0 group {g!r} (expected one of "
                f"{sorted(R0_GROUP_TO_LANE_PHYS)})")
        out = dict(rec)
        out["s_down_m"] = -float(rec["d"])
        out["lane_phys"] = R0_GROUP_TO_LANE_PHYS[g]
        return out
    raise AdapterError(
        "recon record lacks s_down_m/lane_phys and carries no r0 "
        "(group, d) to map them from; rejecting")


def view_to_model_s(lane_phys: int, s_down_m: float,
                    offsets: Sequence[float]) -> float:
    """s_down (increasing downstream, = -d) -> model-world s."""
    return float(offsets[int(lane_phys)]) + float(s_down_m)


def model_s_to_view_s_down(lane_phys: int, s: float,
                           offsets: Sequence[float]) -> float:
    return float(s) - float(offsets[int(lane_phys)])


class ReconSampleTable:
    """The mapped vehicle table of ONE recon sample."""

    def __init__(self, t_s: float, vehicles: Dict[str, dict],
                 zone_extent_s: Optional[Dict[int, Tuple[float, float]]],
                 cav_block: Dict[str, dict], k: int, K: int,
                 sample_weight: float, raw_header: Mapping):
        self.t_s = float(t_s)
        self.vehicles = vehicles      # {vid: {lane, s, v, a?, is_cav?, length_m?}}
        self.zone_extent_s = zone_extent_s   # {lane_phys: (smin, smax)} model s
        self.cav_block = cav_block
        self.k = int(k)
        self.K = int(K)
        self.sample_weight = float(sample_weight)
        self.raw_header = dict(raw_header)

    @property
    def n_tracked(self) -> int:
        return len(self.vehicles)


def parse_sample(hybrid: Mapping, params: Optional[Mapping] = None
                 ) -> ReconSampleTable:
    """One StateSample.hybrid (the Part-A view) -> ReconSampleTable.

    Required: vehicles with s_down_m + lane_phys (directly or via the r0
    mapping), t_s.  Optional: k/K/sample_weight header, per-lane zone
    extent, per-CAV controller block, per-vehicle is_cav/length_m/
    acceleration_mps2.  Arrivals are NOT part of the hybrid view
    (StateSample.arrivals is ignored upstream, never refused).
    """
    params = params or load_params()
    offsets = params["geometry"]["lane_offsets_m"]
    if not isinstance(hybrid, Mapping):
        raise AdapterError(
            f"hybrid must be a mapping (the Part-A view), got "
            f"{type(hybrid).__name__}")
    recs = hybrid.get("vehicles")
    if recs is None:
        raise AdapterError("the Part-A view needs a 'vehicles' mapping")
    t_s = hybrid.get("t_s", hybrid.get("header", {}).get("t_s"))
    if t_s is None:
        raise AdapterError("the Part-A view needs t_s")
    header = hybrid.get("header", {})
    vehicles: Dict[str, dict] = {}
    for vid, raw in recs.items():
        rec = r0_record_to_view(raw)
        lane_phys = int(rec["lane_phys"])
        if lane_phys not in (LANE_MAIN_LEFT, LANE_MAIN_RIGHT, LANE_ACCEL):
            raise AdapterError(
                f"vehicle {vid!r}: lane_phys {lane_phys} not in (0, 1, 2)")
        s = view_to_model_s(lane_phys, rec["s_down_m"], offsets)
        veh = {
            "lane": lane_phys,
            "s": s,
            "v": float(rec.get("speed_mps", rec.get("v", 0.0))),
        }
        if "acceleration_mps2" in rec:
            veh["a"] = float(rec["acceleration_mps2"])
        if "is_cav" in rec:
            veh["is_cav"] = bool(rec["is_cav"])
        if "length_m" in rec:
            veh["length_m"] = float(rec["length_m"])
        vehicles[str(vid)] = veh
    zone_extent_s = None
    ze = hybrid.get("zone_extent")
    if ze is not None:
        zone_extent_s = {}
        for li_s, rng in ze.items():
            li = int(li_s)
            lo, hi = float(rng[0]), float(rng[1])
            zone_extent_s[li] = (view_to_model_s(li, lo, offsets),
                                 view_to_model_s(li, hi, offsets))
    cav_block = dict(hybrid.get("cav_controller_block", {}) or {})
    return ReconSampleTable(
        t_s=float(t_s), vehicles=vehicles, zone_extent_s=zone_extent_s,
        cav_block=cav_block,
        k=int(header.get("k", hybrid.get("k", 0)) or 0),
        K=int(header.get("K", hybrid.get("K", 1)) or 1),
        sample_weight=float(header.get("sample_weight",
                                       hybrid.get("sample_weight", 1.0))),
        raw_header=dict(header),
    )
