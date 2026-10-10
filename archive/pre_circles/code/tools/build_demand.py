"""Controlled mainline+ramp demand generator for the merge scenario.

Replaces randomTrips network-wide random demand with two explicit flows
(mainline + on-ramp) so that bottleneck inflow is directly controllable in
veh/h. Driver heterogeneity is declared inline as a SUMO vTypeDistribution,
so pure-SUMO baselines and TraCI-controlled runs share the exact same
driver mixture with zero code.
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

MAINLINE_EDGE = "1191554636"
MERGE_EDGE = "1487606179"
DOWNSTREAM_EDGE = "1487606178"
RAMP_EDGE = "11553048#1"

# Realistic driver types (Krauss car-following + LC2013 lane changing).
# speedFactor is relative to the 31.29 m/s mainline limit.
DRIVER_TYPES = {
    "aggressive_car": {
        "tau": "0.95",
        "sigma": "0.6",
        "speedFactor": "1.08",
        "accel": "3.0",
        "decel": "4.5",
        "minGap": "1.5",
        "lcCooperative": "0.3",
        "lcAssertive": "0.6",
        "impatience": "0.7",
    },
    "normal_car": {
        "tau": "1.1",
        "sigma": "0.4",
        "speedFactor": "1.0",
        "accel": "2.6",
        "decel": "4.5",
        "minGap": "2.5",
        "lcCooperative": "0.6",
        "impatience": "0.3",
    },
    "cautious_car": {
        "tau": "1.5",
        "sigma": "0.2",
        "speedFactor": "0.92",
        "accel": "2.0",
        "decel": "3.0",
        "minGap": "3.5",
        "lcCooperative": "1.0",
        "impatience": "0.0",
    },
}

# Mixture presets: [aggressive, normal, cautious]
MIXTURES = {
    "standard": [0.10, 0.70, 0.20],
    "aggressive_heavy": [0.30, 0.60, 0.10],
    "cautious_heavy": [0.05, 0.75, 0.20],
}

_TYPE_ORDER = ["aggressive_car", "normal_car", "cautious_car"]


def build_routes_xml(mainline_vph, ramp_vph, duration_s, mixture, seed):
    """Return rou.xml text with a vTypeDistribution and two flows.

    ``seed`` is currently unused: the generated rou.xml is deterministic
    (fixed flows + vTypeDistribution); all run-to-run randomness is driven by
    SUMO's own ``--seed`` at simulation time.
    """
    probs = MIXTURES[mixture]
    root = ET.Element("routes")
    dist = ET.SubElement(root, "vTypeDistribution", id="hdv_mix")
    for vtype_id, prob in zip(_TYPE_ORDER, probs):
        attrs = dict(DRIVER_TYPES[vtype_id])
        attrs["id"] = vtype_id
        attrs["probability"] = f"{prob:.2f}"
        ET.SubElement(dist, "vType", **attrs)
    ET.SubElement(
        root, "route", id="mainline_route", edges=f"{MAINLINE_EDGE} {MERGE_EDGE} {DOWNSTREAM_EDGE}"
    )
    ET.SubElement(
        root, "route", id="ramp_route", edges=f"{RAMP_EDGE} {MERGE_EDGE} {DOWNSTREAM_EDGE}"
    )
    for flow_id, route_id, vph in (
        ("mainline", "mainline_route", mainline_vph),
        ("ramp", "ramp_route", ramp_vph),
    ):
        ET.SubElement(
            root,
            "flow",
            id=flow_id,
            type="hdv_mix",
            route=route_id,
            begin="0",
            end=str(int(duration_s)),
            vehsPerHour=str(int(vph)),
            departLane="best",
            departSpeed="max",
        )
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="    ")


def write_demand(level, mainline_vph, ramp_vph, duration_s, mixture, out_dir):
    """Write osm.demand.<level>.rou.xml into out_dir; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"osm.demand.{level}.rou.xml"
    path.write_text(
        build_routes_xml(mainline_vph, ramp_vph, duration_s, mixture, seed=42), encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", required=True)
    p.add_argument("--mainline-vph", type=int, required=True)
    p.add_argument("--ramp-vph", type=int, required=True)
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--mixture", default="standard", choices=sorted(MIXTURES))
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    print(write_demand(a.level, a.mainline_vph, a.ramp_vph, a.duration, a.mixture, a.out_dir))
