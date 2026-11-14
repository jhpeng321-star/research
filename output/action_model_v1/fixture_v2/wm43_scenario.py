"""WM-43 micro-roster scenario builder (writes only under the run directory).

The fixture does NOT use the stochastic demand machinery: each arm needs a
specific small designated roster at scripted positions/speeds, so the builder
writes an explicit `requests.rou.xml` + `scenario.sumocfg` under
``<run_dir>/scenario/`` and refuses to touch anything else (the preserved
network and the scenario template are read-only inputs).

Roster derivation is deterministic per (arm, seed):
``random.Random("wm43_wall_mechanics_v1:{arm}:{seed}")`` drives declared
uniform bounds that keep every registered check meaningful:

- wall.0 (lane 0) at 240-300 m, wall.1 (lane 1) 1-4 m ahead of it (initial
  span strictly inside the 6 m assembly tolerance), both at 19-21 m/s;
  A4/A5 (wall-v2.1 closure): wall.1 is 40-60 m ahead of wall.0 instead;
- A2 only: follow.0 (lane 0) 70-100 m behind wall.0 at 22-26 m/s (closing
  on the wall; the damping path must brake it before the rear-end region);
- A3 only: the injection member is wall.1 (always the front member).

The CAV vType copies the template's `normal_car` attributes (probability
dropped, ``speedDev="0"`` frozen for the same reason as
scenarios.build_scenario); the route copies the template `mainline_route`
edge sequence verbatim.  The sumocfg mirrors the registered scenario
processing flags (``time-to-teleport=-1``, ``collision.action=warn``).
"""
from __future__ import annotations

import hashlib
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from traffic_shepherd.config import AdapterConfig, load_config

from wm43_plan import (ARM_SPECS, LOADING, PENETRATION, ROLE, SEEDS_BY_ARM,
                       SEEDS_DERIVATION_PREFIX)


@dataclass(frozen=True)
class RosterVehicle:
    vehicle_id: str
    lane_index: int
    depart_pos_m: float
    depart_speed_mps: float
    is_cav: bool = True


def derive_roster(arm: str, seed: int) -> list[RosterVehicle]:
    """Deterministic per-(arm, seed) roster inside the declared bounds.

    A4 geometry bound (r3 F4 lesson): the A2 wall holds 8 m/s from ~t=6 to
    the planned release at t=300 s on the 2644.22 m pre-merge edge, so the
    roster start position is capped at 170 m (+4 m member offset): worst-
    case front-member absolute position at release is
    174 + ~70 m (assembly decel from <=21.9) + 8 m/s * ~294 s <= 2618 m,
    keeping >= 26 m of margin from the edge end, so no member can cross
    onto the merge edge during the registered hold (the cross-edge
    lane-local span comparison is the r3 false member_span_exceeded
    trigger; the dispatcher patch PROPOSAL fixes the comparison itself).
    The follower gap shrinks to 60-90 m so its start stays >= 30 m.
    """
    spec = ARM_SPECS[arm]
    rng = random.Random(f"{SEEDS_DERIVATION_PREFIX}:{arm}:{seed}")
    # A5 band (the rejected A4 120-170 m bound HID the pre-merge/merge
    # boundary; this one GUARANTEES the wall crosses it during the hold):
    # front member <= 304 m start + <= ~2440 m hold travel puts the crossing
    # at t ~= 288.7-293.5 s for all 5 A2 seeds (>= 6.5 s before the t=300
    # release), so the registered arm re-validates the F4 span fix under
    # real SUMO in every seed (see REGISTRATION.md Amendment A5).
    base_pos = 240.0 + rng.uniform(0.0, 60.0)
    if "wide_offset_m" in spec:
        # wall-v2.1 wide-pair arms (A4/A5): the front member wall.1 starts
        # 40-60 m ahead of wall.0 (beyond the 6 m assembly tolerance; inside
        # the 75 m feasibility cap and the 150 m abort span)
        lo, hi = spec["wide_offset_m"]
        member_b_offset = rng.uniform(lo, hi)
    else:
        member_b_offset = rng.uniform(1.0, 4.0)
    member_speed = 19.0 + rng.uniform(0.0, 2.0)
    roster = [
        RosterVehicle(spec["members"][0], 0, round(base_pos, 3), round(member_speed, 3)),
        RosterVehicle(spec["members"][1], 1, round(base_pos + member_b_offset, 3),
                      round(member_speed, 3)),
    ]
    if "follow.0" in spec["extra_cavs"]:
        roster.append(RosterVehicle(
            "follow.0", 0,
            round(base_pos - rng.uniform(70.0, 100.0), 3),
            round(22.0 + rng.uniform(0.0, 4.0), 3)))
    return roster


def _template_parts(config: AdapterConfig) -> tuple[dict, str]:
    """(normal_car attributes, mainline_route edges string) from the frozen template."""
    root = ET.parse(config.scenario_template_path).getroot()
    vtype = root.find("./vTypeDistribution/vType[@id='normal_car']")
    if vtype is None:
        raise ValueError("scenario template carries no normal_car vType")
    # Drop the template identity: the fixture assigns its own vType id
    # (wm43_cav).  Keeping the template 'id' here silently overwrote it in
    # the attribute merge and produced dangling type references (the
    # WM-43 r1 launch failure); the static validator now rejects that class.
    attrs = {k: v for k, v in vtype.attrib.items()
             if k not in ("probability", "id")}
    attrs["speedDev"] = "0"
    route = root.find("route[@id='mainline_route']")
    if route is None:
        raise ValueError("scenario template carries no mainline_route")
    return attrs, route.attrib["edges"]


def build_scenario(run_dir: Path, world: dict, config: AdapterConfig | None = None
                   ) -> tuple[Path, list[RosterVehicle], dict]:
    """Write the fixture scenario under ``run_dir/scenario``; never overwrite.

    Returns (sumocfg path, roster, manifest).  The roster's vehicles are all
    designated CAVs; ``world`` carries arm/seed/loading/penetration/role.
    """
    config = config if config is not None else load_config()
    arm, seed = world["arm"], int(world["seed"])
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown fixture arm {arm!r}")
    if seed not in set(SEEDS_BY_ARM[arm]):
        raise ValueError(f"seed {seed} is outside the registered WM-43B block for {arm}")
    roster = derive_roster(arm, seed)
    vtype_attrs, route_edges = _template_parts(config)
    scenario_dir = Path(run_dir) / "scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    route = ET.Element("routes")
    ET.SubElement(route, "vType", {"id": "wm43_cav", **vtype_attrs})
    ET.SubElement(route, "route", {"id": "mainline_route", "edges": route_edges})
    for vehicle in roster:
        ET.SubElement(route, "vehicle", {
            "id": vehicle.vehicle_id,
            "type": "wm43_cav",
            "route": "mainline_route",
            "depart": "0",
            "departLane": str(vehicle.lane_index),
            "departPos": str(vehicle.depart_pos_m),
            "departSpeed": str(vehicle.depart_speed_mps),
        })
    ET.indent(route)
    routes_text = ET.tostring(route, encoding="unicode") + "\n"

    scene = ET.Element("configuration")
    inputs = ET.SubElement(scene, "input")
    ET.SubElement(inputs, "net-file", {"value": str(config.network_path)})
    ET.SubElement(inputs, "route-files", {"value": "requests.rou.xml"})
    processing = ET.SubElement(scene, "processing")
    ET.SubElement(processing, "time-to-teleport", {"value": "-1"})
    ET.SubElement(processing, "collision.action", {"value": "warn"})
    ET.indent(scene)
    scene_text = ET.tostring(scene, encoding="unicode") + "\n"

    route_path = scenario_dir / "requests.rou.xml"
    scene_path = scenario_dir / "scenario.sumocfg"
    for path, value in ((route_path, routes_text), (scene_path, scene_text)):
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise ValueError(f"refusing to change an existing scenario file: {path}")
        path.write_text(value, encoding="utf-8")

    spec = ARM_SPECS[arm]
    manifest = {
        "schema": "wm43_fixture_scenario/1",
        "arm": arm,
        "seed": seed,
        "loading": LOADING,
        "penetration": PENETRATION,
        "role": ROLE,
        "roster_derivation": f"{SEEDS_DERIVATION_PREFIX}:{arm}:{seed}",
        "roster": [asdict(v) for v in roster],
        "wall_id": spec["wall_id"],
        "wall_members": list(spec["members"]),
        "wall_speed_mps": world["wall_speed_mps"],
        "release_at_s": spec["release_at_s"],
        "injection": dict(spec["injection"]) if spec["injection"] else None,
        "vtype_source_sha256": _sha(config.scenario_template_path),
        "routes_sha256": _sha(route_path),
        "sumocfg_sha256": _sha(scene_path),
        "network_sha256": _sha(config.network_path),
        "network_path": str(config.network_path),
    }
    manifest_path = scenario_dir / "scenario_manifest.json"
    if manifest_path.exists():
        raise ValueError(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return scene_path, roster, manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
