"""WM-43 static scenario validator: pure python, zero SUMO.

Validates the generated fixture inputs BEFORE they can reach SUMO so the
WM-43 r1 failure class (a vehicle referencing an undefined vType —
'Error: The vehicle type 'wm43_cav' ... is not known') can never occur
again: `validate_scenario(scenario_dir)` parses the generated
``scenario.sumocfg``, the route file it references and the referenced
network (``.gz`` supported) and returns a list of problem strings (empty
means valid).  It runs in the preflight for every planned run, in the run
child immediately after the scenario build (before any SUMO start), and in
the package tests — including against the actual failed r1 evidence.

Rule list (registered):
  V1  every file the sumocfg declares (net-file, route-files,
      additional-files, gui-settings-file) resolves on disk (absolute or
      relative to the sumocfg);
  V2  the network parses (gzip-aware) and exposes its edges/lanes;
  V3  every route edge exists in the network;
  V4  every vehicle's type exists as a declared vType (the SUMO default
      type is NOT accepted for this fixture - every roster vehicle must be
      explicitly typed);
  V5  the fixture vType ``wm43_cav`` exists exactly once, carries
      ``carFollowModel`` and ``speedDev="0"`` (the frozen determinism
      contract the runner's receipts rely on);
  V6  route ids unique; vehicle ids unique and disjoint from routes and
      vTypes (no duplicate ids anywhere);
  V7  every vehicle's route exists and has a nonempty edge list;
  V8  departLane is an integer index < lane count of the route's first
      edge (the lanes the wall assembly spans);
  V9  departPos lies within [0, length] of the depart lane;
  V10 numeric departSpeed does not exceed the depart lane's speed limit;
  V11 scenario_manifest.json agrees with the route file: same vehicle id
      set, same lane indices, departPos/departSpeed equal within 1e-6, and
      every roster lane index is inside its edge lane count.
"""
from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path

FIXTURE_VTYPE = "wm43_cav"
INPUT_TAGS = ("net-file", "route-files", "additional-files", "gui-settings-file")
TOLERANCE = 1e-6


def _parse(path: Path) -> ET.Element:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return ET.parse(stream).getroot()
    return ET.parse(path).getroot()


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base / candidate


def validate_scenario(scenario_dir: Path) -> list[str]:
    """Static validation of a generated scenario directory; no SUMO."""
    scenario_dir = Path(scenario_dir)
    problems: list[str] = []

    config_path = scenario_dir / "scenario.sumocfg"
    routes_path = scenario_dir / "requests.rou.xml"
    for path, label in ((config_path, "scenario.sumocfg"),
                        (routes_path, "requests.rou.xml")):
        if not path.is_file():
            problems.append(f"V0: missing generated file {label}")
    if problems:
        return problems

    # --- V1: every declared input resolves -----------------------------
    config_root = _parse(config_path)
    declared: dict[str, Path] = {}
    for tag in INPUT_TAGS:
        node = config_root.find(f"./input/{tag}")
        if node is None:
            continue
        target = _resolve(scenario_dir, node.get("value", ""))
        declared[tag] = target
        if not target.is_file():
            problems.append(f"V1: sumocfg {tag} does not resolve: {target}")
    if "net-file" not in declared or "route-files" not in declared:
        problems.append("V1: sumocfg lacks net-file or route-files")
        return problems

    # --- V2: the network parses and exposes edges/lanes -----------------
    edges: dict[str, dict] = {}
    lanes: dict[str, dict] = {}
    try:
        net_root = _parse(declared["net-file"])
    except (OSError, ET.ParseError) as exc:
        problems.append(f"V2: network failed to parse: {exc!r}")
        return problems
    for edge in net_root.findall(".//edge"):
        edge_lanes = edge.findall("lane")
        if not edge_lanes:
            continue  # internal edges carry no lanes
        entries = []
        for index, lane in enumerate(edge_lanes):
            entry = {"id": lane.get("id"), "index": index,
                     "length": float(lane.get("length", "0")),
                     "speed": float(lane.get("speed", "0"))}
            entries.append(entry)
            lanes[entry["id"]] = {**entry, "edge": edge.get("id")}
        edges[edge.get("id")] = {"lanes": entries, "lane_count": len(entries)}

    # --- route file ------------------------------------------------------
    routes_root = _parse(routes_path)
    vtypes: dict[str, ET.Element] = {}
    for node in routes_root.findall("vType") + \
            routes_root.findall("vTypeDistribution/vType"):
        if node.get("id") in vtypes:
            problems.append(f"V6: duplicate vType id {node.get('id')!r}")
        vtypes[node.get("id")] = node
    route_defs: dict[str, list[str]] = {}
    for node in routes_root.findall("route"):
        route_id = node.get("id")
        if route_id in route_defs:
            problems.append(f"V6: duplicate route id {route_id!r}")
        route_defs[route_id] = [e for e in (node.get("edges", "").split())]
    vehicles = routes_root.findall("vehicle")

    # --- V5: the fixture vType contract ---------------------------------
    fixture_types = [v for v in vtypes if v == FIXTURE_VTYPE]
    if len(fixture_types) != 1:
        problems.append(f"V5: expected exactly one {FIXTURE_VTYPE!r} vType, "
                        f"found {len(fixture_types)}")
    elif vtypes[FIXTURE_VTYPE].get("carFollowModel") is None \
            or vtypes[FIXTURE_VTYPE].get("speedDev") != "0":
        problems.append("V5: wm43_cav must carry carFollowModel and speedDev='0' "
                        "(frozen determinism contract)")

    # --- V4/V6/V7/V8/V9/V10: per-vehicle checks --------------------------
    vehicle_ids: set[str] = set()
    for vehicle in vehicles:
        vehicle_id = vehicle.get("id")
        if vehicle_id in vehicle_ids or vehicle_id in route_defs \
                or vehicle_id in vtypes:
            problems.append(f"V6: duplicate id {vehicle_id!r}")
        vehicle_ids.add(vehicle_id)
        vtype = vehicle.get("type")
        if vtype is None or vtype not in vtypes:
            problems.append(
                f"V4: vehicle {vehicle_id!r} references undefined vType {vtype!r} "
                "(the WM-43 r1 failure: vehicles referenced 'wm43_cav' while the "
                "route file defined only the template 'normal_car')")
            continue
        route_id = vehicle.get("route")
        if route_id not in route_defs:
            problems.append(f"V7: vehicle {vehicle_id!r} references unknown "
                            f"route {route_id!r}")
            continue
        route_edges = route_defs[route_id]
        if not route_edges:
            problems.append(f"V7: route {route_id!r} has no edges")
            continue
        missing = [e for e in route_edges if e not in edges]
        if missing:
            problems.append(f"V3: route {route_id!r} uses edges missing from the "
                            f"network: {missing}")
            continue
        first_edge = edges[route_edges[0]]
        try:
            depart_lane = int(vehicle.get("departLane", "0"))
        except ValueError:
            problems.append(f"V8: vehicle {vehicle_id!r} departLane is not an "
                            "integer lane index")
            continue
        if not 0 <= depart_lane < first_edge["lane_count"]:
            problems.append(
                f"V8: vehicle {vehicle_id!r} departLane {depart_lane} outside "
                f"0..{first_edge['lane_count'] - 1} of edge {route_edges[0]!r}")
            continue
        lane = first_edge["lanes"][depart_lane]
        try:
            depart_pos = float(vehicle.get("departPos", "0"))
        except ValueError:
            problems.append(f"V9: vehicle {vehicle_id!r} departPos is not numeric")
            continue
        if not 0.0 <= depart_pos <= lane["length"]:
            problems.append(
                f"V9: vehicle {vehicle_id!r} departPos {depart_pos} outside "
                f"[0, {lane['length']}] of lane {lane['id']!r}")
        try:
            depart_speed = float(vehicle.get("departSpeed", "0"))
        except ValueError:
            problems.append(f"V10: vehicle {vehicle_id!r} departSpeed is not numeric")
            continue
        if depart_speed > lane["speed"]:
            problems.append(
                f"V10: vehicle {vehicle_id!r} departSpeed {depart_speed} exceeds "
                f"lane limit {lane['speed']} on {lane['id']!r}")

    # --- V11: manifest agreement ----------------------------------------
    manifest_path = scenario_dir / "scenario_manifest.json"
    if not manifest_path.is_file():
        problems.append("V11: scenario_manifest.json missing")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"V11: scenario_manifest.json unreadable: {exc!r}")
            manifest = None
        if manifest is not None:
            roster = manifest.get("roster", [])
            manifest_ids = sorted(v["vehicle_id"] for v in roster)
            if manifest_ids != sorted(vehicle_ids):
                problems.append(
                    f"V11: manifest roster {manifest_ids} != route vehicles "
                    f"{sorted(vehicle_ids)}")
            by_id = {v.get("id"): v for v in vehicles}
            for row in roster:
                vehicle = by_id.get(row["vehicle_id"])
                if vehicle is None:
                    continue
                if int(vehicle.get("departLane", "-1")) != int(row["lane_index"]):
                    problems.append(f"V11: {row['vehicle_id']} lane_index mismatch")
                if abs(float(vehicle.get("departPos", "nan"))
                       - float(row["depart_pos_m"])) > TOLERANCE:
                    problems.append(f"V11: {row['vehicle_id']} departPos mismatch")
                if abs(float(vehicle.get("departSpeed", "nan"))
                       - float(row["depart_speed_mps"])) > TOLERANCE:
                    problems.append(f"V11: {row['vehicle_id']} departSpeed mismatch")
            wall_ids = set(manifest.get("wall_members", []))
            roster_ids = {v["vehicle_id"] for v in roster}
            extra_ids = roster_ids - wall_ids
            first_edge_id = next(iter(route_defs.values()), [""])[0]
            lane_count = edges.get(first_edge_id, {}).get("lane_count", 0)
            for member in sorted(wall_ids | extra_ids):
                row = next((v for v in roster if v["vehicle_id"] == member), None)
                if row is not None and not 0 <= int(row["lane_index"]) < lane_count:
                    problems.append(
                        f"V11: roster member {member!r} lane "
                        f"{row['lane_index']} outside edge {first_edge_id!r}")
    return problems
