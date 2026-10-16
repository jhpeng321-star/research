"""Exact explicit IDM fixture extracted from idm/damping_scenario.py.

Exogenous Poisson departures, driver types, fixed speedDev=0 and nested CAV
identity draws preserve the tested transfer fixture. Penetration changes only
which unchanged IDM vehicles receive commands. Piecewise demand definition is
versioned scenario logic, not an actor input or an alternative control config.
"""
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from traffic_shepherd.config import load_config
from traffic_shepherd.seeds import require_development_case


@dataclass(frozen=True)
class TripRequest:
    vehicle_id: str
    depart_s: float
    route: str
    driver_type: str
    is_cav: bool


def request_schedule(seed: int, mainline_vph: int, *, penetration: float, role: str = "exposed",
                     cav_vtype: str | None = None, population: str = "legacy3") -> list[TripRequest]:
    """``cav_vtype``: CAVs use this single vType (e.g. "normal_car") instead of the
    driver-type draw; HDV draws are unchanged.  None = legacy (CAVs keep their drawn driver type).

    ``population``: "legacy3" (default) = the three discrete vTypes above;
    "textbook" = one continuous-parameter vType per HDV (``tb.<vehicle_id>``, parameters from
    :func:`traffic_shepherd.c4.population.sample_driver`) and every CAV on ``cav_typical``.
    Departures and CAV identities do not depend on ``population``."""
    from traffic_shepherd.c4.population import (TEXTBOOK_CAV_VTYPE, check_population,
                                                hdv_vtype_id)
    check_population(population)
    textbook = population == "textbook"
    if textbook and cav_vtype not in (None, TEXTBOOK_CAV_VTYPE):
        raise ValueError(f"textbook population: CAVs use {TEXTBOOK_CAV_VTYPE!r}, got {cav_vtype!r}")
    require_development_case(seed, mainline_vph, role)
    if not 1900 <= mainline_vph <= 2800:
        raise ValueError("loading outside declared development family")
    if not 0 <= penetration <= 1:
        raise ValueError("penetration must be in [0, 1]")
    requests = []
    for stream in ("mainline", "ramp"):
        # Independent streams keep ramp/type/assignment draws independent of
        # changes to another stream's vehicle count.
        rng = random.Random(f"damping_requests_v1:{seed}:{stream}")
        rates = (
            [(0, 1200, mainline_vph)]
            + [
                (
                    1200 + 100 * i,
                    1300 + 100 * i,
                    round(mainline_vph + (1500 - mainline_vph) * (i + 1) / 3),
                )
                for i in range(3)
            ]
            + [(1500, 1800, 1500)]
        )
        if stream == "ramp":
            rates = [(0, 1800, 400)]
        ordinal = 0
        for begin, end, rate in rates:
            t = float(begin)
            while True:
                t += rng.expovariate(rate / 3600)
                if t >= end:
                    break
                identity = f"{stream}.{ordinal}"
                draw = random.Random(f"damping_type_v1:{seed}:{identity}").random()
                driver = (
                    "aggressive_car"
                    if draw < 0.1
                    else "normal_car"
                    if draw < 0.8
                    else "cautious_car"
                )
                cav = random.Random(f"damping_cav_v1:{seed}:{identity}").random() < penetration
                if cav and cav_vtype is not None:
                    driver = cav_vtype
                if textbook:
                    driver = TEXTBOOK_CAV_VTYPE if cav else hdv_vtype_id(identity)
                requests.append(TripRequest(identity, round(t, 6), stream + "_route", driver, cav))
                ordinal += 1
    return sorted(requests, key=lambda r: (r.depart_s, r.vehicle_id))


def build_scenario(
    root: Path, seed: int, mainline_vph: int, *, penetration: float, role: str = "exposed", config=None,
    cav_vtype: str | None = None, population: str = "legacy3",
) -> tuple[Path, list[TripRequest], dict]:
    config = config if config is not None else load_config()
    if (config.episode_seconds, config.demand_end_seconds, config.ramp_vph) != (2400, 1800, 400):
        raise ValueError("configuration differs from the tested explicit demand fixture")
    requests = request_schedule(seed, mainline_vph, role=role, penetration=penetration,
                                cav_vtype=cav_vtype, population=population)
    textbook = population == "textbook"
    template = config.scenario_template_path
    original = ET.parse(template).getroot()
    route = ET.Element("routes")
    truth_text = None
    if textbook:
        from traffic_shepherd.c4 import population as pop
        cav_params = dict(pop.TEXTBOOK_CAV_PARAMS)
        ET.SubElement(route, "vType", pop.vtype_attrs(pop.TEXTBOOK_CAV_VTYPE, cav_params))
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator=chr(10))
        writer.writerow(["vehicle_id", "driver_class", "vtype_id", "is_cav", *pop.PARAM_ORDER])
        for request in requests:
            if request.is_cav:
                params, cls = cav_params, pop.TEXTBOOK_CAV_VTYPE
            else:
                params, cls = pop.sample_driver(seed, request.vehicle_id), "textbook"
                ET.SubElement(route, "vType", pop.vtype_attrs(request.driver_type, params))
            writer.writerow([request.vehicle_id, cls, request.driver_type, int(request.is_cav),
                             *(repr(float(params[k])) for k in pop.PARAM_ORDER)])
        truth_text = buffer.getvalue()
    else:
        for vtype in original.findall("./vTypeDistribution/vType"):
            attrs = {k: v for k, v in vtype.attrib.items() if k != "probability"}
            # Freeze exact speed factors, avoiding extra random speed-factor draws
            # when policy-dependent insertion order changes.
            attrs["speedDev"] = "0"
            ET.SubElement(route, "vType", attrs)
    for element in original.findall("route"):
        ET.SubElement(route, "route", element.attrib)
    for request in requests:
        ET.SubElement(
            route,
            "vehicle",
            {
                "id": request.vehicle_id,
                "type": request.driver_type,
                "route": request.route,
                "depart": str(request.depart_s),
                "departLane": "best",
                "departSpeed": "max",
            },
        )
    ET.indent(route)
    text = ET.tostring(route, encoding="unicode")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    route_path = root / "requests.rou.xml"
    scene_path = root / "scenario.sumocfg"
    scene = ET.Element("configuration")
    inputs = ET.SubElement(scene, "input")
    ET.SubElement(inputs, "net-file", value=str(config.network_path))
    ET.SubElement(inputs, "route-files", value=route_path.name)
    processing = ET.SubElement(scene, "processing")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.indent(scene)
    scene_text = ET.tostring(scene, encoding="unicode")
    writes = [(route_path, text), (scene_path, scene_text)]
    if truth_text is not None:
        # EVALUATION-ONLY driver truth (wms.data.driver_truth hetdata_truth.csv format).
        writes.append((root / "hetdata_truth.csv", truth_text))
    for path, value in writes:
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise ValueError("refusing to change an existing scenario")
        path.write_text(value, encoding="utf-8")
    manifest = {
        "schema": "native_damping_explicit_requests/1",
        "seed": seed,
        "mainline_vph": mainline_vph,
        "penetration": penetration,
        "episode_seconds": config.episode_seconds,
        "demand_end_seconds": config.demand_end_seconds,
        "population": len(requests),
        "cav_population": sum(r.is_cav for r in requests),
        "routes_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "network_sha256": hashlib.sha256(config.network_path.read_bytes()).hexdigest(),
        "requests_sha256": hashlib.sha256(
            json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
        ).hexdigest(),
        "assignment": "fixed_explicit_driver_type_and_cav_identity_no_runtime_type_change",
        "cav_vtype": cav_vtype,
    }
    if textbook:   # legacy3 manifests stay byte-identical to the pre-population code
        from traffic_shepherd.c4 import population as pop
        manifest.update({
            "cav_vtype": pop.TEXTBOOK_CAV_VTYPE,
            "driver_population": "textbook",
            "driver_population_version": pop.POPULATION_VERSION,
            "driver_truth_file": "hetdata_truth.csv",
            "driver_truth_sha256": hashlib.sha256(truth_text.encode()).hexdigest(),
            "driver_truth_note": "evaluation-only per-vehicle parameters; never a model input",
        })
    return scene_path, requests, manifest


def scenario_identity(scene, requests, *, network_path=None):
    """Bind active trajectory identity to requests, config, routes AND network bytes.

    This strengthens the historical identity, which only included a network path
    inside the configuration. Historical frozen identities are not rewritten.
    """
    from .research_assets import resolve_research_path

    scene = resolve_research_path(scene)
    if network_path is None:
        reference = ET.parse(scene).getroot().find("./input/net-file")
        if reference is None:
            raise ValueError("scenario does not identify its network")
        network_path = scene.parent / reference.attrib["value"]
    network_digest = hashlib.sha256(resolve_research_path(network_path).read_bytes()).digest()
    return hashlib.sha256(json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
                          + scene.read_bytes() + (scene.parent / "requests.rou.xml").read_bytes()
                          + network_digest).hexdigest()


def _mainline_internal_lanes(network_path, mainline_edges):
    """Internal junction lanes of the mainline path, from the REAL net.

    WM-43 A7: a vehicle traversing pre_merge -> merge or merge ->
    downstream spends part of the crossing on an INTERNAL lane
    (':<junction>_<n>_<m>'), which observation_map previously did not
    map - its distance_to_merge_m was None there and the wall group
    raised the loud member_position_unavailable abort (r6, 4/5 A2 runs
    at t=290-295).  Raw read of the preserved network (gzip XML, no
    sumolib, no live TraCI): every <connection> between mainline edges
    names its via internal LANE; all lanes of that internal edge map
    to the edge the crossing leads into.
    Returns {internal_lane_id: to_edge_id}.
    """
    import gzip
    opener = gzip.open if str(network_path).endswith(".gz") else open
    with opener(network_path, "rb") as stream:
        net_root = ET.parse(stream).getroot()
    lane_target = {}
    for conn in net_root.iter("connection"):
        source_edge, target_edge = conn.get("from"), conn.get("to")
        if source_edge not in mainline_edges or target_edge not in mainline_edges:
            continue
        via_lane = conn.get("via")
        if via_lane:
            lane_target[via_lane] = target_edge
    lanes = {}
    for edge in net_root.iter("edge"):
        if edge.get("function") != "internal":
            continue
        target = None
        for lane in edge.findall("lane"):
            target = lane_target.get(lane.get("id")) or target
        if target is not None:
            for lane in edge.findall("lane"):
                lanes[lane.get("id")] = target
    return lanes


def observation_map(api, config=None):
    config = config if config is not None else load_config()
    edges = config.edge_ids
    mapping = {}
    for role in ("pre_merge", "merge", "downstream", "ramp"):
        edge = edges[role]
        for index in range(api.edge.getLaneNumber(edge)):
            lane = f"{edge}_{index}"
            if role == "downstream":
                mapping[lane] = -api.lane.getLength(f"{edges['merge']}_0") - api.lane.getLength(lane)
            elif role == "merge":
                mapping[lane] = -api.lane.getLength(lane)
            else:
                mapping[lane] = 0.0
    # WM-43 A7: internal junction lanes collapse onto their junction's
    # boundary point.  Both neighbouring edges already assign the SAME
    # value to that point (pre_merge end 0.0 == merge start 0.0; merge
    # end -75.44 == downstream start -75.44), so the collapsed mapping
    # is monotone with travel and comparable with both neighbours; the
    # junction interior simply has no separate length on this axis.
    # Normal-lane values above are bit-identical to the registered map.
    mainline = {edges["pre_merge"], edges["merge"], edges["downstream"]}
    boundary_value = {edges["merge"]: 0.0,
                      edges["downstream"]: -api.lane.getLength(f"{edges['merge']}_0")}
    for lane_id, target_edge in _mainline_internal_lanes(
            config.network_path, mainline).items():
        mapping[lane_id] = boundary_value[target_edge]
    return mapping
