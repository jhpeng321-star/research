"""Explicit exogenous departures for full-horizon damping development.

Driver type, vehicle identity, requested departure and CAV designation are
fixed before a policy runs. Insertion delays cannot change the population.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from idm.contract import NETWORK_PATH, SCENARIO_ROOT


EPISODE_SECONDS = 2400
DEMAND_END_SECONDS = 1800
EXPOSED_DEVELOPMENT_SEEDS = frozenset(range(4107, 4133))


@dataclass(frozen=True)
class TripRequest:
    vehicle_id: str
    depart_s: float
    route: str
    driver_type: str
    is_cav: bool


def request_schedule(seed: int, mainline_vph: int, *, role: str = "exposed") -> list[TripRequest]:
    if role != "exposed":
        from idm.damping_campaign import validate_case

        validate_case(seed, mainline_vph, role)
    elif seed not in EXPOSED_DEVELOPMENT_SEEDS:
        raise ValueError("this development runner admits only already exposed seeds 4107–4132")
    if not 1900 <= mainline_vph <= 2800:
        raise ValueError("loading outside declared development family")
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
                cav = random.Random(f"damping_cav_v1:{seed}:{identity}").random() < 0.2
                requests.append(TripRequest(identity, round(t, 6), stream + "_route", driver, cav))
                ordinal += 1
    return sorted(requests, key=lambda r: (r.depart_s, r.vehicle_id))


def build_scenario(
    root: Path, seed: int, mainline_vph: int, *, role: str = "exposed"
) -> tuple[Path, list[TripRequest], dict]:
    requests = request_schedule(seed, mainline_vph, role=role)
    template = SCENARIO_ROOT / "idm_m2200r400.rou.xml"
    original = ET.parse(template).getroot()
    route = ET.Element("routes")
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
    ET.SubElement(inputs, "net-file", value=str(NETWORK_PATH))
    ET.SubElement(inputs, "route-files", value=route_path.name)
    processing = ET.SubElement(scene, "processing")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.indent(scene)
    scene_text = ET.tostring(scene, encoding="unicode")
    for path, value in ((route_path, text), (scene_path, scene_text)):
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise ValueError("refusing to change an existing scenario")
        path.write_text(value, encoding="utf-8")
    manifest = {
        "schema": "native_damping_explicit_requests/1",
        "seed": seed,
        "mainline_vph": mainline_vph,
        "penetration": 0.2,
        "episode_seconds": EPISODE_SECONDS,
        "demand_end_seconds": DEMAND_END_SECONDS,
        "population": len(requests),
        "cav_population": sum(r.is_cav for r in requests),
        "routes_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "network_sha256": hashlib.sha256(NETWORK_PATH.read_bytes()).hexdigest(),
        "requests_sha256": hashlib.sha256(
            json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
        ).hexdigest(),
        "assignment": "fixed_explicit_driver_type_and_cav_identity_no_runtime_type_change",
    }
    return scene_path, requests, manifest
