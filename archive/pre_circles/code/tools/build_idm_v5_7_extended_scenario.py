"""Build the isolated v5.7 straight-upstream-extension SUMO assets.

The source network is read-only.  The script copies it into the repository,
adds a two-lane 900 m dead-end approach connected only to the original first
mainline node, and rewrites the mainline route to traverse that edge.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from experiments import REPO_ROOT
from idm.contract import NETWORK_PATH, SCENARIO_ROOT


OUTPUT = REPO_ROOT / "scenarios" / "idm_v5_7_extended_zone"
NETWORK = OUTPUT / "osm.v5_7.extended.net.xml"
ROUTE = OUTPUT / "idm_m2700r400.v5_7.rou.xml"
CONFIG = OUTPUT / "osm.dense.idm_m2700r400.v5_7.sumocfg"
MANIFEST = OUTPUT / "manifest.json"
PRE_EDGE = "1191554636"
SOURCE_NODE = "470409306"
EXTENSION_NODE = "idm_v5_7_upstream_source"
UPSTREAM_EXTENSION_EDGE = "idm_v5_7_upstream_extension"
UPSTREAM_EXTENSION_M = 900.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_network() -> None:
    with gzip.open(NETWORK_PATH, "rb") as stream:
        root = ET.fromstring(stream.read())
    if root.find(f"edge[@id='{UPSTREAM_EXTENSION_EDGE}']") is not None:
        raise RuntimeError("source network unexpectedly already contains v5.7 edge")
    source = root.find(f"junction[@id='{SOURCE_NODE}']")
    if source is None:
        raise RuntimeError("source mainline node not found")
    x, y = float(source.attrib["x"]), float(source.attrib["y"])
    new_x = x - UPSTREAM_EXTENSION_M
    junction = ET.Element(
        "junction",
        {"id": EXTENSION_NODE, "type": "dead_end", "x": f"{new_x:.2f}", "y": f"{y:.2f}", "incLanes": "", "intLanes": "", "shape": ""},
    )
    edge = ET.Element(
        "edge",
        {"id": UPSTREAM_EXTENSION_EDGE, "from": EXTENSION_NODE, "to": SOURCE_NODE,
         "priority": "14", "type": "highway.motorway", "spreadType": "center",
         "name": "IDM v5.7 extended approach"},
    )
    for index in range(2):
        ET.SubElement(edge, "lane", {
            "id": f"{UPSTREAM_EXTENSION_EDGE}_{index}", "index": str(index),
            "speed": "31.29", "length": f"{UPSTREAM_EXTENSION_M:.2f}",
            "shape": f"{new_x:.2f},{y - 1.6 + 3.2 * index:.2f} {x:.2f},{y - 1.6 + 3.2 * index:.2f}",
        })
    # Place the direct extension immediately before the original external
    # approach; no original edge, junction, merge connection, or lane is edited.
    children = list(root)
    first_edge_index = next(index for index, child in enumerate(children) if child.tag == "edge")
    root.insert(first_edge_index, edge)
    root.insert(first_edge_index, junction)
    for lane in range(2):
        root.append(ET.Element("connection", {
            "from": UPSTREAM_EXTENSION_EDGE, "to": PRE_EDGE,
            "fromLane": str(lane), "toLane": str(lane), "dir": "s", "state": "M",
        }))
    ET.ElementTree(root).write(NETWORK, encoding="utf-8", xml_declaration=True)


def _write_route_and_config() -> None:
    source_route = SCENARIO_ROOT / "idm_m2700r400.rou.xml"
    text = source_route.read_text(encoding="utf-8")
    text = text.replace(
        'edges="1191554636 1487606179 1487606178"',
        f'edges="{UPSTREAM_EXTENSION_EDGE} 1191554636 1487606179 1487606178"',
        1,
    )
    ROUTE.write_text(text, encoding="utf-8")
    CONFIG.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<sumoConfiguration><input>\n"
        f"<net-file value=\"{NETWORK.name}\"/>\n"
        f"<route-files value=\"{ROUTE.name}\"/>\n"
        "</input><processing><ignore-route-errors value=\"true\"/>"
        "<collision.action value=\"warn\"/><time-to-teleport value=\"-1\"/>"
        "</processing><report><no-step-log value=\"true\"/></report>"
        "</sumoConfiguration>\n",
        encoding="utf-8",
    )


def build() -> dict[str, str]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_network()
    _write_route_and_config()
    result = {
        "schema": "idm_v5_7_extended_scenario/1",
        "source_network": str(NETWORK_PATH), "source_network_sha256": _sha(NETWORK_PATH),
        "derived_network": str(NETWORK), "derived_network_sha256": _sha(NETWORK),
        "derived_route": str(ROUTE), "derived_route_sha256": _sha(ROUTE),
        "derived_config": str(CONFIG), "derived_config_sha256": _sha(CONFIG),
        "extension_edge": UPSTREAM_EXTENSION_EDGE, "extension_length_m": UPSTREAM_EXTENSION_M,
        "preserved_downstream_edges": [PRE_EDGE, "1487606179", "1487606178", "11553048#1"],
    }
    import json
    MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(build())
