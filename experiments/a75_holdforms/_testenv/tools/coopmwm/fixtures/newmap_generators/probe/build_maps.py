"""NEWMAP probe: build candidate SUMO networks (new files only) for group-cooperation research.

Writes plain-XML sources + netconvert output + route definitions + a demand spec into
E:/research-dense/assets/research/newmap_candidates/<name>/.  Never touches existing assets;
refuses to change a file whose content differs.

Maps (all mainline lanes 31.29 m/s = the project's speed limit; ramps 22.22 m/s as the project ramp):
  wz3to2  - 3-lane freeway, rightmost lane closed at x=2700 m (work-zone lane drop), 2 lanes for 1300 m.
  wz2to1  - 2-lane freeway, right lane closed at x=2700 m (zipper merge), 1 lane for 1300 m.
  weave   - 2-lane freeway; on-ramp adds an auxiliary lane at x=2400 m, off-ramp drops it at x=2800 m
            (400 m weaving section); 1000 m downstream.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("E:/research-dense/assets/research/newmap_candidates")
V_MAIN = 31.29
V_RAMP = 22.22


def _write(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refusing to change existing {path}")
        return
    path.write_text(text, encoding="utf-8")


def nodes(rows):
    body = "\n".join(f'    <node id="{i}" x="{x}" y="{y}" type="{t}"/>' for i, x, y, t in rows)
    return f'<nodes>\n{body}\n</nodes>\n'


def edges(rows):
    body = "\n".join(
        f'    <edge id="{i}" from="{a}" to="{b}" numLanes="{n}" speed="{v}" priority="{p}"/>'
        for i, a, b, n, v, p in rows)
    return f'<edges>\n{body}\n</edges>\n'


def conns(rows):
    body = "\n".join(f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>'
                     for a, b, fl, tl in rows)
    return f'<connections>\n{body}\n</connections>\n'


def routes(rows):
    body = "\n".join(f'    <route id="{i}" edges="{e}"/>' for i, e in rows)
    return f'<routes>\n{body}\n</routes>\n'


MAPS = {
    "wz3to2": dict(
        nodes=[("n0", 0, 0, "priority"), ("n1", 2700, 0, "priority"), ("n2", 4000, 0, "priority")],
        edges=[("up", "n0", "n1", 3, V_MAIN, 10), ("wz", "n1", "n2", 2, V_MAIN, 10)],
        conns=[("up", "wz", 1, 0), ("up", "wz", 2, 1)],
        routes=[("main", "up wz")],
        bottleneck=dict(edge="up", x_m=2700.0, dropped_lane=0, through_lanes=[1, 2]),
        streams={"main": dict(route="main", profile="peak", depart_lane="free")},
        description="3->2 work-zone lane drop: lane 0 (rightmost) of edge 'up' ends at x=2700 m "
                    "without a connection, so its vehicles must change lanes before the end.",
    ),
    "wz2to1": dict(
        nodes=[("n0", 0, 0, "priority"), ("n1", 2700, 0, "priority"), ("n2", 4000, 0, "priority")],
        edges=[("up", "n0", "n1", 2, V_MAIN, 10), ("wz", "n1", "n2", 1, V_MAIN, 10)],
        conns=[("up", "wz", 1, 0)],
        routes=[("main", "up wz")],
        bottleneck=dict(edge="up", x_m=2700.0, dropped_lane=0, through_lanes=[1]),
        streams={"main": dict(route="main", profile="peak", depart_lane="free")},
        description="2->1 lane closure (zipper merge): lane 0 of edge 'up' ends at x=2700 m.",
    ),
    "weave": dict(
        nodes=[("m0", 0, 0, "priority"), ("j1", 2400, 0, "priority"), ("j2", 2800, 0, "priority"),
               ("m3", 3800, 0, "priority"), ("r0", 2050, -60, "priority"),
               ("x0", 3150, -60, "priority")],
        edges=[("up", "m0", "j1", 2, V_MAIN, 10), ("weave", "j1", "j2", 3, V_MAIN, 10),
               ("down", "j2", "m3", 2, V_MAIN, 10), ("ramp_in", "r0", "j1", 1, V_RAMP, 5),
               ("ramp_out", "j2", "x0", 1, V_RAMP, 5)],
        conns=[("up", "weave", 0, 1), ("up", "weave", 1, 2), ("ramp_in", "weave", 0, 0),
               ("weave", "ramp_out", 0, 0), ("weave", "down", 1, 0), ("weave", "down", 2, 1)],
        routes=[("through", "up weave down"), ("exit", "up weave ramp_out"),
                ("enter", "ramp_in weave down")],
        bottleneck=dict(edge="weave", x_m=2400.0, weave_len_m=400.0, aux_lane=0),
        streams={"main": dict(route=["through", "exit"], exit_share=0.2, profile="peak",
                              depart_lane="free"),
                 "ramp": dict(route="enter", profile="flat", depart_lane="best")},
        description="400 m weaving section: on-ramp joins as auxiliary lane 0 of edge 'weave'; "
                    "lane 0 diverges to the off-ramp. Entering traffic must move left 1 lane, "
                    "exiting traffic (20% of mainline) must move right 1 lane within 400 m.",
    ),
}


def build(name: str, spec: dict, netconvert: str = "netconvert") -> dict:
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    _write(out / f"{name}.nod.xml", nodes(spec["nodes"]))
    _write(out / f"{name}.edg.xml", edges(spec["edges"]))
    _write(out / f"{name}.con.xml", conns(spec["conns"]))
    _write(out / f"{name}.routes.xml", routes(spec["routes"]))
    net = out / f"{name}.net.xml"
    if not net.exists():
        cmd = [netconvert, "--node-files", str(out / f"{name}.nod.xml"),
               "--edge-files", str(out / f"{name}.edg.xml"),
               "--connection-files", str(out / f"{name}.con.xml"),
               "--no-turnarounds", "true", "--offset.disable-normalization", "true",
               "--output-file", str(net)]
        subprocess.run(cmd, check=True)
    meta = {k: v for k, v in spec.items() if k not in ("nodes", "edges", "conns")}
    meta.update(name=name, net_file=net.name, routes_file=f"{name}.routes.xml",
                speed_limit_mps=V_MAIN, ramp_speed_mps=V_RAMP,
                built_with="netconvert (Eclipse SUMO 1.27.1)",
                builder="runs/active/C4-V7FT-20261017/newmap_probe/build_maps.py")
    _write(out / "MAP_SPEC.json", json.dumps(meta, indent=1) + "\n")
    return meta


if __name__ == "__main__":
    names = sys.argv[1:] or list(MAPS)
    for n in names:
        print(json.dumps(build(n, MAPS[n]))[:200])
