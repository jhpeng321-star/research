"""AMENDMENT 40 (MAIN 2026-11-10, feasibility + build): build the geometry variant wz3to2v80 of the wz3to2 work-zone
lane drop with the SAME builder as wz3to2 (runs/active/C4-V7FT-20261017/newmap_probe/build_maps.py, vendored as
tools/coopmwm/fixtures/newmap_generators/probe/build_maps.py): plain XML -> netconvert 1.27.1 with the same options.

Only change (E:/agent-work/a40/VARIANT_CHOICE.md, sha256 14c64b08..., fixed before any controller arm): the downstream
work-zone edge ``wz`` is posted at 22.22 m/s (80 km/h) instead of 31.29 m/s. Approach, lane drop, connections,
lengths and route are the wz3to2 values. Writes a NEW directory only; refuses to change an existing file; never
touches assets/research/newmap_candidates/wz3to2.

usage: python tools/coopmwm/build_geomvar_a40.py [--netconvert netconvert] [--root assets/research/newmap_candidates]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

NAME = 'wz3to2v80'
V_MAIN = 31.29
V_WZ = 22.22          # 80 km/h work-zone limit (VARIANT_CHOICE.md rule: lowest 10 km/h step >= max frozen v_w 22.0)
V_RAMP = 22.22
BASE = 'wz3to2'

SPEC = dict(
    nodes=[('n0', 0, 0, 'priority'), ('n1', 2700, 0, 'priority'), ('n2', 4000, 0, 'priority')],
    edges=[('up', 'n0', 'n1', 3, V_MAIN, 10), ('wz', 'n1', 'n2', 2, V_WZ, 10)],
    conns=[('up', 'wz', 1, 0), ('up', 'wz', 2, 1)],
    routes=[('main', 'up wz')],
    bottleneck=dict(edge='up', x_m=2700.0, dropped_lane=0, through_lanes=[1, 2]),
    streams={'main': dict(route='main', profile='peak', depart_lane='free')},
    description=('A40 geometry variant of wz3to2: identical 3->2 work-zone lane drop (lane 0 of edge up ends at '
                 'x=2700 m without a connection), but the 2-lane work zone (edge wz, 1300 m) is posted at 22.22 m/s '
                 '(80 km/h) instead of 31.29 m/s.'),
)


def nodes(rows):
    body = '\n'.join(f'    <node id="{i}" x="{x}" y="{y}" type="{t}"/>' for i, x, y, t in rows)
    return f'<nodes>\n{body}\n</nodes>\n'


def edges(rows):
    body = '\n'.join(f'    <edge id="{i}" from="{a}" to="{b}" numLanes="{n}" speed="{v}" priority="{p}"/>'
                     for i, a, b, n, v, p in rows)
    return f'<edges>\n{body}\n</edges>\n'


def conns(rows):
    body = '\n'.join(f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>' for a, b, fl, tl in rows)
    return f'<connections>\n{body}\n</connections>\n'


def routes(rows):
    body = '\n'.join(f'    <route id="{i}" edges="{e}"/>' for i, e in rows)
    return f'<routes>\n{body}\n</routes>\n'


def sources():
    """name -> text of the plain-XML builder inputs (the only difference from wz3to2 is the wz edge speed)."""
    return {f'{NAME}.nod.xml': nodes(SPEC['nodes']), f'{NAME}.edg.xml': edges(SPEC['edges']),
            f'{NAME}.con.xml': conns(SPEC['conns']), f'{NAME}.routes.xml': routes(SPEC['routes'])}


def meta():
    m = {k: v for k, v in SPEC.items() if k not in ('nodes', 'edges', 'conns')}
    m.update(name=NAME, net_file=f'{NAME}.net.xml', routes_file=f'{NAME}.routes.xml', speed_limit_mps=V_MAIN,
             workzone_speed_limit_mps=V_WZ, ramp_speed_mps=V_RAMP, base_map=BASE,
             built_with='netconvert (Eclipse SUMO 1.27.1)', builder='tools/coopmwm/build_geomvar_a40.py',
             variant_choice='E:/agent-work/a40/VARIANT_CHOICE.md')
    return m


def _write(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise SystemExit(f'refusing to change existing {path}')
        return
    path.write_text(text, encoding='utf-8', newline='\n')


def build(root: Path, netconvert: str = 'netconvert') -> dict:
    out = root/NAME
    if out.resolve() == (root/BASE).resolve():
        raise SystemExit('refusing to write into the base map directory')
    out.mkdir(parents=True, exist_ok=True)
    for name, text in sources().items():
        _write(out/name, text)
    net = out/f'{NAME}.net.xml'
    if not net.exists():
        subprocess.run([netconvert, '--node-files', f'{NAME}.nod.xml', '--edge-files', f'{NAME}.edg.xml',
                        '--connection-files', f'{NAME}.con.xml', '--no-turnarounds', 'true',
                        '--offset.disable-normalization', 'true', '--output-file', net.name], check=True, cwd=out)
    m = meta()
    _write(out/'MAP_SPEC.json', json.dumps(m, indent=1)+'\n')
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--netconvert', default='netconvert')
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2]/'assets/research/newmap_candidates'))
    a = p.parse_args()
    print(json.dumps(build(Path(a.root), a.netconvert))[:300])
