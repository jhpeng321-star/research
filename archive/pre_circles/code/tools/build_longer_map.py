"""Build a NEW longer ramp-merge map as a SEPARATE copy (does not touch the
current map). Lengthens the upstream source mainline edge (1191554636) by
``EXTEND_M`` meters of straight approach so CAVs get more runway to match the
merge-zone speed far in advance.

The mainline edge is a dead-end fringe source (nothing connects upstream), so
extending it backward is topologically clean and leaves all edge IDs / merge
detection / RSU features / fairness references valid.
"""

from __future__ import annotations

import gzip
import math
import re
import shutil
from pathlib import Path

SRC = Path(r"D:\research\highway_merge_shepherd")
DST = Path(r"D:\research\highway_merge_shepherd_long")
MAINLINE_EDGE = "1191554636"
EXTEND_M = 1500.0  # extra approach runway
SPACING = 100.0  # polyline point spacing for the added straight part
COPY_EXTRA = True  # also copy _light demand scenes + add files for probes


def _parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--extend", type=float, default=EXTEND_M)
    ap.add_argument(
        "--no-extra", action="store_true", help="skip copying _light/ + _ss/ demand scenes"
    )
    return ap.parse_args()


def extend_shape(shape_str: str, unit: tuple, n_points: int) -> list:
    pts = shape_str.split()
    first = tuple(map(float, pts[0].split(",")))
    extra = []
    step = SPACING
    for k in range(n_points, 0, -1):  # farthest -> nearest to original start
        extra.append((first[0] - unit[0] * k * step, first[1] - unit[1] * k * step))
    out = [f"{x:.2f},{y:.2f}" for x, y in extra] + pts
    return out


def polyline_len(shape_list: list) -> float:
    total = 0.0
    prev = None
    for s in shape_list:
        x, y = tuple(map(float, s.split(",")))
        if prev is not None:
            total += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
    return total


def main() -> None:
    args = _parse_args()
    src, dst, extend_m = args.src, args.dst, args.extend
    raw = gzip.open(src / "osm.net.xml.gz", "rt", encoding="utf-8", errors="replace").read()

    # Unit upstream direction from the first two points of the mainline lane _0.
    m0 = re.search(
        r'<edge id="'
        + MAINLINE_EDGE
        + r'".*?<lane id="'
        + MAINLINE_EDGE
        + r'_0"[^>]*shape="([^"]*)"',
        raw,
        re.S,
    )
    if not m0:
        raise SystemExit("mainline lane _0 not found")
    p0str = m0.group(1).split()[0]
    p1str = m0.group(1).split()[1]
    p0 = tuple(map(float, p0str.split(",")))
    p1 = tuple(map(float, p1str.split(",")))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    norm = math.hypot(dx, dy)
    unit = (dx / norm, dy / norm)
    n_points = int(round(extend_m / SPACING))

    def repl_lane(m: re.Match) -> str:
        block = m.group(0)  # just the lane opening tag: <lane id="X" ... shape="...">
        shape = re.search(r'shape="([^"]*)"', block).group(1)
        new_shape = extend_shape(shape, unit, n_points)
        new_len = polyline_len(new_shape)
        block = re.sub(r'length="[0-9.]+"', f'length="{new_len:.2f}"', block, count=1)
        block = re.sub(r'shape="[^"]*"', 'shape="' + " ".join(new_shape) + '"', block, count=1)
        return block

    lane_pat = re.compile(r'<lane id="' + MAINLINE_EDGE + r'_[0-9]+"[^>]*>')
    new_raw, n_lanes = lane_pat.subn(repl_lane, raw)
    print(f"extended {n_lanes} lanes of {MAINLINE_EDGE} by +{extend_m:.0f}m")

    # Write new (longer) net into a fresh directory.
    dst.mkdir(parents=True, exist_ok=True)
    with gzip.open(dst / "osm.net.xml.gz", "wt", encoding="utf-8") as f:
        f.write(new_raw)

    # Copy the rest of the scene (routes reuse edge IDs, add files reference
    # edges/positions still valid under a longer mainline).
    for name in [
        "osm.demand.dense.rou.xml",
        "osm.demand.medium.rou.xml",
        "osm.demand.light.rou.xml",
        "osm.demand.hyper.rou.xml",
        "osm.dense.sumocfg",
        "osm.medium.sumocfg",
        "osm.light.sumocfg",
        "osm.hyper.sumocfg",
        "osm.sumocfg",
        "osm.view.xml",
        "output.add.xml",
        "output.dense.add.xml",
        "output.medium.add.xml",
        "output.light.add.xml",
        "output.hyper.add.xml",
        "hdv_types.add.xml",
    ]:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
            print("  copied", name)
        else:
            print("  MISSING (skipped)", name)

    # Copy the light-demand scene folder (per-density route files + sumocfgs)
    # so the probe driver can run the same tags on the new map.
    if not args.no_extra:
        for sub in ("_light", "_ss"):
            sdir = src / sub
            if sdir.is_dir():
                shutil.copytree(sdir, dst / sub, dirs_exist_ok=True)
                print("  copied", sub)

    print("\nNew longer map ready at:", dst)

    # quick sanity: report old/new lane length
    def lane_len(fn):
        g = gzip.open(fn, "rt", encoding="utf-8", errors="replace").read()
        m = re.search(
            r'<edge id="'
            + MAINLINE_EDGE
            + r'".*?<lane id="'
            + MAINLINE_EDGE
            + r'_0"[^>]*length="([0-9.]+)"',
            g,
            re.S,
        )
        return float(m.group(1))

    print(
        f"mainline _0 length: {lane_len(src / 'osm.net.xml.gz'):.1f} -> {lane_len(dst / 'osm.net.xml.gz'):.1f} m"
    )


if __name__ == "__main__":
    main()
