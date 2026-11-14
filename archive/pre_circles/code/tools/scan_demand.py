"""Scan mainline inflow rates to find light/medium/dense sweet spots."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_demand import build_routes_xml  # noqa: E402

SUMO_DIR = ROOT.parent / "highway_merge_shepherd"
NET = SUMO_DIR / "osm.net.xml.gz"


def _sumo_binary():
    exe = shutil.which("sumo")
    if exe:
        return exe
    home = os.environ.get("SUMO_HOME")
    if home:
        candidate = Path(home) / "bin" / "sumo"
        if candidate.exists():
            return str(candidate)
    return "sumo"


def parse_sumo_stats(stats_xml_text):
    """Parse SUMO statistic-output XML (layout calibrated on SUMO 1.27.1)."""
    root = ET.fromstring(stats_xml_text)
    veh = root.find("vehicles")
    trip = root.find("vehicleTripStatistics")
    tele = root.find("teleports")
    safe = root.find("safety")
    inserted = int(veh.get("inserted")) if veh is not None else 0
    running = int(veh.get("running")) if veh is not None else 0
    return {
        "arrived": inserted - running,
        "mean_tt": float(trip.get("duration")) if trip is not None else None,
        "mean_time_loss": float(trip.get("timeLoss")) if trip is not None else None,
        "teleports": int(tele.get("total")) if tele is not None else 0,
        "collisions": int(safe.get("collisions")) if safe is not None else 0,
    }


def scan_cell(mainline_vph, ramp_vph, seed, duration_s=900):
    """Run one All-HDV SUMO cell directly via CLI flags; return metrics dict."""
    with tempfile.TemporaryDirectory() as tmp:
        rou = Path(tmp) / "demand.rou.xml"
        rou.write_text(
            build_routes_xml(mainline_vph, ramp_vph, duration_s, "standard", seed), encoding="utf-8"
        )
        stats = Path(tmp) / "stats.xml"
        cmd = [
            _sumo_binary(),
            "-n",
            str(NET),
            "-r",
            str(rou),
            "--statistic-output",
            str(stats),
            # Without this flag the stats file omits <vehicleTripStatistics>.
            "--duration-log.statistics",
            "true",
            "--collision.action",
            "warn",
            "--time-to-teleport",
            "-1",
            "--seed",
            str(seed),
            "--no-step-log",
            "true",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(SUMO_DIR))
        m = parse_sumo_stats(stats.read_text(encoding="utf-8"))
        m.update(mainline_vph=mainline_vph, ramp_vph=ramp_vph, seed=seed)
        return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mainline-rates", default="1400,1800,2200,2600,3000,3400")
    p.add_argument("--ramp-vph", type=int, default=600)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "demand_scan"))
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rates = [int(x) for x in a.mainline_rates.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    for rate in rates:
        for seed in seeds:
            m = scan_cell(rate, a.ramp_vph, seed, a.duration)
            if m["mean_tt"] is None or m["mean_time_loss"] is None:
                # Stats XML lacked <vehicleTripStatistics> (e.g. SUMO run
                # without --duration-log.statistics): skip the cell instead of
                # crashing on f"{None:.1f}".
                print(
                    f"WARNING: main={rate} seed={seed}: no <vehicleTripStatistics> "
                    "in SUMO stats output; skipping cell",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            rows.append(m)
            print(
                f"main={rate:5d} seed={seed}  TT={m['mean_tt']:.1f}  "
                f"timeLoss={m['mean_time_loss']:.1f}  coll={m['collisions']}  "
                f"tele={m['teleports']}  arrived={m['arrived']}",
                flush=True,
            )
    (out / "scan_results.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "| mainline vph | seed | mean TT | timeLoss | collisions | teleports | arrived |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in rows:
        lines.append(
            f"| {m['mainline_vph']} | {m['seed']} | {m['mean_tt']:.1f} "
            f"| {m['mean_time_loss']:.1f} | {m['collisions']} "
            f"| {m['teleports']} | {m['arrived']} |"
        )
    (out / "scan_table.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
