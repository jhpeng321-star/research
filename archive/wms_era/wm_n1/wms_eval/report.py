"""Compact P0 baseline report + median-position-error plot from EVAL.json runs.

Usage::

    python -m wms_eval.report --run keep_speed=DIR --run pure_physics=DIR \
        --run a11_closed=DIR --run a11_open=DIR --run a11_open_fid=DIR \
        --out _out/P0_BASELINE.md --plot _out/plots/median_ds.svg

Only ``run`` outputs already produced by ``python -m wms_eval run`` are read.
The report is deliberately capped at the P0 contract: car errors at fixed
horizons, per-section 30 s traffic, and the paired PI-native difference.  Delay
is the normalized ``sum (1 - v/v_free) * dt`` seconds, never metres.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

HORIZONS = ("5.0", "10.0", "30.0", "60.0", "120.0", "180.0", "300.0")
SECTIONS = ("upstream", "merge", "downstream")
TRAFFIC = ("mean_speed", "density", "stopped_vehicles", "merge_queue_length",
           "merge_flow", "total_delay")
COLORS = ("#174a7e", "#b5651d", "#2e7d32", "#c62828", "#6a1b9a", "#00838f", "#ef6c00")
LABELS = ("5", "10", "30", "60", "120", "180", "300")


def _load(path: Path) -> dict:
    if path.is_dir():
        path = path / "EVAL.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else float((values[mid - 1] + values[mid]) / 2)


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def car_series(payload: dict, arm: str, stratum: str, metric: str, stat: str):
    """Per-horizon aggregate across states for one arm/stratum/metric/stat."""
    out = []
    for horizon in HORIZONS:
        values = []
        for result in payload["results"]:
            block = result["car"].get(arm, {}).get(horizon, {}).get(stratum, {})
            item = block.get(metric, {})
            values.append(item.get(stat))
        out.append(_median(values))
    return out


def _car_line(payload: dict, label: str, arm: str, stratum: str, metric: str,
              stat: str, digits: int = 2) -> str:
    series = car_series(payload, arm, stratum, metric, stat)
    return f"- {label} {stratum} {metric} {stat} @{ '/'.join(LABELS) }: " + \
        " ".join(_fmt(value, digits) for value in series)


def _lane_match(payload: dict, arm: str, stratum: str):
    lane, matched = [], []
    for horizon in HORIZONS:
        lanes, fractions = [], []
        for result in payload["results"]:
            block = result["car"].get(arm, {}).get(horizon, {}).get(stratum, {})
            if block.get("lane_agreement") is not None:
                lanes.append(block["lane_agreement"])
            if block.get("matched_fraction") is not None:
                fractions.append(block["matched_fraction"])
        lane.append(_mean(lanes))
        matched.append(_mean(fractions))
    return lane, matched


def _traffic(payload: dict, arm: str) -> dict:
    """Mean recorded/predicted per section/metric across states and 30 s bins."""
    out = {}
    for section in SECTIONS:
        out[section] = {}
        for metric in TRAFFIC:
            rec, pred = [], []
            for result in payload["results"]:
                traffic = result["traffic"]
                keys = sorted(traffic["predicted"][arm], key=lambda key: int(key))
                for key in keys:
                    p = traffic["predicted"][arm][key]["sections"][section].get(metric)
                    r = traffic["recorded"][arm].get(key, {}).get("sections", {}) \
                        .get(section, {}).get(metric)
                    if p is not None:
                        pred.append(p)
                    if r is not None:
                        rec.append(r)
            out[section][metric] = (_mean(rec), _mean(pred))
    return out


def _paired(payload: dict, arm_native: str = "native") -> dict:
    out = {}
    for section in ("merge",):
        for metric in ("mean_speed", "total_delay"):
            p_pred, p_rec = [], []
            for result in payload["results"]:
                traffic = result["traffic"]
                keys = sorted(traffic["predicted"][arm_native], key=lambda key: int(key))
                for key in keys:
                    if traffic["paired_predicted_pi_minus_native"].get(key) is None:
                        continue
                    pred = traffic["paired_predicted_pi_minus_native"][key]["sections"] \
                        .get(section, {}).get(metric)
                    rec = traffic["paired_recorded_pi_minus_native"].get(key, {}) \
                        .get("sections", {}).get(section, {}).get(metric)
                    if pred is not None:
                        p_pred.append(pred)
                    if rec is not None:
                        p_rec.append(rec)
            out[f"{section}.{metric}"] = (_mean(p_rec), _mean(p_pred))
    return out


def _svg(runs: dict, out_path: Path) -> None:
    """Dependency-free SVG: median HDV |ds| (native) vs horizon per model."""
    series = {label: car_series(payload, "native", "hdv", "abs_delta_s_m", "median")
              for label, payload in runs.items()}
    values = [v for s in series.values() for v in s if v is not None]
    vmax = max([1.0] + values)
    width, height = 640, 360
    left, right, top, bottom = 60, 620, 30, 300
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="60" y="20" font-size="14">median HDV |ds| (m) vs horizon (s)</text>']
    for color, (label, s) in zip(COLORS, series.items()):
        points = []
        for index, value in enumerate(s):
            if value is None:
                continue
            x = left + index * (right - left) / (len(HORIZONS) - 1)
            y = bottom - value * (bottom - top) / vmax
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                         f'points="{" ".join(points)}"/>')
    for index, label in enumerate(LABELS):
        x = left + index * (right - left) / (len(HORIZONS) - 1)
        lines.append(f'<text x="{x:.1f}" y="{bottom + 18}" font-size="11" '
                     f'text-anchor="middle">{label}</text>')
    lines.append(f'<text x="10" y="{top}" font-size="11">{vmax:.0f}</text>')
    for offset, (color, label) in enumerate(zip(COLORS, series)):
        lines.append(f'<text x="{left + 10}" y="{top + 14 + offset * 14}" '
                     f'font-size="11" fill="{color}">{html.escape(label)}</text>')
    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _series(values, digits: int = 1) -> str:
    return " ".join(_fmt(value, digits) for value in values)


def _med_p90(payload: dict, metric: str, digits: int = 1) -> str:
    groups = []
    for stratum in ("hdv", "cav"):
        med = _series(car_series(payload, "native", stratum, metric, "median"), digits)
        p90 = _series(car_series(payload, "native", stratum, metric, "p90"), digits)
        groups.append(f"{stratum} med[{med}] p90[{p90}]")
    return " ".join(groups)


def _traffic_line(payload: dict, label: str) -> str:
    traffic = _traffic(payload, "native")
    chunks = []
    for section in SECTIONS:
        block = traffic[section]
        parts = [f"{key} {_fmt(block[key][0])}->{_fmt(block[key][1])}" for key in TRAFFIC]
        chunks.append(f"{section}[{' '.join(parts)}]")
    return f"- {label} " + " | ".join(chunks)


def _paired_line(payload: dict, label: str) -> str:
    paired = _paired(payload)
    parts = [f"{key} {_fmt(value[0])}->{_fmt(value[1])}" for key, value in paired.items()]
    return f"- {label} " + "; ".join(parts)


def build(run_paths: dict, out_md: Path, plot_path: Path) -> Path:
    runs = {label: _load(Path(path)) for label, path in run_paths.items()}
    lines = ["# P0 baseline - WM-S offline evaluation (TRAIN only; 20 states; no SUMO/GPU)",
             "Status: complete. Delay = normalized delay seconds sum((1 - v/v_free)*dt), never metres.",
             "Arm native; horizons 5/10/30/60/120 s; |ds| m, |dv| m/s; across-state aggregate.",
             "",
             "## Car level (native); each metric HDV then CAV, med then p90 across states"]
    for label, payload in runs.items():
        lines.append(f"- {label} |ds| {_med_p90(payload, 'abs_delta_s_m')}")
    for label, payload in runs.items():
        lines.append(f"- {label} |dv| {_med_p90(payload, 'abs_delta_v_mps')}")
    for label, payload in runs.items():
        lane_h, match_h = _lane_match(payload, "native", "hdv")
        lane_c, _ = _lane_match(payload, "native", "cav")
        lines.append(f"- {label} lane hdv[{_series(lane_h, 2)}] cav[{_series(lane_c, 2)}] "
                     f"matched hdv[{_series(match_h, 2)}]")
    lines.append("## Traffic 30 s bins (recorded->predicted, mean over bins/states; "
                 "mean_speed density stopped queue flow delay)")
    for label, payload in runs.items():
        lines.append(_traffic_line(payload, label))
    lines.append("## Paired PI-native difference (merge; recorded->predicted)")
    for label, payload in runs.items():
        lines.append(_paired_line(payload, label))
    lines.append(f"Plot: `{plot_path.name}` (median HDV |ds| vs horizon, all models). "
                 "Full per-state detail in each run's EVAL.json.")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _svg(runs, plot_path)
    return out_md


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wms_eval.report")
    parser.add_argument("--run", action="append", required=True,
                        help="label=path to an EVAL.json or its run directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot", required=True)
    args = parser.parse_args(argv)
    run_paths = {}
    for item in args.run:
        label, _, path = item.partition("=")
        run_paths[label] = path
    print(build(run_paths, Path(args.out), Path(args.plot)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
