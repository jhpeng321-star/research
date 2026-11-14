"""Compact RUN1 evaluation report for the first trained WM-S checkpoint.

Usage::

    python -m wms_eval.run1_report \
        --run keep_speed=DIR --run pure_physics=DIR --run wms_run1=DIR \
        --run wms_stage10=DIR ... --run1 wms_run1 \
        --out _out/RUN1_EVAL.md --plot _out/plots/median_ds_run1.svg

Reads only EVAL.json outputs already produced by ``python -m wms_eval run``.
Delay is the normalized ``sum (1 - v/v_free) * dt`` seconds, never metres.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from .report import HORIZONS, LABELS, SECTIONS, COLORS, _fmt, _load, _mean

PAIR_METRICS = ("mean_speed", "total_delay")
TRAFFIC_METRICS = ("mean_speed", "stopped_vehicles", "merge_queue_length", "total_delay")
TRAFFIC_SHORT = {"mean_speed": "speed", "stopped_vehicles": "stopped",
                 "merge_queue_length": "queue", "total_delay": "delay"}


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    da, db = [x - ma for x in a], [x - mb for x in b]
    sa = sum(x * x for x in da) ** 0.5
    sb = sum(x * x for x in db) ** 0.5
    if sa == 0.0 or sb == 0.0:
        return None
    return sum(x * y for x, y in zip(da, db)) / (sa * sb)


def rank_correlation(predicted: list[float], recorded: list[float]) -> float | None:
    return _pearson(_rank(predicted), _rank(recorded)) if len(predicted) >= 2 else None


def sign_agreement(predicted: list[float], recorded: list[float]) -> tuple[int, int]:
    agree = total = 0
    for p, r in zip(predicted, recorded):
        sp = 0.0 if abs(p) < 1e-12 else (1.0 if p > 0 else -1.0)
        sr = 0.0 if abs(r) < 1e-12 else (1.0 if r > 0 else -1.0)
        total += 1
        agree += sp == sr
    return agree, total


def _paired(payload: dict, metric: str) -> tuple[list[float], list[float]]:
    pred, rec = [], []
    for result in payload["results"]:
        traffic = result["traffic"]
        for key, block in traffic["paired_predicted_pi_minus_native"].items():
            p = block.get("sections", {}).get("merge", {}).get(metric)
            r = traffic["paired_recorded_pi_minus_native"].get(key, {}) \
                .get("sections", {}).get("merge", {}).get(metric)
            if p is not None and r is not None:
                pred.append(p)
                rec.append(r)
    return pred, rec


def _spread(payload: dict, stratum: str, metric: str) -> tuple[list, list]:
    med = [_median([r["car"]["native"].get(h, {}).get(stratum, {}).get(metric, {}).get("median")
                    for r in payload["results"]]) for h in HORIZONS]
    p90 = [_median([r["car"]["native"].get(h, {}).get(stratum, {}).get(metric, {}).get("p90")
                    for r in payload["results"]]) for h in HORIZONS]
    return med, p90


def _median(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else float((values[mid - 1] + values[mid]) / 2)


def _lane_match(payload: dict, stratum: str) -> tuple[list, list]:
    lane, matched = [], []
    for h in HORIZONS:
        lanes, fractions = [], []
        for result in payload["results"]:
            block = result["car"]["native"].get(h, {}).get(stratum, {})
            if block.get("lane_agreement") is not None:
                lanes.append(block["lane_agreement"])
            if block.get("matched_fraction") is not None:
                fractions.append(block["matched_fraction"])
        lane.append(_mean(lanes))
        matched.append(_mean(fractions))
    return lane, matched


def _traffic(payload: dict, section: str, metric: str) -> tuple[float | None, float | None]:
    rec, pred = [], []
    for result in payload["results"]:
        traffic = result["traffic"]
        for key in sorted(traffic["predicted"]["native"], key=int):
            p = traffic["predicted"]["native"][key]["sections"][section].get(metric)
            r = traffic["recorded"]["native"].get(key, {}).get("sections", {}) \
                .get(section, {}).get(metric)
            if p is not None:
                pred.append(p)
            if r is not None:
                rec.append(r)
    return _mean(rec), _mean(pred)


def _series(values, digits: int = 1) -> str:
    return " ".join("n/a" if v is None else f"{v:.{digits}f}" for v in values)


def _car_line(payload: dict, label: str, metric: str) -> str:
    h_med, h_p90 = _spread(payload, "hdv", metric)
    c_med, c_p90 = _spread(payload, "cav", metric)
    return (f"- {label} hdv {_series(h_med)} / {_series(h_p90)} | "
            f"cav {_series(c_med)} / {_series(c_p90)}")


def _traffic_line(runs: dict, labels: list[str], section: str) -> str:
    parts = []
    for label in labels:
        payload = runs[label]
        chunks = []
        for metric in TRAFFIC_METRICS:
            rec, pred = _traffic(payload, section, metric)
            chunks.append(f"{TRAFFIC_SHORT[metric]} {_fmt(rec)}->{_fmt(pred)}")
        parts.append(f"{label}[{' '.join(chunks)}]")
    return f"- {section} " + " | ".join(parts)


def _verdict(runs: dict, run1: str) -> str:
    base = ["keep_speed", "pure_physics"]
    if run1 not in runs or any(b not in runs for b in base):
        return "n/a (missing run)"
    run1_med, _ = _spread(runs[run1], "all", "abs_delta_s_m")
    beats = []
    for h_index, h in enumerate(HORIZONS):
        rivals = [_spread(runs[b], "all", "abs_delta_s_m")[0][h_index] for b in base]
        if None in rivals or run1_med[h_index] is None:
            beats.append(None)
        else:
            beats.append(run1_med[h_index] < min(rivals))
    contiguous = 0
    for value in beats:
        if value is True:
            contiguous += 1
        else:
            break
    beat_str = "/".join(("Y" if v is True else ("n" if v is False else "?")) for v in beats)
    up_to = LABELS[contiguous - 1] if contiguous else "0"
    dv_med, _ = _spread(runs[run1], "all", "abs_delta_v_mps")
    dv_ok = sum(1 for h_index, h in enumerate(HORIZONS)
                if dv_med[h_index] is not None and all(
                    _spread(runs[b], "all", "abs_delta_v_mps")[0][h_index] is not None
                    and dv_med[h_index] < _spread(runs[b], "all", "abs_delta_v_mps")[0][h_index]
                    for b in base))
    return (f"{run1} beats median |ds| of keep_speed+pure_physics up to {up_to} s "
            f"(per-horizon {beat_str}); beats both on |dv| at {dv_ok}/{len(HORIZONS)} horizons.")


def _svg(runs: dict, out_path: Path) -> None:
    series = {}
    for label, payload in runs.items():
        series[label] = _spread(payload, "all", "abs_delta_s_m")[0]
    values = [v for s in series.values() for v in s if v is not None]
    vmax = max([1.0] + values)
    width, height = 720, 380
    left, right, top, bottom = 60, 700, 30, 320
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="60" y="20" font-size="14">median |ds| (m, all vehicles) vs horizon (s)</text>']
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
    lines.append(f'<text x="10" y="{top + 6}" font-size="11">{vmax:.0f}</text>')
    for offset, (color, label) in enumerate(zip(COLORS, series)):
        lines.append(f'<text x="{left + 10}" y="{top + 14 + offset * 14}" '
                     f'font-size="11" fill="{color}">{html.escape(label)}</text>')
    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(run_paths: dict, run1: str, out_md: Path, plot_path: Path) -> Path:
    runs = {label: _load(Path(path)) for label, path in run_paths.items()}
    labels = list(runs)
    lines = [
        "# RUN1_EVAL - first trained WM-S model vs baselines (TRAIN, 20 states, offline; no SUMO/GPU)",
        "Status: complete. Delay = normalized seconds sum((1-v/v_free)*dt). Arm native; |ds| m, |dv| m/s; "
        "per-state median/p90 aggregated across 20 states.",
        "Checkpoint wm_s.pt (sha256 97d8f998...); physics replayed from TRAIN_RECEIPT "
        "(open_road, sumo_fidelity, sumo_idm). Horizons 5/10/30/60/120/180/300 s.",
        "",
        f"## Car |ds| median / p90 (native) over horizons {','.join(LABELS)}",
    ]
    for label, payload in runs.items():
        lines.append(_car_line(payload, label, "abs_delta_s_m"))
    lines += ["", f"## Car |dv| median / p90 (native) over horizons {','.join(LABELS)}"]
    for label, payload in runs.items():
        lines.append(_car_line(payload, label, "abs_delta_v_mps"))
    main = [label for label in labels if label in ("keep_speed", "pure_physics", run1)]
    lines += ["", "## Lane agreement (HDV; CAV) and matched fraction (HDV), per horizon"]
    for label in main:
        lane_h, match_h = _lane_match(runs[label], "hdv")
        lane_c, _ = _lane_match(runs[label], "cav")
        lines.append(f"- {label} lane hdv[{_series(lane_h, 2)}] cav[{_series(lane_c, 2)}] "
                     f"matched hdv[{_series(match_h, 2)}]")
    lines += ["", "## Traffic per section, 30 s bins (recorded->predicted, mean over bins/states; "
                  "speed/stopped/queue/delay)"]
    for section in SECTIONS:
        lines.append(_traffic_line(runs, main, section))
    lines += ["", "## Paired PI-native (merge, predicted vs recorded): sign agreement / rank corr"]
    for label in main:
        parts = []
        for metric in PAIR_METRICS:
            pred, rec = _paired(runs[label], metric)
            agree, total = sign_agreement(pred, rec)
            rho = rank_correlation(pred, rec)
            parts.append(f"{metric} {agree}/{total} "
                         f"rho={'n/a' if rho is None else f'{rho:.2f}'} "
                         f"({_fmt(_mean(rec))}->{_fmt(_mean(pred))})")
        lines.append(f"- {label} " + "; ".join(parts))
    lines += ["", f"## Verdict ({run1})", _verdict(runs, run1),
              f"Plot: `{plot_path.name}` (median |ds| all vehicles vs horizon, all specs). "
              "Per-state detail in each run's EVAL.json."]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _svg(runs, plot_path)
    return out_md


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wms_eval.run1_report")
    parser.add_argument("--run", action="append", required=True,
                        help="label=path to an EVAL.json or its run directory")
    parser.add_argument("--run1", default="wms_run1", help="label of the first trained model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot", required=True)
    args = parser.parse_args(argv)
    run_paths = {}
    for item in args.run:
        label, _, path = item.partition("=")
        run_paths[label] = path
    print(build(run_paths, args.run1, Path(args.out), Path(args.plot)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
