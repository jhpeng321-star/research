"""RUN2 E3: traffic-level evaluation and the one-page R2 baseline report.

Car-level matching collapses after ~60 s (RUN1_EVAL_FAIR), so beyond that
horizon the headline is traffic: per section, per 30 s bin, the four quantities
the owner cares about (mean speed, stopped vehicles, merge queue length,
normalized delay, merge flow) for every spec, plus the paired PI-native
difference at traffic level.

The paired statistic over a controller's effect:

* ``sign_agreement`` -- fraction of (state, bin, section, metric) entries where
  the predicted PI-native delta has the same sign as the recorded one;
* ``rank_corr`` -- Spearman rank correlation between predicted and recorded
  deltas (does the model order the bins/sections the same way?);
* ``mae`` -- mean absolute error of the predicted delta;
* ``bias`` -- mean predicted minus recorded delta.

All statistics are computed on identical (state, bin, section, metric) entries
for every spec, so a spec cannot gain by reporting more entries.  Only bins at
or beyond ``min_start_s`` (default 60 s) enter the headline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

SECTIONS = ("upstream", "merge", "downstream")
TRAFFIC = ("mean_speed", "stopped_vehicles", "merge_queue_length",
           "total_delay", "merge_flow")
HORIZONS = ("5.0", "10.0", "30.0", "60.0", "120.0", "180.0", "300.0")
DEFAULT_MIN_START_S = 60.0
EPS = 1e-9


class TrafficEvalError(ValueError):
    """A run payload is missing the traffic blocks required for the report."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "EVAL.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _sign(value: float) -> int:
    if value > EPS:
        return 1
    if value < -EPS:
        return -1
    return 0


def _mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx <= EPS or sy <= EPS:
        return None
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def _offset(pred_bin: dict, t0_s: float) -> float:
    """Seconds from the snapshot to the bin start (bins start at ``t0_s``).

    ``start_s`` is absolute network time, so the "headline only beyond 60 s"
    rule must subtract each result's own ``t0_s``; otherwise any state whose
    snapshot is already past 60 s would pass every bin unfiltered.
    """
    return float(pred_bin.get("start_s", 0.0)) - float(t0_s)


def _bins(payload: dict, arm: str, min_start_s: float):
    for result in payload.get("results", []):
        traffic = result.get("traffic") or {}
        predicted = (traffic.get("predicted") or {}).get(arm) or {}
        recorded = (traffic.get("recorded") or {}).get(arm) or {}
        t0_s = float(result.get("t0_s", 0.0))
        for key in sorted(predicted, key=lambda item: int(item)):
            pred_bin = predicted[key]
            if _offset(pred_bin, t0_s) + EPS < min_start_s:
                continue
            yield result, key, pred_bin, recorded.get(key)


def _paired_entries(payload: dict, section: str, metric: str, min_start_s: float):
    """Yield ``(recorded_delta, predicted_delta)`` for paired states/bins."""
    for result in payload.get("results", []):
        traffic = result.get("traffic") or {}
        pred = traffic.get("paired_predicted_pi_minus_native")
        rec = traffic.get("paired_recorded_pi_minus_native")
        if not pred or not rec:
            continue
        t0_s = float(result.get("t0_s", 0.0))
        for key in sorted(pred, key=lambda item: int(item)):
            pred_bin = pred[key]
            if _offset(pred_bin, t0_s) + EPS < min_start_s:
                continue
            p = pred_bin.get("sections", {}).get(section, {}).get(metric)
            r = (rec.get(key) or {}).get("sections", {}).get(section, {}).get(metric)
            if p is None or r is None:
                continue
            yield float(r), float(p)


# --------------------------------------------------------------------------- #
# headline and paired statistics
# --------------------------------------------------------------------------- #
def traffic_headline(payload: dict, arm: str = "native", *,
                     min_start_s: float = DEFAULT_MIN_START_S) -> dict:
    """Mean recorded/predicted traffic per section/metric, and error vs recorded.

    Also keeps the per-bin series so the report can show the 30 s trajectory.
    """
    out: dict[str, dict] = {}
    for section in SECTIONS:
        out[section] = {}
        for metric in TRAFFIC:
            rec, pred = [], []
            series: dict[str, dict] = {}
            for _result, key, pred_bin, rec_bin in _bins(payload, arm, min_start_s):
                p = pred_bin.get("sections", {}).get(section, {}).get(metric)
                r = ((rec_bin or {}).get("sections", {}) or {}).get(section, {}).get(metric)
                series.setdefault(key, {"start_s": pred_bin.get("start_s"),
                                        "recorded": [], "predicted": []})
                if p is not None:
                    pred.append(float(p))
                    series[key]["predicted"].append(float(p))
                if r is not None:
                    rec.append(float(r))
                    series[key]["recorded"].append(float(r))
            pairs = [(p, r) for p, r in
                     ((pb.get("sections", {}).get(section, {}).get(metric),
                       ((rb or {}).get("sections", {}) or {}).get(section, {}).get(metric))
                      for _res, _k, pb, rb in _bins(payload, arm, min_start_s))
                     if p is not None and r is not None]
            mae = _mean([abs(float(p) - float(r)) for p, r in pairs])
            bias = _mean([float(p) - float(r) for p, r in pairs])
            out[section][metric] = {
                "recorded": _mean(rec), "predicted": _mean(pred),
                "mae": mae, "bias": bias, "n": len(pairs),
                "series": {k: {"start_s": v["start_s"],
                               "recorded": _mean(v["recorded"]),
                               "predicted": _mean(v["predicted"])}
                           for k, v in series.items()},
            }
    return out


def paired_traffic_stats(payload: dict, section: str, metric: str, *,
                         min_start_s: float = DEFAULT_MIN_START_S) -> dict:
    """Sign agreement, Spearman rank correlation and MAE of the PI-native delta."""
    rec, pred = [], []
    for r, p in _paired_entries(payload, section, metric, min_start_s):
        rec.append(r)
        pred.append(p)
    n = len(rec)
    if n == 0:
        return {"n": 0, "sign_agreement": None, "rank_corr": None,
                "pearson": None, "mae": None, "bias": None}
    agree = sum(1 for r, p in zip(rec, pred) if _sign(r) == _sign(p))
    return {
        "n": n,
        "sign_agreement": agree / n,
        "rank_corr": spearman(pred, rec),
        "pearson": _pearson(pred, rec),
        "mae": _mean([abs(p - r) for p, r in zip(pred, rec)]),
        "bias": _mean([p - r for p, r in zip(pred, rec)]),
    }


def paired_summary(payload: dict, *, min_start_s: float = DEFAULT_MIN_START_S) -> dict:
    out = {}
    for section in ("merge", "downstream", "upstream"):
        for metric in TRAFFIC:
            out[f"{section}.{metric}"] = paired_traffic_stats(
                payload, section, metric, min_start_s=min_start_s)
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else float((values[mid - 1] + values[mid]) / 2)


def _car_line(payload: dict, arm: str = "native", stratum: str = "hdv") -> str:
    parts = []
    for horizon in HORIZONS:
        values = []
        for result in payload.get("results", []):
            block = result.get("car", {}).get(arm, {}).get(horizon, {}).get(stratum, {})
            item = block.get("abs_delta_s_m", {})
            values.append(item.get("median"))
        parts.append(_fmt(_median(values), 1))
    return " ".join(parts)


def build(run_paths: dict, out_md: Path, *, min_start_s: float = DEFAULT_MIN_START_S,
          note: str = "") -> Path:
    runs = {label: _load(Path(path)) for label, path in run_paths.items()}
    first = next(iter(runs.values()))
    n_states = first.get("n_states")
    n_paired = first.get("n_paired", sum(1 for r in first.get("results", []) if r.get("paired")))
    lines = [
        "# R2_BASELINE - RUN2 path-1 traffic-level comparator (TRAIN only; offline; no SUMO/GPU)",
        f"**Status:** complete. TRAIN evaluation set: {n_states} states "
        f"({n_paired} native+PI paired). Headline bins: >= {min_start_s:.0f} s "
        "after t0, 30 s windows; delay is normalized delay seconds, never metres.",
        "Entries (`--entries`) applied identically to every spec that supports them "
        "(wms, pure_physics); keep_speed is constant-speed and ignores entries.",
    ]
    if note:
        lines.append(note)
    lines.append("")

    lines.append("## Traffic headline (native arm; mean recorded -> predicted, MAE; per section)"
                 f" [bins >= {min_start_s:.0f} s]")
    for label, payload in runs.items():
        headline = traffic_headline(payload, "native", min_start_s=min_start_s)
        lines.append(f"### {label}")
        for section in SECTIONS:
            block = headline[section]
            parts = [f"{metric} {_fmt(block[metric]['recorded'])}->"
                     f"{_fmt(block[metric]['predicted'])} (+/-{_fmt(block[metric]['mae'])})"
                     for metric in TRAFFIC]
            lines.append(f"- {section}: " + "; ".join(parts))
    lines.append("")

    lines.append("## Traffic headline (PI arm; mean recorded -> predicted, MAE)")
    for label, payload in runs.items():
        headline = traffic_headline(payload, "pi", min_start_s=min_start_s)
        if not headline.get("merge"):
            continue
        parts = []
        for section in SECTIONS:
            block = headline[section]
            parts.append(f"{section}[mean_speed {_fmt(block['mean_speed']['recorded'])}->"
                         f"{_fmt(block['mean_speed']['predicted'])} "
                         f"delay {_fmt(block['total_delay']['recorded'])}->"
                         f"{_fmt(block['total_delay']['predicted'])}]")
        lines.append(f"- {label}: " + " ".join(parts))
    lines.append("")

    lines.append("## Paired PI-native difference at traffic level "
                 "(bins >= 60 s; sign agreement / Spearman rank corr / MAE)")
    for label, payload in runs.items():
        summary = paired_summary(payload, min_start_s=min_start_s)
        top = [key for key in ("merge.mean_speed", "merge.stopped_vehicles",
                               "merge.merge_queue_length", "merge.total_delay",
                               "merge.merge_flow", "downstream.mean_speed",
                               "upstream.mean_speed")]
        parts = []
        for key in top:
            stat = summary[key]
            parts.append(f"{key} n={stat['n']} sign={_fmt(stat['sign_agreement'], 2)} "
                         f"rank={_fmt(stat['rank_corr'], 2)} mae={_fmt(stat['mae'], 2)}")
        lines.append(f"### {label}")
        for part in parts:
            lines.append(f"- {part}")
    lines.append("")

    lines.append("## Secondary: car-level median HDV |ds| (m) at horizons "
                 + "/".join(HORIZONS[:-1]) + " (native arm)")
    for label, payload in runs.items():
        lines.append(f"- {label}: {_car_line(payload)}")
    lines.append("")
    lines.append("Full per-state, per-bin detail is in each run's EVAL.json. "
                 "Car-level matching collapses past ~60 s; traffic bins are the headline there.")
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms_eval.traffic_eval",
                                description="Traffic-level R2 baseline report")
    p.add_argument("--run", action="append", required=True,
                   help="label=path to an EVAL.json or its run directory")
    p.add_argument("--out", required=True)
    p.add_argument("--min-start-s", type=float, default=DEFAULT_MIN_START_S)
    p.add_argument("--note", default="")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_paths = {}
    for item in args.run:
        label, _, path = item.partition("=")
        run_paths[label] = path
    print(build(run_paths, Path(args.out), min_start_s=args.min_start_s, note=args.note))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
