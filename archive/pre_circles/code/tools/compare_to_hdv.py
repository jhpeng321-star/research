"""Compare batch-eval results against the All-HDV baseline ("paper win" rule).

Reads ``runs/eval/summary_all.json`` (from ``tools/batch_eval.py``) and one or
more All-HDV baseline JSON files (from ``evaluate_sumo_baseline.py --output``),
judges every (baseline, density, cav_penetration) cell, and writes a markdown
diagnosis table.

Win rule (fair-evaluation task):
  * efficiency improved — mean TT lower OR throughput higher than All-HDV
  * safety not worse    — TTC, collisions and hard braking all <= All-HDV means
  * significant         — the efficiency delta exceeds the All-HDV seed std

Example
-------
python tools/compare_to_hdv.py \
    --summary runs/eval/summary_all.json \
    --baseline runs/eval/all_hdv_light.json runs/eval/all_hdv_medium.json \
    --out runs/eval/diagnosis.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "runs" / "eval"


def _num(value, default: float = 0.0) -> float:
    """Coerce None/NaN/unparseable values to a default (old rows lack metrics)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def _is_missing(value) -> bool:
    """True when a metric value is absent (None/NaN/unparseable).

    Missing values must never be coerced to 0 for the win gates: a fake 0
    makes any safety comparison pass and any efficiency delta look positive.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return v != v


_SAFETY_KEYS = ("ttc", "collisions", "hard_braking")


def judge_cell(policy_metrics: dict, hdv_metrics: dict) -> dict:
    """Judge one policy cell against the All-HDV baseline.

    ``policy_metrics``: tt_mean, throughput_mean, ttc, collisions, hard_braking.
    ``hdv_metrics``:    the same plus tt_std and throughput_std (seed-to-seed).

    Returns win/efficiency_improved/safety_ok/significant booleans, the
    efficiency deltas, and human-readable reasons. Missing safety or
    efficiency data (None/NaN/absent keys, e.g. old summary rows) fails the
    corresponding gate instead of being silently coerced to 0.
    """
    tt_missing = _is_missing(policy_metrics.get("tt_mean")) or _is_missing(
        hdv_metrics.get("tt_mean")
    )
    thr_missing = _is_missing(policy_metrics.get("throughput_mean")) or _is_missing(
        hdv_metrics.get("throughput_mean")
    )
    safety_missing = any(
        _is_missing(metrics.get(key))
        for metrics in (policy_metrics, hdv_metrics)
        for key in _SAFETY_KEYS
    )

    tt_delta = (
        float("nan")
        if tt_missing
        else _num(hdv_metrics["tt_mean"]) - _num(policy_metrics["tt_mean"])
    )
    thr_delta = (
        float("nan")
        if thr_missing
        else _num(policy_metrics["throughput_mean"]) - _num(hdv_metrics["throughput_mean"])
    )

    efficiency_improved = (not tt_missing and tt_delta > 0) or (not thr_missing and thr_delta > 0)
    safety_ok = not safety_missing and (
        _num(policy_metrics["ttc"]) <= _num(hdv_metrics["ttc"])
        and _num(policy_metrics["collisions"]) <= _num(hdv_metrics["collisions"])
        and _num(policy_metrics["hard_braking"]) <= _num(hdv_metrics["hard_braking"])
    )
    significant = (not tt_missing and tt_delta > _num(hdv_metrics.get("tt_std"))) or (
        not thr_missing and thr_delta > _num(hdv_metrics.get("throughput_std"))
    )
    win = efficiency_improved and safety_ok and significant

    if _num(hdv_metrics.get("tt_std")) == 0.0 and _num(hdv_metrics.get("throughput_std")) == 0.0:
        # Zero seed std means a single-seed (or degenerate) All-HDV baseline:
        # the significance gate then passes on any positive delta, so the win
        # verdict is unreliable.
        print(
            "WARNING: All-HDV baseline seed std is 0 (single-seed baseline?); "
            "significance verdict unreliable",
            file=sys.stderr,
        )

    reasons = []
    if tt_missing and thr_missing:
        reasons.append("efficiency data: missing")
    elif not efficiency_improved:
        reasons.append("no efficiency gain (neither TT nor throughput better)")
    if safety_missing:
        reasons.append("safety data: missing")
    elif not safety_ok:
        reasons.append("safety regression (TTC/collisions/hard braking worse)")
    if efficiency_improved and not significant:
        reasons.append("gain within All-HDV seed std (noise)")
    if win:
        reasons.append("paper win")

    return {
        "win": win,
        "efficiency_improved": efficiency_improved,
        "safety_ok": safety_ok,
        "significant": significant,
        "tt_delta": tt_delta,
        "throughput_delta": thr_delta,
        "reasons": reasons,
    }


def load_hdv_baselines(paths) -> dict:
    """Load evaluate_sumo_baseline multi-seed JSONs, keyed by density."""
    baselines = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agg = data["aggregate"]
        baselines[data["density"]] = {
            "tt_mean": agg["all_tt_mean"]["mean"],
            "tt_std": agg["all_tt_mean"]["std"],
            "throughput_mean": agg["throughput_veh_per_hour"]["mean"],
            "throughput_std": agg["throughput_veh_per_hour"]["std"],
            "ttc": agg["ttc_events"]["mean"],
            "collisions": agg["collisions"]["mean"],
            "hard_braking": agg["hard_braking_events"]["mean"],
        }
    return baselines


def _mean_present(values) -> float:
    vals = [float(v) for v in values if v is not None and float(v) == float(v)]
    return statistics.fmean(vals) if vals else float("nan")


def group_summary_rows(rows) -> dict:
    """Group summary_all.json rows by (baseline, density, pen); mean over seeds."""
    groups = defaultdict(list)
    for row in rows:
        if not row.get("ok"):
            continue
        key = (row["baseline"], row["density"], row.get("cav_penetration", 0.0))
        groups[key].append(row)

    cells = {}
    for key, members in groups.items():
        cells[key] = {
            "seeds": len(members),
            "tt_mean": _mean_present(m.get("all_mean") for m in members),
            "throughput_mean": _mean_present(m.get("throughput_veh_per_hour") for m in members),
            "ttc": _mean_present(m.get("ttc_events") for m in members),
            "collisions": _mean_present(m.get("collisions") for m in members),
            "hard_braking": _mean_present(m.get("hard_braking_events") for m in members),
        }
    return cells


def build_diagnosis(cells: dict, hdv_baselines: dict) -> str:
    """Render the markdown diagnosis table."""
    lines = [
        "# Policy vs All-HDV diagnosis",
        "",
        "Win rule: efficiency (TT or throughput) improved "
        "**and** safety (TTC + collisions + hard braking) not worse "
        "**and** the efficiency delta exceeds the All-HDV seed std.",
        "",
        "| density | baseline | pen | seeds | TT mean (s) | ΔTT (s) | TT std (s) | "
        "throughput (veh/h) | TTC | collisions | hard braking | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for (baseline, density, pen), cell in sorted(cells.items()):
        hdv = hdv_baselines.get(density)
        if hdv is None:
            lines.append(
                f"| {density} | {baseline} | {pen:.2f} | {cell['seeds']} | "
                f"{cell['tt_mean']:.2f} | n/a | n/a | "
                f"{cell['throughput_mean']:.0f} | {cell['ttc']:.0f} | "
                f"{cell['collisions']:.1f} | {cell['hard_braking']:.0f} | "
                f"NO_BASELINE |"
            )
            continue
        r = judge_cell(cell, hdv)
        verdict = "**PAPER WIN**" if r["win"] else f"no win ({'; '.join(r['reasons'])})"
        lines.append(
            f"| {density} | {baseline} | {pen:.2f} | {cell['seeds']} | "
            f"{cell['tt_mean']:.2f} | {r['tt_delta']:+.2f} | {hdv['tt_std']:.2f} | "
            f"{cell['throughput_mean']:.0f} | "
            f"{cell['ttc']:.0f}/{hdv['ttc']:.0f} | "
            f"{cell['collisions']:.1f}/{hdv['collisions']:.1f} | "
            f"{cell['hard_braking']:.0f}/{hdv['hard_braking']:.0f} | {verdict} |"
        )
    lines.append("")
    lines.append(
        "Safety cells show `policy/All-HDV` means. ΔTT is All-HDV minus policy "
        "(positive = policy faster)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUT_DIR / "summary_all.json",
        help="batch_eval summary JSON (default: runs/eval/summary_all.json).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        nargs="+",
        required=True,
        help="All-HDV baseline JSON(s) from evaluate_sumo_baseline.py --output.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "diagnosis.md",
        help="Markdown output path (default: runs/eval/diagnosis.md).",
    )
    args = parser.parse_args()

    rows = json.loads(args.summary.read_text(encoding="utf-8")).get("results", [])
    cells = group_summary_rows(rows)
    hdv_baselines = load_hdv_baselines(args.baseline)
    md = build_diagnosis(cells, hdv_baselines)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out} ({len(cells)} cells, baselines: {sorted(hdv_baselines)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
