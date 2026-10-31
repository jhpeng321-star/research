"""Portable credibility section, CSV receipts and dependency-free SVG diagrams."""

from collections import defaultdict
import csv
from html import escape
import json

from .metrics import WorldIntervals, reliability


def group_rows(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row.get(k) for k in keys)].append(row)
    return result


def summarize_rows(data, bootstrap=10000):
    all_rows = data["distributions"] + data["events"]
    metadata = {r["seed"]: (r["split"], r["demand"], r["penetration"]) for r in all_rows}
    intervals = WorldIntervals(metadata, repetitions=bootstrap)
    summaries, diagrams = [], []
    base_keys = ["split", "horizon_s", "target", "region"]
    for scope, extra in (("pooled", []), ("action", ["action", "reference"]),
                         ("cell_action", ["demand", "penetration", "action", "reference"])):
        keys = base_keys + extra
        for source, fields in (("distributions", ["crps", "fair_crps", "pooled_crps", "cell_crps",
                    "crps_minus_pooled", "crps_minus_cell"]),
                ("events", ["probability", "outcome", "brier", "pooled_brier", "cell_brier",
                            "brier_minus_pooled", "brier_minus_cell"])):
            for values, rows in group_rows(data[source], keys).items():
                meta = dict(zip(keys, values), scope=scope)
                for field in fields:
                    # Baseline means use the identical scorable forecast subset.
                    subset = rows
                    if source == "events" and field in ("pooled_brier", "cell_brier"):
                        subset = [r for r in rows if r["brier"] is not None]
                    summaries.append(dict(meta, metric=field, **intervals.mean(subset, field)))
                if source == "events":
                    good = [r for r in rows if r["probability"] is not None
                            and r["outcome"] is not None]
                    summaries.append(dict(meta, metric="unscorable", mean=len(rows)-len(good),
                                          count=len(rows), worlds=len({r['seed'] for r in rows}),
                                          low=None, high=None, uncertainty_available=False))
                    if scope != "cell_action":
                        diagrams.extend(dict(meta, **r) for r in reliability(good, intervals))
        coverage_keys = keys + ["interval", "nominal_level"]
        for values, rows in group_rows(data["coverage"], coverage_keys).items():
            meta = dict(zip(coverage_keys, values), scope=scope)
            for field in ("covered", "width", "continuous_exchangeable_reference"):
                summaries.append(dict(meta, metric=field, **intervals.mean(rows, field)))
    for scope, extra in (("pooled", []), ("incumbent", ["incumbent"]),
                         ("cell_incumbent", ["demand", "penetration", "incumbent"])):
        keys = ["split", "horizon_s"] + extra
        for values, rows in group_rows(data["decisions"], keys).items():
            meta = dict(zip(keys, values), scope=scope, target="picker", region="")
            for field in ("switched", "expected_saving", "observed_saving"):
                summaries.append(dict(meta, metric=field, **intervals.mean(rows, field)))
    return summaries, diagrams


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reliability_svg(rows, target, model):
    """Fixed 0..1 axes, diagonal, empirical points and whole-world error bars."""
    subset = [r for r in rows if r["scope"] == "pooled" and r["target"] == target]
    panels = sorted({(r["split"], r["horizon_s"]) for r in subset})
    width, panel_h = 760, 300
    height = 75 + panel_h * ((len(panels)+1)//2)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img">',
           f'<title>{escape(model)}: {escape(target)} reliability</title>',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<g font-family="sans-serif" font-size="11" fill="#162638">',
           f'<text x="20" y="24" font-size="18">{escape(model)} · {escape(target)} reliability</text>',
           '<text x="20" y="44">Fixed deciles; dots show forecast vs observed rate; bars: pointwise 95% world CIs.</text>',
           '<text x="20" y="60">Sparse / constant bins may have no CI. A diagonal-looking small panel does not qualify a model.</text>']
    colours = {"upstream": "#176fa7", "merge": "#d16a13", "ramp": "#5b52ac", "": "#176fa7"}
    for index, panel in enumerate(panels):
        ox, oy = 55 + (index % 2)*380, 105 + (index//2)*panel_h
        side = 205
        svg.append(f'<text x="{ox}" y="{oy-18}" font-weight="bold">{escape(panel[0])} · {panel[1]} s</text>')
        for tick in (0, .25, .5, .75, 1):
            x, y = ox+tick*side, oy+(1-tick)*side
            svg.extend([f'<path d="M {x} {oy} V {oy+side} M {ox} {y} H {ox+side}" stroke="#e6ebf0"/>',
                        f'<text x="{x}" y="{oy+side+16}" text-anchor="middle">{tick:g}</text>',
                        f'<text x="{ox-8}" y="{y+4}" text-anchor="end">{tick:g}</text>'])
        svg.append(f'<path d="M {ox} {oy+side} L {ox+side} {oy}" stroke="#8795a5" stroke-dasharray="4 4"/>')
        svg.append(f'<text x="{ox+side/2}" y="{oy+side+34}" text-anchor="middle">Predicted probability</text>')
        svg.append(f'<text transform="translate({ox-35} {oy+side/2}) rotate(-90)" text-anchor="middle">Observed rate</text>')
        points = [r for r in subset if (r["split"], r["horizon_s"]) == panel and r["count"]]
        for row in points:
            x = ox+row["mean_probability"]*side
            y = oy+(1-row["observed_rate"])*side
            colour = colours[row["region"]]
            if row["observed_low"] is not None:
                low = oy+(1-row["observed_low"])*side
                high = oy+(1-row["observed_high"])*side
                svg.append(f'<path d="M {x} {low} V {high}" stroke="{colour}" stroke-width="2"/>')
            label = "selected switches" if target == "selected_preference" else "all pairs"
            title = escape(f"{row['region'] or label} n={row['count']}, worlds={row['worlds']}; "
                           f"pred={row['mean_probability']:.3f}, observed={row['observed_rate']:.3f}")
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colour}"><title>{title}</title></circle>')
        for j, region in enumerate(sorted({r["region"] for r in points})):
            label = "selected switches" if target == "selected_preference" else "all pairs"
            svg.append(f'<text x="{ox+side+10}" y="{oy+20+j*18}" fill="{colours[region]}">{region or label}</text>')
    return "\n".join(svg + ["</g></svg>"])


def credibility_section(model, audit, summaries):
    complete = len(audit.get("scored_worlds", audit["complete_worlds"]))
    lines = [f"## Credibility — {model}", "",
             f"Evidence: {complete} complete K8 worlds; {len(audit['missing_tasks'])} missing tasks. "
             "Development and reserved worlds are separate. Diagnostic pilot only; "
             "this section does not award WM-PHYS qualification.", "",
             "Strict paired P(A beats B) is checked against recorded paired delay order. "
             "Ties are non-wins. Climatology uses same-arm / same-pair development worlds, "
             "excluding the entire target world; cell climatology additionally matches demand and penetration.", "",
              "Reliability: [breakdown](reliability_breakdown.svg), "
              "[pairwise preference](reliability_preference.svg), "
              "[selected switches](reliability_selected_preference.svg). Full action/cell metrics, "
             "unknown-label counts and interval coverage are in `summary.csv`; "
              "decile support in `reliability.csv`.", ""]
    if audit.get("world_selection") == "explicit":
        lines.extend([f"Explicit scored worlds: {audit['scored_worlds']}. "
                      f"Other complete worlds excluded: {audit['excluded_complete_worlds']}.", ""])
    pooled = [r for r in summaries if r["scope"] == "pooled"]
    for split, h in sorted({(r["split"], r["horizon_s"]) for r in pooled}):
        lines.append(f"### {split}, {h} s")
        selected = [r for r in pooled if (r["split"], r["horizon_s"]) == (split, h)
                    and r["metric"] in ("crps", "crps_minus_cell", "brier", "brier_minus_cell",
                                        "switched", "expected_saving", "observed_saving")]
        selected += [r for r in pooled if (r["split"], r["horizon_s"]) == (split, h)
                     and ((r["target"] == "selected_preference" and
                           r["metric"] in ("probability", "outcome")) or r["metric"] == "covered")]
        for row in selected:
            value = "unavailable" if row["mean"] is None else f"{row['mean']:.5g}"
            ci = f" [{row['low']:.5g}, {row['high']:.5g}]" if row["low"] is not None else " (CI unavailable)"
            metric = row['metric']
            if metric == "covered":
                metric = (f"central {row['nominal_level']:.0%} coverage" if row['interval'] == 'central'
                          else "min-max coverage")
            lines.append(f"- {row['target']} {row['region']} {metric}: {value}{ci}; "
                         f"{row['worlds']} worlds, {row['count']} outcomes.")
        lines.append("")
    lines.extend(["Uncertainty: pointwise 95% percentile intervals from 10,000 stratified "
                  "whole-world resamples (seed 20261031); arms and pairs are never independent "
                  "replicates. These CIs condition on the saved ensemble and do not include "
                  "finite-K resampling or shared model error. No simultaneous gate is implied.", "",
                  "Central 50/80/90% empirical-quantile coverage is descriptive at K8. "
                  "Inclusive min–max coverage has continuous exchangeability reference 7/9; "
                  "ties may raise it. Wide coverage alone does not establish useful predictions.", "",
                  "Picker rule remains P(candidate beats incumbent) >= 0.8 and lower expected "
                  "full-team delay. That registered confidence rule is not proof of calibrated "
                   "probabilities; use the existing S-RB-EVAL qualification gate for adoption.", ""])
    lines.extend(["Selected-switch diagnostics evaluate only the action actually selected by "
                  "that rule for each hypothetical incumbent. They use the selected direction's "
                  "strict paired probability, including reverse directions and ties, rather than "
                  "an arbitrary unordered-pair orientation. No-switch cases remain in the picker "
                  "switch-rate/saving summaries but are not recoded as wins or calibration cases. "
                  "All five hypothetical incumbents remain clustered within each world.", "",
                  "Recorded paired-arm outcomes permit a retrospective original-state check; "
                  "these picker savings are not closed-loop switching-policy performance or "
                  "same-state probability identification. Real labels never enter selection. "
                  "Probability/coverage/switch-rate values are fractions; delay scores and savings "
                  "are veh-s. Positive saving helps the selected action; negative score differences "
                  "favour the WM over climatology.", ""])
    return "\n".join(lines)
