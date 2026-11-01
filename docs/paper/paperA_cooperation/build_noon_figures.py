"""Final published WM-PHYS summary transcription; no simulation or statistics.

python -B docs/paper/paperA_cooperation/build_noon_figures.py
python -B docs/paper/paperA_cooperation/build_noon_figures.py --check
Only the pinned final report is read; historical artwork is never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re

from build_figures import ROOT, guarded

SNAPSHOT = "source_snapshots/wmphys_morning_44ad6b2a1dd2cc09.md"
SOURCE = "E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md"
SHA = "44ad6b2a1dd2cc09277f8023357aed451ce0b0cdaebe15d3a5c1322746177f77"
STEM = "F4P_final_wmphys_diagnostics"
FIELDS = ["figure", "panel", "series", "horizon_s", "estimate", "unit", "lo", "hi",
          "interval_status", "n_worlds", "source_path", "source_sha256", "snapshot",
          "source_line", "source_field", "transform", "scope"]


def extract_noon():
    data = guarded(ROOT / SNAPSHOT).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA
    lines = data.decode("utf-8-sig").splitlines()
    assert "report — final" in lines[0]
    assert "Registered switching rule FAIL" in lines[1]
    assert any("FULL-v4:reserved n=8, total complete=20/20" in s for s in lines)
    rows = []

    def match(pattern):
        found = [(i, re.search(pattern, s)) for i, s in enumerate(lines, 1)
                 if re.search(pattern, s)]
        assert len(found) == 1, pattern
        return found[0]

    def add(panel, series, horizon, value, unit, line, field):
        rows.append(dict(figure="F4P", panel=panel, series=series, horizon_s=horizon,
                         estimate=float(value), unit=unit, lo="", hi="",
                         interval_status="full-world CI NA; singleton cells; no invented whiskers",
                         n_worlds=8, source_path=SOURCE, source_sha256=SHA, snapshot=SNAPSHOT,
                         source_line=line, source_field=field,
                         transform="identity at published summary precision",
                         scope="reserved-world final point diagnostic; not student efficacy or C_G fidelity"))

    line, m = match(r"^- Same-world absolute delay CRPS120/300/600:WM2 ([\d/]+); FULL-v4 ([\d/]+)veh-s")
    for series, values in zip(("WM2", "FULL-v4"), m.groups()):
        for horizon, value in zip((120, 300, 600), values.split("/")):
            add("delay_crps", series, horizon, value, "veh-s", line, f"{series} CRPS{horizon}")

    line, m = match(r"^- FULL-v4:reserved n=8, total complete=20/20; ΔCRPS120/300/600=([\d/-]+)veh-s; Δ300s rate-distance=([\d.]+)pp")
    for horizon, value in zip((120, 300, 600), m[1].split("/")):
        add("paired_crps_change", "FULL-v4 minus WM2", horizon, value, "veh-s", line, f"paired CRPS{horizon}")
    add("rate_distance_change", "FULL-v4 minus WM2", 300, m[2], "pp", line, "paired breakdown-distance")

    line, m = match(r"^- Same-world upstream rates120/300/600 \(%\):real ([\d./]+); WM2 ([\d./]+); FULL-v4 ([\d./]+)")
    for series, values in zip(("Recorded", "WM2", "FULL-v4"), m.groups()):
        # Only the registered 300s breakdown condition is plotted here.
        add("breakdown_rate", series, 300, values.split("/")[1], "%", line, f"{series} upstream300 rate")

    line, m = match(r"^- HDV60 mean-world-median error difference \(m\): (-?[\d.]+);95%CI NA; common HDV/arm endpoints=(\d+)")
    assert m[2] == "1237"
    add("hdv_position_change", "FULL-v4 minus WM2", 60, m[1], "m", line, "HDV60; 1237 common HDV-arm endpoints")

    for arm in ("mappo", "pi", "circles"):
        line, m = match(rf"^- Expected600s {arm}−native delay=(-?[\d.]+) vs recorded=(-?[\d.]+)veh-s")
        for series, value in zip(("FULL-v4", "Recorded"), m.groups()):
            add("controller_effect", f"{arm}:{series}", 600, value, "veh-s", line, f"{arm} minus native; {series}")

    triple = r"(\d+(?:\.\d+)?/\d+(?:\.\d+)?/\d+(?:\.\d+)?)"
    line, m = match(r"^- Same-source selected0.8calls observed win%=" + triple +
                    " vs forecast%=" + triple + "; range coverage%=" + triple)
    for panel, series, values in zip(("selected_reliability", "selected_reliability", "range_coverage"),
                                    ("Observed wins", "Forecast probability", "Absolute-delay range"), m.groups()):
        for horizon, value in zip((120, 300, 600), values.split("/")):
            add(panel, series, horizon, value, "%", line, f"{series} at {horizon}s")
    assert len(rows) == 29
    assert all(r["estimate"] >= 0 for r in rows if r["panel"] in ("delay_crps", "breakdown_rate"))
    return rows, {"source_path": SOURCE, "source_sha256": SHA, "snapshot": SNAPSHOT}


def serialized(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def render_noon(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, gray = "#0072B2", "#D55E00", "#555555"
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.20, hspace=.67, wspace=.38)

    def cells(panel, series):
        return [r["estimate"] for r in rows if r["panel"] == panel and r["series"] == series]

    x = np.arange(3)
    ax = axes[0, 0]
    for offset, label, color in ((-.18, "WM2", blue), (.18, "FULL-v4", orange)):
        ax.bar(x + offset, cells("delay_crps", label), width=.34, label=label, color=color)
    ax.set(title="(a) Better delay distribution (point scores)", ylabel="Delay CRPS (veh-s; lower better)",
           xlabel="Prediction horizon (s)", xticks=x, xticklabels=["120", "300", "600"])
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    rates = [cells("breakdown_rate", label)[0] for label in ("Recorded", "WM2", "FULL-v4")]
    bars = ax.bar(x, rates, width=.55, color=(gray, blue, orange))
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set(title="(b) Breakdown condition FAILS", ylabel="300 s upstream breakdown rate (%)",
           xticks=x, xticklabels=["Recorded", "WM2", "FULL-v4"], ylim=(0, 108))
    ax.text(.5, -.24, "Rate-distance +5.63 pp vs WM2 (worse)", transform=ax.transAxes,
            ha="center", fontsize=8, color=gray)

    ax = axes[1, 0]
    for offset, series, color in ((-.18, "Recorded", gray), (.18, "FULL-v4", orange)):
        values = [cells("controller_effect", f"{arm}:{series}")[0] for arm in ("mappo", "pi", "circles")]
        ax.bar(x + offset, values, width=.34, label=series, color=color)
    ax.axhline(0, color=gray, lw=.8)
    ax.set(title="(c) Correct aggregate effect signs", ylabel="600 s arm minus native delay (veh-s)",
           xticks=x, xticklabels=["MAPPO", "PI", "CIRCLES"], ylim=(-2700, 6500))
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[1, 1]
    for offset, series, color in ((-.18, "Forecast probability", orange), (.18, "Observed wins", blue)):
        ax.bar(x + offset, cells("selected_reliability", series), width=.34, label=series, color=color)
    ax.set(title="(d) Selected calls remain overconfident", ylabel="Selected-call win / forecast (%)",
           xlabel="Prediction horizon (s)", xticks=x, xticklabels=["120", "300", "600"], ylim=(0, 120))
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7)

    fig.suptitle("Final WM-PHYS FULL-v4: better prediction is not action qualification", fontsize=12, y=.98)
    fig.text(.10, .92, "Final reserved-world diagnostics · 8 worlds · complete K8 futures · source 22cc32f6", fontsize=9)
    fig.text(.10, .035,
             "Source: WMPHYS_MORNING.md final report; published point precision only, no invented whiskers.\n"
             "Full-world CIs NA (singleton cells); earlier numeric CIs retracted. HDV60 error change: −15.26 m.\n"
             "All switching conditions required: final FAIL. Selected-call diagnostics are retrospective, not closed-loop efficacy.\n"
             "Different metrics/units and estimands; correct aggregate effect signs do not establish option calibration or $C_G$ fidelity.",
             fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(guarded(ROOT / "figures" / f"{STEM}.{suffix}", output=True), dpi=300, bbox_inches="tight")
    plt.close(fig)


def check_noon(rows, source):
    from PIL import Image

    assert guarded(ROOT / "figures" / f"{STEM}.csv").read_bytes() == serialized(rows)
    artifacts = {}
    for suffix in ("csv", "png", "pdf"):
        relative = f"figures/{STEM}.{suffix}"
        path = guarded(ROOT / relative)
        data = path.read_bytes()
        if suffix == "png":
            with Image.open(path) as image:
                assert image.width > 2000 and image.height > 1500
                image.verify()
        elif suffix == "pdf":
            assert data.startswith(b"%PDF")
        artifacts[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    result = {"status": "PASS", "source_cells": len(rows), "source": source, "artifacts": artifacts,
              "scope": "Immutable final-report summary transcription and CSV/image integrity; no new statistics",
              "intervals": "Full-world CI NA; source absolutes and paired changes independently rounded"}
    guarded(ROOT / "noon_figure_validation.json", output=True).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows, source = extract_noon()
    if not args.check:
        guarded(ROOT / "figures" / f"{STEM}.csv", output=True).write_bytes(serialized(rows))
        render_noon(rows)
    print(json.dumps(check_noon(rows, source), indent=2))


if __name__ == "__main__":
    main()
