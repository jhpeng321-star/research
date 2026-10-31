"""Render published night-summary cells only; no simulator, model or new statistics.

python -B docs/paper/paperA_cooperation/build_night_figures.py
python -B docs/paper/paperA_cooperation/build_night_figures.py --check
python -B docs/paper/paperA_cooperation/build_night_figures.py --only F4D

Pins are immutable report snapshots, independent of later producer refreshes.
Summary precision and published intervals are preserved; no inferred intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re

from build_figures import ROOT, guarded

PINS = {
    "backup": ("sumocf_run_1c6095d7bfa31670.md",
               "E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md"),
    "calibration": ("wmcalib_report_be56ae652810e360.md",
                    "E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md"),
    "queue": ("wmqueue_report_3fbe64e7da2e44f6.md",
              "E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md"),
    "data_screen": ("wmfix_v2data_3e0bd4cf680cefa2.md",
                    "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md"),
}
STEMS = {"F3B": "F3B_stopped_sumo_backup", "F4N": "F4N_failed_tail_remedies",
         "F4D": "F4D_expanded_data_screen"}
FIELDS = ["figure", "panel", "series", "update", "horizon_s", "estimate", "lo", "hi",
          "n_worlds", "n_missing", "n_contrasts", "unit", "source_path", "source_sha256",
          "snapshot", "source_line", "source_column", "transform", "scope"]
NUM = r"[+−-]?\d+(?:\.\d+)?"


def extract_night():
    rows = []
    sources = {}

    def load(key):
        name, original = PINS[key]
        relative = "source_snapshots/" + name
        data = guarded(ROOT / relative).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        assert sha.startswith(name.rsplit("_", 1)[1].split(".")[0]), name
        sources[key] = dict(source_path=original, source_sha256=sha, snapshot=relative)
        return data.decode("utf-8-sig").splitlines()

    def add(key, fig, panel, series, horizon, estimate, line, column, *,
            update="", lo="", hi="", worlds="", missing="", contrasts="", scope=""):
        row = dict(figure=fig, panel=panel, series=series, update=update, horizon_s=horizon,
                   estimate=estimate, lo=lo, hi=hi, n_worlds=worlds, n_missing=missing,
                   n_contrasts=contrasts, unit="veh-s", **sources[key], source_line=line,
                   source_column=column, transform="identity at published summary precision",
                   scope=scope)
        if lo != "":
            assert float(lo) <= float(estimate) <= float(hi)
        rows.append(row)

    backup = load("backup")
    assert "STOPPED" in backup[2]
    for line, text in enumerate(backup, 1):
        if not re.match(r"\| u(?:3|7|12) \|", text):
            continue
        cells = [s.strip() for s in text.strip("|").split("|")]
        update = int(cells[0][1:])
        for column, (horizon, reference) in enumerate(
                ((600, "MAPPO"), (900, "MAPPO"), (600, "CTRL0"), (900, "CTRL0")), 1):
            match = re.fullmatch(rf"({NUM}) \[({NUM}),({NUM})\]", cells[column])
            assert match, cells[column]
            estimate, lo, hi = (float(s) for s in match.groups())
            add("backup", "F3B", "cost", reference, horizon, estimate, line, column,
                update=update, lo=lo, hi=hi, worlds=24,
                scope="fixed DEV; learned-minus-reference; control-start clock; exploratory")
    coverage = {3: (2, 1), 7: (2, 1), 12: (3, 0)}
    # Each final-section line publishes the interaction estimate and its interval.
    for line, text in enumerate(backup, 1):
        match = re.match(rf"- u(3|7|12): C_G=({NUM}),95%CI\[({NUM}),({NUM})\]", text)
        if match:
            update = int(match[1])
            estimate, lo, hi = (float(match[i]) for i in (2, 3, 4))
            worlds, missing = coverage[update]
            # Independently bind coverage to the nearby published share/missing fields.
            nearby = " ".join(backup[line - 1:line + 5])
            assert f"/{worlds};" in nearby or f"/{worlds}; " in nearby
            assert f"{missing}missing group" in nearby
            add("backup", "F3B", "interaction", "deployed pair C_G", 600, estimate, line,
                "C_G,95%CI; coverage from adjacent lines", update=update, lo=lo, hi=hi,
                worlds=worlds, missing=missing,
                scope="fixed pairs; MAPPO-mean background/learned continuation; NOT native audit")

    calibration = load("calibration")
    assert "NOT USABLE" in calibration[1]
    for line, text in enumerate(calibration, 1):
        if text.startswith("## SPEEDHEAD"):
            break
        if not re.match(r"\| (120|300|600) \|", text):
            continue
        cells = [s.strip() for s in text.strip("|").split("|")]
        if cells[1] not in ("raw WM2", "arm-mean (no WM)", "ridge +start", "median ridge +start"):
            continue
        series = {"raw WM2": "Raw WM2", "arm-mean (no WM)": "Arm mean",
                  "ridge +start": "Selected calibrator", "median ridge +start": "Selected calibrator"}[cells[1]]
        add("calibration", "F4N", "grouped_cv", series, int(cells[0]),
            float(cells[2].split("/")[0].strip().replace(",", "")), line, "MAE (first tuple cell)",
            worlds=40, contrasts=120, scope="world-grouped CV; selected calibrator CV optimistic; final gate FAIL")

    queue = load("queue")
    assert "KILL" in queue[2]
    for line, text in enumerate(queue, 1):
        match = re.fullmatch(
            r"- (120|300|600) s: queue (\d+)/[^;]+; state-only (\d+)/[^;]+; arm-mean (\d+)/.*", text)
        if match:
            for series, value in zip(("Queue prototype", "State only", "Arm mean"), match.groups()[1:]):
                add("queue", "F4N", "queue_test", series, int(match[1]), float(value), line,
                    series + " MAE (first tuple cell)", worlds=8, contrasts=48,
                    scope="complete reserved TRAIN panel; four cached-WM arms; fixed prototype KILLED")
    assert len([r for r in rows if r["figure"] == "F3B"]) == 15
    assert len([r for r in rows if r["figure"] == "F4N"]) == 18
    assert {r["update"] for r in rows if r["figure"] == "F3B"} == {3, 7, 12}
    data_screen = load("data_screen")
    final = " ".join(data_screen[:42])
    assert "not a full-gate PASS/FAIL" in final
    assert "step10 failed; steps 20/30 not scored" in final
    assert "Full52-world panel,104 contrasts/horizon" in final
    for line, text in enumerate(data_screen[:42], 1):
        match = re.search(rf"\*\*(120|300)s({NUM}) -> ({NUM})\*\*", text)
        if match:
            for update, value in zip((0, 10), match.groups()[1:]):
                add("data_screen", "F4D", "data_screen", f"Step {update}", int(match[1]),
                    float(value), line, f"paired-J MAE; step{update}", update=update,
                    worlds=52, contrasts=104,
                    scope="TRAIN panel; amended step10 early STOP; steps20/30 unscored; NOT full-gate verdict")
    assert len([r for r in rows if r["figure"] == "F4D"]) == 4
    return rows, sources


def serialized(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def render_night(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, gray = "#0072B2", "#D55E00", "#555555"

    def save(fig, figure):
        for suffix in ("png", "pdf"):
            fig.savefig(guarded(ROOT / "figures" / (STEMS[figure] + "." + suffix), output=True),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(bottom=.28, top=.80, left=.10, right=.98, wspace=.48)
    for horizon, offset, color, marker in ((600, -.10, orange, "o"), (900, .10, blue, "s")):
        cells = [r for r in rows if r["figure"] == "F3B" and r["panel"] == "cost"
                 and r["series"] == "MAPPO" and r["horizon_s"] == horizon]
        y = np.array([r["estimate"] for r in cells])
        axes[0].errorbar(np.arange(3) + offset, y,
                         yerr=[y - [r["lo"] for r in cells], [r["hi"] for r in cells] - y],
                         fmt=marker, color=color, capsize=3, label=f"J{horizon}")
    axes[0].axhline(0, color=gray, lw=.8)
    axes[0].set(title="(a) Traffic cost: no primary-horizon win", ylabel="Learned − MAPPO delay (veh-s)",
                xlabel="Optimizer checkpoint (fixed DEV panel)", xticks=range(3), xticklabels=["u3", "u7", "u12"])
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    cells = [r for r in rows if r["panel"] == "interaction"]
    y = np.array([r["estimate"] for r in cells])
    axes[1].errorbar(range(3), y, yerr=[y - [r["lo"] for r in cells], [r["hi"] for r in cells] - y],
                     fmt="o", color=gray, capsize=3)
    axes[1].axhline(0, color=gray, lw=.8)
    axes[1].set(title="(b) Sparse deployed-pair interaction", ylabel="$C_G$ (veh-s)",
                xlabel="Checkpoint; eligible / missing worlds", xticks=range(3),
                xticklabels=["u3\n2 / 1", "u7\n2 / 1", "u12\n3 / 0"])
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("SUMO-trained backup stopped: useful cooperation not established", fontsize=11, y=.98)
    fig.text(.10, .04, "Source: SUMOCF_RUN.md final report · 24 DEV worlds · copied 95% world-bootstrap CIs\n"
             "Pair diagnostic: MAPPO-mean background, learned continuation; not native-release audit. Different y-axis scales.",
             fontsize=8)
    save(fig, "F3B")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    fig.subplots_adjust(bottom=.31, top=.77, left=.09, right=.98, wspace=.36)
    specs = [("grouped_cv", ("Raw WM2", "Arm mean", "Selected calibrator"),
              "(a) Calibration: NOT USABLE", "40-world grouped CV; 120 contrasts/clock"),
             ("queue_test", ("Queue prototype", "State only", "Arm mean"),
              "(b) Queue/service prototype: KILL", "8 reserved TRAIN worlds; 48 contrasts/clock")]
    for ax, (panel, series, title, scope) in zip(axes, specs):
        for offset, label, color in zip((-.24, 0, .24), series, (orange, blue, gray)):
            cells = [r for r in rows if r["panel"] == panel and r["series"] == label]
            assert [r["horizon_s"] for r in cells] == [120, 300, 600]
            ax.bar(np.arange(3) + offset, [r["estimate"] for r in cells], width=.22, color=color, label=label)
        ax.set(title=title, xlabel="Outcome horizon after control start (s)",
               ylabel="Paired-effect MAE (veh-s)", xticks=range(3), xticklabels=["120", "300", "600"])
        ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.suptitle("Long-tail remedies failed their registered gates", fontsize=11, y=.98)
    fig.text(.09, .02, "(a) 40-world grouped CV; 120 contrasts/clock. (b) 8 reserved TRAIN worlds; 48 contrasts/clock.\n"
             "Sources: WMCALIB.md / WMQUEUE.md · point estimates only; no new CIs\n"
             "Separate contracts. Calibrator CV gains did not qualify on SPEEDHEAD; neither panel measures $C_G$ fidelity.",
             fontsize=8)
    save(fig, "F4N")


def check_night(rows, sources):
    from PIL import Image

    artifacts = {}
    for figure, stem in STEMS.items():
        data_path = guarded(ROOT / "figures" / (stem + ".csv"))
        assert data_path.read_bytes() == serialized([r for r in rows if r["figure"] == figure])
        for suffix in ("csv", "png", "pdf"):
            relative = "figures/" + stem + "." + suffix
            path = guarded(ROOT / relative)
            data = path.read_bytes()
            if suffix == "png":
                with Image.open(path) as image:
                    assert image.width > 1500 and image.height > 700
                    image.verify()
            elif suffix == "pdf":
                assert data.startswith(b"%PDF")
            artifacts[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    result = {"status": "PASS", "source_cells": len(rows), "sources": sources, "artifacts": artifacts,
              "scope": "Published summary transcription, copied CIs, immutable input hashes, CSV/image integrity; no new statistics",
              "precision": "Published rounded summaries, not an independently rerun per-world analysis"}
    guarded(ROOT / "night_figure_validation.json", output=True).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def render_data_screen(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    fig.subplots_adjust(left=.10, right=.98, bottom=.29, top=.76, wspace=.36)
    for ax, horizon, title in zip(axes, (120, 300),
                                 ("(a) Short clock: slight improvement",
                                  "(b) Long clock: worsened")):
        cells = [r for r in rows if r["figure"] == "F4D" and r["horizon_s"] == horizon]
        assert [r["update"] for r in cells] == [0, 10]
        bars = ax.bar([0, 1], [r["estimate"] for r in cells],
                      color=["#555555", "#D55E00"], width=.55)
        ax.bar_label(bars, labels=[f"{r['estimate']:.2f}" for r in cells], padding=4)
        ax.set(title=title, ylabel="Paired-effect MAE (veh-s)",
               xlabel=f"Outcome horizon: {horizon} s",
               xticks=[0, 1], xticklabels=["Entry-path baseline", "Trained step 10"])
        ax.set_ylim(0, max(r["estimate"] for r in cells) * 1.23)
    fig.suptitle("Expanded-data merge head: amended step-10 screen stopped", fontsize=11, y=.97)
    fig.text(.10, .035,
             "Source: WMFIX_PILOT_V2DATA.md FINAL section · 52 TRAIN worlds · 104 contrasts/clock\n"
             "Both horizons had to improve. Steps 20/30 unscored: not a full-gate PASS/FAIL.\n"
             "Published point estimates; no invented CIs. Separate y-axis scales, both starting at zero.",
             fontsize=8)
    for suffix in ("png", "pdf"):
        path = guarded(ROOT / "figures" / (STEMS["F4D"] + "." + suffix), output=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", choices=["F4D"], help="Add late panel without rebuilding inherited artwork")
    args = parser.parse_args()
    rows, sources = extract_night()
    if not args.check:
        for figure, stem in STEMS.items():
            if args.only and figure != args.only:
                continue
            guarded(ROOT / "figures" / (stem + ".csv"), output=True).write_bytes(
                serialized([r for r in rows if r["figure"] == figure]))
        if not args.only:
            render_night(rows)
        render_data_screen(rows)
    print(json.dumps(check_night(rows, sources), indent=2))


if __name__ == "__main__":
    main()
