"""Render existing evening reports/CSVs, without inference or resampling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA, FIG = ROOT / "data", ROOT / "figures"
INPUTS = {
    "pidiag.source.md": "E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md",
    "calibration.source.md": "E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md",
    "switch.source.md": "E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md",
    "recorded_winners.source.csv": "E:/agent-work/c4-c1switch/docs/paper_results/c1switch/recorded_world_winners.csv",
    "step0b.source.csv": "E:/agent-work/c4-c1switch/docs/paper_results/c1switch/step0b_decisions.csv",
    "floor.source.md": "E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR.md",
    "floor.source.csv": "E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR_error_horizon.csv",
}
DENIED = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def snapshot():
    blobs = {}
    for name, filename in INPUTS.items():
        path = Path(filename).resolve()
        if any(term in str(path).lower() for term in DENIED):
            raise ValueError(path)
        blobs[name] = path.read_bytes()
    manifest = {}
    for name, blob in blobs.items():
        (DATA / name).write_bytes(blob)
        manifest[name] = dict(source_file=INPUTS[name], sha256=sha(blob), bytes=len(blob))
    (DATA / "EVENING_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify():
    manifest = json.loads((DATA / "EVENING_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == set(INPUTS)
    for name, info in manifest.items():
        blob = (DATA / name).read_bytes()
        assert info == dict(source_file=INPUTS[name], sha256=sha(blob), bytes=len(blob)), name


def text(name):
    return (DATA / name).read_text(encoding="utf-8-sig")


def rows(name):
    return list(csv.DictReader(io.StringIO(text(name))))


def write_csv(name, records):
    with (DATA / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def contrast(cell):
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", cell.replace("−", "-"))
    assert len(numbers) == 3, cell
    estimate, lo, hi = map(float, numbers)
    assert lo <= estimate <= hi
    return dict(estimate=estimate, lo=lo, hi=hi)


def extract():
    report = text("pidiag.source.md")
    original, shifted = report.split("## Genuine-history test", 1)
    records = []
    for panel, section in [("Original snapshot starts", original), ("Shifted full-history starts", shifted)]:
        line = next(line for line in section.splitlines() if line.startswith("| PI / WM2 |"))
        cells = [s.strip() for s in line.strip("|").split("|")]
        for h, cell in zip([60, 120], cells[2:4]):
            records.append(dict(panel=panel, contrast="WM2 minus BB-traj", horizon_s=h,
                                **contrast(cell), units="m", source_file=INPUTS["pidiag.source.md"],
                                source_selector=f"{panel};PI / WM2;paired delta {h}s"))
    history = report.split("WM2 PI full60−snapshot **", 1)[1].split(" m**", 1)[0]
    records.append(dict(panel="Within identical shifted starts", contrast="Full history minus snapshot (WM2)",
                        horizon_s=60, **contrast(history), units="m",
                        source_file=INPUTS["pidiag.source.md"], source_selector="History attribution;WM2 PI full60-snapshot;60s"))
    write_csv("pi_diagnosis.csv", records)

    report = text("calibration.source.md")
    cv, test = report.split("## SPEEDHEAD evaluation-only", 1)
    test = test.split("## 963887 MAPPO stress case", 1)[0]
    records = []
    for split, section in [("Grouped CV", cv), ("SPEEDHEAD transfer", test)]:
        for line in section.splitlines():
            if not re.match(r"\| (120|300|600) \|", line):
                continue
            h, signal, values = [s.strip() for s in line.strip("|").split("|")]
            if signal not in ("raw WM2", "arm-mean (no WM)", "ridge +start", "median ridge +start"):
                continue
            mae, sign, pearson, spearman, order = map(float, values.replace(",", "").split("/"))
            records.append(dict(split=split, horizon_s=int(h), signal=signal, mae_veh_s=mae,
                                sign_accuracy=sign, pearson=pearson, spearman=spearman, arm_order=order,
                                source_file=INPUTS["calibration.source.md"],
                                source_selector=f"{split};{h}s;{signal}"))
    assert len(records) == 18
    write_csv("calibration.csv", records)

    records = []
    sources = [("Recorded worlds", "recorded_winners.source.csv", "clock_s", "winner", 600),
               ("Same-state decisions", "step0b.source.csv", "horizon_s", "unique_winner", 60),
               ("Same-state decisions", "step0b.source.csv", "horizon_s", "unique_winner", 90)]
    for panel, source, clock, winner, h in sources:
        selected = [r for r in rows(source) if int(r[clock]) == h]
        counts = Counter((r[winner] or "ties") for r in selected)
        assert sum(counts.values()) == len(selected)
        for controller in ["native", "ctrl0", "mappo", "pi", "circles", "ties"]:
            records.append(dict(panel=panel, horizon_s=h, controller=controller, count=counts[controller],
                                denominator=len(selected), units="worlds" if panel == "Recorded worlds" else "states",
                                source_file=INPUTS[source], source_selector=f"{clock}={h};unique winner={controller}"))
    write_csv("switch_headroom.csv", records)


def finish(fig, name, source):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", metadata={"Description": source})
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"Subject": source})
    plt.close(fig)


def render():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    pi = rows("pi_diagnosis.csv")
    for ax, selected, title in zip(axes, [pi[:2], pi[2:4], pi[4:]],
                                    ["Original starts: snapshot", "Shifted starts: full history", "History effect, same shifted starts"]):
        for i, r in enumerate(selected):
            e, lo, hi = (float(r[k]) for k in ("estimate", "lo", "hi"))
            ax.errorbar(e, i, xerr=[[e-lo], [hi-e]], fmt="o", color="#176879", capsize=4)
        ax.set_yticks(range(len(selected)), [f"{r['horizon_s']} s" for r in selected])
        ax.set_ylim(-0.6, max(len(selected)-0.4, 0.6))
        ax.invert_yaxis()
        ax.axvline(0, color="0.55", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Paired endpoint-error difference (m)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Forecast horizon")
    fig.suptitle("PI teacher-forced diagnosis: separate start panels, not one horizon curve", fontsize=11)
    fig.text(0.5, 0.01, "First two: WM2 minus BB-traj. Third: WM2 full history minus snapshot. Negative favors WM2/history.\nSource world-bootstrap intervals; recorded CAVs, conditional presence, entries OFF; not an autonomous feedback win.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, .93))
    finish(fig, "fig_pi_diagnosis", INPUTS["pidiag.source.md"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    calib = rows("calibration.csv")
    for ax, split in zip(axes, ["Grouped CV", "SPEEDHEAD transfer"]):
        for offset, label, color in [(-.24, "raw WM2", "#176879"), (0, "arm-mean (no WM)", "#92999f"), (.24, "Selected calibration", "#c2823f")]:
            selected = [r for r in calib if r["split"] == split and
                        (r["signal"] == label if label != "Selected calibration" else "ridge" in r["signal"])]
            assert len(selected) == 3
            ax.bar([i+offset for i in range(3)], [float(r["mae_veh_s"]) for r in selected], width=.23, label=label, color=color)
        ax.set_xticks(range(3), ["120", "300", "600"])
        ax.set_xlabel("Forecast horizon (s)")
        ax.set_ylabel("Paired-effect MAE (veh-s)")
        ax.set_title(split)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle("Calibration: CV improvement does not survive the frozen transfer gate", fontsize=11)
    fig.text(.5, .01, "Selected per horizon using grouped CV; bars have no inferred CIs. Gate also requires sign and rank improvement.\nArm-minus-CTRL0 objective J; final verdict NOT USABLE for long-horizon policy credit.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, .93))
    finish(fig, "fig_calibration", INPUTS["calibration.source.md"])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    headroom = rows("switch_headroom.csv")
    for ax, h, panel, title in zip(axes, [600, 60, 90], ["Recorded worlds", "Same-state decisions", "Same-state decisions"],
                                  ["Recorded fixed episodes: 600 s", "Same native-trunk state: 60 s", "Same native-trunk state: 90 s"]):
        selected = [r for r in headroom if r["panel"] == panel and int(r["horizon_s"]) == h]
        ax.barh(range(6), [int(r["count"]) for r in selected], color=["#176879"]*5+["#92999f"])
        ax.set_yticks(range(6), ["Native", "CTRL0", "MAPPO", "PI", "CIRCLES", "Tied states"])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Unique-winning " + selected[0]["units"] + " (ties separate)")
        for i, r in enumerate(selected):
            ax.text(int(r["count"])+.15, i, r["count"], va="center", fontsize=8)
        ax.set_xlim(0, max(int(r["count"]) for r in selected)*1.23)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Controller-choice opportunity is not demonstrated model-guided switching benefit", fontsize=11)
    fig.text(.5, .01, "Recorded: 90 worlds. Fresh headroom: 16 states / 8 worlds; full-rerun prefixes and native repeats match.\nSame-state median best–second margins: 1.832371 / 6.853775 veh-s (60 / 90 s), including ties. Main efficacy pending.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, .93))
    finish(fig, "fig_switch_headroom", INPUTS["recorded_winners.source.csv"] + "; " + INPUTS["step0b.source.csv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-inputs", action="store_true")
    args = parser.parse_args()
    assert DATA.is_dir() and FIG.is_dir()
    if args.refresh_inputs:
        snapshot()
    verify()
    extract()
    render()
    print("Evening existing-evidence artwork rendered; no inference, fitting or bootstrap.")
