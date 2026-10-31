"""Render completed overnight exports; no model execution, fitting or bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA, FIG = ROOT / "data", ROOT / "figures"
INPUTS = {
    "closed.source.csv": "E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED_bootstrap.csv",
    "closed.source.md": "E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md",
    "closed_runtime.source.md": "E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED_RUNTIME.md",
    "distribution.source.csv": "E:/agent-work/c4-wmbreakdown/docs/paper_results/wmbreakdown/distribution_summary.csv",
    "distribution.source.md": "E:/agent-work/c4-wmbreakdown/docs/paper_results/WMBREAKDOWN.md",
    "value.source.md": "E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md",
    "repair_stop.source.md": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md",
    "switch_partial.source.md": "E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH.md",
    "rng.source.md": "E:/agent-work/c4-ensprobe/docs/paper_results/ENSPROBE.md",
    "sumocore.source.md": "E:/agent-work/c4-sumocore/docs/paper_results/SUMOCORE.md",
}
DENIED = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")


def digest(blob):
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
        manifest[name] = dict(source_file=INPUTS[name], sha256=digest(blob), bytes=len(blob))
    (DATA / "OVERNIGHT_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify():
    manifest = json.loads((DATA / "OVERNIGHT_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == set(INPUTS)
    for name, info in manifest.items():
        blob = (DATA / name).read_bytes()
        assert info == dict(source_file=INPUTS[name], sha256=digest(blob), bytes=len(blob)), name


def text(name):
    return (DATA / name).read_text(encoding="utf-8-sig")


def rows(name):
    return list(csv.DictReader(io.StringIO(text(name))))


def write_csv(name, records):
    assert records
    with (DATA / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def extract():
    records = []
    for row in rows("closed.source.csv"):
        if (row["model"] == "wm2-minus-bbtraj" and row["plant"] == "common"
                and row["arm"] in ("pi", "mappo") and row["metric"] in ("position", "J_ae")):
            selector = ";".join(f"{k}={row[k]}" for k in ("model", "plant", "arm", "horizon_s", "metric"))
            records.append(dict(row, source_file=INPUTS["closed.source.csv"], source_selector=selector))
    assert len(records) == 28
    assert len({r["source_selector"] for r in records}) == len(records)
    write_csv("closed_loop.csv", records)

    records = []
    for row in rows("distribution.source.csv"):
        if (row["split"] == "reserved8" and row["clock"] == "team" and row["arm"] == "pooled"
                and row["target"] in ("absolute", "effect")):
            selector = ";".join(f"{k}={row[k]}" for k in ("split", "target", "horizon_s", "clock", "arm"))
            records.append(dict(row, source_file=INPUTS["distribution.source.csv"], source_selector=selector))
    assert len(records) == 6, [(r["target"], r["horizon_s"]) for r in records]
    write_csv("distribution.csv", records)

    records = []
    # Transcribe the producer's existing intervals; do not resample report cells.
    for line in text("value.source.md").splitlines():
        if not re.match(r"\| [BCDE]:", line):
            continue
        method, sign, rank, mae = [s.strip() for s in line.strip("|").split("|")]
        for metric, cell, units in [("sign_accuracy", sign, "%"), ("spearman", rank, "correlation"), ("mae", mae, "veh-s")]:
            values = re.findall(r"[+-]?\d+(?:\.\d+)?", cell)
            assert len(values) == 3, cell
            estimate, lo, hi = values
            assert float(lo) <= float(estimate) <= float(hi)
            records.append(dict(method=method, metric=metric, estimate=estimate, lo=lo, hi=hi,
                                units=units, source_file=INPUTS["value.source.md"],
                                source_selector=f"Method table;{method};{metric}"))
    assert len(records) == 12
    write_csv("value_tail.csv", records)


def finish(fig, name, source):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", metadata={"Description": source})
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"Subject": source})
    plt.close(fig)


def render():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "pdf.fonttype": 42})
    closed = rows("closed_loop.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for arm, label, color, marker in [("pi", "PI", "#176879", "o"), ("mappo", "MAPPO", "#c2823f", "s")]:
        for ax, metric, horizons, scale in [(axes[0], "position", [10, 30, 60, 90, 120], 1),
                                             (axes[1], "J_ae", [60, 120, 180, 300], 1000)]:
            selected = sorted([r for r in closed if r["arm"] == arm and r["metric"] == metric
                               and int(r["horizon_s"]) in horizons], key=lambda r: int(r["horizon_s"]))
            x = [int(r["horizon_s"]) for r in selected]
            e, lo, hi = [[float(r[k]) / scale for r in selected] for k in ("estimate", "lo", "hi")]
            ax.plot(x, e, marker=marker, label=label, color=color)
            ax.fill_between(x, lo, hi, color=color, alpha=.12)
    for ax in axes:
        ax.axhline(0, color="0.5", lw=.8)
        ax.set_xlabel("Forecast horizon (s)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    axes[0].set_xticks([10, 30, 60, 90, 120])
    axes[1].set_xticks([60, 120, 180, 300])
    axes[0].set_title("Common-HDV position: 60 s advantage inconclusive")
    axes[0].set_ylabel("Paired WM2 minus BB-traj error (m)")
    axes[1].set_title("Aggregate delay: relative gain is not decision trust")
    axes[1].set_ylabel("Paired absolute-J-error difference (10³ veh-s)")
    fig.suptitle("Completed autonomous common-plant comparison", fontsize=11)
    fig.text(.5, .01, "Source 95% world-bootstrap intervals; negative favors WM2. Genuine 60 s history; autonomous entries/exits/source.\nEndpoint n=40 through 90 s, n=33 at 120 s; sparse 180 s and unavailable 300 s endpoints retained in CSV. J: n=40.\nApproximate shared CAV plant; initialization lineage incomplete. Curves connect tested clocks only.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .19, 1, .94))
    finish(fig, "fig_closed_loop", INPUTS["closed.source.csv"])

    distribution = rows("distribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    for ax, target, title in zip(axes, ["absolute", "effect"], ["Absolute team delay", "Arm minus CTRL0 effect"]):
        selected = sorted([r for r in distribution if r["target"] == target], key=lambda r: int(r["horizon_s"]))
        baseline_key = "cell_arm_climatology_crps" if target == "absolute" else "climatology_crps"
        baseline_label = "Cell × arm climatology" if target == "absolute" else "Paired-effect climatology"
        for offset, key, label, color in [(-.18, "wm_crps", "WM2 K8", "#176879"),
                                         (.18, baseline_key, baseline_label, "#92999f")]:
            ax.bar([i+offset for i in range(3)], [float(r[key]) / 1000 for r in selected], width=.34, label=label, color=color)
        ax.set_xticks(range(3), [r["horizon_s"] for r in selected])
        ax.set_xlabel("Forecast horizon (s)")
        ax.set_ylabel("CRPS (10³ veh-s; lower is better)")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7)
    fig.suptitle("Distributional failure persists beyond individual-outcome error", fontsize=11)
    fig.text(.5, .01, "Separate eight-world evaluation slice, previously exposed to calibration; no invented marginal CI bars.\nClimatology uses development worlds only. Cost-only futures cannot score imagined event timing/rates.\nAbsolute-delay K8 range coverage at 600 s: 12.5%; real below every future: 87.5%.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .19, 1, .94))
    finish(fig, "fig_distribution", INPUTS["distribution.source.csv"])

    value = rows("value_tail.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    labels = ["WM60 + V", "WM90 + V", "V + commands", "Arm/time mean"]
    for ax, metric, title, scale in [(axes[0], "sign_accuracy", "Sign accuracy (%)", 1),
                                      (axes[1], "mae", "MAE (10³ veh-s)", 1000)]:
        selected = [r for r in value if r["metric"] == metric]
        e, lo, hi = [[float(r[k]) / scale for r in selected] for k in ("estimate", "lo", "hi")]
        for i in range(4):
            ax.errorbar(e[i], i, xerr=[[e[i]-lo[i]], [hi[i]-e[i]]], fmt="o", capsize=4,
                        color="#176879" if i < 2 else "#686f75")
        ax.set_yticks(range(4), labels)
        ax.set_ylim(3.6, -.6)
        ax.set_xlabel(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Short WM + real-trained value: frozen necessary-condition gate FAIL", fontsize=11)
    fig.text(.5, .01, "Eight reserved FIT worlds / 400 pairs; source 95% world-bootstrap CIs.\nBoth short methods fail the D/E sign-and-MAE gate; paired gate contrasts are in the source report.\nRemaining fleet + source cost to absolute 900 s; long-WM A is NA, not plotted or estimated from partial runs.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .19, 1, .94))
    finish(fig, "fig_value_tail", INPUTS["value.source.md"])


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
    print("Completed overnight exports rendered; no experiment, fitting or bootstrap.")
