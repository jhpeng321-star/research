"""Render Paper B's existing evidence only; no project/runtime imports.

First render: --refresh-inputs snapshots a fixed allowlist of CSV/reports.
Subsequent renders use only this package's frozen data. No model execution,
fitting, resampling, subprocesses, directory crawling, or simulator access.
"""
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
INPUTS = {
    "direction.source.csv": "E:/research-dense/docs/paper_results/E2_direction_horizon.csv",
    "fair.source.csv": "E:/research-dense/docs/paper_results/E1_C1_fair.csv",
    "wall.source.md": "E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md",
    "regional.source.md": "E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md",
    "entry.source.md": "E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md",
}
DENIED = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")


def allowed(path: Path) -> Path:
    resolved = path.resolve()
    if any(x in str(resolved).lower() for x in DENIED):
        raise ValueError(f"Excluded path: {resolved}")
    return resolved


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def snapshot() -> None:
    # Read all inputs successfully before writing any snapshot.
    blobs = {name: allowed(Path(path)).read_bytes() for name, path in INPUTS.items()}
    manifest = {}
    for name, blob in blobs.items():
        (DATA / name).write_bytes(blob)
        manifest[name] = {"source_file": INPUTS[name], "sha256": digest(blob), "bytes": len(blob)}
    (DATA / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify_snapshots() -> None:
    manifest = json.loads((DATA / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == set(INPUTS)
    for name, info in manifest.items():
        blob = (DATA / name).read_bytes()
        assert info["source_file"] == INPUTS[name]
        assert info["sha256"] == digest(blob), name
        assert info["bytes"] == len(blob), name


def source_text(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8-sig")


def rows(name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(source_text(name))))


def write_csv(name: str, records: list[dict]) -> None:
    assert records
    with (DATA / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def table_rows(text: str, start: str, end: str) -> list[list[str]]:
    section = text.split(start, 1)[1].split(end, 1)[0]
    return [[x.strip() for x in line.strip().strip("|").split("|")]
            for line in section.splitlines() if line.startswith("| ")]


def extract() -> None:
    direction = rows("direction.source.csv")
    assert [int(x["horizon_s"]) for x in direction] == [10, 30, 60, 90, 120, 180]
    for row in direction:
        row["source_file"] = INPUTS["direction.source.csv"]
        row["source_selector"] = f"horizon_s={row['horizon_s']}"
    write_csv("direction.csv", direction)

    fair = rows("fair.source.csv")
    records = []
    selections = [
        ("WM-13E", "wm13e", "regret_oracle"),
        ("BB-traj", "bbtraj", "regret_oracle"),
        ("Always MAPPO", "wm13e", "always_mappo_regret"),
        ("Always native", "wm13e", "always_native_regret"),
    ]
    for label, model, metric in selections:
        panel = f"decisions/B_history_strict/{model}/60/team"
        found = [r for r in fair if r["panel"] == panel and r["metric"] == metric]
        assert len(found) == 1, (panel, metric)
        r = found[0]
        assert int(r["n_worlds"]) == 21 and int(r["n_states"]) == 32
        records.append({"selector": label, **r, "units": "veh-s",
                        "source_file": INPUTS["fair.source.csv"],
                        "source_selector": f"panel={panel};metric={metric}"})
    write_csv("regret.csv", records)

    panel = "paired_decisions/B_history_strict/60/team/13e_minus_bbtraj"
    paired = [r for r in fair if r["panel"] == panel and r["metric"] == "regret_oracle"]
    assert len(paired) == 1
    write_csv("regret_paired.csv", [{**paired[0], "units": "veh-s",
              "source_file": INPUTS["fair.source.csv"],
              "source_selector": f"panel={panel};metric=regret_oracle"}])

    wall = table_rows(source_text("wall.source.md"),
                      "## HDV |ds| on `wall_same`", "## HDV |ds| on `wall_pilot`")
    records = []
    for r in wall:
        if r[0] in ("struct", "blackbox_traj", "blackbox_id"):
            records.append(dict(model=r[0], horizon_s=r[1].removesuffix("s"),
                                n_states=r[2], median_m=r[3], iqr_m=r[4], p90_m=r[5],
                                blowup_fraction=r[6],
                                source_file=INPUTS["wall.source.md"],
                                source_selector=f"wall_same;model={r[0]};h={r[1]}"))
    assert len(records) == 12
    write_csv("wall_teacher_forced.csv", records)

    regional = table_rows(source_text("regional.source.md"),
                          "## Additive source/upstream/downstream errors", "## Corrected verdict")
    records = []
    for r in regional:
        if not r[0].isdigit():
            continue
        matched = r[3].replace("−", "-").replace("%", "")
        values = re.findall(r"([SUD])\s+([+-]?[\d.]+)/([\d.]+)/([+-]?[\d.]+)", matched)
        assert len(values) == 3, matched
        for component, bias, mae, share in values:
            records.append(dict(horizon_s=r[0], component=component, bias_veh_s=bias,
                                mae_veh_s=mae, covariance_share_pct=share,
                                clock="one-tick WMREQUAL3", reduction="paired K8 median, shared order weights",
                                source_file=INPUTS["regional.source.md"],
                                source_selector=f"Additive errors;Matched;h={r[0]};{component}"))
    assert len(records) == 15
    write_csv("regional.csv", records)

    entry = source_text("entry.source.md")
    # Extract published cells; no new statistic is estimated.
    source_block = entry.split("component error (", 1)[1].split("veh-s MAE)", 1)[0]
    clocks, changes = source_block.split(":", 1)
    horizons = [int(x) for x in re.findall(r"\d+", clocks)]
    pairs = re.findall(r"([\d.]+)\s*->\s*([\d.]+)", changes)
    assert len(horizons) == len(pairs) == 5
    records = []
    for h, (before, after) in zip(horizons, pairs):
        records.append(dict(horizon_s=h, component="Source", wm2_mae_veh_s=before,
                            entry_mae_veh_s=after, clock="labels left-boundary corrected",
                            source_file=INPUTS["entry.source.md"],
                            source_selector="Attribution;SOURCE component MAE"))
    for component, pattern in [
        ("Upstream", r"\((1456\.9)->(1393\.0) at 300 s\)"),
        ("Total J", r"J300 (1449\.1)->(1536\.2)"),
    ]:
        match = re.search(pattern, entry)
        assert match, component
        records.append(dict(horizon_s=300, component=component,
                            wm2_mae_veh_s=match[1], entry_mae_veh_s=match[2],
                            clock="labels left-boundary corrected",
                            source_file=INPUTS["entry.source.md"],
                            source_selector=f"Result;{component};300s"))
    write_csv("entry.csv", records)


def finish(fig, name: str, source: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight",
                metadata={"Description": source})
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"Subject": source})
    plt.close(fig)


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)


def render() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    accent = "#176879"
    r = rows("direction.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
    x, y, lo, hi = ([float(row[k]) for row in r] for k in ("horizon_s", "accuracy", "lo", "hi"))
    ax.errorbar(x, y, yerr=[[a-b for a,b in zip(y, lo)], [b-a for a,b in zip(y, hi)]],
                fmt="o-", color=accent, capsize=2, label="Accuracy + source CI")
    ax.axhline(float(r[0]["threshold"]), color="0.3", linestyle="--", linewidth=1,
               label="Registered threshold")
    for a, b, row in zip(x, y, r):
        ax.annotate(row["verdict"], (a,b), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=6)
    ax.set(title="Frozen direction panel\nCheckpoint binding pending",
           xlabel="Prediction horizon (s)", ylabel="Controller-effect sign accuracy (fraction)",
           ylim=(0.45, 1.03), xticks=x)
    ax.legend(fontsize=6, loc="lower left", frameon=False)
    clean(ax)
    finish(fig, "fig_direction", INPUTS["direction.source.csv"])

    r = rows("regret.csv")
    fig, (ax, paired_ax) = plt.subplots(2, 1, figsize=(3.5, 3.2), layout="constrained",
                                      gridspec_kw={"height_ratios": [3, 1]})
    for i, row in enumerate(r):
        v, low, high = (float(row[k]) for k in ("estimate", "lo", "hi"))
        ax.errorbar(v, i, xerr=[[v-low], [high-v]], fmt="o", capsize=3,
                    color=accent if i == 0 else "0.35")
    ax.set(yticks=range(len(r)), yticklabels=[row["selector"] for row in r],
           xlabel="Oracle regret (veh-s; lower is better)", xlim=(0, 75),
           title="Strict-history team selection at 60 s")
    ax.invert_yaxis()
    clean(ax)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    p = rows("regret_paired.csv")[0]
    v, low, high = (float(p[k]) for k in ("estimate", "lo", "hi"))
    paired_ax.errorbar(v, 0, xerr=[[v-low], [high-v]], fmt="o", color=accent, capsize=3)
    paired_ax.axvline(0, color="0.35", linestyle="--", linewidth=.8)
    paired_ax.set(yticks=[], ylim=(-.5,.5), xlim=(-23,5),
                  title="Paired WM minus BB-traj regret",
                  xlabel="Regret difference (veh-s; negative favors WM)")
    clean(paired_ax)
    finish(fig, "fig_regret", INPUTS["fair.source.csv"] + "; B_history_strict/*/60/team")

    r = rows("wall_teacher_forced.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
    for model, label, color, marker in [("struct", "Structured", accent, "o"),
                                       ("blackbox_traj", "BB-traj (teacher matched)", "0.35", "s"),
                                       ("blackbox_id", "BB-id (learned CAV)", "0.6", "^")]:
        subset = [row for row in r if row["model"] == model]
        ax.plot([float(row["horizon_s"]) for row in subset],
                [float(row["median_m"]) for row in subset], marker=marker,
                label=label, color=color)
    ax.set(xlabel="Prediction horizon (s)", ylabel="Median HDV absolute position error (m)",
           title="Historical unseen wall\nTeacher-forced response",
           xticks=[10,30,60,120])
    ax.legend(fontsize=6, frameon=False)
    clean(ax)
    finish(fig, "fig_wall_teacher_forced", INPUTS["wall.source.md"] + "; wall_same; no new CIs")

    r = [row for row in rows("regional.csv") if row["horizon_s"] == "300"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3), layout="constrained")
    ax.bar(["Source", "Upstream / merge", "Downstream"],
           [float(row["covariance_share_pct"]) for row in r],
           color=["0.6", accent, "0.6"], width=0.55)
    ax.axhline(0, color="0.25", linewidth=0.7)
    for i, row in enumerate(r):
        value = float(row["covariance_share_pct"])
        ax.text(i, value + (3 if value >= 0 else -3), f"{value:+.1f}%",
                ha="center", va="bottom" if value >= 0 else "top")
    ax.set(title="Corrected timeline J error at 300 s",
           xlabel="Additive error component (one-tick clock)",
           ylabel="Squared-error covariance share (%)", ylim=(-24, 112))
    clean(ax)
    finish(fig, "fig_regional", INPUTS["regional.source.md"] + "; covariance shares, not causal fractions")

    r = [row for row in rows("entry.csv") if row["horizon_s"] == "300"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    x = list(range(len(r)))
    before = [float(row["wm2_mae_veh_s"]) for row in r]
    after = [float(row["entry_mae_veh_s"]) for row in r]
    ax.bar([i-.19 for i in x], before, width=.36, color="0.6", label="WM2 clock-corrected")
    ax.bar([i+.19 for i in x], after, width=.36, color=accent, label="Zero-head entry path")
    for i, a, b in zip(x, before, after):
        ax.text(i-.19, a+20, f"{a:.1f}", ha="center", fontsize=6)
        ax.text(i+.19, b+20, f"{b:.1f}", ha="center", fontsize=6)
    ax.set(xticks=x, xticklabels=[row["component"] for row in r], ylim=(0,1900),
           title="Entry repair at 300 s\nSource improves; total J worsens",
           xlabel="Error metric at 300 s (left-boundary clock)", ylabel="Paired-effect MAE (veh-s)")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    clean(ax)
    finish(fig, "fig_entry", INPUTS["entry.source.md"] + "; source/upstream/total MAE; no CIs available")

    architecture(accent)


def architecture(accent: str) -> None:
    # Family-level diagram, not a claim that all historical checkpoints share wiring.
    nodes = {
        "history": ((.01,.62,.19,.28), "Observed history\n+ neighbors"),
        "settings": ((.27,.62,.20,.28), "Settings + gated\nlearned residual"),
        "traffic": ((.55,.62,.22,.28), "Physics + lane /\nmerge transition"),
        "next": ((.82,.62,.17,.28), "Predicted road\n+ source state"),
        "controller": ((.28,.12,.22,.27), "Executable CAV code\non predicted\nobservations"),
        "entries": ((.57,.12,.22,.27), "Requests / admissions\nexits + queue ledger"),
    }
    edges = [("history","settings"), ("settings","traffic"), ("traffic","next"),
             ("next","controller"), ("controller","traffic"), ("entries","traffic")]
    records = [{"from": a, "to": b,
                "source_file": "E:/research-dense/docs/WM_STATE_PLAN.md; E:/research-dense/docs/WM_CHANGE_LOG.md; E:/agent-work/c4-paperb/src/wms/model/spec.py",
                "source_selector": "family architecture; historical settings/gate/lane variants differ"}
               for a,b in edges]
    write_csv("architecture_edges.csv", records)
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    for name, ((x,y,w,h), text) in nodes.items():
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.005",
                                   edgecolor=accent if name=="controller" else "0.4",
                                   facecolor="white", linewidth=1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8)
    for a,b in edges:
        x,y,w,h = nodes[a][0]
        xx,yy,ww,hh = nodes[b][0]
        if (a,b) == ("next","controller"):
            route = [(x+w/2,y), (x+w/2,.48), (.24,.48), (.24,yy+hh/2), (xx,yy+hh/2)]
        elif (a,b) == ("controller","traffic"):
            route = [(x+w,y+h/2), (.53,y+h/2), (.53,.69), (xx,.69)]
        elif y < yy:
            route = [(x+w/2,y+h), (xx+ww/2,yy)]
        else:
            route = [(x+w,y+h/2), (xx,yy+hh/2)]
        for i, (p,q) in enumerate(zip(route,route[1:])):
            ax.add_patch(FancyArrowPatch(p,q, arrowstyle="->" if i==len(route)-2 else "-",
                                        mutation_scale=9, color="0.35", linewidth=.9))
    ax.set(xlim=(-.01,1.01), ylim=(0,1))
    ax.axis("off")
    fig.text(.5,.02,"WM-S family schematic: historical residual-only settings are not wired into physics; later hybrid settings are.",
             ha="center", fontsize=7)
    finish(fig, "fig_architecture", "WM_STATE_PLAN.md; WM_CHANGE_LOG.md; local model/spec.py; schematic only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-inputs", action="store_true", help="snapshot only the fixed allowed source files")
    args = parser.parse_args()
    assert ROOT.is_dir() and DATA.is_dir() and FIG.is_dir()
    if args.refresh_inputs:
        snapshot()
    verify_snapshots()
    extract()
    render()
    print("Rendered existing evidence; no model execution or new uncertainty estimates.")


if __name__ == "__main__":
    main()
