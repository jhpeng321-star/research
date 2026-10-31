"""Render existing, explicitly allowlisted evidence. No experiment/model imports.

Initial capture: python -B build_figures.py --freeze-inputs
Portable rebuild: python -B build_figures.py
Validation only: python -B build_figures.py --check
All writes are confined to this package; existing source snapshots are immutable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth", "vtype")
SOURCES = {
    "cooperation": Path("E:/research-dense/docs/paper_results/COOPMEAS2.csv"),
    "historical_direction": Path("E:/research-dense/docs/paper_results/E2_direction_horizon.csv"),
    "historical_efficacy": Path("E:/research-dense/docs/paper_results/E3FINAL_A.csv"),
    "corrected_wm": Path("E:/agent-work/c4-wmrequal3/tools/wmrequal3/evidence/ANALYSIS.json"),
}
SNAPSHOTS = {
    "cooperation": "inputs/COOPMEAS2.csv",
    "historical_direction": "inputs/E2_direction_horizon.csv",
    "historical_efficacy": "inputs/E3FINAL_A.csv",
    "corrected_wm": "inputs/WMREQUAL3_projection.json",
}
FIELDS = ["figure", "panel", "series", "metric", "horizon_s", "estimate", "lo", "hi",
          "n_worlds", "n_contrasts", "sign_n", "source_path", "source_sha256",
          "source_locator", "source_columns", "transform", "scope"]


def guarded(path: Path, *, output: bool = False) -> Path:
    path = path.resolve()
    if any(word in str(path).lower() for word in FORBIDDEN):
        raise ValueError(f"Restricted path: {path}")
    if output and not path.is_relative_to(ROOT):
        raise ValueError(f"Output outside package: {path}")
    return path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> None:
    path = guarded(path, output=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def capture() -> None:
    identities = {}
    for key, original in SOURCES.items():
        path = guarded(original)
        data = path.read_bytes()
        identities[key] = {"path": path.as_posix(), "sha256": digest(data), "bytes": len(data)}
        if key == "corrected_wm":
            full = json.loads(data)
            # Copy summary fields only; no raw rollout rows, batches or model files.
            value = {"contract": full["contract"], "matched": {}}
            for horizon in ("60", "90", "120", "180", "300"):
                cell = full["matched"][horizon]
                value["matched"][horizon] = {
                    "J_median": cell["J_median"], "decomposition": cell["decomposition"]}
            data = (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode()
        target = guarded(ROOT / SNAPSHOTS[key], output=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Immutable snapshot differs; retain it and version any refresh: {target}")
        if not target.exists():
            target.write_bytes(data)
        identities[key]["snapshot"] = SNAPSHOTS[key]
        identities[key]["snapshot_sha256"] = digest(data)
        # Verify against the originally captured digest, including for JSON projections.
        if digest(path.read_bytes()) != identities[key]["sha256"]:
            raise ValueError(f"Source changed while capturing: {path}")
    target = ROOT / "inputs/source_manifest.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != identities:
        raise ValueError("Existing source manifest differs; do not overwrite frozen provenance")
    dump(target, identities)


def csv_rows(key):
    text = guarded(ROOT / SNAPSHOTS[key]).read_text(encoding="utf-8-sig")
    return list(enumerate(csv.DictReader(io.StringIO(text)), start=2))


def extract():
    manifest = json.loads((ROOT / "inputs/source_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest.values():
        assert digest(guarded(ROOT / entry["snapshot"]).read_bytes()) == entry["snapshot_sha256"]
    output = []

    def add(key, fig, panel, series, metric, horizon, est, lo, hi, locator, columns,
            worlds, scope, contrasts="", sign_n=""):
        row = dict(zip(FIELDS, [fig, panel, series, metric, int(horizon), float(est),
                              "" if lo is None else float(lo), "" if hi is None else float(hi),
                              worlds, contrasts, sign_n, manifest[key]["path"],
                              manifest[key]["sha256"], locator, columns, "identity", scope]))
        assert math.isfinite(row["estimate"])
        if lo is not None:
            assert float(lo) <= float(est) <= float(hi)
        output.append(row)

    coop = csv_rows("cooperation")
    for line, r in coop:
        if r["cohort"] != "all":
            continue
        if r["clock_s"] == "300" and r["contrast"] in ("ours", "mappo"):
            add("cooperation", "F2H", "components", r["contrast"], r["term"], 300,
                r["est"], r["lo"], r["hi"], f"line:{line}", "est,lo,hi", int(r["n_worlds"]),
                "historical lane-head MAPPO aecd7d70 / cycle-1 64f70af4; native-release")
        if r["term"] == "C_G" and r["contrast"] == "ours_minus_mappo":
            add("cooperation", "F2H", "paired_interaction", r["contrast"], "C_G", r["clock_s"],
                r["est"], r["lo"], r["hi"], f"line:{line}", "est,lo,hi", int(r["n_worlds"]),
                "historical; primary 300 s; other clocks secondary")
    for arm in ("ours", "mappo"):
        cells = {r["metric"]: r["estimate"] for r in output
                 if r["figure"] == "F2H" and r["panel"] == "components" and r["series"] == arm}
        assert math.isclose(cells["D_G"] - cells["sum_D_i"], cells["C_G"], abs_tol=1e-9)

    for line, r in csv_rows("historical_direction"):
        add("historical_direction", "F4", "historical_direction", "historical E2", "sign_agreement",
            r["horizon_s"], r["accuracy"], r["lo"], r["hi"], f"line:{line}", "accuracy,lo,hi",
            int(r["worlds"]), "historical frozen direction panel; checkpoint alias unbound")
        add("historical_direction", "F4", "historical_direction", "source threshold", "threshold",
            r["horizon_s"], r["threshold"], None, None, f"line:{line}", "threshold",
            int(r["worlds"]), "threshold belongs only to historical E2 panel")

    wm = json.loads((ROOT / SNAPSHOTS["corrected_wm"]).read_text(encoding="utf-8"))
    for horizon, block in wm["matched"].items():
        j = block["J_median"]
        common = (wm["contract"]["worlds"],
                  "corrected recorded native-lane interventions; ours_c1 + coopoff_c1; MAPPO excluded")
        base = f"/matched/{horizon}"
        add("corrected_wm", "F4", "corrected_direction", "WM2 matched native lanes", "sign_agreement",
            horizon, j["sign"], *j["sign_ci95"], base + "/J_median", "sign,sign_ci95",
            *common, contrasts=j["n"], sign_n=j["sign_n"])
        add("corrected_wm", "F4", "corrected_error", "Total effect", "mae", horizon,
            j["mae"], None, None, base + "/J_median/mae", "mae", *common, contrasts=j["n"])
        for region, values in block["decomposition"].items():
            add("corrected_wm", "F4", "corrected_error", region, "mae", horizon,
                values["mae"], None, None, base + f"/decomposition/{region}/mae", "mae",
                *common, contrasts=j["n"])

    for line, r in csv_rows("historical_efficacy"):
        if (r["share"] == r["demand_vph"] == "All" and r["metric"] == "team_delay_vs"
                and (r["treatment"], r["reference"]) in
                (("ours", "mappo"), ("ours", "native"), ("mappo", "native"))):
            add("historical_efficacy", "F3H", "efficacy", r["treatment"] + " minus " + r["reference"],
                "team_delay_vs", r["horizon_s"], r["treatment_minus_reference"], r["ci95_lo"],
                r["ci95_hi"], f"line:{line}", "treatment_minus_reference,ci95_lo,ci95_hi",
                int(r["n_worlds"]), "historical frozen E3 final A; equal-cell; NOT new speed-only learner")
    return output, manifest


def export_rows(rows, filename):
    path = guarded(ROOT / filename, output=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, gray = "#0072B2", "#D55E00", "#555555"

    # Symbolic protocol figure: no illustrative empirical values or invented results.
    fig, ax = plt.subplots(figsize=(8.4, 3.0), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    nodes = [
        ("freeze", .14, .58, "Freeze observed group G\nphysical + RNG state\nrequests + controller memory"),
        ("branches", .46, .58, "Exact audit branches\nB: native release\nGROUP: G acts\nONLY:i: one member acts"),
        ("measure", .82, .58, "$D(S)=J(B)-J(S)$\n$C_G=D(G)-\\sum_i D(\\{i\\})$\nReport group / solo gains\nand world uncertainty"),
        ("interpret", .46, .12, "Fleet + source outcomes; entrant behavior and clocks declared\nBehavior signatures describe the process; C_G measures interaction"),
    ]
    for _, x, y, label in nodes:
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="square,pad=.7", facecolor="white", edgecolor=gray, lw=.8))
    for left, right in ((.285, .33), (.605, .67)):
        ax.annotate("", xy=(right, .58), xytext=(left, .58),
                    arrowprops=dict(arrowstyle="->", color=gray, lw=1))
    ax.text(.82, .92, "WM-S: candidate surrogate\nvalidate against exact branch effects", ha="center",
            va="center", fontsize=8, color=blue)
    ax.annotate("", xy=(.82, .79), xytext=(.82, .84),
                arrowprops=dict(arrowstyle="->", color=blue, ls="--", lw=1))
    fig.suptitle("Cooperation audit: common context, different acting subsets", fontsize=11)
    fig.supxlabel("Design schematic | Source: AUDIT_DESIGN.md; COOPMEAS2.md | learner mean-background credit is a separate intervention", fontsize=7)
    out = guarded(ROOT / "figures/F1_exact_replay_protocol.csv", output=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["node", "symbolic_label", "source_path", "scope"])
        for key, _, _, label in nodes:
            writer.writerow([key, label, "E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md",
                             "symbolic protocol; no measured quantities"])
    fig.savefig(ROOT / "figures/F1_exact_replay_protocol.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "figures/F1_exact_replay_protocol.pdf", bbox_inches="tight")
    plt.close(fig)

    def selected(fig, panel, **filters):
        return [r for r in rows if r["figure"] == fig and r["panel"] == panel
                and all(r[k] == v for k, v in filters.items())]

    def errors(data):
        return np.array([[r["estimate"] - r["lo"] for r in data],
                         [r["hi"] - r["estimate"] for r in data]])

    def save(fig, key, filename):
        data = [r for r in rows if r["figure"] == key]
        export_rows(data, f"figures/{filename}.csv")
        fig.savefig(ROOT / f"figures/{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(ROOT / f"figures/{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained")
    metrics = ["D_G", "sum_D_i", "C_G", "D_all"]
    labels = ["Group saving", "Sum of solo savings", "Interaction $C_G$", "Fleet saving (entrants act)"]
    for arm, color, offset, label, marker in [("ours", blue, -.12, "Historical cycle-1", "o"),
                                             ("mappo", orange, .12, "Historical MAPPO (lane-head)", "s")]:
        data = sorted(selected("F2H", "components", series=arm), key=lambda r: metrics.index(r["metric"]))
        axes[0].errorbar([r["estimate"] for r in data], np.arange(len(data)) + offset,
                         xerr=errors(data), fmt=marker, color=color, capsize=2, label=label)
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, xlabel="Saved delay (veh-s/world)",
                title="(a) Historical 300 s outcomes")
    axes[0].invert_yaxis()
    axes[0].set_ylim(4.2, -.5)  # Reserve a blank legend area below all data rows.
    axes[0].legend(loc="lower right", frameon=False)
    data = sorted(selected("F2H", "paired_interaction"), key=lambda r: r["horizon_s"])
    axes[1].errorbar([r["estimate"] for r in data], range(len(data)), xerr=errors(data),
                     fmt="D", color=gray, capsize=3)
    axes[1].set(yticks=range(len(data)), yticklabels=[str(r["horizon_s"]) for r in data],
                ylabel="Outcome horizon after branch (s)", xlabel="Paired cycle-1 minus MAPPO $C_G$ (veh-s/world)",
                title="(b) Interaction advantage not established")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color="black", lw=.7, ls="--")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("COOPMEAS2: historical interaction, not the new speed-only audit", fontsize=11)
    fig.supxlabel("Source: COOPMEAS2.csv | 28 worlds | source pointwise 95% world CIs; 300 s primary", fontsize=8)
    save(fig, "F2H", "F2H_historical_cooperation")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), layout="constrained")
    data = selected("F4", "historical_direction", metric="sign_agreement")
    axes[0].errorbar([r["horizon_s"] for r in data], [r["estimate"] for r in data],
                     yerr=errors(data), fmt="o", color=blue, capsize=2)
    threshold = selected("F4", "historical_direction", metric="threshold")[0]["estimate"]
    axes[0].axhline(threshold, color=gray, ls="--", lw=.9, label="Historical source threshold")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].set(title="(a) Historical E2 direction only\n54 worlds; checkpoint alias unbound",
                ylabel="Effect-sign agreement (fraction)", ylim=(0, 1.04), xlabel="Outcome horizon (s)")
    data = selected("F4", "corrected_direction")
    axes[1].errorbar([r["horizon_s"] for r in data], [r["estimate"] for r in data],
                     yerr=errors(data), fmt="s", color=orange, capsize=2)
    axes[1].set(title="(b) Corrected native-lane contract\n36 worlds; 72 contrasts; NOT QUALIFIED",
                ylabel="Effect-sign agreement (fraction)", ylim=(0, 1.04), xlabel="Outcome horizon (s)")
    for series, color, marker, label in [("Total effect", "black", "o", "Total effect"),
                                        ("upstream", orange, "s", "Upstream / merge road"),
                                        ("source", blue, "^", "Source waiting"),
                                        ("downstream", gray, "D", "Downstream road")]:
        data = selected("F4", "corrected_error", series=series)
        axes[2].plot([r["horizon_s"] for r in data], [r["estimate"] for r in data],
                     marker=marker, color=color, lw=.9, label=label)
    axes[2].set(title="(c) Corrected effect-magnitude error\nRegional and total MAEs need not add",
                ylabel="MAE of arm-minus-native delay (veh-s)", xlabel="Outcome horizon (s)")
    axes[2].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("World-model reliability is contract-dependent; panels are not pooled", fontsize=11)
    fig.supxlabel("Sources: E2_direction_horizon.csv; WMREQUAL3 ANALYSIS.json | existing 95% world CIs for signs; no MAE CIs exported here", fontsize=8)
    save(fig, "F4", "F4_surrogate_reliability")

    fig, ax = plt.subplots(figsize=(6.8, 3.4), layout="constrained")
    horizons = [120, 300, 600]
    for series, color, offset, marker in [("ours minus mappo", orange, -.14, "D"),
                                         ("ours minus native", blue, 0, "o"),
                                         ("mappo minus native", gray, .14, "s")]:
        data = sorted(selected("F3H", "efficacy", series=series), key=lambda r: r["horizon_s"])
        ax.errorbar(np.arange(len(data)) + offset, [r["estimate"] for r in data], yerr=errors(data),
                    fmt=marker, color=color, capsize=3, label=series.replace("ours", "cycle-1"))
    ax.axhline(0, color="black", ls="--", lw=.8)
    ax.set(xticks=range(len(horizons)), xticklabels=[str(h) for h in horizons],
           xlabel="Cumulative intervention horizon (s)", ylabel="Team cost difference (veh-s/world)",
           title="Historical frozen imagined policy: native gain, MAPPO loss\nE3 final A; not the new speed-only warm-started learner")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=.15)
    fig.supxlabel("Source: E3FINAL_A.csv | 228 paired worlds; equal-cell means | 600 s cycle-1 minus MAPPO primary; negative cost is better", fontsize=7)
    save(fig, "F3H", "F3H_historical_policy_failure")


def check(rows, manifest):
    from PIL import Image

    names = {"F2H": "F2H_historical_cooperation", "F4": "F4_surrogate_reliability",
             "F3H": "F3H_historical_policy_failure"}
    checks = {}
    symbolic = ROOT / "figures/F1_exact_replay_protocol.csv"
    with symbolic.open(encoding="utf-8", newline="") as handle:
        symbolic_rows = list(csv.DictReader(handle))
    assert {r["node"] for r in symbolic_rows} == {"freeze", "branches", "measure", "interpret"}
    with Image.open(ROOT / "figures/F1_exact_replay_protocol.png") as im:
        im.verify()
    checks["F1"] = {"symbolic_nodes": len(symbolic_rows), "csv": symbolic.relative_to(ROOT).as_posix()}
    for key, name in names.items():
        with (ROOT / f"figures/{name}.csv").open(encoding="utf-8", newline="") as handle:
            exported = list(csv.DictReader(handle))
        expected = [r for r in rows if r["figure"] == key]
        assert len(exported) == len(expected)
        for got, wanted in zip(exported, expected, strict=True):
            assert got == {k: str(v) for k, v in wanted.items()}
        image = ROOT / f"figures/{name}.png"
        with Image.open(image) as im:
            im.verify()
        checks[key] = {"source_values_verified": len(expected), "csv": f"figures/{name}.csv"}
    # Check source extract identities rather than reconstructing any marginal CI.
    primary = next(r for r in rows if r["figure"] == "F3H" and r["series"] == "ours minus mappo"
                   and r["horizon_s"] == 600)
    assert primary["lo"] > 0
    old_primary = next(r for r in rows if r["figure"] == "F2H" and r["panel"] == "paired_interaction"
                       and r["horizon_s"] == 300)
    assert old_primary["lo"] < 0 < old_primary["hi"]
    artifacts = {}
    for name in ["F1_exact_replay_protocol", *names.values()]:
        for ext in ("png", "pdf", "csv"):
            path = ROOT / f"figures/{name}.{ext}"
            data = path.read_bytes()
            artifacts[path.relative_to(ROOT).as_posix()] = {"sha256": digest(data), "bytes": len(data)}
    report = {"status": "PASS", "scope": "presentation-only; no new statistical estimates or model inference",
              "figures": checks, "sources": manifest, "artifacts": artifacts,
              "source_check": "frozen CSV bytes / explicit JSON summary projection; no live-source requirement"}
    dump(ROOT / "figure_validation.json", report)
    print(json.dumps({"status": "PASS", "figures": checks}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-inputs", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.freeze_inputs:
        capture()
    rows, manifest = extract()
    if not args.check:
        render(rows)
    check(rows, manifest)


if __name__ == "__main__":
    main()
