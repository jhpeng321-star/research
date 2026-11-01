"""Render completed morning receipts offline; never infer, simulate or bootstrap.

Source pins are immutable, independent of the moving report-refresh manifest.
All intervals are copied from producer outputs, not recomputed from summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json

from build_figures import ROOT, guarded

PINS = {
    "audit": ("audit_v2_csv_0abe14ec6f66b030.csv",
              "E:/agent-work/c4-coopmeasure/tools/coopaudit/evidence/v2_analysis/COOPAUDIT.csv"),
    "value": ("wmvalue_short_test_7dcebed8a07b9132.json",
              "E:/agent-work/c4-wmvalue/tools/wmvalue/evidence/short_test.json"),
    "planner": ("coopplan_summary_13054b89c4a37fdf.json",
                "E:/agent-work/c4-coopplan/docs/paper_results/coopplan/extension/final/SUMMARY.json"),
    "h300": ("coopplan_h300_summary_1a5ec2b0dbf310e4.json",
             "E:/agent-work/c4-coopplan/docs/paper_results/coopplan/h300/final/SUMMARY.json"),
    "worlds": ("coopplan_worlds_c8ad986376cfa3ff.csv",
               "E:/agent-work/c4-coopplan/docs/paper_results/coopplan/extension/final/pooled36.csv"),
}
STEMS = {"F2": "F2_corrected_controller_audit", "F3V": "F3V_failed_wm_value_gate",
         "F6": "F6_exact_model_planner"}
FIELDS = ["figure", "panel", "series", "metric", "horizon_s", "estimate", "lo", "hi",
          "n_worlds", "n_events", "unit", "source_path", "source_sha256", "snapshot",
          "source_locator", "transform", "scope"]


def load_pin(key):
    name, original = PINS[key]
    relative = "source_snapshots/" + name
    data = guarded(ROOT / relative).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert sha.startswith(name.rsplit("_", 1)[1].split(".")[0]), name
    return data, dict(source_path=original, source_sha256=sha, snapshot=relative)


def extract_morning():
    rows, sources = [], {}

    def add(key, fig, panel, series, metric, horizon, value, lo, hi, n, locator,
            *, events="", unit="veh-s", scope="", transform="identity"):
        assert float(lo) <= float(value) <= float(hi), (series, metric)
        rows.append(dict(figure=fig, panel=panel, series=series, metric=metric,
                         horizon_s=horizon, estimate=value, lo=lo, hi=hi,
                         n_worlds=n, n_events=events, unit=unit, **sources[key],
                         source_locator=locator, transform=transform, scope=scope))

    data, sources["audit"] = load_pin("audit")
    audit = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    for row in audit:
        if row["cohort"] == "all28" and row["outcome"] == "600":
            assert row["mean"] == row["lo"] == row["hi"] == ""
        if row["cohort"] != "eligible21" or row["outcome"] != "600":
            continue
        assert int(row["n_worlds"]) == 21
        metric = row["metric"]
        if metric not in ("C_G", "D_GROUP", "sum_D_ONLY", "frac_positive"):
            continue
        add("audit", "F2", "audit", row["controller"], metric, 600,
            *[float(row[k]) for k in ("mean", "lo", "hi")], 21,
            f"cohort=eligible21;outcome=600;controller={row['controller']};metric={metric}",
            unit="fraction" if metric == "frac_positive" else "veh-s",
            scope="conditional eligible21;28 allocated/7 missing/3 wall feasible;native release60s")
    assert len(rows) == 12
    for arm in ("mappo", "wall", "sumocf"):
        terms = {r["metric"]: r["estimate"] for r in rows if r["series"] == arm}
        assert abs(terms["D_GROUP"] - terms["sum_D_ONLY"] - terms["C_G"]) < 1e-9

    data, sources["value"] = load_pin("value")
    value = json.loads(data)
    assert value["verdict"] == "FAIL" and value["survivors"] == []
    assert not value["A_evaluated"] and not value["policy_training_authorized"]
    assert value["n_worlds"] == 8 and value["n_pairs"] == 400
    for method, metrics in value["metrics"].items():
        for metric in ("sign_accuracy", "mae", "spearman"):
            cell = metrics[metric]
            factor = 100 if metric == "sign_accuracy" else 1
            add("value", "F3V", "metrics", method, metric, "to900",
                cell["value"] * factor, *[x * factor for x in cell["ci95"]], 8,
                f"/metrics/{method}/{metric}",
                unit="percent" if factor == 100 else "veh-s" if metric == "mae" else "rho",
                transform="multiply by100 (proportion to percent)" if factor == 100 else "identity",
                scope="400 paired differences;arm-specific continuation;A NA;frozen FAIL")
    for contrast, cells in value["contrasts"].items():
        for metric, ci, factor in (("sign_gain", "sign_ci95", 100), ("mae_reduction", "mae_ci95", 1)):
            add("value", "F3V", "contrasts", contrast, metric, "to900",
                cells[metric] * factor, *[x * factor for x in cells[ci]], 8,
                f"/contrasts/{contrast}/{metric};{ci}", unit="pp" if factor == 100 else "veh-s",
                transform="multiply by100 (proportion to pp)" if factor == 100 else "identity",
                scope="paired world CI;positive improves;no surviving short method")

    for key, cohorts, horizon in (("planner", ("original12", "extension24", "pooled36"), 120),
                                  ("h300", ("confirmation18", "wall6"), 300)):
        data, sources[key] = load_pin(key)
        summary = json.loads(data)
        for cohort in cohorts:
            c = summary["cohorts"][cohort]
            for clock in ("600", "900"):
                cell = c["savings"][clock]
                add(key, "F6", "cost", cohort, "saving_J" + clock, int(clock),
                    cell["mean"], cell["lower"], cell["upper"], c["worlds"],
                    f"/cohorts/{cohort}/savings/{clock}",
                    scope=f"MAPPO-planner;fresh DEV;H{horizon} lookahead;episode clock after warmup")
            cell = c["credit"]
            add(key, "F6", "interaction", cohort, "C_G", horizon,
                cell["mean"], cell["lower"], cell["upper"], c["worlds"],
                f"/cohorts/{cohort}/credit", events=cell["events"], unit="veh-s/action",
                scope=f"event-weighted;whole-world bootstrap;O0 reference;H{horizon};inactive worlds retained")

    data, sources["worlds"] = load_pin("worlds")
    worlds = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    assert len(worlds) == 36 and len({r["seed"] for r in worlds}) == 36
    for clock in ("600", "900"):
        for row in worlds:
            saving = float(row["saving_J" + clock])
            assert abs(float(row["MAPPO_J" + clock]) - float(row["planner_J" + clock]) - saving) < 1e-8
        mean = sum(float(r["saving_J" + clock]) for r in worlds) / len(worlds)
        cell = next(r for r in rows if r["series"] == "pooled36" and r["metric"] == "saving_J" + clock)
        assert abs(mean - cell["estimate"]) < 1e-8
    assert len(rows) == 47
    return rows, sources


def serialized(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def render_morning(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, gray = "#0072B2", "#D55E00", "#555555"

    def save(fig, figure):
        for suffix in ("png", "pdf"):
            fig.savefig(guarded(ROOT / "figures" / (STEMS[figure] + "." + suffix), output=True),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def forest(ax, cells, labels, xlabel, color=blue):
        y = np.arange(len(cells))
        means = np.array([r["estimate"] for r in cells])
        ax.errorbar(means, y, xerr=[means - [r["lo"] for r in cells],
                                  [r["hi"] for r in cells] - means],
                    fmt="o", color=color, capsize=3)
        ax.axvline(0, color=gray, lw=.8)
        ax.set(yticks=y, yticklabels=labels, xlabel=xlabel)
        ax.invert_yaxis()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.17, right=.98, bottom=.30, top=.79, wspace=.60)
    cells = [r for r in rows if r["figure"] == "F2" and r["metric"] == "C_G"]
    labels = ["MAPPO-speed", "Wall", "SUMOCF u15"]
    forest(axes[0], cells, labels, "$C_G$ (veh-s)")
    axes[0].set_title("(a) Mean interaction unresolved")
    for metric, offset, color, label in (("D_GROUP", -.12, blue, "$D_G$"),
                                         ("sum_D_ONLY", .12, orange, "$\\sum D_i$")):
        cells = [r for r in rows if r["figure"] == "F2" and r["metric"] == metric]
        axes[1].bar(np.arange(3) + offset, [r["estimate"] for r in cells], width=.23,
                    color=color, label=label)
    axes[1].axhline(0, color=gray, lw=.8)
    axes[1].set(title="(b) Group / solo means", ylabel="Delay saving (veh-s)",
                xlabel="Frozen controller", xticks=range(3), xticklabels=["MAPPO", "Wall", "u15"])
    axes[1].legend(frameon=False)
    fig.suptitle("Corrected native-release audit: complete, conditional on eligible pairs", y=.98, fontsize=11)
    fig.text(.05, .035, "Source: COOPAUDIT.csv v2 · J600 after t300 · 60 s intervention · 21 eligible / 28 allocated\n"
             "(a) copied 95% stratified-world CIs; (b) point means only. 7 missing groups; full-allocation CIs undefined.\n"
             "Only 3 pairs wall-feasible; wall interaction approximately zero. No missing groups imputed.", fontsize=8)
    save(fig, "F2")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    fig.subplots_adjust(left=.18, right=.98, bottom=.27, top=.79, wspace=.60)
    labels = ["WM60 + V", "WM90 + V", "V + commands", "FIT arm/time mean"]
    for ax, metric, title, xlabel in zip(axes, ("sign_accuracy", "mae"),
                                       ("(a) Sign accuracy", "(b) Effect magnitude error"),
                                       ("Sign agreement (%)", "Paired-effect MAE (veh-s)")):
        cells = [r for r in rows if r["figure"] == "F3V" and r["panel"] == "metrics" and r["metric"] == metric]
        forest(ax, cells, labels, xlabel, orange)
        ax.set_title(title)
        ax.set_xlim(left=0, right=100 if metric == "sign_accuracy" else None)
    fig.suptitle("Frozen WM + real-value gate: FAIL against no-WM baselines", y=.98, fontsize=11)
    fig.text(.04, .035, "Source: WMVALUE short_test.json · 8 reserved FIT worlds / 400 paired differences · copied 95% world CIs\n"
             "Remaining cost to absolute t900; later starts are arm-specific states, not same-state group interventions.\n"
             "Full WM A is NA: both short methods fail the necessary D/E gate; no policy-training authorization.", fontsize=8)
    save(fig, "F3V")

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.5))
    fig.subplots_adjust(left=.19, right=.98, bottom=.19, top=.82, hspace=.95, wspace=.63)
    for i, (cohorts, labels, prefix) in enumerate([
            (("original12", "extension24", "pooled36"), ["Original (12)", "Extension (24)", "Pooled (36)"], "H120"),
            (("confirmation18", "wall6"), ["Confirmation (18)", "Wall panel (6)"], "H300")]):
        ax = axes[i, 0]
        for metric, offset, color, marker in (("saving_J600", -.10, blue, "o"), ("saving_J900", .10, orange, "s")):
            cells = [next(r for r in rows if r["figure"] == "F6" and r["series"] == c and r["metric"] == metric) for c in cohorts]
            means = np.array([r["estimate"] for r in cells])
            ax.errorbar(means, np.arange(len(cohorts)) + offset,
                        xerr=[means - [r["lo"] for r in cells], [r["hi"] for r in cells] - means],
                        fmt=marker, color=color, capsize=3, label=metric.replace("saving_", ""))
        ax.axvline(0, color=gray, lw=.8)
        ax.set(yticks=range(len(cohorts)), yticklabels=labels,
               xlabel="MAPPO − planner delay (veh-s/world)")
        ax.set_title(f"{prefix}: episode savings", pad=28)
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.00))
        cells = [next(r for r in rows if r["figure"] == "F6" and r["series"] == c and r["metric"] == "C_G") for c in cohorts]
        forest(axes[i, 1], cells, [f"{r['n_events']} actions" for r in cells],
               "Event-weighted $C_G$ (veh-s/action)", gray)
        axes[i, 1].set_title(f"{prefix}: exact all-action interaction")
    fig.suptitle("Exact-model planning improves delay; positive mean synergy is not robust", y=.97, fontsize=11)
    fig.text(.04, .025, "Source: COOPPLAN extension/final + h300/final SUMMARY.json · copied 95% whole-world bootstrap CIs\n"
             "Episode J600/J900 differ from local credit H120/H300. Original and extension pooled only within H120.\n"
             "H300 uses different fresh panels and quota-pruned options: not a paired causal horizon comparison.\n"
             "No learned FREE proposals at freeze; synchronous offline SUMO-oracle planning, not WM-trained policy.", fontsize=8)
    save(fig, "F6")


def check_morning(rows, sources):
    from PIL import Image
    artifacts = {}
    for figure, stem in STEMS.items():
        path = guarded(ROOT / "figures" / (stem + ".csv"))
        assert path.read_bytes() == serialized([r for r in rows if r["figure"] == figure])
        for suffix in ("csv", "png", "pdf"):
            relative = "figures/" + stem + "." + suffix
            data = guarded(ROOT / relative).read_bytes()
            if suffix == "png":
                with Image.open(ROOT / relative) as image:
                    assert image.width > 1500 and image.height > 700
                    image.verify()
            elif suffix == "pdf":
                assert data.startswith(b"%PDF")
            artifacts[relative] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    world_path = ROOT / "figures/F6_pooled36_source_worlds.csv"
    assert world_path.read_bytes() == load_pin("worlds")[0]
    artifacts[world_path.relative_to(ROOT).as_posix()] = dict(
        sha256=hashlib.sha256(world_path.read_bytes()).hexdigest(), bytes=world_path.stat().st_size)
    result = dict(status="PASS", source_cells=len(rows), sources=sources, artifacts=artifacts,
                  scope="Pinned source values/copied CIs/algebra/world-CSV identities/image integrity; no new statistics")
    guarded(ROOT / "morning_figure_validation.json", output=True).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows, sources = extract_morning()
    if not args.check:
        for fig, stem in STEMS.items():
            guarded(ROOT / "figures" / (stem + ".csv"), output=True).write_bytes(
                serialized([r for r in rows if r["figure"] == fig]))
        guarded(ROOT / "figures/F6_pooled36_source_worlds.csv", output=True).write_bytes(load_pin("worlds")[0])
        render_morning(rows)
    print(json.dumps(check_morning(rows, sources), indent=2))


if __name__ == "__main__":
    main()
