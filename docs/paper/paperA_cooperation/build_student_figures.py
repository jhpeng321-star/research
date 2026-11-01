"""Render the completed exact-teacher student from frozen producer JSON only.

python -B docs/paper/paperA_cooperation/build_student_figures.py [--check]
No fitting, simulator, inference, new bootstrap or replacement of older assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math

from build_figures import ROOT, guarded
from build_noon_figures import serialized

STEM = "F3T_exact_teacher_student"
PINS = {
    "summary": ("distill_live_summary_ece80f45ff3d9608.json", "SUMMARY.json"),
    "credits": ("distill_live_credits_079cc7f6d10114bd.json", "CREDITS.json"),
    "worlds": ("distill_live_worlds_c39d8627cf23372e.json", "WORLD_METRICS.json"),
    "qa": ("distill_live_qa_44e38727bddc62ed.json", "QA.json"),
}


def extract_student():
    sources, data = {}, {}
    for key, (name, original) in PINS.items():
        relative = "source_snapshots/" + name
        raw = guarded(ROOT / relative).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha.startswith(name.rsplit("_", 1)[1].split(".")[0])
        sources[key] = dict(source_path="E:/agent-work/c4-distill-1101/docs/paper_results/distill_live/" + original,
                            source_sha256=sha, snapshot=relative)
        data[key] = json.loads(raw)
    s, events, worlds = data["summary"], data["credits"], data["worlds"]
    assert data["qa"]["pass_"] and data["qa"]["complete_accounting"] and data["qa"]["all_event_twins"]
    assert len(worlds) == s["worlds"] == 24 and len({w["world_id"] for w in worlds}) == 24
    assert {w["world_id"] for w in worlds} == set(range(984000, 984024))
    assert len(events) == s["credit"]["events"] == 6 and s["budget_pruned"] == 0
    assert len({e["world_id"] for e in events}) == s["credit"]["active_worlds"] == 5
    assert sum(e["C_G"] > 0 for e in events) == s["credit"]["positive"] == 2
    assert not any(e["C_G"] > 0 and e["D_G"] > 0 for e in events)
    assert s["charged"] == s["actual"] == 72960
    assert sum(w["executed_moves"] == 0 for w in worlds) == s["all_O0_bitwise_worlds"] == 19
    for horizon in (600, 900):
        for w in worlds:
            assert math.isclose(w[f"MAPPO_J{horizon}"] - w[f"student_J{horizon}"],
                                w[f"saving_J{horizon}"], abs_tol=1e-8)
        assert math.isclose(sum(w[f"saving_J{horizon}"] for w in worlds)/len(worlds),
                            s["savings"][str(horizon)]["mean"], abs_tol=1e-8)
    for e in events:
        assert math.isclose(e["D_G"] - sum(e["member_alone_D"]), e["C_G"], abs_tol=1e-8)
    assert math.isclose(sum(e["C_G"] for e in events)/len(events), s["credit"]["mean"], abs_tol=1e-8)
    rows = []

    def add(key, panel, series, horizon, value, pointer, ci=None, n=24):
        rows.append(dict(figure="F3T", panel=panel, series=series, horizon_s=horizon,
                         estimate=value, unit="veh-s", lo=ci[0] if ci else "", hi=ci[1] if ci else "",
                         interval_status="copied producer 95% whole-world CI" if ci else "descriptive event; CI not supplied",
                         n_worlds=n, **sources[key], source_line="JSON", source_field=pointer,
                         transform="identity; sum_D_i is explicit sum of source singleton values",
                         scope="exact-teacher student; episode saving and H120 event credit are distinct"))
    for horizon in (600, 900):
        cell = s["savings"][str(horizon)]
        add("summary", "episode_saving", f"J{horizon}", horizon, cell["mean"],
            f"/savings/{horizon}", ci=cell["ci95"])
    add("summary", "event_mean", "C_G", 120, s["credit"]["mean"], "/credit", ci=s["credit"]["ci95"])
    for i, e in enumerate(events):
        for term, value in (("D_G", e["D_G"]), ("sum_D_i", sum(e["member_alone_D"])), ("C_G", e["C_G"])):
            add("credits", "event", f"{i}:{term}", e["horizon_s"], value,
                f"/{i}/" + ("member_alone_D [sum]" if term == "sum_D_i" else term), n=1)
    assert len(rows) == 21
    return rows, sources, worlds, events


def world_csv(worlds):
    output = io.StringIO(newline="")
    fields = list(worlds[0])
    assert all(set(w) == set(fields) for w in worlds)
    writer = csv.DictWriter(output, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(worlds)
    return output.getvalue().encode("utf-8")


def render_student(rows, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.9), gridspec_kw={"width_ratios": [1.1, 1, 1.35]})
    fig.subplots_adjust(left=.08, right=.98, top=.78, bottom=.33, wspace=.52)
    for ax, cells, title, ylabel, labels in (
        (axes[0], rows[:2], "(a) No established MAPPO win", "MAPPO − student delay (veh-s/world)", ["J600", "J900"]),
        (axes[1], rows[2:3], "(b) Mean synergy unresolved", "$C_G$ (veh-s/action)", ["All 6 groups"])):
        y = [r["estimate"] for r in cells]
        ax.errorbar(range(len(cells)), y, yerr=[[v-r["lo"] for v, r in zip(y, cells)],
                    [r["hi"]-v for v, r in zip(y, cells)]], fmt="o", color="#0072B2", capsize=4)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set(title=title, ylabel=ylabel, xticks=range(len(cells)), xticklabels=labels,
               xlim=(-.6, len(cells)-.4))
    ax = axes[2]
    ax.scatter([e["D_G"] for e in events], [e["C_G"] for e in events], color="#D55E00", s=35)
    ax.axhline(0, color="#555555", lw=.8)
    ax.axvline(0, color="#555555", lw=.8)
    ax.set(title="(c) No helpful superadditive move", xlabel="$D_G$ (veh-s; positive helps)",
           ylabel="$C_G$ (veh-s)", xlim=(-140, 275), ylim=(-140, 45))
    ax.text(.98, .95, "0/6: $D_G>0$ and $C_G>0$", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.suptitle("Exact-teacher observable student: completed fresh development test", fontsize=11, y=.98)
    fig.text(.08, .87, "24 worlds · frozen EXAM-only D600 student · six executed moves · zero quota prunes", fontsize=9)
    fig.text(.08, .04,
             "Sources: DISTILL.md final / distill_live JSON · source 95% world-bootstrap CIs; no new statistics.\n"
             "Episode savings (control-start J600/J900) differ from same-prefix H120 event credit. Both positive-$C_G$ groups harm.\n"
             "Event CI resamples all worlds; 35/10000 zero-event draws excluded. 19 all-O0 worlds exactly match MAPPO.\n"
             "This is the exact-teacher student, not a WM-trained result; incident zeros do not establish safety equivalence.", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(guarded(ROOT / "figures" / f"{STEM}.{suffix}", output=True), dpi=300, bbox_inches="tight")
    plt.close(fig)


def check_student(rows, sources, worlds):
    from PIL import Image

    assert guarded(ROOT / "figures" / f"{STEM}.csv").read_bytes() == serialized(rows)
    assert guarded(ROOT / "figures" / f"{STEM}_worlds.csv").read_bytes() == world_csv(worlds)
    artifacts = {}
    for name in (f"{STEM}.csv", f"{STEM}_worlds.csv", f"{STEM}.png", f"{STEM}.pdf"):
        path = guarded(ROOT / "figures" / name)
        raw = path.read_bytes()
        if name.endswith(".png"):
            with Image.open(path) as image:
                assert image.width > 2000 and image.height > 900
                image.verify()
        elif name.endswith(".pdf"):
            assert raw.startswith(b"%PDF")
        artifacts["figures/" + name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    result = dict(status="PASS", source_cells=len(rows), sources=sources, artifacts=artifacts,
                  scope="source hashes/QA/CSV identity, world pairing/means, all-event interaction algebra; no new CIs")
    guarded(ROOT / "student_figure_validation.json", output=True).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows, sources, worlds, events = extract_student()
    if not args.check:
        guarded(ROOT / "figures" / f"{STEM}.csv", output=True).write_bytes(serialized(rows))
        guarded(ROOT / "figures" / f"{STEM}_worlds.csv", output=True).write_bytes(world_csv(worlds))
        render_student(rows, events)
    print(json.dumps(check_student(rows, sources, worlds), indent=2))


if __name__ == "__main__":
    main()
