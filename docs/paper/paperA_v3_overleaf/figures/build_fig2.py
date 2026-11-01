"""Build Fig. 2 (opportunity above MAPPO) from existing, read-only result files.

Sources (read-only):
  (a) E:/agent-work/c4-coopplan/docs/paper_results/coopplan/extension/final/pooled36.csv
      (per-world saving_J600 = MAPPO minus planner, veh-s; COOPPLAN.md 'Pooled36')
  (b) E:/agent-work/c4-coopplan/docs/paper_results/coopplan/extension/final/ALL_GROUP_CREDITS.json
      (52 executed group maneuvers with both member-alone twins; C_G, veh-s)
  (c) E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md, 'Team J600', mappo row
      (D_G +893.181 = sum D_i +825.502 + C_G +67.679, C_G 95% CI [-1179.061, +1154.090])

Run:  C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe figures/build_fig2.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

COOPPLAN = Path("E:/agent-work/c4-coopplan/docs/paper_results/coopplan/extension/final")
COOPAUDIT = Path("E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md")
OUT = Path(__file__).resolve().parent / "fig2_opportunity.pdf"

POOLED_MEAN, POOLED_LO, POOLED_HI = 648.407, 209.699, 1173.616  # COOPPLAN.md Pooled36 J600

BLUE = "#2a6fdb"
GRAY = "#8a8f98"
INK = "#222222"


def load_worlds():
    rows = list(csv.DictReader(open(COOPPLAN / "pooled36.csv", newline="")))
    assert len(rows) == 36, len(rows)
    mean = np.mean([float(r["saving_J600"]) for r in rows])
    assert abs(mean - POOLED_MEAN) < 1e-3, mean
    return rows


def load_credits():
    credits = json.load(open(COOPPLAN / "ALL_GROUP_CREDITS.json"))
    cg = np.array([float(c["C_G"]) for c in credits])
    assert len(cg) == 52 and int((cg > 0).sum()) == 24, (len(cg), (cg > 0).sum())
    return cg


def load_audit():
    text = COOPAUDIT.read_text(encoding="utf-8")
    sec = text.split("## Team J600", 1)[1].split("##", 1)[0]
    m = re.search(
        r"- mappo: C_G ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\];.*?D_G ([+-][\d.]+); sumD_i ([+-][\d.]+)",
        sec,
    )
    cg, lo, hi, dg, sdi = (float(x) for x in m.groups())
    assert abs(dg - (sdi + cg)) < 2e-3, (dg, sdi, cg)
    return dg, sdi, cg, lo, hi


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.titlesize": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })
    rows = load_worlds()
    cg = load_credits()
    dg, sdi, acg, alo, ahi = load_audit()

    fig = plt.figure(figsize=(3.4, 2.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.35, 1.0],
                          hspace=0.62, wspace=0.62)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # (a) per-world saving by cell (penetration x demand)
    pens = sorted({float(r["penetration"]) for r in rows})
    dems = sorted({int(r["demand"]) for r in rows})
    cells = [(p, d) for p in pens for d in dems]
    rng = np.random.default_rng(0)
    for i, (p, d) in enumerate(cells):
        ys = [float(r["saving_J600"]) for r in rows
              if float(r["penetration"]) == p and int(r["demand"]) == d]
        xs = i + rng.uniform(-0.18, 0.18, len(ys))
        cols = [BLUE if y > 0 else GRAY for y in ys]
        ax_a.scatter(xs, ys, s=9, c=cols, edgecolors="white", linewidths=0.4, zorder=3)
    ax_a.axhline(0, color=INK, lw=0.5, zorder=1)
    ax_a.axhline(POOLED_MEAN, color=BLUE, lw=0.8, ls="--", zorder=2)
    ax_a.text(-0.4, POOLED_MEAN / 1.6,
              f"pooled mean +{POOLED_MEAN:.0f} [{POOLED_LO:.0f}, {POOLED_HI:.0f}]",
              ha="left", va="top", fontsize=5.5, color=BLUE)
    ax_a.set_yscale("symlog", linthresh=10)
    ax_a.set_ylim(-2000, 30000)
    ticks = [-1000, -100, -10, 0, 10, 100, 1000, 10000]
    ax_a.set_yticks(ticks)
    ax_a.set_yticklabels(["$-10^3$", "", "", "0", "", "", "$10^3$", "$10^4$"])
    ax_a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_a.set_xticks(range(len(cells)))
    ax_a.set_xticklabels([f"{d / 1000:.1f}k" for _, d in cells], rotation=0)
    for j, p in enumerate(pens):
        ax_a.text(j * len(dems) + (len(dems) - 1) / 2, 1.0, f"{p * 100:.0f}% CAV",
                  transform=ax_a.get_xaxis_transform(), ha="center", va="bottom", fontsize=6)
        if j:
            ax_a.axvline(j * len(dems) - 0.5, color="#dddddd", lw=0.5, zorder=0)
    ax_a.set_xlim(-0.6, len(cells) - 0.4)
    ax_a.set_ylabel("J600 saving (veh-s)")
    ax_a.set_xlabel("demand (veh/h)", labelpad=1)
    ax_a.set_title("(a) exact planner vs. MAPPO, 36 worlds", loc="left", pad=9)

    # (b) distribution of C_G for 52 executed group maneuvers
    bins = np.arange(-700, 1000, 100)
    ax_b.hist(cg[cg <= 0], bins=bins, color=GRAY, edgecolor="white", linewidth=0.5,
              label=f"$C_G\\leq0$ ({int((cg <= 0).sum())})")
    ax_b.hist(cg[cg > 0], bins=bins, color=BLUE, edgecolor="white", linewidth=0.5,
              label=f"$C_G>0$ ({int((cg > 0).sum())})")
    ax_b.axvline(0, color=INK, lw=0.5)
    ax_b.set_xlabel("$C_G$ (veh-s)")
    ax_b.set_ylabel("maneuvers")
    ax_b.legend(frameon=False, fontsize=5.5, loc="upper right", handlelength=1.0)
    ax_b.set_title("(b) synergy, 52 maneuvers", loc="left")

    # (c) MAPPO group effect decomposition, J600
    labels = ["$D_G$", "$\\Sigma_i D_i$", "$C_G$"]
    vals = [dg, sdi, acg]
    ax_c.bar(range(3), vals, width=0.6, color=[INK, GRAY, BLUE], edgecolor="white", linewidth=0.5)
    ax_c.errorbar([2], [acg], yerr=[[acg - alo], [ahi - acg]], fmt="none",
                  ecolor=INK, elinewidth=0.6, capsize=2)
    for i, v in enumerate(vals[:2]):
        ax_c.text(i, v + 40, f"{v:+.0f}", ha="center", va="bottom", fontsize=5.5, color=INK)
    ax_c.text(2, ahi + 40, f"{acg:+.0f}", ha="center", va="bottom", fontsize=5.5, color=INK)
    ax_c.axhline(0, color=INK, lw=0.5)
    ax_c.set_xticks(range(3))
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("J600 effect (veh-s)", labelpad=1)
    ax_c.set_title("(c) MAPPO, 21 pairs", loc="left")

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
