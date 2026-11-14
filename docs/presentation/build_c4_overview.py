"""Build the C4 / WM-S research-showcase slides (landscape PDF).

Usage (from the repo root):
    python docs/presentation/build_c4_overview.py

All numbers live in the DATA section directly below. To add Friday's final results,
fill the FINAL_* structures (replace None with numbers) and re-run; the
"TO BE FILLED FRIDAY" banners disappear automatically once a slot is complete.
Figures are written to docs/presentation/figs/, the PDF to
docs/presentation/C4_WMS_overview_20261016.pdf.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import Paragraph, Table, TableStyle  # noqa: E402

# ======================================================================================
# DATA -- edit here. None = not yet available (rendered as "TBD").
# ======================================================================================

TITLE = "Learning cooperative CAV control inside a physics-informed traffic world model"
SUBTITLE = "WM-S world model and C4 multi-CAV policy: how it works, results so far"
DATE_LINE = "Research showcase, 16 October 2026"
AUTHOR_LINE = "Jinghan Peng"

HORIZONS_S = [10, 30, 60, 120]

# (a) Median HDV position error (m) on 36 new unseen-wall worlds, textbook population.
#     value = (errors at 10/30/60/120 s, blow-ups "k/n")
WALL_ERR_36 = {
    "v5": ([4.26, 15.6, 32.2, 76.4], "22/139"),
    "v3": ([5.19, 18.3, 36.9, 95.9], "27/139"),
    "old C4": ([3.53, 16.9, 35.3, 164.0], "35/139"),
    "v4": ([6.07, 20.9, 47.9, 215.0], "50/139"),
}
# Same wall sets, 60 s of history given to the encoder. Blow-ups only reported for v5.
WALL_ERR_60H = {
    "v5": ([3.64, 12.9, 27.9, 92.2], "25"),
    "v3": ([4.95, 18.4, 50.5, 112.0], None),
    "old C4": ([3.99, 17.2, 47.2, 169.0], None),
    "v4": ([6.27, 25.7, 58.0, 157.0], None),
}
BLOWUP_DEF = "blow-up = a state whose median HDV error exceeds 50 m at 60 s and 100 m at 120 s"

# (b) Wall-benefit prediction (v5).
WALL_BENEFIT = {
    "sign_agree_30": 0.99,
    "sign_agree_60": 0.93,
    "rho": 0.95,
    "pred_120_vehs": -166,
    "true_120_vehs": -108,
    "benefits_caught": (20, 51),
    "gain_captured_frac": 0.43,
}

# (c) Action-effect prediction on held-out paired branches (v5).
ACTION_EFFECT = {
    "sign_agree": (0.85, 0.90),  # 30 s, 60 s
    "pearson": (0.89, 0.87),
    "close_calls": 0.67,
}

# (d) Diagnosis: physics with TRUE driver settings vs v5, same 72 states.
TRUTH_PROBE = {
    "True settings (physics only)": ([1.18, 5.38, 12.6, 47.5], 0),
    "v5": ([4.15, 15.1, 28.5, 56.3], 10),
}
V5_SETTING_FLOOR = {"a": 1.78, "b": 2.87}  # lowest values the v5 encoder could reach
MERGE_DELAY_S = {"v5 model": 2.5, "SUMO": 41.0}  # median accel-lane wait before merging
ENTRY_REFUSED = "99-100%"

# (e) SUMO policy comparison, development worlds (12 seeds x demand 1900/2400 x pen 10/20/40%).
#     Change in system time per vehicle vs native (s); negative = better. (mean, +/-)
SUMO_DEV = [
    ("MAPPO (trained in SUMO)", -1.51, 0.89),
    ("Always max acceleration", -0.56, 0.77),
    ("v5 policy (learned in WM)", 0.97, 0.86),
]
V5_POLICY_ENTRY_WAIT_VEHS = 597
# Earlier legacy driver population, heavy demand (no interval quoted here).
SUMO_LEGACY = [
    ("MAPPO", -4.27, ""),
    ("CIRCLES damping", 1.56, ""),
    ("PI", -0.28, "n.s."),
]

# (f) v6 driver estimates (early) vs v5 vs truth (population medians).
DRIVER_SETTINGS = {
    # setting: (v6, v5, truth, unit)
    "a": (0.86, 1.84, 1.2, "m/s²"),
    "b": (1.94, 2.99, 2.0, "m/s²"),
    "T": (1.29, 1.56, 1.2, "s"),
    "s0": (1.96, 1.94, 2.0, "m"),
}
V6_VAL10_GAIN = "5-23%"

# ---------------------------------------------------------------------------------------
# FRIDAY SLOTS -- replace None with final numbers and re-run.
# ---------------------------------------------------------------------------------------

# Slot 1: v6 vs v5 on the unseen-wall worlds (median HDV error, m, 10/30/60/120 s; blow-ups).
FINAL_WM_WALL = {
    "v6": ([None, None, None, None], None),
    "v5": ([4.26, 15.6, 32.2, 76.4], "22/139"),
}
FINAL_WM_WALL_NOTE = "Same 36 unseen-wall worlds and scorer as the v5 table."
# Wall-benefit prediction, same definitions as slide (b): (v6, v5)
FINAL_WM_BENEFIT = {
    "Sign agreement 30 s": (None, 0.99),
    "Sign agreement 60 s": (None, 0.93),
    "Real 120 s benefits caught": (None, "20/51"),
    "Share of best gain captured": (None, "43%"),
}

# Slot 2: final SUMO evaluation. Change in system time per vehicle vs native (s), mean.
FINAL_EVAL_GRID = {
    "demands_veh_h": [1900, 2100, 2400, 2700],
    "pen_seen_pct": [5, 10, 20, 40],
    "pen_unseen_pct": [15, 30, 60],
    "seeds": 42,
    "light_demands": [1900],  # light = near capacity; the road saturates ~1,930 veh/h mainline
    "heavy_demands": [2400, 2700],
}
FINAL_SUMO_COLUMNS = ["All", "Seen pen.", "Unseen pen.", "Light", "Heavy"]
FINAL_SUMO = {
    "v6 policy": [None, None, None, None, None],
    "v5 policy": [None, None, None, None, None],
    "MAPPO": [None, None, None, None, None],
    "CIRCLES damping": [None, None, None, None, None],
    "PI": [None, None, None, None, None],
    "Always max accel.": [None, None, None, None, None],
}

# Slot 3: cooperative-behaviour analysis (free text values or numbers).
FINAL_COOP = {
    "CAVs side by side across lanes (share of time)": None,
    "Synchronised CAV slowdowns before the merge (events/h)": None,
    "Gap hand-offs to ramp vehicles (events/h)": None,
    "Change with penetration (individual -> group moves)": None,
    "WM-predicted vs SUMO-measured gain": None,
}

# ======================================================================================
# STYLE
# ======================================================================================

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figs"
OUT_PDF = HERE / "C4_WMS_overview_20261016.pdf"

INK = "#1f2937"
MUTED = "#6b7280"
ACCENT = "#1d4ed8"
LIGHT = "#eef2ff"
AMBER = "#b45309"
AMBER_BG = "#fff7ed"
HDV_C = "#9ca3af"
CAV_C = "#2563eb"
RSU_C = "#0f766e"

MODEL_COLORS = {
    "v6": "#15803d",
    "v5": "#2563eb",
    "v3": "#64748b",
    "old C4": "#a855f7",
    "v4": "#ea580c",
    "Truth": "#111827",
    "True settings (physics only)": "#111827",
}
CTRL_COLORS = {
    "MAPPO": "#7c3aed",
    "Always max acceleration": "#94a3b8",
    "v5 policy": "#2563eb",
    "CIRCLES damping": "#ea580c",
    "PI": "#0d9488",
}

PAGE_W, PAGE_H = 960.0, 540.0  # 16:9 points

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
    }
)


def _register_fonts() -> None:
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DV", str(ttf / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", str(ttf / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", str(ttf / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DV-BI", str(ttf / "DejaVuSans-BoldOblique.ttf")))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily

    registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")


def fmt(v, nd=None):
    if v is None:
        return "TBD"
    if isinstance(v, str):
        return v
    if nd is None:
        nd = 2 if abs(v) < 10 else (1 if abs(v) < 100 else 0)
    return f"{v:.{nd}f}".replace("-", "−")


def slot_complete(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return all(slot_complete(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(slot_complete(v) for v in obj)
    return True


# ======================================================================================
# FIGURES
# ======================================================================================


def _box(ax, x, y, w, h, text, fc, ec=None, fs=12, tc=INK, bold=False, r=0.02):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={r}",
            fc=fc, ec=ec or fc, lw=1.6,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, p0, p1, color=INK, lw=1.8, style="-|>", ls="-", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16, color=color, lw=lw,
                        linestyle=ls, connectionstyle=f"arc3,rad={rad}")
    )


def _car(ax, x, y, color, L=0.055, Hh=0.03):
    ax.add_patch(FancyBboxPatch((x - L / 2, y - Hh / 2), L, Hh,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                fc=color, ec="white", lw=0.8))


def _poly_arrow(ax, pts, color=INK, lw=1.8, ls="-"):
    """Polyline whose last segment ends in an arrow head."""
    xs = [q[0] for q in pts[:-1]]
    ys = [q[1] for q in pts[:-1]]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt")
    _arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)


def fig_system(path):
    import random

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 2.6)
    ax.set_ylim(-0.46, 0.74)
    ax.axis("off")
    road = "#e5e7eb"
    lane_h = 0.075
    y0 = 0.25
    # mainline: 3 lanes
    ax.add_patch(Rectangle((0.15, y0), 2.4, 3 * lane_h, fc=road, ec="none"))
    for k in range(1, 3):
        ax.plot([0.15, 2.55], [y0 + k * lane_h] * 2, color="white", lw=1.5, ls=(0, (6, 6)))
    # acceleration lane (1.25-1.70) with taper, and a straight ramp of constant lane height
    ax.add_patch(Rectangle((1.25, y0 - lane_h), 0.45, lane_h, fc=road, ec="none"))
    ax.add_patch(Polygon([[1.70, y0 - lane_h], [1.70, y0], [1.85, y0]], fc=road, ec="none"))
    r0 = (0.55, -0.33)
    ax.add_patch(Polygon([[r0[0], r0[1]], [r0[0], r0[1] + lane_h], [1.25, y0], [1.25, y0 - lane_h]],
                         fc=road, ec="none"))
    ax.plot([1.25, 1.70], [y0, y0], color="white", lw=1.5, ls=(0, (6, 6)))
    # merge zone highlight
    ax.add_patch(Rectangle((1.22, y0 - lane_h - 0.02), 0.66, 4 * lane_h + 0.04, fc="none", ec=AMBER,
                           lw=2, ls="--"))
    ax.text(1.92, y0 - lane_h - 0.02, "merge bottleneck\n(saturates near 1,930 veh/h mainline)",
            ha="left", va="top", fontsize=12, color=AMBER)
    # mainline vehicles (denser near the merge)
    rnd = random.Random(3)
    for lane in range(3):
        yc = y0 + (lane + 0.5) * lane_h
        x = 0.25 + rnd.random() * 0.05
        while x < 2.5:
            _car(ax, x, yc, CAV_C if rnd.random() < 0.25 else HDV_C)
            x += 0.075 if 1.0 < x < 1.8 else 0.13 + rnd.random() * 0.08
    # ramp vehicles along the ramp centre line
    c0 = (r0[0], r0[1] + lane_h / 2)
    c1 = (1.25, y0 - lane_h / 2)
    for t, col in [(0.25, HDV_C), (0.55, CAV_C), (0.85, HDV_C)]:
        _car(ax, c0[0] + t * (c1[0] - c0[0]), c0[1] + t * (c1[1] - c0[1]), col)
    for xr in (1.36, 1.52):
        _car(ax, xr, y0 - lane_h / 2, HDV_C)
    # entry queues (vehicles waiting to enter; their wait counts in system time)
    for k in range(4):
        _car(ax, 0.07, y0 - 0.05 + k * 0.06, CAV_C if k == 2 else HDV_C, L=0.05, Hh=0.035)
    ax.text(0.07, y0 - 0.11, "entry\nqueue", ha="center", va="top", fontsize=11, color=MUTED)
    for k in range(3):
        _car(ax, 0.47 - k * 0.06, -0.30, HDV_C, L=0.05, Hh=0.035)
    ax.text(0.41, -0.36, "ramp entry queue", ha="center", va="top", fontsize=11, color=MUTED)
    # RSU
    rx, ry = 1.30, 0.55
    ax.plot([rx, rx], [y0 + 3 * lane_h + 0.01, ry], color=RSU_C, lw=3)
    ax.add_patch(Rectangle((rx - 0.04, ry - 0.01), 0.08, 0.06, fc=RSU_C, ec="none"))
    ax.text(rx - 0.06, ry + 0.02, "RSU (roadside unit)\nreceives CAV reports", fontsize=12, color=RSU_C,
            va="center", ha="right")
    for cx, cy in [(1.75, y0 + 2.5 * lane_h), (2.12, y0 + 1.5 * lane_h)]:
        ax.plot([rx + 0.04, cx], [ry + 0.02, cy + 0.02], color=RSU_C, lw=1.2, ls=":")
    # scale + direction
    ax.annotate("", xy=(2.55, 0.70), xytext=(0.15, 0.70),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
    ax.text(1.35, 0.715, "~3.2 km of road (schematic, not to scale)", fontsize=11, color=MUTED,
            va="bottom", ha="center")
    ax.annotate("", xy=(2.55, -0.03), xytext=(2.25, -0.03),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.text(2.22, -0.03, "traffic direction", ha="right", va="center", fontsize=11, color=INK)
    # legend
    _car(ax, 1.95, -0.29, HDV_C)
    ax.text(2.0, -0.29, "HDV: human-driven (IDM)", va="center", fontsize=12)
    _car(ax, 1.95, -0.38, CAV_C)
    ax.text(2.0, -0.38, "CAV: connected automated, 5-40%", va="center", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_wms_arch(path):
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.35, 5.85)
    ax.axis("off")
    blue_bg, green_bg, grey_bg, amber_bg = "#dbeafe", "#dcfce7", "#f3f4f6", "#ffedd5"
    _box(ax, 0.2, 3.9, 2.6, 1.2, "Observed history\nof each HDV\n(up to 60 s)", grey_bg, fs=12)
    _box(ax, 3.4, 3.9, 2.6, 1.2, "Driver encoder\n(learned from\nprediction error)", blue_bg, ACCENT, fs=12)
    _box(ax, 6.6, 3.9, 2.8, 1.2, "Driver settings\na, b, T, s0, v0,\nlane parameters", blue_bg, fs=12)
    _box(ax, 0.2, 1.95, 2.6, 1.3, "Current state\n(positions, speeds,\nlanes, neighbours)\n+ CAV commands",
         grey_bg, fs=11.5)
    _box(ax, 3.4, 2.2, 2.6, 0.95, "Gated residual\n(attention + temporal,\nbounded)", amber_bg, AMBER, fs=11.5)
    _box(ax, 6.6, 2.2, 2.8, 0.95, "IDM car-following\ncore (physics)", green_bg, "#15803d", fs=12.5, bold=True)
    _box(ax, 3.4, 0.25, 2.6, 1.15, "Lane model (hazard_v5)\nshared net; heads:\novertake / return / merge",
         amber_bg, AMBER, fs=11)
    _box(ax, 6.6, 0.25, 2.8, 1.15, "Entry model\n(vehicles entering\nthe road)", amber_bg, AMBER, fs=11.5)
    ax.text(10.0, 2.67, "+", fontsize=26, ha="center", va="center", color=INK)
    _box(ax, 10.5, 0.55, 2.3, 3.25, "Integrate\n0.5 s step\n\n→ next state\nof every\nvehicle",
         "#e0e7ff", ACCENT, fs=12.5)
    # identification path
    _arrow(ax, (2.8, 4.5), (3.4, 4.5))
    _arrow(ax, (6.0, 4.5), (6.6, 4.5))
    _arrow(ax, (8.0, 3.9), (8.0, 3.15))
    # state feeds IDM (routed above the residual), residual and lane model
    _poly_arrow(ax, [(2.8, 3.05), (3.1, 3.05), (3.1, 3.5), (7.0, 3.5), (7.0, 3.15)])
    _arrow(ax, (2.8, 2.67), (3.4, 2.67))
    _arrow(ax, (2.8, 2.1), (3.4, 0.95))
    # IDM + residual -> sum -> integrate
    _arrow(ax, (9.4, 2.67), (9.8, 2.67))
    _poly_arrow(ax, [(4.7, 2.2), (4.7, 1.8), (10.0, 1.8), (10.0, 2.45)], color=AMBER)
    ax.text(7.95, 1.86, "Δa (gated correction)", fontsize=11, ha="center", va="bottom", color=AMBER)
    _arrow(ax, (10.2, 2.67), (10.5, 2.67))
    # lanes and entries -> integrate
    _poly_arrow(ax, [(4.7, 0.25), (4.7, 0.05), (11.65, 0.05), (11.65, 0.55)], color=AMBER)
    ax.text(8.0, 0.0, "lane changes (sampled)", fontsize=11, ha="center", va="top", color=AMBER)
    _arrow(ax, (9.4, 0.82), (10.5, 0.82), color=AMBER)
    # feedback loop
    _poly_arrow(ax, [(11.65, 3.8), (11.65, 5.45), (1.5, 5.45), (1.5, 5.1)], color=MUTED, ls="--")
    ax.text(6.6, 5.52, "roll forward step by step (10-120 s)", fontsize=11.5, color=MUTED, ha="center",
            va="bottom")
    # legend
    for k, (fc, ec, lab) in enumerate([(green_bg, "#15803d", "physics"), (amber_bg, AMBER, "learned"),
                                       (blue_bg, ACCENT, "learned identification")]):
        yy = 1.25 - k * 0.38
        ax.add_patch(Rectangle((0.3, yy), 0.25, 0.2, fc=fc, ec=ec))
        ax.text(0.65, yy + 0.1, lab, va="center", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_dreamer_loop(path):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    _box(ax, 0.2, 3.3, 3.0, 1.5, "SUMO data\nnative + undirected random\n+ paired branches\n(+ policy rollouts)", "#f3f4f6", fs=11.5)
    _box(ax, 4.2, 3.3, 2.6, 1.5, "Train WM-S\n(world model)", "#dcfce7", "#15803d", fs=13, bold=True)
    _box(ax, 7.8, 3.3, 3.0, 1.5, "Imagine 30 s rollouts\nall CAVs act with the\nshared policy", "#e0e7ff", ACCENT, fs=12)
    _box(ax, 7.8, 0.6, 3.0, 1.6, "Reward: system time\n(incl. entry-queue wait)\n+ critic, discount 0.995\n(~200 s of value)", "#ffedd5", AMBER, fs=11.5)
    _box(ax, 4.2, 0.6, 2.6, 1.6, "Update policy\nanalytic gradients\nthrough the model", "#dbeafe", ACCENT, fs=12)
    _box(ax, 0.2, 0.6, 3.0, 1.6, "Check in SUMO\n(select checkpoints)", "#f3f4f6", fs=12)
    _arrow(ax, (3.2, 4.05), (4.2, 4.05))
    _arrow(ax, (6.8, 4.05), (7.8, 4.05))
    _arrow(ax, (9.3, 3.3), (9.3, 2.2))
    _arrow(ax, (7.8, 1.4), (6.8, 1.4))
    _arrow(ax, (5.5, 2.2), (8.3, 3.3), color=ACCENT, ls="--")
    ax.text(6.25, 2.75, "repeat", fontsize=11, color=ACCENT, rotation=20)
    _arrow(ax, (4.2, 1.4), (3.2, 1.4))
    _arrow(ax, (1.7, 2.2), (1.7, 3.3), color=MUTED, ls="--")
    _box(ax, 11.1, 2.3, 1.8, 2.5, "Counter-\nfactual\ncredit:\ngroup vs\nindividual\nrows", "#fef3c7", AMBER, fs=11)
    _arrow(ax, (11.1, 2.6), (10.8, 1.6), color=AMBER)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _err_panel(ax, data, title):
    xs = list(range(len(HORIZONS_S)))
    for name, (vals, blow) in data.items():
        lab = f"{name}" + (f"  (blow-ups {blow})" if blow else "")
        ax.plot(xs, vals, marker="o", lw=2.6 if name == "v5" else 1.8, ms=7,
                color=MODEL_COLORS[name], label=lab, zorder=3 if name == "v5" else 2)
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{h} s" for h in HORIZONS_S])
    ax.set_yticks([3, 10, 30, 100, 200], ["3", "10", "30", "100", "200"])
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("median HDV position error (m, log)")
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.legend(fontsize=11, loc="upper left")


def fig_wall_error(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    _err_panel(axes[0], WALL_ERR_36, "36 new unseen-wall worlds")
    _err_panel(axes[1], WALL_ERR_60H, "same wall sets, 60 s history")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_truth_probe(path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = list(range(len(HORIZONS_S)))
    for name, (vals, blow) in TRUTH_PROBE.items():
        c = MODEL_COLORS.get(name, "#111827")
        ax.plot(xs, vals, marker="o", lw=2.4, ms=7, color=c, ls="--" if "True" in name else "-",
                label=f"{name}  ({blow} blow-ups)")
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{h} s" for h in HORIZONS_S])
    ax.set_yticks([1, 3, 10, 30, 60], ["1", "3", "10", "30", "60"])
    ax.set_ylabel("median HDV error (m, log)")
    ax.set_title("same 72 wall states", fontsize=13, loc="left")
    ax.legend(fontsize=10.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_sumo_policy(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    names = [n for n, _, _ in SUMO_DEV][::-1]
    means = [m for _, m, _ in SUMO_DEV][::-1]
    errs = [e for _, _, e in SUMO_DEV][::-1]
    cols = []
    for n in names:
        key = next((k for k in CTRL_COLORS if n.startswith(k)), None)
        cols.append(CTRL_COLORS.get(key, "#6b7280"))
    ax.barh(names, means, xerr=errs, color=cols, height=0.55, capsize=6,
            error_kw=dict(ecolor=INK, lw=1.4))
    ax.axvline(0, color=INK, lw=1.2)
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(max(m + e, 0) + 0.12, i, f"{m:+.2f} ± {e:.2f} s".replace("-", "−"),
                va="center", ha="left", fontsize=12)
    ax.set_xlim(-3.4, 3.0)
    ax.set_xlabel("Δ system time per vehicle vs native (s)   ← better")
    ax.set_title("textbook population, development worlds", fontsize=13, loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    names = [n for n, _, _ in SUMO_LEGACY][::-1]
    means = [m for _, m, _ in SUMO_LEGACY][::-1]
    notes = [s for _, _, s in SUMO_LEGACY][::-1]
    cols = [CTRL_COLORS.get(n, "#6b7280") for n in names]
    ax.barh(names, means, color=cols, height=0.55, alpha=0.75)
    ax.axvline(0, color=INK, lw=1.2)
    for i, (m, s) in enumerate(zip(means, notes)):
        lab = f"{m:+.2f} s".replace("-", "−") + (f" ({s})" if s else "")
        ax.text(m + 0.15 if m >= 0 else 0.15, i, lab, va="center", ha="left", fontsize=12)
    ax.set_xlim(-6.5, 4.0)
    ax.set_xlabel("Δ system time per vehicle vs native (s)")
    ax.set_title("earlier: legacy population, heavy demand", fontsize=13, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_driver_settings(path):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (k, (v6, v5, truth, unit)) in zip(axes, DRIVER_SETTINGS.items()):
        ax.bar([0, 1], [v5, v6], color=[MODEL_COLORS["v5"], MODEL_COLORS["v6"]], width=0.6)
        ax.axhline(truth, color=MODEL_COLORS["Truth"], lw=2, ls="--")
        ax.text(1.35, truth, f"truth {truth:.1f}", va="bottom", ha="left", fontsize=11)
        for x, v in [(0, v5), (1, v6)]:
            ax.text(x, v * 0.5, f"{v:.2f}", ha="center", va="center", fontsize=13, color="white",
                    fontweight="bold")
        ax.set_xticks([0, 1], ["v5", "v6"])
        ax.set_xlim(-0.6, 2.1)
        ax.set_ylim(0, 1.25 * max(v5, v6, truth))
        ax.set_title(f"{k} ({unit})", fontsize=13)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_final_sumo(path) -> bool:
    """Only drawn once the 'All' column of FINAL_SUMO is filled."""
    vals = {k: v[0] for k, v in FINAL_SUMO.items()}
    if not all(v is not None for v in vals.values()):
        return False
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    names = list(vals)[::-1]
    pal = {"v6 policy": MODEL_COLORS["v6"], "v5 policy": MODEL_COLORS["v5"], "MAPPO": CTRL_COLORS["MAPPO"],
           "CIRCLES damping": CTRL_COLORS["CIRCLES damping"], "PI": CTRL_COLORS["PI"]}
    ax.barh(names, [vals[k] for k in names], color=[pal.get(k, "#94a3b8") for k in names], height=0.55)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlabel("Δ system time / veh vs native (s)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


# ======================================================================================
# SLIDE HELPERS
# ======================================================================================

class Deck:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
        self.c.setTitle("C4 / WM-S overview, 2026-10-16")
        self.c.setAuthor(AUTHOR_LINE)
        self.n = 0
        self.total = None

    # ---- primitives ---------------------------------------------------------------
    def style(self, size=17, color=INK, lead=None, align=TA_LEFT, bold=False):
        return ParagraphStyle(
            "s", fontName="DV-B" if bold else "DV", fontSize=size, bulletFontName="DV", bulletFontSize=size,
            leading=lead or size * 1.32, textColor=colors.HexColor(color), alignment=align,
        )

    def para(self, text, x, ytop, w, size=17, color=INK, align=TA_LEFT, bold=False):
        p = Paragraph(text, self.style(size, color, align=align, bold=bold))
        _, h = p.wrap(w, PAGE_H)
        p.drawOn(self.c, x, ytop - h)
        return ytop - h

    def bullets(self, items, x, ytop, w, size=17, gap=7):
        y = ytop
        for it in items:
            if isinstance(it, (list, tuple)):
                for sub in it:
                    st = self.style(size - 2.5, "#374151")
                    st.leftIndent, st.bulletIndent = 40, 22
                    p = Paragraph(sub, st, bulletText="–")
                    _, h = p.wrap(w, PAGE_H)
                    p.drawOn(self.c, x, y - h)
                    y -= h + gap * 0.5
                y -= gap * 0.4
                continue
            st = self.style(size)
            st.leftIndent, st.bulletIndent = 20, 2
            st.bulletColor = colors.HexColor(ACCENT)
            p = Paragraph(it, st, bulletText="•")
            _, h = p.wrap(w, PAGE_H)
            p.drawOn(self.c, x, y - h)
            y -= h + gap
        return y

    def image(self, path, x, y, w, h):
        self.c.drawImage(str(path), x, y, width=w, height=h, preserveAspectRatio=True,
                         anchor="c", mask="auto")

    def table(self, data, x, ytop, col_widths, size=14, header_bg=LIGHT, highlight_rows=(),
              tbd_color=AMBER, first_left=False):
        cells = []
        for r, row in enumerate(data):
            out = []
            for j, v in enumerate(row):
                txt = str(v)
                col = tbd_color if txt == "TBD" else INK
                bold = r == 0 or (r in highlight_rows)
                al = TA_LEFT if (first_left and j == 0) else TA_CENTER
                out.append(Paragraph(txt, self.style(size, col, align=al, bold=bold)))
            cells.append(out)
        t = Table(cells, colWidths=col_widths)
        ts = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.HexColor(ACCENT)),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for r in highlight_rows:
            ts.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f0f9ff")))
        t.setStyle(TableStyle(ts))
        _, h = t.wrap(sum(col_widths), PAGE_H)
        t.drawOn(self.c, x, ytop - h)
        return ytop - h

    def tag(self, text, x, y, fg=AMBER, bg=AMBER_BG, size=12):
        c = self.c
        w = pdfmetrics.stringWidth(text, "DV-B", size) + 16
        c.setFillColor(colors.HexColor(bg))
        c.setStrokeColor(colors.HexColor(fg))
        c.setLineWidth(1)
        c.roundRect(x, y - 5, w, size + 10, 5, fill=1, stroke=1)
        c.setFillColor(colors.HexColor(fg))
        c.setFont("DV-B", size)
        c.drawString(x + 8, y + 1, text)
        return w

    def placeholder_banner(self, done: bool):
        if done:
            return
        c = self.c
        c.setFillColor(colors.HexColor(AMBER_BG))
        c.setStrokeColor(colors.HexColor(AMBER))
        c.setDash(6, 4)
        c.setLineWidth(1.5)
        c.roundRect(PAGE_W - 330, PAGE_H - 70, 300, 40, 6, fill=1, stroke=1)
        c.setDash()
        c.setFillColor(colors.HexColor(AMBER))
        c.setFont("DV-B", 17)
        c.drawCentredString(PAGE_W - 180, PAGE_H - 56, "TO BE FILLED FRIDAY")

    def stat(self, x, y, w, h, value, label, color=ACCENT):
        c = self.c
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(colors.HexColor("#e5e7eb"))
        c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
        c.setFillColor(colors.HexColor(color))
        c.setFont("DV-B", 30)
        c.drawCentredString(x + w / 2, y + h - 44, value)
        self.para(label, x + 8, y + h - 54, w - 16, size=13, color="#374151", align=TA_CENTER)

    # ---- page frame ----------------------------------------------------------------
    def new(self, title, section=None, tag=None):
        if self.n:
            self.c.showPage()
        self.n += 1
        c = self.c
        c.setFillColor(colors.HexColor(ACCENT))
        c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
        if section:
            c.setFillColor(colors.HexColor(MUTED))
            c.setFont("DV", 12)
            c.drawString(40, PAGE_H - 30, section.upper())
        c.setFillColor(colors.HexColor(INK))
        c.setFont("DV-B", 27)
        c.drawString(40, PAGE_H - 62, title)
        if tag:
            tw = pdfmetrics.stringWidth(title, "DV-B", 27)
            self.tag(tag, 40 + tw + 16, PAGE_H - 60)
        c.setStrokeColor(colors.HexColor("#e5e7eb"))
        c.setLineWidth(1)
        c.line(40, PAGE_H - 76, PAGE_W - 40, PAGE_H - 76)
        c.setFillColor(colors.HexColor(MUTED))
        c.setFont("DV", 10)
        c.drawString(40, 16, "C4 / WM-S overview  ·  16 Oct 2026")
        c.drawRightString(PAGE_W - 40, 16, f"{self.n}" + (f" / {self.total}" if self.total else ""))

    def save(self):
        self.c.showPage()
        self.c.save()


# ======================================================================================
# SLIDES
# ======================================================================================

def build(total_pages=None) -> int:
    figs = {
        "system": FIG_DIR / "fig_system.png",
        "arch": FIG_DIR / "fig_wms_arch.png",
        "loop": FIG_DIR / "fig_dreamer_loop.png",
        "wall": FIG_DIR / "fig_wall_error.png",
        "probe": FIG_DIR / "fig_truth_probe.png",
        "sumo": FIG_DIR / "fig_sumo_policy.png",
        "drivers": FIG_DIR / "fig_driver_settings.png",
        "final_sumo": FIG_DIR / "fig_final_sumo.png",
    }
    d = Deck(OUT_PDF)
    d.total = total_pages
    c = d.c
    TOP = PAGE_H - 100
    L, R = 48, PAGE_W - 48
    FULL = R - L

    # 1. Title ------------------------------------------------------------------------
    d.n += 1
    c.setFillColor(colors.HexColor(ACCENT))
    c.rect(0, 0, 18, PAGE_H, fill=1, stroke=0)
    d.para(TITLE, 70, PAGE_H - 140, 800, size=34, bold=True)
    d.para(SUBTITLE, 70, PAGE_H - 275, 800, size=20, color="#374151")
    d.para(f"{AUTHOR_LINE}<br/>{DATE_LINE}", 70, PAGE_H - 345, 800, size=16, color=MUTED)
    d.para("Work in progress. Results marked <b>preliminary</b> are from development worlds; "
           "final-evaluation slots are marked <font color='%s'><b>TO BE FILLED FRIDAY</b></font>." % AMBER,
           70, 110, 800, size=14, color=MUTED)

    # 2. Problem and setting ------------------------------------------------------------
    d.new("Problem: a highway on-ramp merge", "1 · Problem and setting")
    d.image(figs["system"], L, 195, FULL, 250)
    d.bullets([
        "<b>Goal:</b> learn per-CAV acceleration and lane commands that reduce <b>total system time</b> "
        "(time on the road + time waiting in entry queues).",
        "Let cooperative behaviour among CAVs emerge from learning, not from a hand-designed rule.",
    ], L, 180, FULL, size=16)

    # 3. Setting details ----------------------------------------------------------------
    d.new("Setting and terms", "1 · Problem and setting")
    d.bullets([
        "<b>Simulator:</b> SUMO, ~3.2 km highway with one on-ramp merge; 0.5 s physics, 1 s control.",
        "<b>Mixed traffic:</b> human-driven vehicles (<b>HDVs</b>) and connected automated vehicles "
        "(<b>CAVs</b>) at 5-40% penetration. CAVs are typical cars, not a special vehicle type.",
        "<b>RSU</b> (roadside unit): sees timestamped CAV reports (own state + detected neighbours), "
        "not the full traffic state.",
        "<b>Drivers:</b> continuous \"textbook\" IDM population centred on Treiber & Kesting values, "
        "with per-driver variation:",
        ["median a = 1.2 m/s² (max accel.), b = 2.0 m/s² (comfortable braking), "
         "T = 1.2 s (time headway), s0 = 2.0 m (standstill gap)"],
        "<b>Demand</b> 1900-2700 veh/h. The road saturates near <b>1,930 veh/h</b> mainline, so heavy demand "
        "builds entry queues; queue wait is part of the metric.",
        "<b>IDM</b> = Intelligent Driver Model (car following); <b>penetration</b> = CAV share of vehicles.",
    ], L, TOP, FULL, size=17)

    # 4. Why a world model --------------------------------------------------------------
    d.new("Why a world model?", "2 · Approach")
    y = d.bullets([
        "Learning directly in SUMO (MAPPO, multi-agent PPO) works but is <b>slow and opaque</b>: "
        "it does not tell us <i>why</i> an action helps.",
        "A learned <b>world model</b> predicts how traffic evolves given CAV actions. It lets us",
        ["train a policy by imagination (Dreamer-style), with gradients through the model;",
         "predict and rank what CAV actions do to <b>human</b> drivers before acting."],
    ], L, TOP, FULL, size=18)
    y -= 14
    c.setFillColor(colors.HexColor(LIGHT))
    c.roundRect(L, y - 168, FULL, 168, 8, fill=1, stroke=0)
    d.para("<b>Ground rules set by the project owner</b>", L + 16, y - 12, FULL - 32, size=17, color=ACCENT)
    d.bullets([
        "No designed or scripted controllers in world-model training data: only native driving, "
        "undirected random actions and the policy's own rollouts.",
        "Driver characteristics are learned from prediction error only, never from type labels.",
        "CAVs have no driver type.  ·  No control zone.",
    ], L + 16, y - 44, FULL - 32, size=16, gap=6)

    # 5. WM-S architecture --------------------------------------------------------------
    d.new("WM-S: a per-vehicle, per-step world model", "3 · How the world model works")
    d.image(figs["arch"], L, 70, FULL, 370)
    d.para("Every vehicle is advanced every 0.5 s. Physics gives the structure; learned parts correct it.",
           L, 70, FULL, size=15, color=MUTED, align=TA_CENTER)

    # 6. WM-S detail ---------------------------------------------------------------------
    d.new("WM-S: components and training", "3 · How the world model works")
    colw = FULL / 2 - 12
    d.bullets([
        "<b>IDM core.</b> Each HDV's settings (a, b, T, s0, v0, lane parameters) are inferred by a "
        "driver encoder from up to 60 s of its own history.",
        "<b>Gated residual.</b> Bounded learned acceleration correction (attention over neighbours + temporal).",
        "<b>Lanes (hazard_v5).</b> One shared network, heads for overtake / return / merge; features only, "
        "no driver inputs. A head is sampled only if its ranking quality (AP) passes a gate.",
        "<b>Entry model</b> inserts vehicles that enter the road.",
    ], L, TOP, colw, size=15)
    d.bullets([
        "<b>Curriculum:</b> 10 / 30 / 60 / 120 s rollouts.",
        "<b>Losses:</b> trajectory; aggregate (flow, field, reward); lane events.",
        "<b>Paired-difference loss</b> on SUMO branch pairs (same state, different CAV actions): "
        "teaches action effects, not just trajectories.",
        "<b>Jam oversampling.</b>",
        "<b>Data:</b> 480 native + 480 undirected random-action episodes + paired branches",
        ["v5: 36 base worlds, 60 s branches", "v6: 184 base worlds, 120 s branches"],
    ], L + colw + 24, TOP, colw, size=15)

    # 7. C4 loop -----------------------------------------------------------------------
    d.new("C4: learning CAV control inside the world model", "4 · Policy learning")
    d.image(figs["loop"], L, 95, FULL, 345)
    d.para("Dreamer-style actor-critic. The world model is the training environment; SUMO is the judge.",
           L, 92, FULL, size=15, color=MUTED, align=TA_CENTER)

    # 8. C4 detail -----------------------------------------------------------------------
    d.new("C4: policy and baselines", "4 · Policy learning")
    d.bullets([
        "<b>One shared policy</b> runs on every CAV: acceleration head + lane head (stay / left / right).",
        "<b>Imagined 30 s rollouts</b>; analytic gradients flow through the world model "
        "(CAV actions reach HDVs through their IDM response).",
        "<b>Critic</b> with discount 0.995 carries roughly 200 s of value beyond the rollout.",
        "<b>Counterfactual cooperation credit:</b> compare group vs individual counterfactual rows to "
        "estimate whether coordinated actions help.",
        "<b>SUMO-consistent execution cap:</b> imagined CAVs cannot exceed what SUMO would execute "
        "(realised ≈ min(command, own IDM acceleration)).",
        "<b>Baselines:</b> native driving, CIRCLES damping, PI controller, MAPPO trained directly in SUMO, "
        "and a trivial always-max-acceleration arm.",
    ], L, TOP, FULL, size=17)

    # 9. Contributions -----------------------------------------------------------------
    d.new("Contributions we are aiming for", "5 · Contributions")
    rows = [
        ("C1", "Zero-shot prediction of unseen controllers",
         "HDV response to CAV behaviour never seen in training. Earlier: unseen wall action, structured WM "
         "8.71 m vs black-box 25.4 / 40.1 m median HDV error at 60 s.", "earlier result"),
        ("C2", "Per-driver identification", "Driver settings recovered from prediction error alone "
         "(no labels). v6 population medians now near the truth.", "preliminary"),
        ("C3", "Fragility analysis", "When does the model (and the policy) break?", "deferred"),
        ("C4", "Learned cooperative multi-CAV control", "Policy learned inside the WM vs SUMO-trained "
         "MAPPO and classic controllers, across penetrations incl. unseen 15 / 30 / 60%.", "Friday"),
    ]
    y = TOP
    for cid, head, body, status in rows:
        c.setFillColor(colors.HexColor(ACCENT))
        c.roundRect(L, y - 70, 64, 64, 8, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("DV-B", 24)
        c.drawCentredString(L + 32, y - 46, cid)
        d.para(f"<b>{head}</b>", L + 82, y - 4, 640, size=18)
        d.para(body, L + 82, y - 30, 640, size=14, color="#374151")
        st_col = {"deferred": MUTED, "Friday": AMBER}.get(status, "#15803d")
        d.tag(status, R - 118, y - 30, fg=st_col, bg="#ffffff")
        y -= 92

    # 10. Result (a) --------------------------------------------------------------------
    d.new("(a) World-model accuracy on unseen walls", "6 · Results so far")
    d.image(figs["wall"], L, 140, FULL, 290)
    d.bullets([
        "A <b>wall</b> = upstream CAVs slow together across lanes; never in WM training data, so this tests C1. "
        "Textbook population.",
        f"v5 is best at 30 s and beyond; paired, v5 is significantly better than v4 at every horizon. "
        f"Long horizons remain hard ({BLOWUP_DEF}).",
    ], L, 132, FULL, size=14, gap=4)

    # 11. Result (b)+(c) ----------------------------------------------------------------
    d.new("(b, c) Predicting the effect of actions", "6 · Results so far")
    wb, ae = WALL_BENEFIT, ACTION_EFFECT
    d.para("<b>(b) Wall benefit</b> vs native driving", L, TOP, FULL / 2, size=17, color=ACCENT)
    tw = (FULL / 2 - 30) / 2
    d.stat(L, TOP - 160, tw, 120, f"{wb['sign_agree_30']:.2f} / {wb['sign_agree_60']:.2f}",
           f"sign agreement at 30 / 60 s (rho {wb['rho']:.2f})")
    d.stat(L + tw + 14, TOP - 160, tw, 120, f"~{wb['gain_captured_frac'] * 100:.0f}%",
           "of the best possible gain captured by choosing wall vs native from predictions")
    d.bullets([
        f"At 120 s the cost is over-predicted ({fmt(wb['pred_120_vehs'])} vs {fmt(wb['true_120_vehs'])} veh·s) "
        f"and only {wb['benefits_caught'][0]} of {wb['benefits_caught'][1]} real benefits are caught.",
    ], L, TOP - 180, FULL / 2 - 20, size=15)
    x2 = L + FULL / 2 + 10
    d.para("<b>(c) Action effects</b> on held-out paired branches", x2, TOP, FULL / 2, size=17, color=ACCENT)
    d.stat(x2, TOP - 160, tw, 120, f"{ae['sign_agree'][0]:.2f} / {ae['sign_agree'][1]:.2f}",
           "sign agreement at 30 / 60 s")
    d.stat(x2 + tw + 14, TOP - 160, tw, 120, f"{ae['pearson'][0]:.2f} / {ae['pearson'][1]:.2f}",
           "Pearson correlation at 30 / 60 s")
    d.bullets([
        f"Large effects are ranked well; <b>close calls</b> between actions only ~{ae['close_calls']:.2f} right.",
        "Random actions almost always hurt, so effects in this set are large.",
    ], x2, TOP - 180, FULL / 2 - 20, size=15)
    d.para("v5 world model. Paired branch = same SUMO state, different CAV actions.", L, 70, FULL,
           size=13, color=MUTED)

    # 12. Result (d) diagnosis -----------------------------------------------------------
    d.new("(d) Diagnosis: what limits v5?", "6 · Results so far")
    d.image(figs["probe"], L - 6, 85, 430, 340)
    d.bullets([
        "<b>True-settings probe:</b> plain IDM physics given the <i>true</i> driver settings is far more "
        "accurate, with 0 blow-ups. <b>Identification is the main lever.</b>",
        f"v5's encoder could not go below a = {V5_SETTING_FLOOR['a']} and b = {V5_SETTING_FLOOR['b']} m/s² "
        "(range limit), above the true 1.2 / 2.0.",
        f"Most blow-ups come from the merge zone: ramp vehicles merge after "
        f"{MERGE_DELAY_S['v5 model']:g} s in the model vs {MERGE_DELAY_S['SUMO']:g} s in SUMO (medians).",
        f"On saturated entrances the entry insertion rule refused {ENTRY_REFUSED} of SUMO's actual "
        "insertions (fixed for v6).",
    ], L + 440, TOP, FULL - 440, size=15.5)
    d.para("True settings are used for evaluation only, never as model input.", L + 440, 85, FULL - 440,
           size=13, color=MUTED)

    # 13. Result (e) SUMO -----------------------------------------------------------------
    d.new("(e) Policies in SUMO", "6 · Results so far", tag="preliminary")
    d.image(figs["sumo"], L, 215, FULL, 225)
    d.para("12 seeds × demand 1900 / 2400 × penetration 10 / 20 / 40%. Mean ± interval as reported. "
           f"The v5 policy's loss is mostly entry-queue wait (+{V5_POLICY_ENTRY_WAIT_VEHS} veh·s).",
           L, 212, FULL, size=13, color=MUTED)
    d.para("<b>Why CIRCLES and PI do not help here</b>", L, 172, FULL, size=16, color=ACCENT)
    d.bullets([
        "They are designed to damp stop-and-go waves; this bottleneck is <b>capacity-limited</b> at the merge.",
        "Larger gaps cost capacity, and lost capacity becomes entry-queue wait, which the metric counts.",
    ], L, 146, FULL, size=15, gap=4)

    # 14. Result (f) v6 drivers ---------------------------------------------------------
    d.new("(f) v6 world model: driver identification", "6 · Results so far", tag="early")
    d.image(figs["drivers"], L, 180, FULL, 255)
    d.bullets([
        "Population median of the encoder's per-driver estimates. v6 is close to the truth for b, T and s0; "
        "a is now <i>under</i>-estimated (0.86 vs 1.2).",
        f"10 s validation error is {V6_VAL10_GAIN} lower than v5 at matched training steps. "
        "Long-horizon v6 scores: Friday.",
    ], L, 170, FULL, size=15)

    # 15. What v6 changes ---------------------------------------------------------------
    d.new("What v6 changes", "7 · v6")
    colw = FULL / 2 - 12
    d.para("<b>World model</b>", L, TOP, colw, size=18, color=ACCENT)
    d.bullets([
        "Reachable driver-parameter ranges with a learned centre (still prediction-error only).",
        "Emergency braking.",
        "Stop-within-step integration.",
        "Cleaner paired training: shared randomness, a delay-difference target, a ranking term.",
        "120 s long branches.",
        "Merge-zone fix as a v6m fine-tune <i>(in progress)</i>.",
    ], L, TOP - 34, colw, size=15.5)
    d.para("<b>Policy</b>", L + colw + 24, TOP, colw, size=18, color=ACCENT)
    d.bullets([
        "Longer imagination credit (gradient decay 0.98).",
        "Auto-tuned exploration target (the v5 policy's exploration collapsed).",
        "SUMO-consistent entry rule.",
        "Zero-centred start.",
    ], L + colw + 24, TOP - 34, colw, size=15.5)

    # 16. Placeholder: WM wall table -----------------------------------------------------
    done1 = slot_complete(FINAL_WM_WALL) and slot_complete(FINAL_WM_BENEFIT)
    d.new("v6 vs v5: unseen-wall accuracy", "8 · Final results")
    d.placeholder_banner(done1)
    hdr = ["Model"] + [f"{h} s" for h in HORIZONS_S] + ["Blow-ups"]
    data = [hdr] + [[m] + [fmt(v) for v in vals] + [fmt(b)] for m, (vals, b) in FINAL_WM_WALL.items()]
    d.table(data, L + 120, TOP - 20, [130, 110, 110, 110, 110, 130], size=17, highlight_rows=(1,))
    d.para(f"Median HDV position error (m). {FINAL_WM_WALL_NOTE} {BLOWUP_DEF.capitalize()}.",
           L + 120, TOP - 122, 700, size=13, color=MUTED)
    data = [["Wall-benefit prediction", "v6", "v5"]] + [
        [k, fmt(v6), fmt(v5)] for k, (v6, v5) in FINAL_WM_BENEFIT.items()]
    d.table(data, L + 120, TOP - 175, [400, 150, 150], size=15, first_left=True)
    d.para("Question: does better driver identification (slide f) carry over to 60-120 s accuracy, "
           "fewer blow-ups and better wall decisions?", L + 120, 95, 700, size=14, color="#374151")

    # 17. Placeholder: SUMO final eval ----------------------------------------------------
    done2 = slot_complete(FINAL_SUMO)
    g = FINAL_EVAL_GRID
    d.new("Final SUMO evaluation", "8 · Final results")
    d.placeholder_banner(done2)
    hdr = ["Controller"] + FINAL_SUMO_COLUMNS
    data = [hdr] + [[k] + [fmt(v) for v in vals] for k, vals in FINAL_SUMO.items()]
    have_fig = fig_final_sumo(figs["final_sumo"])
    tw_total = 560 if have_fig else 740
    cw = [tw_total * 0.31] + [tw_total * 0.69 / len(FINAL_SUMO_COLUMNS)] * len(FINAL_SUMO_COLUMNS)
    d.table(data, L, TOP - 10, cw, size=14.5, highlight_rows=(1,))
    if have_fig:
        d.image(figs["final_sumo"], L + tw_total + 15, 130, FULL - tw_total - 15, 280)
    seen = "/".join(str(p) for p in g["pen_seen_pct"])
    unseen = "/".join(str(p) for p in g["pen_unseen_pct"])
    dem = "/".join(str(x) for x in g["demands_veh_h"])
    d.para(
        f"Δ system time per vehicle vs native (s), negative = better. Final evaluation worlds: "
        f"{len(g['demands_veh_h'])} demands ({dem} veh/h) × {len(g['pen_seen_pct']) + len(g['pen_unseen_pct'])} "
        f"penetrations × {g['seeds']} seeds. Seen penetrations {seen}%; unseen {unseen}%. "
        f"Light = {'/'.join(map(str, g['light_demands']))}, heavy = {'/'.join(map(str, g['heavy_demands']))} veh/h.",
        L, 128, FULL, size=13, color=MUTED)
    d.para("Also report: entry-queue wait, braking, collisions/teleports, and intervals for every cell.",
           L, 80, FULL, size=13, color=MUTED)

    # 18. Placeholder: cooperation -------------------------------------------------------
    done3 = slot_complete(FINAL_COOP)
    d.new("Does cooperation emerge?", "8 · Final results")
    d.placeholder_banner(done3)
    data = [["Behaviour measured in SUMO (v6 policy)", "Result"]] + [[k, fmt(v)] for k, v in FINAL_COOP.items()]
    d.table(data, L + 60, TOP - 10, [560, 200], size=15, first_left=True)
    d.para("No wall or cooperation rule is given to the policy or the world model; any coordinated formation "
           "was found by learning. Compare against individual damping at low penetration.",
           L + 60, 140, 760, size=14, color=MUTED)

    # 19. Limitations and next steps ------------------------------------------------------
    d.new("Limitations and next steps", "9 · Outlook")
    colw = FULL / 2 - 12
    d.para("<b>Limitations</b>", L, TOP, colw, size=18, color=ACCENT)
    d.bullets([
        "Long-horizon accuracy is still limited: blow-ups, 120 s error.",
        "The learned residual still carries about half of each acceleration.",
        "The policy learns in an imperfect model, so checkpoints are selected by SUMO checks.",
        "So far the learned policy has not beaten native in SUMO; MAPPO has.",
    ], L, TOP - 34, colw, size=16)
    d.para("<b>Next steps</b>", L + colw + 24, TOP, colw, size=18, color=ACCENT)
    d.bullets([
        "Merge-zone fix (v6m).",
        "Longer paired branches.",
        "Uncertainty-aware planning: act on the model only where it is confident.",
        "C3 fragility analysis.",
    ], L + colw + 24, TOP - 34, colw, size=16)

    # 20. Summary -----------------------------------------------------------------------
    d.new("Summary", "")
    d.bullets([
        "<b>WM-S</b>: IDM physics + per-driver identification + bounded learned corrections, rolled out per vehicle.",
        "<b>C1</b>: structured model predicts unseen controllers better than black boxes; v5 improves on earlier "
        "versions but long horizons remain hard.",
        "<b>Diagnosis</b> points at driver identification and the merge zone; v6 targets both.",
        "<b>C4</b>: policy learned in the model; in preliminary SUMO checks the v5 policy did not beat native "
        "(MAPPO did). Final v6 evaluation: see the final-results slides.",
    ], L, TOP, FULL, size=18, gap=12)

    d.save()
    return d.n


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _register_fonts()
    fig_system(FIG_DIR / "fig_system.png")
    fig_wms_arch(FIG_DIR / "fig_wms_arch.png")
    fig_dreamer_loop(FIG_DIR / "fig_dreamer_loop.png")
    fig_wall_error(FIG_DIR / "fig_wall_error.png")
    fig_truth_probe(FIG_DIR / "fig_truth_probe.png")
    fig_sumo_policy(FIG_DIR / "fig_sumo_policy.png")
    fig_driver_settings(FIG_DIR / "fig_driver_settings.png")
    n = build()  # first pass counts pages
    build(total_pages=n)
    print(f"wrote {OUT_PDF} ({n} pages)")


if __name__ == "__main__":
    os.chdir(HERE.parents[1])
    main()
