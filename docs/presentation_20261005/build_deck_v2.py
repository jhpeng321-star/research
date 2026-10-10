"""World-model-only deck (v2). Regenerate WM_world_model_20261005.pptx, OUTLINE.md and figures/*.png from numbers.json.

Usage (rl_driving env):  python build_deck.py
Edit numbers.json only: every null value in repl / closed_loop / a12 renders as a
'PENDING (result expected HH:MM)' placeholder; fill it and rebuild.

numbers.json keys
  taus                       list of horizons (s)
  models[]                   name/change/ckpt rows for the progression table
  slice_regret.{A11,A10,A9,A8,always_native,random}   regret (s) per tau; a11_vs_native_300/570 {a11,native,ci90}
  pi_native_crps.{A11,A10,A9,A8,M1FIX1,climatology}   CRPS (s) per tau; a11_300_ci
  noise.{ratio,ci_lo,ci_hi,n1_sd,pi_sd}               PI-native sd / one-CAV-nudge sd per tau
  t4g.{clim_crps,hgb_crps,clim_sign,hgb_sign}         predictability from t0 state
  dspeed.*                                            direct-speed campaign counts
  repl.{expected, a11_regret_300, native_regret_300, ci90_diff_300 ("[lo, hi]" string or list),
        a10_regret_300, a9_regret_300, other_taus{30,60,120,570}, verdict}   PENDING until filled
  closed_loop.{expected, pl01_a11, fleet_wm, always_native, always_pi, random, pl01_a11_walls,
        diff_pl01_vs_native_ci90, walls_cutoff_note}                          PENDING until filled
        (set pl01_a11_walls to the string "NOT DONE" if the 17:00 cutoff is missed)
  a13.{expected, crps (list of 5 or null), verdict}   PENDING until filled
  a12.{expected, crps_30, crps_300, ref{...}}                                  PENDING until filled
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf-8"))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
TAUS = N["taus"]

NAVY, TEAL, GREY, LIGHT = "1F3A5F", "0E7C86", "444444", "EEF3F7"
AMBER_BG, AMBER = "FFF3CD", "C77700"
C = {"A11": "#0E7C86", "A10": "#6BAED6", "A9": "#9E9AC8", "A8": "#BDBDBD", "native": "#1F3A5F",
     "random": "#E08A5B", "clim": "#C77700", "M1": "#B03A48"}
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
TAU_LABELS = [f"{t} s" for t in TAUS]


def rgb(h): return RGBColor.from_string(h)
def pending(v): return v is None or (isinstance(v, str) and v.strip().upper() == "PENDING")
def ptxt(exp): return f"PENDING (result expected {exp})"


def show(v, exp, fmt="{}"):
    if pending(v): return ptxt(exp)
    if isinstance(v, (list, tuple)): return "[" + ", ".join(str(x) for x in v) + "]"
    if isinstance(v, (int, float)): return fmt.format(v)
    return str(v)


# ----------------------------------------------------------------- figures
def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); return p


def fig_noise():
    n = N["noise"]; r = n["ratio"]
    lo = [a - b for a, b in zip(r, n["ci_lo"])]; hi = [b - a for a, b in zip(r, n["ci_hi"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(TAU_LABELS, r, color=C["A11"], yerr=[lo, hi], capsize=6, ecolor="#333")
    ax.axhline(1, color="#B03A48", ls="--", lw=2, label="noise floor (ratio 1)"); ax.legend(loc="upper right", fontsize=12)
    ax.set_yscale("log"); ax.set_ylim(0.8, 3000)
    for i, v in enumerate(r): ax.text(i + 0.12, v * 1.05, f"{v:g}x", ha="left", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("sd(PI-native) / sd(one-CAV nudge)\n(log scale, 90% CI)"); ax.set_xlabel("prediction horizon")
    ax.set_title("PI effect spread vs null-perturbation noise floor (delay, s)", fontsize=14)
    return save(fig, "noise_floor.png")


def fig_regret():
    s = N["slice_regret"]
    ser = [("A11", s["A11"], C["A11"]), ("A10", s["A10"], C["A10"]), ("A9", s["A9"], C["A9"]), ("A8", s["A8"], C["A8"]),
           ("always-native", s["always_native"], C["native"]), ("random", s["random"], C["random"])]
    fig, ax = plt.subplots(figsize=(11, 5.2)); w = 0.13; x = range(len(TAUS))
    for k, (lab, v, col) in enumerate(ser):
        ax.bar([i + (k - 2.5) * w for i in x], v, w, label=lab, color=col, edgecolor="white")
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(TAU_LABELS)
    ax.set_ylabel("decision regret, s of total delay\n(lower is better, log scale)"); ax.set_xlabel("decision horizon")
    ax.legend(ncol=3, fontsize=11, loc="upper left")
    i = TAUS.index(300)
    ax.annotate(f"A11 {s['A11'][i]:g} s\nvs native {s['always_native'][i]:g} s", xy=(i - 2.5 * w, s["A11"][i]), xytext=(1.1, 4500),
                fontsize=12, fontweight="bold", color=C["A11"], arrowprops=dict(arrowstyle="->", color=C["A11"]))
    ax.set_ylim(0.3, 3e4)
    return save(fig, "regret_slice.png")


def fig_crps():
    d = N["pi_native_crps"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, col, ls, lw in [("M1FIX1", C["M1"], ":", 2), ("A8", C["A8"], "-", 2), ("A9", C["A9"], "-", 2),
                           ("A10", C["A10"], "-", 2), ("A11", C["A11"], "-", 3.5), ("climatology", C["clim"], "--", 3)]:
        ax.plot(TAU_LABELS, d[k], ls, color=col, lw=lw, marker="o", label=k if k != "climatology" else "TRAIN climatology")
    ax.set_yscale("log"); ax.set_ylabel("CRPS, seconds of paired delay\n(lower is better, log scale)"); ax.set_xlabel("prediction horizon")
    ax.legend(fontsize=11, ncol=2); ax.set_title("PI-vs-native paired effect, CAL-tune, 11 decisions", fontsize=14)
    return save(fig, "crps_models.png")


def fig_t4g():
    t = N["t4g"]
    fig, ax = plt.subplots(figsize=(9, 5)); w = 0.38; x = range(len(TAUS))
    ax.bar([i - w / 2 for i in x], t["clim_crps"], w, color=C["clim"], label="climatology (marginal)")
    ax.bar([i + w / 2 for i in x], t["hgb_crps"], w, color=C["A10"], label="HGB on 50 t0-state features")
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(TAU_LABELS)
    ax.set_ylabel("CRPS, s of delay (lower is better, log)"); ax.set_xlabel("prediction horizon"); ax.legend(fontsize=12)
    ax.set_title("World-grouped 5-fold CV, 161 PI-native pairs (TRAIN)", fontsize=14)
    return save(fig, "t4g_predictability.png")


def box(ax, x, y, w, h, text, fc="#EEF3F7", ec="#1F3A5F", fs=12, bold=False, tc="#1F3A5F"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=2, color="#444"))


def fig_model():
    fig, ax = plt.subplots(figsize=(13, 4.8)); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.1, 3.0, 2.3, 1.4, "RSU-observable\nstate at t0\n(CAV self state +\ndetected neighbours)", fs=11)
    box(ax, 0.1, 0.3, 2.3, 1.8, "Candidate plan\n(controller / wall /\ndirect-speed;\nplan summary\n120 s)", fc="#FFF3CD", ec="#C77700", tc="#7A4B00", fs=11)
    box(ax, 3.2, 2.9, 3.4, 1.7, "Rollout simulator\nCAVs: real controller code\nHDVs: IDM/MOBIL +\nneural residual", fs=12, bold=True)
    box(ax, 3.2, 0.6, 3.4, 1.4, "Predicted CAV/HDV\ntrajectories", fs=12)
    box(ax, 7.4, 1.3, 3.0, 2.4, "Plan-aware effect head\nclimatology anchor\n+ gated plan-aware\nresidual\n(per-tau gate in A12)", fc="#D5EEF0", ec="#0E7C86", tc="#0A5A62", fs=11, bold=True)
    box(ax, 11.0, 1.3, 1.9, 2.4, "Quantiles of\ndelay change\nvs native at\n30-570 s", fs=12)
    arrow(ax, 2.4, 3.7, 3.2, 3.7); arrow(ax, 2.4, 1.3, 3.2, 3.3); arrow(ax, 4.9, 2.9, 4.9, 2.0)
    arrow(ax, 6.6, 1.3, 7.4, 2.0); arrow(ax, 6.6, 3.7, 7.4, 3.0); arrow(ax, 10.4, 2.5, 11.0, 2.5)
    ax.text(8.9, 0.35, "trained with ranking loss within each decision\n(rank loss only at tau >= 60 s in A12)", ha="center", fontsize=11, color="#555", style="italic")
    return save(fig, "model_diagram.png")


def fig_planner():
    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = ["1. Individual check\nper CAV: native,\ndamping, PI\n(others held)",
             "2. Group CAVs by\nmeasured interaction\n(step-1 predicted\ntrajectories)",
             "3. Group joint\ncheck (incl. wall\ncandidates)",
             "4. Switch rule\nvs current plan",
             "5. Combine +\nsafety fallback"]
    w, g = 2.25, 0.4
    for i, s in enumerate(steps):
        x = 0.1 + i * (w + g)
        hi = i in (0, 2)
        box(ax, x, 0.8, w, 2.2, s, fs=11, fc="#D5EEF0" if hi else "#EEF3F7", ec=C["A11"] if hi else "#1F3A5F")
        if i < 4: arrow(ax, x + w, 1.9, x + w + g, 1.9)
    ax.text(6.5, 0.25, "re-decision every 60 s;  predictor = world model (A11) behind the planner Predictor interface", ha="center", fontsize=12, color="#555", style="italic")
    return save(fig, "planner_diagram.png")


def fig_setting():
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis("off")
    for y in (3.2, 2.6): ax.add_patch(Rectangle((0, y), 10, 0.55, fc="#D9D9D9", ec="#888"))
    ax.plot([0, 3.5, 5.5], [1.3, 1.3, 2.6], color="#888", lw=22, solid_capstyle="butt", alpha=0.5)
    ax.text(1.2, 0.7, "on-ramp", fontsize=12, color="#555"); ax.text(8.2, 3.95, "mainline", fontsize=12, color="#555")
    for x, y, c in [(0.8, 3.35, "h"), (2.0, 3.35, "c"), (3.3, 3.35, "h"), (4.6, 3.35, "h"), (6.2, 3.35, "c"), (7.5, 3.35, "h"),
                    (1.1, 2.75, "h"), (2.8, 2.75, "c"), (4.0, 2.75, "h"), (6.9, 2.75, "h"), (8.4, 2.75, "c"), (2.0, 1.45, "c")]:
        ax.add_patch(Rectangle((x, y - 0.1), 0.55, 0.3, fc="#0E7C86" if c == "c" else "#9E9E9E", ec="#222"))
    ax.plot([5.5, 5.5], [3.8, 4.7], color="#C77700", lw=3); ax.plot(5.5, 4.75, "^", color="#C77700", ms=16)
    ax.text(5.5, 5.1, "RSU: timestamped CAV state + detected neighbours\n(no true penetration, no hidden driver type)", ha="center", fontsize=11, color="#7A4B00")
    ax.add_patch(Rectangle((0.2, 0.1), 0.4, 0.22, fc="#0E7C86", ec="#222")); ax.text(0.7, 0.12, "CAV", fontsize=12)
    ax.add_patch(Rectangle((1.6, 0.1), 0.4, 0.22, fc="#9E9E9E", ec="#222")); ax.text(2.1, 0.12, "HDV (IDM, never commanded)", fontsize=12)
    return save(fig, "setting.png")


def fig_closed():
    c = N["closed_loop"]
    items = [("PL01+A11", c["pl01_a11"]), ("fleet-WM", c["fleet_wm"]), ("always-native", c["always_native"]),
             ("always-PI", c["always_pi"]), ("random", c["random"]), ("PL01+A11+walls", c["pl01_a11_walls"])]
    items = [(k, v) for k, v in items if isinstance(v, (int, float))]
    p = os.path.join(FIG, "closed_loop.png")
    if len(items) < 2:
        if os.path.exists(p): os.remove(p)
        return None
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar([k for k, _ in items], [v for _, v in items], color=[C["A11"] if k.startswith("PL01") else C["native"] for k, _ in items])
    ax.set_ylabel("total delay, s (lower is better)"); plt.setp(ax.get_xticklabels(), rotation=20)
    return save(fig, "closed_loop.png")


def build_figs():
    return {k: f() for k, f in [("noise", fig_noise), ("regret", fig_regret), ("crps", fig_crps), ("t4g", fig_t4g),
                                ("model", fig_model), ("planner", fig_planner), ("setting", fig_setting), ("closed", fig_closed)]}


# ----------------------------------------------------------------- deck helpers
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
OUTLINE = []; WARN = []


def est_lines(text, width_in, pt):
    cpl = max(1, int(width_in * 72 / (pt * 0.50)))
    return sum(max(1, -(-len(p) // cpl)) for p in text.split("\n"))


def tb(s, x, y, w, h, paras, size=22, color=GREY, bold=False, align=None, anchor=MSO_ANCHOR.TOP, fill=None, line=None, name=""):
    shp = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill: shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
    if line: shp.line.color.rgb = rgb(line); shp.line.width = Pt(2)
    tf = shp.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.12)
    if isinstance(paras, str): paras = [paras]
    total = 0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sz, b, col = size, bold, color
        if isinstance(p, tuple): p, opts = p; sz = opts.get("size", sz); b = opts.get("bold", b); col = opts.get("color", col)
        r = para.add_run(); r.text = p; r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = rgb(col); r.font.name = "Calibri"
        if align: para.alignment = align
        para.space_after = Pt(6)
        total += est_lines(p, w - 0.3, sz) * sz * 1.2 + 6
    if total / 72 > h + 0.05:
        WARN.append(f"possible overflow slide {len(prs.slides)} box '{name or str(paras[0])[:30]}': need {total/72:.2f}in have {h}in")
    return shp


def new_slide(title, notes, outline_lines):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = rgb(TEAL); bar.line.fill.background()
    tb(s, 0.5, 0.3, 12.3, 1.1, title, size=30 if len(title) < 70 else 26, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="title")
    tb(s, 0.5, 7.05, 9, 0.35, FOOT, size=11, color="888888")
    tb(s, 12.2, 7.05, 0.8, 0.35, str(len(prs.slides)), size=11, color="888888", align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    OUTLINE.append((len(prs.slides), title, outline_lines, notes))
    return s


def pic(s, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def table(s, x, y, w, rows, col_w=None, size=15, header_fill=NAVY, row_h=0.42):
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr)).table
    if col_w:
        for i, cw in enumerate(col_w): gt.columns[i].width = Inches(cw)
    for r in range(nr):
        gt.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = gt.cell(r, c); v = str(rows[r][c]); cell.text = v
            cell.margin_left = cell.margin_right = Inches(0.06); cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]; f = p.runs[0].font; f.size = Pt(size); f.name = "Calibri"
            is_p = v.startswith("PENDING")
            cell.fill.solid()
            if r == 0: cell.fill.fore_color.rgb = rgb(header_fill); f.color.rgb = rgb("FFFFFF"); f.bold = True
            elif is_p: cell.fill.fore_color.rgb = rgb(AMBER_BG); f.color.rgb = rgb("7A4B00"); f.bold = True
            else: cell.fill.fore_color.rgb = rgb("FFFFFF" if r % 2 else LIGHT); f.color.rgb = rgb("222222")
            if c > 0 and r > 0 and not is_p and len(v) < 28: p.alignment = PP_ALIGN.CENTER
            cw = (col_w[c] if col_w else w / nc) - 0.15
            if est_lines(v, cw, size) * size * 1.2 / 72 > row_h + 0.05:
                WARN.append(f"possible table cell overflow slide {len(prs.slides)} r{r}c{c}: '{v[:30]}'")
    return gt


def pending_box(s, x, y, w, h, exp, what):
    tb(s, x, y, w, h, [(ptxt(exp), {"bold": True, "size": 24, "color": "7A4B00"}), (what, {"size": 16, "color": "7A4B00"})],
       fill=AMBER_BG, line=AMBER, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, name="pending")


def takeaway(s, text, y=6.3, h=0.7):
    tb(s, 0.5, y, 12.3, h, text, size=20, color="FFFFFF", bold=True, fill=TEAL, anchor=MSO_ANCHOR.MIDDLE, name="takeaway")


def bullets(s, items, x=0.5, y=1.5, w=12.3, h=4.6, size=22):
    return tb(s, x, y, w, h, ["• " + i if not i.startswith(" ") else "    – " + i.strip() for i in items], size=size, name="bullets")




# ================================================================= v2 (world-model-only deck)
def fig_inputs():
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 0.3, 4.4, 4.0, "RSU snapshot at time t0\n\nup to 192 vehicles, each:\n  position, speed, lane\n  CAV or detected neighbour\n\nentrance counters\n(vehicles entering, per 10 s)", fs=15)
    box(ax, 5.1, 2.5, 3.7, 1.8, "State encoder\n(one token per\nvehicle + counters)", fc="#D5EEF0", ec="#0E7C86", tc="#0A5A62", fs=12, bold=True)
    box(ax, 9.5, 2.5, 3.4, 1.8, "Learned scene\nrepresentation\n(to dynamics +\neffect head)", fs=12)
    box(ax, 5.3, 0.3, 7.5, 1.5, "NOT available: hidden driver types, true CAV\npenetration, full traffic state, any global metric", fc="#F6C9C9", ec="#B03A48", tc="#7A1F2B", fs=12)
    arrow(ax, 4.5, 3.4, 5.1, 3.4); arrow(ax, 8.8, 3.4, 9.5, 3.4)
    return save(fig, "wm_inputs.png")


def fig_arch():
    fig, ax = plt.subplots(figsize=(14, 6.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
    box(ax, 0.1, 2.6, 2.0, 1.4, "Snapshot\nat t0", fs=14, bold=True)
    box(ax, 2.7, 2.6, 2.0, 1.4, "State\nencoder", fc="#D5EEF0", ec="#0E7C86", tc="#0A5A62", fs=14, bold=True)
    box(ax, 5.2, 4.2, 5.6, 2.3, "(a) Dynamics, rolled forward every 0.5 s\nCAVs: REAL controller code (candidate plan)\nHuman drivers: IDM car-following\n+ MOBIL lane change + learned neural\ncorrection (acceleration distribution\n+ lane change)", fs=11, bold=True)
    box(ax, 11.3, 4.5, 2.6, 1.7, "Predicted trajectories\n+ 120 s CAV summary\n(count, mean speed,\nmin speed per 10 s)", fs=11)
    box(ax, 5.4, 0.2, 5.2, 3.4, "(b) Effect head (plan-aware)\nchange in total delay vs native\nat 30 / 60 / 120 / 300 / 570 s\nas 19 quantiles (5% .. 95%)\n\n= TRAIN average effect\n+ gate x learned correction", fc="#D5EEF0", ec="#0E7C86", tc="#0A5A62", fs=13, bold=True)
    box(ax, 11.1, 0.9, 2.8, 2.0, "Delay-change\ndistribution\nper horizon\n(with uncertainty)", fs=13, bold=True)
    box(ax, 0.1, 5.2, 2.6, 1.2, "Candidate CAV\ncontroller (plan)", fc="#FFF3CD", ec="#C77700", tc="#7A4B00", fs=13)
    arrow(ax, 2.1, 3.3, 2.7, 3.3); arrow(ax, 4.7, 3.6, 5.2, 4.8); arrow(ax, 4.7, 3.0, 5.4, 2.3)
    arrow(ax, 2.7, 5.8, 5.2, 5.6); arrow(ax, 10.8, 5.35, 11.3, 5.35)
    arrow(ax, 12.5, 4.6, 10.6, 2.6); arrow(ax, 10.6, 1.9, 11.1, 1.9)
    ax.text(0.2, 1.4, "Training: ranking loss within a decision;\nper-horizon loss scaling.\nTraining-only signals: arrival head,\ndriver-response contrast.", fontsize=11, color="#555", style="italic", va="center")
    return save(fig, "wm_arch.png")


def fig_learn():
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 1.6, 2.6, 1.6, "Freeze one SUMO\ntraffic moment\n(identical start)", fs=14, bold=True)
    for lab, y in [("native", 3.35), ("PI controller", 1.95), ("another plan", 0.55)]:
        box(ax, 3.6, y, 2.8, 1.0, f"replay under\n{lab}", fs=13, fc="#FFF3CD", ec="#C77700", tc="#7A4B00")
        arrow(ax, 2.7, 2.4, 3.6, y + 0.5); arrow(ax, 6.4, y + 0.5, 7.2, 2.4)
    box(ax, 7.2, 1.6, 2.6, 1.6, "total delay of\neach replay (s)", fs=14)
    arrow(ax, 9.8, 2.4, 10.4, 2.4)
    box(ax, 10.4, 1.4, 2.5, 2.0, "TRUE effect label\n= delay(plan) -\ndelay(native)\nsame start", fc="#D5EEF0", ec="#0E7C86", tc="#0A5A62", fs=13, bold=True)
    return save(fig, "wm_learn.png")


def fig_quality():
    d = N["pi_native_crps"]; clim = d["climatology"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    series = [("A8", d["A8"], C["A8"], 2.2, 0.9), ("A9", d["A9"], C["A9"], 2.2, 0.9), ("A10", d["A10"], C["A10"], 2.2, 0.9),
              ("A11", d["A11"], C["A11"], 3.2, 1), ("A12", N["a12"]["crps"], "#B03A48", 3.2, 1)]
    for lab, v, col, lw, al in series:
        ax.plot(TAU_LABELS, [a / b for a, b in zip(v, clim)], "-o", color=col, lw=lw, alpha=al, label=lab)
    a13 = N["a13"]["crps"]
    if not pending(a13) and isinstance(a13, list):
        ax.plot(TAU_LABELS, [a / b for a, b in zip(a13, clim)], "-s", color="#222", lw=3.2, label="A13")
    else:
        ax.text(0.97, 0.05, "A13: PENDING (expected %s)" % N["a13"]["expected"], transform=ax.transAxes, ha="right", fontsize=13,
                fontweight="bold", color="#7A4B00", bbox=dict(fc="#"+AMBER_BG, ec="#"+AMBER))
    ax.axhline(1, color=C["clim"], ls="--", lw=2.5); ax.text(4.1, 1.0, "TRAIN average\n(climatology)", color=C["clim"], fontsize=11, va="center")
    ax.set_xlim(-0.2, 5.1)
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_ylabel("CRPS / CRPS of TRAIN average\n(lower is better; <1 beats the average)"); ax.set_xlabel("prediction horizon")
    ax.set_title("PI-vs-native effect, CAL-tune, 11 decisions", fontsize=14); ax.legend(fontsize=11, ncol=2, loc="upper right")
    return save(fig, "wm_quality.png")


def fig_repl():
    sr, rp = N["slice_regret"]["a11_vs_native_300"], N["repl"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2)); w = 0.36
    g = [("Evaluation slice\n(11 worlds; A11 chosen here)", sr["a11"], sr["native"]), ("Pre-registered replication\n(30 fresh worlds)", rp["a11_regret_300"], rp["native_regret_300"])]
    for i, (lab, a, b) in enumerate(g):
        ax.bar(i - w / 2, a, w, color=C["A11"], label="A11" if i == 0 else None)
        ax.bar(i + w / 2, b, w, color=C["native"], label="always-native" if i == 0 else None)
        ax.text(i - w / 2, a + 25, f"{a:g}", ha="center", fontsize=14, fontweight="bold"); ax.text(i + w / 2, b + 25, f"{b:g}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels([x[0] for x in g], fontsize=12); ax.set_ylim(0, 1900)
    ax.set_ylabel("decision regret at 300 s, s of total delay\n(lower is better)"); ax.legend(fontsize=12, loc="upper right")
    return save(fig, "wm_repl.png")


def fig_ncav():
    d = N["diag1"]["n_cav_bins"]; x = [10 * (i + 1) for i in range(12)]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    st = {"recorded (real future)": ("#222", "-", 3.5), "closed road (A11 rollout)": ("#B03A48", "-", 3), "open road (exits+entries)": (C["A11"], "--", 3)}
    for k, v in d.items():
        c, ls, lw = st[k]; ax.plot(x, v, ls, color=c, lw=lw, marker="o", label=k)
    ax.set_xlabel("time into the 120 s plan window, s"); ax.set_ylabel("CAVs on the road, count per 10 s bin")
    ax.set_title("TRAIN state w8676m2700p20t360", fontsize=14); ax.legend(fontsize=12)
    return save(fig, "wm_ncav.png")


def fig_mae():
    e = N["entry_fix"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cols = ["#B03A48", "#E08A5B", C["A11"], "#444444"]
    ax.bar(range(4), e["mae_s"], color=cols)
    for i, v in enumerate(e["mae_s"]): ax.text(i, v + 50, f"{v:g}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(["closed\nroad", "exits\nonly", "open road\n(exits+entries)", "real recorded\nplan"], fontsize=12)
    ax.set_ylabel("mean abs error of PI effect at 300 s, s\n(lower is better; 8 TRAIN states)"); ax.set_ylim(0, 4100)
    return save(fig, "wm_mae.png")


def build_figs():
    return {k: f() for k, f in [("noise", fig_noise), ("setting", fig_setting), ("inputs", fig_inputs), ("arch", fig_arch), ("learn", fig_learn),
                                ("quality", fig_quality), ("repl", fig_repl), ("ncav", fig_ncav), ("mae", fig_mae)]}


FOOT = "World model update | 2026-10-05 | lower is better for all delay / regret / CRPS numbers (seconds)"


def build_deck(F):
    sr, rp, cl, a13 = N["slice_regret"], N["repl"], N["closed_loop"], N["a13"]
    pexp = a13["expected"]
    ci = rp["ci90_diff_300"]

    # 1 title
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = rgb(NAVY); bg.line.fill.background()
    tb(s, 0.7, 0.6, 12, 0.8, "World model update | 2026-10-05", size=22, color="9FD3D8", bold=True)
    head = ("A hybrid physics + neural world model that predicts how human drivers react to CAV controllers, "
            "and how much each plan changes traffic delay, with uncertainty.")
    tb(s, 0.7, 1.8, 12, 3.2, head, size=36, color="FFFFFF", bold=True, name="headline")
    tb(s, 0.7, 5.5, 12, 1.2, ["Where it works, where it does not yet, and what we found tonight.", "Numbers come from verified reports; amber boxes mark pending results."], size=20, color="CFE3EA")
    notes = ("One sentence: the model is a hybrid. Physics (real controller code, IDM car-following, MOBIL lane change) plus a neural correction for how human drivers respond, "
             "and a head that outputs a distribution of the change in total delay for a candidate CAV plan. The talk is an honest status of that model.")
    s.notes_slide.notes_text_frame.text = notes
    OUTLINE.append((1, "Title", [head], notes))

    # 2 why
    s = new_slide("Why a world model: whether a controller helps depends on how human drivers react",
        "Context only. An RSU at a highway merge can ask CAVs to drive in different ways (controllers). Whether a controller reduces total delay depends on how the surrounding human drivers react, and that reaction is not directly observable. So we need a model of that reaction that can score a candidate controller before it is used. The rest of the talk is about that model.",
        ["RSU chooses CAV controllers", "Benefit depends on human-driver reaction", "Need a model of the reaction: predicted delay change + uncertainty"])
    pic(s, F["setting"], 0.4, 1.5, w=6.4)
    bullets(s, ["An RSU can choose how CAVs drive (the controller)", "Whether that cuts total delay depends on how human drivers react",
                "That reaction is not observable and not simple physics", "So: a model that predicts the reaction and the delay change, with uncertainty"], x=7.0, y=1.5, w=6.0, h=4.6, size=22)
    takeaway(s, "We need to predict the human reaction before choosing a controller")

    # 3 inputs
    s = new_slide("Inputs: only what the RSU can see, turned into a state encoding",
        "The model input is a snapshot of up to 192 vehicles: for each, position, speed and lane, tagged as CAV self-reported or detected neighbour, plus entrance counters. Hidden driver types and true penetration are never inputs. A state encoder turns the snapshot into one representation used by both the dynamics and the effect head.",
        ["Snapshot: up to 192 vehicles (position, speed, lane); entrance counters", "No hidden driver types or penetration", "Snapshot -> state encoder"])
    pic(s, F["inputs"], 0.6, 1.4, w=11.6)
    bullets(s, ["CAV self-state + locally detected neighbours; entrance counters", "No hidden driver types, no true penetration"], y=5.75, h=1.2, size=22)

    # 4 architecture
    s = new_slide("Architecture: physics rollout for the dynamics, a plan-aware head for the delay change",
        "Snapshot goes through the encoder, then two parts. (a) Dynamics: CAVs are driven by the real controller code under the candidate plan; human drivers follow IDM car-following and MOBIL lane change plus a learned neural correction that gives an acceleration distribution and a lane-change decision; everything is rolled forward every 0.5 s. (b) Effect head: outputs the change in total delay versus native at 30, 60, 120, 300 and 570 s as 19 quantiles from 5% to 95%. It equals the TRAIN average effect plus a gate times a learned correction, so an untrained head reproduces the average. It is plan aware: it reads a 120 s summary of CAV motion (count, mean speed, min speed per 10 s bin). Training adds a ranking loss within a decision and per-horizon loss scaling. The arrival head and the driver-response contrast are training signals only.",
        ["Snapshot -> encoder -> (a) dynamics (real controller code; IDM + MOBIL + neural correction; 0.5 s) and (b) effect head", "Effect head: 19 quantiles at 30/60/120/300/570 s = TRAIN average + gate x correction; 120 s plan summary", "Ranking loss within a decision; per-horizon loss scaling", "Arrival head and driver-response contrast: training signals"])
    pic(s, F["arch"], 0.5, 1.4, w=12.3)

    # 5 learning
    s = new_slide("How it learns: replay the same traffic moment under different controllers",
        "We freeze one SUMO traffic moment, so the start is identical, and replay it under different controllers. The difference in total delay between a plan and native, from the identical start, is the true effect label. Losses: one-step and multi-step driver acceleration loss for the neural correction, a quantile (pinball) loss on the effect, and a ranking loss within each decision.",
        ["Freeze one SUMO moment; replay under native / PI / other plans", "Label = delay(plan) - delay(native) from identical start", "Losses: one-step + multi-step driver acceleration, effect quantile loss, ranking loss"])
    pic(s, F["learn"], 0.4, 1.4, w=12.5)
    bullets(s, ["Losses: one-step + multi-step driver acceleration; effect quantile loss; ranking loss within a decision"], y=5.6, h=1.2, size=22)

    # 6 noise floor
    s = new_slide("Is there something to predict? Controller effects are 5-27x larger than noise",
        "90 runs, 30 TRAIN worlds x (R0 true native, N1 one-CAV nudge, S1 seed+1). R0 reproduces native bit-exactly. S1 is bit-identical to R0, so the seed floor is exactly zero. sd(PI-native)/sd(N1-native): 27.17 [16.32, 44.71] at 30 s, 9.42, 6.40, 20.82, 4.96 [3.00, 1724.41] at 570 s. Caveat: the nudge spread is itself heavy-tailed at 570 s. Source: WM-N1-NOISEFLOOR-20261004/RESULT.md.",
        ["Chart: sd ratio per horizon with 90% CI (log scale)", "All lower CI bounds > 1 (16.3, 5.1, 4.0, 12.2, 3.0)", "Seed change = exactly 0"])
    pic(s, F["noise"], 0.4, 1.4, w=7.6)
    bullets(s, ["Ratio = spread of the controller effect / spread from a tiny one-CAV nudge", "All 90% CIs above 1 at every horizon (lower bounds 16.3, 5.1, 4.0, 12.2, 3.0)",
                "Seed change after snapshot restore: exactly zero", "Caveat: 570 s is heavy-tailed even for the nudge"], x=8.2, y=1.5, w=4.8, h=4.6, size=19)
    takeaway(s, "The effect is real: there is a signal for the model to learn")

    # 7 quality
    s = new_slide("Prediction quality: calibrated and better than the average at long horizon, worse at 30 s",
        "CRPS (lower better) on CAL-tune PI-native, 11 decisions, divided by the TRAIN-average (climatology) CRPS; below 1 beats the average. A11 CRPS: 11.0 / 37.3 / 135.3 / 431.7 / 1567.4 vs climatology 5.3 / 25.0 / 129.1 / 528.2 / 1563.6. A12: 11.4 / 40.8 / 135.8 / 427.1 / 1553.6 (REPORT.md of WM-N1-SCORE-A12-20261005). A8-A10 from MORNING_SUMMARY_20261005. A12 did not fix 30 s (11.4 vs 5.3, significantly worse); keeps the 300 s gain (427.1 vs 528.2, CI [-261, 38]). A13 = per-horizon gate + penalty on the actual shift away from the average + controller-type-balanced training; result expected " + pexp + ".",
        ["Chart: CRPS ratio to TRAIN average per horizon, A8..A12 (+A13 pending)", "A11 / A12 at 30 s: 11.0 / 11.4 vs 5.3; at 300 s: 431.7 / 427.1 vs 528.2", "A13 = per-horizon gate + shift penalty + type-balanced training: PENDING"])
    pic(s, F["quality"], 0.3, 1.35, w=7.4)
    cr = N["pi_native_crps"]; a12c = N["a12"]["crps"]; a13c = a13["crps"]
    r13 = ["PENDING"] * 5 if pending(a13c) else [f"{v:g}" for v in a13c]
    rows = [["CRPS, s"] + [f"{t} s" for t in TAUS], ["TRAIN avg"] + [f"{v:g}" for v in cr["climatology"]], ["A11"] + [f"{v:g}" for v in cr["A11"]],
            ["A12"] + [f"{v:g}" for v in a12c], ["A13"] + r13]
    table(s, 7.7, 1.5, 5.4, rows, col_w=[0.9, 0.85, 0.85, 0.9, 0.9, 1.0], size=11, row_h=0.5)
    tb(s, 7.7, 4.3, 5.4, 1.3, ["A13 = per-horizon gate + penalty on the shift away from the average + controller-type-balanced training. PENDING (result expected %s)." % pexp],
       size=15, color="7A4B00", fill=AMBER_BG, line=AMBER, name="a13note")
    takeaway(s, "Long horizon: calibrated, better than the average (300 s: 431.7 vs 528.2). 30 s: worse (11.0 vs 5.3)")

    # 8 decisions
    s = new_slide("Does the prediction lead to better choices? Not yet: the replication did not confirm it",
        f"Decision test of the model's ranking ability: choose the plan with the best predicted delay and compare the regret with always-native (regret = s of total delay vs the best plan, lower is better). Evaluation slice (11 worlds, A11 chosen here): A11 972.3 vs always-native 1555.8 at 300 s, 90% CI of the difference [-1339, -3]. Pre-registered replication on 30 fresh worlds: A11 {rp['a11_regret_300']:g} vs native {rp['native_regret_300']:g}, difference CI [{ci[0]}, {ci[1]}] includes 0, so NOT replicated; A9/A10/A12 also tie native. Message: the model does not yet separate situations well enough; the replication caught a selection effect (the model was picked on the slice).",
        ["Chart: regret at 300 s, evaluation slice vs fresh-world replication", "Slice: 972.3 vs 1555.8, CI [-1339, -3]", f"Replication: {rp['a11_regret_300']:g} vs {rp['native_regret_300']:g}, diff CI [{ci[0]}, {ci[1]}] -> not replicated", "A9 / A10 / A12 also tie native"])
    pic(s, F["repl"], 0.3, 1.35, w=7.4)
    bullets(s, ["Slice (11 worlds, model chosen here): 972.3 vs 1555.8 s, CI [-1339, -3]",
                f"30 fresh worlds, pre-registered: {rp['a11_regret_300']:g} vs {rp['native_regret_300']:g} s, diff CI [{ci[0]}, {ci[1]}]: not replicated",
                "A9, A10, A12 also tie always-native"], x=7.9, y=1.4, w=5.2, h=3.6, size=19)
    tb(s, 7.9, 5.0, 5.2, 1.1, ["Replication caught a selection effect: the win appeared only on the slice used to pick the model."], size=15, color="7A4B00", fill=AMBER_BG, line=AMBER, name="caveat")
    takeaway(s, "The model does not yet separate situations well enough to beat always-native")

    # 9 diagnosis
    dg = N["diag1"]
    s = new_slide("Diagnosis: the model's imagined future was a closed road",
        f"The rollout animated only the cars in the snapshot: no vehicle left the road and none entered. CAVs therefore never exited, so the plan summaries (count, speed) fed to the effect head were unrealistic, which inflated the predictions. Same TRAIN situation (w8676m2700p20t360, PI at 300 s): summary from the real recorded future gives +{dg['real_summary_dv_300']} s, the closed-road rollout summary gives +{dg['closed_road_dv_300']} s, the true effect is +{dg['true_dv_300']} s. Source: E:/rd-wt/rollout-entry/_out/ENTRY_FIX.md (the brief quoted +1356 for the real-future value; the report table says 1355, JSON 1354.67).",
        ["Rollout animated only the snapshot cars: no exits or entries", "Same TRAIN situation, PI effect at 300 s: real-future summary +1355 s, closed-road +3732 s, true +1123 s", "Chart: CAVs on the road per 10 s bin: recorded vs closed road vs open road"])
    pic(s, F["ncav"], 0.3, 1.35, w=7.4)
    rows = [["Same TRAIN situation (PI effect at 300 s)", "s"], ["True effect", f"+{dg['true_dv_300']}"],
            ["Head, real-future summary", f"+{dg['real_summary_dv_300']}"], ["Head, closed-road summary", f"+{dg['closed_road_dv_300']}"]]
    table(s, 7.9, 1.5, 5.2, rows, col_w=[4.0, 1.2], size=16, row_h=0.62)
    bullets(s, ["Rollout kept all 25 CAVs on a road they would have left", "Unrealistic summary -> inflated prediction"], x=7.9, y=4.2, w=5.2, h=1.9, size=19)
    takeaway(s, "Closed-road imagination inflated predictions: +3732 s vs true +1123 s")

    # 10 fix
    e = N["entry_fix"]
    s = new_slide("Fix (tonight, exploratory, TRAIN): open-road rollout; magnitude fixed, ranking still weak",
        "Open-road rollout: vehicles exit at the road end and new vehicles enter using the observed entrance counters (rate = mean of the 12 most recent 10 s bins, Poisson per 0.5 s step). 8 TRAIN states, PI effect at 300 s, mean absolute error vs the true effect: closed road 3564 s, exits only 961, open road 947, real recorded plan 761; correlation with the true effect 0.45 / 0.35 / 0.37 / 0.52. Closed-loop smoke (TRAIN, 180 s): 360 rows, 0 collisions, 0 teleports. Predictions are now +80..+540 s instead of a flat +3700 s, and the planner still picks native. Remaining gap: rollout speeds too fast and smooth (mean 22-27 vs recorded 16-23 m/s), no queues, ranking weak. Exploratory: not a claim. Source: E:/rd-wt/rollout-entry/_out/ENTRY_FIX.md.",
        ["8 TRAIN states: MAE closed 3564 -> exits 961 -> open road 947 vs real plan 761; corr 0.45/0.35/0.37/0.52", "Smoke: 0 collisions; predictions +80..+540 s (was ~+3700), still picks native", "Remaining: rollout too fast/smooth (22-27 vs 16-23 m/s), no queues; weak ranking"])
    pic(s, F["mae"], 0.3, 1.35, w=6.6)
    rows = [["Rollout", "MAE, s", "Corr. with truth"]] + [[l, f"{m:g}", f"{c:g}"] for l, m, c in zip(["closed road", "exits only", "open road", "real recorded plan"], e["mae_s"], e["corr"])]
    table(s, 7.1, 1.4, 6.0, rows, col_w=[2.6, 1.4, 2.0], size=15, row_h=0.5)
    bullets(s, ["Predictions now +80..+540 s, not a flat +3700 s", "Closed-loop smoke safe: 0 collisions", "Gap: rollout too fast and smooth (22-27 vs 16-23 m/s), no queues; ranking weak"], x=7.1, y=4.1, w=6.0, h=2.1, size=17)
    takeaway(s, "Magnitude fixed (MAE 3564 -> 947 s); discrimination is not (corr. about 0.4)")

    # 11 blend
    b = N["blend"]
    s = new_slide("Physics vs learning at short horizons: the closed-road rollout is nearly blind to the effect",
        "Blend check, TRAIN only, fixed 60 PI-vs-native pairs, no new simulation. The closed-road rollout's effect has mean near 0 and sd 0.67 / 3.06 / 8.04 s at 30 / 60 / 120 s versus true sd 24.95 / 104.6 / 327.3 s. Its CRPS is significantly worse than the TRAIN average at all three horizons (CI excludes 0). A rollout+head blend is not significantly better than the average at 30 and 60 s and worse at 120 s. Conclusion: need the open-road, queue-faithful rollout before giving physics weight at short horizons. Source: E:/rd-wt/blend-check/_out/BLEND_CHECK.md.",
        ["60 TRAIN pairs", "Rollout sd 0.7 vs true 25 s at 30 s", "Rollout CRPS worse than average at 30/60/120 s; blend not significantly better"])
    rows = [["Horizon", "True effect sd, s", "Rollout sd, s", "CRPS: average / rollout, s", "Rollout minus average (90% CI)", "Blend minus average (90% CI)"]]
    for i, t in enumerate(b["taus"]):
        rows.append([f"{t} s", f"{b['sd_true'][i]:g}", f"{b['sd_rollout'][i]:g}", f"{b['crps_clim'][i]:g} / {b['crps_rollout'][i]:g}", b["d_rollout"][i] + " worse", b["d_blend"][i] + " n.s."])
    table(s, 0.4, 1.5, 12.5, rows, col_w=[1.1, 1.8, 1.6, 2.6, 2.8, 2.6], size=14, row_h=0.8)
    bullets(s, ["Closed-road rollout: effect sd 0.7 vs true 25 s at 30 s; nearly blind to the effect", "Blend with the head: not significantly better at 30 / 60 s, worse at 120 s (negative = better than average)"], y=4.6, h=1.5, size=19)
    takeaway(s, "Physics only helps once the rollout is open-road and queue-faithful", y=6.3, h=0.65)

    # 12 status
    s = new_slide("Status against the claims, and next steps (world model only)",
        f"Calibrated long horizon: yes (coverage 0.82-1.00 for A10/A11; 300 s CRPS 431.7 / 427.1 vs 528.2). Short horizon: not yet (A12 11.4 vs 5.3); A13 pending (expected {pexp}). Situation discrimination: not yet (replication). Open-road imagination: fixed in magnitude (MAE 3564 -> 947 s), queues and speeds next. Next: queue and speed fidelity in the dynamics; train the effect head on rollout-built summaries too; horizon-dependent blend; seed-variance check; then re-test the decisions on fresh worlds.",
        ["Calibrated long horizon: YES; short horizon: NOT YET (A13 pending); discrimination: NOT YET; open-road imagination: magnitude fixed", "Next: queue/speed fidelity; train head on rollout-built summaries; horizon blend; seed-variance check; re-test decisions"])
    rows = [["Claim", "Status", "Evidence"],
            ["Calibrated, better-than-average long horizon", "YES", "300 s CRPS 431.7 / 427.1 vs 528.2"],
            ["Reliable short horizon", "NOT YET", f"A12 11.4 vs 5.3 at 30 s; A13 PENDING ({pexp})"],
            ["Separates situations / better choices", "NOT YET", f"Replication: {rp['a11_regret_300']:g} vs {rp['native_regret_300']:g}, CI [{ci[0]}, {ci[1]}]"],
            ["Realistic open-road imagination", "PARTLY", "Magnitude fixed (MAE 3564 -> 947 s); queues, speeds next"]]
    gt = table(s, 0.5, 1.4, 12.3, rows, col_w=[4.6, 1.5, 6.2], size=15, row_h=0.55)
    cm = {"PARTLY": "FCE5B0", "NOT YET": "F6C9C9", "YES": "A9DCB8"}
    for r_ in range(1, 5):
        c_ = gt.cell(r_, 1); c_.fill.solid(); c_.fill.fore_color.rgb = rgb(cm[rows[r_][1]]); c_.text_frame.paragraphs[0].runs[0].font.bold = True
    tb(s, 0.5, 4.3, 12.3, 0.45, "Next steps", size=22, color=NAVY, bold=True)
    bullets(s, ["Queue and speed fidelity in the dynamics", "Train the effect head on rollout-built summaries too", "Horizon-dependent blend; seed-variance check", "Then re-test the decisions on fresh worlds"], y=4.75, h=2.2, size=19)

    # 13 appendix
    exp = cl["expected"]
    s = new_slide("Appendix: supporting system (planner + closed loop) ran safely",
        f"Supporting system only. The planner plus the world model were run in closed loop in SUMO. Open-road smoke on one TRAIN world (180 s): 360 rows, 0 collisions, 0 teleports. The registered closed-loop test on 30 fresh worlds (600 s) is PENDING (expected {exp}); fill numbers.json closed_loop.* and rebuild.",
        ["Smoke (TRAIN, 180 s): 0 collisions, 0 teleports", f"Closed-loop 30-world numbers PENDING (expected {exp})"])
    rows = [["Policy (30 fresh worlds, 600 s)", "Mean total delay, s (lower is better)"],
            ["Planner + world model", show(cl["pl01_a11"], exp, "{:g}")], ["always-native", show(cl["always_native"], exp, "{:g}")],
            ["Difference (planner - native), 90% CI", show(cl["diff_pl01_vs_native_ci90"], exp)]]
    table(s, 0.6, 1.5, 12.1, rows, col_w=[6.3, 5.8], size=18, row_h=0.7)
    bullets(s, ["Closed-loop smoke (TRAIN, 180 s): 360 rows, 0 collisions, 0 teleports"], y=4.6, h=1.0, size=22)


def write_outline():
    out = ["# World-model deck - outline and speaker notes (generated by build_deck.py)\n"]
    for n, title, lines, notes in OUTLINE:
        out.append(f"## Slide {n}: {title}\n")
        out += [f"- {l}" for l in lines]
        out.append(f"\n**Notes:** {notes}\n")
    open(os.path.join(HERE, "OUTLINE.md"), "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    F = build_figs(); build_deck(F)
    prs.save(os.path.join(HERE, "WM_world_model_20261005.pptx")); write_outline()
    print("slides:", len(prs.slides)); print("\n".join(WARN) or "no overflow warnings")
